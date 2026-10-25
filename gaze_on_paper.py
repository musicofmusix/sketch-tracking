"""Map Pupil Labs Neon gaze onto a printed sheet with an AprilTag in each corner.

The sheet (a4_apriltags_36h11.pdf: A4, tag36h11 ids 0-3 clockwise from
top-left, 40 mm tags 10 mm from the edges; these are the defaults) has one tag
per corner. For every scene frame we fit a homography from undistorted image pixels to paper coordinates
(millimetres, origin = paper top-left, y down) using whichever tags are
visible, so hands covering some tags are fine. Gaze samples are mapped onto
the paper and rendered on a separately provided top-down photo of the
finished sheet, which is aligned via the same tags.

Usage:
    .venv/bin/python gaze_on_paper.py recording2 --final-image photo.jpg --out output2
    # add --timelapse (and optionally --speed 10) for a timelapse video
"""

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

SCENE = "Neon Scene Camera v1 ps1"


# --------------------------------------------------------------------------- io

def load_calibration(rec: Path):
    b = (rec / "calibration.bin").read_bytes()
    K = np.frombuffer(b[7:79], "<f8").reshape(3, 3).copy()
    D = np.frombuffer(b[79:143], "<f8").copy()
    return K, D


def load_gaze(rec: Path):
    t = np.fromfile(rec / "gaze ps1.time", "<u8").astype(np.int64)
    xy = np.fromfile(rec / "gaze ps1.raw", "<f4").reshape(-1, 2).astype(np.float64)
    worn = np.fromfile(rec / "worn ps1.raw", "u1")
    if len(worn) == len(t):
        t, xy = t[worn > 0], xy[worn > 0]
    return t, xy


# ------------------------------------------------------------------ paper layout

def paper_layout(paper_w, paper_h, tag_size, margin, ids):
    """Paper-mm positions of each tag's four corners (TL, TR, BR, BL of the tag's
    black outer border, tags printed upright). `ids` are given clockwise from
    the paper's top-left corner."""
    s, m = tag_size, margin
    origins = [(m, m), (paper_w - m - s, m),
               (paper_w - m - s, paper_h - m - s), (m, paper_h - m - s)]
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)
    return {tid: unit * s + np.array(o) for tid, o in zip(ids, origins)}


# ------------------------------------------------------------------ tag tracking

def make_detector():
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        cv2.aruco.DetectorParameters(),
    )
    params = det.getDetectorParameters()
    # Subpixel refinement: ~9x faster than CORNER_REFINE_APRILTAG, corners
    # within ~0.5 px of it (far below gaze accuracy).
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det.setDetectorParameters(params)
    return det


def find_tags(det, gray, ids):
    """{tag_id: (4, 2) image corners} for the wanted ids found in `gray`."""
    c, found, _ = det.detectMarkers(gray)
    if found is None:
        return {}
    return {int(t): cc.reshape(4, 2) for cc, t in zip(c, found.ravel()) if t in ids}


def _detect_chunk(video, ids, start, stop):
    """Tag corners for frames [start, stop) of `video` (runs in a worker)."""
    cv2.setNumThreads(1)  # one process per core already
    det = make_detector()
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    corners = np.full((stop - start, len(ids), 4, 2), np.nan)
    for j in range(stop - start):
        ok, frame = cap.read()
        if not ok:
            break
        for tid, cc in find_tags(det, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), ids).items():
            corners[j, ids.index(tid)] = cc
    return start, corners


def detect_tags(rec: Path, ids, cache: Path, workers=None):
    """Per-frame image corners of each tag, shape (frames, len(ids), 4, 2), NaN
    where a tag was not seen. The video is split into chunks processed in
    parallel; the result is cached because this is the slow step."""
    if cache.exists():
        d = np.load(cache)
        if "ids" in d.files and list(d["ids"]) == list(ids):
            return d["corners"]
    video = str(rec / f"{SCENE}.mp4")
    n = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT))
    workers = workers or os.cpu_count() or 1
    bounds = np.linspace(0, n, workers * 2 + 1).astype(int)  # 2 chunks/worker for balance
    corners = np.full((n, len(ids), 4, 2), np.nan)
    with ProcessPoolExecutor(workers) as pool:
        jobs = [pool.submit(_detect_chunk, video, list(ids), a, b)
                for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        for k, job in enumerate(jobs, 1):
            start, c = job.result()
            corners[start:start + len(c)] = c
            print(f"  tag detection: chunk {k}/{len(jobs)}")
    np.savez(cache, corners=corners, ids=np.array(ids))
    return corners


def undistort(pts, K, D):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, D, P=K).reshape(-1, 2)


def fit_homography(img_pts, paper_pts):
    """Image -> paper homography from corresponding points (>= 4)."""
    if len(img_pts) == 4:
        return cv2.getPerspectiveTransform(img_pts.astype(np.float32),
                                           paper_pts.astype(np.float32))
    H, _ = cv2.findHomography(img_pts, paper_pts, cv2.RANSAC, 3.0)
    return H


def frame_homographies(corners, ids, layout, K, D, max_gap=15):
    """Undistorted image -> paper-mm homography per frame, from all visible tags.

    Frames with no tag visible are filled by interpolating the (normalised)
    homographies of the neighbouring frames across gaps up to `max_gap`.
    """
    n = len(corners)
    H = np.full((n, 3, 3), np.nan)
    paper = np.stack([layout[t] for t in ids])
    for i in range(n):
        seen = ~np.isnan(corners[i, :, 0, 0])
        if not seen.any():
            continue
        img = undistort(corners[i, seen].reshape(-1, 2), K, D)
        h = fit_homography(img, paper[seen].reshape(-1, 2))
        if h is not None:
            H[i] = h / h[2, 2]

    good = np.where(~np.isnan(H[:, 0, 0]))[0]
    for a, b in zip(good[:-1], good[1:]):
        if 1 < b - a <= max_gap:
            w = ((np.arange(a + 1, b) - a) / (b - a))[:, None, None]
            H[a + 1:b] = H[a] * (1 - w) + H[b] * w
    return H


def apply_h(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


# ------------------------------------------------------------------- rendering

class Canvas:
    """The paper (plus `pad` mm around it) rendered at `ppm` pixels per mm."""

    def __init__(self, paper_w, paper_h, pad, ppm):
        self.ppm = ppm
        self.x0, self.y0 = -pad, -pad
        self.w = int(round((paper_w + 2 * pad) * ppm))
        self.h = int(round((paper_h + 2 * pad) * ppm))
        # paper mm -> canvas pixels
        self.T = np.array([[ppm, 0, pad * ppm], [0, ppm, pad * ppm], [0, 0, 1]])

    def to_px(self, pts):
        return apply_h(self.T, pts)


def rectify(frame, H, maps, canvas):
    """Undistort a scene frame and warp it top-down onto the canvas."""
    und = cv2.remap(frame, *maps, cv2.INTER_LINEAR)
    return cv2.warpPerspective(und, canvas.T @ H, (canvas.w, canvas.h),
                               flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def photo_background(path, ids, layout, canvas):
    """Warp a top-down photo of the finished sheet onto the canvas via its tags.

    The photo is assumed to be (close to) distortion-free, as phone/DSLR photos
    are.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = make_detector()
    tags = find_tags(det, gray, ids)
    if len(tags) < len(ids):
        # Very large photos can defeat the detector; retry downscaled.
        f = 2000 / max(gray.shape)
        small = find_tags(det, cv2.resize(gray, None, fx=f, fy=f), ids)
        if len(small) > len(tags):
            tags = {t: c / f for t, c in small.items()}
    if not tags:
        raise SystemExit(f"none of tags {ids} found in {path}")
    missing = [t for t in ids if t not in tags]
    if missing:
        print(f"  warning: tags {missing} not found in final photo")
    img_pts = np.concatenate([tags[t] for t in tags])
    paper_pts = np.concatenate([layout[t] for t in tags])
    Hp = fit_homography(img_pts, paper_pts)
    return cv2.warpPerspective(img, canvas.T @ Hp, (canvas.w, canvas.h),
                               flags=cv2.INTER_AREA, borderValue=(255, 255, 255))


# Trace colours run early -> late through this palette.
PALETTE = ["#FA0080", "#00DDFA", "#FADD00", "#A59837", "#7A3D5D"]


def palette_lut(n=256):
    """(n, 3) uint8 BGR colours interpolated through PALETTE."""
    rgb = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in PALETTE], float)
    x = np.linspace(0, 1, len(rgb))
    q = np.linspace(0, 1, n)
    lut = np.stack([np.interp(q, x, rgb[:, c]) for c in range(3)], axis=1)
    return lut[:, ::-1].round().astype(np.uint8)


def time_colors(t, t0, t1):
    lut = palette_lut()
    tn = np.clip((np.asarray(t, float) - t0) / max(1e-9, t1 - t0), 0, 1)
    return [tuple(int(v) for v in lut[int(x * 255)]) for x in tn]


# ------------------------------------------------------------- fixations

def detect_fixations(t, xy, max_dispersion=8.0, min_duration=0.1, max_gap=0.1):
    """Dispersion-threshold (I-DT) fixations on paper coordinates.

    A fixation is a run of samples lasting >= `min_duration` s whose x and y
    ranges both stay within `max_dispersion` (mm). Runs are broken by missing
    samples (NaN) or time gaps > `max_gap` s. Works the same for recordings
    and live streams. Returns an (n, 5) array: start_s, end_s, x, y, dwell_s.
    """
    t, xy = np.asarray(t, float), np.asarray(xy, float)
    out, n, i = [], len(t), 0
    while i < n:
        if np.isnan(xy[i, 0]):
            i += 1
            continue
        lo = hi = xy[i].copy()
        j = i + 1
        while j < n and not np.isnan(xy[j, 0]) and t[j] - t[j - 1] <= max_gap:
            nlo, nhi = np.minimum(lo, xy[j]), np.maximum(hi, xy[j])
            if (nhi - nlo).max() > max_dispersion:
                break
            lo, hi = nlo, nhi
            j += 1
        if t[j - 1] - t[i] >= min_duration:
            c = xy[i:j].mean(0)
            out.append((t[i], t[j - 1], c[0], c[1], t[j - 1] - t[i]))
            i = j
        else:
            i += 1
    return np.array(out, float).reshape(-1, 5)


def cluster_fixations(fix, radius=25.0):
    """Coarser view: merge consecutive fixations whose centre lies within
    `radius` mm of the running (dwell-weighted) centre of the current cluster.
    Same column layout as detect_fixations; dwell is the summed fixation time."""
    out = []
    for f in fix:
        if out:
            c = out[-1]
            if np.hypot(f[2] - c[2], f[3] - c[3]) <= radius:
                w = c[4] + f[4]
                c[2] = (c[2] * c[4] + f[2] * f[4]) / w
                c[3] = (c[3] * c[4] + f[3] * f[4]) / w
                c[1], c[4] = f[1], w
                continue
        out.append(list(f))
    return np.array(out, float).reshape(-1, 5)


def save_fixations_csv(fix, path):
    with open(path, "w") as f:
        f.write("start_sec,end_sec,paper_x_mm,paper_y_mm,dwell_sec\n")
        for r in fix:
            f.write(f"{r[0]:.3f},{r[1]:.3f},{r[2]:.2f},{r[3]:.2f},{r[4]:.3f}\n")


# --------------------------------------------------------- trace overlays

DWELL_RADIUS_MM = 4.0  # circle radius for 1 s of dwell; area grows with dwell


def _blend_circle(img, center, radius, color, alpha):
    """Filled, anti-aliased translucent circle (blends only its bounding box)."""
    x, y = center
    r = int(np.ceil(radius)) + 2
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(w, x + r + 1), min(h, y + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    roi = img[y0:y1, x0:x1]
    layer = roi.copy()
    cv2.circle(layer, (x - x0, y - y0), int(round(radius)), color, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_trace_fine(bg, pts_px, t_sec, out_path, t_range=None):
    """Fine view: the gaze path as connected dots, coloured early -> late."""
    t0, t1 = t_range or (float(t_sec.min()), float(t_sec.max()))
    img = bg.copy()
    overlay = img.copy()
    colors = time_colors(t_sec, t0, t1)
    p = np.round(pts_px).astype(int)
    for k in range(1, len(p)):
        cv2.line(overlay, tuple(p[k - 1]), tuple(p[k]), colors[k], 1, cv2.LINE_AA)
    for k in range(len(p)):
        cv2.circle(overlay, tuple(p[k]), 2, colors[k], -1, cv2.LINE_AA)
    img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
    cv2.imwrite(str(out_path), img)


def draw_dwell_circles(bg, fix, canvas, out_path, t_range):
    """Circle view: one translucent circle per fixation/cluster, area
    proportional to dwell time, coloured by time and linked in order.

    Circles use DWELL_RADIUS_MM per sqrt(second), shrunk for this image if
    needed so the longest dwell stays within ~12% of the image."""
    t0, t1 = t_range
    img = bg.copy()
    k = DWELL_RADIUS_MM * canvas.ppm  # px radius per sqrt(s)
    max_dwell = float(fix[:, 4].max()) if len(fix) else 1.0
    limit = 0.12 * min(img.shape[:2])
    if k * max_dwell ** 0.5 > limit:
        k = limit / max_dwell ** 0.5
    if len(fix):
        c = np.round(canvas.to_px(fix[:, 2:4])).astype(int)
        colors = time_colors((fix[:, 0] + fix[:, 1]) / 2, t0, t1)
        links = img.copy()
        for i in range(1, len(c)):
            cv2.line(links, tuple(c[i - 1]), tuple(c[i]), (110, 110, 110), 1, cv2.LINE_AA)
        img = cv2.addWeighted(links, 0.6, img, 0.4, 0)
        for i in range(len(c)):
            r = max(2.0, fix[i, 4] ** 0.5 * k)
            _blend_circle(img, tuple(c[i]), r, colors[i], 0.55)
    cv2.imwrite(str(out_path), img)


def paper_background(paper_w, paper_h, ppm, layout=None):
    """Clean background for the *_clean views: the calibrated sheet only, as
    a white page with a hairline border and the corner tags as pale squares."""
    canvas = Canvas(paper_w, paper_h, 0, ppm)
    img = np.full((canvas.h, canvas.w, 3), 255, np.uint8)
    for corners in (layout or {}).values():
        q = np.round(canvas.to_px(corners)).astype(np.int32)
        cv2.fillPoly(img, [q], (242, 242, 242), cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (canvas.w - 1, canvas.h - 1), (216, 216, 216), 1)
    return img, canvas


def render_trace_overlays(bg, t_sec, paper_xy, canvas, out_dir, fine_step=10,
                          max_dispersion=8.0, cluster_radius=25.0,
                          paper_size=None, layout=None):
    """All trace views of one session, from fine to coarse:
    trace_fine.png (gaze path), trace_fixations.png (fixations, circle area =
    dwell) and trace_clusters.png (fixations merged into regions of attention),
    plus fixations.csv / clusters.csv. `paper_xy` in mm, NaN where unmapped.

    With `paper_size` (mm), each view is also drawn as *_clean.png: no photo,
    just a white sheet cropped to the tag-calibrated paper, showing only the
    gaze that falls on it (paths are clipped at the paper edge)."""
    out_dir = Path(out_dir)
    t_sec, paper_xy = np.asarray(t_sec, float), np.asarray(paper_xy, float)
    if len(t_sec) < 2:
        return {}
    t_range = (max(0.0, float(t_sec.min())), float(t_sec.max()))
    ok = ~np.isnan(paper_xy[:, 0])
    px = canvas.to_px(paper_xy[ok])
    draw_trace_fine(bg, px[::fine_step], t_sec[ok][::fine_step], out_dir / "trace_fine.png", t_range)

    fix = detect_fixations(t_sec, paper_xy, max_dispersion)
    clusters = cluster_fixations(fix, cluster_radius)
    save_fixations_csv(fix, out_dir / "fixations.csv")
    save_fixations_csv(clusters, out_dir / "clusters.csv")
    draw_dwell_circles(bg, fix, canvas, out_dir / "trace_fixations.png", t_range)
    draw_dwell_circles(bg, clusters, canvas, out_dir / "trace_clusters.png", t_range)

    if paper_size is not None:
        pw, ph = paper_size
        white, clean = paper_background(pw, ph, canvas.ppm, layout)
        xy = paper_xy[ok][::fine_step]
        tt = t_sec[ok][::fine_step]
        inside = (xy[:, 0] >= 0) & (xy[:, 0] <= pw) & (xy[:, 1] >= 0) & (xy[:, 1] <= ph)
        # keep a point if it or a neighbour is on the paper, so paths entering or
        # leaving the sheet are drawn up to its edge (the image border clips them)
        keep = inside | np.r_[inside[1:], False] | np.r_[False, inside[:-1]]
        draw_trace_fine(white, clean.to_px(xy[keep]), tt[keep], out_dir / "trace_fine_clean.png",
                        t_range)
        on = lambda f: f[(f[:, 2] >= 0) & (f[:, 2] <= pw) & (f[:, 3] >= 0) & (f[:, 3] <= ph)]
        draw_dwell_circles(white, on(fix), clean, out_dir / "trace_fixations_clean.png", t_range)
        draw_dwell_circles(white, on(clusters), clean, out_dir / "trace_clusters_clean.png", t_range)
    return {"fixations": len(fix), "clusters": len(clusters),
            "fixation_dwell_sec": float(fix[:, 4].sum()) if len(fix) else 0.0}


# Heatmap bands, lowest -> highest dwell (the trace palette, reordered so the
# light colours sit at the edges and the dark ones at the peaks), and the
# band thresholds as fractions of the peak dwell density.
HEAT_BANDS = ["#00DDFA", "#FADD00", "#A59837", "#FA0080", "#7A3D5D"]
HEAT_LEVELS = [0.015, 0.05, 0.13, 0.3, 0.6]


def _bgr(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (5, 3, 1))


def _hatch(mask, spacing, color, img, thickness=1, angle=45):
    """Parallel hatch lines clipped to `mask`, cartography style."""
    h, w = mask.shape
    lines = np.zeros((h, w), np.uint8)
    d = h + w
    for off in range(-d, d, spacing):
        if angle == 45:
            p0, p1 = (off, 0), (off + h, h)
        else:
            p0, p1 = (off, h), (off + h, 0)
        cv2.line(lines, p0, p1, 255, thickness, cv2.LINE_AA)
    a = (lines.astype(np.float32) / 255 * mask)[..., None]
    img[:] = (img * (1 - a * 0.55) + np.array(color, np.float32) * a * 0.55).astype(np.uint8)


def draw_heatmap(bg, pts_px, weights, sigma_px, out_path, raw_path, ppm=None):
    """Dwell map drawn like a stylised contour map: the smoothed dwell
    density is cut into flat bands (HEAT_LEVELS of the peak) filled with the
    palette, alternate bands hatched, each band edged with a thin contour line.
    `heatmap.npy` keeps the continuous density (seconds per pixel)."""
    h, w = bg.shape[:2]
    acc = np.zeros((h, w), np.float64)
    p = np.round(pts_px).astype(int)
    m = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    np.add.at(acc, (p[m, 1], p[m, 0]), weights[m])
    heat = gaussian_filter(acc, sigma_px)
    np.save(raw_path, heat)
    peak = heat.max()

    img = bg.copy()
    u = max(1, int(round(w / 900)))
    if peak > 0:
        # soften the photo a little so the flat colours read as a map layer
        img = cv2.addWeighted(img, 0.8, np.full_like(img, 255), 0.2, 0)
        hn = heat / peak
        for k, (lvl, hex_color) in enumerate(zip(HEAT_LEVELS, HEAT_BANDS)):
            band = hn >= lvl
            if not band.any():
                continue
            col = np.array(_bgr(hex_color), np.float32)
            a = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 0.8)[..., None] * 0.62
            img = (img * (1 - a) + col * a).astype(np.uint8)
            if k % 2 == 1:  # hatch every other band, like the reference map
                dark = tuple(int(v) for v in (col * 0.55))
                _hatch(band.astype(np.float32), 7 * u, dark, img, 1, 45 if k == 1 else -45)
            contours, _ = cv2.findContours(band.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            edge = tuple(int(v) for v in (col * 0.6))
            cv2.drawContours(img, contours, -1, edge, u, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    # band minima, s/cm^2 of dwell (recorded in summary.json since the image has no legend)
    to_cm2 = (ppm * 10) ** 2 if ppm else 1.0
    return [float(lvl * peak * to_cm2) for lvl in HEAT_LEVELS]


def render_heatmaps(bg, t_sec, paper_xy, canvas, out_dir, sigma_mm=5.0,
                    paper_size=None, layout=None):
    """heatmap.png on the photo and, with `paper_size`, heatmap_clean.png on the
    white calibrated sheet. Each 200 Hz sample adds its duration of dwell."""
    out_dir = Path(out_dir)
    t_sec, paper_xy = np.asarray(t_sec, float), np.asarray(paper_xy, float)
    ok = ~np.isnan(paper_xy[:, 0])
    if ok.sum() < 2:
        return {}
    dt = np.clip(np.diff(t_sec[ok], append=t_sec[ok][-1]), 0, 0.02)
    bands = draw_heatmap(bg, canvas.to_px(paper_xy[ok]), dt, sigma_mm * canvas.ppm,
                         out_dir / "heatmap.png", out_dir / "heatmap.npy", ppm=canvas.ppm)
    if paper_size is not None:
        white, clean = paper_background(*paper_size, canvas.ppm, layout)
        draw_heatmap(white, clean.to_px(paper_xy[ok]), dt, sigma_mm * canvas.ppm,
                     out_dir / "heatmap_clean.png", out_dir / "heatmap_clean.npy", ppm=canvas.ppm)
    return {"heatmap_band_min_s_per_cm2": bands}


class VideoOut:
    """H.264 via ffmpeg when available (plays everywhere), else OpenCV mp4v."""

    def __init__(self, path, w, h, fps):
        self.proc = self.cv = None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
                stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, img):
        if self.proc:
            self.proc.stdin.write(img.tobytes())
        else:
            self.cv.write(img)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


class TimelapseComposer:
    """Draws the building-up gaze trace onto successive top-down views.

    The full history is drawn faintly (coloured early -> late through PALETTE,
    as in trace_fine.png); the last `trail_sec` seconds are drawn bright with a
    ring at the current gaze, plus an elapsed-time label.
    """

    def __init__(self, canvas_w, canvas_h, gaze_px, gaze_t, speed, trail_sec=2.0):
        self.colors = time_colors(gaze_t, gaze_t.min(), gaze_t.max()) if len(gaze_t) else []
        self.p = np.round(gaze_px).astype(int)
        self.t = gaze_t
        self.speed, self.trail_sec = speed, trail_sec
        self.history = np.zeros((canvas_h, canvas_w, 3), np.uint8)
        self.drawn = 0

    def frame(self, view, now):
        """`view`: top-down image at time `now` (seconds since start)."""
        p, colors = self.p, self.colors
        upto = int(np.searchsorted(self.t, now, side="right"))
        for k in range(max(self.drawn, 1), upto):
            cv2.line(self.history, tuple(p[k - 1]), tuple(p[k]), colors[k], 1, cv2.LINE_AA)
        self.drawn = max(self.drawn, upto)

        img = view.copy()
        mask = self.history.any(axis=2)
        img[mask] = (img[mask] * 0.45 + self.history[mask] * 0.55).astype(np.uint8)
        start = int(np.searchsorted(self.t, now - self.trail_sec))
        for k in range(max(start, 1), upto):
            cv2.line(img, tuple(p[k - 1]), tuple(p[k]), colors[k], 2, cv2.LINE_AA)
        if upto > 0 and now - self.t[upto - 1] < 0.5:
            cv2.circle(img, tuple(p[upto - 1]), 8, (0, 255, 255), 2, cv2.LINE_AA)
        label = f"{int(now // 60):02d}:{int(now % 60):02d}  x{self.speed}"
        cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return img


def render_timelapse(rec, H, K, D, canvas, gaze_px, gaze_t, speed, out_path,
                     trail_sec=2.0):
    """Top-down timelapse of the sketching with the gaze trace building up.

    Every `speed`-th scene frame is rectified onto the paper and written at the
    scene frame rate, so playback is `speed`x real time. Frames where the paper
    could not be located repeat the previous view.
    """
    cap = cv2.VideoCapture(str(rec / f"{SCENE}.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scene_t = np.fromfile(rec / f"{SCENE}.time", "<u8").astype(np.int64)
    fps = (len(scene_t) - 1) / ((scene_t[-1] - scene_t[0]) / 1e9)
    t_frame = (scene_t - scene_t[0]) / 1e9
    maps = cv2.initUndistortRectifyMap(K, D, None, K, (fw, fh), cv2.CV_32FC1)

    w, h = canvas.w - canvas.w % 2, canvas.h - canvas.h % 2  # yuv420p needs even
    writer = VideoOut(out_path, w, h, fps)
    comp = TimelapseComposer(canvas.w, canvas.h, gaze_px, gaze_t, speed, trail_sec)
    view = np.zeros((canvas.h, canvas.w, 3), np.uint8)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if i % speed:
            continue
        if not np.isnan(H[i, 0, 0]):
            view = rectify(frame, H[i], maps, canvas)
        img = comp.frame(view, t_frame[min(i, len(t_frame) - 1)])
        writer.write(np.ascontiguousarray(img[:h, :w]))
        if i % 1000 < speed:
            print(f"  timelapse {i}/{n}")
    writer.close()


def add_trace_args(ap):
    """Fixation/cluster options, shared with live_gaze.py."""
    ap.add_argument("--fix-dispersion", type=float, default=8,
                    help="max spread (mm) of gaze within one fixation")
    ap.add_argument("--cluster-radius", type=float, default=25,
                    help="fixations within this many mm are merged into one cluster")


def add_layout_args(ap):
    """Paper/tag layout options, shared with live_gaze.py."""
    ap.add_argument("--tag-ids", type=int, nargs=4, default=[0, 1, 2, 3],
                    metavar=("TL", "TR", "BR", "BL"),
                    help="tag ids clockwise from the paper's top-left corner")
    ap.add_argument("--paper-size", type=float, nargs=2, default=[210, 297],
                    metavar=("W", "H"), help="paper size in mm (default A4 portrait)")
    ap.add_argument("--tag-size", type=float, default=40,
                    help="printed tag edge length in mm (outer black border)")
    ap.add_argument("--tag-margin", type=float, default=10,
                    help="distance in mm from paper edge to tag edge")


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", type=Path)
    ap.add_argument("--final-image", type=Path, required=True,
                    help="top-down photo of the finished sheet, all corner tags visible")
    ap.add_argument("--out", type=Path, default=Path("output"))
    add_layout_args(ap)
    ap.add_argument("--pad", type=float, default=20,
                    help="mm of space around the paper kept in the output")
    ap.add_argument("--ppm", type=float, default=4, help="output pixels per mm")
    ap.add_argument("--sigma", type=float, default=5, help="heatmap blur, mm")
    add_trace_args(ap)
    ap.add_argument("--timelapse", action="store_true",
                    help="also render timelapse.mp4: top-down video of the sketching "
                         "with the gaze trace building up")
    ap.add_argument("--speed", type=int, default=5, help="timelapse speed-up factor")
    args = ap.parse_args()

    rec, out = args.recording, args.out
    out.mkdir(parents=True, exist_ok=True)
    ids = list(args.tag_ids)
    pw, ph = args.paper_size
    layout = paper_layout(pw, ph, args.tag_size, args.tag_margin, ids)
    K, D = load_calibration(rec)
    scene_t = np.fromfile(rec / f"{SCENE}.time", "<u8").astype(np.int64)

    print("detecting tags…")
    corners = detect_tags(rec, ids, out / "tag_corners.npz")[: len(scene_t)]
    seen = ~np.isnan(corners[:, :, 0, 0])
    for t, frac in zip(ids, seen.mean(0)):
        print(f"  tag {t}: {frac * 100:.1f}% of frames")
    print(f"  any tag: {seen.any(1).mean() * 100:.1f}% of frames")
    H = frame_homographies(corners, ids, layout, K, D)

    # Gaze -> nearest scene frame -> paper mm.
    gt, gxy = load_gaze(rec)
    fi = np.clip(np.searchsorted(scene_t, gt), 1, len(scene_t) - 1)
    fi = np.where(np.abs(scene_t[fi - 1] - gt) < np.abs(scene_t[fi] - gt), fi - 1, fi)
    gu = undistort(gxy, K, D)
    paper = np.full((len(gt), 2), np.nan)
    for i in np.unique(fi):
        if np.isnan(H[i, 0, 0]):
            continue
        sel = fi == i
        paper[sel] = apply_h(H[i], gu[sel])
    ok = ~np.isnan(paper[:, 0])
    on_paper = ok & (paper[:, 0] >= 0) & (paper[:, 0] <= pw) & (paper[:, 1] >= 0) & (paper[:, 1] <= ph)
    print(f"  {ok.mean() * 100:.1f}% of gaze samples mapped, "
          f"{on_paper.mean() * 100:.1f}% on the paper")

    t0 = scene_t[0]
    with open(out / "gaze_on_paper.csv", "w") as f:
        f.write("timestamp_ns,t_sec,scene_frame,gaze_x_px,gaze_y_px,paper_x_mm,paper_y_mm,on_paper\n")
        for k in np.where(ok)[0]:
            f.write(f"{gt[k]},{(gt[k] - t0) / 1e9:.4f},{fi[k]},{gxy[k, 0]:.2f},"
                    f"{gxy[k, 1]:.2f},{paper[k, 0]:.2f},{paper[k, 1]:.2f},{int(on_paper[k])}\n")

    canvas = Canvas(pw, ph, args.pad, args.ppm)
    bg = photo_background(args.final_image, ids, layout, canvas)
    cv2.imwrite(str(out / "final_photo.png"), bg)

    gp = canvas.to_px(paper[ok])
    gts = (gt[ok] - t0) / 1e9

    # Trace overlays, fine -> coarse (fixations detected on the paper coordinates).
    stats = render_trace_overlays(bg, (gt - t0) / 1e9, paper, canvas, out,
                                  max_dispersion=args.fix_dispersion,
                                  cluster_radius=args.cluster_radius,
                                  paper_size=(pw, ph), layout=layout)
    stats.update(render_heatmaps(bg, (gt - t0) / 1e9, paper, canvas, out, args.sigma,
                                 paper_size=(pw, ph), layout=layout))
    step = 10  # 200 Hz -> 20 Hz for the timelapse trace

    if args.timelapse:
        print(f"rendering timelapse (x{args.speed})…")
        render_timelapse(rec, H, K, D, canvas, gp[::step], gts[::step], args.speed,
                         out / "timelapse.mp4")

    json.dump({"paper_mm": [pw, ph], "tag_ids": ids, "tag_size_mm": args.tag_size,
               "tag_margin_mm": args.tag_margin, "pad_mm": args.pad, "ppm": args.ppm,
               "tag_seen_fraction": dict(zip(map(str, ids), seen.mean(0).tolist())),
               "frames": len(scene_t), "gaze_mapped_fraction": float(ok.mean()),
               "gaze_on_paper_fraction": float(on_paper.mean()),
               **stats},
              open(out / "summary.json", "w"), indent=2)
    print("done ->", out)


if __name__ == "__main__":
    main()
