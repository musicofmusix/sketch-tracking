"""Timelapse for older recordings with a single AprilTag beside the paper
(e.g. recording1, tag on a phone screen) instead of the four-corner sheet.

Reuses gaze_on_paper's tag detection and timelapse renderer. Coordinates are in
tag units (1 = tag edge); the view is rotated in 90° steps so that the side
nearest the wearer is at the bottom. Parts of the paper outside the camera's
view stay black.

Usage:
    .venv/bin/python timelapse_single_tag.py recording1 --out output --speed 5
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

import gaze_on_paper as gp

TAG_UNIT = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)


class Canvas:
    """Window [x0, x1] x [y0, y1] in tag units at `ppu` pixels per unit."""

    def __init__(self, x0, y0, x1, y1, ppu):
        self.w = int(round((x1 - x0) * ppu))
        self.h = int(round((y1 - y0) * ppu))
        self.T = np.array([[ppu, 0, -x0 * ppu], [0, ppu, -y0 * ppu], [0, 0, 1]])

    def to_px(self, pts):
        return gp.apply_h(self.T, pts)


def single_tag_corners(rec, tag_id, cache):
    """(frames, 4, 2) image corners of one tag, NaN where unseen."""
    if cache.exists():
        d = np.load(cache)
        c = d["corners"]
        if c.ndim == 3:  # cache written by the earlier single-tag version
            return c
        if "ids" in d.files and tag_id in list(d["ids"]):
            return c[:, list(d["ids"]).index(tag_id)]
    return gp.detect_tags(rec, [tag_id], cache)[:, 0]


def homographies(corners, K, D, max_gap=15):
    n = len(corners)
    H = np.full((n, 3, 3), np.nan)
    for i in range(n):
        if not np.isnan(corners[i, 0, 0]):
            und = gp.undistort(corners[i], K, D).astype(np.float32)
            H[i] = cv2.getPerspectiveTransform(und, TAG_UNIT)
    good = np.where(~np.isnan(H[:, 0, 0]))[0]
    for a, b in zip(good[:-1], good[1:]):
        if 1 < b - a <= max_gap:
            w = ((np.arange(a + 1, b) - a) / (b - a))[:, None, None]
            H[a + 1:b] = H[a] * (1 - w) + H[b] * w
    return H


def orient_to_viewer(H, img_size):
    """Rotate about the tag centre (90° steps) so image-down maps to canvas-down."""
    w, h = img_size
    good = np.where(~np.isnan(H[:, 0, 0]))[0]
    angles = []
    for i in good[:: max(1, len(good) // 200)]:
        a, b = gp.apply_h(H[i], np.array([[w / 2, h / 2], [w / 2, h / 2 + 50]]))
        angles.append(np.arctan2(b[1] - a[1], b[0] - a[0]))
    theta = np.angle(np.mean(np.exp(1j * np.array(angles))))
    k = int(np.round((np.pi / 2 - theta) / (np.pi / 2))) % 4
    c, s = np.round(np.cos(k * np.pi / 2)), np.round(np.sin(k * np.pi / 2))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    C = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    return np.einsum("ij,njk->nik", C @ R @ np.linalg.inv(C), H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", type=Path)
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--tag-id", type=int, default=42)
    ap.add_argument("--region", type=float, nargs=4, default=[-5, -2.5, 6, 8],
                    metavar=("X0", "Y0", "X1", "Y1"), help="view window in tag units")
    ap.add_argument("--ppu", type=float, default=80, help="pixels per tag unit")
    ap.add_argument("--speed", type=int, default=5, help="timelapse speed-up factor")
    args = ap.parse_args()

    rec, out = args.recording, args.out
    out.mkdir(parents=True, exist_ok=True)
    K, D = gp.load_calibration(rec)
    scene_t = np.fromfile(rec / f"{gp.SCENE}.time", "<u8").astype(np.int64)
    cap = cv2.VideoCapture(str(rec / f"{gp.SCENE}.mp4"))
    img_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    corners = single_tag_corners(rec, args.tag_id, out / "tag_corners.npz")[: len(scene_t)]
    H = orient_to_viewer(homographies(corners, K, D), img_size)

    gt, gxy = gp.load_gaze(rec)
    fi = np.clip(np.searchsorted(scene_t, gt), 1, len(scene_t) - 1)
    fi = np.where(np.abs(scene_t[fi - 1] - gt) < np.abs(scene_t[fi] - gt), fi - 1, fi)
    gu = gp.undistort(gxy, K, D)
    plane = np.full((len(gt), 2), np.nan)
    for i in np.unique(fi):
        if not np.isnan(H[i, 0, 0]):
            sel = fi == i
            plane[sel] = gp.apply_h(H[i], gu[sel])
    ok = ~np.isnan(plane[:, 0])

    canvas = Canvas(*args.region, args.ppu)
    pts = canvas.to_px(plane[ok])
    ts = (gt[ok] - scene_t[0]) / 1e9
    step = 10  # 200 Hz -> 20 Hz, as in gaze_on_paper
    print(f"rendering timelapse (x{args.speed})…")
    gp.render_timelapse(rec, H, K, D, canvas, pts[::step], ts[::step], args.speed,
                        out / "timelapse.mp4")
    print("done ->", out / "timelapse.mp4")


if __name__ == "__main__":
    main()
