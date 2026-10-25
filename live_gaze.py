"""SketchTracking: real-time gaze-on-paper with a browser GUI, streaming from the Neon Companion app.

Connects to the Companion app on the local network, finds the four corner tags
on the printed sheet in every scene frame and maps gaze onto the paper live.
The GUI (http://localhost:8000) shows the scene video with gaze point and trace,
a top-down view of the paper, and start/stop recording. When a recording stops,
the session folder gets the same outputs as gaze_on_paper.py: timelapse,
gaze-path overlay, dwell heatmap and a CSV. A top-down photo of the finished
sketch can be added (before or after stopping) and is used as the background
for the stills.

The source (auto-discover, phone IP, or replay of a recording folder) can be
switched from the GUI; the options below only set the initial one.

Usage:
    .venv/bin/python live_gaze.py                       # auto-discover the phone
    .venv/bin/python live_gaze.py --host 192.168.1.23   # or connect by IP
    .venv/bin/python live_gaze.py --replay recording2   # replay a recording folder
"""

import argparse
import asyncio
import json
import shutil
import socket
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web

import gaze_on_paper as gp

UI_DIR = Path(__file__).parent / "live_ui"
TIMELAPSE_FPS = 30
LIVE_TRAIL_SEC = 3.0


# ------------------------------------------------------------------- sources

class Cancelled(Exception):
    """The GUI switched source while this one was still connecting."""


class NeonSource:
    """Scene frames and 200 Hz gaze from the Companion app's real-time API."""

    def __init__(self, host=None, port=8080, cancelled=lambda: False):
        from pupil_labs.realtime_api.simple import Device, discover_one_device

        if host:
            try:  # fail fast with a clear message instead of a 30 s library timeout
                socket.create_connection((host, port), timeout=3).close()
            except OSError:
                raise ConnectionError(f"no Companion app reachable at {host}:{port}") from None
            self.device = Device(host, port)
        else:
            # search in short slices so a source change in the GUI takes effect quickly
            self.device = None
            while self.device is None:
                if cancelled():
                    raise Cancelled()
                self.device = discover_one_device(1.0)
        calib = self.device.get_calibration()
        self.K = np.asarray(calib.scene_camera_matrix, np.float64).reshape(3, 3)
        self.D = np.asarray(calib.scene_distortion_coefficients, np.float64).ravel()
        self.name = f"{self.device.phone_name} ({self.device.phone_ip})"
        self.can_record = True
        self._gaze = deque(maxlen=2000)
        self._running = True
        threading.Thread(target=self._gaze_loop, daemon=True).start()

    def _gaze_loop(self):
        # The simple API only keeps the latest item per stream, so read gaze
        # in its own tight loop to get (nearly) every 200 Hz sample.
        while self._running:
            g = self.device.receive_gaze_datum(timeout_seconds=1)
            if g is not None:
                self._gaze.append((g.timestamp_unix_seconds, g.x, g.y, bool(g.worn)))

    def read_frame(self):
        f = self.device.receive_scene_video_frame(timeout_seconds=1)
        return None if f is None else (f.bgr_pixels, f.timestamp_unix_seconds)

    def drain_gaze(self, upto):
        out = []
        while self._gaze and self._gaze[0][0] <= upto:
            out.append(self._gaze.popleft())
        return out

    def recording_start(self):
        self.device.recording_start()

    def recording_stop(self):
        self.device.recording_stop_and_save()

    def info(self):
        try:
            return {"battery": self.device.battery_level_percent}
        except Exception:
            return {}

    def close(self):
        self._running = False
        self.device.close()


def is_recording_dir(p: Path):
    """A native Neon recording folder (as exported from the Companion app)."""
    return all((p / f).is_file() for f in (f"{gp.SCENE}.mp4", f"{gp.SCENE}.time",
                                           "gaze ps1.raw", "gaze ps1.time", "calibration.bin"))


class ReplaySource:
    """Plays a native Neon recording folder in real time (loops), for testing
    the GUI and pipeline without the glasses."""

    def __init__(self, rec: Path):
        if not is_recording_dir(rec):
            raise ValueError(f"{rec} is not a Neon recording folder")
        self.rec = rec
        self.K, self.D = gp.load_calibration(rec)
        self.name = f"replay: {rec.name}"
        self.can_record = False
        st = np.fromfile(rec / f"{gp.SCENE}.time", "<u8").astype(np.int64)
        gt, gxy = gp.load_gaze(rec)
        self.frame_t = (st - st[0]) / 1e9
        self.gaze = np.c_[(gt - st[0]) / 1e9, gxy]
        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(str(self.rec / f"{gp.SCENE}.mp4"))
        self.i = 0
        self.gi = 0
        self.t0 = time.time()

    def read_frame(self):
        if self.i >= len(self.frame_t):
            self._open()
        ok, frame = self.cap.read()
        if not ok:
            self._open()
            ok, frame = self.cap.read()
        t = self.frame_t[self.i]
        self.i += 1
        wait = self.t0 + t - time.time()
        if wait > 0:
            time.sleep(wait)
        return frame, self.t0 + t

    def drain_gaze(self, upto):
        out = []
        while self.gi < len(self.gaze) and self.t0 + self.gaze[self.gi, 0] <= upto:
            t, x, y = self.gaze[self.gi]
            out.append((self.t0 + t, x, y, True))
            self.gi += 1
        return out

    def info(self):
        return {}

    def close(self):
        pass


# ------------------------------------------------------------------- session

class Session:
    """One recording: gaze rows plus a clean top-down video for the timelapse."""

    def __init__(self, root: Path, canvas, speed):
        self.dir = root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.speed = speed
        self.t0 = None  # device clock (stream timestamps)
        self.wall0 = time.time()  # host clock, for the GUI timer
        self.rows = []  # (t_rel, raw_x, raw_y, paper_x, paper_y, worn)
        self.tl_times = []
        self.next_tl = 0.0
        w, h = canvas.w - canvas.w % 2, canvas.h - canvas.h % 2
        self.tl_size = (w, h)
        self.tl_raw = gp.VideoOut(self.dir / "timelapse_clean.mp4", w, h, TIMELAPSE_FPS)
        self.last_view = None
        self.sketch = None
        self.outputs = []
        self.state = "recording"  # -> processing -> done | error
        self.error = None

    def elapsed(self):
        return time.time() - self.wall0


def list_outputs(d: Path):
    return sorted(p.name for p in d.iterdir() if p.suffix in (".png", ".mp4", ".csv", ".json"))


# -------------------------------------------------------------------- engine

class Engine:
    def __init__(self, args):
        self.args = args
        self.ids = list(args.tag_ids)
        self.pw, self.ph = args.paper_size
        self.layout = gp.paper_layout(self.pw, self.ph, args.tag_size, args.tag_margin, self.ids)
        self.paper_pts = np.stack([self.layout[t] for t in self.ids])
        self.det = gp.make_detector()
        self.preview_canvas = gp.Canvas(self.pw, self.ph, 10, 2)
        self.out_canvas = gp.Canvas(self.pw, self.ph, args.pad, args.ppm)
        self.sessions_dir = args.out.resolve()

        self.lock = threading.Lock()
        self.source = None
        self.status = {"connected": False, "device": None, "message": "connecting…"}
        self.scene_jpg = self.paper_jpg = None
        self.frame_id = 0
        self.fps = 0.0
        self.tags_seen = []
        self.raw_trail = deque()    # (t, x, y) raw image gaze, for the live trail
        self.paper_trail = deque()  # (t, x, y) paper mm, for the live paper view
        self.session = None
        self.last_session = None
        self.maps = None
        self._stop = False
        # what to connect to; changed from the GUI via set_target()
        if args.replay:
            self.target = {"mode": "replay", "path": str(Path(args.replay).resolve())}
        elif args.host:
            self.target = {"mode": "ip", "host": args.host, "port": args.port}
        else:
            self.target = {"mode": "discover"}
        self.gen = 0  # bumped on every target change
        self._show_placeholder("connecting…")

    # ---- lifecycle

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def set_target(self, target):
        """Switch source (from the GUI). Stops a running recording first."""
        mode = target.get("mode")
        if mode == "replay":
            path = Path(target.get("path", "")).expanduser()
            if not is_recording_dir(path):
                raise ValueError(f"not a Neon recording folder: {path}")
            target = {"mode": "replay", "path": str(path.resolve())}
        elif mode == "ip":
            host = (target.get("host") or "").strip()
            if not host:
                raise ValueError("enter the phone's IP address")
            target = {"mode": "ip", "host": host, "port": int(target.get("port") or 8080)}
        elif mode not in ("discover", "none"):
            raise ValueError(f"unknown source mode {mode!r}")
        else:
            target = {"mode": mode}
        self.stop_recording()
        self.target = target
        self.gen += 1

    def set_out_dir(self, path):
        """Change where new sessions are saved (each gets a timestamped subfolder)."""
        if self.session is not None and self.session.state in ("recording", "processing"):
            raise ValueError("wait until the current recording is saved")
        path = (path or "").strip()
        if not path:
            raise ValueError("enter a folder")
        d = Path(path).expanduser()
        if not d.is_absolute():
            d = Path.cwd() / d
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_test"
            probe.touch()
            probe.unlink()
        except OSError as e:
            raise ValueError(f"cannot write to {d}: {e.strerror or e}") from None
        self.sessions_dir = d.resolve()

    def _describe(self, t):
        return {"discover": "searching for Neon Companion…",
                "ip": f"connecting to {t.get('host')}:{t.get('port')}…",
                "replay": f"opening {Path(t.get('path', '')).name}…",
                "none": "no source selected"}[t["mode"]]

    def _show_placeholder(self, msg):
        img = np.full((720, 960, 3), 241, np.uint8)  # matches the GUI's light grey
        size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(img, msg, ((960 - size[0]) // 2, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (118, 118, 118), 2, cv2.LINE_AA)
        self.scene_jpg = self._encode(img)
        c = self.preview_canvas
        self.paper_jpg = self._encode(np.full((c.h, c.w, 3), 241, np.uint8))

    def _disconnect(self):
        if self.source is not None:
            try:
                self.source.close()
            except Exception:
                pass
        self.source = None
        self.fps, self.tags_seen = 0.0, []
        self.raw_trail.clear()
        self.paper_trail.clear()

    def _open_source(self, target, gen):
        """Open the source in a helper thread so a slow or hanging connection
        never blocks switching to another source from the GUI."""
        result = {}

        def work():
            try:
                result["src"] = ReplaySource(Path(target["path"])) if target["mode"] == "replay" \
                    else NeonSource(target.get("host"), target.get("port", 8080),
                                    cancelled=lambda: self.gen != gen)
            except BaseException as e:
                result["err"] = e
            if self.gen != gen and "src" in result:  # abandoned: clean up
                result["src"].close()

        t = threading.Thread(target=work, daemon=True)
        t.start()
        while t.is_alive():
            if self.gen != gen:
                raise Cancelled()
            t.join(0.1)
        if "err" in result:
            raise result["err"]
        if self.gen != gen:
            result["src"].close()
            raise Cancelled()
        return result["src"]

    def _run(self):
        while not self._stop:
            gen, target = self.gen, self.target
            msg = self._describe(target)
            self.status.update(connected=False, device=None, message=msg)
            self._show_placeholder(msg)
            if target["mode"] == "none":
                while self.gen == gen and not self._stop:
                    time.sleep(0.1)
                continue
            try:
                src = self._open_source(target, gen)
                self.source, self.maps = src, None
                self.status.update(connected=True, device=src.name, message="")
                self._loop(gen)
            except Cancelled:
                continue
            except Exception as e:  # keep the GUI alive; retry the connection
                traceback.print_exc()
                err = f"{type(e).__name__}: {e}"
                self.status.update(connected=False, message=err)
                self._show_placeholder(err[:70])
                for _ in range(20):  # retry after 2 s unless the target changes
                    if self.gen != gen:
                        break
                    time.sleep(0.1)
            finally:
                self._disconnect()

    # ---- per frame

    def _loop(self, gen):
        src = self.source
        times = deque(maxlen=30)
        while not self._stop and self.gen == gen:
            item = src.read_frame()
            if item is None:
                continue
            frame, t = item
            times.append(time.time())
            if len(times) > 1:
                self.fps = (len(times) - 1) / max(1e-6, times[-1] - times[0])
            if self.maps is None:
                h, w = frame.shape[:2]
                self.maps = cv2.initUndistortRectifyMap(src.K, src.D, None, src.K, (w, h),
                                                        cv2.CV_32FC1)
            self._process(frame, t)

    def _process(self, frame, t):
        src = self.source
        self.last_frame = frame  # for "Capture result"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = gp.find_tags(self.det, gray, self.ids)
        H = None
        if tags:
            img_pts = gp.undistort(np.concatenate([tags[i] for i in tags]), src.K, src.D)
            paper = np.concatenate([self.layout[i] for i in tags])
            H = gp.fit_homography(img_pts, paper)
        self.tags_seen = sorted(tags)

        gaze = src.drain_gaze(t)
        mapped = []
        if gaze:
            g = np.array([(x, y) for _, x, y, _ in gaze])
            pp = gp.apply_h(H, gp.undistort(g, src.K, src.D)) if H is not None else \
                np.full((len(g), 2), np.nan)
            mapped = [(gt, x, y, px, py, worn)
                      for (gt, x, y, worn), (px, py) in zip(gaze, pp)]
        for gt, x, y, px, py, worn in mapped:
            self.raw_trail.append((gt, x, y))
            if not np.isnan(px):
                self.paper_trail.append((gt, px, py))
        while self.raw_trail and self.raw_trail[0][0] < t - LIVE_TRAIL_SEC:
            self.raw_trail.popleft()
        while self.paper_trail and self.paper_trail[0][0] < t - self.args.paper_trail:
            self.paper_trail.popleft()

        s = self.session
        if s is not None and s.state == "recording":
            self._record(s, frame, t, H, mapped)

        self.frame_id += 1
        self.scene_jpg = self._encode(self._draw_scene(frame, tags, H, gaze[-1] if gaze else None))
        if self.frame_id % 2 == 0:  # top-down preview at half rate
            self.paper_jpg = self._encode(self._draw_paper(frame, H))

    def _record(self, s, frame, t, H, mapped):
        if s.t0 is None:
            s.t0 = t
        for gt, x, y, px, py, worn in mapped:
            if gt >= s.t0:
                s.rows.append((gt - s.t0, x, y, px, py, int(worn)))
        now = t - s.t0
        if now >= s.next_tl:
            if H is not None:
                c = self.out_canvas
                und = cv2.remap(frame, *self.maps, cv2.INTER_LINEAR)
                s.last_view = cv2.warpPerspective(und, c.T @ H, (c.w, c.h))
            if s.last_view is not None:
                w, h = s.tl_size
                s.tl_raw.write(np.ascontiguousarray(s.last_view[:h, :w]))
                s.tl_times.append(now)
            s.next_tl += s.speed / TIMELAPSE_FPS

    # ---- drawing

    @staticmethod
    def _encode(img, q=80):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        return buf.tobytes() if ok else None

    def _paper_to_raw(self, pts):
        """Paper mm -> distorted scene-image pixels (inverse of the mapping)."""
        src = self.source
        und = gp.apply_h(np.linalg.inv(self._H), pts)
        norm = cv2.undistortPoints(und.reshape(-1, 1, 2), src.K, np.zeros(5))  # K^-1
        obj = np.c_[norm.reshape(-1, 2), np.ones(len(pts))]
        raw, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), src.K, src.D)
        return raw.reshape(-1, 2)

    def _draw_scene(self, frame, tags, H, latest):
        img = frame.copy()
        self._H = H
        for c in tags.values():
            cv2.polylines(img, [np.round(c).astype(np.int32)], True, (90, 220, 120), 2, cv2.LINE_AA)
        if H is not None:
            edge = np.array([[0, 0], [self.pw, 0], [self.pw, self.ph], [0, self.ph]], float)
            # sample the edges so the outline follows lens distortion
            ts = np.linspace(0, 1, 20)[:, None]
            outline = np.concatenate([edge[k] * (1 - ts) + edge[(k + 1) % 4] * ts for k in range(4)])
            cv2.polylines(img, [np.round(self._paper_to_raw(outline)).astype(np.int32)], True,
                          (90, 220, 120), 1, cv2.LINE_AA)
        # trail: newest is brightest
        pts = np.array([(x, y) for _, x, y in self.raw_trail])
        if len(pts) > 1:
            n = len(pts)
            for k in range(1, n):
                a = k / n
                col = (int(60 + 195 * a), int(120 + 80 * a), int(255 * a))
                cv2.line(img, tuple(np.round(pts[k - 1]).astype(int)),
                         tuple(np.round(pts[k]).astype(int)), col, 2, cv2.LINE_AA)
        if latest is not None:
            p = (int(round(latest[1])), int(round(latest[2])))
            cv2.circle(img, p, 22, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.circle(img, p, 22, (40, 80, 255), 3, cv2.LINE_AA)
        return cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

    def _draw_paper(self, frame, H):
        c = self.preview_canvas
        if H is None:
            img = np.full((c.h, c.w, 3), 241, np.uint8)
            cv2.putText(img, "paper not found", (c.w // 2 - 100, c.h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (118, 118, 118), 2, cv2.LINE_AA)
        else:
            und = cv2.remap(frame, *self.maps, cv2.INTER_LINEAR)
            img = cv2.warpPerspective(und, c.T @ H, (c.w, c.h))
        s = self.session
        pts = [(px, py) for _, _, _, px, py, _ in s.rows[-6000:]] if s and s.state == "recording" \
            else [(x, y) for _, x, y in self.paper_trail]
        pts = np.array([p for p in pts if not np.isnan(p[0])])
        if len(pts) > 1:
            q = np.round(c.to_px(pts[::2])).astype(np.int32)
            cv2.polylines(img, [q], False, (60, 170, 255), 1, cv2.LINE_AA)
            cv2.circle(img, tuple(q[-1]), 6, (40, 80, 255), 2, cv2.LINE_AA)
        return img

    # ---- controls

    def start_recording(self, speed, on_companion):
        if self.session and self.session.state == "recording":
            return
        if self.source is None:
            raise RuntimeError("not connected")
        if on_companion and self.source.can_record:
            self.source.recording_start()
        s = Session(self.sessions_dir, self.out_canvas, speed)
        s.on_companion = on_companion and self.source.can_record
        self.session = s

    def stop_recording(self):
        s = self.session
        if s is None or s.state != "recording":
            return
        s.state = "processing"
        if getattr(s, "on_companion", False):
            try:
                self.source.recording_stop()
            except Exception as e:
                print("companion stop failed:", e)
        threading.Thread(target=self._finalize, args=(s,), daemon=True).start()

    def capture_sketch(self):
        """Use the current scene frame as the final sketch photo.

        The frame is undistorted first, because the photo alignment assumes a
        distortion-free image. Needs at least one corner tag in view.
        """
        frame = getattr(self, "last_frame", None)
        if self.source is None or frame is None or self.maps is None:
            raise ValueError("no live frame to capture")
        und = cv2.remap(frame, *self.maps, cv2.INTER_LINEAR)
        tags = gp.find_tags(self.det, cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), self.ids)
        if not tags:
            raise ValueError("no corner tags visible in the current frame; "
                             "point the camera at the sheet and try again")
        ok, buf = cv2.imencode(".png", und)
        label = f"Captured frame, {datetime.now():%H:%M:%S} ({len(tags)} of {len(self.ids)} tags)"
        self.set_sketch(buf.tobytes(), "capture.png", label)
        return label

    def set_sketch(self, data: bytes, filename: str, label=None):
        s = self.session or self.last_session
        label = label or filename
        if s is None:
            # no session yet: keep it for the next one
            self.pending_sketch = (data, filename, label)
            return None
        ext = Path(filename).suffix.lower() or ".jpg"
        path = s.dir / f"final_sketch{ext}"
        for old in s.dir.glob("final_sketch.*"):  # a capture may replace an upload
            old.unlink()
        path.write_bytes(data)
        s.sketch, s.sketch_label = path, label
        if s.state == "done":
            threading.Thread(target=self._render_stills, args=(s,), daemon=True).start()
        return path

    # ---- outputs

    def _finalize(self, s):
        try:
            s.tl_raw.close()
            pending = getattr(self, "pending_sketch", None)
            if pending and s.sketch is None:  # photo chosen before recording
                self.pending_sketch = None
                self.set_sketch(*pending)
            rows = np.array(s.rows, float).reshape(-1, 6)
            with open(s.dir / "gaze_on_paper.csv", "w") as f:
                f.write("t_sec,gaze_x_px,gaze_y_px,paper_x_mm,paper_y_mm,worn\n")
                for r in rows:
                    f.write(f"{r[0]:.4f},{r[1]:.2f},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{int(r[5])}\n")
            self._render_stills(s)
            self._render_timelapse(s)
            ok = ~np.isnan(rows[:, 3])
            pw, ph = self.pw, self.ph
            on = ok & (rows[:, 3] >= 0) & (rows[:, 3] <= pw) & (rows[:, 4] >= 0) & (rows[:, 4] <= ph)
            json.dump({"duration_sec": float(rows[-1, 0]) if len(rows) else 0.0,
                       "gaze_samples": len(rows), "gaze_mapped_fraction": float(ok.mean()) if len(rows) else 0,
                       "gaze_on_paper_fraction": float(on.mean()) if len(rows) else 0,
                       "timelapse_speed": s.speed, "paper_mm": [pw, ph], "tag_ids": self.ids,
                       **getattr(s, "trace_stats", {})},
                      open(s.dir / "summary.json", "w"), indent=2)
            s.outputs = list_outputs(s.dir)
            s.state = "done"
        except Exception as e:
            traceback.print_exc()
            s.state, s.error = "error", f"{type(e).__name__}: {e}"
        finally:
            if self.session is s:
                self.session, self.last_session = None, s

    def _gaze_px(self, s):
        rows = np.array(s.rows, float).reshape(-1, 6)
        ok = ~np.isnan(rows[:, 3])
        return rows[ok], self.out_canvas.to_px(rows[ok][:, 3:5])

    def _render_stills(self, s):
        c = self.out_canvas
        bg = None
        if s.sketch is not None:
            try:
                bg = gp.photo_background(s.sketch, self.ids, self.layout, c)
                cv2.imwrite(str(s.dir / "final_sketch_topdown.png"), bg)
                s.error = None
            except SystemExit as e:  # e.g. tags not found in the photo
                s.error = f"sketch photo: {e}"
        if bg is not None:
            pass
        elif s.last_view is not None:
            bg = s.last_view
            cv2.imwrite(str(s.dir / "final_frame.png"), bg)
        else:
            bg = np.full((c.h, c.w, 3), 255, np.uint8)
        all_rows = np.array(s.rows, float).reshape(-1, 6)
        rows, px = self._gaze_px(s)
        if len(rows) < 2:
            return
        # fine -> coarse trace views; NaN rows (paper not found) break fixations
        s.trace_stats = gp.render_trace_overlays(
            bg, all_rows[:, 0], all_rows[:, 3:5], c, s.dir,
            max_dispersion=self.args.fix_dispersion, cluster_radius=self.args.cluster_radius,
            paper_size=(self.pw, self.ph), layout=self.layout)
        s.trace_stats.update(gp.render_heatmaps(
            bg, all_rows[:, 0], all_rows[:, 3:5], c, s.dir, self.args.sigma,
            paper_size=(self.pw, self.ph), layout=self.layout))
        s.outputs = list_outputs(s.dir)

    def _render_timelapse(self, s):
        rows, px = self._gaze_px(s)
        if not s.tl_times:
            return
        cap = cv2.VideoCapture(str(s.dir / "timelapse_clean.mp4"))
        w, h = s.tl_size
        out = gp.VideoOut(s.dir / "timelapse.mp4", w, h, TIMELAPSE_FPS)
        comp = gp.TimelapseComposer(self.out_canvas.w, self.out_canvas.h,
                                    px[::10], rows[::10, 0], s.speed)
        for now in s.tl_times:
            ok, view = cap.read()
            if not ok:
                break
            full = np.zeros((self.out_canvas.h, self.out_canvas.w, 3), np.uint8)
            full[:h, :w] = view
            out.write(np.ascontiguousarray(comp.frame(full, now)[:h, :w]))
        out.close()
        s.outputs = list_outputs(s.dir)

    # ---- status

    def snapshot(self):
        s = self.session or self.last_session
        st = dict(self.status)
        st.update(fps=round(self.fps, 1), tags_seen=self.tags_seen, tag_ids=self.ids,
                  target=self.target, out_dir=str(self.sessions_dir),
                  can_record_on_companion=bool(self.source and self.source.can_record),
                  pending_sketch=(getattr(self, "pending_sketch", None) or (None, None, None))[2])
        if self.source is not None:
            st.update(self.source.info())
        if s is not None:
            st["session"] = {"name": s.dir.name, "path": str(s.dir.resolve()), "state": s.state,
                             "elapsed": s.elapsed() if s.state == "recording" else
                             (s.rows[-1][0] if s.rows else 0),
                             "samples": len(s.rows), "speed": s.speed, "error": s.error,
                             "sketch": getattr(s, "sketch_label", None) if s.sketch else None,
                             "outputs": s.outputs}
        return st


# ----------------------------------------------------------------------- web

def make_app(engine: Engine):
    app = web.Application(client_max_size=64 * 1024 ** 2)

    async def index(_):
        return web.FileResponse(UI_DIR / "index.html")

    async def mjpeg(request):
        attr = "scene_jpg" if request.match_info["which"] == "scene" else "paper_jpg"
        resp = web.StreamResponse(headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-cache"})
        await resp.prepare(request)
        last = None
        try:
            while True:
                jpg = getattr(engine, attr)
                if jpg is not None and jpg is not last:
                    last = jpg
                    await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                await asyncio.sleep(1 / 40)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    async def status(_):
        return web.json_response(engine.snapshot())

    async def record(request):
        body = await request.json()
        try:
            if body.get("action") == "start":
                engine.start_recording(int(body.get("speed", 5)), bool(body.get("companion")))
            else:
                engine.stop_recording()
        except Exception as e:
            return web.json_response({"error": f"{e}"}, status=400)
        return web.json_response(engine.snapshot())

    async def clear(_):
        engine.raw_trail.clear()
        engine.paper_trail.clear()
        return web.json_response({})

    async def sketch(request):
        reader = await request.multipart()
        part = await reader.next()
        data = await part.read()
        try:
            engine.set_sketch(bytes(data), part.filename or "sketch.jpg")
        except SystemExit as e:  # photo_background reports problems this way
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response(engine.snapshot())

    async def capture(_):
        try:
            label = engine.capture_sketch()
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"label": label, **engine.snapshot()})

    async def source(request):
        try:
            engine.set_target(await request.json())
        except (ValueError, TypeError) as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response(engine.snapshot())

    async def recordings(_):
        """Recording folders under the working directory, newest first."""
        root = Path.cwd()
        found = [p for p in [root, *root.glob("*"), *root.glob("*/*")]
                 if p.is_dir() and is_recording_dir(p)]
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return web.json_response([{"name": str(p.relative_to(root)) if p != root else p.name,
                                   "path": str(p.resolve())} for p in found])

    async def browse(request):
        """Native folder picker on the machine running this server."""
        body = await request.json() if request.can_read_body else {}
        purpose = body.get("purpose", "recording")
        prompt = "Choose where to save sessions" if purpose == "output" \
            else "Choose a Neon recording folder"
        if sys.platform == "darwin" and shutil.which("osascript"):
            cmd = ["osascript", "-e", f'POSIX path of (choose folder with prompt "{prompt}")']
        else:
            cmd = [sys.executable, "-c",
                   "import sys, tkinter as t, tkinter.filedialog as f; r=t.Tk(); r.withdraw(); "
                   "r.attributes('-topmost', True); print(f.askdirectory(title=sys.argv[1]))", prompt]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.DEVNULL)
        out, _ = await proc.communicate()
        path = out.decode().strip()
        if proc.returncode != 0 or not path:
            return web.json_response({"path": None})  # cancelled
        path = path.rstrip("/") or "/"
        return web.json_response({"path": path, "valid": is_recording_dir(Path(path))})

    async def out_dir(request):
        try:
            engine.set_out_dir((await request.json()).get("path"))
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response(engine.snapshot())

    async def session_file(request):
        # only files of the current/last session, wherever it was saved
        s = next((x for x in (engine.session, engine.last_session)
                  if x is not None and x.dir.name == request.match_info["name"]), None)
        if s is None:
            raise web.HTTPNotFound()
        path = (s.dir / request.match_info["file"]).resolve()
        if path.parent != s.dir.resolve() or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    app.router.add_get("/", index)
    app.router.add_get("/stream/{which}.mjpg", mjpeg)
    app.router.add_get("/api/status", status)
    app.router.add_post("/api/record", record)
    app.router.add_post("/api/clear", clear)
    app.router.add_post("/api/sketch", sketch)
    app.router.add_post("/api/source", source)
    app.router.add_post("/api/capture", capture)
    app.router.add_get("/api/recordings", recordings)
    app.router.add_post("/api/browse", browse)
    app.router.add_post("/api/outdir", out_dir)
    app.router.add_get("/sessions/{name}/{file}", session_file)
    app.router.add_static("/ui/", UI_DIR)
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", help="initial source: Companion phone IP (default: auto-discover)")
    ap.add_argument("--port", type=int, default=8080, help="Companion app port")
    ap.add_argument("--replay", help="initial source: replay a recording folder")
    ap.add_argument("--out", type=Path, default=Path("sessions"),
                    help="initial folder for recorded sessions (changeable in the GUI)")
    ap.add_argument("--gui-port", type=int, default=8000)
    gp.add_layout_args(ap)
    gp.add_trace_args(ap)
    ap.add_argument("--pad", type=float, default=20, help="mm around the paper in outputs")
    ap.add_argument("--ppm", type=float, default=4, help="output pixels per mm")
    ap.add_argument("--sigma", type=float, default=5, help="heatmap blur, mm")
    ap.add_argument("--paper-trail", type=float, default=10,
                    help="seconds of trace in the live paper view when not recording")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    engine = Engine(args)
    engine.start()
    print(f"GUI: http://localhost:{args.gui_port}")
    web.run_app(make_app(engine), host="127.0.0.1", port=args.gui_port, print=None)


if __name__ == "__main__":
    main()
