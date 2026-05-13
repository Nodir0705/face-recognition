"""Recognition daemon — runs continuously, watches the camera, logs events.

Run directly for testing:
    python src/recognize.py

Or via systemd (see scripts/attendance.service).
"""

import sys
import time
import logging
import signal
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import AttendanceDB

# cv2, FaceEngine and LivenessTracker are imported lazily inside main() so
# this module can be imported by tests that only exercise the pure-Python
# helpers (decide_event_type, _today_start, Tracker).


log = logging.getLogger("attendance")


def load_config():
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def open_camera(cfg):
    """Return a callable `read() -> (ok, bgr_frame)`."""
    import cv2
    w = cfg["camera"]["width"]
    h = cfg["camera"]["height"]
    fr = cfg["camera"]["framerate"]
    try:
        from picamera2 import Picamera2
        picam = Picamera2()
        config = picam.create_video_configuration(
            main={"format": "RGB888", "size": (w, h)},
            controls={"FrameRate": fr},
        )
        picam.configure(config)
        picam.start()
        time.sleep(1.0)

        def read():
            rgb = picam.capture_array()
            return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        def close():
            picam.stop()

        return read, close
    except ImportError:
        log.warning("picamera2 unavailable, falling back to /dev/video0")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        def read():
            return cap.read()

        def close():
            cap.release()

        return read, close


def decide_event_type(db: AttendanceDB, emp_id: str, cfg: dict) -> str | None:
    """Return 'IN', 'OUT', or None if we should skip (cooldown)."""
    last_type, last_ts = db.last_event(emp_id)
    now = int(time.time())

    if last_ts and (now - last_ts) < cfg["attendance"]["cooldown_sec"]:
        log.debug(f"cooldown active for {emp_id} ({now - last_ts}s ago)")
        return None

    if cfg["attendance"]["toggle_mode"]:
        # First event of "the day" = IN, otherwise toggle from last.
        today_start = _today_start(cfg["attendance"]["day_start_hour"])
        if not last_ts or last_ts < today_start:
            return "IN"
        return "OUT" if last_type == "IN" else "IN"
    else:
        # Day-half mode: morning = IN, afternoon = OUT
        hour = datetime.now().hour
        return "IN" if hour < 12 else "OUT"


def _today_start(start_hour: int) -> int:
    now = datetime.now()
    today = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    if now.hour < start_hour:
        today = today.replace(day=now.day - 1)
    return int(today.timestamp())


class Tracker:
    """Very simple IoU-based tracker so we can debounce per-person, not per-frame."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 8):
        self.iou_th = iou_threshold
        self.max_missed = max_missed
        self.tracks: dict[int, dict] = {}
        self._next_id = 0

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    def update(self, detections):
        """Assign track IDs. Returns list of (track_id, detection) pairs."""
        assigned = []
        used = set()
        for det in detections:
            best_id, best_iou = None, 0.0
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                i = self._iou(det.bbox, tr["bbox"])
                if i > best_iou and i > self.iou_th:
                    best_iou, best_id = i, tid
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self.tracks[best_id] = {
                    "bbox": det.bbox, "missed": 0,
                    "match_history": deque(maxlen=10),
                }
            else:
                self.tracks[best_id]["bbox"] = det.bbox
                self.tracks[best_id]["missed"] = 0
            used.add(best_id)
            assigned.append((best_id, det))

        # Increment "missed" for tracks not seen this frame
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]

        return assigned

    def history(self, tid):
        return self.tracks.get(tid, {}).get("match_history")


def show_greeting(name: str, event: str, lang: str = "ko"):
    """Quick stub — print to console + flash an overlay window if display attached."""
    if lang == "ko":
        word = "출근" if event == "IN" else "퇴근"
        msg = f"{name}님, {word}하셨습니다 ✓"
    else:
        msg = f"{name}, {'CHECKED IN' if event == 'IN' else 'CHECKED OUT'} ✓"
    log.info(f"GREETING: {msg}")
    # On a touchscreen-attached Pi you'd render this to a fullscreen window.
    # Keeping it minimal for now.


def main():
    import cv2  # noqa: F401  (used by open_camera and the loop)
    from face_engine import FaceEngine
    from liveness import LivenessTracker

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = load_config()
    project_root = Path(__file__).resolve().parent.parent

    db = AttendanceDB(str(project_root / cfg["paths"]["db"]))
    engine = FaceEngine(
        model_pack=cfg["recognition"]["model_pack"],
        det_size=tuple(cfg["recognition"]["det_size"]),
    )
    liveness = LivenessTracker(
        blink_timeout_sec=cfg["liveness"]["blink_timeout_sec"],
        min_motion_px=cfg["liveness"]["min_motion_px"],
    )
    tracker = Tracker()

    # Load gallery once; refresh periodically.
    emp_ids, names, gallery = db.load_all_embeddings()
    log.info(f"Loaded {len(set(emp_ids))} employees ({gallery.shape[0]} embeddings)")
    last_reload = time.time()
    RELOAD_EVERY = 60  # seconds — picks up new enrollments

    read, close = open_camera(cfg)
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    min_face = cfg["recognition"]["min_face_px"]
    threshold = cfg["recognition"]["match_threshold"]
    needed_consecutive = cfg["recognition"]["consecutive_frames"]
    liveness_on = cfg["liveness"]["enabled"]

    try:
        while running:
            ok, frame = read()
            if not ok:
                time.sleep(0.1)
                continue

            # Periodic gallery refresh
            if time.time() - last_reload > RELOAD_EVERY:
                emp_ids, names, gallery = db.load_all_embeddings()
                last_reload = time.time()
                log.debug(f"Reloaded gallery: {gallery.shape[0]} embeddings")

            faces = engine.detect(frame)
            # Filter tiny faces
            faces = [f for f in faces
                     if (f.bbox[2] - f.bbox[0]) >= min_face]

            assigned = tracker.update(faces)

            for tid, det in assigned:
                idx, sim = engine.match(det.embedding, gallery, threshold)
                hist = tracker.history(tid)
                if hist is None:
                    continue

                if idx == -1:
                    hist.append(None)
                    continue

                emp_id = emp_ids[idx]
                hist.append((emp_id, sim))

                # Need N consecutive matches *to the same person*
                recent = list(hist)[-needed_consecutive:]
                if len(recent) < needed_consecutive:
                    continue
                if any(r is None for r in recent):
                    continue
                if len(set(r[0] for r in recent)) != 1:
                    continue  # not stable on one identity

                # Liveness gate
                if liveness_on:
                    status = liveness.update(
                        f"t{tid}", frame, det.bbox,
                        getattr(det, "landmarks_106", None),
                    )
                    if not liveness.is_live(f"t{tid}"):
                        if status["timed_out"]:
                            log.warning(f"liveness failed for {emp_id} (track {tid})")
                            liveness.reset(f"t{tid}")
                            hist.clear()
                        continue

                # Decide and log
                event = decide_event_type(db, emp_id, cfg)
                if event is None:
                    continue
                avg_conf = float(np.mean([r[1] for r in recent]))
                row_id = db.log_event(emp_id, event, avg_conf)
                log.info(f"[#{row_id}] {emp_id} {names[idx]} → {event} "
                         f"(sim={avg_conf:.3f})")
                show_greeting(names[idx], event, cfg["display"]["language"])
                hist.clear()
                liveness.reset(f"t{tid}")
    finally:
        close()
        log.info("recognition daemon stopped")


if __name__ == "__main__":
    main()
