"""Combined web server: enrollment wizard + recognition overlay + admin.

This is the single Flask app that owns the camera. It runs three things:

  1. Background recognition thread — reads frames, matches faces, logs events.
     Exposes "what's on screen right now" state so the overlay can draw
     a green tick + name.

  2. Enrollment endpoints — guided capture, pose detection, sample storage.

  3. Admin endpoints — employee list, logs, manual entry.

Routes:
  GET  /                  Admin dashboard
  GET  /enroll            Enrollment wizard (the iPhone-Face-ID-style flow)
  GET  /kiosk             Fullscreen recognition view with green-tick overlay
  GET  /api/stream        MJPEG live preview (raw, no overlay)
  GET  /api/kiosk_stream  MJPEG with recognition overlay
  GET  /api/enroll_stream MJPEG with the enrollment oval guide
  POST /api/enroll/start  Begin a new enrollment session
  POST /api/enroll/capture Capture a sample for a specific target pose
  POST /api/enroll/finish Finalize, compute embeddings, save
  GET  /api/enroll/status Polling endpoint — current pose, quality, instruction
  GET  /api/state         Recognition state (who's recognized right now)
"""

import sys
import os
import io
import csv
import time
import uuid
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
from collections import deque

import numpy as np
import cv2
import yaml
from flask import (Flask, request, jsonify, redirect, url_for, abort,
                   render_template, Response, send_from_directory)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db import AttendanceDB
from src.face_engine import FaceEngine
from src.camera import CameraSource
from src.pose import Pose, evaluate_face, laplacian_sharpness  # noqa: F401
from src import occlusion
from src.sheets import SheetsSync, parse_sheet_input


# ---------- Setup ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("web")

# Dedicated enrollment trace log — every state transition + throttled per-poll
# pose snapshot. Written to /tmp/enrollment.log so we can `tail -f` and grep
# during debugging without drowning in the main app log.
ENROLL_LOG = logging.getLogger("enroll.trace")
ENROLL_LOG.propagate = False
ENROLL_LOG.setLevel(logging.DEBUG)
_enroll_fh = logging.FileHandler("/tmp/enrollment.log", mode="a")
_enroll_fh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-5s %(message)s",
                                            datefmt="%Y-%m-%d %H:%M:%S"))
ENROLL_LOG.addHandler(_enroll_fh)
ENROLL_LOG.info("=" * 60)
ENROLL_LOG.info("attendance app started — enrollment trace log opened")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
with open(PROJECT_ROOT / "config" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

DB = AttendanceDB(str(PROJECT_ROOT / CFG["paths"]["db"]))

def _load_engine():
    """Pick the recognition backend at startup based on config."""
    backend = CFG["recognition"].get("backend", "python")
    if backend == "hailo":
        log.info("loading Hailo face engine…")
        # Lazy import — keeps Python-only installs free of HailoRT dependency
        sys.path.insert(0, str(PROJECT_ROOT / "hailo"))
        from engine_adapter import HailoFaceEngine
        det_path = str(PROJECT_ROOT / CFG["recognition"]["hailo_det_hef"])
        rec_path = str(PROJECT_ROOT / CFG["recognition"]["hailo_rec_hef"])
        return HailoFaceEngine(det_hef=det_path, rec_hef=rec_path)
    log.info("loading Python face engine (InsightFace + ONNXRuntime CPU)…")
    return FaceEngine(
        model_pack=CFG["recognition"]["model_pack"],
        det_size=tuple(CFG["recognition"]["det_size"]),
    )


ENGINE = _load_engine()

log.info("starting Google Sheets sync…")
SHEETS = SheetsSync(
    db=DB,
    credentials_path=PROJECT_ROOT / CFG["google_sheets"]["credentials_path"],
    sync_interval_sec=int(CFG["google_sheets"].get("sync_interval_sec", 30)),
    batch_limit=int(CFG["google_sheets"].get("max_retries", 50)) and 50,
)
# One-time migration: if config.yaml has a placeholder spreadsheet_id and
# settings table is empty, leave it empty so the UI prompts for it. If it has
# a real ID and no setting yet, copy it across so the existing config keeps
# working without admin re-entering.
_legacy_sid = (CFG.get("google_sheets", {}).get("spreadsheet_id") or "").strip()
if _legacy_sid and _legacy_sid != "PASTE_YOUR_SPREADSHEET_ID_HERE" \
        and not DB.get_setting("sheets.spreadsheet_id"):
    SHEETS.configure(_legacy_sid,
                      worksheet=CFG["google_sheets"].get("worksheet", "Attendance"))
    log.info("migrated spreadsheet_id from config.yaml into settings table")
SHEETS.start_background()

log.info("starting camera…")
CAMERA = CameraSource(
    width=CFG["camera"]["width"],
    height=CFG["camera"]["height"],
    framerate=CFG["camera"]["framerate"],
)
CAMERA.start()


# ---------- Shared state ----------

class RecognitionState:
    """What the recognition thread currently sees. Read by the overlay."""
    def __init__(self):
        self.lock = threading.Lock()
        self.faces = []  # list of {bbox, name, emp_id, sim, recognized, just_logged_at}

    def update(self, faces):
        with self.lock:
            self.faces = faces

    def snapshot(self):
        with self.lock:
            return list(self.faces)


STATE = RecognitionState()


class EnrollmentSession:
    """4-step sweep enrollment with live progress feedback.

    Three phases:
      PHASE_FIT   — wait for face to fit the oval. Captures a centered
                    embedding for free at the moment the user is well-positioned.
      PHASE_SWEEP — guided sequence: turn LEFT, turn RIGHT, tilt UP, tilt DOWN.
                    Each step shows a live progress bar (yaw/pitch value as %
                    of threshold) so the user can SEE they're getting closer.
                    Threshold low + forgiving (8 in geometric_pose units) so
                    a small head turn triggers. 5-frame smoothing kills jitter.
                    Per-step timeout (STEP_TIMEOUT_SEC) auto-captures whatever
                    embedding we have and advances — never gets stuck.
      PHASE_DONE  — frontend triggers /api/enroll/finish, which adds
                    horizontal-flip augmented embeddings (5 captured + 5 mirrored
                    = 10 total) for additional pose variation.

    Convention (geometric_pose, no mirror):
      yaw  positive = user's nose pointing camera-right = HEAD TURNED LEFT
      pitch positive = nose pointing up = HEAD TILTED UP
      So "Turn LEFT" means yaw must rise above +threshold.
    """
    # (label, axis, sign, threshold)
    # Thresholds in geometric_pose's "approximate degrees" scale.
    # Tuned LOW (5/4) so a natural ~15° head turn at kiosk distance triggers.
    # Anyone struggling to hit these values tells us it's a scale/sign bug
    # rather than the user not turning enough.
    SWEEP_SEQUENCE = [
        ("left",  "yaw",   +1, 18.0),
        ("right", "yaw",   -1, 18.0),
        ("up",    "pitch", +1, 13.0),
        ("down",  "pitch", -1, 13.0),
    ]
    SMOOTH_WINDOW    = 3    # smaller window = faster response to actual turns
    HOLD_FRAMES      = 1
    STEP_TIMEOUT_SEC = 8.0
    SHARPNESS_MIN    = 40.0
    DET_SCORE_MIN    = 0.60

    PHASE_FIT   = "fit"
    PHASE_SWEEP = "sweep"
    PHASE_DONE  = "done"

    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.emp_id = ""
        self.name = ""
        self.department = ""
        self.email = ""
        self.session_id = ""
        self.started_at = 0.0

        self.phase = self.PHASE_FIT
        self.step_idx = 0
        self.center_embedding: np.ndarray | None = None
        self.center_crop: np.ndarray | None = None
        self.sweep_embeddings: list[np.ndarray] = []
        self.sweep_crops: list[np.ndarray] = []

        # Pose smoothing
        self._yaw_hist: deque = deque(maxlen=self.SMOOTH_WINDOW)
        self._pitch_hist: deque = deque(maxlen=self.SMOOTH_WINDOW)
        self._step_started_at = 0.0
        # "Armed" gate: each new step refuses to capture until the user
        # returns near-neutral. Without this, residual pose from the previous
        # step (e.g. still-extreme-left after a strong left turn) gets stolen
        # by the next step ("right") because the inverted sign makes it look
        # like a satisfying value.
        self._step_armed = False
        # Best frame this step in case we need to fall back at timeout
        self._best_value = 0.0
        self._best_embedding: np.ndarray | None = None
        self._best_crop: np.ndarray | None = None

        self.last_status: dict = {}
        # Last detection result, cached for the MJPEG overlay to draw debug
        # geometry over the live frame. Only the overlay reads it; status
        # endpoint writes it. Threading: a stale read is fine — at worst we
        # draw landmarks one frame behind.
        self.last_detection: dict | None = None

    # ----- lifecycle -----

    def start(self, emp_id, name, dept, email):
        with self.lock:
            self.active = True
            self.emp_id = emp_id.strip()
            self.name = name.strip()
            self.department = dept.strip()
            self.email = email.strip()
            self.session_id = uuid.uuid4().hex
            self.started_at = time.time()
            self.phase = self.PHASE_FIT
            self.step_idx = 0
            self.center_embedding = None
            self.center_crop = None
            self.sweep_embeddings = []
            self.sweep_crops = []
            self._yaw_hist.clear()
            self._pitch_hist.clear()
            self._step_started_at = 0.0
            self._step_armed = False
            self._best_value = 0.0
            self._best_embedding = None
            self._best_crop = None

    def reset(self):
        with self.lock:
            self.active = False
            self.session_id = ""
            self.phase = self.PHASE_FIT
            self.step_idx = 0
            self.center_embedding = None
            self.center_crop = None
            self.sweep_embeddings = []
            self.sweep_crops = []
            self._yaw_hist.clear()
            self._pitch_hist.clear()
            self._step_started_at = 0.0
            self._step_armed = False
            self._best_value = 0.0
            self._best_embedding = None
            self._best_crop = None

    def is_complete(self) -> bool:
        with self.lock:
            return self.phase == self.PHASE_DONE

    # ----- phase transitions -----

    def fit_ok(self, embedding: np.ndarray | None = None,
                aligned_crop: np.ndarray | None = None):
        """Promote FIT → SWEEP. Saves the centered embedding + crop as a
        free first sample (no head turn required). The first step starts
        ARMED since FIT just verified a near-neutral pose."""
        with self.lock:
            if self.phase == self.PHASE_FIT:
                self.phase = self.PHASE_SWEEP
                if embedding is not None:
                    self.center_embedding = embedding.astype(np.float32)
                if aligned_crop is not None:
                    self.center_crop = aligned_crop.copy()
                self._step_started_at = time.time()
                self._yaw_hist.clear()
                self._pitch_hist.clear()
                self._step_armed = True
                self._best_value = 0.0
                self._best_embedding = None
                self._best_crop = None

    def update_sweep(self, yaw: float, pitch: float, embedding: np.ndarray,
                     aligned_crop: np.ndarray | None = None) -> dict:
        """Push one frame's pose into the smoother + check the current
        sweep step. Returns a status dict with: smoothed pose, current step
        info, live progress 0..100 %, hold counter, and whether this call
        captured + advanced.

        Per-step timeout: if STEP_TIMEOUT_SEC elapses, we capture the BEST
        pose value we saw during the step (fallback to the current frame if
        we never saw anything good) and advance. Never blocks.
        """
        out = {
            "smoothed_yaw": 0.0, "smoothed_pitch": 0.0,
            "progress": 0.0, "captured": False, "advanced": False,
            "current_step": None, "step_value": 0.0, "step_target": 0.0,
            "timeout_sec_left": 0.0,
        }
        with self.lock:
            if self.phase != self.PHASE_SWEEP \
                    or self.step_idx >= len(self.SWEEP_SEQUENCE):
                return out

            self._yaw_hist.append(float(yaw))
            self._pitch_hist.append(float(pitch))
            sm_yaw = sum(self._yaw_hist) / len(self._yaw_hist)
            sm_pitch = sum(self._pitch_hist) / len(self._pitch_hist)

            label, axis, sign, threshold = self.SWEEP_SEQUENCE[self.step_idx]
            value = sm_yaw if axis == "yaw" else sm_pitch
            signed = value * sign  # always positive when we're in the right direction
            progress = max(0.0, min(1.0, signed / threshold)) * 100.0

            # ARM gate — once the smoothed value is near neutral (signed below
            # 30% of threshold), the step becomes "armed" and a subsequent
            # threshold crossing will fire a capture. Without this, residual
            # pose from the previous step (e.g. still extreme-left after the
            # LEFT step) could trigger the next step's threshold instantly
            # via inverted sign.
            arm_thresh = threshold * 0.3
            if not self._step_armed:
                if abs(value) < arm_thresh:
                    self._step_armed = True

            # Track the BEST frame this step sees (for timeout fallback) —
            # only count frames after we're armed, so residuals don't pollute.
            if self._step_armed and signed > self._best_value:
                self._best_value = signed
                self._best_embedding = embedding.astype(np.float32)
                if aligned_crop is not None:
                    self._best_crop = aligned_crop.copy()

            captured = False
            advanced = False
            timed_out = (time.time() - self._step_started_at) > self.STEP_TIMEOUT_SEC

            if self._step_armed and signed >= threshold:
                # Threshold met (and we're armed) — capture this exact frame
                self.sweep_embeddings.append(embedding.astype(np.float32))
                if aligned_crop is not None:
                    self.sweep_crops.append(aligned_crop.copy())
                captured = True
                advanced = True
            elif timed_out and self._best_embedding is not None \
                    and self._best_value >= threshold * 0.5:
                # Timed out with a decent best frame — keep it (≥50% of target).
                # If the best is weaker than that, the embedding would be
                # near-frontal and useless; better to fail than ship junk.
                self.sweep_embeddings.append(self._best_embedding)
                if self._best_crop is not None:
                    self.sweep_crops.append(self._best_crop)
                captured = True
                advanced = True

            if advanced:
                self.step_idx += 1
                self._yaw_hist.clear()
                self._pitch_hist.clear()
                self._step_started_at = time.time()
                self._step_armed = False
                self._best_value = 0.0
                self._best_embedding = None
                self._best_crop = None
                if self.step_idx >= len(self.SWEEP_SEQUENCE):
                    self.phase = self.PHASE_DONE

            # Determine "what we did" for accurate logging — the trigger
            # branch we actually entered, not a reverse-engineered guess.
            capture_reason = ""
            if captured:
                capture_reason = ("threshold" if (signed >= threshold)
                                  else "timeout-fallback")
            out.update({
                "smoothed_yaw":     sm_yaw,
                "smoothed_pitch":   sm_pitch,
                "progress":         progress,
                "captured":         captured,
                "advanced":         advanced,
                "armed":            self._step_armed,
                "current_step":     label,
                "step_value":       round(signed, 1),
                "step_target":      threshold,
                "best_so_far":      round(self._best_value, 1),
                "capture_reason":   capture_reason,
                "timeout_sec_left": max(0.0, self.STEP_TIMEOUT_SEC -
                                              (time.time() - self._step_started_at)),
            })
            return out

    def all_embeddings(self) -> np.ndarray:
        with self.lock:
            embs = []
            if self.center_embedding is not None:
                embs.append(self.center_embedding)
            embs.extend(self.sweep_embeddings)
            if not embs:
                return np.zeros((0, 512), dtype=np.float32)
            return np.stack(embs).astype(np.float32)

    def all_crops(self) -> list[np.ndarray]:
        with self.lock:
            crops = []
            if self.center_crop is not None:
                crops.append(self.center_crop)
            crops.extend(self.sweep_crops)
            return crops

    def progress(self) -> dict:
        with self.lock:
            steps = []
            for i, (label, _, _, _) in enumerate(self.SWEEP_SEQUENCE):
                if i < self.step_idx:
                    state = "done"
                elif i == self.step_idx and self.phase == self.PHASE_SWEEP:
                    state = "active"
                else:
                    state = "pending"
                steps.append({"label": label, "state": state})
            current = (self.SWEEP_SEQUENCE[self.step_idx][0]
                       if self.phase == self.PHASE_SWEEP
                          and self.step_idx < len(self.SWEEP_SEQUENCE)
                       else None)
            return {
                "session_id":   self.session_id,
                "active":       self.active,
                "emp_id":       self.emp_id,
                "name":         self.name,
                "phase":        self.phase,
                "step_idx":     self.step_idx,
                "total_steps":  len(self.SWEEP_SEQUENCE),
                "current_step": current,
                "steps":        steps,
                "complete":     self.phase == self.PHASE_DONE,
                "last_status":  self.last_status,
            }


SESSION = EnrollmentSession()


# ---------- Recognition background thread ----------

def recognition_loop():
    """Continuously process frames for recognition. Pauses during enrollment."""
    emp_ids, names, gallery = DB.load_all_embeddings()
    log.info(f"recognition: loaded {len(set(emp_ids))} employees")
    last_reload = time.time()
    RELOAD_EVERY = 60

    threshold = CFG["recognition"]["match_threshold"]
    consec = CFG["recognition"]["consecutive_frames"]
    min_face = CFG["recognition"]["min_face_px"]
    cooldown = CFG["attendance"]["cooldown_sec"]

    # Per-bbox-tracking debounce (very simple: by approximate position)
    history: dict[str, deque] = {}

    def bbox_key(bbox):
        x1, y1, x2, y2 = bbox
        return f"{(x1 // 80)}:{(y1 // 80)}"  # coarse spatial bin

    while True:
        # Pause recognition while someone is being enrolled
        if SESSION.active:
            STATE.update([])
            time.sleep(0.2)
            continue

        frame = CAMERA.latest_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        if time.time() - last_reload > RELOAD_EVERY:
            emp_ids, names, gallery = DB.load_all_embeddings()
            last_reload = time.time()

        faces_seen = ENGINE.detect(frame)
        out = []
        for f in faces_seen:
            x1, y1, x2, y2 = f.bbox
            if (x2 - x1) < min_face:
                continue

            occ = occlusion.detect(frame, f.landmarks, f.bbox)
            key = bbox_key(f.bbox)
            hist = history.setdefault(key, deque(maxlen=10))

            entry = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "recognized": False,
                "name": "",
                "emp_id": "",
                "sim": 0.0,
                "just_logged_at": 0,
                "occluded": occ.any,
                "occlusion_reason": occ.reason,
            }

            # Don't even attempt to match an occluded face — embeddings on a
            # masked / sunglassed face are unreliable and can produce false
            # matches against the wrong enrolled employee.
            if occ.any:
                hist.clear()
                out.append(entry)
                continue

            idx, sim = ENGINE.match(f.embedding, gallery, threshold)
            entry["sim"] = float(sim)

            if idx >= 0:
                emp_id = emp_ids[idx]
                hist.append(emp_id)
                stable = (len(hist) >= consec
                          and len(set(list(hist)[-consec:])) == 1)
                entry["recognized"] = stable
                entry["name"] = names[idx]
                entry["emp_id"] = emp_id

                if stable:
                    # Decide IN/OUT with cooldown
                    last_type, last_ts = DB.last_event(emp_id)
                    now = int(time.time())
                    if not last_ts or (now - last_ts) >= cooldown:
                        if CFG["attendance"]["toggle_mode"]:
                            event = "OUT" if last_type == "IN" else "IN"
                        else:
                            event = "IN" if datetime.now().hour < 12 else "OUT"
                        DB.log_event(emp_id, event, float(sim))
                        entry["just_logged_at"] = now
                        entry["event"] = event
                        log.info(f"recognized {emp_id} {names[idx]} -> {event}")
                        hist.clear()
            else:
                hist.append(None)

            out.append(entry)

        STATE.update(out)
        # Cap recognition at ~12 fps (~83 ms cycle). Hailo can go much faster,
        # but on Pi 5 the GIL is the bottleneck — a short recognition cycle
        # starves the MJPEG generator and the kiosk video gets choppy.
        # 12 fps recognition still feels instant to the user (banner appears
        # within ~250 ms of stable detection thanks to the 200 ms /api/state
        # poll on the kiosk page).
        time.sleep(0.08)


threading.Thread(target=recognition_loop, daemon=True).start()


# ---------- Auth ----------

ADMIN_USER = os.environ.get("ATTENDANCE_ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ATTENDANCE_ADMIN_PASS", "changeme")


def basic_auth(view):
    @wraps(view)
    def wrapper(*a, **kw):
        auth = request.authorization
        if not auth or auth.username != ADMIN_USER or auth.password != ADMIN_PASS:
            return ("Auth required", 401,
                    {"WWW-Authenticate": 'Basic realm="attendance"'})
        return view(*a, **kw)
    return wrapper


# ---------- Flask app ----------

TEMPLATES = Path(__file__).resolve().parent / "templates"
STATIC = Path(__file__).resolve().parent / "static"
app = Flask(__name__, template_folder=str(TEMPLATES), static_folder=str(STATIC))


# ---------- Page routes ----------

@app.route("/")
@basic_auth
def admin_home():
    with DB._conn() as c:
        emps = c.execute(
            """SELECT emp_id, name, department, enrolled_at FROM employees
               WHERE active = 1 ORDER BY name"""
        ).fetchall()
        logs = c.execute(
            """SELECT a.emp_id, e.name, a.event_type, a.timestamp,
                      a.confidence, a.synced
               FROM attendance a JOIN employees e ON e.emp_id = a.emp_id
               ORDER BY a.timestamp DESC LIMIT 50"""
        ).fetchall()
    return render_template(
        "admin.html",
        employees=[dict(r) for r in emps],
        logs=[dict(r) for r in logs],
        fmt_time=lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S"),
        fmt_date=lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d"),
    )


@app.route("/enroll")
@basic_auth
def enroll_page():
    return render_template("enroll.html")


@app.route("/kiosk")
def kiosk_page():
    # Intentionally no auth — this is the public-facing display at the door
    return render_template("kiosk.html",
                           greeting_duration=CFG["display"]["greeting_duration_sec"])


# ---------- Streaming ----------

def _draw_kiosk_overlay(frame: np.ndarray) -> np.ndarray:
    """Draw face boxes + green tick + name under each recognized face."""
    faces = STATE.snapshot()
    h, w = frame.shape[:2]
    for f in faces:
        x1, y1, x2, y2 = f["bbox"]
        if f.get("occluded"):
            color = (40, 130, 230)   # amber-ish in BGR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f.get("occlusion_reason") or "Please face the camera"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            lx = max(0, (x1 + x2) // 2 - tw // 2)
            ly = max(th + 12, y1 - 12)
            cv2.rectangle(frame, (lx - 8, ly - th - 6),
                          (lx + tw + 8, ly + 6), (40, 40, 40), -1)
            cv2.putText(frame, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            continue
        if f["recognized"]:
            color = (99, 199, 89)   # green BGR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            # Green tick circle below the face
            cx = (x1 + x2) // 2
            cy = min(y2 + 50, h - 30)
            cv2.circle(frame, (cx, cy), 22, color, -1)
            # White checkmark
            cv2.line(frame, (cx - 10, cy + 1), (cx - 3, cy + 8), (255, 255, 255), 3)
            cv2.line(frame, (cx - 3, cy + 8), (cx + 12, cy - 8), (255, 255, 255), 3)
            # Name label below the tick
            label = f["name"]
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            lx = cx - tw // 2
            ly = cy + 22 + th + 8
            # background pill
            cv2.rectangle(frame, (lx - 10, ly - th - 6),
                          (lx + tw + 10, ly + 6),
                          (40, 40, 40), -1)
            cv2.putText(frame, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # "CHECK IN" / "CHECK OUT" banner if just logged
            if f.get("just_logged_at") and (time.time() - f["just_logged_at"]) < 3:
                event = f.get("event", "")
                msg = "출근 완료" if event == "IN" else "퇴근 완료"
                if CFG["display"]["language"] == "en":
                    msg = "Checked in" if event == "IN" else "Checked out"
                (bw, bh), _ = cv2.getTextSize(
                    msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                bx = cx - bw // 2
                by = max(y1 - 20, bh + 20)
                cv2.rectangle(frame, (bx - 15, by - bh - 10),
                              (bx + bw + 15, by + 10), color, -1)
                cv2.putText(frame, msg, (bx, by),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
    return frame


def _draw_enrollment_overlay(frame: np.ndarray) -> np.ndarray:
    """Mirrors the frame (selfie-style) so the user's perception of left/right
    matches their physical motion, then dims outside a portrait oval and
    draws live debug geometry on top of the face.

    Important: the mirror is DISPLAY-ONLY. The compute path
    (engine.detect → geometric_pose → SWEEP_SEQUENCE) still runs on the
    unmirrored frame, so yaw/pitch values stay mathematically consistent.
    To draw landmarks correctly on the mirrored display, we mirror their
    x coordinates too: x_mirrored = w - x.
    """
    # Mirror the frame first — selfie-style display
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    rx = int(min(w, h) * 0.22)
    ry = int(min(w, h) * 0.30)

    # Dim outside the oval guide
    mask = np.ones((h, w), dtype=np.uint8) * 255
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 0, -1)
    dark = (frame * 0.45).astype(np.uint8)
    out = np.where(mask[:, :, None] > 0, dark, frame).astype(np.uint8)
    cv2.ellipse(out, (cx, cy), (rx, ry), 0, 0, 360, (240, 240, 240), 2)

    det = SESSION.last_detection
    if not det or det.get("landmarks") is None:
        return out

    kps_raw = det["landmarks"]
    if kps_raw.shape != (5, 2):
        return out

    # Mirror landmark x coords so they line up with the mirrored video
    kps = kps_raw.copy()
    kps[:, 0] = w - kps[:, 0]

    # IMPORTANT: after the mirror, what InsightFace called "L_eye" (the
    # USER's left eye, originally on camera-right side / image-right) now
    # appears on screen-LEFT. We swap the labels so eye-pair drawing/text
    # makes intuitive sense in the mirrored display.
    R_eye, L_eye, nose, R_mouth, L_mouth = (tuple(map(int, p)) for p in kps)
    eye_mid = ((L_eye[0] + R_eye[0]) // 2, (L_eye[1] + R_eye[1]) // 2)
    mouth_mid = ((L_mouth[0] + R_mouth[0]) // 2, (L_mouth[1] + R_mouth[1]) // 2)

    # ----- skeleton -----

    # Eye line — cyan
    cv2.line(out, L_eye, R_eye, (255, 255, 0), 2, cv2.LINE_AA)
    # Mouth line — magenta
    cv2.line(out, L_mouth, R_mouth, (255, 0, 255), 2, cv2.LINE_AA)
    # Vertical reference: eye midpoint → mouth midpoint (gray, dashed-ish)
    cv2.line(out, eye_mid, mouth_mid, (180, 180, 180), 1, cv2.LINE_AA)

    # ----- yaw indicator: horizontal line from eye midpoint at eye_y to nose_x -----
    # Length and direction of this YELLOW line = yaw. Long left or long right
    # means high yaw magnitude. If you turn your head and this line doesn't
    # extend, the landmarks aren't following — i.e., a model issue.
    nose_proj_eye_y = (nose[0], eye_mid[1])
    cv2.line(out, eye_mid, nose_proj_eye_y, (0, 220, 255), 4, cv2.LINE_AA)

    # ----- pitch indicator: vertical line from eye line at eye_mid_x to nose_y -----
    # MAGENTA line. Length = pitch magnitude.
    nose_proj_eye_x = (eye_mid[0], nose[1])
    cv2.line(out, eye_mid, nose_proj_eye_x, (255, 100, 255), 4, cv2.LINE_AA)

    # ----- 5 landmark dots, color-coded -----
    for pt, color in zip(
        [L_eye, R_eye, nose, L_mouth, R_mouth],
        [(80, 255, 80), (80, 255, 80), (0, 165, 255),
         (200, 200, 200), (200, 200, 200)],
    ):
        cv2.circle(out, pt, 4, color, -1, cv2.LINE_AA)
    # Eye midpoint — orange, slightly bigger
    cv2.circle(out, eye_mid, 5, (0, 140, 255), -1, cv2.LINE_AA)

    # ----- text readouts (top-left corner) -----
    yaw   = det.get("smoothed_yaw",   det.get("yaw",   0.0))
    pitch = det.get("smoothed_pitch", det.get("pitch", 0.0))

    # Background pill so text is readable on any video
    pad = 8
    lines = [
        f"yaw  {yaw:+6.1f}",
        f"pitch {pitch:+6.1f}",
    ]
    step = det.get("step")
    if step:
        target = det.get("step_target", 0.0)
        value = det.get("step_value", 0.0)
        progress = det.get("progress", 0.0)
        best = det.get("best_so_far", 0.0)
        armed = det.get("armed", False)
        arm_tag = "ARMED" if armed else "not armed (return to center)"
        lines.append(f"-> {step:5}: {value:+5.1f} / {target:.1f}  ({progress:5.1f}%)")
        lines.append(f"   best so far: {best:+5.1f}   [{arm_tag}]")

    line_h = 24
    box_h = pad * 2 + line_h * len(lines)
    box_w = 380
    cv2.rectangle(out, (10, 10), (10 + box_w, 10 + box_h), (40, 40, 40), -1)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10 + pad, 10 + pad + line_h * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return out


@app.route("/api/kiosk_stream")
def api_kiosk_stream():
    return Response(
        CAMERA.mjpeg_generator(draw_overlay=_draw_kiosk_overlay, fps=30,
                                preview_size=(960, 540)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/enroll_stream")
@basic_auth
def api_enroll_stream():
    return Response(
        CAMERA.mjpeg_generator(draw_overlay=_draw_enrollment_overlay, fps=30,
                                preview_size=(960, 540)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ---------- Enrollment API ----------

@app.route("/api/enroll/start", methods=["POST"])
@basic_auth
def api_enroll_start():
    data = request.get_json(force=True)
    emp_id = (data.get("emp_id") or "").strip()
    name = (data.get("name") or "").strip()
    if not emp_id or not name:
        return jsonify(error="emp_id and name required"), 400
    SESSION.start(emp_id, name,
                  data.get("department", ""),
                  data.get("email", ""))
    log.info(f"enrollment session started: {emp_id} / {name}")
    ENROLL_LOG.info(f"START sid={SESSION.session_id[:8]} emp_id={emp_id} name={name!r}")
    return jsonify(SESSION.progress())


@app.route("/api/enroll/status")
@basic_auth
def api_enroll_status():
    """Polled by the client every ~150ms.

    Drives the whole state machine and auto-captures embeddings on dot
    transitions — there is no separate /capture call from the client now.
    """
    if not SESSION.active:
        return jsonify(active=False)

    frame = CAMERA.latest_frame()
    if frame is None:
        return jsonify(active=True, error="no frame")

    faces = ENGINE.detect(frame)
    prog = SESSION.progress()

    if not faces:
        prog["live"] = {
            "face_count": 0,
            "instruction": "Stand in front of the camera",
            "yaw": 0, "pitch": 0,
        }
        return jsonify(prog)

    # Largest face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    occ = occlusion.detect(frame, face.landmarks, face.bbox)

    # Geometry — does the face fit the on-screen oval?
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2
    fx = (face.bbox[0] + face.bbox[2]) / 2
    fy = (face.bbox[1] + face.bbox[3]) / 2
    fw = face.bbox[2] - face.bbox[0]
    # Tightened centering — was 12% (too loose, allowed clearly off-center
    # faces). 7% means the face center must sit within the inner core of
    # the oval to count as "in".
    centered = (abs(fx - cx) / w < 0.07) and (abs(fy - cy) / h < 0.10)
    big_enough = fw >= 200
    not_too_big = fw <= 360
    fit_ok = centered and big_enough and not_too_big and not occ.any

    # Sharpness gate — Laplacian variance on the face crop only (cheap)
    x1, y1, x2, y2 = face.bbox
    crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
    sharp_value = (laplacian_sharpness(crop) if crop.size > 0 else 0.0)
    sharp = sharp_value >= SESSION.SHARPNESS_MIN

    yaw, pitch, _ = face.pose

    # Cache the latest detection so the MJPEG overlay can draw live debug
    # geometry on top of the video (landmarks, eye line, yaw/pitch indicator).
    SESSION.last_detection = {
        "landmarks": np.asarray(face.landmarks, dtype=np.float32).copy(),
        "bbox":      tuple(int(v) for v in face.bbox),
        "yaw":       float(yaw),
        "pitch":     float(pitch),
    }

    quality = {
        "in_oval":      bool(fit_ok),
        "not_occluded": not occ.any,
        "sharp":        bool(sharp),
        "sharp_value":  round(sharp_value, 1),
        "det_score":    round(float(face.det_score), 3),
    }

    live = {
        "face_count":       1,
        "occluded":         occ.any,
        "occlusion_reason": occ.reason,
        "face_in_oval":     bool(fit_ok),
        "yaw":              round(float(yaw), 1),
        "pitch":            round(float(pitch), 1),
        "quality":          quality,
    }

    STEP_PROMPTS = {
        "left":  "Turn your head LEFT",
        "right": "Turn your head RIGHT",
        "up":    "Tilt your head UP",
        "down":  "Tilt your head DOWN",
    }

    # Pre-aligned crop for capture / augmentation
    try:
        aligned = ENGINE.aligned_crop(frame, face.landmarks)
    except Exception:
        aligned = None

    # PHASE FIT — wait for face to be centered + sized in the oval
    if SESSION.phase == SESSION.PHASE_FIT:
        if occ.any:
            live["instruction"] = occ.reason
        elif not big_enough:
            live["instruction"] = "Move closer"
        elif not not_too_big:
            live["instruction"] = "Move back a little"
        elif not centered:
            live["instruction"] = "Center your face in the oval"
        elif not sharp:
            live["instruction"] = "Hold still — too blurry"
        elif face.det_score < SESSION.DET_SCORE_MIN:
            live["instruction"] = "Adjust your position"
        else:
            SESSION.fit_ok(embedding=face.embedding, aligned_crop=aligned)
            live["instruction"] = STEP_PROMPTS["left"]
            ENROLL_LOG.info(f"FIT_OK -> SWEEP  sid={SESSION.session_id[:8]} "
                            f"yaw={yaw:+.1f} pitch={pitch:+.1f} "
                            f"det_score={face.det_score:.2f} sharp={sharp_value:.0f}")
        prog = SESSION.progress()
        prog["live"] = live
        return jsonify(prog)

    # PHASE SWEEP — guided 4-step sequence with live progress feedback
    if SESSION.phase == SESSION.PHASE_SWEEP:
        if not (big_enough and not_too_big and not occ.any and centered):
            live["instruction"] = (occ.reason if occ.any
                                   else "Keep your face centered in the oval")
            prog = SESSION.progress()
            prog["live"] = live
            return jsonify(prog)

        upd = SESSION.update_sweep(yaw, pitch, face.embedding,
                                     aligned_crop=aligned)
        prog = SESSION.progress()

        # Throttled per-poll trace — every ~300ms while in sweep, plus
        # always-on trace for capture and timeout transitions
        now = time.time()
        last = getattr(SESSION, "_last_trace_at", 0.0)
        if now - last >= 0.3 or upd["captured"]:
            SESSION._last_trace_at = now
            ENROLL_LOG.debug(
                f"SWEEP sid={SESSION.session_id[:8]} step={upd['current_step']:5} "
                f"armed={'Y' if upd.get('armed') else 'n'} "
                f"yaw={yaw:+5.1f} pitch={pitch:+5.1f} "
                f"sm_yaw={upd['smoothed_yaw']:+5.1f} sm_pitch={upd['smoothed_pitch']:+5.1f} "
                f"value={upd['step_value']:+5.1f} target={upd['step_target']:.1f} "
                f"best={upd['best_so_far']:+5.1f} "
                f"prog={upd['progress']:5.1f}% "
                f"timeout_left={upd['timeout_sec_left']:.1f}s "
                f"sharp={sharp_value:.0f} det={face.det_score:.2f}"
            )
        if upd["captured"]:
            ENROLL_LOG.info(
                f"CAPTURED sid={SESSION.session_id[:8]} step={upd['current_step']} "
                f"value={upd['step_value']:+5.1f} target={upd['step_target']:.1f} "
                f"best={upd['best_so_far']:+5.1f} "
                f"reason={upd.get('capture_reason', '?')}"
            )

        live.update({
            "smoothed_yaw":     round(upd["smoothed_yaw"], 1),
            "smoothed_pitch":   round(upd["smoothed_pitch"], 1),
            "progress":         round(upd["progress"], 1),
            "step_value":       upd["step_value"],
            "step_target":      upd["step_target"],
            "best_so_far":      upd["best_so_far"],
            "timeout_sec_left": round(upd["timeout_sec_left"], 1),
        })
        # Stash sweep-state into the detection cache too so the overlay can
        # show the per-step progress alongside the geometry.
        if SESSION.last_detection is not None:
            SESSION.last_detection.update({
                "step":         upd["current_step"],
                "step_value":   upd["step_value"],
                "step_target":  upd["step_target"],
                "best_so_far":  upd["best_so_far"],
                "progress":     upd["progress"],
                "armed":        upd.get("armed", False),
                "smoothed_yaw":   upd["smoothed_yaw"],
                "smoothed_pitch": upd["smoothed_pitch"],
            })

        if upd["captured"]:
            log.info(f"enroll: captured '{upd['current_step']}' for {SESSION.emp_id} "
                     f"(value={upd['step_value']}, target={upd['step_target']})")

        if SESSION.phase == SESSION.PHASE_DONE:
            live["instruction"] = "Done!"
        elif prog["current_step"]:
            live["instruction"] = STEP_PROMPTS[prog["current_step"]]
        else:
            live["instruction"] = "Done!"

        prog["live"] = live
        return jsonify(prog)

    # PHASE DONE — frontend will call /finish next
    live["instruction"] = "Done!"
    prog = SESSION.progress()
    prog["live"] = live
    return jsonify(prog)


@app.route("/api/enroll/capture", methods=["POST"])
@basic_auth
def api_enroll_capture():
    """Manual override — usually the new /status endpoint auto-advances on its
    own, but this lets the client force a capture for the current step from
    whatever pose the user is in right now (useful for debugging)."""
    if not SESSION.active:
        return jsonify(error="no session"), 400
    frame = CAMERA.latest_frame()
    if frame is None:
        return jsonify(error="no frame"), 503
    faces = ENGINE.detect(frame)
    if not faces:
        return jsonify(error="no face"), 422
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    occ = occlusion.detect(frame, face.landmarks, face.bbox)
    if occ.any:
        return jsonify(error=occ.reason), 422
    try:
        aligned = ENGINE.aligned_crop(frame, face.landmarks)
    except Exception:
        aligned = None
    if SESSION.phase == SESSION.PHASE_FIT:
        SESSION.fit_ok(embedding=face.embedding, aligned_crop=aligned)
    elif SESSION.phase == SESSION.PHASE_SWEEP:
        # Force-feed the current pose enough times to satisfy the smoother
        yaw, pitch, _ = face.pose
        for _ in range(SESSION.SMOOTH_WINDOW):
            SESSION.update_sweep(yaw, pitch, face.embedding, aligned_crop=aligned)
    return jsonify(SESSION.progress())


@app.route("/api/enroll/finish", methods=["POST"])
@basic_auth
def api_enroll_finish():
    if not SESSION.is_complete():
        return jsonify(error="incomplete"), 400

    # Liveness: capturing 4 different head poses (left, right, up, down) in
    # sequence is essentially impossible for a static photo, so the sweep
    # itself proves liveness — no separate check needed.

    embs = SESSION.all_embeddings()
    if embs.shape[0] == 0:
        return jsonify(error="no embeddings collected"), 400

    # ---------- Augmentation: horizontal flip ----------
    # ArcFace embeddings of flip(face) are similar but not identical to
    # embeddings of face — typically cos_sim ~0.85-0.95 between the two,
    # which gives us free pose variation. Doubles the gallery size for
    # the cost of one rec model pass per crop.
    augmented_count = 0
    try:
        flipped_embs = []
        for crop in SESSION.all_crops():
            flipped = cv2.flip(crop, 1)
            e = ENGINE.embed_aligned(flipped)
            # Defensive normalize — adapter SHOULD have done it, but be sure
            n = float(np.linalg.norm(e))
            if n > 1e-9:
                flipped_embs.append((e / n).astype(np.float32))
        if flipped_embs:
            embs = np.concatenate([embs, np.stack(flipped_embs)], axis=0)
            augmented_count = len(flipped_embs)
    except Exception as e:
        log.warning(f"enroll: augmentation skipped ({type(e).__name__}: {e})")

    DB.upsert_employee(
        emp_id=SESSION.emp_id, name=SESSION.name,
        embeddings=embs,
        department=SESSION.department, email=SESSION.email,
    )
    info = {
        "emp_id": SESSION.emp_id, "name": SESSION.name,
        "samples": int(embs.shape[0]),
        "augmented": augmented_count,
    }
    log.info(f"enrollment complete: {info}")
    ENROLL_LOG.info(f"FINISH sid={SESSION.session_id[:8]} "
                    f"emp_id={SESSION.emp_id} samples={info['samples']} "
                    f"augmented={info.get('augmented', 0)}")
    SESSION.reset()
    return jsonify(ok=True, **info)


@app.route("/api/enroll/cancel", methods=["POST"])
@basic_auth
def api_enroll_cancel():
    SESSION.reset()
    return jsonify(ok=True)


# ---------- Admin API ----------

@app.route("/api/employees/<emp_id>/deactivate", methods=["POST"])
@basic_auth
def api_deactivate(emp_id):
    DB.deactivate_employee(emp_id)
    return jsonify(ok=True)


@app.route("/api/manual", methods=["POST"])
@basic_auth
def api_manual():
    data = request.get_json(force=True)
    emp_id = data.get("emp_id")
    ev = data.get("event_type", "IN")
    if not emp_id or ev not in ("IN", "OUT"):
        return jsonify(error="bad request"), 400
    DB.log_event(emp_id, ev, confidence=1.0)
    return jsonify(ok=True)


@app.route("/api/state")
def api_state():
    """Used by the kiosk page to know when to flash a greeting banner."""
    return jsonify(faces=STATE.snapshot())


# ---------- Reports & exports ----------

def _parse_date(s: str | None, default: datetime) -> datetime:
    if not s:
        return default
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        abort(400, f"bad date: {s!r}, expected YYYY-MM-DD")


def _date_range_from_query(default_days: int = 30) -> tuple[datetime, datetime]:
    """Read ?from=&to= as YYYY-MM-DD. `to` is inclusive (we add a day internally)."""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    default_from = today - timedelta(days=default_days - 1)
    start = _parse_date(request.args.get("from"), default_from)
    end_inclusive = _parse_date(request.args.get("to"), today)
    end_exclusive = end_inclusive + timedelta(days=1)
    return start, end_exclusive


@app.route("/api/export/csv")
@basic_auth
def api_export_csv():
    """Stream a CSV of attendance events.

    Query params: from=YYYY-MM-DD, to=YYYY-MM-DD (inclusive), emp_id=optional.
    Defaults to last 30 days, all employees.
    """
    start, end = _date_range_from_query(default_days=30)
    emp_id = request.args.get("emp_id") or None

    def generate():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["log_id", "emp_id", "name", "department",
                         "date", "time", "event", "confidence", "synced"])
        yield buf.getvalue(); buf.seek(0); buf.truncate(0)
        for r in DB.events_in_range(int(start.timestamp()),
                                     int(end.timestamp()),
                                     emp_id=emp_id):
            dt = datetime.fromtimestamp(r["timestamp"])
            writer.writerow([
                r["id"], r["emp_id"], r["name"], r["department"] or "",
                dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S"),
                r["event_type"], f"{r['confidence']:.4f}",
                "yes" if r["synced"] else "no",
            ])
            yield buf.getvalue(); buf.seek(0); buf.truncate(0)

    end_inclusive_label = (end - timedelta(days=1)).strftime("%Y%m%d")
    fname = (f"attendance_{emp_id or 'all'}_"
             f"{start.strftime('%Y%m%d')}-{end_inclusive_label}.csv")
    return Response(
        generate(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.route("/api/stats/daily")
@basic_auth
def api_stats_daily():
    """Counts of IN/OUT per day. Used by the dashboard chart."""
    days = int(request.args.get("days", 30))
    days = max(1, min(days, 365))
    end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) \
                       + timedelta(days=1)
    start = end - timedelta(days=days)
    rows = DB.events_per_day(int(start.timestamp()), int(end.timestamp()))
    # Pad missing days with zeros so the bar chart has a continuous x-axis
    by_date = {r["date"]: r for r in rows}
    out = []
    for i in range(days):
        d = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        entry = by_date.get(d, {"date": d, "in_count": 0, "out_count": 0})
        out.append(entry)
    return jsonify(days=out)


@app.route("/api/sheets/status")
@basic_auth
def api_sheets_status():
    return jsonify(SHEETS.status())


@app.route("/api/sheets/configure", methods=["POST"])
@basic_auth
def api_sheets_configure():
    data = request.get_json(force=True) or {}
    raw = data.get("url") or data.get("id") or ""
    worksheet = (data.get("worksheet") or "").strip() or None
    try:
        SHEETS.configure(raw, worksheet=worksheet)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True, status=SHEETS.status())


@app.route("/api/sheets/test", methods=["POST"])
@basic_auth
def api_sheets_test():
    ok, msg = SHEETS.test_connection()
    return jsonify(ok=ok, message=msg, status=SHEETS.status())


@app.route("/api/sheets/upload_credentials", methods=["POST"])
@basic_auth
def api_sheets_upload_credentials():
    """Accept a service-account JSON via multipart upload OR JSON-string body.

    Saves to <project>/config/credentials.json after validation. Replaces any
    existing file. After this call the SheetsSync's cached service email is
    invalidated so /api/sheets/status will show the new one immediately.
    """
    json_text = None
    f = request.files.get("file")
    if f is not None:
        try:
            json_text = f.read().decode("utf-8")
        except UnicodeDecodeError:
            return jsonify(ok=False, error="file is not UTF-8 text"), 400
    else:
        body = request.get_json(silent=True) or {}
        json_text = body.get("json")
    if not json_text:
        return jsonify(ok=False, error="no file or 'json' body provided"), 400

    ok, msg = SHEETS.install_credentials(json_text)
    code = 200 if ok else 400
    return jsonify(ok=ok, message=msg, status=SHEETS.status()), code


@app.route("/api/sheets/sync_now", methods=["POST"])
@basic_auth
def api_sheets_sync_now():
    """Force a sync attempt right now instead of waiting for the next interval."""
    count, err = SHEETS.sync_pending()
    return jsonify(ok=err is None, synced=count, error=err, status=SHEETS.status())


@app.route("/employee/<emp_id>")
@basic_auth
def employee_report(emp_id):
    """Per-employee report page with daily summary + hours chart."""
    emp = DB.get_employee(emp_id)
    if not emp:
        abort(404)
    start, end = _date_range_from_query(default_days=30)
    summary = DB.daily_summary(emp_id,
                                int(start.timestamp()),
                                int(end.timestamp()))

    def fmt_hm(ts):
        return datetime.fromtimestamp(ts).strftime("%H:%M") if ts else "—"

    def fmt_hours(secs):
        if not secs:
            return "—"
        h, rem = divmod(secs, 3600)
        m = rem // 60
        return f"{h}h {m:02d}m"

    total_seconds = sum(d["worked_seconds"] for d in summary)
    days_present = sum(1 for d in summary if d["worked_seconds"] > 0)

    return render_template(
        "employee.html",
        emp=emp,
        emp_enrolled_str=datetime.fromtimestamp(emp["enrolled_at"]).strftime("%Y-%m-%d"),
        summary=summary,
        from_str=start.strftime("%Y-%m-%d"),
        to_str=(end - timedelta(days=1)).strftime("%Y-%m-%d"),
        total_hours_str=fmt_hours(total_seconds),
        days_present=days_present,
        fmt_hm=fmt_hm,
        fmt_hours=fmt_hours,
    )


if __name__ == "__main__":
    # threaded=True is required for concurrent MJPEG streams + API calls
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
