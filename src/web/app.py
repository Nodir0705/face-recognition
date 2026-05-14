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
from src.pose import Pose, evaluate_face, laplacian_sharpness
from src import occlusion


# ---------- Setup ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("web")

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
    """One enrollment in progress. Only one at a time."""
    REQUIRED_POSES = [Pose.CENTER, Pose.LEFT, Pose.RIGHT, Pose.UP, Pose.DOWN]

    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.emp_id = ""
        self.name = ""
        self.department = ""
        self.email = ""
        self.samples: dict[Pose, np.ndarray] = {}   # pose -> embedding
        self.crops: dict[Pose, np.ndarray] = {}     # pose -> face crop (for preview)
        self.session_id = ""
        self.started_at = 0.0
        self.current_target: Pose = Pose.CENTER
        self.last_status: dict = {}

    def start(self, emp_id, name, dept, email):
        with self.lock:
            self.active = True
            self.emp_id = emp_id.strip()
            self.name = name.strip()
            self.department = dept.strip()
            self.email = email.strip()
            self.samples = {}
            self.crops = {}
            self.session_id = uuid.uuid4().hex
            self.started_at = time.time()
            self.current_target = Pose.CENTER

    def add_sample(self, pose: Pose, embedding: np.ndarray, crop: np.ndarray):
        with self.lock:
            self.samples[pose] = embedding
            self.crops[pose] = crop
            # advance to next missing pose
            for p in self.REQUIRED_POSES:
                if p not in self.samples:
                    self.current_target = p
                    break

    def is_complete(self) -> bool:
        with self.lock:
            return all(p in self.samples for p in self.REQUIRED_POSES)

    def progress(self) -> dict:
        with self.lock:
            return {
                "session_id": self.session_id,
                "active": self.active,
                "emp_id": self.emp_id,
                "name": self.name,
                "captured": [p.value for p in self.samples.keys()],
                "remaining": [p.value for p in self.REQUIRED_POSES
                              if p not in self.samples],
                "current_target": self.current_target.value,
                "complete": all(p in self.samples for p in self.REQUIRED_POSES),
                "last_status": self.last_status,
            }

    def reset(self):
        with self.lock:
            self.active = False
            self.samples = {}
            self.crops = {}
            self.session_id = ""


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
        time.sleep(0.05)


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
    """Dim outside an oval guide, mark the oval, show pose target."""
    h, w = frame.shape[:2]

    # Build a mask: 0 inside oval, 1 outside
    mask = np.ones((h, w), dtype=np.uint8) * 255
    cx, cy = w // 2, h // 2
    rx, ry = int(w * 0.22), int(h * 0.36)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 0, -1)

    # Darken outside
    dark = (frame * 0.45).astype(np.uint8)
    out = np.where(mask[:, :, None] > 0, dark, frame)

    # Oval ring
    quality = SESSION.last_status.get("quality", {})
    pose_ok = quality.get("pose_matches", False)
    ring_color = (99, 199, 89) if pose_ok else (200, 200, 200)
    cv2.ellipse(out, (cx, cy), (rx, ry), 0, 0, 360, ring_color, 3)

    # Pose-target arrow
    target = SESSION.current_target
    arrow_msg = {
        Pose.CENTER: "Look straight ahead",
        Pose.LEFT: "Turn left",
        Pose.RIGHT: "Turn right",
        Pose.UP: "Tilt up",
        Pose.DOWN: "Tilt down",
    }.get(target, "")

    if arrow_msg:
        (tw, th), _ = cv2.getTextSize(
            arrow_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(out, arrow_msg, ((w - tw) // 2, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Quality reason
    reason = quality.get("reason", "")
    if reason and reason != "ok":
        (tw, th), _ = cv2.getTextSize(
            reason, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(out, reason, ((w - tw) // 2, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 180, 250), 2)

    return out


@app.route("/api/kiosk_stream")
def api_kiosk_stream():
    return Response(
        CAMERA.mjpeg_generator(draw_overlay=_draw_kiosk_overlay, fps=12),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/enroll_stream")
@basic_auth
def api_enroll_stream():
    return Response(
        CAMERA.mjpeg_generator(draw_overlay=_draw_enrollment_overlay, fps=12),
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
    return jsonify(SESSION.progress())


@app.route("/api/enroll/status")
@basic_auth
def api_enroll_status():
    """Polled every ~250ms by the client. Returns current pose quality."""
    if not SESSION.active:
        return jsonify(active=False)

    frame = CAMERA.latest_frame()
    if frame is None:
        return jsonify(active=True, error="no frame")

    faces = ENGINE.detect(frame)
    if not faces:
        SESSION.last_status = {"quality": {
            "pose_matches": False, "reason": "No face detected"
        }}
        prog = SESSION.progress()
        prog["live"] = {"face_count": 0, "reason": "No face detected"}
        return jsonify(prog)

    # Pick the largest face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    q = evaluate_face(face, frame.shape, min_face_px=200, min_det_score=0.85)
    occ = occlusion.detect(frame, face.landmarks, face.bbox)

    # Occlusion takes precedence over pose-quality reason — the user can't
    # adjust pose meaningfully with a mask on.
    reason = occ.reason or q.reason
    pose_matches = (
        not occ.any
        and q.pose == SESSION.current_target
        and q.in_frame and q.big_enough
        and q.centered and q.sharp_enough
    )

    SESSION.last_status = {"quality": {
        "pose_matches": pose_matches,
        "reason": reason,
        "detected_pose": q.pose.value,
        "occluded": occ.any,
    }}

    prog = SESSION.progress()
    prog["live"] = {
        "face_count": len(faces),
        "detected_pose": q.pose.value,
        "target_pose": SESSION.current_target.value,
        "pose_matches": pose_matches,
        "reason": reason,
        "det_score": q.det_score,
        "occluded": occ.any,
        "sunglasses": occ.sunglasses,
        "mask": occ.mask,
    }
    return jsonify(prog)


@app.route("/api/enroll/capture", methods=["POST"])
@basic_auth
def api_enroll_capture():
    """Capture one sample for the current target pose."""
    if not SESSION.active:
        return jsonify(error="no session"), 400

    frame = CAMERA.latest_frame()
    if frame is None:
        return jsonify(error="no frame"), 503

    faces = ENGINE.detect(frame)
    if not faces:
        return jsonify(error="no face"), 422

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    q = evaluate_face(face, frame.shape, min_face_px=200, min_det_score=0.85)

    occ = occlusion.detect(frame, face.landmarks, face.bbox)
    if occ.any:
        return jsonify(error=occ.reason), 422
    if q.pose != SESSION.current_target:
        return jsonify(error=f"wrong pose (got {q.pose.value}, "
                             f"need {SESSION.current_target.value})"), 422
    if not (q.in_frame and q.big_enough and q.centered):
        return jsonify(error=q.reason), 422

    x1, y1, x2, y2 = face.bbox
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    SESSION.add_sample(SESSION.current_target, face.embedding, crop)
    log.info(f"captured {SESSION.current_target.value} for {SESSION.emp_id}")
    return jsonify(SESSION.progress())


@app.route("/api/enroll/finish", methods=["POST"])
@basic_auth
def api_enroll_finish():
    if not SESSION.is_complete():
        return jsonify(error="incomplete"), 400

    embs = np.stack(list(SESSION.samples.values())).astype(np.float32)
    DB.upsert_employee(
        emp_id=SESSION.emp_id, name=SESSION.name,
        embeddings=embs,
        department=SESSION.department, email=SESSION.email,
    )
    info = {
        "emp_id": SESSION.emp_id, "name": SESSION.name,
        "samples": len(embs),
    }
    log.info(f"enrollment complete: {info}")
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
