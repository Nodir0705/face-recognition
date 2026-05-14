#!/usr/bin/env python3
"""hailo/recognize_hailo.py — drop-in Hailo replacement for src/recognize.py.

Owns the camera, runs SCRFD detection + ArcFace embedding on the Hailo-8 NPU,
matches against the gallery loaded from the same SQLite `attendance.db` the
Python web app uses, and writes IN/OUT events back to the same DB.

Run INSTEAD of src/recognize.py — never both at once (they'd fight over the
camera).

Why this file alongside src/recognize.py and cpp/recognize_cpp.cpp:
  * Same models conceptually (SCRFD-500m + ArcFace MobileFaceNet)
  * Inference moves from CPU (Python or C++) to the Hailo-8 NPU
  * Pre/post-processing stays on CPU but is small (<5 ms total)
  * Decision logic (Tracker, IN/OUT cooldown) is reused verbatim from
    src/recognize.py — that part is already pure Python and well-tested.

Measured on jarvis (Pi 5 + Hailo-8): ~5.6 ms detect+embed per face,
vs ~48 ms on the same Pi 5 CPU. ~9x faster, 95x tighter jitter.
"""

import argparse
import logging
import signal
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

# Reuse decision logic + DB layer from the Python implementation.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from src.db import AttendanceDB
from src.recognize import Tracker, decide_event_type, _today_start  # noqa: F401


log = logging.getLogger("recognize_hailo")


# ----- ArcFace 5-pt alignment template (same constants as cpp/pipeline.hpp) -----

ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],   # L eye
    [73.5318, 51.5014],   # R eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # L mouth
    [70.7299, 92.2041],   # R mouth
], dtype=np.float32)
ALIGNED_SIZE = 112


def align_face(bgr, kps_5):
    """5-point similarity-transform alignment to 112x112 (ArcFace-standard)."""
    src = np.asarray(kps_5, dtype=np.float32).reshape(5, 2)
    M, _ = cv2.estimateAffinePartial2D(src, ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(bgr, M, (ALIGNED_SIZE, ALIGNED_SIZE),
                           flags=cv2.INTER_LINEAR, borderValue=0)


# ----- SCRFD wrapper (preprocess + infer + decode + NMS) -----

class HailoSCRFD:
    """SCRFD-500m on Hailo. Outputs are grouped by stride via tensor shape:
       channels=2 → score, channels=8 → bbox, channels=20 → kps.
       Spatial size determines the stride (80→8, 40→16, 20→32 for 640 input).
    """
    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2

    def __init__(self, vdevice, hef_path, score_threshold=0.5, nms_threshold=0.4):
        from hailo_platform import (HEF, ConfigureParams, FormatType,
                                     HailoStreamInterface, InferVStreams,
                                     InputVStreamParams, OutputVStreamParams)
        self._FormatType = FormatType
        self._InferVStreams = InferVStreams
        self._InputVStreamParams = InputVStreamParams
        self._OutputVStreamParams = OutputVStreamParams

        hef = HEF(hef_path)
        params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        # vdevice was created with scheduling enabled (see main()) — under that
        # mode we don't manually activate; the scheduler swaps network groups
        # in/out per inference call.
        self.network_group = vdevice.configure(hef, params)[0]
        in_info = self.network_group.get_input_vstream_infos()[0]
        self.input_name = in_info.name
        self.input_h, self.input_w = in_info.shape[:2]   # expect 640, 640

        # Index outputs by (stride, role). role in {"score","bbox","kps"}.
        self.outputs = {}
        for vsi in self.network_group.get_output_vstream_infos():
            sh = vsi.shape   # (H, W, C)
            stride = self.input_h // sh[0]
            role = {2: "score", 8: "bbox", 20: "kps"}.get(sh[2])
            if role is None:
                raise RuntimeError(f"unexpected SCRFD output shape {sh} on {vsi.name}")
            self.outputs[(stride, role)] = vsi.name

        # Pre-compute anchor centers per stride. With 2 anchors per location
        # we just duplicate the (cx, cy) pairs.
        self.anchors = {}
        for s in self.STRIDES:
            gh = self.input_h // s
            gw = self.input_w // s
            xs = np.arange(gw, dtype=np.float32) * s
            ys = np.arange(gh, dtype=np.float32) * s
            gx, gy = np.meshgrid(xs, ys)
            ctrs = np.stack([gx.ravel(), gy.ravel()], axis=1)
            ctrs = np.repeat(ctrs, self.NUM_ANCHORS, axis=0)
            self.anchors[s] = ctrs   # shape (gh*gw*NA, 2)

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        # Build the InferVStreams pipeline once. Activated by the caller.
        self._in_params = InputVStreamParams.make(self.network_group,
                                                    format_type=FormatType.UINT8)
        self._out_params = OutputVStreamParams.make(self.network_group,
                                                      format_type=FormatType.FLOAT32)

    def infer_pipeline(self):
        return self._InferVStreams(self.network_group, self._in_params, self._out_params)

    def preprocess(self, bgr):
        """BGR → letterboxed RGB uint8 NHWC, returns (input, scale, pad_w, pad_h)."""
        h, w = bgr.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(bgr, (new_w, new_h))
        padded = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        return rgb[None, ...], scale

    def decode(self, outputs_dict):
        """Decode + NMS. Returns list of dicts: {bbox: (x1,y1,x2,y2), kps: (5,2), score}.
        Coordinates are in INPUT-IMAGE space (640x640); caller divides by scale.
        """
        all_boxes = []
        all_kps = []
        all_scores = []

        for stride in self.STRIDES:
            # Each tensor is (1, H, W, C); reshape to (H*W*NA, per-anchor-channels)
            scores = outputs_dict[self.outputs[(stride, "score")]].reshape(-1)
            bboxes = outputs_dict[self.outputs[(stride, "bbox")]].reshape(-1, 4)
            kps    = outputs_dict[self.outputs[(stride, "kps")]].reshape(-1, 10)

            mask = scores >= self.score_threshold
            if not mask.any():
                continue
            anchors = self.anchors[stride][mask]
            s = scores[mask]
            b = bboxes[mask] * stride   # decode distances → pixels
            k = kps[mask] * stride

            # bbox: anchor_center (cx, cy), distances (l, t, r, b)
            x1 = anchors[:, 0] - b[:, 0]
            y1 = anchors[:, 1] - b[:, 1]
            x2 = anchors[:, 0] + b[:, 2]
            y2 = anchors[:, 1] + b[:, 3]
            all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))

            # kps: 5 (dx, dy) pairs from anchor center
            kps_xy = anchors[:, None, :] + k.reshape(-1, 5, 2)
            all_kps.append(kps_xy)

            all_scores.append(s)

        if not all_boxes:
            return []
        boxes = np.concatenate(all_boxes)
        kps_all = np.concatenate(all_kps)
        scores = np.concatenate(all_scores)

        # NMS (cv2 wants xywh; we have x1y1x2y2)
        wh = boxes[:, 2:] - boxes[:, :2]
        keep = cv2.dnn.NMSBoxes(np.column_stack([boxes[:, :2], wh]).tolist(),
                                  scores.tolist(),
                                  self.score_threshold, self.nms_threshold)
        if len(keep) == 0:
            return []
        keep = np.array(keep).flatten()
        return [
            {"bbox": tuple(map(float, boxes[i])),
             "kps":  kps_all[i].astype(np.float32),
             "score": float(scores[i])}
            for i in keep
        ]


# ----- ArcFace wrapper -----

class HailoArcFace:
    """ArcFace MobileFaceNet on Hailo. Input 112x112 uint8 NHWC, output (512,)."""
    EMBED_DIM = 512

    def __init__(self, vdevice, hef_path):
        from hailo_platform import (HEF, ConfigureParams, FormatType,
                                     HailoStreamInterface, InferVStreams,
                                     InputVStreamParams, OutputVStreamParams)
        self._InferVStreams = InferVStreams

        hef = HEF(hef_path)
        params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        self.network_group = vdevice.configure(hef, params)[0]
        self.input_name = self.network_group.get_input_vstream_infos()[0].name
        self.output_name = self.network_group.get_output_vstream_infos()[0].name

        self._in_params = InputVStreamParams.make(self.network_group,
                                                    format_type=FormatType.UINT8)
        self._out_params = OutputVStreamParams.make(self.network_group,
                                                      format_type=FormatType.FLOAT32)

    def infer_pipeline(self):
        return self._InferVStreams(self.network_group, self._in_params, self._out_params)

    @staticmethod
    def preprocess(aligned_bgr):
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        return rgb[None, ...]

    def embed(self, infer_pipe, aligned_bgr):
        x = self.preprocess(aligned_bgr)
        out = infer_pipe.infer({self.input_name: x})[self.output_name]
        v = out.flatten().astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v


# ----- Gallery loaded from SQLite -----

class Gallery:
    """In-memory gallery of L2-normalized embeddings, with periodic reload."""
    def __init__(self, db: AttendanceDB, reload_every_sec: int = 60):
        self.db = db
        self.reload_every = reload_every_sec
        self.last_reload = 0.0
        self.emp_ids: list[str] = []
        self.names: list[str] = []
        self.matrix = np.zeros((0, 512), dtype=np.float32)
        self.reload(force=True)

    def reload(self, force=False):
        if not force and (time.time() - self.last_reload) < self.reload_every:
            return
        emp_ids, names, matrix = self.db.load_all_embeddings()
        self.emp_ids = list(emp_ids)
        self.names = list(names)
        self.matrix = matrix
        self.last_reload = time.time()
        log.info(f"gallery: {len(set(emp_ids))} employees, {matrix.shape[0]} embeddings")

    def match(self, probe: np.ndarray, threshold: float):
        if self.matrix.shape[0] == 0:
            return -1, 0.0
        sims = self.matrix @ probe
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        return (idx if best >= threshold else -1), best


# ----- Main loop -----

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, help="path to scrfd_500m.hef")
    ap.add_argument("--rec", required=True, help="path to arcface_mobilefacenet.hef")
    ap.add_argument("--db", required=True, help="path to attendance.db")
    ap.add_argument("--config", help="path to config.yaml (overrides defaults)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--threshold", type=float, default=0.42)
    ap.add_argument("--min-face-px", type=int, default=80)
    ap.add_argument("--consecutive", type=int, default=3)
    ap.add_argument("--cooldown", type=int, default=300)
    ap.add_argument("--no-camera", action="store_true",
                    help="exit after configuring everything (smoke test)")
    return ap.parse_args()


def load_config_overrides(args):
    if not args.config:
        return
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    rec = cfg.get("recognition", {})
    if "match_threshold" in rec:    args.threshold = float(rec["match_threshold"])
    if "min_face_px" in rec:        args.min_face_px = int(rec["min_face_px"])
    if "consecutive_frames" in rec: args.consecutive = int(rec["consecutive_frames"])
    att = cfg.get("attendance", {})
    if "cooldown_sec" in att:       args.cooldown = int(att["cooldown_sec"])
    return cfg


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    cfg = load_config_overrides(args) or {}

    # Bundle the cooldown/toggle config in the shape decide_event_type expects.
    decide_cfg = {"attendance": {
        "cooldown_sec": args.cooldown,
        "toggle_mode": cfg.get("attendance", {}).get("toggle_mode", True),
        "day_start_hour": cfg.get("attendance", {}).get("day_start_hour", 4),
    }}

    db = AttendanceDB(args.db)
    gallery = Gallery(db)

    log.info("opening Hailo device with scheduler…")
    from hailo_platform import VDevice, HailoSchedulingAlgorithm
    vparams = VDevice.create_params()
    # Round-robin scheduler lets us configure both SCRFD and ArcFace and
    # have inferences fairly interleaved without manual activate/deactivate.
    vparams.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    vdevice = VDevice(vparams)

    log.info(f"loading SCRFD: {args.det}")
    det = HailoSCRFD(vdevice, args.det)
    log.info(f"loading ArcFace: {args.rec}")
    rec = HailoArcFace(vdevice, args.rec)

    if args.no_camera:
        log.info("--no-camera given, exiting after setup")
        return

    log.info(f"opening camera {args.camera} @ {args.width}x{args.height}")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        log.error(f"could not open camera {args.camera}")
        sys.exit(2)

    tracker = Tracker()
    running = True
    def stop(*_): nonlocal_running.append(False)
    nonlocal_running = [True]
    signal.signal(signal.SIGINT, lambda *_: nonlocal_running.append(False))
    signal.signal(signal.SIGTERM, lambda *_: nonlocal_running.append(False))

    # With the scheduler enabled, no manual activate() — just open the two
    # InferVStreams pipelines and call .infer() on either one as needed.
    try:
        with det.infer_pipeline() as det_pipe, rec.infer_pipeline() as rec_pipe:
            if True:
                last_log = time.time()
                frame_count = 0
                while nonlocal_running[-1]:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        time.sleep(0.05)
                        continue
                    gallery.reload()  # cheap when within reload window

                    # ---- Detection ----
                    blob, scale = det.preprocess(frame)
                    det_outs = det_pipe.infer({det.input_name: blob})
                    detections = det.decode(det_outs)

                    # Map back to original image coords + filter tiny faces
                    valid = []
                    for d in detections:
                        x1, y1, x2, y2 = (v / scale for v in d["bbox"])
                        if (x2 - x1) < args.min_face_px:
                            continue
                        d_remapped = {
                            "bbox": (x1, y1, x2, y2),
                            "kps":  d["kps"] / scale,
                            "score": d["score"],
                        }
                        valid.append(d_remapped)

                    # ---- Tracking ----
                    # The shared Tracker takes objects with `.bbox = (x1,y1,x2,y2)`.
                    fake_dets = [type("D", (), {"bbox": d["bbox"]})() for d in valid]
                    assignments = tracker.update(fake_dets)

                    # ---- Per-track recognition ----
                    for tid, det_obj in assignments:
                        d = next((v for v in valid if v["bbox"] == det_obj.bbox), None)
                        if d is None:
                            continue
                        aligned = align_face(frame, d["kps"])
                        if aligned is None:
                            continue
                        emb = rec.embed(rec_pipe, aligned)
                        idx, sim = gallery.match(emb, args.threshold)

                        hist = tracker.history(tid)
                        if hist is None:
                            continue
                        if idx < 0:
                            hist.append(None)
                            continue

                        emp_id = gallery.emp_ids[idx]
                        hist.append((emp_id, sim))

                        recent = list(hist)[-args.consecutive:]
                        if len(recent) < args.consecutive:                  continue
                        if any(r is None for r in recent):                  continue
                        if len({r[0] for r in recent}) != 1:                continue

                        event = decide_event_type(db, emp_id, decide_cfg)
                        if event is None:
                            continue
                        avg_conf = float(np.mean([r[1] for r in recent]))
                        row_id = db.log_event(emp_id, event, avg_conf)
                        log.info(f"[#{row_id}] {emp_id} {gallery.names[idx]} → {event} "
                                 f"(sim={avg_conf:.3f})")
                        hist.clear()

                    frame_count += 1
                    if time.time() - last_log >= 5.0:
                        fps = frame_count / (time.time() - last_log)
                        log.info(f"~{fps:.1f} fps over last 5s ({len(detections)} faces this frame)")
                        last_log = time.time()
                        frame_count = 0
    finally:
        cap.release()
        log.info("recognition daemon stopped")


if __name__ == "__main__":
    main()
