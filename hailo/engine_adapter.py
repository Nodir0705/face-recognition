"""hailo/engine_adapter.py — HailoFaceEngine with the FaceEngine API.

Lets src/web/app.py swap out InsightFace + ONNXRuntime for a Hailo-8 NPU
without touching the recognition loop or enrollment endpoints. Just selects
the right class at startup based on `recognition.backend` in config.yaml.

Drop-in API match for src.face_engine.FaceEngine:
  * .detect(frame_bgr) -> list[DetectedFace]
  * .match(probe, gallery, threshold) -> (best_index, best_similarity)

Important constraint (see hailo/embedding_compat.py):
  Hailo's arcface_mobilefacenet HEF is a different trained checkpoint from
  InsightFace's buffalo_sc/w600k_mbf.onnx. Embeddings live in completely
  different vector spaces (cos_sim ≈ 0 between the two). This means:

  * Switching backend = re-enroll everyone.
  * The Flask app should refuse to mix gallery embeddings from different
    backends. Easiest enforcement: clear `data/attendance.db`'s employees
    table before flipping the backend flag.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.face_engine import DetectedFace
# Reuse the SCRFD + ArcFace wrappers from the daemon
sys.path.insert(0, str(PROJECT_ROOT / "hailo"))
from recognize_hailo import HailoSCRFD, HailoArcFace, align_face


class HailoFaceEngine:
    """Same surface as src.face_engine.FaceEngine but backed by a Hailo-8."""

    def __init__(self, det_hef: str, rec_hef: str,
                 score_threshold: float = 0.5, nms_threshold: float = 0.4):
        from hailo_platform import VDevice, HailoSchedulingAlgorithm
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        # The VDevice is owned for the lifetime of this engine. The Flask
        # process exits will close it implicitly.
        self._vdevice = VDevice(params)

        self._det = HailoSCRFD(self._vdevice, det_hef,
                                score_threshold=score_threshold,
                                nms_threshold=nms_threshold)
        self._rec = HailoArcFace(self._vdevice, rec_hef)

        # Open both InferVStreams pipelines once; reuse for the process lifetime.
        # These are context managers, but we manually call __enter__ to keep
        # them alive across detect() calls.
        self._det_pipe_cm = self._det.infer_pipeline()
        self._det_pipe = self._det_pipe_cm.__enter__()
        self._rec_pipe_cm = self._rec.infer_pipeline()
        self._rec_pipe = self._rec_pipe_cm.__enter__()

    def close(self):
        # Best-effort — Flask doesn't normally call us back, but tests do.
        for cm in (self._rec_pipe_cm, self._det_pipe_cm):
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        """Detect faces and compute embeddings in a single pass — same shape
        as FaceEngine.detect()."""
        blob, scale = self._det.preprocess(frame_bgr)
        outs = self._det_pipe.infer({self._det.input_name: blob})
        decoded = self._det.decode(outs)

        out: list[DetectedFace] = []
        for d in decoded:
            x1, y1, x2, y2 = (v / scale for v in d["bbox"])
            kps5 = d["kps"] / scale
            aligned = align_face(frame_bgr, kps5)
            if aligned is None:
                continue
            emb = self._rec.embed(self._rec_pipe, aligned)
            out.append(DetectedFace(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                embedding=emb.astype(np.float32),
                det_score=float(d["score"]),
                landmarks=kps5.astype(np.float32),
                pose=(0.0, 0.0, 0.0),  # Hailo HEFs don't provide pose
            ))
        return out

    @staticmethod
    def match(probe: np.ndarray, gallery: np.ndarray,
              threshold: float) -> tuple[int, float]:
        """Identical to FaceEngine.match — both vectors L2-normalized,
        cosine sim is the dot product."""
        if gallery.shape[0] == 0:
            return -1, 0.0
        sims = gallery @ probe
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        return (idx, best) if best >= threshold else (-1, best)
