"""Wrapper around InsightFace for detection + ArcFace embeddings.

We intentionally hide InsightFace behind this thin interface so we can swap
to dlib/face_recognition later without touching the rest of the code.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    embedding: np.ndarray             # shape (512,), L2-normalized float32
    det_score: float                  # detection confidence 0..1
    landmarks: np.ndarray             # shape (5, 2): eyes, nose, mouth corners
    pose: tuple[float, float, float]  # yaw, pitch, roll (degrees)


class FaceEngine:
    def __init__(self, model_pack: str = "buffalo_sc",
                 det_size: tuple[int, int] = (320, 320)):
        # Deferred so `match()` (pure math) can be unit-tested without the model
        # being installed. providers=['CPUExecutionProvider'] is correct for Pi 4.
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(
            name=model_pack,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition", "landmark_2d_106"],
        )
        # ctx_id=-1 forces CPU. det_size controls the detection input resolution.
        self.app.prepare(ctx_id=-1, det_size=det_size)

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        """Detect faces and compute embeddings in a single pass."""
        faces = self.app.get(frame_bgr)
        out = []
        for f in faces:
            emb = f.normed_embedding.astype(np.float32)  # already L2-normalized
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            pose = tuple(float(v) for v in (f.pose if f.pose is not None
                                            else (0.0, 0.0, 0.0)))
            out.append(DetectedFace(
                bbox=(x1, y1, x2, y2),
                embedding=emb,
                det_score=float(f.det_score),
                landmarks=f.kps.astype(np.float32),
                pose=pose,
            ))
        return out

    @staticmethod
    def match(probe: np.ndarray, gallery: np.ndarray,
              threshold: float) -> tuple[int, float]:
        """Return (best_index, best_similarity). best_index = -1 if no match.

        Since both probe and gallery rows are L2-normalized, cosine similarity
        is just a dot product.
        """
        if gallery.shape[0] == 0:
            return -1, 0.0
        sims = gallery @ probe   # shape (N,)
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        return (idx, best) if best >= threshold else (-1, best)
