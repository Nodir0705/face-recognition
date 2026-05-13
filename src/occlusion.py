"""Heuristic detection for sunglasses and masks.

Why not a model? Adding a dedicated occlusion classifier roughly triples
inference cost on the Pi 4 and requires re-tuning. These two heuristics
catch the obvious cases (which are 95% of what employees actually do) and
fail in the safe direction — a false "you're wearing a mask" prompt is
much less harmful than letting a masked face match the wrong person.

Signals used:

* **Sunglasses**: a dark, low-variance patch around each eye landmark.
  Real eyes have bright sclera + dark pupil = high variance. Sunglasses
  are uniformly dark.

* **Mask**: a low-texture patch around the mouth midpoint. Lips and the
  philtrum have skin micro-texture; cloth/surgical masks are much flatter.
  We compare the mouth region's Laplacian variance to a forehead-area
  control patch from the same frame so we're robust to overall image
  blur or low-light flatness.

Both signals are derived from the 5 landmarks InsightFace always returns
(left eye, right eye, nose, mouth-left, mouth-right) — no dependence on
the optional 106-point module.
"""

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class Occlusion:
    sunglasses: bool
    mask: bool
    eye_mean_v: float        # 0..255
    eye_var: float
    mouth_lapvar: float
    forehead_lapvar: float
    reason: str              # "" if no occlusion

    @property
    def any(self) -> bool:
        return self.sunglasses or self.mask


# Tunables. These were eyeballed against typical office lighting
# (~250-500 lux indoor LED) and may need a small bump on darker installs.
EYE_PATCH = 14         # half-size of the square patch around each eye
EYE_DARK_V_MAX = 60    # mean V channel below this counts as "dark"
EYE_LOW_VAR_MAX = 220  # combined variance below this counts as "uniform"

MOUTH_PATCH = 18
FOREHEAD_PATCH = 22
# A mouth region is "flat" if its Laplacian variance is < this fraction
# of the forehead patch's variance. Mask cloth has much less micro-texture
# than skin, even in poor lighting.
MOUTH_FLAT_RATIO = 0.45
# Absolute floor — if the forehead patch itself is essentially flat
# (low-quality frame, very smooth skin) we don't trust the ratio.
MIN_FOREHEAD_LAPVAR = 8.0


def _safe_patch(frame: np.ndarray, cx: float, cy: float, half: int):
    """Return a square patch centered on (cx, cy), or None if it falls outside."""
    h, w = frame.shape[:2]
    cx, cy = int(cx), int(cy)
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2]


def _lapvar(patch_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect(
    frame_bgr: np.ndarray,
    landmarks_5: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
) -> Occlusion:
    """Run both heuristics. `landmarks_5` shape is (5, 2): [Leye, Reye, nose, Lmouth, Rmouth].

    `bbox` is optional but improves the forehead control patch — without it
    we estimate the forehead position from landmark geometry instead.
    """
    pts = np.asarray(landmarks_5, dtype=np.float32)
    if pts.shape != (5, 2):
        # Bad landmarks: defensively report no occlusion
        return Occlusion(False, False, 0, 0, 0, 0, "")

    leye, reye, nose, lmouth, rmouth = pts

    # ---------- Sunglasses ----------
    eye_patches = []
    for cx, cy in (leye, reye):
        patch = _safe_patch(frame_bgr, cx, cy, EYE_PATCH)
        if patch is not None:
            eye_patches.append(patch)
    if eye_patches:
        joined = np.concatenate([p.reshape(-1, 3) for p in eye_patches])
        v = cv2.cvtColor(joined.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)[0, :, 2]
        eye_mean_v = float(v.mean())
        eye_var = float(v.var())
    else:
        eye_mean_v, eye_var = 255.0, 1000.0  # no patches → assume not occluded

    sunglasses = eye_mean_v < EYE_DARK_V_MAX and eye_var < EYE_LOW_VAR_MAX

    # ---------- Mask ----------
    mouth_cx = (lmouth[0] + rmouth[0]) / 2
    mouth_cy = (lmouth[1] + rmouth[1]) / 2
    mouth_patch = _safe_patch(frame_bgr, mouth_cx, mouth_cy, MOUTH_PATCH)
    mouth_lapvar = _lapvar(mouth_patch) if mouth_patch is not None else 0.0

    # Forehead control: above the eye line by ~half the eye-to-mouth distance.
    eye_cy = (leye[1] + reye[1]) / 2
    eye_to_mouth = max(20.0, mouth_cy - eye_cy)
    forehead_cx = (leye[0] + reye[0]) / 2
    forehead_cy = eye_cy - eye_to_mouth * 0.55
    forehead_patch = _safe_patch(frame_bgr, forehead_cx, forehead_cy, FOREHEAD_PATCH)
    forehead_lapvar = _lapvar(forehead_patch) if forehead_patch is not None else 0.0

    mask = (
        forehead_lapvar >= MIN_FOREHEAD_LAPVAR
        and mouth_lapvar < forehead_lapvar * MOUTH_FLAT_RATIO
    )

    # Compose a human-readable reason (UI uses this verbatim)
    reason = ""
    if sunglasses and mask:
        reason = "Please remove sunglasses and mask"
    elif sunglasses:
        reason = "Please remove sunglasses"
    elif mask:
        reason = "Please remove face mask"

    return Occlusion(
        sunglasses=sunglasses,
        mask=mask,
        eye_mean_v=eye_mean_v,
        eye_var=eye_var,
        mouth_lapvar=mouth_lapvar,
        forehead_lapvar=forehead_lapvar,
        reason=reason,
    )
