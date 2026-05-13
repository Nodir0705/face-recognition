"""Pose classification for guided enrollment.

InsightFace's `pose` attribute returns (yaw, pitch, roll) in degrees:
  yaw   — turning head left/right    (negative = right, positive = left, conventions vary)
  pitch — tilting head up/down       (negative = down, positive = up)
  roll  — tilting head shoulder-ward (we don't use this)

We map continuous yaw/pitch into 5 discrete pose buckets, with hysteresis
so the user gets clear "you're in the right pose now" feedback rather than
the classification flickering on the boundary.

Plus a quality gate: face must be roughly centered in the frame, big enough,
and the detection score must be high. We don't want enrollment images that
are blurry, partial, or at an extreme distance.
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np


class Pose(str, Enum):
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    NONE = "none"


# Angle bands (degrees). Tuned for typical enrollment distances (~80cm).
# These intentionally overlap a little so the "center" bucket is generous —
# we want one easy sample, then small head movements for the other four.
YAW_LEFT_MIN = 15      # turning head left -> yaw > +15
YAW_RIGHT_MIN = 15     # turning right -> yaw < -15
PITCH_UP_MIN = 10
PITCH_DOWN_MIN = 10
CENTER_YAW_MAX = 8
CENTER_PITCH_MAX = 8


@dataclass
class PoseQuality:
    pose: Pose
    in_frame: bool
    big_enough: bool
    centered: bool
    sharp_enough: bool
    det_score: float
    reason: str  # human-readable status for the UI


def classify_pose(yaw: float, pitch: float) -> Pose:
    # InsightFace yaw sign: positive = looking to the user's left (camera's right).
    # We frame instructions from the user's perspective.
    if abs(yaw) <= CENTER_YAW_MAX and abs(pitch) <= CENTER_PITCH_MAX:
        return Pose.CENTER
    # Up/down takes priority when the pitch is extreme
    if pitch >= PITCH_UP_MIN and abs(yaw) < YAW_LEFT_MIN:
        return Pose.UP
    if pitch <= -PITCH_DOWN_MIN and abs(yaw) < YAW_LEFT_MIN:
        return Pose.DOWN
    if yaw >= YAW_LEFT_MIN:
        return Pose.LEFT
    if yaw <= -YAW_RIGHT_MIN:
        return Pose.RIGHT
    return Pose.NONE


def evaluate_face(
    detected,
    frame_shape: tuple[int, int],
    min_face_px: int = 200,         # bigger threshold for enrollment than recognition
    min_det_score: float = 0.85,
    center_tolerance: float = 0.25,  # face center must be within ±25% of frame center
) -> PoseQuality:
    """Decide if this detected face is good enough for one enrollment sample."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = detected.bbox
    face_w = x2 - x1
    face_h = y2 - y1
    fx = (x1 + x2) / 2.0
    fy = (y1 + y2) / 2.0

    in_frame = x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h
    big_enough = face_w >= min_face_px and face_h >= min_face_px
    centered = (abs(fx - w / 2) / w) < center_tolerance and \
               (abs(fy - h / 2) / h) < center_tolerance
    sharp_enough = detected.det_score >= min_det_score

    yaw, pitch, _ = detected.pose
    pose = classify_pose(yaw, pitch)

    reason = "ok"
    if not in_frame:
        reason = "Move into the frame"
    elif not big_enough:
        reason = "Come closer"
    elif not centered:
        reason = "Move to the center"
    elif not sharp_enough:
        reason = "Hold still"
    elif pose == Pose.NONE:
        reason = "Adjust head position"

    return PoseQuality(
        pose=pose,
        in_frame=in_frame,
        big_enough=big_enough,
        centered=centered,
        sharp_enough=sharp_enough,
        det_score=float(detected.det_score),
        reason=reason,
    )


def laplacian_sharpness(face_bgr: np.ndarray) -> float:
    """Variance of the Laplacian. >100 is usually sharp enough, <30 is blurry."""
    import cv2
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
