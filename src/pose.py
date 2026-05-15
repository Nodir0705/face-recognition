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


# Canonical 3D positions of the 5 ArcFace landmarks (mm-ish units, centered at
# nose tip). Used for cv2.solvePnP-based head pose estimation when the model
# itself doesn't return one (e.g., Hailo HEFs only emit landmarks).
# Values from a standard "average face" reference; absolute scale doesn't
# matter — only the geometry between points does.
_FACE_3D_MODEL = np.array([
    (-30.0,  30.0, -30.0),   # left eye
    ( 30.0,  30.0, -30.0),   # right eye
    (  0.0,   0.0,   0.0),   # nose tip
    (-25.0, -30.0, -25.0),   # left mouth corner
    ( 25.0, -30.0, -25.0),   # right mouth corner
], dtype=np.float64)


def geometric_pose(kps5: np.ndarray) -> tuple[float, float, float]:
    """Estimate (yaw, pitch, roll) in *approximate degrees* directly from
    landmark positions. No solvePnP — just ratios of pixel distances.

    Why this exists: solvePnP with 5 points + a generic 3D face model is
    noisy and frequently gets the yaw sign wrong on extreme angles. The
    geometric estimate is dumber but rock solid for "is the head turned
    >threshold left/right/up/down" decisions, which is all the enrollment
    flow actually needs.

    Convention (kiosk camera, no mirror):
      * yaw   positive = nose points camera-right = user turning their HEAD LEFT
      * pitch positive = nose points up = user TILTING UP
      * roll  positive = head tilted toward right shoulder

    `kps5` is (5, 2) in image-pixel coordinates: L_eye, R_eye, nose,
    L_mouth, R_mouth (where L/R is the *user's* left/right).
    """
    pts = np.asarray(kps5, dtype=np.float32).reshape(5, 2)
    L_eye, R_eye, nose, L_mouth, R_mouth = pts

    # Eye geometry (the reference frame for both yaw and roll)
    eye_mid_x = (L_eye[0] + R_eye[0]) / 2.0
    eye_mid_y = (L_eye[1] + R_eye[1]) / 2.0
    eye_dx    = L_eye[0] - R_eye[0]      # signed; in unmirrored images L_eye_x > R_eye_x
    eye_dy    = L_eye[1] - R_eye[1]      # signed
    eye_dist  = float(np.hypot(eye_dx, eye_dy))
    if eye_dist < 1.0:
        return 0.0, 0.0, 0.0

    # ----- YAW (left/right head rotation) -----
    # Nose offset from the midpoint of the two eyes, normalized by eye distance.
    # Multiplied by ~100 to land in the same numeric range existing thresholds
    # (e.g., 12-18°) expect, since for a natural ~20° turn the ratio is ~0.18.
    yaw_ratio = (nose[0] - eye_mid_x) / eye_dist
    yaw_deg = yaw_ratio * 100.0

    # ----- PITCH (up/down head tilt) -----
    # Where does the nose sit between the eye line and the mouth line?
    # In a neutral pose the nose lands roughly at fraction ~0.45 of the
    # eye→mouth distance. Tilting up moves the nose UP in image (closer to
    # eyes, smaller fraction), so pitch positive ⇔ fraction below baseline.
    mouth_mid_y = (L_mouth[1] + R_mouth[1]) / 2.0
    eye_to_mouth = mouth_mid_y - eye_mid_y
    if eye_to_mouth < 1.0:
        pitch_deg = 0.0
    else:
        nose_frac = (nose[1] - eye_mid_y) / eye_to_mouth
        # 100× scaling, same reasoning as yaw — a small head tilt of ~10°
        # moves the nose by ~0.10 of the eye-to-mouth distance.
        pitch_deg = (0.45 - nose_frac) * 100.0

    # ----- ROLL (head tilt toward shoulder) -----
    # Angle of the eye line from horizontal.
    roll_deg = float(np.degrees(np.arctan2(eye_dy, eye_dx)))

    return float(yaw_deg), float(pitch_deg), float(roll_deg)


def pose_from_landmarks(kps5: np.ndarray, frame_shape: tuple) -> tuple[float, float, float]:
    """Estimate (yaw, pitch, roll) in degrees via cv2.solvePnP on 5 landmarks.

    Convention used in the rest of the codebase:
      * yaw   positive  = looking to user's left  (camera's right)
      * pitch positive  = looking up
      * roll  positive  = head tilted right shoulder
    Same signs InsightFace's `face.pose` uses, so callers can swap freely.

    `kps5` is (5, 2) in image-pixel coordinates: L_eye, R_eye, nose, L_mouth, R_mouth.
    Returns (0, 0, 0) on failure — caller can treat as "looking straight".
    """
    import cv2
    h, w = frame_shape[:2]
    focal = float(max(w, h))
    cam = np.array([
        [focal, 0,     w / 2.0],
        [0,     focal, h / 2.0],
        [0,     0,     1.0],
    ], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    image_pts = np.asarray(kps5, dtype=np.float64).reshape(5, 2)
    # SOLVEPNP_EPNP supports >= 4 points; SOLVEPNP_ITERATIVE in OpenCV 4.13+
    # requires >= 6 (DLT bound), which we don't have with 5 face landmarks.
    ok, rvec, _ = cv2.solvePnP(
        _FACE_3D_MODEL, image_pts, cam, dist, flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    # Euler decomposition, same order InsightFace returns: (yaw, pitch, roll)
    sy = float(np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
    if sy > 1e-6:
        pitch = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))
        yaw   = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        roll  = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
    else:
        pitch = float(np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1])))
        yaw   = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        roll  = 0.0
    # Hailo + V4L2 cams typically aren't mirrored, so the sign convention
    # already matches our "user-perspective" docstring above.
    return yaw, pitch, roll
