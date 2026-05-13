"""Tests for pose classification and quality gating."""

from dataclasses import dataclass
import numpy as np
import pytest

from src.pose import Pose, classify_pose, evaluate_face


# ---------- classify_pose ----------

@pytest.mark.parametrize("yaw,pitch,expected", [
    (0,    0,   Pose.CENTER),
    (3,   -4,   Pose.CENTER),
    (8,    8,   Pose.CENTER),     # boundary
    (20,   0,   Pose.LEFT),
    (-25,  0,   Pose.RIGHT),
    (0,   15,   Pose.UP),
    (0,  -20,   Pose.DOWN),
    # In the dead-zone: |yaw|>8 (not CENTER) but |yaw|<15 (not LEFT/RIGHT),
    # and |pitch|<10 (not UP/DOWN) → no clear bucket.
    (12,   5,   Pose.NONE),
])
def test_classify_pose(yaw, pitch, expected):
    assert classify_pose(yaw, pitch) is expected


# ---------- evaluate_face ----------

@dataclass
class FakeDetected:
    bbox: tuple
    det_score: float
    pose: tuple


def _frame_shape(w=1280, h=720):
    return (h, w, 3)


def _centered_face(face_w=300, frame_w=1280, frame_h=720):
    """Bbox roughly centered in the frame."""
    cx, cy = frame_w // 2, frame_h // 2
    half = face_w // 2
    return (cx - half, cy - half, cx + half, cy + half)


def test_evaluate_face_happy_path():
    bbox = _centered_face()
    det = FakeDetected(bbox=bbox, det_score=0.95, pose=(0.0, 0.0, 0.0))
    q = evaluate_face(det, _frame_shape())
    assert q.pose is Pose.CENTER
    assert q.in_frame and q.big_enough and q.centered and q.sharp_enough
    assert q.reason == "ok"


def test_evaluate_face_too_small():
    bbox = _centered_face(face_w=80)
    det = FakeDetected(bbox=bbox, det_score=0.95, pose=(0, 0, 0))
    q = evaluate_face(det, _frame_shape())
    assert not q.big_enough
    assert q.reason == "Come closer"


def test_evaluate_face_off_center():
    # face top-left corner, far from center
    bbox = (10, 10, 310, 310)
    det = FakeDetected(bbox=bbox, det_score=0.95, pose=(0, 0, 0))
    q = evaluate_face(det, _frame_shape())
    assert not q.centered
    assert q.reason == "Move to the center"


def test_evaluate_face_low_det_score():
    bbox = _centered_face()
    det = FakeDetected(bbox=bbox, det_score=0.5, pose=(0, 0, 0))
    q = evaluate_face(det, _frame_shape())
    assert not q.sharp_enough
    assert q.reason == "Hold still"


def test_evaluate_face_pose_none_reports_adjust():
    bbox = _centered_face()
    # See test_classify_pose: (12, 5) sits in the dead-zone between buckets.
    det = FakeDetected(bbox=bbox, det_score=0.95, pose=(12, 5, 0))
    q = evaluate_face(det, _frame_shape())
    assert q.pose is Pose.NONE
    assert q.reason == "Adjust head position"


def test_evaluate_face_priority_in_frame_first():
    # A face partly outside the frame: should report "Move into the frame"
    bbox = (-50, 100, 250, 400)
    det = FakeDetected(bbox=bbox, det_score=0.95, pose=(0, 0, 0))
    q = evaluate_face(det, _frame_shape())
    assert not q.in_frame
    assert q.reason == "Move into the frame"
