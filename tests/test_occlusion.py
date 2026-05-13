"""Tests for the occlusion heuristics.

We construct synthetic frames with controlled patches, not real photos —
the heuristics are supposed to work off pixel statistics, so this is
exactly what they should be tested against.
"""

import numpy as np
import pytest

from src.occlusion import (
    detect, EYE_DARK_V_MAX, EYE_LOW_VAR_MAX,
    MIN_FOREHEAD_LAPVAR, MOUTH_FLAT_RATIO,
)


# Frame layout: a 480x640 BGR canvas, face takes the middle.
H, W = 480, 640
LEYE = (260.0, 200.0)
REYE = (380.0, 200.0)
NOSE = (320.0, 250.0)
LMOUTH = (280.0, 320.0)
RMOUTH = (360.0, 320.0)
LANDMARKS = np.array([LEYE, REYE, NOSE, LMOUTH, RMOUTH], dtype=np.float32)


def _empty_frame(fill=128):
    """Solid-color frame; lapvar is ~0 everywhere."""
    return np.full((H, W, 3), fill, dtype=np.uint8)


def _add_skin_texture(frame, cx, cy, half=30, seed=0):
    """Drop random noise inside a square — gives a forehead patch some texture."""
    rng = np.random.default_rng(seed)
    cx, cy = int(cx), int(cy)
    patch = rng.integers(80, 200, size=(half * 2, half * 2, 3), dtype=np.uint8)
    frame[cy - half:cy + half, cx - half:cx + half] = patch


def _draw_eye(frame, cx, cy, dark=False):
    """Either real-eye-like (bright sclera + dark pupil = high variance)
    or sunglasses (uniform very dark). Drawn as a 28x28 patch."""
    cx, cy = int(cx), int(cy)
    half = 14
    if dark:
        frame[cy - half:cy + half, cx - half:cx + half] = 12  # near-black
    else:
        # bright sclera
        frame[cy - half:cy + half, cx - half:cx + half] = 220
        # dark pupil in the middle = high variance
        frame[cy - 4:cy + 4, cx - 4:cx + 4] = 10


def _draw_mouth(frame, cx, cy, flat=False, seed=1):
    """Flat = mask (uniform color). Otherwise textured like skin/lips."""
    cx, cy = int(cx), int(cy)
    half = 18
    if flat:
        frame[cy - half:cy + half, cx - half:cx + half] = 200
    else:
        rng = np.random.default_rng(seed)
        patch = rng.integers(60, 180, size=(half * 2, half * 2, 3), dtype=np.uint8)
        frame[cy - half:cy + half, cx - half:cx + half] = patch


# ---------- Sunglasses ----------

def test_no_sunglasses_when_eyes_have_pupils():
    frame = _empty_frame()
    _draw_eye(frame, *LEYE, dark=False)
    _draw_eye(frame, *REYE, dark=False)
    _draw_mouth(frame, (LMOUTH[0]+RMOUTH[0])/2, (LMOUTH[1]+RMOUTH[1])/2)
    _add_skin_texture(frame, 320, 150)  # forehead
    occ = detect(frame, LANDMARKS)
    assert occ.sunglasses is False


def test_sunglasses_when_eyes_are_uniformly_dark():
    frame = _empty_frame()
    _draw_eye(frame, *LEYE, dark=True)
    _draw_eye(frame, *REYE, dark=True)
    _draw_mouth(frame, (LMOUTH[0]+RMOUTH[0])/2, (LMOUTH[1]+RMOUTH[1])/2)
    _add_skin_texture(frame, 320, 150)
    occ = detect(frame, LANDMARKS)
    assert occ.sunglasses is True
    assert "sunglasses" in occ.reason.lower()
    assert occ.eye_mean_v < EYE_DARK_V_MAX


# ---------- Mask ----------

def test_no_mask_when_mouth_has_texture():
    frame = _empty_frame()
    _draw_eye(frame, *LEYE, dark=False)
    _draw_eye(frame, *REYE, dark=False)
    _draw_mouth(frame, (LMOUTH[0]+RMOUTH[0])/2, (LMOUTH[1]+RMOUTH[1])/2,
                flat=False)
    _add_skin_texture(frame, 320, 150)
    occ = detect(frame, LANDMARKS)
    assert occ.mask is False


def test_mask_when_mouth_is_flat_relative_to_forehead():
    frame = _empty_frame()
    _draw_eye(frame, *LEYE, dark=False)
    _draw_eye(frame, *REYE, dark=False)
    _draw_mouth(frame, (LMOUTH[0]+RMOUTH[0])/2, (LMOUTH[1]+RMOUTH[1])/2,
                flat=True)
    _add_skin_texture(frame, 320, 150)  # textured forehead → ratio kicks in
    occ = detect(frame, LANDMARKS)
    assert occ.mask is True
    assert "mask" in occ.reason.lower()
    assert occ.mouth_lapvar < occ.forehead_lapvar * MOUTH_FLAT_RATIO


def test_mask_not_flagged_when_forehead_is_also_flat():
    """Low-quality / blurred frame: don't trust the mouth/forehead ratio."""
    frame = _empty_frame()
    _draw_eye(frame, *LEYE, dark=False)
    _draw_eye(frame, *REYE, dark=False)
    _draw_mouth(frame, (LMOUTH[0]+RMOUTH[0])/2, (LMOUTH[1]+RMOUTH[1])/2,
                flat=True)
    # NO forehead texture
    occ = detect(frame, LANDMARKS)
    assert occ.mask is False
    assert occ.forehead_lapvar < MIN_FOREHEAD_LAPVAR


# ---------- Combined / safety ----------

def test_both_at_once():
    frame = _empty_frame()
    _draw_eye(frame, *LEYE, dark=True)
    _draw_eye(frame, *REYE, dark=True)
    _draw_mouth(frame, (LMOUTH[0]+RMOUTH[0])/2, (LMOUTH[1]+RMOUTH[1])/2,
                flat=True)
    _add_skin_texture(frame, 320, 150)
    occ = detect(frame, LANDMARKS)
    assert occ.sunglasses and occ.mask
    assert occ.any
    assert "sunglasses" in occ.reason.lower() and "mask" in occ.reason.lower()


def test_bad_landmarks_returns_no_occlusion():
    """Defensive: bad landmark shape → don't block enrollment."""
    frame = _empty_frame()
    occ = detect(frame, np.zeros((3, 2)))   # wrong shape
    assert not occ.any
    assert occ.reason == ""
