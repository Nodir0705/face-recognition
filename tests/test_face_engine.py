"""Tests for the FaceEngine math.

We only test the pure `match()` static method here — it doesn't need the
InsightFace model. End-to-end detection tests live behind @pytest.mark.slow.
"""

import numpy as np
import pytest


def _norm_rows(arr):
    arr = arr.astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def test_match_returns_best_above_threshold():
    from src.face_engine import FaceEngine
    rng = np.random.default_rng(0)
    gallery = _norm_rows(rng.standard_normal((10, 512)))
    probe = gallery[3].copy()
    idx, sim = FaceEngine.match(probe, gallery, threshold=0.5)
    assert idx == 3
    assert sim > 0.99


def test_match_returns_minus_one_when_below_threshold():
    from src.face_engine import FaceEngine
    rng = np.random.default_rng(1)
    gallery = _norm_rows(rng.standard_normal((10, 512)))
    probe = _norm_rows(rng.standard_normal((1, 512)))[0]
    idx, sim = FaceEngine.match(probe, gallery, threshold=0.95)
    assert idx == -1
    # similarity is still reported even when below threshold
    assert -1.0 <= sim <= 1.0


def test_match_handles_empty_gallery():
    from src.face_engine import FaceEngine
    probe = np.zeros(512, dtype=np.float32)
    probe[0] = 1.0
    idx, sim = FaceEngine.match(probe, np.zeros((0, 512), dtype=np.float32),
                                 threshold=0.42)
    assert idx == -1 and sim == 0.0


@pytest.mark.slow
def test_engine_loads_real_model():
    from src.face_engine import FaceEngine
    eng = FaceEngine(model_pack="buffalo_sc", det_size=(320, 320))
    assert eng.app is not None
