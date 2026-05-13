"""Shared pytest fixtures.

The face engine itself (InsightFace) is heavy and downloads ~50MB the first
time it runs, so by default we skip anything that needs it. Tests that *do*
need it must be marked `@pytest.mark.slow` and are only collected when you
pass `--run-slow`.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make `src/` importable as a top-level package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="run @pytest.mark.slow tests (loads InsightFace, ~50MB download)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow: pass --run-slow to enable")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def tmp_db(tmp_path):
    """A fresh AttendanceDB instance pointing at a tmp file."""
    from src.db import AttendanceDB
    return AttendanceDB(str(tmp_path / "att.db"))


@pytest.fixture
def fake_embedding():
    """A deterministic L2-normalized 512-d vector."""
    rng = np.random.default_rng(42)
    v = rng.standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v)
    return v
