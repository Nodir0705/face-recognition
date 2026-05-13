"""Tests for the recognition daemon's pure-logic helpers.

We don't import recognize.py at module level because it pulls in cv2/insightface.
Each test imports inside the function to keep collection cheap and to allow
running these on a dev box without InsightFace installed.
"""

import time
from datetime import datetime, timedelta


def _cfg(toggle=True, cooldown=300, day_start=4):
    return {
        "attendance": {
            "toggle_mode": toggle,
            "cooldown_sec": cooldown,
            "day_start_hour": day_start,
        }
    }


def test_decide_event_first_event_today_is_in(tmp_db):
    from src.recognize import decide_event_type

    tmp_db.upsert_employee("E001", "Alice", _emb())
    # No prior events at all → first event of the day is IN
    assert decide_event_type(tmp_db, "E001", _cfg()) == "IN"


def test_decide_event_toggles_after_in(tmp_db):
    from src.recognize import decide_event_type
    tmp_db.upsert_employee("E001", "Alice", _emb())
    # Log an IN well before the cooldown window
    tmp_db.log_event("E001", "IN", 0.9, ts=int(time.time()) - 600)
    assert decide_event_type(tmp_db, "E001", _cfg()) == "OUT"


def test_decide_event_cooldown_blocks(tmp_db):
    from src.recognize import decide_event_type
    tmp_db.upsert_employee("E001", "Alice", _emb())
    tmp_db.log_event("E001", "IN", 0.9, ts=int(time.time()) - 10)
    # Within cooldown window → None (skip)
    assert decide_event_type(tmp_db, "E001", _cfg()) is None


def test_decide_event_day_half_mode(tmp_db, monkeypatch):
    from src import recognize
    tmp_db.upsert_employee("E001", "Alice", _emb())

    class FakeDT:
        @staticmethod
        def now():
            return datetime(2026, 5, 13, 9, 0, 0)
        @staticmethod
        def fromtimestamp(ts):
            return datetime.fromtimestamp(ts)
    monkeypatch.setattr(recognize, "datetime", FakeDT)
    assert recognize.decide_event_type(tmp_db, "E001", _cfg(toggle=False)) == "IN"

    class FakeDTPM(FakeDT):
        @staticmethod
        def now():
            return datetime(2026, 5, 13, 14, 0, 0)
    monkeypatch.setattr(recognize, "datetime", FakeDTPM)
    assert recognize.decide_event_type(tmp_db, "E001", _cfg(toggle=False)) == "OUT"


def _emb():
    import numpy as np
    rng = np.random.default_rng(0)
    v = rng.standard_normal((1, 512)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v
