"""Tests for the report helpers in db.py."""

from datetime import datetime, timedelta
import numpy as np


def _emb():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((1, 512)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _ts(y, m, d, hh=9, mm=0):
    return int(datetime(y, m, d, hh, mm).timestamp())


def _setup(db, emp_id="E001", name="Alice"):
    db.upsert_employee(emp_id, name, _emb(), department="Eng")


def test_get_employee_returns_record(tmp_db):
    _setup(tmp_db)
    rec = tmp_db.get_employee("E001")
    assert rec["name"] == "Alice"
    assert rec["department"] == "Eng"
    assert rec["active"] == 1


def test_get_employee_missing_returns_none(tmp_db):
    assert tmp_db.get_employee("nope") is None


def test_events_in_range_filters_by_emp_and_time(tmp_db):
    _setup(tmp_db, "E001", "Alice")
    _setup(tmp_db, "E002", "Bob")

    tmp_db.log_event("E001", "IN", 0.9, ts=_ts(2026, 5, 10, 9))
    tmp_db.log_event("E001", "OUT", 0.9, ts=_ts(2026, 5, 10, 18))
    tmp_db.log_event("E002", "IN", 0.9, ts=_ts(2026, 5, 10, 9, 30))
    tmp_db.log_event("E001", "IN", 0.9, ts=_ts(2026, 5, 12, 9))

    # All of 2026-05-10
    rows = tmp_db.events_in_range(_ts(2026, 5, 10, 0), _ts(2026, 5, 11, 0))
    assert len(rows) == 3
    assert [r["event_type"] for r in rows] == ["IN", "IN", "OUT"]  # time-ordered

    # Just Alice in that window
    rows = tmp_db.events_in_range(_ts(2026, 5, 10, 0), _ts(2026, 5, 11, 0),
                                   emp_id="E001")
    assert len(rows) == 2
    assert all(r["emp_id"] == "E001" for r in rows)


def test_daily_summary_pairs_in_with_last_out(tmp_db):
    _setup(tmp_db)
    tmp_db.log_event("E001", "IN", 0.9, ts=_ts(2026, 5, 10, 9))
    tmp_db.log_event("E001", "OUT", 0.9, ts=_ts(2026, 5, 10, 12))
    tmp_db.log_event("E001", "IN", 0.9, ts=_ts(2026, 5, 10, 13))
    tmp_db.log_event("E001", "OUT", 0.9, ts=_ts(2026, 5, 10, 18))

    summary = tmp_db.daily_summary("E001",
                                    _ts(2026, 5, 1, 0),
                                    _ts(2026, 5, 31, 0))
    assert len(summary) == 1
    row = summary[0]
    assert row["date"] == "2026-05-10"
    # worked = last_out (18:00) - first_in (09:00) = 9h
    assert row["worked_seconds"] == 9 * 3600
    assert len(row["events"]) == 4


def test_daily_summary_only_in_yields_zero_hours(tmp_db):
    _setup(tmp_db)
    tmp_db.log_event("E001", "IN", 0.9, ts=_ts(2026, 5, 10, 9))
    summary = tmp_db.daily_summary("E001",
                                    _ts(2026, 5, 1, 0),
                                    _ts(2026, 5, 31, 0))
    assert summary[0]["worked_seconds"] == 0
    assert summary[0]["last_out"] is None


def test_events_per_day_groups_correctly(tmp_db):
    _setup(tmp_db)
    _setup(tmp_db, "E002", "Bob")
    tmp_db.log_event("E001", "IN", 0.9, ts=_ts(2026, 5, 10, 9))
    tmp_db.log_event("E002", "IN", 0.9, ts=_ts(2026, 5, 10, 9, 30))
    tmp_db.log_event("E001", "OUT", 0.9, ts=_ts(2026, 5, 10, 18))
    tmp_db.log_event("E001", "IN", 0.9, ts=_ts(2026, 5, 11, 9))

    rows = tmp_db.events_per_day(_ts(2026, 5, 1, 0), _ts(2026, 6, 1, 0))
    by_date = {r["date"]: r for r in rows}
    assert by_date["2026-05-10"]["in_count"] == 2
    assert by_date["2026-05-10"]["out_count"] == 1
    assert by_date["2026-05-11"]["in_count"] == 1
    assert by_date["2026-05-11"]["out_count"] == 0
