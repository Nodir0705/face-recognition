"""Tests for the SQLite layer."""

import time
import numpy as np
import pytest


def _emb(seed: int, n: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, 512)).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def test_schema_initializes(tmp_db):
    # Just exercising the constructor created the tables
    with tmp_db._conn() as c:
        names = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"employees", "attendance", "sync_failures"}.issubset(names)


def test_upsert_and_load_roundtrip(tmp_db):
    e1 = _emb(1, n=2)
    tmp_db.upsert_employee("E001", "Alice", e1, department="Eng")
    e2 = _emb(2, n=3)
    tmp_db.upsert_employee("E002", "Bob", e2, department="Sales")

    emp_ids, names, gallery = tmp_db.load_all_embeddings()
    assert gallery.shape == (5, 512)
    # Each emp_id appears once per row
    assert emp_ids.count("E001") == 2
    assert emp_ids.count("E002") == 3
    assert "Alice" in names and "Bob" in names


def test_upsert_replaces_existing(tmp_db):
    tmp_db.upsert_employee("E001", "Alice", _emb(1, n=2))
    tmp_db.upsert_employee("E001", "Alice (updated)", _emb(3, n=4))

    emp_ids, names, gallery = tmp_db.load_all_embeddings()
    assert gallery.shape[0] == 4   # replaced, not appended
    assert names[0] == "Alice (updated)"


def test_deactivate_hides_from_load(tmp_db):
    tmp_db.upsert_employee("E001", "Alice", _emb(1))
    tmp_db.upsert_employee("E002", "Bob", _emb(2))
    tmp_db.deactivate_employee("E001")

    emp_ids, _, gallery = tmp_db.load_all_embeddings()
    assert "E001" not in emp_ids
    assert gallery.shape[0] == 1


def test_log_event_and_last_event(tmp_db):
    tmp_db.upsert_employee("E001", "Alice", _emb(1))

    tmp_db.log_event("E001", "IN", 0.92, ts=1_000_000)
    tmp_db.log_event("E001", "OUT", 0.88, ts=1_000_500)

    typ, ts = tmp_db.last_event("E001")
    assert typ == "OUT"
    assert ts == 1_000_500


def test_last_event_returns_none_for_unknown(tmp_db):
    typ, ts = tmp_db.last_event("nope")
    assert typ is None and ts is None


def test_pending_sync_and_mark(tmp_db):
    tmp_db.upsert_employee("E001", "Alice", _emb(1))
    rid1 = tmp_db.log_event("E001", "IN", 0.9)
    rid2 = tmp_db.log_event("E001", "OUT", 0.9)

    pending = tmp_db.pending_sync()
    assert {p["id"] for p in pending} == {rid1, rid2}

    tmp_db.mark_synced([rid1])
    pending = tmp_db.pending_sync()
    assert {p["id"] for p in pending} == {rid2}


def test_event_type_check_constraint(tmp_db):
    tmp_db.upsert_employee("E001", "Alice", _emb(1))
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        tmp_db.log_event("E001", "BOGUS", 1.0)


def test_record_sync_failure_increments(tmp_db):
    tmp_db.upsert_employee("E001", "Alice", _emb(1))
    rid = tmp_db.log_event("E001", "IN", 0.9)
    tmp_db.record_sync_failure(rid, "first")
    tmp_db.record_sync_failure(rid, "second")
    with tmp_db._conn() as c:
        row = c.execute(
            "SELECT retries, last_error FROM sync_failures WHERE attendance_id = ?",
            (rid,),
        ).fetchone()
    assert row["retries"] == 2
    assert row["last_error"] == "second"
