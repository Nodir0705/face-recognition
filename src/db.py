"""SQLite layer for the attendance system.

Schema:
    employees    — registered people and their face embeddings
    attendance   — every IN/OUT event (the source of truth)
    sync_queue   — rows waiting to be pushed to Google Sheets
"""

import sqlite3
import json
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    emp_id          TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    department      TEXT,
    email           TEXT,
    embedding       BLOB NOT NULL,       -- np.float32 array, shape (N, 512)
    enrolled_at     INTEGER NOT NULL,    -- unix epoch
    active          INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS attendance (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_id          TEXT NOT NULL,
    event_type      TEXT NOT NULL CHECK(event_type IN ('IN', 'OUT')),
    timestamp       INTEGER NOT NULL,    -- unix epoch
    confidence      REAL NOT NULL,
    synced          INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (emp_id) REFERENCES employees(emp_id)
);

CREATE INDEX IF NOT EXISTS idx_attendance_emp_time
    ON attendance(emp_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_attendance_synced
    ON attendance(synced, timestamp);

CREATE TABLE IF NOT EXISTS sync_failures (
    attendance_id   INTEGER PRIMARY KEY,
    retries         INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    last_attempt    INTEGER
);
"""


class AttendanceDB:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # WAL gives us concurrent reads while the recognition daemon writes.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---------- Employees ----------

    def upsert_employee(
        self,
        emp_id: str,
        name: str,
        embeddings: np.ndarray,
        department: str = "",
        email: str = "",
    ) -> None:
        """Insert or update an employee. `embeddings` is shape (N, 512) float32."""
        assert embeddings.dtype == np.float32, "embeddings must be float32"
        assert embeddings.ndim == 2 and embeddings.shape[1] == 512
        blob = embeddings.tobytes()
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO employees (emp_id, name, department, email,
                                       embedding, enrolled_at, active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(emp_id) DO UPDATE SET
                    name = excluded.name,
                    department = excluded.department,
                    email = excluded.email,
                    embedding = excluded.embedding,
                    enrolled_at = excluded.enrolled_at,
                    active = 1
                """,
                (emp_id, name, department, email, blob, int(time.time())),
            )

    def deactivate_employee(self, emp_id: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE employees SET active = 0 WHERE emp_id = ?", (emp_id,))

    def load_all_embeddings(self):
        """Return (emp_ids, names, stacked_embeddings) for matching.

        stacked_embeddings is shape (total_N, 512), and emp_ids/names are
        parallel lists of length total_N indicating who each row belongs to.
        """
        emp_ids, names, embs = [], [], []
        with self._conn() as c:
            rows = c.execute(
                "SELECT emp_id, name, embedding FROM employees WHERE active = 1"
            ).fetchall()
        for r in rows:
            arr = np.frombuffer(r["embedding"], dtype=np.float32).reshape(-1, 512)
            for vec in arr:
                emp_ids.append(r["emp_id"])
                names.append(r["name"])
                embs.append(vec)
        if not embs:
            return [], [], np.zeros((0, 512), dtype=np.float32)
        return emp_ids, names, np.stack(embs)

    # ---------- Attendance ----------

    def last_event(self, emp_id: str):
        with self._conn() as c:
            row = c.execute(
                """SELECT event_type, timestamp FROM attendance
                   WHERE emp_id = ? ORDER BY timestamp DESC LIMIT 1""",
                (emp_id,),
            ).fetchone()
        return (row["event_type"], row["timestamp"]) if row else (None, None)

    def log_event(
        self, emp_id: str, event_type: str, confidence: float, ts: int | None = None
    ) -> int:
        ts = ts or int(time.time())
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO attendance (emp_id, event_type, timestamp,
                                           confidence, synced)
                   VALUES (?, ?, ?, ?, 0)""",
                (emp_id, event_type, ts, confidence),
            )
            return cur.lastrowid

    def pending_sync(self, limit: int = 50):
        with self._conn() as c:
            return c.execute(
                """SELECT a.id, a.emp_id, e.name, e.department, a.event_type,
                          a.timestamp, a.confidence
                   FROM attendance a
                   JOIN employees e ON e.emp_id = a.emp_id
                   WHERE a.synced = 0
                   ORDER BY a.timestamp ASC
                   LIMIT ?""",
                (limit,),
            ).fetchall()

    def mark_synced(self, attendance_ids: list[int]) -> None:
        if not attendance_ids:
            return
        with self._conn() as c:
            c.executemany(
                "UPDATE attendance SET synced = 1 WHERE id = ?",
                [(i,) for i in attendance_ids],
            )

    def record_sync_failure(self, attendance_id: int, error: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO sync_failures (attendance_id, retries,
                                              last_error, last_attempt)
                   VALUES (?, 1, ?, ?)
                   ON CONFLICT(attendance_id) DO UPDATE SET
                     retries = retries + 1,
                     last_error = excluded.last_error,
                     last_attempt = excluded.last_attempt""",
                (attendance_id, error, int(time.time())),
            )

    # ---------- Reports ----------

    def get_employee(self, emp_id: str):
        with self._conn() as c:
            row = c.execute(
                """SELECT emp_id, name, department, email, enrolled_at, active
                   FROM employees WHERE emp_id = ?""",
                (emp_id,),
            ).fetchone()
        return dict(row) if row else None

    def events_in_range(
        self,
        start_ts: int,
        end_ts: int,
        emp_id: str | None = None,
    ):
        """Return all attendance rows within [start_ts, end_ts), oldest first.

        Joins employee name+department so callers don't need a second query.
        """
        sql = """SELECT a.id, a.emp_id, e.name, e.department,
                        a.event_type, a.timestamp, a.confidence, a.synced
                 FROM attendance a
                 JOIN employees e ON e.emp_id = a.emp_id
                 WHERE a.timestamp >= ? AND a.timestamp < ?"""
        params: list = [start_ts, end_ts]
        if emp_id:
            sql += " AND a.emp_id = ?"
            params.append(emp_id)
        sql += " ORDER BY a.timestamp ASC"
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def daily_summary(
        self,
        emp_id: str,
        start_ts: int,
        end_ts: int,
    ):
        """Per-day summary for one employee.

        Pairs each IN with the next OUT of the same day (local time). Days
        with only an IN or only an OUT show the missing side as None and
        worked_seconds as 0. Days are bucketed by local date — we let SQLite
        do it via the `localtime` modifier.

        Returns a list of dicts: {date, first_in, last_out, worked_seconds, events}
        """
        rows = self.events_in_range(start_ts, end_ts, emp_id=emp_id)
        from collections import defaultdict
        from datetime import datetime, timezone

        by_day: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            day = datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d")
            by_day[day].append(r)

        out = []
        for day in sorted(by_day):
            events = by_day[day]
            ins = [e for e in events if e["event_type"] == "IN"]
            outs = [e for e in events if e["event_type"] == "OUT"]
            first_in = ins[0]["timestamp"] if ins else None
            last_out = outs[-1]["timestamp"] if outs else None
            worked = (last_out - first_in) if (first_in and last_out
                                               and last_out > first_in) else 0
            out.append({
                "date": day,
                "first_in": first_in,
                "last_out": last_out,
                "worked_seconds": worked,
                "events": events,
            })
        return out

    def events_per_day(self, start_ts: int, end_ts: int):
        """Counts of IN and OUT events per local date in the range.

        Returns list of dicts ordered by date: {date, in_count, out_count}.
        """
        with self._conn() as c:
            # SQLite: convert unix epoch → local date string
            rows = c.execute(
                """SELECT date(timestamp, 'unixepoch', 'localtime') AS d,
                          event_type, COUNT(*) AS n
                   FROM attendance
                   WHERE timestamp >= ? AND timestamp < ?
                   GROUP BY d, event_type
                   ORDER BY d ASC""",
                (start_ts, end_ts),
            ).fetchall()
        agg: dict[str, dict] = {}
        for r in rows:
            day = r["d"]
            entry = agg.setdefault(day, {"date": day, "in_count": 0, "out_count": 0})
            if r["event_type"] == "IN":
                entry["in_count"] = r["n"]
            else:
                entry["out_count"] = r["n"]
        return list(agg.values())
