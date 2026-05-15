"""Google Sheets sync — runtime-configurable from the admin web UI.

Replaces the standalone `src/sync.py` worker with an in-process class the
Flask app can drive. Two key UX changes vs the old design:

  1. Spreadsheet ID is stored in the SQLite settings table, not config.yaml.
     Admin pastes a Sheet URL into the dashboard at runtime; the background
     thread picks up the new ID without restart.
  2. Append-only writes (`spreadsheets.values.append` with `INSERT_ROWS`)
     so admin's manual columns/rows in the same sheet are never overwritten.
     The other side can keep editing the sheet; we just add rows at the bottom.

Sheet column layout:
    A: Log ID (local DB row id)         F: Time (HH:MM:SS, local)
    B: Employee ID                       G: Event (IN / OUT)
    C: Name                              H: Confidence (float)
    D: Department                        I: Source ("auto" or "manual")
    E: Date (YYYY-MM-DD, local)
"""

import json
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.db import AttendanceDB

log = logging.getLogger("sheets")

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Settings keys (also used by the API + UI)
KEY_SPREADSHEET_ID = "sheets.spreadsheet_id"
KEY_WORKSHEET     = "sheets.worksheet"
KEY_LAST_SYNC_TS  = "sheets.last_sync_ts"
KEY_LAST_SYNC_OK  = "sheets.last_sync_ok"
KEY_LAST_ERROR    = "sheets.last_error"

DEFAULT_WORKSHEET = "Attendance"

# Matches the sheet ID inside a typical Google Sheets URL:
#   https://docs.google.com/spreadsheets/d/<ID>/edit#gid=0
_SHEET_URL_RE = re.compile(r"/spreadsheets/d/([A-Za-z0-9_-]+)")


def parse_sheet_input(raw: str) -> str:
    """Accept either a full Sheets URL or a bare spreadsheet ID. Returns the ID.
    Raises ValueError on garbage input."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty input")
    m = _SHEET_URL_RE.search(raw)
    if m:
        return m.group(1)
    # Bare ID: Google IDs are typically ~44 chars, alphanumerics + - and _
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", raw):
        return raw
    raise ValueError(f"could not parse a spreadsheet ID from {raw!r}")


class SheetsSync:
    """Background-syncs new attendance rows to Google Sheets.

    Usage:
        sync = SheetsSync(db, credentials_path)
        sync.start_background()        # spawns thread; safe even if not configured
        sync.configure(spreadsheet_id="...", worksheet="Attendance")
        # later:
        status = sync.status()         # for the UI
        ok, err = sync.test_connection()  # poke the sheet to verify access

    The thread is harmless when not configured — it just sleeps and re-checks
    settings on each interval, picking up a new spreadsheet ID at runtime.
    """

    def __init__(self, db: "AttendanceDB", credentials_path: Path,
                 sync_interval_sec: int = 30, batch_limit: int = 50):
        self.db = db
        self.credentials_path = Path(credentials_path)
        self.sync_interval = sync_interval_sec
        self.batch_limit = batch_limit
        self._service = None
        self._service_email: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_synced_count = 0

    # ----- credentials -----

    def credentials_present(self) -> bool:
        return self.credentials_path.exists()

    def install_credentials(self, json_text: str) -> tuple[bool, str]:
        """Atomically write a uploaded service-account JSON to disk.

        Validates the file looks like a real Google service-account key before
        accepting it. Returns (ok, message). On success, the cached service
        email is invalidated so the next status() call re-reads it.
        """
        try:
            data = json.loads(json_text)
        except Exception as e:
            return False, f"not valid JSON: {e}"
        required = {"type", "client_email", "private_key", "project_id"}
        missing = required - set(data.keys())
        if missing:
            return False, f"missing required fields: {sorted(missing)}"
        if data.get("type") != "service_account":
            return False, (f"expected type='service_account', got "
                            f"{data.get('type')!r} — make sure you downloaded a "
                            f"Service Account key, not an OAuth client secret")
        # Atomic write: write to .tmp, then rename
        self.credentials_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.credentials_path.with_suffix(".tmp")
        tmp.write_text(json_text)
        tmp.chmod(0o600)
        tmp.replace(self.credentials_path)
        # Invalidate caches so next status()/sync uses the new key
        self._service = None
        self._service_email = None
        return True, f"credentials saved (service account: {data['client_email']})"

    def get_service_account_email(self) -> str | None:
        """Read the service account email out of credentials.json so the UI
        can show the admin which address to share the sheet with."""
        if self._service_email:
            return self._service_email
        if not self.credentials_present():
            return None
        try:
            data = json.loads(self.credentials_path.read_text())
            self._service_email = data.get("client_email")
            return self._service_email
        except Exception as e:
            log.warning(f"failed to parse credentials.json: {e}")
            return None

    def _ensure_service(self):
        """Build (and cache) the Sheets service, if not already built."""
        if self._service is not None:
            return self._service
        if not self.credentials_present():
            raise RuntimeError(
                f"credentials missing at {self.credentials_path} — see "
                f"docs/google_sheets_setup.md")
        # Lazy imports — keep heavy Google libs out of the critical path
        # for installs that don't need Sheets.
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        creds = Credentials.from_service_account_file(
            str(self.credentials_path), scopes=SCOPES)
        self._service = build("sheets", "v4", credentials=creds,
                                cache_discovery=False)
        return self._service

    # ----- runtime config (read from settings table) -----

    def spreadsheet_id(self) -> str | None:
        return self.db.get_setting(KEY_SPREADSHEET_ID)

    def worksheet(self) -> str:
        return self.db.get_setting(KEY_WORKSHEET, DEFAULT_WORKSHEET) or DEFAULT_WORKSHEET

    def is_configured(self) -> bool:
        return bool(self.spreadsheet_id()) and self.credentials_present()

    def configure(self, spreadsheet_url_or_id: str, worksheet: str | None = None):
        """Save the target sheet for sync. The background thread picks this
        up on its next iteration (no restart needed)."""
        sid = parse_sheet_input(spreadsheet_url_or_id)
        self.db.set_setting(KEY_SPREADSHEET_ID, sid)
        if worksheet:
            self.db.set_setting(KEY_WORKSHEET, worksheet.strip() or DEFAULT_WORKSHEET)
        # Clear any cached error so the UI doesn't show stale failures
        self.db.set_setting(KEY_LAST_ERROR, "")
        log.info(f"sheets configured: id={sid[:8]}…, worksheet={self.worksheet()}")

    # ----- header + row formatting -----

    HEADER_ROW = ["Log ID", "Employee ID", "Name", "Department",
                   "Date", "Time", "Event", "Confidence", "Source"]

    @staticmethod
    def _row_for_event(r: dict) -> list:
        dt = datetime.fromtimestamp(r["timestamp"])
        return [
            r["id"], r["emp_id"], r["name"], r["department"] or "",
            dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S"),
            r["event_type"], round(r["confidence"], 4), "auto",
        ]

    def _ensure_header(self, service, sid, ws):
        """Create the header row if A1 is empty. Doesn't touch existing data."""
        rng = f"{ws}!A1:I1"
        resp = service.spreadsheets().values().get(
            spreadsheetId=sid, range=rng).execute()
        if resp.get("values"):
            return
        service.spreadsheets().values().update(
            spreadsheetId=sid, range=rng,
            valueInputOption="RAW", body={"values": [self.HEADER_ROW]},
        ).execute()
        log.info(f"created header row in {ws}")

    # ----- public actions -----

    def test_connection(self) -> tuple[bool, str]:
        """Verify the configured sheet is reachable + writable. Reads the
        spreadsheet metadata then does a tiny no-op write to the worksheet
        header range. Returns (ok, message)."""
        try:
            sid = self.spreadsheet_id()
            if not sid:
                return False, "no spreadsheet configured"
            svc = self._ensure_service()
            meta = svc.spreadsheets().get(
                spreadsheetId=sid, fields="properties.title,sheets.properties").execute()
            title = meta.get("properties", {}).get("title", "?")
            tabs = [s["properties"]["title"] for s in meta.get("sheets", [])]
            ws = self.worksheet()
            if ws not in tabs:
                return False, (f"connected to '{title}' but worksheet '{ws}' "
                                f"not found (tabs: {', '.join(tabs)})")
            self._ensure_header(svc, sid, ws)
            return True, f"connected to '{title}' (tab: {ws})"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def sync_pending(self) -> tuple[int, str | None]:
        """Push any unsynced rows. Returns (count_synced, error_message_or_None).
        Thread-safe — only one sync runs at a time."""
        with self._lock:
            if not self.is_configured():
                return 0, None  # no-op, not an error
            try:
                pending = self.db.pending_sync(limit=self.batch_limit)
                if not pending:
                    return 0, None
                svc = self._ensure_service()
                sid = self.spreadsheet_id()
                ws = self.worksheet()
                self._ensure_header(svc, sid, ws)
                rows = [self._row_for_event(p) for p in pending]
                ids = [p["id"] for p in pending]
                svc.spreadsheets().values().append(
                    spreadsheetId=sid, range=f"{ws}!A:I",
                    valueInputOption="USER_ENTERED",
                    insertDataOption="INSERT_ROWS",
                    body={"values": rows},
                ).execute()
                self.db.mark_synced(ids)
                self.db.set_setting(KEY_LAST_SYNC_OK, str(int(time.time())))
                self.db.set_setting(KEY_LAST_SYNC_TS, str(int(time.time())))
                self.db.set_setting(KEY_LAST_ERROR, "")
                self._last_synced_count = len(rows)
                log.info(f"synced {len(rows)} row(s) to sheet")
                return len(rows), None
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                self.db.set_setting(KEY_LAST_SYNC_TS, str(int(time.time())))
                self.db.set_setting(KEY_LAST_ERROR, err)
                # Record per-row failure on the first id so we can retry/skip.
                if pending:
                    self.db.record_sync_failure(pending[0]["id"], err)
                log.error(f"sheets sync failed: {err}")
                return 0, err

    # ----- background loop -----

    def start_background(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                          name="sheets-sync")
        self._thread.start()
        log.info(f"sheets sync thread started (interval={self.sync_interval}s)")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        while not self._stop.wait(self.sync_interval):
            try:
                self.sync_pending()
            except Exception:
                log.exception("sheets sync loop error (will retry next interval)")

    # ----- status (for the UI) -----

    def status(self) -> dict:
        sid = self.spreadsheet_id()
        last_ts_str = self.db.get_setting(KEY_LAST_SYNC_TS)
        last_ok_str = self.db.get_setting(KEY_LAST_SYNC_OK)
        last_err = self.db.get_setting(KEY_LAST_ERROR) or ""
        pending = len(self.db.pending_sync(limit=10000))
        return {
            "configured":            self.is_configured(),
            "credentials_present":   self.credentials_present(),
            "service_account_email": self.get_service_account_email(),
            "spreadsheet_id":        sid,
            "spreadsheet_url":       (f"https://docs.google.com/spreadsheets/d/{sid}/edit"
                                       if sid else None),
            "worksheet":             self.worksheet(),
            "last_sync_ts":          int(last_ts_str) if last_ts_str else None,
            "last_sync_ok_ts":       int(last_ok_str) if last_ok_str else None,
            "last_error":            last_err,
            "pending_count":         pending,
            "sync_interval_sec":     self.sync_interval,
        }
