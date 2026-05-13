"""Google Sheets sync worker.

Runs as a separate process. Polls the local DB for unsynced rows and pushes
them to Google Sheets via the Sheets API. Designed to be resilient to:
  * intermittent network loss (rows stay in DB until acknowledged)
  * Sheets API rate limits (batches up to 50 rows per request)
  * credential expiry (raises clearly, doesn't silently fail)

Sheet column layout (configured to match what we write here):
    A: Log ID       (local DB row id)
    B: Employee ID
    C: Name
    D: Department
    E: Date         (YYYY-MM-DD, local time)
    F: Time         (HH:MM:SS, local time)
    G: Event        (IN / OUT)
    H: Confidence   (float, 0..1)
    I: Source       ("auto" — manual entries from the web UI use "manual")
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import yaml
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import AttendanceDB


log = logging.getLogger("sync")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def load_config():
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def build_service(creds_path: Path):
    creds = Credentials.from_service_account_file(str(creds_path), scopes=SCOPES)
    # cache_discovery=False avoids a deprecation warning on Pi
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def ensure_header(service, spreadsheet_id: str, worksheet: str):
    """Create the header row if the sheet is empty."""
    range_ = f"{worksheet}!A1:I1"
    resp = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id, range=range_,
    ).execute()
    if resp.get("values"):
        return
    header = [["Log ID", "Employee ID", "Name", "Department",
               "Date", "Time", "Event", "Confidence", "Source"]]
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id, range=range_,
        valueInputOption="RAW", body={"values": header},
    ).execute()
    log.info("created header row")


def rows_from_pending(pending):
    rows = []
    ids = []
    for r in pending:
        dt = datetime.fromtimestamp(r["timestamp"])
        rows.append([
            r["id"],
            r["emp_id"],
            r["name"],
            r["department"] or "",
            dt.strftime("%Y-%m-%d"),
            dt.strftime("%H:%M:%S"),
            r["event_type"],
            round(r["confidence"], 4),
            "auto",
        ])
        ids.append(r["id"])
    return rows, ids


def push_batch(service, spreadsheet_id: str, worksheet: str, rows: list[list]):
    body = {"values": rows}
    service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=f"{worksheet}!A:I",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [sync] %(message)s",
    )
    cfg = load_config()
    project_root = Path(__file__).resolve().parent.parent

    db = AttendanceDB(str(project_root / cfg["paths"]["db"]))
    gs = cfg["google_sheets"]
    creds_path = project_root / gs["credentials_path"]
    if not creds_path.exists():
        log.error(f"credentials missing at {creds_path} — see docs/google_sheets_setup.md")
        sys.exit(2)

    service = build_service(creds_path)
    ensure_header(service, gs["spreadsheet_id"], gs["worksheet"])

    interval = gs["sync_interval_sec"]
    max_retries = gs["max_retries"]

    log.info(f"sync worker started (interval={interval}s)")
    while True:
        try:
            pending = db.pending_sync(limit=50)
            if pending:
                rows, ids = rows_from_pending(pending)
                push_batch(service, gs["spreadsheet_id"], gs["worksheet"], rows)
                db.mark_synced(ids)
                log.info(f"synced {len(rows)} row(s)")
        except HttpError as e:
            # Per-row failure tracking is approximate here — Sheets append is
            # all-or-nothing. We record the failure against the first row.
            log.error(f"Sheets API error: {e}")
            if pending:
                db.record_sync_failure(pending[0]["id"], str(e))
        except Exception as e:
            log.exception(f"unexpected sync error: {e}")

        time.sleep(interval)


if __name__ == "__main__":
    main()
