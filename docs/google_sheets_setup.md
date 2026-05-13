# Google Sheets API setup

The sync worker pushes attendance rows to a Google Sheet using a **service account** — no OAuth pop-up, no user login flow, runs headless. This is the right pattern for a kiosk.

## 1. Create a Google Cloud project

1. Go to <https://console.cloud.google.com/>
2. Top-left dropdown → **New Project** → call it e.g. `attendance-system`
3. Wait ~30 sec, switch to that project

## 2. Enable the Sheets API

1. In the search bar at the top, type **Google Sheets API** → Enable

## 3. Create a service account

1. Side menu → **IAM & Admin** → **Service Accounts** → **Create Service Account**
2. Name it `attendance-pi`, click **Create and Continue**
3. Skip the role assignment (we'll grant access on the Sheet itself)
4. Click **Done**
5. Click the new service account → **Keys** tab → **Add Key** → **Create new key** → JSON
6. A `.json` file downloads. **Rename it to `credentials.json`** and place at `config/credentials.json` on the Pi.

The JSON contains a `client_email` field that looks like `attendance-pi@<project>.iam.gserviceaccount.com`. Copy that — you need it in step 5.

## 4. Create the spreadsheet

1. Go to <https://sheets.google.com/> → Blank spreadsheet
2. Rename it (e.g., "Office Attendance 2026")
3. Rename the first tab to **Attendance** (must match `worksheet:` in config.yaml)
4. Copy the spreadsheet ID from the URL:
   `https://docs.google.com/spreadsheets/d/`**`THIS_IS_THE_ID`**`/edit`
5. Paste into `config/config.yaml` under `google_sheets.spreadsheet_id`

## 5. Share the sheet with the service account

1. Click **Share** in Google Sheets
2. Paste the service account's `client_email`
3. Give it **Editor** permission
4. Uncheck "Notify people" → Share

## 6. Test

```bash
source .venv/bin/activate
python src/sync.py
```

You should see `[sync] sync worker started` and `created header row`. Open the sheet — the header row appears.

## Optional: monthly summary

If you also want a formatted summary view (Sheets has the data, but you want pretty totals), add a second sheet tab called `Summary` with a formula like:

```
=QUERY(Attendance!A:I, "SELECT B, C, COUNT(A) WHERE G='IN' GROUP BY B, C LABEL COUNT(A) 'Days present'")
```

That gives a live "days present" tally per employee without any extra code on the Pi.

## Quotas

The Sheets API gives you 60 write requests/minute per project per user. Our sync worker batches up to 50 rows per request and only fires when there are rows pending — even a busy 45-person office won't come close.
