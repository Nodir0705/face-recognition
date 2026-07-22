# Sheets + Telegram setup (one-time per kiosk)

The kiosk POSTs every IN/OUT event to two places: a Google Apps Script
webhook (writes a row to your sheet) and the Telegram Bot API (sends a
message to your admin chat). Both are optional; events that can't be
delivered are queued in SQLite and retried on each subsequent event or
app launch.

## 1. Google Sheets

### Create the sheet
1. Open a new Google Sheet. The first row should be your headers:

       Timestamp | Employee ID | Name | Event

   You can also leave it empty — the script appends rows regardless.

### Bind an Apps Script

1. Extensions → **Apps Script**
2. Replace any existing code with the script below.
3. Click **Deploy** → **New deployment** → gear icon → **Web app**:
   - Description: anything (e.g. "Attendance kiosk")
   - Execute as: **Me**
   - Who has access: **Anyone**
4. Click **Deploy** → grant the requested permissions → copy the **Web app URL**
   (looks like `https://script.google.com/macros/s/AKfy.../exec`).
5. Paste that URL into the kiosk: gear icon → **Konfiguratsiya** → Google Sheet
   field → **Saqlash**.

### Apps Script

```js
function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    sheet.appendRow([
      data.ts || new Date(data.ts_epoch_ms).toISOString(),
      data.emp_id || '',
      data.name || '',
      data.event || ''
    ]);
    return ContentService
      .createTextOutput(JSON.stringify({ ok: true }))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService
      .createTextOutput(JSON.stringify({ ok: false, error: String(err) }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}
```

The tablet sends JSON like:
```json
{
  "emp_id": "E001",
  "name": "Nodir",
  "event": "IN",
  "ts": "2026-05-28T14:23:45+05:00",
  "ts_epoch_ms": 1748424225000
}
```

### Updating the script

If you change the script, you must **Deploy → Manage deployments → pencil
icon → "New version" → Deploy** for the change to take effect. Keeping the
same Web app URL is fine.

## 2. Telegram

### Create the bot
1. In Telegram, talk to **@BotFather**: `/newbot` → pick a name and username
   (must end in `bot`). BotFather replies with a token like
   `1234567890:ABCdefGhIjKlMnOpQrStUvWxYzAbCdEfGh`.
2. Save that token securely. **This is what goes in the "Bot tokeni" field.**

### Find your chat ID
1. Send any message to your new bot (open it via the username, tap Start, send "hi").
2. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser.
3. Find `"chat":{"id":...}` in the JSON. That number is your chat ID.
   **That's what goes in the "Admin chat ID" field.**
4. For a group: add the bot to the group, then `/start@yourbotname` in the
   group, then check `getUpdates` again — the group chat ID will be negative.

### Paste into the kiosk
Gear icon → **Konfiguratsiya** → Bot tokeni + Admin chat ID → **Saqlash**.

You'll start receiving messages like:
```
🟢 Nodir kirdi · 14:23
🔴 Nodir chiqdi · 18:05
```

## What the kiosk does with these

- On every recognition match (IN or OUT), the event is written to SQLite
  immediately. The green pill shows regardless of sync state.
- A background pass tries to POST to Sheets and Telegram. Both are
  independent — if Sheets is down, Telegram still goes through, and the
  Sheets row gets retried on the next event or app launch.
- Each row in the events table tracks `sheets_sent` and `telegram_sent`
  flags independently. Until both are 1, the row is "pending" and stays
  in the retry queue.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| No row appearing in Sheet | Wrong Web app URL, or it points to an older deployment | Re-deploy ("New version"), re-paste URL |
| 403 / "Authorization required" in adb logcat | Apps Script not deployed as "Anyone" | Re-deploy with access = Anyone |
| No Telegram message | Wrong bot token / chat ID | Double-check `getUpdates` output |
| Messages stop arriving after admin blocks bot | Telegram returns 403 forbidden | Unblock the bot, re-send /start |
| Events stack up offline | Expected — they flush on next sync trigger | Reconnect WiFi and approach the kiosk once |
