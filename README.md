# Face Recognition Attendance System

A Raspberry Pi attendance kiosk with iPhone-Face-ID-style guided enrollment, live recognition with on-screen feedback, and Google Sheets logging.

## How it looks

- **Kiosk view** (`/kiosk`) — Employee walks up, camera draws a green box around their face. When recognition is confident, a green tick appears under the face with their name, and a banner announces "출근 완료" or "퇴근 완료".
- **Enrollment wizard** (`/enroll`) — Admin opens this on a phone or laptop. New hire stands in front of the kiosk camera. The wizard shows a live oval guide and asks them to look center, left, right, up, down. Auto-captures when each pose is held steady for ~1 second.
- **Admin dashboard** (`/`) — List of enrolled employees, recent attendance logs with sync status, manual-entry fallback.

## Architecture

```
                  Pi Camera Module 3
                          │
                          ▼
              ┌─────────────────────────┐
              │  Single Flask process   │
              │  (only one can own      │
              │   the camera at a time) │
              │                         │
              │  • Camera reader thread │
              │  • Recognition thread   │──▶  SQLite (attendance.db)
              │  • Flask routes         │
              │                         │
              └────────────┬────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          /kiosk      /enroll        /  (admin)
       (no auth)   (basic auth)  (basic auth)

                  Separate sync process
                          │
                          ▼
                  ┌───────────────┐
                  │ Google Sheets │
                  └───────────────┘
```

Key idea: **the camera is a singleton resource.** Only one process can read from `picamera2` at a time. So recognition runs as a thread inside the same Flask app that serves the web pages — this way enrollment and recognition share the exact same camera feed and lighting.

## Hardware

- Raspberry Pi 4 (4GB minimum, 8GB recommended)
- Raspberry Pi Camera Module 3 (standard, **not Wide**, **not NoIR**)
- Optional: 7" touchscreen or HDMI monitor for the kiosk view
- Stable mount at ~150 cm height, 80–120 cm from where employees stand
- Even front lighting — avoid windows behind the subject

## Software stack

- **Detection:** RetinaFace (InsightFace `buffalo_sc`) — light enough for Pi 4
- **Recognition:** ArcFace 512-d embeddings, cosine similarity, threshold 0.42
- **Pose estimation:** InsightFace returns yaw/pitch — we bucket into 5 enrollment poses
- **Anti-spoofing:** blink detection + frame-to-frame motion check
- **Web:** Flask + MJPEG streaming + plain HTML/JS (no SPA, no build step)
- **Storage:** SQLite locally (WAL mode for concurrent reads/writes), Google Sheets remotely
- **Service:** systemd

## Performance on Pi 4 (8GB), buffalo_sc, 320x320 detection

- End-to-end recognition latency: ~400–600 ms
- Throughput: ~2–3 recognitions/second — enough for one entrance
- MJPEG preview to browser: 12 fps on the LAN

## Project layout

```
attendance_system/
├── src/
│   ├── camera.py          # Singleton camera source with MJPEG streaming
│   ├── db.py              # SQLite schema + access + report queries
│   ├── face_engine.py     # InsightFace wrapper
│   ├── liveness.py        # Anti-spoofing
│   ├── pose.py            # Pose classification for enrollment
│   ├── occlusion.py       # Heuristic mask/sunglasses detection
│   ├── enroll.py          # CLI enrollment (alternative to the web wizard)
│   ├── recognize.py       # Standalone recognition daemon (alternative to the web app's thread)
│   ├── sync.py            # Google Sheets sync worker (separate process)
│   └── web/
│       ├── app.py         # The main Flask app (camera owner)
│       ├── static/
│       │   └── chart.umd.min.js   # Vendored Chart.js — works offline
│       └── templates/
│           ├── admin.html     # /  — dashboard with daily chart + CSV export
│           ├── employee.html  # /employee/<id> — per-employee report
│           ├── enroll.html    # /enroll — iPhone-style wizard
│           ├── kiosk.html     # /kiosk — recognition view with green tick
│           └── base.html
├── tests/                 # pytest suite (see "Tests" below)
├── config/
│   ├── config.yaml
│   └── credentials.json   # you provide (Google service account)
├── data/
│   └── attendance.db      # auto-created
├── scripts/
│   ├── install.sh
│   ├── attendance.service       # main web app
│   └── attendance-sync.service  # sheets sync worker
├── docs/
│   ├── enrollment.md
│   ├── google_sheets_setup.md
│   ├── remote_access.md         # for admin laptop ↔ Pi
│   └── privacy_pipa.md
├── requirements.txt
├── pytest.ini
├── Makefile               # make install / run / sync / test
└── tasks/                 # build plans + lessons (developer notes)
```

## Quick start

1. Flash Raspberry Pi OS Bookworm 64-bit; connect Camera Module 3; enable camera in `raspi-config`.
2. Clone this project to `/home/pi/attendance_system`.
3. Run `bash scripts/install.sh` (~5 minutes; downloads ~50MB of model files). Or `make install`.
4. Set up Google Sheets (see `docs/google_sheets_setup.md`).
5. Test the web app:
   ```bash
   make run        # equivalent to: PYTHONPATH=. .venv/bin/python src/web/app.py
   ```
6. Open `http://<pi-ip>:5000/enroll` on a phone/laptop and enroll yourself.
7. Open `http://<pi-ip>:5000/kiosk` on the kiosk display.
8. Install systemd services so everything runs on boot.

## URLs

| Path | Auth | Purpose |
|------|------|---------|
| `/`               | admin basic-auth | Dashboard: employee list, recent logs, manual entry, daily chart, CSV export |
| `/enroll`         | admin basic-auth | iPhone-style enrollment wizard |
| `/employee/<id>`  | admin basic-auth | Per-employee report — daily summary, hours chart, print-to-PDF |
| `/kiosk`          | none (LAN only)  | Fullscreen camera view with recognition overlay |
| `/api/state`      | none             | JSON: which faces are currently recognized |
| `/api/stats/daily?days=N` | admin basic-auth | JSON: IN/OUT counts per day for the dashboard chart |
| `/api/export/csv?from=&to=&emp_id=` | admin basic-auth | Streamed CSV download |

`/kiosk` deliberately has no auth — it's the public-facing screen at the door. Make sure your network blocks external access to the Pi.

## Tests

```bash
make test           # fast suite (pure Python, no model load)
make test-slow      # everything including the InsightFace loader test (~50MB download)
```

The fast suite covers the SQLite layer, pose/occlusion classifiers, recognition decision logic, report queries, and the `match()` math. Tests requiring the actual InsightFace model are marked `@pytest.mark.slow` and gated behind `--run-slow`.

## Mask & sunglasses handling

The system detects masks and sunglasses heuristically (eye-region brightness & variance for sunglasses; mouth-vs-forehead Laplacian variance for masks). Behavior:

- **Enrollment** refuses to capture a pose while the face is occluded and shows the reason on screen ("Please remove face mask").
- **Recognition** does not match an occluded face against the gallery at all — it would be unreliable and could falsely match the wrong employee. The kiosk shows an amber box with the prompt instead of a green tick.

Tunables are in `src/occlusion.py`. See `docs/enrollment.md` for the runtime details.
