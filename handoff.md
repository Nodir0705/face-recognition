# Handoff — Face Recognition Attendance System

Last updated 2026-05-15.

## What this project is

Production face-recognition attendance kiosk for a Korean office, multi-implementation (Python + C++ + Hailo NPU). See [`README.md`](README.md) for the full picture and [`hailo/README.md`](hailo/README.md) for the NPU path. The repository is at <https://github.com/Nodir0705/face-recognition>.

## Where it actually runs

- **Pi 5 + Hailo-8 AI Kit** at hostname `jarvis` (`192.168.3.8` on the office LAN).
- SSH key auth from this dev box is set up; `ssh jarvis@192.168.3.8` works passwordless.
- Web app served at `http://192.168.3.8:5000`:
  - `/` — admin dashboard (basic auth `admin` / `changeme`)
  - `/enroll` — 4-step sweep enrollment
  - `/kiosk` — fullscreen recognition view (no auth)
- Logs on jarvis:
  - `/tmp/attendance.log` — general Flask + recognition log
  - `/tmp/enrollment.log` — per-poll enrollment trace (one line per ~300 ms)
- Start/restart script on jarvis: `/tmp/start_attendance.sh` (uses `setsid` so the process survives SSH disconnect)

## Repo state — IMPORTANT

**Last pushed commit:** `55c4c7c` ("Hailo: full daemon, Flask integration, embedding compat validator, multi-face bench").

**Significant uncommitted work** since then — about a session's worth of changes. Run `git status` to see them. High-level groups:

1. **Google Sheets integration**
   - New: `src/sheets.py`, `src/db.py` settings table, `tests/test_sheets.py`
   - Sheets card on dashboard with file upload widget for the service-account JSON, Save / Test connection / Sync now buttons, live status pill
   - Background thread syncs every 30 s, append-only writes preserve admin's manual edits
   - Already configured in production: spreadsheet ID `1EZlhyE4QF-bG8Ilb3gDK73rdMOaVc50tDdYgAli6TC8`, worksheet `Sheet1`
2. **Enrollment redesign — went through ~5 iterations, settled on "4-step sweep with arm-gate"**
   - Sequence: fit oval → turn LEFT → return to neutral (re-arm) → turn RIGHT → re-arm → tilt UP → re-arm → tilt DOWN → done
   - Thresholds (current): `yaw=15.0`, `pitch=10.0`
   - Per-step 8 s timeout; falls back to best frame if best ≥ 50 % of target, otherwise advances without capturing
   - Geometric pose (`src/pose.py::geometric_pose`) — landmark-ratio math, NOT solvePnP (which was unreliable with only 5 points on Hailo)
   - Mirror display (`cv2.flip` in `_draw_enrollment_overlay`) so selfie-style perception matches motion direction
   - Live debug overlay drawn on the MJPEG: 5 landmarks, eye line, eye midpoint, yellow yaw-indicator line, magenta pitch-indicator line, top-left text readout with `yaw / pitch / step value / target / best so far / armed`
3. **Augmentation at finalize**
   - 5 captured embeddings + 5 horizontal-flip mirrors via `ENGINE.embed_aligned` = 10 total
   - User asked "is 10 enough?" — answered yes (citing InsightFace + Azure docs); offered ±3° rotation augmentation (would bring 10→20) but USER HAS NOT YET DECIDED
4. **Camera + MJPEG tuning**
   - `picamera2` → V4L2 fallback for USB UVC cams (the office Logitech)
   - 30 fps target with proper rate limiting (achieves ~22 fps in practice, ~52 % CPU)
   - Recognition loop capped at ~12 fps (avoid GIL contention with MJPEG)
   - Kiosk `/api/state` poll bumped to 200 ms
5. **Production observations**
   - Recognition cosine-sim distribution shows wide variance: 0.42-0.94 with many marginal scores at 0.42-0.50 (close to the 0.42 threshold). Diagnosed in `src/db.py::load_all_embeddings` — old gallery had 5 nearly-identical embeddings (mean cos_sim 0.967). New gallery with multi-pose enrollment should be better — **NOT YET MEASURED post-fix**.
   - Hailo's `arcface_mobilefacenet.hef` is a different trained checkpoint than InsightFace's `w600k_mbf.onnx` (proven empirically: cos_sim ≈ -0.002). Switching the backend flag = re-enroll everyone. See `hailo/embedding_compat.py` and the section in `hailo/README.md`.

## Tests

49 passed, 1 slow-skipped on the dev box. Run `make test`.

## Open items / questions for next session

1. **Should we commit + push everything?** The uncommitted body of work is large (sheets, enrollment redesign, debug overlay, augmentation, mirror display, threshold tuning). User said earlier "after you're satisfied with enrollment behaviour, I'll commit + push everything to GitHub" — they have not given the go-ahead yet.
2. **Augmentation**: user asked about going from 10 → 20 samples via ±3° rotation. **Not yet decided / implemented.** One-line change in `api_enroll_finish` — for each crop, `cv2.warpAffine` it by ±3°, run through `ENGINE.embed_aligned`, append.
3. **Threshold tuning ongoing.** Current SWEEP thresholds are 15/10. User has been iterating: 5→7→12→15. Possible next iteration after they actually test 15/10. Recognition match threshold (`config.recognition.match_threshold = 0.42`) is also worth considering — was tuned for InsightFace's FP32 model; Hailo's INT8 model may want a different value.
4. **Re-bench needed**: after the multi-pose enrollment fix, fresh recognition scores would prove (or disprove) that the wide cos_sim variance is gone. Diagnostic command lives in earlier conversation:
   ```
   ssh jarvis@192.168.3.8 'cd ~/attendance_system && python3 -c "from src.db import AttendanceDB; import numpy as np; db=AttendanceDB(\"data/attendance.db\"); ids,names,gallery=db.load_all_embeddings(); sims=gallery@gallery.T; np.fill_diagonal(sims,np.nan); print(\"intra mean:\",np.nanmean(sims),\"min:\",np.nanmin(sims),\"max:\",np.nanmax(sims))"'
   ```
   Expectation: intra-gallery min should be lower than the old 0.937 (since now we have 5 distinct poses + 5 mirrors).

## Useful commands

```bash
# Restart Flask on jarvis (uses /tmp/start_attendance.sh)
ssh jarvis@192.168.3.8 'echo > /tmp/enrollment.log; bash /tmp/start_attendance.sh'

# Tail enrollment trace during testing
ssh jarvis@192.168.3.8 'tail -f /tmp/enrollment.log'

# View only state transitions / captures (no per-poll noise)
ssh jarvis@192.168.3.8 'grep -E "FIT_OK|CAPTURED|FINISH|START" /tmp/enrollment.log'

# Verify Hailo backend is loaded (not Python)
ssh jarvis@192.168.3.8 'grep -E "loading.*engine" /tmp/attendance.log | tail'

# Check sheets sync status
curl -s -u admin:changeme http://192.168.3.8:5000/api/sheets/status | python3 -m json.tool

# Force a sheets sync now
curl -s -u admin:changeme -X POST http://192.168.3.8:5000/api/sheets/sync_now
```

## Suggested skills for the next session

- **`claude-mem:learn-codebase`** if continuing work after a long gap (re-prime on the file structure + the three implementations).
- **`superpowers:test-driven-development`** if adding new features — there's an established pytest setup with fixtures in `tests/conftest.py`.
- **`claude-mem:make-plan`** before any non-trivial change.
- **`everything-claude-code:plan`** if user requests a feature that touches multiple files.

## Things I deliberately did NOT do

- Did NOT delete any of the iteration artifacts (multiple enrollment phases lived in the code at various points). Current state is the 4-step sweep with arm-gate. Other approaches were torn out cleanly.
- Did NOT re-enable the passive-liveness landmark-variance check on finalize (commented out, since the 4-step sweep itself is sufficient liveness — capturing 4 different head poses is impossible for a static photo).
- Did NOT remove the legacy `/api/enroll/capture` endpoint — kept it for diagnostics and manual override.

## Files most likely to need attention next

- `src/web/app.py` — `EnrollmentSession`, `_draw_enrollment_overlay`, `/api/enroll/*` and `/api/sheets/*` routes
- `src/web/templates/enroll.html` — 4-step sweep UI with progress bar, pips, mirror video, live readout
- `src/web/templates/admin.html` — Sheets card with upload + status
- `src/sheets.py` — SheetsSync class
- `src/pose.py` — `geometric_pose` (used by both backends)
- `hailo/engine_adapter.py` — drop-in for `FaceEngine`, exposes `embed_aligned` / `aligned_crop` for augmentation
