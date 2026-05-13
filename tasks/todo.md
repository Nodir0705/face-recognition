# Attendance system — gap-closing build plan

Goal: close the gaps surfaced from the codebase review without disturbing the
camera-singleton architecture or the no-SPA / no-build-step philosophy.

## Phase 1 — Foundations (quick, no behavior change)

- [ ] Create `requirements.txt` with pinned versions matching `scripts/install.sh`
- [ ] Create `tests/` with `pytest` config, `conftest.py`, and first smoke tests:
  - [ ] `test_db.py` — schema init, upsert/load embeddings, log_event, sync queue, deactivate
  - [ ] `test_pose.py` — `classify_pose` boundaries, `evaluate_face` rejection reasons
  - [ ] `test_recognize.py` — `decide_event_type` (toggle + day-half modes, cooldown, day-start)
  - [ ] `test_face_engine.py` — `match()` math only (mocked gallery, no model load)
- [ ] Add `Makefile` shortcuts: `make test`, `make run`, `make sync`
- [ ] Delete `src/admin_web.py` and `scripts/attendance-admin.service` (superseded by `src/web/app.py`'s `/` route)
- [ ] Update `README.md` to remove the admin-service references and add a "Tests" section

## Phase 2 — Reports, exports, charts

- [ ] DB helpers in `src/db.py`:
  - [ ] `events_in_range(start_ts, end_ts, emp_id=None)` — JOIN with employees
  - [ ] `daily_summary(emp_id, start_date, end_date)` — first IN, last OUT, total hours per day
  - [ ] `events_per_day(start_ts, end_ts)` — count grouped by date for the dashboard chart
- [ ] `src/web/app.py` routes:
  - [ ] `GET /api/export/csv?from=YYYY-MM-DD&to=YYYY-MM-DD&emp_id=...` — streamed CSV download
  - [ ] `GET /employee/<emp_id>` — per-employee report page (daily summary table + simple chart)
  - [ ] `GET /api/stats/daily?days=30` — JSON counts for dashboard chart
- [ ] Templates:
  - [ ] `templates/employee.html` — summary table + Chart.js bars + print-friendly CSS (`@media print`)
  - [ ] Update `templates/admin.html` — add "events per day (last 30)" Chart.js bar, "Export CSV" form, link names → `/employee/<id>`
- [ ] PDF: lean on browser print-to-PDF (Ctrl+P → "Save as PDF") via `@media print` CSS — no server-side PDF lib. Add a "Print / Save PDF" button on the employee page.
- [ ] Drop Chart.js into `src/web/static/chart.umd.min.js` (single file, no build step) so it works on Pis without internet.

## Phase 3 — Mask / glasses awareness

- [ ] Pick approach (see decision below)
- [ ] Implement detection + UX path
- [ ] Add tests around the heuristic
- [ ] Document the new behavior in `docs/enrollment.md`

## Phase 4 — Wrap

- [ ] Run full test suite
- [ ] Update top of `README.md` with what changed
- [ ] Add a short "Review" section to this `todo.md`

## Decisions to pin down before Phase 3

1. **Mask/glasses approach** — three options with real tradeoffs (see question)
2. **Chart vendoring** — Chart.js as a single static file vs CDN. (Defaulting to local file — kiosk Pis often don't have outbound internet.)
3. **PDF** — Browser print-to-PDF vs server-side ReportLab. (Defaulting to browser print — zero new deps.)

## Out of scope for this round

- Filling `src/web/static/` beyond what's needed (Chart.js + maybe a favicon)
- Mask-tolerant face recognition model swap (would need full retraining/eval)
- Any changes to the camera/recognition core loop unless tests force one

---

## Review (2026-05-13)

**All 11 tasks completed. 43 tests pass, 1 skipped (`--run-slow`).**

### What changed

| Area | File | Change |
|---|---|---|
| Deps | `requirements.txt` (new) | Pinned versions mirroring install.sh; pytest added for dev |
| Tests | `tests/` (new), `pytest.ini` (new), `conftest.py` (new) | First suites: db, pose, recognize, face_engine, reports, occlusion. `--run-slow` flag gates the InsightFace loader. |
| Build | `Makefile` (new) | install / run / sync / test / test-slow / clean |
| Cleanup | deleted `src/admin_web.py`, `scripts/attendance-admin.service` | Both fully superseded by `src/web/app.py`'s `/` route |
| Reports | `src/db.py` | New: `get_employee`, `events_in_range`, `daily_summary`, `events_per_day`. Local-date bucketing via SQLite's `localtime` modifier. |
| Reports | `src/web/app.py` | New routes: `/api/export/csv`, `/api/stats/daily`, `/employee/<id>` |
| Reports | `src/web/templates/employee.html` (new) | Per-employee report: stats cards + Chart.js hours bar + summary table + print-to-PDF CSS |
| Reports | `src/web/templates/admin.html` | Daily IN/OUT chart, CSV export form, employee names link to their report |
| Vendoring | `src/web/static/chart.umd.min.js` (new) | Chart.js 4.4.4 (~200KB) — works without internet on the Pi |
| Mask/glasses | `src/occlusion.py` (new) | Heuristics: dark+uniform eye region (sunglasses); flat mouth-vs-forehead Laplacian variance (mask). 7 unit tests. |
| Mask/glasses | `src/web/app.py` | Enrollment refuses occluded captures; recognition skips matching for occluded faces (avoids false positives); kiosk overlay shows amber "Please remove …" instead of a wrong-name green tick |
| Mask/glasses | `src/web/templates/enroll.html` | Live status badge gives occlusion priority over pose-quality reason |
| Refactor | `src/face_engine.py`, `src/recognize.py` | Deferred heavy imports (InsightFace, cv2) so pure-Python helpers and the pure `match()` math are unit-testable on a dev box without InsightFace installed |
| Docs | `README.md`, `docs/enrollment.md` | New URLs table, Tests section, mask/sunglasses behavior documented |

### What I deliberately did NOT do

- Filling `static/` with extra assets — Chart.js is the only thing actually needed
- Server-side PDF generation — browser print-to-PDF via `@media print` CSS gives the same output with zero new deps
- Mask-tolerant face recognition model swap — out of scope and would require re-enrolling everyone

### Notes for next time

- The `decide_event_type` test for "day-half mode" patches `recognize.datetime` — mirror this pattern for any future time-of-day logic.
- `src/web/app.py` is now ~340 lines and starting to get crowded. If we add another major feature (e.g. multi-camera, audit log middleware), consider splitting routes into a `src/web/views/` subpackage.
- Occlusion thresholds in `src/occlusion.py` were tuned against synthetic test patches. They probably need a small bump on a real Pi in dim lighting — re-tune from real frames after deployment.
