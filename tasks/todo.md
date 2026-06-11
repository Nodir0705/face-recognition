# Task: Optimization batch 1 (2026-06-11)

Five small, independent diffs from the optimization review:

- [x] 1. Vectorize `DB.load_all_embeddings()` (src/db.py) — replace nested per-vector loop with vstack
- [x] 2. Add `DB.count_pending_sync()` and use it in `SheetsSync.status()` (src/sheets.py) — stop fetching 10k rows to count
- [x] 3. Wrap recognition loop body in try/except with backoff (src/web/app.py) — no more silent thread death
- [x] 4. Cache last detect() result (50ms TTL) shared by enroll status/capture endpoints (src/web/app.py)
- [x] 5. Batch Hailo ArcFace embedding for multi-face frames (hailo/recognize_hailo.py + engine_adapter.py)
- [x] Verify: pytest — 49 passed, 1 skipped (slow)

## Review

- **db.py** `load_all_embeddings()` rewritten: single `np.vstack` over per-row
  `frombuffer` views instead of a per-vector Python loop. Verified identical
  output (ids, names, matrix) vs old implementation on a synthetic
  100-employee × 10-embedding gallery; 1.38 → 0.86 ms/call on the dev box
  (bigger relative win expected on Pi). Added `count_pending_sync()`.
- **sheets.py** `status()` now uses `COUNT(*)` instead of fetching up to
  10,000 joined rows per dashboard poll.
- **app.py** recognition loop body extracted into a `tick()` closure called
  under `try/except` — a transient engine/DB/frame error now logs and backs
  off 1s instead of killing the daemon thread silently.
- **app.py** new `detect_cached()` (50ms TTL, lock-protected, frame+faces
  cached together) used by `/api/enroll/status` and `/api/enroll/capture`.
  Note: the review claim of "70% redundant inference" was wrong — the
  recognition loop pauses during enrollment. Real benefit: bounds inference
  rate under concurrent pollers (second tab, status+capture overlap).
- **hailo** `HailoArcFace.embed_batch()` embeds all faces of a frame in one
  `infer()` call (N×112×112×3 batch); `embed()` is now a thin wrapper.
  Both the standalone daemon loop and `HailoFaceEngine.detect()` use it.
  ⚠ NOT yet smoke-tested on the Hailo device (no NPU on this dev box) —
  run `recognize_hailo.py` on jarvis before deploying; if HailoRT rejects
  multi-frame input dicts, revert to the per-face `embed()` path.

Tests: `pytest -q` → 49 passed, 1 skipped. All modified files byte-compile.

# Task: Optimization batch 2 (2026-06-12)

From the ultracode second-pass sweep (41 candidates, 34 adversarially refuted,
7 confirmed):

- [x] 1. Enrollment overlay: cached oval mask + cv2.convertScaleAbs blend +
  np.copyto ROI restore (src/web/app.py). Verified: max pixel diff 1
  (round vs truncate), 7.78 → 1.35 ms/frame on dev box (5.8x).
- [x] 2. ~~PRAGMA synchronous=NORMAL after WAL~~ REVERTED after review: a
  non-durable mark_synced commit rolled back by power cut would replay the
  Sheets append → duplicate spreadsheet rows. Benefit was ~1s/day of fsync.
  Kept a comment in db.py documenting the decision.
- [x] 3. NMSBoxes gets numpy float32 directly, no .tolist() (hailo/recognize_hailo.py).
  Verified NMSBoxes accepts ndarrays incl. empty-result path.
- [x] 4. TrackerDet namedtuple replaces per-frame type() class creation
  (hailo/recognize_hailo.py). Tracker only reads .bbox — verified.
- [x] 5. Removed redundant .astype(np.float32) on embed_batch rows
  (hailo/engine_adapter.py)
- [x] 6. kiosk.html MJPEG watchdog: pollState-driven src reset after server
  outage (onerror doesn't fire on frozen MJPEG streams). Hardened after
  review: in-flight poll guard (no stale-fetch races), >=3 consecutive
  failures before reconnect (no single-blip restarts), r.ok check, onerror
  backstop for load-time failures. Known accepted limitation: a TCP reset of
  only the stream socket while Flask stays healthy still freezes the frame —
  no browser event exists for that; a periodic forced refresh would flicker.
- [ ] SKIPPED: preallocating the 640x640 preprocess buffer — verifier measured
  only ~0.06 ms/frame saving (lazy mmap zeroing) and buffer reuse risks stale
  pad pixels with variable input sizes. Not worth the hazard.
- [x] Verify: pytest 49 passed; adversarial review panel on the diff

## Review (batch 2)

Adversarial 3-lens panel on the diff: 0 blockers, 4 minors — all addressed:
- watchdog spurious reconnect on single failed poll → 3-failure threshold
- stale in-flight fetch re-arming the flag after recovery → pollBusy guard
- no r.ok check / no load-failure backstop → added both
- synchronous=NORMAL duplicate-Sheets-rows hazard → reverted the PRAGMA

Verification: pytest 49 passed / 1 skipped; overlay blend benchmarked
7.78 → 1.35 ms/frame (max pixel diff 1); NMSBoxes numpy path tested incl.
empty result; kiosk JS passes node --check.
