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
