# Task: Fix slow kiosk face-detector open after enrollment

## Problem
After finishing enrollment and returning to the kiosk (`MainActivity`), the
face **detector** (MediaPipe `FaceLandmarker`) takes a noticeable moment to
"open" / start recognizing.

## Root cause
- The kiosk `FaceLandmarker` has **no warm-up** — unlike `FaceEmbedder`, which
  primes its GPU shaders in its constructor (`FaceEmbedder.kt:50-58`). The first
  real camera frame after returning pays the full GPU-graph init + shader
  compile (~0.5–1.5s).
- On low-RAM kiosk tablets `MainActivity` can be **destroyed** while
  `EnrollActivity` is on top, so `FaceAnalyzer` (and its landmarker) is rebuilt
  from cold on return — and built **synchronously on the main thread**
  (`MainActivity.kt:123`), freezing the UI.
- Plus the unavoidable camera teardown/rebind on `onResume`.

## Plan
- [x] Make the kiosk `FaceLandmarker` an **app-scoped singleton** in
  `AttendanceApp` (like `embedder`), built + warmed **once**, reused across
  `MainActivity` recreations, never rebuilt.
- [x] Add a **warm-up** dummy `detect()` (blank bitmap) so GPU shaders compile
  off the visible path.
- [x] Build/fetch the landmarker **off the main thread** in `MainActivity`, then
  bind the camera once ready — no UI freeze even on a cold rebuild.
- [x] `FaceAnalyzer.close()` must **not** close the shared landmarker; close it
  in `AttendanceApp.onTerminate`.
- [x] Verify: `./gradlew compileDebugKotlin` → BUILD SUCCESSFUL.
- [x] On-device check (device 5200427e95c446b5):
  - Built + installed, no crashes.
  - `Landmarker warm-up took 88–127ms` logged on TID ≠ main → runs **off the
    main thread**.
  - Forced worst case with `always_finish_activities 1` (destroys MainActivity
    when Settings covers it), opened Settings, went back: timeline shows
    **exactly ONE** "Landmarker warm-up" / "FaceLandmarker backend" — the
    singleton is reused, NOT rebuilt on return. (Reset the setting to 0.)
  - Warm detection after return: `detect=33ms`.

## Review
**Changed files (android-spike):**
- `FaceAnalyzer.kt` — constructor now takes a ready `landmarker` + `backend`
  instead of building one; `buildLandmarker()` + new `warmUp()` moved to the
  companion object; `close()` no longer closes the landmarker (app owns it).
- `AttendanceApp.kt` — added `kioskLandmarker()` (lazy, synchronized,
  warmed-once singleton) + `kioskBackend`; pre-builds it on a background thread
  at startup; frees it in `onTerminate`.
- `MainActivity.kt` — builds the `FaceAnalyzer` on a background thread, then
  binds the camera on the main thread once ready (no UI freeze on cold build).

**Effect:** the kiosk detector's GPU graph is built + shader-warmed exactly once
per process and survives `MainActivity` being destroyed while enrolling, so
returning to the kiosk no longer pays a cold-start stall.

---

## Follow-up investigation: "FaceEmbedder constructed twice"
- **Not a real bug — no fix needed.** App is single-process (no
  `android:process`). On clean `adb logcat -c` + `-b main` captures,
  `FaceEmbedder` is built **exactly once**: across 3 separate launches, and
  across launch→Settings→Enroll→back. The earlier "double" was a stale logcat
  multi-buffer dump from prior installs, not a real second construction.
- **`AndroidManifest.xml` is clean** (42 lines, one `<activity>` each:
  Main/Enroll/Settings/Config). I initially misread garbled tool output as
  duplicate declarations — there are none. Confirmed by direct file Read.
- No code changes resulted from this follow-up.

---

## Fix: enrollment sweep thresholds asymmetric (down hard, left/up easy)
**Root cause:** sweep targets were absolute (from 0°), but the resting head pose
against the kiosk camera isn't 0° (you look slightly up/sideways at the tablet).
A non-zero neutral made one direction trivial and its opposite very hard for the
same threshold magnitude — exactly "down hard, up/left easy".

**Fix (EnrollActivity.kt):** measure the resting `neutralYaw`/`neutralPitch`
during FIT (seed on first frame, light EMA after), then make every sweep target
**relative to neutral**: `handleSweep(value, neutral, ±THRESHOLD)` with progress
`= (value - neutral) / delta`. Now each of L/R/U/D needs the same head movement
regardless of mounting angle / posture. Dropped the unused `monotonic` param.

**Verified:** `assembleDebug` BUILD SUCCESSFUL; installed; EnrollActivity opens,
0 crashes. Final "does it feel balanced" needs an on-face sweep — please test
enrollment and tell me if any direction still feels off (I can then tune
`YAW_THRESHOLD`/`PITCH_THRESHOLD`, currently 15°/12°).

---

## Enhancement: average multiple frames per pose (5→quality, not 5→10)
Decision (user): instead of inflating the gallery to 10 raw images, keep 5 poses
but **average ~3 frames per pose**. Rationale: `EmbeddingStore.bestMatch` takes
the max cosine over ALL of a person's embeddings vs a fixed 0.55 threshold, so
more stored vectors = more recall BUT more false-accept surface. Averaging
denoises each pose vector without growing the gallery (stays 5).

**Changes (EnrollActivity.kt):**
- `FRAMES_PER_POSE = 3`; `poseFrames` buffer; collect N frames per pose then
  store `averageNormalized(poseFrames)` (element-wise mean → L2 re-normalize).
- Burst restarts if the face drops mid-pose.

**Verified:** BUILD SUCCESSFUL (fresh APK 17:36), installed, EnrollActivity
opens, 0 crashes. The denoising benefit itself needs an on-face enrollment to
feel; logic verified to run. `FRAMES_PER_POSE` is the knob if 3 feels slow/fast.

Note: first build attempt failed — `mean[i] /= x` (augmented assign on an array
index) doesn't compile under this Kotlin/array combo ("No set method providing
array access"). Rewrote with explicit `mean[i] = mean[i] / x`. Lesson logged.

---

## Enhancement: loading veil while camera spins up in enrollment
After submitting the ID/name form there's a ~3-4s gap (CameraX provider init +
first frames) where the user saw a black/empty preview. Added a loading veil.

**Changes:**
- `strings.xml`: `camera_starting` = "Kamera tayyorlanmoqda…".
- `activity_enroll.xml`: opaque `#0B1020` `captureLoading` LinearLayout (spinner +
  text) as the last child of the capture FrameLayout (draws on top).
- `EnrollActivity.kt`: `showCaptureLoading()` in `resetCaptureState()` (on entry);
  `hideCaptureLoading()` (250ms fade) on the **first** analyzed frame in
  `handleFrame()`; also hidden in the camera-bind `catch` so it can't get stuck.

**Correction:** my first "verified" pass here was WRONG — the build had actually
FAILED (XML edits didn't apply: anchor strings didn't exist) and `adb install`
reused the stale 17:39 APK, so the screenshots were the old app (both black,
identical bytes). Re-did it properly:
- Patched strings.xml (`camera_starting`) + activity_enroll.xml (`captureLoading`
  veil before the last `</FrameLayout>`) via a Python patcher (robust to exact
  anchors). XML validated well-formed.
- **BUILD SUCCESSFUL, gradle exit 0, APK rebuilt 17:59** (newer than stale).
  Compiles WITH the real `R.id.captureLoading` / `R.string.camera_starting`.

**Verified (what's provable headlessly):**
- BUILD SUCCESSFUL, gradle exit 0, APK rebuilt 17:56, XML well-formed.
- Pulled the INSTALLED apk off the device (`/data/app/...`); `aapt2 dump
  resources` confirms it ships `id/captureLoading` + `string/camera_starting`.
  So the fresh build (not the stale one) is on the device.

**NOT visually verified, and why:** `EnrollActivity` is `android:exported="false"`,
so `adb am start` can't deep-link to it — the capture screen can only be reached
by tapping kiosk→gear→Settings→add-employee→fill form→Continue, which I can't
reliably script. Also camera-preview surfaces screenshot as black via screencap.
→ Needs a human to open enrollment once and confirm the spinner + "Kamera
tayyorlanmoqda…" shows during the ~3-4s before the camera appears.

(Also note: earlier-session "EnrollActivity opens, 0 crashes" checks were hollow
for the same exported=false reason — `am start` was silently blocked.)

---

## FPS optimization — RESULTS (measured on SM-T583, Exynos 7870, Mali-T830)

Baseline (FaceLandmarker 478-pt mesh, GPU, full-res):
  steady-state NO-FACE: ~21 fps, detect ~40ms

After Tier 1 + cheap wins (BlazeFace short-range detector, GPU, 640x480 cap,
kiosk.html removed):
  steady-state NO-FACE: ~25 fps, detect ~32ms   →  +20% fps, -20% detect latency

Experiments run (this is why we measured, not guessed):
- CPU delegate for BlazeFace: WORSE (18 fps / 46ms). GPU wins on this device.
  Reverted to GPU. (My pre-measurement guess of ~40fps was wrong — GPU per-frame
  overhead on the tiny 128x128 tensor caps the gain.)

Key NEW finding (next lever): when a face is PRESENT, the **embedder**
(MobileFaceNet f32, GPU) costs ~105-114ms/frame and drops fps to ~6. Steady-state
fps (no face) is detector-bound; recognition-moment fps is embedder-bound. To
speed up the actual recognition moment, the embedder is the target:
  - INT8 / quantized MobileFaceNet, or a smaller embedder (EdgeFace-XS), OR
  - don't embed every frame — embed every Nth frame / only when bbox is stable.

Files changed:
- FaceAnalyzer.kt: rewritten for MediaPipe FaceDetector (BlazeFace) + PERF counter.
- AttendanceApp.kt: kioskLandmarker() → faceDetector() singleton.
- MainActivity.kt: builds FaceAnalyzer(detector=...); 640x480 analysis cap.
- EnrollActivity.kt: 640x480 analysis cap (enrollment still uses FaceLandmarker
  for sweep POSE — unchanged).
- assets: + blaze_face_short_range.tflite (230KB); - kiosk.html (dead).

ACCURACY CAVEAT (must verify on-face): kiosk now crops from a BlazeFace bbox,
but ENROLLMENT still crops from the FaceLandmarker bbox (EnrollFaceAnalyzer
needs pose). The two bbox conventions differ slightly → embeddings may be
marginally less comparable → could nudge match scores. Re-enroll the existing
user and confirm recognition still fires at MATCH_THRESHOLD=0.55. If matches get
flaky, either (a) make EnrollFaceAnalyzer ALSO crop via the shared BlazeFace
detector (keep landmarker only for yaw/pitch), or (b) retune the threshold.
This is the cleanest remaining correctness task.

STILL VALID PENDING: face_landmarker.task (3.6MB) is still shipped because
enrollment uses it. If enrollment is switched to crop via BlazeFace, that asset
can be dropped too (−3.6MB APK) — but enrollment still needs SOME pose source.

---

## INT8 investigation — measured, decided AGAINST (for this device)

User asked to try INT8. Gated it on a free CPU-vs-GPU embedder benchmark first
(added FaceEmbedder.benchmark(), ran at startup, then removed the startup call;
method kept for future use). Measured on SM-T583 (20 iters each, isolated):

  EMBED-BENCH GPu          ~85 ms
  EMBED-BENCH CPU-2thr     ~370 ms   (this was the OLD default in README!)
  EMBED-BENCH CPU-4thr-XNN ~83 ms

Findings:
- GPU ≈ CPU+XNNPACK (~84ms). The earlier ~110ms was detector contention, not
  delegate overhead. Embedder is ~85ms of real COMPUTE → no free delegate win.
- INT8 only accelerates on CPU/XNNPACK (TFLite GPU delegate is float-only; this
  Exynos has no usable NPU). float32 CPU+XNNPACK is already ~83ms; INT8 best case
  ~45ms — modest — and costs: model download, SYNTHETIC calibration (no face
  crops in repo → real accuracy risk), CPU-only, and RE-ENROLL everyone.
- Verdict: poor reward/risk. Not pursued. If revisited, needs real calibration
  face crops + an accuracy harness, not synthetic data.

No source model in repo (hailo/models/ empty, no .onnx). TF 2.19 / onnx2tf ARE
available on the dev box if we ever do want to convert.

RECOMMENDED cheap win instead (recognition-moment FPS, zero model/accuracy/
enroll cost): gate the embedder — embed every Nth frame or only when the bbox is
stable, instead of every frame. Not yet implemented (awaiting go-ahead).

Net embedder change shipped this round: NONE (benchmark call reverted). Detector
swap + 640x480 + kiosk.html removal from prior round stand.

---

## Embedder throttle (recognition-moment FPS) — IMPLEMENTED

The embedder is ~85ms of real compute (vs detect ~32ms), so embedding EVERY frame
is what drops FPS to ~6 while a face is present. Fix: throttle embeds, keep detect
every frame. Zero model/accuracy/re-enroll cost.

FaceAnalyzer.kt:
- EMBED_MIN_INTERVAL_MS = 150L. With a face present, only crop+embed if >=150ms
  since the last embed (<=~6 embeds/sec). Detect still runs every frame.
- Throttle-skipped frames re-emit the LAST decision (Match if confirmed, else
  FaceVisibleNoMatch) with the FRESH box -> overlay tracks at full detector FPS.
- State lastEmbedAtMs/lastConfirmed/lastMatchId/lastMatchName reset on NoFace
  (next person embeds on frame 1; no stale match lingers).
- Enrollment test-capture (pendingEnrollId) bypasses the throttle.
- Safe re-emit: maybeShowRecognition debounces same-empId; insertEvent is
  first-of-day only -> no double-record.

VERIFIED: BUILD SUCCESSFUL, installed, backend=GPU, no crash.

NOT VERIFIED (honest gaps):
1. Face-present FPS gain — could not put a face in front of the camera headlessly;
   every window showed `embed 0/30`. Human must confirm logcat `PERF ... embed
   N/30` (N>0) shows fps ~18-22 (not ~6) and recognition still fires.
2. PERF ANOMALY: steady-state (no-face) this session = ~14.6 fps / detect ~40ms,
   DOWN from ~25 fps / ~32ms on 05-29. Throttle does NOT touch the no-face path;
   a FRESH process (30s force-stop) still showed ~14.7fps, and CPU0 was at max
   freq. Most likely thermal heat-soak after ~15min of builds/benches on a
   passively-cooled tablet, NOT a code regression — but must be re-measured after
   a real reboot / minutes idle. If it stays ~14fps cold, investigate.
