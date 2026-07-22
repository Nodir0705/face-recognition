# Android Spike — face recognition on a Galaxy Tab A (2016)

Throwaway prototype to answer one question:

> Can we run detect + embed + match **entirely on the tablet**, no Hailo, no server,
> on the cheapest old hardware we have lying around?

If this works acceptably on a 10-year-old Snapdragon 410 with a 2MP front camera,
it'll fly on anything modern (Galaxy Tab A9+ or better).

## Stack

| Stage | Library | Runs on |
|---|---|---|
| Camera preview + analysis frames | CameraX 1.3 (RGBA_8888 output) | CPU |
| Face detection | ML Kit `face-detection:16.1.7` (bundled — model in APK, no Play Services dep) | CPU |
| Face embedding | TFLite 2.14 + MobileFaceNet (112×112 → 192-d) | CPU, 2 threads |
| Match | cosine similarity, in-memory `LinkedHashMap` | CPU |

Total per-frame cost on Tab A 2016: ~150–500ms (detect 80–150ms, embed 100–300ms).
That's ~2–5 fps for the analysis loop, plenty for "stand in front of the kiosk".

## Build

1. Open `android-spike/` in Android Studio Hedgehog or newer.
2. Drop a MobileFaceNet `.tflite` into `app/src/main/assets/mobile_face_net.tflite`.
   See `app/src/main/assets/PLACE_MODEL_HERE.txt` for sources.
3. Connect the Tab A via USB, enable Developer Options → USB debugging.
   - **If install fails with `INSTALL_FAILED_OLDER_SDK`**: the tablet is below API 23
     (some early Tab A 2016 SKUs shipped Android 5.1). Update via Samsung's
     Smart Switch first, or lower `minSdk` to 21 in `app/build.gradle` and the
     `androidx.activity` dep to `1.6.1` (the last version that supported API 21).
4. Run → Run 'app'.
5. Grant camera permission.

## What you'll see

- Full-screen front-camera preview.
- Status bar at the bottom showing detect/embed timings and the match verdict.
- **Enroll next face** — captures the next detected face under id `user_1`, `user_2`, …
- **Clear enrollments** — wipes the in-memory store.

## What you're testing

This spike answers a yes/no question. Try this exact sequence:

1. Stand at ~50cm from the tablet in your actual deployment lighting.
2. Tap **Enroll next face** — your face becomes `user_1`.
3. Walk away, come back. Score should be >0.7 (very confident match).
4. Have **3–5 other people** enroll. They should all match themselves consistently.
5. Each person should *not* match each other (cross-scores < 0.4 ideally, < 0.5 acceptable).
6. Repeat at ~1m distance. **This is where the 2MP camera will likely fail.** If
   matches stay reliable at 1m, the camera is the floor and any newer tablet wins easily.
7. Repeat with glasses on/off and partial smile/neutral — embeddings should be stable.

Record the numbers in `tasks/todo.md` or wherever you track deployment decisions —
those are your input for "do we standardize on this hardware or upgrade".

## Threshold tuning

`FaceAnalyzer.MATCH_THRESHOLD = 0.6f` is a sane starting point for MobileFaceNet.
After running step 5 above, set the threshold halfway between the lowest
same-person score and the highest cross-person score you observed.

## What this spike intentionally does **NOT** do

- Persist enrollments (restart wipes them — use SQLite or sync to Flask in production).
- Anti-spoof / liveness (a photo of someone's face will match).
- Multi-face handling (we pick the largest detected face only).
- Pose / quality gating (the production code in `src/pose.py` should be ported next).
- Network sync to the existing Flask backend / Google Sheets.

All of that is straightforward to add **after** the pipeline itself is validated.
Don't build them into the spike — let it stay throwaway.

## If you want to skip this entirely

Plug the tablet in, sideload the APK once built, and the same yes/no can be
answered in 10 minutes per device. The hard part isn't the code, it's the model
input quality from a $200 device's front camera. Run the spike.
