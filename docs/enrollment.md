# Enrollment guide

## How enrollment works

When a new employee starts, an admin opens the **enrollment wizard** in any browser on the office LAN:

```
http://<pi-ip>:5000/enroll
```

The wizard has three steps:

### Step 1 — Details

Admin types the new hire's Employee ID, full name, and (optionally) department and email. The Employee ID is the unique key — pick a stable scheme like `E001`, `E002`, etc.

### Step 2 — Face capture (the iPhone-style part)

The page shows a **live video feed from the Pi's own camera** with an oval guide. The new hire stands at the kiosk, ~80–120 cm from the camera, looking at it.

The system asks them to hold five different poses in sequence:

1. **Center** — look straight at the camera
2. **Left** — slowly turn the head about 20° to their left
3. **Right** — slowly turn the head about 20° to their right
4. **Up** — slowly tilt the head up
5. **Down** — slowly tilt the head down

For each pose, the wizard:
- Shows a green status badge over the video when the pose is correct
- Auto-captures after the pose is held steady for ~1 second
- Marks the pose pill green with a checkmark
- Advances to the next pose

If the system detects problems, the badge turns orange and shows the reason:
- *"Come closer"* — face is too small in the frame
- *"Move to the center"* — face is off-center
- *"Hold still"* — detection score too low (likely motion blur)
- *"No face detected"* — face not visible
- *"Please remove sunglasses"* — eye region is too dark/uniform to be real eyes
- *"Please remove face mask"* — mouth region is too flat compared to the forehead
- *"Please remove sunglasses and mask"* — both at once

### Step 3 — Done

After all five poses are captured, the system computes the face embeddings (one 512-dim vector per pose) and writes them to the local database. **No photos are stored** by default — embeddings cannot be reversed into images.

The admin can then enroll another employee or return to the dashboard.

## Why these five poses?

Recognition at the kiosk happens with the same camera, but the employee won't always stand perfectly center or face dead-on. By enrolling at five orientations, the system can match someone who's glancing slightly off or wearing the camera at a slight angle. Five samples is the sweet spot — fewer hurts robustness, more adds enrollment time without much accuracy gain.

## Best practices

- **Enroll at the kiosk itself, not from someone's desk.** Same camera, same lighting — much higher accuracy at recognition time.
- **Same conditions as daily use.** If the kiosk faces a window, enroll near the window too.
- **Neutral expression.** A small smile is fine; an exaggerated grin or full laugh is not.
- **Glasses on or off — pick one.** Enroll the way the person looks 90% of the time. If they wear glasses daily, enroll with glasses.
- **No masks, no sunglasses.** The system detects both heuristically (eye-region brightness/variance for sunglasses; mouth-vs-forehead texture ratio for masks) and refuses to enroll those poses. If your office mandates masks day-to-day, you need a different model (mask-aware recognition) — ask before deploying.

## Mask / sunglasses detection at runtime

The same heuristics also run at the kiosk during recognition:

- A face flagged as masked or sunglassed is **not matched against the gallery** at all — the embedding from a covered face is unreliable and could falsely match the wrong enrolled employee.
- The kiosk shows an amber box with the message (e.g. "Please remove face mask") instead of a green tick.
- Once the obstruction is removed and the face is clean for `consecutive_frames`, recognition proceeds normally.

The thresholds live in `src/occlusion.py`. Tune them if your office lighting trips false positives.

## Re-enrolling someone

If recognition is unreliable for a particular employee (e.g. major haircut, new glasses), just run enrollment again with the same Employee ID. The new embeddings fully replace the old.

## Removing an employee

When someone leaves:

1. Open `/` (admin dashboard).
2. Click **Deactivate** next to their name.

This hides them from recognition but keeps their historical attendance logs. For full deletion (per PIPA right-to-erasure):

```bash
sqlite3 data/attendance.db "DELETE FROM employees WHERE emp_id = 'E001';"
sqlite3 data/attendance.db "DELETE FROM attendance WHERE emp_id = 'E001';"
```

## Manual fallback

If the camera fails or someone forgets to clock in, an admin can log an event manually from the dashboard. These rows are marked `source=manual` in the Google Sheet so payroll can distinguish them from automatic logs.

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| Wizard says "no face detected" but I'm there | Camera not connected; ribbon flipped | Check `journalctl -u attendance` for camera errors |
| Pose never advances from "Center" | Yaw threshold mistuned; lighting issue | Move to better lighting; if persistent, lower thresholds in `src/pose.py` |
| Recognition works but tick doesn't appear in kiosk view | Browser didn't reach `/api/state` | Check browser console; ensure same LAN |
| Recognition is wrong (says wrong name) | Match threshold too low | Raise `recognition.match_threshold` in `config.yaml` from 0.42 to 0.45 |
| Recognition rejects a real employee | Match threshold too high, or bad enrollment | Re-enroll that person at the kiosk; if still bad, lower threshold to 0.38 |
