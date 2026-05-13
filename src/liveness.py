"""Light anti-spoofing.

We don't want someone to clock in a colleague using a phone photo.
Two cheap checks that work reasonably well together:

1. Blink detection via Eye Aspect Ratio (EAR) on the 5 facial landmarks
   that InsightFace gives us. Truthfully, 5 points is too coarse for proper
   EAR — but we *do* have the 106-point landmark module loaded, which
   includes the full eye contour. We use those points when available.

2. Frame-to-frame motion within the face bbox. A photo held perfectly still
   would still need to "appear" — but it produces almost no inter-frame
   pixel difference once stable. A real person micro-moves.

For high-security needs you'd want a depth camera (RealSense, OAK-D Lite)
or a proper anti-spoofing model like Silent-Face-Anti-Spoofing. For a
45-person office these heuristics are usually enough.
"""

import time
import numpy as np
import cv2


class LivenessTracker:
    def __init__(self, blink_timeout_sec: float = 4.0,
                 min_motion_px: float = 5.0):
        self.blink_timeout = blink_timeout_sec
        self.min_motion = min_motion_px
        self._first_seen: dict[str, float] = {}
        self._blink_seen: dict[str, bool] = {}
        self._ear_history: dict[str, list[float]] = {}
        self._last_face_crop: dict[str, np.ndarray] = {}
        self._motion_ok: dict[str, bool] = {}

    @staticmethod
    def _eye_aspect_ratio(eye_pts: np.ndarray) -> float:
        # eye_pts: (6, 2) — standard EAR formula
        if eye_pts.shape[0] < 6:
            return 1.0  # not enough points, assume open
        a = np.linalg.norm(eye_pts[1] - eye_pts[5])
        b = np.linalg.norm(eye_pts[2] - eye_pts[4])
        c = np.linalg.norm(eye_pts[0] - eye_pts[3])
        return (a + b) / (2.0 * c + 1e-6)

    def update(self, track_key: str, frame: np.ndarray,
               bbox: tuple, landmarks_106: np.ndarray | None) -> dict:
        """Update liveness state for a track and return its status."""
        now = time.time()
        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):y2, max(0, x1):x2]

        if track_key not in self._first_seen:
            self._first_seen[track_key] = now
            self._blink_seen[track_key] = False
            self._ear_history[track_key] = []
            self._motion_ok[track_key] = False

        # --- Motion check ---
        if track_key in self._last_face_crop and crop.size > 0:
            prev = self._last_face_crop[track_key]
            try:
                resized = cv2.resize(crop, (prev.shape[1], prev.shape[0]))
                diff = float(np.mean(cv2.absdiff(prev, resized)))
                if diff >= self.min_motion:
                    self._motion_ok[track_key] = True
            except cv2.error:
                pass
        self._last_face_crop[track_key] = crop.copy() if crop.size > 0 else \
            self._last_face_crop.get(track_key, crop)

        # --- Blink check (needs 106 landmarks) ---
        if landmarks_106 is not None and len(landmarks_106) >= 106:
            # InsightFace 106-point indices for left/right eye contour
            # (approximate — exact indices depend on the model schema)
            left_eye = landmarks_106[33:39]
            right_eye = landmarks_106[87:93]
            ear = (self._eye_aspect_ratio(left_eye)
                   + self._eye_aspect_ratio(right_eye)) / 2.0
            hist = self._ear_history[track_key]
            hist.append(ear)
            if len(hist) > 20:
                hist.pop(0)
            # Blink = EAR dropped well below baseline and recovered
            if len(hist) >= 5:
                baseline = np.median(hist[:-3]) if len(hist) > 3 else np.median(hist)
                if min(hist[-3:]) < baseline * 0.7 and hist[-1] > baseline * 0.85:
                    self._blink_seen[track_key] = True

        elapsed = now - self._first_seen[track_key]
        return {
            "blink_seen": self._blink_seen[track_key],
            "motion_ok": self._motion_ok[track_key],
            "elapsed_sec": elapsed,
            "timed_out": elapsed > self.blink_timeout,
        }

    def is_live(self, track_key: str) -> bool:
        s = self._blink_seen.get(track_key, False)
        m = self._motion_ok.get(track_key, False)
        return s and m

    def reset(self, track_key: str) -> None:
        for d in (self._first_seen, self._blink_seen, self._ear_history,
                  self._last_face_crop, self._motion_ok):
            d.pop(track_key, None)
