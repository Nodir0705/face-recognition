"""Shared camera source.

A SINGLE camera can only be opened by one process at a time on the Pi.
Since we want both the recognition daemon AND the web enrollment server to
see live frames, we run the camera inside ONE process (the web server) and
have the recognition logic run there too as a background thread.

This module wraps picamera2 (or USB fallback) and provides:
  * a thread-safe `latest_frame()` getter (always returns the newest BGR frame)
  * an MJPEG generator for streaming to the browser
  * shutdown handling
"""

import threading
import time
import logging
import cv2
import numpy as np


log = logging.getLogger("camera")


class CameraSource:
    def __init__(self, width: int = 1280, height: int = 720, framerate: int = 15):
        self.width = width
        self.height = height
        self.framerate = framerate
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reader = None
        self._closer = None

    def _open(self):
        # Try picamera2 first (CSI Pi camera is its happy path), fall back to
        # cv2.VideoCapture for USB UVC cams. Many USB cams don't support the
        # 1280x720 RGB888 mode picamera2 asks for here, so we ALSO fall back
        # if picamera2 itself raises during configure/start, not just on
        # ImportError.
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            config = picam.create_video_configuration(
                main={"format": "RGB888", "size": (self.width, self.height)},
                controls={"FrameRate": self.framerate},
            )
            picam.configure(config)
            picam.start()
            time.sleep(1.0)

            def read():
                rgb = picam.capture_array()
                return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            def close():
                picam.stop()

            log.info(f"camera: picamera2 {self.width}x{self.height}@{self.framerate}")
            return read, close
        except ImportError:
            log.info("camera: picamera2 not installed, using V4L2 fallback")
        except Exception as e:
            log.warning(f"camera: picamera2 failed ({type(e).__name__}: {e}); "
                        f"falling back to V4L2")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("could not open camera at /dev/video0 either")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.framerate)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"camera: V4L2 /dev/video0  requested {self.width}x{self.height}, "
                 f"got {actual_w}x{actual_h}")
        return cap.read, cap.release

    def start(self):
        self._reader, self._closer = self._open()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        # Sleep just long enough between reads to yield the GIL and not
        # spin a core. cv2.VideoCapture on V4L2 sometimes buffers frames
        # internally and read() returns immediately, so without ANY sleep
        # this loop eats ~75% of a core. With sleep ~half the camera period,
        # we still consume frames as fast as the camera produces them but
        # leave the GIL available for the MJPEG + recognition threads.
        interval = 0.5 / max(self.framerate, 1)
        while not self._stop.is_set():
            ok, frame = self._reader()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
            time.sleep(interval)

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._closer:
            self._closer()

    def mjpeg_generator(self, draw_overlay=None, fps: int = 25,
                         jpeg_quality: int = 60,
                         preview_size: tuple[int, int] | None = None):
        """Yield multipart MJPEG bytes for HTTP streaming.

        draw_overlay(frame) -> frame  — optional hook to annotate the frame
        (e.g. with face boxes, green tick, oval guide) before streaming.

        preview_size — (width, height) to downscale to AFTER overlay drawing
        and BEFORE JPEG encoding. Roughly halves encode cost vs full 1280×720
        when set to (960, 540). Recognition is unaffected because it always
        runs on the full-resolution `latest_frame()`.
        """
        boundary = b"--frame"
        interval = 1.0 / fps
        next_yield = time.perf_counter()
        while not self._stop.is_set():
            frame = self.latest_frame()
            if frame is None:
                time.sleep(0.02)
                continue
            if draw_overlay is not None:
                try:
                    frame = draw_overlay(frame)
                except Exception:
                    log.exception("overlay error")
            if preview_size is not None and \
                (frame.shape[1], frame.shape[0]) != preview_size:
                frame = cv2.resize(frame, preview_size,
                                    interpolation=cv2.INTER_LINEAR)
            ok, jpg = cv2.imencode(".jpg", frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                continue
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpg.tobytes() + b"\r\n")
            # Rate-limit: sleep only the REMAINING time until the next slot,
            # accounting for the work we just did. The previous "always sleep
            # the full interval" pattern was capping us well below `fps`
            # whenever per-frame work was non-trivial (encode, overlay).
            next_yield += interval
            now = time.perf_counter()
            if now < next_yield:
                time.sleep(next_yield - now)
            else:
                # We're behind — reset the slot to NOW so we don't burst-catch-up
                next_yield = now
