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
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            log.info(f"camera: /dev/video0 fallback {self.width}x{self.height}")
            return cap.read, cap.release

    def start(self):
        self._reader, self._closer = self._open()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self._reader()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
            # Don't burn CPU faster than the camera produces frames
            time.sleep(1.0 / max(self.framerate, 1))

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._closer:
            self._closer()

    def mjpeg_generator(self, draw_overlay=None, fps: int = 12):
        """Yield multipart MJPEG bytes for HTTP streaming.

        draw_overlay(frame) -> frame  — optional hook to annotate the frame
        (e.g. with face boxes, green tick, oval guide) before streaming.
        """
        boundary = b"--frame"
        interval = 1.0 / fps
        while not self._stop.is_set():
            frame = self.latest_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            if draw_overlay is not None:
                try:
                    frame = draw_overlay(frame)
                except Exception:
                    log.exception("overlay error")
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                continue
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpg.tobytes() + b"\r\n")
            time.sleep(interval)
