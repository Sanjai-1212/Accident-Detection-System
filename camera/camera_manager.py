"""
Camera / video source manager.

Supports webcam indices, video files, and RTSP URLs.
Provides FPS measurement and graceful error handling.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional, Tuple, Union

import cv2

import config
from utils.logger import setup_logger

logger = setup_logger("camera")


class CameraManager:
    """Open, read, and release a video source.

    Parameters
    ----------
    source : int | str | None
        Webcam index, file path, or RTSP URL.
        Defaults to ``config.CAMERA_SOURCE``.
    """

    def __init__(self, source: Union[int, str, None] = None) -> None:
        self.source = source if source is not None else config.CAMERA_SOURCE
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps_history: deque = deque(maxlen=60)
        self._last_read_time: float = 0.0
        self._frame_count: int = 0

    # ── open / close ─────────────────────────────────────────────

    def open(self) -> bool:
        """Open the video source. Returns ``True`` on success."""
        try:
            src = self.source
            if isinstance(src, str) and src.isdigit():
                src = int(src)
            self.cap = cv2.VideoCapture(src)

            if not self.cap.isOpened():
                logger.error("Cannot open camera source: %s", self.source)
                return False

            logger.info(
                "Camera opened  — source=%s  resolution=%dx%d  native_fps=%.1f",
                self.source,
                self.width,
                self.height,
                self.native_fps,
            )
            return True
        except Exception as exc:
            logger.error("Camera error: %s", exc)
            return False

    def release(self) -> None:
        """Release the capture device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera released")

    # ── reading ──────────────────────────────────────────────────

    def read(self) -> Tuple[Optional["numpy.ndarray"], float]:
        """Read one frame.

        Returns ``(frame, current_fps)`` or ``(None, 0.0)`` on failure.
        """
        if self.cap is None or not self.cap.isOpened():
            return None, 0.0

        now = time.perf_counter()
        ok, frame = self.cap.read()

        if not ok or frame is None:
            return None, 0.0

        # FPS calculation
        if self._last_read_time > 0:
            dt = now - self._last_read_time
            if dt > 0:
                self._fps_history.append(1.0 / dt)
        self._last_read_time = now
        self._frame_count += 1

        return frame, self.fps

    # ── properties ───────────────────────────────────────────────

    @property
    def fps(self) -> float:
        """Rolling average FPS over the last ~60 frames."""
        if self._fps_history:
            return sum(self._fps_history) / len(self._fps_history)
        return 0.0

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else 0

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else 0

    @property
    def native_fps(self) -> float:
        return float(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap else 0.0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
