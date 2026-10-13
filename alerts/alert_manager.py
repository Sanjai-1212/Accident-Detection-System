"""
Alert manager — sound playback, evidence capture, auto-cleanup.

Alerts are triggered **only** by confirmed accidents (never by
a single frame detection).
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

import config
from utils.logger import setup_logger

logger = setup_logger("alerts")


class AlertManager:
    """Handles alarm sound, evidence screenshots, and event recording."""

    def __init__(self) -> None:
        self._alarm_sound = None
        self._alarm_playing = False
        self._sound_ok = False
        self._last_evidence_path: Optional[str] = None

        os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
        self._init_sound()

    # ── sound ────────────────────────────────────────────────────

    def _init_sound(self) -> None:
        if not config.ENABLE_SOUND:
            return
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            if os.path.exists(config.ALARM_SOUND_PATH):
                self._alarm_sound = pygame.mixer.Sound(config.ALARM_SOUND_PATH)
                self._sound_ok = True
                logger.info("Alarm sound loaded")
            else:
                logger.warning("Alarm sound not found: %s", config.ALARM_SOUND_PATH)
        except Exception as exc:
            logger.warning("Sound init failed: %s", exc)

    def _play_timed(self) -> None:
        """Play alarm for ``SOUND_DURATION_SEC`` then stop."""
        if not self._sound_ok or self._alarm_sound is None:
            return
        if self._alarm_playing:
            return
        try:
            self._alarm_sound.play()
            self._alarm_playing = True

            def _auto_stop():
                time.sleep(config.SOUND_DURATION_SEC)
                self.stop_sound()

            threading.Thread(target=_auto_stop, daemon=True).start()
        except Exception as exc:
            logger.error("Sound playback error: %s", exc)

    def stop_sound(self) -> None:
        """Immediately stop the alarm."""
        if self._alarm_playing and self._alarm_sound is not None:
            try:
                self._alarm_sound.stop()
            except Exception:
                pass
            self._alarm_playing = False

    # ── alert trigger ────────────────────────────────────────────

    def trigger_alert(
        self,
        frame: np.ndarray,
        confidence: float,
        duration: float = 0.0,
    ) -> None:
        """Called when the accident engine enters CONFIRMED state.

        * Plays the alarm sound (auto-stops after configured duration).
        * Saves an evidence screenshot.
        """
        logger.warning(
            "ALERT TRIGGERED  — confidence=%.2f  build-up=%.1fs",
            confidence,
            duration,
        )
        self._play_timed()

        if config.SAVE_EVIDENCE:
            self._save_evidence(frame, confidence)

    # ── evidence ─────────────────────────────────────────────────

    def _save_evidence(self, frame: np.ndarray, confidence: float) -> None:
        try:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fname = f"accident_{ts}_conf{int(confidence * 100)}.jpg"
            fpath = os.path.join(config.EVIDENCE_DIR, fname)
            cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self._last_evidence_path = fpath
            logger.info("Evidence saved: %s", fname)
            self._cleanup_old_evidence()
        except Exception as exc:
            logger.error("Evidence save failed: %s", exc)

    def _cleanup_old_evidence(self) -> None:
        """Delete oldest evidence files when the limit is exceeded."""
        try:
            files = sorted(
                (
                    os.path.join(config.EVIDENCE_DIR, f)
                    for f in os.listdir(config.EVIDENCE_DIR)
                    if f.lower().endswith((".jpg", ".png", ".mp4"))
                ),
                key=os.path.getmtime,
            )
            while len(files) > config.MAX_EVIDENCE_FILES:
                os.remove(files.pop(0))
        except Exception:
            pass

    @property
    def last_evidence_path(self) -> Optional[str]:
        return self._last_evidence_path

    # ── cleanup ──────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release sound resources."""
        self.stop_sound()
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.quit()
        except Exception:
            pass
