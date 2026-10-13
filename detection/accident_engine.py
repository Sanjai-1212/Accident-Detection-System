"""
Accident Detection State Machine — the core false-positive prevention engine.

State flow
----------
::

    MONITORING  ──►  CANDIDATE  ──►  CONFIRMED  ──►  COOLDOWN  ──►  MONITORING
        ▲                │                                              │
        └────────────────┘  (detection gap exceeded MAX_GAP_FRAMES)    │
        └──────────────────────────────────────────────────────────────┘

* **MONITORING** — no accident evidence; waiting for detections.
* **CANDIDATE**  — consecutive detection frames accumulating;
                   resets if gap exceeds ``MAX_GAP_FRAMES``.
* **CONFIRMED**  — enough evidence gathered; alert should fire.
                   Holds for ``CONFIRMED_DISPLAY_SEC`` seconds.
* **COOLDOWN**   — suppresses duplicate alerts; returns to
                   MONITORING after ``ALERT_COOLDOWN_SEC``.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import config
from utils.logger import setup_logger

logger = setup_logger("engine")


class AccidentState(Enum):
    MONITORING = "monitoring"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    COOLDOWN = "cooldown"


@dataclass
class AccidentEvent:
    """A single confirmed accident record."""
    timestamp: float
    confidence: float
    duration_sec: float
    evidence_frames: int


class AccidentEngine:
    """Multi-frame accident confirmation engine.

    Call :meth:`update` once per frame with the detection result.
    The method returns an event string whenever a state transition
    occurs, or ``None`` when nothing changed.

    Returned events
    ---------------
    ``"candidate_start"``
        First suspicious frame detected.
    ``"candidate_rejected"``
        Detection gap exceeded — back to monitoring.
    ``"confirmed"``
        Accident confirmed after enough consecutive evidence.
    ``"cooldown_start"``
        Confirmed display period ended; entering cooldown.
    ``"cooldown_end"``
        Cooldown expired; system returns to monitoring.
    """

    def __init__(self) -> None:
        # ── state ────────────────────────────────────────────────
        self.state: AccidentState = AccidentState.MONITORING
        self.evidence_count: int = 0       # consecutive positive frames
        self.gap_count: int = 0            # consecutive negative frames
        self.candidate_start: float = 0.0
        self.confirmed_start: float = 0.0
        self.cooldown_start: float = 0.0
        self.peak_confidence: float = 0.0

        # ── history ──────────────────────────────────────────────
        self.confidence_history: deque = deque(maxlen=60)
        self.events: List[AccidentEvent] = []

        # ── counters ─────────────────────────────────────────────
        self.total_confirmed: int = 0
        self.total_candidates_rejected: int = 0

        # ── thresholds (from config) ─────────────────────────────
        self.confirmation_frames: int = config.CONFIRMATION_FRAMES
        self.max_gap: int = config.MAX_GAP_FRAMES
        self.confirmed_display_sec: float = config.CONFIRMED_DISPLAY_SEC
        self.cooldown_sec: float = config.ALERT_COOLDOWN_SEC

    # ── main entry point ─────────────────────────────────────────

    def update(
        self,
        has_detection: bool,
        confidence: float,
        timestamp: Optional[float] = None,
    ) -> Optional[str]:
        """Process one frame and return an event string or ``None``."""
        if timestamp is None:
            timestamp = time.time()

        self.confidence_history.append(confidence)

        handler = {
            AccidentState.MONITORING: self._on_monitoring,
            AccidentState.CANDIDATE: self._on_candidate,
            AccidentState.CONFIRMED: self._on_confirmed,
            AccidentState.COOLDOWN: self._on_cooldown,
        }[self.state]

        return handler(has_detection, confidence, timestamp)

    # ── state handlers ───────────────────────────────────────────

    def _on_monitoring(
        self, has_det: bool, conf: float, ts: float
    ) -> Optional[str]:
        if has_det:
            self.state = AccidentState.CANDIDATE
            self.evidence_count = 1
            self.gap_count = 0
            self.candidate_start = ts
            self.peak_confidence = conf
            logger.info(
                "Candidate started  (conf=%.2f)", conf,
            )
            return "candidate_start"
        return None

    def _on_candidate(
        self, has_det: bool, conf: float, ts: float
    ) -> Optional[str]:
        if has_det:
            self.evidence_count += 1
            self.gap_count = 0
            self.peak_confidence = max(self.peak_confidence, conf)

            if self.evidence_count >= self.confirmation_frames:
                # ── CONFIRMED ────────────────────────────────────
                self.state = AccidentState.CONFIRMED
                self.confirmed_start = ts
                self.total_confirmed += 1
                duration = ts - self.candidate_start

                self.events.append(AccidentEvent(
                    timestamp=ts,
                    confidence=self.peak_confidence,
                    duration_sec=duration,
                    evidence_frames=self.evidence_count,
                ))
                logger.warning(
                    "ACCIDENT CONFIRMED  conf=%.2f  evidence=%d frames  "
                    "build-up=%.1fs",
                    self.peak_confidence,
                    self.evidence_count,
                    duration,
                )
                return "confirmed"
        else:
            self.gap_count += 1
            if self.gap_count > self.max_gap:
                self._reject_candidate()
                return "candidate_rejected"
        return None

    def _on_confirmed(
        self, _has_det: bool, _conf: float, ts: float
    ) -> Optional[str]:
        # Hold CONFIRMED for visual feedback, then transition.
        if (ts - self.confirmed_start) >= self.confirmed_display_sec:
            self.state = AccidentState.COOLDOWN
            self.cooldown_start = ts
            self.evidence_count = 0
            self.gap_count = 0
            logger.info("Entering cooldown (%.0fs)", self.cooldown_sec)
            return "cooldown_start"
        return None

    def _on_cooldown(
        self, _has_det: bool, _conf: float, ts: float
    ) -> Optional[str]:
        if (ts - self.cooldown_start) >= self.cooldown_sec:
            self.state = AccidentState.MONITORING
            self.peak_confidence = 0.0
            logger.info("Cooldown expired — back to monitoring")
            return "cooldown_end"
        return None

    # ── helpers ───────────────────────────────────────────────────

    def _reject_candidate(self) -> None:
        self.state = AccidentState.MONITORING
        self.total_candidates_rejected += 1
        self.evidence_count = 0
        self.gap_count = 0
        self.peak_confidence = 0.0
        logger.info(
            "Candidate rejected  (gap > %d frames)", self.max_gap,
        )

    @property
    def cooldown_remaining(self) -> float:
        """Seconds left in cooldown, or 0."""
        if self.state == AccidentState.COOLDOWN:
            return max(0.0, self.cooldown_sec - (time.time() - self.cooldown_start))
        return 0.0

    @property
    def avg_confidence(self) -> float:
        if self.confidence_history:
            return sum(self.confidence_history) / len(self.confidence_history)
        return 0.0

    def reset(self) -> None:
        """Reset engine to initial monitoring state."""
        self.state = AccidentState.MONITORING
        self.evidence_count = 0
        self.gap_count = 0
        self.peak_confidence = 0.0
        self.candidate_start = 0.0
        self.confirmed_start = 0.0
        self.cooldown_start = 0.0
        self.confidence_history.clear()
