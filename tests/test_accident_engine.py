"""
Unit tests for the AccidentEngine state machine.

Run with:  python -m pytest tests/ -v
Or:        python -m unittest tests.test_accident_engine -v
"""
from __future__ import annotations

import unittest

# Patch config before importing engine so tests are self-contained
import sys, types  # noqa: E401,E402
_cfg = types.ModuleType("config")
_cfg.CONFIRMATION_FRAMES = 5
_cfg.MAX_GAP_FRAMES = 2
_cfg.CONFIRMED_DISPLAY_SEC = 1.0
_cfg.ALERT_COOLDOWN_SEC = 5.0
_cfg.LOG_LEVEL = "WARNING"
_cfg.LOG_TO_FILE = False
_cfg.LOG_DIR = "logs"
sys.modules["config"] = _cfg

from detection.accident_engine import AccidentEngine, AccidentState  # noqa: E402


class TestAccidentEngine(unittest.TestCase):
    """Core state-machine tests."""

    def setUp(self):
        self.engine = AccidentEngine()
        self.t = 0.0  # simulated timestamp

    def _tick(self, has_det: bool, conf: float = 0.8, dt: float = 0.033):
        """Advance one simulated frame."""
        self.t += dt
        return self.engine.update(has_det, conf, self.t)

    # ── 1. Normal traffic → NO ALERT ────────────────────────────

    def test_no_detection_stays_monitoring(self):
        for _ in range(100):
            event = self._tick(False, 0.0)
            self.assertIsNone(event)
        self.assertEqual(self.engine.state, AccidentState.MONITORING)

    # ── 2. Single false detection → NO ALERT ────────────────────

    def test_single_frame_detection_no_alert(self):
        self._tick(True)   # 1 frame
        self._tick(False)
        self._tick(False)
        self._tick(False)  # gap > MAX_GAP
        self.assertEqual(self.engine.state, AccidentState.MONITORING)
        self.assertEqual(self.engine.total_confirmed, 0)

    # ── 3. Short suspicious event → NO ALERT ────────────────────

    def test_short_candidate_rejected(self):
        # 3 frames detection (< CONFIRMATION_FRAMES=5)
        for _ in range(3):
            self._tick(True)
        self.assertEqual(self.engine.state, AccidentState.CANDIDATE)
        # Then gap
        for _ in range(3):
            self._tick(False)
        self.assertEqual(self.engine.state, AccidentState.MONITORING)
        self.assertEqual(self.engine.total_candidates_rejected, 1)
        self.assertEqual(self.engine.total_confirmed, 0)

    # ── 4. Persistent evidence → ALERT ───────────────────────────

    def test_confirmed_after_enough_frames(self):
        events = []
        for _ in range(5):
            ev = self._tick(True, 0.85)
            if ev:
                events.append(ev)

        self.assertIn("candidate_start", events)
        self.assertIn("confirmed", events)
        self.assertEqual(self.engine.state, AccidentState.CONFIRMED)
        self.assertEqual(self.engine.total_confirmed, 1)
        self.assertAlmostEqual(self.engine.peak_confidence, 0.85, places=2)

    # ── 5. Duplicate frames → no duplicate alerts ────────────────

    def test_no_duplicate_alerts(self):
        # Confirm first
        for _ in range(5):
            self._tick(True)
        self.assertEqual(self.engine.total_confirmed, 1)

        # Continue detecting — should NOT confirm again
        for _ in range(20):
            self._tick(True)
        self.assertEqual(self.engine.total_confirmed, 1)

    # ── 6. Cooldown → no repeated alerts ─────────────────────────

    def test_cooldown_suppresses_new_alerts(self):
        # First accident
        for _ in range(5):
            self._tick(True)
        # Wait through confirmed display
        for _ in range(60):
            self._tick(True, dt=0.033)

        self.assertIn(
            self.engine.state,
            {AccidentState.COOLDOWN, AccidentState.CONFIRMED},
        )
        # No second confirmation during cooldown
        self.assertEqual(self.engine.total_confirmed, 1)

    # ── 7. Accident ends → back to monitoring ────────────────────

    def test_returns_to_monitoring_after_cooldown(self):
        # Confirm
        for _ in range(5):
            self._tick(True)
        # Advance past confirmed display (1s)
        self._tick(False, dt=1.5)
        self.assertEqual(self.engine.state, AccidentState.COOLDOWN)

        # Advance past cooldown (5s)
        self._tick(False, dt=6.0)
        self.assertEqual(self.engine.state, AccidentState.MONITORING)

    # ── 8. Gap tolerance works ───────────────────────────────────

    def test_gap_tolerance(self):
        """Small gaps (≤ MAX_GAP) don't reset the candidate."""
        self._tick(True)   # evidence=1
        self._tick(True)   # evidence=2
        self._tick(False)  # gap=1 (allowed)
        self._tick(True)   # evidence=3, gap reset
        self._tick(True)   # evidence=4
        self._tick(True)   # evidence=5 → CONFIRMED
        self.assertEqual(self.engine.state, AccidentState.CONFIRMED)
        self.assertEqual(self.engine.total_confirmed, 1)

    # ── 9. Reset works ───────────────────────────────────────────

    def test_reset(self):
        for _ in range(5):
            self._tick(True)
        self.engine.reset()
        self.assertEqual(self.engine.state, AccidentState.MONITORING)
        self.assertEqual(self.engine.evidence_count, 0)

    # ── 10. Peak confidence tracked ──────────────────────────────

    def test_peak_confidence_tracked(self):
        self._tick(True, 0.50)
        self._tick(True, 0.70)
        self._tick(True, 0.90)
        self._tick(True, 0.60)
        self._tick(True, 0.55)
        self.assertAlmostEqual(self.engine.peak_confidence, 0.90, places=2)


if __name__ == "__main__":
    unittest.main()
