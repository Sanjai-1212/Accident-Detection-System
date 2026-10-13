"""
Tracking analysis for accident detection.

Maintains a registry of tracked detection regions and computes
persistence metrics used by the accident engine for evidence scoring.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrackInfo:
    """Accumulated information about one tracked detection region."""
    track_id: int
    first_seen: float          # timestamp
    last_seen: float           # timestamp
    frame_count: int = 1
    peak_confidence: float = 0.0

    @property
    def duration_sec(self) -> float:
        return self.last_seen - self.first_seen


class TrackAnalyzer:
    """Lightweight layer on top of YOLO tracking results.

    It does **not** implement a tracker — YOLO already provides
    ByteTrack / BoT-SORT IDs.  This class records how long each
    tracked region has been visible, which the accident engine
    uses as additional evidence.
    """

    def __init__(self, max_age_sec: float = 10.0) -> None:
        self._tracks: Dict[int, TrackInfo] = {}
        self._max_age = max_age_sec

    # ── public API ───────────────────────────────────────────────

    def update(self, detections: list, timestamp: float) -> None:
        """Feed current-frame detections (with ``track_id``)."""
        seen: set = set()

        for det in detections:
            tid = det.track_id
            if tid is None:
                continue
            seen.add(tid)

            if tid in self._tracks:
                info = self._tracks[tid]
                info.last_seen = timestamp
                info.frame_count += 1
                info.peak_confidence = max(info.peak_confidence, det.confidence)
            else:
                self._tracks[tid] = TrackInfo(
                    track_id=tid,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    frame_count=1,
                    peak_confidence=det.confidence,
                )

        # Purge stale tracks
        expired = [
            tid for tid, info in self._tracks.items()
            if (timestamp - info.last_seen) > self._max_age
        ]
        for tid in expired:
            del self._tracks[tid]

    def get_persistent_tracks(self, min_frames: int = 3) -> List[TrackInfo]:
        """Return tracks seen for at least *min_frames* frames."""
        return [t for t in self._tracks.values() if t.frame_count >= min_frames]

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        """Clear all tracking history."""
        self._tracks.clear()
