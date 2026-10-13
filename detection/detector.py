"""
YOLO-based accident detector with optional ByteTrack tracking.

Wraps the Ultralytics YOLO model, handles device selection,
and returns structured detection results.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

import config
from utils.logger import setup_logger

logger = setup_logger("detector")


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class Detection:
    """A single bounding-box detection."""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


@dataclass
class FrameResult:
    """Aggregated result for one frame of inference."""
    detections: List[Detection] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None
    inference_time_ms: float = 0.0
    has_accident: bool = False
    max_confidence: float = 0.0


# ── Detector ─────────────────────────────────────────────────────

class AccidentDetector:
    """Loads the YOLO model once and exposes a ``detect()`` method.

    Supports optional ByteTrack / BoT-SORT tracking via
    ``model.track(persist=True)``.
    """

    def __init__(self) -> None:
        self.model: Optional[YOLO] = None
        self.device: str = config.DEVICE
        self._loaded: bool = False

    # ── loading ──────────────────────────────────────────────────

    def load(self) -> bool:
        """Load model weights.  Returns *True* on success."""
        try:
            logger.info("Loading model from %s …", config.MODEL_PATH)
            self.model = YOLO(config.MODEL_PATH)

            # Auto-detect device
            if not self.device:
                try:
                    import torch
                    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    self.device = "cpu"

            logger.info("Model loaded  — classes: %s", self.model.names)
            logger.info("Device: %s", self.device)
            self._loaded = True
            return True
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def class_names(self) -> dict:
        return self.model.names if self.model else {}

    # ── inference ────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> FrameResult:
        """Run YOLO (with optional tracking) on *frame*.

        Returns a :class:`FrameResult` with detections, annotated
        frame, latency, and summary flags.
        """
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded — call load() first")

        t0 = time.perf_counter()

        common_kwargs = dict(
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            imgsz=config.INPUT_SIZE,
            device=self.device,
            verbose=False,
        )

        try:
            if config.ENABLE_TRACKING:
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker=config.TRACKER_TYPE,
                    **common_kwargs,
                )
            else:
                results = self.model(frame, **common_kwargs)
        except Exception as exc:
            logger.warning("Inference error (falling back to non-track): %s", exc)
            results = self.model(frame, **common_kwargs)

        inference_ms = (time.perf_counter() - t0) * 1000.0

        # ── extract structured detections ────────────────────────
        result = results[0]
        boxes = result.boxes
        detections: List[Detection] = []

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(boxes)

            for i in range(len(boxes)):
                detections.append(Detection(
                    bbox=xyxy[i],
                    confidence=float(confs[i]),
                    class_id=int(clss[i]),
                    class_name=self.model.names.get(int(clss[i]), "unknown"),
                    track_id=int(ids[i]) if ids[i] is not None else None,
                ))

        has_accident = len(detections) > 0
        max_conf = max((d.confidence for d in detections), default=0.0)

        return FrameResult(
            detections=detections,
            annotated_frame=result.plot(),
            inference_time_ms=inference_ms,
            has_accident=has_accident,
            max_confidence=max_conf,
        )

    def reset_tracker(self) -> None:
        """Reset the internal YOLO tracker state (call between videos)."""
        if self.model is not None:
            try:
                self.model.predictor = None
            except Exception:
                pass
