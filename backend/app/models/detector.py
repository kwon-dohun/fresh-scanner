"""YOLOv8 object detector wrapper for FreshScanner."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single object detection result."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords
    class_name: str
    confidence: float
    class_id: int


class YOLODetector:
    """Wrapper around Ultralytics YOLOv8 for produce / meat detection.

    Model is loaded lazily on the first call to :meth:`detect`.
    If *model_path* does not exist on disk the official weights are
    auto-downloaded by ultralytics.
    """

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        self._model_path = model_path
        self._model = None  # lazy

    # ------------------------------------------------------------------
    # Lazy loader
    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO

            logger.info("Loading YOLO model from %s …", self._model_path)
            self._model = YOLO(self._model_path)
            logger.info("YOLO model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load YOLO model: %s", exc)
            raise RuntimeError(f"Cannot load YOLO model '{self._model_path}': {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> List[Detection]:
        """Run YOLOv8 inference on a BGR/RGB numpy image.

        Parameters
        ----------
        image:
            HxWxC numpy array (uint8).  Accepts both BGR and RGB —
            ultralytics handles conversion internally.
        conf_threshold:
            Minimum confidence to keep a detection.

        Returns
        -------
        list[Detection]
            Detections sorted by confidence (descending).
        """
        self._ensure_model()

        if image is None or image.size == 0:
            logger.warning("detect() received an empty image.")
            return []

        try:
            results = self._model.predict(
                source=image,
                conf=conf_threshold,
                verbose=False,
            )
        except Exception as exc:
            logger.error("YOLO inference failed: %s", exc)
            return []

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = result.names.get(cls_id, str(cls_id))

                detections.append(
                    Detection(
                        bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        class_name=cls_name,
                        confidence=round(conf, 4),
                        class_id=cls_id,
                    )
                )

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
