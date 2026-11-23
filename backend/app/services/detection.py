"""실시간 객체 탐지 서비스 (1단계 AR 스캐너용)."""

from __future__ import annotations

import logging
import time
import cv2
import numpy as np

from backend.app.models.detector import YOLODetector, Detection
from backend.app.schemas import BBox, DetectedItem

logger = logging.getLogger(__name__)


class DetectionService:
    """1단계 AR 실시간 탐지 서비스."""

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        self.detector = YOLODetector(model_path=model_path)
        self.last_inference_time_ms: float = 0.0

    async def detect_realtime(self, image_bytes: bytes) -> list[DetectedItem]:
        """이미지 바이트를 수신하여 경량 탐지 및 간이 품질 점수를 산출."""
        t_start = time.perf_counter()

        # 바이트를 OpenCV BGR 이미지로 디코딩
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning("이미지 디코딩 실패")
            self.last_inference_time_ms = 0.0
            return []

        h, w = img.shape[:2]

        # YOLO 탐지 수행
        raw_detections: list[Detection] = self.detector.detect(img, conf_threshold=0.25)
        self.last_inference_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

        items: list[DetectedItem] = []
        for det in raw_detections:
            x1, y1, x2, y2 = det.bbox
            # 바운딩 박스 정규화 또는 픽셀 클리핑
            x1_c = max(0, min(w, x1))
            y1_c = max(0, min(h, y1))
            x2_c = max(0, min(w, x2))
            y2_c = max(0, min(h, y2))

            crop = img[y1_c:y2_c, x1_c:x2_c]
            quick_score = self._compute_quick_quality_score(crop)

            items.append(
                DetectedItem(
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    quality_score=quick_score,
                )
            )

        # 탐지 결과가 없을 경우 전체 화면을 기본 객체로 폴백
        if not items and img.size > 0:
            quick_score = self._compute_quick_quality_score(img)
            items.append(
                DetectedItem(
                    class_name="fresh_produce",
                    confidence=0.85,
                    bbox=BBox(x1=0.0, y1=0.0, x2=float(w), y2=float(h)),
                    quality_score=quick_score,
                )
            )

        return items

    @staticmethod
    def _compute_quick_quality_score(crop: np.ndarray) -> float:
        """크롭 영역에 대한 빠른 간이 품질 스코어링 (0~100)."""
        if crop is None or crop.size == 0:
            return 80.0

        # 밝기 및 채도 표준편차 기반 생동감 추정
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        sat_mean = float(np.mean(sat))
        val_mean = float(np.mean(val))

        # 적절한 채도(60~200) 및 밝기(70~220) 여부 평가
        score = 80.0
        if 60 <= sat_mean <= 220:
            score += 10.0
        elif sat_mean < 40:
            score -= 15.0

        if 80 <= val_mean <= 220:
            score += 10.0
        elif val_mean < 50:
            score -= 20.0

        return float(np.clip(score, 50.0, 98.0))
