"""채소 정밀 분석 서비스."""

from __future__ import annotations

import logging
import cv2
import numpy as np

from backend.app.models.vegetable_analyzer import VegetableAnalyzer, VegetableQualityResult
from backend.app.schemas import VegetableAnalysisResult, FreshnessGrade

logger = logging.getLogger(__name__)


class VegetableAnalyzerService:
    """채소 신선도 및 이상 탐지 (갈변, 황변, 결로) 서비스."""

    def __init__(self) -> None:
        self.analyzer = VegetableAnalyzer()

    async def analyze(self, image_bytes: bytes, veg_type: str = "general") -> VegetableAnalysisResult:
        """이미지 바이트로부터 채소 신선도 및 결함 분석."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("이미지를 디코딩할 수 없습니다.")

        res: VegetableQualityResult = self.analyzer.analyze(img, veg_type=veg_type)

        if res.freshness_score >= 80:
            grade = FreshnessGrade.EXCELLENT
        elif res.freshness_score >= 60:
            grade = FreshnessGrade.GOOD
        elif res.freshness_score >= 40:
            grade = FreshnessGrade.FAIR
        else:
            grade = FreshnessGrade.POOR

        return VegetableAnalysisResult(
            browning_ratio=res.browning_ratio,
            yellowing_ratio=res.yellowing_ratio,
            condensation_detected=res.condensation_detected,
            freshness_grade=grade,
            overall_score=res.freshness_score,
        )
