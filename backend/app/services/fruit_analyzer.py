"""과일 정밀 분석 서비스."""

from __future__ import annotations

import logging
import cv2
import numpy as np

from backend.app.models.fruit_analyzer import FruitAnalyzer, FruitQualityResult
from backend.app.services.ocr_service import OCRService, PriceTag
from backend.app.services.kamis_api import KAMISClient, PriceComparison
from backend.app.schemas import FruitAnalysisResult, FreshnessGrade

logger = logging.getLogger(__name__)


class FruitAnalyzerService:
    """과일 외관 품질 분석 및 KAMIS 시세 비교 서비스."""

    def __init__(self) -> None:
        self.analyzer = FruitAnalyzer()
        self.ocr_service = OCRService()
        self.kamis_client = KAMISClient()

    async def analyze(self, image_bytes: bytes, fruit_type: str = "apple") -> FruitAnalysisResult:
        """이미지 바이트로부터 과일 표면 품질 분석, 가격표 OCR 및 시세 비교."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("이미지를 디코딩할 수 없습니다.")

        # 1. 외관 품질 분석
        # 지원 타입: watermelon, apple, hallabong, grape, shine_muscat, strawberry
        if fruit_type not in FruitAnalyzer.SUPPORTED_TYPES:
            fruit_type = "apple"

        qual_res: FruitQualityResult = self.analyzer.analyze(img, fruit_type=fruit_type)

        # 2. 가격표 OCR
        price_tag: PriceTag | None = self.ocr_service.extract_price_tag(img)
        store_price = float(price_tag.price) if price_tag and price_tag.price else None

        # 3. KAMIS 시세 비교
        kamis_price = None
        price_eval = "가격표를 인식하지 못했습니다."
        if store_price is not None:
            k_name = "사과"
            if fruit_type == "watermelon":
                k_name = "수박"
            elif fruit_type in ("hallabong", "tangerine", "citrus"):
                k_name = "감귤"
            elif fruit_type in ("grape", "shine_muscat"):
                k_name = "포도"
            elif fruit_type == "strawberry":
                k_name = "딸기"

            comp: PriceComparison = await self.kamis_client.compare_to_market(k_name, store_price)
            kamis_price = float(comp.market_avg) if comp.market_avg > 0 else None
            price_eval = comp.verdict_kr

        # 4. 신선도 등급 결정
        if qual_res.quality_score >= 85:
            grade = FreshnessGrade.EXCELLENT
        elif qual_res.quality_score >= 70:
            grade = FreshnessGrade.GOOD
        elif qual_res.quality_score >= 50:
            grade = FreshnessGrade.FAIR
        else:
            grade = FreshnessGrade.POOR

        defect_ratio = round(qual_res.defect_count * 0.05, 2)

        return FruitAnalysisResult(
            surface_uniformity=qual_res.surface_pattern_score,
            stem_freshness=qual_res.stem_freshness_score,
            defect_ratio=defect_ratio,
            freshness_grade=grade,
            ocr_price=store_price,
            kamis_market_price=kamis_price,
            price_evaluation=price_eval,
        )
