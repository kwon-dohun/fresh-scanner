"""육류 정밀 분석 서비스."""

from __future__ import annotations

import logging
import cv2
import numpy as np

from backend.app.models.meat_segmentor import MeatSegmentor, MeatSegResult
from backend.app.models.optical_calibrator import WhiteLabelReferenceWB
from backend.app.services.ocr_service import OCRService, MeatLabel
from backend.app.schemas import MeatAnalysisResult, FreshnessGrade

logger = logging.getLogger(__name__)


class MeatAnalyzerService:
    """육류 정밀 분석 서비스 (세그멘테이션 + 라벨 OCR + 신선도 등급)."""

    def __init__(self) -> None:
        self.segmentor = MeatSegmentor(use_sam=False)
        self.ocr_service = OCRService()

    async def analyze(self, image_bytes: bytes) -> MeatAnalysisResult:
        """이미지 바이트로부터 살코기/지방/뼈 분할, 라벨 OCR 및 점수 산출."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("이미지를 디코딩할 수 없습니다.")

        # 0. 정육 팩 흰색 가격 라벨 기준 광학 화이트밸런스 보정 (붉은 조명 보정)
        calibrated_img, is_wb_calibrated = WhiteLabelReferenceWB.calibrate_with_label(img)
        analysis_img = calibrated_img if is_wb_calibrated else img

        # 1. 픽셀 단위 세그멘테이션
        seg_res: MeatSegResult = self.segmentor.segment(analysis_img)

        # 2. 정육 라벨 OCR 분석
        label: MeatLabel | None = self.ocr_service.extract_meat_label(img)
        price_per_100g = label.price_per_100g if label else None
        ocr_text = f"단가: {price_per_100g}원/100g, 부위: {label.cut_type if label else '미식별'}" if label else ""

        # 3. 마블링 및 색상 점수 산출
        # 구이용 기준 최적 지방 비율은 20%~30% 내외
        fat_penalty = abs(seg_res.fat_ratio - 0.25) * 40
        marbling_score = max(1.0, min(10.0, 9.0 - fat_penalty))

        # 색상 점수
        if "fresh" in seg_res.color_assessment or "선홍색" in seg_res.color_assessment:
            color_score = 95.0
            grade = FreshnessGrade.EXCELLENT
        elif "aged" in seg_res.color_assessment or "적갈색" in seg_res.color_assessment:
            color_score = 75.0
            grade = FreshnessGrade.GOOD
        elif "browning" in seg_res.color_assessment or "갈변" in seg_res.color_assessment:
            color_score = 45.0
            grade = FreshnessGrade.POOR
        else:
            color_score = 80.0
            grade = FreshnessGrade.FAIR

        if seg_res.has_blood_spots:
            color_score = max(0.0, color_score - 10.0)

        return MeatAnalysisResult(
            lean_ratio=seg_res.lean_ratio,
            fat_ratio=seg_res.fat_ratio,
            bone_ratio=seg_res.bone_ratio,
            marbling_score=round(marbling_score, 1),
            color_score=round(color_score, 1),
            price_per_100g=price_per_100g,
            ocr_label_text=ocr_text,
            freshness_grade=grade,
        )
