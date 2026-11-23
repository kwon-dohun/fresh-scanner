"""정밀 품질 분석 및 설명 가능한 AI (XAI) 3단 브리핑 오케스트레이터."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
import cv2
import numpy as np

from backend.app.config import settings
from backend.app.models.detector import YOLODetector, Detection
from backend.app.models.meat_segmentor import MeatSegmentor, MeatSegResult
from backend.app.models.fruit_analyzer import FruitAnalyzer, FruitQualityResult
from backend.app.models.vegetable_analyzer import VegetableAnalyzer, VegetableQualityResult
from backend.app.services.ocr_service import OCRService
from backend.app.schemas import (
    PreciseScanResponse,
    FreshnessGrade,
    XAIExplanation,
)

logger = logging.getLogger(__name__)


class QualityAnalysisService:
    """2단계 정밀 분석 및 XAI 보고서 생성 서비스."""

    def __init__(self) -> None:
        self.detector = YOLODetector()
        self.meat_segmentor = MeatSegmentor(use_sam=False)
        self.fruit_analyzer = FruitAnalyzer()
        self.vegetable_analyzer = VegetableAnalyzer()
        self.ocr_service = OCRService()

    async def analyze_precise(self, image_bytes: bytes) -> PreciseScanResponse:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("이미지를 디코딩할 수 없습니다.")

        detections: list[Detection] = self.detector.detect(img, conf_threshold=0.20)
        class_name = detections[0].class_name if detections else "meat"
        category = self._categorize(class_name)

        explanations: list[XAIExplanation] = []
        overall_score = 88.0
        grade = FreshnessGrade.GOOD
        mask_url = None

        if category == "meat":
            seg_res: MeatSegResult = self.meat_segmentor.segment(img)
            lean_pct = int(seg_res.lean_ratio * 100)
            fat_pct = int(seg_res.fat_ratio * 100)
            bone_pct = int(seg_res.bone_ratio * 100)

            explanations.append(
                XAIExplanation(
                    factor="[황금 비율 & 마블링]",
                    score=float(min(lean_pct + seg_res.bms_grade * 2, 98)),
                    description=f"살코기 {lean_pct}% : 지방 {fat_pct}% (뼈 {bone_pct}%) 비율이며, 근내 마블링 등급은 BMS No.{seg_res.bms_grade} 수준입니다.",
                )
            )

            defect_desc = "한쪽에 뭉친 떡지방이 발견되지 않았습니다."
            if seg_res.has_tteok_jibang:
                defect_desc = f"주의: 한쪽에 뭉친 떡지방이 약 {int(seg_res.tteok_jibang_area_ratio * 100)}% 면적으로 감지되었습니다."

            explanations.append(
                XAIExplanation(
                    factor="[결함 및 떡지방]",
                    score=92.0 if not seg_res.has_tteok_jibang else 58.0,
                    description=defect_desc,
                )
            )

            explanations.append(
                XAIExplanation(
                    factor="[특수조명 역보정 육색]",
                    score=95.0 if "선홍" in seg_res.color_assessment else 70.0,
                    description=f"매대 조명 보정 후 실제 육색: {seg_res.color_assessment}.",
                )
            )

            overall_score = round(
                (lean_pct * 0.35)
                + (92 if not seg_res.has_tteok_jibang else 55) * 0.35
                + (seg_res.bms_grade * 3.5)
                + 18.0,
                1,
            )
            overall_score = float(np.clip(overall_score, 50.0, 98.0))
            grade = FreshnessGrade.EXCELLENT if overall_score >= 88 else FreshnessGrade.GOOD
            mask_url = self._save_colored_mask(img, seg_res)

        elif category == "fruit":
            f_type = "watermelon" if "watermelon" in class_name.lower() else "apple"
            fruit_res: FruitQualityResult = self.fruit_analyzer.analyze(img, fruit_type=f_type)
            overall_score = fruit_res.quality_score

            explanations.append(
                XAIExplanation(
                    factor="[표면 줄무늬 & 색택]",
                    score=fruit_res.surface_pattern_score,
                    description=f"가보 필터 기반 패턴 분석 결과 선명도 {fruit_res.surface_pattern_score}점, 숙기 상태: {fruit_res.ripeness_level}.",
                )
            )
            explanations.append(
                XAIExplanation(
                    factor="[꼭지 신선도]",
                    score=fruit_res.stem_freshness_score,
                    description=f"꼭지/줄기 녹색도 및 탈수 분석 스코어 {fruit_res.stem_freshness_score}점.",
                )
            )
            explanations.append(
                XAIExplanation(
                    factor="[표면 결함 검출]",
                    score=max(0.0, 100.0 - fruit_res.defect_count * 12),
                    description=f"멍 또는 상처 의심 영역 {fruit_res.defect_count}개 발견.",
                )
            )
            grade = FreshnessGrade.EXCELLENT if overall_score >= 85 else FreshnessGrade.GOOD

        else:  # vegetable
            veg_res: VegetableQualityResult = self.vegetable_analyzer.analyze(img, veg_type="general")
            overall_score = veg_res.freshness_score

            explanations.append(
                XAIExplanation(
                    factor="[식생 엽록소 지수(GLI)]",
                    score=round((veg_res.gli_index + 0.5) * 70, 1),
                    description=f"Green Leaf Index {veg_res.gli_index}로 세포 활성도 및 엽록소 유지 양호.",
                )
            )
            explanations.append(
                XAIExplanation(
                    factor="[짓무름 및 황변도]",
                    score=round((1.0 - veg_res.browning_ratio - veg_res.yellowing_ratio) * 100, 1),
                    description=f"갈변 무름율 {round(veg_res.browning_ratio * 100, 1)}%, 황변율 {round(veg_res.yellowing_ratio * 100, 1)}%.",
                )
            )
            explanations.append(
                XAIExplanation(
                    factor="[포장재 결로 감지]",
                    score=90.0 if not veg_res.condensation_detected else 75.0,
                    description="내부 결로 감지됨 — 빠른 소진 권장" if veg_res.condensation_detected else "내부 결로 없음.",
                )
            )
            grade = FreshnessGrade.EXCELLENT if overall_score >= 80 else FreshnessGrade.GOOD

        ocr_results = self.ocr_service.extract_text(img)
        ocr_text = " / ".join(r.text for r in ocr_results[:5]) if ocr_results else "라벨 텍스트 없음"

        return PreciseScanResponse(
            class_name=class_name,
            freshness_grade=grade,
            overall_score=overall_score,
            explanations=explanations,
            ocr_text=ocr_text,
            segmentation_mask_url=mask_url,
        )

    @staticmethod
    def _categorize(class_name: str) -> str:
        cn = class_name.lower()
        if any(w in cn for w in ["meat", "pork", "beef", "chicken", "steak", "bacon"]):
            return "meat"
        if any(w in cn for w in ["apple", "orange", "banana", "watermelon", "grape", "fruit", "strawberry"]):
            return "fruit"
        if any(w in cn for w in ["vegetable", "lettuce", "cabbage", "spinach", "carrot", "broccoli", "onion"]):
            return "vegetable"
        return "meat"

    @staticmethod
    def _save_colored_mask(image: np.ndarray, seg_res: MeatSegResult) -> str:
        overlay = image.copy()
        overlay[seg_res.lean_mask] = (0, 0, 220)      # 살코기: 붉은색
        overlay[seg_res.fat_mask] = (0, 225, 225)     # 지방: 황금빛 노란색
        overlay[seg_res.bone_mask] = (220, 100, 0)    # 뼈/연골: 푸른색

        blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

        filename = f"mask_{uuid.uuid4().hex[:8]}.jpg"
        save_dir = Path(settings.UPLOAD_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        cv2.imwrite(str(save_path), blended)

        return f"/uploads/{filename}"
