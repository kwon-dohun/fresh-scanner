"""정밀 품질 분석 및 설명 가능한 AI (XAI) 서비스."""

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
    """2단계 정밀 분석 및 XAI 리포트 생성 서비스."""

    def __init__(self) -> None:
        self.detector = YOLODetector()
        self.meat_segmentor = MeatSegmentor(use_sam=False)
        self.fruit_analyzer = FruitAnalyzer()
        self.vegetable_analyzer = VegetableAnalyzer()
        self.ocr_service = OCRService()

    async def analyze_precise(self, image_bytes: bytes) -> PreciseScanResponse:
        """고해상도 이미지에 대해 객체 식별 후 카테고리별 정밀 분석 및 XAI 보고서 생성."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("이미지를 디코딩할 수 없습니다.")

        # 1. 메인 객체 탐지
        detections: list[Detection] = self.detector.detect(img, conf_threshold=0.20)
        class_name = detections[0].class_name if detections else "meat"

        # 카테고리 판별 (정육 vs 과일 vs 채소)
        category = self._categorize(class_name)

        explanations: list[XAIExplanation] = []
        overall_score = 85.0
        grade = FreshnessGrade.GOOD
        mask_url = None

        # 2. 카테고리별 특화 분석
        if category == "meat":
            seg_res: MeatSegResult = self.meat_segmentor.segment(img)
            lean_pct = int(seg_res.lean_ratio * 100)
            fat_pct = int(seg_res.fat_ratio * 100)
            bone_pct = int(seg_res.bone_ratio * 100)

            explanations.append(
                XAIExplanation(
                    factor="[황금 비율]",
                    score=float(lean_pct),
                    description=f"살코기 {lean_pct}% : 지방 {fat_pct}% (뼈 {bone_pct}%) 비율로 구이용 최적 비율에 가깝습니다.",
                )
            )

            defect_desc = "한쪽에 뭉친 떡지방이 발견되지 않았습니다."
            if seg_res.has_tteok_jibang:
                defect_desc = f"주의: 한쪽에 뭉친 떡지방이 약 {int(seg_res.tteok_jibang_area_ratio * 100)}% 면적으로 감지되었습니다."

            explanations.append(
                XAIExplanation(
                    factor="[결함 및 떡지방]",
                    score=90.0 if not seg_res.has_tteok_jibang else 60.0,
                    description=defect_desc,
                )
            )

            color_stat = "선홍빛 신선 육색 유지" if "fresh" in seg_res.color_assessment or "선홍색" in seg_res.color_assessment else "숙성/갈변 상태"
            explanations.append(
                XAIExplanation(
                    factor="[육색 및 핏물]",
                    score=92.0 if "fresh" in seg_res.color_assessment else 70.0,
                    description=f"표면 육색 분석 결과: {color_stat} ({seg_res.color_assessment}).",
                )
            )

            overall_score = round((lean_pct * 0.4) + (80 if not seg_res.has_tteok_jibang else 50) * 0.3 + 92 * 0.3, 1)
            grade = FreshnessGrade.EXCELLENT if overall_score >= 88 else FreshnessGrade.GOOD

            # 컬러 마스크 생성 및 저장
            mask_url = self._save_colored_mask(img, seg_res)

        elif category == "fruit":
            fruit_res: FruitQualityResult = self.fruit_analyzer.analyze(img, fruit_type="apple")
            overall_score = fruit_res.quality_score

            explanations.append(
                XAIExplanation(
                    factor="[표면 패턴 및 색택]",
                    score=fruit_res.surface_pattern_score,
                    description=f"표면 균일도 및 색상 분포 스코어 {fruit_res.surface_pattern_score}점 산출.",
                )
            )
            explanations.append(
                XAIExplanation(
                    factor="[꼭지 신선도]",
                    score=fruit_res.stem_freshness_score,
                    description=f"꼭지/줄기 영역의 녹색도 및 건조 상태 분석 스코어 {fruit_res.stem_freshness_score}점.",
                )
            )
            explanations.append(
                XAIExplanation(
                    factor="[표면 결함]",
                    score=max(0.0, 100.0 - fruit_res.defect_count * 15),
                    description=f"멍 또는 상처 의심 영역 {fruit_res.defect_count}개 발견.",
                )
            )
            grade = FreshnessGrade.EXCELLENT if overall_score >= 85 else FreshnessGrade.GOOD

        else:  # vegetable
            veg_res: VegetableQualityResult = self.vegetable_analyzer.analyze(img, veg_type="general")
            overall_score = veg_res.freshness_score

            explanations.append(
                XAIExplanation(
                    factor="[잎 끝 갈변도]",
                    score=round((1.0 - veg_res.browning_ratio) * 100, 1),
                    description=f"갈변 및 짓무름 면적 비율 {round(veg_res.browning_ratio * 100, 1)}%.",
                )
            )
            explanations.append(
                XAIExplanation(
                    factor="[황변 비율]",
                    score=round((1.0 - veg_res.yellowing_ratio) * 100, 1),
                    description=f"엽록소 소실(황변) 영역 비율 {round(veg_res.yellowing_ratio * 100, 1)}%.",
                )
            )
            explanations.append(
                XAIExplanation(
                    factor="[포장 내부 결로]",
                    score=85.0 if not veg_res.condensation_detected else 75.0,
                    description="포장재 내부 물방울(결로) 맺힘 여부 감지됨." if veg_res.condensation_detected else "내부 결로 없음.",
                )
            )
            grade = FreshnessGrade.EXCELLENT if overall_score >= 80 else FreshnessGrade.GOOD

        # 3. 텍스트 OCR
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
        """YOLO 클래스명으로부터 육류 / 과일 / 채소 카테고리 매핑."""
        cn = class_name.lower()
        if any(w in cn for w in ["meat", "pork", "beef", "chicken", "steak", "bacon"]):
            return "meat"
        if any(w in cn for w in ["apple", "orange", "banana", "watermelon", "grape", "fruit", "strawberry"]):
            return "fruit"
        if any(w in cn for w in ["vegetable", "lettuce", "cabbage", "spinach", "carrot", "broccoli", "onion"]):
            return "vegetable"
        return "meat"  # 기본값

    @staticmethod
    def _save_colored_mask(image: np.ndarray, seg_res: MeatSegResult) -> str:
        """세그멘테이션 마스크를 컬러 오버레이 이미지로 만들어 static 파일로 저장."""
        overlay = image.copy()
        # 살코기: 붉은색 오버레이
        overlay[seg_res.lean_mask] = (0, 0, 220)
        # 지방: 노란색 오버레이
        overlay[seg_res.fat_mask] = (0, 220, 220)
        # 뼈: 푸른색 오버레이
        overlay[seg_res.bone_mask] = (220, 100, 0)

        blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

        filename = f"mask_{uuid.uuid4().hex[:8]}.jpg"
        save_dir = Path(settings.UPLOAD_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        cv2.imwrite(str(save_path), blended)

        return f"/uploads/{filename}"
