"""FreshScanner 핵심 광학 및 컴퓨터 비전 알고리즘 단위 테스트 (100% Apache-2.0 / MIT / BSD)."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from backend.app.models.optical_calibrator import (
    PreCaptureQualityGate,
    WhiteLabelReferenceWB,
)
from backend.app.models.meat_segmentor import MeatSegmentor
from backend.app.models.fruit_analyzer import FruitAnalyzer
from backend.app.models.vegetable_analyzer import VegetableAnalyzer
from backend.app.services.ocr_service import KoreanMartLabelParser


class TestOpticalCalibrator:
    """광학 캘리브레이션 알고리즘 검증."""

    def test_white_label_wb_pink_light_correction(self):
        """정육점 붉은색 조명(R 편향) 하에서 라벨 기준 폰 크리스 화이트밸런스 보정 검증."""
        h, w = 300, 300
        # 붉은색 조명에 편향된 이미지 (B=80, G=80, R=220)
        img = np.full((h, w, 3), (80, 80, 220), dtype=np.uint8)
        # 우측 상단에 흰색 라벨 부착 (조명 영향으로 B=120, G=120, R=250)
        label_box = (200, 20, 280, 80)
        x1, y1, x2, y2 = label_box
        img[y1:y2, x1:x2] = (120, 120, 250)

        corrected, success = WhiteLabelReferenceWB.calibrate_with_label(img, label_bbox=label_box)
        assert success is True

        label_roi = corrected[y1:y2, x1:x2]
        mean_b = float(np.mean(label_roi[:, :, 0]))
        mean_r = float(np.mean(label_roi[:, :, 2]))

        # 원본 라벨의 R(250) - B(120) = 130 편차가 대폭 축소됨
        diff_rb = abs(mean_r - mean_b)
        assert diff_rb < 70.0

    def test_specular_glare_inpainting(self):
        """비닐 랩 반사광 영역 마스킹 및 복원 검증."""
        img = np.full((100, 100, 3), (40, 40, 200), dtype=np.uint8)  # 붉은 살코기
        # 중앙에 강한 반사광 (V=255, S=0)
        cv2.circle(img, (50, 50), 10, (255, 255, 255), -1)

        inpainted = MeatSegmentor.remove_packaging_glare(img)
        assert inpainted is not None
        assert inpainted.shape == img.shape

    def test_pre_capture_quality_gate(self):
        """라플라시안 흐림 및 반사광 임계값 게이트 검증."""
        # 1. 선명하고 깨끗한 체커보드 패턴 이미지
        sharp_img = np.zeros((200, 200, 3), dtype=np.uint8)
        sharp_img[::20, :] = 255
        sharp_img[:, ::20] = 255
        res_sharp = PreCaptureQualityGate.evaluate_frame(sharp_img)
        assert res_sharp.can_capture is True
        assert res_sharp.blur_score > 90.0
        assert res_sharp.glare_ratio < 0.16

        # 2. 극도로 흐린(블러) 이미지
        blurry_img = cv2.GaussianBlur(sharp_img, (51, 51), 0)
        res_blur = PreCaptureQualityGate.evaluate_frame(blurry_img)
        assert res_blur.can_capture is False
        assert res_blur.blur_score < 90.0
        assert any("초점" in m or "스마트폰" in m for m in res_blur.messages)


class TestMeatAnalysis:
    """정육 살코기/지방 분할, 마블링, 떡지방 검출 검증."""

    def test_segmentation_real_image(self):
        sample_path = Path("data/sample_images/real_pork_belly_pack_01.jpg")
        assert sample_path.exists(), "실제 마트 샘플 이미지가 존재해야 합니다."
        img = cv2.imread(str(sample_path))

        segmentor = MeatSegmentor()
        result = segmentor.segment(img)
        assert result.fat_ratio > 0.05
        assert result.lean_ratio > 0.10
        assert result.bms_grade >= 1


class TestFruitAnalysis:
    """수박 줄무늬 가보 필터 및 완숙도 검증."""

    def test_watermelon_gabor_stripes(self):
        analyzer = FruitAnalyzer()
        img = np.full((200, 200, 3), (50, 150, 50), dtype=np.uint8)
        for x in range(20, 180, 30):
            cv2.rectangle(img, (x, 0), (x + 12, 200), (20, 50, 20), -1)

        result = analyzer.analyze(img, fruit_type="watermelon")
        assert result.surface_pattern_score > 30.0
        assert result.quality_score > 0.0


class TestVegetableAnalysis:
    """채소 Green Leaf Index (GLI) 및 짓무름 검증."""

    def test_gli_and_wilting(self):
        evaluator = VegetableAnalyzer()
        # 신선한 녹색 채소
        fresh_leaf = np.full((100, 100, 3), (30, 200, 40), dtype=np.uint8)
        res_fresh = evaluator.analyze(fresh_leaf, veg_type="leafy")
        assert res_fresh.freshness_score > 70.0
        assert res_fresh.browning_ratio < 0.10

        # 갈변 및 짓무른 채소
        brown_leaf = np.full((100, 100, 3), (30, 70, 110), dtype=np.uint8)  # 갈색 톤
        res_brown = evaluator.analyze(brown_leaf, veg_type="leafy")
        assert res_brown.freshness_score < res_fresh.freshness_score
        assert res_brown.browning_ratio > 0.30


class TestKoreanMartLabelParser:
    """국내 3대 마트(이마트/홈플러스/롯데마트) 정육 라벨 OCR 파서 단위 테스트."""

    def test_emart_price_and_weight_parsing(self):
        lines = [
            "국내산 삼겹살 (구이용)",
            "이력번호: 002123456789",
            "100g당 2,680원",
            "중량: 450g",
            "판매금액: 12,060원",
        ]
        price_per_100g, weight, total_price = KoreanMartLabelParser.parse_label_lines(lines)
        assert price_per_100g == 2680
        assert weight == 450
        assert total_price == 12060

    def test_membership_discount_priority(self):
        """회원 할인가와 정상가가 동시에 기재된 경우 실제 결제 단가 추출."""
        lines = [
            "한우 1등급 등심 300g",
            "정상가 100g당 11,000원",
            "신세계포인트 적립시 100g당 8,800원",
            "결제예정금액: 26,400원",
        ]
        price_per_100g, weight, total_price = KoreanMartLabelParser.parse_label_lines(lines)
        assert price_per_100g == 8800  # 할인가 우선 인식
        assert weight == 300
        assert total_price == 26400


class TestONNXInference:
    """온디바이스 배포용 DeepLabV3-MobileNetV3 ONNX 모델 무결성 검증."""

    def test_onnx_segmentation_runtime(self):
        onnx_path = Path("models/meat_deeplabv3_mobilenet.onnx")
        if not onnx_path.exists():
            pytest.skip("ONNX 모델 파일이 존재하지 않습니다.")

        from backend.app.models.deep_meat_segmentor import DeepMeatSegmentor
        segmentor = DeepMeatSegmentor(weights_path=str(onnx_path), use_onnx=True)

        dummy = np.full((256, 256, 3), (40, 40, 200), dtype=np.uint8)
        lean, fat, bone = segmentor.predict_mask(dummy)

        assert lean.shape == (256, 256)
        assert fat.shape == (256, 256)
        assert bone.shape == (256, 256)

