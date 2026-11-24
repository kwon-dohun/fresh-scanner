"""Unit tests for FreshScanner Expert Scoring Engine & Extractors."""

import numpy as np
import pytest

from backend.app.models.expert_scoring_engine import (
    DynamicExpertScoringEngine,
    FactorResult,
)
from backend.app.models.optical_calibrator import WhiteLabelReferenceWB
from backend.app.models.part_geometry_engine import PartGeometryEngine
from backend.app.models.track_a_extractors import (
    extract_fat_oxidation_and_color,
    extract_marbling_and_fat_clump,
    extract_shinemuscat_ripeness,
    extract_skin_texture_score,
)
from backend.app.models.track_b_extractors import (
    extract_apple_bottom_ripeness,
    extract_mandarin_aspect_ratio,
    extract_pork_belly_cartilage,
    extract_strawberry_calyx_and_shoulder,
    extract_watermelon_navel,
)


def test_dynamic_expert_scoring_engine_normal():
    engine = DynamicExpertScoringEngine()
    factors = [
        FactorResult("오돌뼈 위치", 95.0, 0.90, 0.30, "오돌뼈 중앙 꽃삼겹"),
        FactorResult("살코기 비율", 90.0, 0.95, 0.40, "살코기 황금비율"),
        FactorResult("지방 색상", 85.0, 0.85, 0.30, "순백색 신선"),
    ]
    res = engine.evaluate(factors)
    assert res["status"] == "OK"
    assert res["final_score"] is not None
    assert 88.0 <= res["final_score"] <= 93.0
    assert len(res["reasons"]) == 2
    assert len(res["veto_alerts"]) == 0


def test_dynamic_expert_scoring_engine_veto_penalty():
    engine = DynamicExpertScoringEngine()
    factors = [
        FactorResult("오돌뼈 위치", 95.0, 0.90, 0.50, "오돌뼈 중앙"),
        FactorResult("살코기 비율", 95.0, 0.95, 0.50, "살코기 황금비"),
        FactorResult(
            "떡지방 뭉침", 30.0, 0.92, 0.0, "⚠️ 단일 떡지방 18% 뭉침",
            is_veto=True, veto_penalty=20.0,
        ),
    ]
    res = engine.evaluate(factors)
    assert res["status"] == "OK"
    # 95.0 - 20.0 = 75.0
    assert res["final_score"] == 75.0
    assert len(res["veto_alerts"]) == 1
    assert "떡지방" in res["veto_alerts"][0]["name"]


def test_dynamic_expert_scoring_engine_missing_feature_guide():
    engine = DynamicExpertScoringEngine()
    # 수박 배꼽이 각도상 안 보임 (신뢰도 0.0)
    factors = [
        FactorResult("줄무늬 선명도", 90.0, 0.88, 0.40, "줄무늬 선명"),
        FactorResult(
            "수박 배꼽 비율", 0.0, 0.0, 0.60, "",
            missing_guide="수박을 살짝 돌려 배꼽(아랫면)을 비춰주세요 🍉",
        ),
    ]
    res = engine.evaluate(factors)
    assert res["status"] == "OK"
    # 배꼽이 빠졌으므로 남은 줄무늬 가중치가 100%로 재조정되어 90.0점이 됨 (0점 안 먹음!)
    assert res["final_score"] == 90.0
    assert res["guide_prompt"] == "수박을 살짝 돌려 배꼽(아랫면)을 비춰주세요 🍉"


def test_marbling_and_fat_clump():
    # 100x100 meat mask with high marbling (checkerboard)
    fat_mask = np.zeros((100, 100), dtype=np.uint8)
    fat_mask[::2, ::2] = 1
    meat_mask = np.ones((100, 100), dtype=np.uint8)

    res = extract_marbling_and_fat_clump(fat_mask, meat_mask)
    assert res["confidence"] > 0.80
    assert res["marbling_dispersion_score"] >= 70.0
    assert not res["is_fat_clump_veto"]


def test_part_geometry_engine_pork_belly():
    # Red meat + white fat image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :60] = [30, 30, 180]  # Red lean meat
    img[:, 60:] = [220, 220, 220]  # White fat

    eval_res = PartGeometryEngine.evaluate_pork_belly(img)
    assert eval_res.item_name == "한돈 삼겹살"
    assert 50 <= eval_res.total_score <= 100
    assert "살코기" in eval_res.core_evidence
    # Track B 미전달 시 임시 휴리스틱 오염 없이 가이드 안내가 정상 생성되는지 검증
    assert eval_res.guide_prompt == "삼겹살 팩 단면이 잘 보이게 비춰주세요 🥩"
    assert eval_res.feature_table["오돌뼈 상대 위치"] == "미검출 (촬영 가이드 활성화)"


def test_part_geometry_engine_apple():
    # Pure red image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [30, 30, 200]  # Red apple

    eval_res = PartGeometryEngine.evaluate_apple(img)
    assert eval_res.item_name == "부사 사과"
    assert eval_res.total_score >= 80
    assert eval_res.guide_prompt == "사과 아랫면(밑동)을 살짝 비춰주세요 🍎"


def test_optical_calibrator_white_reference():
    # 붉은 조명 왜곡된 고기 이미지 (R이 과도하게 큼) + 흰색 가격 라벨 패치
    img = np.full((120, 120, 3), (20, 30, 210), dtype=np.uint8)  # 붉은 조명
    # 우측 상단 흰색 라벨 패치 (30x30)
    img[10:40, 80:110] = (230, 230, 240)

    calibrated, is_calibrated = WhiteLabelReferenceWB.calibrate_with_label(img)
    assert is_calibrated is True
    assert calibrated is not None
    assert calibrated.shape == img.shape
