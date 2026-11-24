"""FreshScanner Track B 부위 탐지(Part Detection) 계측 모듈.

YOLOX-Nano / RT-DETR (Apache-2.0 상용 안전) 모델이 뱉어내는
부위별 바운딩 박스([{"label": str, "conf": float, "bbox": [x1, y1, x2, y2]}])를 받아
DynamicExpertScoringEngine에 즉시 꽂히는 FactorResult로 변환.

1. 수박: 배꼽 상대 지름비 (A_navel / A_body)
2. 부사 사과: 밑동(꽃자리) 엽록소 잔류율 (<2% 황금빛 완숙)
3. 제주 감귤: 측면 편평비 (Height / Width, 0.72~0.84 조생 최적)
4. 딸기: 꼭지 젖힘 각도 & 어깨(상단 25%) 착색도
5. 삼겹살: 오돌뼈 장축 상대 위치 (0.35~0.65 중앙 꽃삼겹 부위)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from backend.app.models.expert_scoring_engine import FactorResult


def _get_best_det(detections: List[Dict[str, Any]], label: str, min_conf: float = 0.60) -> Optional[Dict[str, Any]]:
    candidates = [d for d in detections if d.get("label") == label and d.get("conf", 0.0) >= min_conf]
    return max(candidates, key=lambda x: x["conf"]) if candidates else None


def _box_area(bbox: List[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


# 1. 수박: 배꼽 상대 지름비
def extract_watermelon_navel(detections: List[Dict[str, Any]]) -> FactorResult:
    body = _get_best_det(detections, "watermelon_body")
    navel = _get_best_det(detections, "watermelon_navel")

    if not body or not navel:
        return FactorResult(
            name="수박 배꼽 비율",
            score=0.0,
            confidence=0.0,
            base_weight=0.45,
            reason_text="",
            missing_guide="수박을 살짝 돌려 배꼽(아랫면)을 비춰주세요 🍉",
        )

    # 면적 기반 등가지름 비율 (%)
    ratio_pct = float(np.sqrt(_box_area(navel["bbox"]) / (_box_area(body["bbox"]) + 1e-6)) * 100.0)

    if ratio_pct <= 4.0:
        score, text = 96.0, f"초소형 배꼽({ratio_pct:.1f}%) · 껍질 얇고 당도 우수"
    elif ratio_pct >= 7.5:
        score, text = 70.0, f"확장된 배꼽({ratio_pct:.1f}%) · 껍질이 두꺼울 수 있음"
    else:
        score = float(np.interp(ratio_pct, [4.0, 7.5], [95.0, 72.0]))
        text = f"적정 배꼽 비율({ratio_pct:.1f}%)"

    return FactorResult(
        name="수박 배꼽 비율",
        score=round(score, 1),
        confidence=min(body["conf"], navel["conf"]),
        base_weight=0.45,
        reason_text=text,
        overlay_data={"type": "navel_box", "bbox": navel["bbox"], "ratio_pct": round(ratio_pct, 1)},
    )


# 2. 부사 사과: 밑동(배꼽) 엽록소 잔류율
def extract_apple_bottom_ripeness(image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> FactorResult:
    bottom = _get_best_det(detections, "apple_bottom")
    if not bottom:
        return FactorResult(
            name="사과 밑동 완숙도",
            score=0.0,
            confidence=0.0,
            base_weight=0.30,
            reason_text="",
            missing_guide="사과 아랫면(밑동)을 살짝 비춰주세요 🍎",
        )

    x1, y1, x2, y2 = map(int, bottom["bbox"])
    roi = image_bgr[max(0, y1) : y2, max(0, x1) : x2]
    if roi.size == 0:
        return FactorResult(name="사과 밑동 완숙도", score=0.0, confidence=0.0, base_weight=0.30, reason_text="")

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # OpenCV Hue 범위 (0~179): 초록색 (35~70)
    green_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 70) & (hsv[:, :, 1] >= 40)
    green_pct = float(np.count_nonzero(green_mask) / (roi.shape[0] * roi.shape[1] + 1e-6) * 100.0)

    if green_pct < 2.0:
        score, text = 94.0, f"밑동까지 황금 완숙 (초록기 {green_pct:.1f}%)"
    elif green_pct >= 12.0:
        score, text = 72.0, f"밑동 초록기 잔류({green_pct:.1f}%) · 신맛 가능성"
    else:
        score = float(np.interp(green_pct, [2.0, 12.0], [93.0, 74.0]))
        text = f"밑동 완숙도 양호 (초록기 {green_pct:.1f}%)"

    return FactorResult(
        name="사과 밑동 완숙도",
        score=round(score, 1),
        confidence=bottom["conf"],
        base_weight=0.30,
        reason_text=text,
        overlay_data={"type": "apple_bottom", "bbox": bottom["bbox"], "green_pct": round(green_pct, 1)},
    )


# 3. 제주 감귤: 측면 편평비 (Aspect Ratio)
def extract_mandarin_aspect_ratio(detections: List[Dict[str, Any]]) -> FactorResult:
    side = _get_best_det(detections, "mandarin_side")
    if not side:
        return FactorResult(
            name="감귤 측면 편평비",
            score=0.0,
            confidence=0.0,
            base_weight=0.35,
            reason_text="",
            missing_guide="귤을 들어 옆면이 보이게 비춰주세요 🍊",
        )

    x1, y1, x2, y2 = side["bbox"]
    ar = float((y2 - y1) / ((x2 - x1) + 1e-6))

    if 0.72 <= ar <= 0.84:
        score, text = 95.0, f"납작한 완숙 밀감 (편평비 {ar:.2f}) · 껍질 얇음"
    elif ar >= 0.95:
        score, text = 74.0, f"동그란 구형 (편평비 {ar:.2f}) · 껍질 두꺼울 수 있음"
    else:
        score = float(np.interp(ar, [0.84, 0.95], [93.0, 76.0]))
        text = f"감귤 형태 양호 (편평비 {ar:.2f})"

    return FactorResult(
        name="감귤 측면 편평비",
        score=round(score, 1),
        confidence=side["conf"],
        base_weight=0.35,
        reason_text=text,
        overlay_data={"type": "mandarin_side", "bbox": side["bbox"], "ar": round(ar, 2)},
    )


# 4. 딸기: 꼭지 젖힘 & 어깨 착색도
def extract_strawberry_calyx_and_shoulder(image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> FactorResult:
    body = _get_best_det(detections, "strawberry_body")
    calyx = _get_best_det(detections, "strawberry_calyx")
    if not body or not calyx:
        return FactorResult(
            name="딸기 꼭지·어깨 완숙도",
            score=0.0,
            confidence=0.0,
            base_weight=0.40,
            reason_text="",
            missing_guide="딸기 상단 꼭지가 보이게 비춰주세요 🍓",
        )

    bx1, by1, bx2, by2 = map(int, body["bbox"])
    cx1, cy1, cx2, cy2 = map(int, calyx["bbox"])

    # 꼭지 중심 Y가 과육 최상단(by1)보다 위로 올라가 있을수록 꼭지가 위로 젖혀진 완숙 상태
    calyx_cy = (cy1 + cy2) / 2.0
    body_h = max(1, by2 - by1)
    lift_ratio = (by1 - calyx_cy) / body_h

    # 어깨 영역 (과육 상단 25%) 붉은 착색도 계측
    shoulder_roi = image_bgr[max(0, by1) : by1 + int(body_h * 0.25), max(0, bx1) : bx2]
    red_pct = 0.0
    if shoulder_roi.size > 0:
        hsv = cv2.cvtColor(shoulder_roi, cv2.COLOR_BGR2HSV)
        red_mask = ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] >= 70)
        red_pct = float(np.count_nonzero(red_mask) / (shoulder_roi.shape[0] * shoulder_roi.shape[1] + 1e-6) * 100.0)

    is_lifted = lift_ratio > -0.05
    score = min(98.0, (red_pct * 0.65) + (35.0 if is_lifted else 15.0))
    text = f"꼭지 젖힘 양호 · 어깨 착색 {red_pct:.0f}%" if is_lifted else f"어깨 착색 {red_pct:.0f}% (꼭지 덜 젖혀짐)"

    return FactorResult(
        name="딸기 꼭지·어깨 완숙도",
        score=round(score, 1),
        confidence=min(body["conf"], calyx["conf"]),
        base_weight=0.40,
        reason_text=text,
        overlay_data={"type": "strawberry_calyx", "bbox": calyx["bbox"], "red_pct": round(red_pct, 1)},
    )


# 5. 삼겹살: 오돌뼈 장축 상대 위치
def extract_pork_belly_cartilage(detections: List[Dict[str, Any]]) -> FactorResult:
    strip = _get_best_det(detections, "meat_strip")
    cartilage = _get_best_det(detections, "cartilage")

    if not strip or not cartilage:
        return FactorResult(
            name="오돌뼈 상대 위치",
            score=0.0,
            confidence=0.0,
            base_weight=0.30,
            reason_text="",
            missing_guide="삼겹살 팩 단면이 잘 보이게 비춰주세요 🥩",
        )

    sx1, sy1, sx2, sy2 = strip["bbox"]
    cx1, cy1, cx2, cy2 = cartilage["bbox"]

    strip_w = max(1.0, sx2 - sx1)
    cartilage_cx = (cx1 + cx2) / 2.0
    rel_x = float((cartilage_cx - sx1) / strip_w)

    if 0.35 <= rel_x <= 0.65:
        score = 96.0
        text = f"오돌뼈 중앙({int(rel_x*100)}%) · 5~8번 갈비 최고 꽃삼겹 부위"
    elif rel_x < 0.25 or rel_x > 0.75:
        score = 68.0
        text = f"오돌뼈 가장자리({int(rel_x*100)}%) · 퍽퍽한 미추리 부위"
    else:
        score = 85.0
        text = f"오돌뼈 위치 양호({int(rel_x*100)}%)"

    return FactorResult(
        name="오돌뼈 상대 위치",
        score=round(score, 1),
        confidence=min(strip["conf"], cartilage["conf"]),
        base_weight=0.30,
        reason_text=text,
        overlay_data={"type": "cartilage_box", "bbox": cartilage["bbox"], "rel_x": round(rel_x * 100, 1)},
    )
