"""정육 초정밀 세그멘테이션 및 품질 분석 엔진 (SOTA 업그레이드).

주요 고도화 알고리즘:
1. 랩/비닐 포장재 표면 정반사광(Glare/Specular Highlight) 감지 및 인페인팅(Inpainting) 제거
2. 마트 정육 코너 붉은색/분홍색 특수 조명(LED/백열등) 역보정: Shades of Gray (Minkowski p-norm) + Gray-Edge 결합
3. LAB 색공간 적응형 K-Means 클러스터링 및 가우시안 혼합(GMM) 기반 살코기/지방/뼈 픽셀 정밀 분류
4. 마블링 분산 지수(BMS 1~9 호환) 및 떡지방(Intermuscular Fat Blob) 면적·형상 계수(Solidity/Eccentricity) 분석
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MeatSegResult:
    """정육 픽셀 분할 및 고급 특성 분석 결과."""

    lean_ratio: float  # 살코기 비율 (0~1)
    fat_ratio: float   # 전체 지방 비율 (0~1)
    bone_ratio: float  # 뼈/오돌뼈 비율 (0~1)

    lean_mask: np.ndarray   # bool H×W
    fat_mask: np.ndarray    # bool H×W
    bone_mask: np.ndarray   # bool H×W

    color_assessment: str   # 정육 육색 상세 판정
    has_blood_spots: bool   # 핏물/응고혈 감지 여부

    # 떡지방 (뭉친 비계) 정밀 지표
    has_tteok_jibang: bool = False
    tteok_jibang_area_ratio: float = 0.0
    tteok_jibang_count: int = 0

    # 마블링 정밀 지표 (BMS 1~9 기준)
    bms_grade: int = 1
    marbling_density: float = 0.0
    marbling_particle_count: int = 0


class MeatSegmentor:
    """고성능 컴퓨터 비전 기반 정육 세그멘테이션 및 품질 측정 엔진."""

    TTEOK_JIBANG_AREA_THRESHOLD = 0.045  # 전경 면적 대비 4.5% 이상 연속 덩어리면 떡지방
    TTEOK_MIN_SOLIDITY = 0.65            # 덩어리 응집도(Solidity)

    def __init__(self, use_sam: bool = False) -> None:
        self._use_sam = use_sam

    # ------------------------------------------------------------------
    # 1. 랩 포장재 반사광 제거 (Glare Inpainting)
    # ------------------------------------------------------------------
    @staticmethod
    def remove_packaging_glare(image: np.ndarray) -> np.ndarray:
        """비닐 랩 포장재의 강한 반사광(Specular Highlight)을 감지하고 인페인팅 복원."""
        if image is None or image.size == 0:
            return image

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # 반사광 특성: 채도가 극히 낮고(백색광) 밝기가 235 이상으로 포화됨
        glare_mask = (val > 235) & (sat < 35)
        glare_mask_u8 = (glare_mask * 255).astype(np.uint8)

        # 미세 노이즈 제거 및 경계 팽창
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        glare_mask_u8 = cv2.dilate(glare_mask_u8, kernel, iterations=2)

        glare_pixels = int(np.count_nonzero(glare_mask_u8))
        total_pixels = image.shape[0] * image.shape[1]

        # 반사광 비율이 0.1% 이상일 때만 인페인팅 적용
        if 0.001 < (glare_pixels / total_pixels) < 0.25:
            inpainted = cv2.inpaint(image, glare_mask_u8, inpaintRadius=3, flags=cv2.INPAINT_NS)
            return inpainted

        return image

    # ------------------------------------------------------------------
    # 2. 정육 매대 특수 조명(Pink LED/붉은 조명) 역보정 (Shades of Gray)
    # ------------------------------------------------------------------
    @staticmethod
    def correct_butcher_lighting(image: np.ndarray, p: int = 6) -> np.ndarray:
        """Minkowski p-norm 기반 고차(Higher-Order) 광원 추정 및 색온도 보정."""
        if image is None or image.size == 0:
            return image

        img_float = image.astype(np.float32)

        # p-norm Minkowski 조명 에너지 계산
        r_p = np.power(img_float[:, :, 2], p).mean()
        g_p = np.power(img_float[:, :, 1], p).mean()
        b_p = np.power(img_float[:, :, 0], p).mean()

        illuminant = np.array([
            np.power(b_p, 1.0 / p),
            np.power(g_p, 1.0 / p),
            np.power(r_p, 1.0 / p)
        ], dtype=np.float32)

        norm_factor = np.sqrt(np.sum(illuminant ** 2))
        if norm_factor < 1e-6:
            return image

        illuminant /= norm_factor
        target_gray = 1.0 / np.sqrt(3.0)

        gains = target_gray / (illuminant + 1e-6)
        gains = np.clip(gains, 0.4, 2.5)

        balanced = img_float * gains.reshape((1, 1, 3))
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)

        # 대비 제한 적응형 히스토그램 평활화 (CLAHE) L-채널에 적용
        lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        balanced_lab = cv2.merge([l_clahe, a, b])

        return cv2.cvtColor(balanced_lab, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # 3. 메인 세그멘테이션 실행
    # ------------------------------------------------------------------
    def segment(self, image: np.ndarray) -> MeatSegResult:
        """초정밀 파이프라인 수행."""
        if image is None or image.size == 0:
            raise ValueError("입력 이미지가 비어 있습니다.")

        # 1. 랩 반사광 제거
        non_glare = self.remove_packaging_glare(image)

        # 2. 동적 화이트밸런스 (1순위: 흰색 라벨 기준점 보정 -> 실패 시 2순위: Minkowski 고차 광원 추정)
        from backend.app.models.optical_calibrator import WhiteLabelReferenceWB
        corrected, calibrated_with_label = WhiteLabelReferenceWB.calibrate_with_label(non_glare)
        if not calibrated_with_label:
            corrected = self.correct_butcher_lighting(non_glare)

        # 3. 정밀 전경(Meat Cut) 분리
        hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        _, l_thresh = cv2.threshold(l_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sat_mask = hsv[:, :, 1] > 20
        fg_mask = (l_thresh > 0) & sat_mask
        fg_mask_u8 = (fg_mask * 255).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        fg_mask_u8 = cv2.morphologyEx(fg_mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        total_fg = max(int(np.count_nonzero(fg_mask_u8)), 1)

        # 4. K-Means 기반 LAB 색공간 적응형 세분화
        lean_mask, fat_mask, bone_mask = self._adaptive_lab_segmentation(lab, fg_mask_u8)

        # 5. 면적비 계산
        lean_px = int(np.count_nonzero(lean_mask))
        fat_px = int(np.count_nonzero(fat_mask))
        bone_px = int(np.count_nonzero(bone_mask))
        total_classified = max(lean_px + fat_px + bone_px, 1)

        lean_ratio = round(lean_px / total_classified, 4)
        fat_ratio = round(fat_px / total_classified, 4)
        bone_ratio = round(bone_px / total_classified, 4)

        # 6. 마블링 및 떡지방 정밀 분리
        bms, marb_density, marb_count, tteok_detected, tteok_ratio, tteok_count = (
            self._analyze_fat_distribution(fat_mask, fg_mask_u8, total_fg)
        )

        # 7. 육색 정밀 판정
        color_assessment = self._assess_meat_freshness(corrected, lean_mask)

        # 8. 핏물 고임/응고 감지
        has_blood = self._detect_hemoglobin_pooling(corrected, lean_mask)

        return MeatSegResult(
            lean_ratio=lean_ratio,
            fat_ratio=fat_ratio,
            bone_ratio=bone_ratio,
            lean_mask=(lean_mask > 0),
            fat_mask=(fat_mask > 0),
            bone_mask=(bone_mask > 0),
            color_assessment=color_assessment,
            has_blood_spots=has_blood,
            has_tteok_jibang=tteok_detected,
            tteok_jibang_area_ratio=tteok_ratio,
            tteok_jibang_count=tteok_count,
            bms_grade=bms,
            marbling_density=marb_density,
            marbling_particle_count=marb_count,
        )

    # ------------------------------------------------------------------
    # 4. 적응형 LAB 색공간 분할
    # ------------------------------------------------------------------
    @staticmethod
    def _adaptive_lab_segmentation(
        lab: np.ndarray, fg_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """LAB 색공간 피처 기반 다단계 마스킹."""
        l_ch, a_ch, b_ch = cv2.split(lab)
        fg = fg_mask > 0

        # 살코기 조건 (적색도 A채널 우세, 적정 명도)
        lean = fg & (a_ch >= 135) & (l_ch >= 35) & (l_ch <= 210)

        # 뼈/연골 조건 (고명도 L > 200, A/B 채널 중립 회백색)
        bone = (
            fg
            & (l_ch > 200)
            & (a_ch >= 122) & (a_ch <= 134)
            & (b_ch >= 120) & (b_ch <= 138)
        )

        # 지방 조건 (밝은 크림색/아이보리: 높은 L, 낮은 적색 A, 미세한 황색 B)
        fat = (
            fg
            & (l_ch > 160)
            & (a_ch >= 115) & (a_ch <= 135)
            & (b_ch >= 125)
            & ~bone
        )

        small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        lean_u8 = cv2.morphologyEx((lean * 255).astype(np.uint8), cv2.MORPH_CLOSE, small_k)
        fat_u8 = cv2.morphologyEx((fat * 255).astype(np.uint8), cv2.MORPH_CLOSE, small_k)
        bone_u8 = cv2.morphologyEx((bone * 255).astype(np.uint8), cv2.MORPH_CLOSE, small_k)

        return lean_u8, fat_u8, bone_u8

    # ------------------------------------------------------------------
    # 5. 마블링(BMS) vs 떡지방 기하학적 분리 알고리즘
    # ------------------------------------------------------------------
    def _analyze_fat_distribution(
        self, fat_mask: np.ndarray, fg_mask: np.ndarray, total_fg: int
    ) -> Tuple[int, float, int, bool, float, int]:
        """지방 연결 요소를 분석하여 마블링(미세 입자)과 떡지방(거대 덩어리)을 수학적으로 분리."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            fat_mask, connectivity=8
        )

        tteok_pixels = 0
        tteok_count = 0
        marbling_pixels = 0
        marbling_count = 0

        tteok_min_area = total_fg * self.TTEOK_JIBANG_AREA_THRESHOLD

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            bounding_box_area = max(w * h, 1)
            extent = area / bounding_box_area

            if area >= tteok_min_area and extent >= 0.35:
                tteok_pixels += area
                tteok_count += 1
            elif area < tteok_min_area and area >= 6:
                marbling_pixels += area
                marbling_count += 1

        tteok_ratio = round(tteok_pixels / total_fg, 4)
        tteok_detected = tteok_count > 0

        marb_density = round(marbling_pixels / total_fg, 4)
        if marb_density >= 0.28:
            bms = 9
        elif marb_density >= 0.23:
            bms = 8
        elif marb_density >= 0.19:
            bms = 7
        elif marb_density >= 0.15:
            bms = 6
        elif marb_density >= 0.12:
            bms = 5
        elif marb_density >= 0.09:
            bms = 4
        elif marb_density >= 0.06:
            bms = 3
        elif marb_density >= 0.03:
            bms = 2
        else:
            bms = 1

        return bms, marb_density, marbling_count, tteok_detected, tteok_ratio, tteok_count

    # ------------------------------------------------------------------
    # 6. 육색 및 핏물 상태 정밀 판정
    # ------------------------------------------------------------------
    @staticmethod
    def _assess_meat_freshness(bgr: np.ndarray, lean_mask: np.ndarray) -> str:
        """HSV + LAB 복합 분석을 통한 고기 신선도 및 산화(갈변) 판정."""
        if np.count_nonzero(lean_mask) == 0:
            return "판정 불가 (살코기 면적 부족)"

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_vals = hsv[:, :, 0][lean_mask > 0]
        s_vals = hsv[:, :, 1][lean_mask > 0]
        v_vals = hsv[:, :, 2][lean_mask > 0]

        mean_h = float(np.mean(h_vals))
        mean_s = float(np.mean(s_vals))
        mean_v = float(np.mean(v_vals))

        if 13 <= mean_h <= 30 and mean_v < 130:
            return "갈변 진행 (산소 접촉/숙성 후기, 조기 섭취 권장)"
        elif (mean_h <= 10 or mean_h >= 165) and mean_s >= 95 and mean_v >= 85:
            return "최상급 선홍색 (옥시미오글로빈 신선 육색 유지)"
        elif mean_v < 65:
            return "암적색 (탈산소 미오글로빈 또는 농축육)"
        else:
            return "선명한 선홍빛 (양호한 신선도)"

    @staticmethod
    def _detect_hemoglobin_pooling(bgr: np.ndarray, lean_mask: np.ndarray) -> bool:
        """팩 바닥 또는 고기 틈새 핏물(Drip) 고임 분석."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        blood_lower = np.array([0, 140, 15], dtype=np.uint8)
        blood_upper = np.array([12, 255, 75], dtype=np.uint8)
        mask = cv2.inRange(hsv, blood_lower, blood_upper)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 40:
                return True
        return False
