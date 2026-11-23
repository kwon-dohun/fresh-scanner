"""채소 엽록소 지수(GLI/VARI) 및 무름·황변·결로 정밀 분석 엔진."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VegetableQualityResult:
    """채소 품질 및 신선도 상세 리포트."""

    freshness_score: float          # 0~100
    browning_ratio: float           # 0~1 (갈변/짓무름 비율)
    yellowing_ratio: float          # 0~1 (황변/엽록소 소실 비율)
    condensation_detected: bool     # 포장지 내부 결로 여부
    edge_wilting_score: float       # 0~100 (에지 시듦 지수)
    gli_index: float                # Green Leaf Index (-1.0 ~ 1.0)
    recommendations: List[str]


class VegetableAnalyzer:
    """분광 지수 및 텍스처 엔트로피 기반 채소 신선도 측정기."""

    def analyze(self, image: np.ndarray, veg_type: str = "leafy") -> VegetableQualityResult:
        if image is None or image.size == 0:
            raise ValueError("입력 이미지가 비어 있습니다.")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. 전경 분리
        fg_mask = self._foreground_mask(gray)
        total_fg = max(int(np.count_nonzero(fg_mask)), 1)

        # 2. GLI (Green Leaf Index) 계산: (2*G - R - B) / (2*G + R + B)
        gli_mean = self._compute_gli(image, fg_mask)

        # 3. 색상 비율 (녹색 / 황변 / 갈변)
        green_r, yellow_r, brown_r = self._color_ratios(hsv, fg_mask, total_fg)

        # 4. 잎 가장자리 무름 및 시듦(Wilting) 정밀 분석
        wilting_score = self._edge_wilting_score(gray, fg_mask)

        # 5. 비닐 포장 내부 결로(물방울) 감지
        condensation = self._detect_condensation(gray, fg_mask)

        # 6. 종합 신선도 산출
        freshness = self._compute_freshness(gli_mean, yellow_r, brown_r, wilting_score, condensation)

        recs = self._recommendations(freshness, green_r, yellow_r, brown_r, wilting_score, condensation, veg_type)

        return VegetableQualityResult(
            freshness_score=round(freshness, 1),
            browning_ratio=round(brown_r, 4),
            yellowing_ratio=round(yellow_r, 4),
            condensation_detected=condensation,
            edge_wilting_score=round(wilting_score, 1),
            gli_index=round(gli_mean, 3),
            recommendations=recs,
        )

    @staticmethod
    def _foreground_mask(gray: np.ndarray) -> np.ndarray:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        return mask

    @staticmethod
    def _compute_gli(bgr: np.ndarray, fg_mask: np.ndarray) -> float:
        """Green Leaf Index (식생 활성도 지수) 연산."""
        img_f = bgr.astype(np.float32)
        b = img_f[:, :, 0]
        g = img_f[:, :, 1]
        r = img_f[:, :, 2]

        num = 2.0 * g - r - b
        denom = 2.0 * g + r + b + 1e-6

        gli = num / denom
        gli_fg = gli[fg_mask > 0]
        if len(gli_fg) == 0:
            return 0.0
        return float(np.mean(gli_fg))

    @staticmethod
    def _color_ratios(hsv: np.ndarray, fg_mask: np.ndarray, total_fg: int) -> Tuple[float, float, float]:
        green_m = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255]))
        yellow_m = cv2.inRange(hsv, np.array([15, 40, 80]), np.array([35, 255, 255]))
        brown_m = cv2.inRange(hsv, np.array([5, 30, 20]), np.array([20, 200, 160]))

        green_m = cv2.bitwise_and(green_m, fg_mask)
        yellow_m = cv2.bitwise_and(yellow_m, fg_mask)
        brown_m = cv2.bitwise_and(brown_m, fg_mask)

        return (
            int(np.count_nonzero(green_m)) / total_fg,
            int(np.count_nonzero(yellow_m)) / total_fg,
            int(np.count_nonzero(brown_m)) / total_fg,
        )

    @staticmethod
    def _edge_wilting_score(gray: np.ndarray, fg_mask: np.ndarray) -> float:
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        eroded = cv2.erode(fg_mask, np.ones((14, 14), np.uint8))
        edge_band = cv2.subtract(fg_mask, eroded)

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        edge_pixels = grad_mag[edge_band > 0]
        interior_pixels = grad_mag[eroded > 0]

        if len(edge_pixels) == 0 or len(interior_pixels) == 0:
            return 0.0

        edge_e = float(np.mean(edge_pixels))
        interior_e = float(np.mean(interior_pixels))

        if interior_e < 1e-6:
            return 0.0

        ratio = edge_e / interior_e
        if ratio >= 0.95:
            return 5.0
        return float(np.clip((0.95 - ratio) * 110.0, 0.0, 95.0))

    @staticmethod
    def _detect_condensation(gray: np.ndarray, fg_mask: np.ndarray) -> bool:
        _, bright = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
        bright = cv2.bitwise_and(bright, fg_mask)

        circles = cv2.HoughCircles(
            bright, cv2.HOUGH_GRADIENT, dp=1.5, minDist=8, param1=45, param2=11, minRadius=2, maxRadius=18
        )
        return bool(circles is not None and len(circles[0]) >= 3)

    @staticmethod
    def _compute_freshness(
        gli: float, yellow_r: float, brown_r: float, wilting: float, condensation: bool
    ) -> float:
        # GLI 정상치(0.15~0.45) 기반 활성도
        base = 65.0 + (gli * 70.0)
        base -= yellow_r * 80.0
        base -= brown_r * 150.0
        base -= wilting * 0.25

        if condensation:
            base -= 5.0  # 과도한 결로는 무름병을 촉진할 수 있음

        return float(np.clip(base, 20.0, 98.0))

    @staticmethod
    def _recommendations(
        freshness: float, green_r: float, yellow_r: float, brown_r: float,
        wilting: float, condensation: bool, veg_type: str
    ) -> List[str]:
        recs: List[str] = []
        if freshness >= 85:
            recs.append("잎맥이 살아있고 녹색도가 뛰어난 최상급 채소입니다.")
        elif freshness >= 65:
            recs.append("전반적으로 신선하나 수확 후 시간이 다소 경과했습니다.")
        else:
            recs.append("짓무름 또는 황변이 발생하여 상품성이 떨어집니다.")

        if brown_r > 0.05:
            recs.append(f"잎 끝 짓무름/갈변 면적 {brown_r * 100:.1f}% 감지 — 손질 시 손실 예상.")
        if yellow_r > 0.10:
            recs.append(f"황변(엽록소 파괴) 진행 {yellow_r * 100:.1f}%.")
        if condensation:
            recs.append("비닐 내부에 물방울이 맺혀 있어 개봉 후 빠른 소진을 권장합니다.")

        return recs
