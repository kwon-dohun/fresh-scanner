"""Vegetable freshness analysis via HSV colour ratios, edge wilting
detection, and condensation (water-droplet) detection.

Analyses green-vs-yellow-vs-brown pixel ratios, edge-region wilting
using gradient analysis, and water droplets via circular Hough /
blob detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VegetableQualityResult:
    """Quality report for a vegetable item."""

    freshness_score: float          # 0-100
    browning_ratio: float           # 0-1
    yellowing_ratio: float          # 0-1
    condensation_detected: bool
    edge_wilting_score: float       # 0-100 (100 = severe wilting)
    recommendations: List[str]


# ---------------------------------------------------------------------------
# HSV range definitions
# ---------------------------------------------------------------------------

# Green vegetation
_GREEN_LOWER = np.array([30, 30, 30], dtype=np.uint8)
_GREEN_UPPER = np.array([90, 255, 255], dtype=np.uint8)

# Yellowing
_YELLOW_LOWER = np.array([15, 40, 80], dtype=np.uint8)
_YELLOW_UPPER = np.array([35, 255, 255], dtype=np.uint8)

# Browning (oxidised tissue)
_BROWN_LOWER = np.array([5, 30, 20], dtype=np.uint8)
_BROWN_UPPER = np.array([20, 200, 160], dtype=np.uint8)


# ---------------------------------------------------------------------------
# VegetableAnalyzer
# ---------------------------------------------------------------------------

class VegetableAnalyzer:
    """Analyse vegetable freshness from a BGR image.

    Supported veg_type values: any string (analysis is generic but
    thresholds may be tuned per type in future).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self, image: np.ndarray, veg_type: str
    ) -> VegetableQualityResult:
        """Return a :class:`VegetableQualityResult` for the given image."""
        if image is None or image.size == 0:
            raise ValueError("analyze() received an empty image.")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Foreground mask
        fg_mask = self._foreground_mask(gray)
        total_fg = max(int(fg_mask.sum() / 255), 1)

        # 2. Colour ratios
        green_ratio, yellow_ratio, brown_ratio = self._color_ratios(
            hsv, fg_mask, total_fg
        )

        # 3. Edge wilting
        wilting_score = self._edge_wilting_score(gray, fg_mask)

        # 4. Condensation detection
        condensation = self._detect_condensation(gray, fg_mask)

        # 5. Composite freshness
        freshness = self._compute_freshness(
            green_ratio, yellow_ratio, brown_ratio, wilting_score, condensation
        )

        # 6. Recommendations
        recs = self._recommendations(
            freshness, green_ratio, yellow_ratio, brown_ratio,
            wilting_score, condensation, veg_type
        )

        return VegetableQualityResult(
            freshness_score=round(freshness, 1),
            browning_ratio=round(brown_ratio, 4),
            yellowing_ratio=round(yellow_ratio, 4),
            condensation_detected=condensation,
            edge_wilting_score=round(wilting_score, 1),
            recommendations=recs,
        )

    # ------------------------------------------------------------------
    # Foreground extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _foreground_mask(gray: np.ndarray) -> np.ndarray:
        """Otsu + morphology foreground mask."""
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    # ------------------------------------------------------------------
    # HSV colour ratios
    # ------------------------------------------------------------------

    @staticmethod
    def _color_ratios(
        hsv: np.ndarray, fg_mask: np.ndarray, total_fg: int
    ) -> Tuple[float, float, float]:
        """Return (green_ratio, yellow_ratio, brown_ratio) in [0, 1]."""
        green_mask = cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)
        yellow_mask = cv2.inRange(hsv, _YELLOW_LOWER, _YELLOW_UPPER)
        brown_mask = cv2.inRange(hsv, _BROWN_LOWER, _BROWN_UPPER)

        # Only count foreground pixels
        green_mask = cv2.bitwise_and(green_mask, fg_mask)
        yellow_mask = cv2.bitwise_and(yellow_mask, fg_mask)
        brown_mask = cv2.bitwise_and(brown_mask, fg_mask)

        green_r = int(green_mask.sum() / 255) / total_fg
        yellow_r = int(yellow_mask.sum() / 255) / total_fg
        brown_r = int(brown_mask.sum() / 255) / total_fg

        return green_r, yellow_r, brown_r

    # ------------------------------------------------------------------
    # Edge-region wilting
    # ------------------------------------------------------------------

    @staticmethod
    def _edge_wilting_score(gray: np.ndarray, fg_mask: np.ndarray) -> float:
        """Score edge wilting (0-100, higher = worse).

        Wilted edges show high-frequency texture loss and reduced contrast
        compared to the interior.  We compare gradient energy at the
        contour fringe to the interior.
        """
        # Find contours of the foreground
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0

        # Create a 15-pixel wide edge band
        edge_width = 15
        eroded = cv2.erode(fg_mask, np.ones((edge_width * 2, edge_width * 2), np.uint8))
        edge_band = cv2.subtract(fg_mask, eroded)

        # Gradient magnitude via Sobel
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Mean gradient in edge band vs interior
        edge_pixels = grad_mag[edge_band > 0]
        interior_pixels = grad_mag[eroded > 0]

        if len(edge_pixels) == 0 or len(interior_pixels) == 0:
            return 0.0

        edge_energy = float(np.mean(edge_pixels))
        interior_energy = float(np.mean(interior_pixels))

        if interior_energy < 1e-6:
            return 0.0

        # Wilted edges: low gradient at boundary (smooth, mushy texture)
        ratio = edge_energy / interior_energy
        # ratio < 0.5 → significant wilting; ratio > 1.0 → crisp edges
        if ratio >= 1.0:
            return 0.0  # Crisp
        wilting = (1.0 - ratio) * 100
        return float(np.clip(wilting, 0, 100))

    # ------------------------------------------------------------------
    # Condensation / water-droplet detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_condensation(gray: np.ndarray, fg_mask: np.ndarray) -> bool:
        """Detect small circular bright highlights (water droplets) using
        blob detection and circular Hough transform.
        """
        # Method 1: Circular Hough on bright specular highlights
        # Threshold to find bright spots
        _, bright = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        bright = cv2.bitwise_and(bright, fg_mask)

        circles = cv2.HoughCircles(
            bright,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=10,
            param1=50,
            param2=12,
            minRadius=2,
            maxRadius=20,
        )
        if circles is not None and len(circles[0]) >= 3:
            return True

        # Method 2: SimpleBlobDetector for small bright blobs
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = 8
        params.maxArea = 500
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = True
        params.minConvexity = 0.6

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(bright)

        return len(keypoints) >= 3

    # ------------------------------------------------------------------
    # Composite freshness
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_freshness(
        green_r: float,
        yellow_r: float,
        brown_r: float,
        wilting: float,
        condensation: bool,
    ) -> float:
        """Compute overall freshness score (0-100)."""
        # Base: green ratio contributes positively
        base = green_r * 100

        # Penalties
        base -= yellow_r * 60
        base -= brown_r * 120
        base -= wilting * 0.3

        # Condensation: slight positive (indicates just-picked / refrigerated)
        if condensation:
            base += 5

        return float(np.clip(base, 0, 100))

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def _recommendations(
        freshness: float,
        green_r: float,
        yellow_r: float,
        brown_r: float,
        wilting: float,
        condensation: bool,
        veg_type: str,
    ) -> List[str]:
        """Generate actionable Korean-language recommendations."""
        recs: List[str] = []

        if freshness >= 80:
            recs.append("신선도가 높습니다. 구매를 추천합니다.")
        elif freshness >= 50:
            recs.append("보통 수준의 신선도입니다.")
        else:
            recs.append("신선도가 낮습니다. 다른 상품을 추천합니다.")

        if brown_r > 0.10:
            recs.append(f"갈변 비율이 {brown_r*100:.1f}%로 높습니다. 산화가 진행되었을 수 있습니다.")

        if yellow_r > 0.15:
            recs.append(f"황변 비율이 {yellow_r*100:.1f}%입니다. 보관 기간이 길었을 수 있습니다.")

        if wilting > 40:
            recs.append("가장자리가 시들어 보입니다. 잎 채소라면 물에 담가 살려볼 수 있습니다.")

        if condensation:
            recs.append("표면에 수분이 감지되었습니다. 냉장 보관 중이었을 수 있습니다.")

        return recs
