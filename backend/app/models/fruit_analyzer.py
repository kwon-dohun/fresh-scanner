"""Fruit quality analysis via classical computer-vision techniques.

Supports: watermelon, apple, hallabong, grape, shine_muscat, strawberry.
Uses colour-histogram analysis, edge detection (stripe clarity), stem
freshness scoring, and anomaly detection for defects/bruises.
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
class FruitQualityResult:
    """Comprehensive quality report for a single fruit item."""

    quality_score: float              # 0-100 overall
    surface_pattern_score: float      # 0-100
    stem_freshness_score: float       # 0-100
    defect_count: int
    defect_locations: List[Tuple[int, int, int, int]]  # list of (x1,y1,x2,y2)
    color_distribution: Dict[str, float]               # color_name -> ratio
    recommendations: List[str]


# ---------------------------------------------------------------------------
# Fruit type configs
# ---------------------------------------------------------------------------

_FRUIT_COLOR_RANGES: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = {
    # (label, hsv_lower, hsv_upper) — OpenCV HSV: H 0-179, S/V 0-255
    "watermelon": [
        ("dark_green", np.array([35, 60, 30]), np.array([85, 255, 120])),
        ("light_green", np.array([35, 40, 120]), np.array([85, 255, 255])),
        ("yellow_stripe", np.array([20, 40, 120]), np.array([35, 255, 255])),
    ],
    "apple": [
        ("red", np.array([0, 80, 60]), np.array([10, 255, 255])),
        ("red_wrap", np.array([170, 80, 60]), np.array([179, 255, 255])),
        ("green", np.array([35, 40, 40]), np.array([85, 255, 255])),
        ("yellow", np.array([15, 40, 100]), np.array([35, 255, 255])),
    ],
    "hallabong": [
        ("orange", np.array([10, 100, 100]), np.array([25, 255, 255])),
        ("green_stem", np.array([35, 40, 40]), np.array([85, 255, 200])),
    ],
    "grape": [
        ("purple", np.array([120, 30, 30]), np.array([160, 255, 200])),
        ("dark_red", np.array([0, 50, 20]), np.array([10, 255, 150])),
        ("green", np.array([35, 30, 60]), np.array([85, 255, 255])),
    ],
    "shine_muscat": [
        ("yellow_green", np.array([25, 40, 100]), np.array([50, 255, 255])),
        ("green", np.array([35, 40, 80]), np.array([85, 255, 255])),
    ],
    "strawberry": [
        ("red", np.array([0, 100, 80]), np.array([10, 255, 255])),
        ("red_wrap", np.array([170, 100, 80]), np.array([179, 255, 255])),
        ("green_calyx", np.array([35, 40, 40]), np.array([85, 255, 255])),
        ("white_unripe", np.array([0, 0, 180]), np.array([179, 50, 255])),
    ],
}


# ---------------------------------------------------------------------------
# FruitAnalyzer
# ---------------------------------------------------------------------------

class FruitAnalyzer:
    """Analyse fruit quality from a BGR image.

    All heavy computation uses OpenCV / NumPy — no ML model required.
    """

    SUPPORTED_TYPES = frozenset(_FRUIT_COLOR_RANGES.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self, image: np.ndarray, fruit_type: str
    ) -> FruitQualityResult:
        """Return a :class:`FruitQualityResult` for the given fruit image."""
        if fruit_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported fruit_type '{fruit_type}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_TYPES))}"
            )
        if image is None or image.size == 0:
            raise ValueError("analyze() received an empty image.")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        color_dist = self._color_distribution(hsv, fruit_type)
        surface_score = self._surface_pattern_score(gray, hsv, fruit_type)
        stem_score = self._stem_freshness_score(hsv, fruit_type)
        defects, defect_bboxes = self._detect_defects(image, hsv, fruit_type)

        # Composite quality
        quality = self._composite_quality(
            surface_score, stem_score, defects, color_dist, fruit_type
        )

        recommendations = self._generate_recommendations(
            quality, surface_score, stem_score, defects, fruit_type
        )

        return FruitQualityResult(
            quality_score=round(quality, 1),
            surface_pattern_score=round(surface_score, 1),
            stem_freshness_score=round(stem_score, 1),
            defect_count=defects,
            defect_locations=defect_bboxes,
            color_distribution={k: round(v, 4) for k, v in color_dist.items()},
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Colour distribution
    # ------------------------------------------------------------------

    def _color_distribution(
        self, hsv: np.ndarray, fruit_type: str
    ) -> Dict[str, float]:
        """Compute the fraction of fruit pixels falling into each named
        colour range defined for *fruit_type*."""
        ranges = _FRUIT_COLOR_RANGES[fruit_type]
        total_pixels = max(hsv.shape[0] * hsv.shape[1], 1)
        dist: Dict[str, float] = {}

        for label, lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            count = int(mask.sum() / 255)
            dist[label] = count / total_pixels

        # Merge red + red_wrap if both exist
        if "red" in dist and "red_wrap" in dist:
            dist["red"] = dist["red"] + dist.pop("red_wrap")

        # "other" bucket
        accounted = sum(dist.values())
        dist["other"] = max(0.0, 1.0 - accounted)
        return dist

    # ------------------------------------------------------------------
    # Surface-pattern analysis
    # ------------------------------------------------------------------

    def _surface_pattern_score(
        self, gray: np.ndarray, hsv: np.ndarray, fruit_type: str
    ) -> float:
        """Score surface pattern clarity (0-100).

        For watermelons this measures stripe regularity via edge density.
        For other fruits it evaluates colour uniformity.
        """
        if fruit_type == "watermelon":
            return self._watermelon_stripe_score(gray)

        # General: low variance in V-channel of foreground → uniform → good
        v_ch = hsv[:, :, 2]
        # Simple foreground: Otsu on gray
        _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_pixels = v_ch[fg > 0]
        if len(fg_pixels) == 0:
            return 50.0
        cv_coeff = float(np.std(fg_pixels)) / max(float(np.mean(fg_pixels)), 1.0)
        # Lower CV → more uniform → higher score
        score = max(0.0, 100.0 - cv_coeff * 300)
        return np.clip(score, 0, 100)

    @staticmethod
    def _watermelon_stripe_score(gray: np.ndarray) -> float:
        """Measure stripe clarity using Canny edge density in horizontal
        band through the centre."""
        h, w = gray.shape[:2]
        # Central horizontal band (40% of height)
        y1, y2 = int(h * 0.3), int(h * 0.7)
        band = gray[y1:y2, :]

        edges = cv2.Canny(band, 30, 100)
        edge_density = float(edges.sum() / 255) / max(band.size, 1)

        # Good stripes → edge density ~ 0.03-0.08
        if edge_density < 0.01:
            return 30.0  # too few edges — blurry or no stripes
        elif edge_density > 0.12:
            return 50.0  # too noisy
        else:
            # Map [0.02, 0.08] → [60, 100]
            score = 60 + (edge_density - 0.02) / 0.06 * 40
            return float(np.clip(score, 30, 100))

    # ------------------------------------------------------------------
    # Stem freshness
    # ------------------------------------------------------------------

    def _stem_freshness_score(self, hsv: np.ndarray, fruit_type: str) -> float:
        """Evaluate stem / calyx freshness via green-channel ratio in the
        top 20 % of the image (where the stem typically appears)."""
        h, w = hsv.shape[:2]
        top_region = hsv[: int(h * 0.20), :]

        # Green mask in top region
        green_lower = np.array([35, 40, 40], dtype=np.uint8)
        green_upper = np.array([85, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(top_region, green_lower, green_upper)

        green_ratio = float(green_mask.sum() / 255) / max(top_region.shape[0] * top_region.shape[1], 1)

        # Brown / yellow in top region (wilted stem)
        brown_lower = np.array([10, 40, 20], dtype=np.uint8)
        brown_upper = np.array([30, 200, 150], dtype=np.uint8)
        brown_mask = cv2.inRange(top_region, brown_lower, brown_upper)
        brown_ratio = float(brown_mask.sum() / 255) / max(top_region.shape[0] * top_region.shape[1], 1)

        # Score: more green → fresher
        if fruit_type in ("strawberry",):
            # Calyx prominence expected
            if green_ratio > 0.10:
                score = 90.0
            elif green_ratio > 0.04:
                score = 70.0
            else:
                score = 40.0
        else:
            if green_ratio > 0.05:
                score = 85.0
            elif green_ratio > 0.02:
                score = 65.0
            else:
                score = 45.0

        # Penalise browning
        score -= brown_ratio * 200
        return float(np.clip(score, 0, 100))

    # ------------------------------------------------------------------
    # Defect / bruise detection
    # ------------------------------------------------------------------

    def _detect_defects(
        self,
        bgr: np.ndarray,
        hsv: np.ndarray,
        fruit_type: str,
    ) -> Tuple[int, List[Tuple[int, int, int, int]]]:
        """Detect surface defects / bruises via colour anomaly on small
        patches.

        Uses a sliding-window approach: divide image into patches, compute
        mean LAB colour, then flag patches whose L*a*b* distance from the
        global mean exceeds a threshold (simple isolation-style anomaly).
        """
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        h, w = lab.shape[:2]

        # Foreground mask
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Global mean colour of foreground
        fg_bool = fg > 0
        if fg_bool.sum() == 0:
            return 0, []
        global_mean = lab[fg_bool].mean(axis=0)  # shape (3,)

        patch_size = max(16, min(h, w) // 15)
        threshold = 35.0  # LAB Euclidean distance

        defect_bboxes: List[Tuple[int, int, int, int]] = []

        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch_fg = fg[y: y + patch_size, x: x + patch_size]
                if patch_fg.sum() / 255 < patch_size * patch_size * 0.5:
                    continue  # mostly background
                patch_lab = lab[y: y + patch_size, x: x + patch_size]
                patch_mean = patch_lab[patch_fg > 0].mean(axis=0)
                dist = float(np.linalg.norm(patch_mean - global_mean))
                if dist > threshold:
                    defect_bboxes.append((x, y, x + patch_size, y + patch_size))

        # Merge overlapping / adjacent boxes
        defect_bboxes = self._merge_bboxes(defect_bboxes, gap=patch_size)
        return len(defect_bboxes), defect_bboxes

    @staticmethod
    def _merge_bboxes(
        bboxes: List[Tuple[int, int, int, int]], gap: int = 0
    ) -> List[Tuple[int, int, int, int]]:
        """Greedy merge of overlapping / adjacent bounding boxes."""
        if not bboxes:
            return []

        merged = list(bboxes)
        changed = True
        while changed:
            changed = False
            new_merged: List[Tuple[int, int, int, int]] = []
            used = set()
            for i in range(len(merged)):
                if i in used:
                    continue
                x1, y1, x2, y2 = merged[i]
                for j in range(i + 1, len(merged)):
                    if j in used:
                        continue
                    bx1, by1, bx2, by2 = merged[j]
                    # Check overlap / adjacency
                    if (
                        x1 - gap <= bx2
                        and x2 + gap >= bx1
                        and y1 - gap <= by2
                        and y2 + gap >= by1
                    ):
                        x1 = min(x1, bx1)
                        y1 = min(y1, by1)
                        x2 = max(x2, bx2)
                        y2 = max(y2, by2)
                        used.add(j)
                        changed = True
                new_merged.append((x1, y1, x2, y2))
                used.add(i)
            merged = new_merged
        return merged

    # ------------------------------------------------------------------
    # Composite quality & recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def _composite_quality(
        surface: float,
        stem: float,
        defect_count: int,
        color_dist: Dict[str, float],
        fruit_type: str,
    ) -> float:
        """Weighted composite quality score (0-100)."""
        # Weights depend on fruit type
        w_surface, w_stem, w_defect, w_color = 0.30, 0.15, 0.30, 0.25

        defect_penalty = min(defect_count * 10, 60)
        defect_score = max(0, 100 - defect_penalty)

        # Color score: ratio of "desirable" colour
        desirable_keys = {
            "watermelon": ["dark_green", "light_green"],
            "apple": ["red"],
            "hallabong": ["orange"],
            "grape": ["purple", "dark_red"],
            "shine_muscat": ["yellow_green", "green"],
            "strawberry": ["red"],
        }
        desired_ratio = sum(color_dist.get(k, 0) for k in desirable_keys.get(fruit_type, []))
        color_score = min(desired_ratio * 200, 100)  # scale up

        total = (
            w_surface * surface
            + w_stem * stem
            + w_defect * defect_score
            + w_color * color_score
        )
        return float(np.clip(total, 0, 100))

    @staticmethod
    def _generate_recommendations(
        quality: float,
        surface: float,
        stem: float,
        defect_count: int,
        fruit_type: str,
    ) -> List[str]:
        """Generate actionable recommendations in Korean."""
        recs: List[str] = []

        if quality >= 85:
            recs.append("전반적으로 우수한 품질입니다. 구매를 추천합니다.")
        elif quality >= 60:
            recs.append("양호한 품질이나 일부 주의 사항이 있습니다.")
        else:
            recs.append("품질이 낮습니다. 다른 상품을 비교해 보세요.")

        if surface < 50:
            if fruit_type == "watermelon":
                recs.append("줄무늬가 흐릿합니다 — 덜 익었을 수 있습니다.")
            else:
                recs.append("표면 색상이 고르지 않습니다.")

        if stem < 50:
            recs.append("꼭지 부분이 갈변되어 신선도가 떨어질 수 있습니다.")

        if defect_count > 0:
            recs.append(f"표면 결점 {defect_count}개가 발견되었습니다. 멍이나 상처를 확인하세요.")

        return recs
