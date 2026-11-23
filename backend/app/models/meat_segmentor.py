"""Meat segmentation module — lean / fat / bone pixel-level analysis.

Uses LAB colour-space thresholding with connected-component blob analysis
to separate lean meat (red-pink), fat (white-yellow), and bone (hard white).
Optionally leverages MobileSAM for instance-level masks when available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MeatSegResult:
    """Pixel-level segmentation result for a meat cut."""

    lean_ratio: float  # 0-1
    fat_ratio: float   # 0-1
    bone_ratio: float  # 0-1

    lean_mask: np.ndarray   # bool H×W
    fat_mask: np.ndarray    # bool H×W
    bone_mask: np.ndarray   # bool H×W

    color_assessment: str   # e.g. "선홍색 (fresh)", "갈변 (browning)"
    has_blood_spots: bool

    # 떡지방 (large contiguous fat blob) detection
    has_tteok_jibang: bool = False
    tteok_jibang_area_ratio: float = 0.0


# ---------------------------------------------------------------------------
# MeatSegmentor
# ---------------------------------------------------------------------------

class MeatSegmentor:
    """Segment meat images into lean / fat / bone regions.

    The primary pipeline uses classical CV (LAB thresholds + morphology).
    If *use_sam* is True the class tries to load MobileSAM for a refinement
    pass, falling back gracefully to the classical pipeline.
    """

    # Minimum area (fraction of total foreground) to count as 떡지방
    TTEOK_JIBANG_AREA_THRESHOLD = 0.05

    # LAB-space ranges  (L, A, B)
    # Lean meat: moderate L, high A (red), moderate-high B
    # Fat:       high L, low-mid A, mid-high B (yellowish white)
    # Bone:      very high L, low A, low B (hard neutral white)

    def __init__(self, use_sam: bool = False) -> None:
        self._use_sam = use_sam
        self._sam_model = None  # lazy

    # ------------------------------------------------------------------
    # Lighting correction
    # ------------------------------------------------------------------

    @staticmethod
    def correct_lighting(image: np.ndarray) -> np.ndarray:
        """White-balance correction using the gray-world assumption.

        Each channel is scaled so that its mean equals the global mean
        across all channels.
        """
        if image is None or image.size == 0:
            return image

        img = image.astype(np.float32)
        channel_means = img.mean(axis=(0, 1))  # per-channel means
        global_mean = channel_means.mean()

        # Avoid division by zero
        scale = np.where(channel_means > 1e-6, global_mean / channel_means, 1.0)
        corrected = np.clip(img * scale, 0, 255).astype(np.uint8)
        return corrected

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, image: np.ndarray) -> MeatSegResult:
        """Run the full segmentation pipeline on a BGR image.

        Returns a :class:`MeatSegResult` with ratios, masks, and
        qualitative colour assessment.
        """
        if image is None or image.size == 0:
            raise ValueError("segment() received an empty image.")

        # 1. Lighting correction
        corrected = self.correct_lighting(image)

        # 2. Convert to LAB
        lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        # 3. Build foreground mask (remove background via Otsu on L)
        _, fg_mask = cv2.threshold(l_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        total_fg = max(int(fg_mask.sum() / 255), 1)

        # 4. Classify each foreground pixel
        lean_mask, fat_mask, bone_mask = self._classify_pixels(
            l_ch, a_ch, b_ch, fg_mask
        )

        # 5. Morphological cleanup per mask
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for m in (lean_mask, fat_mask, bone_mask):
            cv2.morphologyEx(m, cv2.MORPH_CLOSE, small_kernel, iterations=1, dst=m)
            cv2.morphologyEx(m, cv2.MORPH_OPEN, small_kernel, iterations=1, dst=m)

        # 6. Ratios
        lean_px = int(lean_mask.sum() / 255)
        fat_px = int(fat_mask.sum() / 255)
        bone_px = int(bone_mask.sum() / 255)
        total_classified = max(lean_px + fat_px + bone_px, 1)

        lean_ratio = round(lean_px / total_classified, 4)
        fat_ratio = round(fat_px / total_classified, 4)
        bone_ratio = round(bone_px / total_classified, 4)

        # 7. Colour assessment
        color_assessment = self._assess_color(corrected, lean_mask)

        # 8. Blood-spot detection
        has_blood_spots = self._detect_blood_spots(corrected, lean_mask)

        # 9. 떡지방 detection (large contiguous fat blobs)
        has_tteok, tteok_ratio = self._detect_tteok_jibang(fat_mask, total_fg)

        return MeatSegResult(
            lean_ratio=lean_ratio,
            fat_ratio=fat_ratio,
            bone_ratio=bone_ratio,
            lean_mask=(lean_mask > 0),
            fat_mask=(fat_mask > 0),
            bone_mask=(bone_mask > 0),
            color_assessment=color_assessment,
            has_blood_spots=has_blood_spots,
            has_tteok_jibang=has_tteok,
            tteok_jibang_area_ratio=tteok_ratio,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_pixels(
        l_ch: np.ndarray,
        a_ch: np.ndarray,
        b_ch: np.ndarray,
        fg_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Threshold LAB channels into lean / fat / bone masks.

        LAB ranges (OpenCV 0-255 encoding):
          - L:   0-255 (lightness)
          - A: 128 = neutral; >128 red, <128 green
          - B: 128 = neutral; >128 yellow, <128 blue

        Lean (red-pink meat):  A > 138,  L in 40-200
        Fat  (white-yellow):   L > 170,  A in 118-138,  B > 130
        Bone (hard white):     L > 190,  A in 120-135,  B in 118-138
        """
        fg = fg_mask > 0

        # Lean: distinctly red (high A), not too bright
        lean = fg & (a_ch > 138) & (l_ch > 40) & (l_ch < 200)

        # Bone: very bright, nearly neutral a/b
        bone = (
            fg
            & (l_ch > 190)
            & (a_ch >= 120) & (a_ch <= 135)
            & (b_ch >= 118) & (b_ch <= 138)
        )

        # Fat: bright, low-redness, yellowish
        fat = (
            fg
            & (l_ch > 170)
            & (a_ch >= 118) & (a_ch <= 138)
            & (b_ch > 130)
            & ~bone  # bone takes priority over fat
        )

        lean_mask = np.zeros_like(l_ch, dtype=np.uint8)
        fat_mask = np.zeros_like(l_ch, dtype=np.uint8)
        bone_mask = np.zeros_like(l_ch, dtype=np.uint8)

        lean_mask[lean] = 255
        fat_mask[fat] = 255
        bone_mask[bone] = 255

        return lean_mask, fat_mask, bone_mask

    @staticmethod
    def _assess_color(bgr: np.ndarray, lean_mask: np.ndarray) -> str:
        """Qualitative color assessment of the lean-meat region."""
        if lean_mask.sum() == 0:
            return "판별 불가 (insufficient lean area)"

        # Extract lean pixels in HSV
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_vals = hsv[:, :, 0][lean_mask > 0]
        s_vals = hsv[:, :, 1][lean_mask > 0]
        v_vals = hsv[:, :, 2][lean_mask > 0]

        mean_h = float(np.mean(h_vals))
        mean_s = float(np.mean(s_vals))
        mean_v = float(np.mean(v_vals))

        # Hue ~0-10 or ~170-180 = red region in OpenCV HSV
        if mean_s > 100 and mean_v > 100:
            if (mean_h < 12) or (mean_h > 165):
                return "선홍색 (bright cherry-red, fresh)"
            elif mean_h < 25:
                return "적갈색 (reddish-brown, slightly aged)"
            else:
                return "갈변 (browning, check freshness)"
        elif mean_v < 80:
            return "어두운 색 (dark, possibly oxidized)"
        else:
            return "연한 분홍 (pale pink, may be low myoglobin)"

    @staticmethod
    def _detect_blood_spots(bgr: np.ndarray, lean_mask: np.ndarray) -> bool:
        """Detect dark-red blood spots within the lean region.

        Blood spots appear as small, very dark & highly saturated red
        patches compared to surrounding lean tissue.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Dark saturated red
        lower = np.array([0, 120, 20], dtype=np.uint8)
        upper = np.array([10, 255, 100], dtype=np.uint8)
        blood_mask = cv2.inRange(hsv, lower, upper)

        # Also wrap-around red hue
        lower2 = np.array([170, 120, 20], dtype=np.uint8)
        upper2 = np.array([180, 255, 100], dtype=np.uint8)
        blood_mask |= cv2.inRange(hsv, lower2, upper2)

        # Only count inside lean region
        blood_mask = cv2.bitwise_and(blood_mask, lean_mask)

        # Require at least a few connected pixels
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            blood_mask, connectivity=8
        )
        min_area = 30  # pixels
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                return True
        return False

    def _detect_tteok_jibang(
        self, fat_mask: np.ndarray, total_fg_pixels: int
    ) -> Tuple[bool, float]:
        """Detect 떡지방 — large contiguous fat blobs.

        Returns (detected: bool, largest_blob_ratio: float).
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            fat_mask, connectivity=8
        )

        max_ratio = 0.0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            ratio = area / total_fg_pixels
            if ratio > max_ratio:
                max_ratio = ratio

        detected = max_ratio >= self.TTEOK_JIBANG_AREA_THRESHOLD
        return detected, round(max_ratio, 4)

    # ------------------------------------------------------------------
    # Optional SAM refinement (graceful fallback)
    # ------------------------------------------------------------------

    def _load_sam(self) -> None:
        """Try to load MobileSAM. Non-fatal if unavailable."""
        if not self._use_sam or self._sam_model is not None:
            return
        try:
            from ultralytics import SAM

            self._sam_model = SAM("mobile_sam.pt")
            logger.info("MobileSAM loaded for refinement.")
        except Exception as exc:
            logger.warning("MobileSAM not available, using classical pipeline only: %s", exc)
            self._use_sam = False
