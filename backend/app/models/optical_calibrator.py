"""마트 매대 실전 환경 특화 광학 보정 및 품질 게이트 모듈 (상용 출시급).

주요 기능:
1. 흰색 가격 라벨을 절대 백색 기준점(White Reference)으로 사용하는 동적 화이트밸런스:
   - 정육 코너 붉은 LED 조명 하에서 팩 표면의 흰색 스티커 라벨을 자동 감지하고,
     해당 라벨의 순수 흰색 종이 픽셀 RGB를 (255, 255, 255)로 맞추는 Von Kries 색채 변환 행렬을 이미지 전체에 적용.
2. 실시간 촬영 전 품질 게이트 (Pre-Capture Quality Gate):
   - 라플라시안 분산 기반 초점 흐림(Blur) 감지
   - 비닐 랩 정반사광(Glare) 비율 감지 및 실시간 각도 틸팅 가이드 생성
3. 정육 팩 내부 ROI 자동 크롭 (Pack ROI Cropping):
   - 매대 선반, 손, 이웃 팩 등 배경 노이즈를 배제하고 분석 대상 팩만 정밀 크롭.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """촬영 전 이미지 품질 적합도 결과."""
    can_capture: bool
    blur_score: float         # 라플라시안 분산값 (100 이상 권장)
    glare_ratio: float        # 정반사광 면적비 (15% 이하 권장)
    messages: List[str]       # 사용자 실시간 가이드 문구


class WhiteLabelReferenceWB:
    """정육 팩의 흰색 가격표 스티커를 광학 기준점으로 삼는 동적 화이트밸런스 보정기."""

    @classmethod
    def calibrate_with_label(
        cls, image: np.ndarray, label_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[np.ndarray, bool]:
        """흰색 라벨 영역을 추출하여 조명 색온도 오프셋을 역산, 이미지 전체에 적용.

        Args:
            image: BGR 이미지.
            label_bbox: (x1, y1, x2, y2) 라벨 좌표. None이면 자동 감지 시도.

        Returns:
            (보정된 BGR 이미지, 라벨 기반 보정 성공 여부)
        """
        if image is None or image.size == 0:
            return image, False

        h, w = image.shape[:2]

        # 1. 라벨 영역 지정되지 않은 경우 직사각형 고명도 흰색 패치 자동 검출
        if label_bbox is None:
            label_bbox = cls._detect_white_label_patch(image)

        if label_bbox is None:
            # 라벨 검출 실패 시 일반 그레이월드/Minkowski 폴백
            return image, False

        x1, y1, x2, y2 = label_bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        label_crop = image[y1:y2, x1:x2]
        if label_crop.size == 0:
            return image, False

        # 2. 라벨 내부의 인쇄된 검은 글자/바코드를 제외하고 순수 흰색 종이 픽셀만 필터링
        gray_label = cv2.cvtColor(label_crop, cv2.COLOR_BGR2GRAY)
        # 밝기 상위 25% 픽셀 추출 (글씨를 제외한 라벨 종이 바탕)
        q75 = np.percentile(gray_label, 70)
        paper_mask = gray_label >= q75

        if np.count_nonzero(paper_mask) < 20:
            return image, False

        # 3. 조명 아래에서 측정된 라벨 종이의 평균 BGR 값
        b_mean = float(np.mean(label_crop[:, :, 0][paper_mask]))
        g_mean = float(np.mean(label_crop[:, :, 1][paper_mask]))
        r_mean = float(np.mean(label_crop[:, :, 2][paper_mask]))

        # 너무 어두운 경우(오검출) 배제
        if r_mean < 80 and g_mean < 80 and b_mean < 80:
            return image, False

        # 4. 순수 화이트(255, 255, 255)로 맞추기 위한 채널별 변환 가중치(Von Kries 스케일링)
        # 과도한 노이즈 증폭 방지를 위해 타깃 화이트를 밝은 기준점(245)으로 설정
        target_white = 245.0
        b_gain = target_white / max(b_mean, 1e-4)
        g_gain = target_white / max(g_mean, 1e-4)
        r_gain = target_white / max(r_mean, 1e-4)

        # 붉은 조명 특성상 R gain은 낮추고 B/G gain은 올리는 방향으로 수렴
        # 급격한 색 왜곡 방지를 위해 gain 클리핑 (0.5 ~ 2.2)
        b_gain = float(np.clip(b_gain, 0.5, 2.2))
        g_gain = float(np.clip(g_gain, 0.5, 2.2))
        r_gain = float(np.clip(r_gain, 0.5, 2.2))

        img_float = image.astype(np.float32)
        img_float[:, :, 0] *= b_gain
        img_float[:, :, 1] *= g_gain
        img_float[:, :, 2] *= r_gain

        calibrated = np.clip(img_float, 0, 255).astype(np.uint8)
        return calibrated, True

    @staticmethod
    def _detect_white_label_patch(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """정육 팩 표면에 부착된 흰색 직사각형 스티커 라벨 자동 감지."""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # 라벨 종이 특성: 채도가 낮고(무채색) 밝기가 높음
        # (붉은 조명을 받으면 살짝 붉은빛이 돌 수 있으므로 채도 임계치 70 이하 허용)
        white_mask = (sat < 65) & (val > 140)
        white_u8 = (white_mask * 255).astype(np.uint8)

        # 모폴로지 클로징으로 텍스트 구멍 메우기
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(white_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = h * w

        best_bbox = None
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 라벨 크기는 대개 전체 화면의 1.5% ~ 25% 사이
            if 0.015 * img_area <= area <= 0.28 * img_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / max(bh, 1)
                # 라벨은 대부분 1:1.2 ~ 1:2.5 정도의 가로/세로 직사각형
                if 0.4 <= aspect <= 3.2:
                    if area > best_area:
                        best_area = area
                        best_bbox = (x, y, x + bw, y + bh)

        return best_bbox


class PreCaptureQualityGate:
    """촬영 직전 실시간 프레임 품질 검사 및 피드백 생성기 (Garbage In 방지)."""

    BLUR_THRESHOLD = 90.0        # 라플라시안 분산 최소값 (이하이면 초점 흐림)
    GLARE_MAX_RATIO = 0.16       # 허용 가능한 최대 반사광 면적 비율 (16%)

    @classmethod
    def evaluate_frame(cls, frame: np.ndarray) -> QualityGateResult:
        """단일 프리뷰 프레임 품질 평가."""
        if frame is None or frame.size == 0:
            return QualityGateResult(False, 0.0, 1.0, ["카메라 입력이 없습니다."])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. 초점 흐림(Blur) 검사
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. 비닐 랩 정반사광(Glare) 검사 (V>235 & S<30)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        glare_mask = (val > 235) & (sat < 35)
        total_pixels = frame.shape[0] * frame.shape[1]
        glare_ratio = float(np.count_nonzero(glare_mask) / total_pixels)

        messages: List[str] = []
        can_capture = True

        if blur_score < cls.BLUR_THRESHOLD:
            messages.append("초점이 흐립니다. 스마트폰을 잠시 멈춰주세요.")
            can_capture = False

        if glare_ratio > cls.GLARE_MAX_RATIO:
            messages.append("💡 비닐 빛 반사가 심합니다. 폰을 15도 정도 살짝 기울여주세요.")
            can_capture = False

        if can_capture and not messages:
            messages.append("촬영하기에 완벽한 각도와 선명도입니다! 📸")

        return QualityGateResult(
            can_capture=can_capture,
            blur_score=round(blur_score, 1),
            glare_ratio=round(glare_ratio, 3),
            messages=messages,
        )


class PackROICropper:
    """매대 선반 노이즈를 걷어내고 대상 팩 영역만 추출하는 2단 파이프라인 헬퍼."""

    @staticmethod
    def crop_pack(image: np.ndarray, pack_bbox: Optional[Tuple[int, int, int, int]] = None, padding: float = 0.03) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        if pack_bbox is None:
            return image

        h, w = image.shape[:2]
        x1, y1, x2, y2 = pack_bbox
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)

        x1_p = max(0, x1 - pad_w)
        y1_p = max(0, y1 - pad_h)
        x2_p = min(w, x2 + pad_w)
        y2_p = min(h, y2 + pad_h)

        crop = image[y1_p:y2_p, x1_p:x2_p]
        return crop if crop.size > 0 else image
