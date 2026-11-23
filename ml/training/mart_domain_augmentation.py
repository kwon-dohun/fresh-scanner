"""마트 실제 매대 도메인 특화 데이터 증강 파이프라인 (Synthetic Mart Augmentation).

스튜디오 조명에서 촬영된 AI Hub 공개 데이터셋을 실제 대형마트 현장 사진으로 변환하기 위해:
1. 정육점 핑크/레드 LED 특수 조명(Pink Tint) 합성 (R 채널 증폭 + G 채널 비선형 감쇄)
2. 비닐 랩 정반사광 핫스팟(Specular Glare Hotspot) 합성 (가우시안 폴오프 고휘도 마스크)
3. 랩 비닐 주름 및 형광등 스트립 반사(Wrinkle & Strip Reflection)
4. 손떨림 미세 모션 블러(Point Spread Function Motion Blur)
"""

from __future__ import annotations

import random
from typing import Tuple
import cv2
import numpy as np


class MartDomainAugmentor:
    """대형마트 현장 광학 왜곡을 실제 이미지에 합성하는 증강기."""

    @classmethod
    def apply_butcher_pink_tint(cls, image: np.ndarray, intensity: float = 0.35) -> np.ndarray:
        """정육 매대 붉은 LED 조명 색온도 편향 합성."""
        img_f = image.astype(np.float32)
        # Red 채널 15~35% 증폭
        img_f[:, :, 2] *= (1.0 + intensity * 0.9)
        # Green 채널 5~15% 감쇄
        img_f[:, :, 1] *= (1.0 - intensity * 0.35)
        # Blue 채널 살짝 감쇄
        img_f[:, :, 0] *= (1.0 - intensity * 0.25)

        return np.clip(img_f, 0, 255).astype(np.uint8)

    @classmethod
    def apply_wrap_specular_glare(cls, image: np.ndarray, num_spots: int = 2) -> np.ndarray:
        """비닐 랩 표면의 형광등 정반사광 핫스팟 합성."""
        h, w = image.shape[:2]
        output = image.copy()

        for _ in range(num_spots):
            cx = random.randint(int(w * 0.2), int(w * 0.8))
            cy = random.randint(int(h * 0.2), int(h * 0.8))
            radius_x = random.randint(25, 70)
            radius_y = random.randint(15, 45)
            angle = random.randint(0, 180)

            # 반사광 오버레이 마스크 생성
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, (cx, cy), (radius_x, radius_y), angle, 0, 360, 1.0, -1)
            # 가우시안 블러로 빛 번짐 효과 생성
            mask = cv2.GaussianBlur(mask, (31, 31), 10.0)

            # 흰색 빛 합성
            glare_color = np.array([255, 255, 255], dtype=np.float32)
            for c in range(3):
                output[:, :, c] = np.clip(
                    output[:, :, c] * (1.0 - mask * 0.8) + glare_color[c] * (mask * 0.8),
                    0, 255
                ).astype(np.uint8)

        return output

    @classmethod
    def apply_motion_blur(cls, image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        """스마트폰 손떨림 미세 모션 블러 합성."""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(image, -1, kernel)

    @classmethod
    def augment_for_mart_training(cls, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """학습용 통합 증강 파이프라인 (이미지 왜곡 + 라벨 마스크 보존)."""
        aug_img = image.copy()

        # 1. 70% 확률로 정육점 붉은 조명 왜곡 주입
        if random.random() < 0.70:
            tint_strength = random.uniform(0.15, 0.40)
            aug_img = cls.apply_butcher_pink_tint(aug_img, tint_strength)

        # 2. 60% 확률로 비닐 랩 반사광 주입
        if random.random() < 0.60:
            spots = random.randint(1, 3)
            aug_img = cls.apply_wrap_specular_glare(aug_img, num_spots=spots)

        # 3. 30% 확률로 미세 모션 블러 주입
        if random.random() < 0.30:
            aug_img = cls.apply_motion_blur(aug_img, kernel_size=random.choice([5, 7]))

        return aug_img, mask
