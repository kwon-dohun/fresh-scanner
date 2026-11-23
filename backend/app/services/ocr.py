"""OCR 서비스 어댑터 모듈."""

from __future__ import annotations

import cv2
import numpy as np

from backend.app.services.ocr_service import (
    OCRService as BaseOCRService,
    OCRResult,
    MeatLabel,
    PriceTag,
    Receipt,
    ReceiptItem,
)


class OCRService(BaseOCRService):
    """비동기 바이트 입력 및 편의 메서드를 지원하는 OCR 서비스."""

    async def extract_text(
        self,
        image_input: bytes | np.ndarray,
        region: tuple[int, int, int, int] | None = None,
    ) -> str:
        """이미지(바이트 또는 ndarray)에서 전체 텍스트를 추출하여 하나의 문자열로 반환."""
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = image_input

        if img is None:
            return ""

        results: list[OCRResult] = super().extract_text(img, region=region)
        return "\n".join(r.text for r in results)


__all__ = ["OCRService", "OCRResult", "MeatLabel", "PriceTag", "Receipt", "ReceiptItem"]
