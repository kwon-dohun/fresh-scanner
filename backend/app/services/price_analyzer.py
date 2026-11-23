"""가격 및 가성비 분석 서비스."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from backend.app.services.ocr_service import OCRService, MeatLabel, PriceTag
from backend.app.services.kamis_api import KAMISClient, PriceComparison


@dataclass
class MeatValueResult:
    label_price_per_100g: int
    lean_ratio: float
    effective_lean_price: int
    value_verdict: str


@dataclass
class ProduceValueResult:
    store_price: float
    market_price: float
    difference_pct: float
    verdict: str


class PriceAnalyzer:
    """단가 OCR과 비전 분석을 결합하여 실질 가성비 도출."""

    def __init__(self, ocr_service: OCRService | None = None, kamis_client: KAMISClient | None = None) -> None:
        self.ocr_service = ocr_service or OCRService()
        self.kamis_client = kamis_client or KAMISClient()

    def analyze_meat_value(self, image: np.ndarray, lean_ratio: float) -> MeatValueResult | None:
        """팩 라벨 단가 × 살코기 비율 = 실질 살코기 가성비 계산."""
        label: MeatLabel | None = self.ocr_service.extract_meat_label(image)
        if not label or not label.price_per_100g:
            return None

        unit_p = label.price_per_100g
        effective_p = int(unit_p / (lean_ratio + 1e-6))

        verdict = f"표기 가격은 {unit_p}원이나, 순수 살코기 기준 가성비 단가는 100g당 {effective_p}원입니다."

        return MeatValueResult(
            label_price_per_100g=unit_p,
            lean_ratio=lean_ratio,
            effective_lean_price=effective_p,
            value_verdict=verdict,
        )

    async def analyze_produce_value(self, image: np.ndarray, item_name: str) -> ProduceValueResult | None:
        """매대 POP 가격표 OCR × KAMIS 시세 API 대조."""
        tag: PriceTag | None = self.ocr_service.extract_price_tag(image)
        if not tag or not tag.price:
            return None

        comp: PriceComparison = await self.kamis_client.compare_to_market(item_name, float(tag.price))

        return ProduceValueResult(
            store_price=float(tag.price),
            market_price=float(comp.market_avg),
            difference_pct=comp.difference_pct,
            verdict=comp.verdict_kr,
        )
