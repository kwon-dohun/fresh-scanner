"""KAMIS (한국농수산식품유통공사) 공공 시세 연동 및 지역별 벤치마크 엔진."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

BENCHMARK_PATH = Path(__file__).resolve().parents[3] / "data" / "benchmark_kamis_dataset.json"
TODAY_PATH = Path(__file__).resolve().parents[3] / "data" / "kamis_today.json"


@dataclass
class KAMISPrice:
    item_name: str
    date: str
    wholesale_price: int
    retail_price: int
    unit: str
    change_rate: float
    grade: str = "상품"
    region: str = "전국"


@dataclass
class PriceComparison:
    market_avg: int
    store_price: float
    difference_pct: float
    verdict_kr: str
    min_fair_price: int = 0
    max_fair_price: int = 0
    is_bargain: bool = False


# KAMIS 공공 품목 코드 매핑
KAMIS_ITEM_MAP: dict[str, tuple[str, str, str, str]] = {
    "사과": ("411", "01", "10개", "400"),
    "배": ("412", "01", "10개", "400"),
    "수박": ("413", "01", "1통", "400"),
    "참외": ("414", "01", "10개", "400"),
    "포도": ("415", "01", "2kg", "400"),
    "샤인머스캣": ("415", "03", "2kg", "400"),
    "감귤": ("418", "01", "10kg", "400"),
    "한라봉": ("418", "03", "10kg", "400"),
    "딸기": ("419", "01", "100g", "400"),
    "바나나": ("420", "01", "100g", "400"),
    "배추": ("211", "01", "1포기", "200"),
    "양배추": ("212", "01", "1포기", "200"),
    "시금치": ("213", "01", "1단", "200"),
    "상추": ("214", "01", "100g", "200"),
    "깻잎": ("215", "01", "100g", "200"),
    "양파": ("226", "01", "1kg", "200"),
    "대파": ("246", "01", "1kg", "200"),
    "마늘": ("231", "01", "1kg", "200"),
    "감자": ("222", "01", "100g", "200"),
    "고구마": ("223", "01", "100g", "200"),
    "당근": ("232", "01", "1kg", "200"),
    "오이": ("243", "01", "10개", "200"),
    "애호박": ("243", "02", "1개", "200"),
    "토마토": ("225", "01", "1kg", "200"),
    "삼겹살": ("534", "02", "100g", "500"),
    "목살": ("534", "03", "100g", "500"),
    "한우등심": ("531", "01", "100g", "500"),
    "한우갈비": ("531", "02", "100g", "500"),
    "닭고기": ("535", "01", "1kg", "500"),
    "계란": ("536", "01", "30개", "500"),
}


class KAMISClient:
    """KAMIS 실시간 Open API 및 로컬 벤치마크 고속 하이브리드 클라이언트."""

    BASE_URL = "http://www.kamis.or.kr/service/price/xml.do"

    def __init__(self, api_key: str = "", api_id: str = "") -> None:
        self._api_key = api_key
        self._api_id = api_id
        self._cache: dict[str, KAMISPrice] = {}
        self._benchmark_data: dict[str, Any] = self._load_benchmark()

    @staticmethod
    def _load_benchmark() -> dict[str, Any]:
        """로컬 고품질 벤치마크 및 일일 동기화 데이터셋 로드."""
        # 1. GitHub Actions가 오늘 아침 동기화한 최신 파일 확인
        if TODAY_PATH.exists():
            try:
                with open(TODAY_PATH, "r", encoding="utf-8") as f:
                    today_data = json.load(f).get("items", {})
                    if today_data:
                        return today_data
            except Exception:
                pass

        # 2. 벤치마크 데이터셋 확인
        if BENCHMARK_PATH.exists():
            try:
                with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
                    return json.load(f).get("commodities", {})
            except Exception as e:
                logger.warning("벤치마크 데이터 로드 실패: %s", e)
        return {}

    async def get_daily_price(
        self, item_name: str, date: str | None = None, region: str = "전국"
    ) -> KAMISPrice:
        """품목명 및 지역 기준 당일 도소매 시세 조회."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # 1. 로컬 벤치마크 우선 매칭
        for key, info in self._benchmark_data.items():
            if key in item_name or item_name in key:
                retail = info.get("regional_prices", {}).get(region, info.get("national_retail_avg", 0))
                return KAMISPrice(
                    item_name=key,
                    date=date,
                    wholesale_price=info.get("national_wholesale_avg", 0),
                    retail_price=retail,
                    unit=info.get("unit", ""),
                    change_rate=info.get("weekly_trend", [0.0])[-1],
                    grade=info.get("grade", "상품"),
                    region=region,
                )

        # 2. Open API 온라인 호출
        lookup = None
        for k, v in KAMIS_ITEM_MAP.items():
            if k in item_name or item_name in k:
                lookup = v
                break

        if lookup and self._api_key:
            item_code, kind_code, unit, _ = lookup
            try:
                params = {
                    "action": "dailyPriceByCategoryList",
                    "p_cert_key": self._api_key,
                    "p_cert_id": self._api_id,
                    "p_returntype": "json",
                    "p_product_cls_code": "02",
                    "p_item_code": item_code,
                    "p_kind_code": kind_code,
                    "p_yyyy": date[:4],
                    "p_period": "day",
                }
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.BASE_URL, params=params, timeout=5) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            item_data = data.get("data", {}).get("item", [])
                            if item_data and isinstance(item_data, list):
                                row = item_data[0]
                                p_str = str(row.get("dpr1", "0")).replace(",", "")
                                return KAMISPrice(
                                    item_name=item_name,
                                    date=date,
                                    wholesale_price=int(p_str) if p_str.isdigit() else 0,
                                    retail_price=int(p_str) if p_str.isdigit() else 0,
                                    unit=unit,
                                    change_rate=0.0,
                                )
            except Exception:
                pass

        # 3. 폴백 기본 데이터
        return KAMISPrice(
            item_name=item_name,
            date=date,
            wholesale_price=3000,
            retail_price=4000,
            unit="개/단",
            change_rate=0.0,
        )

    async def compare_to_market(
        self, item_name: str, store_price: float, region: str = "전국"
    ) -> PriceComparison:
        """매대 가격 vs 전국/지역 평균 시세 비교 브리핑 산출."""
        kamis = await self.get_daily_price(item_name, region=region)
        market_avg = kamis.retail_price

        # 벤치마크 적정가 범위
        bench = None
        for k, v in self._benchmark_data.items():
            if k in item_name or item_name in k:
                bench = v
                break

        min_fair = bench.get("min_fair_price", int(market_avg * 0.85)) if bench else int(market_avg * 0.85)
        max_fair = bench.get("max_fair_price", int(market_avg * 1.15)) if bench else int(market_avg * 1.15)

        if market_avg == 0:
            return PriceComparison(
                market_avg=0,
                store_price=store_price,
                difference_pct=0.0,
                verdict_kr="시세 데이터 수집 중",
            )

        diff_pct = ((store_price - market_avg) / market_avg) * 100.0
        abs_pct = abs(round(diff_pct, 1))

        is_bargain = store_price <= min_fair
        if diff_pct <= -10:
            verdict = f"오늘 전국 평균가({market_avg:,}원) 대비 {abs_pct}% 저렴한 특가 상품!"
        elif diff_pct < -3:
            verdict = f"시세 대비 {abs_pct}% 저렴 (구매 적기)"
        elif diff_pct > 8:
            verdict = f"시세 대비 {abs_pct}% 다소 비쌈 (다른 매대 비교 추천)"
        else:
            verdict = f"오늘 평균 시세({market_avg:,}원) 수준의 적정가"

        return PriceComparison(
            market_avg=market_avg,
            store_price=store_price,
            difference_pct=round(diff_pct, 1),
            verdict_kr=verdict,
            min_fair_price=min_fair,
            max_fair_price=max_fair,
            is_bargain=is_bargain,
        )
