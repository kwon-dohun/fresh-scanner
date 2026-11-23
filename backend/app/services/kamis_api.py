"""KAMIS (한국농수산식품유통공사) API 클라이언트.

한국 농축수산물 도·소매 시세를 조회하고 매장 가격과 비교한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class KAMISPrice:
    """KAMIS 가격 조회 결과."""
    item_name: str
    date: str
    wholesale_price: int
    retail_price: int
    unit: str
    change_rate: float  # percent change vs previous period


@dataclass
class PriceComparison:
    """매장 가격 vs 시세 비교 결과."""
    market_avg: int
    store_price: float
    difference_pct: float
    verdict_kr: str  # e.g. '시세 대비 14% 저렴'


# ---------------------------------------------------------------------------
# KAMIS item_code / kind_code lookup
# ---------------------------------------------------------------------------

# 주요 한국 농축수산물 품목 코드 사전 (20+ 항목)
# (item_code, kind_code, unit, item_category_code)
KAMIS_ITEM_MAP: dict[str, tuple[str, str, str, str]] = {
    # 과일
    "사과":     ("411", "01", "10개",  "400"),
    "배":       ("412", "01", "10개",  "400"),
    "수박":     ("413", "01", "1개",   "400"),
    "참외":     ("414", "01", "10개",  "400"),
    "포도":     ("415", "01", "2kg",   "400"),
    "감귤":     ("418", "01", "10kg",  "400"),
    "한라봉":   ("418", "03", "10kg",  "400"),
    "딸기":     ("419", "01", "2kg",   "400"),
    "바나나":   ("420", "01", "100g",  "400"),
    "키위":     ("421", "02", "10개",  "400"),
    # 채소
    "배추":     ("211", "01", "1포기", "200"),
    "양배추":   ("212", "01", "1포기", "200"),
    "시금치":   ("213", "01", "1단",   "200"),
    "상추":     ("214", "01", "100g",  "200"),
    "깻잎":     ("215", "01", "100g",  "200"),
    "양파":     ("226", "01", "1kg",   "200"),
    "대파":     ("246", "01", "1kg",   "200"),
    "마늘":     ("231", "01", "1kg",   "200"),
    "감자":     ("222", "01", "100g",  "200"),
    "고구마":   ("223", "01", "100g",  "200"),
    "당근":     ("232", "01", "1kg",   "200"),
    "오이":     ("243", "01", "10개",  "200"),
    "토마토":   ("225", "01", "1kg",   "200"),
    # 축산
    "삼겹살":   ("534", "02", "1kg",   "500"),
    "목살":     ("534", "03", "1kg",   "500"),
    "한우등심": ("531", "01", "1kg",   "500"),
    "한우갈비": ("531", "02", "1kg",   "500"),
    "닭고기":   ("535", "01", "1kg",   "500"),
    "계란":     ("536", "01", "30개",  "500"),
    "우유":     ("537", "01", "1L",    "500"),
    # 수산
    "고등어":   ("641", "01", "1마리", "600"),
    "갈치":     ("642", "01", "1마리", "600"),
    "오징어":   ("644", "01", "1마리", "600"),
}


# ---------------------------------------------------------------------------
# KAMIS API Client
# ---------------------------------------------------------------------------

class KAMISClient:
    """KAMIS (농산물유통정보) Open API 클라이언트."""

    BASE_URL = "http://www.kamis.or.kr/service/price/xml.do"

    def __init__(self, api_key: str = "", api_id: str = "") -> None:
        self._api_key = api_key
        self._api_id = api_id
        self._cache: dict[str, KAMISPrice] = {}

    # -- Internal -----------------------------------------------------------

    def _base_params(self) -> dict[str, str]:
        return {
            "action": "dailyPriceByCategoryList",
            "p_cert_key": self._api_key,
            "p_cert_id": self._api_id,
            "p_returntype": "json",
        }

    async def _fetch(self, params: dict[str, str]) -> dict[str, Any] | None:
        """KAMIS API 호출 — 실패 시 None 반환."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        logger.warning("KAMIS API HTTP %s", resp.status)
                        return None
                    data = await resp.json(content_type=None)
                    return data
        except Exception as exc:
            logger.warning("KAMIS API 호출 실패: %s", exc)
            return None

    @staticmethod
    def _parse_price_str(val: str) -> int:
        """'12,345' → 12345.  빈 문자열이나 '-' 은 0."""
        if not val or val.strip() in ("-", ""):
            return 0
        return int(val.replace(",", "").replace(" ", ""))

    def _parse_item_row(self, row: dict[str, Any]) -> KAMISPrice:
        """KAMIS JSON row → KAMISPrice."""
        change_str = str(row.get("dpr5", "0")).replace("%", "").strip()
        try:
            change_rate = float(change_str)
        except ValueError:
            change_rate = 0.0

        return KAMISPrice(
            item_name=row.get("item_name", ""),
            date=row.get("day1", ""),
            wholesale_price=self._parse_price_str(str(row.get("dpr1", "0"))),
            retail_price=self._parse_price_str(str(row.get("dpr1", "0"))),
            unit=row.get("unit", ""),
            change_rate=change_rate,
        )

    def _lookup(self, item_name: str) -> tuple[str, str, str, str] | None:
        """품목명 → (item_code, kind_code, unit, category_code) 조회."""
        # 정확히 매칭
        if item_name in KAMIS_ITEM_MAP:
            return KAMIS_ITEM_MAP[item_name]
        # 부분 매칭
        for key, val in KAMIS_ITEM_MAP.items():
            if key in item_name or item_name in key:
                return val
        return None

    # -- Public API ---------------------------------------------------------

    async def get_daily_price(
        self, item_code: str, kind_code: str, date: str | None = None
    ) -> KAMISPrice:
        """특정 품목의 당일 도·소매 시세를 조회한다.

        Args:
            item_code: KAMIS 품목 코드 (예: '534' 돼지고기).
            kind_code: 품종 코드 (예: '02' 삼겹살).
            date: 조회일자 'YYYY-MM-DD'. None이면 오늘.

        Returns:
            KAMISPrice 객체.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        cache_key = f"{item_code}_{kind_code}_{date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        params = self._base_params()
        params.update({
            "p_product_cls_code": "02",  # 소매
            "p_item_code": item_code,
            "p_kind_code": kind_code,
            "p_yyyy": date[:4],
            "p_period": "day",
            "p_convert_kg_yn": "Y",
        })

        data = await self._fetch(params)

        if data and "data" in data:
            items = data["data"].get("item", [])
            if isinstance(items, list) and items:
                result = self._parse_item_row(items[0])
                self._cache[cache_key] = result
                return result

        # 폴백: 빈 결과
        fallback = KAMISPrice(
            item_name=f"item_{item_code}",
            date=date,
            wholesale_price=0,
            retail_price=0,
            unit="",
            change_rate=0.0,
        )
        return fallback

    async def get_price_trend(
        self, item_code: str, kind_code: str, days: int = 7
    ) -> list[KAMISPrice]:
        """최근 N일간 가격 추이를 반환한다."""
        results: list[KAMISPrice] = []
        today = datetime.now()
        for offset in range(days):
            d = today - timedelta(days=offset)
            date_str = d.strftime("%Y-%m-%d")
            price = await self.get_daily_price(item_code, kind_code, date_str)
            if price.retail_price > 0:
                results.append(price)
        results.reverse()  # 오래된 날짜 → 최근 순
        return results

    async def compare_to_market(
        self, item_name: str, store_price: float
    ) -> PriceComparison:
        """매장 가격과 시세를 비교한다.

        Args:
            item_name: 한글 품목명 (예: '삼겹살').
            store_price: 매장 가격 (원).

        Returns:
            PriceComparison 객체.
        """
        lookup = self._lookup(item_name)
        market_avg = 0

        if lookup:
            item_code, kind_code, _, _ = lookup
            kamis_price = await self.get_daily_price(item_code, kind_code)
            market_avg = kamis_price.retail_price

        # 시세를 가져오지 못한 경우 → 비교 불가
        if market_avg == 0:
            return PriceComparison(
                market_avg=0,
                store_price=store_price,
                difference_pct=0.0,
                verdict_kr="시세 정보를 조회할 수 없습니다",
            )

        diff_pct = ((store_price - market_avg) / market_avg) * 100
        abs_pct = abs(round(diff_pct, 1))

        if diff_pct < -3:
            verdict = f"시세 대비 {abs_pct}% 저렴"
        elif diff_pct > 3:
            verdict = f"시세 대비 {abs_pct}% 비쌈"
        else:
            verdict = "시세와 비슷한 수준"

        return PriceComparison(
            market_avg=market_avg,
            store_price=store_price,
            difference_pct=round(diff_pct, 1),
            verdict_kr=verdict,
        )
