"""식재료 텍스트 파싱 및 표준화 LLM/NLP 서비스."""

from __future__ import annotations

import logging
import re
from typing import Sequence

from backend.app.schemas import ParsedIngredient

logger = logging.getLogger(__name__)

# 한국어 식재료 정규화 동의어 사전
SYNONYM_MAP: dict[str, tuple[str, str, str]] = {
    # 키워드: (표준식재료명, 기본단위, 카테고리)
    "삼겹": ("삼겹살", "g", "정육"),
    "목심": ("목살", "g", "정육"),
    "목살": ("목살", "g", "정육"),
    "등심": ("소고기 등심", "g", "정육"),
    "한우": ("한우", "g", "정육"),
    "대파": ("대파", "단", "채소"),
    "쪽파": ("쪽파", "단", "채소"),
    "양파": ("양파", "개", "채소"),
    "마늘": ("깐마늘", "g", "채소"),
    "두부": ("두부", "모", "가공/두부"),
    "찌개용두부": ("두부", "모", "가공/두부"),
    "부침용두부": ("두부", "모", "가공/두부"),
    "애호박": ("애호박", "개", "채소"),
    "호박": ("애호박", "개", "채소"),
    "감자": ("감자", "개", "채소"),
    "고구마": ("고구마", "개", "채소"),
    "당근": ("당근", "개", "채소"),
    "시금치": ("시금치", "단", "채소"),
    "상추": ("상추", "봉", "채소"),
    "깻잎": ("깻잎", "봉", "채소"),
    "계란": ("계란", "구", "축산"),
    "달걀": ("계란", "구", "축산"),
    "사과": ("사과", "개", "과일"),
    "배": ("배", "개", "과일"),
    "수박": ("수박", "통", "과일"),
    "포도": ("포도", "송이", "과일"),
    "샤인머스캣": ("샤인머스캣", "송이", "과일"),
    "딸기": ("딸기", "팩", "과일"),
    "콩나물": ("콩나물", "봉", "채소"),
    "숙주": ("숙주", "봉", "채소"),
    "무": ("무", "개", "채소"),
    "배추": ("배추", "포기", "채소"),
    "김치": ("김치", "kg", "가공"),
    "오이": ("오이", "개", "채소"),
    "버섯": ("버섯", "팩", "채소"),
    "팽이버섯": ("팽이버섯", "봉", "채소"),
    "새송이": ("새송이버섯", "팩", "채소"),
    "표고": ("표고버섯", "팩", "채소"),
    "청양고추": ("청양고추", "봉", "채소"),
    "풋고추": ("풋고추", "봉", "채소"),
    "고추": ("고추", "봉", "채소"),
    "된장": ("된장", "개", "양념/소스"),
    "고추장": ("고추장", "개", "양념/소스"),
    "간장": ("간장", "병", "양념/소스"),
    "참기름": ("참기름", "병", "양념/소스"),
    "우유": ("우유", "개", "유제품"),
}


class LLMService:
    """영수증 OCR 및 온라인 주문 텍스트 정규화 파서."""

    async def normalize_ingredients(self, raw_text: str) -> list[ParsedIngredient]:
        """영수증 OCR 원문 텍스트로부터 표준 식재료 리스트 추출."""
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        results: list[ParsedIngredient] = []
        seen = set()

        for line in lines:
            item = self._match_synonym(line)
            if item and item.name not in seen:
                results.append(item)
                seen.add(item.name)

        # 텍스트에서 매칭되지 않았을 때의 샘플 폴백
        if not results and raw_text:
            results.append(
                ParsedIngredient(name="신선채소", quantity=1.0, unit="팩", category="채소")
            )

        return results

    async def parse_online_order(self, raw_text: str, source: str = "coupang") -> list[ParsedIngredient]:
        """쿠팡/마켓컬리 주문내역 스크린샷 텍스트나 알림톡 복붙 내용 파싱."""
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        results: list[ParsedIngredient] = []
        seen = set()

        for line in lines:
            # 수량 패턴 추출 (예: '2개', '1박스', '500g')
            qty = 1.0
            qty_match = re.search(r"(\d+)\s*(개|팩|봉|박스|통|단|kg|g)", line)
            if qty_match:
                qty = float(qty_match.group(1))

            item = self._match_synonym(line)
            if item and item.name not in seen:
                item.quantity = qty
                results.append(item)
                seen.add(item.name)

        if not results:
            # 기본 파싱
            results.append(
                ParsedIngredient(name="주문 식재료", quantity=1.0, unit="개", category="기타")
            )

        return results

    @staticmethod
    def _match_synonym(text: str) -> ParsedIngredient | None:
        """한 줄 텍스트에서 사전 키워드 매칭."""
        for keyword, (std_name, unit, cat) in SYNONYM_MAP.items():
            if keyword in text:
                # 숫자 추출
                m = re.search(r"(\d+)\s*(개|팩|봉|통|단)", text)
                qty = float(m.group(1)) if m else 1.0
                matched_unit = m.group(2) if m else unit

                return ParsedIngredient(
                    name=std_name,
                    quantity=qty,
                    unit=matched_unit,
                    category=cat,
                )
        return None
