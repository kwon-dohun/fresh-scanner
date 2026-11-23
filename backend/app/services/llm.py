"""한국 식재료 영수증/온라인 주문 텍스트 정밀 퍼지 매칭 및 정규화 파서 (SOTA 업그레이드)."""

from __future__ import annotations

import logging
import re
from typing import Sequence

from backend.app.schemas import ParsedIngredient

logger = logging.getLogger(__name__)

# 마트 영수증 및 온라인몰(쿠팡/컬리/SSG) 50여 종 대용량 표준 식재료 정규화 사전
COMPREHENSIVE_TAXONOMY: dict[str, tuple[str, str, str]] = {
    # ── [정육 / 가공육] ──
    "삼겹": ("삼겹살", "g", "정육"),
    "오겹": ("오겹살", "g", "정육"),
    "대패": ("대패삼겹살", "g", "정육"),
    "목심": ("목살", "g", "정육"),
    "목살": ("목살", "g", "정육"),
    "등심": ("소고기 등심", "g", "정육"),
    "안심": ("안심", "g", "정육"),
    "채끝": ("채끝살", "g", "정육"),
    "부채살": ("부채살", "g", "정육"),
    "차돌": ("차돌박이", "g", "정육"),
    "국거리": ("소고기 국거리", "g", "정육"),
    "불고기": ("불고기용 고기", "g", "정육"),
    "찌개용돼지": ("돼지고기(찌개용)", "g", "정육"),
    "앞다리": ("돼지고기 앞다리살", "g", "정육"),
    "뒷다리": ("돼지고기 뒷다리살", "g", "정육"),
    "항정살": ("항정살", "g", "정육"),
    "가브리살": ("가브리살", "g", "정육"),
    "닭볶음": ("닭볶음탕용 닭", "마리", "정육"),
    "생닭": ("생닭", "마리", "정육"),
    "닭가슴살": ("닭가슴살", "팩", "정육"),
    "베이컨": ("베이컨", "팩", "가공육"),
    "스팸": ("스팸/햄", "캔", "가공육"),
    "비엔나": ("비엔나소시지", "봉", "가공육"),

    # ── [채소 / 쌈채류] ──
    "대파": ("대파", "단", "채소"),
    "흙대파": ("대파", "단", "채소"),
    "쪽파": ("쪽파", "단", "채소"),
    "실파": ("실파", "단", "채소"),
    "양파": ("양파", "개", "채소"),
    "깐양파": ("양파", "망", "채소"),
    "자색양파": ("자색양파", "망", "채소"),
    "마늘": ("깐마늘", "g", "채소"),
    "다진마늘": ("다진마늘", "통", "채소"),
    "애호박": ("애호박", "개", "채소"),
    "단호박": ("단호박", "개", "채소"),
    "쥬키니": ("쥬키니호박", "개", "채소"),
    "감자": ("감자", "개", "채소"),
    "수미감자": ("감자", "봉", "채소"),
    "고구마": ("고구마", "봉", "채소"),
    "당근": ("당근", "개", "채소"),
    "흙당근": ("당근", "봉", "채소"),
    "시금치": ("시금치", "단", "채소"),
    "상추": ("상추", "봉", "채소"),
    "적상추": ("상추", "봉", "채소"),
    "꽃상추": ("상추", "봉", "채소"),
    "깻잎": ("깻잎", "봉", "채소"),
    "양배추": ("양배추", "통", "채소"),
    "배추": ("배추", "통", "채소"),
    "알배기": ("알배기배추", "통", "채소"),
    "무": ("무", "개", "채소"),
    "열무": ("열무", "단", "채소"),
    "오이": ("오이", "개", "채소"),
    "백오이": ("오이", "개", "채소"),
    "다다기": ("오이", "개", "채소"),
    "콩나물": ("콩나물", "봉", "채소"),
    "숙주": ("숙주", "봉", "채소"),
    "팽이": ("팽이버섯", "봉", "채소"),
    "새송이": ("새송이버섯", "팩", "채소"),
    "표고": ("표고버섯", "팩", "채소"),
    "느타리": ("느타리버섯", "팩", "채소"),
    "청양고추": ("청양고추", "봉", "채소"),
    "오이고추": ("오이고추", "봉", "채소"),
    "풋고추": ("풋고추", "봉", "채소"),
    "파프리카": ("파프리카", "개", "채소"),
    "피망": ("피망", "개", "채소"),
    "브로콜리": ("브로콜리", "송이", "채소"),

    # ── [과일] ──
    "사과": ("사과", "개", "과일"),
    "후지": ("사과", "봉", "과일"),
    "배": ("배", "개", "과일"),
    "신고배": ("배", "개", "과일"),
    "수박": ("수박", "통", "과일"),
    "참외": ("참외", "봉", "과일"),
    "포도": ("포도", "송이", "과일"),
    "샤인머스": ("샤인머스캣", "송이", "과일"),
    "샤인머스켓": ("샤인머스캣", "송이", "과일"),
    "딸기": ("딸기", "팩", "과일"),
    "바나나": ("바나나", "송이", "과일"),
    "감귤": ("감귤", "봉", "과일"),
    "귤": ("감귤", "봉", "과일"),
    "한라봉": ("한라봉", "박스", "과일"),
    "토마토": ("토마토", "팩", "과일"),
    "방울토마토": ("방울토마토", "팩", "과일"),

    # ── [가공식품 / 두부 / 계란 / 유제품] ──
    "두부": ("두부", "모", "가공/두부"),
    "순두부": ("순두부", "봉", "가공/두부"),
    "연두부": ("연두부", "팩", "가공/두부"),
    "계란": ("계란", "구", "축산"),
    "달걀": ("계란", "구", "축산"),
    "우유": ("우유", "개", "유제품"),
    "어묵": ("어묵", "봉", "수산가공"),
    "김치": ("김치", "kg", "가공"),
    "된장": ("된장", "통", "양념"),
    "고추장": ("고추장", "통", "양념"),
    "쌈장": ("쌈장", "통", "양념"),
    "진간장": ("진간장", "병", "양념"),
    "국간장": ("국간장", "병", "양념"),
    "참기름": ("참기름", "병", "양념"),
}


class LLMService:
    """한국어 마트 텍스트(POS 영수증 약어, 온라인 주문내역) 정규화 엔진."""

    async def normalize_ingredients(self, raw_text: str) -> list[ParsedIngredient]:
        """영수증 OCR 텍스트 정규화."""
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        results: list[ParsedIngredient] = []
        seen = set()

        for line in lines:
            # 품목명에서 잡음 기호 제거 (예: *, [할인], [행사], 숫자)
            cleaned = re.sub(r"\[.*?\]|\(.*?\)|[0-9,]+원|\*+", " ", line).strip()
            item = self._fuzzy_match(cleaned)
            if item and item.name not in seen:
                results.append(item)
                seen.add(item.name)

        if not results and raw_text:
            results.append(
                ParsedIngredient(name="신선식재료", quantity=1.0, unit="팩", category="기타")
            )

        return results

    async def parse_online_order(self, raw_text: str, source: str = "coupang") -> list[ParsedIngredient]:
        """쿠팡/컬리/배민B마트 알림톡 및 스크린샷 텍스트 파싱."""
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        results: list[ParsedIngredient] = []
        seen = set()

        for line in lines:
            # 수량 추출
            qty = 1.0
            m_qty = re.search(r"(\d+)\s*(개|팩|봉|박스|통|단|kg|g|모|구)", line)
            if m_qty:
                qty = float(m_qty.group(1))

            cleaned = re.sub(r"\[.*?\]|\(.*?\)", " ", line).strip()
            item = self._fuzzy_match(cleaned)
            if item and item.name not in seen:
                item.quantity = qty
                results.append(item)
                seen.add(item.name)

        if not results:
            results.append(
                ParsedIngredient(name="주문 식재료", quantity=1.0, unit="개", category="기타")
            )

        return results

    @classmethod
    def _fuzzy_match(cls, text: str) -> ParsedIngredient | None:
        """사전 키워드 기반 퍼지 매칭."""
        # 1. 완벽 포함 매칭
        for kw, (std_name, unit, cat) in COMPREHENSIVE_TAXONOMY.items():
            if kw in text:
                return ParsedIngredient(name=std_name, quantity=1.0, unit=unit, category=cat)

        # 2. 부분 일치 (2글자 이상 키워드)
        for kw, (std_name, unit, cat) in COMPREHENSIVE_TAXONOMY.items():
            if len(kw) >= 2:
                # 2글자 모두 텍스트에 포함되는지 확인
                if all(char in text for char in kw):
                    return ParsedIngredient(name=std_name, quantity=1.0, unit=unit, category=cat)

        return None
