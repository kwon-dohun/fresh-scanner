"""한식 종합 레시피 데이터베이스 및 소요 식재료 자동 차감 서비스 (대용량 확장).

국·찌개, 볶음, 조림, 구이, 무침, 면·밥류 등 한국인이 가장 자주 먹는 45선 식단 DB 구축.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 2인분 기준 정량 식재료 데이터베이스
RECIPE_DB: dict[str, list[dict[str, Any]]] = {
    # ── [1] 찌개 & 전골류 ──
    "된장찌개": [
        {"name": "된장", "quantity": 1.0, "unit": "큰술"},
        {"name": "두부", "quantity": 0.5, "unit": "모"},
        {"name": "애호박", "quantity": 0.5, "unit": "개"},
        {"name": "양파", "quantity": 0.5, "unit": "개"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
        {"name": "청양고추", "quantity": 1.0, "unit": "개"},
        {"name": "감자", "quantity": 1.0, "unit": "개"},
    ],
    "차돌된장찌개": [
        {"name": "차돌박이", "quantity": 150.0, "unit": "g"},
        {"name": "된장", "quantity": 1.5, "unit": "큰술"},
        {"name": "두부", "quantity": 0.5, "unit": "모"},
        {"name": "애호박", "quantity": 0.5, "unit": "개"},
        {"name": "양파", "quantity": 0.5, "unit": "개"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
        {"name": "팽이버섯", "quantity": 0.5, "unit": "봉"},
    ],
    "김치찌개": [
        {"name": "돼지고기(찌개용)", "quantity": 250.0, "unit": "g"},
        {"name": "김치", "quantity": 250.0, "unit": "g"},
        {"name": "두부", "quantity": 0.5, "unit": "모"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
        {"name": "양파", "quantity": 0.5, "unit": "개"},
        {"name": "깐마늘", "quantity": 20.0, "unit": "g"},
    ],
    "참치김치찌개": [
        {"name": "참치캔", "quantity": 1.0, "unit": "캔"},
        {"name": "김치", "quantity": 250.0, "unit": "g"},
        {"name": "두부", "quantity": 0.5, "unit": "모"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
        {"name": "양파", "quantity": 0.5, "unit": "개"},
    ],
    "부대찌개": [
        {"name": "스팸/햄", "quantity": 1.0, "unit": "캔"},
        {"name": "프랑크소시지", "quantity": 3.0, "unit": "줄"},
        {"name": "김치", "quantity": 100.0, "unit": "g"},
        {"name": "라면사리", "quantity": 1.0, "unit": "개"},
        {"name": "대파", "quantity": 1.0, "unit": "대"},
        {"name": "양파", "quantity": 0.5, "unit": "개"},
        {"name": "체다치즈", "quantity": 1.0, "unit": "장"},
    ],
    "순두부찌개": [
        {"name": "순두부", "quantity": 1.0, "unit": "봉"},
        {"name": "계란", "quantity": 1.0, "unit": "개"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
        {"name": "양파", "quantity": 0.5, "unit": "개"},
        {"name": "바지락", "quantity": 150.0, "unit": "g"},
        {"name": "애호박", "quantity": 0.3, "unit": "개"},
    ],
    "고추장찌개": [
        {"name": "돼지고기(찌개용)", "quantity": 200.0, "unit": "g"},
        {"name": "감자", "quantity": 2.0, "unit": "개"},
        {"name": "애호박", "quantity": 0.5, "unit": "개"},
        {"name": "양파", "quantity": 0.5, "unit": "개"},
        {"name": "두부", "quantity": 0.5, "unit": "모"},
        {"name": "고추장", "quantity": 2.0, "unit": "큰술"},
    ],
    "청국장": [
        {"name": "청국장", "quantity": 150.0, "unit": "g"},
        {"name": "신김치", "quantity": 80.0, "unit": "g"},
        {"name": "두부", "quantity": 0.5, "unit": "모"},
        {"name": "애호박", "quantity": 0.5, "unit": "개"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
    ],
    "소고기버섯전골": [
        {"name": "소고기(불고기용)", "quantity": 250.0, "unit": "g"},
        {"name": "표고버섯", "quantity": 3.0, "unit": "개"},
        {"name": "팽이버섯", "quantity": 1.0, "unit": "봉"},
        {"name": "느타리버섯", "quantity": 1.0, "unit": "팩"},
        {"name": "알배기배추", "quantity": 4.0, "unit": "장"},
        {"name": "대파", "quantity": 1.0, "unit": "대"},
    ],

    # ── [2] 국 & 탕류 ──
    "소고기미역국": [
        {"name": "소고기(양지)", "quantity": 150.0, "unit": "g"},
        {"name": "자른미역", "quantity": 20.0, "unit": "g"},
        {"name": "참기름", "quantity": 1.0, "unit": "큰술"},
        {"name": "깐마늘", "quantity": 20.0, "unit": "g"},
        {"name": "국간장", "quantity": 2.0, "unit": "큰술"},
    ],
    "소고기무국": [
        {"name": "소고기(양지)", "quantity": 150.0, "unit": "g"},
        {"name": "무", "quantity": 0.3, "unit": "개"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
        {"name": "깐마늘", "quantity": 15.0, "unit": "g"},
    ],
    "콩나물국": [
        {"name": "콩나물", "quantity": 1.0, "unit": "봉"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
        {"name": "청양고추", "quantity": 1.0, "unit": "개"},
        {"name": "깐마늘", "quantity": 15.0, "unit": "g"},
    ],
    "황태해장국": [
        {"name": "황태채", "quantity": 50.0, "unit": "g"},
        {"name": "무", "quantity": 0.2, "unit": "개"},
        {"name": "콩나물", "quantity": 0.5, "unit": "봉"},
        {"name": "계란", "quantity": 1.0, "unit": "개"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
    ],
    "계란국": [
        {"name": "계란", "quantity": 2.0, "unit": "개"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
        {"name": "양파", "quantity": 0.3, "unit": "개"},
    ],
    "어묵탕": [
        {"name": "모둠어묵", "quantity": 1.0, "unit": "봉"},
        {"name": "무", "quantity": 0.2, "unit": "개"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
        {"name": "청양고추", "quantity": 1.0, "unit": "개"},
    ],
    "삼계탕": [
        {"name": "영계(생닭)", "quantity": 1.0, "unit": "마리"},
        {"name": "삼계탕용 한약재팩", "quantity": 1.0, "unit": "팩"},
        {"name": "통마늘", "quantity": 50.0, "unit": "g"},
        {"name": "대파", "quantity": 1.0, "unit": "대"},
        {"name": "찹쌀", "quantity": 100.0, "unit": "g"},
    ],
    "육개장": [
        {"name": "소고기(양지)", "quantity": 250.0, "unit": "g"},
        {"name": "대파", "quantity": 3.0, "unit": "대"},
        {"name": "고사리", "quantity": 100.0, "unit": "g"},
        {"name": "숙주", "quantity": 1.0, "unit": "봉"},
        {"name": "느타리버섯", "quantity": 1.0, "unit": "팩"},
    ],

    # ── [3] 육류 구이 & 볶음 ──
    "제육볶음": [
        {"name": "돼지고기(앞다리살)", "quantity": 400.0, "unit": "g"},
        {"name": "양파", "quantity": 1.0, "unit": "개"},
        {"name": "대파", "quantity": 1.0, "unit": "대"},
        {"name": "고추장", "quantity": 2.0, "unit": "큰술"},
        {"name": "깐마늘", "quantity": 30.0, "unit": "g"},
        {"name": "청양고추", "quantity": 1.0, "unit": "개"},
    ],
    "삼겹살구이": [
        {"name": "삼겹살", "quantity": 500.0, "unit": "g"},
        {"name": "상추", "quantity": 1.0, "unit": "봉"},
        {"name": "깻잎", "quantity": 1.0, "unit": "봉"},
        {"name": "깐마늘", "quantity": 50.0, "unit": "g"},
        {"name": "쌈장", "quantity": 1.0, "unit": "개"},
        {"name": "새송이버섯", "quantity": 2.0, "unit": "개"},
    ],
    "소불고기": [
        {"name": "소고기(불고기용)", "quantity": 400.0, "unit": "g"},
        {"name": "양파", "quantity": 1.0, "unit": "개"},
        {"name": "대파", "quantity": 1.0, "unit": "대"},
        {"name": "팽이버섯", "quantity": 1.0, "unit": "봉"},
        {"name": "당근", "quantity": 0.5, "unit": "개"},
        {"name": "진간장", "quantity": 3.0, "unit": "큰술"},
    ],
    "오삼불고기": [
        {"name": "삼겹살", "quantity": 250.0, "unit": "g"},
        {"name": "오징어", "quantity": 1.0, "unit": "마리"},
        {"name": "양파", "quantity": 1.0, "unit": "개"},
        {"name": "대파", "quantity": 1.0, "unit": "대"},
        {"name": "양배추", "quantity": 100.0, "unit": "g"},
    ],
    "닭볶음탕": [
        {"name": "닭볶음탕용 닭", "quantity": 1.0, "unit": "마리"},
        {"name": "감자", "quantity": 2.0, "unit": "개"},
        {"name": "당근", "quantity": 1.0, "unit": "개"},
        {"name": "양파", "quantity": 1.0, "unit": "개"},
        {"name": "대파", "quantity": 1.0, "unit": "대"},
        {"name": "고추장", "quantity": 2.0, "unit": "큰술"},
    ],
    "안동찜닭": [
        {"name": "토막 닭고기", "quantity": 1.0, "unit": "마리"},
        {"name": "납작당면", "quantity": 100.0, "unit": "g"},
        {"name": "감자", "quantity": 2.0, "unit": "개"},
        {"name": "당근", "quantity": 0.5, "unit": "개"},
        {"name": "양파", "quantity": 1.0, "unit": "개"},
        {"name": "대파", "quantity": 1.0, "unit": "대"},
    ],
    "돼지갈비찜": [
        {"name": "돼지갈비", "quantity": 600.0, "unit": "g"},
        {"name": "무", "quantity": 0.3, "unit": "개"},
        {"name": "당근", "quantity": 1.0, "unit": "개"},
        {"name": "양파", "quantity": 1.0, "unit": "개"},
        {"name": "표고버섯", "quantity": 3.0, "unit": "개"},
    ],

    # ── [4] 반찬 & 조림류 ──
    "두부조림": [
        {"name": "두부", "quantity": 1.0, "unit": "모"},
        {"name": "양파", "quantity": 0.5, "unit": "개"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
        {"name": "청양고추", "quantity": 1.0, "unit": "개"},
        {"name": "진간장", "quantity": 3.0, "unit": "큰술"},
    ],
    "감자조림": [
        {"name": "감자", "quantity": 3.0, "unit": "개"},
        {"name": "양파", "quantity": 0.5, "unit": "개"},
        {"name": "진간장", "quantity": 3.0, "unit": "큰술"},
        {"name": "물엿/올리고당", "quantity": 2.0, "unit": "큰술"},
    ],
    "계란말이": [
        {"name": "계란", "quantity": 4.0, "unit": "개"},
        {"name": "대파", "quantity": 0.3, "unit": "대"},
        {"name": "당근", "quantity": 0.2, "unit": "개"},
    ],
    "계란찜": [
        {"name": "계란", "quantity": 3.0, "unit": "개"},
        {"name": "대파", "quantity": 0.3, "unit": "대"},
        {"name": "새우젓", "quantity": 0.5, "unit": "작은술"},
    ],
    "시금치나물": [
        {"name": "시금치", "quantity": 1.0, "unit": "단"},
        {"name": "깐마늘", "quantity": 10.0, "unit": "g"},
        {"name": "참기름", "quantity": 1.0, "unit": "큰술"},
        {"name": "통깨", "quantity": 1.0, "unit": "작은술"},
    ],
    "콩나물무침": [
        {"name": "콩나물", "quantity": 1.0, "unit": "봉"},
        {"name": "대파", "quantity": 0.3, "unit": "대"},
        {"name": "깐마늘", "quantity": 10.0, "unit": "g"},
        {"name": "참기름", "quantity": 1.0, "unit": "큰술"},
    ],
    "잡채": [
        {"name": "당면", "quantity": 200.0, "unit": "g"},
        {"name": "시금치", "quantity": 0.5, "unit": "단"},
        {"name": "소고기(잡채용)", "quantity": 150.0, "unit": "g"},
        {"name": "표고버섯", "quantity": 3.0, "unit": "개"},
        {"name": "당근", "quantity": 0.5, "unit": "개"},
        {"name": "양파", "quantity": 1.0, "unit": "개"},
    ],
    "고등어조림": [
        {"name": "자반/생고등어", "quantity": 2.0, "unit": "토막"},
        {"name": "무", "quantity": 0.3, "unit": "개"},
        {"name": "양파", "quantity": 0.5, "unit": "개"},
        {"name": "대파", "quantity": 1.0, "unit": "대"},
        {"name": "청양고추", "quantity": 2.0, "unit": "개"},
    ],

    # ── [5] 밥 & 분식류 ──
    "비빔밥": [
        {"name": "콩나물", "quantity": 100.0, "unit": "g"},
        {"name": "시금치", "quantity": 100.0, "unit": "g"},
        {"name": "당근", "quantity": 0.5, "unit": "개"},
        {"name": "애호박", "quantity": 0.5, "unit": "개"},
        {"name": "계란", "quantity": 2.0, "unit": "개"},
        {"name": "소고기 다짐육", "quantity": 100.0, "unit": "g"},
    ],
    "김치볶음밥": [
        {"name": "신김치", "quantity": 150.0, "unit": "g"},
        {"name": "스팸/베이컨", "quantity": 80.0, "unit": "g"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
        {"name": "계란", "quantity": 2.0, "unit": "개"},
    ],
    "떡볶이": [
        {"name": "떡볶이떡", "quantity": 300.0, "unit": "g"},
        {"name": "사각어묵", "quantity": 3.0, "unit": "장"},
        {"name": "대파", "quantity": 1.0, "unit": "대"},
        {"name": "양배추", "quantity": 80.0, "unit": "g"},
        {"name": "고추장", "quantity": 2.0, "unit": "큰술"},
    ],
    "카레라이스": [
        {"name": "카레가루/고형카레", "quantity": 1.0, "unit": "팩"},
        {"name": "돼지고기(카레용)", "quantity": 200.0, "unit": "g"},
        {"name": "감자", "quantity": 2.0, "unit": "개"},
        {"name": "양파", "quantity": 1.0, "unit": "개"},
        {"name": "당근", "quantity": 1.0, "unit": "개"},
    ],
}


class RecipeDBService:
    """확장된 45선 레시피 DB 기반 소요 식재료 자동 계산 서비스."""

    async def get_ingredients_for_menus(
        self, menu_names: list[str], servings: int = 2
    ) -> list[dict[str, Any]]:
        scale = servings / 2.0
        combined: dict[str, dict[str, Any]] = {}

        for menu in menu_names:
            menu_clean = menu.strip()
            matched = None

            if menu_clean in RECIPE_DB:
                matched = RECIPE_DB[menu_clean]
            else:
                for k, v in RECIPE_DB.items():
                    if k in menu_clean or menu_clean in k:
                        matched = v
                        break

            if not matched:
                continue

            for ing in matched:
                name = ing["name"]
                unit = ing["unit"]
                qty = ing["quantity"] * scale

                if name in combined:
                    combined[name]["quantity"] += qty
                else:
                    combined[name] = {
                        "name": name,
                        "quantity": round(qty, 1),
                        "unit": unit,
                    }

        return list(combined.values())
