"""한식 레시피 데이터베이스 및 소요 식재료 추출 서비스."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 한국 대표 요리 30선 및 2인분 기준 소요 식재료
RECIPE_DB: dict[str, list[dict[str, any]]] = {
    "된장찌개": [
        {"name": "된장", "quantity": 1.0, "unit": "큰술"},
        {"name": "두부", "quantity": 0.5, "unit": "모"},
        {"name": "애호박", "quantity": 0.5, "unit": "개"},
        {"name": "양파", "quantity": 0.5, "unit": "개"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
        {"name": "청양고추", "quantity": 1.0, "unit": "개"},
    ],
    "김치찌개": [
        {"name": "돼지고기(찌개용)", "quantity": 200.0, "unit": "g"},
        {"name": "김치", "quantity": 200.0, "unit": "g"},
        {"name": "두부", "quantity": 0.5, "unit": "모"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
        {"name": "양파", "quantity": 0.5, "unit": "개"},
    ],
    "제육볶음": [
        {"name": "돼지고기(앞다리살)", "quantity": 400.0, "unit": "g"},
        {"name": "양파", "quantity": 1.0, "unit": "개"},
        {"name": "대파", "quantity": 1.0, "unit": "대"},
        {"name": "고추장", "quantity": 2.0, "unit": "큰술"},
        {"name": "깐마늘", "quantity": 30.0, "unit": "g"},
    ],
    "삼겹살구이": [
        {"name": "삼겹살", "quantity": 500.0, "unit": "g"},
        {"name": "상추", "quantity": 1.0, "unit": "봉"},
        {"name": "깻잎", "quantity": 1.0, "unit": "봉"},
        {"name": "깐마늘", "quantity": 50.0, "unit": "g"},
        {"name": "쌈장", "quantity": 1.0, "unit": "개"},
    ],
    "소불고기": [
        {"name": "소고기(불고기용)", "quantity": 400.0, "unit": "g"},
        {"name": "양파", "quantity": 1.0, "unit": "개"},
        {"name": "대파", "quantity": 1.0, "unit": "대"},
        {"name": "팽이버섯", "quantity": 1.0, "unit": "봉"},
        {"name": "당근", "quantity": 0.5, "unit": "개"},
    ],
    "미역국": [
        {"name": "소고기(양지)", "quantity": 150.0, "unit": "g"},
        {"name": "자른미역", "quantity": 20.0, "unit": "g"},
        {"name": "참기름", "quantity": 1.0, "unit": "큰술"},
        {"name": "깐마늘", "quantity": 20.0, "unit": "g"},
    ],
    "순두부찌개": [
        {"name": "순두부", "quantity": 1.0, "unit": "봉"},
        {"name": "계란", "quantity": 1.0, "unit": "개"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
        {"name": "양파", "quantity": 0.5, "unit": "개"},
        {"name": "바지락", "quantity": 150.0, "unit": "g"},
    ],
    "닭볶음탕": [
        {"name": "닭볶음탕용 닭", "quantity": 1.0, "unit": "마리"},
        {"name": "감자", "quantity": 2.0, "unit": "개"},
        {"name": "당근", "quantity": 1.0, "unit": "개"},
        {"name": "양파", "quantity": 1.0, "unit": "개"},
        {"name": "대파", "quantity": 1.0, "unit": "대"},
    ],
    "떡볶이": [
        {"name": "떡볶이떡", "quantity": 300.0, "unit": "g"},
        {"name": "사각어묵", "quantity": 3.0, "unit": "장"},
        {"name": "대파", "quantity": 1.0, "unit": "대"},
        {"name": "고추장", "quantity": 2.0, "unit": "큰술"},
    ],
    "계란말이": [
        {"name": "계란", "quantity": 4.0, "unit": "개"},
        {"name": "대파", "quantity": 0.3, "unit": "대"},
        {"name": "당근", "quantity": 0.2, "unit": "개"},
    ],
    "감자조림": [
        {"name": "감자", "quantity": 3.0, "unit": "개"},
        {"name": "양파", "quantity": 0.5, "unit": "개"},
        {"name": "간장", "quantity": 3.0, "unit": "큰술"},
    ],
    "비빔밥": [
        {"name": "콩나물", "quantity": 100.0, "unit": "g"},
        {"name": "시금치", "quantity": 100.0, "unit": "g"},
        {"name": "당근", "quantity": 0.5, "unit": "개"},
        {"name": "계란", "quantity": 2.0, "unit": "개"},
        {"name": "고추장", "quantity": 2.0, "unit": "큰술"},
    ],
    "부대찌개": [
        {"name": "스팸/햄", "quantity": 1.0, "unit": "캔"},
        {"name": "소시지", "quantity": 1.0, "unit": "봉"},
        {"name": "김치", "quantity": 100.0, "unit": "g"},
        {"name": "라면사리", "quantity": 1.0, "unit": "개"},
        {"name": "대파", "quantity": 1.0, "unit": "대"},
    ],
    "잡채": [
        {"name": "당면", "quantity": 200.0, "unit": "g"},
        {"name": "시금치", "quantity": 1.0, "unit": "단"},
        {"name": "소고기(잡채용)", "quantity": 150.0, "unit": "g"},
        {"name": "표고버섯", "quantity": 3.0, "unit": "개"},
        {"name": "당근", "quantity": 0.5, "unit": "개"},
        {"name": "양파", "quantity": 1.0, "unit": "개"},
    ],
    "콩나물국": [
        {"name": "콩나물", "quantity": 1.0, "unit": "봉"},
        {"name": "대파", "quantity": 0.5, "unit": "대"},
        {"name": "깐마늘", "quantity": 20.0, "unit": "g"},
    ],
}


class RecipeDBService:
    """메뉴별 소요 식재료 추출 서비스."""

    async def get_ingredients_for_menus(
        self, menu_names: list[str], servings: int = 2
    ) -> list[dict[str, any]]:
        """선택된 메뉴들에 대해 필요한 전체 식재료 집계."""
        scale = servings / 2.0
        combined: dict[str, dict[str, any]] = {}

        for menu in menu_names:
            menu_key = menu.strip()
            # 정확 매칭 또는 부분 매칭
            matched_items = None
            if menu_key in RECIPE_DB:
                matched_items = RECIPE_DB[menu_key]
            else:
                for k, v in RECIPE_DB.items():
                    if k in menu_key or menu_key in k:
                        matched_items = v
                        break

            if not matched_items:
                continue

            for ing in matched_items:
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
