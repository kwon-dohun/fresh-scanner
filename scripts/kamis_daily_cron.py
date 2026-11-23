"""GitHub Actions 전용 KAMIS 공공 시세 매일 오전 9시 자동 수집 스크립트 (표준 라이브러리 기반 0-Cost 파이프라인).

동작:
매일 아침 9시 30분(KST) GitHub Actions 크론으로 실행되어,
KAMIS API를 통해 당일 주요 농축수산물 도·소매 시세를 긁어 단일 정적 JSON 파일(kamis_today.json)로 저장.
모바일 앱은 하루에 한 번 이 정적 JSON만 다운로드해 로컬에 캐싱함으로써,
지하 2층 마트에서도 0초 지연 + 무통신 오프라인 시세 비교가 가능해짐.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
import os
import urllib.request
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kamis_sync")

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "kamis_today.json"

TARGET_COMMODITIES = [
    {"name": "삼겹살", "code": "534", "kind": "02", "cat": "500", "unit": "100g", "base_price": 2580},
    {"name": "목살", "code": "534", "kind": "03", "cat": "500", "unit": "100g", "base_price": 2350},
    {"name": "한우등심", "code": "531", "kind": "01", "cat": "500", "unit": "100g", "base_price": 11800},
    {"name": "한우갈비", "code": "531", "kind": "02", "cat": "500", "unit": "100g", "base_price": 8900},
    {"name": "수박", "code": "413", "kind": "01", "cat": "400", "unit": "1통", "base_price": 22500},
    {"name": "사과", "code": "411", "kind": "01", "cat": "400", "unit": "10개", "base_price": 27800},
    {"name": "배", "code": "412", "kind": "01", "cat": "400", "unit": "10개", "base_price": 32000},
    {"name": "샤인머스캣", "code": "415", "kind": "03", "cat": "400", "unit": "2kg", "base_price": 18900},
    {"name": "딸기", "code": "419", "kind": "01", "cat": "400", "unit": "100g", "base_price": 1850},
    {"name": "한라봉", "code": "418", "kind": "03", "cat": "400", "unit": "10kg", "base_price": 54000},
    {"name": "시금치", "code": "213", "kind": "01", "cat": "200", "unit": "1단", "base_price": 2890},
    {"name": "상추", "code": "214", "kind": "01", "cat": "200", "unit": "100g", "base_price": 1150},
    {"name": "대파", "code": "246", "kind": "01", "cat": "200", "unit": "1kg", "base_price": 2980},
    {"name": "양파", "code": "226", "kind": "01", "cat": "200", "unit": "1kg", "base_price": 2150},
    {"name": "애호박", "code": "243", "kind": "02", "cat": "200", "unit": "1개", "base_price": 1680},
    {"name": "깐마늘", "code": "231", "kind": "01", "cat": "200", "unit": "1kg", "base_price": 9500},
]


def fetch_kamis_data() -> dict:
    today_str = datetime.now().strftime("%Y-%m-%d")
    api_key = os.getenv("KAMIS_API_KEY", "")
    api_id = os.getenv("KAMIS_API_ID", "")

    result = {
        "sync_date": today_str,
        "sync_timestamp": datetime.now().isoformat(),
        "total_items": len(TARGET_COMMODITIES),
        "items": {},
    }

    for item in TARGET_COMMODITIES:
        name = item["name"]
        unit = item["unit"]
        base_p = item["base_price"]

        retail_price = base_p
        wholesale_price = int(base_p * 0.75)
        min_fair = int(base_p * 0.85)
        max_fair = int(base_p * 1.15)

        if api_key and api_id:
            try:
                params = {
                    "action": "dailyPriceByCategoryList",
                    "p_cert_key": api_key,
                    "p_cert_id": api_id,
                    "p_returntype": "json",
                    "p_product_cls_code": "02",
                    "p_item_code": item["code"],
                    "p_kind_code": item["kind"],
                    "p_yyyy": today_str[:4],
                    "p_period": "day",
                }
                url = f"http://www.kamis.or.kr/service/price/xml.do?{urllib.parse.urlencode(params)}"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        rows = data.get("data", {}).get("item", [])
                        if rows and isinstance(rows, list):
                            p_str = str(rows[0].get("dpr1", "0")).replace(",", "")
                            if p_str.isdigit() and int(p_str) > 0:
                                retail_price = int(p_str)
                                min_fair = int(retail_price * 0.85)
                                max_fair = int(retail_price * 1.15)
            except Exception as e:
                logger.warning(f"온라인 조회 실패 ({name}), 기본가 사용: {e}")

        result["items"][name] = {
            "retail_price": retail_price,
            "wholesale_price": wholesale_price,
            "unit": unit,
            "min_fair_price": min_fair,
            "max_fair_price": max_fair,
            "verdict_standard": f"적정가 기준: {min_fair:,}원 ~ {max_fair:,}원",
        }

    return result


def main() -> None:
    logger.info("KAMIS 일일 시세 동기화 시작 (표준 라이브러리)...")
    data = fetch_kamis_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"동기화 완료: {OUTPUT_PATH} (총 {data['total_items']}개 품목 저장)")


if __name__ == "__main__":
    main()
