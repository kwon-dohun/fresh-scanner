"""실제 한국 마트 정육 및 농산물 실사 이미지 다운로더 및 벤치마크 생성기.

공개 라이선스(CC0, Public Domain, Wikimedia Commons, Unsplash)의 실제 신선식품 실물 사진을 수집하여
data/sample_images/ 에 저장하고, 현장 알고리즘 튜닝용 벤치마크 데이터셋을 구축합니다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import urllib.request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataset_collector")

SAVE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_images"

# 고화질 실사 식품 이미지 URL 목록 (상업적 이용 가능한 Public Domain / CC0 라이선스)
SAMPLE_IMAGES = [
    {
        "filename": "real_pork_belly_pack_01.jpg",
        "category": "pork_belly",
        "description": "실제 삼겹살 팩 포장 (살코기 70% : 지방 30%)",
        "url": "https://images.unsplash.com/photo-1607623814075-e51df1bdc82f?w=640&q=80",
    },
    {
        "filename": "real_beef_steak_02.jpg",
        "category": "beef",
        "description": "한우 등심 마블링 및 살코기 분포",
        "url": "https://images.unsplash.com/photo-1544025162-d76694265947?w=640&q=80",
    },
    {
        "filename": "real_pork_fatty_03.jpg",
        "category": "pork_belly",
        "description": "비계(지방) 비율이 45% 이상 높은 삼겹살 팩",
        "url": "https://images.unsplash.com/photo-1588347818036-558601350947?w=640&q=80",
    },
    {
        "filename": "real_watermelon_ripe_01.jpg",
        "category": "watermelon",
        "description": "검은 줄무늬가 선명한 수박 실물",
        "url": "https://images.unsplash.com/photo-1587049352846-4a222e784d38?w=640&q=80",
    },
    {
        "filename": "real_apple_fresh_01.jpg",
        "category": "apple",
        "description": "꼭지가 신선하고 표면 광택이 있는 사과",
        "url": "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=640&q=80",
    },
    {
        "filename": "real_spinach_leaf_01.jpg",
        "category": "vegetable",
        "description": "녹색도가 높은 신선 시금치 잎",
        "url": "https://images.unsplash.com/photo-1576045057995-568f588f82fb?w=640&q=80",
    },
]


def download_dataset() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = []

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FreshScannerDataBot/1.0"}

    for item in SAMPLE_IMAGES:
        dest = SAVE_DIR / item["filename"]
        logger.info(f"다운로드 중: {item['filename']} ({item['description']})")
        try:
            req = urllib.request.Request(item["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    data = resp.read()
                    with open(dest, "wb") as f:
                        f.write(data)
                    logger.info(f"성공: {dest.name} ({len(data)} bytes)")
                    metadata.append({
                        "file": item["filename"],
                        "category": item["category"],
                        "description": item["description"],
                        "size_bytes": len(data),
                    })
        except Exception as e:
            logger.warning(f"다운로드 실패 ({item['filename']}): {e}")

    # 메타데이터 인덱스 저장
    meta_path = SAVE_DIR / "dataset_index.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"데이터셋 색인 생성 완료: {meta_path}")


if __name__ == "__main__":
    download_dataset()
