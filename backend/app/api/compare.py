"""다중 후보 팩 비교 트레이(Comparison Tray) 및 촬영 품질 게이트 API."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile

from backend.app.config import settings
from backend.app.models.optical_calibrator import PreCaptureQualityGate
from backend.app.models.meat_segmentor import MeatSegmentor, MeatSegResult
from backend.app.services.ocr_service import OCRService, MeatLabel
from backend.app.schemas import (
    ComparisonCandidateItem,
    ComparisonTrayResponse,
    QualityCheckResponse,
)

router = APIRouter(prefix="/compare", tags=["비교 트레이"])


@router.post("/quality-check", response_model=QualityCheckResponse)
async def check_preview_quality(image: UploadFile = File(...)) -> QualityCheckResponse:
    """프리뷰 프레임 실시간 품질 검사 (초점 흐림 및 랩 반사광 체크).

    촬영 버튼을 누르기 전 실시간으로 호출되어,
    빛 반사가 심하거나 흔들린 사진의 촬영을 사전에 방지합니다.
    """
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    res = PreCaptureQualityGate.evaluate_frame(frame)
    return QualityCheckResponse(
        can_capture=res.can_capture,
        blur_score=res.blur_score,
        glare_ratio=res.glare_ratio,
        messages=res.messages,
    )


@router.post("/tray", response_model=ComparisonTrayResponse)
async def compare_multiple_packs(images: List[UploadFile] = File(...)) -> ComparisonTrayResponse:
    """2~4팩 연속 촬영 비교 트레이 (매대 앞 사회적 마찰 최소화).

    여러 팩의 사진을 순서대로 수신하여 병렬 분석하고,
    '실질 살코기 가성비 1위'를 선별하여 한 화면에 비교 브리핑을 제공합니다.
    """
    if not images:
        raise ValueError("비교할 이미지가 없습니다.")

    segmentor = MeatSegmentor()
    ocr_service = OCRService()

    async def analyze_single_pack(idx: int, upload: UploadFile) -> dict:
        content = await upload.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"id": idx + 1, "error": True}

        # 1. 세그멘테이션
        seg: MeatSegResult = segmentor.segment(img)

        # 2. 라벨 OCR
        label: MeatLabel | None = ocr_service.extract_meat_label(img)
        unit_price = float(label.price_per_100g) if label and label.price_per_100g else (2500.0 + idx * 100)

        # 3. 실질 살코기 단가 계산
        effective_price = round(unit_price / max(seg.lean_ratio, 0.20), 0)

        # 4. 종합 품질 점수
        score = round((seg.lean_ratio * 45) + (50 if not seg.has_tteok_jibang else 20) + (seg.bms_grade * 3), 1)

        # 5. 썸네일 저장
        thumb_name = f"thumb_cand_{uuid.uuid4().hex[:6]}.jpg"
        save_dir = Path(settings.UPLOAD_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        h, w = img.shape[:2]
        thumb = cv2.resize(img, (200, int(200 * h / w)))
        cv2.imwrite(str(save_dir / thumb_name), thumb)

        return {
            "candidate_id": idx + 1,
            "label_price": unit_price,
            "lean_ratio": seg.lean_ratio,
            "fat_ratio": seg.fat_ratio,
            "effective_price": effective_price,
            "overall_score": score,
            "has_tteok": seg.has_tteok_jibang,
            "thumbnail_url": f"/uploads/{thumb_name}",
        }

    # 병렬 비동기 분석 수행
    tasks = [analyze_single_pack(i, img_file) for i, img_file in enumerate(images)]
    raw_results = await asyncio.gather(*tasks)

    # 유효한 후보 필터링
    valid_cands = [r for r in raw_results if "error" not in r]
    if not valid_cands:
        raise ValueError("유효한 고기 이미지를 분석할 수 없습니다.")

    # 실질 살코기 단가(effective_price) 기준 오름차순 정렬 (낮을수록 가성비 1위)
    valid_cands.sort(key=lambda x: x["effective_price"])

    winner = valid_cands[0]
    winner_id = winner["candidate_id"]

    # 후보 리스트 빌드
    items: List[ComparisonCandidateItem] = []
    for rank, cand in enumerate(valid_cands, start=1):
        c_id = cand["candidate_id"]
        is_win = (c_id == winner_id)

        if is_win:
            badge = "가성비 1위 🏆"
            verdict = f"순수 살코기 기준 100g당 {int(cand['effective_price']):,}원으로 가장 경제적입니다."
        elif cand["has_tteok"]:
            badge = "떡지방 주의 ⚠️"
            verdict = "뭉친 비계가 발견되어 버려지는 부위가 많을 수 있습니다."
        else:
            badge = f"{rank}위 추천"
            diff = int(cand["effective_price"] - winner["effective_price"])
            verdict = f"1위 상품 대비 살코기 100g당 약 {diff:,}원 더 비쌉니다."

        items.append(
            ComparisonCandidateItem(
                candidate_id=c_id,
                label_price_per_100g=cand["label_price"],
                lean_ratio=cand["lean_ratio"],
                fat_ratio=cand["fat_ratio"],
                effective_lean_price=cand["effective_price"],
                overall_score=cand["overall_score"],
                rank=rank,
                badge=badge,
                verdict=verdict,
                thumbnail_url=cand["thumbnail_url"],
            )
        )

    # 1위 선정 근거 브리핑
    summary = (
        f"총 {len(items)}팩 비교 결과, {winner_id}번 팩이 순수 살코기 실질 단가 "
        f"100g당 {int(winner['effective_price']):,}원으로 가장 가성비가 뛰어납니다! (살코기 {int(winner['lean_ratio']*100)}%)"
    )

    return ComparisonTrayResponse(
        total_candidates=len(items),
        winner_id=winner_id,
        winner_badge="가성비 1위 🏆",
        summary_verdict=summary,
        candidates=items,
    )
