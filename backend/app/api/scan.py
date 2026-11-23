"""FreshScanner 스캔 API — 1단계 AR 실시간 탐지 및 2단계 정밀분석."""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from backend.app.schemas import (
    DetectedItem,
    PreciseScanResponse,
    RealtimeScanResponse,
)

# 서비스 레이어 (다른 워커가 구현 중)
from backend.app.services.detection import DetectionService
from backend.app.services.quality import QualityAnalysisService

router = APIRouter(prefix="/scan", tags=["스캔"])


@router.post("/realtime", response_model=RealtimeScanResponse)
async def realtime_scan(image: UploadFile = File(...)) -> RealtimeScanResponse:
    """1단계 AR 실시간 스캔.

    경량 YOLO 모델로 이미지 내 식재료를 탐지하고,
    바운딩 박스·클래스·신뢰도·간이 품질 점수를 반환합니다.
    """
    image_bytes = await image.read()
    detection_service = DetectionService()
    items: list[DetectedItem] = await detection_service.detect_realtime(image_bytes)
    inference_time = detection_service.last_inference_time_ms
    return RealtimeScanResponse(items=items, inference_time_ms=inference_time)


@router.post("/precise", response_model=PreciseScanResponse)
async def precise_scan(image: UploadFile = File(...)) -> PreciseScanResponse:
    """2단계 정밀분석.

    고해상도 이미지를 대상으로 세그멘테이션·OCR·품질 분석을 수행하고,
    XAI 기반 상세 리포트를 반환합니다.
    """
    image_bytes = await image.read()
    quality_service = QualityAnalysisService()
    result = await quality_service.analyze_precise(image_bytes)
    return result
