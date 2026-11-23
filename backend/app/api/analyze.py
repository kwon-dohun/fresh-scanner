"""FreshScanner 카테고리별 정밀 분석 API — 육류·과일·채소."""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from backend.app.schemas import (
    FruitAnalysisResult,
    MeatAnalysisResult,
    VegetableAnalysisResult,
)

# 서비스 레이어 (다른 워커가 구현 중)
from backend.app.services.meat_analyzer import MeatAnalyzerService
from backend.app.services.fruit_analyzer import FruitAnalyzerService
from backend.app.services.vegetable_analyzer import VegetableAnalyzerService
from backend.app.services.onnx_produce_classifier import produce_classifier
import cv2
import numpy as np

router = APIRouter(prefix="/analyze", tags=["분석"])


@router.post("/produce/deeplearning")
async def analyze_produce_deeplearning(image: UploadFile = File(...)) -> dict:
    """Colab에서 파인튜닝된 ONNX 딥러닝 모델로 품목(감귤/사과/삼겹살/채소) 및 품질을 판별."""
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image payload"}

    result = produce_classifier.predict(img)
    return result


@router.post("/meat", response_model=MeatAnalysisResult)
async def analyze_meat(image: UploadFile = File(...)) -> MeatAnalysisResult:
    """육류 정밀 분석.

    이미지에서 살코기·지방·뼈 비율을 세그멘테이션으로 계산하고,
    가격 라벨 OCR을 수행하여 MeatAnalysisResult를 반환합니다.
    """
    image_bytes = await image.read()
    service = MeatAnalyzerService()
    result = await service.analyze(image_bytes)
    return result


@router.post("/fruit", response_model=FruitAnalysisResult)
async def analyze_fruit(image: UploadFile = File(...)) -> FruitAnalysisResult:
    """과일 품질 분석.

    표면 패턴·꼭지 신선도·결점을 분석하고,
    가격표 OCR + KAMIS 시세 비교를 수행하여 FruitAnalysisResult를 반환합니다.
    """
    image_bytes = await image.read()
    service = FruitAnalyzerService()
    result = await service.analyze(image_bytes)
    return result


@router.post("/vegetable", response_model=VegetableAnalysisResult)
async def analyze_vegetable(image: UploadFile = File(...)) -> VegetableAnalysisResult:
    """채소 신선도 분석.

    갈변 비율·황변·결로 감지 등을 분석하여
    VegetableAnalysisResult를 반환합니다.
    """
    image_bytes = await image.read()
    service = VegetableAnalyzerService()
    result = await service.analyze(image_bytes)
    return result
