"""FreshScanner Pydantic 스키마 정의."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ───────────────────────── 공통 ─────────────────────────

class BBox(BaseModel):
    """바운딩 박스 좌표 (xyxy)."""
    x1: float
    y1: float
    x2: float
    y2: float


class FreshnessGrade(str, Enum):
    """신선도 등급."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


# ───────────────────────── Scan ─────────────────────────

class DetectedItem(BaseModel):
    """YOLO 탐지 결과 단일 항목."""
    class_name: str = Field(..., description="탐지된 객체 클래스명")
    confidence: float = Field(..., ge=0, le=1, description="탐지 신뢰도")
    bbox: BBox
    quality_score: float = Field(0.0, ge=0, le=100, description="간이 품질 점수")


class RealtimeScanResponse(BaseModel):
    """1단계 AR 실시간 스캔 응답."""
    items: list[DetectedItem] = []
    inference_time_ms: float = 0.0


class XAIExplanation(BaseModel):
    """XAI 기반 품질 분석 설명."""
    factor: str = Field(..., description="분석 요인 (예: 색상, 질감)")
    score: float = Field(..., description="해당 요인 점수")
    description: str = Field("", description="상세 설명")


class PreciseScanResponse(BaseModel):
    """2단계 정밀분석 응답."""
    class_name: str
    freshness_grade: FreshnessGrade
    overall_score: float = Field(..., ge=0, le=100)
    explanations: list[XAIExplanation] = []
    ocr_text: str = ""
    segmentation_mask_url: Optional[str] = None


# ───────────────────────── Analyze ─────────────────────────

class MeatAnalysisResult(BaseModel):
    """육류 분석 결과."""
    lean_ratio: float = Field(..., ge=0, le=1, description="살코기 비율")
    fat_ratio: float = Field(..., ge=0, le=1, description="지방 비율")
    bone_ratio: float = Field(..., ge=0, le=1, description="뼈 비율")
    marbling_score: float = Field(0.0, ge=0, le=10, description="마블링 점수")
    color_score: float = Field(0.0, ge=0, le=100, description="색상 점수")
    price_per_100g: Optional[float] = None
    ocr_label_text: str = ""
    freshness_grade: FreshnessGrade = FreshnessGrade.GOOD


class FruitAnalysisResult(BaseModel):
    """과일 분석 결과."""
    surface_uniformity: float = Field(0.0, ge=0, le=100, description="표면 균일도")
    stem_freshness: float = Field(0.0, ge=0, le=100, description="꼭지 신선도")
    defect_ratio: float = Field(0.0, ge=0, le=1, description="결점 비율")
    freshness_grade: FreshnessGrade = FreshnessGrade.GOOD
    ocr_price: Optional[float] = None
    kamis_market_price: Optional[float] = None
    price_evaluation: str = ""


class VegetableAnalysisResult(BaseModel):
    """채소 분석 결과."""
    browning_ratio: float = Field(0.0, ge=0, le=1, description="갈변 비율")
    yellowing_ratio: float = Field(0.0, ge=0, le=1, description="황변 비율")
    condensation_detected: bool = Field(False, description="결로 감지 여부")
    freshness_grade: FreshnessGrade = FreshnessGrade.GOOD
    overall_score: float = Field(0.0, ge=0, le=100)


# ───────────────────────── Inventory ─────────────────────────

class InventoryItem(BaseModel):
    """냉장고 재고 항목."""
    id: int
    name: str = Field(..., description="표준 식재료명")
    quantity: float = Field(1.0, description="수량")
    unit: str = Field("개", description="단위")
    category: str = Field("기타", description="카테고리")
    purchase_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    source: str = Field("manual", description="입력 경로 (manual/receipt/online)")


class InventoryListResponse(BaseModel):
    """재고 목록 응답."""
    items: list[InventoryItem] = []
    total: int = 0


class ReceiptOCRRequest(BaseModel):
    """영수증 OCR 요청 (이미지는 UploadFile로 별도 수신)."""
    store_name: Optional[str] = None


class OnlineOrderRequest(BaseModel):
    """온라인 주문 텍스트 파싱 요청."""
    raw_text: str = Field(..., description="쿠팡/마켓컬리 등에서 복사한 주문 내역 텍스트")
    source: str = Field("coupang", description="출처 (coupang/kurly/etc)")


class ConfirmPurchaseRequest(BaseModel):
    """장보기 목록 → 냉장고 확정 요청."""
    item_ids: list[int] = Field(..., description="확정할 쇼핑 목록 항목 ID 리스트")


class ParsedIngredient(BaseModel):
    """파싱/정규화된 식재료."""
    name: str
    quantity: float = 1.0
    unit: str = "개"
    category: str = "기타"


class IngredientParseResponse(BaseModel):
    """OCR/텍스트 파싱 결과."""
    parsed_items: list[ParsedIngredient] = []
    added_count: int = 0


# ───────────────────────── Shopping List ─────────────────────────

class ShoppingGenerateRequest(BaseModel):
    """장보기 목록 생성 요청."""
    menu_names: list[str] = Field(..., description="만들 메뉴 이름 리스트")
    servings: int = Field(2, description="인분 수")


class ShoppingItem(BaseModel):
    """장보기 목록 개별 항목."""
    id: int
    name: str
    quantity: float = 1.0
    unit: str = "개"
    checked: bool = False
    in_fridge: float = Field(0.0, description="냉장고에 이미 있는 수량")
    need: float = Field(0.0, description="추가로 필요한 수량")


class ShoppingListResponse(BaseModel):
    """장보기 목록 응답."""
    items: list[ShoppingItem] = []
    total: int = 0
    checked_count: int = 0
