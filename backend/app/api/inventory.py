"""FreshScanner 냉장고 재고 관리 API."""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from backend.app.schemas import (
    ConfirmPurchaseRequest,
    IngredientParseResponse,
    InventoryItem,
    InventoryListResponse,
    OnlineOrderRequest,
)

# 서비스 레이어 (다른 워커가 구현 중)
from backend.app.services.inventory import InventoryService
from backend.app.services.ocr import OCRService
from backend.app.services.llm import LLMService

router = APIRouter(prefix="/inventory", tags=["재고관리"])


@router.get("/", response_model=InventoryListResponse)
async def list_inventory() -> InventoryListResponse:
    """냉장고 재고 전체 목록 조회.

    현재 냉장고에 등록된 모든 식재료를 반환합니다.
    """
    service = InventoryService()
    items = await service.list_all()
    return InventoryListResponse(items=items, total=len(items))


@router.post("/receipt-ocr", response_model=IngredientParseResponse)
async def receipt_ocr(image: UploadFile = File(...)) -> IngredientParseResponse:
    """영수증 OCR → 재고 등록.

    영수증 이미지를 OCR로 읽고, LLM을 사용해
    표준 식재료명으로 정규화한 뒤 냉장고 재고에 추가합니다.
    """
    image_bytes = await image.read()
    ocr_service = OCRService()
    raw_text = await ocr_service.extract_text(image_bytes)

    llm_service = LLMService()
    parsed = await llm_service.normalize_ingredients(raw_text)

    inventory_service = InventoryService()
    added = await inventory_service.add_items(parsed)
    return IngredientParseResponse(parsed_items=parsed, added_count=added)


@router.post("/online-order", response_model=IngredientParseResponse)
async def online_order(request: OnlineOrderRequest) -> IngredientParseResponse:
    """온라인 주문 텍스트 파싱 → 재고 등록.

    쿠팡·마켓컬리 등에서 복사한 주문 내역 텍스트를
    LLM으로 파싱하여 표준 식재료로 변환 후 냉장고 재고에 추가합니다.
    """
    llm_service = LLMService()
    parsed = await llm_service.parse_online_order(
        raw_text=request.raw_text,
        source=request.source,
    )

    inventory_service = InventoryService()
    added = await inventory_service.add_items(parsed)
    return IngredientParseResponse(parsed_items=parsed, added_count=added)


@router.post("/confirm-purchase", response_model=InventoryListResponse)
async def confirm_purchase(request: ConfirmPurchaseRequest) -> InventoryListResponse:
    """장보기 완료 확정 — 쇼핑 목록 항목을 냉장고 재고로 이동.

    장보기 목록에서 체크된 항목들을 냉장고 재고로 확정 등록합니다.
    """
    service = InventoryService()
    items = await service.confirm_purchase(request.item_ids)
    return InventoryListResponse(items=items, total=len(items))


@router.delete("/{item_id}")
async def delete_item(item_id: int) -> dict:
    """재고 항목 삭제.

    냉장고 재고에서 특정 항목을 제거합니다.
    """
    service = InventoryService()
    await service.delete_item(item_id)
    return {"detail": f"항목 {item_id} 삭제 완료"}
