"""FreshScanner 장보기 목록 API."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.schemas import (
    ShoppingGenerateRequest,
    ShoppingItem,
    ShoppingListResponse,
)

# 서비스 레이어 (다른 워커가 구현 중)
from backend.app.services.shopping import ShoppingService
from backend.app.services.inventory import InventoryService
from backend.app.services.recipe_db import RecipeDBService

router = APIRouter(prefix="/shopping", tags=["장보기"])


@router.post("/generate", response_model=ShoppingListResponse)
async def generate_shopping_list(
    request: ShoppingGenerateRequest,
) -> ShoppingListResponse:
    """장보기 목록 자동 생성.

    메뉴 이름 목록을 받아 한식 레시피 DB에서 필요한 식재료를 조회하고,
    냉장고 재고를 차감한 뒤 장보기 체크리스트를 반환합니다.
    """
    recipe_service = RecipeDBService()
    required = await recipe_service.get_ingredients_for_menus(
        menu_names=request.menu_names,
        servings=request.servings,
    )

    inventory_service = InventoryService()
    fridge_items = await inventory_service.list_all()

    shopping_service = ShoppingService()
    shopping_list = await shopping_service.generate(
        required_ingredients=required,
        fridge_items=fridge_items,
    )
    return ShoppingListResponse(
        items=shopping_list,
        total=len(shopping_list),
        checked_count=sum(1 for item in shopping_list if item.checked),
    )


@router.get("/current", response_model=ShoppingListResponse)
async def get_current_shopping_list() -> ShoppingListResponse:
    """현재 장보기 목록 조회.

    가장 최근 생성된 장보기 목록을 반환합니다.
    """
    service = ShoppingService()
    items = await service.get_current()
    return ShoppingListResponse(
        items=items,
        total=len(items),
        checked_count=sum(1 for item in items if item.checked),
    )


@router.patch("/{item_id}/check", response_model=ShoppingItem)
async def toggle_check(item_id: int) -> ShoppingItem:
    """장보기 항목 체크 토글.

    해당 항목의 체크 상태를 반전시킵니다.
    """
    service = ShoppingService()
    item = await service.toggle_check(item_id)
    return item
