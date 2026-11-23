"""장보기 목록 관리 및 냉장고 차감 서비스."""

from __future__ import annotations

import logging
from sqlalchemy import select, delete
from backend.app.db.database import async_session_factory, init_db
from backend.app.db.models import ShoppingListItem as DBShoppingListItem
from backend.app.schemas import (
    ShoppingItem as SchemaShoppingItem,
    InventoryItem as SchemaInventoryItem,
)

logger = logging.getLogger(__name__)


class ShoppingService:
    """장보기 목록 자동 생성 및 상태 관리 서비스."""

    async def generate(
        self,
        required_ingredients: list[dict],
        fridge_items: list[SchemaInventoryItem],
    ) -> list[SchemaShoppingItem]:
        """필요 식재료에서 냉장고 보유량을 차감하여 장보기 목록 생성 및 저장."""
        await init_db()

        # 냉장고 재고 매핑 (이름 -> 총 보유수량)
        fridge_map: dict[str, float] = {}
        for f in fridge_items:
            fridge_map[f.name] = fridge_map.get(f.name, 0.0) + f.quantity

        new_items: list[SchemaShoppingItem] = []

        async with async_session_factory() as session:
            # 기존 장보기 목록 초기화
            await session.execute(delete(DBShoppingListItem))

            for req in required_ingredients:
                name = req["name"]
                needed_qty = float(req["quantity"])
                unit = req.get("unit", "개")

                in_fridge = fridge_map.get(name, 0.0)
                # 부분 일치 확인 (예: '대파' in '흙대파')
                if in_fridge == 0.0:
                    for f_name, f_qty in fridge_map.items():
                        if name in f_name or f_name in name:
                            in_fridge += f_qty

                actual_need = max(0.0, needed_qty - in_fridge)

                # 장보기가 필요한 경우만 DB에 저장
                if actual_need > 0:
                    db_item = DBShoppingListItem(
                        name=name,
                        quantity=actual_need,
                        unit=unit,
                        checked=False,
                    )
                    session.add(db_item)
                    await session.flush()

                    new_items.append(
                        SchemaShoppingItem(
                            id=db_item.id,
                            name=name,
                            quantity=actual_need,
                            unit=unit,
                            checked=False,
                            in_fridge=in_fridge,
                            need=actual_need,
                        )
                    )

            await session.commit()

        return new_items

    async def get_current(self) -> list[SchemaShoppingItem]:
        """현재 저장된 장보기 목록 반환."""
        await init_db()
        async with async_session_factory() as session:
            stmt = select(DBShoppingListItem).order_by(DBShoppingListItem.id.asc())
            res = await session.execute(stmt)
            rows = res.scalars().all()

            return [
                SchemaShoppingItem(
                    id=r.id,
                    name=r.name,
                    quantity=r.quantity,
                    unit=r.unit,
                    checked=r.checked,
                    in_fridge=0.0,
                    need=r.quantity,
                )
                for r in rows
            ]

    async def toggle_check(self, item_id: int) -> SchemaShoppingItem:
        """장보기 항목 체크/체크해제 토글."""
        await init_db()
        async with async_session_factory() as session:
            stmt = select(DBShoppingListItem).where(DBShoppingListItem.id == item_id)
            res = await session.execute(stmt)
            item = res.scalar_one_or_none()

            if not item:
                raise ValueError(f"쇼핑 항목 {item_id}을(를) 찾을 수 없습니다.")

            item.checked = not item.checked
            await session.commit()

            return SchemaShoppingItem(
                id=item.id,
                name=item.name,
                quantity=item.quantity,
                unit=item.unit,
                checked=item.checked,
                in_fridge=0.0,
                need=item.quantity,
            )
