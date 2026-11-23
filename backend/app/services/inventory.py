"""냉장고 재고 관리 서비스."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Sequence

from sqlalchemy import select, delete
from backend.app.db.database import async_session_factory, init_db
from backend.app.db.models import InventoryItem as DBInventoryItem, ShoppingListItem as DBShoppingListItem
from backend.app.schemas import InventoryItem as SchemaInventoryItem, ParsedIngredient

logger = logging.getLogger(__name__)


class InventoryService:
    """냉장고 재고 비즈니스 로직 서비스."""

    def __init__(self) -> None:
        pass

    async def list_all(self) -> list[SchemaInventoryItem]:
        """냉장고의 모든 재고 조회."""
        await init_db()
        async with async_session_factory() as session:
            stmt = select(DBInventoryItem).order_by(DBInventoryItem.added_at.desc())
            res = await session.execute(stmt)
            rows = res.scalars().all()

            return [
                SchemaInventoryItem(
                    id=row.id,
                    name=row.name,
                    quantity=row.quantity,
                    unit=row.unit,
                    category=row.category or "기타",
                    purchase_date=row.added_at,
                    expiry_date=row.expiry_estimate,
                    source=row.source or "manual",
                )
                for row in rows
            ]

    async def add_items(self, parsed_items: list[ParsedIngredient]) -> int:
        """파싱된 식재료 리스트를 냉장고 재고에 일괄 추가."""
        await init_db()
        added_count = 0
        async with async_session_factory() as session:
            for item in parsed_items:
                expiry = datetime.now() + timedelta(days=7)  # 기본 7일 유통기한 추정
                db_item = DBInventoryItem(
                    name=item.name,
                    quantity=item.quantity,
                    unit=item.unit,
                    category=item.category,
                    added_at=datetime.now(),
                    source="auto_parsed",
                    expiry_estimate=expiry,
                )
                session.add(db_item)
                added_count += 1
            await session.commit()
        return added_count

    async def confirm_purchase(self, item_ids: list[int]) -> list[SchemaInventoryItem]:
        """쇼핑 목록에서 구매 완료된 아이템들을 냉장고 재고로 이전."""
        await init_db()
        async with async_session_factory() as session:
            # 1. 쇼핑 리스트에서 항목 조회
            stmt = select(DBShoppingListItem).where(DBShoppingListItem.id.in_(item_ids))
            res = await session.execute(stmt)
            shop_items = res.scalars().all()

            # 2. 냉장고 재고에 추가
            for s_item in shop_items:
                inv = DBInventoryItem(
                    name=s_item.name,
                    quantity=s_item.quantity,
                    unit=s_item.unit,
                    category="구매확정",
                    added_at=datetime.now(),
                    source="shopping_confirmed",
                    expiry_estimate=datetime.now() + timedelta(days=5),
                )
                session.add(inv)
                # 쇼핑 목록에서는 삭제
                await session.delete(s_item)

            await session.commit()

        return await self.list_all()

    async def delete_item(self, item_id: int) -> None:
        """재고 항목 삭제."""
        await init_db()
        async with async_session_factory() as session:
            stmt = delete(DBInventoryItem).where(DBInventoryItem.id == item_id)
            await session.execute(stmt)
            await session.commit()
