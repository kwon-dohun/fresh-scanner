"""SQLAlchemy ORM 모델 정의."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.database import Base


class InventoryItem(Base):
    """냉장고 재고 아이템."""

    __tablename__ = "inventory_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    quantity: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    unit: Mapped[str] = mapped_column(String(50), nullable=False, default="개")
    category: Mapped[str] = mapped_column(String(100), nullable=True, index=True)
    added_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    source: Mapped[str] = mapped_column(
        String(100), nullable=True, comment="receipt / online_order / manual"
    )
    expiry_estimate: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<InventoryItem(id={self.id}, name='{self.name}', qty={self.quantity}{self.unit})>"


class ShoppingListItem(Base):
    """장보기 목록 아이템."""

    __tablename__ = "shopping_list_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    unit: Mapped[str] = mapped_column(String(50), nullable=False, default="개")
    checked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    menu_source: Mapped[str | None] = mapped_column(
        String(200), nullable=True, comment="이 아이템이 필요한 메뉴 이름"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<ShoppingListItem(id={self.id}, name='{self.name}', checked={self.checked})>"
