"""FreshScanner 메인 FastAPI 애플리케이션."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings

logger = logging.getLogger("fresh_scanner")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """앱 시작·종료 시 실행되는 라이프사이클 이벤트.

    시작 시 ML 모델 로딩, 디렉터리 생성 등을 수행합니다.
    """
    # ── 시작 ──
    logger.info("FreshScanner 서버 시작 — 모델 로딩 중...")

    # 업로드·모델 디렉터리 보장
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.MODEL_DIR).mkdir(parents=True, exist_ok=True)

    # ML 모델 초기화 (서비스 레이어가 구현되면 여기서 로딩)
    # from backend.app.services.detection import DetectionService
    # app.state.detection_service = DetectionService()
    # await app.state.detection_service.load_model()

    logger.info("FreshScanner 서버 준비 완료.")
    yield

    # ── 종료 ──
    logger.info("FreshScanner 서버 종료 중...")


app = FastAPI(
    title="FreshScanner API",
    description="신선식품 AI 분석 및 냉장고 재고·장보기 관리 API",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS 미들웨어 ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 프론트엔드 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 라우터 등록 ──
from backend.app.api.scan import router as scan_router  # noqa: E402
from backend.app.api.analyze import router as analyze_router  # noqa: E402
from backend.app.api.inventory import router as inventory_router  # noqa: E402
from backend.app.api.shopping_list import router as shopping_router  # noqa: E402
from backend.app.api.compare import router as compare_router  # noqa: E402

app.include_router(scan_router)
app.include_router(analyze_router)
app.include_router(inventory_router)
app.include_router(shopping_router)
app.include_router(compare_router)

# ── 정적 파일 (업로드 이미지) ──
uploads_path = Path(settings.UPLOAD_DIR)
uploads_path.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_path)), name="uploads")


# ── 헬스체크 ──
@app.get("/health")
async def health_check() -> dict:
    """서버 상태 확인 엔드포인트."""
    return {"status": "healthy", "version": app.version}
