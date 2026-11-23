"""FreshScanner 백엔드 설정 모듈."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """앱 전역 설정. 환경변수 또는 .env 파일에서 로드."""

    # 한국농수산식품유통공사(KAMIS) API 키
    KAMIS_API_KEY: str = ""

    # SQLite DB 경로
    DATABASE_URL: str = "sqlite:///./fresh_scanner.db"

    # YOLO / 세그멘테이션 모델 디렉터리
    MODEL_DIR: str = str(Path(__file__).resolve().parent.parent / "models")

    # 업로드 이미지 저장 디렉터리
    UPLOAD_DIR: str = str(Path(__file__).resolve().parent.parent / "uploads")

    # OpenAI API 키 (LLM 기반 OCR 정규화 등에 사용)
    OPENAI_API_KEY: str = ""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


settings = Settings()
