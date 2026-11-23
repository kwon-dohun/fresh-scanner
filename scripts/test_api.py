"""FreshScanner 통합 엔드포인트 동작 검증 스크립트."""

from __future__ import annotations

import io
import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app


def create_dummy_meat_image() -> bytes:
    """테스트용 합성 고기 이미지 생성 (살코기 붉은색 + 지방 흰색)."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # 살코기 영역 (진한 선홍색)
    img[:, :] = (30, 30, 200)
    # 지방 마블링 영역 (아이보리 화이트)
    cv2.circle(img, (150, 150), 40, (230, 240, 240), -1)
    cv2.rectangle(img, (20, 20), (80, 80), (220, 230, 230), -1)

    is_success, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


def run_tests() -> None:
    client = TestClient(app)
    dummy_bytes = create_dummy_meat_image()

    print("\n--- 1. 헬스체크 검증 ---")
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print("[PASS] /health:", res.json())

    print("\n--- 2. 1단계 실시간 AR 스캔 (/scan/realtime) ---")
    files = {"image": ("dummy.jpg", dummy_bytes, "image/jpeg")}
    res = client.post("/scan/realtime", files=files)
    assert res.status_code == 200, f"Realtime scan failed: {res.text}"
    print("[PASS] /scan/realtime:", res.json())

    print("\n--- 3. 2단계 정밀 분석 (/scan/precise) ---")
    files = {"image": ("dummy.jpg", dummy_bytes, "image/jpeg")}
    res = client.post("/scan/precise", files=files)
    assert res.status_code == 200, f"Precise scan failed: {res.text}"
    print("[PASS] /scan/precise:", res.json())

    print("\n--- 4. 카테고리별 정밀 분석 (/analyze/meat) ---")
    files = {"image": ("dummy.jpg", dummy_bytes, "image/jpeg")}
    res = client.post("/analyze/meat", files=files)
    assert res.status_code == 200, f"Analyze meat failed: {res.text}"
    print("[PASS] /analyze/meat:", res.json())

    print("\n--- 5. 온라인 주문 텍스트 파싱 (/inventory/online-order) ---")
    payload = {
        "raw_text": "[쿠팡 로켓프레시 주문완료]\n1. 국내산 한돈 삼겹살 500g 1개\n2. 친환경 무농약 대파 1단\n3. 찌개용 두부 1모",
        "source": "coupang",
    }
    res = client.post("/inventory/online-order", json=payload)
    assert res.status_code == 200, f"Online order parse failed: {res.text}"
    print("[PASS] /inventory/online-order:", res.json())

    print("\n--- 6. 냉장고 재고 목록 조회 (/inventory/) ---")
    res = client.get("/inventory/")
    assert res.status_code == 200, f"Inventory list failed: {res.text}"
    print("[PASS] /inventory/: total items =", res.json().get("total"))

    print("\n--- 7. 장보기 목록 자동 생성 (/shopping/generate) ---")
    req = {
        "menu_names": ["된장찌개", "제육볶음"],
        "servings": 2,
    }
    res = client.post("/shopping/generate", json=req)
    assert res.status_code == 200, f"Shopping list generate failed: {res.text}"
    print("[PASS] /shopping/generate:", res.json())

    print("\n🎉 모든 핵심 엔드포인트 통합 검증 성공!\n")


if __name__ == "__main__":
    run_tests()
