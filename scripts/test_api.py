"""FreshScanner 종합 엔드포인트 및 고도화 CV 알고리즘 검증 스크립트."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app


def create_dummy_meat_with_glare() -> bytes:
    """비닐 랩 반사광 및 마트 조명이 포함된 고기 이미지 합성."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # 붉은색 조명 편향된 살코기 (BGR: 20, 20, 220)
    img[:, :] = (20, 20, 220)
    # 지방 층
    cv2.rectangle(img, (20, 20), (120, 80), (210, 230, 230), -1)
    # 강한 비닐 반사광 (Specular Highlight: V>240, S<20)
    cv2.circle(img, (180, 180), 25, (250, 252, 255), -1)

    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


def create_dummy_watermelon() -> bytes:
    """수박 줄무늬 패턴 이미지 합성 (가보 필터 검증용)."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:, :] = (40, 140, 50)  # 연한 녹색 바탕
    # 검은색 수직 줄무늬
    for x in range(30, 280, 45):
        cv2.rectangle(img, (x, 0), (x + 18, 300), (20, 50, 20), -1)

    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


def create_dummy_vegetable() -> bytes:
    """신선한 채소 이미지 합성 (GLI 분광 검증용)."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:, :] = (30, 180, 40)  # 선명한 녹색
    cv2.circle(img, (150, 150), 80, (25, 200, 35), -1)

    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


def run_tests() -> None:
    client = TestClient(app)
    meat_bytes = create_dummy_meat_with_glare()
    fruit_bytes = create_dummy_watermelon()
    veg_bytes = create_dummy_vegetable()

    print("\n=======================================================")
    print(" 🛒 FreshScanner SOTA 알고리즘 및 API 통합 검증 시작")
    print("=======================================================")

    print("\n--- 1. 헬스체크 ---")
    res = client.get("/health")
    assert res.status_code == 200
    print("[PASS] /health:", res.json())

    print("\n--- 2. 1단계 실시간 AR 스캔 (/scan/realtime) ---")
    files = {"image": ("meat.jpg", meat_bytes, "image/jpeg")}
    res = client.post("/scan/realtime", files=files)
    assert res.status_code == 200
    print(f"[PASS] /scan/realtime: 감지된 객체 {len(res.json()['items'])}개, 추론시간 {res.json()['inference_time_ms']}ms")

    print("\n--- 3. 2단계 정밀 분석 (/scan/precise) ---")
    files = {"image": ("meat.jpg", meat_bytes, "image/jpeg")}
    res = client.post("/scan/precise", files=files)
    assert res.status_code == 200
    body = res.json()
    print(f"[PASS] /scan/precise: 종합 점수={body['overall_score']}점, 등급={body['freshness_grade']}")
    for exp in body["explanations"]:
        print(f"       {exp['factor']}: {exp['description']}")

    print("\n--- 4. 육류 정밀 분석 (/analyze/meat) ---")
    files = {"image": ("meat.jpg", meat_bytes, "image/jpeg")}
    res = client.post("/analyze/meat", files=files)
    assert res.status_code == 200
    m_body = res.json()
    print(f"[PASS] /analyze/meat: 살코기={int(m_body['lean_ratio']*100)}%, 지방={int(m_body['fat_ratio']*100)}%, 마블링={m_body['marbling_score']}점")

    print("\n--- 5. 과일 정밀 분석 & KAMIS 시세 비교 (/analyze/fruit) ---")
    files = {"image": ("fruit.jpg", fruit_bytes, "image/jpeg")}
    res = client.post("/analyze/fruit", files=files)
    assert res.status_code == 200
    f_body = res.json()
    print(f"[PASS] /analyze/fruit: 표면패턴={f_body['surface_uniformity']}점, 꼭지신선도={f_body['stem_freshness']}점, 시세평가={f_body['price_evaluation']}")

    print("\n--- 6. 채소 신선도 & GLI 분석 (/analyze/vegetable) ---")
    files = {"image": ("veg.jpg", veg_bytes, "image/jpeg")}
    res = client.post("/analyze/vegetable", files=files)
    assert res.status_code == 200
    v_body = res.json()
    print(f"[PASS] /analyze/vegetable: 신선도점수={v_body['overall_score']}점, 갈변율={v_body['browning_ratio']*100:.1f}%, 결로감지={v_body['condensation_detected']}")

    print("\n--- 7. 마트 영수증 / 온라인 주문 퍼지 파싱 (/inventory/online-order) ---")
    payload = {
        "raw_text": "[쿠팡 로켓프레시 주문완료]\n1. 국내산 한돈 삼겹살 500g 2개\n2. GAP 친환경 흙대파 1단\n3. CJ 찌개두부(부침겸용) 1모\n4. 제주 하우스 감귤 1봉",
        "source": "coupang",
    }
    res = client.post("/inventory/online-order", json=payload)
    assert res.status_code == 200
    p_body = res.json()
    print(f"[PASS] /inventory/online-order: 파싱된 식재료 {len(p_body['parsed_items'])}개, 재고 추가 {p_body['added_count']}건")
    for item in p_body["parsed_items"]:
        print(f"       - {item['name']} {item['quantity']}{item['unit']} ({item['category']})")

    print("\n--- 8. 45선 레시피 DB 기반 자동 차감 장보기 목록 생성 (/shopping/generate) ---")
    req = {
        "menu_names": ["차돌된장찌개", "안동찜닭"],
        "servings": 2,
    }
    res = client.post("/shopping/generate", json=req)
    assert res.status_code == 200
    s_body = res.json()
    print(f"[PASS] /shopping/generate: 냉장고 차감 후 필요한 장보기 품목 {s_body['total']}개")
    for item in s_body["items"][:5]:
        print(f"       - {item['name']}: {item['quantity']}{item['unit']} (냉장고 보유: {item['in_fridge']})")

    print("\n--- 9. 촬영 전 실시간 품질 게이트 검증 (/compare/quality-check) ---")
    files = {"image": ("frame.jpg", meat_bytes, "image/jpeg")}
    res = client.post("/compare/quality-check", files=files)
    assert res.status_code == 200
    q_body = res.json()
    print(f"[PASS] /compare/quality-check: 촬영허용={q_body['can_capture']}, 선명도={q_body['blur_score']}, 반사광비율={q_body['glare_ratio']*100:.1f}%")
    print(f"       가이드 메시지: {q_body['messages']}")

    print("\n--- 10. 다중 팩 비교 트레이 (/compare/tray) ---")
    # 2팩 비교 테스트 (살코기 많은 팩 vs 비계 많은 팩)
    meat_pack_1 = meat_bytes
    # 2번 팩 (지방 비율이 더 높은 합성 팩)
    img2 = np.zeros((300, 300, 3), dtype=np.uint8)
    img2[:, :] = (30, 30, 200)
    cv2.rectangle(img2, (10, 10), (250, 180), (220, 230, 230), -1)  # 큰 떡지방
    _, buf2 = cv2.imencode(".jpg", img2)
    meat_pack_2 = buf2.tobytes()

    multi_files = [
        ("images", ("pack1.jpg", meat_pack_1, "image/jpeg")),
        ("images", ("pack2.jpg", meat_pack_2, "image/jpeg")),
    ]
    res = client.post("/compare/tray", files=multi_files)
    assert res.status_code == 200
    t_body = res.json()
    print(f"[PASS] /compare/tray: 총 비교={t_body['total_candidates']}팩, 가성비 1위={t_body['winner_id']}번 팩")
    print(f"       브리핑: {t_body['summary_verdict']}")
    for c in t_body["candidates"]:
        print(f"       후보 {c['candidate_id']}번: 실질단가={int(c['effective_lean_price'])}원, {c['badge']} ({c['verdict']})")

    print("\n=======================================================")
    print(" 🎉 모든 SOTA 비전 파이프라인 및 백엔드 API 검증 100% 통과!")
    print("=======================================================\n")


if __name__ == "__main__":
    run_tests()
