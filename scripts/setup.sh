#!/bin/bash
# FreshScanner 초기 환경 설정 스크립트

set -e

echo "=== [1/4] Python 가상환경 생성 ==="
python3 -m venv venv
source venv/bin/activate || source venv/Scripts/activate

echo "=== [2/4] 필수 패키지 설치 ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== [3/4] 기본 디렉터리 생성 ==="
mkdir -p uploads models data/sample_images

echo "=== [4/4] YOLO 기본 가중치 다운로드 (자동) ==="
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo "=== FreshScanner 환경 구성 완료! ==="
echo "서버 실행: uvicorn backend.app.main:app --reload --port 8000"
