"""
FreshScanner — 정육 세그멘테이션 모델 학습 스크립트
==================================================

YOLO 세그멘테이션 모델을 사용하여 정육(고기) 이미지에서
지방(fat), 살코기(lean), 뼈(bone), 라벨(label) 영역을 분할합니다.

데이터셋 준비 가이드:
---------------------
1. LabelMe로 이미지 어노테이션 수행
   - 각 영역(lean, fat, bone, label)을 폴리곤으로 마킹
   - JSON 형식으로 저장

2. LabelMe → YOLO 형식 변환
   - data_pipeline.py의 convert_labelme_to_yolo() 함수 사용
   - 또는: labelme2yolo --json_dir ./annotations --output_dir ./dataset

3. 디렉토리 구조 (YOLO segmentation format):
   dataset/
   ├── images/
   │   ├── train/     # 학습 이미지
   │   └── val/       # 검증 이미지
   └── labels/
       ├── train/     # 학습 라벨 (.txt, 폴리곤 좌표)
       └── val/       # 검증 라벨

4. 라벨 파일 형식 (각 줄):
   <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
   - 좌표는 이미지 크기로 정규화 (0~1)
   - class_id: 0=lean, 1=fat, 2=bone, 3=label

사용법:
-------
    python meat_segmentation_train.py \\
        --data ../configs/meat_seg.yaml \\
        --epochs 100 \\
        --batch-size 16 \\
        --img-size 640 \\
        --model-base yolov8n-seg

AI Hub 데이터셋 활용:
    - "축산물 품질 판별용 이미지" 데이터셋 다운로드
    - data_pipeline.py의 download_aihub_dataset() 참고
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics 패키지가 필요합니다. pip install ultralytics")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FreshScanner 정육 세그멘테이션 모델 학습"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ml/configs/meat_seg.yaml",
        help="YOLO 데이터셋 설정 파일 경로 (default: ml/configs/meat_seg.yaml)",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default="yolov8n-seg",
        help="기반 모델 (default: yolov8n-seg). yolov8s-seg, yolov8m-seg 등 가능",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="학습 에포크 수 (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="배치 크기 (default: 16)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="입력 이미지 크기 (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="학습 디바이스 (예: 'cpu', '0', '0,1'). 비워두면 자동 감지",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="이전 학습에서 이어서 학습",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/meat_seg",
        help="결과 저장 경로 (default: runs/meat_seg)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="실험 이름 (default: train)",
    )
    return parser.parse_args()


def build_augmentation_config() -> dict:
    """
    정육 촬영 환경에 최적화된 데이터 증강 설정.

    - 매장 조명 차이를 시뮬레이션하기 위해 밝기/대비 변화 적용
    - 수평 뒤집기로 좌우 대칭 데이터 증강
    - 과도한 색상 변환은 고기 색상 판별에 악영향이므로 제한
    """
    return {
        # 밝기 변화: 매장 조명 차이 시뮬레이션 (±40%)
        "hsv_h": 0.01,   # 색조 변화 최소화 (고기 색상 보존)
        "hsv_s": 0.3,    # 채도 약간 변화
        "hsv_v": 0.4,    # 밝기 변화 (조명 차이 시뮬레이션)
        # 기하학적 변환
        "degrees": 5.0,       # 약간의 회전 (진열대 각도 차이)
        "translate": 0.1,     # 이동
        "scale": 0.3,         # 스케일 변화
        "fliplr": 0.5,        # 수평 뒤집기 50%
        "flipud": 0.0,        # 상하 뒤집기 없음 (고기 방향 중요)
        "mosaic": 0.5,        # 모자이크 증강
        "mixup": 0.0,         # 믹스업 비활성화 (세그멘테이션 경계 보존)
    }


def train(args: argparse.Namespace) -> None:
    """모델 학습 실행."""
    print("=" * 60)
    print("🥩 FreshScanner — 정육 세그멘테이션 모델 학습")
    print("=" * 60)
    print(f"  모델 베이스: {args.model_base}")
    print(f"  데이터셋:    {args.data}")
    print(f"  에포크:      {args.epochs}")
    print(f"  배치 크기:   {args.batch_size}")
    print(f"  이미지 크기: {args.img_size}")
    print(f"  디바이스:    {args.device or '자동 감지'}")
    print("=" * 60)

    # 데이터셋 설정 파일 존재 확인
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: 데이터셋 설정 파일을 찾을 수 없습니다: {data_path}")
        print("  ml/configs/meat_seg.yaml 파일을 확인해주세요.")
        sys.exit(1)

    # 모델 로드 (사전학습 가중치 사용)
    if args.resume:
        # 이전 학습 이어서
        last_checkpoint = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_checkpoint.exists():
            print(f"Error: 체크포인트를 찾을 수 없습니다: {last_checkpoint}")
            sys.exit(1)
        model = YOLO(str(last_checkpoint))
        print(f"  체크포인트에서 재개: {last_checkpoint}")
    else:
        # 새로 학습 시작 (pretrained weights)
        model = YOLO(f"{args.model_base}.pt")
        print(f"  사전학습 모델 로드: {args.model_base}.pt")

    # 데이터 증강 설정
    augmentation = build_augmentation_config()
    print(f"\n  데이터 증강 설정:")
    for k, v in augmentation.items():
        print(f"    {k}: {v}")

    # 학습 실행
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device or None,
        project=args.project,
        name=args.name,
        # 학습 하이퍼파라미터
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # 데이터 증강
        **augmentation,
        # 기타
        patience=20,       # 조기 종료: 20 에포크 동안 개선 없으면 중단
        save=True,
        save_period=10,    # 10 에포크마다 체크포인트 저장
        plots=True,        # 학습 그래프 저장
        verbose=True,
    )

    # 결과 출력
    print("\n" + "=" * 60)
    print("✅ 학습 완료!")
    print(f"  결과 저장 경로: {results.save_dir}")
    print(f"  최적 가중치:   {results.save_dir}/weights/best.pt")
    print("=" * 60)

    # 검증 실행
    print("\n📊 검증 결과:")
    metrics = model.val()
    print(f"  mAP50:    {metrics.seg.map50:.4f}")
    print(f"  mAP50-95: {metrics.seg.map:.4f}")

    # 모델 export 안내
    print("\n💡 모바일 배포를 위한 ONNX 변환:")
    print(f"  python ml/export/export_onnx.py --model {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    args = parse_args()
    train(args)
