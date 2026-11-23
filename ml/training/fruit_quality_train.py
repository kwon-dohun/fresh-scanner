"""
FreshScanner — 과일 품질 분류 모델 학습 스크립트
================================================

ResNet18 기반 전이학습으로 과일의 품질 등급(A/B/C/D)을 분류하고
결함(defect) 여부를 동시에 판별하는 멀티태스크 모델을 학습합니다.

데이터 구조:
-----------
    dataset/
    ├── train/
    │   ├── grade_A/      # 1등급 (최상)
    │   ├── grade_B/      # 2등급 (양호)
    │   ├── grade_C/      # 3등급 (보통)
    │   └── grade_D/      # 등급 외 (불량/결함)
    └── val/
        ├── grade_A/
        ├── grade_B/
        ├── grade_C/
        └── grade_D/

사용법:
-------
    python fruit_quality_train.py \\
        --data-dir ./dataset/fruit_quality \\
        --epochs 50 \\
        --batch-size 32 \\
        --lr 0.001

AI Hub 활용:
    - "농산물 품질(등급) 분류 이미지" 데이터셋 다운로드
    - 사과, 배, 감귤, 포도, 딸기 등 다양한 품목 포함
"""

import argparse
import os
import sys
import copy
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms
except ImportError:
    print("Error: torch, torchvision 패키지가 필요합니다.")
    print("  pip install torch torchvision")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FreshScanner 과일 품질 분류 모델 학습"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="데이터셋 루트 디렉토리 (train/, val/ 하위폴더 포함)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="학습 에포크 수 (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="배치 크기 (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="학습률 (default: 0.001)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="입력 이미지 크기 (default: 224)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="품질 등급 수 (default: 4, A/B/C/D)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/fruit_quality",
        help="모델 저장 경로 (default: runs/fruit_quality)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="학습 디바이스 (예: 'cpu', 'cuda:0'). 비워두면 자동 감지",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="ImageNet 사전학습 가중치 사용 (default: True)",
    )
    return parser.parse_args()


class FruitQualityModel(nn.Module):
    """
    ResNet18 기반 과일 품질 분류 + 결함 탐지 멀티태스크 모델.

    출력:
    - quality_grade: 품질 등급 (A/B/C/D) — 4-class classification
    - has_defect: 결함 여부 — binary classification
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()

        # ResNet18 backbone (사전학습 가중치)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # 원래 FC 레이어의 입력 차원
        in_features = self.backbone.fc.in_features

        # 원래 FC 제거
        self.backbone.fc = nn.Identity()

        # 공유 특성에서 분기하는 두 개의 헤드
        # Head 1: 품질 등급 분류 (A/B/C/D)
        self.grade_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        # Head 2: 결함 탐지 (있음/없음)
        self.defect_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        grade_logits = self.grade_head(features)
        defect_logit = self.defect_head(features)
        return grade_logits, defect_logit


def build_data_transforms(img_size: int) -> dict:
    """학습/검증용 데이터 변환 파이프라인."""
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(
                brightness=0.3,   # 조명 변화
                contrast=0.2,
                saturation=0.2,
                hue=0.05,         # 색조 변화 최소 (과일 색상 보존)
            ),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]),
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_grade: nn.Module,
    criterion_defect: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """1 에포크 학습. 반환: (loss, accuracy)"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 결함 라벨: grade_D (index 3) → 1, 나머지 → 0
        defect_labels = (labels == 3).float().unsqueeze(1)

        optimizer.zero_grad()

        grade_logits, defect_logit = model(inputs)

        # 멀티태스크 손실
        loss_grade = criterion_grade(grade_logits, labels)
        loss_defect = criterion_defect(defect_logit, defect_labels)
        loss = loss_grade + 0.3 * loss_defect  # 등급 분류에 가중치

        loss.backward()
        optimizer.step()

        _, preds = torch.max(grade_logits, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_grade: nn.Module,
    criterion_defect: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """검증. 반환: (loss, accuracy)"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        defect_labels = (labels == 3).float().unsqueeze(1)

        grade_logits, defect_logit = model(inputs)

        loss_grade = criterion_grade(grade_logits, labels)
        loss_defect = criterion_defect(defect_logit, defect_labels)
        loss = loss_grade + 0.3 * loss_defect

        _, preds = torch.max(grade_logits, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


def train(args: argparse.Namespace) -> None:
    """메인 학습 루프."""
    print("=" * 60)
    print("🍎 FreshScanner — 과일 품질 분류 모델 학습")
    print("=" * 60)

    # 디바이스 설정
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  디바이스: {device}")

    # 데이터 로딩
    data_dir = Path(args.data_dir)
    if not (data_dir / "train").exists():
        print(f"Error: {data_dir / 'train'} 디렉토리가 존재하지 않습니다.")
        sys.exit(1)

    data_transforms = build_data_transforms(args.img_size)

    image_datasets = {
        phase: datasets.ImageFolder(str(data_dir / phase), data_transforms[phase])
        for phase in ["train", "val"]
    }

    dataloaders = {
        phase: DataLoader(
            image_datasets[phase],
            batch_size=args.batch_size,
            shuffle=(phase == "train"),
            num_workers=4,
            pin_memory=True,
        )
        for phase in ["train", "val"]
    }

    class_names = image_datasets["train"].classes
    num_classes = len(class_names)
    print(f"  클래스: {class_names}")
    print(f"  학습 데이터: {len(image_datasets['train'])}장")
    print(f"  검증 데이터: {len(image_datasets['val'])}장")
    print(f"  에포크: {args.epochs}, LR: {args.lr}")
    print("=" * 60)

    # 모델 생성
    model = FruitQualityModel(
        num_classes=num_classes,
        pretrained=args.pretrained,
    ).to(device)

    # 손실 함수
    criterion_grade = nn.CrossEntropyLoss()
    criterion_defect = nn.BCEWithLogitsLoss()

    # 옵티마이저: backbone은 작은 LR, 헤드는 큰 LR
    optimizer = optim.Adam([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.grade_head.parameters(), "lr": args.lr},
        {"params": model.defect_head.parameters(), "lr": args.lr},
    ])

    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # 출력 디렉토리
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Early stopping 변수
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    # 학습 루프
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        # 학습
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"],
            criterion_grade, criterion_defect,
            optimizer, device,
        )

        # 검증
        val_loss, val_acc = validate(
            model, dataloaders["val"],
            criterion_grade, criterion_defect,
            device,
        )

        scheduler.step()

        print(
            f"  Train — Loss: {train_loss:.4f}, Acc: {train_acc:.4f}  |  "
            f"Val — Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        # 최적 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0

            checkpoint_path = output_dir / "best_model.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": best_model_wts,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_acc,
                "class_names": class_names,
                "num_classes": num_classes,
            }, checkpoint_path)
            print(f"  💾 최적 모델 저장: {checkpoint_path} (Acc: {best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹️  Early stopping: {args.patience} 에포크 동안 개선 없음")
                break

        # 주기적 체크포인트
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, output_dir / f"checkpoint_epoch_{epoch + 1}.pth")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ 학습 완료! ({elapsed / 60:.1f}분)")
    print(f"  최고 검증 정확도: {best_acc:.4f}")
    print(f"  최적 모델: {output_dir / 'best_model.pth'}")
    print("=" * 60)

    # 최적 가중치로 복원
    model.load_state_dict(best_model_wts)

    # 최종 모델 저장 (추론용)
    final_path = output_dir / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"  최종 모델 저장: {final_path}")

    print("\n💡 ONNX 변환:")
    print(f"  python ml/export/export_onnx.py --model {final_path} --type torchvision")


if __name__ == "__main__":
    args = parse_args()
    train(args)
