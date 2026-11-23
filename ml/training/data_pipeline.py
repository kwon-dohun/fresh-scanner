"""
FreshScanner — 데이터 파이프라인 유틸리티
=========================================

ML 모델 학습을 위한 데이터 준비, 변환, 증강 함수 모음.

포함 기능:
- AI Hub 데이터셋 다운로드 가이드
- LabelMe → YOLO 형식 변환
- 학습/검증/테스트 데이터 분할
- 조명 변화 시뮬레이션 증강
"""

import json
import os
import random
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None


# ============================================
# 1. AI Hub 데이터셋 다운로드
# ============================================

def download_aihub_dataset(output_dir: str = "./data/aihub") -> None:
    """
    AI Hub 한국 농축산물 데이터셋 다운로드 안내.

    직접 다운로드는 AI Hub 인증이 필요하므로,
    이 함수는 가이드와 디렉토리 구조만 생성합니다.

    추천 데이터셋:
    ─────────────────────────────────────────────
    1. 축산물 품질 판별용 AI 학습 데이터
       - URL: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71399
       - 소/돼지/닭 부위별 이미지 + 등급 라벨
       - 활용: 정육 세그멘테이션 모델 학습

    2. 농산물 품질(등급) 분류 이미지
       - URL: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=142
       - 사과, 배, 감귤, 딸기 등 품목별 등급 이미지
       - 활용: 과일 품질 분류 모델 학습

    3. 농산물 신선도 판별 이미지
       - URL: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100
       - 채소류 신선도 등급별 이미지
       - 활용: 채소 신선도 판별 모델 학습

    다운로드 절차:
    1. https://aihub.or.kr 회원가입 & 로그인
    2. 데이터셋 페이지에서 "다운로드" 신청
    3. 승인 후 다운로드 (보통 1~2일 소요)
    4. 압축 해제 후 output_dir에 배치
    """
    output_path = Path(output_dir)

    # 디렉토리 구조 생성
    dirs = [
        "meat/raw",
        "meat/annotations",
        "fruit/raw",
        "fruit/annotations",
        "vegetable/raw",
        "vegetable/annotations",
    ]

    for d in dirs:
        (output_path / d).mkdir(parents=True, exist_ok=True)

    # README 생성
    readme = output_path / "README.md"
    readme.write_text(
        "# AI Hub 데이터셋\n\n"
        "## 다운로드 방법\n"
        "1. https://aihub.or.kr 회원가입\n"
        "2. 데이터셋 검색 및 다운로드 신청\n"
        "3. 승인 후 다운로드\n"
        "4. 각 폴더에 압축 해제\n\n"
        "## 디렉토리 구조\n"
        "- meat/raw/ — 정육 이미지\n"
        "- meat/annotations/ — 정육 어노테이션 (JSON)\n"
        "- fruit/raw/ — 과일 이미지\n"
        "- fruit/annotations/ — 과일 어노테이션\n"
        "- vegetable/raw/ — 채소 이미지\n"
        "- vegetable/annotations/ — 채소 어노테이션\n",
        encoding="utf-8",
    )

    print(f"✅ 디렉토리 구조 생성: {output_path}")
    print("📋 다운로드 가이드:")
    print("   1. https://aihub.or.kr 방문")
    print('   2. "축산물 품질 판별용 AI 학습 데이터" 검색')
    print("   3. 다운로드 신청 → 승인 대기 → 다운로드")
    print(f"   4. {output_path}/meat/raw/ 에 이미지 저장")


# ============================================
# 2. LabelMe → YOLO 형식 변환
# ============================================

def convert_labelme_to_yolo(
    labelme_dir: str,
    output_dir: str,
    class_map: Optional[dict] = None,
) -> None:
    """
    LabelMe JSON 어노테이션을 YOLO 세그멘테이션 형식으로 변환.

    Parameters
    ----------
    labelme_dir : str
        LabelMe JSON 파일이 있는 디렉토리
    output_dir : str
        YOLO 형식 라벨 파일 출력 디렉토리
    class_map : dict, optional
        클래스 이름 → ID 매핑. 기본값:
        {"lean": 0, "fat": 1, "bone": 2, "label": 3}

    변환 형식:
    ----------
    LabelMe JSON:
        {"shapes": [{"label": "lean", "points": [[x1,y1], [x2,y2], ...]}]}

    YOLO segmentation txt (각 줄):
        <class_id> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ...
        (좌표는 이미지 너비/높이로 정규화, 0~1)
    """
    if class_map is None:
        class_map = {"lean": 0, "fat": 1, "bone": 2, "label": 3}

    labelme_path = Path(labelme_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = list(labelme_path.glob("*.json"))
    print(f"📂 LabelMe 파일 {len(json_files)}개 발견")

    converted = 0
    skipped = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            img_width = data.get("imageWidth", 0)
            img_height = data.get("imageHeight", 0)

            if img_width == 0 or img_height == 0:
                print(f"  ⚠️ 이미지 크기 정보 없음, 건너뜀: {json_file.name}")
                skipped += 1
                continue

            # YOLO 라벨 파일 생성
            label_file = output_path / json_file.with_suffix(".txt").name
            lines = []

            for shape in data.get("shapes", []):
                label_name = shape.get("label", "").lower()
                if label_name not in class_map:
                    print(f"  ⚠️ 알 수 없는 클래스 '{label_name}', 건너뜀")
                    continue

                class_id = class_map[label_name]
                points = shape.get("points", [])

                if len(points) < 3:
                    continue  # 최소 3점 필요 (삼각형)

                # 좌표 정규화 (0~1)
                normalized = []
                for x, y in points:
                    nx = max(0.0, min(1.0, x / img_width))
                    ny = max(0.0, min(1.0, y / img_height))
                    normalized.extend([f"{nx:.6f}", f"{ny:.6f}"])

                line = f"{class_id} " + " ".join(normalized)
                lines.append(line)

            if lines:
                label_file.write_text("\n".join(lines) + "\n")
                converted += 1
            else:
                skipped += 1

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ❌ 파싱 실패: {json_file.name} — {e}")
            skipped += 1

    print(f"✅ 변환 완료: {converted}개 성공, {skipped}개 건너뜀")
    print(f"   출력 디렉토리: {output_path}")


# ============================================
# 3. 데이터셋 분할
# ============================================

def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_by_folder: bool = True,
) -> dict:
    """
    데이터셋을 train/val/test로 분할 (층화 추출).

    Parameters
    ----------
    source_dir : str
        원본 이미지 디렉토리.
        - 단일 폴더: 모든 이미지를 무작위 분할
        - 클래스 하위 폴더: 클래스별 비율 유지 (층화 추출)
    output_dir : str
        분할 결과 출력 디렉토리
    train_ratio, val_ratio, test_ratio : float
        각 세트의 비율 (합이 1.0)
    seed : int
        랜덤 시드
    stratify_by_folder : bool
        하위 폴더(클래스)별 비율 유지 여부

    Returns
    -------
    dict
        {"train": count, "val": count, "test": count}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train + val + test 비율의 합이 1.0이어야 합니다"

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # 하위 폴더(클래스) 존재 여부 확인
    subdirs = [d for d in source_path.iterdir() if d.is_dir()]

    if subdirs and stratify_by_folder:
        # 클래스별 층화 추출
        all_files = []
        all_labels = []

        for subdir in subdirs:
            class_name = subdir.name
            files = [
                f for f in subdir.iterdir()
                if f.suffix.lower() in image_exts
            ]
            all_files.extend(files)
            all_labels.extend([class_name] * len(files))

        print(f"📂 {len(subdirs)}개 클래스, 총 {len(all_files)}개 이미지")
    else:
        # 단일 폴더
        all_files = [
            f for f in source_path.iterdir()
            if f.suffix.lower() in image_exts
        ]
        all_labels = ["all"] * len(all_files)
        print(f"📂 총 {len(all_files)}개 이미지")

    if len(all_files) == 0:
        print("❌ 이미지를 찾을 수 없습니다.")
        return {"train": 0, "val": 0, "test": 0}

    # train_test_split 사용 (층화 추출)
    if train_test_split is None:
        print("⚠️ scikit-learn이 필요합니다. pip install scikit-learn")
        return {"train": 0, "val": 0, "test": 0}

    # 1차: train+val vs test
    files_trainval, files_test, labels_trainval, _ = train_test_split(
        all_files, all_labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=all_labels if stratify_by_folder else None,
    )

    # 2차: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    files_train, files_val, _, _ = train_test_split(
        files_trainval, labels_trainval,
        test_size=val_ratio_adjusted,
        random_state=seed,
        stratify=labels_trainval if stratify_by_folder else None,
    )

    # 파일 복사
    counts = {}
    for split_name, split_files in [
        ("train", files_train),
        ("val", files_val),
        ("test", files_test),
    ]:
        for src_file in split_files:
            # 클래스 폴더 구조 유지
            if stratify_by_folder and subdirs:
                class_name = src_file.parent.name
                dst_dir = output_path / split_name / class_name
            else:
                dst_dir = output_path / split_name

            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_dir / src_file.name)

            # 대응하는 라벨 파일도 복사 (있으면)
            label_file = src_file.with_suffix(".txt")
            if label_file.exists():
                label_dst = output_path / split_name.replace("images", "labels")
                label_dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label_file, label_dst / label_file.name)

        counts[split_name] = len(split_files)

    print(f"✅ 분할 완료:")
    print(f"   Train: {counts['train']}개")
    print(f"   Val:   {counts['val']}개")
    print(f"   Test:  {counts['test']}개")
    print(f"   출력:  {output_path}")

    return counts


# ============================================
# 4. 조명 변화 시뮬레이션 증강
# ============================================

def augment_meat_images(
    input_dir: str,
    output_dir: str,
    num_augmented: int = 5,
    seed: int = 42,
) -> int:
    """
    정육 이미지에 조명 변화 증강을 적용하여 다양한 매장 환경을 시뮬레이션.

    마트/정육점마다 조명 종류(형광등, LED, 할로겐)와 밝기가 달라
    같은 고기도 색상이 다르게 보일 수 있습니다.
    이를 학습 데이터에 반영하여 모델의 일반화 성능을 향상시킵니다.

    증강 종류:
    1. 밝기 조절: 어두운 매장 ~ 밝은 매장
    2. 색온도 변화: 따뜻한 조명 (할로겐) ~ 차가운 조명 (형광등)
    3. 그림자 시뮬레이션: 진열대 그림자, 고기 표면 그림자
    4. 화이트 밸런스 왜곡: 카메라 자동 화이트밸런스 차이

    Parameters
    ----------
    input_dir : str
        원본 이미지 디렉토리
    output_dir : str
        증강 이미지 출력 디렉토리
    num_augmented : int
        원본 1장당 생성할 증강 이미지 수
    seed : int
        랜덤 시드

    Returns
    -------
    int
        생성된 총 증강 이미지 수
    """
    random.seed(seed)
    np.random.seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_exts
    ]

    print(f"📂 원본 이미지 {len(image_files)}개 발견")
    total_generated = 0

    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        for i in range(num_augmented):
            augmented = img.copy()

            # 1. 밝기 조절 (0.6 ~ 1.4)
            brightness_factor = random.uniform(0.6, 1.4)
            hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] *= brightness_factor
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # 2. 색온도 변화 (따뜻한/차가운 조명)
            if random.random() > 0.5:
                temp_shift = random.uniform(-20, 20)
                b, g, r = cv2.split(augmented)
                if temp_shift > 0:
                    # 따뜻한 조명 (할로겐): 빨강/노랑 증가
                    r = np.clip(r.astype(np.float32) + temp_shift, 0, 255).astype(np.uint8)
                    b = np.clip(b.astype(np.float32) - temp_shift * 0.5, 0, 255).astype(np.uint8)
                else:
                    # 차가운 조명 (형광등): 파랑 증가
                    b = np.clip(b.astype(np.float32) - temp_shift, 0, 255).astype(np.uint8)
                    r = np.clip(r.astype(np.float32) + temp_shift * 0.5, 0, 255).astype(np.uint8)
                augmented = cv2.merge([b, g, r])

            # 3. 약간의 가우시안 노이즈 (카메라 센서 노이즈)
            if random.random() > 0.7:
                noise = np.random.normal(0, 5, augmented.shape).astype(np.float32)
                augmented = np.clip(
                    augmented.astype(np.float32) + noise, 0, 255
                ).astype(np.uint8)

            # 4. 약간의 블러 (핸드폰 카메라 흔들림)
            if random.random() > 0.8:
                kernel_size = random.choice([3, 5])
                augmented = cv2.GaussianBlur(augmented, (kernel_size, kernel_size), 0)

            # 저장
            stem = img_file.stem
            suffix = img_file.suffix
            out_name = f"{stem}_aug_{i:02d}{suffix}"
            cv2.imwrite(str(output_path / out_name), augmented)
            total_generated += 1

    print(f"✅ 증강 완료: {total_generated}개 이미지 생성")
    print(f"   출력: {output_path}")
    return total_generated


# ============================================
# CLI 인터페이스
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FreshScanner 데이터 파이프라인")
    subparsers = parser.add_subparsers(dest="command", help="실행할 명령")

    # download 서브커맨드
    p_download = subparsers.add_parser("download", help="AI Hub 데이터셋 안내")
    p_download.add_argument("--output-dir", default="./data/aihub")

    # convert 서브커맨드
    p_convert = subparsers.add_parser("convert", help="LabelMe → YOLO 변환")
    p_convert.add_argument("--input-dir", required=True, help="LabelMe JSON 디렉토리")
    p_convert.add_argument("--output-dir", required=True, help="YOLO 라벨 출력 디렉토리")

    # split 서브커맨드
    p_split = subparsers.add_parser("split", help="데이터셋 분할")
    p_split.add_argument("--source-dir", required=True, help="원본 디렉토리")
    p_split.add_argument("--output-dir", required=True, help="출력 디렉토리")
    p_split.add_argument("--train-ratio", type=float, default=0.7)
    p_split.add_argument("--val-ratio", type=float, default=0.2)
    p_split.add_argument("--test-ratio", type=float, default=0.1)

    # augment 서브커맨드
    p_augment = subparsers.add_parser("augment", help="정육 이미지 증강")
    p_augment.add_argument("--input-dir", required=True, help="원본 이미지 디렉토리")
    p_augment.add_argument("--output-dir", required=True, help="증강 이미지 출력 디렉토리")
    p_augment.add_argument("--num-augmented", type=int, default=5, help="원본당 증강 수")

    args = parser.parse_args()

    if args.command == "download":
        download_aihub_dataset(args.output_dir)
    elif args.command == "convert":
        convert_labelme_to_yolo(args.input_dir, args.output_dir)
    elif args.command == "split":
        split_dataset(
            args.source_dir, args.output_dir,
            args.train_ratio, args.val_ratio, args.test_ratio,
        )
    elif args.command == "augment":
        augment_meat_images(args.input_dir, args.output_dir, args.num_augmented)
    else:
        parser.print_help()
