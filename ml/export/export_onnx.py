"""모바일 온디바이스(AR) 배포용 ONNX 모델 내보내기 스크립트."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_yolo_to_onnx(model_path: str, output_path: str | None = None, imgsz: int = 640) -> str:
    """Ultralytics YOLO 모델을 ONNX 형식으로 변환."""
    try:
        from ultralytics import YOLO

        print(f"[*] YOLO 모델 로드 중: {model_path}")
        model = YOLO(model_path)
        print(f"[*] ONNX 내보내기 시작 (imgsz={imgsz}, dynamic=False, simplify=True)...")
        exported = model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)
        print(f"[+] 변환 성공: {exported}")
        return exported
    except Exception as e:
        print(f"[-] YOLO ONNX 변환 오류: {e}")
        raise


def export_torch_to_onnx(model_path: str, output_path: str, num_classes: int = 4, imgsz: int = 224) -> None:
    """PyTorch ResNet/Vision 모델을 ONNX 형식으로 변환."""
    import torch
    import torchvision.models as models

    print(f"[*] PyTorch 모델 로드 중: {model_path}")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(1, 3, imgsz, imgsz)

    print(f"[*] ONNX 파일 저장 중: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"[+] 변환 완료: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreshScanner ONNX Model Exporter")
    parser.add_argument("--type", choices=["yolo", "torch"], default="yolo", help="모델 타입")
    parser.add_argument("--model-path", type=str, required=True, help="입력 가중치 파일 경로")
    parser.add_argument("--output-path", type=str, default="model.onnx", help="출력 ONNX 경로")
    parser.add_argument("--imgsz", type=int, default=640, help="입력 이미지 해상도")

    args = parser.parse_args()

    if args.type == "yolo":
        export_yolo_to_onnx(args.model_path, imgsz=args.imgsz)
    else:
        export_torch_to_onnx(args.model_path, args.output_path, imgsz=args.imgsz)
