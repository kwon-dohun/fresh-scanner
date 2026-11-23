"""모바일 온디바이스(iOS CoreML / Android ONNX Runtime) 배포용 ONNX 변환 스크립트 (100% Apache-2.0 / BSD 안전 라이선스)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("onnx_export")


def export_deeplabv3_to_onnx(model_path: str | None = None, output_path: str = "meat_deeplabv3_mobilenet.onnx", imgsz: int = 256) -> str:
    """Torchvision DeepLabV3-MobileNetV3 정육 세그멘테이션 모델을 ONNX 형식으로 변환."""
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

    logger.info("DeepLabV3-MobileNetV3 (BSD-3 라이선스) 로드 중...")
    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=4)
    if model_path and Path(model_path).exists():
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, imgsz, imgsz)

    logger.info(f"ONNX 변환 시작 (imgsz={imgsz}, opset=14) -> {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["image_input"],
        output_names=["segmentation_output"],
        dynamic_axes={"image_input": {0: "batch_size"}, "segmentation_output": {0: "batch_size"}},
    )
    logger.info(f"변환 완료 성공: {output_path}")
    return output_path


def export_ssdlite_to_onnx(output_path: str = "food_ssdlite_mobilenet.onnx", imgsz: int = 320) -> str:
    """Torchvision SSDLite-MobileNetV3 객체 탐지 모델 ONNX 변환."""
    import torchvision.models.detection as detection

    logger.info("SSDLite-MobileNetV3 (BSD 라이선스) 로드 중...")
    model = detection.ssdlite320_mobilenet_v3_large(weights=None)
    model.eval()

    dummy_input = [torch.randn(3, imgsz, imgsz)]

    logger.info(f"SSDLite 탐지기 ONNX 내보내기 -> {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
    )
    logger.info(f"변환 완료 성공: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreshScanner Apache-2.0 ONNX Exporter")
    parser.add_argument("--type", choices=["segmentation", "detection"], default="segmentation")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="model.onnx")
    parser.add_argument("--imgsz", type=int, default=256)

    args = parser.parse_args()

    if args.type == "segmentation":
        export_deeplabv3_to_onnx(args.model_path, args.output_path, imgsz=args.imgsz)
    else:
        export_ssdlite_to_onnx(args.output_path, imgsz=args.imgsz)
