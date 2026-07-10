from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
try:
    import torch
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e


def _read_image_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _center_crop(image: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h < crop_h or w < crop_w:
        image = cv2.resize(image, (max(w, crop_w), max(h, crop_h)), interpolation=cv2.INTER_LINEAR)
        h, w = image.shape[:2]
    y0 = (h - crop_h) // 2 if h > crop_h else 0
    x0 = (w - crop_w) // 2 if w > crop_w else 0
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out-mask", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=768)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--encoder", type=str, default="efficientnet-b3")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--postprocess-preset", type=str, default=None)
    args = parser.parse_args()

    import segmentation_models_pytorch as smp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is not available, using CPU.")
    print(f"Device: {device}")

    model = smp.UnetPlusPlus(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights if args.encoder_weights else None,
        in_channels=3,
        classes=int(args.num_classes),
    )
    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)

    img = _read_image_rgb(args.image)
    img = _center_crop(img, args.input_size, args.input_size)
    img = preprocessing_fn(img)
    x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    if args.postprocess_preset:
        from postprocess import postprocess_multiclass_mask

        pred = postprocess_multiclass_mask(pred, preset=str(args.postprocess_preset))

    args.out_mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_mask), pred)
    print(f"Saved mask: {args.out_mask}")


if __name__ == "__main__":
    main()
