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


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise SystemExit("PyYAML is not installed. Install with:\n  py -m pip install pyyaml") from e
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid config: expected a dict at root, got {type(data).__name__}")
    return data


def _center_crop_resize_rgb(image_rgb: np.ndarray, crop: int) -> np.ndarray:
    crop = int(crop)
    h, w = image_rgb.shape[:2]
    if h < crop or w < crop:
        new_h = max(h, crop)
        new_w = max(w, crop)
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = new_h, new_w
    y0 = (h - crop) // 2 if h > crop else 0
    x0 = (w - crop) // 2 if w > crop else 0
    return image_rgb[y0 : y0 + crop, x0 : x0 + crop, :]


def _overlay_rgb(image_rgb_u8: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = image_rgb_u8.copy()
    a = float(alpha)
    m1 = mask_u8 == 1
    if m1.any():
        out[m1] = (out[m1].astype(np.float32) * (1 - a) + np.array([0, 255, 0], dtype=np.float32) * a).astype(np.uint8)
    m2 = mask_u8 == 2
    if m2.any():
        out[m2] = (out[m2].astype(np.float32) * (1 - a) + np.array([255, 0, 0], dtype=np.float32) * a).astype(np.uint8)
    return out


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"CUDA device: {props.name}")
    return device


def _deep_supervision_enabled(cfg: dict) -> bool:
    m = cfg.get("model") or {}
    return bool(m.get("deep_supervision", False)) if isinstance(m, dict) else False


def _forward_unetpp_deep_supervision(self: torch.nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    features = self.encoder(x)
    feats = features[1:]
    feats = feats[::-1]

    depth = int(getattr(self.decoder, "depth"))
    in_channels = list(getattr(self.decoder, "in_channels"))
    blocks = getattr(self.decoder, "blocks")

    dense_x: dict[str, torch.Tensor] = {}
    for layer_idx in range(len(in_channels) - 1):
        for depth_idx in range(depth - layer_idx):
            if layer_idx == 0:
                out = blocks[f"x_{depth_idx}_{depth_idx}"](feats[depth_idx], feats[depth_idx + 1])
                dense_x[f"x_{depth_idx}_{depth_idx}"] = out
            else:
                dense_l_i = depth_idx + layer_idx
                cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                cat_features = torch.cat(cat_features + [feats[dense_l_i + 1]], dim=1)
                dense_x[f"x_{depth_idx}_{dense_l_i}"] = blocks[f"x_{depth_idx}_{dense_l_i}"](
                    dense_x[f"x_{depth_idx}_{dense_l_i - 1}"], cat_features
                )

    dense_x[f"x_0_{depth}"] = blocks[f"x_0_{depth}"](dense_x[f"x_0_{depth - 1}"])

    keys = [f"x_0_{depth}", f"x_0_{depth - 1}", f"x_0_{depth - 2}", f"x_0_{depth - 3}"]
    outs: list[torch.Tensor] = []
    for k in keys:
        v = dense_x.get(k, None)
        if v is not None:
            outs.append(self.segmentation_head(v))
    return outs


def _build_model(cfg: dict) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    m = cfg.get("model") or {}
    if not isinstance(m, dict):
        raise SystemExit("Config: model must be a dict")

    encoder = m.get("encoder") or m.get("encoder_name")
    if not encoder:
        raise SystemExit("Config: model.encoder (or model.encoder_name) is required")
    classes = int(m.get("classes", 3))
    in_channels = int(m.get("in_channels", 3))
    encoder_weights = m.get("encoder_weights", None)

    model = smp.UnetPlusPlus(
        encoder_name=str(encoder),
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    if _deep_supervision_enabled(cfg):
        import types

        model.forward = types.MethodType(_forward_unetpp_deep_supervision, model)
    return model


def _build_preprocess(cfg: dict):
    import segmentation_models_pytorch as smp

    m = cfg.get("model") or {}
    encoder = m.get("encoder") or m.get("encoder_name")
    encoder_weights = m.get("encoder_weights", None)
    if encoder_weights is None:
        def fn(image_rgb_u8: np.ndarray) -> np.ndarray:
            return image_rgb_u8.astype(np.float32) / 255.0
        return fn
    return smp.encoders.get_preprocessing_fn(str(encoder), encoder_weights)


def _iter_images(input_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    device = _select_device()

    model = _build_model(cfg).to(device)
    ckpt = torch.load(str(args.checkpoint), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    input_size = int((cfg.get("model") or {}).get("input_size", 768))
    preprocess = _build_preprocess(cfg)

    out_dir = args.output_dir.resolve()
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    from postprocess import apply_postprocess

    files = _iter_images(args.input_dir.resolve())
    if not files:
        raise SystemExit(f"No images found in {args.input_dir.resolve()}")

    variants = [
        ("orig", lambda t: t, lambda t: t),
        ("hflip", lambda t: torch.flip(t, dims=[3]), lambda t: torch.flip(t, dims=[3])),
        ("vflip", lambda t: torch.flip(t, dims=[2]), lambda t: torch.flip(t, dims=[2])),
        ("rot90", lambda t: torch.rot90(t, k=1, dims=[2, 3]), lambda t: torch.rot90(t, k=-1, dims=[2, 3])),
        ("rot180", lambda t: torch.rot90(t, k=2, dims=[2, 3]), lambda t: torch.rot90(t, k=-2, dims=[2, 3])),
        ("rot270", lambda t: torch.rot90(t, k=3, dims=[2, 3]), lambda t: torch.rot90(t, k=-3, dims=[2, 3])),
    ]

    use_amp = bool((cfg.get("train") or {}).get("amp", False)) and (device.type == "cuda")
    autocast_ctx = torch.amp.autocast("cuda", enabled=True) if use_amp else torch.no_grad()

    for p in files:
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Skip unreadable image: {p}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = _center_crop_resize_rgb(img_rgb, crop=input_size)
        x_np = preprocess(img_rgb)
        x = torch.from_numpy(x_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

        logits_sum = None
        with torch.no_grad():
            for _, fwd, inv in variants:
                x_aug = fwd(x)
                with autocast_ctx:
                    logits = model(x_aug)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                logits = inv(logits)
                logits_sum = logits if logits_sum is None else (logits_sum + logits)
        logits_avg = logits_sum / float(len(variants))
        pred = torch.argmax(logits_avg, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        pred = apply_postprocess(pred, cfg.get("postprocess", None))

        stem = p.stem
        mask_path = out_dir / "masks" / f"{stem}.png"
        overlay_path = out_dir / "overlays" / f"{stem}.png"
        cv2.imwrite(str(mask_path), pred)
        overlay = _overlay_rgb(img_rgb.astype(np.uint8), pred)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
