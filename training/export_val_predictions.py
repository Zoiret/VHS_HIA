from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e

from augmentations import get_val_augmentations
from dataset import SegmentationDataset
from postprocess import apply_postprocess


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_properties(0).name}")

    model = _build_model(cfg).to(device)
    ckpt = torch.load(str(args.checkpoint), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    incompat = model.load_state_dict(state, strict=False)
    missing = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
    unexpected = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
    if missing or unexpected:
        print(f"Checkpoint load (non-strict): missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    ds_root = Path(cfg["dataset"]["root"]).resolve()
    val_txt = Path(cfg["dataset"]["val_txt"]).resolve()
    num_classes = int(cfg["model"]["classes"])
    input_size = int(cfg["model"]["input_size"])
    preprocess = _build_preprocess(cfg)

    dataset_cfg = cfg.get("dataset") or {}
    target = dataset_cfg.get("target", None)
    crop_mode = dataset_cfg.get("crop_mode", None)
    crop_padding = float(dataset_cfg.get("crop_padding", 0.0)) if isinstance(dataset_cfg, dict) else 0.0
    boundary_cfg = dataset_cfg.get("boundary", None) if isinstance(dataset_cfg, dict) else None

    val_ds = SegmentationDataset(
        dataset_root=ds_root,
        split_txt=val_txt,
        num_classes=num_classes,
        target=target,
        crop_mode=crop_mode,
        crop_padding=crop_padding,
        boundary_cfg=boundary_cfg,
        augment_fn=get_val_augmentations(input_size, input_size),
        preprocessing_fn=preprocess,
    )

    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    max_n = int(args.limit) if args.limit is not None else None
    n = 0
    use_amp = bool((cfg.get("train") or {}).get("amp", False)) and (device.type == "cuda")

    with torch.no_grad():
        for batch in loader:
            image_t = batch["image"].to(device, non_blocking=True)
            mask_t = batch["mask"].to(device, non_blocking=True)
            image_path = batch.get("image_path", ["sample"])[0]
            sample_id = Path(str(image_path)).stem

            with torch.amp.autocast("cuda", enabled=use_amp) if device.type == "cuda" else torch.no_grad():
                logits = model(image_t)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            if logits.shape[-2:] != mask_t.shape[-2:]:
                logits = torch.nn.functional.interpolate(logits, size=mask_t.shape[-2:], mode="bilinear", align_corners=False)
            pred_t = torch.argmax(logits, dim=1)

            img_np = image_t[0].detach().cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
            img_u8 = (img_np * 255.0 + 0.5).astype(np.uint8)
            gt_u8 = mask_t[0].detach().cpu().numpy().astype(np.uint8)
            pred_u8 = pred_t[0].detach().cpu().numpy().astype(np.uint8)

            pred_u8 = apply_postprocess(pred_u8, cfg.get("postprocess", None))

            sample_dir = out_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(sample_dir / "image.png"), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(sample_dir / "gt.png"), gt_u8)
            cv2.imwrite(str(sample_dir / "pred.png"), pred_u8)

            overlay_gt = _overlay_rgb(img_u8, gt_u8)
            overlay_pred = _overlay_rgb(img_u8, pred_u8)
            cv2.imwrite(str(sample_dir / "overlay.png"), cv2.cvtColor(overlay_pred, cv2.COLOR_RGB2BGR))

            compare = np.concatenate([img_u8, overlay_gt, overlay_pred], axis=1)
            cv2.imwrite(str(sample_dir / "compare.png"), cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))

            n += 1
            if max_n is not None and n >= max_n:
                break

    print(f"Exported: {n} samples to {out_root}")


if __name__ == "__main__":
    main()
