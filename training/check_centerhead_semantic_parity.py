from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from augmentations import get_val_augmentations
from dataset import SegmentationDataset
from metrics import compute_per_class_metrics_from_logits
from models_centerhead import UnetPlusPlusSemanticCenterHead, load_semantic_checkpoint_non_strict


def _read_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise SystemExit("pyyaml is not installed. Install training deps with:\n  py -m pip install -r requirements-train.txt") from e
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"Config root must be a dict: {path}")
    return obj


def _simple_preprocess_uint8_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    return (img_rgb_u8.astype(np.float32) / 255.0).astype(np.float32)


def _load_checkpoint_state(checkpoint_path: str) -> dict:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt.get("model") if isinstance(ckpt, dict) else None
    if state is None:
        state = ckpt
    if not isinstance(state, dict):
        raise SystemExit(f"Unsupported checkpoint format: {checkpoint_path}")
    return state


def _build_val_loader(cfg: dict, *, batch_size: int, num_workers: int, device: torch.device):
    ds_root = Path(cfg["dataset"]["root"]).resolve()
    val_txt = Path(cfg["dataset"]["val_txt"]).resolve()

    num_classes = int(cfg["model"]["classes"])
    input_size = int(cfg["model"]["input_size"])

    import segmentation_models_pytorch as smp

    encoder = cfg["model"].get("encoder") or cfg["model"].get("encoder_name")
    if not encoder:
        raise SystemExit("Config: model.encoder_name is required")
    encoder_weights = cfg["model"].get("encoder_weights", None)
    if encoder_weights is None:
        preprocessing_fn = _simple_preprocess_uint8_rgb
    else:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(str(encoder), encoder_weights)

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
        preprocessing_fn=preprocessing_fn,
    )

    if device.type != "cuda":
        num_workers = 0

    dl_kwargs = {}
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = False
        dl_kwargs["prefetch_factor"] = 2

    loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        **dl_kwargs,
    )
    return loader


def _build_original_model(cfg: dict) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    encoder = cfg["model"].get("encoder") or cfg["model"].get("encoder_name")
    if not encoder:
        raise SystemExit("Config: model.encoder_name is required")

    return smp.UnetPlusPlus(
        encoder_name=str(encoder),
        encoder_weights=cfg["model"].get("encoder_weights", None),
        in_channels=int(cfg["model"]["in_channels"]),
        classes=int(cfg["model"]["classes"]),
    )


def _maybe_unwrap_logits(out: torch.Tensor | list | tuple) -> torch.Tensor:
    if isinstance(out, (list, tuple)):
        if not out:
            raise RuntimeError("Model returned empty outputs list/tuple")
        return out[0]
    return out


@torch.no_grad()
def _run_parity(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    num_classes: int,
    max_batches: int | None,
) -> dict:
    model_a.eval()
    model_b.eval()

    sum_abs = 0.0
    cnt_abs = 0
    max_abs = 0.0

    disagree = 0
    total_pix = 0

    dice_a_sum = [0.0 for _ in range(int(num_classes))]
    dice_b_sum = [0.0 for _ in range(int(num_classes))]
    samples = 0

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= int(max_batches):
            break

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits_a = _maybe_unwrap_logits(model_a(images))
        out_b = model_b(images)
        if not isinstance(out_b, dict) or "semantic" not in out_b:
            raise RuntimeError("Wrapper model must return dict with key 'semantic'")
        logits_b = out_b["semantic"]

        if tuple(logits_a.shape) != tuple(logits_b.shape):
            raise RuntimeError(f"Logits shape mismatch: original={tuple(logits_a.shape)} wrapper={tuple(logits_b.shape)}")

        diff = (logits_a - logits_b).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        sum_abs += float(diff.sum().item())
        cnt_abs += int(diff.numel())

        pred_a = torch.argmax(logits_a, dim=1)
        pred_b = torch.argmax(logits_b, dim=1)
        d = (pred_a != pred_b)
        disagree += int(d.sum().item())
        total_pix += int(d.numel())

        m_a = compute_per_class_metrics_from_logits(logits_a, masks, num_classes=int(num_classes))
        m_b = compute_per_class_metrics_from_logits(logits_b, masks, num_classes=int(num_classes))

        bs = int(images.shape[0])
        samples += bs
        for c in range(int(num_classes)):
            dice_a_sum[c] += float(m_a.dice[c]) * bs
            dice_b_sum[c] += float(m_b.dice[c]) * bs

    dice_a = [float(x / max(samples, 1)) for x in dice_a_sum]
    dice_b = [float(x / max(samples, 1)) for x in dice_b_sum]

    if int(num_classes) == 3:
        mean_fg_a = float((dice_a[1] + dice_a[2]) / 2.0)
        mean_fg_b = float((dice_b[1] + dice_b[2]) / 2.0)
    else:
        mean_fg_a = float(dice_a[1]) if int(num_classes) > 1 else float(dice_a[0])
        mean_fg_b = float(dice_b[1]) if int(num_classes) > 1 else float(dice_b[0])

    return {
        "samples": int(samples),
        "max_abs_logits_diff": float(max_abs),
        "mean_abs_logits_diff": float(sum_abs / max(cnt_abs, 1)),
        "argmax_disagreement_rate": float(disagree / max(total_pix, 1)),
        "original": {
            "dice_per_class": dice_a,
            "dice_leaflet": float(dice_a[1]) if int(num_classes) > 1 else None,
            "dice_ring": float(dice_a[2]) if int(num_classes) > 2 else None,
            "mean_fg": float(mean_fg_a),
        },
        "wrapper": {
            "dice_per_class": dice_b,
            "dice_leaflet": float(dice_b[1]) if int(num_classes) > 1 else None,
            "dice_ring": float(dice_b[2]) if int(num_classes) > 2 else None,
            "mean_fg": float(mean_fg_b),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    cfg = _read_yaml(Path(args.config))
    num_classes = int(cfg["model"]["classes"])

    device = torch.device(args.device) if str(args.device).strip() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = str(args.checkpoint)
    state = _load_checkpoint_state(ckpt_path)

    model_orig = _build_original_model(cfg)
    incompat = model_orig.load_state_dict(state, strict=True)
    if incompat is not None and (getattr(incompat, "missing_keys", None) or getattr(incompat, "unexpected_keys", None)):
        raise RuntimeError("Original model strict load produced missing/unexpected keys")

    model_wrap = UnetPlusPlusSemanticCenterHead(
        encoder_name=str(cfg["model"].get("encoder") or cfg["model"].get("encoder_name")),
        encoder_weights=cfg["model"].get("encoder_weights", None),
        in_channels=int(cfg["model"]["in_channels"]),
        classes=int(cfg["model"]["classes"]),
    )
    missing, unexpected = load_semantic_checkpoint_non_strict(model_wrap, ckpt_path)

    loader = _build_val_loader(cfg, batch_size=int(args.batch_size), num_workers=int(args.num_workers), device=device)

    model_orig = model_orig.to(device)
    model_wrap = model_wrap.to(device)

    max_batches = int(args.max_batches) if int(args.max_batches) > 0 else None
    res = _run_parity(model_orig, model_wrap, loader, device, num_classes=num_classes, max_batches=max_batches)
    res["checkpoint"] = ckpt_path
    res["config"] = str(Path(args.config))
    res["device"] = str(device)
    res["wrapper_load_report"] = {"missing_keys": missing, "unexpected_keys": unexpected}

    out_path = str(args.out).strip()
    if out_path:
        p = Path(out_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("# SEMANTIC PARITY")
    print(f"- original mean_fg: {res['original']['mean_fg']:.6f}")
    print(f"- wrapper mean_fg: {res['wrapper']['mean_fg']:.6f}")
    print(f"- argmax disagreement: {res['argmax_disagreement_rate']:.8f}")
    print(f"- max logits difference: {res['max_abs_logits_diff']:.8e}")
    print(f"- mean logits difference: {res['mean_abs_logits_diff']:.8e}")
    print(f"- parity passed: {(res['argmax_disagreement_rate'] < 1e-6 and res['max_abs_logits_diff'] < 1e-5)}")


if __name__ == "__main__":
    main()

