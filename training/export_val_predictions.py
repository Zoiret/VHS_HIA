from __future__ import annotations

import argparse
import csv
import json
import shutil
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


def _find_contours(mask01_u8: np.ndarray):
    res = cv2.findContours(mask01_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 2:
        contours, hierarchy = res
        return contours, hierarchy
    _, contours, hierarchy = res
    return contours, hierarchy


def _draw_mask_contours_rgb(
    image_rgb_u8: np.ndarray,
    mask_u8: np.ndarray,
    *,
    class_id: int,
    color_rgb: tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    m = (mask_u8 == int(class_id)).astype(np.uint8) * 255
    if not np.any(m):
        return image_rgb_u8
    contours, _ = _find_contours(m)
    if not contours:
        return image_rgb_u8
    cv2.drawContours(image_rgb_u8, contours, contourIdx=-1, color=tuple(int(x) for x in color_rgb), thickness=int(thickness))
    return image_rgb_u8


def _overlay_contours_rgb(image_rgb_u8: np.ndarray, gt_u8: np.ndarray, pred_u8: np.ndarray) -> np.ndarray:
    out = image_rgb_u8.copy()
    out = _draw_mask_contours_rgb(out, gt_u8, class_id=1, color_rgb=(0, 255, 0), thickness=2)
    out = _draw_mask_contours_rgb(out, gt_u8, class_id=2, color_rgb=(255, 0, 0), thickness=2)
    out = _draw_mask_contours_rgb(out, pred_u8, class_id=1, color_rgb=(0, 255, 255), thickness=1)
    out = _draw_mask_contours_rgb(out, pred_u8, class_id=2, color_rgb=(255, 0, 255), thickness=1)
    return out


def _count_components(mask01: np.ndarray) -> int:
    m = (mask01.astype(np.uint8) * 255)
    num, _, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return max(0, int(num) - 1)


def _dice(mask_pred01: np.ndarray, mask_gt01: np.ndarray, eps: float = 1e-7) -> float:
    p = mask_pred01.astype(bool)
    t = mask_gt01.astype(bool)
    inter = float(np.sum(p & t))
    ps = float(np.sum(p))
    ts = float(np.sum(t))
    return float((2.0 * inter + eps) / (ps + ts + eps))


def _iou(mask_pred01: np.ndarray, mask_gt01: np.ndarray, eps: float = 1e-7) -> float:
    p = mask_pred01.astype(bool)
    t = mask_gt01.astype(bool)
    inter = float(np.sum(p & t))
    union = float(np.sum(p | t))
    return float((inter + eps) / (union + eps))


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


def _infer_run_name(cfg: dict, *, config_path: Path, checkpoint_path: Path | None) -> str:
    train_cfg = cfg.get("train") or {}
    if isinstance(train_cfg, dict):
        save_dir = train_cfg.get("save_dir", None)
        if save_dir:
            return Path(str(save_dir)).name
    if checkpoint_path is not None:
        return checkpoint_path.parent.name
    return config_path.stem


def _infer_default_checkpoint(cfg: dict) -> Path | None:
    train_cfg = cfg.get("train") or {}
    if not isinstance(train_cfg, dict):
        return None
    save_dir = train_cfg.get("save_dir", None)
    if not save_dir:
        return None
    p = Path(str(save_dir)) / "best_mean_fg.pth"
    if p.exists():
        return p
    return None


def _mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def _median(xs: list[float]) -> float | None:
    return float(np.median(np.asarray(xs, dtype=np.float64))) if xs else None


def _std(xs: list[float]) -> float | None:
    return float(np.std(np.asarray(xs, dtype=np.float64), ddof=0)) if xs else None


def _copy_case_bundle(src_sample_dir: Path, dst_sample_dir: Path) -> None:
    dst_sample_dir.mkdir(parents=True, exist_ok=True)
    for name in ["image.png", "gt.png", "pred.png", "overlay.png", "compare.png"]:
        src = src_sample_dir / name
        if src.exists():
            shutil.copy2(src, dst_sample_dir / name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--split", type=str, choices=["val", "test"], default="val")
    ap.add_argument("--split-txt", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = _infer_default_checkpoint(cfg)
    if checkpoint_path is None:
        raise SystemExit(
            "Checkpoint is not specified and default best_mean_fg.pth was not found.\n"
            "Pass --checkpoint PATH_TO_best_mean_fg.pth"
        )
    checkpoint_path = checkpoint_path.resolve()

    run_name = _infer_run_name(cfg, config_path=args.config, checkpoint_path=checkpoint_path)
    analysis_root = (Path("training") / "analysis").resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)

    split_label = str(args.split).strip().lower()
    split_txt = args.split_txt
    if split_txt is None:
        if split_label == "val":
            split_txt = Path(cfg["dataset"]["val_txt"])
        elif split_label == "test":
            split_txt = Path(cfg["dataset"]["test_txt"])
        else:
            raise SystemExit(f"Unsupported split: {split_label!r}")
    split_txt = split_txt.resolve()

    out_root_default_name = f"{split_label}_predictions_{run_name}"
    out_root = args.output_dir.resolve() if args.output_dir is not None else (analysis_root / out_root_default_name)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_root / f"{run_name}_raw_{split_label}_metrics.csv"
    summary_path = analysis_root / f"{run_name}_raw_{split_label}_summary.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_properties(0).name}")

    model = _build_model(cfg).to(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    incompat = model.load_state_dict(state, strict=False)
    missing = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
    unexpected = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
    if missing or unexpected:
        print(f"Checkpoint load (non-strict): missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    ds_root = Path(cfg["dataset"]["root"]).resolve()
    val_txt = split_txt
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

    max_n = int(args.limit) if args.limit is not None else None
    n = 0
    use_amp = bool((cfg.get("train") or {}).get("amp", False)) and (device.type == "cuda")

    rows: list[dict] = []
    dice_leaflet_vals: list[float] = []
    dice_ring_vals: list[float] = []
    dice_fg_vals: list[float] = []
    iou_leaflet_vals: list[float] = []
    iou_ring_vals: list[float] = []
    pred_leaflet_components_vals: list[int] = []
    gt_leaflet_components_vals: list[int] = []
    pred_ring_components_vals: list[int] = []
    gt_ring_components_vals: list[int] = []
    leaflet_pixels_gt_vals: list[int] = []
    leaflet_pixels_pred_vals: list[int] = []
    ring_pixels_gt_vals: list[int] = []
    ring_pixels_pred_vals: list[int] = []

    with torch.no_grad():
        for batch in loader:
            image_t = batch["image"].to(device, non_blocking=True)
            mask_t = batch["mask"].to(device, non_blocking=True)
            image_path = batch.get("image_path", ["sample"])[0]
            mask_path = batch.get("mask_path", [None])[0]
            sample_id = Path(str(image_path)).stem

            with torch.amp.autocast("cuda", enabled=use_amp) if device.type == "cuda" else torch.no_grad():
                logits = model(image_t)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            if logits.shape[-2:] != mask_t.shape[-2:]:
                logits = torch.nn.functional.interpolate(logits, size=mask_t.shape[-2:], mode="bilinear", align_corners=False)
            pred_raw_t = torch.argmax(logits, dim=1)

            gt_u8 = mask_t[0].detach().cpu().numpy().astype(np.uint8)
            pred_raw_u8 = pred_raw_t[0].detach().cpu().numpy().astype(np.uint8)

            encoder_weights = (cfg.get("model") or {}).get("encoder_weights", None)
            if encoder_weights is None:
                img_np = image_t[0].detach().cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
                img_u8 = (img_np * 255.0 + 0.5).astype(np.uint8)
            else:
                raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if raw_bgr is None:
                    raise FileNotFoundError(f"Failed to read image: {image_path}")
                img_u8 = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                if mask_path is None:
                    raise SystemExit("mask_path is missing in dataloader batch")
                raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if raw_mask is None:
                    raise FileNotFoundError(f"Failed to read mask: {mask_path}")
                if raw_mask.ndim == 3:
                    raw_mask = raw_mask[:, :, 0]
                raw_mask = raw_mask.astype(np.uint8)

                if crop_mode == "anatomy_bbox":
                    fg = raw_mask > 0
                    if fg.any():
                        ys, xs = np.where(fg)
                        y0 = int(ys.min())
                        y1 = int(ys.max()) + 1
                        x0 = int(xs.min())
                        x1 = int(xs.max()) + 1
                        bbox_h = y1 - y0
                        bbox_w = x1 - x0
                        pad = int(round(max(bbox_h, bbox_w) * float(crop_padding)))
                        y0 = max(0, y0 - pad)
                        x0 = max(0, x0 - pad)
                        y1 = min(raw_mask.shape[0], y1 + pad)
                        x1 = min(raw_mask.shape[1], x1 + pad)
                        img_u8 = img_u8[y0:y1, x0:x1, :]
                        raw_mask = raw_mask[y0:y1, x0:x1]

                img_u8, _ = get_val_augmentations(input_size, input_size)(img_u8, raw_mask)

            sample_dir = out_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(sample_dir / "image.png"), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(sample_dir / "gt.png"), gt_u8)
            cv2.imwrite(str(sample_dir / "pred.png"), pred_raw_u8)

            overlay = _overlay_contours_rgb(img_u8, gt_u8, pred_raw_u8)
            cv2.imwrite(str(sample_dir / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            compare = np.concatenate([img_u8, overlay], axis=1)
            cv2.imwrite(str(sample_dir / "compare.png"), cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))

            dice_leaflet = _dice(pred_raw_u8 == 1, gt_u8 == 1) if num_classes >= 2 else None
            dice_ring = _dice(pred_raw_u8 == 2, gt_u8 == 2) if num_classes >= 3 else None
            dice_mean_fg = None
            if dice_leaflet is not None and dice_ring is not None:
                dice_mean_fg = float((dice_leaflet + dice_ring) / 2.0)

            iou_leaflet = _iou(pred_raw_u8 == 1, gt_u8 == 1) if num_classes >= 2 else None
            iou_ring = _iou(pred_raw_u8 == 2, gt_u8 == 2) if num_classes >= 3 else None

            pred_leaflet_components = _count_components(pred_raw_u8 == 1) if num_classes >= 2 else None
            gt_leaflet_components = _count_components(gt_u8 == 1) if num_classes >= 2 else None
            pred_ring_components = _count_components(pred_raw_u8 == 2) if num_classes >= 3 else None
            gt_ring_components = _count_components(gt_u8 == 2) if num_classes >= 3 else None

            leaflet_pixels_gt = int(np.sum(gt_u8 == 1)) if num_classes >= 2 else None
            leaflet_pixels_pred = int(np.sum(pred_raw_u8 == 1)) if num_classes >= 2 else None
            ring_pixels_gt = int(np.sum(gt_u8 == 2)) if num_classes >= 3 else None
            ring_pixels_pred = int(np.sum(pred_raw_u8 == 2)) if num_classes >= 3 else None

            row = {
                "filename": Path(str(image_path)).name,
                "dice_leaflet": dice_leaflet,
                "dice_ring": dice_ring,
                "mean_fg": dice_mean_fg,
                "iou_leaflet": iou_leaflet,
                "gt_leaflet_components": gt_leaflet_components,
                "pred_leaflet_components": pred_leaflet_components,
                "iou_ring": iou_ring,
                "gt_ring_components": gt_ring_components,
                "pred_ring_components": pred_ring_components,
                "leaflet_pixels_gt": leaflet_pixels_gt,
                "leaflet_pixels_pred": leaflet_pixels_pred,
                "ring_pixels_gt": ring_pixels_gt,
                "ring_pixels_pred": ring_pixels_pred,
                "sample_id": sample_id,
                "image_path": str(image_path),
            }
            rows.append(row)
            if dice_leaflet is not None:
                dice_leaflet_vals.append(float(dice_leaflet))
            if dice_ring is not None:
                dice_ring_vals.append(float(dice_ring))
            if dice_mean_fg is not None:
                dice_fg_vals.append(float(dice_mean_fg))
            if iou_leaflet is not None:
                iou_leaflet_vals.append(float(iou_leaflet))
            if iou_ring is not None:
                iou_ring_vals.append(float(iou_ring))
            if pred_leaflet_components is not None:
                pred_leaflet_components_vals.append(int(pred_leaflet_components))
            if gt_leaflet_components is not None:
                gt_leaflet_components_vals.append(int(gt_leaflet_components))
            if pred_ring_components is not None:
                pred_ring_components_vals.append(int(pred_ring_components))
            if gt_ring_components is not None:
                gt_ring_components_vals.append(int(gt_ring_components))
            if leaflet_pixels_gt is not None:
                leaflet_pixels_gt_vals.append(int(leaflet_pixels_gt))
            if leaflet_pixels_pred is not None:
                leaflet_pixels_pred_vals.append(int(leaflet_pixels_pred))
            if ring_pixels_gt is not None:
                ring_pixels_gt_vals.append(int(ring_pixels_gt))
            if ring_pixels_pred is not None:
                ring_pixels_pred_vals.append(int(ring_pixels_pred))

            n += 1
            if max_n is not None and n >= max_n:
                break

    fieldnames = [
        "filename",
        "dice_leaflet",
        "dice_ring",
        "mean_fg",
        "iou_leaflet",
        "iou_ring",
        "gt_leaflet_components",
        "pred_leaflet_components",
        "pred_ring_components",
        "gt_ring_components",
        "leaflet_pixels_gt",
        "leaflet_pixels_pred",
        "ring_pixels_gt",
        "ring_pixels_pred",
        "sample_id",
        "image_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    worst_root = out_root / "worst_cases"
    best10_root = worst_root / "best_10"
    worst10_root = worst_root / "worst_10"
    best10_root.mkdir(parents=True, exist_ok=True)
    worst10_root.mkdir(parents=True, exist_ok=True)

    sortable = []
    for r in rows:
        v = r.get("mean_fg", None)
        if v is None:
            continue
        sortable.append((float(v), str(r.get("sample_id"))))
    sortable.sort(key=lambda x: x[0])
    worst10 = sortable[:10]
    best10 = sortable[-10:][::-1]

    for rank, (v, sid) in enumerate(worst10, start=1):
        _copy_case_bundle(out_root / sid, worst10_root / f"{rank:02d}__{sid}__mean_fg_{v:.4f}")
    for rank, (v, sid) in enumerate(best10, start=1):
        _copy_case_bundle(out_root / sid, best10_root / f"{rank:02d}__{sid}__mean_fg_{v:.4f}")

    summary = {
        "run_name": run_name,
        "config": str(args.config.resolve()),
        "checkpoint": str(checkpoint_path),
        "split": split_label,
        "split_txt": str(split_txt),
        "count": int(n),
        "dice": {
            "leaflet": {"mean": _mean(dice_leaflet_vals), "median": _median(dice_leaflet_vals), "std": _std(dice_leaflet_vals)},
            "ring": {"mean": _mean(dice_ring_vals), "median": _median(dice_ring_vals), "std": _std(dice_ring_vals)},
            "mean_fg": {"mean": _mean(dice_fg_vals), "median": _median(dice_fg_vals), "std": _std(dice_fg_vals)},
        },
        "iou": {
            "leaflet": {"mean": _mean(iou_leaflet_vals), "median": _median(iou_leaflet_vals), "std": _std(iou_leaflet_vals)},
            "ring": {"mean": _mean(iou_ring_vals), "median": _median(iou_ring_vals), "std": _std(iou_ring_vals)},
        },
        "components": {
            "gt_leaflet": {"mean": _mean([float(x) for x in gt_leaflet_components_vals]), "median": _median([float(x) for x in gt_leaflet_components_vals]), "std": _std([float(x) for x in gt_leaflet_components_vals])},
            "pred_leaflet": {"mean": _mean([float(x) for x in pred_leaflet_components_vals]), "median": _median([float(x) for x in pred_leaflet_components_vals]), "std": _std([float(x) for x in pred_leaflet_components_vals])},
            "gt_ring": {"mean": _mean([float(x) for x in gt_ring_components_vals]), "median": _median([float(x) for x in gt_ring_components_vals]), "std": _std([float(x) for x in gt_ring_components_vals])},
            "pred_ring": {"mean": _mean([float(x) for x in pred_ring_components_vals]), "median": _median([float(x) for x in pred_ring_components_vals]), "std": _std([float(x) for x in pred_ring_components_vals])},
        },
        "pixels": {
            "leaflet_gt": {"mean": _mean([float(x) for x in leaflet_pixels_gt_vals]), "median": _median([float(x) for x in leaflet_pixels_gt_vals]), "std": _std([float(x) for x in leaflet_pixels_gt_vals])},
            "leaflet_pred": {"mean": _mean([float(x) for x in leaflet_pixels_pred_vals]), "median": _median([float(x) for x in leaflet_pixels_pred_vals]), "std": _std([float(x) for x in leaflet_pixels_pred_vals])},
            "ring_gt": {"mean": _mean([float(x) for x in ring_pixels_gt_vals]), "median": _median([float(x) for x in ring_pixels_gt_vals]), "std": _std([float(x) for x in ring_pixels_gt_vals])},
            "ring_pred": {"mean": _mean([float(x) for x in ring_pixels_pred_vals]), "median": _median([float(x) for x in ring_pixels_pred_vals]), "std": _std([float(x) for x in ring_pixels_pred_vals])},
        },
        "sort": {
            "best10": [{"sample_id": sid, "mean_fg": v} for v, sid in best10],
            "worst10": [{"sample_id": sid, "mean_fg": v} for v, sid in worst10],
        },
        "outputs": {
            "predictions_dir": str(out_root),
            "raw_metrics_csv": str(csv_path),
            "raw_summary_json": str(summary_path),
            "worst_cases_dir": str(worst_root),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Exported: {n} val samples")
    print(f"Predictions: {out_root}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
