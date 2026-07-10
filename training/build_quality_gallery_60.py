from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
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

from postprocess import postprocess_multiclass_mask


@dataclass(frozen=True)
class SampleInfo:
    sample_id: str
    quality_class: str
    split_source: str
    image_path: Path
    gt_path: Path


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise SystemExit(
            "PyYAML is not installed. Install training deps with:\n"
            "  py -m pip install -r requirements-train.txt"
        ) from e
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid config: expected a dict at root, got {type(data).__name__}")
    return data


def _read_rgb_u8(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_u8(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(str(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def _center_crop_pair(image_rgb: np.ndarray, mask_u8: np.ndarray, crop: int) -> tuple[np.ndarray, np.ndarray]:
    crop_h = int(crop)
    crop_w = int(crop)
    h, w = image_rgb.shape[:2]
    if h < crop_h or w < crop_w:
        new_w = max(w, crop_w)
        new_h = max(h, crop_h)
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_u8 = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        h, w = image_rgb.shape[:2]
    y0 = (h - crop_h) // 2 if h > crop_h else 0
    x0 = (w - crop_w) // 2 if w > crop_w else 0
    return image_rgb[y0 : y0 + crop_h, x0 : x0 + crop_w], mask_u8[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _find_contours(mask01_u8: np.ndarray):
    res = cv2.findContours(mask01_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 2:
        contours, hierarchy = res
        return contours, hierarchy
    _, contours, hierarchy = res
    return contours, hierarchy


def _draw_mask_contours_rgb(image_rgb_u8: np.ndarray, mask_u8: np.ndarray, *, class_id: int, color_rgb, thickness: int) -> None:
    m = (mask_u8 == int(class_id)).astype(np.uint8) * 255
    if not np.any(m):
        return
    contours, _ = _find_contours(m)
    if not contours:
        return
    cv2.drawContours(image_rgb_u8, contours, contourIdx=-1, color=tuple(int(x) for x in color_rgb), thickness=int(thickness))


def _overlay_contours_rgb(image_rgb_u8: np.ndarray, gt_u8: np.ndarray, pred_u8: np.ndarray) -> np.ndarray:
    out = image_rgb_u8.copy()
    _draw_mask_contours_rgb(out, gt_u8, class_id=1, color_rgb=(0, 255, 0), thickness=2)
    _draw_mask_contours_rgb(out, gt_u8, class_id=2, color_rgb=(255, 0, 0), thickness=2)
    _draw_mask_contours_rgb(out, pred_u8, class_id=1, color_rgb=(0, 255, 255), thickness=1)
    _draw_mask_contours_rgb(out, pred_u8, class_id=2, color_rgb=(255, 0, 255), thickness=1)
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


def _metric_pack(gt_u8: np.ndarray, pred_u8: np.ndarray) -> dict:
    dice_leaflet = _dice(pred_u8 == 1, gt_u8 == 1)
    dice_ring = _dice(pred_u8 == 2, gt_u8 == 2)
    mean_fg = float((dice_leaflet + dice_ring) / 2.0)
    iou_leaflet = _iou(pred_u8 == 1, gt_u8 == 1)
    iou_ring = _iou(pred_u8 == 2, gt_u8 == 2)
    mean_iou = float((iou_leaflet + iou_ring) / 2.0)

    gt_leaflet_pixels = int(np.sum(gt_u8 == 1))
    pred_leaflet_pixels = int(np.sum(pred_u8 == 1))
    gt_ring_pixels = int(np.sum(gt_u8 == 2))
    pred_ring_pixels = int(np.sum(pred_u8 == 2))

    gt_leaflet_components = _count_components(gt_u8 == 1)
    pred_leaflet_components = _count_components(pred_u8 == 1)
    gt_ring_components = _count_components(gt_u8 == 2)
    pred_ring_components = _count_components(pred_u8 == 2)

    return {
        "dice_leaflet": float(dice_leaflet),
        "dice_ring": float(dice_ring),
        "mean_fg": float(mean_fg),
        "iou_leaflet": float(iou_leaflet),
        "iou_ring": float(iou_ring),
        "mean_iou": float(mean_iou),
        "gt_leaflet_pixels": int(gt_leaflet_pixels),
        "pred_leaflet_pixels": int(pred_leaflet_pixels),
        "gt_ring_pixels": int(gt_ring_pixels),
        "pred_ring_pixels": int(pred_ring_pixels),
        "gt_leaflet_components": int(gt_leaflet_components),
        "pred_leaflet_components": int(pred_leaflet_components),
        "gt_ring_components": int(gt_ring_components),
        "pred_ring_components": int(pred_ring_components),
    }


def _colorize_mask(mask_u8: np.ndarray) -> np.ndarray:
    out = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 3), dtype=np.uint8)
    out[mask_u8 == 1] = (0, 255, 0)
    out[mask_u8 == 2] = (255, 0, 0)
    return out


def _text(img: np.ndarray, x: int, y: int, s: str, *, scale: float = 0.6, color=(255, 255, 255), thickness: int = 1) -> None:
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, float(scale), tuple(int(c) for c in color), int(thickness), cv2.LINE_AA)


def _make_compare(
    *,
    original_rgb: np.ndarray,
    gt_u8: np.ndarray,
    pred_raw_u8: np.ndarray,
    pred_r2_u8: np.ndarray,
    title_lines: list[str],
    legend_lines: list[str],
) -> np.ndarray:
    h, w = original_rgb.shape[:2]
    panel_gt = _colorize_mask(gt_u8)
    panel_raw = _colorize_mask(pred_raw_u8)
    panel_r2 = _colorize_mask(pred_r2_u8)
    panels = [original_rgb, panel_gt, panel_raw, panel_r2]
    grid = np.concatenate(panels, axis=1)

    header_h = 160
    header = np.zeros((header_h, grid.shape[1], 3), dtype=np.uint8)
    header[:] = (20, 20, 20)

    y = 24
    for line in title_lines:
        _text(header, 12, y, line, scale=0.65, thickness=2)
        y += 24

    y = 24
    x0 = int(grid.shape[1] * 0.58)
    for line in legend_lines:
        _text(header, x0, y, line, scale=0.55, thickness=1)
        y += 20

    out = np.concatenate([header, grid], axis=0)

    caption_y = header_h + 28
    cap_color = (255, 255, 255)
    _text(out, 12, caption_y, "ORIGINAL", scale=0.8, color=cap_color, thickness=2)
    _text(out, 12 + w, caption_y, "GT (colors)", scale=0.8, color=cap_color, thickness=2)
    _text(out, 12 + 2 * w, caption_y, "PRED RAW (colors)", scale=0.8, color=cap_color, thickness=2)
    _text(out, 12 + 3 * w, caption_y, "PRED R2 (colors)", scale=0.8, color=cap_color, thickness=2)
    return out


def _mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def _median(xs: list[float]) -> float | None:
    return float(np.median(np.asarray(xs, dtype=np.float64))) if xs else None


def _std(xs: list[float]) -> float | None:
    return float(np.std(np.asarray(xs, dtype=np.float64), ddof=0)) if xs else None


def _stats(xs: list[float]) -> dict:
    return {"mean": _mean(xs), "median": _median(xs), "std": _std(xs)}


def _read_split_ids(split_txt: Path) -> set[str]:
    ids: set[str] = set()
    with split_txt.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise SystemExit(f"Invalid line in {split_txt}: {line!r}")
            img_rel = parts[0]
            ids.add(Path(img_rel).stem)
    return ids


def _select_n(ids: list[str], *, split_priority: list[str], split_map: dict[str, str], n: int) -> list[str]:
    out = []
    used = set()
    for split in split_priority:
        for sid in ids:
            if sid in used:
                continue
            if split_map.get(sid) != split:
                continue
            out.append(sid)
            used.add(sid)
            if len(out) >= int(n):
                return out
    for sid in ids:
        if sid in used:
            continue
        out.append(sid)
        used.add(sid)
        if len(out) >= int(n):
            break
    return out


def _build_model(cfg: dict) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    m = cfg.get("model") or {}
    encoder = m.get("encoder") or m.get("encoder_name")
    if not encoder:
        raise SystemExit("Config: model.encoder (or model.encoder_name) is required")
    model = smp.UnetPlusPlus(
        encoder_name=str(encoder),
        encoder_weights=m.get("encoder_weights", None),
        in_channels=int(m.get("in_channels", 3)),
        classes=int(m.get("classes", 3)),
    )
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--curation-json", type=Path, default=Path("server_assets/curation/curation_result.json"))
    ap.add_argument("--dataset-root", type=Path, default=Path("datasets/converted_full_multiclass"))
    ap.add_argument("--full-split-dir", type=Path, default=Path("datasets/converted_full_multiclass"))
    ap.add_argument("--out-dir", type=Path, default=Path("training/analysis/quality_gallery_60"))
    ap.add_argument("--n-per-class", type=int, default=20)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    input_size = int((cfg.get("model") or {}).get("input_size", 768))

    curation_path = args.curation_json.resolve()
    curation = json.loads(curation_path.read_text(encoding="utf-8"))
    if not isinstance(curation, dict):
        raise SystemExit(f"Invalid curation json: {curation_path}")

    for k in ["clean", "medium", "bad"]:
        if k not in curation or not isinstance(curation.get(k), list):
            raise SystemExit(f"Curated categories not found in {curation_path}. Missing key: {k!r}")

    clean_ids = [str(x) for x in curation.get("clean", [])]
    medium_ids = [str(x) for x in curation.get("medium", [])]
    bad_ids = [str(x) for x in curation.get("bad", [])]

    dup = set(clean_ids) & set(medium_ids) | set(clean_ids) & set(bad_ids) | set(medium_ids) & set(bad_ids)
    if dup:
        raise SystemExit(f"Curation categories overlap: {sorted(list(dup))[:10]}")

    full_split_dir = args.full_split_dir.resolve()
    train_ids = _read_split_ids(full_split_dir / "train.txt")
    val_ids = _read_split_ids(full_split_dir / "val.txt")
    test_ids = _read_split_ids(full_split_dir / "test.txt")
    split_map = {}
    for sid in train_ids:
        split_map[sid] = "train"
    for sid in val_ids:
        split_map[sid] = "val"
    for sid in test_ids:
        split_map[sid] = "test"

    split_priority = ["test", "val", "train"]
    n = int(args.n_per_class)
    selected_clean = _select_n(clean_ids, split_priority=split_priority, split_map=split_map, n=n)
    selected_medium = _select_n(medium_ids, split_priority=split_priority, split_map=split_map, n=n)
    selected_bad = _select_n(bad_ids, split_priority=split_priority, split_map=split_map, n=n)

    selected = []
    for sid in selected_clean:
        selected.append(("clean", sid))
    for sid in selected_medium:
        selected.append(("medium", sid))
    for sid in selected_bad:
        selected.append(("bad", sid))

    seen = set()
    for q, sid in selected:
        if sid in seen:
            raise SystemExit(f"Duplicate selected: {sid}")
        seen.add(sid)

    ds_root = args.dataset_root.resolve()
    items: list[SampleInfo] = []
    for q, sid in selected:
        img_path = (ds_root / "images" / f"{sid}.png").resolve()
        gt_path = (ds_root / "masks" / f"{sid}.png").resolve()
        if not img_path.exists():
            raise SystemExit(f"Missing image: {img_path}")
        if not gt_path.exists():
            raise SystemExit(f"Missing gt mask: {gt_path}")
        split_source = split_map.get(sid, "unknown")
        if split_source not in {"train", "val", "test"}:
            raise SystemExit(f"Sample {sid} not found in full split files (train/val/test)")
        items.append(SampleInfo(sample_id=sid, quality_class=q, split_source=split_source, image_path=img_path, gt_path=gt_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg).to(device)
    ckpt = torch.load(str(args.checkpoint), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    per_item_metrics = {}

    legend_lines = [
        "Legend:",
        "Mask colors: bg=black, leaflet=green, ring=red",
        "Contours: GT leaf=green thick, GT ring=red thick",
        "Contours: Pred leaf=cyan thin, Pred ring=magenta thin",
        "Postprocess: ring_erosion_r2 (ring only, leaflet unchanged)",
    ]

    for info in items:
        img = _read_rgb_u8(info.image_path)
        gt = _read_u8(info.gt_path)

        uniq = set(np.unique(gt).tolist())
        if not uniq.issubset({0, 1, 2}):
            raise SystemExit(f"Unexpected class IDs in GT for {info.sample_id}: {sorted(list(uniq))}")

        img_c, gt_c = _center_crop_pair(img, gt, input_size)
        x_np = img_c.astype(np.float32) / 255.0
        x = torch.from_numpy(x_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        pred_raw = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        pred_r2 = postprocess_multiclass_mask(pred_raw, preset="ring_erosion_r2")
        if not np.array_equal(pred_r2 == 1, pred_raw == 1):
            raise SystemExit(f"Leaflet changed by ring_erosion_r2 for {info.sample_id}")

        raw_m = _metric_pack(gt_c, pred_raw)
        r2_m = _metric_pack(gt_c, pred_r2)

        delta_ring = float(r2_m["dice_ring"] - raw_m["dice_ring"])
        delta_mean_fg = float(r2_m["mean_fg"] - raw_m["mean_fg"])

        sample_dir = out_root / info.quality_class / info.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(sample_dir / "original.png"), cv2.cvtColor(img_c, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sample_dir / "gt.png"), gt_c.astype(np.uint8))
        cv2.imwrite(str(sample_dir / "pred_raw.png"), pred_raw.astype(np.uint8))
        cv2.imwrite(str(sample_dir / "pred_r2.png"), pred_r2.astype(np.uint8))

        overlay_raw = _overlay_contours_rgb(img_c, gt_c, pred_raw)
        overlay_r2 = _overlay_contours_rgb(img_c, gt_c, pred_r2)
        cv2.imwrite(str(sample_dir / "overlay_raw.png"), cv2.cvtColor(overlay_raw, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sample_dir / "overlay_r2.png"), cv2.cvtColor(overlay_r2, cv2.COLOR_RGB2BGR))

        title_lines = [
            f"sample={info.sample_id}  quality={info.quality_class}  split={info.split_source}",
            f"RAW:  Dice leaf={raw_m['dice_leaflet']:.4f}  Dice ring={raw_m['dice_ring']:.4f}  mean_fg={raw_m['mean_fg']:.4f}  IoU leaf={raw_m['iou_leaflet']:.4f}  IoU ring={raw_m['iou_ring']:.4f}",
            f"R2 :  Dice leaf={r2_m['dice_leaflet']:.4f}  Dice ring={r2_m['dice_ring']:.4f}  mean_fg={r2_m['mean_fg']:.4f}  IoU leaf={r2_m['iou_leaflet']:.4f}  IoU ring={r2_m['iou_ring']:.4f}",
            f"Δ (R2-RAW):  ΔDice ring={delta_ring:+.4f}  Δmean_fg={delta_mean_fg:+.4f}",
        ]
        compare = _make_compare(
            original_rgb=img_c,
            gt_u8=gt_c,
            pred_raw_u8=pred_raw,
            pred_r2_u8=pred_r2,
            title_lines=title_lines,
            legend_lines=legend_lines,
        )
        compare_path = sample_dir / "compare.png"
        cv2.imwrite(str(compare_path), cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))

        metrics_obj = {
            "sample": info.sample_id,
            "quality_class": info.quality_class,
            "split_source": info.split_source,
            "image_path": str(info.image_path),
            "gt_path": str(info.gt_path),
            "raw": raw_m,
            "r2": r2_m,
            "delta_ring_dice": delta_ring,
            "delta_mean_fg": delta_mean_fg,
            "preset": "ring_erosion_r2",
        }
        (sample_dir / "metrics.json").write_text(json.dumps(metrics_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        row = {
            "sample": info.sample_id,
            "quality_class": info.quality_class,
            "split_source": info.split_source,
            "raw_dice_leaflet": raw_m["dice_leaflet"],
            "raw_dice_ring": raw_m["dice_ring"],
            "raw_mean_fg": raw_m["mean_fg"],
            "raw_iou_leaflet": raw_m["iou_leaflet"],
            "raw_iou_ring": raw_m["iou_ring"],
            "raw_mean_iou": raw_m["mean_iou"],
            "r2_dice_leaflet": r2_m["dice_leaflet"],
            "r2_dice_ring": r2_m["dice_ring"],
            "r2_mean_fg": r2_m["mean_fg"],
            "r2_iou_leaflet": r2_m["iou_leaflet"],
            "r2_iou_ring": r2_m["iou_ring"],
            "r2_mean_iou": r2_m["mean_iou"],
            "delta_ring_dice": delta_ring,
            "delta_mean_fg": delta_mean_fg,
            "image_path": str(info.image_path),
            "gt_path": str(info.gt_path),
            "compare_path": str(compare_path),
        }
        all_rows.append(row)
        per_item_metrics[info.sample_id] = metrics_obj

    csv_path = out_root / "quality_samples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    def _summarize(rows: list[dict]) -> dict:
        raw_leaf = [float(r["raw_dice_leaflet"]) for r in rows]
        raw_ring = [float(r["raw_dice_ring"]) for r in rows]
        raw_fg = [float(r["raw_mean_fg"]) for r in rows]
        raw_iou_leaf = [float(r["raw_iou_leaflet"]) for r in rows]
        raw_iou_ring = [float(r["raw_iou_ring"]) for r in rows]
        raw_iou_mean = [float(r["raw_mean_iou"]) for r in rows]

        r2_leaf = [float(r["r2_dice_leaflet"]) for r in rows]
        r2_ring = [float(r["r2_dice_ring"]) for r in rows]
        r2_fg = [float(r["r2_mean_fg"]) for r in rows]
        r2_iou_leaf = [float(r["r2_iou_leaflet"]) for r in rows]
        r2_iou_ring = [float(r["r2_iou_ring"]) for r in rows]
        r2_iou_mean = [float(r["r2_mean_iou"]) for r in rows]

        delta_ring = [float(r["delta_ring_dice"]) for r in rows]
        delta_fg = [float(r["delta_mean_fg"]) for r in rows]
        improved = sum(1 for d in delta_ring if d > 1e-9)
        worsened = sum(1 for d in delta_ring if d < -1e-9)
        neutral = int(len(delta_ring) - improved - worsened)

        split_counts = {"train": 0, "val": 0, "test": 0}
        for r in rows:
            s = str(r["split_source"])
            if s in split_counts:
                split_counts[s] += 1

        return {
            "count": int(len(rows)),
            "split_counts": split_counts,
            "raw": {
                "dice_leaflet": _stats(raw_leaf),
                "dice_ring": _stats(raw_ring),
                "mean_fg": _stats(raw_fg),
                "iou_leaflet": _stats(raw_iou_leaf),
                "iou_ring": _stats(raw_iou_ring),
                "mean_iou": _stats(raw_iou_mean),
            },
            "r2": {
                "dice_leaflet": _stats(r2_leaf),
                "dice_ring": _stats(r2_ring),
                "mean_fg": _stats(r2_fg),
                "iou_leaflet": _stats(r2_iou_leaf),
                "iou_ring": _stats(r2_iou_ring),
                "mean_iou": _stats(r2_iou_mean),
            },
            "delta": {
                "improved": int(improved),
                "worsened": int(worsened),
                "unchanged": int(neutral),
                "mean_delta_ring_dice": _mean(delta_ring),
                "mean_delta_mean_fg": _mean(delta_fg),
            },
        }

    by_quality = {}
    for q in ["clean", "medium", "bad"]:
        by_quality[q] = _summarize([r for r in all_rows if str(r["quality_class"]) == q])

    by_split = {}
    for s in ["train", "val", "test"]:
        by_split[s] = _summarize([r for r in all_rows if str(r["split_source"]) == s])

    quality_summary_json = out_root / "quality_summary.json"
    quality_summary_csv = out_root / "quality_summary.csv"
    split_summary_json = out_root / "split_summary.json"
    manifest_json = out_root / "selection_manifest.json"

    quality_summary_json.write_text(json.dumps(by_quality, ensure_ascii=False, indent=2), encoding="utf-8")
    split_summary_json.write_text(json.dumps(by_split, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "quality_label_source": str(curation_path),
        "expert_labels": True,
        "mapping": {"clean": "clean", "medium": "medium", "bad": "bad"},
        "requested_n_per_class": n,
        "selected_counts": {
            "clean": len(selected_clean),
            "medium": len(selected_medium),
            "bad": len(selected_bad),
        },
        "selected": [
            {
                "sample_id": i.sample_id,
                "quality_class": i.quality_class,
                "split_source": i.split_source,
                "image_path": str(i.image_path),
                "gt_path": str(i.gt_path),
            }
            for i in items
        ],
        "split_overlap_full": {
            "train_val": int(len(train_ids & val_ids)),
            "train_test": int(len(train_ids & test_ids)),
            "val_test": int(len(val_ids & test_ids)),
        },
    }
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    flat_rows = []
    for q, obj in by_quality.items():
        flat_rows.append(
            {
                "group": f"quality:{q}",
                "count": obj["count"],
                "raw_mean_fg_mean": obj["raw"]["mean_fg"]["mean"],
                "raw_dice_ring_mean": obj["raw"]["dice_ring"]["mean"],
                "raw_dice_leaflet_mean": obj["raw"]["dice_leaflet"]["mean"],
                "raw_mean_iou_mean": obj["raw"]["mean_iou"]["mean"],
                "r2_mean_fg_mean": obj["r2"]["mean_fg"]["mean"],
                "r2_dice_ring_mean": obj["r2"]["dice_ring"]["mean"],
                "r2_dice_leaflet_mean": obj["r2"]["dice_leaflet"]["mean"],
                "r2_mean_iou_mean": obj["r2"]["mean_iou"]["mean"],
                "improved": obj["delta"]["improved"],
                "worsened": obj["delta"]["worsened"],
                "unchanged": obj["delta"]["unchanged"],
                "mean_delta_ring_dice": obj["delta"]["mean_delta_ring_dice"],
                "mean_delta_mean_fg": obj["delta"]["mean_delta_mean_fg"],
                "train": obj["split_counts"]["train"],
                "val": obj["split_counts"]["val"],
                "test": obj["split_counts"]["test"],
            }
        )
    for s, obj in by_split.items():
        flat_rows.append(
            {
                "group": f"split:{s}",
                "count": obj["count"],
                "raw_mean_fg_mean": obj["raw"]["mean_fg"]["mean"],
                "raw_dice_ring_mean": obj["raw"]["dice_ring"]["mean"],
                "raw_dice_leaflet_mean": obj["raw"]["dice_leaflet"]["mean"],
                "raw_mean_iou_mean": obj["raw"]["mean_iou"]["mean"],
                "r2_mean_fg_mean": obj["r2"]["mean_fg"]["mean"],
                "r2_dice_ring_mean": obj["r2"]["dice_ring"]["mean"],
                "r2_dice_leaflet_mean": obj["r2"]["dice_leaflet"]["mean"],
                "r2_mean_iou_mean": obj["r2"]["mean_iou"]["mean"],
                "improved": obj["delta"]["improved"],
                "worsened": obj["delta"]["worsened"],
                "unchanged": obj["delta"]["unchanged"],
                "mean_delta_ring_dice": obj["delta"]["mean_delta_ring_dice"],
                "mean_delta_mean_fg": obj["delta"]["mean_delta_mean_fg"],
                "train": obj["split_counts"]["train"],
                "val": obj["split_counts"]["val"],
                "test": obj["split_counts"]["test"],
            }
        )
    with quality_summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        w.writeheader()
        for r in flat_rows:
            w.writerow(r)

    index_html = out_root / "index.html"
    cards = []
    for r in all_rows:
        rel_compare = Path(r["compare_path"]).relative_to(out_root).as_posix()
        cards.append(
            {
                "sample": r["sample"],
                "quality_class": r["quality_class"],
                "split_source": r["split_source"],
                "compare": rel_compare,
                "raw_mean_fg": float(r["raw_mean_fg"]),
                "raw_dice_ring": float(r["raw_dice_ring"]),
                "r2_mean_fg": float(r["r2_mean_fg"]),
                "r2_dice_ring": float(r["r2_dice_ring"]),
                "delta_ring": float(r["delta_ring_dice"]),
            }
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>quality_gallery_60</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    .controls {{ position: sticky; top: 0; background: #fff; padding: 8px 0; border-bottom: 1px solid #ddd; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 8px; }}
    .meta {{ font-size: 13px; color: #222; margin: 6px 0; }}
    img {{ width: 100%; height: auto; border-radius: 6px; border: 1px solid #eee; }}
    .tag {{ display: inline-block; padding: 2px 6px; border-radius: 6px; background: #f3f3f3; margin-right: 6px; }}
  </style>
</head>
<body>
  <h2>quality_gallery_60</h2>
  <div class="controls">
    <span class="tag">Filter split:</span>
    <label><input type="checkbox" id="f_train" checked> train</label>
    <label><input type="checkbox" id="f_val" checked> val</label>
    <label><input type="checkbox" id="f_test" checked> test</label>
  </div>
  <div id="root"></div>

  <script>
    const data = {json.dumps(cards, ensure_ascii=False)};
    function okSplit(s) {{
      if (s === "train") return document.getElementById("f_train").checked;
      if (s === "val") return document.getElementById("f_val").checked;
      if (s === "test") return document.getElementById("f_test").checked;
      return true;
    }}
    function render() {{
      const root = document.getElementById("root");
      root.innerHTML = "";
      for (const q of ["clean","medium","bad"]) {{
        const h = document.createElement("h3");
        h.textContent = q;
        root.appendChild(h);
        const grid = document.createElement("div");
        grid.className = "grid";
        for (const item of data.filter(x => x.quality_class === q && okSplit(x.split_source))) {{
          const c = document.createElement("div");
          c.className = "card";
          c.innerHTML = `
            <div class="meta">
              <span class="tag">${{item.sample}}</span>
              <span class="tag">split=${{item.split_source}}</span>
              <span class="tag">raw mean_fg=${{item.raw_mean_fg.toFixed(4)}}</span>
              <span class="tag">raw ring=${{item.raw_dice_ring.toFixed(4)}}</span>
              <span class="tag">r2 mean_fg=${{item.r2_mean_fg.toFixed(4)}}</span>
              <span class="tag">r2 ring=${{item.r2_dice_ring.toFixed(4)}}</span>
              <span class="tag">Δring=${{item.delta_ring.toFixed(4)}}</span>
            </div>
            <a href="${{item.compare}}" target="_blank"><img src="${{item.compare}}" loading="lazy"/></a>
          `;
          grid.appendChild(c);
        }}
        root.appendChild(grid);
      }}
    }}
    document.getElementById("f_train").addEventListener("change", render);
    document.getElementById("f_val").addEventListener("change", render);
    document.getElementById("f_test").addEventListener("change", render);
    render();
  </script>
</body>
</html>
"""
    index_html.write_text(html, encoding="utf-8")

    print(f"Selected: clean={len(selected_clean)} medium={len(selected_medium)} bad={len(selected_bad)} total={len(items)}")
    print(f"Output dir: {out_root}")
    print(f"CSV: {csv_path}")
    print(f"Quality summary: {quality_summary_json}")
    print(f"Split summary: {split_summary_json}")


if __name__ == "__main__":
    main()
