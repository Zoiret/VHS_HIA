from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Sample:
    sample_id: str
    sample_dir: Path
    image_path: Path
    gt_path: Path
    pred_path: Path
    image_rgb: np.ndarray
    gt: np.ndarray
    pred: np.ndarray


def _read_u8(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(str(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def _read_rgb_u8(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


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


def _remove_small_objects(mask01: np.ndarray, min_area: int) -> np.ndarray:
    a = int(min_area)
    if a <= 0:
        return mask01
    m = (mask01.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return mask01
    out = np.zeros_like(mask01, dtype=np.uint8)
    for lab in range(1, int(num)):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= a:
            out[labels == lab] = 1
    return out


def _remove_small_holes(mask01: np.ndarray, max_hole_area: int) -> np.ndarray:
    a = int(max_hole_area)
    if a <= 0:
        return mask01
    fg = (mask01.astype(np.uint8) > 0).astype(np.uint8)
    inv = (1 - fg) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask01
    h, w = fg.shape[:2]
    out = fg.copy()
    for lab in range(1, int(num)):
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        area = int(stats[lab, cv2.CC_STAT_AREA])
        touches_border = (x == 0) or (y == 0) or (x + bw >= w) or (y + bh >= h)
        if touches_border:
            continue
        if area <= a:
            out[labels == lab] = 1
    return out.astype(np.uint8)


def _morph(mask01: np.ndarray, op: str, radius: int) -> np.ndarray:
    r = int(radius)
    if r <= 0:
        return mask01
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    src = (mask01.astype(np.uint8) * 255)
    if op == "opening":
        dst = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel, iterations=1)
    elif op == "closing":
        dst = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations=1)
    elif op == "erosion":
        dst = cv2.erode(src, kernel, iterations=1)
    elif op == "dilation":
        dst = cv2.dilate(src, kernel, iterations=1)
    else:
        raise ValueError(op)
    return (dst > 0).astype(np.uint8)


def _keep_largest_components(mask01: np.ndarray, n_keep: int) -> np.ndarray:
    n = int(n_keep)
    if n <= 0:
        return np.zeros_like(mask01, dtype=np.uint8)
    m = (mask01.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return mask01
    areas = []
    for lab in range(1, int(num)):
        areas.append((int(stats[lab, cv2.CC_STAT_AREA]), lab))
    areas.sort(reverse=True, key=lambda x: x[0])
    keep_labels = {lab for _, lab in areas[:n]}
    out = np.zeros_like(mask01, dtype=np.uint8)
    for lab in keep_labels:
        out[labels == lab] = 1
    return out


def _apply_single_op(mask01: np.ndarray, operation: str, value: int) -> np.ndarray:
    if operation == "remove_small_objects":
        return _remove_small_objects(mask01, min_area=int(value))
    if operation == "remove_small_holes":
        return _remove_small_holes(mask01, max_hole_area=int(value))
    if operation in {"opening", "closing", "erosion", "dilation"}:
        return _morph(mask01, op=operation, radius=int(value))
    if operation == "keep_largest_components":
        return _keep_largest_components(mask01, n_keep=int(value))
    raise ValueError(operation)


def _apply_on_multiclass(pred_u8: np.ndarray, *, class_id: int, operation: str, value: int) -> np.ndarray:
    out = pred_u8.copy()
    src01 = (pred_u8 == int(class_id)).astype(np.uint8)
    dst01 = _apply_single_op(src01, operation=operation, value=int(value))
    out[pred_u8 == int(class_id)] = 0
    out[dst01.astype(bool)] = int(class_id)
    return out


def _safe_div(a: float, b: float) -> float:
    if b == 0.0:
        return 0.0
    return float(a / b)


def _metric_pack(gt: np.ndarray, pred: np.ndarray) -> dict:
    dice_leaflet = _dice(pred == 1, gt == 1)
    dice_ring = _dice(pred == 2, gt == 2)
    mean_fg = float((dice_leaflet + dice_ring) / 2.0)
    iou_leaflet = _iou(pred == 1, gt == 1)
    iou_ring = _iou(pred == 2, gt == 2)

    gt_leaflet_components = _count_components(gt == 1)
    pred_leaflet_components = _count_components(pred == 1)
    gt_ring_components = _count_components(gt == 2)
    pred_ring_components = _count_components(pred == 2)

    leaflet_pixels_gt = int(np.sum(gt == 1))
    leaflet_pixels_pred = int(np.sum(pred == 1))
    ring_pixels_gt = int(np.sum(gt == 2))
    ring_pixels_pred = int(np.sum(pred == 2))

    leaflet_area_delta_rel = _safe_div(float(leaflet_pixels_pred - leaflet_pixels_gt), float(max(leaflet_pixels_gt, 1)))
    ring_area_delta_rel = _safe_div(float(ring_pixels_pred - ring_pixels_gt), float(max(ring_pixels_gt, 1)))

    return {
        "dice_leaflet": float(dice_leaflet),
        "dice_ring": float(dice_ring),
        "mean_fg": float(mean_fg),
        "iou_leaflet": float(iou_leaflet),
        "iou_ring": float(iou_ring),
        "gt_leaflet_components": int(gt_leaflet_components),
        "pred_leaflet_components": int(pred_leaflet_components),
        "gt_ring_components": int(gt_ring_components),
        "pred_ring_components": int(pred_ring_components),
        "leaflet_pixels_gt": int(leaflet_pixels_gt),
        "leaflet_pixels_pred": int(leaflet_pixels_pred),
        "ring_pixels_gt": int(ring_pixels_gt),
        "ring_pixels_pred": int(ring_pixels_pred),
        "leaflet_area_delta_rel": float(leaflet_area_delta_rel),
        "ring_area_delta_rel": float(ring_area_delta_rel),
    }


def _mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def _median(xs: list[float]) -> float | None:
    return float(np.median(np.asarray(xs, dtype=np.float64))) if xs else None


def _std(xs: list[float]) -> float | None:
    return float(np.std(np.asarray(xs, dtype=np.float64), ddof=0)) if xs else None


def _collect_samples(predictions_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for d in sorted(predictions_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name == "worst_cases":
            continue
        pred_path = d / "pred.png"
        gt_path = d / "gt.png"
        image_path = d / "image.png"
        if not (pred_path.exists() and gt_path.exists() and image_path.exists()):
            continue
        pred = _read_u8(pred_path)
        gt = _read_u8(gt_path)
        if pred.shape != gt.shape:
            raise SystemExit(f"Shape mismatch for {d.name}: pred={pred.shape} gt={gt.shape}")
        img = _read_rgb_u8(image_path)
        if img.shape[:2] != pred.shape[:2]:
            raise SystemExit(f"Image shape mismatch for {d.name}: image={img.shape} pred={pred.shape}")
        samples.append(
            Sample(
                sample_id=d.name,
                sample_dir=d,
                image_path=image_path,
                gt_path=gt_path,
                pred_path=pred_path,
                image_rgb=img,
                gt=gt,
                pred=pred,
            )
        )
    if not samples:
        raise SystemExit(f"No samples found in: {predictions_dir}")
    return samples


def _write_overlay(out_path: Path, image_rgb: np.ndarray, gt_u8: np.ndarray, pred_u8: np.ndarray) -> None:
    overlay = _overlay_contours_rgb(image_rgb, gt_u8, pred_u8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def _copy_image_gt(out_dir: Path, sample: Sample) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.image_path, out_dir / "image.png")
    shutil.copy2(sample.gt_path, out_dir / "gt.png")


def _write_pred(out_path: Path, pred_u8: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), pred_u8.astype(np.uint8))


def _format_value(operation: str, value: int) -> str:
    if operation in {"opening", "closing", "erosion", "dilation"}:
        return f"r={int(value)}"
    if operation in {"remove_small_objects", "remove_small_holes"}:
        return f"area={int(value)}"
    if operation == "keep_largest_components":
        return f"n={int(value)}"
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("training/analysis/val_predictions_unetpp_effb3_a100_multiclass_curated_finetune_stage2_lr1e5_100ep"),
    )
    ap.add_argument("--output-dir", type=Path, default=Path("training/postprocess_sensitivity"))
    args = ap.parse_args()

    predictions_dir = args.predictions_dir.resolve()
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    samples = _collect_samples(predictions_dir)

    baseline_by_id = {}
    for s in samples:
        baseline_by_id[s.sample_id] = _metric_pack(s.gt, s.pred)

    operation_space = [
        ("remove_small_objects", [0, 20, 50, 100, 200, 300, 500]),
        ("remove_small_holes", [0, 20, 50, 100, 200, 300, 500]),
        ("opening", [0, 1, 2, 3]),
        ("closing", [0, 1, 2, 3, 4]),
        ("erosion", [0, 1, 2, 3]),
        ("dilation", [0, 1, 2, 3]),
        ("keep_largest_components", [1, 2, 3]),
    ]

    class_targets = [
        ("leaflet", 1),
        ("ring", 2),
    ]

    results_rows = []
    summary_rows = []

    best_setting_per_op = {}

    for class_name, class_id in class_targets:
        for op_name, values in operation_space:
            best_mean_target = -math.inf
            best_value = None
            best_variant_by_id = None
            best_metrics_agg = None

            for v in values:
                deltas_target = []
                deltas_mean_fg = []

                dice_leaflet_vals = []
                dice_ring_vals = []
                mean_fg_vals = []
                iou_leaflet_vals = []
                iou_ring_vals = []

                gt_leaflet_components_vals = []
                pred_leaflet_components_vals = []
                gt_ring_components_vals = []
                pred_ring_components_vals = []

                leaflet_pixels_gt_vals = []
                leaflet_pixels_pred_vals = []
                ring_pixels_gt_vals = []
                ring_pixels_pred_vals = []

                leaflet_area_delta_rel_vals = []
                ring_area_delta_rel_vals = []

                variant_by_id = {}

                for s in samples:
                    pred2 = _apply_on_multiclass(s.pred, class_id=class_id, operation=op_name, value=int(v))
                    variant_by_id[s.sample_id] = pred2
                    m = _metric_pack(s.gt, pred2)

                    base = baseline_by_id[s.sample_id]
                    delta_mean_fg = float(m["mean_fg"] - base["mean_fg"])
                    delta_target = float(
                        m["dice_leaflet"] - base["dice_leaflet"] if class_id == 1 else m["dice_ring"] - base["dice_ring"]
                    )
                    deltas_mean_fg.append(delta_mean_fg)
                    deltas_target.append(delta_target)

                    results_rows.append(
                        {
                            "target_class": class_name,
                            "operation": op_name,
                            "value": int(v),
                            "value_label": _format_value(op_name, int(v)),
                            "sample_id": s.sample_id,
                            "dice_leaflet": m["dice_leaflet"],
                            "dice_ring": m["dice_ring"],
                            "mean_fg": m["mean_fg"],
                            "iou_leaflet": m["iou_leaflet"],
                            "iou_ring": m["iou_ring"],
                            "gt_leaflet_components": m["gt_leaflet_components"],
                            "pred_leaflet_components": m["pred_leaflet_components"],
                            "gt_ring_components": m["gt_ring_components"],
                            "pred_ring_components": m["pred_ring_components"],
                            "leaflet_pixels_gt": m["leaflet_pixels_gt"],
                            "leaflet_pixels_pred": m["leaflet_pixels_pred"],
                            "ring_pixels_gt": m["ring_pixels_gt"],
                            "ring_pixels_pred": m["ring_pixels_pred"],
                            "leaflet_area_delta_rel": m["leaflet_area_delta_rel"],
                            "ring_area_delta_rel": m["ring_area_delta_rel"],
                            "delta_dice_leaflet": float(m["dice_leaflet"] - base["dice_leaflet"]),
                            "delta_dice_ring": float(m["dice_ring"] - base["dice_ring"]),
                            "delta_mean_fg": delta_mean_fg,
                        }
                    )

                    dice_leaflet_vals.append(float(m["dice_leaflet"]))
                    dice_ring_vals.append(float(m["dice_ring"]))
                    mean_fg_vals.append(float(m["mean_fg"]))
                    iou_leaflet_vals.append(float(m["iou_leaflet"]))
                    iou_ring_vals.append(float(m["iou_ring"]))

                    gt_leaflet_components_vals.append(float(m["gt_leaflet_components"]))
                    pred_leaflet_components_vals.append(float(m["pred_leaflet_components"]))
                    gt_ring_components_vals.append(float(m["gt_ring_components"]))
                    pred_ring_components_vals.append(float(m["pred_ring_components"]))

                    leaflet_pixels_gt_vals.append(float(m["leaflet_pixels_gt"]))
                    leaflet_pixels_pred_vals.append(float(m["leaflet_pixels_pred"]))
                    ring_pixels_gt_vals.append(float(m["ring_pixels_gt"]))
                    ring_pixels_pred_vals.append(float(m["ring_pixels_pred"]))

                    leaflet_area_delta_rel_vals.append(float(m["leaflet_area_delta_rel"]))
                    ring_area_delta_rel_vals.append(float(m["ring_area_delta_rel"]))

                improved_mean_fg = sum(1 for d in deltas_mean_fg if d > 1e-9)
                worsened_mean_fg = sum(1 for d in deltas_mean_fg if d < -1e-9)
                improved_target = sum(1 for d in deltas_target if d > 1e-9)
                worsened_target = sum(1 for d in deltas_target if d < -1e-9)

                mean_delta_mean_fg = _mean([float(x) for x in deltas_mean_fg]) or 0.0
                mean_delta_target = _mean([float(x) for x in deltas_target]) or 0.0
                max_improve_mean_fg = max(deltas_mean_fg) if deltas_mean_fg else 0.0
                max_worsen_mean_fg = min(deltas_mean_fg) if deltas_mean_fg else 0.0
                max_improve_target = max(deltas_target) if deltas_target else 0.0
                max_worsen_target = min(deltas_target) if deltas_target else 0.0

                mean_dice_leaflet = _mean(dice_leaflet_vals) or 0.0
                mean_dice_ring = _mean(dice_ring_vals) or 0.0
                mean_mean_fg = _mean(mean_fg_vals) or 0.0

                mean_target = float(mean_dice_leaflet if class_id == 1 else mean_dice_ring)
                if mean_target > best_mean_target:
                    best_mean_target = mean_target
                    best_value = int(v)
                    best_variant_by_id = variant_by_id
                    best_metrics_agg = {
                        "mean_dice_leaflet": mean_dice_leaflet,
                        "mean_dice_ring": mean_dice_ring,
                        "mean_mean_fg": mean_mean_fg,
                        "mean_iou_leaflet": _mean(iou_leaflet_vals) or 0.0,
                        "mean_iou_ring": _mean(iou_ring_vals) or 0.0,
                    }

                summary_rows.append(
                    {
                        "target_class": class_name,
                        "operation": op_name,
                        "value": int(v),
                        "value_label": _format_value(op_name, int(v)),
                        "val_count": int(len(samples)),
                        "mean_dice_leaflet": mean_dice_leaflet,
                        "mean_dice_ring": mean_dice_ring,
                        "mean_mean_fg": mean_mean_fg,
                        "mean_iou_leaflet": _mean(iou_leaflet_vals),
                        "mean_iou_ring": _mean(iou_ring_vals),
                        "mean_gt_leaflet_components": _mean(gt_leaflet_components_vals),
                        "mean_pred_leaflet_components": _mean(pred_leaflet_components_vals),
                        "mean_gt_ring_components": _mean(gt_ring_components_vals),
                        "mean_pred_ring_components": _mean(pred_ring_components_vals),
                        "mean_leaflet_pixels_gt": _mean(leaflet_pixels_gt_vals),
                        "mean_leaflet_pixels_pred": _mean(leaflet_pixels_pred_vals),
                        "mean_ring_pixels_gt": _mean(ring_pixels_gt_vals),
                        "mean_ring_pixels_pred": _mean(ring_pixels_pred_vals),
                        "mean_leaflet_area_delta_rel": _mean(leaflet_area_delta_rel_vals),
                        "mean_ring_area_delta_rel": _mean(ring_area_delta_rel_vals),
                        "improved_mean_fg_count": int(improved_mean_fg),
                        "worsened_mean_fg_count": int(worsened_mean_fg),
                        "mean_delta_mean_fg": float(mean_delta_mean_fg),
                        "max_improve_mean_fg": float(max_improve_mean_fg),
                        "max_worsen_mean_fg": float(max_worsen_mean_fg),
                        "improved_target_dice_count": int(improved_target),
                        "worsened_target_dice_count": int(worsened_target),
                        "mean_delta_target_dice": float(mean_delta_target),
                        "max_improve_target_dice": float(max_improve_target),
                        "max_worsen_target_dice": float(max_worsen_target),
                    }
                )

            best_setting_per_op[f"{class_name}::{op_name}"] = {
                "target_class": class_name,
                "class_id": int(class_id),
                "operation": op_name,
                "best_value": int(best_value) if best_value is not None else None,
                "best_value_label": _format_value(op_name, int(best_value)) if best_value is not None else None,
                "metrics": best_metrics_agg,
            }

            if best_value is None or best_variant_by_id is None:
                continue

            op_vis_root = out_root / "visuals" / class_name / op_name / _format_value(op_name, int(best_value))
            best_dir = op_vis_root / "best_10"
            worst_dir = op_vis_root / "worst_10"
            best_dir.mkdir(parents=True, exist_ok=True)
            worst_dir.mkdir(parents=True, exist_ok=True)

            deltas = []
            for s in samples:
                pred2 = best_variant_by_id[s.sample_id]
                m2 = _metric_pack(s.gt, pred2)
                base = baseline_by_id[s.sample_id]
                delta = float(m2["dice_leaflet"] - base["dice_leaflet"]) if class_id == 1 else float(m2["dice_ring"] - base["dice_ring"])
                deltas.append((delta, s))
            deltas.sort(key=lambda x: x[0], reverse=True)
            top = deltas[:10]
            bot = deltas[-10:][::-1]

            for rank, (delta, s) in enumerate(top, start=1):
                pred2 = best_variant_by_id[s.sample_id]
                case_dir = best_dir / f"{rank:02d}__{s.sample_id}__delta_{delta:+.4f}"
                _copy_image_gt(case_dir, s)
                _write_pred(case_dir / "pred.png", pred2)
                _write_overlay(case_dir / "overlay.png", s.image_rgb, s.gt, pred2)

            for rank, (delta, s) in enumerate(bot, start=1):
                pred2 = best_variant_by_id[s.sample_id]
                case_dir = worst_dir / f"{rank:02d}__{s.sample_id}__delta_{delta:+.4f}"
                _copy_image_gt(case_dir, s)
                _write_pred(case_dir / "pred.png", pred2)
                _write_overlay(case_dir / "overlay.png", s.image_rgb, s.gt, pred2)

    results_path = out_root / "operation_results.csv"
    summary_path = out_root / "operation_summary.csv"
    best_path = out_root / "best_operation_per_metric.json"

    if results_rows:
        with results_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
            w.writeheader()
            for r in results_rows:
                w.writerow(r)

    if summary_rows:
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)

    def _best_by(metric_name: str):
        best = None
        best_v = -math.inf
        for r in summary_rows:
            v = r.get(metric_name, None)
            if v is None:
                continue
            fv = float(v)
            if fv > best_v:
                best_v = fv
                best = r
        return best

    best_leaflet = _best_by("mean_dice_leaflet")
    best_ring = _best_by("mean_dice_ring")
    best_mean_fg = _best_by("mean_mean_fg")

    out = {
        "predictions_dir": str(predictions_dir),
        "output_dir": str(out_root),
        "val_count": int(len(samples)),
        "best_by_metric": {
            "mean_dice_leaflet": best_leaflet,
            "mean_dice_ring": best_ring,
            "mean_mean_fg": best_mean_fg,
        },
        "best_setting_per_operation": best_setting_per_op,
    }
    best_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Samples: {len(samples)}")
    print(f"Operation variants: {len(summary_rows)}")
    print(f"Output: {out_root}")
    print(f"Summary CSV: {summary_path}")
    print(f"Results CSV: {results_path}")
    print(f"Best JSON: {best_path}")


if __name__ == "__main__":
    main()
