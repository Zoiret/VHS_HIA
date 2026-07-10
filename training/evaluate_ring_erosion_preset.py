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

from postprocess import postprocess_multiclass_mask


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


def _metric_pack(gt: np.ndarray, pred: np.ndarray) -> dict:
    dice_leaflet = _dice(pred == 1, gt == 1)
    dice_ring = _dice(pred == 2, gt == 2)
    mean_fg = float((dice_leaflet + dice_ring) / 2.0)
    iou_ring = _iou(pred == 2, gt == 2)

    ring_pixels_gt = int(np.sum(gt == 2))
    ring_pixels_pred = int(np.sum(pred == 2))
    ring_area_error = int(ring_pixels_pred - ring_pixels_gt)

    gt_ring_components = _count_components(gt == 2)
    pred_ring_components = _count_components(pred == 2)

    return {
        "dice_leaflet": float(dice_leaflet),
        "dice_ring": float(dice_ring),
        "mean_fg": float(mean_fg),
        "iou_ring": float(iou_ring),
        "ring_pixels_gt": int(ring_pixels_gt),
        "ring_pixels_pred": int(ring_pixels_pred),
        "ring_area_error": int(ring_area_error),
        "gt_ring_components": int(gt_ring_components),
        "pred_ring_components": int(pred_ring_components),
    }


def _mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def _median(xs: list[float]) -> float | None:
    return float(np.median(np.asarray(xs, dtype=np.float64))) if xs else None


def _std(xs: list[float]) -> float | None:
    return float(np.std(np.asarray(xs, dtype=np.float64), ddof=0)) if xs else None


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs, dtype=np.float64), float(p)))


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


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_u8_png(path: Path, mask_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask_u8.astype(np.uint8))


def _write_rgb_png(path: Path, image_rgb_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb_u8, cv2.COLOR_RGB2BGR))


def _copy_case_artifacts(src_sample_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ["image.png", "gt.png", "pred.png"]:
        src = src_sample_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--val-predictions-dir",
        type=Path,
        default=Path("training/analysis/val_predictions_unetpp_effb3_a100_multiclass_curated_finetune_stage2_lr1e5_100ep"),
    )
    ap.add_argument("--output-dir", type=Path, default=Path("training/postprocess_ring_erosion_r2"))
    ap.add_argument("--curated-split-dir", type=Path, default=Path("datasets/converted_full_multiclass_curated"))
    ap.add_argument("--run-test-if-available", action="store_true")
    args = ap.parse_args()

    val_pred_dir = args.val_predictions_dir.resolve()
    out_root = args.output_dir.resolve()
    _ensure_dir(out_root)

    curated_split_dir = args.curated_split_dir.resolve()
    train_txt = curated_split_dir / "train.txt"
    val_txt = curated_split_dir / "val.txt"
    test_txt = curated_split_dir / "test.txt"

    split_info = {"val": str(val_txt), "independent_test": None, "overlap_checked": False}
    if train_txt.exists() and val_txt.exists():
        train_ids = _read_split_ids(train_txt)
        val_ids = _read_split_ids(val_txt)
        test_ids = _read_split_ids(test_txt) if test_txt.exists() else set()
        split_info["independent_test"] = str(test_txt) if test_txt.exists() else None
        split_info["overlap_checked"] = True
        split_info["overlap"] = {
            "train_val": len(train_ids & val_ids),
            "train_test": len(train_ids & test_ids),
            "val_test": len(val_ids & test_ids),
        }
        split_info["counts"] = {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}

    samples = _collect_samples(val_pred_dir)

    radii = [0, 1, 2, 3]
    per_sample_rows: list[dict] = []
    per_radius = {}

    for r in radii:
        deltas: list[float] = []
        delta_mean_fg: list[float] = []
        ratios: list[float] = []
        abs_area_err: list[float] = []
        signed_area_err: list[float] = []
        over_count = 0
        under_count = 0
        comp_diff_before: list[int] = []
        comp_diff_after: list[int] = []
        removed_component_cases: list[str] = []
        reduced_components_cases: list[str] = []
        farther_from_gt_cases: list[str] = []
        improve_001 = []
        worsen_001 = []

        for s in samples:
            base = _metric_pack(s.gt, s.pred)
            pp = postprocess_multiclass_mask(s.pred, ring_erosion_radius=int(r))
            after = _metric_pack(s.gt, pp)

            gt_ring_pixels = int(base["ring_pixels_gt"])
            pred_before = int(base["ring_pixels_pred"])
            pred_after = int(after["ring_pixels_pred"])
            err_before = int(base["ring_area_error"])
            err_after = int(after["ring_area_error"])

            ratio = float(pred_after / max(gt_ring_pixels, 1))
            ratios.append(ratio)
            abs_area_err.append(float(abs(err_after)))
            signed_area_err.append(float(err_after))
            if err_after > 0:
                over_count += 1
            elif err_after < 0:
                under_count += 1

            d_ring = float(after["dice_ring"] - base["dice_ring"])
            d_fg = float(after["mean_fg"] - base["mean_fg"])
            deltas.append(d_ring)
            delta_mean_fg.append(d_fg)

            gt_c = int(base["gt_ring_components"])
            before_c = int(base["pred_ring_components"])
            after_c = int(after["pred_ring_components"])
            comp_diff_before.append(abs(before_c - gt_c))
            comp_diff_after.append(abs(after_c - gt_c))

            if pred_before > 0 and pred_after == 0 and gt_c > 0:
                removed_component_cases.append(s.sample_id)
            if after_c < before_c:
                reduced_components_cases.append(s.sample_id)
            if abs(after_c - gt_c) > abs(before_c - gt_c):
                farther_from_gt_cases.append(s.sample_id)

            if d_ring > 0.01:
                improve_001.append(s.sample_id)
            if d_ring < -0.01:
                worsen_001.append(s.sample_id)

            per_sample_rows.append(
                {
                    "radius": int(r),
                    "sample_id": s.sample_id,
                    "dice_leaflet": float(after["dice_leaflet"]),
                    "dice_ring": float(after["dice_ring"]),
                    "mean_fg": float(after["mean_fg"]),
                    "iou_ring": float(after["iou_ring"]),
                    "ring_pixels_gt": gt_ring_pixels,
                    "ring_pixels_pred_before": pred_before,
                    "ring_pixels_pred_after": pred_after,
                    "ring_area_error_before": err_before,
                    "ring_area_error_after": err_after,
                    "gt_ring_components": gt_c,
                    "pred_ring_components_before": before_c,
                    "pred_ring_components_after": after_c,
                    "delta_dice_ring": d_ring,
                    "delta_mean_fg": d_fg,
                }
            )

        improved = sum(1 for d in deltas if d > 1e-9)
        worsened = sum(1 for d in deltas if d < -1e-9)
        neutral = int(len(deltas) - improved - worsened)

        per_radius[str(r)] = {
            "radius": int(r),
            "count": int(len(samples)),
            "improved": int(improved),
            "worsened": int(worsened),
            "neutral": int(neutral),
            "delta_dice_ring": {
                "mean": _mean(deltas),
                "median": _median(deltas),
                "std": _std(deltas),
                "min": float(min(deltas)) if deltas else None,
                "max": float(max(deltas)) if deltas else None,
                "p10": _percentile(deltas, 10),
                "p90": _percentile(deltas, 90),
            },
            "delta_mean_fg": {
                "mean": _mean(delta_mean_fg),
                "median": _median(delta_mean_fg),
                "std": _std(delta_mean_fg),
                "min": float(min(delta_mean_fg)) if delta_mean_fg else None,
                "max": float(max(delta_mean_fg)) if delta_mean_fg else None,
                "p10": _percentile(delta_mean_fg, 10),
                "p90": _percentile(delta_mean_fg, 90),
            },
            "area_bias": {
                "mean_ratio_pred_to_gt": _mean(ratios),
                "median_ratio_pred_to_gt": _median(ratios),
                "mean_abs_area_error": _mean(abs_area_err),
                "mean_signed_area_error": _mean(signed_area_err),
                "pct_overprediction": float(100.0 * over_count / len(samples)) if samples else None,
                "pct_underprediction": float(100.0 * under_count / len(samples)) if samples else None,
            },
            "component_error": {
                "mean_abs_error_before": _mean([float(x) for x in comp_diff_before]),
                "mean_abs_error_after": _mean([float(x) for x in comp_diff_after]),
            },
            "flags": {
                "worsened_gt_0p01": worsen_001,
                "improved_gt_0p01": improve_001,
                "ring_component_removed": removed_component_cases,
                "ring_components_reduced": reduced_components_cases,
                "ring_components_farther_from_gt": farther_from_gt_cases,
            },
        }

    csv_path = out_root / "ring_erosion_robustness.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
        w.writeheader()
        for r in per_sample_rows:
            w.writerow(r)

    summary_path = out_root / "ring_erosion_summary.json"
    _write_json(
        summary_path,
        {
            "val_predictions_dir": str(val_pred_dir),
            "output_dir": str(out_root),
            "splits": split_info,
            "radii": radii,
            "per_radius": per_radius,
        },
    )

    test_result_path = out_root / "test_split_result.json"
    if test_txt.exists() and args.run_test_if_available:
        _write_json(
            test_result_path,
            {
                "status": "not_executed",
                "reason": "test split exists but raw predictions are not available in this workspace; export test predictions first",
                "test_split": str(test_txt),
            },
        )
    else:
        _write_json(
            test_result_path,
            {
                "status": "absent" if not test_txt.exists() else "skipped",
                "test_split": str(test_txt) if test_txt.exists() else None,
            },
        )

    r2_root = out_root
    improved_dir = r2_root / "improved_10"
    worsened_dir = r2_root / "worsened_10"
    area_corr_dir = r2_root / "largest_area_correction_10"
    comp_loss_dir = r2_root / "component_loss_cases"
    _ensure_dir(improved_dir)
    _ensure_dir(worsened_dir)
    _ensure_dir(area_corr_dir)
    _ensure_dir(comp_loss_dir)

    per_sample_r2 = [r for r in per_sample_rows if int(r["radius"]) == 2]
    per_sample_r2.sort(key=lambda x: float(x["delta_dice_ring"]), reverse=True)
    top10 = per_sample_r2[:10]
    bot10 = per_sample_r2[-10:][::-1]

    by_id = {s.sample_id: s for s in samples}
    for rank, row in enumerate(top10, start=1):
        sid = str(row["sample_id"])
        s = by_id[sid]
        pp = postprocess_multiclass_mask(s.pred, ring_erosion_radius=2)
        case_dir = improved_dir / f"{rank:02d}__{sid}__dRing_{float(row['delta_dice_ring']):+.4f}"
        _copy_case_artifacts(s.sample_dir, case_dir)
        _write_u8_png(case_dir / "pred_post.png", pp)
        _write_rgb_png(case_dir / "overlay_raw.png", _overlay_contours_rgb(s.image_rgb, s.gt, s.pred))
        _write_rgb_png(case_dir / "overlay_post.png", _overlay_contours_rgb(s.image_rgb, s.gt, pp))
        _write_rgb_png(case_dir / "compare.png", np.concatenate([s.image_rgb, _overlay_contours_rgb(s.image_rgb, s.gt, pp)], axis=1))
        _write_json(case_dir / "metrics.json", row)

    for rank, row in enumerate(bot10, start=1):
        sid = str(row["sample_id"])
        s = by_id[sid]
        pp = postprocess_multiclass_mask(s.pred, ring_erosion_radius=2)
        case_dir = worsened_dir / f"{rank:02d}__{sid}__dRing_{float(row['delta_dice_ring']):+.4f}"
        _copy_case_artifacts(s.sample_dir, case_dir)
        _write_u8_png(case_dir / "pred_post.png", pp)
        _write_rgb_png(case_dir / "overlay_raw.png", _overlay_contours_rgb(s.image_rgb, s.gt, s.pred))
        _write_rgb_png(case_dir / "overlay_post.png", _overlay_contours_rgb(s.image_rgb, s.gt, pp))
        _write_rgb_png(case_dir / "compare.png", np.concatenate([s.image_rgb, _overlay_contours_rgb(s.image_rgb, s.gt, pp)], axis=1))
        _write_json(case_dir / "metrics.json", row)

    scored = []
    for row in per_sample_r2:
        err_b = float(row["ring_area_error_before"])
        err_a = float(row["ring_area_error_after"])
        corr = abs(err_b) - abs(err_a)
        scored.append((float(corr), row))
    scored.sort(key=lambda x: x[0], reverse=True)
    for rank, (corr, row) in enumerate(scored[:10], start=1):
        sid = str(row["sample_id"])
        s = by_id[sid]
        pp = postprocess_multiclass_mask(s.pred, ring_erosion_radius=2)
        case_dir = area_corr_dir / f"{rank:02d}__{sid}__corr_{corr:+.0f}"
        _copy_case_artifacts(s.sample_dir, case_dir)
        _write_u8_png(case_dir / "pred_post.png", pp)
        _write_rgb_png(case_dir / "overlay_raw.png", _overlay_contours_rgb(s.image_rgb, s.gt, s.pred))
        _write_rgb_png(case_dir / "overlay_post.png", _overlay_contours_rgb(s.image_rgb, s.gt, pp))
        _write_json(case_dir / "metrics.json", row)

    comp_loss = [
        row
        for row in per_sample_r2
        if int(row["ring_pixels_pred_before"]) > 0 and int(row["ring_pixels_pred_after"]) == 0 and int(row["gt_ring_components"]) > 0
    ]
    for rank, row in enumerate(comp_loss, start=1):
        sid = str(row["sample_id"])
        s = by_id[sid]
        pp = postprocess_multiclass_mask(s.pred, ring_erosion_radius=2)
        case_dir = comp_loss_dir / f"{rank:02d}__{sid}"
        _copy_case_artifacts(s.sample_dir, case_dir)
        _write_u8_png(case_dir / "pred_post.png", pp)
        _write_rgb_png(case_dir / "overlay_raw.png", _overlay_contours_rgb(s.image_rgb, s.gt, s.pred))
        _write_rgb_png(case_dir / "overlay_post.png", _overlay_contours_rgb(s.image_rgb, s.gt, pp))
        _write_json(case_dir / "metrics.json", row)

    print(f"Val samples: {len(samples)}")
    print(f"Output: {out_root}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Visual report: {r2_root}")


if __name__ == "__main__":
    main()

