from __future__ import annotations

import argparse
import csv
import json
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


def _metric_pack(gt: np.ndarray, pred: np.ndarray) -> dict:
    dice_leaflet = _dice(pred == 1, gt == 1)
    dice_ring = _dice(pred == 2, gt == 2)
    mean_fg = float((dice_leaflet + dice_ring) / 2.0)
    iou_ring = _iou(pred == 2, gt == 2)

    gt_ring_pixels = int(np.sum(gt == 2))
    pred_ring_pixels = int(np.sum(pred == 2))
    signed_area_error = int(pred_ring_pixels - gt_ring_pixels)
    abs_area_error = int(abs(signed_area_error))
    ratio = float(pred_ring_pixels / max(gt_ring_pixels, 1))

    gt_ring_components = _count_components(gt == 2)
    pred_ring_components = _count_components(pred == 2)
    abs_component_err = int(abs(pred_ring_components - gt_ring_components))

    return {
        "dice_leaflet": float(dice_leaflet),
        "dice_ring": float(dice_ring),
        "mean_fg": float(mean_fg),
        "iou_ring": float(iou_ring),
        "area_ratio": float(ratio),
        "signed_area_error": int(signed_area_error),
        "abs_area_error": int(abs_area_error),
        "gt_ring_components": int(gt_ring_components),
        "pred_ring_components": int(pred_ring_components),
        "abs_component_err": int(abs_component_err),
        "gt_ring_pixels": int(gt_ring_pixels),
        "pred_ring_pixels": int(pred_ring_pixels),
    }


def _mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def _median(xs: list[float]) -> float | None:
    return float(np.median(np.asarray(xs, dtype=np.float64))) if xs else None


def _std(xs: list[float]) -> float | None:
    return float(np.std(np.asarray(xs, dtype=np.float64), ddof=0)) if xs else None


def _bootstrap_ci_mean(xs: list[float], *, n_boot: int = 5000, seed: int = 1337) -> dict:
    if not xs:
        return {"n": 0, "n_boot": int(n_boot), "seed": int(seed), "mean": None, "ci95": [None, None]}
    arr = np.asarray(xs, dtype=np.float64)
    n = int(arr.shape[0])
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(arr[idx]))
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return {"n": n, "n_boot": int(n_boot), "seed": int(seed), "mean": float(np.mean(arr)), "ci95": [lo, hi]}


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


def _copy_base_artifacts(src_sample_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ["image.png", "gt.png", "pred.png"]:
        src = src_sample_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def _aggregate(rows: list[dict], key: str) -> dict:
    vals = [float(r[key]) for r in rows]
    return {"mean": _mean(vals), "median": _median(vals), "std": _std(vals), "min": float(min(vals)), "max": float(max(vals))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--test-predictions-dir",
        type=Path,
        default=Path("training/analysis/test_predictions_unetpp_effb3_a100_multiclass_curated_finetune_stage2_lr1e5_100ep"),
    )
    ap.add_argument("--output-dir", type=Path, default=Path("training/postprocess_ring_erosion_r2_test"))
    ap.add_argument("--test-split-txt", type=Path, default=Path("datasets/converted_full_multiclass_curated/test.txt"))
    ap.add_argument("--bootstrap", type=int, default=5000)
    args = ap.parse_args()

    test_pred_dir = args.test_predictions_dir.resolve()
    out_root = args.output_dir.resolve()
    _ensure_dir(out_root)

    test_split_txt = args.test_split_txt.resolve()
    test_split_exists = test_split_txt.exists()

    samples = _collect_samples(test_pred_dir)

    rows: list[dict] = []
    deltas_ring: list[float] = []
    improved = 0
    worsened = 0
    unchanged = 0
    below_m01 = []
    above_p01 = []
    underprediction_cases = []

    for s in samples:
        base = _metric_pack(s.gt, s.pred)
        pp = postprocess_multiclass_mask(s.pred, ring_erosion_radius=2)
        after = _metric_pack(s.gt, pp)

        d_ring = float(after["dice_ring"] - base["dice_ring"])
        deltas_ring.append(d_ring)
        if d_ring > 1e-9:
            improved += 1
        elif d_ring < -1e-9:
            worsened += 1
        else:
            unchanged += 1
        if d_ring < -0.01:
            below_m01.append(s.sample_id)
        if d_ring > 0.01:
            above_p01.append(s.sample_id)
        if int(after["signed_area_error"]) < 0:
            underprediction_cases.append(s.sample_id)

        rows.append(
            {
                "sample_id": s.sample_id,
                "baseline_dice_leaflet": base["dice_leaflet"],
                "baseline_dice_ring": base["dice_ring"],
                "baseline_mean_fg": base["mean_fg"],
                "baseline_iou_ring": base["iou_ring"],
                "baseline_area_ratio": base["area_ratio"],
                "baseline_signed_area_error": base["signed_area_error"],
                "baseline_abs_area_error": base["abs_area_error"],
                "baseline_gt_ring_components": base["gt_ring_components"],
                "baseline_pred_ring_components": base["pred_ring_components"],
                "baseline_abs_component_err": base["abs_component_err"],
                "r2_dice_leaflet": after["dice_leaflet"],
                "r2_dice_ring": after["dice_ring"],
                "r2_mean_fg": after["mean_fg"],
                "r2_iou_ring": after["iou_ring"],
                "r2_area_ratio": after["area_ratio"],
                "r2_signed_area_error": after["signed_area_error"],
                "r2_abs_area_error": after["abs_area_error"],
                "r2_gt_ring_components": after["gt_ring_components"],
                "r2_pred_ring_components": after["pred_ring_components"],
                "r2_abs_component_err": after["abs_component_err"],
                "delta_dice_ring": d_ring,
                "gt_ring_pixels": base["gt_ring_pixels"],
                "pred_ring_pixels_baseline": base["pred_ring_pixels"],
                "pred_ring_pixels_r2": after["pred_ring_pixels"],
            }
        )

    metrics_csv = out_root / "ring_erosion_r2_test_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    baseline_rows = [
        {
            "dice_leaflet": float(r["baseline_dice_leaflet"]),
            "dice_ring": float(r["baseline_dice_ring"]),
            "mean_fg": float(r["baseline_mean_fg"]),
            "iou_ring": float(r["baseline_iou_ring"]),
            "area_ratio": float(r["baseline_area_ratio"]),
            "signed_area_error": float(r["baseline_signed_area_error"]),
            "abs_area_error": float(r["baseline_abs_area_error"]),
            "abs_component_err": float(r["baseline_abs_component_err"]),
            "gt_ring_components": float(r["baseline_gt_ring_components"]),
            "pred_ring_components": float(r["baseline_pred_ring_components"]),
        }
        for r in rows
    ]
    r2_rows = [
        {
            "dice_leaflet": float(r["r2_dice_leaflet"]),
            "dice_ring": float(r["r2_dice_ring"]),
            "mean_fg": float(r["r2_mean_fg"]),
            "iou_ring": float(r["r2_iou_ring"]),
            "area_ratio": float(r["r2_area_ratio"]),
            "signed_area_error": float(r["r2_signed_area_error"]),
            "abs_area_error": float(r["r2_abs_area_error"]),
            "abs_component_err": float(r["r2_abs_component_err"]),
            "gt_ring_components": float(r["r2_gt_ring_components"]),
            "pred_ring_components": float(r["r2_pred_ring_components"]),
        }
        for r in rows
    ]

    def _pct(rows_list: list[dict], *, key: str) -> float:
        n = len(rows_list)
        if n == 0:
            return 0.0
        return float(100.0 * sum(1 for r in rows_list if float(r[key]) > 0.0) / n)

    def _pct_under(rows_list: list[dict], *, key: str) -> float:
        n = len(rows_list)
        if n == 0:
            return 0.0
        return float(100.0 * sum(1 for r in rows_list if float(r[key]) < 0.0) / n)

    baseline_summary = {
        "leaflet_dice": _aggregate(baseline_rows, "dice_leaflet"),
        "ring_dice": _aggregate(baseline_rows, "dice_ring"),
        "mean_fg": _aggregate(baseline_rows, "mean_fg"),
        "iou_ring": _aggregate(baseline_rows, "iou_ring"),
        "area_ratio": _aggregate(baseline_rows, "area_ratio"),
        "mean_signed_area_error": _mean([float(r["signed_area_error"]) for r in baseline_rows]),
        "mean_abs_area_error": _mean([float(r["abs_area_error"]) for r in baseline_rows]),
        "pct_overprediction": _pct(baseline_rows, key="signed_area_error"),
        "pct_underprediction": _pct_under(baseline_rows, key="signed_area_error"),
        "mean_gt_ring_components": _mean([float(r["gt_ring_components"]) for r in baseline_rows]),
        "mean_pred_ring_components": _mean([float(r["pred_ring_components"]) for r in baseline_rows]),
        "mean_abs_component_err": _mean([float(r["abs_component_err"]) for r in baseline_rows]),
    }
    r2_summary = {
        "leaflet_dice": _aggregate(r2_rows, "dice_leaflet"),
        "ring_dice": _aggregate(r2_rows, "dice_ring"),
        "mean_fg": _aggregate(r2_rows, "mean_fg"),
        "iou_ring": _aggregate(r2_rows, "iou_ring"),
        "area_ratio": _aggregate(r2_rows, "area_ratio"),
        "mean_signed_area_error": _mean([float(r["signed_area_error"]) for r in r2_rows]),
        "mean_abs_area_error": _mean([float(r["abs_area_error"]) for r in r2_rows]),
        "pct_overprediction": _pct(r2_rows, key="signed_area_error"),
        "pct_underprediction": _pct_under(r2_rows, key="signed_area_error"),
        "mean_gt_ring_components": _mean([float(r["gt_ring_components"]) for r in r2_rows]),
        "mean_pred_ring_components": _mean([float(r["pred_ring_components"]) for r in r2_rows]),
        "mean_abs_component_err": _mean([float(r["abs_component_err"]) for r in r2_rows]),
    }

    delta_stats = {
        "improved": int(improved),
        "worsened": int(worsened),
        "unchanged": int(unchanged),
        "mean_delta_dice_ring": _mean(deltas_ring),
        "median_delta_dice_ring": _median(deltas_ring),
        "min_delta_dice_ring": float(min(deltas_ring)) if deltas_ring else None,
        "max_delta_dice_ring": float(max(deltas_ring)) if deltas_ring else None,
        "samples_delta_lt_-0p01": below_m01,
        "samples_delta_gt_+0p01": above_p01,
    }

    bootstrap = _bootstrap_ci_mean(deltas_ring, n_boot=int(args.bootstrap), seed=1337)

    test_split_result = {
        "status": "executed",
        "test_split_txt": str(test_split_txt),
        "test_split_exists": bool(test_split_exists),
        "predictions_dir": str(test_pred_dir),
        "count": int(len(samples)),
        "preset": "ring_erosion_r2",
        "ring_erosion_radius": 2,
    }

    summary_json = out_root / "ring_erosion_r2_test_summary.json"
    _write_json(
        summary_json,
        {
            "test_split": test_split_result,
            "baseline": baseline_summary,
            "r2": r2_summary,
            "r2_delta": delta_stats,
            "bootstrap_ci": bootstrap,
            "underprediction_cases": underprediction_cases,
            "outputs": {
                "metrics_csv": str(metrics_csv),
                "summary_json": str(summary_json),
            },
        },
    )
    _write_json(out_root / "test_split_result.json", test_split_result)

    improved_dir = out_root / "improved_10"
    worsened_dir = out_root / "worsened_10"
    area_corr_dir = out_root / "largest_area_correction_10"
    under_dir = out_root / "underprediction_cases"
    _ensure_dir(improved_dir)
    _ensure_dir(worsened_dir)
    _ensure_dir(area_corr_dir)
    _ensure_dir(under_dir)

    by_id = {s.sample_id: s for s in samples}
    rows_sorted = sorted(rows, key=lambda r: float(r["delta_dice_ring"]), reverse=True)
    top10 = rows_sorted[:10]
    bot10 = rows_sorted[-10:][::-1]

    for rank, row in enumerate(top10, start=1):
        sid = str(row["sample_id"])
        s = by_id[sid]
        pp = postprocess_multiclass_mask(s.pred, ring_erosion_radius=2)
        case_dir = improved_dir / f"{rank:02d}__{sid}__dRing_{float(row['delta_dice_ring']):+.4f}"
        _copy_base_artifacts(s.sample_dir, case_dir)
        _write_u8_png(case_dir / "pred_r2.png", pp)
        _write_rgb_png(case_dir / "overlay_raw.png", _overlay_contours_rgb(s.image_rgb, s.gt, s.pred))
        _write_rgb_png(case_dir / "overlay_r2.png", _overlay_contours_rgb(s.image_rgb, s.gt, pp))
        _write_rgb_png(case_dir / "compare.png", np.concatenate([s.image_rgb, _overlay_contours_rgb(s.image_rgb, s.gt, pp)], axis=1))
        _write_json(case_dir / "metrics.json", row)

    for rank, row in enumerate(bot10, start=1):
        sid = str(row["sample_id"])
        s = by_id[sid]
        pp = postprocess_multiclass_mask(s.pred, ring_erosion_radius=2)
        case_dir = worsened_dir / f"{rank:02d}__{sid}__dRing_{float(row['delta_dice_ring']):+.4f}"
        _copy_base_artifacts(s.sample_dir, case_dir)
        _write_u8_png(case_dir / "pred_r2.png", pp)
        _write_rgb_png(case_dir / "overlay_raw.png", _overlay_contours_rgb(s.image_rgb, s.gt, s.pred))
        _write_rgb_png(case_dir / "overlay_r2.png", _overlay_contours_rgb(s.image_rgb, s.gt, pp))
        _write_rgb_png(case_dir / "compare.png", np.concatenate([s.image_rgb, _overlay_contours_rgb(s.image_rgb, s.gt, pp)], axis=1))
        _write_json(case_dir / "metrics.json", row)

    scored = []
    for row in rows:
        err_b = float(row["baseline_signed_area_error"])
        err_a = float(row["r2_signed_area_error"])
        corr = abs(err_b) - abs(err_a)
        scored.append((float(corr), row))
    scored.sort(key=lambda x: x[0], reverse=True)
    for rank, (corr, row) in enumerate(scored[:10], start=1):
        sid = str(row["sample_id"])
        s = by_id[sid]
        pp = postprocess_multiclass_mask(s.pred, ring_erosion_radius=2)
        case_dir = area_corr_dir / f"{rank:02d}__{sid}__corr_{corr:+.0f}"
        _copy_base_artifacts(s.sample_dir, case_dir)
        _write_u8_png(case_dir / "pred_r2.png", pp)
        _write_rgb_png(case_dir / "overlay_raw.png", _overlay_contours_rgb(s.image_rgb, s.gt, s.pred))
        _write_rgb_png(case_dir / "overlay_r2.png", _overlay_contours_rgb(s.image_rgb, s.gt, pp))
        _write_rgb_png(case_dir / "compare.png", np.concatenate([s.image_rgb, _overlay_contours_rgb(s.image_rgb, s.gt, pp)], axis=1))
        _write_json(case_dir / "metrics.json", row)

    for sid in underprediction_cases:
        s = by_id[sid]
        pp = postprocess_multiclass_mask(s.pred, ring_erosion_radius=2)
        case_dir = under_dir / sid
        _copy_base_artifacts(s.sample_dir, case_dir)
        _write_u8_png(case_dir / "pred_r2.png", pp)
        _write_rgb_png(case_dir / "overlay_raw.png", _overlay_contours_rgb(s.image_rgb, s.gt, s.pred))
        _write_rgb_png(case_dir / "overlay_r2.png", _overlay_contours_rgb(s.image_rgb, s.gt, pp))
        _write_rgb_png(case_dir / "compare.png", np.concatenate([s.image_rgb, _overlay_contours_rgb(s.image_rgb, s.gt, pp)], axis=1))
        row = next((r for r in rows if str(r["sample_id"]) == sid), None)
        if row is not None:
            _write_json(case_dir / "metrics.json", row)

    print(f"Test samples: {len(samples)}")
    print(f"Output: {out_root}")
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
