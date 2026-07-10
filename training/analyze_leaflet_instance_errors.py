from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PredSample:
    sample_id: str
    split: str
    sample_dir: Path
    image_path: Path
    gt_path: Path
    pred_path: Path


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


def _label_cc(mask01: np.ndarray) -> tuple[np.ndarray, list[int]]:
    m = (mask01.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = []
    for lab in range(1, int(num)):
        areas.append(int(stats[lab, cv2.CC_STAT_AREA]))
    return labels.astype(np.int32), areas


def _count_cc(mask01: np.ndarray) -> int:
    m = (mask01.astype(np.uint8) * 255)
    num, _, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return max(0, int(num) - 1)


def _largest_areas(areas: list[int], k: int = 5) -> list[int]:
    return sorted([int(a) for a in areas], reverse=True)[: int(k)]


def _pair_iou_matrix(gt_labels: np.ndarray, pred_labels: np.ndarray, gt_k: int, pred_k: int) -> np.ndarray:
    m = np.zeros((int(gt_k), int(pred_k)), dtype=np.float64)
    if gt_k == 0 or pred_k == 0:
        return m
    for gi in range(1, int(gt_k) + 1):
        g = gt_labels == gi
        g_sum = float(np.sum(g))
        if g_sum <= 0:
            continue
        for pi in range(1, int(pred_k) + 1):
            p = pred_labels == pi
            p_sum = float(np.sum(p))
            if p_sum <= 0:
                continue
            inter = float(np.sum(g & p))
            if inter <= 0:
                continue
            union = g_sum + p_sum - inter
            m[gi - 1, pi - 1] = inter / max(union, 1.0)
    return m


def _significant_overlaps_counts(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    gt_k: int,
    pred_k: int,
    *,
    min_intersection_over_gt: float = 0.10,
    min_intersection_over_pred: float = 0.10,
) -> tuple[list[int], list[int]]:
    gt_counts = [0 for _ in range(int(pred_k))]
    pred_counts = [0 for _ in range(int(gt_k))]
    if gt_k == 0 or pred_k == 0:
        return gt_counts, pred_counts

    gt_areas = [float(np.sum(gt_labels == (i + 1))) for i in range(int(gt_k))]
    pred_areas = [float(np.sum(pred_labels == (i + 1))) for i in range(int(pred_k))]
    for gi in range(int(gt_k)):
        g = gt_labels == (gi + 1)
        if gt_areas[gi] <= 0:
            continue
        for pi in range(int(pred_k)):
            p = pred_labels == (pi + 1)
            if pred_areas[pi] <= 0:
                continue
            inter = float(np.sum(g & p))
            if inter <= 0:
                continue
            if (inter / max(gt_areas[gi], 1.0) >= float(min_intersection_over_gt)) or (
                inter / max(pred_areas[pi], 1.0) >= float(min_intersection_over_pred)
            ):
                gt_counts[pi] += 1
                pred_counts[gi] += 1
    return gt_counts, pred_counts


def _perm_assignment_max_iou(iou: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    gt_k, pred_k = iou.shape[0], iou.shape[1]
    if gt_k == 0 or pred_k == 0:
        return 0.0, []
    k = min(int(gt_k), int(pred_k))
    best = -1.0
    best_pairs: list[tuple[int, int]] = []
    for cols in itertools.permutations(range(int(pred_k)), k):
        s = 0.0
        pairs = []
        for r, c in enumerate(cols):
            s += float(iou[r, c])
            pairs.append((r, c))
        if s > best:
            best = s
            best_pairs = pairs
    return float(best), best_pairs


def _palette_rgb(n: int) -> list[tuple[int, int, int]]:
    base = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
        (0, 128, 128),
        (230, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (255, 215, 180),
        (0, 0, 128),
        (128, 128, 128),
    ]
    if n <= len(base):
        return base[:n]
    out = []
    for i in range(n):
        c = base[i % len(base)]
        k = 1 + (i // len(base))
        out.append((max(0, c[0] - 15 * k), max(0, c[1] - 10 * k), max(0, c[2] - 5 * k)))
    return out


def _colorize_labels(labels: np.ndarray, k: int) -> np.ndarray:
    h, w = labels.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    colors = _palette_rgb(int(k))
    for i in range(1, int(k) + 1):
        out[labels == i] = colors[i - 1]
    return out


def _find_contours(mask01_u8: np.ndarray):
    res = cv2.findContours(mask01_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 2:
        contours, hierarchy = res
        return contours, hierarchy
    _, contours, hierarchy = res
    return contours, hierarchy


def _draw_contours_rgb(image_rgb_u8: np.ndarray, mask01: np.ndarray, *, color_rgb, thickness: int) -> None:
    m = (mask01.astype(np.uint8) * 255)
    if not np.any(m):
        return
    contours, _ = _find_contours(m)
    if not contours:
        return
    cv2.drawContours(image_rgb_u8, contours, -1, tuple(int(x) for x in color_rgb), int(thickness))


def _overlay(image_rgb: np.ndarray, gt_leaf01: np.ndarray, pred_leaf01: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    _draw_contours_rgb(out, gt_leaf01, color_rgb=(0, 255, 0), thickness=2)
    _draw_contours_rgb(out, pred_leaf01, color_rgb=(255, 0, 255), thickness=1)
    return out


def _text(img: np.ndarray, x: int, y: int, s: str, *, scale: float = 0.6, color=(255, 255, 255), thickness: int = 1) -> None:
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, float(scale), tuple(int(c) for c in color), int(thickness), cv2.LINE_AA)


def _make_compare(sample_id: str, split: str, gt_k: int, pred_k: int, merged: bool, fragmented: bool, dice: float, *, original: np.ndarray, gt_col: np.ndarray, pr_col: np.ndarray, ov: np.ndarray) -> np.ndarray:
    h, w = original.shape[:2]
    grid = np.concatenate([original, gt_col, pr_col, ov], axis=1)
    header_h = 90
    header = np.zeros((header_h, grid.shape[1], 3), dtype=np.uint8)
    header[:] = (20, 20, 20)
    _text(header, 12, 28, f"sample={sample_id} split={split}  GT_cc={gt_k}  Pred_cc={pred_k}  dice_leaflet={dice:.4f}", scale=0.7, thickness=2)
    _text(header, 12, 58, f"merged={str(bool(merged)).lower()}  fragmented={str(bool(fragmented)).lower()}", scale=0.7, thickness=2)
    out = np.concatenate([header, grid], axis=0)
    _text(out, 12, header_h + 28, "ORIGINAL", scale=0.8, thickness=2)
    _text(out, 12 + w, header_h + 28, "GT leaflets (instances)", scale=0.8, thickness=2)
    _text(out, 12 + 2 * w, header_h + 28, "PRED leaflets (instances)", scale=0.8, thickness=2)
    _text(out, 12 + 3 * w, header_h + 28, "OVERLAY (GT green, Pred magenta)", scale=0.8, thickness=2)
    return out


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


def _collect_prediction_exports(pred_dir: Path, split: str) -> list[PredSample]:
    out: list[PredSample] = []
    for d in sorted(pred_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name == "worst_cases":
            continue
        img = d / "image.png"
        gt = d / "gt.png"
        pr = d / "pred.png"
        if not (img.exists() and gt.exists() and pr.exists()):
            continue
        out.append(PredSample(sample_id=d.name, split=split, sample_dir=d, image_path=img, gt_path=gt, pred_path=pr))
    if not out:
        raise SystemExit(f"No prediction samples found in: {pred_dir}")
    return out


def _map_title_to_leaflet(title: str) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return False
    if "calc" in t or "кальц" in t:
        return False
    leaflet_markers = [
        "leaf",
        "leaflet",
        "leaflets",
        "leaflet_1",
        "leaflet_2",
        "leaflet_3",
        "створка",
        "створки",
        "створ",
    ]
    return any(m in t for m in leaflet_markers)


def _audit_supervisely_instances(ann_dir: Path) -> dict:
    ann_paths = sorted([p for p in ann_dir.glob("*.json") if p.is_file()], key=lambda p: p.name.lower())
    per_sample = []
    any_multi = 0
    title_counter = {}
    for p in ann_paths:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        objs = obj.get("objects")
        if not isinstance(objs, list):
            continue
        leaf_objs = []
        for o in objs:
            title = str(o.get("classTitle") or o.get("class_title") or o.get("title") or "")
            if _map_title_to_leaflet(title):
                leaf_objs.append(o)
                title_counter[title] = title_counter.get(title, 0) + 1
        leaf_n = int(len(leaf_objs))
        if leaf_n >= 2:
            any_multi += 1
        per_sample.append(
            {
                "sample_id": Path(p.name).stem.replace(".png", ""),
                "leaflet_objects": leaf_n,
                "leaflet_object_ids": [int(o.get("id")) for o in leaf_objs if isinstance(o.get("id"), int)],
                "leaflet_titles": sorted(list({str(o.get("classTitle") or "") for o in leaf_objs})),
                "has_objectId_field_nonnull": any(o.get("objectId") is not None for o in leaf_objs),
            }
        )
    return {
        "ann_dir": str(ann_dir),
        "ann_files": int(len(ann_paths)),
        "samples_with_2plus_leaflet_objects": int(any_multi),
        "leaflet_titles_hist": dict(sorted(title_counter.items(), key=lambda x: (-x[1], x[0]))),
        "per_sample": per_sample[:200],
    }


def _watershed_from_markers(topography_u8: np.ndarray, markers: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    if topography_u8.ndim == 2:
        topo3 = cv2.cvtColor(topography_u8, cv2.COLOR_GRAY2BGR)
    else:
        topo3 = topography_u8
    mk = markers.astype(np.int32).copy()
    mk[mask01.astype(bool) == 0] = 1
    cv2.watershed(topo3, mk)
    seg = np.zeros_like(mk, dtype=np.int32)
    seg[(mk > 1) & (mask01.astype(bool) == 1)] = mk[(mk > 1) & (mask01.astype(bool) == 1)]
    seg[seg < 0] = 0
    return seg


def _limit_instances(labels: np.ndarray, max_instances: int) -> tuple[np.ndarray, int]:
    max_k = int(max_instances)
    if max_k <= 0:
        return np.zeros_like(labels, dtype=np.int32), 0
    k = int(labels.max())
    if k <= max_k:
        return labels, k
    areas = []
    for i in range(1, k + 1):
        areas.append((int(np.sum(labels == i)), i))
    areas.sort(reverse=True, key=lambda x: x[0])
    keep = [i for _, i in areas[:max_k]]
    out = np.zeros_like(labels, dtype=np.int32)
    for new_i, old_i in enumerate(keep, start=1):
        out[labels == old_i] = int(new_i)
    return out, int(max_instances)


def _keep_topk_by_area(labels: np.ndarray, k: int) -> tuple[np.ndarray, int, list[int]]:
    kk = int(k)
    if kk <= 0:
        return np.zeros_like(labels, dtype=np.int32), 0, []
    n = int(labels.max())
    if n <= 0:
        return labels.astype(np.int32), 0, []
    areas = []
    for i in range(1, n + 1):
        areas.append((int(np.sum(labels == i)), i))
    areas.sort(reverse=True, key=lambda x: x[0])
    keep = [lab for _, lab in areas[:kk]]
    out = np.zeros_like(labels, dtype=np.int32)
    kept_areas = []
    for new_i, old_i in enumerate(keep, start=1):
        out[labels == old_i] = int(new_i)
        kept_areas.append(int(np.sum(labels == old_i)))
    return out, int(len(keep)), kept_areas


def _split_method_dt_watershed(pred01: np.ndarray, *, peak_dt_thresh: int, max_instances: int) -> np.ndarray:
    m = pred01.astype(np.uint8)
    if int(m.sum()) == 0:
        return np.zeros_like(m, dtype=np.int32)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    d = dist.copy()
    d[m == 0] = 0
    dil = cv2.dilate(d, np.ones((3, 3), np.uint8))
    peaks = (d == dil) & (d >= float(max(1, int(peak_dt_thresh))))
    peaks_u8 = peaks.astype(np.uint8)
    _, mk = cv2.connectedComponents(peaks_u8)
    mk = mk.astype(np.int32)
    mk[mk > 0] += 1
    seg = _watershed_from_markers(255 - dist_u8, mk, m)
    seg, _ = _limit_instances(seg, max_instances=int(max_instances))
    return seg


def _split_method_grad_watershed(image_rgb: np.ndarray, pred01: np.ndarray, *, peak_dt_thresh: int, max_instances: int) -> np.ndarray:
    m = pred01.astype(np.uint8)
    if int(m.sum()) == 0:
        return np.zeros_like(m, dtype=np.int32)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    d = dist.copy()
    d[m == 0] = 0
    dil = cv2.dilate(d, np.ones((3, 3), np.uint8))
    peaks = (d == dil) & (d >= float(max(1, int(peak_dt_thresh))))
    peaks_u8 = peaks.astype(np.uint8)
    _, mk = cv2.connectedComponents(peaks_u8)
    mk = mk.astype(np.int32)
    mk[mk > 0] += 1
    seg = _watershed_from_markers(mag_u8, mk, m)
    seg, _ = _limit_instances(seg, max_instances=int(max_instances))
    return seg


def _split_method_neck_dt(pred01: np.ndarray, *, neck_dt_thresh: int, max_instances: int) -> np.ndarray:
    m = pred01.astype(np.uint8)
    if int(m.sum()) == 0:
        return np.zeros_like(m, dtype=np.int32)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    cut = (dist <= float(max(1, int(neck_dt_thresh)))) & (m > 0)
    m2 = m.copy()
    m2[cut] = 0
    labels, _ = _label_cc(m2 > 0)
    labels, _ = _limit_instances(labels, max_instances=int(max_instances))
    if int(labels.max()) <= 1:
        labels, _ = _label_cc(m > 0)
        labels, _ = _limit_instances(labels, max_instances=int(max_instances))
        return labels
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mk = labels.astype(np.int32)
    mk[mk > 0] += 1
    seg = _watershed_from_markers(255 - dist_u8, mk, m)
    seg, _ = _limit_instances(seg, max_instances=int(max_instances))
    return seg


def _split_method_erosion_markers(pred01: np.ndarray, *, erosion_r: int, max_instances: int) -> np.ndarray:
    m = pred01.astype(np.uint8)
    if int(m.sum()) == 0:
        return np.zeros_like(m, dtype=np.int32)
    r = int(erosion_r)
    if r <= 0:
        labels, _ = _label_cc(m > 0)
        labels, _ = _limit_instances(labels, max_instances=int(max_instances))
        return labels
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    er = cv2.erode(m * 255, kernel, iterations=1)
    er01 = (er > 0).astype(np.uint8)
    labels, _ = _label_cc(er01 > 0)
    labels, _ = _limit_instances(labels, max_instances=int(max_instances))
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mk = labels.astype(np.int32)
    mk[mk > 0] += 1
    seg = _watershed_from_markers(255 - dist_u8, mk, m)
    seg, _ = _limit_instances(seg, max_instances=int(max_instances))
    return seg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--val-predictions-dir",
        type=Path,
        default=Path("training/analysis/val_predictions_unetpp_effb3_a100_multiclass_curated_finetune_stage2_lr1e5_100ep"),
    )
    ap.add_argument(
        "--test-predictions-dir",
        type=Path,
        default=Path("training/analysis/test_predictions_unetpp_effb3_a100_multiclass_curated_finetune_stage2_lr1e5_100ep"),
    )
    ap.add_argument("--dataset-root", type=Path, default=Path("datasets/converted_full_multiclass"))
    ap.add_argument("--curation-json", type=Path, default=Path("server_assets/curation/curation_result.json"))
    ap.add_argument("--supervisely-ann-dir", type=Path, default=Path("exports/supervisely_sdk/Срезы 2026/ann"))
    ap.add_argument("--out-dir", type=Path, default=Path("training/analysis/leaflet_instance_errors"))
    ap.add_argument("--max-leaflet-instances", type=int, default=3)
    ap.add_argument("--run-splitting-benchmark", action="store_true")
    args = ap.parse_args()

    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ds_root = args.dataset_root.resolve()
    train_txt = ds_root / "train.txt"
    val_txt = ds_root / "val.txt"
    test_txt = ds_root / "test.txt"
    train_ids = _read_split_ids(train_txt) if train_txt.exists() else set()
    val_ids = _read_split_ids(val_txt) if val_txt.exists() else set()
    test_ids = _read_split_ids(test_txt) if test_txt.exists() else set()

    curated = json.loads(args.curation_json.resolve().read_text(encoding="utf-8"))
    clean_ids = [str(x) for x in curated.get("clean", [])] if isinstance(curated, dict) else []
    medium_ids = [str(x) for x in curated.get("medium", [])] if isinstance(curated, dict) else []
    bad_ids = [str(x) for x in curated.get("bad", [])] if isinstance(curated, dict) else []

    def _gt_leaflet_cc(sid: str) -> int:
        m = _read_u8(ds_root / "masks" / f"{sid}.png")
        return _count_cc(m == 1)

    gt_audit = {"splits": {}, "curated_quality": {}, "overlap": {}}
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        if not ids:
            continue
        ccs = [_gt_leaflet_cc(s) for s in sorted(list(ids))[:]]
        gt_audit["splits"][split_name] = {
            "count": int(len(ccs)),
            "leaflet_cc_hist": {str(k): int(ccs.count(k)) for k in sorted(set(ccs))},
            "pct_ge_2": float(100.0 * sum(1 for x in ccs if x >= 2) / len(ccs)),
            "pct_ge_3": float(100.0 * sum(1 for x in ccs if x >= 3) / len(ccs)),
        }
    for q, ids in [("clean", clean_ids), ("medium", medium_ids), ("bad", bad_ids)]:
        if not ids:
            continue
        ccs = [_gt_leaflet_cc(s) for s in ids]
        gt_audit["curated_quality"][q] = {
            "count": int(len(ccs)),
            "leaflet_cc_hist": {str(k): int(ccs.count(k)) for k in sorted(set(ccs))},
            "pct_ge_2": float(100.0 * sum(1 for x in ccs if x >= 2) / len(ccs)),
            "pct_ge_3": float(100.0 * sum(1 for x in ccs if x >= 3) / len(ccs)),
        }
    gt_audit["overlap"] = {
        "train_val": int(len(train_ids & val_ids)),
        "train_test": int(len(train_ids & test_ids)),
        "val_test": int(len(val_ids & test_ids)),
    }

    supervisely_audit = _audit_supervisely_instances(args.supervisely_ann_dir.resolve())

    val_samples = _collect_prediction_exports(args.val_predictions_dir.resolve(), "val")
    test_samples = _collect_prediction_exports(args.test_predictions_dir.resolve(), "test")
    pred_samples = val_samples + test_samples

    metrics_rows = []
    by_split = {"val": [], "test": []}

    case_lists = {"val": {"merged": [], "fragmented": [], "mixed": [], "correct": []}, "test": {"merged": [], "fragmented": [], "mixed": [], "correct": []}}
    merged_case_samples: list[dict] = []

    vis_root = out_root
    (vis_root / "merged_cases").mkdir(parents=True, exist_ok=True)
    (vis_root / "fragmented_cases").mkdir(parents=True, exist_ok=True)
    (vis_root / "mixed_cases").mkdir(parents=True, exist_ok=True)
    (vis_root / "correct_cases").mkdir(parents=True, exist_ok=True)

    for s in pred_samples:
        img = _read_rgb_u8(s.image_path)
        gt = _read_u8(s.gt_path)
        pr = _read_u8(s.pred_path)
        if gt.shape != pr.shape:
            raise SystemExit(f"Shape mismatch: {s.sample_id} gt={gt.shape} pred={pr.shape}")

        gt_leaf = (gt == 1)
        pr_leaf = (pr == 1)

        gt_labels_all, gt_areas_all = _label_cc(gt_leaf)
        pr_labels_all, pr_areas_all = _label_cc(pr_leaf)
        gt_k_all = int(gt_labels_all.max())
        pr_k_all = int(pr_labels_all.max())

        gt_labels, gt_k, gt_areas = _keep_topk_by_area(gt_labels_all, int(args.max_leaflet_instances))
        pr_labels, pr_k, pr_areas = _keep_topk_by_area(pr_labels_all, int(args.max_leaflet_instances))

        gt_per_pred, pred_per_gt = _significant_overlaps_counts(gt_labels, pr_labels, gt_k, pr_k)
        max_gt_per_pred = int(max(gt_per_pred) if gt_per_pred else 0)
        max_pred_per_gt = int(max(pred_per_gt) if pred_per_gt else 0)

        merged_by_count = bool(gt_k >= 2 and pr_k < gt_k)
        fragmented_by_count = bool(pr_k > gt_k)
        merged_by_overlap = bool(gt_k >= 2 and max_gt_per_pred >= 2)
        fragmented_by_overlap = bool(max_pred_per_gt >= 2)

        dice_leaf = _dice(pr_leaf, gt_leaf)
        iou_leaf = _iou(pr_leaf, gt_leaf)

        iou_mat = _pair_iou_matrix(gt_labels, pr_labels, gt_k, pr_k)
        sum_iou, pairs = _perm_assignment_max_iou(iou_mat) if (gt_k > 0 and pr_k > 0 and gt_k <= 6 and pr_k <= 6) else (0.0, [])
        mean_matched_iou = float(sum_iou / max(gt_k, 1)) if gt_k > 0 else None

        if merged_by_overlap and fragmented_by_overlap:
            case_type = "mixed"
        elif merged_by_overlap:
            case_type = "merged"
        elif fragmented_by_overlap:
            case_type = "fragmented"
        else:
            case_type = "correct"

        by_split[s.split].append(case_type)
        case_lists[s.split][case_type].append(s.sample_id)

        row = {
            "sample": s.sample_id,
            "split": s.split,
            "gt_leaflet_components_all": int(gt_k_all),
            "pred_leaflet_components_all": int(pr_k_all),
            "gt_leaflet_components": int(gt_k),
            "pred_leaflet_components": int(pr_k),
            "component_count_delta": int(pr_k - gt_k),
            "merged_leaflet_failure_count_rule": int(1 if merged_by_count else 0),
            "fragmented_leaflet_failure_count_rule": int(1 if fragmented_by_count else 0),
            "merged_leaflet_failure_overlap_rule": int(1 if merged_by_overlap else 0),
            "fragmented_leaflet_failure_overlap_rule": int(1 if fragmented_by_overlap else 0),
            "case_type": case_type,
            "max_gt_components_covered_by_one_pred": int(max_gt_per_pred),
            "max_pred_components_covering_one_gt": int(max_pred_per_gt),
            "gt_largest_areas_all": json.dumps(_largest_areas(gt_areas_all, 5), ensure_ascii=False),
            "pred_largest_areas_all": json.dumps(_largest_areas(pr_areas_all, 5), ensure_ascii=False),
            "gt_largest_areas_kept": json.dumps([int(a) for a in gt_areas], ensure_ascii=False),
            "pred_largest_areas_kept": json.dumps([int(a) for a in pr_areas], ensure_ascii=False),
            "dice_leaflet": float(dice_leaf),
            "iou_leaflet": float(iou_leaf),
            "mean_matched_instance_iou": mean_matched_iou,
        }
        metrics_rows.append(row)

        case_dir = vis_root / f"{case_type}_cases" / s.sample_id
        case_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(case_dir / "original.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        gt_col = _colorize_labels(gt_labels, gt_k)
        pr_col = _colorize_labels(pr_labels, pr_k)
        cv2.imwrite(str(case_dir / "gt_leaflet_components.png"), cv2.cvtColor(gt_col, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(case_dir / "pred_leaflet_components.png"), cv2.cvtColor(pr_col, cv2.COLOR_RGB2BGR))
        ov = _overlay(img, gt_leaf, pr_leaf)
        cv2.imwrite(str(case_dir / "overlay.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        cmp_img = _make_compare(s.sample_id, s.split, gt_k, pr_k, merged_by_overlap, fragmented_by_overlap, float(dice_leaf), original=img, gt_col=gt_col, pr_col=pr_col, ov=ov)
        cv2.imwrite(str(case_dir / "compare.png"), cv2.cvtColor(cmp_img, cv2.COLOR_RGB2BGR))
        (case_dir / "metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

        if case_type == "merged":
            merged_case_samples.append(
                {
                    "sample": s.sample_id,
                    "split": s.split,
                    "image_path": str(s.image_path),
                    "gt_path": str(s.gt_path),
                    "pred_path": str(s.pred_path),
                }
            )

    metrics_csv = out_root / "leaflet_instance_error_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        w.writeheader()
        for r in metrics_rows:
            w.writerow(r)

    def _counts(split: str) -> dict:
        items = case_lists[split]
        return {k: int(len(v)) for k, v in items.items()}

    summary = {
        "ground_truth_audit": gt_audit,
        "supervisely_audit": supervisely_audit,
        "instance_error_counts": {"val": _counts("val"), "test": _counts("test")},
        "case_lists": case_lists,
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "summary_json": str(out_root / "leaflet_instance_error_summary.json"),
            "visual_report_dir": str(out_root),
        },
    }

    benchmark_rows = []
    best_method = None
    best_score = -math.inf

    if args.run_splitting_benchmark and merged_case_samples:
        methods = [
            ("dt_watershed", [{"peak_dt_thresh": 1}, {"peak_dt_thresh": 2}, {"peak_dt_thresh": 3}]),
            ("grad_watershed", [{"peak_dt_thresh": 1}, {"peak_dt_thresh": 2}, {"peak_dt_thresh": 3}]),
            ("neck_dt", [{"neck_dt_thresh": 1}, {"neck_dt_thresh": 2}, {"neck_dt_thresh": 3}]),
            ("erosion_markers", [{"erosion_r": 1}, {"erosion_r": 2}, {"erosion_r": 3}]),
        ]

        def _apply(method: str, *, image_rgb: np.ndarray, pred01: np.ndarray, params: dict) -> np.ndarray:
            if method == "dt_watershed":
                return _split_method_dt_watershed(pred01, peak_dt_thresh=int(params["peak_dt_thresh"]), max_instances=int(args.max_leaflet_instances))
            if method == "grad_watershed":
                return _split_method_grad_watershed(image_rgb, pred01, peak_dt_thresh=int(params["peak_dt_thresh"]), max_instances=int(args.max_leaflet_instances))
            if method == "neck_dt":
                return _split_method_neck_dt(pred01, neck_dt_thresh=int(params["neck_dt_thresh"]), max_instances=int(args.max_leaflet_instances))
            if method == "erosion_markers":
                return _split_method_erosion_markers(pred01, erosion_r=int(params["erosion_r"]), max_instances=int(args.max_leaflet_instances))
            raise ValueError(method)

        for method, param_list in methods:
            for params in param_list:
                per_sample = []
                merged_fail = 0
                frag_fail = 0
                mean_iou_vals = []
                dice_before_vals = []
                dice_after_vals = []
                degraded = 0
                for s0 in merged_case_samples:
                    sid = str(s0["sample"])
                    split = str(s0["split"])
                    sample_dir = (args.val_predictions_dir if split == "val" else args.test_predictions_dir).resolve() / sid
                    img = _read_rgb_u8(sample_dir / "image.png")
                    gt = _read_u8(sample_dir / "gt.png")
                    pr = _read_u8(sample_dir / "pred.png")
                    gt_leaf = (gt == 1)
                    pr_leaf = (pr == 1)
                    gt_labels_all, _ = _label_cc(gt_leaf)
                    gt_labels, gt_k, _ = _keep_topk_by_area(gt_labels_all, int(args.max_leaflet_instances))

                    inst = _apply(method, image_rgb=img, pred01=pr_leaf.astype(np.uint8), params=params)
                    inst, inst_k = _limit_instances(inst, int(args.max_leaflet_instances))
                    pred_union = inst > 0
                    dice_before = float(_dice(pr_leaf, gt_leaf))
                    dice_after = float(_dice(pred_union, gt_leaf))
                    dice_before_vals.append(dice_before)
                    dice_after_vals.append(dice_after)
                    if dice_after < dice_before - 0.02:
                        degraded += 1

                    if gt_k >= 2 and inst_k < gt_k:
                        merged_fail += 1
                    if inst_k > gt_k:
                        frag_fail += 1

                    iou_mat = _pair_iou_matrix(gt_labels, inst.astype(np.int32), gt_k, inst_k) if inst_k > 0 else np.zeros((gt_k, 0))
                    sum_iou, _ = _perm_assignment_max_iou(iou_mat) if (gt_k > 0 and inst_k > 0 and gt_k <= 6 and inst_k <= 6) else (0.0, [])
                    mean_iou = float(sum_iou / max(gt_k, 1)) if gt_k > 0 else None
                    if mean_iou is not None:
                        mean_iou_vals.append(mean_iou)

                    per_sample.append((sid, split, gt_k, inst_k, mean_iou, dice_before, dice_after))

                mean_iou_all = float(sum(mean_iou_vals) / len(mean_iou_vals)) if mean_iou_vals else None
                mean_dice_before = float(sum(dice_before_vals) / len(dice_before_vals)) if dice_before_vals else None
                mean_dice_after = float(sum(dice_after_vals) / len(dice_after_vals)) if dice_after_vals else None

                bench = {
                    "method": method,
                    "params": json.dumps(params, ensure_ascii=False),
                    "merged_cases_count": int(len(merged_case_samples)),
                    "merged_failures": int(merged_fail),
                    "fragmented_failures": int(frag_fail),
                    "mean_matched_instance_iou": mean_iou_all,
                    "mean_semantic_dice_before": mean_dice_before,
                    "mean_semantic_dice_after": mean_dice_after,
                    "pct_semantic_dice_degraded_gt_0p02": float(100.0 * degraded / len(merged_case_samples)) if merged_case_samples else None,
                }
                benchmark_rows.append(bench)

                score = 0.0
                score += float((1.0 - merged_fail / max(len(merged_case_samples), 1)) * 10.0)
                score -= float((frag_fail / max(len(merged_case_samples), 1)) * 3.0)
                if mean_iou_all is not None:
                    score += float(mean_iou_all * 5.0)
                if mean_dice_after is not None and mean_dice_before is not None:
                    score += float((mean_dice_after - mean_dice_before) * 10.0)
                score -= float((degraded / max(len(merged_case_samples), 1)) * 5.0)

                if score > best_score:
                    best_score = score
                    best_method = {"method": method, "params": params, "summary": bench, "score": float(score)}

        bench_csv = out_root / "instance_separation_benchmark.csv"
        with bench_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(benchmark_rows[0].keys()))
            w.writeheader()
            for r in benchmark_rows:
                w.writerow(r)
        summary["outputs"]["instance_benchmark_csv"] = str(bench_csv)
        summary["outputs"]["best_method_json"] = str(out_root / "best_method_summary.json")

    (out_root / "leaflet_instance_error_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if best_method is not None:
        (out_root / "best_method_summary.json").write_text(json.dumps(best_method, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Merged cases (overlap rule): {len(merged_case_samples)}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
