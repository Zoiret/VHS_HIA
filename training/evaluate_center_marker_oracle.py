from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Sample:
    sample_id: str
    split: str
    quality: str | None


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


def _read_u16(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(str(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    return arr


def _center_crop_to_shape(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == target_h and w == target_w:
        return arr
    if h < target_h or w < target_w:
        raise ValueError(f"Cannot crop {target_h}x{target_w} from {h}x{w}")
    y0 = (h - target_h) // 2
    x0 = (w - target_w) // 2
    if arr.ndim == 2:
        return arr[y0 : y0 + target_h, x0 : x0 + target_w]
    return arr[y0 : y0 + target_h, x0 : x0 + target_w, :]


def _read_split_ids(split_txt: Path) -> list[str]:
    ids = []
    with split_txt.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise SystemExit(f"Invalid line in {split_txt}: {line!r}")
            ids.append(Path(parts[0]).stem)
    return ids


def _local_maxima_by_cc(center01: np.ndarray, leaf_union: np.ndarray, thr: float, max_markers: int = 3) -> list[tuple[int, int, float]]:
    c = center01.astype(np.float32).copy()
    c[~leaf_union.astype(bool)] = 0.0
    m = (c >= float(thr)).astype(np.uint8)
    if int(m.sum()) == 0:
        return []
    n, labels = cv2.connectedComponents(m)
    pts: list[tuple[int, int, float]] = []
    for lab in range(1, int(n)):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        vals = c[ys, xs]
        k = int(np.argmax(vals))
        y = int(ys[k])
        x = int(xs[k])
        pts.append((y, x, float(c[y, x])))
    pts.sort(key=lambda t: t[2], reverse=True)
    return pts[: int(max_markers)]


def _nms_markers(center01: np.ndarray, leaf_union: np.ndarray, thr: float, min_dist: int, max_markers: int = 3) -> list[tuple[int, int, float]]:
    c = center01.astype(np.float32).copy()
    c[~leaf_union.astype(bool)] = 0.0
    ys, xs = np.where(c >= float(thr))
    if ys.size == 0:
        return []
    vals = c[ys, xs]
    order = np.argsort(-vals)
    pts = []
    r2 = float(int(min_dist) * int(min_dist))
    for idx in order.tolist():
        y = int(ys[idx])
        x = int(xs[idx])
        v = float(vals[idx])
        ok = True
        for (py, px, _) in pts:
            if float((py - y) * (py - y) + (px - x) * (px - x)) < r2:
                ok = False
                break
        if ok:
            pts.append((y, x, v))
        if len(pts) >= int(max_markers):
            break
    return pts


def _fallback_marker_from_distance(dist01: np.ndarray, component01: np.ndarray) -> tuple[int, int, float] | None:
    d = dist01.astype(np.float32).copy()
    d[~component01.astype(bool)] = -1.0
    max_d = float(d.max())
    if max_d <= 0.0:
        ys, xs = np.where(component01)
        if ys.size == 0:
            return None
        return int(ys[0]), int(xs[0]), 0.0
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return int(y), int(x), float(max_d)


def _connected_components(mask01: np.ndarray) -> tuple[np.ndarray, int]:
    m = (mask01.astype(np.uint8) > 0).astype(np.uint8) * 255
    n, labels = cv2.connectedComponents(m, connectivity=8)
    return labels.astype(np.int32), max(0, int(n) - 1)


def _rgb_gradient_u8(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag_u8


def _geometry_topo_u8(component01: np.ndarray) -> np.ndarray:
    m = component01.astype(np.uint8)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3).astype(np.float32)
    if float(dt.max()) > 0.0:
        dt = dt / float(dt.max())
    topo = (1.0 - dt) * 255.0
    return topo.astype(np.uint8)


def _watershed(
    component01: np.ndarray,
    markers_yx: list[tuple[int, int]],
    topo_u8: np.ndarray,
) -> np.ndarray:
    h, w = component01.shape[:2]
    mk = np.zeros((h, w), dtype=np.int32)
    mk[component01.astype(bool) == 0] = 1
    for idx, (y, x) in enumerate(markers_yx, start=2):
        if 0 <= y < h and 0 <= x < w and bool(component01[y, x]):
            mk[y, x] = int(idx)
    topo3 = cv2.cvtColor(topo_u8, cv2.COLOR_GRAY2BGR)
    cv2.watershed(topo3, mk)
    out = np.zeros((h, w), dtype=np.uint8)
    labs = sorted([int(v) for v in np.unique(mk) if int(v) > 1])
    for new_i, lab in enumerate(labs, start=1):
        out[(mk == lab) & component01.astype(bool)] = np.uint8(new_i)
    out[component01.astype(bool) == 0] = 0
    return out


def _keep_top3_by_area(labels_u8: np.ndarray) -> tuple[np.ndarray, int]:
    k = int(labels_u8.max())
    if k <= 3:
        return labels_u8, k
    areas = []
    for i in range(1, k + 1):
        areas.append((int(np.sum(labels_u8 == i)), i))
    areas.sort(reverse=True, key=lambda t: t[0])
    keep = [lab for _, lab in areas[:3]]
    out = np.zeros_like(labels_u8, dtype=np.uint8)
    for new_i, old_i in enumerate(keep, start=1):
        out[labels_u8 == old_i] = np.uint8(new_i)
    return out, 3


def _iou_matrix(gt_u8: np.ndarray, pred_u8: np.ndarray, gt_k: int, pred_k: int) -> np.ndarray:
    m = np.zeros((int(gt_k), int(pred_k)), dtype=np.float64)
    if gt_k == 0 or pred_k == 0:
        return m
    for gi in range(1, int(gt_k) + 1):
        g = gt_u8 == gi
        g_sum = float(np.sum(g))
        if g_sum <= 0:
            continue
        for pi in range(1, int(pred_k) + 1):
            p = pred_u8 == pi
            p_sum = float(np.sum(p))
            if p_sum <= 0:
                continue
            inter = float(np.sum(g & p))
            if inter <= 0:
                continue
            union = g_sum + p_sum - inter
            m[gi - 1, pi - 1] = inter / max(union, 1.0)
    return m


def _best_perm_sum(iou: np.ndarray) -> float:
    gt_k, pred_k = iou.shape[0], iou.shape[1]
    if gt_k == 0 or pred_k == 0:
        return 0.0
    k = min(int(gt_k), int(pred_k))
    best = -1.0
    import itertools

    for cols in itertools.permutations(range(int(pred_k)), k):
        s = 0.0
        for r, c in enumerate(cols):
            s += float(iou[r, c])
        if s > best:
            best = s
    return float(best if best >= 0 else 0.0)


def _dice(mask_a: np.ndarray, mask_b: np.ndarray, eps: float = 1e-7) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = float(np.sum(a & b))
    sa = float(np.sum(a))
    sb = float(np.sum(b))
    return float((2.0 * inter + eps) / (sa + sb + eps))


def _colorize_instances(labels_u8: np.ndarray) -> np.ndarray:
    out = np.zeros((labels_u8.shape[0], labels_u8.shape[1], 3), dtype=np.uint8)
    pal = {1: (230, 25, 75), 2: (60, 180, 75), 3: (255, 225, 25)}
    for i in [1, 2, 3]:
        out[labels_u8 == i] = pal[i]
    return out


def _heatmap_u16(u16: np.ndarray) -> np.ndarray:
    x = (u16.astype(np.float32) / 65535.0) * 255.0
    x8 = np.clip(x, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(x8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def _draw_markers(img_rgb: np.ndarray, markers: list[tuple[int, int]], *, color=(255, 255, 255)) -> np.ndarray:
    out = img_rgb.copy()
    for idx, (y, x) in enumerate(markers, start=1):
        cv2.circle(out, (int(x), int(y)), 6, tuple(int(c) for c in color), thickness=2)
        cv2.putText(out, str(idx), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple(int(c) for c in color), 2, cv2.LINE_AA)
    return out


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
    except Exception:
        import shutil

        shutil.copy2(src, dst)


def _extract_metadata_centers(meta_path: Path) -> list[tuple[int, int]]:
    obj = json.loads(meta_path.read_text(encoding="utf-8"))
    centers = []
    for inst in obj.get("instances") or []:
        yx = inst.get("center_yx")
        if isinstance(yx, list) and len(yx) == 2 and isinstance(yx[0], int) and isinstance(yx[1], int):
            centers.append((int(yx[0]), int(yx[1])))
    return centers[:3]


def _build_markers(
    *,
    method: str,
    center01: np.ndarray,
    leaf_union: np.ndarray,
    meta_centers: list[tuple[int, int]] | None,
    thr: float | None,
    min_dist: int | None,
    max_markers: int = 3,
) -> list[tuple[int, int]]:
    if method == "metadata":
        pts = meta_centers or []
        pts = [(int(y), int(x)) for (y, x) in pts if 0 <= y < leaf_union.shape[0] and 0 <= x < leaf_union.shape[1] and bool(leaf_union[y, x])]
        return pts[: int(max_markers)]
    if method == "cc":
        assert thr is not None
        pts = _local_maxima_by_cc(center01, leaf_union, float(thr), max_markers=max_markers)
        return [(int(y), int(x)) for (y, x, _) in pts]
    if method == "nms":
        assert thr is not None and min_dist is not None
        pts = _nms_markers(center01, leaf_union, float(thr), int(min_dist), max_markers=max_markers)
        return [(int(y), int(x)) for (y, x, _) in pts]
    raise ValueError(method)


def _hybrid_reconstruct(
    *,
    leaf_union01: np.ndarray,
    dist01: np.ndarray,
    img_rgb: np.ndarray,
    markers_yx: list[tuple[int, int]],
    watershed_variant: str,
) -> tuple[np.ndarray, dict]:
    labels_cc, cc_k = _connected_components(leaf_union01)
    h, w = leaf_union01.shape[:2]
    pred = np.zeros((h, w), dtype=np.uint8)

    used_fallback = 0
    components_unchanged = 0
    merged_components_split = 0
    watershed_calls = 0
    incorrect_splits = 0

    marker_map = np.zeros((h, w), dtype=np.uint8)
    for y, x in markers_yx:
        if 0 <= y < h and 0 <= x < w:
            marker_map[y, x] = 1

    next_label = 1

    for comp_id in range(1, int(cc_k) + 1):
        comp01 = labels_cc == comp_id
        ys, xs = np.where(comp01)
        if ys.size == 0:
            continue
        in_markers = [(y, x) for (y, x) in markers_yx if bool(comp01[int(y), int(x)])]
        if len(in_markers) == 0:
            fb = _fallback_marker_from_distance(dist01, comp01)
            if fb is not None:
                in_markers = [(int(fb[0]), int(fb[1]))]
                used_fallback += 1
        if len(in_markers) <= 1:
            pred[comp01] = np.uint8(next_label)
            next_label += 1
            components_unchanged += 1
            continue

        merged_components_split += 1
        watershed_calls += 1
        if watershed_variant == "dt":
            topo_u8 = (1.0 - np.clip(dist01, 0.0, 1.0)) * 255.0
            topo_u8 = topo_u8.astype(np.uint8)
        elif watershed_variant == "rgb_grad":
            topo_u8 = _rgb_gradient_u8(img_rgb)
        elif watershed_variant == "geom":
            topo_u8 = _geometry_topo_u8(comp01.astype(np.uint8))
        else:
            raise ValueError(watershed_variant)

        seg = _watershed(comp01.astype(np.uint8), in_markers, topo_u8)
        seg, seg_k = _keep_top3_by_area(seg)
        if seg_k <= 1:
            pred[comp01] = np.uint8(next_label)
            next_label += 1
            continue

        for local in range(1, int(seg_k) + 1):
            pred[seg == local] = np.uint8(next_label)
            next_label += 1

    pred, pred_k = _keep_top3_by_area(pred)

    return pred, {
        "cc_components": int(cc_k),
        "markers_total": int(len(markers_yx)),
        "components_unchanged": int(components_unchanged),
        "merged_components_split": int(merged_components_split),
        "watershed_calls": int(watershed_calls),
        "used_fallback_markers": int(used_fallback),
        "incorrect_splits": int(incorrect_splits),
        "pred_instances": int(pred_k),
    }


def _case_type(gt_k: int, pred_k: int) -> str:
    merged = gt_k >= 2 and pred_k < gt_k
    fragmented = pred_k > gt_k
    if merged and fragmented:
        return "mixed"
    if merged:
        return "merged"
    if fragmented:
        return "fragmented"
    return "correct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance-root", type=Path, default=Path("datasets/converted_leaflet_distance"))
    ap.add_argument("--source-instances-root", type=Path, default=Path("datasets/converted_leaflet_instances"))
    ap.add_argument("--out-dir", type=Path, default=Path("training/analysis/center_marker_oracle"))
    ap.add_argument("--max-instances", type=int, default=3)
    ap.add_argument("--save-visuals", action="store_true")
    args = ap.parse_args()

    d_root = args.distance_root.resolve()
    s_root = args.source_instances_root.resolve()

    train_ids = _read_split_ids(d_root / "train.txt")
    val_ids = _read_split_ids(d_root / "val.txt")
    test_ids = _read_split_ids(d_root / "test.txt")
    split_map = {sid: "train" for sid in train_ids}
    split_map.update({sid: "val" for sid in val_ids})
    split_map.update({sid: "test" for sid in test_ids})

    summary_path = d_root / "distance_dataset_summary.json"
    quality_map = {}
    if summary_path.exists():
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        q = obj.get("quality_counts")
        _ = q

    manifest_path = d_root / "distance_dataset_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    manifest_rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    for r in manifest_rows:
        if r.get("quality"):
            quality_map[str(r["sample"])] = str(r["quality"])

    samples = [Sample(sample_id=str(r["sample"]), split=str(r["split"]), quality=(quality_map.get(str(r["sample"])) or None)) for r in manifest_rows]

    marker_methods = []
    for thr in [0.3, 0.5, 0.7]:
        marker_methods.append(("cc", {"thr": thr}))
    for thr in [0.3, 0.5, 0.7]:
        for md in [4, 8, 12, 16]:
            marker_methods.append(("nms", {"thr": thr, "min_dist": md}))
    marker_methods.append(("metadata", {}))
    watershed_variants = ["dt", "rgb_grad", "geom"]

    results = []

    def eval_config(method: str, params: dict, watershed_variant: str) -> dict:
        rows = []
        per_group = defaultdict(list)
        by_inst_count = {1: [], 2: [], 3: []}
        by_quality = {"clean": [], "medium": [], "bad": [], "unknown": []}
        unchanged_sum = 0
        split_sum = 0
        incorrect_split_sum = 0
        for s in samples:
            sid = s.sample_id
            img_p = d_root / "images" / f"{sid}.png"
            sem_p = d_root / "semantic_masks" / f"{sid}.png"
            dist_p = d_root / "distance_maps" / f"{sid}.png"
            center_p = d_root / "center_maps" / f"{sid}.png"
            meta_p = d_root / "metadata" / f"{sid}.json"
            inst_p = s_root / "instance_masks" / f"{sid}.png"

            img = _read_rgb_u8(img_p)
            sem = _read_u8(sem_p)
            dist_u16 = _read_u16(dist_p)
            center_u16 = _read_u16(center_p) if center_p.exists() else np.zeros_like(dist_u16)
            inst_src = _read_u8(inst_p)
            inst = _center_crop_to_shape(inst_src, sem.shape[0], sem.shape[1])

            leaf_union = sem == 1
            dist01 = dist_u16.astype(np.float32) / 65535.0
            center01 = center_u16.astype(np.float32) / 65535.0

            meta_centers = _extract_metadata_centers(meta_p) if meta_p.exists() else []
            markers = _build_markers(
                method=method,
                center01=center01,
                leaf_union=leaf_union,
                meta_centers=meta_centers,
                thr=params.get("thr"),
                min_dist=params.get("min_dist"),
                max_markers=int(args.max_instances),
            )

            pred, dbg = _hybrid_reconstruct(
                leaf_union01=leaf_union.astype(np.uint8),
                dist01=dist01,
                img_rgb=img,
                markers_yx=markers,
                watershed_variant=watershed_variant,
            )

            gt_k = int(len([i for i in [1, 2, 3] if int(np.sum(inst == i)) > 0]))
            pred_k = int(len([i for i in [1, 2, 3] if int(np.sum(pred == i)) > 0]))

            iou_mat = _iou_matrix(inst, pred, gt_k, pred_k)
            sum_iou = _best_perm_sum(iou_mat)
            mean_iou = float(sum_iou / max(gt_k, 1)) if gt_k > 0 else 0.0

            case = _case_type(gt_k, pred_k)
            perfect = int((pred_k == gt_k) and (mean_iou >= 0.90))
            exact_count = int(pred_k == gt_k)

            semantic_before = 1.0
            semantic_after = float(_dice(pred > 0, leaf_union))

            row = {
                "sample": sid,
                "split": s.split,
                "quality": s.quality or "unknown",
                "gt_instances": gt_k,
                "pred_instances": pred_k,
                "case_type": case,
                "exact_count": exact_count,
                "mean_matched_instance_iou": mean_iou,
                "perfect": perfect,
                "semantic_dice_before": semantic_before,
                "semantic_dice_after": semantic_after,
                "markers": int(len(markers)),
                "components_unchanged": int(dbg["components_unchanged"]),
                "merged_components_split": int(dbg["merged_components_split"]),
                "used_fallback_markers": int(dbg["used_fallback_markers"]),
                "watershed_calls": int(dbg["watershed_calls"]),
            }
            rows.append(row)

            by_inst_count.get(gt_k, []).append(row)
            by_quality.get(s.quality or "unknown", []).append(row)
            unchanged_sum += int(dbg["components_unchanged"])
            split_sum += int(dbg["merged_components_split"])
            incorrect_split_sum += int(dbg["incorrect_splits"])

        exact_acc = float(np.mean([r["exact_count"] for r in rows])) if rows else 0.0
        mean_iou_all = float(np.mean([r["mean_matched_instance_iou"] for r in rows])) if rows else 0.0
        med_iou_all = float(np.median([r["mean_matched_instance_iou"] for r in rows])) if rows else 0.0
        perfect_rate = float(np.mean([r["perfect"] for r in rows])) if rows else 0.0
        merged_n = int(sum(1 for r in rows if r["case_type"] == "merged"))
        frag_n = int(sum(1 for r in rows if r["case_type"] == "fragmented"))
        mixed_n = int(sum(1 for r in rows if r["case_type"] == "mixed"))

        return {
            "method": method,
            "params": params,
            "watershed_variant": watershed_variant,
            "samples": int(len(rows)),
            "exact_instance_count_accuracy": exact_acc,
            "merged": merged_n,
            "fragmented": frag_n,
            "mixed": mixed_n,
            "mean_matched_instance_iou": mean_iou_all,
            "median_matched_instance_iou": med_iou_all,
            "perfect_recovery_rate": perfect_rate,
            "samples_left_unchanged_components_sum": int(unchanged_sum),
            "merged_components_split_sum": int(split_sum),
            "incorrect_splits_sum": int(incorrect_split_sum),
            "rows": rows,
            "by_instance_count": by_inst_count,
            "by_quality": by_quality,
        }

    comparison_rows = []
    best_metadata = None
    best_metadata_score = -1e9
    for variant in watershed_variants:
        r = eval_config("metadata", {}, variant)
        score = float(r["mean_matched_instance_iou"]) * 10.0 + float(r["exact_instance_count_accuracy"]) * 2.0 - float(r["fragmented"]) / max(r["samples"], 1)
        if score > best_metadata_score:
            best_metadata_score = score
            best_metadata = r
        comparison_rows.append(
            {
                "method": "metadata",
                "params": json.dumps({}, ensure_ascii=False),
                "watershed_variant": variant,
                "samples": r["samples"],
                "exact_count_acc": r["exact_instance_count_accuracy"],
                "mean_iou": r["mean_matched_instance_iou"],
                "median_iou": r["median_matched_instance_iou"],
                "perfect_rate": r["perfect_recovery_rate"],
                "merged": r["merged"],
                "fragmented": r["fragmented"],
                "mixed": r["mixed"],
            }
        )

    assert best_metadata is not None
    best_variant = str(best_metadata["watershed_variant"])

    best_cm = None
    best_cm_score = -1e9
    for method, params in marker_methods:
        if method == "metadata":
            continue
        r = eval_config(method, params, best_variant)
        score = float(r["mean_matched_instance_iou"]) * 10.0 + float(r["exact_instance_count_accuracy"]) * 2.0 - float(r["fragmented"]) / max(r["samples"], 1)
        if score > best_cm_score:
            best_cm_score = score
            best_cm = r
        comparison_rows.append(
            {
                "method": method,
                "params": json.dumps(params, ensure_ascii=False),
                "watershed_variant": best_variant,
                "samples": r["samples"],
                "exact_count_acc": r["exact_instance_count_accuracy"],
                "mean_iou": r["mean_matched_instance_iou"],
                "median_iou": r["median_matched_instance_iou"],
                "perfect_rate": r["perfect_recovery_rate"],
                "merged": r["merged"],
                "fragmented": r["fragmented"],
                "mixed": r["mixed"],
            }
        )

    assert best_cm is not None

    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    comparison_csv = out_root / "marker_method_comparison.csv"
    with comparison_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        w.writeheader()
        for r in comparison_rows:
            w.writerow(r)

    best_json = out_root / "best_center_marker_method.json"
    _write_json(
        best_json,
        {
            "metadata_upper_bound": {
                "watershed_variant": best_variant,
                "exact_instance_count_accuracy": best_metadata["exact_instance_count_accuracy"],
                "merged": best_metadata["merged"],
                "fragmented": best_metadata["fragmented"],
                "mixed": best_metadata["mixed"],
                "mean_matched_instance_iou": best_metadata["mean_matched_instance_iou"],
                "median_matched_instance_iou": best_metadata["median_matched_instance_iou"],
                "perfect_recovery_rate": best_metadata["perfect_recovery_rate"],
            },
            "best_center_map_extraction": {
                "method": best_cm["method"],
                "params": best_cm["params"],
                "watershed_variant": best_variant,
                "exact_instance_count_accuracy": best_cm["exact_instance_count_accuracy"],
                "merged": best_cm["merged"],
                "fragmented": best_cm["fragmented"],
                "mixed": best_cm["mixed"],
                "mean_matched_instance_iou": best_cm["mean_matched_instance_iou"],
                "median_matched_instance_iou": best_cm["median_matched_instance_iou"],
                "perfect_recovery_rate": best_cm["perfect_recovery_rate"],
            },
        },
    )

    def summarize_rows(rows: list[dict]) -> dict:
        if not rows:
            return {"n": 0}
        return {
            "n": int(len(rows)),
            "exact_count_accuracy": float(np.mean([r["exact_count"] for r in rows])),
            "merged": int(sum(1 for r in rows if r["case_type"] == "merged")),
            "fragmented": int(sum(1 for r in rows if r["case_type"] == "fragmented")),
            "mixed": int(sum(1 for r in rows if r["case_type"] == "mixed")),
            "mean_matched_instance_iou": float(np.mean([r["mean_matched_instance_iou"] for r in rows])),
            "median_matched_instance_iou": float(np.median([r["mean_matched_instance_iou"] for r in rows])),
            "perfect_recovery_rate": float(np.mean([r["perfect"] for r in rows])),
        }

    summary = {
        "distance_root": str(d_root),
        "source_instances_root": str(s_root),
        "outputs": {
            "results_csv": str(out_root / "center_marker_oracle_results.csv"),
            "summary_json": str(out_root / "center_marker_oracle_summary.json"),
            "comparison_csv": str(comparison_csv),
            "best_method_json": str(best_json),
            "visual_dir": str(out_root / "visuals"),
        },
        "metadata_upper_bound": summarize_rows(best_metadata["rows"]),
        "best_center_map_extraction": {
            "method": best_cm["method"],
            "params": best_cm["params"],
            "watershed_variant": best_variant,
            "summary": summarize_rows(best_cm["rows"]),
        },
        "hybrid_logic": {
            "best_center_map": {
                "samples_left_unchanged_components_sum": int(best_cm["samples_left_unchanged_components_sum"]),
                "merged_components_split_sum": int(best_cm["merged_components_split_sum"]),
                "incorrect_splits_sum": int(best_cm["incorrect_splits_sum"]),
            }
        },
        "by_instance_count": {
            "metadata": {str(k): summarize_rows(best_metadata["by_instance_count"].get(k, [])) for k in [1, 2, 3]},
            "best_center_map": {str(k): summarize_rows(best_cm["by_instance_count"].get(k, [])) for k in [1, 2, 3]},
        },
        "by_quality": {
            "metadata": {k: summarize_rows(best_metadata["by_quality"].get(k, [])) for k in ["clean", "medium", "bad", "unknown"]},
            "best_center_map": {k: summarize_rows(best_cm["by_quality"].get(k, [])) for k in ["clean", "medium", "bad", "unknown"]},
        },
    }
    _write_json(out_root / "center_marker_oracle_summary.json", summary)

    results_csv = out_root / "center_marker_oracle_results.csv"
    rows_out = []
    for tag, r0 in [("metadata", best_metadata), ("best_center_map", best_cm)]:
        for row in r0["rows"]:
            r = dict(row)
            r["oracle_tag"] = tag
            r["marker_method"] = r0["method"]
            r["marker_params"] = json.dumps(r0["params"], ensure_ascii=False)
            r["watershed_variant"] = r0["watershed_variant"]
            rows_out.append(r)
    with results_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    if args.save_visuals:
        vis_root = out_root / "visuals"
        vis_root.mkdir(parents=True, exist_ok=True)

        def render_set(tag: str, run: dict) -> None:
            tag_root = vis_root / tag
            for c in ["correct", "merged", "fragmented", "mixed"]:
                (tag_root / c).mkdir(parents=True, exist_ok=True)
            for row in run["rows"]:
                sid = row["sample"]
                case = row["case_type"]
                sample_dir = tag_root / case / sid
                sample_dir.mkdir(parents=True, exist_ok=True)

                img_p = d_root / "images" / f"{sid}.png"
                sem_p = d_root / "semantic_masks" / f"{sid}.png"
                dist_p = d_root / "distance_maps" / f"{sid}.png"
                center_p = d_root / "center_maps" / f"{sid}.png"
                meta_p = d_root / "metadata" / f"{sid}.json"
                inst_p = s_root / "instance_masks" / f"{sid}.png"

                _ensure_link_or_copy(img_p, sample_dir / "original.png")
                _ensure_link_or_copy(sem_p, sample_dir / "semantic_mask.png")
                _ensure_link_or_copy(dist_p, sample_dir / "distance_map.png")
                if center_p.exists():
                    _ensure_link_or_copy(center_p, sample_dir / "center_map.png")

                img = _read_rgb_u8(img_p)
                sem = _read_u8(sem_p)
                dist_u16 = _read_u16(dist_p)
                center_u16 = _read_u16(center_p) if center_p.exists() else np.zeros_like(dist_u16)
                inst_src = _read_u8(inst_p)
                inst = _center_crop_to_shape(inst_src, sem.shape[0], sem.shape[1])
                cv2.imwrite(str(sample_dir / "gt_instance_mask.png"), inst.astype(np.uint8))

                leaf_union = sem == 1
                dist01 = dist_u16.astype(np.float32) / 65535.0
                center01 = center_u16.astype(np.float32) / 65535.0
                meta_centers = _extract_metadata_centers(meta_p) if meta_p.exists() else []
                markers = _build_markers(
                    method=run["method"],
                    center01=center01,
                    leaf_union=leaf_union,
                    meta_centers=meta_centers,
                    thr=run["params"].get("thr"),
                    min_dist=run["params"].get("min_dist"),
                    max_markers=int(args.max_instances),
                )
                pred, _ = _hybrid_reconstruct(
                    leaf_union01=leaf_union.astype(np.uint8),
                    dist01=dist01,
                    img_rgb=img,
                    markers_yx=markers,
                    watershed_variant=run["watershed_variant"],
                )

                markers_img = _draw_markers(img, markers)
                cv2.imwrite(str(sample_dir / "extracted_markers.png"), cv2.cvtColor(markers_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(sample_dir / "watershed_result.png"), pred.astype(np.uint8))

                panels = [
                    img,
                    _colorize_instances(inst),
                    _heatmap_u16(center_u16),
                    _draw_markers(_heatmap_u16(center_u16), markers, color=(255, 255, 255)),
                    _colorize_instances(pred),
                ]
                compare = np.concatenate(panels, axis=1)
                cv2.imwrite(str(sample_dir / "comparison.png"), cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))
                _write_json(sample_dir / "metrics.json", row)

        render_set("metadata_oracle", best_metadata)
        render_set("best_center_map", best_cm)

        meta_fail = [r for r in best_metadata["rows"] if r["perfect"] == 0 and r["mean_matched_instance_iou"] < 0.50]
        cm_fail = [r for r in best_cm["rows"] if (r["markers"] != r["gt_instances"]) or (r["case_type"] in {"fragmented", "merged", "mixed"})]
        fail_root = vis_root
        (fail_root / "metadata_oracle_failures").mkdir(parents=True, exist_ok=True)
        (fail_root / "center_map_extraction_failures").mkdir(parents=True, exist_ok=True)
        for r in meta_fail[:200]:
            _write_json(fail_root / "metadata_oracle_failures" / f"{r['sample']}.json", r)
        for r in cm_fail[:200]:
            _write_json(fail_root / "center_map_extraction_failures" / f"{r['sample']}.json", r)

    print(f"Best watershed variant (metadata upper bound): {best_variant}")
    print(f"Metadata upper bound mean IoU: {best_metadata['mean_matched_instance_iou']:.4f}")
    print(f"Best center-map method: {best_cm['method']} params={best_cm['params']}")
    print(f"Best center-map mean IoU: {best_cm['mean_matched_instance_iou']:.4f}")
    print(f"Outputs: {out_root}")


if __name__ == "__main__":
    main()
