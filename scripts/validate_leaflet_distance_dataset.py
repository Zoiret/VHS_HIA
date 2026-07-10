from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


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


def _read_split(split_txt: Path) -> list[str]:
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


def _center_crop_to_shape(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == target_h and w == target_w:
        return arr
    if h < target_h or w < target_w:
        raise SystemExit(f"Cannot crop {target_h}x{target_w} from {h}x{w}")
    y0 = (h - target_h) // 2
    x0 = (w - target_w) // 2
    if arr.ndim == 2:
        return arr[y0 : y0 + target_h, x0 : x0 + target_w]
    return arr[y0 : y0 + target_h, x0 : x0 + target_w, :]


def _local_maxima_points(map01: np.ndarray, mask01: np.ndarray, *, min_value: float = 0.3) -> list[tuple[int, int, float]]:
    m = map01.copy().astype(np.float32)
    m[~mask01.astype(bool)] = 0.0
    if float(m.max()) <= 0.0:
        return []
    dil = cv2.dilate(m, np.ones((3, 3), np.uint8))
    peaks = (m == dil) & (m >= float(min_value)) & (m > 0.0) & mask01.astype(bool)
    peaks_u8 = peaks.astype(np.uint8)
    n, labels = cv2.connectedComponents(peaks_u8)
    pts = []
    for lab in range(1, int(n)):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        vals = m[ys, xs]
        k = int(np.argmax(vals))
        y = int(ys[k])
        x = int(xs[k])
        pts.append((y, x, float(m[y, x])))
    pts.sort(key=lambda t: t[2], reverse=True)
    return pts


def _iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-7) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = float(np.sum(aa & bb))
    union = float(np.sum(aa | bb))
    return float((inter + eps) / (union + eps))


def _perm_assignment_max_iou(iou: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    gt_k, pred_k = iou.shape[0], iou.shape[1]
    if gt_k == 0 or pred_k == 0:
        return 0.0, []
    k = min(int(gt_k), int(pred_k))
    best = -1.0
    best_pairs: list[tuple[int, int]] = []
    import itertools

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


def _iou_matrix(gt_labels: np.ndarray, pred_labels: np.ndarray, gt_k: int, pred_k: int) -> np.ndarray:
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


def _oracle_watershed_from_distance(leaf_union01: np.ndarray, dist01: np.ndarray, *, max_instances: int = 3) -> np.ndarray:
    mask = leaf_union01.astype(np.uint8)
    if int(mask.sum()) == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    d = dist01.astype(np.float32)
    d[mask == 0] = 0.0

    inst_guess = int(max_instances)
    pts = []
    for thr in [0.5, 0.3, 0.1]:
        pts = _local_maxima_points(d, mask, min_value=float(thr))
        if len(pts) >= 1:
            break
    if not pts:
        yx = np.argwhere(mask > 0)
        y, x = int(yx[0][0]), int(yx[0][1])
        pts = [(y, x, float(d[y, x]))]

    pts = pts[: int(inst_guess)]

    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[mask == 0] = 1
    for k, (y, x, _) in enumerate(pts, start=2):
        markers[int(y), int(x)] = int(k)

    topo = (1.0 - np.clip(d, 0.0, 1.0)) * 255.0
    topo_u8 = topo.astype(np.uint8)
    topo3 = cv2.cvtColor(topo_u8, cv2.COLOR_GRAY2BGR)
    cv2.watershed(topo3, markers)

    out = np.zeros(mask.shape, dtype=np.uint8)
    lbls = sorted([int(x) for x in np.unique(markers) if int(x) > 1])
    for new_i, lab in enumerate(lbls, start=1):
        out[markers == lab] = np.uint8(new_i)
    out[mask == 0] = 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance-root", type=Path, default=Path("datasets/converted_leaflet_distance"))
    ap.add_argument("--source-instances-root", type=Path, default=Path("datasets/converted_leaflet_instances"))
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--oracle-watershed", action="store_true")
    ap.add_argument("--perfect-iou-threshold", type=float, default=0.90)
    args = ap.parse_args()

    d_root = args.distance_root.resolve()
    s_root = args.source_instances_root.resolve()

    for p in [d_root / "train.txt", d_root / "val.txt", d_root / "test.txt"]:
        if not p.exists():
            raise SystemExit(f"Missing split file: {p}")
    for p in [s_root / "train.txt", s_root / "val.txt", s_root / "test.txt"]:
        if not p.exists():
            raise SystemExit(f"Missing source split file: {p}")

    d_train = _read_split(d_root / "train.txt")
    d_val = _read_split(d_root / "val.txt")
    d_test = _read_split(d_root / "test.txt")
    s_train = _read_split(s_root / "train.txt")
    s_val = _read_split(s_root / "val.txt")
    s_test = _read_split(s_root / "test.txt")

    split_same = {"train_same": set(d_train) == set(s_train), "val_same": set(d_val) == set(s_val), "test_same": set(d_test) == set(s_test)}
    split_overlap = {
        "train_val": int(len(set(d_train) & set(d_val))),
        "train_test": int(len(set(d_train) & set(d_test))),
        "val_test": int(len(set(d_val) & set(d_test))),
    }

    errors = []
    rows = []

    oracle_stats = {
        "enabled": bool(args.oracle_watershed),
        "perfect_iou_threshold": float(args.perfect_iou_threshold),
        "samples": 0,
        "exact_instance_count_accuracy": None,
        "merged": None,
        "fragmented": None,
        "mean_matched_instance_iou": None,
        "perfect_recovery_rate": None,
    }
    oracle_samples = []

    def check_one(sid: str, split: str) -> None:
        img_p = d_root / "images" / f"{sid}.png"
        sem_p = d_root / "semantic_masks" / f"{sid}.png"
        dist_p = d_root / "distance_maps" / f"{sid}.png"
        center_p = d_root / "center_maps" / f"{sid}.png"
        inst_p = s_root / "instance_masks" / f"{sid}.png"
        s_sem_p = s_root / "semantic_masks" / f"{sid}.png"

        if not img_p.exists() or not sem_p.exists() or not dist_p.exists() or not inst_p.exists() or not s_sem_p.exists():
            errors.append({"sample": sid, "split": split, "error": "missing_files"})
            return

        try:
            img = _read_rgb_u8(img_p)
            sem = _read_u8(sem_p)
            dist_u16 = _read_u16(dist_p)
            inst_src = _read_u8(inst_p)
            src_sem_src = _read_u8(s_sem_p)
        except Exception as e:
            errors.append({"sample": sid, "split": split, "error": f"read_error: {e}"})
            return

        h, w = sem.shape[:2]
        if img.shape[:2] != (h, w) or dist_u16.shape[:2] != (h, w):
            errors.append({"sample": sid, "split": split, "error": "shape_mismatch"})
            return

        inst = _center_crop_to_shape(inst_src, h, w)
        src_sem = _center_crop_to_shape(src_sem_src, h, w)
        if inst.shape[:2] != (h, w) or src_sem.shape[:2] != (h, w):
            errors.append({"sample": sid, "split": split, "error": "shape_mismatch_after_crop"})
            return

        sem_ids = set(np.unique(sem).tolist())
        if not sem_ids.issubset({0, 1, 2}):
            errors.append({"sample": sid, "split": split, "error": f"bad_semantic_ids={sorted(list(sem_ids))}"})
            return

        inst_ids = set(np.unique(inst).tolist())
        if not inst_ids.issubset({0, 1, 2, 3}):
            errors.append({"sample": sid, "split": split, "error": f"bad_instance_ids={sorted(list(inst_ids))}"})
            return

        leaf_union = sem == 1
        ring = sem == 2

        dist01 = dist_u16.astype(np.float32) / 65535.0
        if np.any(~np.isfinite(dist01)):
            errors.append({"sample": sid, "split": split, "error": "nan_or_inf_in_distance"})
            return
        if int(np.sum(dist01[~leaf_union] != 0.0)) != 0:
            errors.append({"sample": sid, "split": split, "error": "distance_nonzero_outside_leaflet"})
            return

        present = [i for i in [1, 2, 3] if int(np.sum(inst == i)) > 0]
        for iid in present:
            m = inst == iid
            if float(dist01[m].max()) < 0.95:
                errors.append({"sample": sid, "split": split, "error": f"instance_max_lt_0p95 inst={iid} max={float(dist01[m].max())}"})
                return
            if int(np.sum(dist01[m] > 0.0)) == 0:
                errors.append({"sample": sid, "split": split, "error": f"instance_all_zero inst={iid}"})
                return

        if center_p.exists():
            cm_u16 = _read_u16(center_p)
            if cm_u16.shape != sem.shape:
                errors.append({"sample": sid, "split": split, "error": "center_shape_mismatch"})
                return
            cm = cm_u16.astype(np.float32) / 65535.0
            if np.any(~np.isfinite(cm)):
                errors.append({"sample": sid, "split": split, "error": "nan_or_inf_in_center"})
                return
            if int(np.sum(cm[~leaf_union] != 0.0)) != 0:
                errors.append({"sample": sid, "split": split, "error": "center_nonzero_outside_leaflet"})
                return

        rows.append({"sample": sid, "split": split, "instance_count": len(present), "ok": 1, "error": ""})

        if args.oracle_watershed:
            gt_ids = [i for i in [1, 2, 3] if int(np.sum(inst == i)) > 0]
            gt_k = int(len(gt_ids))
            pred_lbl = _oracle_watershed_from_distance(leaf_union.astype(np.uint8), dist01, max_instances=3)
            pred_ids = sorted([i for i in np.unique(pred_lbl).tolist() if int(i) > 0])
            pred_k = int(len(pred_ids))

            merged = int(gt_k >= 2 and pred_k < gt_k)
            fragmented = int(pred_k > gt_k)

            iou_mat = _iou_matrix(inst.astype(np.int32), pred_lbl.astype(np.int32), gt_k, pred_k) if (gt_k > 0 and pred_k > 0) else np.zeros((gt_k, pred_k))
            sum_iou, _ = _perm_assignment_max_iou(iou_mat) if (gt_k > 0 and pred_k > 0) else (0.0, [])
            mean_iou = float(sum_iou / max(gt_k, 1)) if gt_k > 0 else 0.0

            perfect = int((pred_k == gt_k) and (mean_iou >= float(args.perfect_iou_threshold)))
            oracle_samples.append(
                {
                    "sample": sid,
                    "split": split,
                    "gt_instances": gt_k,
                    "pred_instances": pred_k,
                    "merged": merged,
                    "fragmented": fragmented,
                    "mean_matched_instance_iou": mean_iou,
                    "perfect": perfect,
                }
            )

    for split, ids in [("train", d_train), ("val", d_val), ("test", d_test)]:
        for sid in ids:
            check_one(sid, split)

    if args.oracle_watershed and oracle_samples:
        oracle_stats["samples"] = int(len(oracle_samples))
        oracle_stats["exact_instance_count_accuracy"] = float(sum(1 for s in oracle_samples if s["gt_instances"] == s["pred_instances"]) / len(oracle_samples))
        oracle_stats["merged"] = int(sum(int(s["merged"]) for s in oracle_samples))
        oracle_stats["fragmented"] = int(sum(int(s["fragmented"]) for s in oracle_samples))
        oracle_stats["mean_matched_instance_iou"] = float(np.mean([float(s["mean_matched_instance_iou"]) for s in oracle_samples]))
        oracle_stats["perfect_recovery_rate"] = float(sum(int(s["perfect"]) for s in oracle_samples) / len(oracle_samples))

    report = {
        "distance_root": str(d_root),
        "source_instances_root": str(s_root),
        "split_counts": {"train": len(d_train), "val": len(d_val), "test": len(d_test)},
        "split_overlap": split_overlap,
        "split_same_as_source": split_same,
        "samples_checked": int(len(d_train) + len(d_val) + len(d_test)),
        "errors_count": int(len(errors)),
        "errors": errors[:200],
        "oracle_watershed": oracle_stats,
    }

    out_json = d_root / "validation_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_csv = d_root / "validation_report.csv"
    if rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    if args.oracle_watershed and oracle_samples:
        oracle_csv = d_root / "oracle_watershed_results.csv"
        with oracle_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(oracle_samples[0].keys()))
            w.writeheader()
            for r in oracle_samples:
                w.writerow(r)
        report["oracle_watershed"]["results_csv"] = str(oracle_csv)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Checked: {report['samples_checked']}")
    print(f"Errors: {report['errors_count']}")
    print(f"Split overlap: {split_overlap}")
    print(f"Split same as source: {split_same}")
    if args.oracle_watershed:
        print(f"Oracle exact count acc: {oracle_stats['exact_instance_count_accuracy']}")
        print(f"Oracle mean matched IoU: {oracle_stats['mean_matched_instance_iou']}")
        print(f"Oracle perfect rate: {oracle_stats['perfect_recovery_rate']}")
    print(f"Report: {out_json}")
    if args.strict and report["errors_count"] != 0:
        raise SystemExit("Validation failed (strict)")


if __name__ == "__main__":
    main()
