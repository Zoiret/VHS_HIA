from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SourceItem:
    sample_id: str
    split: str
    image_path: Path
    semantic_path: Path
    instance_path: Path
    metadata_path: Path


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
        raise SystemExit(f"Cannot center-crop {crop_h}x{crop_w} from {h}x{w}")
    y0 = (h - crop_h) // 2 if h > crop_h else 0
    x0 = (w - crop_w) // 2 if w > crop_w else 0
    return image_rgb[y0 : y0 + crop_h, x0 : x0 + crop_w], mask_u8[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _center_crop3(image_rgb: np.ndarray, sem_u8: np.ndarray, inst_u8: np.ndarray, crop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img2, sem2 = _center_crop_pair(image_rgb, sem_u8, crop)
    _, inst2 = _center_crop_pair(image_rgb, inst_u8, crop)
    return img2, sem2, inst2


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


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _write_u8_png(path: Path, mask_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask_u8.astype(np.uint8))


def _kernel(radius: int) -> np.ndarray:
    r = int(radius)
    k = 2 * r + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _distance_to_mask(mask01: np.ndarray) -> np.ndarray:
    src = (mask01.astype(np.uint8) > 0).astype(np.uint8)
    inv = (1 - src).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return dist.astype(np.float32)


def _compute_boundary(
    inst_u8: np.ndarray,
    leaf_union01: np.ndarray,
    ring01: np.ndarray,
    *,
    boundary_width: int,
    max_gap: int,
) -> tuple[np.ndarray, dict]:
    leaf_union01 = (leaf_union01.astype(np.uint8) > 0).astype(np.uint8)
    ring01 = (ring01.astype(np.uint8) > 0).astype(np.uint8)
    safe_leaf = (leaf_union01 > 0) & (ring01 == 0)

    present = [i for i in [1, 2, 3] if int(np.sum(inst_u8 == i)) > 0]
    if len(present) <= 1:
        return np.zeros_like(inst_u8, dtype=np.uint8), {"present_instances": present, "pairs_considered": 0, "pairs_close": 0}

    masks01 = {i: (inst_u8 == i).astype(np.uint8) for i in present}
    dists = {i: _distance_to_mask(masks01[i]) for i in present}

    pairs = []
    for a_idx in range(len(present)):
        for b_idx in range(a_idx + 1, len(present)):
            pairs.append((present[a_idx], present[b_idx]))

    boundary01 = np.zeros_like(inst_u8, dtype=np.uint8)
    pairs_close = 0
    pairs_considered = 0
    pairs_no_boundary = []

    if int(max_gap) <= 0:
        for i, j in pairs:
            pairs_considered += 1
            mi = masks01[i]
            mj = masks01[j]
            di = cv2.dilate(mi, _kernel(1), iterations=1)
            dj = cv2.dilate(mj, _kernel(1), iterations=1)
            contact = (di > 0) & (dj > 0) & safe_leaf
            if bool(np.any(contact)):
                pairs_close += 1
                boundary01[contact] = 1
    else:
        dist_stack = np.stack([dists[i] for i in present], axis=0)
        order = np.argsort(dist_stack, axis=0)
        best_idx = order[0]
        second_idx = order[1]
        best = np.take_along_axis(dist_stack, best_idx[None, :, :], axis=0)[0]
        second = np.take_along_axis(dist_stack, second_idx[None, :, :], axis=0)[0]
        diff = second - best

        core = safe_leaf & (second <= float(max_gap)) & (diff <= 1.0)
        boundary01[core] = 1

        for i, j in pairs:
            pairs_considered += 1
            dist_i = dists[i]
            dist_j = dists[j]
            near_both = (dist_i <= float(max_gap)) & (dist_j <= float(max_gap)) & safe_leaf
            if not bool(np.any(near_both)):
                continue
            pairs_close += 1
            if not bool(np.any(core & near_both)):
                pairs_no_boundary.append({"i": int(i), "j": int(j)})

    width = int(boundary_width)
    if width <= 0:
        return np.zeros_like(inst_u8, dtype=np.uint8), {"present_instances": present, "pairs_considered": pairs_considered, "pairs_close": pairs_close}

    if width > 1 and bool(np.any(boundary01)):
        r = max(0, (width - 1) // 2)
        if r > 0:
            boundary01 = (cv2.dilate(boundary01 * 255, _kernel(r), iterations=1) > 0).astype(np.uint8)

    boundary01 = (boundary01 > 0) & safe_leaf
    boundary01 = boundary01.astype(np.uint8)

    return boundary01, {
        "present_instances": present,
        "pairs_considered": int(pairs_considered),
        "pairs_close": int(pairs_close),
        "pairs_no_boundary": pairs_no_boundary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("datasets/converted_leaflet_instances"))
    ap.add_argument("--output-root", type=Path, default=Path("datasets/converted_leaflet_boundary"))
    ap.add_argument("--curation-json", type=Path, default=Path("server_assets/curation/curation_result.json"))
    ap.add_argument("--boundary-width", type=int, default=3)
    ap.add_argument("--max-boundary-gap", type=int, default=2)
    ap.add_argument("--crop-size", type=int, default=768)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_root = args.input_root.resolve()
    out_root = args.output_root.resolve()

    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output dir exists and is not empty: {out_root} (use --overwrite)")

    images_out = out_root / "images"
    masks_out = out_root / "masks"
    meta_out = out_root / "metadata"
    for p in [images_out, masks_out, meta_out]:
        _safe_mkdir(p)

    train_txt = in_root / "train.txt"
    val_txt = in_root / "val.txt"
    test_txt = in_root / "test.txt"
    for p in [train_txt, val_txt, test_txt]:
        if not p.exists():
            raise SystemExit(f"Missing split file: {p}")

    train_ids = _read_split(train_txt)
    val_ids = _read_split(val_txt)
    test_ids = _read_split(test_txt)

    split_overlaps = {
        "train_val": int(len(set(train_ids) & set(val_ids))),
        "train_test": int(len(set(train_ids) & set(test_ids))),
        "val_test": int(len(set(val_ids) & set(test_ids))),
    }
    if any(v != 0 for v in split_overlaps.values()):
        raise SystemExit(f"Split overlap in source dataset: {split_overlaps}")

    quality_map = {}
    curation_path = args.curation_json.resolve()
    if curation_path.exists():
        cur = json.loads(curation_path.read_text(encoding="utf-8"))
        if isinstance(cur, dict):
            for k in ["clean", "medium", "bad"]:
                for sid in cur.get(k, []) or []:
                    quality_map[str(sid)] = k

    items: list[SourceItem] = []
    for split, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        for sid in ids:
            items.append(
                SourceItem(
                    sample_id=sid,
                    split=split,
                    image_path=in_root / "images" / f"{sid}.png",
                    semantic_path=in_root / "semantic_masks" / f"{sid}.png",
                    instance_path=in_root / "instance_masks" / f"{sid}.png",
                    metadata_path=in_root / "metadata" / f"{sid}.json",
                )
            )

    summary = {
        "input_root": str(in_root),
        "output_root": str(out_root),
        "crop_size": int(args.crop_size),
        "boundary_width": int(args.boundary_width),
        "max_boundary_gap": int(args.max_boundary_gap),
        "method": "distance-transform bisector inside leaflet union; only if instances are within max_gap (or touching if max_gap=0); then dilate to boundary_width",
        "label_priority": [3, 2, 1, 0],
        "total": int(len(items)),
        "success": 0,
        "failed": 0,
        "failures": [],
        "split_counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "split_overlap": split_overlaps,
        "quality_counts": Counter(quality_map.values()),
        "instance_count_distribution": Counter(),
        "samples_with_boundary": 0,
        "samples_without_boundary": 0,
        "boundary_pixels": {"mean": None, "median": None, "std": None},
        "boundary_fraction": {"mean": None, "median": None, "std": None, "p95": None},
        "suspicious_large_boundary_fraction": [],
        "close_instances_but_no_boundary": [],
        "ring_pixels_preserved_checks": 0,
    }

    manifest_rows = []
    boundary_pixels_all = []
    boundary_frac_all = []
    suspicious = []

    for it in items:
        try:
            img = _read_rgb_u8(it.image_path)
            sem = _read_u8(it.semantic_path)
            inst = _read_u8(it.instance_path)
        except Exception as e:
            summary["failed"] += 1
            summary["failures"].append({"sample": it.sample_id, "split": it.split, "reason": "read_error", "error": str(e)})
            continue

        img_c, sem_c, inst_c = _center_crop3(img, sem, inst, int(args.crop_size))

        sem_ids = set(np.unique(sem_c).tolist())
        inst_ids = set(np.unique(inst_c).tolist())
        if not sem_ids.issubset({0, 1, 2}):
            summary["failed"] += 1
            summary["failures"].append({"sample": it.sample_id, "split": it.split, "reason": "bad_semantic_ids", "ids": sorted(list(sem_ids))})
            continue
        if not inst_ids.issubset({0, 1, 2, 3}):
            summary["failed"] += 1
            summary["failures"].append({"sample": it.sample_id, "split": it.split, "reason": "bad_instance_ids", "ids": sorted(list(inst_ids))})
            continue

        leaf_union01 = (sem_c == 1).astype(np.uint8)
        ring01 = (sem_c == 2).astype(np.uint8)
        present = [i for i in [1, 2, 3] if int(np.sum(inst_c == i)) > 0]
        summary["instance_count_distribution"][str(len(present))] += 1

        boundary01, dbg = _compute_boundary(
            inst_c,
            leaf_union01,
            ring01,
            boundary_width=int(args.boundary_width),
            max_gap=int(args.max_boundary_gap),
        )

        leaf_union = leaf_union01.astype(bool)
        ring = ring01.astype(bool)
        boundary = boundary01.astype(bool)

        if bool(np.any(boundary)):
            summary["samples_with_boundary"] += 1
        else:
            summary["samples_without_boundary"] += 1

        leaflet_pixels = int(np.sum(leaf_union))
        boundary_pixels = int(np.sum(boundary))
        frac = float(boundary_pixels / max(leaflet_pixels, 1))
        boundary_pixels_all.append(boundary_pixels)
        boundary_frac_all.append(frac)

        if dbg.get("pairs_no_boundary"):
            summary["close_instances_but_no_boundary"].append({"sample": it.sample_id, "split": it.split, "pairs": dbg.get("pairs_no_boundary")})

        target = np.zeros_like(sem_c, dtype=np.uint8)
        target[leaf_union] = 1
        target[ring] = 2
        target[boundary] = 3
        target[ring] = 2

        if int(np.sum((target == 3) & (target == 2))) != 0:
            summary["failed"] += 1
            summary["failures"].append({"sample": it.sample_id, "split": it.split, "reason": "boundary_overlaps_ring"})
            continue

        union_target = (target == 1) | (target == 3)
        mismatch_union = int(np.sum(union_target != leaf_union))
        if mismatch_union != 0:
            summary["failed"] += 1
            summary["failures"].append({"sample": it.sample_id, "split": it.split, "reason": "leaflet_union_mismatch", "pixels": mismatch_union})
            continue

        mismatch_ring = int(np.sum((target == 2) != ring))
        summary["ring_pixels_preserved_checks"] += 1
        if mismatch_ring != 0:
            summary["failed"] += 1
            summary["failures"].append({"sample": it.sample_id, "split": it.split, "reason": "ring_mismatch", "pixels": mismatch_ring})
            continue

        out_img = images_out / f"{it.sample_id}.png"
        out_mask = masks_out / f"{it.sample_id}.png"
        out_meta = meta_out / f"{it.sample_id}.json"

        _write_rgb_png(out_img, img_c)
        _write_u8_png(out_mask, target)

        src_meta = {}
        if it.metadata_path.exists():
            try:
                src_meta = json.loads(it.metadata_path.read_text(encoding="utf-8"))
            except Exception:
                src_meta = {}

        inst_stats = []
        for iid in [1, 2, 3]:
            area = int(np.sum(inst_c == iid))
            inst_stats.append({"instance_id": iid, "present": bool(area > 0), "area": area})

        meta = {
            "sample": it.sample_id,
            "split": it.split,
            "quality": quality_map.get(it.sample_id, None),
            "crop_size": int(args.crop_size),
            "boundary_width": int(args.boundary_width),
            "max_boundary_gap": int(args.max_boundary_gap),
            "source": {
                "image": str(it.image_path),
                "semantic_mask": str(it.semantic_path),
                "instance_mask": str(it.instance_path),
                "instance_metadata": str(it.metadata_path),
            },
            "source_instance_summary": {
                "leaflet_source_objects_total": src_meta.get("leaflet_source_objects_total"),
                "leaflet_selected_instances": src_meta.get("leaflet_selected_instances"),
                "leaflet_selected_source_object_ids": src_meta.get("leaflet_selected_source_object_ids"),
            },
            "present_instances": present,
            "instance_stats": inst_stats,
            "leaflet_pixels": leaflet_pixels,
            "ring_pixels": int(np.sum(ring)),
            "boundary_pixels": boundary_pixels,
            "boundary_fraction": frac,
            "debug": dbg,
        }
        out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest_rows.append(
            {
                "sample": it.sample_id,
                "split": it.split,
                "quality": quality_map.get(it.sample_id, ""),
                "instance_count": int(len(present)),
                "leaflet_pixels": leaflet_pixels,
                "ring_pixels": int(np.sum(ring)),
                "boundary_pixels": boundary_pixels,
                "boundary_fraction": frac,
                "close_pairs_considered": int(dbg.get("pairs_considered", 0)),
                "close_pairs": int(dbg.get("pairs_close", 0)),
                "close_pairs_no_boundary": int(len(dbg.get("pairs_no_boundary") or [])),
                "image_rel": f"images/{it.sample_id}.png",
                "mask_rel": f"masks/{it.sample_id}.png",
                "metadata_rel": f"metadata/{it.sample_id}.json",
                "conversion_ok": 1,
            }
        )

        summary["success"] += 1

    if boundary_pixels_all:
        arr = np.asarray(boundary_pixels_all, dtype=np.float64)
        summary["boundary_pixels"] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=0)),
        }
    if boundary_frac_all:
        arr = np.asarray(boundary_frac_all, dtype=np.float64)
        p95 = float(np.percentile(arr, 95))
        summary["boundary_fraction"] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=0)),
            "p95": p95,
        }
        for r in manifest_rows:
            if float(r["boundary_fraction"]) > p95:
                suspicious.append(r["sample"])
        summary["suspicious_large_boundary_fraction"] = sorted(list(set(suspicious)))

    def _write_split(name: str, ids: list[str]) -> None:
        out = out_root / name
        with out.open("w", encoding="utf-8") as f:
            for sid in ids:
                f.write(f"images/{sid}.png\tmasks/{sid}.png\n")

    _write_split("train.txt", train_ids)
    _write_split("val.txt", val_ids)
    _write_split("test.txt", test_ids)

    manifest_csv = out_root / "boundary_dataset_manifest.csv"
    if manifest_rows:
        with manifest_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            for r in manifest_rows:
                w.writerow(r)

    summary_path = out_root / "boundary_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total: {summary['total']}")
    print(f"Success: {summary['success']}")
    print(f"Failed: {summary['failed']}")
    print(f"Boundary width: {args.boundary_width}px")
    print(f"Max gap: {args.max_boundary_gap}px")
    print(f"Output: {out_root}")
    print(f"Manifest: {manifest_csv}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

