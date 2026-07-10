from __future__ import annotations

import argparse
import csv
import json
import math
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


def _write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _write_u8_png(path: Path, mask_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask_u8.astype(np.uint8))


def _write_u16_png(path: Path, arr_u16: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr_u16.astype(np.uint16))


def _normalize_dt_inside_instance(dist: np.ndarray, mask01: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int] | None]:
    inside = mask01.astype(bool)
    if not bool(np.any(inside)):
        return np.zeros_like(dist, dtype=np.float32), 0.0, None
    d = dist.astype(np.float32)
    d[~inside] = 0.0
    max_d = float(d.max())
    if max_d <= 0.0:
        return np.zeros_like(dist, dtype=np.float32), 0.0, None

    if max_d <= 1.0:
        norm = np.zeros_like(d, dtype=np.float32)
        norm[inside] = 1.0
    else:
        norm = np.zeros_like(d, dtype=np.float32)
        norm[inside] = (d[inside] - 1.0) / (max_d - 1.0)
        norm = np.clip(norm, 0.0, 1.0)

    yx = None
    max_pos = np.argwhere(d == max_d)
    if max_pos.size > 0:
        yx = (int(max_pos[0][0]), int(max_pos[0][1]))
    return norm, max_d, yx


def _gaussian_add(map01: np.ndarray, center_yx: tuple[int, int], sigma: float) -> None:
    y0, x0 = int(center_yx[0]), int(center_yx[1])
    h, w = map01.shape[:2]
    s = float(sigma)
    if s <= 0.0:
        if 0 <= y0 < h and 0 <= x0 < w:
            map01[y0, x0] = 1.0
        return
    r = int(math.ceil(3.0 * s))
    y1 = max(0, y0 - r)
    y2 = min(h, y0 + r + 1)
    x1 = max(0, x0 - r)
    x2 = min(w, x0 + r + 1)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    g = np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2.0 * s * s)).astype(np.float32)
    patch = map01[y1:y2, x1:x2]
    np.maximum(patch, g, out=patch)


def _write_split(out_root: Path, name: str, ids: list[str]) -> None:
    out = out_root / name
    with out.open("w", encoding="utf-8") as f:
        for sid in ids:
            f.write(f"images/{sid}.png\tsemantic_masks/{sid}.png\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("datasets/converted_leaflet_instances"))
    ap.add_argument("--output-root", type=Path, default=Path("datasets/converted_leaflet_distance"))
    ap.add_argument("--curation-json", type=Path, default=Path("server_assets/curation/curation_result.json"))
    ap.add_argument("--crop-size", type=int, default=768)
    ap.add_argument("--center-sigma", type=int, default=4)
    ap.add_argument("--no-center-map", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_root = args.input_root.resolve()
    out_root = args.output_root.resolve()

    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output dir exists and is not empty: {out_root} (use --overwrite)")

    images_out = out_root / "images"
    semantic_out = out_root / "semantic_masks"
    dist_out = out_root / "distance_maps"
    center_out = out_root / "center_maps"
    meta_out = out_root / "metadata"
    for p in [images_out, semantic_out, dist_out, meta_out]:
        _safe_mkdir(p)
    if not args.no_center_map:
        _safe_mkdir(center_out)

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
        "distance_map_format": "png_u16",
        "distance_map_range": "0..65535",
        "distance_map_normalization": "per-instance: (dt-1)/(maxdt-1) if maxdt>1 else 1 inside instance; then pixelwise max across instances; outside leaflet=0",
        "center_map": (not bool(args.no_center_map)),
        "center_sigma": int(args.center_sigma),
        "center_map_format": "png_u16" if not args.no_center_map else None,
        "center_map_range": "0..65535" if not args.no_center_map else None,
        "total": int(len(items)),
        "success": 0,
        "failed": 0,
        "failures": [],
        "split_counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "split_overlap": split_overlaps,
        "quality_counts": Counter(quality_map.values()),
        "instance_count_distribution": Counter(),
        "warnings_count": 0,
    }

    manifest_rows = []

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

        leaf_union = (sem_c == 1)
        ring = (sem_c == 2)
        present = [i for i in [1, 2, 3] if int(np.sum(inst_c == i)) > 0]
        summary["instance_count_distribution"][str(len(present))] += 1

        dist_map = np.zeros(sem_c.shape, dtype=np.float32)
        center_map = np.zeros(sem_c.shape, dtype=np.float32) if not args.no_center_map else None
        warnings = []

        per_inst = []
        for iid in [1, 2, 3]:
            m01 = (inst_c == iid).astype(np.uint8)
            if int(m01.sum()) == 0:
                continue
            dt = cv2.distanceTransform(m01, cv2.DIST_L2, 3)
            norm, max_d, center_yx = _normalize_dt_inside_instance(dt, m01)
            dist_map = np.maximum(dist_map, norm)
            if center_map is not None and center_yx is not None:
                _gaussian_add(center_map, center_yx, float(args.center_sigma))
            per_inst.append(
                {
                    "instance_id": int(iid),
                    "area": int(np.sum(m01 > 0)),
                    "max_dt": float(max_d),
                    "center_yx": [int(center_yx[0]), int(center_yx[1])] if center_yx is not None else None,
                }
            )

        dist_map[~leaf_union] = 0.0
        if center_map is not None:
            center_map[~leaf_union] = 0.0

        dist_u16 = np.clip(dist_map, 0.0, 1.0)
        dist_u16 = (dist_u16 * 65535.0 + 0.5).astype(np.uint16)
        center_u16 = None
        if center_map is not None:
            cm = np.clip(center_map, 0.0, 1.0)
            center_u16 = (cm * 65535.0 + 0.5).astype(np.uint16)

        out_img = images_out / f"{it.sample_id}.png"
        out_sem = semantic_out / f"{it.sample_id}.png"
        out_dist = dist_out / f"{it.sample_id}.png"
        out_center = center_out / f"{it.sample_id}.png" if center_u16 is not None else None
        out_meta = meta_out / f"{it.sample_id}.json"

        _write_rgb_png(out_img, img_c)
        _write_u8_png(out_sem, sem_c)
        _write_u16_png(out_dist, dist_u16)
        if out_center is not None and center_u16 is not None:
            _write_u16_png(out_center, center_u16)

        meta = {
            "sample": it.sample_id,
            "split": it.split,
            "quality": quality_map.get(it.sample_id, None),
            "crop_size": int(args.crop_size),
            "distance_map_format": "png_u16",
            "center_map_format": "png_u16" if center_u16 is not None else None,
            "source": {
                "image": str(it.image_path),
                "semantic_mask": str(it.semantic_path),
                "instance_mask": str(it.instance_path),
                "instance_metadata": str(it.metadata_path),
            },
            "instance_count": int(len(present)),
            "source_instance_ids": present,
            "instances": per_inst,
            "warnings": warnings,
            "paths": {
                "distance_map_rel": f"distance_maps/{it.sample_id}.png",
                "center_map_rel": (f"center_maps/{it.sample_id}.png" if out_center is not None else None),
            },
        }
        out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        if warnings:
            summary["warnings_count"] += 1

        manifest_rows.append(
            {
                "sample": it.sample_id,
                "split": it.split,
                "quality": quality_map.get(it.sample_id, ""),
                "instance_count": int(len(present)),
                "instance_1_area": int(np.sum(inst_c == 1)),
                "instance_2_area": int(np.sum(inst_c == 2)),
                "instance_3_area": int(np.sum(inst_c == 3)),
                "distance_rel": f"distance_maps/{it.sample_id}.png",
                "center_rel": (f"center_maps/{it.sample_id}.png" if out_center is not None else ""),
                "image_rel": f"images/{it.sample_id}.png",
                "semantic_rel": f"semantic_masks/{it.sample_id}.png",
                "metadata_rel": f"metadata/{it.sample_id}.json",
                "conversion_ok": 1,
            }
        )

        summary["success"] += 1

    _write_split(out_root, "train.txt", train_ids)
    _write_split(out_root, "val.txt", val_ids)
    _write_split(out_root, "test.txt", test_ids)

    manifest_csv = out_root / "distance_dataset_manifest.csv"
    if manifest_rows:
        with manifest_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            for r in manifest_rows:
                w.writerow(r)

    summary_path = out_root / "distance_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total: {summary['total']}")
    print(f"Success: {summary['success']}")
    print(f"Failed: {summary['failed']}")
    print(f"Distance maps: {dist_out} (png uint16)")
    if not args.no_center_map:
        print(f"Center maps: {center_out} (png uint16)")
    print(f"Output: {out_root}")
    print(f"Manifest: {manifest_csv}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

