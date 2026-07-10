from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import shutil
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _norm_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def _is_leaflet_title(title: str) -> bool:
    t = _norm_title(title)
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


def _is_annulus_title(title: str) -> bool:
    t = _norm_title(title)
    if not t:
        return False
    ring_markers = [
        "aortic valve base",
        "valve base",
        "fibrous_ring",
        "fibrous ring",
        "annulus",
        "ring",
        "фиброз",
        "кольц",
    ]
    return any(m in t for m in ring_markers)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_size_from_ann_or_image(ann: dict, image_path: Path) -> tuple[int, int]:
    size = ann.get("size") or {}
    h = size.get("height")
    w = size.get("width")
    if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
        return int(h), int(w)
    try:
        with Image.open(image_path) as img:
            w2, h2 = img.size
            return int(h2), int(w2)
    except Exception:
        img_bgr = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise
        return int(img_bgr.shape[0]), int(img_bgr.shape[1])


def _bitmap_to_full_mask(bitmap: dict, full_h: int, full_w: int) -> tuple[np.ndarray, bool]:
    origin = bitmap.get("origin") or bitmap.get("origin_xy") or [0, 0]
    if not (isinstance(origin, list) and len(origin) == 2):
        origin = [0, 0]
    x0 = int(origin[0])
    y0 = int(origin[1])

    data_b64 = bitmap.get("data")
    if not isinstance(data_b64, str) or not data_b64:
        return np.zeros((full_h, full_w), dtype=np.uint8), False

    raw = base64.b64decode(data_b64)
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        try:
            raw2 = zlib.decompress(raw)
            decoded = cv2.imdecode(np.frombuffer(raw2, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception:
            decoded = None
    if decoded is None:
        return np.zeros((full_h, full_w), dtype=np.uint8), False

    if decoded.ndim == 2:
        local = decoded > 0
    else:
        if decoded.shape[2] == 4:
            local = decoded[:, :, 3] > 0
        else:
            local = np.any(decoded[:, :, :3] > 0, axis=2)

    local = local.astype(np.uint8)
    out = np.zeros((full_h, full_w), dtype=np.uint8)
    h, w = local.shape[:2]

    x1 = max(0, x0)
    y1 = max(0, y0)
    x2 = min(full_w, x0 + w)
    y2 = min(full_h, y0 + h)
    if x2 <= x1 or y2 <= y1:
        return out, True

    lx1 = x1 - x0
    ly1 = y1 - y0
    lx2 = lx1 + (x2 - x1)
    ly2 = ly1 + (y2 - y1)

    out[y1:y2, x1:x2] = local[ly1:ly2, lx1:lx2]
    return out, True


def _fill_polygon(mask: np.ndarray, points: object, value: int) -> bool:
    if points is None:
        return False

    exterior = None
    interiors = []

    if isinstance(points, dict):
        exterior = points.get("exterior")
        interiors = points.get("interior") or points.get("interiors") or []
    elif isinstance(points, list):
        exterior = points
    else:
        return False

    if not (isinstance(exterior, list) and len(exterior) >= 3):
        return False

    ext = np.array(exterior, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [ext], int(value))

    if isinstance(interiors, list):
        for hole in interiors:
            if isinstance(hole, list) and len(hole) >= 3:
                hole_pts = np.array(hole, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [hole_pts], 0)

    return True


def _extract_objects(ann: dict) -> list[dict]:
    objs = ann.get("objects")
    if isinstance(objs, list):
        return objs
    return []


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_png_u8(path: Path, arr_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8.astype(np.uint8), mode="L").save(path)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _read_split_ids(split_txt: Path) -> list[str]:
    ids: list[str] = []
    with split_txt.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise SystemExit(f"Invalid line in {split_txt}: {line!r}")
            img_rel = parts[0]
            ids.append(Path(img_rel).stem)
    return ids


def _bbox_and_centroid(mask01: np.ndarray) -> tuple[list[int] | None, list[float] | None]:
    ys, xs = np.where(mask01.astype(bool))
    if ys.size == 0:
        return None, None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    cx = float(xs.mean())
    cy = float(ys.mean())
    return [x1, y1, x2, y2], [cx, cy]


def _resolve_overlap_by_distance(masks01: list[np.ndarray], *, tie_break_order: list[int]) -> tuple[np.ndarray, int]:
    if not masks01:
        raise ValueError("masks01 is empty")
    h, w = masks01[0].shape[:2]
    for m in masks01:
        if m.shape[:2] != (h, w):
            raise ValueError("shape mismatch")

    dists = []
    for m in masks01:
        src = (m.astype(np.uint8) > 0).astype(np.uint8)
        if int(src.sum()) == 0:
            dists.append(np.zeros((h, w), dtype=np.float32))
            continue
        dist = cv2.distanceTransform(src, cv2.DIST_L2, 3)
        dists.append(dist.astype(np.float32))

    stack = np.stack(dists, axis=0)
    max_dist = stack.max(axis=0)
    winners = stack == max_dist[None, :, :]

    label = np.zeros((h, w), dtype=np.uint8)
    overlap_pixels = 0
    for idx in tie_break_order:
        win = winners[idx]
        src = masks01[idx].astype(bool)
        label[(label == 0) & src & win] = np.uint8(idx + 1)

    union = np.zeros((h, w), dtype=bool)
    for m in masks01:
        union |= m.astype(bool)
    unlabeled = union & (label == 0)
    if bool(np.any(unlabeled)):
        for idx in tie_break_order:
            src = masks01[idx].astype(bool)
            add = unlabeled & src
            if bool(np.any(add)):
                label[add] = np.uint8(idx + 1)
                unlabeled[add] = False
            if not bool(np.any(unlabeled)):
                break

    if stack.shape[0] >= 2:
        sum_masks = np.zeros((h, w), dtype=np.uint8)
        for m in masks01:
            sum_masks += (m.astype(np.uint8) > 0).astype(np.uint8)
        overlap_pixels = int(np.sum(sum_masks >= 2))

    return label, overlap_pixels


def _render_object_mask(obj: dict, *, full_h: int, full_w: int, bitmap_decode_failed: Counter[str]) -> np.ndarray:
    geom = obj.get("geometry") if isinstance(obj.get("geometry"), dict) else {}
    points = obj.get("points")
    if points is None and isinstance(geom, dict):
        points = geom.get("points")
    bitmap = obj.get("bitmap")
    if bitmap is None and isinstance(geom, dict):
        bitmap = geom.get("bitmap")

    if isinstance(bitmap, dict):
        local, ok = _bitmap_to_full_mask(bitmap, full_h, full_w)
        if not ok:
            title = str(obj.get("classTitle") or "")
            bitmap_decode_failed[title] += 1
        return (local > 0).astype(np.uint8)

    poly = np.zeros((full_h, full_w), dtype=np.uint8)
    if _fill_polygon(poly, points, 1):
        return (poly > 0).astype(np.uint8)

    return np.zeros((full_h, full_w), dtype=np.uint8)


@dataclass(frozen=True)
class LeafObject:
    source_id: int | None
    class_title: str
    mask01: np.ndarray
    area: int
    bbox: list[int] | None
    centroid: list[float] | None
    geometry_type: str


def _select_leaf_objects(leaf_objs: list[LeafObject], *, max_instances: int = 3) -> tuple[list[LeafObject], list[dict]]:
    valid = []
    excluded = []
    for o in leaf_objs:
        if int(o.area) <= 0:
            excluded.append({"source_id": o.source_id, "class_title": o.class_title, "reason": "zero_area"})
            continue
        valid.append(o)

    valid.sort(key=lambda o: (-int(o.area), float(o.centroid[0]) if o.centroid else 0.0, float(o.centroid[1]) if o.centroid else 0.0, int(o.source_id or 0)))
    selected = valid[: int(max_instances)]
    for o in valid[int(max_instances) :]:
        excluded.append({"source_id": o.source_id, "class_title": o.class_title, "reason": "excess_over_3"})
    return selected, excluded


def _assign_instance_ids(selected: list[LeafObject]) -> list[LeafObject]:
    def key(o: LeafObject):
        cx = float(o.centroid[0]) if o.centroid else 0.0
        cy = float(o.centroid[1]) if o.centroid else 0.0
        sid = int(o.source_id or 0)
        return (cx, cy, sid)

    return sorted(selected, key=key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("exports/supervisely_sdk/Срезы 2026"))
    ap.add_argument("--output-root", type=Path, default=Path("datasets/converted_leaflet_instances"))
    ap.add_argument("--reference-splits-root", type=Path, default=Path("datasets/converted_full_multiclass"))
    ap.add_argument("--curation-json", type=Path, default=Path("server_assets/curation/curation_result.json"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    input_root = args.input_root.resolve()
    ann_dir = (input_root / "ann").resolve()
    img_dir = (input_root / "img").resolve()
    if not ann_dir.exists():
        raise SystemExit(f"Missing ann dir: {ann_dir}")
    if not img_dir.exists():
        raise SystemExit(f"Missing img dir: {img_dir}")

    out_root = args.output_root.resolve()
    if out_root.exists():
        has_any = any(out_root.iterdir())
        if has_any and not args.overwrite:
            raise SystemExit(f"Output dir exists and is not empty: {out_root} (use --overwrite to proceed)")
    _safe_mkdir(out_root)

    images_out = out_root / "images"
    semantic_out = out_root / "semantic_masks"
    instance_out = out_root / "instance_masks"
    meta_out = out_root / "metadata"
    for d in [images_out, semantic_out, instance_out, meta_out]:
        _safe_mkdir(d)

    splits_root = args.reference_splits_root.resolve()
    train_txt = splits_root / "train.txt"
    val_txt = splits_root / "val.txt"
    test_txt = splits_root / "test.txt"
    if not train_txt.exists() or not val_txt.exists() or not test_txt.exists():
        raise SystemExit(f"Missing reference split files in: {splits_root}")

    train_ids = _read_split_ids(train_txt)
    val_ids = _read_split_ids(val_txt)
    test_ids = _read_split_ids(test_txt)
    overlap_counts = {
        "train_val": int(len(set(train_ids) & set(val_ids))),
        "train_test": int(len(set(train_ids) & set(test_ids))),
        "val_test": int(len(set(val_ids) & set(test_ids))),
    }
    if any(v != 0 for v in overlap_counts.values()):
        raise SystemExit(f"Reference splits overlap: {overlap_counts}")

    all_ids = []
    all_ids.extend([("train", x) for x in train_ids])
    all_ids.extend([("val", x) for x in val_ids])
    all_ids.extend([("test", x) for x in test_ids])

    quality_map = {}
    curation_path = args.curation_json.resolve()
    if curation_path.exists():
        cur = json.loads(curation_path.read_text(encoding="utf-8"))
        if isinstance(cur, dict):
            for k in ["clean", "medium", "bad"]:
                for sid in cur.get(k, []) or []:
                    quality_map[str(sid)] = k

    summary = {
        "input_root": str(input_root),
        "output_root": str(out_root),
        "reference_splits_root": str(splits_root),
        "total_annotations_requested": int(len(all_ids)),
        "success": 0,
        "failed": 0,
        "failures": [],
        "leaf_objects_distribution": Counter(),
        "selected_instances_distribution": Counter(),
        "excluded_objects_total": 0,
        "annotations_with_overlaps": 0,
        "overlap_pixels_total": 0,
        "annotations_with_warnings": 0,
        "union_mismatch_count": 0,
        "split_counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "split_overlaps": overlap_counts,
        "quality_counts": Counter(quality_map.values()),
        "overlap_handling_rule": "distance_to_boundary (cv2.distanceTransform), tie-break by deterministic instance order",
    }

    manifest_rows = []
    bitmap_decode_failed_total: Counter[str] = Counter()

    for split, sid in all_ids:
        ann_path = ann_dir / f"{sid}.png.json"
        img_path = img_dir / f"{sid}.png"
        if not ann_path.exists():
            summary["failed"] += 1
            summary["failures"].append({"sample": sid, "split": split, "reason": "missing_annotation", "path": str(ann_path)})
            continue
        if not img_path.exists():
            summary["failed"] += 1
            summary["failures"].append({"sample": sid, "split": split, "reason": "missing_image", "path": str(img_path)})
            continue

        warnings = []
        bitmap_decode_failed: Counter[str] = Counter()
        try:
            ann = _read_json(ann_path)
            h, w = _infer_size_from_ann_or_image(ann, img_path)
            objs = _extract_objects(ann)
        except Exception as e:
            summary["failed"] += 1
            summary["failures"].append({"sample": sid, "split": split, "reason": "read_error", "error": str(e)})
            continue

        leaf_objs: list[LeafObject] = []
        annulus_objs = 0
        for obj in objs:
            title = str(obj.get("classTitle") or obj.get("class_title") or obj.get("title") or "")
            geom_type = str(obj.get("geometryType") or obj.get("geometry_type") or "unknown")
            src_id = obj.get("id")
            src_id_int = int(src_id) if isinstance(src_id, int) else None

            if _is_leaflet_title(title):
                m01 = _render_object_mask(obj, full_h=h, full_w=w, bitmap_decode_failed=bitmap_decode_failed)
                area = int(np.sum(m01 > 0))
                bbox, centroid = _bbox_and_centroid(m01)
                leaf_objs.append(
                    LeafObject(
                        source_id=src_id_int,
                        class_title=title,
                        mask01=m01,
                        area=area,
                        bbox=bbox,
                        centroid=centroid,
                        geometry_type=geom_type,
                    )
                )
            elif _is_annulus_title(title):
                annulus_objs += 1
            else:
                continue

        summary["leaf_objects_distribution"][str(len(leaf_objs))] += 1

        selected, excluded = _select_leaf_objects(leaf_objs, max_instances=3)
        summary["excluded_objects_total"] += int(len(excluded))

        selected = _assign_instance_ids(selected)
        selected_n = int(len(selected))
        summary["selected_instances_distribution"][str(selected_n)] += 1

        ring_mask01 = np.zeros((h, w), dtype=np.uint8)
        for obj in objs:
            title = str(obj.get("classTitle") or obj.get("class_title") or obj.get("title") or "")
            if not _is_annulus_title(title):
                continue
            local = _render_object_mask(obj, full_h=h, full_w=w, bitmap_decode_failed=bitmap_decode_failed)
            if bool(np.any(local)):
                ring_mask01[local.astype(bool)] = 1

        leaf_masks01 = []
        for o in selected:
            m = o.mask01.copy()
            if bool(np.any(ring_mask01)):
                m[ring_mask01.astype(bool)] = 0
            leaf_masks01.append(m)

        overlap_pixels = 0
        instance_label = np.zeros((h, w), dtype=np.uint8)
        if selected_n >= 1:
            tie_break_order = list(range(selected_n))
            instance_label, overlap_pixels = _resolve_overlap_by_distance(leaf_masks01, tie_break_order=tie_break_order)
            if overlap_pixels > 0:
                summary["annotations_with_overlaps"] += 1
                summary["overlap_pixels_total"] += int(overlap_pixels)
                warnings.append(f"leaflet_overlap_pixels={overlap_pixels}")

        semantic = np.zeros((h, w), dtype=np.uint8)
        semantic[instance_label > 0] = 1
        semantic[ring_mask01.astype(bool)] = 2

        union_instance = instance_label > 0
        union_sem_leaf = semantic == 1
        union_mismatch = int(np.sum(union_instance != union_sem_leaf))
        if union_mismatch != 0:
            summary["union_mismatch_count"] += 1
            warnings.append(f"union_instance_vs_semantic_mismatch_pixels={union_mismatch}")

        inst_ids = sorted(list(set(np.unique(instance_label).tolist()) - {0}))
        if inst_ids and inst_ids != list(range(1, max(inst_ids) + 1)):
            warnings.append(f"non_contiguous_instance_ids={inst_ids}")

        inst_info = []
        for inst_id in [1, 2, 3]:
            m = instance_label == inst_id
            area = int(np.sum(m))
            bbox, centroid = _bbox_and_centroid(m.astype(np.uint8))
            inst_info.append({"instance_id": inst_id, "present": bool(area > 0), "area": area, "bbox": bbox, "centroid": centroid})

        if bitmap_decode_failed:
            for k, v in bitmap_decode_failed.items():
                bitmap_decode_failed_total[k] += int(v)
            warnings.append(f"bitmap_decode_failed_titles={dict(bitmap_decode_failed)}")

        if warnings:
            summary["annotations_with_warnings"] += 1

        out_img = images_out / f"{sid}.png"
        out_sem = semantic_out / f"{sid}.png"
        out_inst = instance_out / f"{sid}.png"
        out_meta = meta_out / f"{sid}.json"

        _copy_file(img_path, out_img)
        _save_png_u8(out_sem, semantic)
        _save_png_u8(out_inst, instance_label)

        selected_meta = []
        for inst_local_id, obj in enumerate(selected, start=1):
            selected_meta.append(
                {
                    "local_instance_id": int(inst_local_id),
                    "source_object_id": obj.source_id,
                    "class_title": obj.class_title,
                    "area": int(obj.area),
                    "bbox": obj.bbox,
                    "centroid": obj.centroid,
                    "geometry_type": obj.geometry_type,
                }
            )

        meta = {
            "sample": sid,
            "split": split,
            "quality": quality_map.get(sid, None),
            "image_size": {"height": int(h), "width": int(w)},
            "source": {"image": str(img_path), "annotation": str(ann_path)},
            "leaflet_source_objects_total": int(len(leaf_objs)),
            "leaflet_selected_instances": int(selected_n),
            "leaflet_selected_source_object_ids": [x.get("source_object_id") for x in selected_meta],
            "leaflet_excluded_objects": excluded,
            "leaflet_instance_assignment_rule": "sort by centroid_x, then centroid_y, then source_object_id (technical order, not anatomical numbering)",
            "leaflet_selected": selected_meta,
            "instance_mask_stats": inst_info,
            "annulus_objects_count": int(annulus_objs),
            "overlap_pixels": int(overlap_pixels),
            "warnings": warnings,
        }
        with out_meta.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        manifest_rows.append(
            {
                "sample": sid,
                "split": split,
                "quality": quality_map.get(sid, ""),
                "source_leaf_objects": int(len(leaf_objs)),
                "selected_instances": int(selected_n),
                "excluded_instances": int(len(excluded)),
                "instance_1_area": int(inst_info[0]["area"]),
                "instance_2_area": int(inst_info[1]["area"]),
                "instance_3_area": int(inst_info[2]["area"]),
                "overlap_pixels": int(overlap_pixels),
                "warning_count": int(len(warnings)),
                "conversion_ok": int(1 if union_mismatch == 0 else 0),
                "image_rel": f"images/{sid}.png",
                "semantic_rel": f"semantic_masks/{sid}.png",
                "instance_rel": f"instance_masks/{sid}.png",
                "metadata_rel": f"metadata/{sid}.json",
            }
        )

        summary["success"] += 1

    def _write_split(name: str, ids: list[str]) -> None:
        out = out_root / name
        with out.open("w", encoding="utf-8") as f:
            for sid in ids:
                f.write(f"images/{sid}.png\tsemantic_masks/{sid}.png\n")

    _write_split("train.txt", train_ids)
    _write_split("val.txt", val_ids)
    _write_split("test.txt", test_ids)

    summary_path = out_root / "conversion_summary.json"
    summary["bitmap_decode_failed_titles"] = dict(bitmap_decode_failed_total)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_csv = out_root / "instance_dataset_manifest.csv"
    if manifest_rows:
        with manifest_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            for r in manifest_rows:
                w.writerow(r)

    print(f"Total requested: {summary['total_annotations_requested']}")
    print(f"Success: {summary['success']}")
    print(f"Failed: {summary['failed']}")
    print(f"Overlaps: {summary['annotations_with_overlaps']} (pixels total={summary['overlap_pixels_total']})")
    print(f"Warnings: {summary['annotations_with_warnings']}")
    print(f"Union mismatches: {summary['union_mismatch_count']}")
    print(f"Output: {out_root}")
    print(f"Summary: {summary_path}")
    print(f"Manifest: {manifest_csv}")


if __name__ == "__main__":
    main()

