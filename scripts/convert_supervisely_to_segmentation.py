import argparse
import base64
import json
import os
import random
import re
import shutil
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _norm_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def map_class_title_to_id(title: str) -> int | None:
    t = _norm_title(title)
    if not t:
        return None
    if "calc" in t or "кальц" in t:
        return None

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
    if any(m in t for m in leaflet_markers):
        return 1

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
    if any(m in t for m in ring_markers):
        return 2

    return None


def is_ignored_title_no_warning(title: str) -> bool:
    t = _norm_title(title)
    if not t:
        return True
    if "calc" in t or "кальц" in t:
        return True
    return False


def _safe_rel_filename(input_root: Path, file_path: Path) -> str:
    rel = os.path.relpath(str(file_path), str(input_root))
    rel = rel.replace("\\", "/")
    rel = re.sub(r"[^A-Za-z0-9._/-]+", "_", rel)
    rel = rel.replace("/", "__")
    rel = rel.lstrip("._")
    if not rel:
        rel = file_path.name
    return rel


def _ann_to_image_basename(ann_path: Path) -> str:
    name = ann_path.name
    if name.lower().endswith(".json"):
        name = name[:-5]
    return name


def _scan_ann_files(input_root: Path) -> list[Path]:
    ann_dirs = [p for p in input_root.rglob("ann") if p.is_dir() and p.name.lower() == "ann"]
    ann_files: list[Path] = []
    for d in ann_dirs:
        for p in d.rglob("*.json"):
            if p.is_file():
                ann_files.append(p)
    ann_files.sort(key=lambda p: str(p).lower())
    return ann_files


def _scan_images(input_root: Path) -> list[Path]:
    images: list[Path] = []
    for p in input_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    images.sort(key=lambda p: str(p).lower())
    return images


def _build_image_index(roots: list[Path]) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = defaultdict(list)
    for r in roots:
        for p in _scan_images(r):
            idx[p.name.lower()].append(p)
    return idx


def _find_matching_image(ann_path: Path, image_index: dict[str, list[Path]]) -> tuple[Path | None, str]:
    base = _ann_to_image_basename(ann_path)
    ann_dir = ann_path.parent

    candidate_dirs = []
    if ann_dir.name.lower() == "ann":
        candidate_dirs.append(ann_dir.parent / "img")
        candidate_dirs.append(ann_dir.parent / "images")
    for d in candidate_dirs:
        p = d / base
        if p.exists() and p.is_file():
            return p, "sibling_img"

    hits = image_index.get(base.lower(), [])
    if len(hits) == 1:
        return hits[0], "global_basename"
    if len(hits) > 1:
        return hits[0], "global_basename_ambiguous"

    stem = Path(base).stem
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]:
        hits2 = image_index.get((stem + ext).lower(), [])
        if len(hits2) == 1:
            return hits2[0], "global_stem_ext"
        if len(hits2) > 1:
            return hits2[0], "global_stem_ext_ambiguous"

    return None, "no_matching_image"


def _discover_pairs(input_root: Path) -> tuple[list[tuple[Path, Path, str]], dict]:
    masks_sibling = input_root.parent / f"{input_root.name}_masks"
    roots = [input_root]
    if masks_sibling.exists() and masks_sibling.is_dir():
        roots.append(masks_sibling.resolve())

    ann_files = _scan_ann_files(input_root)
    image_index = _build_image_index(roots)

    pairs: list[tuple[Path, Path, str]] = []
    missing_images = 0
    for ann_path in ann_files:
        img_path, reason = _find_matching_image(ann_path, image_index=image_index)
        if img_path is None:
            missing_images += 1
            continue
        pairs.append((img_path, ann_path, reason))

    info = {
        "roots": [str(r) for r in roots],
        "ann_files_total": int(len(ann_files)),
        "img_files_total": int(sum(len(_scan_images(r)) for r in roots)),
        "missing_images_for_ann": int(missing_images),
        "match_reasons": Counter([r for _, _, r in pairs]),
    }
    return pairs, info


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_size_from_ann_or_image(ann: dict, image_path: Path) -> tuple[int, int]:
    size = ann.get("size") or {}
    h = size.get("height")
    w = size.get("width")
    if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
        return h, w
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


def _extract_objects(ann: dict) -> list[dict]:
    objs = ann.get("objects")
    if isinstance(objs, list):
        return objs
    return []


def _fill_polygon(mask: np.ndarray, points: object, class_id: int) -> bool:
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
    cv2.fillPoly(mask, [ext], int(class_id))

    if isinstance(interiors, list):
        for hole in interiors:
            if isinstance(hole, list) and len(hole) >= 3:
                hole_pts = np.array(hole, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [hole_pts], 0)

    return True


def _render_mask_for_image(
    image_path: Path,
    ann_path: Path,
    target: str,
) -> tuple[np.ndarray, dict, Counter, Counter, Counter, Counter, bool]:
    ann = _read_json(ann_path)
    h, w = _infer_size_from_ann_or_image(ann, image_path)
    mask = np.zeros((h, w), dtype=np.uint8)

    unknown_titles: Counter[str] = Counter()
    used_titles: Counter[str] = Counter()
    unsupported_geometries: Counter[str] = Counter()
    bitmap_decode_failed: Counter[str] = Counter()

    for obj in _extract_objects(ann):
        title = obj.get("classTitle") or obj.get("class_title") or obj.get("title") or ""
        class_id = map_class_title_to_id(str(title))
        if class_id is None:
            if not is_ignored_title_no_warning(str(title)) and _norm_title(str(title)):
                unknown_titles[str(title)] += 1
            continue

        used_titles[str(title)] += 1
        geometry_type = obj.get("geometryType") or obj.get("geometry_type") or "unknown"

        geom = obj.get("geometry") if isinstance(obj.get("geometry"), dict) else {}

        points = obj.get("points")
        if points is None and isinstance(geom, dict):
            points = geom.get("points")

        bitmap = obj.get("bitmap")
        if bitmap is None and isinstance(geom, dict):
            bitmap = geom.get("bitmap")

        if isinstance(bitmap, dict):
            local, ok = _bitmap_to_full_mask(bitmap, h, w)
            if not ok:
                bitmap_decode_failed[str(title)] += 1
            if local.any():
                mask[local.astype(bool)] = np.maximum(mask[local.astype(bool)], class_id)
            continue

        if _fill_polygon(mask, points, class_id):
            continue

        unsupported_geometries[str(geometry_type)] += 1

    t = str(target).strip().lower()
    has_leaf = bool(np.any(mask == 1))
    has_ring = bool(np.any(mask == 2))
    usable = False
    if t == "leaflet_only":
        mask = (mask == 1).astype(np.uint8)
        usable = has_leaf
    else:
        usable = has_leaf or has_ring

    meta = {
        "image_size": {"height": int(h), "width": int(w)},
        "source": {"image": str(image_path), "annotation": str(ann_path)},
    }
    return mask, meta, used_titles, unknown_titles, unsupported_geometries, bitmap_decode_failed, usable


def _save_mask_png(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(out_path)


def _copy_image(image_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, out_path)


@dataclass(frozen=True)
class ConvertedItem:
    image_rel: str
    mask_rel: str
    meta_rel: str


def _write_split_files(items: list[ConvertedItem], out_root: Path, seed: int, val_ratio: float, test_ratio: float) -> None:
    rnd = random.Random(seed)
    idx = list(range(len(items)))
    rnd.shuffle(idx)
    items_shuffled = [items[i] for i in idx]

    n = len(items_shuffled)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_test = min(n_test, n)
    n_val = min(n_val, n - n_test)
    n_train = n - n_val - n_test

    train = items_shuffled[:n_train]
    val = items_shuffled[n_train : n_train + n_val]
    test = items_shuffled[n_train + n_val :]

    def write_list(name: str, subset: list[ConvertedItem]) -> None:
        p = out_root / name
        with p.open("w", encoding="utf-8") as f:
            for it in subset:
                f.write(f"{it.image_rel}\t{it.mask_rel}\n")

    write_list("train.txt", train)
    write_list("val.txt", val)
    write_list("test.txt", test)


def convert_supervisely_export(
    input_root: Path,
    output_root: Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    target: str,
) -> None:
    images_out = output_root / "images"
    masks_out = output_root / "masks"
    meta_out = output_root / "meta"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

    pairs, discovery_info = _discover_pairs(input_root)
    pairs.sort(key=lambda t: (str(t[0]).lower(), str(t[1]).lower()))
    input_items = pairs
    images_found = discovery_info["img_files_total"]
    anns_found = discovery_info["ann_files_total"]
    missing_images_for_ann = discovery_info["missing_images_for_ann"]

    items: list[ConvertedItem] = []
    unknown_titles_total: Counter[str] = Counter()
    used_titles_total: Counter[str] = Counter()
    unsupported_geometries_total: Counter[str] = Counter()
    bitmap_decode_failed_total: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()

    used_output_names: set[str] = set()

    for img_path, ann_path, match_reason in tqdm(input_items, desc="Converting", unit="ann"):
        mask, meta, used_titles, unknown_titles, unsupported_geometries, bitmap_decode_failed, usable = _render_mask_for_image(
            img_path, ann_path, target=target
        )
        used_titles_total.update(used_titles)
        unknown_titles_total.update(unknown_titles)
        unsupported_geometries_total.update(unsupported_geometries)
        bitmap_decode_failed_total.update(bitmap_decode_failed)

        if not usable:
            skipped_reasons["empty_or_not_applicable_for_target"] += 1
            continue

        sample_id = Path(_ann_to_image_basename(ann_path)).stem
        ext = img_path.suffix.lower() if img_path.suffix else ".png"
        safe_img_name = f"{sample_id}{ext}"
        if safe_img_name in used_output_names:
            stem, ext = os.path.splitext(safe_img_name)
            k = 2
            while f"{stem}__{k}{ext}" in used_output_names:
                k += 1
            safe_img_name = f"{stem}__{k}{ext}"
        used_output_names.add(safe_img_name)

        safe_stem = os.path.splitext(safe_img_name)[0]
        out_img_path = images_out / safe_img_name
        out_mask_path = masks_out / f"{safe_stem}.png"
        out_meta_path = meta_out / f"{safe_stem}.json"

        _copy_image(img_path, out_img_path)
        _save_mask_png(mask, out_mask_path)

        meta["output"] = {
            "image": str(out_img_path),
            "mask": str(out_mask_path),
        }
        meta["target"] = str(target)
        meta["matching"] = {"reason": str(match_reason)}
        meta["used_class_titles"] = dict(used_titles)
        meta["unknown_class_titles"] = dict(unknown_titles)
        meta["unsupported_geometries"] = dict(unsupported_geometries)
        meta["bitmap_decode_failed"] = dict(bitmap_decode_failed)

        with out_meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        items.append(
            ConvertedItem(
                image_rel=f"images/{safe_img_name}",
                mask_rel=f"masks/{safe_stem}.png",
                meta_rel=f"meta/{safe_stem}.json",
            )
        )

    _write_split_files(items, output_root, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)

    print()
    print("Done.")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Target: {target}")
    print(f"Images found (in search roots): {images_found}")
    print(f"Annotations found (under ann/): {anns_found}")
    if missing_images_for_ann:
        print(f"Annotations without matching image file: {missing_images_for_ann}")
    if skipped_reasons:
        print("Skipped:")
        for k, v in skipped_reasons.most_common():
            print(f"  - {k}: {v}")
    print(f"Converted: {len(items)}")
    print(f"Objects used (mapped to classes): {sum(used_titles_total.values())}")
    print()

    if unsupported_geometries_total:
        print("Warnings: unsupported geometry types (top 20):")
        for gt, cnt in unsupported_geometries_total.most_common(20):
            print(f"  - {gt}: {cnt}")
        print()

    if unknown_titles_total:
        print("Warnings: unknown / ignored class titles (top 30):")
        for title, cnt in unknown_titles_total.most_common(30):
            print(f"  - {title}: {cnt}")

    if bitmap_decode_failed_total:
        print()
        print("Warnings: bitmap decode failed (top 30):")
        for title, cnt in bitmap_decode_failed_total.most_common(30):
            print(f"  - {title}: {cnt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Supervisely export to simple semantic segmentation dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("exports/supervisely"),
        help="Path to Supervisely export root (default: exports/supervisely).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/converted"),
        help="Output dataset root (default: datasets/converted).",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--target", type=str, choices=["multiclass", "leaflet_only"], default="multiclass")
    args = parser.parse_args()

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if not input_root.exists():
        raise SystemExit(f"Input path does not exist: {input_root}")

    if args.val_ratio < 0 or args.test_ratio < 0 or (args.val_ratio + args.test_ratio) >= 1.0:
        raise SystemExit("Invalid split ratios: require val_ratio>=0, test_ratio>=0, val_ratio+test_ratio < 1.0")

    convert_supervisely_export(
        input_root=input_root,
        output_root=output_root,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        target=str(args.target),
    )


if __name__ == "__main__":
    main()
