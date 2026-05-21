import argparse
import base64
import json
import os
import re
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _ann_to_image_basename(ann_path: Path) -> str:
    name = ann_path.name
    if name.lower().endswith(".json"):
        name = name[:-5]
    return name


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_image_rgb_any(path: Path) -> np.ndarray | None:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is not None:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        with Image.open(path) as img:
            img.load()
            return np.array(img.convert("RGB"))
    except Exception:
        return None


def _decode_supervisely_bitmap_to_local_alpha(bitmap: dict) -> tuple[np.ndarray | None, bool]:
    data_b64 = bitmap.get("data")
    if not isinstance(data_b64, str) or not data_b64:
        return None, False

    try:
        raw = base64.b64decode(data_b64)
    except Exception:
        return None, False

    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        try:
            raw2 = zlib.decompress(raw)
            decoded = cv2.imdecode(np.frombuffer(raw2, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception:
            decoded = None
    if decoded is None:
        return None, False

    if decoded.ndim == 2:
        alpha = (decoded > 0).astype(np.uint8)
    else:
        if decoded.shape[2] == 4:
            alpha = (decoded[:, :, 3] > 0).astype(np.uint8)
        else:
            alpha = (np.any(decoded[:, :, :3] > 0, axis=2)).astype(np.uint8)

    return alpha, True


def _scan_ann_files(root: Path) -> list[Path]:
    ann_dirs = [p for p in root.rglob("ann") if p.is_dir() and p.name.lower() == "ann"]
    ann_files: list[Path] = []
    for d in ann_dirs:
        for p in d.rglob("*.json"):
            if p.is_file():
                ann_files.append(p)
    ann_files.sort(key=lambda p: str(p).lower())
    return ann_files


def _scan_images(root: Path) -> list[Path]:
    images: list[Path] = []
    for p in root.rglob("*"):
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


@dataclass(frozen=True)
class AnnAudit:
    ann_path: Path
    image_path: Path | None
    match_reason: str
    objects_total: int
    objects_by_title: Counter[str]
    images_has_leaf: bool
    images_has_ring: bool
    empty_objects: bool
    unsupported_geometry_count: int
    bitmap_decode_failed: int
    bitmap_objects: int
    image_unreadable: bool


def _audit_annotation(ann_path: Path, image_path: Path | None, match_reason: str) -> AnnAudit:
    objects_by_title: Counter[str] = Counter()
    unsupported_geometry_count = 0
    bitmap_decode_failed = 0
    bitmap_objects = 0
    objects_total = 0
    has_leaf = False
    has_ring = False

    image_unreadable = False
    if image_path is not None:
        image_unreadable = _read_image_rgb_any(image_path) is None

    try:
        ann = _read_json(ann_path)
    except Exception:
        ann = {}

    objs = ann.get("objects")
    if not isinstance(objs, list):
        objs = []
    objects_total = len(objs)

    for obj in objs:
        title = str(obj.get("classTitle") or obj.get("class_title") or obj.get("title") or "")
        objects_by_title[title] += 1
        t = _norm(title)
        if "leaf" in t:
            has_leaf = True
        if "aortic valve base" in t or "valve base" in t or "ring" in t or "annulus" in t:
            has_ring = True

        geom_type = str(obj.get("geometryType") or obj.get("geometry_type") or "unknown")
        geom = obj.get("geometry") if isinstance(obj.get("geometry"), dict) else {}
        bitmap = obj.get("bitmap")
        if bitmap is None and isinstance(geom, dict):
            bitmap = geom.get("bitmap")

        if isinstance(bitmap, dict):
            bitmap_objects += 1
            _, ok = _decode_supervisely_bitmap_to_local_alpha(bitmap)
            if not ok:
                bitmap_decode_failed += 1
            continue

        points = obj.get("points")
        if points is None and isinstance(geom, dict):
            points = geom.get("points")
        if points is not None:
            continue

        unsupported_geometry_count += 1

    return AnnAudit(
        ann_path=ann_path,
        image_path=image_path,
        match_reason=match_reason,
        objects_total=objects_total,
        objects_by_title=objects_by_title,
        images_has_leaf=has_leaf,
        images_has_ring=has_ring,
        empty_objects=(objects_total == 0),
        unsupported_geometry_count=unsupported_geometry_count,
        bitmap_decode_failed=bitmap_decode_failed,
        bitmap_objects=bitmap_objects,
        image_unreadable=image_unreadable,
    )


def _load_converted_source_index(converted_root: Path) -> dict[str, str]:
    meta_dir = (converted_root / "meta").resolve()
    if not meta_dir.exists():
        return {}
    idx: dict[str, str] = {}
    for p in meta_dir.rglob("*.json"):
        if not p.is_file():
            continue
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        src = meta.get("source") if isinstance(meta, dict) else None
        if not isinstance(src, dict):
            continue
        img = src.get("image")
        if not isinstance(img, str) or not img:
            continue
        b = Path(img).name.lower()
        idx[b] = p.stem
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Supervisely export layout and matching vs converted dataset.")
    parser.add_argument("--input", type=Path, required=True, help="Export root, e.g. exports/supervisely/328010_HIA")
    parser.add_argument("--converted-root", type=Path, default=Path("datasets/converted"))
    args = parser.parse_args()

    input_root = args.input.resolve()
    if not input_root.exists():
        raise SystemExit(f"Input does not exist: {input_root}")

    masks_sibling = input_root.parent / f"{input_root.name}_masks"
    roots = [input_root]
    if masks_sibling.exists() and masks_sibling.is_dir():
        roots.append(masks_sibling.resolve())

    image_index = _build_image_index(roots)
    ann_files = _scan_ann_files(input_root)

    img_files_total = sum(len(_scan_images(r)) for r in roots)
    ann_files_total = len(ann_files)

    audits: list[AnnAudit] = []
    for ann_path in tqdm(ann_files, desc="Audit", unit="ann"):
        img_path, reason = _find_matching_image(ann_path, image_index=image_index)
        audits.append(_audit_annotation(ann_path, img_path, reason))

    ann_with_matching_image = sum(1 for a in audits if a.image_path is not None)
    image_with_matching_ann = len({a.image_path for a in audits if a.image_path is not None})

    objects_total = sum(a.objects_total for a in audits)
    objects_by_title = Counter()
    for a in audits:
        objects_by_title.update(a.objects_by_title)

    images_with_leaf = sum(1 for a in audits if a.images_has_leaf)
    images_with_ring = sum(1 for a in audits if a.images_has_ring)
    images_with_both = sum(1 for a in audits if a.images_has_leaf and a.images_has_ring)
    images_only_leaf = sum(1 for a in audits if a.images_has_leaf and not a.images_has_ring)
    images_only_ring = sum(1 for a in audits if a.images_has_ring and not a.images_has_leaf)

    empty_anns = sum(1 for a in audits if a.empty_objects)
    unsupported_geom = sum(a.unsupported_geometry_count for a in audits)
    bitmap_decode_failed = sum(a.bitmap_decode_failed for a in audits)
    bitmap_objects = sum(a.bitmap_objects for a in audits)
    image_unreadable = sum(1 for a in audits if a.image_unreadable)

    match_reasons = Counter(a.match_reason for a in audits)

    print()
    print("Supervisely audit")
    print(f"Input root: {input_root}")
    if len(roots) > 1:
        print(f"Extra image root: {roots[1]}")
    print()
    print(f"Image files found: {img_files_total}")
    print(f"Annotation json found (under ann/): {ann_files_total}")
    print(f"Annotations with matching image: {ann_with_matching_image}")
    print(f"Images with matching annotation: {image_with_matching_ann}")
    print()
    print(f"Objects total: {objects_total}")
    print(f"Images containing Leaf: {images_with_leaf}")
    print(f"Images containing Aortic valve base: {images_with_ring}")
    print(f"Images containing both: {images_with_both}")
    print(f"Images only Leaf: {images_only_leaf}")
    print(f"Images only Aortic valve base: {images_only_ring}")
    print()
    print(f"Empty annotations (0 objects): {empty_anns}")
    print(f"Unsupported geometryType objects: {unsupported_geom}")
    print(f"Bitmap objects: {bitmap_objects}")
    print(f"Bitmap decode failed: {bitmap_decode_failed}")
    print(f"Images unreadable: {image_unreadable}")
    print()
    print("Matching reasons (top):")
    for k, v in match_reasons.most_common(20):
        print(f"  - {k}: {v}")

    print()
    print("Objects by classTitle (top 30):")
    for k, v in objects_by_title.most_common(30):
        print(f"  - {k or '<empty>'}: {v}")

    converted_root = args.converted_root.resolve()
    converted_idx = _load_converted_source_index(converted_root)
    if converted_idx:
        raw_ids = {_ann_to_image_basename(a.ann_path).lower() for a in audits}
        converted_src_basenames = set(converted_idx.keys())

        missing = sorted([x for x in raw_ids if x not in converted_src_basenames])
        print()
        print("Compare with converted")
        print(f"Converted root: {converted_root}")
        print(f"Raw annotated samples (by ann filename): {len(raw_ids)}")
        print(f"Converted samples (by meta source.image basename): {len(converted_src_basenames)}")
        print(f"Missing in converted: {len(missing)}")

        reason_by_id: dict[str, str] = {}
        audit_by_id = {(_ann_to_image_basename(a.ann_path).lower()): a for a in audits}
        for mid in missing[:200]:
            a = audit_by_id.get(mid)
            if a is None:
                reason_by_id[mid] = "other"
                continue
            if a.image_path is None:
                reason_by_id[mid] = "no_matching_image"
            elif a.image_unreadable:
                reason_by_id[mid] = "image_unreadable"
            elif a.empty_objects:
                reason_by_id[mid] = "empty_annotation"
            elif a.unsupported_geometry_count > 0:
                reason_by_id[mid] = "unsupported_geometry"
            elif a.bitmap_decode_failed > 0:
                reason_by_id[mid] = "bitmap_decode_failed"
            else:
                reason_by_id[mid] = "other"

        reason_counts = Counter(reason_by_id.values())
        print()
        print("Missing reasons (sampled):")
        for k, v in reason_counts.most_common():
            print(f"  - {k}: {v}")
        if missing:
            print()
            print("Missing sample ids (first 50):")
            for mid in missing[:50]:
                print(f"  - {mid}  ({reason_by_id.get(mid, 'other')})")
    else:
        print()
        print("Compare with converted: skipped (no meta/ in converted root)")


if __name__ == "__main__":
    main()

