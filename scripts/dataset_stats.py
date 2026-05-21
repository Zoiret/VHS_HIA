import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _read_mask(path: Path) -> np.ndarray | None:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute basic statistics for converted segmentation dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/converted"),
        help="Converted dataset root (default: datasets/converted).",
    )
    args = parser.parse_args()

    ds_root = args.dataset_root.resolve()
    images_dir = ds_root / "images"
    masks_dir = ds_root / "masks"
    meta_dir = ds_root / "meta"

    if not masks_dir.exists():
        raise SystemExit(f"Masks dir not found: {masks_dir}")

    mask_paths = [p for p in masks_dir.rglob("*.png") if p.is_file()]
    mask_paths.sort(key=lambda p: str(p).lower())

    class_pixel_counts = Counter({0: 0, 1: 0, 2: 0})
    empty_masks = 0
    images_with_leaflet = 0
    images_with_ring = 0
    bad_masks = 0

    for mp in tqdm(mask_paths, desc="Stats", unit="mask"):
        m = _read_mask(mp)
        if m is None:
            bad_masks += 1
            continue

        if m.size == 0:
            bad_masks += 1
            continue

        class_pixel_counts[0] += int(np.sum(m == 0))
        class_pixel_counts[1] += int(np.sum(m == 1))
        class_pixel_counts[2] += int(np.sum(m == 2))

        if int(m.max()) == 0:
            empty_masks += 1
        if np.any(m == 1):
            images_with_leaflet += 1
        if np.any(m == 2):
            images_with_ring += 1

    unknown_titles = Counter()
    unsupported_geometries = Counter()
    if meta_dir.exists():
        meta_paths = [p for p in meta_dir.rglob("*.json") if p.is_file()]
        for mp in meta_paths:
            try:
                with mp.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                u = meta.get("unknown_class_titles") or {}
                if isinstance(u, dict):
                    for k, v in u.items():
                        if isinstance(v, int):
                            unknown_titles[str(k)] += v

                ug = meta.get("unsupported_geometries") or {}
                if isinstance(ug, dict):
                    for k, v in ug.items():
                        if isinstance(v, int):
                            unsupported_geometries[str(k)] += v
            except Exception:
                continue

    total_images = 0
    if images_dir.exists():
        total_images = len([p for p in images_dir.rglob("*") if p.is_file()])

    print()
    print("Dataset stats")
    print(f"Dataset root: {ds_root}")
    print(f"Total images: {total_images}")
    print(f"Total masks: {len(mask_paths)}")
    if bad_masks:
        print(f"Bad masks (failed to read): {bad_masks}")
    print()
    print("Class pixel counts (mask uint8: 0 background, 1 leaflet, 2 fibrous_ring)")
    print(f"  background (0): {class_pixel_counts[0]}")
    print(f"  leaflet (1): {class_pixel_counts[1]}")
    print(f"  fibrous_ring (2): {class_pixel_counts[2]}")
    print()
    print(f"Empty masks: {empty_masks}")
    print(f"Images with leaflet: {images_with_leaflet}")
    print(f"Images with fibrous_ring: {images_with_ring}")
    print()

    if unknown_titles:
        print("Warnings: unknown / ignored class titles (top 50)")
        for title, cnt in unknown_titles.most_common(50):
            print(f"  - {title}: {cnt}")
        print()

    if unsupported_geometries:
        print("Warnings: unsupported geometry types (top 20)")
        for gt, cnt in unsupported_geometries.most_common(20):
            print(f"  - {gt}: {cnt}")


if __name__ == "__main__":
    main()
