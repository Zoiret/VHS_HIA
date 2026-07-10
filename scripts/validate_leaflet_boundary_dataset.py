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


def _center_crop_pair(image_rgb: np.ndarray, mask_u8: np.ndarray, crop: int) -> tuple[np.ndarray, np.ndarray]:
    crop_h = int(crop)
    crop_w = int(crop)
    h, w = image_rgb.shape[:2]
    if h < crop_h or w < crop_w:
        raise SystemExit(f"Cannot center-crop {crop_h}x{crop_w} from {h}x{w}")
    y0 = (h - crop_h) // 2 if h > crop_h else 0
    x0 = (w - crop_w) // 2 if w > crop_w else 0
    return image_rgb[y0 : y0 + crop_h, x0 : x0 + crop_w], mask_u8[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _crop3(img: np.ndarray, sem: np.ndarray, inst: np.ndarray, crop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img2, sem2 = _center_crop_pair(img, sem, crop)
    _, inst2 = _center_crop_pair(img, inst, crop)
    return img2, sem2, inst2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary-root", type=Path, default=Path("datasets/converted_leaflet_boundary"))
    ap.add_argument("--source-instances-root", type=Path, default=Path("datasets/converted_leaflet_instances"))
    ap.add_argument("--crop-size", type=int, default=768)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    b_root = args.boundary_root.resolve()
    s_root = args.source_instances_root.resolve()
    crop = int(args.crop_size)

    for p in [b_root / "train.txt", b_root / "val.txt", b_root / "test.txt"]:
        if not p.exists():
            raise SystemExit(f"Missing boundary split file: {p}")
    for p in [s_root / "train.txt", s_root / "val.txt", s_root / "test.txt"]:
        if not p.exists():
            raise SystemExit(f"Missing source split file: {p}")

    b_train = _read_split(b_root / "train.txt")
    b_val = _read_split(b_root / "val.txt")
    b_test = _read_split(b_root / "test.txt")
    s_train = _read_split(s_root / "train.txt")
    s_val = _read_split(s_root / "val.txt")
    s_test = _read_split(s_root / "test.txt")

    split_overlaps = {
        "train_val": int(len(set(b_train) & set(b_val))),
        "train_test": int(len(set(b_train) & set(b_test))),
        "val_test": int(len(set(b_val) & set(b_test))),
    }
    if any(v != 0 for v in split_overlaps.values()):
        raise SystemExit(f"Boundary splits overlap: {split_overlaps}")

    split_same = {
        "train_same": set(b_train) == set(s_train),
        "val_same": set(b_val) == set(s_val),
        "test_same": set(b_test) == set(s_test),
    }

    errors = []
    warnings = []
    rows = []

    def check_one(sid: str, split: str) -> None:
        b_img_p = b_root / "images" / f"{sid}.png"
        b_mask_p = b_root / "masks" / f"{sid}.png"
        if not b_img_p.exists() or not b_mask_p.exists():
            errors.append({"sample": sid, "split": split, "error": "missing_boundary_files"})
            return

        s_img_p = s_root / "images" / f"{sid}.png"
        s_sem_p = s_root / "semantic_masks" / f"{sid}.png"
        s_inst_p = s_root / "instance_masks" / f"{sid}.png"
        if not s_img_p.exists() or not s_sem_p.exists() or not s_inst_p.exists():
            errors.append({"sample": sid, "split": split, "error": "missing_source_files"})
            return

        try:
            b_img = _read_rgb_u8(b_img_p)
            b_mask = _read_u8(b_mask_p)
            s_img = _read_rgb_u8(s_img_p)
            s_sem = _read_u8(s_sem_p)
            s_inst = _read_u8(s_inst_p)
        except Exception as e:
            errors.append({"sample": sid, "split": split, "error": f"read_error: {e}"})
            return

        _, s_sem_c, s_inst_c = _crop3(s_img, s_sem, s_inst, crop)

        if b_img.shape[:2] != b_mask.shape[:2]:
            errors.append({"sample": sid, "split": split, "error": "shape_mismatch_boundary_image_mask"})
            return
        if b_img.shape[:2] != s_sem_c.shape[:2] or b_img.shape[:2] != s_inst_c.shape[:2]:
            errors.append({"sample": sid, "split": split, "error": "shape_mismatch_with_source_crop"})
            return

        b_ids = set(np.unique(b_mask).tolist())
        if not b_ids.issubset({0, 1, 2, 3}):
            errors.append({"sample": sid, "split": split, "error": f"invalid_boundary_labels={sorted(list(b_ids))}"})
            return

        ring_src = s_sem_c == 2
        leaf_src = s_sem_c == 1
        inst_union = s_inst_c > 0

        ring_preserved = int(np.sum((b_mask == 2) != ring_src))
        if ring_preserved != 0:
            errors.append({"sample": sid, "split": split, "error": f"ring_not_preserved_pixels={ring_preserved}"})
            return

        union_target = (b_mask == 1) | (b_mask == 3)
        union_mismatch = int(np.sum(union_target != leaf_src))
        if union_mismatch != 0:
            errors.append({"sample": sid, "split": split, "error": f"leaflet_union_mismatch_pixels={union_mismatch}"})
            return

        union_vs_inst = int(np.sum(union_target != inst_union))
        if union_vs_inst != 0:
            errors.append({"sample": sid, "split": split, "error": f"union_vs_instance_union_mismatch_pixels={union_vs_inst}"})
            return

        boundary = b_mask == 3
        if int(np.sum(boundary & ring_src)) != 0:
            errors.append({"sample": sid, "split": split, "error": "boundary_overlaps_ring"})
            return
        if int(np.sum(boundary & (~leaf_src))) != 0:
            errors.append({"sample": sid, "split": split, "error": "boundary_outside_leaflet_union"})
            return

        inst_present = [i for i in [1, 2, 3] if int(np.sum(s_inst_c == i)) > 0]
        if len(inst_present) <= 1 and int(np.sum(boundary)) != 0:
            errors.append({"sample": sid, "split": split, "error": "boundary_present_with_single_instance"})
            return

        rows.append(
            {
                "sample": sid,
                "split": split,
                "instance_count": int(len(inst_present)),
                "boundary_pixels": int(np.sum(boundary)),
                "leaflet_pixels": int(np.sum(leaf_src)),
                "ok": 1,
                "error": "",
            }
        )

    for split, ids in [("train", b_train), ("val", b_val), ("test", b_test)]:
        for sid in ids:
            check_one(sid, split)

    report = {
        "boundary_root": str(b_root),
        "source_instances_root": str(s_root),
        "crop_size": crop,
        "split_counts": {"train": len(b_train), "val": len(b_val), "test": len(b_test)},
        "split_overlap": split_overlaps,
        "split_same_as_source": split_same,
        "samples_checked": int(len(b_train) + len(b_val) + len(b_test)),
        "errors_count": int(len(errors)),
        "warnings_count": int(len(warnings)),
        "errors": errors[:200],
        "warnings": warnings[:200],
    }

    out_json = b_root / "validation_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_csv = b_root / "validation_report.csv"
    if rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(f"Checked: {report['samples_checked']}")
    print(f"Errors: {report['errors_count']}")
    print(f"Warnings: {report['warnings_count']}")
    print(f"Split overlap: {split_overlaps}")
    print(f"Split same as source: {split_same}")
    print(f"Report: {out_json}")
    if args.strict and report["errors_count"] != 0:
        raise SystemExit("Validation failed (strict)")


if __name__ == "__main__":
    main()

