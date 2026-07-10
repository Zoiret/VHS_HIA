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


def _read_split(split_txt: Path) -> list[tuple[str, str]]:
    rows = []
    with split_txt.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise SystemExit(f"Invalid line in {split_txt}: {line!r}")
            rows.append((parts[0], parts[1]))
    return rows


def _ids_from_split(rows: list[tuple[str, str]]) -> set[str]:
    out = set()
    for img_rel, _ in rows:
        out.add(Path(img_rel).stem)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path("datasets/converted_leaflet_instances"))
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    root = args.dataset_root.resolve()
    if not root.exists():
        raise SystemExit(f"Missing dataset root: {root}")

    train_txt = root / "train.txt"
    val_txt = root / "val.txt"
    test_txt = root / "test.txt"
    for p in [train_txt, val_txt, test_txt]:
        if not p.exists():
            raise SystemExit(f"Missing split file: {p}")

    train_rows = _read_split(train_txt)
    val_rows = _read_split(val_txt)
    test_rows = _read_split(test_txt)

    train_ids = _ids_from_split(train_rows)
    val_ids = _ids_from_split(val_rows)
    test_ids = _ids_from_split(test_rows)

    overlaps = {
        "train_val": int(len(train_ids & val_ids)),
        "train_test": int(len(train_ids & test_ids)),
        "val_test": int(len(val_ids & test_ids)),
    }

    errors = []
    warnings = []
    per_sample_rows = []

    manifest_csv = root / "instance_dataset_manifest.csv"
    if not manifest_csv.exists():
        warnings.append({"type": "missing_manifest_csv", "path": str(manifest_csv)})

    def check_one(sample_id: str, split: str) -> None:
        img_p = root / "images" / f"{sample_id}.png"
        sem_p = root / "semantic_masks" / f"{sample_id}.png"
        inst_p = root / "instance_masks" / f"{sample_id}.png"
        meta_p = root / "metadata" / f"{sample_id}.json"
        row = {"sample": sample_id, "split": split, "ok": 1, "error": "", "warning_count": 0}

        try:
            img = _read_rgb_u8(img_p)
            sem = _read_u8(sem_p)
            inst = _read_u8(inst_p)
        except Exception as e:
            row["ok"] = 0
            row["error"] = f"read_error: {e}"
            errors.append({"sample": sample_id, "split": split, "error": row["error"]})
            per_sample_rows.append(row)
            return

        if img.shape[:2] != sem.shape[:2] or sem.shape != inst.shape:
            row["ok"] = 0
            row["error"] = f"shape_mismatch image={img.shape} sem={sem.shape} inst={inst.shape}"
            errors.append({"sample": sample_id, "split": split, "error": row["error"]})
            per_sample_rows.append(row)
            return

        sem_ids = set(np.unique(sem).tolist())
        inst_ids = set(np.unique(inst).tolist())
        if not sem_ids.issubset({0, 1, 2}):
            row["ok"] = 0
            row["error"] = f"bad_semantic_ids: {sorted(list(sem_ids))}"
            errors.append({"sample": sample_id, "split": split, "error": row["error"]})
            per_sample_rows.append(row)
            return
        if not inst_ids.issubset({0, 1, 2, 3}):
            row["ok"] = 0
            row["error"] = f"bad_instance_ids: {sorted(list(inst_ids))}"
            errors.append({"sample": sample_id, "split": split, "error": row["error"]})
            per_sample_rows.append(row)
            return

        inst_union = inst > 0
        sem_leaf = sem == 1
        mismatch = int(np.sum(inst_union != sem_leaf))
        if mismatch != 0:
            row["ok"] = 0
            row["error"] = f"union_mismatch_pixels={mismatch}"
            errors.append({"sample": sample_id, "split": split, "error": row["error"]})
            per_sample_rows.append(row)
            return

        if int(np.sum((inst > 0) & (sem == 2))) != 0:
            row["ok"] = 0
            row["error"] = "annulus_pixels_in_instance_mask"
            errors.append({"sample": sample_id, "split": split, "error": row["error"]})
            per_sample_rows.append(row)
            return

        present = sorted([i for i in [1, 2, 3] if int(np.sum(inst == i)) > 0])
        if present and present != list(range(1, max(present) + 1)):
            row["ok"] = 0
            row["error"] = f"non_contiguous_instance_ids_present={present}"
            errors.append({"sample": sample_id, "split": split, "error": row["error"]})
            per_sample_rows.append(row)
            return

        if not meta_p.exists():
            row["warning_count"] += 1
            warnings.append({"sample": sample_id, "split": split, "type": "missing_metadata", "path": str(meta_p)})
        else:
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                stats = meta.get("instance_mask_stats") or []
                if isinstance(stats, list):
                    for st in stats:
                        inst_id = int(st.get("instance_id"))
                        area_meta = int(st.get("area"))
                        area_mask = int(np.sum(inst == inst_id))
                        if area_meta != area_mask:
                            row["ok"] = 0
                            row["error"] = f"metadata_area_mismatch inst={inst_id} meta={area_meta} mask={area_mask}"
                            errors.append({"sample": sample_id, "split": split, "error": row["error"]})
                            per_sample_rows.append(row)
                            return
            except Exception as e:
                row["warning_count"] += 1
                warnings.append({"sample": sample_id, "split": split, "type": "bad_metadata_json", "error": str(e)})

        per_sample_rows.append(row)

    for split, ids in [("train", sorted(list(train_ids))), ("val", sorted(list(val_ids))), ("test", sorted(list(test_ids)))]:
        for sid in ids:
            check_one(sid, split)

    report = {
        "dataset_root": str(root),
        "counts": {"train": int(len(train_ids)), "val": int(len(val_ids)), "test": int(len(test_ids))},
        "split_overlaps": overlaps,
        "samples_checked": int(len(train_ids) + len(val_ids) + len(test_ids)),
        "errors_count": int(len(errors)),
        "warnings_count": int(len(warnings)),
        "errors": errors[:200],
        "warnings": warnings[:200],
    }
    out_json = root / "validation_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = root / "validation_report.csv"
    if per_sample_rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
            w.writeheader()
            for r in per_sample_rows:
                w.writerow(r)

    print(f"Checked: {report['samples_checked']}")
    print(f"Errors: {report['errors_count']}")
    print(f"Warnings: {report['warnings_count']}")
    print(f"Splits overlap: {overlaps}")
    print(f"Report: {out_json}")
    if args.strict and report["errors_count"] != 0:
        raise SystemExit("Validation failed (strict)")


if __name__ == "__main__":
    main()

