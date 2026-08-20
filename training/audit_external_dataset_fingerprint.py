from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROP_H = 768
DEFAULT_CROP_W = 768


def _read_split_rows(path: Path) -> list[str]:
    return [
        row.strip()
        for row in path.read_text(encoding="utf-8-sig").splitlines()
        if row.strip()
    ]


def canonical_split_sha256(path: Path) -> tuple[list[str], str]:
    rows = _read_split_rows(path)
    payload = ("\n".join(rows) + "\n").encode("utf-8")
    return rows, hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sample_id_from_row(row: str) -> str:
    image_rel = row.split("\t", 1)[0].strip()
    return Path(image_rel).stem


def _patient_id_from_sample_id(sample_id: str) -> str:
    parts = str(sample_id).split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return str(sample_id)


def _center_crop_like_validation(image: np.ndarray, crop_h: int, crop_w: int, *, is_mask: bool) -> np.ndarray:
    h, w = image.shape[:2]
    if h < crop_h or w < crop_w:
        new_h = max(h, crop_h)
        new_w = max(w, crop_w)
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        image = cv2.resize(image, (new_w, new_h), interpolation=interp)
        h, w = image.shape[:2]
    y0 = (h - crop_h) // 2 if h > crop_h else 0
    x0 = (w - crop_w) // 2 if w > crop_w else 0
    if image.ndim == 2:
        return image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w, :]


def _read_u8(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def _positive_instance_count_after_crop(instance_path: Path, *, crop_h: int, crop_w: int) -> int | None:
    if not instance_path.exists():
        return None
    inst = _read_u8(instance_path)
    inst = _center_crop_like_validation(inst, crop_h, crop_w, is_mask=True)
    labels = [int(v) for v in np.unique(inst) if int(v) > 0]
    return int(len(labels))


def _stable_relative_text(path: Path, *bases: Path) -> str:
    path = path.resolve()
    for base in bases:
        try:
            return path.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return path.name


def _hash_asset(path: Path, *, relative_path: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(relative_path).replace("\\", "/"),
            "exists": False,
            "size": None,
            "sha256": None,
        }
    return {
        "path": str(relative_path).replace("\\", "/"),
        "exists": True,
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _duplicate_rows(rows: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    duplicates: list[str] = []
    for row in rows:
        counts[row] = counts.get(row, 0) + 1
        if counts[row] == 2:
            duplicates.append(row)
    return duplicates


def _parse_split_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"Invalid --split value {value!r}; expected name=path")
    name, raw_path = value.split("=", 1)
    split_name = name.strip()
    if not split_name:
        raise SystemExit(f"Invalid --split value {value!r}; empty split name")
    return split_name, Path(raw_path.strip())


def _resolve_path(path_like: Path, base_root: Path) -> Path:
    if path_like.is_absolute():
        return path_like.resolve()
    return (base_root / path_like).resolve()


def _asset_triplet_for_row(
    *,
    row: str,
    dataset_root: Path,
    instance_root: Path,
    crop_h: int,
    crop_w: int,
) -> tuple[str, dict[str, Any], int | None]:
    parts = row.split("\t")
    if len(parts) != 2:
        raise ValueError(f"Invalid split row: expected 2 tab-separated columns, got {row!r}")
    image_rel, semantic_rel = [x.strip().replace("\\", "/") for x in parts]
    sample_id = Path(image_rel).stem
    image_path = (dataset_root / image_rel).resolve()
    semantic_path = (dataset_root / semantic_rel).resolve()
    instance_path = (instance_root / "instance_masks" / f"{sample_id}.png").resolve()
    gt_count = _positive_instance_count_after_crop(instance_path, crop_h=crop_h, crop_w=crop_w)
    asset_entry = {
        "sample_id": str(sample_id),
        "patient_id": _patient_id_from_sample_id(sample_id),
        "image": _hash_asset(image_path, relative_path=image_rel),
        "semantic_mask": _hash_asset(semantic_path, relative_path=semantic_rel),
        "instance_mask": _hash_asset(instance_path, relative_path=f"instance_masks/{sample_id}.png"),
        "gt_count_after_locked_768_crop": gt_count,
    }
    return sample_id, asset_entry, gt_count


def build_fingerprint(
    *,
    dataset_root: Path,
    instance_root: Path,
    splits: dict[str, Path],
    crop_h: int = DEFAULT_CROP_H,
    crop_w: int = DEFAULT_CROP_W,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    instance_root = instance_root.resolve()
    contract_splits: dict[str, Any] = {}
    contract_assets: dict[str, Any] = {}
    diagnostics_splits: dict[str, Any] = {}
    total_missing_assets = 0
    total_duplicate_rows = 0
    split_base = dataset_root.parent

    for split_name, split_path in sorted(splits.items()):
        split_path = split_path.resolve()
        rows, split_sha = canonical_split_sha256(split_path)
        ordered_sample_ids = [_sample_id_from_row(row) for row in rows]
        unique_patients = sorted({_patient_id_from_sample_id(sample_id) for sample_id in ordered_sample_ids})
        gt_distribution: dict[str, int] = {}
        split_missing_assets = 0
        duplicates = _duplicate_rows(rows)
        for row in rows:
            sample_id, asset_entry, gt_count = _asset_triplet_for_row(
                row=row,
                dataset_root=dataset_root,
                instance_root=instance_root,
                crop_h=crop_h,
                crop_w=crop_w,
            )
            if sample_id not in contract_assets:
                contract_assets[sample_id] = asset_entry
            if gt_count is not None:
                key = str(gt_count)
                gt_distribution[key] = int(gt_distribution.get(key, 0) + 1)
            for asset_key in ("image", "semantic_mask", "instance_mask"):
                if not bool(asset_entry[asset_key]["exists"]):
                    split_missing_assets += 1
        total_missing_assets += int(split_missing_assets)
        total_duplicate_rows += int(len(duplicates))
        contract_splits[split_name] = {
            "split_relative_path": _stable_relative_text(split_path, REPO_ROOT, split_base),
            "canonical_split_sha256": split_sha,
            "logical_row_count": int(len(rows)),
            "ordered_rows": rows,
            "ordered_sample_ids": ordered_sample_ids,
            "unique_patient_ids": unique_patients,
            "patient_count": int(len(unique_patients)),
            "duplicate_rows": duplicates,
            "duplicate_row_count": int(len(duplicates)),
            "gt_distribution_after_locked_768_crop": {k: int(v) for k, v in sorted(gt_distribution.items())},
            "missing_asset_count": int(split_missing_assets),
        }
        diagnostics_splits[split_name] = {
            "absolute_split_path": str(split_path),
        }

    contract_payload = {
        "contract_version": 1,
        "crop_contract": {
            "height": int(crop_h),
            "width": int(crop_w),
            "mode": "deterministic_center_crop",
        },
        "dataset_root": _stable_relative_text(dataset_root, REPO_ROOT, dataset_root.parent.parent),
        "instance_root": _stable_relative_text(instance_root, REPO_ROOT, instance_root.parent.parent),
        "splits": contract_splits,
        "assets": dict(sorted(contract_assets.items())),
    }
    contract_json = json.dumps(contract_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    dataset_contract_sha256 = hashlib.sha256(contract_json.encode("utf-8")).hexdigest()
    return {
        "dataset_contract_sha256": dataset_contract_sha256,
        "dataset_contract": contract_payload,
        "summary": {
            "split_names": sorted(contract_splits.keys()),
            "total_missing_assets": int(total_missing_assets),
            "total_duplicate_rows": int(total_duplicate_rows),
            "asset_sample_count": int(len(contract_assets)),
        },
        "diagnostics": {
            "repo_root": str(REPO_ROOT),
            "absolute_dataset_root": str(dataset_root),
            "absolute_instance_root": str(instance_root),
            "splits": diagnostics_splits,
        },
    }


def _rows_only_in_a(rows_a: list[str], rows_b: list[str]) -> list[str]:
    set_b = set(rows_b)
    return [row for row in rows_a if row not in set_b]


def compare_fingerprints(payload_a: dict[str, Any], payload_b: dict[str, Any]) -> dict[str, Any]:
    contract_a = payload_a.get("dataset_contract") or {}
    contract_b = payload_b.get("dataset_contract") or {}
    splits_a = contract_a.get("splits") or {}
    splits_b = contract_b.get("splits") or {}
    assets_a = contract_a.get("assets") or {}
    assets_b = contract_b.get("assets") or {}

    split_names = sorted(set(splits_a.keys()) | set(splits_b.keys()))
    split_diffs: dict[str, Any] = {}
    any_split_difference = False
    missing_assets_in_contract = False
    for split_name in split_names:
        sa = splits_a.get(split_name)
        sb = splits_b.get(split_name)
        if sa is None or sb is None:
            any_split_difference = True
            split_diffs[split_name] = {
                "present_in_a": sa is not None,
                "present_in_b": sb is not None,
            }
            continue
        rows_a = list(sa.get("ordered_rows") or [])
        rows_b = list(sb.get("ordered_rows") or [])
        sample_ids_a = list(sa.get("ordered_sample_ids") or [])
        sample_ids_b = list(sb.get("ordered_sample_ids") or [])
        patient_ids_a = list(sa.get("unique_patient_ids") or [])
        patient_ids_b = list(sb.get("unique_patient_ids") or [])
        diff = {
            "ordered_equal": bool(rows_a == rows_b),
            "same_sample_set_different_order": bool(set(sample_ids_a) == set(sample_ids_b) and rows_a != rows_b),
            "rows_only_in_a": _rows_only_in_a(rows_a, rows_b),
            "rows_only_in_b": _rows_only_in_a(rows_b, rows_a),
            "patient_ids_only_in_a": sorted(set(patient_ids_a) - set(patient_ids_b)),
            "patient_ids_only_in_b": sorted(set(patient_ids_b) - set(patient_ids_a)),
            "gt_distribution_a": sa.get("gt_distribution_after_locked_768_crop"),
            "gt_distribution_b": sb.get("gt_distribution_after_locked_768_crop"),
        }
        if not diff["ordered_equal"] or diff["rows_only_in_a"] or diff["rows_only_in_b"] or diff["patient_ids_only_in_a"] or diff["patient_ids_only_in_b"] or diff["gt_distribution_a"] != diff["gt_distribution_b"]:
            any_split_difference = True
        split_diffs[split_name] = diff
        if int(sa.get("missing_asset_count", 0)) > 0 or int(sb.get("missing_asset_count", 0)) > 0:
            missing_assets_in_contract = True

    asset_ids = sorted(set(assets_a.keys()) | set(assets_b.keys()))
    asset_diffs: dict[str, Any] = {
        "image_sha_mismatch": [],
        "semantic_mask_sha_mismatch": [],
        "instance_mask_sha_mismatch": [],
        "missing_in_a": [],
        "missing_in_b": [],
        "missing_files": [],
    }
    any_asset_mismatch = False
    for sample_id in asset_ids:
        aa = assets_a.get(sample_id)
        bb = assets_b.get(sample_id)
        if aa is None:
            asset_diffs["missing_in_a"].append(sample_id)
            missing_assets_in_contract = True
            continue
        if bb is None:
            asset_diffs["missing_in_b"].append(sample_id)
            missing_assets_in_contract = True
            continue
        for asset_key, bucket in (
            ("image", "image_sha_mismatch"),
            ("semantic_mask", "semantic_mask_sha_mismatch"),
            ("instance_mask", "instance_mask_sha_mismatch"),
        ):
            a_asset = aa.get(asset_key) or {}
            b_asset = bb.get(asset_key) or {}
            if not bool(a_asset.get("exists", False)) or not bool(b_asset.get("exists", False)):
                asset_diffs["missing_files"].append({"sample_id": sample_id, "asset": asset_key})
                missing_assets_in_contract = True
                continue
            if str(a_asset.get("sha256")) != str(b_asset.get("sha256")):
                asset_diffs[bucket].append(sample_id)
                any_asset_mismatch = True

    if missing_assets_in_contract:
        classification = "INCOMPLETE_COMPARISON"
    elif any_asset_mismatch:
        classification = "DIFFERENT_ASSETS"
    elif any_split_difference or str(payload_a.get("dataset_contract_sha256")) != str(payload_b.get("dataset_contract_sha256")):
        classification = "SAME_ASSETS_DIFFERENT_SPLIT"
    else:
        classification = "IDENTICAL_DATASET"

    return {
        "classification": classification,
        "dataset_contract_sha256_a": payload_a.get("dataset_contract_sha256"),
        "dataset_contract_sha256_b": payload_b.get("dataset_contract_sha256"),
        "split_differences": split_diffs,
        "asset_differences": asset_diffs,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit external dataset identity for split files and referenced assets.")
    parser.add_argument("--dataset-root", type=Path, help="Dataset root containing images/ and masks/.")
    parser.add_argument("--instance-root", type=Path, help="Instance dataset root containing instance_masks/.")
    parser.add_argument("--split", action="append", default=[], help="Split definition in the form name=path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--crop-height", type=int, default=DEFAULT_CROP_H)
    parser.add_argument("--crop-width", type=int, default=DEFAULT_CROP_W)
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None, help="Compare two fingerprint JSON files.")
    args = parser.parse_args()

    if args.compare is not None:
        path_a = _resolve_path(Path(args.compare[0]), REPO_ROOT)
        path_b = _resolve_path(Path(args.compare[1]), REPO_ROOT)
        payload_a = json.loads(path_a.read_text(encoding="utf-8"))
        payload_b = json.loads(path_b.read_text(encoding="utf-8"))
        result = compare_fingerprints(payload_a, payload_b)
        if args.output is not None:
            _write_json(_resolve_path(args.output, REPO_ROOT), result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.dataset_root is None or args.instance_root is None or not args.split:
        raise SystemExit("Fingerprint mode requires --dataset-root, --instance-root, and at least one --split name=path.")

    dataset_root = _resolve_path(args.dataset_root, REPO_ROOT)
    instance_root = _resolve_path(args.instance_root, REPO_ROOT)
    split_map = {name: _resolve_path(path, REPO_ROOT) for name, path in (_parse_split_arg(v) for v in args.split)}
    payload = build_fingerprint(
        dataset_root=dataset_root,
        instance_root=instance_root,
        splits=split_map,
        crop_h=int(args.crop_height),
        crop_w=int(args.crop_width),
    )
    if args.output is not None:
        _write_json(_resolve_path(args.output, REPO_ROOT), payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
