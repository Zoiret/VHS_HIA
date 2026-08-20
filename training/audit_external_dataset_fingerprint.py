from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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


def raw_split_sha256(path: Path) -> str:
    return sha256_file(path)


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


def _semantic_mask_fingerprint(path: Path, *, relative_path: str) -> dict[str, Any]:
    base = _hash_asset(path, relative_path=relative_path)
    if not bool(base["exists"]):
        return {
            "path": base["path"],
            "exists": False,
            "file_size": None,
            "file_sha256": None,
            "decoded_shape": None,
            "decoded_dtype": None,
            "decoded_unique_values": [],
            "decoded_class_counts": {},
            "decoded_pixel_sha256": None,
        }
    decoded = _read_u8(path)
    decoded_c = np.ascontiguousarray(decoded.astype(np.uint8))
    unique, counts = np.unique(decoded_c, return_counts=True)
    return {
        "path": base["path"],
        "exists": True,
        "file_size": base["size"],
        "file_sha256": base["sha256"],
        "decoded_shape": [int(v) for v in decoded_c.shape],
        "decoded_dtype": str(decoded_c.dtype),
        "decoded_unique_values": [int(v) for v in unique.tolist()],
        "decoded_class_counts": {str(int(k)): int(v) for k, v in zip(unique.tolist(), counts.tolist())},
        "decoded_pixel_sha256": hashlib.sha256(decoded_c.tobytes(order="C")).hexdigest(),
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
        "semantic_mask": _semantic_mask_fingerprint(semantic_path, relative_path=semantic_rel),
        "instance_mask": _hash_asset(instance_path, relative_path=f"instance_masks/{sample_id}.png"),
        "gt_count_after_locked_768_crop": gt_count,
    }
    return sample_id, asset_entry, gt_count


def _logical_contract_payload(contract_payload: dict[str, Any]) -> dict[str, Any]:
    logical = json.loads(json.dumps(contract_payload))
    for split_entry in (logical.get("splits") or {}).values():
        if isinstance(split_entry, dict):
            split_entry.pop("raw_split_sha256", None)
    for asset_entry in (logical.get("assets") or {}).values():
        if not isinstance(asset_entry, dict):
            continue
        semantic_entry = asset_entry.get("semantic_mask")
        if isinstance(semantic_entry, dict):
            semantic_entry.pop("file_sha256", None)
            semantic_entry.pop("file_size", None)
    return logical


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
        split_raw_sha = raw_split_sha256(split_path)
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
            "raw_split_sha256": split_raw_sha,
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
    logical_contract_payload = _logical_contract_payload(contract_payload)
    contract_json = json.dumps(logical_contract_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
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


def _sample_pool_summary(contract: dict[str, Any]) -> dict[str, Any]:
    splits = contract.get("splits") or {}
    sample_ids: list[str] = []
    for split_entry in splits.values():
        sample_ids.extend(list((split_entry or {}).get("ordered_sample_ids") or []))
    unique_ids = sorted(set(sample_ids))
    return {
        "unique_sample_count": int(len(unique_ids)),
        "ordered_unique_sample_ids": unique_ids,
    }


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
        "instance_mask_sha_mismatch": [],
        "semantic_mask_sha_mismatch": [],
        "semantic_file_bytes_differ_pixels_identical": [],
        "semantic_pixels_differ": [],
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
        a_sem = aa.get("semantic_mask") or {}
        b_sem = bb.get("semantic_mask") or {}
        if not bool(a_sem.get("exists", False)) or not bool(b_sem.get("exists", False)):
            asset_diffs["missing_files"].append({"sample_id": sample_id, "asset": "semantic_mask"})
            missing_assets_in_contract = True
            continue
        if str(a_sem.get("file_sha256")) != str(b_sem.get("file_sha256")):
            asset_diffs["semantic_mask_sha_mismatch"].append(sample_id)
        if str(a_sem.get("decoded_pixel_sha256")) != str(b_sem.get("decoded_pixel_sha256")):
            asset_diffs["semantic_pixels_differ"].append(sample_id)
            any_asset_mismatch = True
        elif str(a_sem.get("file_sha256")) != str(b_sem.get("file_sha256")):
            asset_diffs["semantic_file_bytes_differ_pixels_identical"].append(sample_id)

    pool_a = _sample_pool_summary(contract_a)
    pool_b = _sample_pool_summary(contract_b)
    if missing_assets_in_contract:
        classification = "INCOMPLETE_COMPARISON"
    elif any_asset_mismatch:
        classification = "DIFFERENT_ASSETS"
    elif any_split_difference:
        classification = "SAME_ASSETS_DIFFERENT_SPLIT"
    else:
        classification = "IDENTICAL_DATASET"

    return {
        "classification": classification,
        "dataset_contract_sha256_a": payload_a.get("dataset_contract_sha256"),
        "dataset_contract_sha256_b": payload_b.get("dataset_contract_sha256"),
        "sample_pool_a": pool_a,
        "sample_pool_b": pool_b,
        "sample_pool_comparison": {
            "exact_same_sample_id_pool": bool(pool_a["ordered_unique_sample_ids"] == pool_b["ordered_unique_sample_ids"]),
            "sample_ids_only_in_a": sorted(set(pool_a["ordered_unique_sample_ids"]) - set(pool_b["ordered_unique_sample_ids"])),
            "sample_ids_only_in_b": sorted(set(pool_b["ordered_unique_sample_ids"]) - set(pool_a["ordered_unique_sample_ids"])),
        },
        "split_differences": split_diffs,
        "asset_differences": asset_diffs,
    }


def compare_split_dirs(canonical_dir: Path, external_dir: Path) -> dict[str, Any]:
    canonical_dir = canonical_dir.resolve()
    external_dir = external_dir.resolve()
    split_names = ("train", "val", "test")
    report: dict[str, Any] = {"status": "MATCH", "splits": {}, "canonical_dir": str(canonical_dir), "external_dir": str(external_dir)}
    for split_name in split_names:
        canonical_path = canonical_dir / f"{split_name}.txt"
        external_path = external_dir / f"{split_name}.txt"
        canonical_rows, canonical_sha = canonical_split_sha256(canonical_path)
        external_exists = external_path.exists()
        external_rows = _read_split_rows(external_path) if external_exists else []
        external_sha = hashlib.sha256(("\n".join(external_rows) + "\n").encode("utf-8")).hexdigest() if external_exists else None
        match = bool(external_exists and canonical_rows == external_rows)
        if not match:
            report["status"] = "DRIFT"
        report["splits"][split_name] = {
            "match": match,
            "canonical_path": str(canonical_path),
            "external_path": str(external_path),
            "canonical_sha256": canonical_sha,
            "external_sha256": external_sha,
            "external_exists": external_exists,
        }
    return report


def sync_split_dirs(canonical_dir: Path, external_dir: Path) -> dict[str, Any]:
    canonical_dir = canonical_dir.resolve()
    external_dir = external_dir.resolve()
    split_names = ("train", "val", "test")
    backup_dir = external_dir / "_backup_before_canonical_sync"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for split_name in split_names:
        canonical_path = canonical_dir / f"{split_name}.txt"
        external_path = external_dir / f"{split_name}.txt"
        if external_path.exists():
            shutil.copy2(str(external_path), str(backup_dir / f"{split_name}.txt"))
        external_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(canonical_path), str(external_path))
    verify_report = compare_split_dirs(canonical_dir, external_dir)
    return {
        "performed": True,
        "backup_dir": str(backup_dir),
        "verify": verify_report,
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
    parser.add_argument("--compare-splits", nargs=2, metavar=("CANONICAL_DIR", "EXTERNAL_DIR"), default=None, help="Compare tracked canonical split manifests against an external split directory.")
    parser.add_argument("--sync", action="store_true", help="With --compare-splits, back up and copy canonical manifests into the external split directory.")
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

    if args.compare_splits is not None:
        canonical_dir = _resolve_path(Path(args.compare_splits[0]), REPO_ROOT)
        external_dir = _resolve_path(Path(args.compare_splits[1]), REPO_ROOT)
        report = compare_split_dirs(canonical_dir, external_dir)
        if args.sync:
            report["sync"] = sync_split_dirs(canonical_dir, external_dir)
        else:
            report["sync"] = {"performed": False}
        if args.output is not None:
            _write_json(_resolve_path(args.output, REPO_ROOT), report)
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
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
