from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_reconstruction_policies import _write_json_atomic


def _read_identity_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise SystemExit(f"Invalid JSONL row in {path}: expected object")
        rows.append(obj)
    return rows


def _duplicates(rows: list[dict]) -> list[str]:
    counts = {}
    for row in rows:
        sample = str(row["sample"])
        counts[sample] = counts.get(sample, 0) + 1
    return sorted([sample for sample, count in counts.items() if int(count) > 1])


def _row_key(row: dict) -> tuple:
    return (
        str(row.get("split")),
        str(row.get("sample")),
        str(row.get("image_relative_path")),
        str(row.get("semantic_gt_relative_path")),
        str(row.get("instance_gt_relative_path")),
        str(row.get("center_gt_relative_path")),
    )


def _row_without_index(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "sample_index"}


def _compare_rows(local_rows: list[dict], server_rows: list[dict]) -> dict:
    local_by_sample = {str(row["sample"]): row for row in local_rows}
    server_by_sample = {str(row["sample"]): row for row in server_rows}
    local_samples = set(local_by_sample)
    server_samples = set(server_by_sample)
    shared = sorted(local_samples & server_samples)
    split_diffs = []
    gt_count_diffs = []
    relpath_diffs = []
    image_hash_diffs = []
    semantic_hash_diffs = []
    instance_hash_diffs = []
    center_hash_diffs = []
    for sample in shared:
        left = local_by_sample[sample]
        right = server_by_sample[sample]
        if str(left["split"]) != str(right["split"]):
            split_diffs.append({"sample": sample, "local": left["split"], "server": right["split"]})
        if int(left["gt_instance_count"]) != int(right["gt_instance_count"]):
            gt_count_diffs.append({"sample": sample, "local": int(left["gt_instance_count"]), "server": int(right["gt_instance_count"])})
        rel_diffs = {}
        for key in ("image_relative_path", "semantic_gt_relative_path", "instance_gt_relative_path", "center_gt_relative_path"):
            if str(left[key]) != str(right[key]):
                rel_diffs[key] = {"local": str(left[key]), "server": str(right[key])}
        if rel_diffs:
            relpath_diffs.append({"sample": sample, "differences": rel_diffs})
        for key, bucket in (
            ("image_sha256", image_hash_diffs),
            ("semantic_gt_sha256", semantic_hash_diffs),
            ("instance_gt_sha256", instance_hash_diffs),
            ("center_gt_sha256", center_hash_diffs),
        ):
            if str(left[key]) != str(right[key]):
                bucket.append({"sample": sample, "local": str(left[key]), "server": str(right[key])})
    ordering_only = bool(
        len(local_rows) == len(server_rows)
        and sorted((_row_without_index(row) for row in local_rows), key=lambda item: _row_key(item))
        == sorted((_row_without_index(row) for row in server_rows), key=lambda item: _row_key(item))
        and [_row_key(row) for row in local_rows] != [_row_key(row) for row in server_rows]
    )
    same_identity = not any(
        [
            sorted(local_samples - server_samples),
            sorted(server_samples - local_samples),
            split_diffs,
            gt_count_diffs,
            relpath_diffs,
            image_hash_diffs,
            semantic_hash_diffs,
            instance_hash_diffs,
            center_hash_diffs,
            _duplicates(local_rows),
            _duplicates(server_rows),
        ]
    )
    if same_identity:
        status = "exact_match"
    elif not image_hash_diffs and not semantic_hash_diffs and not instance_hash_diffs and not center_hash_diffs:
        status = "legacy_path_dependent_hash_mismatch"
    else:
        status = "dataset_content_mismatch"
    return {
        "status": status,
        "same_canonical_identity": bool(same_identity),
        "sample_ids_missing_locally": sorted(server_samples - local_samples),
        "sample_ids_missing_on_server": sorted(local_samples - server_samples),
        "duplicate_samples": {
            "local": _duplicates(local_rows),
            "server": _duplicates(server_rows),
        },
        "split_differences": split_diffs,
        "gt_count_differences": gt_count_diffs,
        "relative_path_differences": relpath_diffs,
        "image_hash_differences": image_hash_diffs,
        "semantic_gt_hash_differences": semantic_hash_diffs,
        "instance_gt_hash_differences": instance_hash_diffs,
        "center_gt_hash_differences": center_hash_diffs,
        "ordering_only_differences": bool(ordering_only),
        "execution_root_only_differences": bool(status == "legacy_path_dependent_hash_mismatch"),
        "local_row_count": int(len(local_rows)),
        "server_row_count": int(len(server_rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-identity-manifest", type=str, required=True)
    ap.add_argument("--server-identity-manifest", type=str, required=True)
    ap.add_argument("--output", type=str, default="training/analysis/manifest_diff.json")
    args = ap.parse_args()

    local_path = Path(args.local_identity_manifest).resolve()
    server_path = Path(args.server_identity_manifest).resolve()
    output_path = Path(args.output).resolve()
    diff = _compare_rows(_read_identity_manifest(local_path), _read_identity_manifest(server_path))
    diff["local_identity_manifest"] = str(local_path)
    diff["server_identity_manifest"] = str(server_path)
    _write_json_atomic(output_path, diff)
    print(json.dumps(diff, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
