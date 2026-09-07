from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bridge_presence_gate_v4 as gate_v4
import bridge_presence_gate_v4_patient_disjoint_dev as dev
import bridge_suppression_head as bridge
import run_bridge_presence_gate_v4_patient_disjoint_dev_preflight as preflight_runner
import train_bridge_presence_gate_v4_patient_disjoint_dev as train_runner


DEFAULT_CONFIG = bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v4_patient_disjoint_dev_v1.yaml"
DEFAULT_AUDIT_DIR = bridge.REPO_ROOT / "training" / "analysis" / "bridge_presence_gate_v4_scalar_baseline_provenance_audit"
ROW_FIELD_ORDER = [
    "sample_id",
    "patient_id",
    "gt_count",
    "bridge_target",
    "candidate_pixels",
    "candidate_fraction",
    "candidate_fraction_denominator",
    "candidate_mask_shape",
    "scalar_source",
    "dtype",
]
HISTORICAL_SOURCE_COMMIT = "5eb63e5"
MERGED_SOURCE_COMMIT = "1d5e0b1"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELD_ORDER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _canonical_row_payload(rows: list[dict[str, Any]]) -> bytes:
    lines = []
    for row in rows:
        ordered = [str(row[field]) for field in ROW_FIELD_ORDER]
        lines.append("\t".join(ordered))
    return ("\n".join(lines) + "\n").encode("utf-8")


def canonical_row_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_row_payload(rows)).hexdigest()


def canonical_scientific_selector_input_sha256(rows: list[dict[str, Any]]) -> str:
    return dev._scientific_selector_input_sha256(rows)


def _normalize_shape(shape: Any) -> str:
    if isinstance(shape, str):
        return shape
    if isinstance(shape, (list, tuple)):
        return "x".join(str(int(v)) for v in shape)
    return str(shape)


def build_forensic_selector_rows(
    *,
    records: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    scalar_source_function: str,
    candidate_fraction_source: str,
    target_source: str,
) -> list[dict[str, Any]]:
    by_feature = {str(row["sample_id"]): dict(row) for row in feature_rows}
    rows: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        feature = by_feature[sample_id]
        mask_np = record.get("candidate_mask_np")
        if mask_np is None:
            mask_tensor = record.get("candidate_mask")
            if mask_tensor is None:
                raise SystemExit(f"Missing candidate mask for forensic row {sample_id}")
            mask_np = np.asarray(mask_tensor)
        mask_shape = tuple(int(v) for v in np.asarray(mask_np).shape[-2:])
        denominator = int(np.prod(mask_shape))
        rows.append(
            {
                "sample_id": sample_id,
                "patient_id": str(record["patient_id"]),
                "gt_count": int(record["gt_count"]),
                "bridge_target": int(feature["bridge_positive_target"]),
                "candidate_pixels": int(record["candidate_pixels"]),
                "candidate_fraction": format(float(feature["candidate_fraction"]), ".17g"),
                "candidate_fraction_denominator": int(denominator),
                "candidate_mask_shape": _normalize_shape(mask_shape),
                "scalar_source": str(scalar_source_function),
                "dtype": str(
                    f"candidate_fraction=float64;candidate_mask=uint8;target=int64;"
                    f"candidate_fraction_source={candidate_fraction_source};target_source={target_source}"
                ),
            }
        )
    return sorted(rows, key=lambda row: str(row["sample_id"]))


def summarize_forensic_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt_counts = [int(row["gt_count"]) for row in rows]
    targets = [int(row["bridge_target"]) for row in rows]
    fractions = np.asarray([float(row["candidate_fraction"]) for row in rows], dtype=np.float64)
    return {
        "row_count": int(len(rows)),
        "gt1_count": int(sum(1 for value in gt_counts if int(value) == 1)),
        "gt2_count": int(sum(1 for value in gt_counts if int(value) == 2)),
        "gt3_count": int(sum(1 for value in gt_counts if int(value) == 3)),
        "positive_count": int(sum(targets)),
        "negative_count": int(len(rows) - sum(targets)),
        "candidate_fraction_min": float(np.min(fractions)) if fractions.size else 0.0,
        "candidate_fraction_mean": float(np.mean(fractions)) if fractions.size else 0.0,
        "candidate_fraction_max": float(np.max(fractions)) if fractions.size else 0.0,
        "scientific_selector_input_sha256": canonical_scientific_selector_input_sha256(rows),
        "audit_row_sha256": canonical_row_sha256(rows),
        "sample_ids": [str(row["sample_id"]) for row in rows],
    }


def diff_forensic_row_sets(reference_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ref_map = {str(row["sample_id"]): dict(row) for row in reference_rows}
    cur_map = {str(row["sample_id"]): dict(row) for row in current_rows}
    missing_sample_ids = sorted(sample_id for sample_id in ref_map if sample_id not in cur_map)
    extra_sample_ids = sorted(sample_id for sample_id in cur_map if sample_id not in ref_map)
    target_differences: list[dict[str, Any]] = []
    candidate_fraction_differences: list[dict[str, Any]] = []
    for sample_id in sorted(set(ref_map) & set(cur_map)):
        ref = ref_map[sample_id]
        cur = cur_map[sample_id]
        if int(ref["bridge_target"]) != int(cur["bridge_target"]):
            target_differences.append(
                {
                    "sample_id": str(sample_id),
                    "reference_target": int(ref["bridge_target"]),
                    "current_target": int(cur["bridge_target"]),
                }
            )
        ref_fraction = float(ref["candidate_fraction"])
        cur_fraction = float(cur["candidate_fraction"])
        ref_pixels = int(ref["candidate_pixels"])
        cur_pixels = int(cur["candidate_pixels"])
        if abs(ref_fraction - cur_fraction) > 0.0 or ref_pixels != cur_pixels:
            candidate_fraction_differences.append(
                {
                    "sample_id": str(sample_id),
                    "target": int(ref["bridge_target"]),
                    "reference_candidate_pixels": int(ref_pixels),
                    "current_candidate_pixels": int(cur_pixels),
                    "reference_candidate_fraction": float(ref_fraction),
                    "current_candidate_fraction": float(cur_fraction),
                    "absolute_difference": float(abs(ref_fraction - cur_fraction)),
                }
            )
    return {
        "reference_row_count": int(len(reference_rows)),
        "current_row_count": int(len(current_rows)),
        "reference_scientific_selector_input_sha256": canonical_scientific_selector_input_sha256(reference_rows),
        "current_scientific_selector_input_sha256": canonical_scientific_selector_input_sha256(current_rows),
        "reference_audit_row_sha256": canonical_row_sha256(reference_rows),
        "current_audit_row_sha256": canonical_row_sha256(current_rows),
        "missing_sample_ids": missing_sample_ids,
        "extra_sample_ids": extra_sample_ids,
        "target_differences": target_differences,
        "candidate_fraction_differences": candidate_fraction_differences,
    }


def run_historical_selector(feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(gate_v4.simple_scalar_threshold_audit(feature_rows)["best_scalar"])


def _selector_result_payload(feature_rows: list[dict[str, Any]], *, selector_function: str) -> dict[str, Any]:
    selected_rule = run_historical_selector(feature_rows)
    frozen_threshold = float(dev.FROZEN_SIMPLE_SCALAR_RULE["threshold"])
    actual_threshold = float(selected_rule["threshold"])
    return {
        **selected_rule,
        "selector_function": str(selector_function),
        "matches_frozen_threshold": bool(abs(actual_threshold - frozen_threshold) <= 1.0e-12),
        "frozen_rule_reproduced": bool(abs(actual_threshold - frozen_threshold) <= 1.0e-12),
        "expected_frozen_threshold": float(frozen_threshold),
    }


def find_original_artifacts(root: Path) -> list[str]:
    candidates: list[str] = []
    patterns = ("*bridge_presence_gate_v4*json", "*bridge_presence_gate_v4*csv", "*preflight*json", "*selector*json", "*selector*csv")
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            rel = str(path.resolve())
            if "scalar_baseline_provenance_audit" in rel:
                continue
            if "patient_disjoint" in rel or "selector" in rel:
                candidates.append(rel)
    return sorted(dict.fromkeys(candidates))


def _direct_prepare_rows(
    *,
    cfg: dict[str, Any],
    sample_ids: list[str],
    device: Any,
    frozen_model: Any,
) -> dict[str, Any]:
    split_txt = bridge._resolve_repo_path((cfg.get("dataset") or {}).get("train_txt", dev.DEFAULT_SOURCE_SPLIT), dev.DEFAULT_SOURCE_SPLIT)
    records = bridge.mine_bridge_records_for_split(
        cfg=cfg,
        split_txt=split_txt,
        model=frozen_model,
        device=device,
        cache_features=True,
        selected_sample_ids=sample_ids,
    )
    cached_records = bridge.cache_microset_features(records)
    annotated: list[dict[str, Any]] = []
    for row in cached_records:
        current = dict(row)
        current["gate_target"] = int(current["bridge_positive"])
        annotated.append(current)
    logits, _ = gate_v4.compute_frozen_v2_bridge_logits(frozen_model, annotated, device, return_diagnostics=True)
    feature_rows, _, _, _ = gate_v4.extract_gate_feature_rows(annotated, logits)
    return {
        "records": records,
        "cached_records": annotated,
        "feature_rows": feature_rows,
    }


def _package_path_audit(
    *,
    rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    historical_selector_source: str,
) -> dict[str, Any]:
    return {
        "rows": rows,
        "summary": summarize_forensic_rows(rows),
        "selector_result": {
            **run_historical_selector(feature_rows),
            "selector_function": str(historical_selector_source),
        },
    }


def run_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    audit_dir = bridge._resolve_repo_path(((cfg.get("analysis") or {}).get("scalar_baseline_provenance_audit_dir")), DEFAULT_AUDIT_DIR)
    audit_dir.mkdir(parents=True, exist_ok=True)
    bridge._seed_everything(int(cfg.get("seed", 1337)))
    manifest_stage = preflight_runner._prepare_manifest(cfg)
    contract = dict(manifest_stage["manifest"]["contract"])
    device = manifest_stage["device"]
    frozen_model, frozen_v2_info = gate_v4.load_frozen_v2_pixel_model_from_cfg(cfg, device)

    historical_preflight = _direct_prepare_rows(
        cfg=cfg,
        sample_ids=list(contract["train_sample_ids"]),
        device=device,
        frozen_model=frozen_model,
    )
    current_preflight = dev._prepare_split_preflight_core(
        cfg=cfg,
        sample_ids=list(contract["train_sample_ids"]),
        device=device,
        frozen_model=frozen_model,
        enforce_frozen_scalar_rule=False,
        caller="forensic_audit.current_preflight",
        source_function="bridge_presence_gate_v4_scalar_baseline_provenance_audit.run_pipeline",
    )
    current_training = train_runner._prepare_training_inputs_core(cfg, enforce_frozen_scalar_rule=False)["train_prepared"]

    historical_rows = build_forensic_selector_rows(
        records=historical_preflight["cached_records"],
        feature_rows=historical_preflight["feature_rows"],
        scalar_source_function="gate_v4.extract_gate_feature_rows@5eb63e5 -> gate_v4.simple_scalar_threshold_audit@5eb63e5",
        candidate_fraction_source="np.mean(candidate_mask_np.astype(np.float32) > 0)",
        target_source="bridge.mine_bridge_records_for_split.bridge_positive",
    )
    current_preflight_rows = build_forensic_selector_rows(
        records=current_preflight["cached_records"],
        feature_rows=current_preflight["feature_rows"],
        scalar_source_function="dev.prepare_split_preflight -> gate_v4.extract_gate_feature_rows",
        candidate_fraction_source="np.mean(candidate_mask_np.astype(np.float32) > 0)",
        target_source="dev.prepare_split_preflight.gate_target=bridge_positive",
    )
    current_training_rows = build_forensic_selector_rows(
        records=current_training["cached_records"],
        feature_rows=current_training["feature_rows"],
        scalar_source_function="train_runner._prepare_training_inputs -> dev.prepare_split_preflight -> gate_v4.extract_gate_feature_rows",
        candidate_fraction_source="np.mean(candidate_mask_np.astype(np.float32) > 0)",
        target_source="dev.prepare_split_preflight.gate_target=bridge_positive",
    )

    historical_audit = _package_path_audit(
        rows=historical_rows,
        feature_rows=historical_preflight["feature_rows"],
        historical_selector_source="gate_v4.simple_scalar_threshold_audit@5eb63e5",
    )
    current_preflight_audit = _package_path_audit(
        rows=current_preflight_rows,
        feature_rows=current_preflight["feature_rows"],
        historical_selector_source="gate_v4.simple_scalar_threshold_audit@64a3561",
    )
    current_training_audit = _package_path_audit(
        rows=current_training_rows,
        feature_rows=current_training["feature_rows"],
        historical_selector_source="gate_v4.simple_scalar_threshold_audit@64a3561",
    )

    diff_hist_vs_preflight = diff_forensic_row_sets(historical_rows, current_preflight_rows)
    diff_hist_vs_training = diff_forensic_row_sets(historical_rows, current_training_rows)
    diff_preflight_vs_training = diff_forensic_row_sets(current_preflight_rows, current_training_rows)
    artifact_hits = find_original_artifacts(bridge.REPO_ROOT / "training")

    for name, rows in (
        ("historical_preflight_rows", historical_rows),
        ("current_preflight_rows", current_preflight_rows),
        ("current_training_rows", current_training_rows),
    ):
        _write_csv(audit_dir / f"{name}.csv", rows)
        bridge._write_json(audit_dir / f"{name}.json", rows)

    bridge._write_json(audit_dir / "historical_vs_current_preflight_diff.json", diff_hist_vs_preflight)
    bridge._write_json(audit_dir / "historical_vs_current_training_diff.json", diff_hist_vs_training)
    bridge._write_json(audit_dir / "current_preflight_vs_training_diff.json", diff_preflight_vs_training)

    selector_results = {
        "historical_preflight_rows": _selector_result_payload(
            historical_preflight["feature_rows"],
            selector_function="gate_v4.simple_scalar_threshold_audit@5eb63e5",
        ),
        "current_preflight_rows": _selector_result_payload(
            current_preflight["feature_rows"],
            selector_function="gate_v4.simple_scalar_threshold_audit@64a3561",
        ),
        "current_training_rows": _selector_result_payload(
            current_training["feature_rows"],
            selector_function="gate_v4.simple_scalar_threshold_audit@64a3561",
        ),
    }
    bridge._write_json(audit_dir / "selector_results.json", selector_results)

    output = {
        "audit_execution_success": True,
        "historical_path": {
            "source_commit": HISTORICAL_SOURCE_COMMIT,
            "merged_commit": MERGED_SOURCE_COMMIT,
            "row_builder": [
                "bridge.mine_bridge_records_for_split",
                "bridge.cache_microset_features",
                "gate_v4.compute_frozen_v2_bridge_logits",
                "gate_v4.extract_gate_feature_rows",
                "gate_v4.simple_scalar_threshold_audit",
            ],
            "target_source": "bridge.mine_bridge_records_for_split -> bridge_positive -> dev.prepare_split_preflight gate_target",
            "candidate_fraction_source": "gate_v4.extract_gate_feature_rows -> np.mean(candidate_mask_np.astype(np.float32))",
            "selector": "gate_v4.simple_scalar_threshold_audit",
            "underlying_function_diffs_from_5eb63e5_to_head": {
                "mine_bridge_records_for_split": False,
                "cache_microset_features": False,
                "extract_gate_feature_rows": False,
                "simple_scalar_threshold_audit": False,
            },
            "reconstruction_kind": "reconstructed_historical_path_using_byte-identical current implementations for the inspected functions",
        },
        "row_sets": {
            "historical_preflight_rows": historical_audit,
            "current_preflight_rows": current_preflight_audit,
            "current_training_rows": current_training_audit,
        },
        "subset_differences": {
            "historical_vs_current_preflight": diff_hist_vs_preflight,
            "historical_vs_current_training": diff_hist_vs_training,
            "current_preflight_vs_current_training": diff_preflight_vs_training,
        },
        "target_differences": {
            "historical_vs_current_preflight": diff_hist_vs_preflight["target_differences"],
            "historical_vs_current_training": diff_hist_vs_training["target_differences"],
            "current_preflight_vs_current_training": diff_preflight_vs_training["target_differences"],
        },
        "candidate_fraction_differences": {
            "historical_vs_current_preflight": diff_hist_vs_preflight["candidate_fraction_differences"],
            "historical_vs_current_training": diff_hist_vs_training["candidate_fraction_differences"],
            "current_preflight_vs_current_training": diff_preflight_vs_training["candidate_fraction_differences"],
            "definition": "candidate_fraction = candidate_pixels / (candidate_mask_shape_h * candidate_mask_shape_w)",
        },
        "selector_results": selector_results,
        "original_artifact_search": {
            "found": bool(artifact_hits),
            "paths": artifact_hits,
            "contains_actual_selector_rows": False,
        },
        "frozen_v2_checkpoint": frozen_v2_info,
    }
    bridge._write_json(audit_dir / "provenance_summary.json", output)
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    cfg_path = bridge._resolve_repo_path(args.config, DEFAULT_CONFIG)
    cfg = bridge._read_yaml(cfg_path)
    cfg["_config_path"] = str(cfg_path.resolve())
    result = run_pipeline(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
