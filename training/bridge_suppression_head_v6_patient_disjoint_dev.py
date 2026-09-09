from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e

import bridge_presence_gate_v4_patient_disjoint_dev as gate_dev
import bridge_suppression_head as bridge


DEFAULT_ANALYSIS_DIR = bridge.REPO_ROOT / "training" / "analysis" / "bridge_suppression_head_v6_patient_disjoint_dev"
DEFAULT_RUN_DIR = bridge.REPO_ROOT / "training" / "runs" / "unetpp_effb3_bridge_suppression_frozen_semantic_patient_disjoint_v6_dev"
DEFAULT_CONFIG = bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_patient_disjoint_v6_dev.yaml"
HISTORICAL_FULL_CONFIG = bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_100ep.yaml"
V6_EXPERIMENT_VERSION = "clean_full_data_v2_bridge_head_patient_disjoint_v6_dev"
V6_CHECKPOINT_SELECTION_VERSION = "gate_train_reconstruction_v1"
V6_SAFETY_MAX_NEGATIVE_REGRESSIONS = 10
V6_SAFETY_MAX_NEGATIVE_TOPOLOGY_CHANGES = 12
STACKED_CACHE_FIELDS = ("x_0_4", "x_2_2", "p_leaf", "candidate_mask", "bridge_target")
V6_REQUIRED_METADATA_FIELDS = ("sample_id", "patient_id", "gt_count", "bridge_positive", "candidate_pixels", "bridge_pixels")

DEVELOPMENT_REFERENCES = {
    "closed": {
        "positive_success50": int(gate_dev.LOCKED_VAL_REFERENCES["always_closed"]["positive_success50"]),
        "positive_mean_matched_iou": float(gate_dev.LOCKED_VAL_REFERENCES["always_closed"]["positive_mean_matched_iou"]),
        "negative_regressions": 0,
        "negative_topology_changes": 0,
    },
    "historical_v2_open": {
        "positive_success50": int(gate_dev.LOCKED_VAL_REFERENCES["always_open"]["positive_success50"]),
        "positive_mean_matched_iou": float(gate_dev.LOCKED_VAL_REFERENCES["always_open"]["positive_mean_matched_iou"]),
        "negative_regressions": int(gate_dev.LOCKED_VAL_REFERENCES["always_open"]["negative_regressions"]),
        "negative_topology_changes": int(gate_dev.LOCKED_VAL_REFERENCES["always_open"]["negative_topology_changes"]),
    },
    "historical_safe_oracle": {
        "positive_success50": int(gate_dev.LOCKED_VAL_REFERENCES["safe_two_state_oracle"]["positive_success50"]),
        "positive_mean_matched_iou": float(gate_dev.LOCKED_VAL_REFERENCES["safe_two_state_oracle"]["positive_mean_matched_iou"]),
        "negative_regressions": int(gate_dev.LOCKED_VAL_REFERENCES["safe_two_state_oracle"]["negative_regressions"]),
        "negative_topology_changes": int(gate_dev.LOCKED_VAL_REFERENCES["safe_two_state_oracle"]["negative_topology_changes"]),
    },
}


def load_historical_micro_manifest() -> dict[str, Any]:
    return bridge.read_locked_micro_manifest(bridge.MICRO_MANIFEST_V2_PATH)


def compute_microset_overlap(contract: dict[str, Any], micro_manifest: dict[str, Any]) -> dict[str, Any]:
    micro_sample_ids = {str(v) for v in micro_manifest.get("sample_ids") or []}
    micro_patient_ids = {str(row["patient_id"]) for row in micro_manifest.get("rows") or []}
    train_sample_ids = {str(v) for v in contract.get("train_sample_ids") or []}
    val_sample_ids = {str(v) for v in contract.get("val_sample_ids") or []}
    train_patient_ids = {str(v) for v in contract.get("train_patient_ids") or []}
    val_patient_ids = {str(v) for v in contract.get("val_patient_ids") or []}
    return {
        "train_overlap_count": int(len(micro_sample_ids & train_sample_ids)),
        "val_overlap_count": int(len(micro_sample_ids & val_sample_ids)),
        "train_overlap_samples": sorted(micro_sample_ids & train_sample_ids),
        "val_overlap_samples": sorted(micro_sample_ids & val_sample_ids),
        "overlapping_patients": sorted(micro_patient_ids & (train_patient_ids | val_patient_ids)),
    }


def summarize_false_bridge_targets(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_pixels = [int(row["candidate_pixels"]) for row in records]
    bridge_pixels = [int(row["bridge_pixels"]) for row in records]
    positive_rows = [row for row in records if int(row["bridge_pixels"]) > 0]
    zero_rows = [row for row in records if int(row["bridge_pixels"]) == 0]
    by_gt: dict[str, dict[str, Any]] = {}
    by_patient: dict[str, dict[str, Any]] = {}
    per_sample = []
    for row in records:
        gt_key = f"GT{int(row['gt_count'])}"
        by_gt.setdefault(
            gt_key,
            {
                "sample_count": 0,
                "positive_target_samples": 0,
                "zero_target_samples": 0,
                "candidate_pixels": 0,
                "false_bridge_pixels": 0,
            },
        )
        by_patient.setdefault(
            str(row["patient_id"]),
            {
                "sample_count": 0,
                "positive_target_samples": 0,
                "zero_target_samples": 0,
                "candidate_pixels": 0,
                "false_bridge_pixels": 0,
            },
        )
        for bucket in (by_gt[gt_key], by_patient[str(row["patient_id"])]):
            bucket["sample_count"] += 1
            bucket["positive_target_samples"] += int(int(row["bridge_pixels"]) > 0)
            bucket["zero_target_samples"] += int(int(row["bridge_pixels"]) == 0)
            bucket["candidate_pixels"] += int(row["candidate_pixels"])
            bucket["false_bridge_pixels"] += int(row["bridge_pixels"])
        per_sample.append(
            {
                "sample_id": str(row["sample_id"]),
                "patient_id": str(row["patient_id"]),
                "gt_count": int(row["gt_count"]),
                "bridge_positive": int(row["bridge_positive"]),
                "candidate_pixels": int(row["candidate_pixels"]),
                "false_bridge_pixels": int(row["bridge_pixels"]),
                "false_bridge_over_candidate": float(int(row["bridge_pixels"]) / max(int(row["candidate_pixels"]), 1)),
            }
        )
    for bucket in list(by_gt.values()) + list(by_patient.values()):
        bucket["false_bridge_over_candidate"] = float(bucket["false_bridge_pixels"] / max(bucket["candidate_pixels"], 1))
    return {
        "sample_count": int(len(records)),
        "positive_target_samples": int(len(positive_rows)),
        "zero_target_samples": int(len(zero_rows)),
        "total_candidate_pixels": int(sum(candidate_pixels)),
        "total_false_bridge_pixels": int(sum(bridge_pixels)),
        "false_bridge_over_candidate": float(sum(bridge_pixels) / max(sum(candidate_pixels), 1)),
        "per_sample_distribution": {
            "false_bridge_pixels_min": int(min(bridge_pixels)) if bridge_pixels else 0,
            "false_bridge_pixels_mean": float(np.mean(bridge_pixels)) if bridge_pixels else 0.0,
            "false_bridge_pixels_median": float(np.median(bridge_pixels)) if bridge_pixels else 0.0,
            "false_bridge_pixels_max": int(max(bridge_pixels)) if bridge_pixels else 0,
        },
        "by_gt_count": by_gt,
        "by_patient": by_patient,
        "per_sample_rows": per_sample,
    }


def build_v6_model_with_fresh_bridge_head(cfg: dict[str, Any], device: torch.device) -> tuple[bridge.FrozenSemanticBridgeSuppressionModel, dict[str, Any], dict[str, Any]]:
    model = bridge.build_model_from_cfg(cfg).to(device)
    semantic_info = bridge.load_semantic_checkpoint(
        model,
        bridge._resolve_repo_path((cfg.get("train") or {}).get("init_checkpoint"), bridge.DEFAULT_SEMANTIC_CHECKPOINT),
    )
    optimizer, optimizer_meta = bridge.build_optimizer(model, cfg)
    head_state = {
        "context_projection_state_sha256": bridge.canonical_model_state_sha256(model.context_projection.state_dict()),
        "bridge_head_state_sha256": bridge.canonical_model_state_sha256(model.bridge_head.state_dict()),
    }
    del optimizer
    return model, semantic_info, {**optimizer_meta, **head_state}


def build_v6_cached_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_by_sample: dict[str, dict[str, Any]] = {}
    for row in records:
        sample_id = str(row.get("sample_id"))
        if not sample_id:
            raise SystemExit("V6 cached-record construction failed: raw record missing sample_id.")
        if sample_id in raw_by_sample:
            raise SystemExit(f"V6 cached-record construction failed: duplicate raw sample_id={sample_id}.")
        raw_by_sample[sample_id] = row
    tensorized = bridge.cache_microset_features(records)
    tensor_by_sample: dict[str, dict[str, Any]] = {}
    for row in tensorized:
        sample_id = str(row.get("sample_id"))
        if not sample_id:
            raise SystemExit("V6 cached-record construction failed: tensorized record missing sample_id.")
        if sample_id in tensor_by_sample:
            raise SystemExit(f"V6 cached-record construction failed: duplicate tensorized sample_id={sample_id}.")
        tensor_by_sample[sample_id] = row
    raw_ids = set(raw_by_sample)
    tensor_ids = set(tensor_by_sample)
    if raw_ids != tensor_ids:
        raise SystemExit(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "v6_cached_record_sample_set_mismatch",
                    "raw_only_sample_ids": sorted(raw_ids - tensor_ids),
                    "tensorized_only_sample_ids": sorted(tensor_ids - raw_ids),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    merged: list[dict[str, Any]] = []
    for sample_id in sorted(raw_ids):
        raw_row = raw_by_sample[sample_id]
        tensor_row = dict(tensor_by_sample[sample_id])
        if str(tensor_row.get("sample_id")) != str(raw_row.get("sample_id")):
            raise SystemExit(f"V6 cached-record construction failed: sample_id mismatch for sample_id={sample_id}.")
        current = dict(raw_row)
        current.update(tensor_row)
        merged.append(current)
    return merged


def _tensor_contract_row(value: Any) -> dict[str, Any]:
    if torch.is_tensor(value):
        return {
            "python_type": type(value).__name__,
            "dtype": str(value.dtype),
            "shape": [int(v) for v in value.shape],
        }
    if isinstance(value, np.ndarray):
        return {
            "python_type": type(value).__name__,
            "dtype": str(value.dtype),
            "shape": [int(v) for v in value.shape],
        }
    return {
        "python_type": type(value).__name__ if value is not None else None,
        "dtype": None,
        "shape": None,
    }


def summarize_cached_record_contract(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"sample_count": 0, "fields": {}, "metadata_fields": {}, "unique_sample_ids": 0, "patient_count": 0}
    first = records[0]
    sample_ids = [str(row.get("sample_id")) for row in records]
    patient_ids = [str(row.get("patient_id")) for row in records if row.get("patient_id") is not None]
    return {
        "sample_count": int(len(records)),
        "unique_sample_ids": int(len(set(sample_ids))),
        "patient_count": int(len(set(patient_ids))),
        "fields": {
            str(key): _tensor_contract_row(first.get(key))
            for key in STACKED_CACHE_FIELDS
        },
        "metadata_fields": {
            str(key): type(first.get(key)).__name__ if first.get(key) is not None else None
            for key in V6_REQUIRED_METADATA_FIELDS
        },
    }


def metadata_contract_table() -> list[dict[str, Any]]:
    return [
        {"key": "sample_id", "raw_mined_record_present": True, "historical_tensorized_record_present": True, "v6_evaluator_required": True, "dtype_type": "str", "classification": "scientific_identity_and_reporting"},
        {"key": "patient_id", "raw_mined_record_present": True, "historical_tensorized_record_present": True, "v6_evaluator_required": True, "dtype_type": "str", "classification": "reporting_only"},
        {"key": "gt_count", "raw_mined_record_present": True, "historical_tensorized_record_present": True, "v6_evaluator_required": True, "dtype_type": "int", "classification": "scientific_and_reporting"},
        {"key": "bridge_positive", "raw_mined_record_present": True, "historical_tensorized_record_present": True, "v6_evaluator_required": True, "dtype_type": "int", "classification": "scientific_and_reporting"},
        {"key": "candidate_pixels", "raw_mined_record_present": True, "historical_tensorized_record_present": True, "v6_evaluator_required": True, "dtype_type": "int", "classification": "scientific_and_reporting"},
        {"key": "bridge_pixels", "raw_mined_record_present": True, "historical_tensorized_record_present": True, "v6_evaluator_required": True, "dtype_type": "int", "classification": "scientific_and_reporting"},
        {"key": "image_path", "raw_mined_record_present": True, "historical_tensorized_record_present": True, "v6_evaluator_required": False, "dtype_type": "str", "classification": "diagnostic_only"},
    ]


def validate_train_cache_contract(records: list[dict[str, Any]], *, expected_sample_ids: list[str] | None = None) -> dict[str, Any]:
    if not records:
        raise SystemExit("V6 train cache validation failed: no cached GATE_TRAIN records.")
    summary = summarize_cached_record_contract(records)
    errors: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for row in records:
        sample_id = str(row.get("sample_id"))
        if not sample_id:
            errors.append({"sample_id": None, "field": "sample_id", "reason": "missing_metadata"})
        elif sample_id in seen_sample_ids:
            errors.append({"sample_id": sample_id, "field": "sample_id", "reason": "duplicate_metadata"})
        else:
            seen_sample_ids.add(sample_id)
        for key in V6_REQUIRED_METADATA_FIELDS:
            if row.get(key) is None:
                errors.append({"sample_id": sample_id, "field": str(key), "reason": "missing_metadata"})
        p_leaf_shape: tuple[int, ...] | None = None
        for key in STACKED_CACHE_FIELDS:
            value = row.get(key)
            if not torch.is_tensor(value):
                errors.append({"sample_id": sample_id, "field": str(key), "reason": "python_type", "actual_type": type(value).__name__})
                continue
            if str(value.dtype) != "torch.float32":
                errors.append({"sample_id": sample_id, "field": str(key), "reason": "dtype", "actual_dtype": str(value.dtype)})
            if int(value.ndim) != 3:
                errors.append({"sample_id": sample_id, "field": str(key), "reason": "ndim", "actual_ndim": int(value.ndim)})
            if not bool(torch.isfinite(value).all().item()):
                errors.append({"sample_id": sample_id, "field": str(key), "reason": "non_finite"})
            if str(key) == "p_leaf":
                p_leaf_shape = tuple(int(v) for v in value.shape)
                if p_leaf_shape[:1] != (1,):
                    errors.append({"sample_id": sample_id, "field": str(key), "reason": "channel_dim", "actual_shape": list(p_leaf_shape)})
            elif str(key) in {"candidate_mask", "bridge_target"} and p_leaf_shape is not None:
                current_shape = tuple(int(v) for v in value.shape)
                if current_shape != p_leaf_shape:
                    errors.append(
                        {
                            "sample_id": sample_id,
                            "field": str(key),
                            "reason": "shape_mismatch_vs_p_leaf",
                            "actual_shape": list(current_shape),
                            "expected_shape": list(p_leaf_shape),
                        }
                    )
    if expected_sample_ids is not None:
        expected = {str(v) for v in expected_sample_ids}
        actual = {str(row.get("sample_id")) for row in records if row.get("sample_id") is not None}
        if actual != expected:
            errors.append(
                {
                    "field": "sample_id",
                    "reason": "manifest_sample_set_mismatch",
                    "missing_from_cache": sorted(expected - actual),
                    "unexpected_in_cache": sorted(actual - expected),
                }
            )
    if errors:
        raise SystemExit(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "v6_train_cache_contract_mismatch",
                    "summary": summary,
                    "errors": errors[:20],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return summary


def build_loss_hparam_parity_report(v6_cfg: dict[str, Any], historical_cfg: dict[str, Any]) -> dict[str, Any]:
    hist_loss = historical_cfg.get("loss") or {}
    v6_loss = v6_cfg.get("loss") or {}
    hist_train = historical_cfg.get("train") or {}
    v6_train = v6_cfg.get("train") or {}
    return {
        "loss": {
            "candidate_bce_terms": "balanced_bce = 0.5 * positive_bce + 0.5 * negative_bce",
            "positive_negative_weighting": "0.5 / 0.5",
            "dice_term": True,
            "lambda_negative_mean": float(v6_loss.get("lambda_negative_mean", 0.0)),
            "lambda_negative_hard": float(v6_loss.get("lambda_negative_hard", 0.0)),
            "negative_hard_topk_fraction": float(v6_loss.get("negative_hard_topk_fraction", 0.01)),
            "historical_lambda_negative_mean": float(hist_loss.get("lambda_negative_mean", 0.0)),
            "historical_lambda_negative_hard": float(hist_loss.get("lambda_negative_hard", 0.0)),
            "historical_negative_hard_topk_fraction": float(hist_loss.get("negative_hard_topk_fraction", 0.01)),
        },
        "optimizer": "AdamW",
        "learning_rate": float(v6_train.get("lr", 1.0e-3)),
        "lr_context_projection": float(v6_train.get("lr_context_projection", v6_train.get("lr", 1.0e-3))),
        "lr_bridge_head": float(v6_train.get("lr_bridge_head", v6_train.get("lr", 1.0e-3))),
        "weight_decay": float(v6_train.get("weight_decay", 1.0e-5)),
        "batch_size": int(v6_train.get("batch_size", 16)),
        "epoch_count": int(v6_train.get("epochs", 100)),
        "step_epoch_semantics": "epoch_based_training_with_train_only_checkpoint_selection",
        "historical_config_path": str(HISTORICAL_FULL_CONFIG.resolve()),
        "historical_batch_size": int(hist_train.get("batch_size", 16)),
        "historical_epoch_count": int(hist_train.get("epochs", 100)),
        "historical_lr": float(hist_train.get("lr", 1.0e-3)),
        "historical_weight_decay": float(hist_train.get("weight_decay", 1.0e-5)),
    }


def build_v6_checkpoint_selection_policy() -> dict[str, Any]:
    return {
        "version": V6_CHECKPOINT_SELECTION_VERSION,
        "metric": "gate_train_reconstruction_pred_success50_then_mean_iou",
        "selection_rule": "maximize predicted_success50 on GATE_TRAIN, then predicted_mean_iou, tie -> earlier epoch",
        "uses_validation": False,
    }


def evaluate_v6_development_decision(val_eval: dict[str, Any]) -> dict[str, Any]:
    pred = val_eval["reconstruction"]
    closed = DEVELOPMENT_REFERENCES["closed"]
    historical_open = DEVELOPMENT_REFERENCES["historical_v2_open"]
    positive_utility = bool(
        int(pred["positive_success50"]) > int(closed["positive_success50"])
        and float(pred["positive_mean_matched_iou"]) > float(closed["positive_mean_matched_iou"])
    )
    safety_ok = bool(
        int(pred["negative_regressions"]) <= int(V6_SAFETY_MAX_NEGATIVE_REGRESSIONS)
        and int(pred["negative_topology_changes"]) <= int(V6_SAFETY_MAX_NEGATIVE_TOPOLOGY_CHANGES)
    )
    return {
        "full_data_v2_representation_useful_on_reused_dev_val": "YES" if (positive_utility and safety_ok) else "NO",
        "local_v2_bridge_representation_insufficient": "YES" if not (positive_utility and safety_ok) else "NO",
        "positive_utility_rule_satisfied": bool(positive_utility),
        "safety_rule_satisfied": bool(safety_ok),
        "historical_open_reference": historical_open,
    }


def evaluate_open_on_cached_records(
    *,
    model: bridge.FrozenSemanticBridgeSuppressionModel,
    cached_records: list[dict[str, Any]],
    device: torch.device,
    threshold: float = bridge.BRIDGE_REMOVE_THRESHOLD,
) -> dict[str, Any]:
    recon = bridge.evaluate_reconstruction_levels_on_cached(model, cached_records, device, threshold=float(threshold))
    batch = bridge.stack_cached_batch(cached_records, device)
    metadata_by_sample: dict[str, dict[str, Any]] = {}
    for row in cached_records:
        sample_id = str(row["sample_id"])
        if sample_id in metadata_by_sample:
            raise SystemExit(f"V6 evaluation failed: duplicate cached sample_id={sample_id}.")
        metadata_by_sample[sample_id] = row
    with torch.no_grad():
        outputs = model.bridge_forward_from_cached(
            x_0_4=batch["x_0_4"],
            x_2_2=batch["x_2_2"],
            p_leaf=batch["p_leaf"],
        )
        bridge_probs = torch.sigmoid(outputs["bridge_logits"])
    pixel = bridge.compute_binary_metrics_from_domain(
        bridge_probs=bridge_probs,
        bridge_target=batch["bridge_target"],
        candidate_mask=batch["candidate_mask"],
        threshold=float(threshold),
    )
    positive_indices = [idx for idx, row in enumerate(cached_records) if int(row["bridge_positive"]) == 1]
    positive_pixel = bridge.compute_binary_metrics_for_subset(
        bridge_probs=bridge_probs,
        bridge_target=batch["bridge_target"],
        candidate_mask=batch["candidate_mask"],
        subset_indices=positive_indices,
        threshold=float(threshold),
    )
    required_recon_keys = {
        "sample_id",
        "bridge_positive",
        "gt_count",
        "candidate_pixels",
        "gt_bridge_pixels",
        "predicted_removed_pixels",
        "predicted_removed_fraction",
        "start_mean_iou",
        "predicted_mean_iou",
        "oracle_mean_iou",
        "start_success50",
        "predicted_success50",
        "oracle_success50",
        "component_topology_changed",
    }
    missing_recon_keys = sorted(required_recon_keys - set(str(v) for v in (recon.get("per_sample") or [])[0].keys())) if recon.get("per_sample") else []
    if missing_recon_keys:
        raise SystemExit(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "v6_reconstruction_per_sample_contract_mismatch",
                    "missing_reconstruction_keys": missing_recon_keys,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    per_sample_rows = [
        {
            "sample_id": str(row["sample_id"]),
            "patient_id": str(metadata_by_sample[str(row["sample_id"])]["patient_id"]),
            "gt_count": int(metadata_by_sample[str(row["sample_id"])]["gt_count"]),
            "bridge_positive": int(row["bridge_positive"]),
            "candidate_pixels": int(row["candidate_pixels"]),
            "gt_bridge_pixels": int(row["gt_bridge_pixels"]),
            "predicted_removed_pixels": int(row["predicted_removed_pixels"]),
            "removed_over_candidate": float(row["predicted_removed_fraction"]),
            "start_mean_iou": float(row["start_mean_iou"]),
            "predicted_mean_iou": float(row["predicted_mean_iou"]),
            "oracle_mean_iou": float(row["oracle_mean_iou"]),
            "start_success50": int(row["start_success50"]),
            "predicted_success50": int(row["predicted_success50"]),
            "oracle_success50": int(row["oracle_success50"]),
            "component_topology_changed": int(row["component_topology_changed"]),
        }
        for row in recon["per_sample"]
    ]
    return {
        "pixel": {
            "bridge_precision": float(pixel["precision"]),
            "bridge_recall": float(pixel["recall"]),
            "bridge_f1": float(pixel["f1"]),
            "positive_only_precision": float(positive_pixel["precision"]),
            "positive_only_recall": float(positive_pixel["recall"]),
            "positive_only_f1": float(positive_pixel["f1"]),
            "removed_over_candidate": float(recon["removal_calibration"]["all_removed_over_candidate"]),
        },
        "reconstruction": {
            "overall_success50": int(recon["reconstruction"]["p50_minus_predicted_bridge"]["all_iou_ge_0.50_count"]),
            "overall_mean_matched_iou": float(recon["reconstruction"]["p50_minus_predicted_bridge"]["mean_matched_iou"]),
            "positive_success50": int(recon["positive_subset"]["reconstruction"]["p50_minus_predicted_bridge"]["all_iou_ge_0.50_count"]),
            "positive_mean_matched_iou": float(recon["positive_subset"]["reconstruction"]["p50_minus_predicted_bridge"]["mean_matched_iou"]),
            "negative_regressions": int(recon["negative_subset"]["num_regresses"]),
            "negative_topology_changes": int(recon["negative_subset"]["num_component_topology_changes"]),
            "negative_removed_fraction": float(recon["negative_subset"]["fraction_of_candidate_pixels_removed"]),
        },
        "per_sample": per_sample_rows,
        "raw_reconstruction": recon,
    }


def patient_level_report(per_sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_patient: dict[str, list[dict[str, Any]]] = {}
    for row in per_sample_rows:
        by_patient.setdefault(str(row["patient_id"]), []).append(row)
    patient_rows: list[dict[str, Any]] = []
    for patient_id, rows in sorted(by_patient.items()):
        positives = [row for row in rows if int(row["bridge_positive"]) == 1]
        negatives = [row for row in rows if int(row["bridge_positive"]) == 0]
        patient_rows.append(
            {
                "patient_id": str(patient_id),
                "sample_count": int(len(rows)),
                "positive_success50": int(sum(int(row["predicted_success50"]) for row in positives)),
                "positive_mean_matched_iou": float(np.mean([float(row["predicted_mean_iou"]) for row in positives])) if positives else 0.0,
                "negative_regressions": int(sum(1 for row in negatives if float(row["predicted_mean_iou"]) + 1.0e-9 < float(row["start_mean_iou"]))),
                "negative_topology_changes": int(sum(int(row["component_topology_changed"]) for row in negatives)),
                "removed_over_candidate": float(np.mean([float(row["removed_over_candidate"]) for row in rows])) if rows else 0.0,
            }
        )
    return {
        "rows": patient_rows,
        "macro_mean": {
            "positive_success50": float(np.mean([float(row["positive_success50"]) for row in patient_rows])) if patient_rows else 0.0,
            "positive_mean_matched_iou": float(np.mean([float(row["positive_mean_matched_iou"]) for row in patient_rows])) if patient_rows else 0.0,
            "negative_regressions": float(np.mean([float(row["negative_regressions"]) for row in patient_rows])) if patient_rows else 0.0,
            "negative_topology_changes": float(np.mean([float(row["negative_topology_changes"]) for row in patient_rows])) if patient_rows else 0.0,
        },
    }
