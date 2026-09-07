from __future__ import annotations

import hashlib
import math
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

import bridge_presence_gate_v4 as gate_v4
import bridge_presence_gate_v4_patient_disjoint_dev as dev
import bridge_suppression_head as bridge


DEFAULT_ANALYSIS_DIR = bridge.REPO_ROOT / "training" / "analysis" / "bridge_presence_gate_v5_utility_aligned_dev"
DEFAULT_RUN_DIR = bridge.REPO_ROOT / "training" / "runs" / "unetpp_effb3_bridge_presence_gate_v5_utility_aligned_dev_v1"
UTILITY_TARGET_VERSION = "utility_aligned_two_state_gate_v5_train_only_v1"
UTILITY_ALIGNED_DECISION_NAME = "OPEN_useful_over_CLOSED"


def _utility_choice_for_positive(state_row: dict[str, Any]) -> tuple[int, str]:
    closed = dict(state_row["closed"])
    open_state = dict(state_row["open"])
    closed_success = int(closed["predicted_success50"])
    open_success = int(open_state["predicted_success50"])
    closed_iou = float(closed["predicted_mean_iou"])
    open_iou = float(open_state["predicted_mean_iou"])
    if open_success > closed_success:
        return 1, "success_improvement"
    if open_success < closed_success:
        return 0, "closed_higher_success50"
    if open_iou > closed_iou:
        return 1, "iou_only_improvement"
    if open_iou < closed_iou:
        return 0, "closed_higher_iou"
    return 0, "exact_tie_closed"


def utility_target_row_from_state(
    *,
    feature_row: dict[str, Any],
    state_row: dict[str, Any],
) -> dict[str, Any]:
    if int(state_row["gate_target"]) == 0:
        utility_target = 0
        decision_reason = "negative_forced_closed"
    else:
        utility_target, decision_reason = _utility_choice_for_positive(state_row)
    out = {
        "sample_id": str(feature_row["sample_id"]),
        "patient_id": str(feature_row.get("patient_id", bridge._make_patient_id(str(feature_row["sample_id"])))),
        "gt_count": int(feature_row.get("gt_count", 0)),
        "bridge_positive_target": int(feature_row["bridge_positive_target"]),
        "utility_gate_target": int(utility_target),
        "utility_gate_target_name": "OPEN" if int(utility_target) == 1 else "CLOSED",
        "utility_target_version": UTILITY_TARGET_VERSION,
        "decision_name": UTILITY_ALIGNED_DECISION_NAME,
        "decision_reason": str(decision_reason),
        "closed_success50": int(state_row["closed"]["predicted_success50"]),
        "open_success50": int(state_row["open"]["predicted_success50"]),
        "closed_mean_matched_iou": float(state_row["closed"]["predicted_mean_iou"]),
        "open_mean_matched_iou": float(state_row["open"]["predicted_mean_iou"]),
        "candidate_pixels": int(feature_row.get("candidate_pixels", 0)),
        "candidate_fraction_denominator": int(feature_row.get("candidate_fraction_denominator", dev.TRAIN_SELECTOR_CROP_AREA_PIXELS)),
        "candidate_fraction": float(feature_row["candidate_fraction"]),
        "candidate_component_count": float(feature_row["candidate_component_count"]),
    }
    for scalar_name in gate_v4.SCALAR_FEATURE_NAMES:
        out[str(scalar_name)] = float(feature_row[str(scalar_name)])
    return out


def build_utility_target_rows(
    *,
    feature_rows: list[dict[str, Any]],
    hard_gate_state_cache: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    state_lookup = {str(row["sample_id"]): row for row in hard_gate_state_cache}
    rows: list[dict[str, Any]] = []
    for feature_row in feature_rows:
        sample_id = str(feature_row["sample_id"])
        state_row = state_lookup.get(sample_id)
        if state_row is None:
            raise SystemExit(f"Missing hard-gate state cache row for utility-target sample_id={sample_id}")
        rows.append(utility_target_row_from_state(feature_row=feature_row, state_row=state_row))
    return rows


def build_utility_targets_tensor(utility_target_rows: list[dict[str, Any]]) -> torch.Tensor:
    values = np.asarray([[float(int(row["utility_gate_target"]))] for row in utility_target_rows], dtype=np.float32)
    return torch.from_numpy(values)


def attach_utility_targets_to_feature_rows(
    feature_rows: list[dict[str, Any]],
    utility_target_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_lookup = {str(row["sample_id"]): row for row in utility_target_rows}
    out: list[dict[str, Any]] = []
    for feature_row in feature_rows:
        sample_id = str(feature_row["sample_id"])
        target_row = target_lookup.get(sample_id)
        if target_row is None:
            raise SystemExit(f"Missing utility target row for sample_id={sample_id}")
        current = dict(feature_row)
        current["utility_gate_target"] = int(target_row["utility_gate_target"])
        out.append(current)
    return out


def override_gate_targets_in_state_cache(
    hard_gate_state_cache: list[dict[str, Any]],
    utility_target_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_lookup = {str(row["sample_id"]): int(row["utility_gate_target"]) for row in utility_target_rows}
    out: list[dict[str, Any]] = []
    for row in hard_gate_state_cache:
        sample_id = str(row["sample_id"])
        if sample_id not in target_lookup:
            raise SystemExit(f"Missing utility target override for sample_id={sample_id}")
        current = dict(row)
        current["gate_target"] = int(target_lookup[sample_id])
        out.append(current)
    return out


def summarize_utility_target_rows(utility_target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    utility_open = [row for row in utility_target_rows if int(row["utility_gate_target"]) == 1]
    utility_closed = [row for row in utility_target_rows if int(row["utility_gate_target"]) == 0]
    positives = [row for row in utility_target_rows if int(row["bridge_positive_target"]) == 1]
    negatives = [row for row in utility_target_rows if int(row["bridge_positive_target"]) == 0]
    by_gt: dict[str, dict[str, int]] = {}
    by_patient: dict[str, dict[str, int]] = {}
    for row in utility_target_rows:
        gt_key = f"GT{int(row['gt_count'])}"
        patient_id = str(row["patient_id"])
        by_gt.setdefault(gt_key, {"utility_open": 0, "utility_closed": 0})
        by_gt[gt_key]["utility_open" if int(row["utility_gate_target"]) == 1 else "utility_closed"] += 1
        by_patient.setdefault(patient_id, {"utility_open": 0, "utility_closed": 0, "bridge_positive": 0, "bridge_negative": 0})
        by_patient[patient_id]["utility_open" if int(row["utility_gate_target"]) == 1 else "utility_closed"] += 1
        by_patient[patient_id]["bridge_positive" if int(row["bridge_positive_target"]) == 1 else "bridge_negative"] += 1
    utility_open_positive = [row for row in utility_open if int(row["bridge_positive_target"]) == 1]
    utility_closed_positive = [row for row in utility_closed if int(row["bridge_positive_target"]) == 1]
    return {
        "sample_count": int(len(utility_target_rows)),
        "utility_open_count": int(len(utility_open)),
        "utility_closed_count": int(len(utility_closed)),
        "bridge_positive": {
            "sample_count": int(len(positives)),
            "utility_open_count": int(sum(int(row["utility_gate_target"]) for row in positives)),
            "utility_closed_count": int(sum(1 for row in positives if int(row["utility_gate_target"]) == 0)),
        },
        "bridge_negative": {
            "sample_count": int(len(negatives)),
            "utility_open_count": int(sum(int(row["utility_gate_target"]) for row in negatives)),
            "utility_closed_count": int(sum(1 for row in negatives if int(row["utility_gate_target"]) == 0)),
        },
        "by_gt_count": by_gt,
        "by_patient": by_patient,
        "success_improvement_open_count": int(sum(1 for row in utility_open_positive if str(row["decision_reason"]) == "success_improvement")),
        "iou_only_open_count": int(sum(1 for row in utility_open_positive if str(row["decision_reason"]) == "iou_only_improvement")),
        "bridge_positive_closed_reason_counts": {
            str(reason): int(sum(1 for row in utility_closed_positive if str(row["decision_reason"]) == str(reason)))
            for reason in sorted({str(row["decision_reason"]) for row in utility_closed_positive})
        },
    }


def bridge_positive_vs_utility_target_agreement(utility_target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    for row in utility_target_rows:
        bridge_positive = int(row["bridge_positive_target"])
        utility_target = int(row["utility_gate_target"])
        if bridge_positive == 1 and utility_target == 1:
            tp += 1
        elif bridge_positive == 0 and utility_target == 0:
            tn += 1
        elif bridge_positive == 0 and utility_target == 1:
            fp += 1
        else:
            fn += 1
    total = max(int(len(utility_target_rows)), 1)
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "agreement_fraction": float((tp + tn) / total),
    }


def _evaluate_scalar_target_candidate(
    utility_target_rows: list[dict[str, Any]],
    *,
    scalar_name: str,
    direction: str,
    threshold: float,
) -> dict[str, Any]:
    values = np.asarray([float(row[scalar_name]) for row in utility_target_rows], dtype=np.float64)
    labels = np.asarray([int(row["utility_gate_target"]) for row in utility_target_rows], dtype=np.int64)
    pred = (values >= float(threshold)).astype(np.int64) if str(direction) == "ge" else (values <= float(threshold)).astype(np.int64)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    positives = max(int(np.sum(labels == 1)), 1)
    negatives = max(int(np.sum(labels == 0)), 1)
    sensitivity = float(tp / positives)
    specificity = float(tn / negatives)
    return {
        "scalar": str(scalar_name),
        "direction": str(direction),
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
    }


def build_simple_scalar_baselines(utility_target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines: list[dict[str, Any]] = []
    for scalar_name in gate_v4.SCALAR_FEATURE_NAMES:
        values = np.asarray([float(row[scalar_name]) for row in utility_target_rows], dtype=np.float64)
        if values.size == 0:
            continue
        thresholds = dev._build_midpoint_threshold_candidates(values)
        for direction in ("ge", "le"):
            best: dict[str, Any] | None = None
            for order, (_kind, threshold) in enumerate(thresholds, start=1):
                current = _evaluate_scalar_target_candidate(
                    utility_target_rows,
                    scalar_name=str(scalar_name),
                    direction=str(direction),
                    threshold=float(threshold),
                )
                current["evaluation_order"] = int(order)
                if best is None:
                    best = current
                    continue
                if float(current["balanced_accuracy"]) > float(best["balanced_accuracy"]) + 1.0e-12:
                    best = current
            if best is not None:
                baselines.append(best)
    return sorted(
        baselines,
        key=lambda row: (
            -float(row["balanced_accuracy"]),
            str(row["scalar"]),
            str(row["direction"]),
            float(row["threshold"]),
        ),
    )


def find_conflicting_duplicate_feature_vectors(
    *,
    features_t: torch.Tensor,
    utility_target_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if int(features_t.shape[0]) != int(len(utility_target_rows)):
        raise SystemExit("Feature tensor row count does not match utility target rows for duplicate-feature audit.")
    feature_np = features_t.detach().cpu().numpy()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for idx, row in enumerate(utility_target_rows):
        vector = np.ascontiguousarray(feature_np[idx])
        digest = hashlib.sha256(vector.tobytes()).hexdigest()
        grouped.setdefault(digest, []).append(
            {
                "sample_id": str(row["sample_id"]),
                "patient_id": str(row["patient_id"]),
                "utility_gate_target": int(row["utility_gate_target"]),
            }
        )
    conflicts: list[dict[str, Any]] = []
    for digest, members in sorted(grouped.items()):
        labels = sorted({int(row["utility_gate_target"]) for row in members})
        if len(members) > 1 and len(labels) > 1:
            conflicts.append(
                {
                    "feature_vector_sha256": str(digest),
                    "sample_count": int(len(members)),
                    "utility_labels": labels,
                    "samples": members,
                }
            )
    return conflicts


def build_feature_feasibility_audit(
    *,
    utility_target_rows: list[dict[str, Any]],
    features_t: torch.Tensor,
) -> dict[str, Any]:
    utility_open_count = int(sum(int(row["utility_gate_target"]) for row in utility_target_rows))
    total = max(int(len(utility_target_rows)), 1)
    return {
        "class_balance": {
            "sample_count": int(len(utility_target_rows)),
            "utility_open_count": int(utility_open_count),
            "utility_closed_count": int(len(utility_target_rows) - utility_open_count),
            "utility_open_fraction": float(utility_open_count / total),
        },
        "simple_scalar_baselines": build_simple_scalar_baselines(utility_target_rows),
        "linear_diagnostic_baseline": {
            "available": False,
            "status": "not_run_no_existing_diagnostic_baseline",
        },
        "conflicting_duplicate_feature_vectors": find_conflicting_duplicate_feature_vectors(
            features_t=features_t,
            utility_target_rows=utility_target_rows,
        ),
    }


def build_v5_training_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    future_training = cfg.get("future_training") or {}
    gate_cfg = cfg.get("gate") or {}
    return {
        "features": int(dev.FEATURE_DIMENSION),
        "trainable_parameters": int(dev.TRAINABLE_GATE_PARAMS),
        "optimizer": str(future_training.get("optimizer", "AdamW")),
        "learning_rate": float(future_training.get("learning_rate", 1.0e-3)),
        "steps": int(future_training.get("max_steps", 300)),
        "seed": int(future_training.get("seed", cfg.get("seed", 1337))),
        "threshold": float(gate_cfg.get("gate_threshold", 0.50)),
        "checkpoint_selection": {
            "metric": str(future_training.get("checkpoint_selection_metric", "gate_train_loss")),
            "tie_break": str(future_training.get("checkpoint_tie_break", "earlier_step")),
        },
    }

