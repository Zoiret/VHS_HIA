from __future__ import annotations

import hashlib
import itertools
import json
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
import bridge_suppression_head as bridge
import train_bridge_presence_gate_v4 as micro_runner


DEFAULT_SOURCE_SPLIT = bridge.DEFAULT_TRAIN_SPLIT
DEFAULT_SOURCE_SHA256 = "f5e920ffaf54c0a0034c457cf3c951f71e186a9f35e3fe67a5eee95737b2ee82"
DEFAULT_MANIFEST_DIR = bridge.REPO_ROOT / "training" / "manifests" / "bridge_presence_gate_v4_patient_disjoint_dev_v1"
DEFAULT_ANALYSIS_DIR = bridge.REPO_ROOT / "training" / "analysis" / "bridge_presence_gate_v4_patient_disjoint_preflight"
SPLIT_ALGORITHM_VERSION = "patient_disjoint_bridge_presence_gate_v4_dev_v1"
TRAIN_SCALAR_SELECTION_VERSION = "train_only_balanced_accuracy_v1"
SUCCESS_CRITERIA_V1_VERSION = "patient_disjoint_v4_val_success_v1"
SUCCESS_CRITERIA_V2_VERSION = "patient_disjoint_v4_val_success_v2_feasibility_corrected"
FEATURE_DIMENSION = 105
TRAINABLE_GATE_PARAMS = 1713
ALLOWED_SPLIT_FIELDS = ("sample_id", "patient_id", "gt_count", "bridge_positive")
FROZEN_V1_EXPECTED_SPLIT = {
    "train": {
        "samples": 121,
        "patients": 17,
        "bridge_positive": 41,
        "bridge_negative": 80,
    },
    "val": {
        "samples": 36,
        "patients": 5,
        "bridge_positive": 12,
        "bridge_negative": 24,
    },
    "patient_overlap": 0,
    "sample_overlap": 0,
}


def _read_source_split_entries(path: Path) -> list[dict[str, str]]:
    bridge._assert_safe_path(path)
    rows = bridge._canonical_split_rows(path)
    entries: list[dict[str, str]] = []
    for row in rows:
        parts = row.split()
        if len(parts) < 2:
            raise SystemExit(f"Malformed split row in {path}: {row}")
        sample_id = Path(parts[0]).stem
        entries.append(
            {
                "sample_id": str(sample_id),
                "patient_id": bridge._make_patient_id(str(sample_id)),
                "row_text": str(row),
            }
        )
    return entries


def build_split_metadata_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in records:
        current = {
            "sample_id": str(row["sample_id"]),
            "patient_id": str(row["patient_id"]),
            "gt_count": int(row["gt_count"]),
            "bridge_positive": int(row["bridge_positive"]),
        }
        out.append(current)
    return sorted(out, key=lambda row: str(row["sample_id"]))


def summarize_split_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    patient_ids = sorted({str(row["patient_id"]) for row in rows})
    gt1 = int(sum(1 for row in rows if int(row["gt_count"]) == 1))
    gt2 = int(sum(1 for row in rows if int(row["gt_count"]) == 2))
    gt3 = int(sum(1 for row in rows if int(row["gt_count"]) == 3))
    pos = int(sum(int(row["bridge_positive"]) for row in rows))
    sample_count = int(len(rows))
    patient_sample_counts: dict[str, int] = {}
    for row in rows:
        pid = str(row["patient_id"])
        patient_sample_counts[pid] = int(patient_sample_counts.get(pid, 0) + 1)
    return {
        "sample_count": sample_count,
        "patient_count": int(len(patient_ids)),
        "patient_ids": patient_ids,
        "bridge_positive_count": pos,
        "bridge_negative_count": int(sample_count - pos),
        "gt1_count": gt1,
        "gt2_count": gt2,
        "gt3_count": gt3,
        "positive_fraction": float(pos / max(sample_count, 1)),
        "patient_sample_counts": {str(k): int(v) for k, v in sorted(patient_sample_counts.items())},
        "sample_ids": [str(row["sample_id"]) for row in rows],
    }


def _split_score(
    *,
    val_summary: dict[str, Any],
    train_summary: dict[str, Any],
    full_summary: dict[str, Any],
    seed: int,
) -> tuple[Any, ...]:
    target_val_samples = int(round(float(full_summary["sample_count"]) * 0.23))
    target_val_pos = int(round(float(full_summary["bridge_positive_count"]) * 0.23))
    target_val_gt2 = int(round(float(full_summary["gt2_count"]) * 0.23))
    target_val_gt3 = int(round(float(full_summary["gt3_count"]) * 0.23))
    val_counts = list((val_summary.get("patient_sample_counts") or {}).values())
    max_patient_load = int(max(val_counts)) if val_counts else 0
    min_patient_load = int(min(val_counts)) if val_counts else 0
    sample_spread = int(max_patient_load - min_patient_load)
    val_patients = list(val_summary.get("patient_ids") or [])
    seeded_tiebreak = hashlib.sha256(f"{int(seed)}:{','.join(val_patients)}".encode("utf-8")).hexdigest()
    return (
        abs(int(val_summary["patient_count"]) - 5),
        abs(int(val_summary["sample_count"]) - int(target_val_samples)),
        abs(int(val_summary["bridge_positive_count"]) - int(target_val_pos)),
        abs(int(val_summary["gt2_count"]) - int(target_val_gt2)) + abs(int(val_summary["gt3_count"]) - int(target_val_gt3)),
        abs(float(val_summary["positive_fraction"]) - float(full_summary["positive_fraction"])),
        abs(float(train_summary["positive_fraction"]) - float(full_summary["positive_fraction"])),
        int(max_patient_load),
        int(sample_spread),
        seeded_tiebreak,
        tuple(val_patients),
    )


def select_patient_disjoint_split(
    metadata_rows: list[dict[str, Any]],
    *,
    seed: int,
    preferred_val_patient_count: int = 5,
    fallback_val_patient_counts: list[int] | tuple[int, ...] = (4, 6),
) -> dict[str, Any]:
    rows = [dict(row) for row in metadata_rows]
    full_summary = summarize_split_metadata(rows)
    patient_ids = list(full_summary["patient_ids"])
    patient_to_rows: dict[str, list[dict[str, Any]]] = {str(pid): [] for pid in patient_ids}
    for row in rows:
        patient_to_rows[str(row["patient_id"])].append(row)
    candidate_sizes = [int(preferred_val_patient_count)] + [int(v) for v in fallback_val_patient_counts if int(v) != int(preferred_val_patient_count)]
    best: dict[str, Any] | None = None
    for val_patient_count in candidate_sizes:
        current_best: dict[str, Any] | None = None
        for val_patients_tuple in itertools.combinations(patient_ids, int(val_patient_count)):
            val_patients = set(str(v) for v in val_patients_tuple)
            val_rows = [row for row in rows if str(row["patient_id"]) in val_patients]
            train_rows = [row for row in rows if str(row["patient_id"]) not in val_patients]
            val_summary = summarize_split_metadata(val_rows)
            train_summary = summarize_split_metadata(train_rows)
            if int(val_summary["bridge_positive_count"]) <= 0 or int(val_summary["bridge_negative_count"]) <= 0:
                continue
            if int(train_summary["bridge_positive_count"]) <= 0 or int(train_summary["bridge_negative_count"]) <= 0:
                continue
            if int(val_summary["gt2_count"]) <= 0 or int(val_summary["gt3_count"]) <= 0:
                continue
            current = {
                "train_rows": train_rows,
                "val_rows": val_rows,
                "train_summary": train_summary,
                "val_summary": val_summary,
                "score": _split_score(val_summary=val_summary, train_summary=train_summary, full_summary=full_summary, seed=int(seed)),
            }
            if current_best is None or tuple(current["score"]) < tuple(current_best["score"]):
                current_best = current
        if current_best is not None:
            best = current_best
            break
    if best is None:
        raise SystemExit("Failed to find a patient-disjoint split satisfying bridge-positive/negative and GT2/GT3 coverage.")
    train_ids = {str(row["sample_id"]) for row in best["train_rows"]}
    val_ids = {str(row["sample_id"]) for row in best["val_rows"]}
    best["patient_overlap"] = int(len(set(best["train_summary"]["patient_ids"]) & set(best["val_summary"]["patient_ids"])))
    best["sample_overlap"] = int(len(train_ids & val_ids))
    best["algorithm"] = {
        "version": SPLIT_ALGORITHM_VERSION,
        "seed": int(seed),
        "preferred_val_patient_count": int(preferred_val_patient_count),
        "fallback_val_patient_counts": [int(v) for v in fallback_val_patient_counts],
        "model_features_used": False,
        "model_performance_used": False,
        "allowed_fields": list(ALLOWED_SPLIT_FIELDS),
        "ranking_rule": [
            "abs(val_patient_count-5)",
            "abs(val_sample_count-target_23pct)",
            "abs(val_bridge_positive_count-target_23pct)",
            "abs(val_gt2_count-target_23pct)+abs(val_gt3_count-target_23pct)",
            "abs(val_positive_fraction-full_positive_fraction)",
            "abs(train_positive_fraction-full_positive_fraction)",
            "max_val_patient_sample_count",
            "val_patient_sample_spread",
            "seeded_sha256_tiebreak",
            "lexicographic_patient_ids",
        ],
    }
    return best


def build_split_texts(
    *,
    source_entries: list[dict[str, str]],
    train_sample_ids: list[str],
    val_sample_ids: list[str],
) -> dict[str, str]:
    train_set = {str(v) for v in train_sample_ids}
    val_set = {str(v) for v in val_sample_ids}
    if train_set & val_set:
        raise SystemExit("Patient-disjoint split writer received overlapping sample IDs.")
    source_ids = [str(entry["sample_id"]) for entry in source_entries]
    missing = sorted((train_set | val_set) - set(source_ids))
    if missing:
        raise SystemExit(f"Split contains sample IDs not present in source split: {missing[:10]}")
    train_rows = [str(entry["row_text"]) for entry in source_entries if str(entry["sample_id"]) in train_set]
    val_rows = [str(entry["row_text"]) for entry in source_entries if str(entry["sample_id"]) in val_set]
    return {
        "train_text": ("\n".join(train_rows) + "\n") if train_rows else "",
        "val_text": ("\n".join(val_rows) + "\n") if val_rows else "",
    }


def build_manifest_contract(
    *,
    source_split_path: Path,
    source_sha256: str,
    split_payload: dict[str, Any],
) -> dict[str, Any]:
    train_summary = dict(split_payload["train_summary"])
    val_summary = dict(split_payload["val_summary"])
    return {
        "source_path": bridge._repo_relative_canonical_path(source_split_path),
        "source_canonical_sha256": str(source_sha256),
        "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
        "seed": int(split_payload["algorithm"]["seed"]),
        "train_patient_ids": list(train_summary["patient_ids"]),
        "val_patient_ids": list(val_summary["patient_ids"]),
        "train_sample_ids": list(train_summary["sample_ids"]),
        "val_sample_ids": list(val_summary["sample_ids"]),
        "train_summary": train_summary,
        "val_summary": val_summary,
        "patient_overlap": int(split_payload["patient_overlap"]),
        "sample_overlap": int(split_payload["sample_overlap"]),
        "algorithm": dict(split_payload["algorithm"]),
    }


def load_frozen_manifest(
    *,
    manifest_dir: Path,
    contract_payload: dict[str, Any],
    train_text: str,
    val_text: str,
) -> dict[str, Any]:
    contract_path = manifest_dir / "contract.json"
    train_path = manifest_dir / "train.txt"
    val_path = manifest_dir / "val.txt"
    existing = [path.exists() for path in (contract_path, train_path, val_path)]
    if not all(existing):
        raise SystemExit(f"Frozen patient-disjoint manifest v1 must already exist and be complete: {manifest_dir}")
    current_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    current_train = train_path.read_text(encoding="utf-8")
    current_val = val_path.read_text(encoding="utf-8")
    if current_contract != contract_payload or current_train != train_text or current_val != val_text:
        raise SystemExit(f"Frozen patient-disjoint manifest v1 differs from regenerated deterministic expectation: {manifest_dir}")
    train_summary = current_contract.get("train_summary") or {}
    val_summary = current_contract.get("val_summary") or {}
    expected = FROZEN_V1_EXPECTED_SPLIT
    checks = [
        (int(train_summary.get("sample_count", -1)), int(expected["train"]["samples"]), "train.sample_count"),
        (int(train_summary.get("patient_count", -1)), int(expected["train"]["patients"]), "train.patient_count"),
        (int(train_summary.get("bridge_positive_count", -1)), int(expected["train"]["bridge_positive"]), "train.bridge_positive_count"),
        (int(train_summary.get("bridge_negative_count", -1)), int(expected["train"]["bridge_negative"]), "train.bridge_negative_count"),
        (int(val_summary.get("sample_count", -1)), int(expected["val"]["samples"]), "val.sample_count"),
        (int(val_summary.get("patient_count", -1)), int(expected["val"]["patients"]), "val.patient_count"),
        (int(val_summary.get("bridge_positive_count", -1)), int(expected["val"]["bridge_positive"]), "val.bridge_positive_count"),
        (int(val_summary.get("bridge_negative_count", -1)), int(expected["val"]["bridge_negative"]), "val.bridge_negative_count"),
        (int(current_contract.get("patient_overlap", -1)), int(expected["patient_overlap"]), "patient_overlap"),
        (int(current_contract.get("sample_overlap", -1)), int(expected["sample_overlap"]), "sample_overlap"),
    ]
    mismatches = [
        {"field": field, "actual": int(actual), "expected": int(expected_value)}
        for actual, expected_value, field in checks
        if int(actual) != int(expected_value)
    ]
    if mismatches:
        raise SystemExit(json.dumps({
            "status": "blocked",
            "reason": "frozen_manifest_v1_mismatch",
            "manifest_dir": str(manifest_dir.resolve()),
            "mismatches": mismatches,
        }, ensure_ascii=False, indent=2))
    return {
        "contract_path": str(contract_path.resolve()),
        "train_path": str(train_path.resolve()),
        "val_path": str(val_path.resolve()),
        "created": False,
        "contract": current_contract,
    }


def load_frozen_semantic_only_model(cfg: dict[str, Any], device: torch.device) -> tuple[bridge.FrozenSemanticBridgeSuppressionModel, dict[str, Any]]:
    model = bridge.build_model_from_cfg(cfg).to(device)
    semantic_info = bridge.load_semantic_checkpoint(
        model,
        bridge._resolve_repo_path((cfg.get("train") or {}).get("init_checkpoint"), bridge.DEFAULT_SEMANTIC_CHECKPOINT),
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model, semantic_info


def summarize_split_record_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_pixels = [int(row["candidate_pixels"]) for row in records]
    bridge_pixels = [int(row["bridge_pixels"]) for row in records]
    return {
        "samples": int(len(records)),
        "patients": int(len({str(row["patient_id"]) for row in records})),
        "bridge_positive": int(sum(int(row["bridge_positive"]) for row in records)),
        "bridge_negative": int(sum(1 for row in records if int(row["bridge_positive"]) == 0)),
        "positive_fraction": float(sum(int(row["bridge_positive"]) for row in records) / max(len(records), 1)),
        "gt1_count": int(sum(1 for row in records if int(row["gt_count"]) == 1)),
        "gt2_count": int(sum(1 for row in records if int(row["gt_count"]) == 2)),
        "gt3_count": int(sum(1 for row in records if int(row["gt_count"]) == 3)),
        "candidate_pixels": {
            "min": int(min(candidate_pixels)) if candidate_pixels else 0,
            "mean": float(np.mean(candidate_pixels)) if candidate_pixels else 0.0,
            "median": float(np.median(candidate_pixels)) if candidate_pixels else 0.0,
            "max": int(max(candidate_pixels)) if candidate_pixels else 0,
        },
        "bridge_pixels": {
            "min": int(min(bridge_pixels)) if bridge_pixels else 0,
            "mean": float(np.mean(bridge_pixels)) if bridge_pixels else 0.0,
            "median": float(np.median(bridge_pixels)) if bridge_pixels else 0.0,
            "max": int(max(bridge_pixels)) if bridge_pixels else 0,
        },
    }


def summarize_hard_gate_states(
    split_records: list[dict[str, Any]],
    hard_gate_state_cache: list[dict[str, Any]],
) -> dict[str, Any]:
    by_state: dict[str, list[dict[str, Any]]] = {
        "closed": [dict(row["closed"], sample_id=row["sample_id"], gate_target=row["gate_target"]) for row in hard_gate_state_cache],
        "open": [dict(row["open"], sample_id=row["sample_id"], gate_target=row["gate_target"]) for row in hard_gate_state_cache],
    }

    def _summary(state_rows: list[dict[str, Any]], *, open_state: bool) -> dict[str, Any]:
        pos_rows = [row for row in state_rows if int(row["gate_target"]) == 1]
        neg_rows = [row for row in state_rows if int(row["gate_target"]) == 0]
        return {
            "success50": int(sum(int(row["predicted_success50"]) for row in state_rows)),
            "mean_matched_iou": float(np.mean([float(row["predicted_mean_iou"]) for row in state_rows])) if state_rows else 0.0,
            "positive_success50": int(sum(int(row["predicted_success50"]) for row in pos_rows)),
            "positive_mean_matched_iou": float(np.mean([float(row["predicted_mean_iou"]) for row in pos_rows])) if pos_rows else 0.0,
            "negative_regressions": int(sum(1 for row in neg_rows if float(row["predicted_mean_iou"]) + 1.0e-9 < float(row["start_mean_iou"]))) if open_state else 0,
            "negative_topology_changes": int(sum(int(row["component_topology_changed"]) for row in neg_rows)) if open_state else 0,
        }

    positive_union_upper_bound = int(
        sum(
            1
            for row in hard_gate_state_cache
            if int(row["gate_target"]) == 1
            and (int(row["closed"]["predicted_success50"]) == 1 or int(row["open"]["predicted_success50"]) == 1)
        )
    )
    return {
        "always_closed": _summary(by_state["closed"], open_state=False),
        "always_open": _summary(by_state["open"], open_state=True),
        "cache_construction_time_seconds": float(sum(float(row[state]["predicted_reconstruction_runtime_seconds"]) for row in hard_gate_state_cache for state in ("closed", "open"))),
        "optimized_normalizer_used": True,
        "record_stats": summarize_split_record_stats(split_records),
        "two_state_positive_success50_union_upper_bound": int(positive_union_upper_bound),
    }


def _oracle_positive_choice(state_row: dict[str, Any]) -> str:
    closed = state_row["closed"]
    open_state = state_row["open"]
    closed_key = (
        int(closed["predicted_success50"]),
        float(closed["predicted_mean_iou"]),
        1,
    )
    open_key = (
        int(open_state["predicted_success50"]),
        float(open_state["predicted_mean_iou"]),
        0,
    )
    return "open" if open_key > closed_key else "closed"


def compute_safe_two_state_oracle(hard_gate_state_cache: list[dict[str, Any]]) -> dict[str, Any]:
    per_sample: list[dict[str, Any]] = []
    positive_open = 0
    positive_closed = 0
    for row in hard_gate_state_cache:
        if int(row["gate_target"]) == 0:
            chosen_name = "closed"
        else:
            chosen_name = _oracle_positive_choice(row)
        chosen = row[str(chosen_name)]
        if int(row["gate_target"]) == 1 and str(chosen_name) == "open":
            positive_open += 1
        if int(row["gate_target"]) == 1 and str(chosen_name) == "closed":
            positive_closed += 1
        per_sample.append(
            {
                "sample_id": str(row["sample_id"]),
                "gate_target": int(row["gate_target"]),
                "chosen_state": str(chosen_name).upper(),
                "bridge_positive": int(row["bridge_positive"]),
                **dict(chosen),
            }
        )
    positives = [row for row in per_sample if int(row["gate_target"]) == 1]
    negatives = [row for row in per_sample if int(row["gate_target"]) == 0]
    return {
        "positive_success50": int(sum(int(row["predicted_success50"]) for row in positives)),
        "positive_mean_matched_iou": float(np.mean([float(row["predicted_mean_iou"]) for row in positives])) if positives else 0.0,
        "overall_success50": int(sum(int(row["predicted_success50"]) for row in per_sample)),
        "overall_mean_matched_iou": float(np.mean([float(row["predicted_mean_iou"]) for row in per_sample])) if per_sample else 0.0,
        "negative_regressions": int(sum(1 for row in negatives if float(row["predicted_mean_iou"]) + 1.0e-9 < float(row["start_mean_iou"]))),
        "negative_topology_changes": int(sum(int(row["component_topology_changed"]) for row in negatives)),
        "positive_open_count": int(positive_open),
        "positive_closed_count": int(positive_closed),
        "per_sample": per_sample,
    }


def select_train_only_scalar_rule(train_feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    audit = gate_v4.simple_scalar_threshold_audit(train_feature_rows)
    return {
        "selection_version": TRAIN_SCALAR_SELECTION_VERSION,
        "train_simple_gate_threshold_exists": bool(audit["simple_gate_threshold_exists"]),
        "selected_rule": audit["best_scalar"],
        "selection_uses_validation_labels": False,
    }


def build_predeclared_baselines() -> dict[str, Any]:
    return {
        "always_closed": {"definition": "gate_open = false for every sample"},
        "always_open": {"definition": "gate_open = true for every sample"},
        "train_derived_simple_scalar_gate": {
            "definition": "single frozen scalar/direction/threshold selected on GATE_TRAIN only",
            "selection_version": TRAIN_SCALAR_SELECTION_VERSION,
        },
        "trained_v4_gate": {
            "definition": "1713-parameter V4 gate with fixed threshold 0.50",
            "feature_dimensionality": FEATURE_DIMENSION,
            "trainable_parameters": TRAINABLE_GATE_PARAMS,
        },
    }


def build_predeclared_success_criteria_v1(
    *,
    val_summary: dict[str, Any],
    always_closed: dict[str, Any],
    always_open: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    val_positive_count = int(val_summary["bridge_positive"])
    min_positive_success50 = int(max(int(always_closed["positive_success50"]) + 1, int(math.ceil(float(val_positive_count) * 0.5))))
    min_positive_mean_iou = float(max(float(always_closed["positive_mean_matched_iou"]) + 1.0e-6, float(always_open["positive_mean_matched_iou"]) - 0.10))
    train_cfg = cfg.get("train") or {}
    gate_cfg = cfg.get("gate") or {}
    return {
        "version": SUCCESS_CRITERIA_V1_VERSION,
        "declared_before_training": True,
        "safety": {
            "negative_regressions": 0,
            "negative_topology_changes": 0,
        },
        "utility": {
            "positive_success50_min": int(min_positive_success50),
            "positive_mean_matched_iou_min": float(min_positive_mean_iou),
            "must_exceed_always_closed_positive_success50": True,
            "must_exceed_always_closed_positive_mean_matched_iou": True,
        },
        "model_selection_metric": "gate_train_loss",
        "checkpoint_tie_break_rule": "earlier_step",
        "gate_threshold_selection_policy": f"fixed_{float(gate_cfg.get('gate_threshold', 0.50)):.2f}",
        "maximum_training_steps": int((cfg.get("future_training") or {}).get("max_steps", train_cfg.get("max_steps", 300))),
        "optimizer": str((cfg.get("future_training") or {}).get("optimizer", "AdamW")),
        "learning_rate": float((cfg.get("future_training") or {}).get("learning_rate", train_cfg.get("lr", 1.0e-3))),
        "seed": int((cfg.get("future_training") or {}).get("seed", cfg.get("seed", 1337))),
    }


def assess_success_criterion_v1_feasibility(
    *,
    criterion_v1: dict[str, Any],
    two_state_positive_success50_union_upper_bound: int,
) -> dict[str, Any]:
    positive_success50_min = int(((criterion_v1.get("utility") or {}).get("positive_success50_min", -1)))
    feasible = bool(positive_success50_min <= int(two_state_positive_success50_union_upper_bound))
    return {
        **criterion_v1,
        "status": "feasible" if feasible else "infeasible",
        "detected_before_training": True,
        "reason": None if feasible else "positive_success50_min exceeds the theoretical two-state upper bound",
        "original_positive_success50_min": int(positive_success50_min),
        "theoretical_two_state_upper_bound": int(two_state_positive_success50_union_upper_bound),
    }


def build_predeclared_success_criteria_v2(
    *,
    always_closed: dict[str, Any],
    safe_two_state_oracle: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    closed_success = int(always_closed["positive_success50"])
    oracle_success = int(safe_two_state_oracle["positive_success50"])
    positive_success50_min = int(closed_success + math.ceil(0.5 * float(oracle_success - closed_success)))
    closed_positive_iou = float(always_closed["positive_mean_matched_iou"])
    oracle_positive_iou = float(safe_two_state_oracle["positive_mean_matched_iou"])
    positive_mean_iou_min = float(closed_positive_iou + 0.5 * float(oracle_positive_iou - closed_positive_iou))
    future_training = cfg.get("future_training") or {}
    return {
        "version": SUCCESS_CRITERIA_V2_VERSION,
        "declared_before_training": True,
        "formula": {
            "positive_success50_min": "closed_success + ceil(0.5 * (oracle_success - closed_success))",
            "positive_mean_matched_iou_min": "closed_positive_iou + 0.5 * (oracle_positive_iou - closed_positive_iou)",
        },
        "derived_values": {
            "closed_success": int(closed_success),
            "oracle_success": int(oracle_success),
            "positive_success50_min": int(positive_success50_min),
            "closed_positive_iou": float(closed_positive_iou),
            "oracle_positive_iou": float(oracle_positive_iou),
            "positive_mean_matched_iou_min": float(positive_mean_iou_min),
        },
        "safety": {
            "negative_regressions": 0,
            "negative_topology_changes": 0,
        },
        "utility": {
            "positive_success50_min": int(positive_success50_min),
            "positive_mean_matched_iou_min": float(positive_mean_iou_min),
            "must_exceed_always_closed_positive_success50": True,
            "must_exceed_always_closed_positive_mean_matched_iou": True,
        },
        "model_selection_metric": str(future_training.get("checkpoint_selection_metric", "gate_train_loss")),
        "checkpoint_tie_break_rule": str(future_training.get("checkpoint_tie_break", "earlier_step")),
        "gate_threshold_selection_policy": str(future_training.get("gate_threshold_selection_policy", "fixed_0p50")),
        "maximum_training_steps": int(future_training.get("max_steps", 300)),
        "optimizer": str(future_training.get("optimizer", "AdamW")),
        "learning_rate": float(future_training.get("learning_rate", 1.0e-3)),
        "seed": int(future_training.get("seed", cfg.get("seed", 1337))),
        "validation_used_for_checkpoint_selection": False,
        "validation_used_for_threshold_selection": False,
    }


def prepare_split_preflight(
    *,
    cfg: dict[str, Any],
    sample_ids: list[str],
    device: torch.device,
    frozen_model: bridge.FrozenSemanticBridgeSuppressionModel,
) -> dict[str, Any]:
    split_txt = bridge._resolve_repo_path((cfg.get("dataset") or {}).get("train_txt", DEFAULT_SOURCE_SPLIT), DEFAULT_SOURCE_SPLIT)
    records = bridge.mine_bridge_records_for_split(
        cfg=cfg,
        split_txt=split_txt,
        model=frozen_model,
        device=device,
        cache_features=True,
        selected_sample_ids=sample_ids,
    )
    cached_records = bridge.cache_microset_features(records)
    annotated = []
    for row in cached_records:
        current = dict(row)
        current["gate_target"] = int(current["bridge_positive"])
        annotated.append(current)
    logits, logit_diag = gate_v4.compute_frozen_v2_bridge_logits(frozen_model, annotated, device, return_diagnostics=True)
    feature_rows, features_t, targets_t, pixel_remove_masks = gate_v4.extract_gate_feature_rows(annotated, logits)
    hard_gate_state_cache, cache_timing = gate_v4.build_hard_gate_state_cache(annotated, pixel_remove_masks)
    gate_model = gate_v4.build_gate_model_from_cfg(cfg, input_dim=int(features_t.shape[1]))
    runtime_report = micro_runner._build_runtime_device_report(
        cfg=cfg,
        prepared={
            "device": device,
            "frozen_model": frozen_model,
            "cached_micro": annotated,
            "frozen_logits": logits,
            "frozen_logit_diagnostics": logit_diag,
        },
        gate_model=gate_model.cpu(),
        gate_features_t=features_t.cpu(),
    )
    return {
        "records": records,
        "cached_records": annotated,
        "feature_rows": feature_rows,
        "features_t": features_t,
        "targets_t": targets_t,
        "hard_gate_state_cache": hard_gate_state_cache,
        "cache_timing": cache_timing,
        "state_summary": summarize_hard_gate_states(records, hard_gate_state_cache),
        "runtime_report": runtime_report,
        "simple_scalar_rule": select_train_only_scalar_rule(feature_rows),
    }


def verify_source_contract(source_split: Path, expected_sha256: str) -> dict[str, Any]:
    actual_sha = bridge._canonical_split_sha256(source_split)
    status = "pass" if str(actual_sha) == str(expected_sha256) else "blocked"
    out = {
        "status": status,
        "source_split": bridge._repo_relative_canonical_path(source_split),
        "expected_sha256": str(expected_sha256),
        "actual_sha256": str(actual_sha),
    }
    if status != "pass":
        raise SystemExit(json.dumps(out, ensure_ascii=False, indent=2))
    return out
