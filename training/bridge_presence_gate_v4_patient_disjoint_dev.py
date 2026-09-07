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
LOCKED_VAL_REFERENCES = {
    "always_closed": {
        "positive_success50": 1,
        "positive_mean_matched_iou": 0.4234538944043546,
    },
    "always_open": {
        "positive_success50": 2,
        "positive_mean_matched_iou": 0.5061690111512952,
        "negative_regressions": 20,
        "negative_topology_changes": 24,
    },
    "safe_two_state_oracle": {
        "positive_success50": 3,
        "positive_mean_matched_iou": 0.5233930737601301,
        "negative_regressions": 0,
        "negative_topology_changes": 0,
        "positive_open_count": 7,
        "positive_closed_count": 5,
    },
    "two_state_positive_success50_union_upper_bound": 3,
}
LOCKED_ACTIVE_SUCCESS_CRITERION_V2 = {
    "positive_success50_min": 2,
    "positive_mean_matched_iou_min": 0.47342348408224233,
    "negative_regressions": 0,
    "negative_topology_changes": 0,
}
FROZEN_SIMPLE_SCALAR_RULE = {
    "scalar": "candidate_fraction",
    "direction": "ge",
    "threshold": 0.1541646271944046,
}
TRAIN_ONLY_SELECTOR_SCALAR = "candidate_fraction"
TRAIN_ONLY_SELECTOR_DIRECTION = "ge"
SELECTOR_SENTINEL_EPS = 1.0e-9
TRAIN_SELECTOR_CROP_AREA_PIXELS = 768 * 768


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
    audit = run_train_only_balanced_accuracy_selector_v1(train_feature_rows)
    validate_frozen_scalar_rule(audit)
    return {
        "selection_version": TRAIN_SCALAR_SELECTION_VERSION,
        "train_simple_gate_threshold_exists": bool(audit["selected_rule"] is not None),
        "selected_rule": dict(FROZEN_SIMPLE_SCALAR_RULE),
        "selection_uses_validation_labels": False,
        "selector_audit": audit,
    }


def build_train_only_selector_input_rows(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in feature_rows:
        sample_id = str(row["sample_id"])
        rows.append(
            {
                "sample_id": sample_id,
                "patient_id": bridge._make_patient_id(sample_id),
                "bridge_target": int(row["bridge_positive_target"]),
                "candidate_fraction": float(np.float64(float(row[TRAIN_ONLY_SELECTOR_SCALAR]))),
            }
        )
    return sorted(rows, key=lambda row: str(row["sample_id"]))


def _selector_input_sha256(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\t".join(
            [
                str(row["sample_id"]),
                str(row["patient_id"]),
                str(int(row["bridge_target"])),
                format(float(row["candidate_fraction"]), ".17g"),
            ]
        )
        for row in rows
    ]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summarize_selector_input_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([float(row["candidate_fraction"]) for row in rows], dtype=np.float64)
    labels = [int(row["bridge_target"]) for row in rows]
    dup_counts: dict[str, int] = {}
    for value in values.tolist():
        key = format(float(value), ".17g")
        dup_counts[key] = int(dup_counts.get(key, 0) + 1)
    duplicates = [
        {"candidate_fraction": str(key), "count": int(count)}
        for key, count in sorted(dup_counts.items())
        if int(count) > 1
    ]
    return {
        "row_count": int(len(rows)),
        "positive_count": int(sum(labels)),
        "negative_count": int(len(rows) - sum(labels)),
        "selector_input_sha256": _selector_input_sha256(rows),
        "candidate_fraction_min": float(np.min(values)) if values.size else 0.0,
        "candidate_fraction_max": float(np.max(values)) if values.size else 0.0,
        "candidate_fraction_mean": float(np.mean(values)) if values.size else 0.0,
        "duplicate_candidate_fraction_values": duplicates,
        "candidate_fraction_dtype_before_selection": "float64",
    }


def compare_selector_input_rows(reference_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ref_map = {str(row["sample_id"]): row for row in reference_rows}
    cur_map = {str(row["sample_id"]): row for row in current_rows}
    differing: list[dict[str, Any]] = []
    for sample_id in sorted(set(ref_map) | set(cur_map)):
        ref = ref_map.get(sample_id)
        cur = cur_map.get(sample_id)
        if ref is None or cur is None:
            differing.append(
                {
                    "sample_id": str(sample_id),
                    "reason": "missing_row",
                    "reference": ref,
                    "current": cur,
                }
            )
            continue
        if int(ref["bridge_target"]) != int(cur["bridge_target"]) or abs(float(ref["candidate_fraction"]) - float(cur["candidate_fraction"])) > 0.0:
            differing.append(
                {
                    "sample_id": str(sample_id),
                    "reference_target": int(ref["bridge_target"]),
                    "current_target": int(cur["bridge_target"]),
                    "reference_candidate_fraction": float(ref["candidate_fraction"]),
                    "current_candidate_fraction": float(cur["candidate_fraction"]),
                    "candidate_fraction_abs_diff": float(abs(float(ref["candidate_fraction"]) - float(cur["candidate_fraction"]))),
                }
            )
    return {
        "reference_row_count": int(len(reference_rows)),
        "current_row_count": int(len(current_rows)),
        "reference_sha256": _selector_input_sha256(reference_rows),
        "current_sha256": _selector_input_sha256(current_rows),
        "identical": bool(_selector_input_sha256(reference_rows) == _selector_input_sha256(current_rows)),
        "differing_samples": differing,
    }


def evaluate_train_only_selector_candidate(
    selector_rows: list[dict[str, Any]],
    *,
    threshold: float,
    direction: str = TRAIN_ONLY_SELECTOR_DIRECTION,
    scalar: str = TRAIN_ONLY_SELECTOR_SCALAR,
    include_prediction_vector: bool = False,
) -> dict[str, Any]:
    values = np.asarray([float(row[scalar]) for row in selector_rows], dtype=np.float64)
    labels = np.asarray([int(row["bridge_target"]) for row in selector_rows], dtype=np.int64)
    pred = (values >= float(threshold)).astype(np.int64) if str(direction) == "ge" else (values <= float(threshold)).astype(np.int64)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    positives = max(int(np.sum(labels == 1)), 1)
    negatives = max(int(np.sum(labels == 0)), 1)
    sensitivity = float(tp / positives)
    specificity = float(tn / negatives)
    out = {
        "scalar": str(scalar),
        "direction": str(direction),
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
    }
    if include_prediction_vector:
        out["prediction_vector"] = [int(v) for v in pred.tolist()]
    return out


def _observed_selector_tiebreak_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    threshold = float(candidate["threshold"])
    kind = str(candidate.get("threshold_kind", "observed"))
    kind_rank = 0 if kind == "observed" else 1
    if str(candidate["direction"]) == "ge":
        return (kind_rank, -threshold)
    return (kind_rank, threshold)


def _build_midpoint_threshold_candidates(values: np.ndarray) -> list[tuple[str, float]]:
    unique = np.unique(values)
    thresholds: list[tuple[str, float]] = [("sentinel_low", float(unique[0] - SELECTOR_SENTINEL_EPS))]
    thresholds.extend(("midpoint", float((a + b) / 2.0)) for a, b in zip(unique[:-1], unique[1:]))
    thresholds.append(("sentinel_high", float(unique[-1] + SELECTOR_SENTINEL_EPS)))
    return thresholds


def _build_observed_threshold_candidates(values: np.ndarray) -> list[tuple[str, float]]:
    unique = np.unique(values)
    thresholds: list[tuple[str, float]] = [("sentinel_low", float(unique[0] - SELECTOR_SENTINEL_EPS))]
    thresholds.extend(("observed", float(v)) for v in unique.tolist())
    thresholds.append(("sentinel_high", float(unique[-1] + SELECTOR_SENTINEL_EPS)))
    return thresholds


def _infer_candidate_pixels(candidate_fraction: float, crop_area_pixels: int = TRAIN_SELECTOR_CROP_AREA_PIXELS) -> int:
    return int(round(float(candidate_fraction) * int(crop_area_pixels)))


def audit_selector_threshold_neighbors(
    selector_rows: list[dict[str, Any]],
    *,
    threshold: float,
    upper_probe_threshold: float | None = None,
    crop_area_pixels: int = TRAIN_SELECTOR_CROP_AREA_PIXELS,
) -> dict[str, Any]:
    values = np.asarray([float(row[TRAIN_ONLY_SELECTOR_SCALAR]) for row in selector_rows], dtype=np.float64)
    unique = np.unique(values)
    lower_values = unique[unique < float(threshold)]
    upper_values = unique[unique > float(threshold)]
    lower_value = float(lower_values[-1]) if lower_values.size else None
    upper_value = float(upper_values[0]) if upper_values.size else None

    def _matching_rows(value: float | None) -> list[dict[str, Any]]:
        if value is None:
            return []
        out: list[dict[str, Any]] = []
        for row in selector_rows:
            if abs(float(row[TRAIN_ONLY_SELECTOR_SCALAR]) - float(value)) <= 0.0:
                out.append(
                    {
                        "sample_id": str(row["sample_id"]),
                        "patient_id": str(row["patient_id"]),
                        "bridge_target": int(row["bridge_target"]),
                    }
                )
        return out

    exact_midpoint = float((float(lower_value) + float(upper_value)) / 2.0) if lower_value is not None and upper_value is not None else None
    upper_bound = float(upper_probe_threshold) if upper_probe_threshold is not None else (float(upper_value) if upper_value is not None else float(threshold))
    values_between = [float(v) for v in unique.tolist() if float(threshold) < float(v) < float(upper_bound)]
    return {
        "lower_observed_fraction": lower_value,
        "lower_candidate_pixels": _infer_candidate_pixels(lower_value, crop_area_pixels) if lower_value is not None else None,
        "lower_samples": _matching_rows(lower_value),
        "upper_observed_fraction": upper_value,
        "upper_candidate_pixels": _infer_candidate_pixels(upper_value, crop_area_pixels) if upper_value is not None else None,
        "upper_samples": _matching_rows(upper_value),
        "exact_midpoint": exact_midpoint,
        "frozen_threshold": float(threshold),
        "midpoint_difference": None if exact_midpoint is None else float(exact_midpoint - float(threshold)),
        "train_values_strictly_between_frozen_and_upper": values_between,
        "any_train_value_strictly_between_frozen_and_upper": bool(values_between),
    }


def train_only_observed_value_selector_reference(
    feature_rows: list[dict[str, Any]],
    *,
    extra_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    selector_rows = build_train_only_selector_input_rows(feature_rows)
    summary = summarize_selector_input_rows(selector_rows)
    values = np.asarray([float(row[TRAIN_ONLY_SELECTOR_SCALAR]) for row in selector_rows], dtype=np.float64)
    if values.size == 0:
        raise SystemExit("Observed-value selector reference cannot run on an empty selector input table.")
    candidates: list[dict[str, Any]] = []
    for evaluation_order, (kind, threshold) in enumerate(_build_observed_threshold_candidates(values), start=1):
        current = evaluate_train_only_selector_candidate(
            selector_rows,
            threshold=float(threshold),
            direction=TRAIN_ONLY_SELECTOR_DIRECTION,
            scalar=TRAIN_ONLY_SELECTOR_SCALAR,
            include_prediction_vector=True,
        )
        current["threshold_kind"] = str(kind)
        current["evaluation_order"] = int(evaluation_order)
        candidates.append(current)
    max_balanced_accuracy = max(float(candidate["balanced_accuracy"]) for candidate in candidates)
    tied_best = [
        dict(candidate)
        for candidate in candidates
        if abs(float(candidate["balanced_accuracy"]) - float(max_balanced_accuracy)) <= 1.0e-12
    ]
    tied_best = sorted(tied_best, key=_observed_selector_tiebreak_key)
    for rank, candidate in enumerate(tied_best, start=1):
        candidate["deterministic_tie_break_rank"] = int(rank)
    probes: list[dict[str, Any]] = []
    for threshold in (extra_thresholds or []):
        current = evaluate_train_only_selector_candidate(
            selector_rows,
            threshold=float(threshold),
            direction=TRAIN_ONLY_SELECTOR_DIRECTION,
            scalar=TRAIN_ONLY_SELECTOR_SCALAR,
            include_prediction_vector=True,
        )
        current["threshold_kind"] = "explicit_probe"
        probes.append(current)
    return {
        "selection_version": TRAIN_SCALAR_SELECTION_VERSION,
        "selector_algorithm": {
            "scalar": TRAIN_ONLY_SELECTOR_SCALAR,
            "direction": TRAIN_ONLY_SELECTOR_DIRECTION,
            "candidate_thresholds": "observed_values_plus_boundary_sentinels",
            "tie_break_rule": "max_balanced_accuracy_then_observed_threshold_then_highest_threshold_for_ge",
            "row_order": "sorted_by_sample_id",
        },
        "selector_input_rows": selector_rows,
        "selector_input_summary": summary,
        "max_balanced_accuracy": float(max_balanced_accuracy),
        "tied_best_thresholds": tied_best,
        "selected_rule": dict(tied_best[0]) if tied_best else None,
        "specific_threshold_checks": probes,
    }


def compare_train_only_selector_semantics(feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    midpoint = train_only_balanced_accuracy_selector_v1(feature_rows, extra_thresholds=[float(FROZEN_SIMPLE_SCALAR_RULE["threshold"])])
    observed = train_only_observed_value_selector_reference(feature_rows, extra_thresholds=[float(FROZEN_SIMPLE_SCALAR_RULE["threshold"])])
    midpoint_rule = midpoint["selected_rule"]
    observed_rule = observed["selected_rule"]
    differing_samples: list[dict[str, Any]] = []
    midpoint_pred = list(midpoint_rule.get("prediction_vector") or []) if midpoint_rule is not None else []
    observed_pred = list(observed_rule.get("prediction_vector") or []) if observed_rule is not None else []
    selector_rows = list(midpoint["selector_input_rows"])
    for idx, row in enumerate(selector_rows):
        if idx >= len(midpoint_pred) or idx >= len(observed_pred):
            break
        if int(midpoint_pred[idx]) != int(observed_pred[idx]):
            differing_samples.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "patient_id": str(row["patient_id"]),
                    "bridge_target": int(row["bridge_target"]),
                    "candidate_fraction": float(row["candidate_fraction"]),
                    "midpoint_prediction": int(midpoint_pred[idx]),
                    "observed_prediction": int(observed_pred[idx]),
                }
            )
    upper_probe_threshold = None
    if observed_rule is not None:
        upper_probe_threshold = float(observed_rule["threshold"])
    return {
        "midpoint_selector": midpoint,
        "observed_value_selector": observed,
        "train_predictions_identical": bool(midpoint_pred == observed_pred),
        "differing_train_samples": differing_samples,
        "neighbor_audit": audit_selector_threshold_neighbors(
            selector_rows,
            threshold=float(FROZEN_SIMPLE_SCALAR_RULE["threshold"]),
            upper_probe_threshold=upper_probe_threshold,
        ),
    }


def train_only_balanced_accuracy_selector_v1(
    feature_rows: list[dict[str, Any]],
    *,
    extra_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    selector_rows = build_train_only_selector_input_rows(feature_rows)
    summary = summarize_selector_input_rows(selector_rows)
    values = np.asarray([float(row[TRAIN_ONLY_SELECTOR_SCALAR]) for row in selector_rows], dtype=np.float64)
    if values.size == 0:
        raise SystemExit("TRAIN-only selector cannot run on an empty selector input table.")
    candidates: list[dict[str, Any]] = []
    for evaluation_order, (kind, threshold) in enumerate(_build_midpoint_threshold_candidates(values), start=1):
        current = evaluate_train_only_selector_candidate(
            selector_rows,
            threshold=float(threshold),
            direction=TRAIN_ONLY_SELECTOR_DIRECTION,
            scalar=TRAIN_ONLY_SELECTOR_SCALAR,
            include_prediction_vector=True,
        )
        current["threshold_kind"] = str(kind)
        current["evaluation_order"] = int(evaluation_order)
        candidates.append(current)
    max_balanced_accuracy = max(float(candidate["balanced_accuracy"]) for candidate in candidates)
    tied_best = [
        dict(candidate)
        for candidate in candidates
        if abs(float(candidate["balanced_accuracy"]) - float(max_balanced_accuracy)) <= 1.0e-12
    ]
    tied_best = sorted(tied_best, key=lambda candidate: int(candidate["evaluation_order"]))
    for rank, candidate in enumerate(tied_best, start=1):
        candidate["deterministic_tie_break_rank"] = int(rank)
    specific_threshold_checks = []
    for threshold in (extra_thresholds or []):
        current = evaluate_train_only_selector_candidate(
            selector_rows,
            threshold=float(threshold),
            direction=TRAIN_ONLY_SELECTOR_DIRECTION,
            scalar=TRAIN_ONLY_SELECTOR_SCALAR,
            include_prediction_vector=True,
        )
        current["threshold_kind"] = "explicit_probe"
        specific_threshold_checks.append(current)
    return {
        "selection_version": TRAIN_SCALAR_SELECTION_VERSION,
        "selector_algorithm": {
            "scalar": TRAIN_ONLY_SELECTOR_SCALAR,
            "direction": TRAIN_ONLY_SELECTOR_DIRECTION,
            "candidate_thresholds": "midpoints_between_sorted_unique_values_plus_boundary_sentinels",
            "tie_break_rule": "first_best_in_ascending_threshold_order_for_fixed_candidate_fraction_ge",
            "row_order": "sorted_by_sample_id",
        },
        "selector_input_rows": selector_rows,
        "selector_input_summary": summary,
        "max_balanced_accuracy": float(max_balanced_accuracy),
        "tied_best_thresholds": tied_best,
        "selected_rule": dict(tied_best[0]) if tied_best else None,
        "specific_threshold_checks": specific_threshold_checks,
    }


def run_train_only_balanced_accuracy_selector_v1(
    feature_rows: list[dict[str, Any]],
    *,
    extra_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    return train_only_balanced_accuracy_selector_v1(feature_rows, extra_thresholds=extra_thresholds)


def validate_frozen_scalar_rule(selector_audit: dict[str, Any]) -> dict[str, Any]:
    locked_rule = dict(FROZEN_SIMPLE_SCALAR_RULE)
    best_scalar = selector_audit.get("selected_rule")
    if best_scalar is None:
        raise SystemExit("Frozen TRAIN-only scalar rule must exist but simple threshold audit returned no candidate.")
    check_fields = ("scalar", "direction")
    for field in check_fields:
        if str(best_scalar.get(field)) != str(locked_rule.get(field)):
            raise SystemExit(f"Frozen TRAIN-only scalar rule mismatch for {field}: expected {locked_rule.get(field)} actual {best_scalar.get(field)}")
    if abs(float(best_scalar.get("threshold")) - float(locked_rule["threshold"])) > 1.0e-12:
        raise SystemExit(
            f"Frozen TRAIN-only scalar rule mismatch for threshold: expected {locked_rule['threshold']} actual {best_scalar.get('threshold')}"
        )
    return {
        "matches_frozen_rule": True,
        "frozen_rule_reproduced": True,
        "expected_rule": locked_rule,
        "actual_rule": dict(best_scalar),
    }


def _classification_from_gate_targets(gate_targets: list[int], gate_open: list[int]) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    for target, open_flag in zip(gate_targets, gate_open):
        target_i = int(target)
        open_i = int(open_flag)
        if open_i == 1 and target_i == 1:
            tp += 1
        elif open_i == 0 and target_i == 0:
            tn += 1
        elif open_i == 1 and target_i == 0:
            fp += 1
        else:
            fn += 1
    positives = max(int(sum(1 for v in gate_targets if int(v) == 1)), 1)
    negatives = max(int(sum(1 for v in gate_targets if int(v) == 0)), 1)
    sensitivity = float(tp / positives)
    specificity = float(tn / negatives)
    precision = float(tp / max(tp + fp, 1))
    f1 = float((2.0 * precision * sensitivity) / max(precision + sensitivity, 1.0e-12))
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "precision": precision,
        "f1": f1,
        "gate_open_count": int(sum(int(v) for v in gate_open)),
        "gate_closed_count": int(sum(1 - int(v) for v in gate_open)),
    }


def evaluate_fixed_scalar_rule(
    hard_gate_state_cache: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    *,
    scalar_rule: dict[str, Any],
    gate_threshold: float = 0.50,
) -> dict[str, Any]:
    scalar = str(scalar_rule["scalar"])
    direction = str(scalar_rule["direction"])
    threshold = float(scalar_rule["threshold"])
    gate_probs: list[float] = []
    for row in feature_rows:
        value = float(row[scalar])
        open_flag = bool(value >= threshold) if direction == "ge" else bool(value <= threshold)
        gate_probs.append(float(gate_threshold if open_flag else 0.0))
    out = gate_v4.evaluate_gate_threshold_on_cached(
        hard_gate_state_cache,
        np.asarray(gate_probs, dtype=np.float64),
        gate_threshold=float(gate_threshold),
    )
    out["selection_rule"] = {
        **dict(scalar_rule),
        "selection_uses_validation_labels": False,
    }
    return out


def _per_sample_with_states(
    hard_gate_state_cache: list[dict[str, Any]],
    eval_payload: dict[str, Any],
    *,
    gate_prob_lookup: dict[str, float],
) -> list[dict[str, Any]]:
    state_lookup = {str(row["sample_id"]): row for row in hard_gate_state_cache}
    rows: list[dict[str, Any]] = []
    for row in eval_payload["per_sample"]:
        source = state_lookup[str(row["sample_id"])]
        closed = source["closed"]
        open_state = source["open"]
        rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "patient_id": bridge._make_patient_id(str(row["sample_id"])),
                "target": int(row["gate_target"]),
                "gate_probability": float(gate_prob_lookup[str(row["sample_id"])]),
                "gate_state": "OPEN" if int(row["gate_open"]) == 1 else "CLOSED",
                "closed_success50": int(closed["predicted_success50"]),
                "closed_mean_iou": float(closed["predicted_mean_iou"]),
                "open_success50": int(open_state["predicted_success50"]),
                "open_mean_iou": float(open_state["predicted_mean_iou"]),
                "chosen_success50": int(row["predicted_success50"]),
                "chosen_mean_iou": float(row["predicted_mean_iou"]),
                "negative_regression_flag": int(int(row["gate_target"]) == 0 and float(row["predicted_mean_iou"]) + 1.0e-9 < float(row["start_mean_iou"])),
                "topology_change_flag": int(row["component_topology_changed"]),
            }
        )
    return rows


def patient_level_exploratory_report(per_sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_sample_rows:
        grouped.setdefault(str(row["patient_id"]), []).append(row)
    per_patient: list[dict[str, Any]] = []
    for patient_id in sorted(grouped):
        rows = grouped[patient_id]
        targets = [int(row["target"]) for row in rows]
        opens = [1 if str(row["gate_state"]) == "OPEN" else 0 for row in rows]
        cls = _classification_from_gate_targets(targets, opens)
        positives = [row for row in rows if int(row["target"]) == 1]
        per_patient.append(
            {
                "patient_id": str(patient_id),
                "classification_balanced_accuracy": float(cls["balanced_accuracy"]),
                "positive_success50": int(sum(int(row["chosen_success50"]) for row in positives)),
                "mean_matched_iou": float(np.mean([float(row["chosen_mean_iou"]) for row in rows])) if rows else 0.0,
                "gate_open_fraction": float(np.mean([float(v) for v in opens])) if rows else 0.0,
                "negative_regressions": int(sum(int(row["negative_regression_flag"]) for row in rows)),
                "topology_changes": int(sum(int(row["topology_change_flag"]) for row in rows)),
            }
        )
    return {
        "per_patient": per_patient,
        "macro_mean": {
            "classification_balanced_accuracy": float(np.mean([float(row["classification_balanced_accuracy"]) for row in per_patient])) if per_patient else 0.0,
            "positive_success50": float(np.mean([float(row["positive_success50"]) for row in per_patient])) if per_patient else 0.0,
            "mean_matched_iou": float(np.mean([float(row["mean_matched_iou"]) for row in per_patient])) if per_patient else 0.0,
            "gate_open_fraction": float(np.mean([float(row["gate_open_fraction"]) for row in per_patient])) if per_patient else 0.0,
            "negative_regressions": float(np.mean([float(row["negative_regressions"]) for row in per_patient])) if per_patient else 0.0,
            "topology_changes": float(np.mean([float(row["topology_changes"]) for row in per_patient])) if per_patient else 0.0,
        },
    }


def evaluate_gate_probabilities(
    *,
    hard_gate_state_cache: list[dict[str, Any]],
    gate_probs: np.ndarray,
    gate_threshold: float,
) -> dict[str, Any]:
    eval_payload = gate_v4.evaluate_gate_threshold_on_cached(
        hard_gate_state_cache,
        gate_probs,
        gate_threshold=float(gate_threshold),
    )
    gate_prob_lookup = {
        str(row["sample_id"]): float(gate_probs[idx])
        for idx, row in enumerate(hard_gate_state_cache)
    }
    per_sample_rows = _per_sample_with_states(
        hard_gate_state_cache,
        eval_payload,
        gate_prob_lookup=gate_prob_lookup,
    )
    return {
        **eval_payload,
        "per_sample_detailed": per_sample_rows,
        "patient_level_exploratory": patient_level_exploratory_report(per_sample_rows),
    }


def compute_gain_fractions(
    *,
    trained_payload: dict[str, Any],
    always_closed_payload: dict[str, Any],
    safe_oracle: dict[str, Any],
) -> dict[str, Any]:
    trained = trained_payload["gated_reconstruction"]
    closed = always_closed_payload["gated_reconstruction"]
    success_den = float(int(safe_oracle["positive_success50"]) - int(closed["positive_success50"]))
    iou_den = float(float(safe_oracle["positive_mean_matched_iou"]) - float(closed["positive_mean_matched_iou"]))
    return {
        "success_gain_fraction": None if abs(success_den) <= 1.0e-12 else float((int(trained["positive_success50"]) - int(closed["positive_success50"])) / success_den),
        "iou_gain_fraction": None if abs(iou_den) <= 1.0e-12 else float((float(trained["positive_mean_matched_iou"]) - float(closed["positive_mean_matched_iou"])) / iou_den),
    }


def evaluate_success_against_locked_v2_criterion(
    *,
    trained_payload: dict[str, Any],
    success_criteria_v2: dict[str, Any],
    always_closed_payload: dict[str, Any],
) -> dict[str, Any]:
    trained = trained_payload["gated_reconstruction"]
    utility = success_criteria_v2["utility"]
    safety = success_criteria_v2["safety"]
    passed = bool(
        int(trained["negative_regressions"]) <= int(safety["negative_regressions"])
        and int(trained["negative_topology_changes"]) <= int(safety["negative_topology_changes"])
        and int(trained["positive_success50"]) >= int(utility["positive_success50_min"])
        and float(trained["positive_mean_matched_iou"]) >= float(utility["positive_mean_matched_iou_min"])
        and int(trained["positive_success50"]) > int(always_closed_payload["gated_reconstruction"]["positive_success50"])
        and float(trained["positive_mean_matched_iou"]) > float(always_closed_payload["gated_reconstruction"]["positive_mean_matched_iou"])
    )
    return {
        "pass": passed,
        "status_text": "YES" if passed else "NO",
    }


def snapshot_frozen_backbone_state(model: bridge.FrozenSemanticBridgeSuppressionModel) -> dict[str, Any]:
    named = [(name, p) for name, p in model.named_parameters()]
    return {
        "named": named,
        "params": bridge._snapshot_named_parameters(named),
        "bn": bridge._collect_batchnorm_stats(model.base),
    }


def frozen_backbone_invariant_deltas(model: bridge.FrozenSemanticBridgeSuppressionModel, snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "semantic_parameter_max_delta": float(bridge._max_parameter_delta_from_snapshot(snapshot["named"], snapshot["params"])),
        "semantic_bn_state_max_delta": float(bridge._max_bn_delta(model.base, snapshot["bn"])),
        "v2_pixel_head_parameter_max_delta": float(
            max(
                float(torch.max(torch.abs(param.detach().cpu() - snapshot["params"][name])).item())
                for name, param in model.named_parameters()
                if "bridge_head" in str(name)
            ) if any("bridge_head" in str(name) for name, _ in model.named_parameters()) else 0.0
        ),
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


def assert_locked_val_references(
    *,
    always_closed: dict[str, Any],
    always_open: dict[str, Any],
    safe_two_state_oracle: dict[str, Any],
    union_upper_bound: int,
) -> None:
    checks = [
        ("always_closed.positive_success50", int(always_closed["positive_success50"]), int(LOCKED_VAL_REFERENCES["always_closed"]["positive_success50"])),
        ("always_closed.positive_mean_matched_iou", float(always_closed["positive_mean_matched_iou"]), float(LOCKED_VAL_REFERENCES["always_closed"]["positive_mean_matched_iou"])),
        ("always_open.positive_success50", int(always_open["positive_success50"]), int(LOCKED_VAL_REFERENCES["always_open"]["positive_success50"])),
        ("always_open.positive_mean_matched_iou", float(always_open["positive_mean_matched_iou"]), float(LOCKED_VAL_REFERENCES["always_open"]["positive_mean_matched_iou"])),
        ("always_open.negative_regressions", int(always_open["negative_regressions"]), int(LOCKED_VAL_REFERENCES["always_open"]["negative_regressions"])),
        ("always_open.negative_topology_changes", int(always_open["negative_topology_changes"]), int(LOCKED_VAL_REFERENCES["always_open"]["negative_topology_changes"])),
        ("safe_two_state_oracle.positive_success50", int(safe_two_state_oracle["positive_success50"]), int(LOCKED_VAL_REFERENCES["safe_two_state_oracle"]["positive_success50"])),
        ("safe_two_state_oracle.positive_mean_matched_iou", float(safe_two_state_oracle["positive_mean_matched_iou"]), float(LOCKED_VAL_REFERENCES["safe_two_state_oracle"]["positive_mean_matched_iou"])),
        ("safe_two_state_oracle.negative_regressions", int(safe_two_state_oracle["negative_regressions"]), int(LOCKED_VAL_REFERENCES["safe_two_state_oracle"]["negative_regressions"])),
        ("safe_two_state_oracle.negative_topology_changes", int(safe_two_state_oracle["negative_topology_changes"]), int(LOCKED_VAL_REFERENCES["safe_two_state_oracle"]["negative_topology_changes"])),
        ("safe_two_state_oracle.positive_open_count", int(safe_two_state_oracle["positive_open_count"]), int(LOCKED_VAL_REFERENCES["safe_two_state_oracle"]["positive_open_count"])),
        ("safe_two_state_oracle.positive_closed_count", int(safe_two_state_oracle["positive_closed_count"]), int(LOCKED_VAL_REFERENCES["safe_two_state_oracle"]["positive_closed_count"])),
        ("two_state_positive_success50_union_upper_bound", int(union_upper_bound), int(LOCKED_VAL_REFERENCES["two_state_positive_success50_union_upper_bound"])),
    ]
    mismatches: list[dict[str, Any]] = []
    for field, actual, expected in checks:
        if isinstance(expected, float):
            if abs(float(actual) - float(expected)) > 1.0e-12:
                mismatches.append({"field": field, "actual": float(actual), "expected": float(expected)})
        elif int(actual) != int(expected):
            mismatches.append({"field": field, "actual": int(actual), "expected": int(expected)})
    if mismatches:
        raise SystemExit(json.dumps({
            "status": "blocked",
            "reason": "locked_val_reference_mismatch",
            "mismatches": mismatches,
        }, ensure_ascii=False, indent=2))


def assert_locked_active_success_criterion_v2(success_criteria_v2: dict[str, Any]) -> None:
    utility = success_criteria_v2["utility"]
    safety = success_criteria_v2["safety"]
    checks = [
        ("positive_success50_min", int(utility["positive_success50_min"]), int(LOCKED_ACTIVE_SUCCESS_CRITERION_V2["positive_success50_min"])),
        ("positive_mean_matched_iou_min", float(utility["positive_mean_matched_iou_min"]), float(LOCKED_ACTIVE_SUCCESS_CRITERION_V2["positive_mean_matched_iou_min"])),
        ("negative_regressions", int(safety["negative_regressions"]), int(LOCKED_ACTIVE_SUCCESS_CRITERION_V2["negative_regressions"])),
        ("negative_topology_changes", int(safety["negative_topology_changes"]), int(LOCKED_ACTIVE_SUCCESS_CRITERION_V2["negative_topology_changes"])),
    ]
    mismatches: list[dict[str, Any]] = []
    for field, actual, expected in checks:
        if isinstance(expected, float):
            if abs(float(actual) - float(expected)) > 1.0e-12:
                mismatches.append({"field": field, "actual": float(actual), "expected": float(expected)})
        elif int(actual) != int(expected):
            mismatches.append({"field": field, "actual": int(actual), "expected": int(expected)})
    if mismatches:
        raise SystemExit(json.dumps({
            "status": "blocked",
            "reason": "locked_success_criterion_v2_mismatch",
            "mismatches": mismatches,
        }, ensure_ascii=False, indent=2))


def _prepare_split_preflight_core(
    *,
    cfg: dict[str, Any],
    sample_ids: list[str],
    device: torch.device,
    frozen_model: bridge.FrozenSemanticBridgeSuppressionModel,
    enforce_frozen_scalar_rule: bool,
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
    selector_audit = run_train_only_balanced_accuracy_selector_v1(
        feature_rows,
        extra_thresholds=[float(FROZEN_SIMPLE_SCALAR_RULE["threshold"])],
    )
    selector_semantics_comparison = compare_train_only_selector_semantics(feature_rows)
    simple_scalar_rule = {
        "selection_version": TRAIN_SCALAR_SELECTION_VERSION,
        "train_simple_gate_threshold_exists": bool(selector_audit["selected_rule"] is not None),
        "selected_rule": dict(FROZEN_SIMPLE_SCALAR_RULE),
        "selection_uses_validation_labels": False,
        "selector_audit": selector_audit,
    }
    frozen_rule_validation = {
        "matches_frozen_rule": False,
        "frozen_rule_reproduced": False,
        "expected_rule": dict(FROZEN_SIMPLE_SCALAR_RULE),
        "actual_rule": dict(selector_audit["selected_rule"]) if selector_audit["selected_rule"] is not None else None,
    }
    if bool(enforce_frozen_scalar_rule):
        frozen_rule_validation = validate_frozen_scalar_rule(selector_audit)
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
        "frozen_logits": logits,
        "frozen_logit_diagnostics": logit_diag,
        "feature_rows": feature_rows,
        "features_t": features_t,
        "targets_t": targets_t,
        "gate_model": gate_model,
        "hard_gate_state_cache": hard_gate_state_cache,
        "cache_timing": cache_timing,
        "state_summary": summarize_hard_gate_states(records, hard_gate_state_cache),
        "runtime_report": runtime_report,
        "simple_scalar_rule": simple_scalar_rule,
        "frozen_rule_validation": frozen_rule_validation,
        "selector_input_rows": list(selector_audit["selector_input_rows"]),
        "selector_audit": {
            **selector_audit,
            "neighbor_audit": selector_semantics_comparison["neighbor_audit"],
            "observed_value_selector_reference": selector_semantics_comparison["observed_value_selector"],
            "midpoint_vs_observed_comparison": {
                "train_predictions_identical": bool(selector_semantics_comparison["train_predictions_identical"]),
                "differing_train_samples": list(selector_semantics_comparison["differing_train_samples"]),
            },
        },
    }


def prepare_split_preflight(
    *,
    cfg: dict[str, Any],
    sample_ids: list[str],
    device: torch.device,
    frozen_model: bridge.FrozenSemanticBridgeSuppressionModel,
) -> dict[str, Any]:
    return _prepare_split_preflight_core(
        cfg=cfg,
        sample_ids=sample_ids,
        device=device,
        frozen_model=frozen_model,
        enforce_frozen_scalar_rule=True,
    )


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
