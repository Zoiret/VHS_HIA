from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bridge_suppression_head as bridge
import bridge_suppression_head_v6_failure_audit as audit
import bridge_suppression_head_v6_patient_disjoint_dev as v6
import run_bridge_presence_gate_v4_patient_disjoint_dev_preflight as split_runner


def _attach_eval_rows(records: list[dict[str, Any]], eval_payload: dict[str, Any]) -> list[dict[str, Any]]:
    per_sample = {str(row["sample_id"]): row for row in eval_payload["per_sample"]}
    out: list[dict[str, Any]] = []
    for row in records:
        sample_id = str(row["sample_id"])
        current = dict(row)
        current["_predicted_component_topology_changed"] = int(per_sample[sample_id]["component_topology_changed"])
        current["_predicted_success50"] = int(per_sample[sample_id]["predicted_success50"])
        current["_predicted_mean_iou"] = float(per_sample[sample_id]["predicted_mean_iou"])
        out.append(current)
    return out


def _write_required_outputs(
    *,
    analysis_dir: Path,
    summary: dict[str, Any],
    score_partition: dict[str, Any],
    damage_decomposition: dict[str, Any],
    removal_component_morphology: dict[str, Any],
    topology_failure_cases: list[dict[str, Any]],
    exact_target_oracle: dict[str, Any],
    train_trajectory_pareto: dict[str, Any],
    historical_v2_vs_v6: dict[str, Any],
    patient_level: dict[str, Any],
) -> None:
    bridge._write_json(analysis_dir / "summary.json", summary)
    bridge._write_json(analysis_dir / "score_partition.json", score_partition)
    bridge._write_json(analysis_dir / "damage_decomposition.json", damage_decomposition)
    bridge._write_json(analysis_dir / "removal_component_morphology.json", removal_component_morphology)
    audit._write_csv(analysis_dir / "topology_failure_cases.csv", topology_failure_cases)
    bridge._write_json(analysis_dir / "exact_target_oracle.json", exact_target_oracle)
    bridge._write_json(analysis_dir / "train_trajectory_pareto.json", train_trajectory_pareto)
    bridge._write_json(analysis_dir / "historical_v2_vs_v6.json", historical_v2_vs_v6)
    bridge._write_json(analysis_dir / "patient_level.json", patient_level)


def _read_train_history(run_dir: Path) -> list[dict[str, Any]]:
    history_path = run_dir / "train_history.json"
    if not history_path.exists():
        raise SystemExit(f"Existing V6 train_history.json not found: {history_path}")
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"Unexpected V6 train history format: {history_path}")
    return payload


def run_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    analysis_dir = bridge._resolve_repo_path(
        ((cfg.get("analysis") or {}).get("failure_audit_dir")),
        audit.DEFAULT_ANALYSIS_DIR,
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    bridge._seed_everything(int(cfg.get("seed", 1337)))

    manifest_stage = split_runner._prepare_manifest(cfg)
    contract = dict(manifest_stage["manifest"]["contract"])
    device = manifest_stage["device"]

    semantic_probe_model, _semantic_probe_info, _ = v6.build_v6_model_with_fresh_bridge_head(cfg, device)
    split_txt = bridge._resolve_repo_path(((cfg.get("dataset") or {}).get("train_txt")), bridge.DEFAULT_TRAIN_SPLIT)
    train_raw = bridge.mine_bridge_records_for_split(
        cfg=cfg,
        split_txt=split_txt,
        model=semantic_probe_model,
        device=device,
        cache_features=True,
        selected_sample_ids=list(contract["train_sample_ids"]),
    )
    val_raw = bridge.mine_bridge_records_for_split(
        cfg=cfg,
        split_txt=split_txt,
        model=semantic_probe_model,
        device=device,
        cache_features=True,
        selected_sample_ids=list(contract["val_sample_ids"]),
    )
    train_records = audit.attach_split_name(v6.build_v6_cached_records(train_raw), "train")
    val_records = audit.attach_split_name(v6.build_v6_cached_records(val_raw), "val")
    all_records = list(train_records) + list(val_records)

    checkpoint_info = audit.verify_frozen_v6_checkpoint(audit.AUTHORITATIVE_V6_CHECKPOINT)
    v6_model, frozen_info = audit.load_v6_model_and_verify_invariants(cfg, audit.AUTHORITATIVE_V6_CHECKPOINT, device)
    train_eval = v6.evaluate_open_on_cached_records(model=v6_model, cached_records=train_records, device=device, threshold=audit.FIXED_REMOVE_THRESHOLD)
    val_eval = v6.evaluate_open_on_cached_records(model=v6_model, cached_records=val_records, device=device, threshold=audit.FIXED_REMOVE_THRESHOLD)
    decision = v6.evaluate_v6_development_decision(val_eval)
    if str(decision["full_data_v2_representation_useful_on_reused_dev_val"]) != "NO":
        raise SystemExit(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "v6_result_no_longer_fails",
                    "decision": decision,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    train_records = _attach_eval_rows(train_records, train_eval)
    val_records = _attach_eval_rows(val_records, val_eval)
    all_records = list(train_records) + list(val_records)
    v6_probs_train = audit._bridge_probs_for_cached_records(v6_model, train_records, device)
    v6_probs_val = audit._bridge_probs_for_cached_records(v6_model, val_records, device)
    v6_probs_all = np.concatenate([v6_probs_train, v6_probs_val], axis=0)

    historical_v2_model, historical_v2_info = audit.load_historical_v2_model(cfg, device)
    historical_v2_probs_train = audit._bridge_probs_for_cached_records(historical_v2_model, train_records, device)
    historical_v2_probs_val = audit._bridge_probs_for_cached_records(historical_v2_model, val_records, device)

    score_partition = {
        "threshold": float(audit.FIXED_REMOVE_THRESHOLD),
        "threshold_sweep_performed": False,
        "train": {
            "v6_open": audit.summarize_score_partition(train_records, v6_probs_train),
            "historical_v2_open": audit.summarize_score_partition(train_records, historical_v2_probs_train),
        },
        "reused_val": {
            "v6_open": audit.summarize_score_partition(val_records, v6_probs_val),
            "historical_v2_open": audit.summarize_score_partition(val_records, historical_v2_probs_val),
        },
    }
    damage_decomposition = {
        "threshold": float(audit.FIXED_REMOVE_THRESHOLD),
        "threshold_sweep_performed": False,
        "v6_open": audit.damage_decomposition(all_records, v6_probs_all, threshold=audit.FIXED_REMOVE_THRESHOLD),
    }
    removal_component_morphology = {
        "threshold": float(audit.FIXED_REMOVE_THRESHOLD),
        "threshold_sweep_performed": False,
        "v6_open": audit.removal_component_morphology(all_records, v6_probs_all, threshold=audit.FIXED_REMOVE_THRESHOLD),
    }
    topology_failure_cases = audit.topology_failure_cases(all_records, v6_probs_all, threshold=audit.FIXED_REMOVE_THRESHOLD)
    exact_target_oracle = {
        "gate_train": audit.exact_target_oracle(train_records),
        "reused_gate_val": audit.exact_target_oracle(val_records),
    }
    train_trajectory_pareto = audit.pareto_frontier(_read_train_history(v6.DEFAULT_RUN_DIR))
    historical_v2_vs_v6 = {
        "threshold": float(audit.FIXED_REMOVE_THRESHOLD),
        "threshold_sweep_performed": False,
        "train": audit.compare_historical_v2_vs_v6(train_records, v6_probs_train, historical_v2_probs_train),
        "reused_val": audit.compare_historical_v2_vs_v6(val_records, v6_probs_val, historical_v2_probs_val),
    }
    historical_v2_vs_v6["interpretation"] = audit.interpret_historical_v2_vs_v6(historical_v2_vs_v6["reused_val"])
    patient_level = audit.patient_level_failure_concentration(val_records, v6_probs_val)
    dominant_failure = audit.classify_dominant_failure(
        score_partition_train=score_partition["train"]["v6_open"],
        exact_target_oracle_val=exact_target_oracle["reused_gate_val"],
        trajectory=train_trajectory_pareto,
    )

    summary = {
        "frozen_result": {
            **checkpoint_info,
            "result": "FAIL",
            "decision": decision,
            "semantic_checkpoint": frozen_info["semantic_checkpoint"],
            "semantic_parameter_max_delta": float(frozen_info["semantic_parameter_max_delta"]),
            "semantic_bn_state_max_delta": float(frozen_info["semantic_bn_state_max_delta"]),
            "historical_v2_checkpoint": historical_v2_info,
        },
        "audit_contract": {
            "authoritative_v6_epoch": int(audit.AUTHORITATIVE_V6_EPOCH),
            "checkpoint_reselected": False,
            "threshold": float(audit.FIXED_REMOVE_THRESHOLD),
            "threshold_sweep_performed": False,
            "no_test_access": True,
            "authoritative_holdout_touched": False,
            "read_only_checkpoint_audit": True,
        },
        "score_separability": {
            "train": score_partition["train"]["v6_open"],
            "reused_val": score_partition["reused_val"]["v6_open"],
        },
        "damage": {
            "inside_gt_removal": damage_decomposition["v6_open"]["aggregate"],
            "dominant_negative_failure": {
                "rules": dict(audit.TOPOLOGY_FAILURE_RULES),
                "counts": {
                    label: int(sum(1 for row in topology_failure_cases if str(row["failure_class"]) == label))
                    for label in audit.TOPOLOGY_FAILURE_RULES.keys()
                },
            },
        },
        "exact_target_oracle": exact_target_oracle,
        "trajectory": train_trajectory_pareto,
        "historical_v2_vs_v6": historical_v2_vs_v6,
        "patient_level": patient_level,
        "next_representation": dominant_failure,
        "frozen_v6_val_metrics": val_eval["reconstruction"],
    }
    _write_required_outputs(
        analysis_dir=analysis_dir,
        summary=summary,
        score_partition=score_partition,
        damage_decomposition=damage_decomposition,
        removal_component_morphology=removal_component_morphology,
        topology_failure_cases=topology_failure_cases,
        exact_target_oracle=exact_target_oracle,
        train_trajectory_pareto=train_trajectory_pareto,
        historical_v2_vs_v6=historical_v2_vs_v6,
        patient_level=patient_level,
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(v6.DEFAULT_CONFIG))
    args = ap.parse_args()
    cfg_path = bridge._resolve_repo_path(args.config, v6.DEFAULT_CONFIG)
    cfg = bridge._read_yaml(cfg_path)
    cfg["_config_path"] = str(cfg_path.resolve())
    result = run_pipeline(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
