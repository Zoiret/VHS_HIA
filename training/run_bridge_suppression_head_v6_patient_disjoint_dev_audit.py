from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bridge_presence_gate_v4_patient_disjoint_dev as gate_dev
import bridge_suppression_head as bridge
import bridge_suppression_head_v6_patient_disjoint_dev as v6
import run_bridge_presence_gate_v4_patient_disjoint_dev_preflight as split_runner


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    analysis_dir = bridge._resolve_repo_path(((cfg.get("analysis") or {}).get("feature_audit_dir")), v6.DEFAULT_ANALYSIS_DIR)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    bridge._seed_everything(int(cfg.get("seed", 1337)))
    historical_cfg = bridge._read_yaml(v6.HISTORICAL_FULL_CONFIG)
    manifest_stage = split_runner._prepare_manifest(cfg)
    contract = dict(manifest_stage["manifest"]["contract"])
    micro_manifest = v6.load_historical_micro_manifest()
    overlap = v6.compute_microset_overlap(contract, micro_manifest)
    device = manifest_stage["device"]
    model, semantic_info, model_meta = v6.build_v6_model_with_fresh_bridge_head(cfg, device)
    train_records = bridge.mine_bridge_records_for_split(
        cfg=cfg,
        split_txt=bridge._resolve_repo_path(((cfg.get("dataset") or {}).get("train_txt")), bridge.DEFAULT_TRAIN_SPLIT),
        model=model,
        device=device,
        cache_features=True,
        selected_sample_ids=list(contract["train_sample_ids"]),
    )
    val_records = bridge.mine_bridge_records_for_split(
        cfg=cfg,
        split_txt=bridge._resolve_repo_path(((cfg.get("dataset") or {}).get("train_txt")), bridge.DEFAULT_TRAIN_SPLIT),
        model=model,
        device=device,
        cache_features=True,
        selected_sample_ids=list(contract["val_sample_ids"]),
    )
    train_target_audit = v6.summarize_false_bridge_targets(train_records)
    val_target_audit = v6.summarize_false_bridge_targets(val_records)
    loss_hparam = v6.build_loss_hparam_parity_report(cfg, historical_cfg)
    summary = {
        "experiment": {
            "version": v6.V6_EXPERIMENT_VERSION,
            "question": "Can the same V2 bridge-suppression representation generalize when trained from scratch on full patient-disjoint GATE_TRAIN?",
        },
        "manifest": manifest_stage["manifest"],
        "microset_overlap": overlap,
        "target_audit": {
            "gate_train": train_target_audit,
            "gate_val_diagnostic_only": val_target_audit,
        },
        "contract": {
            "initialization": {
                "semantic_checkpoint": semantic_info,
                "bridge_head_initialization": "fresh_random_deterministic_seed_1337",
                "historical_v2_weights_loaded": False,
            },
            "architecture": {
                "matches_v2_head": True,
                "candidate_mask": "P_leaf >= 0.50",
                "output": "removal_logits",
                "inference": "P50 AND NOT(sigmoid(removal_logits) >= 0.50)",
                "trainable_parameters": int(model_meta["total_trainable_params"]),
            },
            "loss_hyperparameters": loss_hparam,
            "checkpoint_selection": v6.build_v6_checkpoint_selection_policy(),
        },
        "development_references": dict(v6.DEVELOPMENT_REFERENCES),
        "development_criterion": {
            "positive_utility_rule": "positive_success50 > CLOSED positive_success50 AND positive_mean_matched_iou > CLOSED positive_mean_matched_iou",
            "max_negative_regressions": int(v6.V6_SAFETY_MAX_NEGATIVE_REGRESSIONS),
            "max_negative_topology_changes": int(v6.V6_SAFETY_MAX_NEGATIVE_TOPOLOGY_CHANGES),
        },
        "validation_status": {
            "reused_development_validation": True,
            "authoritative_holdout_touched": False,
        },
    }
    bridge._write_json(analysis_dir / "microset_overlap.json", overlap)
    bridge._write_json(analysis_dir / "gate_train_target_audit.json", train_target_audit)
    bridge._write_json(analysis_dir / "gate_val_target_audit.json", val_target_audit)
    bridge._write_json(analysis_dir / "loss_hyperparameter_parity.json", loss_hparam)
    bridge._write_json(analysis_dir / "development_references.json", v6.DEVELOPMENT_REFERENCES)
    _write_csv(analysis_dir / "gate_train_target_per_sample.csv", list(train_target_audit["per_sample_rows"]))
    _write_csv(analysis_dir / "gate_val_target_per_sample.csv", list(val_target_audit["per_sample_rows"]))
    bridge._write_json(analysis_dir / "v6_audit_summary.json", summary)
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
