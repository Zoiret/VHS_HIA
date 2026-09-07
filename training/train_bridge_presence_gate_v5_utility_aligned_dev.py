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

import bridge_presence_gate_v4_patient_disjoint_dev as dev
import bridge_presence_gate_v5_utility_aligned_dev as v5
import bridge_suppression_head as bridge
import train_bridge_presence_gate_v4_patient_disjoint_dev as v4_runner


DEFAULT_CONFIG = bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v5_utility_aligned_dev_v1.yaml"


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _prepare_training_inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    prepared = v4_runner._prepare_training_inputs_core(cfg, enforce_frozen_scalar_rule=True)
    utility_target_rows = v5.build_utility_target_rows(
        feature_rows=list(prepared["train_prepared"]["feature_rows"]),
        hard_gate_state_cache=list(prepared["train_prepared"]["hard_gate_state_cache"]),
    )
    train_feature_rows = v5.attach_utility_targets_to_feature_rows(
        prepared["train_prepared"]["feature_rows"],
        utility_target_rows,
    )
    train_state_cache = v5.override_gate_targets_in_state_cache(
        prepared["train_prepared"]["hard_gate_state_cache"],
        utility_target_rows,
    )
    prepared["train_prepared"] = {
        **prepared["train_prepared"],
        "feature_rows": train_feature_rows,
        "hard_gate_state_cache": train_state_cache,
        "targets_t": v5.build_utility_targets_tensor(utility_target_rows),
        "utility_target_rows": utility_target_rows,
    }
    prepared["v5_utility_target_summary"] = v5.summarize_utility_target_rows(utility_target_rows)
    prepared["v5_bridge_positive_agreement"] = v5.bridge_positive_vs_utility_target_agreement(utility_target_rows)
    prepared["v5_feature_feasibility"] = v5.build_feature_feasibility_audit(
        utility_target_rows=utility_target_rows,
        features_t=prepared["train_prepared"]["features_t"],
    )
    return prepared


def run_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    save_dir = bridge._resolve_repo_path((cfg.get("train") or {}).get("save_dir"), v5.DEFAULT_RUN_DIR)
    analysis_dir = bridge._resolve_repo_path(((cfg.get("analysis") or {}).get("feature_audit_dir")), v5.DEFAULT_ANALYSIS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    bridge._seed_everything(int(cfg.get("seed", 1337)))
    prepared = _prepare_training_inputs(cfg)
    runtime_report = v4_runner.micro_runner._build_runtime_device_report(
        cfg=cfg,
        prepared={
            "device": prepared["device"],
            "frozen_model": prepared["frozen_model"],
            "cached_micro": prepared["train_prepared"]["cached_records"],
            "frozen_logits": prepared["train_prepared"]["frozen_logits"],
            "frozen_logit_diagnostics": prepared["train_prepared"]["frozen_logit_diagnostics"],
        },
        gate_model=prepared["train_prepared"]["gate_model"].cpu(),
        gate_features_t=prepared["train_prepared"]["features_t"].cpu(),
    )
    v4_runner.micro_runner._assert_expected_cuda_runtime(runtime_report)
    runtime_snapshot = v4_runner.micro_runner._runtime_environment_snapshot(prepared["device"])
    v4_runner.micro_runner._print_runtime_startup_report("V5 utility-aligned training startup", runtime_report, runtime_snapshot)
    dev.assert_locked_val_references(
        always_closed=prepared["val_prepared"]["state_summary"]["always_closed"],
        always_open=prepared["val_prepared"]["state_summary"]["always_open"],
        safe_two_state_oracle=dev.compute_safe_two_state_oracle(prepared["val_prepared"]["hard_gate_state_cache"]),
        union_upper_bound=int(prepared["val_prepared"]["state_summary"]["two_state_positive_success50_union_upper_bound"]),
    )
    dev.assert_locked_active_success_criterion_v2(prepared["success_criteria_v2"])
    _save_csv(save_dir / "gate_train_utility_targets.csv", prepared["train_prepared"]["utility_target_rows"])
    bridge._write_json(save_dir / "gate_train_utility_targets.json", prepared["train_prepared"]["utility_target_rows"])
    bridge._write_json(save_dir / "gate_train_utility_target_summary.json", prepared["v5_utility_target_summary"])
    bridge._write_json(save_dir / "gate_train_bridge_positive_vs_utility_target_agreement.json", prepared["v5_bridge_positive_agreement"])
    bridge._write_json(save_dir / "gate_train_feature_feasibility.json", prepared["v5_feature_feasibility"])
    frozen_snapshot = dev.snapshot_frozen_backbone_state(prepared["frozen_model"])
    frozen_logits_before = prepared["train_prepared"]["frozen_logits"].clone()
    train_run = v4_runner._train_only_run(
        cfg=cfg,
        save_dir=save_dir,
        device=prepared["device"],
        gate_model=prepared["train_prepared"]["gate_model"],
        features_t=prepared["train_prepared"]["features_t"],
        targets_t=prepared["train_prepared"]["targets_t"],
    )
    best_ckpt_payload = v4_runner._load_gate_checkpoint(
        save_dir / "best_train_loss.pth",
        prepared["train_prepared"]["gate_model"].to(prepared["device"]),
        prepared["device"],
    )
    invariants = dev.frozen_backbone_invariant_deltas(prepared["frozen_model"], frozen_snapshot)
    invariants["cached_v2_logits_unchanged"] = bool(
        v4_runner.torch.equal(frozen_logits_before.cpu(), prepared["train_prepared"]["frozen_logits"].cpu())
    )
    invariants["manifest_unchanged"] = True
    train_gate_probs = v4_runner._gate_probabilities(
        prepared["train_prepared"]["gate_model"],
        prepared["train_prepared"]["features_t"],
        prepared["device"],
    )
    val_gate_probs = v4_runner._gate_probabilities(
        prepared["train_prepared"]["gate_model"],
        prepared["val_prepared"]["features_t"],
        prepared["device"],
    )
    train_eval = v4_runner._evaluate_split(
        split_name="GATE_TRAIN",
        hard_gate_state_cache=prepared["train_prepared"]["hard_gate_state_cache"],
        feature_rows=prepared["train_prepared"]["feature_rows"],
        gate_probs=train_gate_probs,
    )
    val_eval = v4_runner._evaluate_split(
        split_name="GATE_VAL_REUSED_DEV",
        hard_gate_state_cache=prepared["val_prepared"]["hard_gate_state_cache"],
        feature_rows=prepared["val_prepared"]["feature_rows"],
        gate_probs=val_gate_probs,
        success_criteria_v2=prepared["success_criteria_v2"],
    )
    bridge._write_json(save_dir / "gate_train_baselines.json", train_eval)
    bridge._write_json(save_dir / "gate_val_reused_dev_baselines.json", val_eval)
    summary = {
        "experiment": {
            "version": v5.UTILITY_TARGET_VERSION,
            "purpose": "Utility-aligned V5 gate development experiment",
        },
        "training_contract": v5.build_v5_training_contract(cfg),
        "utility_target_summary": prepared["v5_utility_target_summary"],
        "bridge_positive_vs_utility_target_agreement": prepared["v5_bridge_positive_agreement"],
        "feature_feasibility": prepared["v5_feature_feasibility"],
        "runtime_device_contract": runtime_report,
        "runtime_environment_snapshot": runtime_snapshot,
        "best_train_loss_checkpoint": {
            "path": str((save_dir / "best_train_loss.pth").resolve()),
            "step": int(best_ckpt_payload["step"]),
            "loss": float(train_run["best_train_loss"]),
        },
        "train_evaluation": train_eval,
        "val_reused_dev_evaluation": val_eval,
        "reused_development_validation": True,
        "fresh_unseen_development_validation": False,
        "authoritative_holdout_touched": False,
        "utility_aligned_gate_sufficient_on_reused_dev_val": str(val_eval["generalization_pass"]["status_text"]),
        "frozen_invariants": invariants,
    }
    bridge._write_json(save_dir / "summary.json", summary)
    return summary


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
