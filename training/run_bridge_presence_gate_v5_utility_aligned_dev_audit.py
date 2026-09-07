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

import bridge_presence_gate_v4 as gate_v4
import bridge_presence_gate_v4_patient_disjoint_dev as dev
import bridge_presence_gate_v5_utility_aligned_dev as v5
import bridge_suppression_head as bridge
import run_bridge_presence_gate_v4_patient_disjoint_dev_preflight as preflight_runner


DEFAULT_CONFIG = bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v5_utility_aligned_dev_v1.yaml"


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
    analysis_dir = bridge._resolve_repo_path(
        ((cfg.get("analysis") or {}).get("feature_audit_dir")),
        v5.DEFAULT_ANALYSIS_DIR,
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    bridge._seed_everything(int(cfg.get("seed", 1337)))
    manifest_stage = preflight_runner._prepare_manifest(cfg)
    contract = dict(manifest_stage["manifest"]["contract"])
    device = manifest_stage["device"]
    frozen_model, frozen_v2_info = gate_v4.load_frozen_v2_pixel_model_from_cfg(cfg, device)
    train_prepared = dev._prepare_split_preflight_core(
        cfg=cfg,
        sample_ids=list(contract["train_sample_ids"]),
        device=device,
        frozen_model=frozen_model,
        enforce_frozen_scalar_rule=True,
        caller="v5_utility_audit.gate_train",
        source_function="run_bridge_presence_gate_v5_utility_aligned_dev_audit.run_pipeline",
        validation_artifact_path=analysis_dir / "gate_train_selector_validation.json",
    )
    utility_target_rows = v5.build_utility_target_rows(
        feature_rows=list(train_prepared["feature_rows"]),
        hard_gate_state_cache=list(train_prepared["hard_gate_state_cache"]),
    )
    utility_target_summary = v5.summarize_utility_target_rows(utility_target_rows)
    agreement = v5.bridge_positive_vs_utility_target_agreement(utility_target_rows)
    feature_feasibility = v5.build_feature_feasibility_audit(
        utility_target_rows=utility_target_rows,
        features_t=train_prepared["features_t"],
    )
    summary = {
        "experiment": {
            "version": v5.UTILITY_TARGET_VERSION,
            "purpose": "V5 utility-aligned two-state decision gate development audit",
            "decision_name": v5.UTILITY_ALIGNED_DECISION_NAME,
        },
        "training_contract": v5.build_v5_training_contract(cfg),
        "manifest": manifest_stage["manifest"],
        "frozen_v2_checkpoint": frozen_v2_info,
        "gate_train": {
            "sample_count": int(len(utility_target_rows)),
            "utility_target_summary": utility_target_summary,
            "bridge_positive_vs_utility_target_agreement": agreement,
            "feature_feasibility": feature_feasibility,
        },
        "validation_status": {
            "fresh_unseen_development_validation": False,
            "reused_development_validation": True,
            "authoritative_holdout_touched": False,
        },
        "fixed_development_benchmark": {
            "version": dev.SUCCESS_CRITERIA_V2_VERSION,
            "criteria": dev.LOCKED_ACTIVE_SUCCESS_CRITERION_V2,
        },
    }
    _write_csv(analysis_dir / "gate_train_utility_targets.csv", utility_target_rows)
    bridge._write_json(analysis_dir / "gate_train_utility_targets.json", utility_target_rows)
    bridge._write_json(analysis_dir / "gate_train_utility_target_summary.json", utility_target_summary)
    bridge._write_json(analysis_dir / "gate_train_bridge_positive_vs_utility_target_agreement.json", agreement)
    bridge._write_json(analysis_dir / "gate_train_feature_feasibility.json", feature_feasibility)
    bridge._write_json(analysis_dir / "gate_train_selector_audit.json", train_prepared["selector_audit"])
    bridge._write_json(analysis_dir / "v5_utility_target_audit_summary.json", summary)
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
