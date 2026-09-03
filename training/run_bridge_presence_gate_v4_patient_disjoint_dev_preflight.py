from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bridge_presence_gate_v4 as gate_v4
import bridge_presence_gate_v4_patient_disjoint_dev as dev
import bridge_suppression_head as bridge
import train_bridge_presence_gate_v4 as micro_runner


DEFAULT_CONFIG = bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v4_patient_disjoint_dev_v1.yaml"


def _prepare_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    split_cfg = cfg.get("development_split") or {}
    source_split = bridge._resolve_repo_path(split_cfg.get("source_split"), dev.DEFAULT_SOURCE_SPLIT)
    source_sha = str(split_cfg.get("source_split_canonical_sha256", dev.DEFAULT_SOURCE_SHA256))
    dev.verify_source_contract(source_split, source_sha)
    device = bridge._select_device()
    model, semantic_info = dev.load_frozen_semantic_only_model(cfg, device)
    t_records_start = time.perf_counter()
    records = bridge.mine_bridge_records_for_split(
        cfg=cfg,
        split_txt=source_split,
        model=model,
        device=device,
        cache_features=False,
    )
    metadata_rows = dev.build_split_metadata_rows(records)
    split_payload = dev.select_patient_disjoint_split(
        metadata_rows,
        seed=int(split_cfg.get("seed", cfg.get("seed", 1337))),
        preferred_val_patient_count=int(split_cfg.get("preferred_val_patient_count", 5)),
        fallback_val_patient_counts=list(split_cfg.get("fallback_val_patient_counts", [4, 6])),
    )
    source_entries = dev._read_source_split_entries(source_split)
    split_texts = dev.build_split_texts(
        source_entries=source_entries,
        train_sample_ids=list(split_payload["train_summary"]["sample_ids"]),
        val_sample_ids=list(split_payload["val_summary"]["sample_ids"]),
    )
    manifest_dir = bridge._resolve_repo_path(split_cfg.get("manifest_dir"), dev.DEFAULT_MANIFEST_DIR)
    contract_payload = dev.build_manifest_contract(
        source_split_path=source_split,
        source_sha256=source_sha,
        split_payload=split_payload,
    )
    manifest_info = dev.load_frozen_manifest(
        manifest_dir=manifest_dir,
        contract_payload=contract_payload,
        train_text=split_texts["train_text"],
        val_text=split_texts["val_text"],
    )
    return {
        "device": device,
        "semantic_model": model,
        "semantic_checkpoint": semantic_info,
        "records": records,
        "metadata_rows": metadata_rows,
        "manifest": manifest_info,
        "split_payload": split_payload,
        "source_split": source_split,
        "source_contract": {
            "source_path": bridge._repo_relative_canonical_path(source_split),
            "source_canonical_sha256": source_sha,
            "source_sample_count": int(len(source_entries)),
            "source_patient_count": int(len({str(entry['patient_id']) for entry in source_entries})),
            "record_mining_seconds": float(time.perf_counter() - t_records_start),
        },
    }


def run_pipeline(cfg: dict[str, Any], *, manifest_only: bool = False) -> dict[str, Any]:
    analysis_dir = bridge._resolve_repo_path(
        ((cfg.get("analysis") or {}).get("feature_audit_dir")),
        dev.DEFAULT_ANALYSIS_DIR,
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    bridge._seed_everything(int(cfg.get("seed", 1337)))
    manifest_stage = _prepare_manifest(cfg)
    contract = dict(manifest_stage["manifest"]["contract"])
    predeclared_baselines = dev.build_predeclared_baselines()
    output: dict[str, Any] = {
        "source": manifest_stage["source_contract"],
        "manifest": {
            "contract_path": manifest_stage["manifest"]["contract_path"],
            "train_path": manifest_stage["manifest"]["train_path"],
            "val_path": manifest_stage["manifest"]["val_path"],
            "created": bool(manifest_stage["manifest"]["created"]),
            "contract": contract,
        },
        "split_selection": {
            "algorithm": dict(manifest_stage["split_payload"]["algorithm"]),
            "patient_overlap": int(manifest_stage["split_payload"]["patient_overlap"]),
            "sample_overlap": int(manifest_stage["split_payload"]["sample_overlap"]),
        },
        "feature_contract": {
            "dimensions": int(dev.FEATURE_DIMENSION),
            "trainable_parameters": int(dev.TRAINABLE_GATE_PARAMS),
            "semantic_frozen": True,
            "v2_pixel_head_frozen": True,
        },
        "future_baselines": predeclared_baselines,
        "status": "manifest_ready",
    }
    if manifest_only:
        bridge._write_json(analysis_dir / "manifest_summary.json", output)
        return output
    pixel_cfg = cfg.get("frozen_v2_pixel_head") or {}
    checkpoint_path = bridge._resolve_repo_path(pixel_cfg.get("checkpoint_path"), gate_v4.DEFAULT_V2_PIXEL_HEAD_CHECKPOINT)
    if not checkpoint_path.exists():
        output["status"] = "blocked"
        output["blocked_reason"] = f"Frozen V2 pixel-head checkpoint not found: {checkpoint_path}"
        bridge._write_json(analysis_dir / "preflight_summary.json", output)
        raise SystemExit(json.dumps(output, ensure_ascii=False, indent=2))
    device = manifest_stage["device"]
    frozen_model, frozen_v2_info = gate_v4.load_frozen_v2_pixel_model_from_cfg(cfg, device)
    micro_runner._assert_expected_cuda_runtime(
        micro_runner._build_runtime_device_report(
            cfg=cfg,
            prepared={
                "device": device,
                "frozen_model": frozen_model,
                "cached_micro": [],
                "frozen_logits": None,
                "frozen_logit_diagnostics": {"input_devices": {}},
            },
            gate_model=None,
            gate_features_t=None,
        )
    )
    train_prepared = dev.prepare_split_preflight(
        cfg=cfg,
        sample_ids=list(contract["train_sample_ids"]),
        device=device,
        frozen_model=frozen_model,
    )
    val_prepared = dev.prepare_split_preflight(
        cfg=cfg,
        sample_ids=list(contract["val_sample_ids"]),
        device=device,
        frozen_model=frozen_model,
    )
    runtime_snapshot = micro_runner._runtime_environment_snapshot(device)
    train_oracle = dev.compute_safe_two_state_oracle(train_prepared["hard_gate_state_cache"])
    val_oracle = dev.compute_safe_two_state_oracle(val_prepared["hard_gate_state_cache"])
    success_criteria_v1 = dev.build_predeclared_success_criteria_v1(
        val_summary=val_prepared["state_summary"]["record_stats"],
        always_closed=val_prepared["state_summary"]["always_closed"],
        always_open=val_prepared["state_summary"]["always_open"],
        cfg=cfg,
    )
    success_criteria_v1 = dev.assess_success_criterion_v1_feasibility(
        criterion_v1=success_criteria_v1,
        two_state_positive_success50_union_upper_bound=int(val_prepared["state_summary"]["two_state_positive_success50_union_upper_bound"]),
    )
    success_criteria_v2 = dev.build_predeclared_success_criteria_v2(
        always_closed=val_prepared["state_summary"]["always_closed"],
        safe_two_state_oracle=val_oracle,
        cfg=cfg,
    )
    output.update(
        {
            "status": "pass",
            "semantic_checkpoint": manifest_stage["semantic_checkpoint"],
            "frozen_v2_checkpoint": frozen_v2_info,
            "runtime_device_contract": val_prepared["runtime_report"],
            "runtime_environment_snapshot": runtime_snapshot,
            "gate_train": {
                **train_prepared["state_summary"]["record_stats"],
                "closed_reconstruction": train_prepared["state_summary"]["always_closed"],
                "open_reconstruction": train_prepared["state_summary"]["always_open"],
                "safe_two_state_oracle": train_oracle,
                "two_state_positive_success50_union_upper_bound": int(train_prepared["state_summary"]["two_state_positive_success50_union_upper_bound"]),
                "train_simple_scalar_rule": train_prepared["simple_scalar_rule"],
                "cache_construction_time_seconds": float(train_prepared["cache_timing"]["total_seconds"]),
            },
            "gate_val": {
                **val_prepared["state_summary"]["record_stats"],
                "closed_reconstruction": val_prepared["state_summary"]["always_closed"],
                "open_reconstruction": val_prepared["state_summary"]["always_open"],
                "safe_two_state_oracle": val_oracle,
                "two_state_positive_success50_union_upper_bound": int(val_prepared["state_summary"]["two_state_positive_success50_union_upper_bound"]),
                "cache_construction_time_seconds": float(val_prepared["cache_timing"]["total_seconds"]),
            },
            "success_criteria": {
                "active_version": dev.SUCCESS_CRITERIA_V2_VERSION,
                dev.SUCCESS_CRITERIA_V1_VERSION: success_criteria_v1,
                dev.SUCCESS_CRITERIA_V2_VERSION: success_criteria_v2,
            },
        }
    )
    bridge._write_json(analysis_dir / "preflight_summary.json", output)
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()
    cfg_path = bridge._resolve_repo_path(args.config, DEFAULT_CONFIG)
    cfg = bridge._read_yaml(cfg_path)
    cfg["_config_path"] = str(cfg_path.resolve())
    result = run_pipeline(cfg, manifest_only=bool(args.manifest_only))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
