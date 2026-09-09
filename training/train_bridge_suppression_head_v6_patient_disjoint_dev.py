from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import torch
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e

import bridge_suppression_head as bridge
import bridge_suppression_head_v6_patient_disjoint_dev as v6
import run_bridge_presence_gate_v4_patient_disjoint_dev_preflight as split_runner


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _prepare_inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    manifest_stage = split_runner._prepare_manifest(cfg)
    contract = dict(manifest_stage["manifest"]["contract"])
    device = manifest_stage["device"]
    model, semantic_info, model_meta = v6.build_v6_model_with_fresh_bridge_head(cfg, device)
    optimizer, optimizer_meta = bridge.build_optimizer(model, cfg)
    loss_fn = bridge.build_bridge_loss_from_cfg(cfg)
    split_txt = bridge._resolve_repo_path(((cfg.get("dataset") or {}).get("train_txt")), bridge.DEFAULT_TRAIN_SPLIT)
    train_records_raw = bridge.mine_bridge_records_for_split(
        cfg=cfg,
        split_txt=split_txt,
        model=model,
        device=device,
        cache_features=True,
        selected_sample_ids=list(contract["train_sample_ids"]),
    )
    val_records_raw = bridge.mine_bridge_records_for_split(
        cfg=cfg,
        split_txt=split_txt,
        model=model,
        device=device,
        cache_features=True,
        selected_sample_ids=list(contract["val_sample_ids"]),
    )
    train_records = v6.build_v6_cached_records(train_records_raw)
    val_records = v6.build_v6_cached_records(val_records_raw)
    train_cache_contract = v6.validate_train_cache_contract(
        train_records,
        expected_sample_ids=list(contract["train_sample_ids"]),
    )
    return {
        "manifest_stage": manifest_stage,
        "contract": contract,
        "device": device,
        "model": model,
        "semantic_info": semantic_info,
        "model_meta": model_meta,
        "optimizer": optimizer,
        "optimizer_meta": optimizer_meta,
        "loss_fn": loss_fn,
        "train_records": train_records,
        "val_records": val_records,
        "train_cache_contract": train_cache_contract,
    }


def _iter_epoch_batches(records: list[dict[str, Any]], *, batch_size: int, seed: int, epoch: int) -> list[list[dict[str, Any]]]:
    order = np.arange(len(records), dtype=np.int64)
    rng = np.random.default_rng(int(seed) + int(epoch))
    rng.shuffle(order)
    out: list[list[dict[str, Any]]] = []
    for start in range(0, len(order), max(1, int(batch_size))):
        idx = order[start : start + max(1, int(batch_size))]
        out.append([records[int(i)] for i in idx.tolist()])
    return out


def _train_v6(
    *,
    cfg: dict[str, Any],
    save_dir: Path,
    model: bridge.FrozenSemanticBridgeSuppressionModel,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_records: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    train_cfg = cfg.get("train") or {}
    epochs = int(train_cfg.get("epochs", 100))
    batch_size = int(train_cfg.get("batch_size", 16))
    seed = int(cfg.get("seed", 1337))
    history: list[dict[str, Any]] = []
    best_key: tuple[int, float] | None = None
    best_payload: dict[str, Any] | None = None
    batch_count = 0
    optimizer_updates = 0
    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        for batch_records in _iter_epoch_batches(train_records, batch_size=batch_size, seed=seed, epoch=epoch):
            batch_count += 1
            batch = bridge.stack_cached_batch(batch_records, device)
            model.train(True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model.bridge_forward_from_cached(
                x_0_4=batch["x_0_4"],
                x_2_2=batch["x_2_2"],
                p_leaf=batch["p_leaf"],
            )
            loss_dict = loss_fn(
                bridge_logits=outputs["bridge_logits"],
                bridge_target=batch["bridge_target"],
                candidate_mask=outputs["candidate_mask"],
                bridge_positive=batch["bridge_positive"],
            )
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer_updates += 1
            epoch_losses.append(float(loss.detach().cpu().item()))
        train_eval = v6.evaluate_open_on_cached_records(model=model, cached_records=train_records, device=device)
        key = (
            int(train_eval["reconstruction"]["positive_success50"]),
            float(train_eval["reconstruction"]["positive_mean_matched_iou"]),
        )
        row = {
            "epoch": int(epoch),
            "train_loss_mean": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "train_positive_success50": int(train_eval["reconstruction"]["positive_success50"]),
            "train_positive_mean_matched_iou": float(train_eval["reconstruction"]["positive_mean_matched_iou"]),
            "train_negative_regressions": int(train_eval["reconstruction"]["negative_regressions"]),
            "train_negative_topology_changes": int(train_eval["reconstruction"]["negative_topology_changes"]),
        }
        history.append(row)
        if best_key is None or tuple(key) > tuple(best_key):
            best_key = key
            best_payload = {
                "epoch": int(epoch),
                "selection_policy": v6.build_v6_checkpoint_selection_policy(),
                "train_evaluation": train_eval,
            }
            bridge.save_checkpoint(save_dir / "best_train_reconstruction.pth", model, optimizer, epoch, cfg, extra=best_payload)
    bridge.save_checkpoint(save_dir / "last.pth", model, optimizer, epochs, cfg, extra={"selection_policy": "last"})
    _save_csv(save_dir / "train_history.csv", history)
    bridge._write_json(save_dir / "train_history.json", history)
    return {
        "epochs": int(epochs),
        "batch_count": int(batch_count),
        "optimizer_updates": int(optimizer_updates),
        "history": history,
        "best_payload": best_payload,
    }


def _load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    payload = torch.load(str(path), map_location=device)
    model.load_state_dict(payload["model"])
    return payload


def run_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    save_dir = bridge._resolve_repo_path((cfg.get("train") or {}).get("save_dir"), v6.DEFAULT_RUN_DIR)
    analysis_dir = bridge._resolve_repo_path(((cfg.get("analysis") or {}).get("feature_audit_dir")), v6.DEFAULT_ANALYSIS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    bridge._seed_everything(int(cfg.get("seed", 1337)))
    prepared = _prepare_inputs(cfg)
    contract_summary = prepared["train_cache_contract"]
    print(
        "V6 TRAIN cache contract | "
        f"samples {int(contract_summary['sample_count'])} | "
        f"tensor_contract PASS | metadata_contract PASS | "
        f"unique_sample_ids {int(contract_summary['unique_sample_ids'])} | "
        f"patients {int(contract_summary['patient_count'])} | "
        f"p_leaf {contract_summary['fields']['p_leaf']['dtype']} {contract_summary['fields']['p_leaf']['shape']} | "
        f"candidate_mask {contract_summary['fields']['candidate_mask']['dtype']} {contract_summary['fields']['candidate_mask']['shape']} | "
        f"bridge_target {contract_summary['fields']['bridge_target']['dtype']} {contract_summary['fields']['bridge_target']['shape']}",
        flush=True,
    )
    semantic_named = [(name, p) for name, p in prepared["model"].named_parameters() if name.startswith("base.")]
    semantic_snap = bridge._snapshot_named_parameters(semantic_named)
    bn_snap = bridge._collect_batchnorm_stats(prepared["model"].base)
    train_run = _train_v6(
        cfg=cfg,
        save_dir=save_dir,
        model=prepared["model"],
        optimizer=prepared["optimizer"],
        loss_fn=prepared["loss_fn"],
        train_records=prepared["train_records"],
        device=prepared["device"],
    )
    best_payload = _load_checkpoint(save_dir / "best_train_reconstruction.pth", prepared["model"], prepared["device"])
    semantic_deltas = {
        "semantic_parameter_max_delta": float(bridge._max_parameter_delta_from_snapshot(semantic_named, semantic_snap)),
        "semantic_bn_state_max_delta": float(bridge._max_bn_delta(prepared["model"].base, bn_snap)),
    }
    train_eval = v6.evaluate_open_on_cached_records(model=prepared["model"], cached_records=prepared["train_records"], device=prepared["device"])
    val_eval = v6.evaluate_open_on_cached_records(model=prepared["model"], cached_records=prepared["val_records"], device=prepared["device"])
    train_patient = v6.patient_level_report(train_eval["per_sample"])
    val_patient = v6.patient_level_report(val_eval["per_sample"])
    decision = v6.evaluate_v6_development_decision(val_eval)
    bridge._write_json(save_dir / "train_eval.json", train_eval)
    bridge._write_json(save_dir / "val_eval_reused_dev.json", val_eval)
    bridge._write_json(save_dir / "train_patient_report.json", train_patient)
    bridge._write_json(save_dir / "val_patient_report.json", val_patient)
    _save_csv(save_dir / "train_per_sample.csv", train_eval["per_sample"])
    _save_csv(save_dir / "val_per_sample.csv", val_eval["per_sample"])
    summary = {
        "experiment": {"version": v6.V6_EXPERIMENT_VERSION},
        "training_contract": {
            "initialization": "fresh_random_bridge_head_seed_1337",
            "architecture": "historical_v2_head_exact",
            "parameters": int(prepared["optimizer_meta"]["total_trainable_params"]),
            "loss": v6.build_loss_hparam_parity_report(cfg, bridge._read_yaml(v6.HISTORICAL_FULL_CONFIG))["loss"],
            "optimizer": "AdamW",
            "learning_rate": float((cfg.get("train") or {}).get("lr", 1.0e-3)),
            "steps": f"{int((cfg.get('train') or {}).get('epochs', 100))} epochs",
            "checkpoint_selection": v6.build_v6_checkpoint_selection_policy(),
        },
        "semantic_checkpoint": prepared["semantic_info"],
        "train_cache_contract": contract_summary,
        "development_references": dict(v6.DEVELOPMENT_REFERENCES),
        "train_evaluation": train_eval,
        "val_reused_dev_evaluation": val_eval,
        "train_patient_report": train_patient,
        "val_patient_report": val_patient,
        "decision": decision,
        "training_progress": {
            "optimizer_updates": int(train_run["optimizer_updates"]),
            "batches_completed": int(train_run["batch_count"]),
            "epochs_completed": int(train_run["epochs"]),
        },
        "validation_status": {
            "reused_development_validation": True,
            "authoritative_holdout_touched": False,
        },
        "best_checkpoint": {
            "path": str((save_dir / "best_train_reconstruction.pth").resolve()),
            "epoch": int(best_payload["step"]),
        },
        "frozen_invariants": semantic_deltas,
    }
    bridge._write_json(save_dir / "summary.json", summary)
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
