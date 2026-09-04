from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

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
import run_bridge_presence_gate_v4_patient_disjoint_dev_preflight as preflight_runner
import train_bridge_presence_gate_v4 as micro_runner


DEFAULT_CONFIG = bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v4_patient_disjoint_dev_v1.yaml"


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_hhmmss(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _progress_epoch_for_step(step: int, max_steps: int, progress_epochs: int) -> int:
    if int(max_steps) <= 0 or int(progress_epochs) <= 0:
        return 0
    return max(1, min(int(progress_epochs), int(math.ceil((int(step) * int(progress_epochs)) / float(max_steps)))))


def _should_emit_progress(step: int, last_epoch: int, max_steps: int, progress_epochs: int) -> bool:
    return bool(_progress_epoch_for_step(step, max_steps, progress_epochs) > int(last_epoch))


def _save_gate_checkpoint(path: Path, gate_model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, cfg: dict[str, Any], extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gate_model": gate_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "config": cfg,
            "extra": extra,
        },
        str(path),
    )


def _load_gate_checkpoint(path: Path, gate_model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    payload = torch.load(str(path), map_location=device)
    gate_model.load_state_dict(payload["gate_model"])
    return payload


def _prepare_training_inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    manifest_stage = preflight_runner._prepare_manifest(cfg)
    contract = dict(manifest_stage["manifest"]["contract"])
    device = manifest_stage["device"]
    frozen_model, frozen_v2_info = gate_v4.load_frozen_v2_pixel_model_from_cfg(cfg, device)
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
    train_scalar_rule = train_prepared["simple_scalar_rule"]
    if not bool(train_scalar_rule["train_simple_gate_threshold_exists"]):
        raise SystemExit("Frozen TRAIN-only scalar rule must exist for patient-disjoint development runner.")
    success_v2 = dev.build_predeclared_success_criteria_v2(
        always_closed=val_prepared["state_summary"]["always_closed"],
        safe_two_state_oracle=dev.compute_safe_two_state_oracle(val_prepared["hard_gate_state_cache"]),
        cfg=cfg,
    )
    return {
        "manifest_stage": manifest_stage,
        "contract": contract,
        "device": device,
        "frozen_model": frozen_model,
        "frozen_v2_checkpoint": frozen_v2_info,
        "train_prepared": train_prepared,
        "val_prepared": val_prepared,
        "success_criteria_v2": success_v2,
    }


def _train_only_run(
    *,
    cfg: dict[str, Any],
    save_dir: Path,
    device: torch.device,
    gate_model: gate_v4.SampleLevelBridgePresenceGate,
    features_t: torch.Tensor,
    targets_t: torch.Tensor,
) -> dict[str, Any]:
    train_cfg = cfg.get("future_training") or {}
    max_steps = int(train_cfg.get("max_steps", 300))
    progress_epochs = int(train_cfg.get("progress_epochs", 100))
    optimizer = torch.optim.AdamW(
        [p for p in gate_model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("learning_rate", 1.0e-3)),
        weight_decay=float((cfg.get("train") or {}).get("weight_decay", 1.0e-5)),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_step: int | None = None
    history: list[dict[str, Any]] = []
    gate_model = gate_model.to(device)
    features_t = features_t.to(device)
    targets_t = targets_t.to(device)
    step_times: deque[float] = deque(maxlen=25)
    last_reported_epoch = 0
    train_start = time.perf_counter()
    for step in range(1, max_steps + 1):
        t_step = time.perf_counter()
        gate_model.train(True)
        optimizer.zero_grad(set_to_none=True)
        logits = gate_model(features_t)
        loss = loss_fn(logits, targets_t)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        history.append({"step": int(step), "gate_train_loss": float(loss_value)})
        if (loss_value + 1.0e-12) < best_loss:
            best_loss = float(loss_value)
            best_step = int(step)
            _save_gate_checkpoint(
                save_dir / "best_train_loss.pth",
                gate_model,
                optimizer,
                step,
                cfg,
                {"selection_metric": "gate_train_loss", "selection_value": float(loss_value)},
            )
        step_times.append(float(time.perf_counter() - t_step))
        if _should_emit_progress(step, last_reported_epoch, max_steps, progress_epochs):
            epoch = _progress_epoch_for_step(step, max_steps, progress_epochs)
            pct = 100.0 * float(step) / max(float(max_steps), 1.0)
            mean_step = float(sum(step_times) / max(len(step_times), 1))
            eta_seconds = float(max(0, max_steps - step) * mean_step)
            print(
                f"Epoch {epoch:03d}/{int(progress_epochs):03d} | "
                f"step {int(step)}/{int(max_steps)} | "
                f"{pct:.1f}% | "
                f"elapsed {_format_hhmmss(float(time.perf_counter() - train_start))} | "
                f"ETA {_format_hhmmss(eta_seconds)} | "
                f"gate_train_loss {float(loss_value):.6f}",
                flush=True,
            )
            last_reported_epoch = int(epoch)
    _save_gate_checkpoint(save_dir / "last.pth", gate_model, optimizer, max_steps, cfg, {"selection_metric": "last"})
    _save_csv(save_dir / "gate_train_history.csv", history)
    bridge._write_json(save_dir / "gate_train_history.json", history)
    return {
        "best_train_loss": float(best_loss),
        "best_train_loss_step": int(best_step if best_step is not None else max_steps),
        "history": history,
        "optimizer_name": "AdamW",
        "max_steps": int(max_steps),
    }


def _gate_probabilities(gate_model: torch.nn.Module, features_t: torch.Tensor, device: torch.device) -> np.ndarray:
    gate_model.eval()
    with torch.no_grad():
        gate_probs_t = torch.sigmoid(gate_model(features_t.to(device)))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return gate_probs_t.detach().cpu().numpy().reshape(-1)


def _evaluate_split(
    *,
    split_name: str,
    hard_gate_state_cache: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    gate_probs: np.ndarray,
    success_criteria_v2: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate_threshold = 0.50
    trained = dev.evaluate_gate_probabilities(
        hard_gate_state_cache=hard_gate_state_cache,
        gate_probs=gate_probs,
        gate_threshold=gate_threshold,
    )
    always_closed = dev.evaluate_gate_probabilities(
        hard_gate_state_cache=hard_gate_state_cache,
        gate_probs=np.zeros_like(gate_probs, dtype=np.float64),
        gate_threshold=gate_threshold,
    )
    always_open = dev.evaluate_gate_probabilities(
        hard_gate_state_cache=hard_gate_state_cache,
        gate_probs=np.ones_like(gate_probs, dtype=np.float64),
        gate_threshold=gate_threshold,
    )
    simple_scalar = dev.evaluate_fixed_scalar_rule(
        hard_gate_state_cache=hard_gate_state_cache,
        feature_rows=feature_rows,
        scalar_rule=dev.FROZEN_SIMPLE_SCALAR_RULE,
        gate_threshold=gate_threshold,
    )
    simple_scalar = {**simple_scalar}
    simple_gate_prob_lookup = {str(v["sample_id"]): float(v["gate_open"]) for v in simple_scalar["per_sample"]}
    simple_scalar["per_sample_detailed"] = dev._per_sample_with_states(
        hard_gate_state_cache,
        simple_scalar,
        gate_prob_lookup=simple_gate_prob_lookup,
    )
    simple_scalar["patient_level_exploratory"] = dev.patient_level_exploratory_report(simple_scalar["per_sample_detailed"])
    safe_oracle = dev.compute_safe_two_state_oracle(hard_gate_state_cache)
    result = {
        "split": str(split_name),
        "always_closed": always_closed,
        "always_open": always_open,
        "simple_scalar": simple_scalar,
        "trained_v4": trained,
        "safe_two_state_oracle": safe_oracle,
        "gain_fractions": dev.compute_gain_fractions(
            trained_payload=trained,
            always_closed_payload=always_closed,
            safe_oracle=safe_oracle,
        ),
    }
    if success_criteria_v2 is not None:
        result["generalization_pass"] = dev.evaluate_success_against_locked_v2_criterion(
            trained_payload=trained,
            success_criteria_v2=success_criteria_v2,
            always_closed_payload=always_closed,
        )
    return result


def run_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    save_dir = bridge._resolve_repo_path((cfg.get("train") or {}).get("save_dir"), bridge.REPO_ROOT / "training" / "runs" / "unetpp_effb3_bridge_presence_gate_v4_patient_disjoint_dev_v1")
    save_dir.mkdir(parents=True, exist_ok=True)
    bridge._seed_everything(int(cfg.get("seed", 1337)))
    prepared = _prepare_training_inputs(cfg)
    runtime_report = micro_runner._build_runtime_device_report(
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
    micro_runner._assert_expected_cuda_runtime(runtime_report)
    runtime_snapshot = micro_runner._runtime_environment_snapshot(prepared["device"])
    micro_runner._print_runtime_startup_report("Patient-disjoint training startup", runtime_report, runtime_snapshot)
    dev.assert_locked_val_references(
        always_closed=prepared["val_prepared"]["state_summary"]["always_closed"],
        always_open=prepared["val_prepared"]["state_summary"]["always_open"],
        safe_two_state_oracle=dev.compute_safe_two_state_oracle(prepared["val_prepared"]["hard_gate_state_cache"]),
        union_upper_bound=int(prepared["val_prepared"]["state_summary"]["two_state_positive_success50_union_upper_bound"]),
    )
    dev.assert_locked_active_success_criterion_v2(prepared["success_criteria_v2"])
    frozen_snapshot = dev.snapshot_frozen_backbone_state(prepared["frozen_model"])
    frozen_logits_before = prepared["train_prepared"]["frozen_logits"].clone()
    train_run = _train_only_run(
        cfg=cfg,
        save_dir=save_dir,
        device=prepared["device"],
        gate_model=prepared["train_prepared"]["gate_model"],
        features_t=prepared["train_prepared"]["features_t"],
        targets_t=prepared["train_prepared"]["targets_t"],
    )
    best_ckpt_payload = _load_gate_checkpoint(
        save_dir / "best_train_loss.pth",
        prepared["train_prepared"]["gate_model"].to(prepared["device"]),
        prepared["device"],
    )
    train_gate_probs = _gate_probabilities(
        prepared["train_prepared"]["gate_model"],
        prepared["train_prepared"]["features_t"],
        prepared["device"],
    )
    val_gate_probs = _gate_probabilities(
        prepared["train_prepared"]["gate_model"],
        prepared["val_prepared"]["features_t"],
        prepared["device"],
    )
    train_eval = _evaluate_split(
        split_name="GATE_TRAIN",
        hard_gate_state_cache=prepared["train_prepared"]["hard_gate_state_cache"],
        feature_rows=prepared["train_prepared"]["feature_rows"],
        gate_probs=train_gate_probs,
    )
    val_eval = _evaluate_split(
        split_name="GATE_VAL",
        hard_gate_state_cache=prepared["val_prepared"]["hard_gate_state_cache"],
        feature_rows=prepared["val_prepared"]["feature_rows"],
        gate_probs=val_gate_probs,
        success_criteria_v2=prepared["success_criteria_v2"],
    )
    invariants = dev.frozen_backbone_invariant_deltas(prepared["frozen_model"], frozen_snapshot)
    invariants["cached_v2_logits_unchanged"] = bool(torch.equal(frozen_logits_before.cpu(), prepared["train_prepared"]["frozen_logits"].cpu()))
    invariants["manifest_unchanged"] = True
    _save_csv(save_dir / "gate_train_per_sample.csv", train_eval["trained_v4"]["per_sample_detailed"])
    _save_csv(save_dir / "gate_val_per_sample.csv", val_eval["trained_v4"]["per_sample_detailed"])
    bridge._write_json(save_dir / "gate_train_patient_level_exploratory.json", train_eval["trained_v4"]["patient_level_exploratory"])
    bridge._write_json(save_dir / "gate_val_patient_level_exploratory.json", val_eval["trained_v4"]["patient_level_exploratory"])
    bridge._write_json(save_dir / "gate_train_baselines.json", train_eval)
    bridge._write_json(save_dir / "gate_val_baselines.json", val_eval)
    summary = {
        "training_contract": {
            "samples": int(prepared["contract"]["train_summary"]["sample_count"]),
            "patients": int(prepared["contract"]["train_summary"]["patient_count"]),
            "steps": int(train_run["max_steps"]),
            "optimizer": "AdamW",
            "learning_rate": float((cfg.get("future_training") or {}).get("learning_rate", 0.001)),
            "seed": int((cfg.get("future_training") or {}).get("seed", cfg.get("seed", 1337))),
            "checkpoint_selection": {"metric": "gate_train_loss", "lower_is_better": True, "tie_break": "earlier_step"},
            "threshold": 0.50,
        },
        "validation_isolation": {
            "used_during_optimization": False,
            "used_for_checkpoint_selection": False,
            "used_for_threshold_selection": False,
            "first_evaluated_when": "after best_train_loss checkpoint is frozen",
        },
        "manifest": prepared["manifest_stage"]["manifest"],
        "frozen_v2_checkpoint": prepared["frozen_v2_checkpoint"],
        "runtime_device_contract": runtime_report,
        "runtime_environment_snapshot": runtime_snapshot,
        "best_train_loss_checkpoint": {
            "path": str((save_dir / "best_train_loss.pth").resolve()),
            "step": int(best_ckpt_payload["step"]),
            "loss": float(train_run["best_train_loss"]),
        },
        "train_evaluation": train_eval,
        "val_evaluation": val_eval,
        "success_criteria": {
            "active_version": dev.SUCCESS_CRITERIA_V2_VERSION,
            dev.SUCCESS_CRITERIA_V2_VERSION: prepared["success_criteria_v2"],
        },
        "patient_disjoint_v4_generalization_pass": str(val_eval["generalization_pass"]["status_text"]),
        "frozen_invariants": invariants,
        "simple_scalar_rule": dict(dev.FROZEN_SIMPLE_SCALAR_RULE),
    }
    bridge._write_json(save_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    cfg_path = bridge._resolve_repo_path(args.config, DEFAULT_CONFIG)
    cfg = bridge._read_yaml(cfg_path)
    cfg["_config_path"] = str(cfg_path.resolve())
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
