from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
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
import bridge_suppression_head as bridge


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


def _tensor_device_text(tensor: Any) -> str | None:
    if torch.is_tensor(tensor):
        return str(tensor.device)
    return None


def _module_device_text(module: torch.nn.Module | None) -> str | None:
    if module is None:
        return None
    for param in module.parameters():
        return str(param.device)
    for buf in module.buffers():
        return str(buf.device)
    return None


def _device_index(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(device.index) if device.index is not None else int(torch.cuda.current_device())


def _runtime_environment_snapshot(device: torch.device) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "gpu_utilization_percent": None,
        "gpu_memory_used_mb": None,
        "gpu_memory_free_mb": None,
        "gpu_memory_total_mb": None,
        "active_gpu_processes": None,
        "cpu_load_percent": None,
    }
    try:
        import psutil  # type: ignore

        snapshot["cpu_load_percent"] = float(psutil.cpu_percent(interval=None))
    except Exception:
        try:
            if hasattr(os, "getloadavg"):
                load1, _load5, _load15 = os.getloadavg()
                cpu_count = max(int(os.cpu_count() or 1), 1)
                snapshot["cpu_load_percent"] = float((float(load1) / float(cpu_count)) * 100.0)
        except Exception:
            snapshot["cpu_load_percent"] = None
    if device.type != "cuda" or not torch.cuda.is_available():
        return snapshot
    try:
        free_b, total_b = torch.cuda.mem_get_info(device)
        snapshot["gpu_memory_free_mb"] = float(free_b / (1024.0 * 1024.0))
        snapshot["gpu_memory_total_mb"] = float(total_b / (1024.0 * 1024.0))
        snapshot["gpu_memory_used_mb"] = float(snapshot["gpu_memory_total_mb"] - snapshot["gpu_memory_free_mb"])
    except Exception:
        pass
    try:
        gpu_index = _device_index(device)
        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        gpu_uuid: str | None = None
        if gpu_query.returncode == 0:
            for line in gpu_query.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 5:
                    continue
                if int(parts[0]) == int(gpu_index):
                    gpu_uuid = parts[1]
                    snapshot["gpu_utilization_percent"] = float(parts[2])
                    snapshot["gpu_memory_used_mb"] = float(parts[3])
                    snapshot["gpu_memory_free_mb"] = float(parts[4])
                    break
        proc_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc_query.returncode == 0:
            count = 0
            for line in proc_query.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 2 and gpu_uuid is not None and parts[0] == gpu_uuid:
                    count += 1
            snapshot["active_gpu_processes"] = int(count)
    except Exception:
        pass
    return snapshot


def _runtime_amp_report(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "semantic_inference_amp_requested": bool((cfg.get("semantic_inference") or {}).get("amp", False)),
        "semantic_inference_amp_enabled": bool(bridge._semantic_inference_amp_enabled(cfg, device)),
        "frozen_v2_pixel_head_inference_amp_enabled": False,
        "v4_gate_training_amp_enabled": False,
    }


def _build_runtime_device_report(
    *,
    cfg: dict[str, Any],
    prepared: dict[str, Any],
    gate_model: torch.nn.Module | None,
    gate_features_t: torch.Tensor | None,
) -> dict[str, Any]:
    device = prepared["device"]
    gpu_index = _device_index(device)
    cached_micro = prepared.get("cached_micro") or []
    first_row = cached_micro[0] if cached_micro else {}
    frozen_logit_diag = prepared.get("frozen_logit_diagnostics") or {}
    return {
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
        "selected_torch_device": str(device),
        "cuda_device_index": gpu_index,
        "cuda_device_name": None if gpu_index is None else torch.cuda.get_device_name(gpu_index),
        "semantic_model_parameter_device": _module_device_text(getattr(prepared["frozen_model"], "base", None)),
        "frozen_v2_pixel_head_parameter_device": _module_device_text(getattr(prepared["frozen_model"], "bridge_head", None)),
        "v4_gate_parameter_device": _module_device_text(gate_model),
        "cached_semantic_feature_tensor_device": _tensor_device_text(first_row.get("x_0_4")),
        "cached_context_feature_tensor_device": _tensor_device_text(first_row.get("x_2_2")),
        "cached_semantic_probability_tensor_device": _tensor_device_text(first_row.get("p_leaf")),
        "cached_v2_logit_tensor_device": _tensor_device_text(prepared.get("frozen_logits")),
        "gate_feature_tensor_device": _tensor_device_text(gate_features_t),
        "frozen_v2_inference_input_devices": dict(frozen_logit_diag.get("input_devices") or {}),
        "amp": _runtime_amp_report(cfg, device),
    }


def _assert_expected_cuda_runtime(report: dict[str, Any]) -> None:
    if str(report.get("selected_torch_device")) != "cuda" and not str(report.get("selected_torch_device", "")).startswith("cuda"):
        return
    errors: list[str] = []
    for field in ("semantic_model_parameter_device", "frozen_v2_pixel_head_parameter_device"):
        value = str(report.get(field, "") or "")
        if not value.startswith("cuda"):
            errors.append(f"{field} expected CUDA but found {value or 'unknown'}")
    for key, value in (report.get("frozen_v2_inference_input_devices") or {}).items():
        value_text = str(value or "")
        if not value_text.startswith("cuda"):
            errors.append(f"frozen_v2_inference_input_devices.{key} expected CUDA but found {value_text or 'unknown'}")
    if errors:
        raise SystemExit(json.dumps({
            "status": "blocked",
            "runtime_device_validation": "blocked",
            "errors": errors,
            "runtime_device_report": report,
        }, ensure_ascii=False, indent=2))


def _print_runtime_startup_report(stage: str, report: dict[str, Any], snapshot: dict[str, Any]) -> None:
    print(
        f"{stage} runtime | "
        f"cuda_available {report.get('torch_cuda_is_available')} | "
        f"selected {report.get('selected_torch_device')} | "
        f"cuda_index {report.get('cuda_device_index')} | "
        f"cuda_name {report.get('cuda_device_name')} | "
        f"semantic {report.get('semantic_model_parameter_device')} | "
        f"frozen_v2 {report.get('frozen_v2_pixel_head_parameter_device')} | "
        f"gate {report.get('v4_gate_parameter_device')} | "
        f"cached_feat {report.get('cached_semantic_feature_tensor_device')} | "
        f"cached_logits {report.get('cached_v2_logit_tensor_device')} | "
        f"gate_feat {report.get('gate_feature_tensor_device')} | "
        f"gpu_util {snapshot.get('gpu_utilization_percent')} | "
        f"gpu_mem_used_mb {snapshot.get('gpu_memory_used_mb')} | "
        f"gpu_mem_free_mb {snapshot.get('gpu_memory_free_mb')} | "
        f"gpu_procs {snapshot.get('active_gpu_processes')} | "
        f"cpu_load {snapshot.get('cpu_load_percent')}",
        flush=True,
    )


def _print_eval_timing(step: int, timing: dict[str, Any]) -> None:
    print(
        "Eval timing | "
        f"step {int(step)} | "
        f"gpu_forward {float(timing.get('gpu_forward_seconds', 0.0)):.3f}s | "
        f"transfer {float(timing.get('transfer_seconds', 0.0)):.3f}s | "
        f"cpu_cc {float(timing.get('cpu_connected_components_seconds', 0.0)):.3f}s | "
        f"cpu_recon {float(timing.get('cpu_reconstruction_seconds', 0.0)):.3f}s | "
        f"checkpoint {float(timing.get('checkpoint_io_seconds', 0.0)):.3f}s | "
        f"total {float(timing.get('total_seconds', 0.0)):.3f}s",
        flush=True,
    )


def _progress_epoch_for_step(step: int, max_steps: int, progress_epochs: int) -> int:
    if int(max_steps) <= 0 or int(progress_epochs) <= 0:
        return 0
    return max(1, min(int(progress_epochs), int(math.ceil((int(step) * int(progress_epochs)) / float(max_steps)))))


def _should_emit_progress(step: int, last_epoch: int, max_steps: int, progress_epochs: int) -> bool:
    return bool(_progress_epoch_for_step(step, max_steps, progress_epochs) > int(last_epoch))


def _is_eval_step(step: int, max_steps: int, log_every: int) -> bool:
    return bool(int(step) == 1 or int(step) % max(1, int(log_every)) == 0 or int(step) == int(max_steps))


def _progress_eta_seconds(
    *,
    current_step: int,
    max_steps: int,
    log_every: int,
    mean_step_seconds: float,
    recent_eval_seconds: list[float],
) -> float:
    remaining_steps = max(0, int(max_steps) - int(current_step))
    step_eta = float(remaining_steps) * float(mean_step_seconds)
    future_eval_steps = [
        step for step in range(int(current_step) + 1, int(max_steps) + 1)
        if _is_eval_step(step, max_steps, log_every)
    ]
    eval_eta = float(len(future_eval_steps)) * float(sum(recent_eval_seconds) / max(len(recent_eval_seconds), 1)) if recent_eval_seconds else 0.0
    return max(0.0, step_eta + eval_eta)


def _safe_useful_status(eval_payload: dict[str, Any]) -> bool:
    gated = eval_payload["gated_reconstruction"]
    return bool(
        int(gated["negative_regressions"]) == 0
        and int(gated["negative_topology_changes"]) == 0
        and int(gated["positive_success50"]) >= 3
    )


def _safe_useful_key(eval_payload: dict[str, Any]) -> tuple[Any, ...]:
    gated = eval_payload["gated_reconstruction"]
    cls = eval_payload["classification"]
    return (
        int(gated["positive_success50"]),
        float(gated["positive_mean_matched_iou"]),
        float(cls["balanced_accuracy"]),
        float(gated["overall_mean_matched_iou"]),
    )


def _progress_line(
    *,
    step: int,
    max_steps: int,
    progress_epochs: int,
    elapsed_seconds: float,
    eta_seconds: float,
    gate_loss: float,
    eval_row: dict[str, Any] | None,
) -> str:
    epoch = _progress_epoch_for_step(step, max_steps, progress_epochs)
    pct = 100.0 * float(step) / max(float(max_steps), 1.0)
    line = (
        f"Epoch {epoch:03d}/{int(progress_epochs):03d} | "
        f"step {int(step)}/{int(max_steps)} | "
        f"{pct:.1f}% | "
        f"elapsed {_format_hhmmss(elapsed_seconds)} | "
        f"ETA {_format_hhmmss(eta_seconds)} | "
        f"gate_loss {float(gate_loss):.4f}"
    )
    if eval_row is not None:
        line += (
            f" | gate_acc {float(eval_row['gate_accuracy']):.4f}"
            f" | gate_pos {int(eval_row['gate_tp'])}/6"
            f" | gate_neg {int(eval_row['gate_tn'])}/4"
            f" | gated_pos_s50 {int(eval_row['positive_success50'])}/6"
            f" | gated_neg_reg {int(eval_row['negative_regressions'])}"
            f" | gated_neg_topo {int(eval_row['negative_topology_changes'])}"
        )
    return line


def _final_human_summary(summary: dict[str, Any], *, total_training_time_seconds: float) -> str:
    best_safe = summary.get("best_safe_useful_gate")
    best_gate_loss = summary.get("best_gate_loss")
    best_safe_step = "NONE" if not isinstance(best_safe, dict) else str(int(best_safe["step"]))
    best_gate_loss_step = "NONE" if not isinstance(best_gate_loss, dict) else str(int(best_gate_loss["step"]))
    final = summary["final"]
    return (
        "Final summary | "
        f"total {_format_hhmmss(total_training_time_seconds)} | "
        f"best_safe_useful {best_safe_step} | "
        f"best_gate_loss {best_gate_loss_step} | "
        f"gated_pos_s50 {int(final['positive_success50'])}/6 | "
        f"gated_pos_iou {float(final['positive_mean_matched_iou']):.4f} | "
        f"gated_neg_reg {int(final['negative_regressions'])} | "
        f"gated_neg_topo {int(final['negative_topology_changes'])}"
    )


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


def _prepare_v4_inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    t_prepare_start = time.perf_counter()
    device = bridge._select_device()
    frozen_model, frozen_ckpt = gate_v4.load_frozen_v2_pixel_model_from_cfg(cfg, device)
    dataset_cfg = cfg.get("dataset") or {}
    train_split = bridge._resolve_repo_path(dataset_cfg.get("train_txt", bridge.DEFAULT_TRAIN_SPLIT), bridge.DEFAULT_TRAIN_SPLIT)
    bridge._assert_safe_path(train_split)
    micro_cfg = cfg.get("micro_overfit") or {}
    manifest_path = bridge._resolve_repo_path(micro_cfg.get("manifest_path"), bridge.MICRO_MANIFEST_V2_PATH)
    manifest_payload = bridge.read_locked_micro_manifest(manifest_path)
    manifest_payload["_manifest_path"] = str(manifest_path.resolve())
    split_validation = bridge.validate_locked_manifest_source_split(
        manifest_payload=manifest_payload,
        configured_train_split=train_split,
    )
    if str(split_validation.get("status")) != "pass":
        raise SystemExit(str(split_validation.get("error")))
    t_record_start = time.perf_counter()
    micro_records = bridge.mine_bridge_records_for_split(
        cfg=cfg,
        split_txt=train_split,
        model=frozen_model,
        device=device,
        cache_features=True,
        selected_sample_ids=[str(v) for v in manifest_payload["sample_ids"]],
    )
    t_record_end = time.perf_counter()
    record_validation = bridge.validate_locked_micro_records(
        manifest_payload=manifest_payload,
        records=micro_records,
        split_txt=train_split,
    )
    if str(record_validation.get("status")) != "pass":
        raise SystemExit(json.dumps(record_validation, ensure_ascii=False, indent=2))
    t_cache_start = time.perf_counter()
    cached_micro = bridge.cache_microset_features(micro_records)
    cached_micro = gate_v4.annotate_cached_records_with_gate_targets(cached_micro, manifest_payload)
    t_cache_end = time.perf_counter()
    frozen_logits, frozen_logit_diagnostics = gate_v4.compute_frozen_v2_bridge_logits(
        frozen_model,
        cached_micro,
        device,
        return_diagnostics=True,
    )
    t_feature_start = time.perf_counter()
    feature_rows, features_t, targets_t, pixel_remove_masks = gate_v4.extract_gate_feature_rows(cached_micro, frozen_logits)
    t_feature_end = time.perf_counter()
    t_state_cache_start = time.perf_counter()
    hard_gate_state_cache, hard_gate_cache_timing = gate_v4.build_hard_gate_state_cache(cached_micro, pixel_remove_masks)
    t_state_cache_end = time.perf_counter()
    gate_model = gate_v4.build_gate_model_from_cfg(cfg, input_dim=int(features_t.shape[1]))
    simple_threshold = gate_v4.simple_scalar_threshold_audit(feature_rows)
    return {
        "device": device,
        "frozen_model": frozen_model,
        "frozen_checkpoint": frozen_ckpt,
        "train_split": train_split,
        "manifest_payload": manifest_payload,
        "manifest_resolution": {
            "split_validation": split_validation,
            "record_validation": record_validation,
        },
        "cached_micro": cached_micro,
        "frozen_logits": frozen_logits,
        "frozen_logit_diagnostics": frozen_logit_diagnostics,
        "pixel_remove_masks": pixel_remove_masks,
        "hard_gate_state_cache": hard_gate_state_cache,
        "feature_rows": feature_rows,
        "features_t": features_t,
        "targets_t": targets_t,
        "gate_model": gate_model,
        "simple_threshold": simple_threshold,
        "prepare_timing": {
            "mine_bridge_records_seconds": float(t_record_end - t_record_start),
            "cache_feature_records_seconds": float(t_cache_end - t_cache_start),
            "frozen_v2_logit_inference": dict(frozen_logit_diagnostics.get("timing") or {}),
            "gate_feature_extraction_seconds": float(t_feature_end - t_feature_start),
            "hard_gate_state_cache_seconds": float(t_state_cache_end - t_state_cache_start),
            "hard_gate_state_cache_timing": hard_gate_cache_timing,
            "total_seconds": float(time.perf_counter() - t_prepare_start),
        },
    }


def _run_gate_micro_overfit(
    *,
    gate_model: gate_v4.SampleLevelBridgePresenceGate,
    features_t: torch.Tensor,
    targets_t: torch.Tensor,
    cached_micro: list[dict[str, Any]],
    hard_gate_state_cache: list[dict[str, Any]],
    frozen_model: bridge.FrozenSemanticBridgeSuppressionModel,
    frozen_logits_before: torch.Tensor,
    device: torch.device,
    save_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    gate_model = gate_model.to(device)
    features_t = features_t.to(device)
    targets_t = targets_t.to(device)
    micro_cfg = cfg.get("micro_overfit") or {}
    max_steps = int(micro_cfg.get("max_steps", 300))
    progress_epochs = int(micro_cfg.get("progress_epochs", 100))
    log_every = int(micro_cfg.get("log_every", 10))
    gate_cfg = cfg.get("gate") or {}
    gate_threshold = float(gate_cfg.get("gate_threshold", 0.50))
    optimizer = torch.optim.AdamW(
        [p for p in gate_model.parameters() if p.requires_grad],
        lr=float((cfg.get("train") or {}).get("lr", 1.0e-3)),
        weight_decay=float((cfg.get("train") or {}).get("weight_decay", 1.0e-5)),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    frozen_named = [(name, p) for name, p in frozen_model.named_parameters()]
    frozen_ref = bridge._snapshot_named_parameters(frozen_named)
    frozen_bn_ref = bridge._collect_batchnorm_stats(frozen_model.base)
    metrics_rows: list[dict[str, Any]] = []
    recent_eval_seconds: deque[float] = deque(maxlen=5)
    step_times: deque[float] = deque(maxlen=25)
    best_gate_loss = float("inf")
    best_gate_loss_payload: dict[str, Any] | None = None
    best_safe_key: tuple[Any, ...] | None = None
    best_safe_payload: dict[str, Any] | None = None
    last_eval_row: dict[str, Any] | None = None
    last_reported_epoch = 0
    eval_timing_rows: list[dict[str, Any]] = []
    train_start = time.perf_counter()
    for step in range(1, max_steps + 1):
        t_step = time.perf_counter()
        gate_model.train(True)
        optimizer.zero_grad(set_to_none=True)
        logits = gate_model(features_t)
        loss = loss_fn(logits, targets_t)
        loss.backward()
        optimizer.step()
        step_times.append(float(time.perf_counter() - t_step))
        if _is_eval_step(step, max_steps, log_every):
            t_eval = time.perf_counter()
            gate_model.eval()
            t_gate_forward_start = time.perf_counter()
            with torch.no_grad():
                gate_probs_t = torch.sigmoid(gate_model(features_t))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            t_gate_forward_end = time.perf_counter()
            t_transfer_start = time.perf_counter()
            gate_probs = gate_probs_t.detach().cpu().numpy().reshape(-1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_transfer_end = time.perf_counter()
            eval_payload = gate_v4.evaluate_gate_threshold_on_cached(
                hard_gate_state_cache,
                gate_probs,
                gate_threshold=gate_threshold,
            )
            cls = eval_payload["classification"]
            gated = eval_payload["gated_reconstruction"]
            row = {
                "step": int(step),
                "gate_loss": float(loss.detach().cpu().item()),
                "gate_accuracy": float((int(cls["tp"]) + int(cls["tn"])) / max(len(hard_gate_state_cache), 1)),
                "gate_tp": int(cls["tp"]),
                "gate_tn": int(cls["tn"]),
                "gate_fp": int(cls["fp"]),
                "gate_fn": int(cls["fn"]),
                "gate_sensitivity": float(cls["sensitivity"]),
                "gate_specificity": float(cls["specificity"]),
                "gate_balanced_accuracy": float(cls["balanced_accuracy"]),
                "positive_success50": int(gated["positive_success50"]),
                "positive_mean_matched_iou": float(gated["positive_mean_matched_iou"]),
                "negative_regressions": int(gated["negative_regressions"]),
                "negative_topology_changes": int(gated["negative_topology_changes"]),
                "negative_removed_fraction": float(gated["negative_removed_fraction"]),
                "overall_success50": int(gated["overall_success50"]),
                "overall_mean_matched_iou": float(gated["overall_mean_matched_iou"]),
                "safe_useful": int(bool(eval_payload["safe_useful"])),
            }
            metrics_rows.append(row)
            last_eval_row = row
            checkpoint_io_seconds = 0.0
            if float(loss.detach().cpu().item()) < float(best_gate_loss):
                best_gate_loss = float(loss.detach().cpu().item())
                best_gate_loss_payload = {"step": int(step), "loss": float(best_gate_loss), "eval": eval_payload}
                t_ckpt_start = time.perf_counter()
                _save_gate_checkpoint(save_dir / "best_gate_loss.pth", gate_model, optimizer, step, cfg, {"best_payload": best_gate_loss_payload})
                checkpoint_io_seconds += float(time.perf_counter() - t_ckpt_start)
            if bool(eval_payload["safe_useful"]):
                current_key = _safe_useful_key(eval_payload)
                if best_safe_key is None or current_key > best_safe_key:
                    best_safe_key = current_key
                    best_safe_payload = {"step": int(step), "eval": eval_payload}
                    t_ckpt_start = time.perf_counter()
                    _save_gate_checkpoint(save_dir / "best_safe_useful_gate.pth", gate_model, optimizer, step, cfg, {"best_payload": best_safe_payload})
                    checkpoint_io_seconds += float(time.perf_counter() - t_ckpt_start)
            eval_seconds = float(time.perf_counter() - t_eval)
            recent_eval_seconds.append(eval_seconds)
            eval_timing = {
                "step": int(step),
                "gpu_forward_seconds": float(t_gate_forward_end - t_gate_forward_start),
                "transfer_seconds": float(t_transfer_end - t_transfer_start),
                "cpu_connected_components_seconds": float((eval_payload.get("timing") or {}).get("cpu_connected_components_seconds", 0.0)),
                "cpu_reconstruction_seconds": float((eval_payload.get("timing") or {}).get("cpu_reconstruction_seconds", 0.0)),
                "checkpoint_io_seconds": float(checkpoint_io_seconds),
                "total_seconds": float(eval_seconds),
            }
            eval_timing_rows.append(eval_timing)
            _print_eval_timing(step, eval_timing)
        if _should_emit_progress(step, last_reported_epoch, max_steps, progress_epochs):
            eta_seconds = _progress_eta_seconds(
                current_step=step,
                max_steps=max_steps,
                log_every=log_every,
                mean_step_seconds=float(sum(step_times) / max(len(step_times), 1)),
                recent_eval_seconds=list(recent_eval_seconds),
            )
            print(
                _progress_line(
                    step=step,
                    max_steps=max_steps,
                    progress_epochs=progress_epochs,
                    elapsed_seconds=float(time.perf_counter() - train_start),
                    eta_seconds=eta_seconds,
                    gate_loss=float(loss.detach().cpu().item()),
                    eval_row=last_eval_row if _is_eval_step(step, max_steps, log_every) else None,
                ),
                flush=True,
            )
            last_reported_epoch = _progress_epoch_for_step(step, max_steps, progress_epochs)
    t_last_ckpt_start = time.perf_counter()
    _save_gate_checkpoint(save_dir / "last.pth", gate_model, optimizer, max_steps, cfg, {"selection_policy": "last"})
    last_checkpoint_io_seconds = float(time.perf_counter() - t_last_ckpt_start)
    _save_csv(save_dir / "gate_micro_overfit_metrics.csv", metrics_rows)
    bridge._write_json(save_dir / "gate_micro_overfit_metrics.json", metrics_rows)
    thresholds = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05).tolist()]
    gate_model.eval()
    t_sweep_forward_start = time.perf_counter()
    with torch.no_grad():
        gate_probs_t = torch.sigmoid(gate_model(features_t))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    t_sweep_forward_end = time.perf_counter()
    t_sweep_transfer_start = time.perf_counter()
    gate_probs = gate_probs_t.detach().cpu().numpy().reshape(-1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_sweep_transfer_end = time.perf_counter()
    sweep = gate_v4.gate_threshold_sweep(hard_gate_state_cache, gate_probs, thresholds)
    bridge._write_json(save_dir / "gate_threshold_sweep.json", sweep)
    frozen_logits_after, frozen_logits_after_diag = gate_v4.compute_frozen_v2_bridge_logits(
        frozen_model,
        cached_micro,
        device,
        return_diagnostics=True,
    )
    final_row = metrics_rows[-1]
    summary = {
        "final": final_row,
        "best_gate_loss": best_gate_loss_payload,
        "best_safe_useful_gate": best_safe_payload,
        "no_safe_useful_gate": bool(best_safe_payload is None),
        "frozen_semantic_parameter_max_delta": float(bridge._max_parameter_delta_from_snapshot(frozen_named, frozen_ref)),
        "frozen_semantic_bn_max_delta": float(bridge._max_bn_delta(frozen_model.base, frozen_bn_ref)),
        "frozen_pixel_logits_max_abs_delta": float(torch.max(torch.abs(frozen_logits_after - frozen_logits_before)).item()),
        "gate_trainable_params": int(gate_v4.count_trainable_parameters(gate_model)),
        "runtime_timing": {
            "evaluation_rows": eval_timing_rows,
            "last_checkpoint_io_seconds": float(last_checkpoint_io_seconds),
            "gate_threshold_sweep_gpu_forward_seconds": float(t_sweep_forward_end - t_sweep_forward_start),
            "gate_threshold_sweep_transfer_seconds": float(t_sweep_transfer_end - t_sweep_transfer_start),
            "frozen_v2_recheck_inference": dict(frozen_logits_after_diag.get("timing") or {}),
        },
    }
    print(_final_human_summary(summary, total_training_time_seconds=float(time.perf_counter() - train_start)), flush=True)
    bridge._write_json(save_dir / "summary.json", summary)
    bridge._write_json(save_dir / "runtime_timing_diagnostics.json", summary["runtime_timing"])
    return summary


def run_pipeline(cfg: dict[str, Any], *, preflight_only: bool = False) -> dict[str, Any]:
    save_dir = bridge._resolve_repo_path((cfg.get("train") or {}).get("save_dir"), bridge.REPO_ROOT / "training" / "runs" / "bridge_presence_gate_v4_micro_overfit")
    future_full_dir = bridge._resolve_repo_path((cfg.get("reserved_full_run") or {}).get("save_dir"), bridge.REPO_ROOT / "training" / "runs" / "bridge_presence_gate_v4_full")
    analysis_dir = bridge._resolve_repo_path(
        ((cfg.get("analysis") or {}).get("feature_audit_dir")),
        bridge.REPO_ROOT / "training" / "analysis" / "bridge_presence_gate_v4_preflight",
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if future_full_dir.exists():
        raise SystemExit(f"Reserved future full run directory must remain nonexistent: {future_full_dir}")
    bridge._seed_everything(int(cfg.get("seed", 1337)))
    prepared = _prepare_v4_inputs(cfg)
    preflight_runtime_report = _build_runtime_device_report(
        cfg=cfg,
        prepared=prepared,
        gate_model=prepared.get("gate_model"),
        gate_features_t=prepared.get("features_t"),
    )
    _assert_expected_cuda_runtime(preflight_runtime_report)
    preflight_runtime_snapshot = _runtime_environment_snapshot(prepared["device"])
    _print_runtime_startup_report("Preflight", preflight_runtime_report, preflight_runtime_snapshot)
    _save_csv(analysis_dir / "gate_features.csv", prepared["feature_rows"])
    bridge._write_json(analysis_dir / "gate_features.json", prepared["feature_rows"])
    preflight = {
        "frozen_v2_pixel_head": prepared["frozen_checkpoint"],
        "micro_manifest": prepared["manifest_payload"],
        "micro_manifest_resolution": prepared["manifest_resolution"],
        "simple_threshold_audit": prepared["simple_threshold"],
        "gate_trainable_params": int(gate_v4.count_trainable_parameters(prepared["gate_model"])),
        "runtime_device_report": preflight_runtime_report,
        "runtime_environment_snapshot": preflight_runtime_snapshot,
        "prepare_timing": prepared.get("prepare_timing"),
        "input_feature_sources": {
            "pooled_semantic_features": ["x_0_4_gap", "x_0_4_gmp", "x_2_2_gap", "x_2_2_gmp"],
            "scalar_bridge_statistics": list(gate_v4.SCALAR_FEATURE_NAMES),
        },
    }
    bridge._write_json(save_dir / "preflight_summary.json", preflight)
    if preflight_only:
        return preflight
    training_gate_model = prepared["gate_model"].to(prepared["device"])
    training_features_t = prepared["features_t"].to(prepared["device"])
    training_targets_t = prepared["targets_t"].to(prepared["device"])
    training_runtime_report = _build_runtime_device_report(
        cfg=cfg,
        prepared=prepared,
        gate_model=training_gate_model,
        gate_features_t=training_features_t,
    )
    _assert_expected_cuda_runtime(training_runtime_report)
    training_runtime_snapshot = _runtime_environment_snapshot(prepared["device"])
    _print_runtime_startup_report("Training startup", training_runtime_report, training_runtime_snapshot)
    summary = _run_gate_micro_overfit(
        gate_model=training_gate_model,
        features_t=training_features_t,
        targets_t=training_targets_t,
        cached_micro=prepared["cached_micro"],
        hard_gate_state_cache=prepared["hard_gate_state_cache"],
        frozen_model=prepared["frozen_model"],
        frozen_logits_before=prepared["frozen_logits"],
        device=prepared["device"],
        save_dir=save_dir,
        cfg=cfg,
    )
    summary["training_startup_runtime_device_report"] = training_runtime_report
    summary["training_startup_runtime_environment_snapshot"] = training_runtime_snapshot
    overall = {**preflight, "micro_overfit": summary}
    bridge._write_json(save_dir / "readiness_summary.json", overall)
    return overall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    cfg_path = bridge._resolve_repo_path(args.config, bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v4_micro_overfit.yaml")
    cfg = bridge._read_yaml(cfg_path)
    cfg["_config_path"] = str(cfg_path.resolve())
    result = run_pipeline(cfg, preflight_only=bool(args.preflight_only))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
