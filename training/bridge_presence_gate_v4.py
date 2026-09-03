from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e

import bridge_suppression_head as bridge


DEFAULT_V2_PIXEL_HEAD_CHECKPOINT = (
    bridge.REPO_ROOT
    / "training"
    / "runs"
    / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2"
    / "best_reconstruction.pth"
)

SCALAR_FEATURE_NAMES = [
    "bridge_score_mean",
    "bridge_score_max",
    "bridge_score_top1pct_mean",
    "bridge_score_top5pct_mean",
    "bridge_score_frac_ge_0p50",
    "bridge_score_frac_ge_0p75",
    "bridge_score_frac_ge_0p90",
    "candidate_fraction",
    "candidate_component_count",
]


class SampleLevelBridgePresenceGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(input_dim), int(hidden_dim))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(float(dropout_p)) if float(dropout_p) > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(int(hidden_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


def _device_text(obj: Any) -> str | None:
    if torch.is_tensor(obj):
        return str(obj.device)
    if isinstance(obj, nn.Module):
        for param in obj.parameters():
            return str(param.device)
        for buf in obj.buffers():
            return str(buf.device)
    return None


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _payload_path_join(base: str, child: str) -> str:
    if not base:
        return str(child)
    if child.startswith("["):
        return f"{base}{child}"
    return f"{base}.{child}"


def _payload_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        summary: dict[str, Any] = {
            "type": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        if int(value.size) <= 16:
            summary["value"] = value.tolist()
        return summary
    if isinstance(value, np.generic):
        return {
            "type": "numpy_scalar",
            "dtype": str(value.dtype),
            "value": value.item(),
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": sorted(str(k) for k in value.keys()),
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": int(len(value)),
            "value": list(value) if len(value) <= 8 else None,
        }
    return {
        "type": type(value).__name__,
        "value": value,
    }


def compare_nested_payloads(
    reference: Any,
    optimized: Any,
    *,
    path: str = "payload",
    float_atol: float = 0.0,
) -> dict[str, Any] | None:
    if isinstance(reference, np.ndarray) or isinstance(optimized, np.ndarray):
        if not (isinstance(reference, np.ndarray) and isinstance(optimized, np.ndarray)):
            return {
                "path": path,
                "reason": "type_mismatch",
                "reference": _payload_summary(reference),
                "optimized": _payload_summary(optimized),
            }
        if reference.shape != optimized.shape:
            return {
                "path": path,
                "reason": "shape_mismatch",
                "reference": _payload_summary(reference),
                "optimized": _payload_summary(optimized),
            }
        if reference.dtype != optimized.dtype:
            return {
                "path": path,
                "reason": "dtype_mismatch",
                "reference": _payload_summary(reference),
                "optimized": _payload_summary(optimized),
            }
        if np.issubdtype(reference.dtype, np.floating) and float(float_atol) > 0.0:
            equal = bool(np.allclose(reference, optimized, rtol=0.0, atol=float(float_atol), equal_nan=True))
        else:
            equal = bool(np.array_equal(reference, optimized))
        if not equal:
            return {
                "path": path,
                "reason": "value_mismatch",
                "reference": _payload_summary(reference),
                "optimized": _payload_summary(optimized),
            }
        return None
    if isinstance(reference, np.generic) or isinstance(optimized, np.generic):
        if not (isinstance(reference, np.generic) and isinstance(optimized, np.generic)):
            return {
                "path": path,
                "reason": "type_mismatch",
                "reference": _payload_summary(reference),
                "optimized": _payload_summary(optimized),
            }
        if reference.dtype != optimized.dtype:
            return {
                "path": path,
                "reason": "dtype_mismatch",
                "reference": _payload_summary(reference),
                "optimized": _payload_summary(optimized),
            }
        ref_value = reference.item()
        opt_value = optimized.item()
        if isinstance(ref_value, float) and isinstance(opt_value, float) and float(float_atol) > 0.0:
            equal = bool(abs(ref_value - opt_value) <= float(float_atol))
        else:
            equal = bool(ref_value == opt_value)
        if not equal:
            return {
                "path": path,
                "reason": "value_mismatch",
                "reference": _payload_summary(reference),
                "optimized": _payload_summary(optimized),
            }
        return None
    if isinstance(reference, dict) or isinstance(optimized, dict):
        if not (isinstance(reference, dict) and isinstance(optimized, dict)):
            return {
                "path": path,
                "reason": "type_mismatch",
                "reference": _payload_summary(reference),
                "optimized": _payload_summary(optimized),
            }
        if set(reference.keys()) != set(optimized.keys()):
            return {
                "path": path,
                "reason": "keys_mismatch",
                "reference": _payload_summary(reference),
                "optimized": _payload_summary(optimized),
            }
        for key in sorted(reference.keys(), key=lambda v: str(v)):
            mismatch = compare_nested_payloads(
                reference[key],
                optimized[key],
                path=_payload_path_join(path, str(key)),
                float_atol=float(float_atol),
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(reference, (list, tuple)) or isinstance(optimized, (list, tuple)):
        if type(reference) is not type(optimized):
            return {
                "path": path,
                "reason": "type_mismatch",
                "reference": _payload_summary(reference),
                "optimized": _payload_summary(optimized),
            }
        if len(reference) != len(optimized):
            return {
                "path": path,
                "reason": "length_mismatch",
                "reference": _payload_summary(reference),
                "optimized": _payload_summary(optimized),
            }
        for idx, (ref_item, opt_item) in enumerate(zip(reference, optimized)):
            mismatch = compare_nested_payloads(
                ref_item,
                opt_item,
                path=_payload_path_join(path, f"[{int(idx)}]"),
                float_atol=float(float_atol),
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(reference, float) or isinstance(optimized, float):
        if not (isinstance(reference, float) and isinstance(optimized, float)):
            return {
                "path": path,
                "reason": "type_mismatch",
                "reference": _payload_summary(reference),
                "optimized": _payload_summary(optimized),
            }
        if float(float_atol) > 0.0:
            equal = bool(abs(float(reference) - float(optimized)) <= float(float_atol))
        else:
            equal = bool(reference == optimized)
        if not equal:
            return {
                "path": path,
                "reason": "value_mismatch",
                "reference": _payload_summary(reference),
                "optimized": _payload_summary(optimized),
            }
        return None
    if reference != optimized:
        return {
            "path": path,
            "reason": "value_mismatch",
            "reference": _payload_summary(reference),
            "optimized": _payload_summary(optimized),
        }
    return None


def format_payload_mismatch(
    *,
    sample_id: str,
    state_name: str,
    mismatch: dict[str, Any],
    category: str,
) -> str:
    return json.dumps(
        {
            "status": "blocked",
            "category": str(category),
            "sample_id": str(sample_id),
            "state": str(state_name),
            "mismatch": mismatch,
        },
        ensure_ascii=False,
        indent=2,
    )


def inspect_bridge_checkpoint(path: Path) -> dict[str, Any]:
    path = path.resolve()
    ckpt = torch.load(str(path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise SystemExit(f"Unsupported frozen V2 checkpoint payload: {path}")
    pixel_key = "model"
    state = ckpt.get(pixel_key)
    if not isinstance(state, dict):
        raise SystemExit(f"Frozen V2 checkpoint missing state-dict key '{pixel_key}': {path}")
    extra = ckpt.get("extra", {}) if isinstance(ckpt, dict) else {}
    best_payload = extra.get("best_payload") if isinstance(extra, dict) else None
    if not isinstance(best_payload, dict):
        best_payload = None
    return {
        "checkpoint_path": str(path),
        "checkpoint_file_sha256": bridge._sha256_file(path),
        "checkpoint_model_state_sha256": bridge.canonical_model_state_sha256(state),
        "step": int(ckpt.get("step", -1)) if isinstance(ckpt, dict) else -1,
        "state_dict_key": pixel_key,
        "selection_policy": None if best_payload is None else best_payload.get("selection_policy"),
        "selection_reason": None if best_payload is None else best_payload.get("selection_reason"),
    }


def validate_expected_bridge_checkpoint_provenance(cfg: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    pixel_cfg = cfg.get("frozen_v2_pixel_head") or {}
    expected_file_sha = str(pixel_cfg.get("expected_file_sha256", "")).strip().lower()
    expected_model_sha = str(pixel_cfg.get("expected_model_state_sha256", "")).strip().lower()
    expected_step_raw = pixel_cfg.get("expected_step")
    expected_key = str(pixel_cfg.get("state_dict_key", "model")).strip()
    actual_file_sha = str(info.get("checkpoint_file_sha256", "")).strip().lower()
    actual_model_sha = str(info.get("checkpoint_model_state_sha256", "")).strip().lower()
    actual_step = int(info.get("step", -1))
    actual_key = str(info.get("state_dict_key", "")).strip()
    errors: list[str] = []
    if expected_file_sha and actual_file_sha != expected_file_sha:
        errors.append(f"Frozen V2 checkpoint file SHA256 mismatch: expected {expected_file_sha} actual {actual_file_sha}")
    if expected_model_sha and actual_model_sha != expected_model_sha:
        errors.append(f"Frozen V2 checkpoint model-state SHA256 mismatch: expected {expected_model_sha} actual {actual_model_sha}")
    if expected_step_raw is not None and actual_step != int(expected_step_raw):
        errors.append(f"Frozen V2 checkpoint step mismatch: expected {int(expected_step_raw)} actual {actual_step}")
    if expected_key and actual_key != expected_key:
        errors.append(f"Frozen V2 checkpoint state-dict key mismatch: expected {expected_key} actual {actual_key}")
    return {
        "status": "pass" if not errors else "blocked",
        "errors": errors,
        "resolved_checkpoint_path": str(info.get("checkpoint_path")),
        "checkpoint_file_sha256": actual_file_sha,
        "checkpoint_model_state_sha256": actual_model_sha,
        "step": actual_step,
        "state_dict_key": actual_key,
        "semantic_frozen": True,
        "v2_pixel_head_frozen": True,
    }


def load_frozen_v2_pixel_model_from_cfg(cfg: dict[str, Any], device: torch.device) -> tuple[bridge.FrozenSemanticBridgeSuppressionModel, dict[str, Any]]:
    pixel_cfg = cfg.get("frozen_v2_pixel_head") or {}
    checkpoint_path = bridge._resolve_repo_path(pixel_cfg.get("checkpoint_path"), DEFAULT_V2_PIXEL_HEAD_CHECKPOINT)
    if not checkpoint_path.exists():
        raise SystemExit(f"Frozen V2 pixel-head checkpoint not found: {checkpoint_path}")
    info = inspect_bridge_checkpoint(checkpoint_path)
    provenance = validate_expected_bridge_checkpoint_provenance(cfg, info)
    if str(provenance.get("status")) != "pass":
        raise SystemExit(json.dumps(provenance, ensure_ascii=False, indent=2))
    model = bridge.build_model_from_cfg(cfg).to(device)
    semantic_info = bridge.load_semantic_checkpoint(
        model,
        bridge._resolve_repo_path((cfg.get("train") or {}).get("init_checkpoint"), bridge.DEFAULT_SEMANTIC_CHECKPOINT),
    )
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt.get(info["state_dict_key"]) if isinstance(ckpt, dict) else None
    if not isinstance(state, dict):
        raise SystemExit(f"Unsupported frozen V2 checkpoint format: {checkpoint_path}")
    incompat = model.load_state_dict(state, strict=True)
    missing = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
    unexpected = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
    if missing or unexpected:
        raise RuntimeError(f"Unexpected frozen V2 checkpoint incompatibility: missing={missing[:5]} unexpected={unexpected[:5]}")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    info["semantic_checkpoint_path"] = str(semantic_info["checkpoint_path"])
    info["semantic_checkpoint_sha256"] = str(semantic_info["checkpoint_sha256"])
    info["provenance_validation"] = provenance
    info["semantic_frozen"] = True
    info["v2_pixel_head_frozen"] = True
    return model, info


def gate_target_map_from_manifest(manifest_payload: dict[str, Any]) -> dict[str, int]:
    rows = manifest_payload.get("rows") or []
    out: dict[str, int] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        out[sample_id] = int(int(row.get("bridge_pixels", 0)) > 0)
    return out


def annotate_cached_records_with_gate_targets(cached_records: list[dict[str, Any]], manifest_payload: dict[str, Any]) -> list[dict[str, Any]]:
    target_map = gate_target_map_from_manifest(manifest_payload)
    out: list[dict[str, Any]] = []
    for row in cached_records:
        sample_id = str(row["sample_id"])
        if sample_id not in target_map:
            raise SystemExit(f"Locked manifest missing gate target for sample: {sample_id}")
        current = dict(row)
        current["gate_target"] = int(target_map[sample_id])
        out.append(current)
    return out


def compute_frozen_v2_bridge_logits(
    model: bridge.FrozenSemanticBridgeSuppressionModel,
    cached_records: list[dict[str, Any]],
    device: torch.device,
    *,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    t_transfer_start = time.perf_counter()
    batch = bridge.stack_cached_batch(cached_records, device)
    _cuda_sync(device)
    t_transfer_end = time.perf_counter()
    model.eval()
    t_forward_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.bridge_forward_from_cached(
            x_0_4=batch["x_0_4"],
            x_2_2=batch["x_2_2"],
            p_leaf=batch["p_leaf"],
        )
        _cuda_sync(device)
    t_forward_end = time.perf_counter()
    t_transfer_back_start = time.perf_counter()
    logits = outputs["bridge_logits"].detach().cpu().float()
    _cuda_sync(device)
    t_transfer_back_end = time.perf_counter()
    diagnostics = {
        "selected_device": str(device),
        "model_parameter_device": _device_text(model),
        "semantic_model_parameter_device": _device_text(model.base),
        "frozen_v2_pixel_head_parameter_device": _device_text(model.bridge_head),
        "input_devices": {
            "x_0_4": _device_text(batch["x_0_4"]),
            "x_2_2": _device_text(batch["x_2_2"]),
            "p_leaf": _device_text(batch["p_leaf"]),
        },
        "output_device": _device_text(logits),
        "amp_enabled": False,
        "timing": {
            "transfer_to_device_seconds": float(t_transfer_end - t_transfer_start),
            "gpu_forward_seconds": float(t_forward_end - t_forward_start),
            "transfer_to_cpu_seconds": float(t_transfer_back_end - t_transfer_back_start),
            "total_seconds": float(t_transfer_back_end - t_transfer_start),
        },
    }
    if return_diagnostics:
        return logits, diagnostics
    return logits


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def _pooled_feature_vector(record: dict[str, Any]) -> np.ndarray:
    x04 = record["x_0_4"].detach().cpu().numpy().astype(np.float32)
    x22 = record["x_2_2"].detach().cpu().numpy().astype(np.float32)
    pooled = [
        x04.mean(axis=(1, 2)),
        x04.max(axis=(1, 2)),
        x22.mean(axis=(1, 2)),
        x22.max(axis=(1, 2)),
    ]
    return np.concatenate(pooled, axis=0).astype(np.float32)


def _masked_topk_mean(values: np.ndarray, frac: float) -> float:
    if values.size == 0:
        return 0.0
    k = max(1, int(math.ceil(float(values.size) * float(frac))))
    top = np.partition(values, -k)[-k:]
    return float(np.mean(top))


def extract_gate_feature_rows(cached_records: list[dict[str, Any]], bridge_logits: torch.Tensor) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, list[np.ndarray]]:
    rows: list[dict[str, Any]] = []
    feature_vectors: list[np.ndarray] = []
    targets: list[float] = []
    pixel_remove_masks: list[np.ndarray] = []
    logits_np = bridge_logits.detach().cpu().numpy().astype(np.float32)
    for idx, record in enumerate(cached_records):
        candidate_mask = record["candidate_mask_np"].astype(np.uint8)
        candidate_bool = candidate_mask > 0
        logits_2d = logits_np[idx, 0]
        scores = _sigmoid_np(logits_2d)
        candidate_scores = scores[candidate_bool]
        pooled = _pooled_feature_vector(record)
        scalar = {
            "bridge_score_mean": float(np.mean(candidate_scores)) if candidate_scores.size else 0.0,
            "bridge_score_max": float(np.max(candidate_scores)) if candidate_scores.size else 0.0,
            "bridge_score_top1pct_mean": _masked_topk_mean(candidate_scores, 0.01),
            "bridge_score_top5pct_mean": _masked_topk_mean(candidate_scores, 0.05),
            "bridge_score_frac_ge_0p50": float(np.mean(candidate_scores >= 0.50)) if candidate_scores.size else 0.0,
            "bridge_score_frac_ge_0p75": float(np.mean(candidate_scores >= 0.75)) if candidate_scores.size else 0.0,
            "bridge_score_frac_ge_0p90": float(np.mean(candidate_scores >= 0.90)) if candidate_scores.size else 0.0,
            "candidate_fraction": float(np.mean(candidate_bool.astype(np.float32))),
            "candidate_component_count": float(record.get("component_count_start", bridge._connected_components(candidate_mask)[1])),
        }
        scalar_vec = np.asarray([scalar[name] for name in SCALAR_FEATURE_NAMES], dtype=np.float32)
        feature_vec = np.concatenate([pooled, scalar_vec], axis=0).astype(np.float32)
        feature_vectors.append(feature_vec)
        targets.append(float(record["gate_target"]))
        pixel_remove_masks.append((scores >= 0.50).astype(np.uint8))
        rows.append(
            {
                "sample_id": str(record["sample_id"]),
                "bridge_positive_target": int(record["gate_target"]),
                **{key: float(value) for key, value in scalar.items()},
                "pooled_feature_dimensionality": int(pooled.size),
                "feature_dimensionality": int(feature_vec.size),
            }
        )
    features_t = torch.from_numpy(np.stack(feature_vectors, axis=0)).float()
    targets_t = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
    return rows, features_t, targets_t, pixel_remove_masks


def simple_scalar_threshold_audit(feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([int(row["bridge_positive_target"]) for row in feature_rows], dtype=np.int64)
    positives = max(int(np.sum(labels == 1)), 1)
    negatives = max(int(np.sum(labels == 0)), 1)
    best: dict[str, Any] | None = None
    perfect: dict[str, Any] | None = None
    for name in SCALAR_FEATURE_NAMES:
        values = np.asarray([float(row[name]) for row in feature_rows], dtype=np.float64)
        unique = np.unique(values)
        thresholds = [float(unique[0] - 1.0e-9)] + [float((a + b) / 2.0) for a, b in zip(unique[:-1], unique[1:])] + [float(unique[-1] + 1.0e-9)]
        for direction in ("ge", "le"):
            for threshold in thresholds:
                pred = (values >= threshold).astype(np.int64) if direction == "ge" else (values <= threshold).astype(np.int64)
                tp = int(np.sum((pred == 1) & (labels == 1)))
                tn = int(np.sum((pred == 0) & (labels == 0)))
                fp = int(np.sum((pred == 1) & (labels == 0)))
                fn = int(np.sum((pred == 0) & (labels == 1)))
                sensitivity = float(tp / positives)
                specificity = float(tn / negatives)
                balanced_accuracy = 0.5 * (sensitivity + specificity)
                current = {
                    "scalar": name,
                    "direction": direction,
                    "threshold": float(threshold),
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "balanced_accuracy": balanced_accuracy,
                }
                if best is None or float(current["balanced_accuracy"]) > float(best["balanced_accuracy"]):
                    best = current
                if tp == positives and tn == negatives:
                    perfect = current
                    break
            if perfect is not None:
                break
        if perfect is not None:
            break
    return {
        "simple_gate_threshold_exists": bool(perfect is not None),
        "best_scalar": perfect if perfect is not None else best,
    }


def build_gate_model_from_cfg(cfg: dict[str, Any], input_dim: int) -> SampleLevelBridgePresenceGate:
    gate_cfg = cfg.get("gate") or {}
    return SampleLevelBridgePresenceGate(
        input_dim=int(input_dim),
        hidden_dim=int(gate_cfg.get("hidden_dim", 16)),
        dropout_p=float(gate_cfg.get("dropout_p", 0.0)),
    )


def count_trainable_parameters(module: nn.Module) -> int:
    return int(sum(int(param.numel()) for param in module.parameters() if param.requires_grad))


def apply_hard_sample_gate(pixel_remove_mask: np.ndarray, gate_open: bool) -> np.ndarray:
    if bool(gate_open):
        return pixel_remove_mask.astype(np.uint8).copy()
    return np.zeros_like(pixel_remove_mask, dtype=np.uint8)


def _build_gate_state_payload(
    *,
    row: dict[str, Any],
    predicted_removed_pixels: int,
    component_count_predicted: int,
    reconstruction: dict[str, Any],
    reconstruction_timing: dict[str, Any],
) -> dict[str, Any]:
    start = row["start_reconstruction"]
    return {
        "candidate_pixels": int(row["candidate_pixels"]),
        "predicted_removed_pixels": int(predicted_removed_pixels),
        "predicted_removed_fraction": float(int(predicted_removed_pixels) / max(int(row["candidate_pixels"]), 1)),
        "start_mean_iou": float(start["metrics"]["instance_mean_matched_iou"]),
        "predicted_mean_iou": float(reconstruction["metrics"]["instance_mean_matched_iou"]),
        "start_success50": int(bool(start["metrics"]["all_iou_ge_0.50"])),
        "predicted_success50": int(bool(reconstruction["metrics"]["all_iou_ge_0.50"])),
        "component_count_start": int(row["component_count_start"]),
        "component_count_predicted": int(component_count_predicted),
        "component_topology_changed": int(int(row["component_count_start"]) != int(component_count_predicted)),
        "predicted_reconstruction_runtime_seconds": float(reconstruction_timing["total_seconds"]),
        "predicted_normalization_runtime_seconds": float(reconstruction_timing["normalization_seconds"]),
        "predicted_metric_runtime_seconds": float(reconstruction_timing["metrics_seconds"]),
        "predicted_topology_runtime_seconds": float(reconstruction_timing["topology_seconds"]),
    }


def build_hard_gate_state_cache(
    cached_records: list[dict[str, Any]],
    pixel_remove_masks: list[np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    t_cache_start = time.perf_counter()
    state_cache: list[dict[str, Any]] = []
    cpu_connected_components_seconds = 0.0
    cpu_reconstruction_seconds = 0.0
    for idx, row in enumerate(cached_records):
        original_v2_remove_pixels = int(np.sum(pixel_remove_masks[idx] > 0))
        candidate_mask = row["candidate_mask_np"].astype(np.uint8)
        closed_state = _build_gate_state_payload(
            row=row,
            predicted_removed_pixels=0,
            component_count_predicted=int(row["component_count_start"]),
            reconstruction=row["start_reconstruction"],
            reconstruction_timing={
                "total_seconds": 0.0,
                "normalization_seconds": 0.0,
                "metrics_seconds": 0.0,
                "topology_seconds": 0.0,
            },
        )
        refined_open = ((candidate_mask > 0) & (pixel_remove_masks[idx] == 0)).astype(np.uint8)
        t_cc_start = time.perf_counter()
        open_component_count = int(bridge._connected_components(refined_open.astype(np.uint8))[1])
        cpu_connected_components_seconds += float(time.perf_counter() - t_cc_start)
        open_reconstruction_timed = bridge.run_locked_reconstruction_with_timing(refined_open, row["gt_instances"])
        cpu_reconstruction_seconds += float(open_reconstruction_timed["timing"]["total_seconds"])
        open_state = _build_gate_state_payload(
            row=row,
            predicted_removed_pixels=original_v2_remove_pixels,
            component_count_predicted=int(open_component_count),
            reconstruction=open_reconstruction_timed["result"],
            reconstruction_timing=open_reconstruction_timed["timing"],
        )
        state_cache.append(
            {
                "sample_id": str(row["sample_id"]),
                "gate_target": int(row["gate_target"]),
                "bridge_positive": int(row["bridge_positive"]),
                "candidate_pixels": int(row["candidate_pixels"]),
                "original_v2_remove_pixels": int(original_v2_remove_pixels),
                "closed": closed_state,
                "open": open_state,
            }
        )
    return state_cache, {
        "cpu_connected_components_seconds": float(cpu_connected_components_seconds),
        "cpu_reconstruction_seconds": float(cpu_reconstruction_seconds),
        "total_seconds": float(time.perf_counter() - t_cache_start),
    }


def profile_hard_gate_reconstruction_states(
    cached_records: list[dict[str, Any]],
    pixel_remove_masks: list[np.ndarray],
    *,
    output_dir: Path,
    reference_implementation: str = "reference",
    optimized_implementation: str = "optimized",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(cached_records):
        candidate_mask = row["candidate_mask_np"].astype(np.uint8)
        states = [
            ("CLOSED", np.zeros_like(pixel_remove_masks[idx], dtype=np.uint8)),
            ("OPEN", pixel_remove_masks[idx].astype(np.uint8)),
        ]
        for state_name, remove_mask in states:
            refined = ((candidate_mask > 0) & (remove_mask == 0)).astype(np.uint8)
            reference = bridge.run_locked_reconstruction_profiled(
                refined,
                row["gt_instances"],
                normalizer_implementation=reference_implementation,
            )
            optimized = bridge.run_locked_reconstruction_profiled(
                refined,
                row["gt_instances"],
                normalizer_implementation=optimized_implementation,
            )
            label_mismatch = compare_nested_payloads(
                reference["result"]["labels"],
                optimized["result"]["labels"],
                path="result.instance_mask",
            )
            if label_mismatch is not None:
                raise SystemExit(
                    format_payload_mismatch(
                        sample_id=str(row["sample_id"]),
                        state_name=state_name,
                        mismatch=label_mismatch,
                        category="instance_mask_parity",
                    )
                )
            metric_mismatch = compare_nested_payloads(
                reference["result"]["metrics"],
                optimized["result"]["metrics"],
                path="metrics",
            )
            if metric_mismatch is not None:
                raise SystemExit(
                    format_payload_mismatch(
                        sample_id=str(row["sample_id"]),
                        state_name=state_name,
                        mismatch=metric_mismatch,
                        category="metric_parity",
                    )
                )
            topology_mismatch = compare_nested_payloads(
                reference["result"]["topology"],
                optimized["result"]["topology"],
                path="topology",
            )
            if topology_mismatch is not None:
                raise SystemExit(
                    format_payload_mismatch(
                        sample_id=str(row["sample_id"]),
                        state_name=state_name,
                        mismatch=topology_mismatch,
                        category="topology_parity",
                    )
                )
            ref_profile = dict(reference["profile"])
            opt_profile = dict(optimized["profile"])
            combined = {
                "sample_id": str(row["sample_id"]),
                "state": str(state_name),
                "candidate_pixels": int(row["candidate_pixels"]),
                "removal_pixels": int(np.sum(remove_mask > 0)),
                "foreground_pixels_entering_normalizer": int(ref_profile["foreground_pixels_entering_normalizer"]),
                "input_component_count": int(ref_profile["input_component_count"]),
                "expected_k": int(ref_profile["expected_k"]),
                "output_component_count": int(ref_profile["output_component_count"]),
                "reference_total_reconstruction_seconds": float(ref_profile["total_reconstruction_seconds"]),
                "optimized_total_reconstruction_seconds": float(opt_profile["total_reconstruction_seconds"]),
                "reference_input_mask_preparation_seconds": float(ref_profile["input_mask_preparation_seconds"]),
                "reference_connected_component_labeling_seconds": float(ref_profile["connected_component_labeling_seconds"]),
                "reference_component_filtering_statistics_seconds": float(ref_profile["component_filtering_statistics_seconds"]),
                "reference_seed_centroid_preparation_seconds": float(ref_profile["seed_centroid_preparation_seconds"]),
                "reference_distance_map_computation_seconds": float(ref_profile["distance_map_computation_seconds"]),
                "reference_centroid_distance_computation_seconds": float(ref_profile["centroid_distance_computation_seconds"]),
                "reference_pixel_to_instance_assignment_seconds": float(ref_profile["pixel_to_instance_assignment_seconds"]),
                "reference_per_component_python_loops_seconds": float(ref_profile["per_component_python_loops_seconds"]),
                "reference_morphology_seconds": float(ref_profile["morphology_seconds"]),
                "reference_output_instance_mask_creation_seconds": float(ref_profile["output_instance_mask_creation_seconds"]),
                "reference_gt_matching_seconds": float(ref_profile["gt_matching_seconds"]),
                "reference_iou_matrix_construction_seconds": float(ref_profile["iou_matrix_construction_seconds"]),
                "reference_success50_aggregate_seconds": float(ref_profile["success50_aggregate_seconds"]),
                "reference_array_copy_dtype_conversion_seconds": float(ref_profile["array_copy_dtype_conversion_seconds"]),
                "optimized_input_mask_preparation_seconds": float(opt_profile["input_mask_preparation_seconds"]),
                "optimized_connected_component_labeling_seconds": float(opt_profile["connected_component_labeling_seconds"]),
                "optimized_component_filtering_statistics_seconds": float(opt_profile["component_filtering_statistics_seconds"]),
                "optimized_seed_centroid_preparation_seconds": float(opt_profile["seed_centroid_preparation_seconds"]),
                "optimized_distance_map_computation_seconds": float(opt_profile["distance_map_computation_seconds"]),
                "optimized_centroid_distance_computation_seconds": float(opt_profile["centroid_distance_computation_seconds"]),
                "optimized_pixel_to_instance_assignment_seconds": float(opt_profile["pixel_to_instance_assignment_seconds"]),
                "optimized_per_component_python_loops_seconds": float(opt_profile["per_component_python_loops_seconds"]),
                "optimized_morphology_seconds": float(opt_profile["morphology_seconds"]),
                "optimized_output_instance_mask_creation_seconds": float(opt_profile["output_instance_mask_creation_seconds"]),
                "optimized_gt_matching_seconds": float(opt_profile["gt_matching_seconds"]),
                "optimized_iou_matrix_construction_seconds": float(opt_profile["iou_matrix_construction_seconds"]),
                "optimized_success50_aggregate_seconds": float(opt_profile["success50_aggregate_seconds"]),
                "optimized_array_copy_dtype_conversion_seconds": float(opt_profile["array_copy_dtype_conversion_seconds"]),
                "reference_call_counts": dict(ref_profile.get("call_counts") or {}),
                "optimized_call_counts": dict(opt_profile.get("call_counts") or {}),
                "matched_iou_per_gt": list(reference["result"]["metrics"]["matched_iou_per_gt"]),
                "success50": float(reference["result"]["metrics"]["all_iou_ge_0.50"]),
                "topology_class": str(reference["result"]["topology"]["topology_class"]),
            }
            rows.append(combined)
    rows = sorted(rows, key=lambda row: float(row["reference_total_reconstruction_seconds"]), reverse=True)
    bridge._write_json(output_dir / "reconstruction_profile.json", rows)
    bridge._write_csv(output_dir / "reconstruction_profile.csv", rows, fieldnames=list(rows[0].keys()) if rows else [])
    total_ref = float(sum(float(row["reference_total_reconstruction_seconds"]) for row in rows))
    total_opt = float(sum(float(row["optimized_total_reconstruction_seconds"]) for row in rows))
    return {
        "rows": rows,
        "reference_total_reconstruction_seconds": total_ref,
        "optimized_total_reconstruction_seconds": total_opt,
        "reference_mean_per_state_seconds": float(total_ref / max(len(rows), 1)),
        "optimized_mean_per_state_seconds": float(total_opt / max(len(rows), 1)),
        "speedup_factor": float(total_ref / max(total_opt, 1.0e-12)),
        "slowest_reference_state": rows[0] if rows else None,
        "slowest_optimized_state": max(rows, key=lambda row: float(row["optimized_total_reconstruction_seconds"])) if rows else None,
    }


def _evaluate_gate_threshold_reference(
    cached_records: list[dict[str, Any]],
    pixel_remove_masks: list[np.ndarray],
    gate_probs: np.ndarray,
    *,
    gate_threshold: float,
) -> dict[str, Any]:
    t_eval_start = time.perf_counter()
    per_sample: list[dict[str, Any]] = []
    tp = tn = fp = fn = 0
    cpu_connected_components_seconds = 0.0
    cpu_reconstruction_seconds = 0.0
    for idx, row in enumerate(cached_records):
        gate_target = int(row["gate_target"])
        gate_prob = float(gate_probs[idx])
        gate_open = bool(gate_prob >= float(gate_threshold))
        if gate_open and gate_target == 1:
            tp += 1
        elif (not gate_open) and gate_target == 0:
            tn += 1
        elif gate_open and gate_target == 0:
            fp += 1
        else:
            fn += 1
        final_remove = apply_hard_sample_gate(pixel_remove_masks[idx], gate_open)
        candidate_mask = row["candidate_mask_np"].astype(np.uint8)
        refined = ((candidate_mask > 0) & (final_remove == 0)).astype(np.uint8)
        t_cc_start = time.perf_counter()
        comp_after = int(bridge._connected_components(refined.astype(np.uint8))[1])
        cpu_connected_components_seconds += float(time.perf_counter() - t_cc_start)
        pred_timed = bridge.run_locked_reconstruction_with_timing(refined, row["gt_instances"])
        pred = pred_timed["result"]
        cpu_reconstruction_seconds += float(pred_timed["timing"]["total_seconds"])
        start = row["start_reconstruction"]
        per_sample.append(
            {
                "sample_id": str(row["sample_id"]),
                "gate_target": int(gate_target),
                "gate_prob": gate_prob,
                "gate_open": int(gate_open),
                "bridge_positive": int(row["bridge_positive"]),
                "original_v2_remove_pixels": int(np.sum(pixel_remove_masks[idx] > 0)),
                **_build_gate_state_payload(
                    row=row,
                    predicted_removed_pixels=int(np.sum(final_remove > 0)),
                    component_count_predicted=int(comp_after),
                    reconstruction=pred,
                    reconstruction_timing=pred_timed["timing"],
                ),
            }
        )
    positives = [row for row in per_sample if int(row["gate_target"]) == 1]
    negatives = [row for row in per_sample if int(row["gate_target"]) == 0]
    pos_success = int(sum(int(v["predicted_success50"]) for v in positives))
    neg_reg = int(sum(1 for v in negatives if float(v["predicted_mean_iou"]) + 1.0e-9 < float(v["start_mean_iou"])))
    neg_topo = int(sum(int(v["component_topology_changed"]) for v in negatives))
    cls_pos = max(len(positives), 1)
    cls_neg = max(len(negatives), 1)
    sensitivity = float(tp / cls_pos)
    specificity = float(tn / cls_neg)
    out = {
        "classification": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": 0.5 * (sensitivity + specificity),
        },
        "gated_reconstruction": {
            "positive_success50": int(pos_success),
            "positive_mean_matched_iou": float(np.mean([float(v["predicted_mean_iou"]) for v in positives])) if positives else 0.0,
            "negative_regressions": int(neg_reg),
            "negative_topology_changes": int(neg_topo),
            "negative_removed_fraction": float(
                sum(int(v["predicted_removed_pixels"]) for v in negatives) / max(sum(int(v["candidate_pixels"]) for v in negatives), 1)
            ) if negatives else 0.0,
            "overall_success50": int(sum(int(v["predicted_success50"]) for v in per_sample)),
            "overall_mean_matched_iou": float(np.mean([float(v["predicted_mean_iou"]) for v in per_sample])) if per_sample else 0.0,
        },
        "gate_open_samples": [str(v["sample_id"]) for v in per_sample if int(v["gate_open"]) == 1],
        "gate_closed_samples": [str(v["sample_id"]) for v in per_sample if int(v["gate_open"]) == 0],
        "per_sample": per_sample,
        "timing": {
            "cpu_connected_components_seconds": float(cpu_connected_components_seconds),
            "cpu_reconstruction_seconds": float(cpu_reconstruction_seconds),
            "total_seconds": float(time.perf_counter() - t_eval_start),
        },
    }
    out["safe_useful"] = bool(
        int(out["gated_reconstruction"]["negative_regressions"]) == 0
        and int(out["gated_reconstruction"]["negative_topology_changes"]) == 0
        and int(out["gated_reconstruction"]["positive_success50"]) >= 3
    )
    return out


def evaluate_gate_threshold_on_cached(
    hard_gate_state_cache: list[dict[str, Any]],
    gate_probs: np.ndarray,
    *,
    gate_threshold: float,
) -> dict[str, Any]:
    t_eval_start = time.perf_counter()
    per_sample: list[dict[str, Any]] = []
    tp = tn = fp = fn = 0
    for idx, row in enumerate(hard_gate_state_cache):
        gate_target = int(row["gate_target"])
        gate_prob = float(gate_probs[idx])
        gate_open = bool(gate_prob >= float(gate_threshold))
        if gate_open and gate_target == 1:
            tp += 1
        elif (not gate_open) and gate_target == 0:
            tn += 1
        elif gate_open and gate_target == 0:
            fp += 1
        else:
            fn += 1
        selected_state = row["open"] if gate_open else row["closed"]
        per_sample.append(
            {
                "sample_id": str(row["sample_id"]),
                "gate_target": int(gate_target),
                "gate_prob": gate_prob,
                "gate_open": int(gate_open),
                "bridge_positive": int(row["bridge_positive"]),
                "original_v2_remove_pixels": int(row["original_v2_remove_pixels"]),
                **{key: value for key, value in selected_state.items()},
            }
        )
    positives = [row for row in per_sample if int(row["gate_target"]) == 1]
    negatives = [row for row in per_sample if int(row["gate_target"]) == 0]
    pos_success = int(sum(int(v["predicted_success50"]) for v in positives))
    neg_reg = int(sum(1 for v in negatives if float(v["predicted_mean_iou"]) + 1.0e-9 < float(v["start_mean_iou"])))
    neg_topo = int(sum(int(v["component_topology_changed"]) for v in negatives))
    cls_pos = max(len(positives), 1)
    cls_neg = max(len(negatives), 1)
    sensitivity = float(tp / cls_pos)
    specificity = float(tn / cls_neg)
    out = {
        "classification": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": 0.5 * (sensitivity + specificity),
        },
        "gated_reconstruction": {
            "positive_success50": int(pos_success),
            "positive_mean_matched_iou": float(np.mean([float(v["predicted_mean_iou"]) for v in positives])) if positives else 0.0,
            "negative_regressions": int(neg_reg),
            "negative_topology_changes": int(neg_topo),
            "negative_removed_fraction": float(
                sum(int(v["predicted_removed_pixels"]) for v in negatives) / max(sum(int(v["candidate_pixels"]) for v in negatives), 1)
            ) if negatives else 0.0,
            "overall_success50": int(sum(int(v["predicted_success50"]) for v in per_sample)),
            "overall_mean_matched_iou": float(np.mean([float(v["predicted_mean_iou"]) for v in per_sample])) if per_sample else 0.0,
        },
        "gate_open_samples": [str(v["sample_id"]) for v in per_sample if int(v["gate_open"]) == 1],
        "gate_closed_samples": [str(v["sample_id"]) for v in per_sample if int(v["gate_open"]) == 0],
        "per_sample": per_sample,
        "timing": {
            "cpu_connected_components_seconds": 0.0,
            "cpu_reconstruction_seconds": 0.0,
            "total_seconds": float(time.perf_counter() - t_eval_start),
        },
    }
    out["safe_useful"] = bool(
        int(out["gated_reconstruction"]["negative_regressions"]) == 0
        and int(out["gated_reconstruction"]["negative_topology_changes"]) == 0
        and int(out["gated_reconstruction"]["positive_success50"]) >= 3
    )
    return out


def gate_threshold_sweep(
    hard_gate_state_cache: list[dict[str, Any]],
    gate_probs: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for threshold in thresholds:
        current = evaluate_gate_threshold_on_cached(
            hard_gate_state_cache,
            gate_probs,
            gate_threshold=float(threshold),
        )
        out.append({"gate_threshold": float(threshold), **current})
    return out


def safe_useful_key(eval_payload: dict[str, Any]) -> tuple[Any, ...]:
    gated = eval_payload["gated_reconstruction"]
    cls = eval_payload["classification"]
    return (
        int(gated["positive_success50"]),
        float(gated["positive_mean_matched_iou"]),
        float(cls["balanced_accuracy"]),
        float(gated["overall_mean_matched_iou"]),
    )
