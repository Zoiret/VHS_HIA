from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e

import audit_semantic_soft_logit_recoverability as soft_audit
import evaluate_semantic_topology_aux_postrun as postrun
import leaflet_oracle_count_geometric_split_audit as base_audit
import leaflet_oracle_count_geometric_split_forensic as forensic
import leaflet_oracle_k_constrained_normalization_audit as k_audit
import semantic_topology_aux as topo_aux
import validate_centerhead as center_metrics
from dataset import read_split_file


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEMANTIC_CONFIG = REPO_ROOT / "training" / "configs" / "unetpp_effb3_semantic_topology_aux_finetune_100ep.yaml"
DEFAULT_SEMANTIC_CHECKPOINT = (
    REPO_ROOT / "training" / "runs" / "unetpp_effb3_a100_multiclass_curated_finetune_stage2_lr1e5_100ep" / "best_mean_fg.pth"
)
DEFAULT_TRAIN_SPLIT = REPO_ROOT / "datasets" / "converted_full_multiclass_curated" / "train.txt"
DEFAULT_VAL_SPLIT = REPO_ROOT / "datasets" / "converted_full_multiclass_curated" / "val.txt"
DEFAULT_TEST_SPLIT = REPO_ROOT / "datasets" / "converted_full_multiclass_curated" / "test.txt"
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "converted_full_multiclass"
DEFAULT_INSTANCE_ROOT = REPO_ROOT / "datasets" / "converted_leaflet_instances"
MICRO_MANIFEST_PATH = REPO_ROOT / "training" / "manifests" / "bridge_suppression_micro_overfit_manifest.json"
MICRO_MANIFEST_V2_PATH = REPO_ROOT / "training" / "manifests" / "bridge_suppression_micro_overfit_v2_manifest.json"
PROHIBITED_PATH_SUBSTRINGS = ("center_full_val_manifest.jsonl", "authoritative_106_holdout", "holdout")
LOCKED_CANDIDATE_THRESHOLD = 0.50
BRIDGE_REMOVE_THRESHOLD = 0.50
MICROSET_POSITIVE_TARGET = 8
MICROSET_NEGATIVE_TARGET = 2


@dataclass(frozen=True)
class SplitAudit:
    split_path: str
    sample_count: int
    patient_count: int
    gt1: int
    gt2: int
    gt3: int
    bridge_positive_samples: int
    sample_overlap_with_train: int
    patient_overlap_with_train: int
    instance_mask_available: int
    gt_instance_count_available: int


def _resolve_repo_path(path_like: str | Path | None, default: Path) -> Path:
    if path_like is None:
        return default.resolve()
    try:
        portable_text = _portable_repo_path_text(path_like, repo_root=REPO_ROOT, platform_name=os.name)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    return Path(portable_text).resolve()


def _is_windows_absolute_text(path_text: str) -> bool:
    text = str(path_text).strip()
    return bool(re.match(r"^[A-Za-z]:[\\/]", text)) or text.startswith("\\\\")


def _is_posix_absolute_text(path_text: str) -> bool:
    text = str(path_text).strip()
    return text.startswith("/")


def _portable_repo_path_text(path_like: str | Path, *, repo_root: str | Path, platform_name: str) -> str:
    raw = str(path_like).strip()
    if not raw:
        raise ValueError("Portable path error: empty path is not allowed.")
    repo_root_text = str(repo_root).strip()
    if platform_name == "nt":
        if _is_posix_absolute_text(raw) and not _is_windows_absolute_text(raw):
            raise ValueError(
                f"Portable path error: POSIX-style absolute path is not valid on Windows platform: {raw}. "
                f"Use a repository-relative path instead."
            )
        if _is_windows_absolute_text(raw):
            return str(PureWindowsPath(raw))
        return str(PureWindowsPath(repo_root_text) / raw)
    if _is_windows_absolute_text(raw):
        raise ValueError(
            f"Portable path error: Windows-style absolute path is not valid on POSIX platform: {raw}. "
            f"Use a repository-relative path instead."
        )
    repo_root_posix = repo_root_text.replace("\\", "/")
    if _is_posix_absolute_text(raw):
        return str(PurePosixPath(raw))
    return str(PurePosixPath(repo_root_posix) / raw.replace("\\", "/"))


def _repo_relative_canonical_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _assert_safe_path(path: Path) -> None:
    text = str(path).replace("\\", "/").lower()
    for token in PROHIBITED_PATH_SUBSTRINGS:
        if token.lower() in text:
            raise SystemExit(f"Prohibited path detected in bridge suppression preparation: {path}")


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise SystemExit(
            "PyYAML is not installed. Install training deps with:\n"
            "  py -m pip install -r requirements-train.txt"
        ) from e
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Expected YAML dict at {path}")
    return data


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical_model_state_sha256(state_dict: dict[str, Any]) -> str:
    """Historical semantic hash for model state payloads.

    This intentionally matches the V2 threshold-sweep contract:
    sorted keys, UTF-8 key bytes, dtype text, JSON shape text, and raw
    contiguous CPU tensor bytes.
    """
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        h.update(str(key).encode("utf-8"))
        if torch.is_tensor(value):
            arr = value.detach().cpu().contiguous()
            h.update(str(arr.dtype).encode("utf-8"))
            h.update(json.dumps(list(arr.shape)).encode("utf-8"))
            h.update(arr.numpy().tobytes(order="C"))
        else:
            h.update(repr(value).encode("utf-8"))
    return h.hexdigest()


def _canonical_split_rows(path: Path) -> list[str]:
    return [
        row.strip()
        for row in path.read_text(encoding="utf-8-sig").splitlines()
        if row.strip()
    ]


def _canonical_split_sha256(path: Path) -> str:
    rows = _canonical_split_rows(path)
    payload = ("\n".join(rows) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_locked_manifest_source_split(
    *,
    manifest_payload: dict[str, Any],
    configured_train_split: Path,
) -> dict[str, Any]:
    summary = {
        "status": "pass",
        "manifest_path": str(_resolve_repo_path(manifest_payload.get("_manifest_path"), MICRO_MANIFEST_V2_PATH)),
        "manifest_source_split": str(manifest_payload.get("source_split", "")),
        "configured_train_split": str(configured_train_split.resolve()),
        "resolved_source_split": None,
        "expected_source_split_canonical_sha256": str(manifest_payload.get("source_split_canonical_sha256", "")),
        "actual_source_split_canonical_sha256": None,
        "error": None,
    }
    source_split = manifest_payload.get("source_split")
    expected_sha = str(manifest_payload.get("source_split_canonical_sha256", "")).strip().lower()
    if not isinstance(source_split, str) or not source_split.strip():
        summary["status"] = "blocked"
        summary["error"] = "Locked micro manifest must contain a non-empty source_split string."
        return summary
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        summary["status"] = "blocked"
        summary["error"] = "Locked micro manifest must contain source_split_canonical_sha256 as a 64-character lowercase hex string."
        return summary
    try:
        resolved_source_split = _resolve_repo_path(source_split, configured_train_split)
    except SystemExit as e:
        summary["status"] = "blocked"
        summary["error"] = str(e)
        return summary
    summary["resolved_source_split"] = str(resolved_source_split)
    if resolved_source_split != configured_train_split.resolve():
        summary["status"] = "blocked"
        summary["error"] = (
            "Locked micro manifest must use train split only. "
            f"Manifest source_split={resolved_source_split} train_split={configured_train_split.resolve()}"
        )
        return summary
    actual_sha = _canonical_split_sha256(resolved_source_split).lower()
    summary["actual_source_split_canonical_sha256"] = actual_sha
    if actual_sha != expected_sha:
        summary["status"] = "blocked"
        summary["error"] = (
            "Locked micro manifest TRAIN canonical SHA256 mismatch. "
            f"manifest={expected_sha} actual={actual_sha} path={resolved_source_split}"
        )
    return summary


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _amp_enabled(cfg: dict[str, Any], device: torch.device) -> bool:
    train_cfg = cfg.get("train") or {}
    v = train_cfg.get("amp", None)
    if v is None:
        return device.type == "cuda"
    return bool(v) and device.type == "cuda"


def _semantic_inference_amp_enabled(cfg: dict[str, Any], device: torch.device) -> bool:
    semantic_cfg = cfg.get("semantic_inference") or {}
    v = semantic_cfg.get("amp", False)
    return bool(v) and device.type == "cuda"


def _semantic_inference_backend_requested(cfg: dict[str, Any]) -> dict[str, Any]:
    semantic_cfg = cfg.get("semantic_inference") or {}
    return {
        "amp_requested": bool(semantic_cfg.get("amp", False)),
        "matmul_allow_tf32": bool(semantic_cfg.get("matmul_allow_tf32", False)),
        "cudnn_allow_tf32": bool(semantic_cfg.get("cudnn_allow_tf32", False)),
        "cudnn_benchmark": bool(semantic_cfg.get("cudnn_benchmark", False)),
        "cudnn_deterministic": bool(semantic_cfg.get("cudnn_deterministic", True)),
    }


def semantic_inference_backend_summary(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    requested = _semantic_inference_backend_requested(cfg)
    return {
        **requested,
        "amp_enabled": bool(_semantic_inference_amp_enabled(cfg, device)),
    }


@contextlib.contextmanager
def _semantic_inference_backend_ctx(cfg: dict[str, Any], device: torch.device):
    requested = _semantic_inference_backend_requested(cfg)
    prev_matmul_allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    prev_cudnn_allow_tf32 = bool(torch.backends.cudnn.allow_tf32)
    prev_cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    prev_cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(requested["matmul_allow_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(requested["cudnn_allow_tf32"])
        torch.backends.cudnn.benchmark = bool(requested["cudnn_benchmark"])
        torch.backends.cudnn.deterministic = bool(requested["cudnn_deterministic"])
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_allow_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_allow_tf32
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark
        torch.backends.cudnn.deterministic = prev_cudnn_deterministic


def _autocast_ctx(device: torch.device, enabled: bool):
    if device.type == "cuda" and bool(enabled):
        return torch.amp.autocast("cuda", enabled=True)
    return contextlib.nullcontext()


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_grad_scaler(device: torch.device, enabled: bool):
    if device.type == "cuda" and bool(enabled):
        return torch.amp.GradScaler("cuda")
    return None


def _make_patient_id(sample_id: str) -> str:
    parts = str(sample_id).split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return str(sample_id)


def _connected_components(mask01: np.ndarray) -> tuple[np.ndarray, int]:
    n, labels = cv2.connectedComponents(mask01.astype(np.uint8), connectivity=8)
    return labels.astype(np.int32), max(int(n) - 1, 0)


def _mask_rgb(mask01: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(mask01.shape + (3,), dtype=np.uint8)
    out[mask01.astype(bool)] = np.asarray(color, dtype=np.uint8)
    return out


def _instance_rgb(labels_u8: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [0, 0, 0],
            [230, 57, 70],
            [29, 153, 243],
            [38, 166, 154],
            [255, 183, 3],
            [171, 71, 188],
            [240, 98, 146],
        ],
        dtype=np.uint8,
    )
    out = np.zeros(labels_u8.shape + (3,), dtype=np.uint8)
    for label in np.unique(labels_u8):
        idx = int(label)
        if idx <= 0:
            continue
        out[labels_u8 == idx] = palette[idx % len(palette)]
    return out


def _probability_heatmap_rgb(p_leaf: np.ndarray) -> np.ndarray:
    scaled = np.clip(np.round(p_leaf * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def _panel(title: str, image_rgb: np.ndarray) -> np.ndarray:
    canvas = np.full((image_rgb.shape[0] + 36, image_rgb.shape[1], 3), 18, dtype=np.uint8)
    canvas[36:, :, :] = image_rgb
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def _save_visual_grid(
    *,
    out_path: Path,
    image_rgb: np.ndarray,
    gt_inst_u8: np.ndarray,
    p_leaf: np.ndarray,
    candidate_mask01: np.ndarray,
    bridge_target01: np.ndarray,
    candidate_minus_oracle01: np.ndarray,
    reconstructed_u8: np.ndarray,
) -> str:
    panels = [
        _panel("RGB", image_rgb),
        _panel("GT Instances", _instance_rgb(gt_inst_u8)),
        _panel("P(leaflet)", _probability_heatmap_rgb(p_leaf)),
        _panel("P>=0.50 Candidate", _mask_rgb(candidate_mask01, (255, 255, 255))),
        _panel("FALSE_BRIDGE_PIXELS", _mask_rgb(bridge_target01, (255, 255, 255))),
        _panel("Candidate Minus Oracle Bridge", _mask_rgb(candidate_minus_oracle01, (255, 255, 255))),
        _panel("Reconstructed Instances", _instance_rgb(reconstructed_u8)),
    ]
    filler = np.full_like(panels[0], 18)
    row1 = np.concatenate(panels[:4], axis=1)
    row2 = np.concatenate([panels[4], panels[5], panels[6], filler], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    return str(out_path.resolve())


def false_bridge_pixels_from_candidate(
    *,
    gt_sem_u8: np.ndarray,
    gt_inst_u8: np.ndarray,
    candidate_mask01: np.ndarray,
) -> np.ndarray:
    bridge_component = soft_audit._bridge_component_mask(gt_inst_u8.astype(np.uint8), candidate_mask01.astype(np.uint8))
    false_leaflet = ((gt_sem_u8 != 1) & (candidate_mask01 > 0)).astype(np.uint8)
    return ((false_leaflet > 0) & (bridge_component > 0)).astype(np.uint8)


def candidate_p50_mask_from_probs(p_leaf: np.ndarray) -> np.ndarray:
    return (p_leaf >= float(LOCKED_CANDIDATE_THRESHOLD)).astype(np.uint8)


def refine_candidate_with_bridge_probs(candidate_mask01: np.ndarray, bridge_probs: np.ndarray, threshold: float = BRIDGE_REMOVE_THRESHOLD) -> np.ndarray:
    remove01 = (bridge_probs >= float(threshold)).astype(np.uint8)
    return ((candidate_mask01 > 0) & (remove01 == 0)).astype(np.uint8)


def run_locked_reconstruction(pred_leaf01: np.ndarray, gt_inst_u8: np.ndarray) -> dict[str, Any]:
    return run_locked_reconstruction_with_timing(pred_leaf01, gt_inst_u8)["result"]


def _compute_detailed_instance_metrics_profiled(gt_inst_u8: np.ndarray, pred_inst_u8: np.ndarray, *, gt_k: int, pred_k: int) -> dict[str, Any]:
    t_iou_start = time.perf_counter()
    iou_mat = center_metrics._iou_matrix(gt_inst_u8, pred_inst_u8, int(gt_k), int(pred_k))
    t_iou_end = time.perf_counter()
    t_base_assign_start = time.perf_counter()
    sum_iou = center_metrics._best_perm_sum(iou_mat)
    t_base_assign_end = time.perf_counter()
    mean_iou = float(sum_iou / max(int(gt_k), 1))
    case = center_metrics._case_type(int(gt_k), int(pred_k))
    base = {
        "gt_instance_count": int(gt_k),
        "pred_instance_count": int(pred_k),
        "case": str(case),
        "instance_exact_count": bool(int(pred_k) == int(gt_k)),
        "instance_exact_count_acc": float(int(int(pred_k) == int(gt_k))),
        "instance_mean_matched_iou": float(mean_iou),
        "instance_merged": bool(case == "merged"),
        "instance_fragmented": bool(case == "fragmented"),
        "instance_mixed": bool(case == "mixed"),
        "instance_merged_rate": float(int(case == "merged")),
        "instance_fragmented_rate": float(int(case == "fragmented")),
        "instance_mixed_rate": float(int(case == "mixed")),
        "instance_perfect": bool((int(pred_k) == int(gt_k)) and (mean_iou >= 0.90)),
        "instance_perfect_rate": float(int((int(pred_k) == int(gt_k)) and (mean_iou >= 0.90))),
        "iou_matrix": iou_mat,
    }
    t_match_start = time.perf_counter()
    assign = base_audit._best_assignment(np.asarray(iou_mat, dtype=np.float64))
    t_match_end = time.perf_counter()
    t_agg_start = time.perf_counter()
    matched_ious = [0.0 for _ in range(int(gt_k))]
    matched_pred_ids: set[int] = set()
    for gi, pi in assign["pairs"]:
        val = float(iou_mat[int(gi), int(pi)])
        matched_ious[int(gi)] = val
        if val > 0.0:
            matched_pred_ids.add(int(pi) + 1)
    unmatched_gt = int(sum(1 for v in matched_ious if float(v) <= 0.0))
    unmatched_pred = int(len([pi for pi in range(1, int(pred_k) + 1) if pi not in matched_pred_ids]))
    thresholds = {
        "all_iou_ge_0.50": float(all(float(v) >= 0.50 for v in matched_ious)) if gt_k > 0 else 0.0,
        "all_iou_ge_0.70": float(all(float(v) >= 0.70 for v in matched_ious)) if gt_k > 0 else 0.0,
        "all_iou_ge_0.80": float(all(float(v) >= 0.80 for v in matched_ious)) if gt_k > 0 else 0.0,
    }
    base.update(
        {
            "matched_iou_per_gt": [float(v) for v in matched_ious],
            "median_matched_iou": float(np.median(np.asarray(matched_ious, dtype=np.float64))) if matched_ious else 0.0,
            "unmatched_gt_instances": unmatched_gt,
            "unmatched_pred_instances": unmatched_pred,
            **thresholds,
        }
    )
    t_agg_end = time.perf_counter()
    return {
        "metrics": base,
        "timing": {
            "iou_matrix_seconds": float(t_iou_end - t_iou_start),
            "base_assignment_seconds": float(t_base_assign_end - t_base_assign_start),
            "gt_matching_seconds": float(t_match_end - t_match_start),
            "success_aggregate_seconds": float(t_agg_end - t_agg_start),
            "total_seconds": float(t_agg_end - t_iou_start),
        },
    }


def run_locked_reconstruction_profiled(
    pred_leaf01: np.ndarray,
    gt_inst_u8: np.ndarray,
    *,
    normalizer_implementation: str = "optimized",
) -> dict[str, Any]:
    t_copy_start = time.perf_counter()
    pred_leaf01 = pred_leaf01.astype(np.uint8)
    gt_inst_u8 = gt_inst_u8.astype(np.uint8)
    t_copy_end = time.perf_counter()
    t_gt_k_start = time.perf_counter()
    gt_k = int(len(topo_aux._positive_instance_ids(gt_inst_u8)))
    t_gt_k_end = time.perf_counter()
    normalization_profile: dict[str, Any] = {"implementation": str(normalizer_implementation)}
    t_norm_start = time.perf_counter()
    normalized = k_audit.normalize_mask_exact_k(
        pred_leaf01,
        gt_k,
        postrun.NORMALIZER_METHOD,
        implementation=str(normalizer_implementation),
        profile=normalization_profile,
    )
    t_norm_end = time.perf_counter()
    t_output_cast_start = time.perf_counter()
    pred_inst = normalized["labels"].astype(np.uint8)
    t_output_cast_end = time.perf_counter()
    metric_profiled = _compute_detailed_instance_metrics_profiled(gt_inst_u8, pred_inst, gt_k=gt_k, pred_k=int(normalized["final_group_count"]))
    t_topology_start = time.perf_counter()
    topology = forensic.classify_semantic_topology(gt_inst_u8, pred_leaf01)
    t_topology_end = time.perf_counter()
    result = {
        "gt_k": gt_k,
        "pred_k": int(normalized["final_group_count"]),
        "labels": pred_inst,
        "metrics": metric_profiled["metrics"],
        "topology": topology,
    }
    return {
        "result": result,
        "timing": {
            "normalization_seconds": float(t_norm_end - t_norm_start),
            "metrics_seconds": float(metric_profiled["timing"]["total_seconds"]),
            "topology_seconds": float(t_topology_end - t_topology_start),
            "total_seconds": float(t_topology_end - t_copy_start),
        },
        "profile": {
            "normalizer_method": str(postrun.NORMALIZER_METHOD),
            "normalizer_implementation": str(normalizer_implementation),
            "input_mask_preparation_seconds": float(t_copy_end - t_copy_start),
            "expected_k_seconds": float(t_gt_k_end - t_gt_k_start),
            "foreground_pixels_entering_normalizer": int(np.sum(pred_leaf01 > 0)),
            "expected_k": int(gt_k),
            "input_component_count": int(_connected_components(pred_leaf01)[1]),
            "output_component_count": int(normalized["final_group_count"]),
            "array_copy_dtype_conversion_seconds": float((t_copy_end - t_copy_start) + (t_output_cast_end - t_output_cast_start)),
            "connected_component_labeling_seconds": float(normalization_profile.get("connected_component_labeling_seconds", 0.0)),
            "component_filtering_statistics_seconds": float(normalization_profile.get("component_filtering_statistics_seconds", 0.0)),
            "seed_centroid_preparation_seconds": float(normalization_profile.get("seed_centroid_preparation_seconds", 0.0)),
            "distance_map_computation_seconds": float(normalization_profile.get("distance_map_computation_seconds", 0.0)),
            "centroid_distance_computation_seconds": float(normalization_profile.get("centroid_distance_computation_seconds", 0.0)),
            "pixel_to_instance_assignment_seconds": float(normalization_profile.get("pixel_to_instance_assignment_seconds", 0.0)),
            "per_component_python_loops_seconds": float(normalization_profile.get("per_component_python_loops_seconds", 0.0)),
            "morphology_seconds": float(normalization_profile.get("morphology_seconds", 0.0)),
            "output_instance_mask_creation_seconds": float(normalization_profile.get("output_instance_mask_creation_seconds", 0.0)),
            "gt_matching_seconds": float(metric_profiled["timing"]["gt_matching_seconds"]),
            "iou_matrix_construction_seconds": float(metric_profiled["timing"]["iou_matrix_seconds"]),
            "success50_aggregate_seconds": float(metric_profiled["timing"]["success_aggregate_seconds"]),
            "base_assignment_seconds": float(metric_profiled["timing"]["base_assignment_seconds"]),
            "topology_seconds": float(t_topology_end - t_topology_start),
            "total_reconstruction_seconds": float(t_topology_end - t_copy_start),
            "call_counts": dict(normalization_profile.get("call_counts") or {}),
        },
    }


def run_locked_reconstruction_with_timing(pred_leaf01: np.ndarray, gt_inst_u8: np.ndarray) -> dict[str, Any]:
    pred_leaf01 = pred_leaf01.astype(np.uint8)
    gt_inst_u8 = gt_inst_u8.astype(np.uint8)
    t0 = time.perf_counter()
    gt_k = int(len(topo_aux._positive_instance_ids(gt_inst_u8)))
    normalized = postrun.run_locked_normalization(pred_leaf01, gt_k)
    t1 = time.perf_counter()
    pred_inst = normalized["labels"].astype(np.uint8)
    metrics = base_audit.compute_detailed_instance_metrics(gt_inst_u8, pred_inst, gt_k=gt_k, pred_k=int(normalized["final_group_count"]))
    t2 = time.perf_counter()
    topology = forensic.classify_semantic_topology(gt_inst_u8, pred_leaf01)
    t3 = time.perf_counter()
    result = {
        "gt_k": gt_k,
        "pred_k": int(normalized["final_group_count"]),
        "labels": pred_inst,
        "metrics": metrics,
        "topology": topology,
    }
    return {
        "result": result,
        "timing": {
            "normalization_seconds": float(t1 - t0),
            "metrics_seconds": float(t2 - t1),
            "topology_seconds": float(t3 - t2),
            "total_seconds": float(t3 - t0),
        },
    }


class ContextProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 8) -> None:
        super().__init__()
        self.proj = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class BridgeSuppressionHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 16) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(int(in_channels), int(hidden_channels), kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(hidden_channels), 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FrozenSemanticBridgeSuppressionModel(nn.Module):
    def __init__(
        self,
        *,
        encoder_name: str,
        encoder_weights: str | None,
        in_channels: int,
        classes: int,
        context_channels_out: int = 8,
        bridge_hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ModuleNotFoundError as e:
            raise SystemExit(
                "segmentation-models-pytorch is not installed. Install training deps with:\n"
                "  py -m pip install -r requirements-train.txt"
            ) from e

        self.base = smp.UnetPlusPlus(
            encoder_name=str(encoder_name),
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=int(classes),
        )
        self._tap_paths = ["base.decoder.blocks.x_0_4", "base.decoder.blocks.x_2_2"]
        self._tap_outputs: dict[str, torch.Tensor] = {}
        modules = dict(self.named_modules())
        for path in self._tap_paths:
            module = modules.get(path, None)
            if module is None:
                raise RuntimeError(f"Feature tap module not found: {path}")
            module.register_forward_hook(self._make_tap_hook(path))

        x04_mod = modules["base.decoder.blocks.x_0_4"]
        x22_mod = modules["base.decoder.blocks.x_2_2"]
        x04_channels = int(self._infer_out_channels(x04_mod))
        x22_channels = int(self._infer_out_channels(x22_mod))
        self.x_0_4_channels = x04_channels
        self.x_2_2_channels = x22_channels
        self.context_projection = ContextProjection(in_channels=x22_channels, out_channels=int(context_channels_out))
        self.bridge_head = BridgeSuppressionHead(in_channels=int(x04_channels + context_channels_out + 1), hidden_channels=int(bridge_hidden_channels))

    def _make_tap_hook(self, path: str):
        def _hook(_module, _inputs, output):
            self._tap_outputs[path] = output

        return _hook

    @staticmethod
    def _infer_out_channels(module: nn.Module) -> int:
        convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
        if not convs:
            raise RuntimeError(f"Failed to infer output channels for module {type(module).__name__}")
        return int(convs[-1].out_channels)

    def freeze_semantic_base(self) -> None:
        for param in self.base.parameters():
            param.requires_grad = False
        self.base.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        self.context_projection.train(mode)
        self.bridge_head.train(mode)
        return self

    def bridge_forward_from_cached(self, *, x_0_4: torch.Tensor, x_2_2: torch.Tensor, p_leaf: torch.Tensor) -> dict[str, torch.Tensor]:
        train_dtype = self.context_projection.proj.weight.dtype
        x_0_4_fp = x_0_4.to(dtype=train_dtype)
        x_2_2_fp = x_2_2.to(dtype=train_dtype)
        p_leaf_fp = p_leaf.to(dtype=train_dtype)
        projected = self.context_projection(x_2_2_fp)
        projected_up = F.interpolate(projected, size=x_0_4_fp.shape[-2:], mode="bilinear", align_corners=False)
        concat = torch.cat([x_0_4_fp, projected_up, p_leaf_fp], dim=1)
        bridge_logits = self.bridge_head(concat)
        candidate_mask = (p_leaf_fp >= float(LOCKED_CANDIDATE_THRESHOLD)).to(dtype=train_dtype)
        return {
            "x_0_4": x_0_4_fp,
            "x_2_2": x_2_2_fp,
            "projected_context": projected_up,
            "p_leaf": p_leaf_fp,
            "candidate_mask": candidate_mask,
            "bridge_logits": bridge_logits,
            "concatenated": concat,
        }

    def forward(self, image_t: torch.Tensor) -> dict[str, torch.Tensor]:
        self._tap_outputs = {}
        with torch.no_grad():
            features = self.base.encoder(image_t)
            decoder_feature = self.base.decoder(features)
            semantic_logits = self.base.segmentation_head(decoder_feature)
        x_0_4 = self._tap_outputs["base.decoder.blocks.x_0_4"]
        x_2_2 = self._tap_outputs["base.decoder.blocks.x_2_2"]
        p_leaf = torch.softmax(semantic_logits.float(), dim=1)[:, 1:2]
        bridge = self.bridge_forward_from_cached(x_0_4=x_0_4, x_2_2=x_2_2, p_leaf=p_leaf)
        bridge["semantic_logits"] = semantic_logits
        return bridge


def build_model_from_cfg(cfg: dict[str, Any]) -> FrozenSemanticBridgeSuppressionModel:
    model_cfg = cfg.get("model") or {}
    bridge_cfg = cfg.get("bridge_head") or {}
    encoder_name = model_cfg.get("encoder_name") or model_cfg.get("encoder")
    if not encoder_name:
        raise SystemExit("Config: model.encoder_name is required")
    model = FrozenSemanticBridgeSuppressionModel(
        encoder_name=str(encoder_name),
        encoder_weights=model_cfg.get("encoder_weights", None),
        in_channels=int(model_cfg["in_channels"]),
        classes=int(model_cfg["classes"]),
        context_channels_out=int(bridge_cfg.get("context_projection_channels", 8)),
        bridge_hidden_channels=int(bridge_cfg.get("hidden_channels", 16)),
    )
    model.freeze_semantic_base()
    return model


def load_semantic_checkpoint(model: FrozenSemanticBridgeSuppressionModel, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise SystemExit(f"Unsupported checkpoint format: {checkpoint_path}")
    incompat = model.base.load_state_dict(state, strict=True)
    missing = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
    unexpected = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
    if missing or unexpected:
        raise RuntimeError(f"Unexpected checkpoint incompatibility: missing={missing[:5]} unexpected={unexpected[:5]}")
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
    }


class CandidateBalancedBCEDiceLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1.0e-6,
        *,
        lambda_negative_mean: float = 0.0,
        lambda_negative_hard: float = 0.0,
        negative_hard_topk_fraction: float = 0.01,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.lambda_negative_mean = float(lambda_negative_mean)
        self.lambda_negative_hard = float(lambda_negative_hard)
        self.negative_hard_topk_fraction = float(negative_hard_topk_fraction)

    def _negative_sample_bce_terms(
        self,
        *,
        bridge_logits: torch.Tensor,
        bridge_target: torch.Tensor,
        candidate_mask: torch.Tensor,
        bridge_positive: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        device = bridge_logits.device
        zero = bridge_logits.sum() * 0.0
        logits_flat = bridge_logits[:, 0]
        target_flat = bridge_target[:, 0]
        candidate_flat = candidate_mask[:, 0] > 0.5
        target_positive = (target_flat > 0.5) & candidate_flat
        if bridge_positive is None:
            bridge_positive = torch.any(target_positive.reshape(target_positive.shape[0], -1), dim=1)
        else:
            bridge_positive = bridge_positive.to(device=device).bool().reshape(-1)
        negative_sample_mask = ~bridge_positive
        if int(torch.sum(negative_sample_mask).item()) == 0:
            return {
                "negative_candidate_mean_bce": zero,
                "negative_candidate_hard_bce": zero,
                "negative_sample_count": torch.tensor(0.0, device=device),
            }
        sample_mean_losses: list[torch.Tensor] = []
        sample_hard_losses: list[torch.Tensor] = []
        for idx in torch.nonzero(negative_sample_mask, as_tuple=False).reshape(-1):
            sample_logits = logits_flat[int(idx)]
            sample_candidate = candidate_flat[int(idx)]
            sample_logits = sample_logits[sample_candidate]
            if int(sample_logits.numel()) == 0:
                sample_mean_losses.append(zero)
                sample_hard_losses.append(zero)
                continue
            sample_mean_losses.append(
                F.binary_cross_entropy_with_logits(sample_logits, torch.zeros_like(sample_logits), reduction="mean")
            )
            k = max(1, int(math.ceil(float(sample_logits.numel()) * self.negative_hard_topk_fraction)))
            top_logits = torch.topk(sample_logits, k=k, largest=True).values
            sample_hard_losses.append(
                F.binary_cross_entropy_with_logits(top_logits, torch.zeros_like(top_logits), reduction="mean")
            )
        return {
            "negative_candidate_mean_bce": torch.stack(sample_mean_losses).mean() if sample_mean_losses else zero,
            "negative_candidate_hard_bce": torch.stack(sample_hard_losses).mean() if sample_hard_losses else zero,
            "negative_sample_count": torch.tensor(float(len(sample_mean_losses)), device=device),
        }

    def forward(
        self,
        *,
        bridge_logits: torch.Tensor,
        bridge_target: torch.Tensor,
        candidate_mask: torch.Tensor,
        bridge_positive: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        candidate = candidate_mask > 0.5
        logits_flat = bridge_logits[candidate]
        target_flat = bridge_target[candidate]
        zero = bridge_logits.sum() * 0.0
        if int(logits_flat.numel()) == 0:
            return {
                "loss": zero,
                "balanced_bce": zero,
                "dice_loss": zero,
                "positive_bce": zero,
                "negative_bce": zero,
                "base_bridge_loss": zero,
                "negative_candidate_mean_bce": zero,
                "negative_candidate_hard_bce": zero,
                "weighted_preservation_loss": zero,
                "candidate_count": torch.tensor(0, device=bridge_logits.device, dtype=torch.float32),
                "positive_count": torch.tensor(0, device=bridge_logits.device, dtype=torch.float32),
                "negative_count": torch.tensor(0, device=bridge_logits.device, dtype=torch.float32),
                "negative_sample_count": torch.tensor(0, device=bridge_logits.device, dtype=torch.float32),
            }
        pos = target_flat > 0.5
        neg = ~pos
        if bool(torch.any(pos)):
            pos_loss = F.binary_cross_entropy_with_logits(logits_flat[pos], torch.ones_like(logits_flat[pos]), reduction="mean")
        else:
            pos_loss = zero
        if bool(torch.any(neg)):
            neg_loss = F.binary_cross_entropy_with_logits(logits_flat[neg], torch.zeros_like(logits_flat[neg]), reduction="mean")
        else:
            neg_loss = zero
        balanced_bce = 0.5 * pos_loss + 0.5 * neg_loss
        probs = torch.sigmoid(logits_flat)
        target_float = target_flat.float()
        intersection = torch.sum(probs * target_float)
        dice_coeff = (2.0 * intersection + self.eps) / (torch.sum(probs) + torch.sum(target_float) + self.eps)
        dice_loss = 1.0 - dice_coeff
        base_bridge_loss = balanced_bce + dice_loss
        preservation = self._negative_sample_bce_terms(
            bridge_logits=bridge_logits,
            bridge_target=bridge_target,
            candidate_mask=candidate_mask,
            bridge_positive=bridge_positive,
        )
        weighted_preservation_loss = (
            float(self.lambda_negative_mean) * preservation["negative_candidate_mean_bce"]
            + float(self.lambda_negative_hard) * preservation["negative_candidate_hard_bce"]
        )
        total = base_bridge_loss + weighted_preservation_loss
        return {
            "loss": total,
            "base_bridge_loss": base_bridge_loss,
            "balanced_bce": balanced_bce,
            "dice_loss": dice_loss,
            "positive_bce": pos_loss,
            "negative_bce": neg_loss,
            "negative_candidate_mean_bce": preservation["negative_candidate_mean_bce"],
            "negative_candidate_hard_bce": preservation["negative_candidate_hard_bce"],
            "weighted_preservation_loss": weighted_preservation_loss,
            "candidate_count": torch.tensor(float(logits_flat.numel()), device=bridge_logits.device),
            "positive_count": torch.tensor(float(torch.sum(pos).item()), device=bridge_logits.device),
            "negative_count": torch.tensor(float(torch.sum(neg).item()), device=bridge_logits.device),
            "negative_sample_count": preservation["negative_sample_count"],
        }


def build_bridge_loss_from_cfg(cfg: dict[str, Any]) -> CandidateBalancedBCEDiceLoss:
    loss_cfg = cfg.get("loss") or {}
    return CandidateBalancedBCEDiceLoss(
        eps=float(loss_cfg.get("eps", 1.0e-6)),
        lambda_negative_mean=float(loss_cfg.get("lambda_negative_mean", 0.0)),
        lambda_negative_hard=float(loss_cfg.get("lambda_negative_hard", 0.0)),
        negative_hard_topk_fraction=float(loss_cfg.get("negative_hard_topk_fraction", 0.01)),
    )


def build_optimizer(model: FrozenSemanticBridgeSuppressionModel, cfg: dict[str, Any]) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    train_cfg = cfg.get("train") or {}
    lr_context = float(train_cfg.get("lr_context_projection", train_cfg.get("lr", 1.0e-3)))
    lr_head = float(train_cfg.get("lr_bridge_head", train_cfg.get("lr", 1.0e-3)))
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-5))
    context_named = [(name, p) for name, p in model.named_parameters() if name.startswith("context_projection.") and p.requires_grad]
    head_named = [(name, p) for name, p in model.named_parameters() if name.startswith("bridge_head.") and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for _n, p in context_named], "lr": lr_context},
            {"params": [p for _n, p in head_named], "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )
    meta = {
        "context_projection_lr": lr_context,
        "bridge_head_lr": lr_head,
        "weight_decay": weight_decay,
        "context_projection_params": int(sum(int(p.numel()) for _n, p in context_named)),
        "bridge_head_params": int(sum(int(p.numel()) for _n, p in head_named)),
        "total_trainable_params": int(sum(int(p.numel()) for _n, p in context_named + head_named)),
        "parameter_names": [name for name, _p in context_named + head_named],
    }
    return optimizer, meta


def _collect_batchnorm_stats(model: nn.Module) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    out: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for name, module in model.named_modules():
        running_mean = getattr(module, "running_mean", None)
        running_var = getattr(module, "running_var", None)
        if running_mean is None or running_var is None:
            continue
        if torch.is_tensor(running_mean) and torch.is_tensor(running_var):
            out.append((name, running_mean.detach().clone(), running_var.detach().clone()))
    return out


def _max_bn_delta(model: nn.Module, ref: list[tuple[str, torch.Tensor, torch.Tensor]]) -> float:
    modules = dict(model.named_modules())
    max_delta = 0.0
    for name, mean_ref, var_ref in ref:
        module = modules.get(name, None)
        if module is None:
            continue
        running_mean = getattr(module, "running_mean", None)
        running_var = getattr(module, "running_var", None)
        if running_mean is None or running_var is None:
            continue
        d1 = float((running_mean.detach() - mean_ref).abs().max().item()) if running_mean.numel() else 0.0
        d2 = float((running_var.detach() - var_ref).abs().max().item()) if running_var.numel() else 0.0
        max_delta = max(max_delta, d1, d2)
    return float(max_delta)


def _snapshot_named_parameters(named_params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    return {str(name): param.detach().clone() for name, param in named_params}


def _max_parameter_delta_from_snapshot(named_params: list[tuple[str, torch.nn.Parameter]], snap: dict[str, torch.Tensor]) -> float:
    max_delta = 0.0
    for name, param in named_params:
        ref = snap.get(str(name), None)
        if ref is None:
            continue
        delta = float((param.detach() - ref).abs().max().item()) if param.numel() else 0.0
        max_delta = max(max_delta, delta)
    return float(max_delta)


def _named_grad_l2_norm(named_params: list[tuple[str, torch.nn.Parameter]]) -> float:
    s = 0.0
    for _name, param in named_params:
        if param.grad is None:
            continue
        s += float(torch.sum(param.grad.detach().float() ** 2).item())
    return float(math.sqrt(max(s, 0.0)))


def _count_present_grads(named_params: list[tuple[str, torch.nn.Parameter]]) -> int:
    return int(sum(1 for _name, param in named_params if param.grad is not None))


def _all_grads_finite(named_params: list[tuple[str, torch.nn.Parameter]]) -> bool:
    for _name, param in named_params:
        if param.grad is None:
            continue
        if not bool(torch.isfinite(param.grad.detach()).all().item()):
            return False
    return True


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, cfg: dict[str, Any], extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "config": cfg,
            "extra": extra,
        },
        str(path),
    )


def _load_image_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _load_u8(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def _center_crop_like_validation(image: np.ndarray, crop_h: int, crop_w: int, *, is_mask: bool) -> np.ndarray:
    h, w = image.shape[:2]
    if h < crop_h or w < crop_w:
        new_h = max(h, crop_h)
        new_w = max(w, crop_w)
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        image = cv2.resize(image, (new_w, new_h), interpolation=interp)
        h, w = image.shape[:2]
    y0 = (h - crop_h) // 2 if h > crop_h else 0
    x0 = (w - crop_w) // 2 if w > crop_w else 0
    if image.ndim == 2:
        return image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w, :]


def _bridge_input_hw_from_cfg(cfg: dict[str, Any]) -> tuple[int, int]:
    model_cfg = cfg.get("model") or {}
    size = int(model_cfg.get("input_size", 768))
    return int(size), int(size)


def _simple_preprocess_uint8_rgb(image_rgb_u8: np.ndarray) -> np.ndarray:
    return image_rgb_u8.astype(np.float32) / 255.0


def _leaflet_probability_threshold_diagnostics(
    p_leaf: np.ndarray,
    *,
    threshold: float = LOCKED_CANDIDATE_THRESHOLD,
) -> dict[str, Any]:
    delta = np.abs(np.asarray(p_leaf, dtype=np.float32) - np.float32(threshold))
    return {
        "min_abs_leaflet_probability_minus_0_5": float(np.min(delta)) if delta.size else 0.0,
        "count_abs_leaflet_probability_minus_0_5_le_1e_6": int(np.sum(delta <= 1.0e-6)),
        "count_abs_leaflet_probability_minus_0_5_le_1e_5": int(np.sum(delta <= 1.0e-5)),
        "count_abs_leaflet_probability_minus_0_5_le_1e_4": int(np.sum(delta <= 1.0e-4)),
    }


def _build_split_items(cfg: dict[str, Any], split_txt: Path) -> list[dict[str, Any]]:
    dataset_cfg = cfg.get("dataset") or {}
    dataset_root = _resolve_repo_path(dataset_cfg.get("root", DEFAULT_DATASET_ROOT), DEFAULT_DATASET_ROOT)
    instance_root = _resolve_repo_path(dataset_cfg.get("instance_root", DEFAULT_INSTANCE_ROOT), DEFAULT_INSTANCE_ROOT)
    items = read_split_file(dataset_root.resolve(), split_txt.resolve())
    out: list[dict[str, Any]] = []
    for item in items:
        sample_id = Path(item.image_path).stem
        instance_path = instance_root / "instance_masks" / f"{sample_id}.png"
        out.append(
            {
                "sample_id": str(sample_id),
                "patient_id": _make_patient_id(sample_id),
                "image_path": str(item.image_path),
                "mask_path": str(item.mask_path),
                "instance_path": str(instance_path.resolve()),
            }
        )
    return out


def mine_bridge_records_for_split(
    *,
    cfg: dict[str, Any],
    split_txt: Path,
    model: FrozenSemanticBridgeSuppressionModel,
    device: torch.device,
    cache_features: bool,
    selected_sample_ids: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[dict[str, Any]]:
    _assert_safe_path(split_txt)
    items = _build_split_items(cfg, split_txt)
    if selected_sample_ids is not None:
        requested_ids = [str(v) for v in selected_sample_ids]
        requested_set = set(requested_ids)
        by_id = {str(item["sample_id"]): item for item in items if str(item["sample_id"]) in requested_set}
        items = [by_id[sample_id] for sample_id in requested_ids if sample_id in by_id]
        resolved_ids = {str(item["sample_id"]) for item in items}
        missing_ids = sorted(requested_set - resolved_ids)
        if missing_ids:
            raise SystemExit(
                f"Locked micro manifest contains sample IDs not resolvable from split {split_txt.resolve()}: {missing_ids}"
            )
    audit_batch_size = int(((cfg.get("train") or {}).get("audit_batch_size", 2 if device.type != "cuda" else 8)))
    semantic_inference_amp = _semantic_inference_amp_enabled(cfg, device)
    out: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(items), audit_batch_size):
            batch_items = items[start : start + audit_batch_size]
            images_np: list[np.ndarray] = []
            gt_sems: list[np.ndarray] = []
            gt_insts: list[np.ndarray] = []
            crop_h, crop_w = _bridge_input_hw_from_cfg(cfg)
            for item in batch_items:
                image_rgb = _load_image_rgb(Path(item["image_path"]))
                gt_sem_full = _load_u8(Path(item["mask_path"]))
                gt_inst_full = _load_u8(Path(item["instance_path"]))
                image_rgb = _center_crop_like_validation(image_rgb, crop_h, crop_w, is_mask=False)
                gt_sem = _center_crop_like_validation(gt_sem_full, crop_h, crop_w, is_mask=True)
                gt_inst = _center_crop_like_validation(gt_inst_full, crop_h, crop_w, is_mask=True)
                images_np.append(_simple_preprocess_uint8_rgb(image_rgb).transpose(2, 0, 1))
                gt_sems.append(gt_sem.astype(np.uint8))
                gt_insts.append(gt_inst.astype(np.uint8))
            image_batch = torch.from_numpy(np.stack(images_np, axis=0)).float().to(device)
            with _semantic_inference_backend_ctx(cfg, device):
                with _autocast_ctx(device, enabled=semantic_inference_amp):
                    outputs = model(image_batch)
            p_leaf_batch = outputs["p_leaf"].detach().cpu().numpy().astype(np.float32)
            x04_batch = outputs["x_0_4"].detach().cpu().float() if cache_features else None
            x22_batch = outputs["x_2_2"].detach().cpu().float() if cache_features else None
            for idx, item in enumerate(batch_items):
                gt_sem = gt_sems[idx]
                gt_inst = gt_insts[idx]
                p_leaf = p_leaf_batch[idx, 0]
                candidate_mask = candidate_p50_mask_from_probs(p_leaf)
                bridge_target = false_bridge_pixels_from_candidate(
                    gt_sem_u8=gt_sem.astype(np.uint8),
                    gt_inst_u8=gt_inst.astype(np.uint8),
                    candidate_mask01=candidate_mask.astype(np.uint8),
                )
                oracle_removed = ((candidate_mask > 0) & (bridge_target == 0)).astype(np.uint8)
                topo_before = forensic.classify_semantic_topology(gt_inst.astype(np.uint8), candidate_mask.astype(np.uint8))
                topo_after = forensic.classify_semantic_topology(gt_inst.astype(np.uint8), oracle_removed.astype(np.uint8))
                labels, region_count = _connected_components(bridge_target.astype(np.uint8))
                region_areas = [int(np.sum(labels == cc_idx)) for cc_idx in range(1, int(region_count) + 1)]
                gt_count = int(len(topo_aux._positive_instance_ids(gt_inst.astype(np.uint8))))
                reconstructed = run_locked_reconstruction(oracle_removed.astype(np.uint8), gt_inst.astype(np.uint8))
                probability_diag = _leaflet_probability_threshold_diagnostics(p_leaf)
                record = {
                    **item,
                    "gt_count": int(gt_count),
                    "candidate_pixels": int(np.sum(candidate_mask > 0)),
                    "bridge_pixels": int(np.sum(bridge_target > 0)),
                    "bridge_positive": int(np.sum(bridge_target > 0) > 0),
                    "bridge_region_count": int(region_count),
                    "bridge_region_areas": region_areas,
                    "topology_changes_if_oracle_removed": int(
                        str(topo_before["topology_class"]) != str(topo_after["topology_class"])
                        or bool(topo_before["bridge"]) != bool(topo_after["bridge"])
                        or bool(topo_before["missing"]) != bool(topo_after["missing"])
                    ),
                    "candidate_mask": candidate_mask.astype(np.uint8),
                    "bridge_target": bridge_target.astype(np.uint8),
                    "p_leaf": p_leaf.astype(np.float32),
                    "gt_semantic": gt_sem.astype(np.uint8),
                    "gt_instances": gt_inst.astype(np.uint8),
                    "oracle_removed_mask": oracle_removed.astype(np.uint8),
                    "oracle_removed_reconstruction": reconstructed["labels"].astype(np.uint8),
                    "bridge_contract_input_hw": [int(crop_h), int(crop_w)],
                    **probability_diag,
                }
                if cache_features and x04_batch is not None and x22_batch is not None:
                    record["x_0_4"] = x04_batch[idx].clone()
                    record["x_2_2"] = x22_batch[idx].clone()
                out.append(record)
    return out


def summarize_bridge_records(records: list[dict[str, Any]], split_path: Path, *, train_sample_ids: set[str] | None = None, train_patient_ids: set[str] | None = None) -> dict[str, Any]:
    train_sample_ids = train_sample_ids or set()
    train_patient_ids = train_patient_ids or set()
    gt1 = sum(1 for row in records if int(row["gt_count"]) == 1)
    gt2 = sum(1 for row in records if int(row["gt_count"]) == 2)
    gt3 = sum(1 for row in records if int(row["gt_count"]) == 3)
    bridge_positive = [row for row in records if int(row["bridge_positive"]) == 1]
    gt2_positive = sum(1 for row in bridge_positive if int(row["gt_count"]) == 2)
    gt3_positive = sum(1 for row in bridge_positive if int(row["gt_count"]) == 3)
    bridge_pixels_per_positive = [int(row["bridge_pixels"]) for row in bridge_positive]
    bridge_region_areas = [int(area) for row in bridge_positive for area in row["bridge_region_areas"]]
    bridge_region_counts = [int(row["bridge_region_count"]) for row in bridge_positive]
    overlap_samples = sorted({str(row["sample_id"]) for row in records if str(row["sample_id"]) in train_sample_ids})
    overlap_patients = sorted({str(row["patient_id"]) for row in records if str(row["patient_id"]) in train_patient_ids})
    return {
        "split_path": str(split_path.resolve()),
        "sample_count": int(len(records)),
        "patient_count": int(len({str(row["patient_id"]) for row in records})),
        "gt1": int(gt1),
        "gt2": int(gt2),
        "gt3": int(gt3),
        "bridge_positive_samples": int(len(bridge_positive)),
        "gt2_positive_samples": int(gt2_positive),
        "gt3_positive_samples": int(gt3_positive),
        "total_candidate_pixels": int(sum(int(row["candidate_pixels"]) for row in records)),
        "total_bridge_positive_pixels": int(sum(int(row["bridge_pixels"]) for row in records)),
        "bridge_positive_fraction_within_candidate": float(
            sum(int(row["bridge_pixels"]) for row in records) / max(sum(int(row["candidate_pixels"]) for row in records), 1)
        ),
        "bridge_pixels_per_positive_sample": {
            "median": float(np.median(bridge_pixels_per_positive)) if bridge_pixels_per_positive else 0.0,
            "p75": float(np.percentile(bridge_pixels_per_positive, 75)) if bridge_pixels_per_positive else 0.0,
            "p90": float(np.percentile(bridge_pixels_per_positive, 90)) if bridge_pixels_per_positive else 0.0,
            "max": int(max(bridge_pixels_per_positive)) if bridge_pixels_per_positive else 0,
        },
        "connected_bridge_regions_per_positive_sample": {
            "median": float(np.median(bridge_region_counts)) if bridge_region_counts else 0.0,
            "p75": float(np.percentile(bridge_region_counts, 75)) if bridge_region_counts else 0.0,
            "p90": float(np.percentile(bridge_region_counts, 90)) if bridge_region_counts else 0.0,
            "max": int(max(bridge_region_counts)) if bridge_region_counts else 0,
        },
        "bridge_region_area_distribution": {
            "median": float(np.median(bridge_region_areas)) if bridge_region_areas else 0.0,
            "p75": float(np.percentile(bridge_region_areas, 75)) if bridge_region_areas else 0.0,
            "p90": float(np.percentile(bridge_region_areas, 90)) if bridge_region_areas else 0.0,
            "max": int(max(bridge_region_areas)) if bridge_region_areas else 0,
        },
        "topology_changing_bridge_samples": int(sum(int(row["topology_changes_if_oracle_removed"]) for row in bridge_positive)),
        "sample_overlap_with_train": int(len(overlap_samples)),
        "patient_overlap_with_train": int(len(overlap_patients)),
        "overlap_samples": overlap_samples,
        "overlap_patients": overlap_patients,
        "instance_mask_available": int(sum(1 for row in records if Path(row["instance_path"]).exists())),
        "gt_instance_count_available": int(sum(1 for row in records if int(row["gt_count"]) > 0)),
    }


def _select_visual_examples(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    positive = [row for row in records if int(row["bridge_positive"]) == 1]
    negatives = [row for row in records if int(row["bridge_positive"]) == 0 and int(row["gt_count"]) in {2, 3}]

    def _pick(rows: list[dict[str, Any]], key_fn) -> dict[str, Any] | None:
        if not rows:
            return None
        rows = sorted(rows, key=key_fn, reverse=True)
        return rows[0]

    return {
        "gt2_bridge_positive": _pick([r for r in positive if int(r["gt_count"]) == 2], lambda r: int(r["bridge_pixels"])),
        "gt3_bridge_positive": _pick([r for r in positive if int(r["gt_count"]) == 3], lambda r: int(r["bridge_pixels"])),
        "multiple_bridge_regions": _pick(positive, lambda r: (int(r["bridge_region_count"]), int(r["bridge_pixels"]))),
        "thin_bridge": _pick(positive, lambda r: -float(np.median(r["bridge_region_areas"])) if r["bridge_region_areas"] else float("-inf")),
        "broad_bridge": _pick(positive, lambda r: int(max(r["bridge_region_areas"])) if r["bridge_region_areas"] else 0),
        "bridge_negative_gt2_gt3": _pick(negatives, lambda r: int(r["candidate_pixels"])),
        "oracle_changes_failure_to_success": _pick(
            [
                r for r in positive
                if bool(run_locked_reconstruction(r["candidate_mask"], r["gt_instances"])["metrics"]["all_iou_ge_0.50"]) is False
                and bool(run_locked_reconstruction(r["oracle_removed_mask"], r["gt_instances"])["metrics"]["all_iou_ge_0.50"]) is True
            ],
            lambda r: int(r["bridge_pixels"]),
        ),
    }


def save_train_target_visual_audit(records: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    visuals_dir = output_dir / "bridge_target_visual_audit"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    chosen = _select_visual_examples(records)
    out: dict[str, str] = {}
    for label, row in chosen.items():
        if row is None:
            continue
        image_rgb = _center_crop_like_validation(
            _load_image_rgb(Path(row["image_path"])),
            row["gt_semantic"].shape[0],
            row["gt_semantic"].shape[1],
            is_mask=False,
        )
        out[label] = _save_visual_grid(
            out_path=visuals_dir / f"{label}_{row['sample_id']}.png",
            image_rgb=image_rgb,
            gt_inst_u8=row["gt_instances"],
            p_leaf=row["p_leaf"],
            candidate_mask01=row["candidate_mask"],
            bridge_target01=row["bridge_target"],
            candidate_minus_oracle01=row["oracle_removed_mask"],
            reconstructed_u8=row["oracle_removed_reconstruction"],
        )
    return out


def _read_existing_micro_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def read_locked_micro_manifest(path: Path) -> dict[str, Any]:
    payload = _read_existing_micro_manifest(path)
    if payload is None:
        raise SystemExit(f"Locked micro manifest is missing or invalid JSON: {path}")
    sample_ids = payload.get("sample_ids")
    rows = payload.get("rows")
    source_split = payload.get("source_split")
    source_split_sha256 = payload.get("source_split_canonical_sha256")
    if not isinstance(sample_ids, list) or not isinstance(rows, list):
        raise SystemExit(f"Locked micro manifest must contain sample_ids and rows lists: {path}")
    if not isinstance(source_split, str) or not source_split.strip():
        raise SystemExit(f"Locked micro manifest must contain non-empty source_split: {path}")
    if not isinstance(source_split_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_split_sha256.strip().lower()):
        raise SystemExit(f"Locked micro manifest must contain source_split_canonical_sha256 as lowercase hex: {path}")
    row_ids = [str(row.get("sample_id", "")) for row in rows if isinstance(row, dict)]
    if [str(v) for v in sample_ids] != row_ids:
        raise SystemExit(f"Locked micro manifest sample_ids/rows mismatch: {path}")
    return payload


def summarize_manifest_expectations(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in payload.get("rows", []) if isinstance(row, dict)]
    return {
        "expected_sample_count": int(len(rows)),
        "expected_positive_count": int(sum(int(row.get("bridge_positive", 0)) for row in rows)),
        "expected_negative_count": int(sum(1 for row in rows if int(row.get("bridge_positive", 0)) == 0)),
        "expected_gt2_count": int(sum(1 for row in rows if int(row.get("gt_count", 0)) == 2)),
        "expected_gt3_count": int(sum(1 for row in rows if int(row.get("gt_count", 0)) == 3)),
        "sample_ids": [str(row.get("sample_id", "")) for row in rows],
    }


def validate_locked_micro_records(
    *,
    manifest_payload: dict[str, Any],
    records: list[dict[str, Any]],
    split_txt: Path,
) -> dict[str, Any]:
    expected_rows = [row for row in manifest_payload.get("rows", []) if isinstance(row, dict)]
    expected_by_id = {str(row["sample_id"]): row for row in expected_rows}
    actual_by_id = {str(row["sample_id"]): row for row in records}
    expected_ids = [str(row["sample_id"]) for row in expected_rows]
    actual_ids = [str(row["sample_id"]) for row in records]

    summary = {
        "manifest_path": str(_resolve_repo_path(manifest_payload.get("_manifest_path"), MICRO_MANIFEST_V2_PATH)),
        "source_split": str(split_txt.resolve()),
        **summarize_manifest_expectations(manifest_payload),
        "actual_sample_count": int(len(records)),
        "actual_positive_count": int(sum(int(row["bridge_positive"]) for row in records)),
        "actual_negative_count": int(sum(1 for row in records if int(row["bridge_positive"]) == 0)),
        "actual_gt2_count": int(sum(1 for row in records if int(row["gt_count"]) == 2)),
        "actual_gt3_count": int(sum(1 for row in records if int(row["gt_count"]) == 3)),
        "actual_sample_ids": actual_ids,
    }

    missing_ids = [sample_id for sample_id in expected_ids if sample_id not in actual_by_id]
    unexpected_ids = [sample_id for sample_id in actual_ids if sample_id not in expected_by_id]
    mismatched_rows: list[dict[str, Any]] = []
    for sample_id in expected_ids:
        expected = expected_by_id.get(sample_id)
        actual = actual_by_id.get(sample_id)
        if expected is None or actual is None:
            continue
        for key in ("patient_id", "gt_count", "bridge_positive", "bridge_pixels", "candidate_pixels", "topology_changes_if_oracle_removed"):
            if key in expected and str(expected.get(key)) != str(actual.get(key)):
                mismatched_rows.append(
                    {
                        "sample_id": sample_id,
                        "field": key,
                        "expected": expected.get(key),
                        "actual": actual.get(key),
                    }
                )
    summary["missing_ids"] = missing_ids
    summary["unexpected_ids"] = unexpected_ids
    summary["mismatched_rows"] = mismatched_rows
    summary["per_sample_portability_diagnostics"] = [
        {
            "sample_id": str(row["sample_id"]),
            "candidate_pixels": int(row["candidate_pixels"]),
            "bridge_pixels": int(row["bridge_pixels"]),
            "min_abs_leaflet_probability_minus_0_5": float(row.get("min_abs_leaflet_probability_minus_0_5", 0.0)),
            "count_abs_leaflet_probability_minus_0_5_le_1e_6": int(row.get("count_abs_leaflet_probability_minus_0_5_le_1e_6", 0)),
            "count_abs_leaflet_probability_minus_0_5_le_1e_5": int(row.get("count_abs_leaflet_probability_minus_0_5_le_1e_5", 0)),
            "count_abs_leaflet_probability_minus_0_5_le_1e_4": int(row.get("count_abs_leaflet_probability_minus_0_5_le_1e_4", 0)),
        }
        for row in records
    ]
    summary["status"] = "pass"

    if (
        missing_ids
        or unexpected_ids
        or mismatched_rows
        or summary["expected_sample_count"] != summary["actual_sample_count"]
        or summary["expected_positive_count"] != summary["actual_positive_count"]
        or summary["expected_negative_count"] != summary["actual_negative_count"]
        or summary["expected_gt2_count"] != summary["actual_gt2_count"]
        or summary["expected_gt3_count"] != summary["actual_gt3_count"]
        or summary["sample_ids"] != summary["actual_sample_ids"]
    ):
        summary["status"] = "blocked"
    return summary


def _default_select_micro_overfit_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positives = [row for row in records if int(row["bridge_positive"]) == 1]
    negatives = [row for row in records if int(row["bridge_positive"]) == 0]
    positives = sorted(
        positives,
        key=lambda r: (
            int(r["topology_changes_if_oracle_removed"]),
            int(r["gt_count"]),
            int(r["bridge_region_count"]),
            int(r["bridge_pixels"]),
            -ord(str(r["sample_id"])[0]) if str(r["sample_id"]) else 0,
        ),
        reverse=True,
    )
    negatives = sorted(
        negatives,
        key=lambda r: (
            int(r["gt_count"]),
            int(r["candidate_pixels"]),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []

    def _take_one(rows: list[dict[str, Any]], predicate) -> None:
        for row in rows:
            if predicate(row) and all(str(row["sample_id"]) != str(existing["sample_id"]) for existing in selected):
                selected.append(row)
                return

    _take_one(positives, lambda r: int(r["gt_count"]) == 3)
    _take_one(positives, lambda r: int(r["gt_count"]) == 2)
    for row in positives:
        if len([r for r in selected if int(r["bridge_positive"]) == 1]) >= MICROSET_POSITIVE_TARGET:
            break
        if all(str(row["sample_id"]) != str(existing["sample_id"]) for existing in selected):
            selected.append(row)
    _take_one(negatives, lambda r: int(r["gt_count"]) == 3)
    _take_one(negatives, lambda r: int(r["gt_count"]) == 2)
    for row in negatives:
        if len([r for r in selected if int(r["bridge_positive"]) == 0]) >= MICROSET_NEGATIVE_TARGET:
            break
        if all(str(row["sample_id"]) != str(existing["sample_id"]) for existing in selected):
            selected.append(row)
    return sorted(selected, key=lambda r: str(r["sample_id"]))


def select_micro_overfit_records(records: list[dict[str, Any]], *, manifest_path: Path = MICRO_MANIFEST_PATH) -> list[dict[str, Any]]:
    existing = _read_existing_micro_manifest(manifest_path)
    if not existing:
        return _default_select_micro_overfit_records(records)

    rows_by_id = {str(row["sample_id"]): row for row in records}
    selected: list[dict[str, Any]] = []
    taken_ids: set[str] = set()
    needs_positive_replacement = 0

    for row in existing.get("rows", []):
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("sample_id", ""))
        current = rows_by_id.get(sample_id)
        if current is None:
            continue
        previous_positive = int(row.get("bridge_positive", 0)) == 1
        current_positive = int(current["bridge_positive"]) == 1
        if previous_positive and not current_positive:
            needs_positive_replacement += 1
            continue
        selected.append(current)
        taken_ids.add(sample_id)

    if needs_positive_replacement <= 0:
        return selected

    positive_pool = [
        row for row in _default_select_micro_overfit_records(records)
        if int(row["bridge_positive"]) == 1 and str(row["sample_id"]) not in taken_ids
    ]
    for row in positive_pool:
        if needs_positive_replacement <= 0:
            break
        selected.append(row)
        taken_ids.add(str(row["sample_id"]))
        needs_positive_replacement -= 1
    return sorted(selected, key=lambda r: str(r["sample_id"]))


def write_micro_manifest(records: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    payload = {
        "source_split": _repo_relative_canonical_path(DEFAULT_TRAIN_SPLIT),
        "source_split_canonical_sha256": _canonical_split_sha256(DEFAULT_TRAIN_SPLIT),
        "locked_candidate_threshold": float(LOCKED_CANDIDATE_THRESHOLD),
        "sample_ids": [str(row["sample_id"]) for row in records],
        "bridge_positive_count": int(sum(int(row["bridge_positive"]) for row in records)),
        "bridge_negative_count": int(sum(1 for row in records if int(row["bridge_positive"]) == 0)),
        "gt2_count": int(sum(1 for row in records if int(row["gt_count"]) == 2)),
        "gt3_count": int(sum(1 for row in records if int(row["gt_count"]) == 3)),
        "rows": [
            {
                "sample_id": str(row["sample_id"]),
                "gt_count": int(row["gt_count"]),
                "bridge_positive": int(row["bridge_positive"]),
                "bridge_pixels": int(row["bridge_pixels"]),
                "candidate_pixels": int(row["candidate_pixels"]),
                "topology_changes_if_oracle_removed": int(row["topology_changes_if_oracle_removed"]),
            }
            for row in records
        ],
    }
    _write_json(path, payload)
    return payload


def cache_microset_features(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in records:
        candidate_mask_np = row["candidate_mask"].astype(np.uint8)
        oracle_removed_mask = row["oracle_removed_mask"].astype(np.uint8)
        gt_instances = row["gt_instances"].astype(np.uint8)
        start_reconstruction = run_locked_reconstruction(candidate_mask_np, gt_instances)
        oracle_reconstruction = run_locked_reconstruction(oracle_removed_mask, gt_instances)
        out.append(
            {
                "sample_id": str(row["sample_id"]),
                "patient_id": str(row["patient_id"]),
                "gt_count": int(row["gt_count"]),
                "bridge_positive": int(row["bridge_positive"]),
                "candidate_pixels": int(row["candidate_pixels"]),
                "bridge_pixels": int(row["bridge_pixels"]),
                "x_0_4": row["x_0_4"].clone(),
                "x_2_2": row["x_2_2"].clone(),
                "p_leaf": torch.from_numpy(row["p_leaf"][None, ...]).float(),
                "candidate_mask": torch.from_numpy(candidate_mask_np[None, ...].astype(np.float32)),
                "bridge_target": torch.from_numpy(row["bridge_target"][None, ...].astype(np.float32)),
                "gt_instances": gt_instances,
                "candidate_mask_np": candidate_mask_np,
                "oracle_removed_mask": oracle_removed_mask,
                "component_count_start": int(_connected_components(candidate_mask_np)[1]),
                "start_reconstruction": start_reconstruction,
                "oracle_reconstruction": oracle_reconstruction,
                "image_path": str(row["image_path"]),
            }
        )
    return out


def stack_cached_batch(records: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    return {
        "x_0_4": torch.stack([row["x_0_4"] for row in records], dim=0).to(device),
        "x_2_2": torch.stack([row["x_2_2"] for row in records], dim=0).to(device),
        "p_leaf": torch.stack([row["p_leaf"] for row in records], dim=0).to(device),
        "candidate_mask": torch.stack([row["candidate_mask"] for row in records], dim=0).to(device),
        "bridge_target": torch.stack([row["bridge_target"] for row in records], dim=0).to(device),
        "bridge_positive": torch.tensor([int(row["bridge_positive"]) for row in records], device=device, dtype=torch.float32),
        "sample_ids": [str(row["sample_id"]) for row in records],
    }


def compute_binary_metrics_from_domain(
    *,
    bridge_probs: torch.Tensor,
    bridge_target: torch.Tensor,
    candidate_mask: torch.Tensor,
    threshold: float = BRIDGE_REMOVE_THRESHOLD,
) -> dict[str, Any]:
    domain = candidate_mask > 0.5
    pred = (bridge_probs >= float(threshold)) & domain
    target = (bridge_target > 0.5) & domain
    tp = int(torch.sum(pred & target).item())
    fp = int(torch.sum(pred & (~target)).item())
    fn = int(torch.sum((~pred) & target).item())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float((2 * tp) / max(2 * tp + fp + fn, 1))
    dice = f1
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": dice,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_binary_metrics_for_subset(
    *,
    bridge_probs: torch.Tensor,
    bridge_target: torch.Tensor,
    candidate_mask: torch.Tensor,
    subset_indices: list[int],
    threshold: float = BRIDGE_REMOVE_THRESHOLD,
) -> dict[str, Any]:
    if not subset_indices:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "dice": 0.0, "tp": 0, "fp": 0, "fn": 0}
    idx = torch.as_tensor(subset_indices, device=bridge_probs.device, dtype=torch.long)
    return compute_binary_metrics_from_domain(
        bridge_probs=bridge_probs.index_select(0, idx),
        bridge_target=bridge_target.index_select(0, idx),
        candidate_mask=candidate_mask.index_select(0, idx),
        threshold=threshold,
    )


def _subset_reconstruction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = int(len(rows))
    success50 = int(sum(int(bool(row["all_iou_ge_0.50"])) for row in rows))
    gt2_n = sum(1 for row in rows if int(row["gt_count"]) == 2)
    gt3_n = sum(1 for row in rows if int(row["gt_count"]) == 3)
    return {
        "n": n,
        "mean_matched_iou": float(np.mean([float(row["instance_mean_matched_iou"]) for row in rows])) if rows else 0.0,
        "all_iou_ge_0.50_count": success50,
        "all_iou_ge_0.50_rate": float(success50 / max(n, 1)),
        "gt2_success": f"{sum(int(bool(row['all_iou_ge_0.50'])) for row in rows if int(row['gt_count']) == 2)}/{gt2_n}",
        "gt3_success": f"{sum(int(bool(row['all_iou_ge_0.50'])) for row in rows if int(row['gt_count']) == 3)}/{gt3_n}",
    }


def _ensure_cached_reconstruction_invariants(row: dict[str, Any]) -> None:
    if "component_count_start" not in row:
        row["component_count_start"] = int(_connected_components(row["candidate_mask_np"].astype(np.uint8))[1])
    if "start_reconstruction" not in row:
        row["start_reconstruction"] = run_locked_reconstruction(row["candidate_mask_np"].astype(np.uint8), row["gt_instances"].astype(np.uint8))
    if "oracle_reconstruction" not in row:
        row["oracle_reconstruction"] = run_locked_reconstruction(row["oracle_removed_mask"].astype(np.uint8), row["gt_instances"].astype(np.uint8))


def evaluate_reconstruction_levels_on_cached(
    model: FrozenSemanticBridgeSuppressionModel,
    records: list[dict[str, Any]],
    device: torch.device,
    *,
    threshold: float = BRIDGE_REMOVE_THRESHOLD,
) -> dict[str, Any]:
    model.eval()
    t_eval_start = time.perf_counter()
    batch = stack_cached_batch(records, device)
    t_forward_start = time.perf_counter()
    with torch.no_grad():
        outputs = model.bridge_forward_from_cached(
            x_0_4=batch["x_0_4"],
            x_2_2=batch["x_2_2"],
            p_leaf=batch["p_leaf"],
        )
        bridge_probs = torch.sigmoid(outputs["bridge_logits"]).detach().cpu()
    t_forward_end = time.perf_counter()
    threshold_seconds = 0.0
    reconstruction_seconds = 0.0
    metrics_seconds = 0.0
    diagnostics_seconds = 0.0
    predicted_metrics: list[dict[str, Any]] = []
    start_metrics: list[dict[str, Any]] = []
    oracle_metrics: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    total_removed_pixels = 0
    total_candidate_pixels = 0
    positive_removed_pixels = 0
    positive_candidate_pixels = 0
    negative_removed_pixels = 0
    negative_candidate_pixels = 0
    positive_gt_bridge_pixels = 0
    for idx, row in enumerate(records):
        _ensure_cached_reconstruction_invariants(row)
        candidate_mask = row["candidate_mask_np"].astype(np.uint8)
        t_threshold_start = time.perf_counter()
        pred_remove = (bridge_probs[idx, 0].numpy() >= float(threshold)).astype(np.uint8)
        refined = ((candidate_mask > 0) & (pred_remove == 0)).astype(np.uint8)
        t_threshold_end = time.perf_counter()
        threshold_seconds += float(t_threshold_end - t_threshold_start)
        predicted_removed_pixels = int(np.sum((candidate_mask > 0) & (pred_remove > 0)))
        candidate_pixels = int(row["candidate_pixels"])
        total_removed_pixels += predicted_removed_pixels
        total_candidate_pixels += candidate_pixels
        t_recon_start = time.perf_counter()
        comp_before = int(row["component_count_start"])
        comp_after = _connected_components(refined.astype(np.uint8))[1]
        pred_timed = run_locked_reconstruction_with_timing(refined, row["gt_instances"])
        pred = pred_timed["result"]
        start = row["start_reconstruction"]
        oracle = row["oracle_reconstruction"]
        t_recon_end = time.perf_counter()
        reconstruction_seconds += float(t_recon_end - t_recon_start)
        start_metrics.append({"sample_id": row["sample_id"], "gt_count": row["gt_count"], **start["metrics"]})
        predicted_metrics.append({"sample_id": row["sample_id"], "gt_count": row["gt_count"], **pred["metrics"]})
        oracle_metrics.append({"sample_id": row["sample_id"], "gt_count": row["gt_count"], **oracle["metrics"]})
        if int(row["bridge_positive"]) == 1:
            positive_indices.append(idx)
            positive_removed_pixels += predicted_removed_pixels
            positive_candidate_pixels += candidate_pixels
            positive_gt_bridge_pixels += int(row["bridge_pixels"])
        else:
            negative_indices.append(idx)
            negative_removed_pixels += predicted_removed_pixels
            negative_candidate_pixels += candidate_pixels
        t_diag_start = time.perf_counter()
        per_sample.append(
            {
                "sample_id": str(row["sample_id"]),
                "bridge_positive": int(row["bridge_positive"]),
                "gt_count": int(row["gt_count"]),
                "candidate_pixels": candidate_pixels,
                "gt_bridge_pixels": int(row["bridge_pixels"]),
                "predicted_removed_pixels": predicted_removed_pixels,
                "predicted_removed_fraction": float(predicted_removed_pixels / max(candidate_pixels, 1)),
                "start_mean_iou": float(start["metrics"]["instance_mean_matched_iou"]),
                "predicted_mean_iou": float(pred["metrics"]["instance_mean_matched_iou"]),
                "oracle_mean_iou": float(oracle["metrics"]["instance_mean_matched_iou"]),
                "start_success50": int(bool(start["metrics"]["all_iou_ge_0.50"])),
                "predicted_success50": int(bool(pred["metrics"]["all_iou_ge_0.50"])),
                "oracle_success50": int(bool(oracle["metrics"]["all_iou_ge_0.50"])),
                "component_count_start": int(comp_before),
                "component_count_predicted": int(comp_after),
                "component_topology_changed": int(comp_before != comp_after),
                "predicted_reconstruction_runtime_seconds": float(pred_timed["timing"]["total_seconds"]),
                "predicted_normalization_runtime_seconds": float(pred_timed["timing"]["normalization_seconds"]),
                "predicted_metric_runtime_seconds": float(pred_timed["timing"]["metrics_seconds"]),
                "predicted_topology_runtime_seconds": float(pred_timed["timing"]["topology_seconds"]),
            }
        )
        diagnostics_seconds += float(time.perf_counter() - t_diag_start)

    t_metrics_start = time.perf_counter()
    binary = compute_binary_metrics_from_domain(
        bridge_probs=bridge_probs,
        bridge_target=batch["bridge_target"].cpu(),
        candidate_mask=batch["candidate_mask"].cpu(),
        threshold=threshold,
    )
    positive_pixel = compute_binary_metrics_for_subset(
        bridge_probs=bridge_probs,
        bridge_target=batch["bridge_target"].cpu(),
        candidate_mask=batch["candidate_mask"].cpu(),
        subset_indices=positive_indices,
        threshold=threshold,
    )
    negative_rows = [row for row in per_sample if int(row["bridge_positive"]) == 0]
    positive_start = [row for row in start_metrics if str(row["sample_id"]) in {str(v["sample_id"]) for v in per_sample if int(v["bridge_positive"]) == 1}]
    positive_pred = [row for row in predicted_metrics if str(row["sample_id"]) in {str(v["sample_id"]) for v in per_sample if int(v["bridge_positive"]) == 1}]
    positive_oracle = [row for row in oracle_metrics if str(row["sample_id"]) in {str(v["sample_id"]) for v in per_sample if int(v["bridge_positive"]) == 1}]
    metrics_seconds += float(time.perf_counter() - t_metrics_start)
    out = {
        "pixel": binary,
        "positive_subset": {
            "pixel": positive_pixel,
            "reconstruction": {
                "p50_start": _subset_reconstruction_summary(positive_start),
                "p50_minus_predicted_bridge": _subset_reconstruction_summary(positive_pred),
                "p50_minus_gt_oracle_bridge": _subset_reconstruction_summary(positive_oracle),
            },
        },
        "negative_subset": {
            "predicted_bridge_pixels": int(sum(int(row["predicted_removed_pixels"]) for row in negative_rows)),
            "fraction_of_candidate_pixels_removed": float(
                sum(int(row["predicted_removed_pixels"]) for row in negative_rows)
                / max(sum(int(row["candidate_pixels"]) for row in negative_rows), 1)
            ),
            "samples_with_zero_predicted_removal": int(sum(1 for row in negative_rows if int(row["predicted_removed_pixels"]) == 0)),
            "starting_mean_matched_iou": float(np.mean([float(row["start_mean_iou"]) for row in negative_rows])) if negative_rows else 0.0,
            "refined_mean_matched_iou": float(np.mean([float(row["predicted_mean_iou"]) for row in negative_rows])) if negative_rows else 0.0,
            "num_improves": int(sum(1 for row in negative_rows if float(row["predicted_mean_iou"]) > float(row["start_mean_iou"]) + 1.0e-9)),
            "num_unchanged": int(sum(1 for row in negative_rows if abs(float(row["predicted_mean_iou"]) - float(row["start_mean_iou"])) <= 1.0e-9)),
            "num_regresses": int(sum(1 for row in negative_rows if float(row["predicted_mean_iou"]) + 1.0e-9 < float(row["start_mean_iou"]))),
            "num_component_topology_changes": int(sum(int(row["component_topology_changed"]) for row in negative_rows)),
        },
        "removal_calibration": {
            "all_removed_over_candidate": float(total_removed_pixels / max(total_candidate_pixels, 1)),
            "positive_removed_over_candidate": float(positive_removed_pixels / max(positive_candidate_pixels, 1)),
            "negative_removed_over_candidate": float(negative_removed_pixels / max(negative_candidate_pixels, 1)),
            "positive_gt_bridge_over_candidate": float(positive_gt_bridge_pixels / max(positive_candidate_pixels, 1)),
        },
        "reconstruction": {
            "p50_start": _subset_reconstruction_summary(start_metrics),
            "p50_minus_predicted_bridge": _subset_reconstruction_summary(predicted_metrics),
            "p50_minus_gt_oracle_bridge": _subset_reconstruction_summary(oracle_metrics),
        },
        "per_sample": per_sample,
    }
    output_timing = {
        "forward_seconds": float(t_forward_end - t_forward_start),
        "threshold_seconds": float(threshold_seconds),
        "reconstruction_seconds": float(reconstruction_seconds),
        "metrics_seconds": float(metrics_seconds),
        "diagnostics_seconds": float(diagnostics_seconds),
        "total_seconds": float(time.perf_counter() - t_eval_start),
        "reconstruction_per_sample_seconds": {
            str(row["sample_id"]): float(row["predicted_reconstruction_runtime_seconds"])
            for row in per_sample
        },
    }
    out["timing"] = output_timing
    return out


def build_validation_audit(
    *,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    val_split: Path,
) -> dict[str, Any]:
    train_samples = {str(row["sample_id"]) for row in train_records}
    train_patients = {str(row["patient_id"]) for row in train_records}
    summary = summarize_bridge_records(val_records, val_split, train_sample_ids=train_samples, train_patient_ids=train_patients)
    summary["verdict"] = "valid_for_bridge_head_development" if int(summary["sample_overlap_with_train"]) == 0 else "blocked"
    return summary
