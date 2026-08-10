from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from diagnose_full_dataset_center_baseline import (
    _align_instance_mask,
    _build_loader,
    _heatmap_margin,
    _load_checkpoint_model,
    _sample_sort_key,
    _sha256_file,
    _write_csv,
    _write_json,
    _write_text,
)
from train_centerhead import _read_yaml
from validate_centerhead import (
    _extract_metadata_centers,
    _marker_contract,
    _markers_from_center_map,
    _patient_id_from_sample,
    _sample_center_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "training" / "runs" / "unetpp_effb3_centerhead_x2_2_adapter_full_dataset_aug_x2_2_unfreeze_100ep"
DEFAULT_CONFIG_PATH = REPO_ROOT / "training" / "configs" / "unetpp_effb3_centerhead_x2_2_adapter_full_dataset_aug_x2_2_unfreeze_100ep.yaml"
DEFAULT_CHECKPOINT_PATH = DEFAULT_RUN_DIR / "best_primary.pth"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "training" / "analysis" / "center_x2_2_marker_extraction_audit"
DEFAULT_TRAIN_MANIFEST = REPO_ROOT / "training" / "manifests" / "center_full_train_manifest.jsonl"
DEFAULT_VAL_MANIFEST = REPO_ROOT / "training" / "manifests" / "center_full_val_manifest.jsonl"
TOPOLOGY_THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)
MAX_MARKERS = 3


def _load_cfg(*, checkpoint_path: Path, run_dir: Path | None, config_path: Path) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_cfg = ckpt.get("config")
    if isinstance(ckpt_cfg, dict):
        return dict(ckpt_cfg)
    if run_dir is not None:
        config_json = run_dir / "config.json"
        if config_json.exists():
            return json.loads(config_json.read_text(encoding="utf-8"))
    return _read_yaml(config_path)


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):.6f}"


def _round_float(x: float | None) -> float | None:
    if x is None:
        return None
    return float(round(float(x), 6))


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _leaf_union_from_semantic(pred_sem: np.ndarray) -> np.ndarray:
    return (pred_sem.astype(np.uint8) == 1)


def _component_centroid(ys: np.ndarray, xs: np.ndarray) -> tuple[float, float]:
    return float(np.mean(ys.astype(np.float64))), float(np.mean(xs.astype(np.float64)))


def _component_weighted_centroid(
    ys: np.ndarray,
    xs: np.ndarray,
    vals: np.ndarray,
) -> tuple[float, float]:
    weights = vals.astype(np.float64)
    total = float(np.sum(weights))
    if total <= 0.0:
        return _component_centroid(ys, xs)
    y = float(np.sum(ys.astype(np.float64) * weights) / total)
    x = float(np.sum(xs.astype(np.float64) * weights) / total)
    return y, x


def _snap_to_component_pixel(
    ys: np.ndarray,
    xs: np.ndarray,
    *,
    target_y: float,
    target_x: float,
) -> tuple[int, int]:
    dy = ys.astype(np.float64) - float(target_y)
    dx = xs.astype(np.float64) - float(target_x)
    d2 = dy * dy + dx * dx
    best = int(np.argmin(d2))
    return int(ys[best]), int(xs[best])


def _connected_components_default(mask01: np.ndarray) -> tuple[int, np.ndarray]:
    return cv2.connectedComponents(mask01.astype(np.uint8))


def _component_stats(
    center_prob: np.ndarray,
    leaf_union: np.ndarray,
    gt_inst: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    masked = center_prob.astype(np.float32).copy()
    masked[~leaf_union.astype(bool)] = 0.0
    binary = (masked >= float(threshold)).astype(np.uint8)
    if int(binary.sum()) == 0:
        return {
            "thresholded_mask_sum": 0,
            "binary_mask": binary,
            "labels": np.zeros_like(binary, dtype=np.int32),
            "components": [],
        }
    n, labels = _connected_components_default(binary)
    components: list[dict[str, Any]] = []
    for lab in range(1, int(n)):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        vals = masked[ys, xs]
        k = int(np.argmax(vals))
        argmax_y = int(ys[k])
        argmax_x = int(xs[k])
        centroid_y, centroid_x = _component_centroid(ys, xs)
        weighted_y, weighted_x = _component_weighted_centroid(ys, xs, vals)
        weighted_y_i, weighted_x_i = _snap_to_component_pixel(
            ys,
            xs,
            target_y=weighted_y,
            target_x=weighted_x,
        )
        gt_ids = sorted(int(v) for v in np.unique(gt_inst[ys, xs]) if int(v) > 0)
        components.append(
            {
                "label": int(lab),
                "area": int(ys.size),
                "max_score": float(masked[argmax_y, argmax_x]),
                "centroid_y": float(centroid_y),
                "centroid_x": float(centroid_x),
                "argmax_y": int(argmax_y),
                "argmax_x": int(argmax_x),
                "weighted_centroid_y": float(weighted_y),
                "weighted_centroid_x": float(weighted_x),
                "weighted_centroid_snapped_y": int(weighted_y_i),
                "weighted_centroid_snapped_x": int(weighted_x_i),
                "gt_instance_ids": gt_ids,
                "intersects_any_gt": bool(gt_ids),
            }
        )
    components.sort(key=lambda row: (-float(row["max_score"]), int(row["argmax_y"]), int(row["argmax_x"])))
    return {
        "thresholded_mask_sum": int(binary.sum()),
        "binary_mask": binary,
        "labels": labels.astype(np.int32),
        "components": components,
    }


def _current_policy_markers(
    center_prob: np.ndarray,
    leaf_union: np.ndarray,
    *,
    threshold: float,
    max_markers: int = MAX_MARKERS,
) -> list[tuple[int, int, float]]:
    return list(_markers_from_center_map(center_prob, leaf_union, float(threshold), max_markers=int(max_markers)))


def _representative_markers_from_components(
    components: list[dict[str, Any]],
    *,
    policy: str,
    max_markers: int | None,
) -> list[tuple[int, int, float]]:
    pts: list[tuple[int, int, float]] = []
    for row in components:
        score = float(row["max_score"])
        if policy == "current":
            y = int(row["argmax_y"])
            x = int(row["argmax_x"])
        elif policy == "component_argmax":
            y = int(row["argmax_y"])
            x = int(row["argmax_x"])
        elif policy == "weighted_centroid":
            y = int(row["weighted_centroid_snapped_y"])
            x = int(row["weighted_centroid_snapped_x"])
        else:
            raise ValueError(f"Unknown representative policy: {policy}")
        pts.append((y, x, score))
    pts.sort(key=lambda item: (-float(item[2]), int(item[0]), int(item[1])))
    if max_markers is None:
        return pts
    return pts[: int(max_markers)]


def _max_pool_nms_markers(
    center_prob: np.ndarray,
    leaf_union: np.ndarray,
    *,
    threshold: float,
    kernel: int,
    max_markers: int | None,
) -> list[tuple[int, int, float]]:
    if int(kernel) % 2 == 0 or int(kernel) <= 0:
        raise ValueError(f"kernel must be a positive odd integer, got {kernel}")
    masked = center_prob.astype(np.float32).copy()
    masked[~leaf_union.astype(bool)] = 0.0
    maxima_base = (masked >= float(threshold))
    if not bool(np.any(maxima_base)):
        return []
    kernel_arr = np.ones((int(kernel), int(kernel)), dtype=np.uint8)
    dilated = cv2.dilate(masked, kernel_arr)
    maxima = maxima_base & (masked >= (dilated - 1e-12))
    if not bool(np.any(maxima)):
        return []
    n, labels = _connected_components_default(maxima.astype(np.uint8))
    pts: list[tuple[int, int, float]] = []
    for lab in range(1, int(n)):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        vals = masked[ys, xs]
        k = int(np.argmax(vals))
        y = int(ys[k])
        x = int(xs[k])
        pts.append((y, x, float(masked[y, x])))
    pts.sort(key=lambda item: (-float(item[2]), int(item[0]), int(item[1])))
    if max_markers is None:
        return pts
    return pts[: int(max_markers)]


def _split_from_previous(
    prev_labels: np.ndarray | None,
    curr_labels: np.ndarray,
) -> dict[str, Any]:
    if prev_labels is None:
        return {
            "split_from_previous_threshold": False,
            "split_parent_component_count": 0,
            "split_child_component_count": 0,
            "raw_count_increase_vs_previous_threshold": False,
        }
    split_parents = 0
    split_children: set[int] = set()
    prev_raw = int(prev_labels.max())
    curr_raw = int(curr_labels.max())
    for lab in range(1, prev_raw + 1):
        overlap = curr_labels[prev_labels == lab]
        child_labels = {int(v) for v in np.unique(overlap) if int(v) > 0}
        if len(child_labels) > 1:
            split_parents += 1
            split_children.update(child_labels)
    return {
        "split_from_previous_threshold": bool(split_parents > 0),
        "split_parent_component_count": int(split_parents),
        "split_child_component_count": int(len(split_children)),
        "raw_count_increase_vs_previous_threshold": bool(curr_raw > prev_raw),
    }


def _raw_and_capped_count_rows(
    rows: list[dict[str, Any]],
    *,
    split_name: str,
    threshold: float,
) -> list[dict[str, Any]]:
    out = []
    for gt_group in ("all", "1", "2", "3"):
        if gt_group == "all":
            subset = list(rows)
        else:
            subset = [row for row in rows if int(row["gt_instance_count"]) == int(gt_group)]
        raw_counts = [int(row["raw_component_count"]) for row in subset]
        capped_counts = [min(int(v), int(MAX_MARKERS)) for v in raw_counts]
        out.append(
            {
                "split": str(split_name),
                "threshold": float(threshold),
                "gt_group": str(gt_group),
                "sample_count": int(len(subset)),
                "raw_count_histogram_json": _json_dumps(dict(sorted(Counter(raw_counts).items()))),
                "capped_count_histogram_json": _json_dumps(dict(sorted(Counter(capped_counts).items()))),
                "fraction_raw_count_gt_3": float(np.mean([1.0 if int(v) > int(MAX_MARKERS) else 0.0 for v in raw_counts])) if raw_counts else None,
                "raw_count_mean": float(np.mean(raw_counts)) if raw_counts else None,
                "raw_count_median": float(np.median(np.asarray(raw_counts, dtype=np.float64))) if raw_counts else None,
            }
        )
    return out


def _evaluate_rows(
    rows: list[dict[str, Any]],
    *,
    split_name: str,
    threshold: float,
    policy_name: str,
    cap_name: str,
) -> dict[str, Any]:
    tp = int(sum(int(row["tp"]) for row in rows))
    fp = int(sum(int(row["fp"]) for row in rows))
    fn = int(sum(int(row["fn"]) for row in rows))
    match_distances = [float(d) for row in rows for d in row["match_distances"]]
    per_sample_loc = [float(row["center_loc_err_px"]) for row in rows]
    predicted_counts = [int(row["predicted_count"]) for row in rows]
    gt_counts = [int(row["gt_instance_count"]) for row in rows]
    center_precision = float(tp / max(tp + fp, 1))
    center_recall = float(tp / max(tp + fn, 1))
    center_f1 = float((2.0 * center_precision * center_recall) / max(center_precision + center_recall, 1e-7))
    return {
        "split": str(split_name),
        "threshold": float(threshold),
        "policy": str(policy_name),
        "cap_policy": str(cap_name),
        "sample_count": int(len(rows)),
        "center_precision": center_precision,
        "center_recall": center_recall,
        "center_f1": center_f1,
        "center_f1_mean_samples": float(np.mean([float(row["center_f1"]) for row in rows])) if rows else None,
        "strict_marker_contract_pass_rate": float(np.mean([1.0 if bool(row["marker_contract_pass"]) else 0.0 for row in rows])) if rows else None,
        "exact_center_count_accuracy": float(np.mean([1.0 if int(row["predicted_count"]) == int(row["gt_instance_count"]) else 0.0 for row in rows])) if rows else None,
        "localization_error_px_pooled_matches": float(np.mean(np.asarray(match_distances, dtype=np.float64))) if match_distances else None,
        "localization_error_px_mean_samples": float(np.mean(np.asarray(per_sample_loc, dtype=np.float64))) if per_sample_loc else None,
        "missing_gt_markers": int(sum(int(row["missing_gt_instance_markers"]) for row in rows)),
        "duplicate_markers": int(sum(int(row["multiple_markers_inside_gt_instances"]) for row in rows)),
        "markers_outside_instances": int(sum(int(row["markers_outside_all_gt_instances"]) for row in rows)),
        "fraction_count_3": float(np.mean([1.0 if int(v) == int(MAX_MARKERS) else 0.0 for v in predicted_counts])) if predicted_counts else None,
        "fraction_raw_count_gt_3": float(np.mean([1.0 if int(row["raw_predicted_count"]) > int(MAX_MARKERS) else 0.0 for row in rows])) if rows else None,
        "predicted_count_histogram_json": _json_dumps(dict(sorted(Counter(predicted_counts).items()))),
        "gt_count_histogram_json": _json_dumps(dict(sorted(Counter(gt_counts).items()))),
    }


def _per_sample_eval_row(
    *,
    split_name: str,
    threshold: float,
    sample_context: dict[str, Any],
    gt_inst: np.ndarray,
    gt_pts: list[tuple[int, int]],
    pred_pts_scored: list[tuple[int, int, float]],
    raw_predicted_count: int,
) -> dict[str, Any]:
    pred_pts = [(int(y), int(x)) for (y, x, _score) in pred_pts_scored]
    center_metrics = _sample_center_metrics(pred_pts, gt_pts)
    tp, fp, fn = int(center_metrics["tp"]), int(center_metrics["fp"]), int(center_metrics["fn"])
    match_distances = []
    if pred_pts and gt_pts:
        used_gt: set[int] = set()
        for py, px in pred_pts:
            best_idx = None
            best_d = None
            for gi, (gy, gx) in enumerate(gt_pts):
                if gi in used_gt:
                    continue
                d = float(np.hypot(float(py - gy), float(px - gx)))
                if best_d is None or float(d) < float(best_d):
                    best_idx = int(gi)
                    best_d = float(d)
            if best_idx is not None and best_d is not None and float(best_d) <= 16.0:
                used_gt.add(int(best_idx))
                match_distances.append(float(best_d))
    contract = _marker_contract(gt_inst, pred_pts)
    return {
        "split": str(split_name),
        "sample": str(sample_context["sample"]),
        "patient_id": str(sample_context["patient_id"]),
        "gt_instance_count": int(sample_context["gt_instance_count"]),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "center_f1": float(center_metrics["center_f1"]),
        "center_loc_err_px": float(center_metrics["center_loc_err_px"]),
        "predicted_count": int(center_metrics["predicted_center_count"]),
        "raw_predicted_count": int(raw_predicted_count),
        "marker_contract_pass": bool(contract["marker_contract_pass"]),
        "missing_gt_instance_markers": int(contract["missing_gt_instance_markers"]),
        "multiple_markers_inside_gt_instances": int(contract["multiple_markers_inside_gt_instances"]),
        "markers_outside_all_gt_instances": int(contract["markers_outside_all_gt_instances"]),
        "match_distances": match_distances,
    }


def _aggregate_topology_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_counts = [int(row["raw_component_count"]) for row in rows]
    split_rows = [row for row in rows if bool(row["split_from_previous_threshold"])]
    return {
        "sample_count": int(len(rows)),
        "raw_component_count_mean": float(np.mean(raw_counts)) if raw_counts else None,
        "raw_component_count_median": float(np.median(np.asarray(raw_counts, dtype=np.float64))) if raw_counts else None,
        "fraction_samples_raw_count_gt_3": float(np.mean([1.0 if int(v) > int(MAX_MARKERS) else 0.0 for v in raw_counts])) if raw_counts else None,
        "samples_with_split_from_previous_threshold": int(len(split_rows)),
        "fraction_samples_with_split_from_previous_threshold": float(len(split_rows) / max(len(rows), 1)),
        "total_split_parent_components": int(sum(int(row["split_parent_component_count"]) for row in rows)),
        "total_split_child_components": int(sum(int(row["split_child_component_count"]) for row in rows)),
        "samples_with_raw_count_increase_vs_previous_threshold": int(sum(1 for row in rows if bool(row["raw_count_increase_vs_previous_threshold"]))),
        "mean_components_outside_all_gt_instances": float(np.mean([float(row["components_outside_all_gt_instances"]) for row in rows])) if rows else None,
        "mean_gt_instances_intersected_by_multiple_components": float(np.mean([float(row["gt_instances_intersected_by_multiple_components"]) for row in rows])) if rows else None,
    }


def _best_row(rows: list[dict[str, Any]], *, metric: str) -> dict[str, Any]:
    best = None
    best_key = None
    for row in rows:
        key = (
            float(row.get(metric) if row.get(metric) is not None else -float("inf")),
            float(row.get("strict_marker_contract_pass_rate") if row.get("strict_marker_contract_pass_rate") is not None else -float("inf")),
            float(row.get("exact_center_count_accuracy") if row.get("exact_center_count_accuracy") is not None else -float("inf")),
            -float(row.get("localization_error_px_pooled_matches") if row.get("localization_error_px_pooled_matches") is not None else float("inf")),
            -float(row.get("threshold", 0.0)),
        )
        if best_key is None or key > best_key:
            best = row
            best_key = key
    if best is None:
        raise RuntimeError(f"No rows available for metric {metric}")
    return best


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _current_extraction_contract(cfg: dict, checkpoint_path: Path) -> dict[str, Any]:
    center_feature_cfg = dict(((cfg.get("model") or {}).get("center_feature")) or {})
    input_size = int((cfg.get("model") or {}).get("input_size", 0))
    native_stride = int(center_feature_cfg.get("native_stride", 1) or 1)
    upsample_logits = bool(center_feature_cfg.get("upsample_logits_to_target", False))
    feature_h = int(input_size // native_stride) if native_stride > 0 else None
    feature_w = int(input_size // native_stride) if native_stride > 0 else None
    final_h = int(input_size) if upsample_logits else feature_h
    final_w = int(input_size) if upsample_logits else feature_w
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "validation_function": "validate_centerhead._markers_from_center_map",
        "diagnosis_function": "validate_centerhead._markers_from_center_map",
        "validation_and_diagnosis_use_same_extraction_code": True,
        "thresholding_creates_binary_mask": True,
        "thresholding_expression": "m = (c >= float(thr)).astype(np.uint8)",
        "connected_components_used": True,
        "connected_components_call": "cv2.connectedComponents(m)",
        "connected_components_connectivity_argument": None,
        "representative_policy": "component_argmax",
        "representative_expression": "k = int(np.argmax(vals)); y = int(ys[k]); x = int(xs[k])",
        "nms_used_in_current_validation_path": False,
        "component_sorting": "descending component representative score",
        "max_markers_applied_after_component_representative_extraction_and_sorting": True,
        "max_markers_value": int(MAX_MARKERS),
        "coordinate_convention": "row_col_yx",
        "center_feature_module_path": center_feature_cfg.get("module_path"),
        "center_feature_native_stride": int(native_stride),
        "center_feature_resolution_hw_before_upsample": [int(feature_h), int(feature_w)] if feature_h is not None and feature_w is not None else None,
        "center_heatmap_resolution_hw_after_model_forward": [int(final_h), int(final_w)] if final_h is not None and final_w is not None else None,
        "conversion_from_heatmap_to_768_image_coordinates": "identity_mapping_after_model_upsample_to_target",
        "trace": {
            "model_forward_path": [
                "models_centerhead.UnetPlusPlusSemanticCenterHead.forward",
                "models_centerhead.UnetPlusPlusSemanticCenterHead.forward_base",
                "models_centerhead.UnetPlusPlusSemanticCenterHead.forward_center",
                "models_centerhead.UnetPlusPlusSemanticCenterHead.resolve_center_features",
                "models_centerhead.UnetPlusPlusSemanticCenterHead.forward_center_from_features",
                "models_centerhead.UnetPlusPlusSemanticCenterHead.upsample_center_logits",
            ],
            "validation_metric_path": [
                "validate_centerhead.validate_centerhead",
                "validate_centerhead._markers_from_center_map",
                "validate_centerhead._sample_center_metrics",
                "validate_centerhead._marker_contract",
            ],
            "training_best_checkpoint_metric_path": [
                "train_centerhead._threshold_sweep",
                "validate_centerhead.validate_centerhead",
                "train_centerhead best_checkpoint_metadata best_threshold_metrics",
            ],
        },
    }


def _make_visual_panel(
    *,
    image_path: Path,
    instance_mask_path: Path,
    center_prob: np.ndarray,
    gt_pts: list[tuple[int, int]],
    markers_a: list[tuple[int, int]],
    markers_b: list[tuple[int, int]],
    title: str,
    subtitle_a: str,
    subtitle_b: str,
    out_path: Path,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    gt_inst = _align_instance_mask(instance_mask_path, target_hw=image.shape[:2])
    heat_u8 = np.clip(center_prob.astype(np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    inst_bgr = np.zeros((*gt_inst.shape, 3), dtype=np.uint8)
    palette = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for inst_id in sorted(int(v) for v in np.unique(gt_inst) if int(v) > 0):
        inst_bgr[gt_inst == inst_id] = palette[inst_id % len(palette)]

    def _draw(base: np.ndarray, pts: list[tuple[int, int]], color: tuple[int, int, int]) -> np.ndarray:
        out = base.copy()
        for y, x in gt_pts:
            cv2.circle(out, (int(x), int(y)), 5, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        for idx, (y, x) in enumerate(pts, start=1):
            cv2.circle(out, (int(x), int(y)), 5, color, 2, lineType=cv2.LINE_AA)
            cv2.putText(out, str(idx), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return out

    header = np.full((96, image.shape[1] * 2, 3), 255, dtype=np.uint8)
    cv2.putText(header, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(header, subtitle_a, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(header, subtitle_b, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    top = np.concatenate([_draw(image, [], (255, 255, 255)), _draw(inst_bgr, [], (255, 255, 255))], axis=1)
    bottom = np.concatenate([_draw(heat_bgr, markers_a, (0, 255, 0)), _draw(heat_bgr, markers_b, (0, 0, 255))], axis=1)
    panel = np.concatenate([header, top, bottom], axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def _visual_category_candidates(
    per_sample_topology_rows: list[dict[str, Any]],
    *,
    val_reference_threshold: float,
    split_lookup: dict[tuple[str, str, float], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    candidates = {}
    val_rows = [
        row
        for row in per_sample_topology_rows
        if str(row["split"]) == "val" and abs(float(row["threshold"]) - float(val_reference_threshold)) < 1e-9
    ]

    def _pick(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None
        return sorted(rows, key=lambda row: (_sample_sort_key(str(row["sample"])), str(row["sample"])))[0]

    candidates["gt1_failure"] = _pick([row for row in val_rows if int(row["gt_instance_count"]) == 1 and not bool(row["current_marker_contract_pass"])])
    candidates["gt2_failure"] = _pick([row for row in val_rows if int(row["gt_instance_count"]) == 2 and not bool(row["current_marker_contract_pass"])])
    candidates["gt3_cap_masked_case"] = _pick(
        [
            row
            for row in val_rows
            if int(row["gt_instance_count"]) == 3
            and int(row["raw_component_count"]) > int(MAX_MARKERS)
            and int(row["current_predicted_count"]) == int(MAX_MARKERS)
        ]
    )
    split_rows = [row for row in per_sample_topology_rows if bool(row["split_from_previous_threshold"]) and str(row["split"]) == "val"]
    candidates["threshold_split_case"] = _pick(split_rows)
    candidates["duplicate_marker_case"] = _pick([row for row in val_rows if int(row["current_multiple_markers_inside_gt_instances"]) > 0])
    candidates["outside_marker_case"] = _pick([row for row in val_rows if int(row["current_markers_outside_all_gt_instances"]) > 0])
    candidates["positive_margin_but_strict_fail"] = _pick(
        [
            row
            for row in val_rows
            if row["margin"] is not None
            and float(row["margin"]) > 0.0
            and not bool(row["current_marker_contract_pass"])
        ]
    )
    return {key: row for key, row in candidates.items() if row is not None}


def _metric_alignment_summary(
    current_policy_rows: list[dict[str, Any]],
    threshold_topology_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    val_rows = [row for row in current_policy_rows if str(row["split"]) == "val"]
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in val_rows:
        grouped[float(row["threshold"])].append(row)
    threshold_metrics = {}
    best_f1_thr = None
    best_strict_thr = None
    best_exact_thr = None
    best_f1_val = None
    best_strict_val = None
    best_exact_val = None
    reference_threshold = None
    for threshold, rows in sorted(grouped.items()):
        corr = _pearson(
            [float(row["center_f1"]) for row in rows],
            [1.0 if bool(row["marker_contract_pass"]) else 0.0 for row in rows],
        )
        strict_pass_rows = [row for row in rows if bool(row["marker_contract_pass"])]
        strict_fail_rows = [row for row in rows if not bool(row["marker_contract_pass"])]
        current_f1 = float(np.mean([float(row["center_f1"]) for row in rows])) if rows else None
        current_strict = float(np.mean([1.0 if bool(row["marker_contract_pass"]) else 0.0 for row in rows])) if rows else None
        current_exact = float(np.mean([1.0 if int(row["predicted_count"]) == int(row["gt_instance_count"]) else 0.0 for row in rows])) if rows else None
        threshold_metrics[_threshold_key(threshold)] = {
            "threshold": float(threshold),
            "f1_strict_pass_correlation": corr,
            "center_f1_mean_samples": current_f1,
            "strict_marker_contract_pass_rate": current_strict,
            "exact_center_count_accuracy": current_exact,
            "center_f1_mean_strict_pass_samples": float(np.mean([float(row["center_f1"]) for row in strict_pass_rows])) if strict_pass_rows else None,
            "center_f1_mean_strict_fail_samples": float(np.mean([float(row["center_f1"]) for row in strict_fail_rows])) if strict_fail_rows else None,
        }
        if best_f1_val is None or (current_f1 is not None and float(current_f1) > float(best_f1_val)):
            best_f1_val = current_f1
            best_f1_thr = float(threshold)
        if best_strict_val is None or (current_strict is not None and float(current_strict) > float(best_strict_val)):
            best_strict_val = current_strict
            best_strict_thr = float(threshold)
        if best_exact_val is None or (current_exact is not None and float(current_exact) > float(best_exact_val)):
            best_exact_val = current_exact
            best_exact_thr = float(threshold)
    reference_threshold = best_f1_thr
    ref_rows = grouped.get(float(reference_threshold), []) if reference_threshold is not None else []
    topology_lookup = {
        (str(row["split"]), str(row["sample"]), float(row["threshold"])): row
        for row in threshold_topology_rows
    }
    positive_margin_strict_fail = []
    for row in ref_rows:
        if row["margin"] is not None and float(row["margin"]) > 0.0 and not bool(row["marker_contract_pass"]):
            topo = topology_lookup.get((str(row["split"]), str(row["sample"]), float(row["threshold"])))
            positive_margin_strict_fail.append(
                {
                    "sample": str(row["sample"]),
                    "gt_instance_count": int(row["gt_instance_count"]),
                    "predicted_count": int(row["predicted_count"]),
                    "raw_component_count": int(topo["raw_component_count"]) if topo is not None else None,
                    "current_missing_gt_instance_markers": int(row["missing_gt_instance_markers"]),
                    "current_multiple_markers_inside_gt_instances": int(row["multiple_markers_inside_gt_instances"]),
                    "current_markers_outside_all_gt_instances": int(row["markers_outside_all_gt_instances"]),
                    "gt_instances_intersected_by_multiple_components": int(topo["gt_instances_intersected_by_multiple_components"]) if topo is not None else None,
                    "components_outside_all_gt_instances": int(topo["components_outside_all_gt_instances"]) if topo is not None else None,
                }
            )
    reason_counts = Counter()
    for row in positive_margin_strict_fail:
        if int(row["current_multiple_markers_inside_gt_instances"]) > 0 or int(row.get("gt_instances_intersected_by_multiple_components") or 0) > 0:
            reason_counts["duplicate_or_fragmented_components_inside_gt"] += 1
        if int(row["current_markers_outside_all_gt_instances"]) > 0 or int(row.get("components_outside_all_gt_instances") or 0) > 0:
            reason_counts["outside_components_or_markers"] += 1
        if int(row["current_missing_gt_instance_markers"]) > 0:
            reason_counts["missing_gt_marker_coverage"] += 1
    explanation = (
        "Positive heatmap margin only proves that at least one GT center score exceeds far-background score. "
        "Strict marker contract can still fail when extraction produces multiple thresholded components inside the same GT instance, "
        "places extra markers outside all GT instances, or misses another GT instance entirely."
    )
    return {
        "validation_per_threshold": threshold_metrics,
        "threshold_maximizing_center_f1_mean_samples": best_f1_thr,
        "threshold_maximizing_strict_marker_contract_pass_rate": best_strict_thr,
        "threshold_maximizing_exact_center_count_accuracy": best_exact_thr,
        "thresholds_differ": bool(len({best_f1_thr, best_strict_thr, best_exact_thr}) > 1),
        "reference_threshold_for_per_sample_relationships": reference_threshold,
        "positive_margin_but_strict_fail_samples": positive_margin_strict_fail,
        "positive_margin_but_strict_fail_reason_counts": dict(reason_counts),
        "heatmap_margin_relationship_explanation": explanation,
    }


def _localization_metric_audit(
    *,
    checkpoint_path: Path,
    current_eval_rows: list[dict[str, Any]],
    current_summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    extra = dict(ckpt.get("extra") or {})
    threshold_sweep = dict(extra.get("threshold_sweep") or {})
    sweep_rows = list(threshold_sweep.get("rows") or [])
    training_by_threshold = {
        float(row["threshold"]): row for row in sweep_rows if row.get("threshold") is not None
    }
    diagnosis_mean_sample_by_threshold = {
        float(row["threshold"]): row for row in current_summary_rows if row.get("threshold") is not None and str(row["policy"]) == "current" and str(row["cap_policy"]) == "capped_top3" and str(row["split"]) == "val"
    }
    diagnosis_pooled = {}
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in current_eval_rows:
        if str(row["split"]) == "val":
            grouped[float(row["threshold"])].append(row)
    for threshold, rows in grouped.items():
        all_matches = [float(d) for row in rows for d in row["match_distances"]]
        diagnosis_pooled[float(threshold)] = float(np.mean(np.asarray(all_matches, dtype=np.float64))) if all_matches else None
    compared_thresholds = sorted(set(training_by_threshold.keys()) & set(diagnosis_mean_sample_by_threshold.keys()))
    comparison_rows = []
    for threshold in compared_thresholds:
        train_loc = training_by_threshold[threshold].get("localization_error_px")
        diag_mean = diagnosis_mean_sample_by_threshold[threshold].get("localization_error_px_mean_samples")
        diag_pooled = diagnosis_pooled.get(threshold)
        comparison_rows.append(
            {
                "threshold": float(threshold),
                "training_localization_error_px": float(train_loc) if train_loc is not None else None,
                "diagnosis_localization_error_px_mean_samples": float(diag_mean) if diag_mean is not None else None,
                "diagnosis_localization_error_px_pooled_matches": float(diag_pooled) if diag_pooled is not None else None,
            }
        )
    cause = "pooled_matched_pairs_vs_mean_of_per_sample_matched_pair_errors"
    return {
        "training_definition": {
            "source": "train_centerhead._threshold_sweep -> validate_centerhead.validate_centerhead",
            "units": "image_space_pixels",
            "aggregation": "pooled_mean_over_all_matched_pairs_in_dataset",
        },
        "diagnosis_definition": {
            "source": "previous diagnose_full_dataset_center_baseline threshold_summary",
            "units": "image_space_pixels",
            "aggregation": "mean_of_per_sample_mean_matched_pair_errors",
        },
        "cause_of_discrepancy": cause,
        "canonical_metric_proposed": {
            "name": "localization_error_px_pooled_matches",
            "units": "image_space_pixels",
            "aggregation": "pooled_mean_over_all_matched_pairs",
            "reason": "matches the training-time validate_centerhead calculation used inside threshold sweep rows",
        },
        "comparison_rows": comparison_rows,
    }


def _classify(
    *,
    representative_rows: list[dict[str, Any]],
    nms_rows: list[dict[str, Any]],
    count_rows: list[dict[str, Any]],
    metric_alignment: dict[str, Any],
    localization_audit: dict[str, Any],
) -> dict[str, Any]:
    val_rep = [row for row in representative_rows if str(row["split"]) == "val" and str(row["cap_policy"]) == "capped_top3"]
    val_nms = [row for row in nms_rows if str(row["split"]) == "val" and str(row["cap_policy"]) == "capped_top3"]
    current_best = _best_row([row for row in val_rep if str(row["policy"]) == "current"], metric="center_f1_mean_samples")
    argmax_best = _best_row([row for row in val_rep if str(row["policy"]) == "component_argmax"], metric="center_f1_mean_samples")
    weighted_best = _best_row([row for row in val_rep if str(row["policy"]) == "weighted_centroid"], metric="center_f1_mean_samples")
    nms_best = _best_row(val_nms, metric="strict_marker_contract_pass_rate") if val_nms else None
    count_all = [row for row in count_rows if str(row["split"]) == "val" and str(row["gt_group"]) == "all"]
    count_best = _best_row(count_all, metric="fraction_raw_count_gt_3") if count_all else None
    loc_rows = list(localization_audit.get("comparison_rows") or [])
    metric_only_issue = all(
        row.get("training_localization_error_px") is None
        or row.get("diagnosis_localization_error_px_pooled_matches") is None
        or abs(float(row["training_localization_error_px"]) - float(row["diagnosis_localization_error_px_pooled_matches"])) < 1e-6
        for row in loc_rows
    )
    rep_strict_gain = float(argmax_best["strict_marker_contract_pass_rate"]) - float(current_best["strict_marker_contract_pass_rate"])
    weighted_strict_gain = float(weighted_best["strict_marker_contract_pass_rate"]) - float(current_best["strict_marker_contract_pass_rate"])
    best_nms_strict_gain = (
        float(nms_best["strict_marker_contract_pass_rate"]) - float(current_best["strict_marker_contract_pass_rate"])
        if nms_best is not None
        else 0.0
    )
    raw_count_masking = float(count_best["fraction_raw_count_gt_3"]) if count_best is not None and count_best["fraction_raw_count_gt_3"] is not None else 0.0

    if not metric_only_issue and loc_rows:
        result = "coordinate_or_metric_mismatch"
        evidence = {
            "localization_comparison_rows": loc_rows,
        }
    elif max(rep_strict_gain, weighted_strict_gain) >= 0.05:
        result = "representative_point_failure"
        evidence = {
            "current_best": current_best,
            "argmax_best": argmax_best,
            "weighted_centroid_best": weighted_best,
        }
    elif best_nms_strict_gain >= 0.10:
        result = "connected_component_fragmentation"
        evidence = {
            "current_best": current_best,
            "best_nms": nms_best,
            "strict_gain_vs_current": best_nms_strict_gain,
        }
    elif (
        raw_count_masking >= 0.50
        and float(current_best["strict_marker_contract_pass_rate"]) >= 0.40
        and best_nms_strict_gain < 0.05
        and max(rep_strict_gain, weighted_strict_gain) <= 0.02
    ):
        result = "max_markers_masking"
        evidence = {
            "current_best": current_best,
            "fraction_raw_count_gt_3_best": raw_count_masking,
        }
    elif best_nms_strict_gain <= 0.05 and max(rep_strict_gain, weighted_strict_gain) <= 0.02:
        result = "heatmap_representation_failure"
        evidence = {
            "current_best": current_best,
            "best_nms": nms_best,
            "argmax_best": argmax_best,
            "weighted_centroid_best": weighted_best,
        }
    else:
        result = "mixed_extraction_failure"
        evidence = {
            "current_best": current_best,
            "best_nms": nms_best,
            "argmax_best": argmax_best,
            "weighted_centroid_best": weighted_best,
            "metric_alignment": metric_alignment,
        }
    return {"result": result, "evidence": evidence}


def run(
    *,
    run_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    train_manifest: Path,
    val_manifest: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_heatmap_dir = output_dir / "raw_heatmaps"
    raw_heatmap_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg(checkpoint_path=checkpoint_path, run_dir=run_dir if run_dir.exists() else None, config_path=config_path)
    extraction_contract = _current_extraction_contract(cfg, checkpoint_path)
    _write_json(output_dir / "extraction_contract.json", extraction_contract)

    model, ckpt_meta = _load_checkpoint_model(cfg, checkpoint_path, device)
    eval_specs = [("train", train_manifest), ("val", val_manifest)]

    per_sample_component_topology: list[dict[str, Any]] = []
    representative_eval_rows: list[dict[str, Any]] = []
    nms_eval_rows: list[dict[str, Any]] = []
    raw_vs_capped_rows: list[dict[str, Any]] = []
    sample_contexts: dict[tuple[str, str], dict[str, Any]] = {}

    current_policy_eval_rows: list[dict[str, Any]] = []

    for split_name, manifest_path in eval_specs:
        loader = _build_loader(cfg, manifest_path)
        previous_labels_by_sample: dict[str, np.ndarray] = {}
        grouped_current_rows: dict[float, list[dict[str, Any]]] = defaultdict(list)
        grouped_rep_rows: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
        grouped_nms_rows: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                out = model(images)
                pred_sem = torch.argmax(out["semantic"], dim=1).detach().cpu().numpy().astype(np.uint8)
                pred_center = torch.sigmoid(out["center"]).detach().cpu().numpy().astype(np.float32)

                for i in range(int(pred_sem.shape[0])):
                    sample = str(batch["sample"][i])
                    patient_id = str(batch["patient_id"][i]) if "patient_id" in batch else _patient_id_from_sample(sample)
                    gt_count = int(batch["gt_instance_count"][i])
                    image_path = Path(str(batch["image_path"][i]))
                    instance_mask_path = Path(str(batch["instance_mask_path"][i]))
                    metadata_path = Path(str(batch["metadata_path"][i]))
                    gt_pts = _extract_metadata_centers(str(metadata_path))
                    gt_inst = _align_instance_mask(instance_mask_path, target_hw=pred_sem[i].shape[:2])
                    center_prob = pred_center[i, 0].astype(np.float32)
                    leaf_union = _leaf_union_from_semantic(pred_sem[i])
                    raw_split_dir = raw_heatmap_dir / str(split_name)
                    raw_split_dir.mkdir(parents=True, exist_ok=True)
                    raw_heatmap_path = raw_split_dir / f"{sample}.npz"
                    np.savez_compressed(raw_heatmap_path, center_prob=center_prob)

                    sample_context = {
                        "split": str(split_name),
                        "sample": str(sample),
                        "patient_id": str(patient_id),
                        "gt_instance_count": int(gt_count),
                        "image_path": str(image_path),
                        "instance_mask_path": str(instance_mask_path),
                        "metadata_path": str(metadata_path),
                        "raw_heatmap_path": str(raw_heatmap_path),
                        "leaf_union": leaf_union.copy(),
                        "gt_points": list(gt_pts),
                    }
                    sample_contexts[(str(split_name), str(sample))] = sample_context

                    for threshold in TOPOLOGY_THRESHOLDS:
                        comp = _component_stats(center_prob, leaf_union, gt_inst, threshold=float(threshold))
                        components = list(comp["components"])
                        prev_labels = previous_labels_by_sample.get(sample)
                        split_stats = _split_from_previous(prev_labels, comp["labels"])
                        previous_labels_by_sample[sample] = comp["labels"]
                        components_per_gt = {}
                        multi_gt = 0
                        outside_components = 0
                        for inst_id in sorted(int(v) for v in np.unique(gt_inst) if int(v) > 0):
                            count = int(sum(1 for row in components if int(inst_id) in list(row["gt_instance_ids"])))
                            components_per_gt[str(inst_id)] = count
                            if count > 1:
                                multi_gt += 1
                        for row in components:
                            if not bool(row["intersects_any_gt"]):
                                outside_components += 1
                        heat_stats = _heatmap_margin(center_prob, gt_inst, gt_pts)
                        current_pts = _current_policy_markers(center_prob, leaf_union, threshold=float(threshold), max_markers=MAX_MARKERS)
                        current_eval = _per_sample_eval_row(
                            split_name=split_name,
                            threshold=float(threshold),
                            sample_context=sample_context,
                            gt_inst=gt_inst,
                            gt_pts=gt_pts,
                            pred_pts_scored=current_pts,
                            raw_predicted_count=int(len(components)),
                        )
                        current_eval["margin"] = heat_stats["margin"]
                        current_eval["raw_component_count"] = int(len(components))
                        grouped_current_rows[float(threshold)].append(current_eval)
                        current_policy_eval_rows.append(current_eval)

                        topology_row = {
                            "split": str(split_name),
                            "sample": str(sample),
                            "patient_id": str(patient_id),
                            "gt_instance_count": int(gt_count),
                            "threshold": float(threshold),
                            "raw_component_count": int(len(components)),
                            "component_areas_json": _json_dumps([int(row["area"]) for row in components]),
                            "component_max_scores_json": _json_dumps([_round_float(float(row["max_score"])) for row in components]),
                            "component_centroids_yx_json": _json_dumps([[float(row["centroid_y"]), float(row["centroid_x"])] for row in components]),
                            "component_argmax_yx_json": _json_dumps([[int(row["argmax_y"]), int(row["argmax_x"])] for row in components]),
                            "component_weighted_centroids_yx_json": _json_dumps([[float(row["weighted_centroid_y"]), float(row["weighted_centroid_x"])] for row in components]),
                            "components_per_gt_instance_json": _json_dumps(components_per_gt),
                            "components_outside_all_gt_instances": int(outside_components),
                            "gt_instances_intersected_by_multiple_components": int(multi_gt),
                            "split_from_previous_threshold": bool(split_stats["split_from_previous_threshold"]),
                            "split_parent_component_count": int(split_stats["split_parent_component_count"]),
                            "split_child_component_count": int(split_stats["split_child_component_count"]),
                            "raw_count_increase_vs_previous_threshold": bool(split_stats["raw_count_increase_vs_previous_threshold"]),
                            "current_predicted_count": int(current_eval["predicted_count"]),
                            "current_marker_contract_pass": bool(current_eval["marker_contract_pass"]),
                            "current_missing_gt_instance_markers": int(current_eval["missing_gt_instance_markers"]),
                            "current_multiple_markers_inside_gt_instances": int(current_eval["multiple_markers_inside_gt_instances"]),
                            "current_markers_outside_all_gt_instances": int(current_eval["markers_outside_all_gt_instances"]),
                            "margin": heat_stats["margin"],
                        }
                        per_sample_component_topology.append(topology_row)

                        for policy in ("current", "component_argmax", "weighted_centroid"):
                            rep_pts = _representative_markers_from_components(components, policy=policy, max_markers=MAX_MARKERS)
                            rep_eval = _per_sample_eval_row(
                                split_name=split_name,
                                threshold=float(threshold),
                                sample_context=sample_context,
                                gt_inst=gt_inst,
                                gt_pts=gt_pts,
                                pred_pts_scored=rep_pts,
                                raw_predicted_count=int(len(components)),
                            )
                            grouped_rep_rows[(policy, float(threshold))].append(rep_eval)

                        for kernel in (3, 5, 9):
                            uncapped_pts = _max_pool_nms_markers(center_prob, leaf_union, threshold=float(threshold), kernel=int(kernel), max_markers=None)
                            capped_pts = uncapped_pts[: int(MAX_MARKERS)]
                            uncapped_eval = _per_sample_eval_row(
                                split_name=split_name,
                                threshold=float(threshold),
                                sample_context=sample_context,
                                gt_inst=gt_inst,
                                gt_pts=gt_pts,
                                pred_pts_scored=uncapped_pts,
                                raw_predicted_count=int(len(uncapped_pts)),
                            )
                            capped_eval = _per_sample_eval_row(
                                split_name=split_name,
                                threshold=float(threshold),
                                sample_context=sample_context,
                                gt_inst=gt_inst,
                                gt_pts=gt_pts,
                                pred_pts_scored=capped_pts,
                                raw_predicted_count=int(len(uncapped_pts)),
                            )
                            grouped_nms_rows[(f"nms_kernel_{kernel}", "uncapped", float(threshold))].append(uncapped_eval)
                            grouped_nms_rows[(f"nms_kernel_{kernel}", "capped_top3", float(threshold))].append(capped_eval)

        for threshold, rows in sorted(grouped_current_rows.items()):
            raw_vs_capped_rows.extend(_raw_and_capped_count_rows(rows, split_name=split_name, threshold=float(threshold)))
        for (policy, threshold), rows in sorted(grouped_rep_rows.items(), key=lambda item: (item[0][0], item[0][1])):
            representative_eval_rows.append(
                _evaluate_rows(rows, split_name=split_name, threshold=float(threshold), policy_name=policy, cap_name="capped_top3")
            )
        for (policy, cap_name, threshold), rows in sorted(grouped_nms_rows.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
            nms_eval_rows.append(
                _evaluate_rows(rows, split_name=split_name, threshold=float(threshold), policy_name=policy, cap_name=cap_name)
            )

    threshold_topology_summary: list[dict[str, Any]] = []
    grouped_topology: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in per_sample_component_topology:
        grouped_topology[(str(row["split"]), float(row["threshold"]))].append(row)
    for (split_name, threshold), rows in sorted(grouped_topology.items(), key=lambda item: (item[0][0], item[0][1])):
        threshold_topology_summary.append(
            {
                "split": str(split_name),
                "threshold": float(threshold),
                **_aggregate_topology_rows(rows),
            }
        )

    representative_policy_comparison_rows = []
    for split_name in ("train", "val"):
        subset = [row for row in representative_eval_rows if str(row["split"]) == split_name]
        if not subset:
            continue
        best_f1 = _best_row(subset, metric="center_f1_mean_samples")
        best_strict = _best_row(subset, metric="strict_marker_contract_pass_rate")
        for row in subset:
            row_out = dict(row)
            row_out["is_best_center_f1_mean_samples"] = bool(str(row["policy"]) == str(best_f1["policy"]) and abs(float(row["threshold"]) - float(best_f1["threshold"])) < 1e-9)
            row_out["is_best_strict_marker_contract_pass_rate"] = bool(str(row["policy"]) == str(best_strict["policy"]) and abs(float(row["threshold"]) - float(best_strict["threshold"])) < 1e-9)
            representative_policy_comparison_rows.append(row_out)

    nms_policy_comparison_rows = []
    for split_name in ("train", "val"):
        subset = [row for row in nms_eval_rows if str(row["split"]) == split_name]
        if not subset:
            continue
        best_f1 = _best_row(subset, metric="center_f1_mean_samples")
        best_strict = _best_row(subset, metric="strict_marker_contract_pass_rate")
        for row in subset:
            row_out = dict(row)
            row_out["is_best_center_f1_mean_samples"] = bool(
                str(row["policy"]) == str(best_f1["policy"])
                and str(row["cap_policy"]) == str(best_f1["cap_policy"])
                and abs(float(row["threshold"]) - float(best_f1["threshold"])) < 1e-9
            )
            row_out["is_best_strict_marker_contract_pass_rate"] = bool(
                str(row["policy"]) == str(best_strict["policy"])
                and str(row["cap_policy"]) == str(best_strict["cap_policy"])
                and abs(float(row["threshold"]) - float(best_strict["threshold"])) < 1e-9
            )
            nms_policy_comparison_rows.append(row_out)

    metric_alignment = _metric_alignment_summary(current_policy_eval_rows, per_sample_component_topology)
    _write_json(output_dir / "metric_alignment_summary.json", metric_alignment)

    localization_audit = _localization_metric_audit(
        checkpoint_path=checkpoint_path,
        current_eval_rows=current_policy_eval_rows,
        current_summary_rows=representative_policy_comparison_rows,
    )
    _write_json(output_dir / "localization_metric_audit.json", localization_audit)

    classification = _classify(
        representative_rows=representative_policy_comparison_rows,
        nms_rows=nms_policy_comparison_rows,
        count_rows=raw_vs_capped_rows,
        metric_alignment=metric_alignment,
        localization_audit=localization_audit,
    )

    _write_csv(
        output_dir / "threshold_topology_summary.csv",
        threshold_topology_summary,
        [
            "split",
            "threshold",
            "sample_count",
            "raw_component_count_mean",
            "raw_component_count_median",
            "fraction_samples_raw_count_gt_3",
            "samples_with_split_from_previous_threshold",
            "fraction_samples_with_split_from_previous_threshold",
            "total_split_parent_components",
            "total_split_child_components",
            "samples_with_raw_count_increase_vs_previous_threshold",
            "mean_components_outside_all_gt_instances",
            "mean_gt_instances_intersected_by_multiple_components",
        ],
    )
    _write_csv(
        output_dir / "per_sample_component_topology.csv",
        per_sample_component_topology,
        [
            "split",
            "sample",
            "patient_id",
            "gt_instance_count",
            "threshold",
            "raw_component_count",
            "component_areas_json",
            "component_max_scores_json",
            "component_centroids_yx_json",
            "component_argmax_yx_json",
            "component_weighted_centroids_yx_json",
            "components_per_gt_instance_json",
            "components_outside_all_gt_instances",
            "gt_instances_intersected_by_multiple_components",
            "split_from_previous_threshold",
            "split_parent_component_count",
            "split_child_component_count",
            "raw_count_increase_vs_previous_threshold",
            "current_predicted_count",
            "current_marker_contract_pass",
            "current_missing_gt_instance_markers",
            "current_multiple_markers_inside_gt_instances",
            "current_markers_outside_all_gt_instances",
            "margin",
        ],
    )
    _write_csv(
        output_dir / "raw_vs_capped_count_distribution.csv",
        raw_vs_capped_rows,
        [
            "split",
            "threshold",
            "gt_group",
            "sample_count",
            "raw_count_histogram_json",
            "capped_count_histogram_json",
            "fraction_raw_count_gt_3",
            "raw_count_mean",
            "raw_count_median",
        ],
    )
    _write_csv(
        output_dir / "representative_policy_comparison.csv",
        representative_policy_comparison_rows,
        [
            "split",
            "threshold",
            "policy",
            "cap_policy",
            "sample_count",
            "center_precision",
            "center_recall",
            "center_f1",
            "center_f1_mean_samples",
            "strict_marker_contract_pass_rate",
            "exact_center_count_accuracy",
            "localization_error_px_pooled_matches",
            "localization_error_px_mean_samples",
            "missing_gt_markers",
            "duplicate_markers",
            "markers_outside_instances",
            "fraction_count_3",
            "fraction_raw_count_gt_3",
            "predicted_count_histogram_json",
            "gt_count_histogram_json",
            "is_best_center_f1_mean_samples",
            "is_best_strict_marker_contract_pass_rate",
        ],
    )
    _write_csv(
        output_dir / "nms_policy_comparison.csv",
        nms_policy_comparison_rows,
        [
            "split",
            "threshold",
            "policy",
            "cap_policy",
            "sample_count",
            "center_precision",
            "center_recall",
            "center_f1",
            "center_f1_mean_samples",
            "strict_marker_contract_pass_rate",
            "exact_center_count_accuracy",
            "localization_error_px_pooled_matches",
            "localization_error_px_mean_samples",
            "missing_gt_markers",
            "duplicate_markers",
            "markers_outside_instances",
            "fraction_count_3",
            "fraction_raw_count_gt_3",
            "predicted_count_histogram_json",
            "gt_count_histogram_json",
            "is_best_center_f1_mean_samples",
            "is_best_strict_marker_contract_pass_rate",
        ],
    )

    visual_review_dir = output_dir / "visual_review"
    visual_candidates = _visual_category_candidates(
        per_sample_component_topology,
        val_reference_threshold=float(metric_alignment.get("reference_threshold_for_per_sample_relationships") or TOPOLOGY_THRESHOLDS[0]),
        split_lookup={},
    )
    for category, row in visual_candidates.items():
        sample_key = (str(row["split"]), str(row["sample"]))
        ctx = sample_contexts.get(sample_key)
        if ctx is None:
            continue
        threshold = float(row["threshold"])
        center_prob = np.load(str(ctx["raw_heatmap_path"]))["center_prob"].astype(np.float32)
        leaf_union = np.asarray(ctx["leaf_union"], dtype=bool)
        current_markers = [(int(y), int(x)) for (y, x, _s) in _current_policy_markers(center_prob, leaf_union, threshold=threshold, max_markers=MAX_MARKERS)]
        alt_threshold = float(threshold)
        if category == "threshold_split_case":
            thr_idx = list(TOPOLOGY_THRESHOLDS).index(float(threshold)) if float(threshold) in TOPOLOGY_THRESHOLDS else 0
            alt_threshold = float(TOPOLOGY_THRESHOLDS[max(thr_idx - 1, 0)])
        alt_markers = [(int(y), int(x)) for (y, x, _s) in _current_policy_markers(center_prob, leaf_union, threshold=alt_threshold, max_markers=MAX_MARKERS)]
        _make_visual_panel(
            image_path=Path(str(ctx["image_path"])),
            instance_mask_path=Path(str(ctx["instance_mask_path"])),
            center_prob=center_prob,
            gt_pts=[(int(y), int(x)) for y, x in ctx["gt_points"]],
            markers_a=current_markers,
            markers_b=alt_markers,
            title=f"{category} :: {ctx['sample']}",
            subtitle_a=f"current threshold={threshold:.3f} pred={len(current_markers)}",
            subtitle_b=f"comparison threshold={alt_threshold:.3f} pred={len(alt_markers)}",
            out_path=visual_review_dir / f"{category}__{ctx['sample']}.png",
        )

    diagnosis_summary = {
        "run_dir": str(run_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": ckpt_meta["checkpoint_sha256"],
        "checkpoint_epoch": ckpt_meta["epoch"],
        "train_manifest": str(train_manifest.resolve()),
        "val_manifest": str(val_manifest.resolve()),
        "classification": classification,
        "best_current_representative_train": _best_row([row for row in representative_policy_comparison_rows if str(row["split"]) == "train" and str(row["policy"]) == "current"], metric="center_f1_mean_samples"),
        "best_current_representative_val": _best_row([row for row in representative_policy_comparison_rows if str(row["split"]) == "val" and str(row["policy"]) == "current"], metric="center_f1_mean_samples"),
        "best_nms_val_capped": _best_row([row for row in nms_policy_comparison_rows if str(row["split"]) == "val" and str(row["cap_policy"]) == "capped_top3"], metric="strict_marker_contract_pass_rate"),
        "metric_alignment": metric_alignment,
        "localization_metric_audit": localization_audit,
    }
    _write_json(output_dir / "diagnosis_summary.json", diagnosis_summary)

    files_to_review = [
        output_dir / "extraction_contract.json",
        output_dir / "threshold_topology_summary.csv",
        output_dir / "per_sample_component_topology.csv",
        output_dir / "raw_vs_capped_count_distribution.csv",
        output_dir / "representative_policy_comparison.csv",
        output_dir / "nms_policy_comparison.csv",
        output_dir / "metric_alignment_summary.json",
        output_dir / "localization_metric_audit.json",
        output_dir / "diagnosis_summary.json",
        output_dir / "visual_review",
    ]
    _write_text(output_dir / "files_to_review.txt", "\n".join(str(path.resolve()) for path in files_to_review) + "\n")
    return diagnosis_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--config-path", type=str, default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--checkpoint-path", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    ap.add_argument("--train-manifest", type=str, default=str(DEFAULT_TRAIN_MANIFEST))
    ap.add_argument("--val-manifest", type=str, default=str(DEFAULT_VAL_MANIFEST))
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    summary = run(
        run_dir=Path(args.run_dir).resolve(),
        config_path=Path(args.config_path).resolve(),
        checkpoint_path=Path(args.checkpoint_path).resolve(),
        train_manifest=Path(args.train_manifest).resolve(),
        val_manifest=Path(args.val_manifest).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        device=torch.device(str(args.device)),
    )
    print(json.dumps(summary["classification"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
