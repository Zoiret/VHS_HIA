from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import bridge_suppression_head as bridge
import bridge_suppression_head_v6_patient_disjoint_dev as v6


DEFAULT_ANALYSIS_DIR = bridge.REPO_ROOT / "training" / "analysis" / "bridge_suppression_head_v6_failure_audit"
AUTHORITATIVE_V6_CHECKPOINT = v6.DEFAULT_RUN_DIR / "best_train_reconstruction.pth"
AUTHORITATIVE_V6_EPOCH = 63
FIXED_REMOVE_THRESHOLD = 0.50
SCORE_PARTITIONS = ("TRUE_BRIDGE", "POSITIVE_SAMPLE_NON_BRIDGE", "ZERO_TARGET_SAMPLE")
BOUNDARY_BINS = ("0_2", "3_5", "6_10", "gt_10")
TOPOLOGY_FAILURE_RULES = {
    "A_leaflet_interior_cut": "inside_gt_removed_pixels > 0 AND far_inside_gt_removed_pixels > 0",
    "B_boundary_erosion_or_fragmentation": "inside_gt_removed_pixels > 0 AND far_inside_gt_removed_pixels == 0",
    "C_harmless_external_fp_cleanup_but_reconstruction_changed": "inside_gt_removed_pixels == 0 AND outside_gt_removed_pixels > 0 AND reconstruction_regressed == 1",
    "D_other": "all remaining cases",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _as_target_mask(row: dict[str, Any]) -> np.ndarray:
    value = row["bridge_target"]
    if torch.is_tensor(value):
        return value[0].detach().cpu().numpy() > 0.5
    arr = np.asarray(value)
    if arr.ndim == 3:
        arr = arr[0]
    return arr > 0.5


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _bool_mask_summary(mask: np.ndarray) -> dict[str, int]:
    return {"pixels": int(np.sum(mask.astype(np.uint8)))}


def _load_checkpoint_payload(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(str(path), map_location=map_location)
    if not isinstance(payload, dict):
        raise SystemExit(f"Unsupported checkpoint payload format: {path}")
    return payload


def verify_frozen_v6_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Frozen V6 checkpoint not found: {path}")
    payload = _load_checkpoint_payload(path, map_location="cpu")
    step = int(payload["step"])
    if step != int(AUTHORITATIVE_V6_EPOCH):
        raise SystemExit(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "unexpected_v6_epoch",
                    "expected_epoch": int(AUTHORITATIVE_V6_EPOCH),
                    "actual_epoch": step,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    model_state = payload["model"]
    return {
        "checkpoint_path": str(path.resolve()),
        "epoch": step,
        "checkpoint_sha256": bridge._sha256_file(path),
        "model_state_sha256": bridge.canonical_model_state_sha256(model_state),
        "extra": payload.get("extra") or {},
    }


def load_v6_model_and_verify_invariants(
    cfg: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[bridge.FrozenSemanticBridgeSuppressionModel, dict[str, Any]]:
    model = bridge.build_model_from_cfg(cfg).to(device)
    semantic_info = bridge.load_semantic_checkpoint(
        model,
        bridge._resolve_repo_path((cfg.get("train") or {}).get("init_checkpoint"), bridge.DEFAULT_SEMANTIC_CHECKPOINT),
    )
    semantic_named = [(name, p) for name, p in model.named_parameters() if name.startswith("base.")]
    semantic_snap = bridge._snapshot_named_parameters(semantic_named)
    bn_snap = bridge._collect_batchnorm_stats(model.base)
    payload = _load_checkpoint_payload(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model"], strict=True)
    semantic_parameter_max_delta = float(bridge._max_parameter_delta_from_snapshot(semantic_named, semantic_snap))
    semantic_bn_state_max_delta = float(bridge._max_bn_delta(model.base, bn_snap))
    if semantic_parameter_max_delta != 0.0 or semantic_bn_state_max_delta != 0.0:
        raise SystemExit(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "frozen_semantic_invariants_changed",
                    "semantic_parameter_max_delta": semantic_parameter_max_delta,
                    "semantic_bn_state_max_delta": semantic_bn_state_max_delta,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return model, {
        "semantic_checkpoint": semantic_info,
        "semantic_parameter_max_delta": semantic_parameter_max_delta,
        "semantic_bn_state_max_delta": semantic_bn_state_max_delta,
    }


def load_historical_v2_model(
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[bridge.FrozenSemanticBridgeSuppressionModel, dict[str, Any]]:
    baseline_cfg = cfg.get("historical_v2_baseline") or {}
    checkpoint_path = bridge._resolve_repo_path(
        baseline_cfg.get("checkpoint_path"),
        bridge.REPO_ROOT / "training" / "runs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2" / "best_reconstruction.pth",
    )
    if not checkpoint_path.exists():
        raise SystemExit(f"Historical V2 checkpoint not found: {checkpoint_path}")
    model = bridge.build_model_from_cfg(cfg).to(device)
    semantic_info = bridge.load_semantic_checkpoint(
        model,
        bridge._resolve_repo_path((cfg.get("train") or {}).get("init_checkpoint"), bridge.DEFAULT_SEMANTIC_CHECKPOINT),
    )
    payload = _load_checkpoint_payload(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model"], strict=True)
    return model, {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": bridge._sha256_file(checkpoint_path),
        "model_state_sha256": bridge.canonical_model_state_sha256(payload["model"]),
        "epoch": int(payload.get("step", -1)),
        "semantic_checkpoint": semantic_info,
    }


def attach_split_name(cached_records: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in cached_records:
        current = dict(row)
        current["_split_name"] = str(split_name)
        out.append(current)
    return out


def _bridge_probs_for_cached_records(
    model: bridge.FrozenSemanticBridgeSuppressionModel,
    cached_records: list[dict[str, Any]],
    device: torch.device,
) -> np.ndarray:
    batch = bridge.stack_cached_batch(cached_records, device)
    model.eval()
    with torch.no_grad():
        outputs = model.bridge_forward_from_cached(
            x_0_4=batch["x_0_4"],
            x_2_2=batch["x_2_2"],
            p_leaf=batch["p_leaf"],
        )
    return torch.sigmoid(outputs["bridge_logits"]).detach().cpu().numpy().astype(np.float32)


def _partition_masks(row: dict[str, Any]) -> dict[str, np.ndarray]:
    target = _as_target_mask(row)
    candidate = row["candidate_mask_np"].astype(bool)
    sample_has_bridge = bool(np.any(target))
    zero_target_mask = candidate & np.full(candidate.shape, not sample_has_bridge, dtype=bool)
    return {
        "TRUE_BRIDGE": target & candidate,
        "POSITIVE_SAMPLE_NON_BRIDGE": (~target) & candidate & np.full(candidate.shape, sample_has_bridge, dtype=bool),
        "ZERO_TARGET_SAMPLE": zero_target_mask,
    }


def summarize_score_partition(cached_records: list[dict[str, Any]], probs: np.ndarray) -> dict[str, Any]:
    out: dict[str, dict[str, Any]] = {}
    for partition in SCORE_PARTITIONS:
        values: list[np.ndarray] = []
        for idx, row in enumerate(cached_records):
            mask = _partition_masks(row)[partition]
            if np.any(mask):
                values.append(probs[idx, 0][mask].astype(np.float64))
        all_values = np.concatenate(values, axis=0) if values else np.asarray([], dtype=np.float64)
        summary = _percentiles(all_values)
        summary.update(
            {
                "pixel_count": int(all_values.size),
                "fraction_ge_0p50": float(np.mean(all_values >= FIXED_REMOVE_THRESHOLD)) if all_values.size else 0.0,
            }
        )
        out[partition] = summary
    return out


def _gt_union_and_boundary(gt_instances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt_union = gt_instances.astype(np.uint8) > 0
    boundary = np.zeros_like(gt_union, dtype=bool)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.roll(gt_union, shift=(dx, dy), axis=(0, 1))
        if dx == 1:
            shifted[0, :] = False
        if dx == -1:
            shifted[-1, :] = False
        if dy == 1:
            shifted[:, 0] = False
        if dy == -1:
            shifted[:, -1] = False
        boundary |= gt_union & (~shifted)
    return gt_union, boundary


def _distance_to_boundary_bins(gt_union: np.ndarray, boundary: np.ndarray, inside_removed_mask: np.ndarray) -> dict[str, int]:
    out = {name: 0 for name in BOUNDARY_BINS}
    removed = np.argwhere(inside_removed_mask.astype(np.uint8) > 0)
    if removed.size == 0:
        return out
    boundary_coords = np.argwhere(boundary.astype(np.uint8) > 0)
    if boundary_coords.size == 0:
        out["gt_10"] = int(len(removed))
        return out
    for coord in removed:
        dist = int(np.min(np.abs(boundary_coords[:, 0] - coord[0]) + np.abs(boundary_coords[:, 1] - coord[1])))
        if dist <= 2:
            out["0_2"] += 1
        elif dist <= 5:
            out["3_5"] += 1
        elif dist <= 10:
            out["6_10"] += 1
        else:
            out["gt_10"] += 1
    return out


def spatial_damage_partition(row: dict[str, Any], pred_remove: np.ndarray) -> dict[str, Any]:
    candidate = row["candidate_mask_np"].astype(bool)
    pred_remove_candidate = pred_remove.astype(bool) & candidate
    gt_union, boundary = _gt_union_and_boundary(row["gt_instances"])
    inside = pred_remove_candidate & gt_union
    outside = pred_remove_candidate & (~gt_union)
    if np.any(inside & outside):
        raise SystemExit("Spatial damage partition overlap detected.")
    if int(np.sum(pred_remove_candidate)) != int(np.sum(inside) + np.sum(outside)):
        raise SystemExit("Spatial damage partition is not exhaustive.")
    boundary_bins = _distance_to_boundary_bins(gt_union, boundary, inside)
    return {
        "pred_remove_candidate": pred_remove_candidate,
        "inside_gt": inside,
        "outside_gt": outside,
        "boundary_bins": boundary_bins,
        "gt_union": gt_union,
    }


def damage_decomposition(cached_records: list[dict[str, Any]], probs: np.ndarray, *, threshold: float = FIXED_REMOVE_THRESHOLD) -> dict[str, Any]:
    aggregate: dict[str, dict[str, Any]] = {}
    per_sample_rows: list[dict[str, Any]] = []
    for split_name in ("train", "val"):
        for polarity in ("positive_target_samples", "zero_target_samples"):
            aggregate[f"{split_name}_{polarity}"] = {
                "inside_gt_removed_pixels": 0,
                "outside_gt_removed_pixels": 0,
                "boundary_distance_bins": {name: 0 for name in BOUNDARY_BINS},
            }
    for idx, row in enumerate(cached_records):
        pred_remove = probs[idx, 0] >= float(threshold)
        spatial = spatial_damage_partition(row, pred_remove)
        split_name = str(row.get("_split_name", "unknown"))
        polarity = "positive_target_samples" if int(row["bridge_positive"]) == 1 else "zero_target_samples"
        bucket = aggregate[f"{split_name}_{polarity}"]
        bucket["inside_gt_removed_pixels"] += int(np.sum(spatial["inside_gt"]))
        bucket["outside_gt_removed_pixels"] += int(np.sum(spatial["outside_gt"]))
        for bin_name, count in spatial["boundary_bins"].items():
            bucket["boundary_distance_bins"][bin_name] += int(count)
        per_sample_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "patient_id": str(row["patient_id"]),
                "split": split_name,
                "bridge_positive": int(row["bridge_positive"]),
                "removed_pixels_total": int(np.sum(spatial["pred_remove_candidate"])),
                "inside_gt_removed_pixels": int(np.sum(spatial["inside_gt"])),
                "outside_gt_removed_pixels": int(np.sum(spatial["outside_gt"])),
                **{f"inside_gt_boundary_{name}": int(value) for name, value in spatial["boundary_bins"].items()},
            }
        )
    return {"aggregate": aggregate, "per_sample_rows": per_sample_rows}


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return bridge._connected_components(mask.astype(np.uint8))


def removal_component_morphology(cached_records: list[dict[str, Any]], probs: np.ndarray, *, threshold: float = FIXED_REMOVE_THRESHOLD) -> dict[str, Any]:
    component_areas: list[float] = []
    total_components = 0
    total_overlapping_true_bridge = 0
    total_inside_gt = 0
    total_outside_gt = 0
    zero_target_components_intersecting_gt_interior = 0
    zero_target_samples_with_topology_change = 0
    per_sample_rows: list[dict[str, Any]] = []
    per_component_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(cached_records):
        pred_remove = probs[idx, 0] >= float(threshold)
        spatial = spatial_damage_partition(row, pred_remove)
        candidate = row["candidate_mask_np"].astype(bool)
        gt_union = spatial["gt_union"]
        true_bridge = _as_target_mask(row)
        target_pixels = int(np.sum(true_bridge))
        labels, comp_n = _connected_components(spatial["pred_remove_candidate"].astype(np.uint8))
        total_components += int(comp_n)
        overlap_true_bridge_components = 0
        inside_gt_only_components = 0
        outside_gt_only_components = 0
        zero_target_gt_intersections = 0
        component_area_values: list[int] = []
        overlap_precisions: list[float] = []
        overlap_recalls: list[float] = []
        for comp_id in range(1, int(comp_n) + 1):
            comp_mask = labels == comp_id
            area = int(np.sum(comp_mask))
            component_areas.append(float(area))
            component_area_values.append(area)
            overlap_true_bridge_pixels = int(np.sum(comp_mask & true_bridge))
            inside_gt_pixels = int(np.sum(comp_mask & gt_union))
            outside_gt_pixels = int(np.sum(comp_mask & (~gt_union) & candidate))
            overlaps_target = overlap_true_bridge_pixels > 0
            entirely_inside_gt = bool(np.all(gt_union[comp_mask])) if area > 0 else False
            entirely_outside_gt = bool(np.all((~gt_union)[comp_mask])) if area > 0 else False
            if overlaps_target:
                overlap_true_bridge_components += 1
                total_overlapping_true_bridge += 1
                overlap_precisions.append(float(overlap_true_bridge_pixels / max(area, 1)))
                overlap_recalls.append(float(overlap_true_bridge_pixels / max(target_pixels, 1)))
            if entirely_inside_gt:
                inside_gt_only_components += 1
                total_inside_gt += 1
            if entirely_outside_gt:
                outside_gt_only_components += 1
                total_outside_gt += 1
            if int(row["bridge_positive"]) == 0 and inside_gt_pixels > 0:
                zero_target_gt_intersections += 1
                zero_target_components_intersecting_gt_interior += 1
            per_component_rows.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "patient_id": str(row["patient_id"]),
                    "split": str(row.get("_split_name", "unknown")),
                    "bridge_positive": int(row["bridge_positive"]),
                    "component_id": int(comp_id),
                    "area": area,
                    "overlaps_true_bridge": int(overlaps_target),
                    "entirely_inside_gt_leaflet": int(entirely_inside_gt),
                    "entirely_outside_gt_leaflet": int(entirely_outside_gt),
                    "precision_against_target": float(overlap_true_bridge_pixels / max(area, 1)),
                    "recall_contribution": float(overlap_true_bridge_pixels / max(target_pixels, 1)),
                    "inside_gt_pixels": inside_gt_pixels,
                    "outside_gt_pixels": outside_gt_pixels,
                }
            )
        topology_changed = int(row.get("_predicted_component_topology_changed", 0))
        if int(row["bridge_positive"]) == 0 and topology_changed == 1:
            zero_target_samples_with_topology_change += 1
        per_sample_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "patient_id": str(row["patient_id"]),
                "split": str(row.get("_split_name", "unknown")),
                "bridge_positive": int(row["bridge_positive"]),
                "component_count": int(comp_n),
                "total_removed_area": int(np.sum(spatial["pred_remove_candidate"])),
                "largest_component_area": int(max(component_area_values)) if component_area_values else 0,
                "median_component_area": float(np.median(component_area_values)) if component_area_values else 0.0,
                "fraction_components_overlapping_true_bridge": float(overlap_true_bridge_components / max(comp_n, 1)),
                "fraction_components_entirely_inside_gt_leaflet": float(inside_gt_only_components / max(comp_n, 1)),
                "fraction_components_entirely_outside_gt_leaflet": float(outside_gt_only_components / max(comp_n, 1)),
                "mean_overlap_precision_against_target": float(np.mean(overlap_precisions)) if overlap_precisions else 0.0,
                "mean_overlap_recall_contribution": float(np.mean(overlap_recalls)) if overlap_recalls else 0.0,
                "zero_target_components_intersecting_gt_interior": int(zero_target_gt_intersections),
                "topology_change": int(topology_changed),
            }
        )
    return {
        "component_area_distribution": _percentiles(np.asarray(component_areas, dtype=np.float64)),
        "aggregate": {
            "total_component_count": int(total_components),
            "fraction_overlapping_true_bridge_target": float(total_overlapping_true_bridge / max(total_components, 1)),
            "fraction_entirely_inside_gt_leaflet": float(total_inside_gt / max(total_components, 1)),
            "fraction_entirely_outside_gt_leaflet": float(total_outside_gt / max(total_components, 1)),
            "zero_target_components_intersecting_gt_interior": int(zero_target_components_intersecting_gt_interior),
            "zero_target_samples_with_topology_change": int(zero_target_samples_with_topology_change),
        },
        "per_sample_rows": per_sample_rows,
        "per_component_rows": per_component_rows,
    }


def classify_topology_failure_case(row: dict[str, Any]) -> str:
    if int(row["inside_gt_removed_pixels"]) > 0 and int(row["far_inside_gt_removed_pixels"]) > 0:
        return "A_leaflet_interior_cut"
    if int(row["inside_gt_removed_pixels"]) > 0:
        return "B_boundary_erosion_or_fragmentation"
    if int(row["outside_gt_removed_pixels"]) > 0 and int(row["reconstruction_regressed"]) == 1:
        return "C_harmless_external_fp_cleanup_but_reconstruction_changed"
    return "D_other"


def topology_failure_cases(cached_records: list[dict[str, Any]], probs: np.ndarray, *, threshold: float = FIXED_REMOVE_THRESHOLD) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(cached_records):
        if int(row["bridge_positive"]) != 0:
            continue
        pred_remove = probs[idx, 0] >= float(threshold)
        spatial = spatial_damage_partition(row, pred_remove)
        refined = ((row["candidate_mask_np"].astype(bool)) & (~spatial["pred_remove_candidate"])).astype(np.uint8)
        pred = bridge.run_locked_reconstruction(refined, row["gt_instances"])
        start = row["start_reconstruction"]
        labels, comp_n = _connected_components(spatial["pred_remove_candidate"].astype(np.uint8))
        largest_component = 0
        for comp_id in range(1, int(comp_n) + 1):
            largest_component = max(largest_component, int(np.sum(labels == comp_id)))
        far_inside_gt_removed_pixels = int(spatial["boundary_bins"]["6_10"] + spatial["boundary_bins"]["gt_10"])
        case_row = {
            "sample_id": str(row["sample_id"]),
            "patient_id": str(row["patient_id"]),
            "split": str(row.get("_split_name", "unknown")),
            "semantic_candidate_component_count_before": int(row["component_count_start"]),
            "component_count_after_v6_removal": int(pred["pred_k"]),
            "gt_instance_count": int(row["gt_count"]),
            "topology_changed": int(int(pred["pred_k"]) != int(row["component_count_start"])),
            "reconstruction_regressed": int(float(pred["metrics"]["instance_mean_matched_iou"]) + 1.0e-9 < float(start["metrics"]["instance_mean_matched_iou"])),
            "inside_gt_removed_pixels": int(np.sum(spatial["inside_gt"])),
            "outside_gt_removed_pixels": int(np.sum(spatial["outside_gt"])),
            "far_inside_gt_removed_pixels": int(far_inside_gt_removed_pixels),
            "largest_destructive_removal_component": int(largest_component),
        }
        case_row["failure_class"] = classify_topology_failure_case(case_row)
        rows.append(case_row)
    return rows


def exact_target_oracle(cached_records: list[dict[str, Any]]) -> dict[str, Any]:
    predicted_metrics: list[dict[str, Any]] = []
    positive_pred: list[dict[str, Any]] = []
    negative_regressions = 0
    negative_topology_changes = 0
    for row in cached_records:
        candidate = row["candidate_mask_np"].astype(bool)
        target = _as_target_mask(row)
        refined = (candidate & (~target)).astype(np.uint8)
        pred = bridge.run_locked_reconstruction(refined, row["gt_instances"])
        start = row["start_reconstruction"]
        metrics = {"sample_id": str(row["sample_id"]), "gt_count": int(row["gt_count"]), **pred["metrics"]}
        predicted_metrics.append(metrics)
        if int(row["bridge_positive"]) == 1:
            positive_pred.append(metrics)
        else:
            if bool(pred["metrics"]["all_iou_ge_0.50"]) != bool(start["metrics"]["all_iou_ge_0.50"]) or abs(float(pred["metrics"]["instance_mean_matched_iou"]) - float(start["metrics"]["instance_mean_matched_iou"])) > 0.0:
                negative_regressions += 1
            if int(pred["pred_k"]) != int(start["pred_k"]):
                negative_topology_changes += 1
    if negative_regressions != 0 or negative_topology_changes != 0:
        raise SystemExit(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "exact_target_oracle_zero_target_not_identical_to_closed",
                    "negative_regressions": int(negative_regressions),
                    "negative_topology_changes": int(negative_topology_changes),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return {
        "overall": bridge._subset_reconstruction_summary(predicted_metrics),
        "positive": bridge._subset_reconstruction_summary(positive_pred),
        "negative_regressions": int(negative_regressions),
        "negative_topology_changes": int(negative_topology_changes),
        "zero_target_identical_to_closed": True,
    }


def pareto_frontier(train_history: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in train_history:
        rows.append(
            {
                "epoch": int(row["epoch"]),
                "positive_success50": int(row["train_positive_success50"]),
                "positive_mean_matched_iou": float(row["train_positive_mean_matched_iou"]),
                "negative_regressions": int(row["train_negative_regressions"]),
                "negative_topology_changes": int(row["train_negative_topology_changes"]),
            }
        )
    frontier: list[dict[str, Any]] = []
    for item in rows:
        dominated = False
        for other in rows:
            if other["epoch"] == item["epoch"]:
                continue
            better_or_equal = (
                other["positive_success50"] >= item["positive_success50"]
                and other["positive_mean_matched_iou"] >= item["positive_mean_matched_iou"]
                and other["negative_regressions"] <= item["negative_regressions"]
                and other["negative_topology_changes"] <= item["negative_topology_changes"]
            )
            strictly_better = (
                other["positive_success50"] > item["positive_success50"]
                or other["positive_mean_matched_iou"] > item["positive_mean_matched_iou"]
                or other["negative_regressions"] < item["negative_regressions"]
                or other["negative_topology_changes"] < item["negative_topology_changes"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(item)
    closed_ref = v6.DEVELOPMENT_REFERENCES["closed"]
    useful_epochs = [
        row
        for row in rows
        if int(row["positive_success50"]) > int(closed_ref["positive_success50"])
        and float(row["positive_mean_matched_iou"]) > float(closed_ref["positive_mean_matched_iou"])
    ]
    safe_negative_epochs = [
        row
        for row in rows
        if int(row["negative_regressions"]) <= int(v6.V6_SAFETY_MAX_NEGATIVE_REGRESSIONS)
        and int(row["negative_topology_changes"]) <= int(v6.V6_SAFETY_MAX_NEGATIVE_TOPOLOGY_CHANGES)
    ]
    safe_useful_epochs = [
        row
        for row in useful_epochs
        if int(row["negative_regressions"]) <= int(v6.V6_SAFETY_MAX_NEGATIVE_REGRESSIONS)
        and int(row["negative_topology_changes"]) <= int(v6.V6_SAFETY_MAX_NEGATIVE_TOPOLOGY_CHANGES)
    ]
    if safe_useful_epochs:
        interpretation = "at_least_one_already_evaluated_epoch_was_useful_and_within_declared_safety_bounds"
    elif useful_epochs:
        interpretation = "destructive_removal_present_across_the_useful_region_of_the_training_trajectory"
    else:
        interpretation = "no_epoch_surpassed_closed_positive_utility"
    return {
        "all_epochs": rows,
        "pareto_frontier": sorted(frontier, key=lambda item: int(item["epoch"])),
        "any_positive_improved_epoch": bool(useful_epochs),
        "any_safe_negative_epoch": bool(safe_negative_epochs),
        "safe_useful_epoch_exists": bool(safe_useful_epochs),
        "interpretation": interpretation,
    }


def _compare_prob_set(cached_records: list[dict[str, Any]], probs: np.ndarray) -> dict[str, Any]:
    target_tuples: list[tuple[int, int, int]] = []
    inside_off_target = 0
    outside_off_target = 0
    positive_successes = 0
    negative_regressions = 0
    negative_topology_changes = 0
    negative_total = 0
    positive_total = 0
    for idx, row in enumerate(cached_records):
        pred_remove = probs[idx, 0] >= FIXED_REMOVE_THRESHOLD
        spatial = spatial_damage_partition(row, pred_remove)
        bridge_target = _as_target_mask(row)
        tp = int(np.sum(spatial["pred_remove_candidate"] & bridge_target))
        fp = int(np.sum(spatial["pred_remove_candidate"] & (~bridge_target)))
        fn = int(np.sum((~spatial["pred_remove_candidate"]) & bridge_target & row["candidate_mask_np"].astype(bool)))
        target_tuples.append((tp, fp, fn))
        inside_off_target += int(np.sum(spatial["inside_gt"] & (~bridge_target)))
        outside_off_target += int(np.sum(spatial["outside_gt"] & (~bridge_target)))
        refined = (row["candidate_mask_np"].astype(bool) & (~spatial["pred_remove_candidate"])).astype(np.uint8)
        pred = bridge.run_locked_reconstruction(refined, row["gt_instances"])
        start = row["start_reconstruction"]
        if int(row["bridge_positive"]) == 1:
            positive_total += 1
            positive_successes += int(bool(pred["metrics"]["all_iou_ge_0.50"]))
        else:
            negative_total += 1
            negative_regressions += int(float(pred["metrics"]["instance_mean_matched_iou"]) + 1.0e-9 < float(start["metrics"]["instance_mean_matched_iou"]))
            negative_topology_changes += int(int(pred["pred_k"]) != int(start["pred_k"]))
    tp = int(sum(item[0] for item in target_tuples))
    fp = int(sum(item[1] for item in target_tuples))
    fn = int(sum(item[2] for item in target_tuples))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float((2 * tp) / max(2 * tp + fp + fn, 1))
    return {
        "true_bridge_precision": precision,
        "true_bridge_recall": recall,
        "true_bridge_f1": f1,
        "off_target_inside_gt_removal": int(inside_off_target),
        "off_target_outside_gt_removal": int(outside_off_target),
        "positive_success50": int(positive_successes),
        "positive_total": int(positive_total),
        "negative_regressions": int(negative_regressions),
        "negative_regression_rate": float(negative_regressions / max(negative_total, 1)),
        "negative_topology_changes": int(negative_topology_changes),
        "negative_topology_change_rate": float(negative_topology_changes / max(negative_total, 1)),
    }


def compare_historical_v2_vs_v6(cached_records: list[dict[str, Any]], v6_probs: np.ndarray, v2_probs: np.ndarray) -> dict[str, Any]:
    return {
        "historical_v2_open": _compare_prob_set(cached_records, v2_probs),
        "v6_open": _compare_prob_set(cached_records, v6_probs),
    }


def interpret_historical_v2_vs_v6(compare_payload: dict[str, Any]) -> dict[str, str]:
    v2_metrics = compare_payload["historical_v2_open"]
    v6_metrics = compare_payload["v6_open"]
    recall_delta = float(v6_metrics["true_bridge_recall"]) - float(v2_metrics["true_bridge_recall"])
    precision_delta = float(v6_metrics["true_bridge_precision"]) - float(v2_metrics["true_bridge_precision"])
    inside_delta = int(v6_metrics["off_target_inside_gt_removal"]) - int(v2_metrics["off_target_inside_gt_removal"])
    outside_delta = int(v6_metrics["off_target_outside_gt_removal"]) - int(v2_metrics["off_target_outside_gt_removal"])
    if recall_delta > max(precision_delta, 0.05) and inside_delta >= 0 and outside_delta >= 0:
        main_improvement = "bridge recall with higher removal aggressiveness"
    elif precision_delta > max(recall_delta, 0.05):
        main_improvement = "bridge precision"
    elif inside_delta < 0 and outside_delta < 0:
        main_improvement = "spatial localization"
    else:
        main_improvement = "removal aggressiveness"
    main_unchanged_failure = "inside-GT destructive negative removal" if int(v6_metrics["negative_regressions"]) > 0 else "negative safety remained acceptable"
    return {
        "main_improvement": main_improvement,
        "main_unchanged_failure": main_unchanged_failure,
    }


def patient_level_failure_concentration(cached_records: list[dict[str, Any]], probs: np.ndarray) -> dict[str, Any]:
    rows_by_patient: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(cached_records):
        patient_id = str(row["patient_id"])
        current = rows_by_patient.setdefault(
            patient_id,
            {
                "patient_id": patient_id,
                "positive_samples": 0,
                "positive_successes": 0,
                "negative_samples": 0,
                "negative_regressions": 0,
                "negative_topology_changes": 0,
                "inside_gt_removed_fraction_values": [],
            },
        )
        pred_remove = probs[idx, 0] >= FIXED_REMOVE_THRESHOLD
        spatial = spatial_damage_partition(row, pred_remove)
        refined = (row["candidate_mask_np"].astype(bool) & (~spatial["pred_remove_candidate"])).astype(np.uint8)
        pred = bridge.run_locked_reconstruction(refined, row["gt_instances"])
        start = row["start_reconstruction"]
        total_removed = int(np.sum(spatial["pred_remove_candidate"]))
        inside_removed = int(np.sum(spatial["inside_gt"]))
        current["inside_gt_removed_fraction_values"].append(float(inside_removed / max(total_removed, 1)))
        if int(row["bridge_positive"]) == 1:
            current["positive_samples"] += 1
            current["positive_successes"] += int(bool(pred["metrics"]["all_iou_ge_0.50"]))
        else:
            current["negative_samples"] += 1
            current["negative_regressions"] += int(float(pred["metrics"]["instance_mean_matched_iou"]) + 1.0e-9 < float(start["metrics"]["instance_mean_matched_iou"]))
            current["negative_topology_changes"] += int(int(pred["pred_k"]) != int(start["pred_k"]))
    rows: list[dict[str, Any]] = []
    for patient_id, row in sorted(rows_by_patient.items()):
        rows.append(
            {
                "patient_id": patient_id,
                "positive_samples": int(row["positive_samples"]),
                "positive_successes": int(row["positive_successes"]),
                "negative_samples": int(row["negative_samples"]),
                "negative_regressions": int(row["negative_regressions"]),
                "negative_topology_changes": int(row["negative_topology_changes"]),
                "mean_inside_gt_removed_fraction": float(np.mean(row["inside_gt_removed_fraction_values"])) if row["inside_gt_removed_fraction_values"] else 0.0,
            }
        )
    return {"rows": rows}


def classify_dominant_failure(
    *,
    score_partition_train: dict[str, Any],
    exact_target_oracle_val: dict[str, Any],
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    labels: list[str] = []
    true_bridge = score_partition_train["TRUE_BRIDGE"]
    positive_non_bridge = score_partition_train["POSITIVE_SAMPLE_NON_BRIDGE"]
    zero_target = score_partition_train["ZERO_TARGET_SAMPLE"]
    if float(true_bridge["median"]) <= max(float(zero_target["p95"]), float(positive_non_bridge["p95"])):
        labels.append("A_SCORE_SEPARABILITY_FAILURE")
    if not bool(exact_target_oracle_val["zero_target_identical_to_closed"]) or int(exact_target_oracle_val["positive"]["all_iou_ge_0.50_count"]) <= int(v6.DEVELOPMENT_REFERENCES["closed"]["positive_success50"]):
        labels.append("B_SPATIAL_TARGET_FAILURE")
    if bool(trajectory["any_positive_improved_epoch"]) and bool(trajectory["any_safe_negative_epoch"]) and not bool(trajectory["safe_useful_epoch_exists"]):
        labels.append("D_CHECKPOINT_TRAJECTORY_CONFLICT")
    if not labels:
        labels.append("C_CONTEXT_FAILURE")
    if "C_CONTEXT_FAILURE" in labels or "A_SCORE_SEPARABILITY_FAILURE" in labels:
        recommended_family = "context-conditioned separation head"
    elif "B_SPATIAL_TARGET_FAILURE" in labels:
        recommended_family = "explicit inter-instance separation/boundary head"
    else:
        recommended_family = "instance embedding / affinity head"
    return {
        "dominant_failure_class": labels,
        "recommended_representation_family": recommended_family,
    }
