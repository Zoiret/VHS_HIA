from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from audit_micro_reconstruction_contract import (
    _annotate_panel,
    _build_loader,
    _build_model_from_cfg,
    _center_prob_to_bgr,
    _draw_markers,
    _load_gt_instance,
    _make_device,
    _marker_contract,
    _read_yaml,
    _resolve_path,
    _resize_panel,
    _sample_center_metrics,
    _seed_all,
)
from compare_reconstruction_policies import (
    PRIMARY_THRESHOLD,
    _assert_policy_contract,
    _labels_to_bgr,
    _overlay_component_ids,
    _policy_metrics,
    _write_csv_atomic,
    _write_json_atomic,
    reconstruct_policy_componentwise,
    run_policy,
)
from validate_centerhead import (
    _connected_components,
    _extract_metadata_centers,
    _fallback_marker,
    _geometry_topo_u8,
    _keep_top3_by_area,
    _markers_from_center_map,
    _watershed,
)
from validate_reconstruction_policies_holdout import (
    AUTHORITATIVE_BEST_CHECKPOINT_SHA256,
    AUTHORITATIVE_SEMANTIC_CHECKPOINT_SHA256,
    DEFAULT_OUTPUT_DIR as HOLDOUT_OUTPUT_DIR,
    EXPECTED_HOLDOUT_MANIFEST_SHA256,
    _canonical_manifest_stdout_payload,
    _aggregate_rows,
    _build_loader_split_file,
    _checkpoint_identity,
    _conditioned_evidence_summary,
    _expected_manifest_identity_sha,
    _inventory_holdout_samples,
    _manifest_identity_status,
    _overall_authoritative_status,
    _safe_git_commit,
    _safe_hostname,
    _scope_invariant_summary,
    _semantic_cc_bucket,
    _semantic_checkpoint_identity,
    _write_holdout_manifest,
)


DEFAULT_OUTPUT_DIR = "training/analysis/centerhead_spatial_x2_2_center_generalization_holdout_diagnosis"
DIAGNOSTIC_THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90)
POLICIES = ("P0_CURRENT", "P1_DROP_UNMARKED")
SCOPES = ("end_to_end", "center_oracle", "full_oracle")
NO_TRAINING_OCCURRED = True
PRODUCTION_FILES_UNCHANGED = True


def _gt_marker_points(gt_pts: list[tuple[int, int]]) -> list[dict]:
    return [{"marker_id": int(idx), "y": int(y), "x": int(x), "score": 1.0} for idx, (y, x) in enumerate(gt_pts, start=1)]


def _marker_labels_from_points(shape: tuple[int, int], marker_points: list[dict]) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.uint8)
    for mp in marker_points:
        y = int(mp["y"])
        x = int(mp["x"])
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            labels[y, x] = np.uint8(int(mp["marker_id"]))
    return labels


def _run_p0_with_explicit_markers(pred_sem: np.ndarray, marker_points: list[dict]) -> tuple[np.ndarray, dict]:
    leaf_union = pred_sem == 1
    labels_cc, cc_k = _connected_components(leaf_union.astype(np.uint8))
    pred_inst = np.zeros_like(pred_sem, dtype=np.uint8)
    next_lab = 1
    component_traces = []
    for comp_id in range(1, int(cc_k) + 1):
        comp01 = labels_cc == comp_id
        in_markers_initial = [(int(mp["y"]), int(mp["x"])) for mp in marker_points if bool(comp01[int(mp["y"]), int(mp["x"])])]
        in_markers = list(in_markers_initial)
        fallback_marker = None
        used_fallback = False
        watershed_local_count_before_keep = 0
        watershed_local_count_after_keep = 0
        output_labels = []
        path = "single_label"
        if len(in_markers) == 0:
            fb = _fallback_marker(comp01)
            if fb is not None:
                fallback_marker = (int(fb[0]), int(fb[1]))
                in_markers = [fallback_marker]
                used_fallback = True
        if len(in_markers) <= 1:
            pred_inst[comp01] = np.uint8(next_lab)
            output_labels = [int(next_lab)]
            next_lab += 1
        else:
            path = "watershed"
            seg = _watershed(comp01.astype(np.uint8), in_markers, _geometry_topo_u8(comp01.astype(np.uint8)))
            watershed_local_count_before_keep = int(seg.max())
            seg, seg_k = _keep_top3_by_area(seg.astype(np.uint8))
            watershed_local_count_after_keep = int(seg_k)
            if seg_k <= 1:
                path = "watershed_collapsed"
                pred_inst[comp01] = np.uint8(next_lab)
                output_labels = [int(next_lab)]
                next_lab += 1
            else:
                for local in range(1, int(seg_k) + 1):
                    pred_inst[seg == local] = np.uint8(next_lab)
                    output_labels.append(int(next_lab))
                    next_lab += 1
        component_traces.append(
            {
                "component_id": int(comp_id),
                "area": int(np.sum(comp01)),
                "marker_count_before_fallback": int(len(in_markers_initial)),
                "marker_count_after_fallback": int(len(in_markers)),
                "markers_before_fallback": [{"y": int(y), "x": int(x)} for (y, x) in in_markers_initial],
                "markers_used": [{"y": int(y), "x": int(x)} for (y, x) in in_markers],
                "used_fallback": bool(used_fallback),
                "fallback_marker": ({"y": int(fallback_marker[0]), "x": int(fallback_marker[1])} if fallback_marker is not None else None),
                "path": path,
                "watershed_local_count_before_keep": int(watershed_local_count_before_keep),
                "watershed_local_count_after_keep": int(watershed_local_count_after_keep),
                "output_labels": [int(v) for v in output_labels],
            }
        )
    raw_inst = pred_inst.copy()
    raw_k = int(raw_inst.max())
    final_inst, final_k = _keep_top3_by_area(pred_inst)
    marker_ids = {int(mp["marker_id"]) for mp in marker_points}
    trace = {
        "leaf_union": leaf_union.astype(np.uint8),
        "semantic_components": labels_cc.astype(np.int32),
        "semantic_component_count": int(cc_k),
        "raw_labels": raw_inst.astype(np.uint8),
        "raw_count": int(raw_k),
        "final_labels": final_inst.astype(np.uint8),
        "final_count": int(final_k),
        "component_assignments": component_traces,
        "labels_without_marker_provenance": [int(v) for v in np.unique(raw_inst) if int(v) > 0 and int(v) not in marker_ids],
        "merged_markers": [],
        "fallback_marker_calls": int(sum(1 for comp in component_traces if bool(comp["used_fallback"]))),
        "keep_top3_call_count": 1 if int(raw_k) != int(final_k) else 0,
        "new_non_marker_label_count": int(len([int(v) for v in np.unique(raw_inst) if int(v) > 0 and int(v) not in marker_ids])),
        "marker_labels": _marker_labels_from_points(pred_sem.shape[:2], marker_points),
    }
    return final_inst.astype(np.uint8), trace


def _run_policy_with_explicit_markers(policy: str, pred_sem: np.ndarray, marker_points: list[dict]) -> tuple[np.ndarray, dict]:
    leaf_union = (pred_sem == 1).astype(np.uint8)
    if policy == "P0_CURRENT":
        return _run_p0_with_explicit_markers(pred_sem, marker_points)
    if policy == "P1_DROP_UNMARKED":
        pred_inst, trace = reconstruct_policy_componentwise(
            leaf_union=leaf_union,
            marker_points=marker_points,
            drop_unmarked=True,
            attach_unmarked=False,
        )
        return pred_inst.astype(np.uint8), trace
    raise ValueError(f"Unsupported explicit-marker policy: {policy}")


def _row_from_scope(
    *,
    sample_entry: dict,
    scope: str,
    policy: str,
    marker_source: str,
    semantic_source: str,
    gt_pts: list[tuple[int, int]],
    marker_points: list[dict],
    gt_inst: np.ndarray,
    pred_sem: np.ndarray,
    pred_inst: np.ndarray,
    trace: dict,
) -> dict:
    marker_contract = _marker_contract(gt_inst, marker_points)
    center_metrics = _sample_center_metrics([(int(mp["y"]), int(mp["x"])) for mp in marker_points], gt_pts)
    policy_metrics = _policy_metrics(
        policy_name=policy,
        gt_inst=gt_inst,
        pred_sem=pred_sem,
        pred_inst=pred_inst,
        marker_points=marker_points,
        trace=trace,
    )
    _assert_policy_contract(policy, policy_metrics)
    row = {
        "scope": scope,
        "sample": str(sample_entry["sample"]),
        "sample_index": int(sample_entry["sample_index"]),
        "split": str(sample_entry["split"]),
        "mouse_id": str(sample_entry["mouse_id"]),
        "marker_source": marker_source,
        "semantic_source": semantic_source,
        "policy": policy,
        "threshold": float(PRIMARY_THRESHOLD),
        "gt_instance_count": int(policy_metrics["counts"]["gt_instance_count"]),
        "marker_count": int(policy_metrics["counts"]["marker_count"]),
        "marker_contract_pass": bool(marker_contract["marker_contract_pass"]),
        "semantic_cc_count": int(policy_metrics["counts"]["semantic_connected_component_count"]),
        "final_output_label_count": int(policy_metrics["counts"]["final_output_label_count"]),
        "output_count_error": int(policy_metrics["counts"]["final_output_label_count"] - policy_metrics["counts"]["gt_instance_count"]),
        "exact_count": bool(policy_metrics["counts"]["exact_count"]),
        "matched_iou": float(policy_metrics["instance_metrics"]["matched_iou"]),
        "mean_matched_dice": policy_metrics["instance_metrics"]["mean_matched_dice"],
        "fragmented": bool(policy_metrics["instance_metrics"]["fragmented"]),
        "merged": bool(policy_metrics["instance_metrics"]["merged"]),
        "mixed": bool(policy_metrics["instance_metrics"]["mixed"]),
        "assigned_area_fraction": float(policy_metrics["area_accounting"]["assigned_area_fraction"]),
        "dropped_area_fraction": float(policy_metrics["area_accounting"]["dropped_area"] / max(int(policy_metrics["area_accounting"]["semantic_leaflet_area"]), 1)),
        "invariant_pass": bool(policy_metrics["contract"]["pass"]),
        "fallback_marker_calls": int(policy_metrics["contract"]["fallback_marker_calls"]),
        "keep_top3_call_count": int(policy_metrics["contract"]["keep_top3_call_count"]),
        "raw_labels_without_marker_provenance": int(policy_metrics["contract"]["raw_labels_without_marker_provenance"]),
        "final_labels_without_marker_provenance": int(policy_metrics["contract"]["final_labels_without_marker_provenance"]),
        "ambiguous_assignments": 0 if policy_metrics["component_assignment"]["ambiguous_assignments"] is None else int(policy_metrics["component_assignment"]["ambiguous_assignments"]),
        "markers_preserved": bool(len(policy_metrics["contract"]["markers_without_output_label"]) == 0),
        "output_count_over_marker_count": bool(int(policy_metrics["counts"]["final_output_label_count"]) > int(policy_metrics["counts"]["marker_count"])),
        "center_precision": float(center_metrics["center_precision"]),
        "center_recall": float(center_metrics["center_recall"]),
        "center_f1": float(center_metrics["center_f1"]),
    }
    return row


def _scores_at_gt_locations(center_prob: np.ndarray, gt_pts: list[tuple[int, int]]) -> list[float]:
    out = []
    for y, x in gt_pts:
        if 0 <= int(y) < center_prob.shape[0] and 0 <= int(x) < center_prob.shape[1]:
            out.append(float(center_prob[int(y), int(x)]))
    return out


def _center_diag_row(
    *,
    sample_entry: dict,
    threshold: float,
    pred_sem: np.ndarray,
    center_prob: np.ndarray,
    gt_inst: np.ndarray,
    gt_pts: list[tuple[int, int]],
) -> dict:
    leaf_union = pred_sem == 1
    markers_scored = _markers_from_center_map(center_prob.astype(np.float32), leaf_union.astype(bool), float(threshold), max_markers=3)
    marker_points = _gt_marker_points([(int(y), int(x)) for (y, x, _score) in markers_scored])
    contract = _marker_contract(gt_inst, marker_points)
    center_metrics = _sample_center_metrics([(int(mp["y"]), int(mp["x"])) for mp in marker_points], gt_pts)
    tp = int(len(center_metrics["center_matches"]))
    fp = int(center_metrics["predicted_center_count"]) - tp
    fn = int(center_metrics["gt_center_count"]) - tp
    far_mask = gt_inst == 0
    return {
        "sample": str(sample_entry["sample"]),
        "sample_index": int(sample_entry["sample_index"]),
        "split": str(sample_entry["split"]),
        "threshold": float(threshold),
        "gt_instance_count": int(len(gt_pts)),
        "predicted_count": int(center_metrics["predicted_center_count"]),
        "exact_center_count": bool(center_metrics["center_count_accuracy"] >= 1.0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "center_precision": float(center_metrics["center_precision"]),
        "center_recall": float(center_metrics["center_recall"]),
        "center_f1": float(center_metrics["center_f1"]),
        "localization_error_px": float(center_metrics["center_loc_err_px"]),
        "marker_contract_pass": bool(contract["marker_contract_pass"]),
        "missing_gt_instances": int(contract["missing_gt_instance_markers"]),
        "duplicate_instances": int(contract["multiple_markers_inside_gt_instances"]),
        "outside_markers": int(contract["markers_outside_all_gt_instances"]),
        "center_scores_at_gt_locations": json.dumps(_scores_at_gt_locations(center_prob, gt_pts)),
        "maximum_far_background_score": float(np.max(center_prob[far_mask])) if bool(np.any(far_mask)) else None,
        "semantic_cc_count": int(_connected_components((pred_sem == 1).astype(np.uint8))[1]),
    }


def _aggregate_center_rows(rows: list[dict], threshold: float) -> dict:
    tp = int(sum(int(row["tp"]) for row in rows))
    fp = int(sum(int(row["fp"]) for row in rows))
    fn = int(sum(int(row["fn"]) for row in rows))
    gt_count_buckets = defaultdict(list)
    for row in rows:
        gt_count_buckets[int(row["gt_instance_count"])].append(row)
    return {
        "threshold": float(threshold),
        "center_precision": float(tp / max(tp + fp, 1)),
        "center_recall": float(tp / max(tp + fn, 1)),
        "center_f1": float((2.0 * tp) / max(2 * tp + fp + fn, 1)),
        "predicted_center_count_mean": float(np.mean([float(row["predicted_count"]) for row in rows])) if rows else None,
        "predicted_center_count_median": float(np.median([float(row["predicted_count"]) for row in rows])) if rows else None,
        "exact_center_count_accuracy": float(np.mean([1.0 if bool(row["exact_center_count"]) else 0.0 for row in rows])) if rows else None,
        "strict_marker_contract_pass_count": int(sum(1 for row in rows if bool(row["marker_contract_pass"]))),
        "strict_marker_contract_pass_rate": float(np.mean([1.0 if bool(row["marker_contract_pass"]) else 0.0 for row in rows])) if rows else None,
        "missing_gt_instances": int(sum(int(row["missing_gt_instances"]) for row in rows)),
        "gt_instances_with_multiple_markers": int(sum(int(row["duplicate_instances"]) for row in rows)),
        "markers_outside_all_gt_instances": int(sum(int(row["outside_markers"]) for row in rows)),
        "localization_error_px": float(np.mean([float(row["localization_error_px"]) for row in rows])) if rows else None,
        "sample_count_gt1": int(len(gt_count_buckets.get(1, []))),
        "sample_count_gt2": int(len(gt_count_buckets.get(2, []))),
        "sample_count_gt3": int(len(gt_count_buckets.get(3, []))),
        "pass_count_gt1": int(sum(1 for row in gt_count_buckets.get(1, []) if bool(row["marker_contract_pass"]))),
        "pass_count_gt2": int(sum(1 for row in gt_count_buckets.get(2, []) if bool(row["marker_contract_pass"]))),
        "pass_count_gt3": int(sum(1 for row in gt_count_buckets.get(3, []) if bool(row["marker_contract_pass"]))),
    }


def _paired_scope_delta(rows: list[dict], left_scope: str, right_scope: str) -> list[dict]:
    out = []
    for policy in POLICIES:
        left = {(row["sample"], row["policy"]): row for row in rows if row["scope"] == left_scope and row["policy"] == policy}
        right = {(row["sample"], row["policy"]): row for row in rows if row["scope"] == right_scope and row["policy"] == policy}
        for key in sorted(set(left) & set(right)):
            lrow = left[key]
            rrow = right[key]
            out.append(
                {
                    "sample": str(lrow["sample"]),
                    "policy": policy,
                    "left_scope": left_scope,
                    "right_scope": right_scope,
                    "gt_instance_count": int(lrow["gt_instance_count"]),
                    "semantic_cc_count": int(lrow["semantic_cc_count"]),
                    "exact_count_delta": int(bool(rrow["exact_count"])) - int(bool(lrow["exact_count"])),
                    "matched_iou_delta": float(rrow["matched_iou"]) - float(lrow["matched_iou"]),
                    "dice_delta": (
                        None
                        if lrow["mean_matched_dice"] is None or rrow["mean_matched_dice"] is None
                        else float(rrow["mean_matched_dice"]) - float(lrow["mean_matched_dice"])
                    ),
                    "assigned_area_delta": float(rrow["assigned_area_fraction"]) - float(lrow["assigned_area_fraction"]),
                    "dropped_area_delta": float(rrow["dropped_area_fraction"]) - float(lrow["dropped_area_fraction"]),
                }
            )
    return out


def _oracle_scope_summary(scope_rows: list[dict]) -> dict:
    summary = {}
    stratified = {}
    for scope in SCOPES:
        summary[scope] = {}
        stratified[scope] = {}
        scope_only = [row for row in scope_rows if row["scope"] == scope]
        for policy in POLICIES:
            policy_rows = [row for row in scope_only if row["policy"] == policy]
            summary[scope][policy] = _aggregate_rows(policy_rows)
            grouped = defaultdict(list)
            for row in policy_rows:
                grouped[("gt_instance_count", str(row["gt_instance_count"]))].append(row)
                grouped[("semantic_cc_bucket", _semantic_cc_bucket(int(row["semantic_cc_count"])))].append(row)
            stratified[scope][policy] = {
                f"{kind}:{value}": _aggregate_rows(group_rows)
                for (kind, value), group_rows in sorted(grouped.items())
            }
    return {"scope_results": summary, "stratified": stratified}


def _p0_count_confusion(rows: list[dict]) -> list[dict]:
    counts = Counter(
        (int(row["gt_instance_count"]), int(row["final_output_label_count"]))
        for row in rows
        if row["scope"] == "end_to_end" and row["policy"] == "P0_CURRENT"
    )
    out = []
    for (gt_count, pred_count), sample_count in sorted(counts.items()):
        out.append(
            {
                "gt_instance_count": int(gt_count),
                "p0_final_output_count": int(pred_count),
                "sample_count": int(sample_count),
            }
        )
    return out


def _classify_bottleneck(scope_summary: dict, full_oracle_invariants: dict) -> dict:
    p1_a = scope_summary["scope_results"]["end_to_end"]["P1_DROP_UNMARKED"]
    p1_b = scope_summary["scope_results"]["center_oracle"]["P1_DROP_UNMARKED"]
    p1_c = scope_summary["scope_results"]["full_oracle"]["P1_DROP_UNMARKED"]
    if int(full_oracle_invariants["P1_DROP_UNMARKED"]["all_samples_invariant_violations"]) > 0 or int(
        full_oracle_invariants["P1_DROP_UNMARKED"]["output_count_over_marker_count"]
    ) > 0:
        status = "reconstruction_policy_bug"
    else:
        exact_gain = (p1_b["exact_count_accuracy"] or 0.0) - (p1_a["exact_count_accuracy"] or 0.0)
        iou_gain = (p1_b["mean_matched_iou"] or 0.0) - (p1_a["mean_matched_iou"] or 0.0)
        semantic_gap = (p1_c["mean_matched_iou"] or 0.0) - (p1_b["mean_matched_iou"] or 0.0)
        high_center_oracle_count = (p1_b["exact_count_accuracy"] or 0.0) >= 0.9
        center_oracle_clean = int(p1_b["invariant_violation_count"]) == 0
        if exact_gain >= 0.2 and center_oracle_clean and high_center_oracle_count:
            if semantic_gap >= 0.05 or (p1_b["dropped_area_fraction"] or 0.0) >= 0.10:
                status = "mixed_center_and_semantic_failure"
            else:
                status = "center_branch_primary_bottleneck"
        elif high_center_oracle_count and ((p1_b["mean_matched_iou"] or 0.0) < 0.8 or (p1_b["dropped_area_fraction"] or 0.0) >= 0.10):
            status = "semantic_fragmentation_primary_bottleneck"
        else:
            status = "mixed_center_and_semantic_failure"
    return {
        "status": status,
        "evidence": {
            "p1_end_to_end_exact_count": p1_a["exact_count_accuracy"],
            "p1_center_oracle_exact_count": p1_b["exact_count_accuracy"],
            "p1_full_oracle_exact_count": p1_c["exact_count_accuracy"],
            "p1_end_to_end_mean_iou": p1_a["mean_matched_iou"],
            "p1_center_oracle_mean_iou": p1_b["mean_matched_iou"],
            "p1_full_oracle_mean_iou": p1_c["mean_matched_iou"],
            "p1_full_oracle_invariants": full_oracle_invariants["P1_DROP_UNMARKED"],
        },
    }


def _center_failure_key(row: dict) -> tuple:
    return (
        0 if not bool(row["marker_contract_pass"]) else 1,
        -int(row["missing_gt_instances"]),
        -int(row["duplicate_instances"]),
        -int(row["outside_markers"]),
        float(row["center_f1"]),
        -int(row["semantic_cc_count"]),
        str(row["sample"]),
    )


def _center_failure_panel(
    *,
    sample: str,
    image_rgb_u8: np.ndarray,
    gt_inst: np.ndarray,
    gt_marker_points: list[dict],
    pred_marker_points: list[dict],
    pred_sem: np.ndarray,
    center_prob: np.ndarray,
    p0_final: np.ndarray,
    p1_end_final: np.ndarray,
    p1_oracle_final: np.ndarray,
    p0_iou: float,
    p1_end_iou: float,
    p1_oracle_iou: float,
) -> np.ndarray:
    original_bgr = cv2.cvtColor(image_rgb_u8, cv2.COLOR_RGB2BGR)
    heatmap = _draw_markers(_center_prob_to_bgr(center_prob), [{"y": int(mp["y"]), "x": int(mp["x"])} for mp in pred_marker_points])
    gt_marker_vis = _draw_markers(original_bgr.copy(), [{"y": int(mp["y"]), "x": int(mp["x"])} for mp in gt_marker_points])
    pred_marker_vis = _draw_markers(original_bgr.copy(), [{"y": int(mp["y"]), "x": int(mp["x"])} for mp in pred_marker_points])
    panels = [
        _annotate_panel(_resize_panel(original_bgr), f"{sample} image"),
        _annotate_panel(_resize_panel(_labels_to_bgr(gt_inst)), "GT instances"),
        _annotate_panel(_resize_panel(gt_marker_vis), "GT centers"),
        _annotate_panel(_resize_panel(heatmap), "predicted center heatmap"),
        _annotate_panel(_resize_panel(pred_marker_vis), "predicted markers @0.03"),
        _annotate_panel(_resize_panel(_overlay_component_ids(pred_sem, _connected_components((pred_sem == 1).astype(np.uint8))[0])), "predicted semantic"),
        _annotate_panel(_resize_panel(_labels_to_bgr(p0_final)), f"P0 final IoU={p0_iou:.4f}"),
        _annotate_panel(_resize_panel(_labels_to_bgr(p1_end_final)), f"P1 end-to-end IoU={p1_end_iou:.4f}"),
        _annotate_panel(_resize_panel(_labels_to_bgr(p1_oracle_final)), f"P1 center-oracle IoU={p1_oracle_iou:.4f}"),
    ]
    width = max(panel.shape[1] for panel in panels)
    height = max(panel.shape[0] for panel in panels)
    normalized = [cv2.resize(panel, (width, height), interpolation=cv2.INTER_NEAREST) for panel in panels]
    top = np.concatenate(normalized[:3], axis=1)
    mid = np.concatenate(normalized[3:6], axis=1)
    bottom = np.concatenate(normalized[6:9], axis=1)
    return np.concatenate([top, mid, bottom], axis=0)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="training/configs/unetpp_effb3_centerhead_spatial_x2_2_adapter_legacy_fp32_micro.yaml")
    ap.add_argument("--run-dir", type=str, default="training/runs/unetpp_effb3_centerhead_spatial_x2_2_adapter_legacy_fp32_micro")
    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--expected-manifest-identity-sha", type=str, default="")
    ap.add_argument("--allow-manifest-mismatch-for-diagnostics", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _seed_all(1337)
    cfg_path = _resolve_path(repo_root, args.config)
    run_dir = _resolve_path(repo_root, args.run_dir)
    out_dir = _resolve_path(repo_root, args.output_dir)
    if cfg_path is None or run_dir is None or out_dir is None:
        raise SystemExit("Failed to resolve config/run-dir/output-dir")
    cfg = _read_yaml(cfg_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "visual_review").mkdir(parents=True, exist_ok=True)

    inventory = _inventory_holdout_samples(cfg, repo_root)
    manifest_path, identity_manifest_path, manifest_meta_path = _write_holdout_manifest(out_dir, inventory)
    manifest_metadata = json.loads(manifest_meta_path.read_text(encoding="utf-8"))
    identity_entries = [json.loads(line) for line in identity_manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps({"manifest_inspection": _canonical_manifest_stdout_payload(manifest_metadata, identity_entries)}, ensure_ascii=False, indent=2))
    eligible_entries = list(inventory["eligible"])
    if not eligible_entries:
        raise SystemExit(json.dumps({"status": "insufficient_holdout_data"}, ensure_ascii=False, indent=2))

    loader_split = _build_loader_split_file(out_dir, eligible_entries)
    device = _make_device(cfg, args.device)
    loader = _build_loader(cfg, repo_root=repo_root, split_txt=loader_split, device=device)
    model = _build_model_from_cfg(cfg, repo_root=repo_root)
    ckpt_info = _checkpoint_identity(run_dir)
    ckpt_info.update(_semantic_checkpoint_identity(cfg, repo_root))
    ckpt_info["authoritative_checkpoint_sha256"] = AUTHORITATIVE_BEST_CHECKPOINT_SHA256
    ckpt_info["authoritative_checkpoint_match"] = bool(
        str(ckpt_info["checkpoint_sha256"]) == str(AUTHORITATIVE_BEST_CHECKPOINT_SHA256)
        and int(ckpt_info["checkpoint_iteration"]) == 75
    )
    ckpt_info["checkpoint_identity_status"] = "exact_match" if bool(ckpt_info["authoritative_checkpoint_match"]) else "checkpoint_identity_mismatch"
    ckpt_info["iteration_matches_authoritative"] = bool(int(ckpt_info["checkpoint_iteration"]) == 75)
    ckpt_info["semantic_checkpoint_expected_sha256"] = AUTHORITATIVE_SEMANTIC_CHECKPOINT_SHA256
    ckpt_info["semantic_checkpoint_identity_status"] = (
        "exact_match"
        if str(ckpt_info["semantic_checkpoint_sha256"]) == str(AUTHORITATIVE_SEMANTIC_CHECKPOINT_SHA256)
        else "semantic_checkpoint_identity_mismatch"
    )
    ckpt_info["hostname"] = _safe_hostname()["value"]
    ckpt_info["git_commit"] = _safe_git_commit(repo_root)["value"]
    ckpt_info["device"] = str(device)
    ckpt_info["execution_manifest_sha256"] = manifest_metadata["execution_manifest_sha256"]
    ckpt_info["legacy_expected_manifest_sha256"] = EXPECTED_HOLDOUT_MANIFEST_SHA256
    ckpt_info["legacy_execution_manifest_sha_matches_expected"] = bool(
        str(manifest_metadata["execution_manifest_sha256"]) == str(EXPECTED_HOLDOUT_MANIFEST_SHA256)
    )
    ckpt_info["manifest_identity_sha256"] = manifest_metadata["canonical_identity_sha256"]
    ckpt_info["expected_manifest_identity_sha256"] = _expected_manifest_identity_sha(args.expected_manifest_identity_sha)
    ckpt_info["manifest_identity_status"] = _manifest_identity_status(
        actual_sha=str(manifest_metadata["canonical_identity_sha256"]),
        expected_sha=ckpt_info["expected_manifest_identity_sha256"],
        unique_sample_count=int(manifest_metadata["unique_sample_count"]),
        row_count=int(manifest_metadata["manifest_row_count"]),
    )
    ckpt_info["overall_authoritative_status"] = _overall_authoritative_status(
        checkpoint_identity_status=ckpt_info["checkpoint_identity_status"],
        semantic_checkpoint_identity_status=ckpt_info["semantic_checkpoint_identity_status"],
        manifest_identity_status=ckpt_info["manifest_identity_status"],
    )
    ckpt_info["diagnosis_execution_status"] = (
        "running_diagnostics"
        if ckpt_info["overall_authoritative_status"] == "exact_match"
        else "diagnostics_completed_but_authoritative_identity_failed"
    )
    checkpoint_identity_json = {key: value for key, value in ckpt_info.items() if key != "state_dict"}
    _write_json_atomic((out_dir / "checkpoint_identity.json").resolve(), checkpoint_identity_json)

    incompat = model.load_state_dict(ckpt_info["state_dict"], strict=False)
    missing = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
    unexpected = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
    if unexpected or missing:
        raise SystemExit(f"Checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()

    entry_by_sample = {entry["sample"]: dict(entry, sample_index=idx) for idx, entry in enumerate(eligible_entries)}
    center_rows = []
    scope_rows = []
    cached_visuals = {}

    for batch in loader:
        images = batch["image"].to(device)
        out = model(images)
        sample_id = Path(str(batch["image_path"][0])).stem
        entry = entry_by_sample[sample_id]
        pred_sem = torch.argmax(out["semantic"], dim=1).detach().cpu().numpy()[0].astype(np.uint8)
        center_prob = torch.sigmoid(out["center"]).detach().cpu().numpy()[0, 0].astype(np.float32)
        gt_sem = batch["mask"].detach().cpu().numpy()[0].astype(np.uint8)
        gt_inst = _load_gt_instance(Path(inventory["instance_root"]), sample_id, pred_sem.shape[:2])
        gt_pts = _extract_metadata_centers(str(batch["metadata_path"][0]))
        gt_marker_points = _gt_marker_points(gt_pts)
        gt_marker_contract = _marker_contract(gt_inst, gt_marker_points)
        if not bool(gt_marker_contract["marker_contract_pass"]):
            raise SystemExit(
                json.dumps(
                    {
                        "status": "oracle_marker_contract_invalid",
                        "sample": sample_id,
                        "gt_marker_contract": gt_marker_contract,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        image_rgb_u8 = (np.clip(batch["image"].detach().cpu().numpy()[0].transpose(1, 2, 0), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        for threshold in DIAGNOSTIC_THRESHOLDS:
            center_rows.append(
                _center_diag_row(
                    sample_entry=entry,
                    threshold=float(threshold),
                    pred_sem=pred_sem,
                    center_prob=center_prob,
                    gt_inst=gt_inst,
                    gt_pts=gt_pts,
                )
            )

        for policy in POLICIES:
            end_pred_inst, end_marker_points, end_trace = run_policy(policy, pred_sem, center_prob, PRIMARY_THRESHOLD)
            end_row = _row_from_scope(
                sample_entry=entry,
                scope="end_to_end",
                policy=policy,
                marker_source="predicted_centers",
                semantic_source="predicted_semantic",
                gt_pts=gt_pts,
                marker_points=end_marker_points,
                gt_inst=gt_inst,
                pred_sem=pred_sem,
                pred_inst=end_pred_inst,
                trace=end_trace,
            )
            scope_rows.append(end_row)

            center_oracle_pred_inst, center_oracle_trace = _run_policy_with_explicit_markers(policy, pred_sem, gt_marker_points)
            center_oracle_row = _row_from_scope(
                sample_entry=entry,
                scope="center_oracle",
                policy=policy,
                marker_source="gt_centers",
                semantic_source="predicted_semantic",
                gt_pts=gt_pts,
                marker_points=gt_marker_points,
                gt_inst=gt_inst,
                pred_sem=pred_sem,
                pred_inst=center_oracle_pred_inst,
                trace=center_oracle_trace,
            )
            scope_rows.append(center_oracle_row)

            full_oracle_pred_inst, full_oracle_trace = _run_policy_with_explicit_markers(policy, gt_sem, gt_marker_points)
            full_oracle_row = _row_from_scope(
                sample_entry=entry,
                scope="full_oracle",
                policy=policy,
                marker_source="gt_centers",
                semantic_source="gt_semantic",
                gt_pts=gt_pts,
                marker_points=gt_marker_points,
                gt_inst=gt_inst,
                pred_sem=gt_sem,
                pred_inst=full_oracle_pred_inst,
                trace=full_oracle_trace,
            )
            scope_rows.append(full_oracle_row)

            cached_visuals.setdefault(sample_id, {})
            if policy == "P0_CURRENT":
                cached_visuals[sample_id]["p0_final"] = end_pred_inst
                cached_visuals[sample_id]["p0_iou"] = float(end_row["matched_iou"])
            else:
                cached_visuals[sample_id]["p1_end_final"] = end_pred_inst
                cached_visuals[sample_id]["p1_end_iou"] = float(end_row["matched_iou"])
                cached_visuals[sample_id]["p1_oracle_final"] = center_oracle_pred_inst
                cached_visuals[sample_id]["p1_oracle_iou"] = float(center_oracle_row["matched_iou"])
            cached_visuals[sample_id]["image_rgb_u8"] = image_rgb_u8
            cached_visuals[sample_id]["gt_inst"] = gt_inst
            cached_visuals[sample_id]["gt_marker_points"] = gt_marker_points
            cached_visuals[sample_id]["pred_sem"] = pred_sem
            cached_visuals[sample_id]["center_prob"] = center_prob
            cached_visuals[sample_id]["pred_marker_points"] = end_marker_points

    center_rows.sort(key=lambda row: (float(row["threshold"]), int(row["sample_index"])))
    scope_rows.sort(key=lambda row: (str(row["scope"]), int(row["sample_index"]), str(row["policy"])))

    center_summary_rows = [_aggregate_center_rows([row for row in center_rows if abs(float(row["threshold"]) - float(thr)) < 1e-9], thr) for thr in DIAGNOSTIC_THRESHOLDS]
    _write_csv_atomic((out_dir / "center_threshold_summary.csv").resolve(), center_summary_rows)
    _write_csv_atomic((out_dir / "per_sample_center_diagnostics.csv").resolve(), center_rows)
    _write_csv_atomic((out_dir / "per_sample_oracle_policy_metrics.csv").resolve(), scope_rows)

    scope_summary = _oracle_scope_summary(scope_rows)
    end_vs_center = _paired_scope_delta(scope_rows, "end_to_end", "center_oracle")
    _write_csv_atomic((out_dir / "end_to_end_vs_center_oracle.csv").resolve(), end_vs_center)

    full_oracle_invariants = {
        policy: _scope_invariant_summary([row for row in scope_rows if row["scope"] == "full_oracle" and row["policy"] == policy])
        for policy in POLICIES
    }
    _write_json_atomic((out_dir / "full_oracle_invariants.json").resolve(), full_oracle_invariants)

    confusion_rows = _p0_count_confusion(scope_rows)
    _write_csv_atomic((out_dir / "p0_gt_count_confusion.csv").resolve(), confusion_rows)

    worst_center_failures = sorted(
        [row for row in center_rows if abs(float(row["threshold"]) - float(PRIMARY_THRESHOLD)) < 1e-9],
        key=_center_failure_key,
    )[:20]
    _write_csv_atomic((out_dir / "worst_center_failures.csv").resolve(), worst_center_failures)

    bottleneck = _classify_bottleneck(scope_summary, full_oracle_invariants)
    _write_json_atomic((out_dir / "bottleneck_decision.json").resolve(), bottleneck)

    visual_dir = (out_dir / "visual_review").resolve()
    for row in worst_center_failures:
        sample_id = str(row["sample"])
        pack = cached_visuals[sample_id]
        panel = _center_failure_panel(
            sample=sample_id,
            image_rgb_u8=pack["image_rgb_u8"],
            gt_inst=pack["gt_inst"],
            gt_marker_points=pack["gt_marker_points"],
            pred_marker_points=pack["pred_marker_points"],
            pred_sem=pack["pred_sem"],
            center_prob=pack["center_prob"],
            p0_final=pack["p0_final"],
            p1_end_final=pack["p1_end_final"],
            p1_oracle_final=pack["p1_oracle_final"],
            p0_iou=float(pack["p0_iou"]),
            p1_end_iou=float(pack["p1_end_iou"]),
            p1_oracle_iou=float(pack["p1_oracle_iou"]),
        )
        cv2.imwrite(str((visual_dir / f"{sample_id}.png").resolve()), panel)

    oracle_summary_payload = {
        "checkpoint_identity": checkpoint_identity_json,
        "holdout_manifest": str(manifest_path),
        "holdout_manifest_identity": str(identity_manifest_path),
        "holdout_manifest_metadata": str(manifest_meta_path),
        "execution_manifest_sha256": manifest_metadata["execution_manifest_sha256"],
        "canonical_identity_sha256": manifest_metadata["canonical_identity_sha256"],
        "manifest_identity_status": ckpt_info["manifest_identity_status"],
        "overall_authoritative_status": ckpt_info["overall_authoritative_status"],
        "diagnosis_execution_status": ckpt_info["diagnosis_execution_status"],
        **scope_summary,
        "end_to_end_vs_center_oracle": {
            policy: _aggregate_rows([row for row in scope_rows if row["scope"] == "center_oracle" and row["policy"] == policy])
            for policy in POLICIES
        },
        "conditioned_evidence_if_center_oracle_used_for_reconstruction": _conditioned_evidence_summary(
            total_holdout_sample_count=len(eligible_entries),
            conditioned_primary_rows=[
                {
                    **row,
                    "marker_contract_pass": True,
                }
                for row in scope_rows
                if row["scope"] == "center_oracle"
            ],
        ),
    }
    _write_json_atomic((out_dir / "oracle_scope_summary.json").resolve(), oracle_summary_payload)

    print(
        json.dumps(
            {
                "status": "done",
                "output_dir": str(out_dir),
                "host": ckpt_info["hostname"],
                "device": ckpt_info["device"],
                "manifest_sha": manifest_metadata["canonical_identity_sha256"],
                "checkpoint_identity_status": ckpt_info["checkpoint_identity_status"],
                "semantic_checkpoint_identity_status": ckpt_info["semantic_checkpoint_identity_status"],
                "manifest_identity_status": ckpt_info["manifest_identity_status"],
                "overall_authoritative_status": ckpt_info["overall_authoritative_status"],
                "diagnosis_execution_status": ckpt_info["diagnosis_execution_status"],
                "bottleneck_status": bottleneck["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if ckpt_info["overall_authoritative_status"] != "exact_match" and not bool(args.allow_manifest_mismatch_for_diagnostics):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
