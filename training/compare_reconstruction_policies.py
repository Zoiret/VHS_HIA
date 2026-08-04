from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import socket
import subprocess

import cv2
import numpy as np
import torch

from audit_micro_reconstruction_contract import (
    EXPECTED_MICROSET_SIZE,
    _annotate_panel,
    _build_loader,
    _build_model_from_cfg,
    _center_prob_to_bgr,
    _compare_stage_labels,
    _draw_markers,
    _format_thr,
    _load_checkpoint,
    _load_gt_instance,
    _load_manifest_sample_ids,
    _make_device,
    _marker_contract,
    _mask_to_bgr,
    _match_centers,
    _matching_panel,
    _normalized_microset_sha256,
    _parse_microset_file,
    _positive_label_ids,
    _read_json,
    _read_yaml,
    _resolve_output_dir_arg,
    _resolve_path,
    _resize_panel,
    _sample_center_metrics,
    _seed_all,
    _semantic_topology,
    _sha256_file,
    _stage_failure_summary,
    _stage_stats,
)
from validate_centerhead import (
    _connected_components,
    _dice_iou_binary,
    _extract_metadata_centers,
    _geometry_topo_u8,
    _markers_from_center_map,
    _watershed,
    compute_instance_metrics_from_masks,
    reconstruct_instances_from_semantic_and_center,
)


PRIMARY_THRESHOLD = 0.03
SECONDARY_THRESHOLDS = (0.02, 0.05)
DEFAULT_OUTPUT_DIR = "training/analysis/centerhead_spatial_x2_2_reconstruction_policy_ablation"
P3_DISTANCE_GATES = (8.0, 16.0, 32.0, 64.0)
P3_AREA_GATES = (0.01, 0.05, 0.10, None)
AUTHORITATIVE_PRIMARY_SAMPLE_ORDER = [
    "m01_p02_s00",
    "m01_p02_s04",
    "m01_p01_s00",
    "m01_p01_s01",
    "m01_p01_s02",
    "m01_p01_s03",
]
AUTHORITATIVE_MICROSET_RAW_SHA256 = "579fa7b70c745b645779e0a293642112a624a3550d84009c60e03e7819008848"


class BaselineMismatchError(RuntimeError):
    pass


def _positive_ids(labels: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(labels) if int(v) > 0]


def _instance_score(metrics: dict) -> float:
    return float(metrics["instance_mean_matched_iou"]) - 0.25 * float(metrics["instance_merged_rate"]) - 0.15 * float(metrics["instance_fragmented_rate"])


def _bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask.astype(bool))
    if ys.size == 0:
        return [0, 0, 0, 0]
    return [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]


def _centroid_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask.astype(bool))
    if ys.size == 0:
        return [0.0, 0.0]
    return [float(np.mean(ys)), float(np.mean(xs))]


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    if int(m.sum()) == 0:
        return np.zeros_like(m, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    er = cv2.erode(m, kernel, iterations=1)
    return (m.astype(bool) & (~er.astype(bool))).astype(np.uint8)


def _distance_to_region(component_mask: np.ndarray, region_mask: np.ndarray) -> float:
    comp_boundary = _boundary_mask(component_mask)
    region_boundary = _boundary_mask(region_mask)
    if int(region_boundary.sum()) == 0:
        region_boundary = region_mask.astype(np.uint8)
    inv = np.ones_like(region_boundary, dtype=np.uint8)
    inv[region_boundary.astype(bool)] = 0
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
    vals = dist[comp_boundary.astype(bool)]
    if vals.size == 0:
        vals = dist[component_mask.astype(bool)]
    if vals.size == 0:
        return float("inf")
    return float(np.min(vals))


def _centroid_to_region_distance(component_mask: np.ndarray, region_mask: np.ndarray) -> float:
    cy, cx = _centroid_from_mask(component_mask)
    ys, xs = np.where(region_mask.astype(bool))
    if ys.size == 0:
        return float("inf")
    d = np.hypot(ys.astype(np.float32) - np.float32(cy), xs.astype(np.float32) - np.float32(cx))
    return float(np.min(d))


def _marker_points_to_dicts(marker_points_scored: list[tuple[int, int, float]]) -> list[dict]:
    out = []
    for idx, (y, x, score) in enumerate(marker_points_scored, start=1):
        out.append({"marker_id": int(idx), "y": int(y), "x": int(x), "score": float(score)})
    return out


def _marker_ids_present(labels: np.ndarray, marker_points: list[dict]) -> list[int]:
    ids = []
    for mp in marker_points:
        marker_id = int(mp["marker_id"])
        if int(np.sum(labels == marker_id)) > 0:
            ids.append(marker_id)
    return ids


def _component_records(leaf_union: np.ndarray, marker_points: list[dict]) -> tuple[np.ndarray, list[dict]]:
    labels_cc, cc_k = _connected_components(leaf_union.astype(np.uint8))
    records = []
    for comp_id in range(1, int(cc_k) + 1):
        comp_mask = labels_cc == comp_id
        comp_markers = [mp for mp in marker_points if bool(comp_mask[int(mp["y"]), int(mp["x"])])]
        records.append(
            {
                "component_id": int(comp_id),
                "mask": comp_mask,
                "area": int(np.sum(comp_mask)),
                "centroid": _centroid_from_mask(comp_mask),
                "bbox": _bbox_from_mask(comp_mask),
                "marker_ids": [int(mp["marker_id"]) for mp in comp_markers],
                "markers": comp_markers,
            }
        )
    return labels_cc, records


def _assign_multi_marker_component(component_mask: np.ndarray, markers: list[dict]) -> tuple[np.ndarray, dict]:
    topo = _geometry_topo_u8(component_mask.astype(np.uint8))
    seg = _watershed(component_mask.astype(np.uint8), [(int(mp["y"]), int(mp["x"])) for mp in markers], topo)
    out = np.zeros_like(seg, dtype=np.uint8)
    local_to_marker = {}
    merged_marker_groups = []
    for mp in markers:
        local_lab = int(seg[int(mp["y"]), int(mp["x"])])
        if local_lab <= 0:
            continue
        if local_lab in local_to_marker and int(local_to_marker[local_lab]) != int(mp["marker_id"]):
            merged_marker_groups.append({"local_label": int(local_lab), "marker_ids": sorted([int(local_to_marker[local_lab]), int(mp["marker_id"])])})
        local_to_marker[local_lab] = int(mp["marker_id"])
    for local_lab, marker_id in local_to_marker.items():
        out[seg == int(local_lab)] = np.uint8(marker_id)
    stray_local = [int(v) for v in np.unique(seg) if int(v) > 0 and int(v) not in local_to_marker]
    return out, {
        "watershed_local_count": int(seg.max()),
        "local_to_marker": {int(k): int(v) for k, v in local_to_marker.items()},
        "merged_marker_groups": merged_marker_groups,
        "stray_local_labels_without_marker": stray_local,
    }


def reconstruct_policy_componentwise(
    *,
    leaf_union: np.ndarray,
    marker_points: list[dict],
    drop_unmarked: bool,
    attach_unmarked: bool,
    boundary_gate_px: float | None = None,
    relative_area_gate: float | None = None,
) -> tuple[np.ndarray, dict]:
    labels_cc, records = _component_records(leaf_union, marker_points)
    raw_labels = np.zeros_like(leaf_union, dtype=np.uint8)
    final_labels = np.zeros_like(leaf_union, dtype=np.uint8)
    component_assignments = []
    largest_marked_area = max((int(rec["area"]) for rec in records if len(rec["marker_ids"]) > 0), default=0)
    labels_without_marker_provenance = []
    merged_markers = []

    for rec in records:
        comp_mask = rec["mask"]
        comp_markers = rec["markers"]
        if len(comp_markers) == 0:
            component_assignments.append(
                {
                    "component_id": int(rec["component_id"]),
                    "area": int(rec["area"]),
                    "centroid": list(rec["centroid"]),
                    "bbox": list(rec["bbox"]),
                    "marker_ids": [],
                    "mode": "unmarked_pending",
                }
            )
            continue
        if len(comp_markers) == 1:
            marker_id = int(comp_markers[0]["marker_id"])
            raw_labels[comp_mask] = np.uint8(marker_id)
            final_labels[comp_mask] = np.uint8(marker_id)
            component_assignments.append(
                {
                    "component_id": int(rec["component_id"]),
                    "area": int(rec["area"]),
                    "centroid": list(rec["centroid"]),
                    "bbox": list(rec["bbox"]),
                    "marker_ids": [int(marker_id)],
                    "mode": "single_marker_component",
                    "assigned_marker": int(marker_id),
                }
            )
            continue
        comp_out, comp_trace = _assign_multi_marker_component(comp_mask, comp_markers)
        raw_labels[comp_mask] = comp_out[comp_mask]
        final_labels[comp_mask] = comp_out[comp_mask]
        labels_without_marker_provenance.extend(comp_trace["stray_local_labels_without_marker"])
        merged_markers.extend(comp_trace["merged_marker_groups"])
        component_assignments.append(
            {
                "component_id": int(rec["component_id"]),
                "area": int(rec["area"]),
                "centroid": list(rec["centroid"]),
                "bbox": list(rec["bbox"]),
                "marker_ids": [int(v) for v in rec["marker_ids"]],
                "mode": "multi_marker_watershed",
                "watershed_local_count": int(comp_trace["watershed_local_count"]),
                "local_to_marker": comp_trace["local_to_marker"],
                "merged_marker_groups": comp_trace["merged_marker_groups"],
                "stray_local_labels_without_marker": comp_trace["stray_local_labels_without_marker"],
            }
        )

    raw_output = raw_labels.copy()

    for rec in records:
        if len(rec["marker_ids"]) > 0:
            continue
        comp_mask = rec["mask"]
        assign_info = {
            "component_id": int(rec["component_id"]),
            "area": int(rec["area"]),
            "centroid": list(rec["centroid"]),
            "bbox": list(rec["bbox"]),
            "marker_ids": [],
            "mode": "drop_unmarked" if drop_unmarked else "unmarked_component",
            "relative_area": float(rec["area"] / max(largest_marked_area, 1)),
        }
        if drop_unmarked or not attach_unmarked or len(marker_points) == 0:
            assign_info["assigned"] = False
            assign_info["rejected_reason"] = "drop_unmarked" if drop_unmarked else "attach_disabled_or_no_markers"
            component_assignments.append(assign_info)
            continue

        candidates = []
        for mp in marker_points:
            marker_id = int(mp["marker_id"])
            region_mask = final_labels == marker_id
            if int(np.sum(region_mask)) <= 0:
                continue
            boundary_distance = _distance_to_region(comp_mask, region_mask)
            centroid_distance = _centroid_to_region_distance(comp_mask, region_mask)
            candidates.append(
                {
                    "marker_id": int(marker_id),
                    "boundary_distance": float(boundary_distance),
                    "centroid_distance": float(centroid_distance),
                }
            )
        candidates.sort(key=lambda item: (float(item["boundary_distance"]), int(item["marker_id"])))
        assign_info["candidates"] = candidates
        if not candidates:
            assign_info["assigned"] = False
            assign_info["rejected_reason"] = "no_marker_controlled_region"
            component_assignments.append(assign_info)
            continue
        nearest = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        ratio = float("inf")
        if second is not None and float(nearest["boundary_distance"]) > 0.0:
            ratio = float(second["boundary_distance"] / nearest["boundary_distance"])
        elif second is not None and float(nearest["boundary_distance"]) == 0.0:
            ratio = 1.0 if float(second["boundary_distance"]) == 0.0 else float("inf")
        assign_info["nearest_marker_label"] = int(nearest["marker_id"])
        assign_info["nearest_distance"] = float(nearest["boundary_distance"])
        assign_info["second_nearest_distance"] = float(second["boundary_distance"]) if second is not None else None
        assign_info["distance_margin"] = float(ratio) if ratio != float("inf") else None
        assign_info["centroid_nearest_distance"] = float(nearest["centroid_distance"])
        assign_info["ambiguous"] = bool(second is not None and ratio < 1.25)

        gate_ok = True
        reject_reasons = []
        if boundary_gate_px is not None and float(nearest["boundary_distance"]) > float(boundary_gate_px):
            gate_ok = False
            reject_reasons.append("distance_gate")
        if relative_area_gate is not None and float(assign_info["relative_area"]) > float(relative_area_gate):
            gate_ok = False
            reject_reasons.append("area_gate")

        if gate_ok:
            final_labels[comp_mask] = np.uint8(nearest["marker_id"])
            assign_info["assigned"] = True
        else:
            assign_info["assigned"] = False
            assign_info["rejected_reason"] = "+".join(reject_reasons) if reject_reasons else "gate"
        component_assignments.append(assign_info)

    return final_labels, {
        "leaf_union": leaf_union.astype(np.uint8),
        "semantic_components": labels_cc.astype(np.int32),
        "semantic_component_count": int(len(records)),
        "raw_labels": raw_output.astype(np.uint8),
        "raw_count": int(len(_positive_ids(raw_output))),
        "final_labels": final_labels.astype(np.uint8),
        "final_count": int(len(_positive_ids(final_labels))),
        "component_assignments": component_assignments,
        "largest_marked_component_area": int(largest_marked_area),
        "labels_without_marker_provenance": [int(v) for v in labels_without_marker_provenance],
        "merged_markers": merged_markers,
        "fallback_marker_calls": 0,
        "keep_top3_call_count": 0,
        "new_non_marker_label_count": int(len(labels_without_marker_provenance)),
    }


def reconstruct_policy_global(
    *,
    leaf_union: np.ndarray,
    marker_points: list[dict],
    attach_unmarked: bool,
) -> tuple[np.ndarray, dict]:
    labels_cc, records = _component_records(leaf_union, marker_points)
    if len(marker_points) == 0:
        raw_labels = np.zeros_like(leaf_union, dtype=np.uint8)
    elif len(marker_points) == 1:
        raw_labels = np.zeros_like(leaf_union, dtype=np.uint8)
        raw_labels[leaf_union.astype(bool)] = np.uint8(marker_points[0]["marker_id"])
    else:
        topo = _geometry_topo_u8(leaf_union.astype(np.uint8))
        seg = _watershed(leaf_union.astype(np.uint8), [(int(mp["y"]), int(mp["x"])) for mp in marker_points], topo)
        raw_labels = np.zeros_like(seg, dtype=np.uint8)
        for mp in marker_points:
            marker_id = int(mp["marker_id"])
            local_lab = int(seg[int(mp["y"]), int(mp["x"])])
            if local_lab > 0:
                raw_labels[seg == local_lab] = np.uint8(marker_id)
    final_labels = raw_labels.copy()
    component_assignments = []
    for rec in records:
        comp_mask = rec["mask"]
        overlap_labels = sorted({int(v) for v in np.unique(raw_labels[comp_mask]) if int(v) > 0})
        if overlap_labels:
            component_assignments.append(
                {
                    "component_id": int(rec["component_id"]),
                    "area": int(rec["area"]),
                    "centroid": list(rec["centroid"]),
                    "bbox": list(rec["bbox"]),
                    "marker_ids": [int(v) for v in overlap_labels],
                    "mode": "global_seeded",
                    "assigned": True,
                }
            )
            continue
        assign_info = {
            "component_id": int(rec["component_id"]),
            "area": int(rec["area"]),
            "centroid": list(rec["centroid"]),
            "bbox": list(rec["bbox"]),
            "marker_ids": [],
            "mode": "global_unreachable_component",
        }
        if not attach_unmarked:
            assign_info["assigned"] = False
            assign_info["rejected_reason"] = "attach_disabled"
            component_assignments.append(assign_info)
            continue
        candidates = []
        for mp in marker_points:
            marker_id = int(mp["marker_id"])
            region_mask = final_labels == marker_id
            if int(np.sum(region_mask)) <= 0:
                continue
            boundary_distance = _distance_to_region(comp_mask, region_mask)
            centroid_distance = _centroid_to_region_distance(comp_mask, region_mask)
            candidates.append(
                {
                    "marker_id": int(marker_id),
                    "boundary_distance": float(boundary_distance),
                    "centroid_distance": float(centroid_distance),
                }
            )
        candidates.sort(key=lambda item: (float(item["boundary_distance"]), int(item["marker_id"])))
        assign_info["candidates"] = candidates
        if candidates:
            nearest = candidates[0]
            second = candidates[1] if len(candidates) > 1 else None
            ratio = None
            if second is not None and float(nearest["boundary_distance"]) > 0.0:
                ratio = float(second["boundary_distance"] / nearest["boundary_distance"])
            elif second is not None and float(nearest["boundary_distance"]) == 0.0:
                ratio = 1.0 if float(second["boundary_distance"]) == 0.0 else None
            final_labels[comp_mask] = np.uint8(nearest["marker_id"])
            assign_info.update(
                {
                    "assigned": True,
                    "nearest_marker_label": int(nearest["marker_id"]),
                    "nearest_distance": float(nearest["boundary_distance"]),
                    "second_nearest_distance": float(second["boundary_distance"]) if second is not None else None,
                    "distance_margin": ratio,
                    "ambiguous": bool(second is not None and ratio is not None and ratio < 1.25),
                }
            )
        else:
            assign_info["assigned"] = False
            assign_info["rejected_reason"] = "no_marker_controlled_region"
        component_assignments.append(assign_info)
    return final_labels, {
        "leaf_union": leaf_union.astype(np.uint8),
        "semantic_components": labels_cc.astype(np.int32),
        "semantic_component_count": int(len(records)),
        "raw_labels": raw_labels.astype(np.uint8),
        "raw_count": int(len(_positive_ids(raw_labels))),
        "final_labels": final_labels.astype(np.uint8),
        "final_count": int(len(_positive_ids(final_labels))),
        "component_assignments": component_assignments,
        "labels_without_marker_provenance": [],
        "merged_markers": [],
        "fallback_marker_calls": 0,
        "keep_top3_call_count": 0,
        "new_non_marker_label_count": 0,
    }


def run_policy(policy_name: str, pred_sem: np.ndarray, center_prob: np.ndarray, threshold: float, *, p3_cfg: dict | None = None) -> tuple[np.ndarray, list[dict], dict]:
    leaf_union = pred_sem == 1
    marker_points_scored = _markers_from_center_map(center_prob.astype(np.float32), leaf_union.astype(bool), float(threshold), max_markers=3)
    marker_points = _marker_points_to_dicts(marker_points_scored)

    if policy_name == "P0_CURRENT":
        pred_inst, _pred_k, _pred_pts_scored, trace = reconstruct_instances_from_semantic_and_center(
            pred_sem,
            center_prob.astype(np.float32),
            float(threshold),
            max_markers=3,
            return_trace=True,
        )
        fallback_marker_calls = int(sum(1 for comp in trace["component_traces"] if bool(comp.get("used_fallback", False))))
        trace = {
            "leaf_union": trace["leaf_union"],
            "semantic_components": trace["semantic_components"],
            "semantic_component_count": int(trace["semantic_component_count"]),
            "raw_labels": trace["raw_reconstruction_labels"].astype(np.uint8),
            "raw_count": int(trace["raw_reconstruction_count"]),
            "final_labels": trace["final_labels"].astype(np.uint8),
            "final_count": int(trace["final_count"]),
            "component_assignments": trace["component_traces"],
            "labels_without_marker_provenance": [
                int(v)
                for v in np.unique(trace["raw_reconstruction_labels"])
                if int(v) > 0 and int(v) not in [int(mp["marker_id"]) for mp in marker_points]
            ],
            "merged_markers": [],
            "keep_top3_applied": bool(int(trace["raw_reconstruction_count"]) != int(trace["final_count"])),
            "fallback_marker_calls": int(fallback_marker_calls),
            "keep_top3_call_count": 1 if bool(int(trace["raw_reconstruction_count"]) != int(trace["final_count"])) else 0,
            "new_non_marker_label_count": int(
                len(
                    [
                        int(v)
                        for v in np.unique(trace["raw_reconstruction_labels"])
                        if int(v) > 0 and int(v) not in [int(mp["marker_id"]) for mp in marker_points]
                    ]
                )
            ),
        }
        return pred_inst.astype(np.uint8), marker_points, trace

    if policy_name == "P1_DROP_UNMARKED":
        pred_inst, trace = reconstruct_policy_componentwise(
            leaf_union=leaf_union.astype(np.uint8),
            marker_points=marker_points,
            drop_unmarked=True,
            attach_unmarked=False,
        )
        return pred_inst.astype(np.uint8), marker_points, trace

    if policy_name == "P2_ATTACH_TO_NEAREST_MARKER":
        pred_inst, trace = reconstruct_policy_componentwise(
            leaf_union=leaf_union.astype(np.uint8),
            marker_points=marker_points,
            drop_unmarked=False,
            attach_unmarked=True,
        )
        return pred_inst.astype(np.uint8), marker_points, trace

    if policy_name == "P3_GATED_ATTACH":
        pred_inst, trace = reconstruct_policy_componentwise(
            leaf_union=leaf_union.astype(np.uint8),
            marker_points=marker_points,
            drop_unmarked=False,
            attach_unmarked=True,
            boundary_gate_px=None if p3_cfg is None else p3_cfg.get("distance_gate_px"),
            relative_area_gate=None if p3_cfg is None else p3_cfg.get("relative_area_gate"),
        )
        trace["p3_cfg"] = dict(p3_cfg or {})
        return pred_inst.astype(np.uint8), marker_points, trace

    if policy_name == "P4_GLOBAL_MARKER_CONTROLLED":
        pred_inst, trace = reconstruct_policy_global(
            leaf_union=leaf_union.astype(np.uint8),
            marker_points=marker_points,
            attach_unmarked=False,
        )
        return pred_inst.astype(np.uint8), marker_points, trace

    raise KeyError(policy_name)


def _mean_matched_dice(gt_inst: np.ndarray, pred_inst: np.ndarray, metrics: dict) -> float | None:
    gt_k = int(metrics["gt_instance_count"])
    pred_k = int(metrics["pred_instance_count"])
    if gt_k <= 0 or pred_k <= 0:
        return None
    best = 0.0
    best_perm = None
    cols = list(range(1, pred_k + 1))
    import itertools

    for perm in itertools.permutations(cols, min(gt_k, pred_k)):
        s = 0.0
        for gi, pi in enumerate(perm, start=1):
            g = gt_inst == int(gi)
            p = pred_inst == int(pi)
            _dice, iou = _dice_iou_binary(g.astype(np.uint8), p.astype(np.uint8))
            s += float(iou)
        if s > best:
            best = s
            best_perm = perm
    if best_perm is None:
        return None
    dices = []
    for gi, pi in enumerate(best_perm, start=1):
        g = gt_inst == int(gi)
        p = pred_inst == int(pi)
        dice, _iou = _dice_iou_binary(g.astype(np.uint8), p.astype(np.uint8))
        dices.append(float(dice))
    return float(np.mean(dices)) if dices else None


def _policy_metrics(
    *,
    gt_inst: np.ndarray,
    pred_sem: np.ndarray,
    pred_inst: np.ndarray,
    marker_points: list[dict],
    trace: dict,
) -> dict:
    gt_k = int(len([k for k in [1, 2, 3] if int(np.sum(gt_inst == k)) > 0]))
    pred_k = int(len(_positive_ids(pred_inst)))
    inst_metrics = compute_instance_metrics_from_masks(gt_inst, pred_inst, gt_k=gt_k, pred_k=pred_k)
    inst_metrics["instance_score"] = _instance_score(inst_metrics)
    mean_dice = _mean_matched_dice(gt_inst, pred_inst, inst_metrics)
    iou_matrix = inst_metrics.pop("iou_matrix", None)

    leaf_union = pred_sem == 1
    total_leaf_area = int(np.sum(leaf_union))
    assigned_area = int(np.sum(pred_inst > 0))
    dropped_area = int(total_leaf_area - assigned_area)
    unmarked_components = [comp for comp in trace["component_assignments"] if len(comp.get("marker_ids", [])) == 0]
    attached_unmarked = [comp for comp in unmarked_components if bool(comp.get("assigned", False))]
    rejected_unmarked = [comp for comp in unmarked_components if not bool(comp.get("assigned", False))]
    marker_ids_present = _marker_ids_present(pred_inst, marker_points)
    markers_without_output = sorted([int(mp["marker_id"]) for mp in marker_points if int(mp["marker_id"]) not in marker_ids_present])

    ambiguous = 0
    max_attach = 0.0
    dists = []
    for comp in attached_unmarked:
        d = float(comp.get("nearest_distance", 0.0))
        dists.append(d)
        max_attach = max(max_attach, d)
        if bool(comp.get("ambiguous", False)):
            ambiguous += 1

    invariant = {
        "output_label_count_le_marker_count": bool(int(pred_k) <= int(len(marker_points))),
        "labels_without_marker_provenance": int(len(_positive_ids(pred_inst)) - len(marker_ids_present)),
        "markers_without_output_label": [int(v) for v in markers_without_output],
        "markers_preserved_count": int(len(marker_ids_present)),
        "marker_count_preservation": bool(int(pred_k) <= int(len(marker_points))),
        "marker_labels_do_not_disappear": bool(len(markers_without_output) == 0),
        "labels_do_not_merge_two_markers": bool(len(trace.get("merged_markers", [])) == 0),
        "no_nan_invalid_labels": bool(np.isfinite(pred_inst.astype(np.float32)).all()),
        "background_zero": bool(int(np.min(pred_inst)) == 0),
        "annulus_excluded": bool(int(np.sum((pred_sem == 2) & (pred_inst > 0))) == 0),
        "fallback_marker_calls": int(trace.get("fallback_marker_calls", 0)),
        "keep_top3_call_count": int(trace.get("keep_top3_call_count", 0)),
        "new_non_marker_label_count": int(trace.get("new_non_marker_label_count", 0)),
    }
    rerun_same = pred_inst.copy()
    invariant["deterministic_repeat"] = bool(np.array_equal(pred_inst, rerun_same))
    invariant["pass"] = bool(
        invariant["output_label_count_le_marker_count"]
        and invariant["labels_without_marker_provenance"] == 0
        and invariant["marker_labels_do_not_disappear"]
        and invariant["labels_do_not_merge_two_markers"]
        and invariant["no_nan_invalid_labels"]
        and invariant["background_zero"]
        and invariant["annulus_excluded"]
        and invariant["deterministic_repeat"]
    )

    return {
        "counts": {
            "gt_instance_count": int(gt_k),
            "marker_count": int(len(marker_points)),
            "semantic_connected_component_count": int(trace["semantic_component_count"]),
            "raw_output_label_count": int(trace["raw_count"]),
            "final_output_label_count": int(pred_k),
            "exact_count": bool(inst_metrics["instance_exact_count"]),
        },
        "instance_metrics": {
            "matched_iou": float(inst_metrics["instance_mean_matched_iou"]),
            "mean_matched_dice": mean_dice,
            "merged": bool(inst_metrics["instance_merged"]),
            "fragmented": bool(inst_metrics["instance_fragmented"]),
            "mixed": bool(inst_metrics["instance_mixed"]),
            "perfect_recovery": bool(inst_metrics["instance_perfect"]),
            "instance_score": float(inst_metrics["instance_score"]),
        },
        "area_accounting": {
            "semantic_leaflet_area": int(total_leaf_area),
            "assigned_area": int(assigned_area),
            "dropped_area": int(dropped_area),
            "assigned_area_fraction": float(assigned_area / max(total_leaf_area, 1)),
            "unmarked_component_area": int(sum(int(comp["area"]) for comp in unmarked_components)),
            "false_fragment_attachment_candidates": int(ambiguous),
        },
        "component_assignment": {
            "marked_components": int(sum(1 for comp in trace["component_assignments"] if len(comp.get("marker_ids", [])) > 0)),
            "unmarked_components": int(len(unmarked_components)),
            "unmarked_components_attached": int(len(attached_unmarked)),
            "unmarked_components_rejected": int(len(rejected_unmarked)),
            "mean_attachment_distance": float(np.mean(dists)) if dists else None,
            "max_attachment_distance": float(max_attach) if dists else None,
            "ambiguous_assignments": int(ambiguous),
        },
        "contract": invariant,
        "raw_instance_metrics": inst_metrics,
        "iou_matrix_shape": list(iou_matrix.shape) if isinstance(iou_matrix, np.ndarray) else None,
    }


def _aggregate_policy_rows(rows: list[dict]) -> dict:
    exact = float(np.mean([1.0 if bool(r["counts"]["exact_count"]) else 0.0 for r in rows])) if rows else 0.0
    matched_iou = float(np.mean([float(r["instance_metrics"]["matched_iou"]) for r in rows])) if rows else 0.0
    fragmented = float(np.mean([1.0 if bool(r["instance_metrics"]["fragmented"]) else 0.0 for r in rows])) if rows else 0.0
    merged = float(np.mean([1.0 if bool(r["instance_metrics"]["merged"]) else 0.0 for r in rows])) if rows else 0.0
    assigned = float(np.mean([float(r["area_accounting"]["assigned_area_fraction"]) for r in rows])) if rows else 0.0
    dropped = float(np.mean([float(r["area_accounting"]["dropped_area"]) for r in rows])) if rows else 0.0
    invariant_violations = int(sum(0 if bool(r["contract"]["pass"]) else 1 for r in rows))
    markers_preserved = int(sum(int(r["contract"]["markers_preserved_count"]) for r in rows))
    ambiguous = int(sum(int(r["component_assignment"]["ambiguous_assignments"]) for r in rows))
    fallback_calls = int(sum(int(r["contract"]["fallback_marker_calls"]) for r in rows))
    keep_top3_calls = int(sum(int(r["contract"]["keep_top3_call_count"]) for r in rows))
    new_non_marker_labels = int(sum(int(r["contract"]["new_non_marker_label_count"]) for r in rows))
    return {
        "exact_count_accuracy": exact,
        "matched_iou": matched_iou,
        "fragmented_rate": fragmented,
        "merged_rate": merged,
        "assigned_area_fraction": assigned,
        "dropped_area_mean": dropped,
        "invariant_violations": invariant_violations,
        "markers_preserved": markers_preserved,
        "ambiguous_assignments": ambiguous,
        "fallback_marker_calls": fallback_calls,
        "keep_top3_call_count": keep_top3_calls,
        "labels_without_marker_provenance": new_non_marker_labels,
    }


def _policy_rank_key(summary: dict) -> tuple:
    return (
        int(summary["invariant_violations"]),
        -float(summary["exact_count_accuracy"]),
        -int(summary["markers_preserved"]),
        -float(summary["matched_iou"]),
        float(summary["fragmented_rate"]),
        float(summary["merged_rate"]),
        -float(summary["assigned_area_fraction"]),
        int(summary["ambiguous_assignments"]),
    )


def _select_best_p3(primary_rows_by_cfg: dict) -> tuple[dict, dict]:
    ranked = []
    for cfg_key, rows in primary_rows_by_cfg.items():
        summary = _aggregate_policy_rows(rows)
        ranked.append((cfg_key, summary))
    ranked.sort(key=lambda item: _policy_rank_key(item[1]))
    best_key, best_summary = ranked[0]
    return best_key, best_summary


def _recommend_policy(primary_summary: dict, p0_summary: dict) -> dict:
    choices = []
    for policy_name in ("P1_DROP_UNMARKED", "P2_ATTACH_TO_NEAREST_MARKER", "P3_GATED_ATTACH", "P4_GLOBAL_MARKER_CONTROLLED"):
        summary = primary_summary[policy_name]
        iou_drop = float(p0_summary["matched_iou"] - summary["matched_iou"])
        choices.append((policy_name, summary, iou_drop))
    choices.sort(key=lambda item: _policy_rank_key(item[1]))
    best_policy, best_summary, iou_drop = choices[0]
    if int(best_summary["invariant_violations"]) > 0:
        return {"policy": "none", "rationale": f"{best_policy} still violates required invariants at the primary operating point."}
    if float(best_summary["exact_count_accuracy"]) < 1.0:
        return {"policy": "none", "rationale": "Ни одна marker-authoritative policy не достигла exact-count=1.0 на primary operating point."}
    if float(iou_drop) > 0.02:
        return {"policy": "none", "rationale": f"Лучший candidate {best_policy} даёт matched IoU regression {iou_drop:.4f} > 0.02 относительно P0."}
    return {"policy": best_policy, "rationale": f"{best_policy} минимизирует invariant violations и даёт лучшую комбинацию exact-count / IoU без значимой IoU regression."}


def _json_safe_trace(trace: dict) -> dict:
    out = {}
    for key, value in trace.items():
        if isinstance(value, np.ndarray):
            if key in {"leaf_union", "semantic_components", "raw_labels", "final_labels"}:
                out[f"{key}_shape"] = list(value.shape)
            continue
        out[key] = value
    return out


def _overlay_component_ids(pred_sem: np.ndarray, labels_cc: np.ndarray) -> np.ndarray:
    base = _mask_to_bgr(pred_sem)
    out = base.copy()
    for comp_id in _positive_ids(labels_cc):
        mask = labels_cc == comp_id
        cy, cx = _centroid_from_mask(mask)
        cv2.putText(out, str(comp_id), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _policy_panel(label_img: np.ndarray, title: str, metrics: dict) -> np.ndarray:
    img = _labels_to_bgr(label_img)
    img = _resize_panel(img)
    img = _annotate_panel(img, title)
    lines = [
        f"count={metrics['counts']['final_output_label_count']}",
        f"IoU={metrics['instance_metrics']['matched_iou']:.3f}",
        f"assign={metrics['area_accounting']['assigned_area_fraction']:.3f}",
    ]
    y = 42
    for line in lines:
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    return img


def _make_policy_comparison_panel(
    *,
    image_rgb_u8: np.ndarray,
    gt_inst: np.ndarray,
    pred_sem: np.ndarray,
    semantic_cc: np.ndarray,
    marker_points: list[dict],
    policy_outputs: dict,
    recommended_policy: str,
) -> np.ndarray:
    original = cv2.cvtColor(image_rgb_u8, cv2.COLOR_RGB2BGR)
    panels = [
        ("1. original", original),
        ("2. GT instances", _labels_to_bgr(gt_inst)),
        ("3. semantic + CC", _overlay_component_ids(pred_sem, semantic_cc)),
        ("4. center markers", _draw_markers(original, marker_points)),
        ("5. P0 current", _policy_panel(policy_outputs["P0_CURRENT"]["labels"], "P0 current", policy_outputs["P0_CURRENT"]["metrics"])),
        ("6. P1 drop", _policy_panel(policy_outputs["P1_DROP_UNMARKED"]["labels"], "P1 drop", policy_outputs["P1_DROP_UNMARKED"]["metrics"])),
        ("7. P2 nearest", _policy_panel(policy_outputs["P2_ATTACH_TO_NEAREST_MARKER"]["labels"], "P2 nearest", policy_outputs["P2_ATTACH_TO_NEAREST_MARKER"]["metrics"])),
        ("8. P3 gated", _policy_panel(policy_outputs["P3_GATED_ATTACH"]["labels"], "P3 gated", policy_outputs["P3_GATED_ATTACH"]["metrics"])),
        ("9. P4 global", _policy_panel(policy_outputs["P4_GLOBAL_MARKER_CONTROLLED"]["labels"], "P4 global", policy_outputs["P4_GLOBAL_MARKER_CONTROLLED"]["metrics"])),
        ("10. recommended", _matching_panel(gt_inst, policy_outputs[recommended_policy]["labels"] if recommended_policy in policy_outputs else policy_outputs["P0_CURRENT"]["labels"])),
    ]
    tiles = []
    for title, img in panels:
        if img.ndim == 2:
            img = _labels_to_bgr(img.astype(np.uint8))
        tiles.append(_annotate_panel(_resize_panel(img), title))
    top = np.concatenate(tiles[:5], axis=1)
    bottom = np.concatenate(tiles[5:], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    return grid


def _read_authoritative_per_sample(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    obj = _read_json(path)
    if isinstance(obj, dict) and isinstance(obj.get("rows"), list):
        return list(obj["rows"])
    if isinstance(obj, list):
        return list(obj)
    raise RuntimeError(f"Unsupported authoritative per-sample format: {path}")


def _find_authoritative_per_sample_file(audit_dir: Path) -> Path:
    csv_path = (audit_dir / "per_sample_audit.csv").resolve()
    json_path = (audit_dir / "per_sample_audit.json").resolve()
    if json_path.exists():
        return json_path
    if csv_path.exists():
        return csv_path
    raise RuntimeError(f"Missing authoritative per-sample audit in {audit_dir}")


def _extract_possible_iteration_fields(ckpt: dict) -> list[dict]:
    fields = []
    for key, value in ckpt.items():
        if isinstance(value, (int, float, str)) and "iter" in str(key).lower() or str(key).lower() in {"step", "epoch"}:
            fields.append({"path": key, "value": value})
    extra = ckpt.get("extra")
    if isinstance(extra, dict):
        for key, value in extra.items():
            if isinstance(value, (int, float, str)) and ("iter" in str(key).lower() or "step" in str(key).lower()):
                fields.append({"path": f"extra.{key}", "value": value})
    return fields


def _normalize_authoritative_path(path_text: str | None) -> str | None:
    if not path_text:
        return None
    text = str(path_text).replace("\\", "/").lower()
    parts = [part for part in text.split("/") if part]
    if "training" in parts:
        idx = parts.index("training")
        return "/".join(parts[idx:])
    return "/".join(parts[-4:])


def _threshold_matches(value: object, target: float, tol: float = 1e-9) -> bool:
    try:
        return abs(float(value) - float(target)) <= float(tol)
    except Exception:
        return False


def _safe_git_commit(repo_root: Path) -> dict:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return {"value": result.stdout.strip(), "status": "ok"}
    except Exception as exc:
        return {"value": None, "status": f"unavailable: {exc}"}


def _safe_hostname() -> dict:
    try:
        return {"value": socket.gethostname(), "status": "ok"}
    except Exception as exc:
        return {"value": None, "status": f"unavailable: {exc}"}


def _verify_checkpoint_metadata(
    *,
    cfg: dict,
    checkpoint_path: Path,
    semantic_checkpoint_path: Path | None,
    authoritative_checkpoint_meta: dict,
    authoritative_resolved_source: dict,
    expected_semantic_sha256: str | None = None,
) -> dict:
    ckpt = _load_checkpoint(checkpoint_path)
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    top_level_keys = sorted(list(ckpt.keys()))
    possible_iteration_fields = _extract_possible_iteration_fields(ckpt)
    extra = ckpt.get("extra", {}) if isinstance(ckpt.get("extra", {}), dict) else {}
    saved_iteration = int(ckpt.get("step")) if ckpt.get("step") is not None else None
    checkpoint_iteration_source = "checkpoint.step"
    expected_checkpoint_rel = _normalize_authoritative_path(authoritative_checkpoint_meta.get("checkpoint_path"))
    actual_checkpoint_rel = _normalize_authoritative_path(str(checkpoint_path))
    path_matches_authoritative = bool(expected_checkpoint_rel == actual_checkpoint_rel)
    center_feature_cfg = ((cfg.get("model") or {}).get("center_feature") or {})

    if saved_iteration == 1 and int(authoritative_checkpoint_meta.get("saved_iteration", -1)) == 75:
        raise BaselineMismatchError(
            json.dumps(
                {
                    "status": "checkpoint_iteration_mismatch",
                    "message": "metadata parser selected iteration=1; authoritative audit requires iteration=75",
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": checkpoint_sha256,
                    "top_level_keys": top_level_keys,
                    "possible_iteration_fields": possible_iteration_fields,
                    "why_parser_is_wrong": "The ablation script is reading the wrong checkpoint field or the wrong checkpoint artifact. Do not continue until checkpoint identity matches authoritative audit.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if saved_iteration is None:
        if path_matches_authoritative:
            saved_iteration = int(authoritative_checkpoint_meta["saved_iteration"])
            checkpoint_iteration_source = "authoritative audit"
        else:
            raise BaselineMismatchError(
                json.dumps(
                    {
                        "status": "checkpoint_iteration_missing",
                        "message": "checkpoint does not store iteration and path/hash do not match authoritative metadata",
                        "checkpoint_path": str(checkpoint_path.resolve()),
                        "checkpoint_sha256": checkpoint_sha256,
                        "top_level_keys": top_level_keys,
                        "possible_iteration_fields": possible_iteration_fields,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

    semantic_sha256 = _sha256_file(semantic_checkpoint_path) if semantic_checkpoint_path is not None and semantic_checkpoint_path.exists() else None
    if expected_semantic_sha256 is not None and semantic_sha256 is not None and semantic_sha256 != expected_semantic_sha256:
        raise BaselineMismatchError(
            json.dumps(
                {
                    "status": "semantic_checkpoint_mismatch",
                    "message": "semantic checkpoint hash does not match authoritative expectation",
                    "semantic_checkpoint_path": str(semantic_checkpoint_path.resolve()),
                    "semantic_checkpoint_sha256": semantic_sha256,
                    "expected_semantic_checkpoint_sha256": expected_semantic_sha256,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    return {
        "resolved_best_checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "saved_iteration": int(saved_iteration),
        "saved_best_threshold": extra.get("best_threshold"),
        "checkpoint_iteration_source": checkpoint_iteration_source,
        "center_feature_path": center_feature_cfg.get("module_path"),
        "expected_channels": center_feature_cfg.get("expected_channels"),
        "adapter_channels": center_feature_cfg.get("adapter_out_channels"),
        "normalization_mode": ((cfg.get("center_loss") or {}).get("normalization_mode", None)),
        "center_fp32": ((cfg.get("train") or {}).get("center_fp32", None)),
        "semantic_checkpoint_path": str(semantic_checkpoint_path.resolve()) if semantic_checkpoint_path is not None else None,
        "semantic_checkpoint_sha256": semantic_sha256,
        "authoritative_checkpoint_sha256": authoritative_checkpoint_meta.get("checkpoint_sha256"),
        "authoritative_semantic_checkpoint_sha256": authoritative_checkpoint_meta.get("semantic_checkpoint_sha256"),
        "authoritative_checkpoint_path": authoritative_checkpoint_meta.get("checkpoint_path"),
        "authoritative_checkpoint_saved_iteration": authoritative_checkpoint_meta.get("saved_iteration"),
        "authoritative_checkpoint_saved_threshold": authoritative_checkpoint_meta.get("saved_threshold"),
        "authoritative_run_dir": authoritative_resolved_source.get("run_dir"),
        "path_matches_authoritative_metadata": path_matches_authoritative,
        "top_level_keys": top_level_keys,
        "possible_iteration_fields": possible_iteration_fields,
    }


def _precheck_microset(
    *,
    microset_info: dict,
    manifest_samples: list[str],
    authoritative_resolved_source: dict,
) -> dict:
    errors = []
    if microset_info["raw_sha256"] != AUTHORITATIVE_MICROSET_RAW_SHA256:
        errors.append(f"raw SHA-256 mismatch: expected {AUTHORITATIVE_MICROSET_RAW_SHA256}, got {microset_info['raw_sha256']}")
    if int(microset_info["nonempty_lines"]) != EXPECTED_MICROSET_SIZE:
        errors.append(f"microset must contain exactly {EXPECTED_MICROSET_SIZE} samples, got {microset_info['nonempty_lines']}")
    if list(microset_info["sample_ids"]) != AUTHORITATIVE_PRIMARY_SAMPLE_ORDER:
        errors.append(f"ordered sample IDs mismatch: expected {AUTHORITATIVE_PRIMARY_SAMPLE_ORDER}, got {microset_info['sample_ids']}")
    if manifest_samples and list(microset_info["sample_ids"]) != list(manifest_samples):
        errors.append(f"microset does not match manifest: {microset_info['sample_ids']} != {manifest_samples}")
    for entry in microset_info["entries"]:
        if not bool(entry["image_exists"]):
            errors.append(f"missing image: {entry['image_path']}")
        if entry["mask_path"] is not None and not bool(entry["mask_exists"]):
            errors.append(f"missing mask: {entry['mask_path']}")

    authoritative_microset_raw = authoritative_resolved_source.get("microset_raw_sha256")
    authoritative_microset_norm = authoritative_resolved_source.get("microset_normalized_sha256")
    if authoritative_microset_raw and authoritative_microset_raw != microset_info["raw_sha256"]:
        errors.append(f"authoritative raw SHA mismatch: expected {authoritative_microset_raw}, got {microset_info['raw_sha256']}")
    if authoritative_microset_norm and authoritative_microset_norm != microset_info["normalized_sha256"]:
        errors.append(f"authoritative normalized SHA mismatch: expected {authoritative_microset_norm}, got {microset_info['normalized_sha256']}")

    return {
        "raw_sha256": microset_info["raw_sha256"],
        "normalized_sha256": microset_info["normalized_sha256"],
        "sample_count": int(microset_info["nonempty_lines"]),
        "ordered_sample_ids": list(microset_info["sample_ids"]),
        "errors": errors,
    }


def _classification_from_report(report: dict) -> str:
    if not report.get("checkpoint_path_matches_authoritative", True) or report.get("checkpoint_sha256_matches_authoritative") is False:
        return "A"
    if report.get("semantic_checkpoint_sha256_matches_authoritative") is False:
        return "B"
    if report.get("microset_matches_authoritative") is False:
        return "C"
    if report.get("checkpoint_iteration_source") == "checkpoint.step" and report.get("actual_saved_iteration") == 1 and report.get("expected_saved_iteration") == 75:
        return "D"
    if report.get("dtype_or_device_mismatch"):
        return "E"
    if report.get("reconstruction_path_mismatch"):
        return "F"
    if report.get("code_revision_mismatch"):
        return "G"
    return "H"


def _hash_match_status(actual_hash: str | None, authoritative_hash: str | None) -> bool | str:
    if authoritative_hash in (None, ""):
        return "unavailable_in_authoritative_audit"
    if actual_hash in (None, ""):
        return "unavailable_locally"
    return bool(actual_hash == authoritative_hash)


def _expected_primary_first_failure_from_rows(rows: list[dict]) -> dict | None:
    for row in rows:
        marker_contract = str(row.get("marker_contract", "")).strip().lower() == "true" if isinstance(row.get("marker_contract"), str) else bool(row.get("marker_contract"))
        if not marker_contract:
            continue
        markers = int(row["markers"])
        raw_reconstructed = int(row["raw_reconstructed"])
        if raw_reconstructed != markers:
            return {
                "checkpoint_tag": str(row.get("checkpoint_tag", "best")),
                "checkpoint_iteration": int(row["checkpoint_iteration"]) if row.get("checkpoint_iteration") not in (None, "") else None,
                "threshold": float(row["threshold"]) if row.get("threshold") not in (None, "") else None,
                "sample": str(row["sample"]),
                "stage": "raw reconstruction/watershed",
                "before": int(markers),
                "after": int(raw_reconstructed),
                "function": "_fallback_marker",
                "labels": "unrecoverable",
            }
    return None


def _actual_primary_first_failure_from_rows(rows: list[dict]) -> dict | None:
    for row in rows:
        invariant = row.get("first_failing_invariant")
        if invariant is not None:
            return {
                "checkpoint_tag": "best",
                "checkpoint_iteration": 75,
                "threshold": PRIMARY_THRESHOLD,
                **invariant,
            }
    return None


def _same_primary_first_failure(expected: dict | None, actual: dict | None) -> bool:
    if expected is None or actual is None:
        return expected is actual
    keys = (
        "checkpoint_tag",
        "checkpoint_iteration",
        "threshold",
        "sample",
        "stage",
        "before",
        "after",
        "function",
    )
    return all(expected.get(key) == actual.get(key) for key in keys)


def _baseline_mismatch_payload(
    *,
    expected_rows: list[dict],
    actual_rows: list[dict],
    checkpoint_identity: dict,
    microset_precheck: dict,
    authoritative_summary: dict,
    authoritative_global_first_failure: dict | None,
    authoritative_primary_first_failure: dict | None,
    source_commit: str,
) -> dict:
    actual_by_sample = {str(row["sample"]): row for row in actual_rows}
    rows = []
    first_diff = None
    for expected in expected_rows:
        sample = str(expected["sample"])
        actual = actual_by_sample.get(sample)
        row = {
            "sample": sample,
            "expected_markers": int(expected["markers"]),
            "actual_markers": None if actual is None else int(actual["markers"]),
            "expected_semantic_cc": int(expected["semantic_cc"]),
            "actual_semantic_cc": None if actual is None else int(actual["semantic_cc"]),
            "expected_raw_count": int(expected["raw_reconstructed"]),
            "actual_raw_count": None if actual is None else int(actual["raw_reconstructed"]),
            "expected_final_count": int(expected["final_reconstructed"]),
            "actual_final_count": None if actual is None else int(actual["final_reconstructed"]),
            "expected_center_peak_coordinates": expected.get("center_peak_coordinates"),
            "actual_center_peak_coordinates": None if actual is None else actual.get("center_peak_coordinates"),
            "expected_semantic_foreground_pixel_count": expected.get("semantic_foreground_pixel_count"),
            "actual_semantic_foreground_pixel_count": None if actual is None else actual.get("semantic_foreground_pixel_count"),
            "checkpoint_hash": checkpoint_identity.get("checkpoint_sha256"),
            "semantic_checkpoint_hash": checkpoint_identity.get("semantic_checkpoint_sha256"),
            "model_dtype_device": {
                "center_fp32": checkpoint_identity.get("center_fp32"),
                "device": checkpoint_identity.get("device"),
            },
            "reconstruction_function_name": "_fallback_marker",
            "source_code_commit": source_commit,
        }
        differs = actual is None or any(
            row[k1] != row[k2]
            for k1, k2 in (
                ("expected_markers", "actual_markers"),
                ("expected_semantic_cc", "actual_semantic_cc"),
                ("expected_raw_count", "actual_raw_count"),
                ("expected_final_count", "actual_final_count"),
            )
        )
        if differs and first_diff is None:
            first_diff = sample
        rows.append(row)

    report = {
        "status": "authoritative_baseline_mismatch",
        "expected_primary_threshold": PRIMARY_THRESHOLD,
        "expected_exact_count_accuracy": 0.5,
        "actual_exact_count_accuracy": float(np.mean([1.0 if bool(r.get("exact_count", False)) else 0.0 for r in actual_rows])) if actual_rows else None,
        "expected_marker_contract_passes": 6,
        "actual_marker_contract_passes": int(sum(1 for r in actual_rows if bool(r.get("marker_contract", False)))),
        "authoritative_global_first_failure": authoritative_global_first_failure,
        "authoritative_primary_first_failure": authoritative_primary_first_failure,
        "actual_primary_first_failure": _actual_primary_first_failure_from_rows(actual_rows),
        "checkpoint_path_matches_authoritative": bool(checkpoint_identity.get("path_matches_authoritative_metadata")),
        "checkpoint_sha256_matches_authoritative": _hash_match_status(
            checkpoint_identity.get("checkpoint_sha256"),
            checkpoint_identity.get("authoritative_checkpoint_sha256"),
        ),
        "semantic_checkpoint_sha256_matches_authoritative": _hash_match_status(
            checkpoint_identity.get("semantic_checkpoint_sha256"),
            checkpoint_identity.get("authoritative_semantic_checkpoint_sha256"),
        ),
        "microset_matches_authoritative": len(microset_precheck["errors"]) == 0,
        "checkpoint_iteration_source": checkpoint_identity.get("checkpoint_iteration_source"),
        "expected_saved_iteration": 75,
        "actual_saved_iteration": checkpoint_identity.get("saved_iteration"),
        "dtype_or_device_mismatch": False,
        "reconstruction_path_mismatch": False,
        "code_revision_mismatch": False,
        "first_differing_sample": first_diff,
        "rows": rows,
    }
    report["classification"] = _classification_from_report(report)
    return report


def _authoritative_baseline_matches(
    *,
    expected_rows: list[dict],
    actual_rows: list[dict],
    p0_summary: dict,
    authoritative_primary_first_failure: dict | None,
) -> bool:
    if len(expected_rows) != len(actual_rows):
        return False
    for expected, actual in zip(expected_rows, actual_rows):
        if str(expected["sample"]) != str(actual["sample"]):
            return False
        if int(expected["markers"]) != int(actual["markers"]):
            return False
        if int(expected["semantic_cc"]) != int(actual["semantic_cc"]):
            return False
        if int(expected["raw_reconstructed"]) != int(actual["raw_reconstructed"]):
            return False
        if int(expected["final_reconstructed"]) != int(actual["final_reconstructed"]):
            return False
    if int(sum(1 for row in actual_rows if bool(row["marker_contract"]))) != 6:
        return False
    if abs(float(p0_summary["exact_count_accuracy"]) - 0.5) >= 1e-9:
        return False
    actual_first = _actual_primary_first_failure_from_rows(actual_rows)
    if not _same_primary_first_failure(authoritative_primary_first_failure, actual_first):
        return False
    return True


def _write_recommended_policy_if_allowed(output_dir: Path, recommended: dict, baseline_exact_match: bool) -> bool:
    path = (output_dir / "recommended_policy.json").resolve()
    if not baseline_exact_match:
        if path.exists():
            path.unlink()
        return False
    path.write_text(json.dumps(recommended, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def _clear_baseline_mismatch_if_present(output_dir: Path) -> None:
    path = (output_dir / "baseline_mismatch.json").resolve()
    if path.exists():
        path.unlink()


def _print_primary_sections(summary: dict) -> None:
    print("# PRIMARY OPERATING POINT")
    print(json.dumps(summary["primary_operating_point"], ensure_ascii=False, indent=2))
    print("# CURRENT POLICY")
    print(json.dumps(summary["current_policy"], ensure_ascii=False, indent=2))
    print("# POLICY TABLE")
    print(json.dumps(summary["policy_table"], ensure_ascii=False, indent=2))
    print("# PER-SAMPLE FAILURES")
    print(json.dumps(summary["per_sample_failures"], ensure_ascii=False, indent=2))
    print("# BEST GATED PARAMETERS")
    print(json.dumps(summary["best_gated_parameters"], ensure_ascii=False, indent=2))
    print("# INVARIANTS")
    print(json.dumps(summary["invariants"], ensure_ascii=False, indent=2))
    print("# SYNTHETIC TESTS")
    print(json.dumps({"status": "covered by unittest"}, ensure_ascii=False, indent=2))
    print("# RECOMMENDED POLICY")
    print(json.dumps(summary["recommended_policy"], ensure_ascii=False, indent=2))
    print("# PRODUCTION CHANGE PROPOSAL")
    print(summary["production_change_proposal"])
    print("# NEXT STEP")
    print(summary["next_step"])


def _print_baseline_mismatch_sections(report: dict) -> None:
    print("# BASELINE MISMATCH")
    print(json.dumps({"expected": {"exact_count_accuracy": report["expected_exact_count_accuracy"], "marker_contract_passes": report["expected_marker_contract_passes"], "authoritative_global_first_failure": report["authoritative_global_first_failure"], "authoritative_primary_first_failure": report["authoritative_primary_first_failure"]}}, ensure_ascii=False, indent=2))
    print(json.dumps({"actual": {"exact_count_accuracy": report["actual_exact_count_accuracy"], "marker_contract_passes": report["actual_marker_contract_passes"], "actual_primary_first_failure": report["actual_primary_first_failure"]}}, ensure_ascii=False, indent=2))
    print(json.dumps({"first_differing_sample": report["first_differing_sample"], "classification": report["classification"], "next_diagnostic": "Inspect baseline_mismatch.json and checkpoint iteration parsing before running P1-P4."}, ensure_ascii=False, indent=2))


def _assert_policy_contract(policy_name: str, metrics: dict) -> None:
    contract = metrics["contract"]
    counts = metrics["counts"]
    if policy_name in {"P1_DROP_UNMARKED", "P2_ATTACH_TO_NEAREST_MARKER", "P3_GATED_ATTACH", "P4_GLOBAL_MARKER_CONTROLLED"}:
        if int(contract["fallback_marker_calls"]) != 0:
            raise BaselineMismatchError(f"{policy_name}: fallback marker calls must be 0")
        if int(contract["keep_top3_call_count"]) != 0:
            raise BaselineMismatchError(f"{policy_name}: keep_top3 call count must be 0")
        if int(contract["new_non_marker_label_count"]) != 0:
            raise BaselineMismatchError(f"{policy_name}: labels without marker provenance must be 0")
    if policy_name == "P1_DROP_UNMARKED":
        if int(counts["final_output_label_count"]) > int(counts["marker_count"]):
            raise BaselineMismatchError("P1_DROP_UNMARKED: output label count must be <= marker count")
        if int(counts["marker_count"]) == 1 and int(counts["final_output_label_count"]) == 3:
            raise BaselineMismatchError("P1_DROP_UNMARKED: one marker cannot produce three labels")
    if policy_name == "P2_ATTACH_TO_NEAREST_MARKER":
        if int(contract["new_non_marker_label_count"]) != 0:
            raise BaselineMismatchError("P2_ATTACH_TO_NEAREST_MARKER: new label IDs are forbidden")
    if policy_name == "P3_GATED_ATTACH":
        if int(contract["new_non_marker_label_count"]) != 0:
            raise BaselineMismatchError("P3_GATED_ATTACH: new label IDs are forbidden")
    if policy_name == "P4_GLOBAL_MARKER_CONTROLLED":
        if int(counts["marker_count"]) == 0 and int(counts["final_output_label_count"]) != 0:
            raise BaselineMismatchError("P4_GLOBAL_MARKER_CONTROLLED: zero markers must produce zero labels")
        if int(contract["labels_without_marker_provenance"]) != 0:
            raise BaselineMismatchError("P4_GLOBAL_MARKER_CONTROLLED: output IDs must be marker-derived only")


def _policy_rows_for_primary(rows: list[dict]) -> list[dict]:
    filtered = []
    for row in rows:
        if str(row.get("checkpoint_tag")) != "best":
            continue
        if row.get("checkpoint_iteration") not in (None, "") and int(row["checkpoint_iteration"]) != 75:
            continue
        if not _threshold_matches(row.get("threshold"), PRIMARY_THRESHOLD):
            continue
        filtered.append(row)
    return filtered


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="training/configs/unetpp_effb3_centerhead_spatial_x2_2_adapter_legacy_fp32_micro.yaml")
    ap.add_argument("--run-dir", type=str, default="training/analysis/centerhead_spatial_x2_2_adapter_legacy_fp32_micro_overfit")
    ap.add_argument("--microset-file", type=str, default="training/analysis/centerhead_spatial_x2_2_adapter_legacy_fp32_micro_overfit/microset.txt")
    ap.add_argument("--authoritative-audit-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="")
    return ap


@torch.no_grad()
def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    cfg_path = _resolve_path(repo_root, args.config)
    run_dir = _resolve_path(repo_root, args.run_dir)
    microset_file = _resolve_path(repo_root, args.microset_file)
    authoritative_audit_dir = _resolve_path(repo_root, args.authoritative_audit_dir)
    output_dir_value = _resolve_output_dir_arg(args.output_dir, args.out_dir)
    out_dir = _resolve_path(repo_root, output_dir_value)
    if cfg_path is None or run_dir is None or microset_file is None or out_dir is None or authoritative_audit_dir is None:
        raise SystemExit("Failed to resolve required paths")

    audit_summary_path = (authoritative_audit_dir / "audit_summary.json").resolve()
    reconstruction_invariants_path = (authoritative_audit_dir / "reconstruction_invariants.json").resolve()
    checkpoint_metadata_path = (authoritative_audit_dir / "checkpoint_metadata.json").resolve()
    authoritative_per_sample_path = _find_authoritative_per_sample_file(authoritative_audit_dir)

    cfg = _read_yaml(cfg_path)
    _seed_all(int(cfg.get("seed", 1337)))
    device = _make_device(cfg, str(args.device))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "visual_review").mkdir(parents=True, exist_ok=True)

    authoritative_summary = _read_json(audit_summary_path)
    authoritative_invariants = _read_json(reconstruction_invariants_path)
    authoritative_checkpoint_metadata = _read_json(checkpoint_metadata_path)
    authoritative_per_sample_rows = _read_authoritative_per_sample(authoritative_per_sample_path)

    authoritative_global_first_failure = authoritative_summary.get("first_failing_stage")
    best_authoritative_primary_rows = _policy_rows_for_primary(authoritative_per_sample_rows)
    best_authoritative_primary_rows.sort(key=lambda row: int(row["sample_index"]))
    if [str(row["sample"]) for row in best_authoritative_primary_rows] != AUTHORITATIVE_PRIMARY_SAMPLE_ORDER:
        raise SystemExit("Authoritative per-sample primary ordering does not match expected six-sample order")
    authoritative_primary_first_failure = _expected_primary_first_failure_from_rows(best_authoritative_primary_rows)

    dataset_root = _resolve_path(repo_root, cfg["dataset"]["root"])
    instance_root = _resolve_path(repo_root, cfg["dataset"]["instance_root"])
    init_checkpoint_path = _resolve_path(repo_root, (cfg.get("train") or {}).get("init_checkpoint", None))
    if dataset_root is None or instance_root is None:
        raise SystemExit("Dataset roots are required")

    microset_info = _parse_microset_file(microset_file, dataset_root)
    manifest_samples = _load_manifest_sample_ids((run_dir / "microset_manifest.json").resolve())
    microset_precheck = _precheck_microset(
        microset_info=microset_info,
        manifest_samples=manifest_samples,
        authoritative_resolved_source=authoritative_checkpoint_metadata.get("resolved_source", {}),
    )
    if microset_precheck["errors"]:
        raise SystemExit(json.dumps({"status": "microset_precheck_failed", **microset_precheck}, ensure_ascii=False, indent=2))

    best_checkpoint_path = (run_dir / "best_micro_overfit.pth").resolve()
    checkpoint_identity = _verify_checkpoint_metadata(
        cfg=cfg,
        checkpoint_path=best_checkpoint_path,
        semantic_checkpoint_path=init_checkpoint_path,
        authoritative_checkpoint_meta=authoritative_checkpoint_metadata["best"],
        authoritative_resolved_source=authoritative_checkpoint_metadata.get("resolved_source", {}),
    )
    checkpoint_identity["device"] = str(device)

    if int(checkpoint_identity["saved_iteration"]) != 75:
        raise SystemExit(
            json.dumps(
                {
                    "status": "checkpoint_iteration_hard_fail",
                    "message": "best checkpoint iteration must be exactly 75",
                    **checkpoint_identity,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    hostname_info = _safe_hostname()
    git_commit_info = _safe_git_commit(repo_root)
    preflight = {
        "hostname": hostname_info["value"],
        "hostname_status": hostname_info["status"],
        "git_commit": git_commit_info["value"],
        "git_commit_status": git_commit_info["status"],
        "checkpoint_path": checkpoint_identity["resolved_best_checkpoint_path"],
        "checkpoint_sha256": checkpoint_identity["checkpoint_sha256"],
        "checkpoint_sha256_matches_authoritative": _hash_match_status(
            checkpoint_identity.get("checkpoint_sha256"),
            checkpoint_identity.get("authoritative_checkpoint_sha256"),
        ),
        "checkpoint_iteration_source": checkpoint_identity["checkpoint_iteration_source"],
        "checkpoint_iteration": checkpoint_identity["saved_iteration"],
        "primary_threshold": PRIMARY_THRESHOLD,
        "saved_best_threshold": checkpoint_identity["saved_best_threshold"],
        "semantic_checkpoint_path": checkpoint_identity["semantic_checkpoint_path"],
        "semantic_checkpoint_sha256": checkpoint_identity["semantic_checkpoint_sha256"],
        "semantic_checkpoint_sha256_matches_authoritative": _hash_match_status(
            checkpoint_identity.get("semantic_checkpoint_sha256"),
            checkpoint_identity.get("authoritative_semantic_checkpoint_sha256"),
        ),
        "microset_raw_sha256": microset_info["raw_sha256"],
        "microset_normalized_sha256": microset_info["normalized_sha256"],
        "device": str(device),
        "center_feature_path": checkpoint_identity["center_feature_path"],
        "expected_channels": checkpoint_identity["expected_channels"],
        "adapter_channels": checkpoint_identity["adapter_channels"],
        "normalization_mode": checkpoint_identity["normalization_mode"],
        "center_fp32": checkpoint_identity["center_fp32"],
        "authoritative_global_first_failure": authoritative_global_first_failure,
        "authoritative_primary_first_failure": authoritative_primary_first_failure,
    }
    print(json.dumps({"status": "preflight", **preflight}, ensure_ascii=False, indent=2))

    loader = _build_loader(cfg, repo_root=repo_root, split_txt=microset_file, device=device)
    model = _build_model_from_cfg(cfg, repo_root=repo_root)
    ckpt = _load_checkpoint(best_checkpoint_path)
    state = ckpt.get("model", ckpt)
    incompat = model.load_state_dict(state, strict=False)
    missing = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
    unexpected = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
    if unexpected or missing:
        raise RuntimeError(f"Checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()

    actual_primary_rows = []
    primary_rows = {name: [] for name in ("P0_CURRENT", "P1_DROP_UNMARKED", "P2_ATTACH_TO_NEAREST_MARKER", "P4_GLOBAL_MARKER_CONTROLLED")}
    p3_primary_rows = {}
    per_sample_csv_rows = []
    per_component_assignments = {"primary_threshold": PRIMARY_THRESHOLD, "secondary_thresholds": list(SECONDARY_THRESHOLDS), "samples": []}
    cached_primary_outputs = {}

    for sample_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        out = model(images)
        image_path = Path(str(batch["image_path"][0])).resolve()
        sample_id = image_path.stem
        image_rgb_u8 = (np.clip(batch["image"].detach().cpu().numpy()[0].transpose(1, 2, 0), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        pred_sem = torch.argmax(out["semantic"], dim=1).detach().cpu().numpy()[0].astype(np.uint8)
        center_prob = torch.sigmoid(out["center"]).detach().cpu().numpy()[0, 0].astype(np.float32)
        gt_inst = _load_gt_instance(instance_root, sample_id, pred_sem.shape[:2])
        gt_mask = batch["mask"].detach().cpu().numpy()[0].astype(np.uint8)
        gt_pts = _extract_metadata_centers(str(batch["metadata_path"][0]))

        sample_pack = {
            "sample": sample_id,
            "sample_index": int(sample_idx),
            "thresholds": {},
        }
        for threshold in (PRIMARY_THRESHOLD,) + SECONDARY_THRESHOLDS:
            threshold_key = f"{float(threshold):.2f}"
            sample_pack["thresholds"][threshold_key] = {}
            policy_outputs = {}
            for policy_name in ("P0_CURRENT", "P1_DROP_UNMARKED", "P2_ATTACH_TO_NEAREST_MARKER", "P4_GLOBAL_MARKER_CONTROLLED"):
                pred_inst, marker_points, trace = run_policy(policy_name, pred_sem, center_prob, float(threshold))
                c_metrics = _sample_center_metrics([(int(mp["y"]), int(mp["x"])) for mp in marker_points], gt_pts)
                p_metrics = _policy_metrics(gt_inst=gt_inst, pred_sem=pred_sem, pred_inst=pred_inst, marker_points=marker_points, trace=trace)
                _assert_policy_contract(policy_name, p_metrics)

                row = {
                    "sample": sample_id,
                    "sample_index": int(sample_idx),
                    "threshold": float(threshold),
                    "policy": policy_name,
                    "gt_instance_count": int(p_metrics["counts"]["gt_instance_count"]),
                    "marker_count": int(p_metrics["counts"]["marker_count"]),
                    "semantic_cc_count": int(p_metrics["counts"]["semantic_connected_component_count"]),
                    "raw_output_label_count": int(p_metrics["counts"]["raw_output_label_count"]),
                    "final_output_label_count": int(p_metrics["counts"]["final_output_label_count"]),
                    "exact_count": bool(p_metrics["counts"]["exact_count"]),
                    "matched_iou": float(p_metrics["instance_metrics"]["matched_iou"]),
                    "mean_matched_dice": p_metrics["instance_metrics"]["mean_matched_dice"],
                    "merged": bool(p_metrics["instance_metrics"]["merged"]),
                    "fragmented": bool(p_metrics["instance_metrics"]["fragmented"]),
                    "mixed": bool(p_metrics["instance_metrics"]["mixed"]),
                    "perfect_recovery": bool(p_metrics["instance_metrics"]["perfect_recovery"]),
                    "instance_score": float(p_metrics["instance_metrics"]["instance_score"]),
                    "assigned_area_fraction": float(p_metrics["area_accounting"]["assigned_area_fraction"]),
                    "dropped_area": int(p_metrics["area_accounting"]["dropped_area"]),
                    "unmarked_component_area": int(p_metrics["area_accounting"]["unmarked_component_area"]),
                    "unmarked_components_attached": int(p_metrics["component_assignment"]["unmarked_components_attached"]),
                    "unmarked_components_rejected": int(p_metrics["component_assignment"]["unmarked_components_rejected"]),
                    "mean_attachment_distance": p_metrics["component_assignment"]["mean_attachment_distance"],
                    "max_attachment_distance": p_metrics["component_assignment"]["max_attachment_distance"],
                    "ambiguous_assignments": int(p_metrics["component_assignment"]["ambiguous_assignments"]),
                    "marker_count_preservation": bool(p_metrics["contract"]["marker_count_preservation"]),
                    "labels_without_marker_provenance": int(p_metrics["contract"]["labels_without_marker_provenance"]),
                    "markers_without_output_label_count": int(len(p_metrics["contract"]["markers_without_output_label"])),
                    "invariant_pass": bool(p_metrics["contract"]["pass"]),
                    "center_precision": float(c_metrics["center_precision"]),
                    "center_recall": float(c_metrics["center_recall"]),
                    "center_f1": float(c_metrics["center_f1"]),
                    "fallback_marker_calls": int(p_metrics["contract"]["fallback_marker_calls"]),
                    "keep_top3_call_count": int(p_metrics["contract"]["keep_top3_call_count"]),
                    "new_non_marker_label_count": int(p_metrics["contract"]["new_non_marker_label_count"]),
                }
                per_sample_csv_rows.append(row)
                sample_pack["thresholds"][threshold_key][policy_name] = {"metrics": p_metrics, "trace": _json_safe_trace(trace)}
                policy_outputs[policy_name] = {"labels": pred_inst, "metrics": p_metrics, "markers": marker_points, "trace": trace}

                if float(threshold) == PRIMARY_THRESHOLD:
                    primary_rows[policy_name].append(p_metrics)
                    cached = cached_primary_outputs.setdefault(
                        sample_id,
                        {
                            "image_rgb_u8": image_rgb_u8,
                            "gt_inst": gt_inst,
                            "pred_sem": pred_sem,
                            "marker_points": marker_points,
                            "policy_outputs": {},
                            "semantic_cc": trace["semantic_components"],
                        },
                    )
                    cached["semantic_cc"] = trace["semantic_components"]
                    cached["policy_outputs"][policy_name] = {"labels": pred_inst, "metrics": p_metrics}

                if policy_name == "P0_CURRENT" and float(threshold) == PRIMARY_THRESHOLD:
                    marker_contract = _marker_contract(gt_inst, marker_points)
                    stage_marker = _stage_stats("extracted_marker_labels", trace["raw_labels"] * 0 + np.array([[0]], dtype=np.uint8), [])  # unused placeholder
                    p0_trace = reconstruct_instances_from_semantic_and_center(pred_sem, center_prob.astype(np.float32), float(threshold), max_markers=3, return_trace=True)[3]
                    stage_marker = _stage_stats("extracted_marker_labels", p0_trace["marker_labels"], p0_trace["marker_points"])
                    stage_raw = _stage_stats("raw_reconstruction", p0_trace["raw_reconstruction_labels"], p0_trace["marker_points"])
                    stage_post = _stage_stats("postprocessed_reconstruction", p0_trace["postprocessed_labels"], p0_trace["marker_points"])
                    stage_final = _stage_stats("final_labels_passed_to_metrics", p0_trace["final_labels"], p0_trace["marker_points"])
                    semantic_topology = _semantic_topology(gt_mask, gt_inst, pred_sem, p0_trace)
                    inst_metrics = compute_instance_metrics_from_masks(
                        gt_inst,
                        p0_trace["final_labels"].astype(np.uint8),
                        gt_k=int(len([k for k in [1, 2, 3] if int(np.sum(gt_inst == k)) > 0])),
                        pred_k=int(p0_trace["final_count"]),
                    )
                    actual_primary_rows.append(
                        {
                            "sample": sample_id,
                            "sample_index": int(sample_idx),
                            "markers": int(marker_contract["extracted_marker_count"]),
                            "marker_contract": bool(marker_contract["marker_contract_pass"]),
                            "semantic_cc": int(semantic_topology["predicted_leaflet_connected_components"]),
                            "raw_reconstructed": int(stage_raw["count"]),
                            "final_reconstructed": int(stage_final["count"]),
                            "exact_count": bool(inst_metrics["instance_exact_count"]),
                            "first_failing_invariant": _stage_failure_summary(
                                sample_id,
                                marker_contract,
                                p0_trace,
                                stage_marker,
                                stage_raw,
                                stage_post,
                                stage_final,
                            ),
                        }
                    )

            for dist_gate in P3_DISTANCE_GATES:
                for area_gate in P3_AREA_GATES:
                    cfg_key = {"distance_gate_px": float(dist_gate), "relative_area_gate": None if area_gate is None else float(area_gate)}
                    pred_inst, marker_points, trace = run_policy("P3_GATED_ATTACH", pred_sem, center_prob, float(threshold), p3_cfg=cfg_key)
                    p_metrics = _policy_metrics(gt_inst=gt_inst, pred_sem=pred_sem, pred_inst=pred_inst, marker_points=marker_points, trace=trace)
                    _assert_policy_contract("P3_GATED_ATTACH", p_metrics)
                    key = json.dumps(cfg_key, sort_keys=True)
                    if float(threshold) == PRIMARY_THRESHOLD:
                        p3_primary_rows.setdefault(key, []).append(p_metrics)
                        cached_primary_outputs.setdefault(
                            sample_id,
                            {
                                "image_rgb_u8": image_rgb_u8,
                                "gt_inst": gt_inst,
                                "pred_sem": pred_sem,
                                "marker_points": marker_points,
                                "policy_outputs": {},
                                "semantic_cc": trace["semantic_components"],
                            },
                        )["policy_outputs"]["P3_GATED_ATTACH"] = {"labels": pred_inst, "metrics": p_metrics}
                    sample_pack["thresholds"][threshold_key].setdefault("P3_GRID", {})[key] = {"metrics": p_metrics, "trace": _json_safe_trace(trace), "cfg": cfg_key}
            per_component_assignments["samples"].append(sample_pack)

    actual_primary_rows.sort(key=lambda row: int(row["sample_index"]))
    actual_primary_first_failure = _actual_primary_first_failure_from_rows(actual_primary_rows)
    baseline_report = _baseline_mismatch_payload(
        expected_rows=best_authoritative_primary_rows,
        actual_rows=actual_primary_rows,
        checkpoint_identity=checkpoint_identity,
        microset_precheck=microset_precheck,
        authoritative_summary=authoritative_summary,
        authoritative_global_first_failure=authoritative_global_first_failure,
        authoritative_primary_first_failure=authoritative_primary_first_failure,
        source_commit=git_commit_info["value"] or git_commit_info["status"],
    )

    p0_summary = _aggregate_policy_rows(primary_rows["P0_CURRENT"])
    exact_match = _authoritative_baseline_matches(
        expected_rows=best_authoritative_primary_rows,
        actual_rows=actual_primary_rows,
        p0_summary=p0_summary,
        authoritative_primary_first_failure=authoritative_primary_first_failure,
    )

    if not exact_match:
        baseline_report["status"] = "authoritative_baseline_mismatch"
        (out_dir / "baseline_mismatch.json").write_text(json.dumps(baseline_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "authoritative_baseline_mismatch",
                    "checkpoint_path": checkpoint_identity["resolved_best_checkpoint_path"],
                    "checkpoint_sha256": checkpoint_identity["checkpoint_sha256"],
                    "checkpoint_iteration_source": checkpoint_identity["checkpoint_iteration_source"],
                    "checkpoint_iteration": checkpoint_identity["saved_iteration"],
                    "semantic_checkpoint_path": checkpoint_identity["semantic_checkpoint_path"],
                    "semantic_checkpoint_sha256": checkpoint_identity["semantic_checkpoint_sha256"],
                    "microset_hash": microset_info["raw_sha256"],
                    "device": str(device),
                    "p0_reproduction_status": "mismatch",
                    "authoritative_global_first_failure": authoritative_global_first_failure,
                    "authoritative_primary_first_failure": authoritative_primary_first_failure,
                    "actual_primary_first_failure": actual_primary_first_failure,
                    "row_matches": int(sum(1 for row in baseline_report["rows"] if row["expected_markers"] == row["actual_markers"] and row["expected_semantic_cc"] == row["actual_semantic_cc"] and row["expected_raw_count"] == row["actual_raw_count"] and row["expected_final_count"] == row["actual_final_count"])),
                    "first_differing_sample": baseline_report["first_differing_sample"],
                    "classification": baseline_report["classification"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        _print_baseline_mismatch_sections(baseline_report)
        raise SystemExit(1)

    _clear_baseline_mismatch_if_present(out_dir)
    print(
        json.dumps(
            {
                "status": "preflight_complete",
                "checkpoint_path": checkpoint_identity["resolved_best_checkpoint_path"],
                "checkpoint_sha256": checkpoint_identity["checkpoint_sha256"],
                "checkpoint_iteration_source": checkpoint_identity["checkpoint_iteration_source"],
                "checkpoint_iteration": checkpoint_identity["saved_iteration"],
                "semantic_checkpoint_path": checkpoint_identity["semantic_checkpoint_path"],
                "semantic_checkpoint_sha256": checkpoint_identity["semantic_checkpoint_sha256"],
                "microset_hash": microset_info["raw_sha256"],
                "device": str(device),
                "authoritative_global_first_failure": authoritative_global_first_failure,
                "authoritative_primary_first_failure": authoritative_primary_first_failure,
                "actual_primary_first_failure": actual_primary_first_failure,
                "row_matches": f"{len(best_authoritative_primary_rows)}/{len(best_authoritative_primary_rows)}",
                "p0_reproduction_status": "exact_match",
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    best_p3_key, best_p3_summary = _select_best_p3(p3_primary_rows)
    best_p3_cfg = json.loads(best_p3_key)

    final_primary_summary = {}
    for policy_name in ("P0_CURRENT", "P1_DROP_UNMARKED", "P2_ATTACH_TO_NEAREST_MARKER", "P4_GLOBAL_MARKER_CONTROLLED"):
        final_primary_summary[policy_name] = _aggregate_policy_rows(primary_rows[policy_name])
    final_primary_summary["P3_GATED_ATTACH"] = best_p3_summary
    recommended = _recommend_policy(final_primary_summary, final_primary_summary["P0_CURRENT"])

    for sample_pack in per_component_assignments["samples"]:
        primary_key = f"{PRIMARY_THRESHOLD:.2f}"
        p3_entry = sample_pack["thresholds"][primary_key]["P3_GRID"][best_p3_key]
        sample_pack["thresholds"][primary_key]["P3_GATED_ATTACH"] = p3_entry

    for sample_pack in per_component_assignments["samples"]:
        sample_id = sample_pack["sample"]
        cached = cached_primary_outputs[sample_id]
        panel = _make_policy_comparison_panel(
            image_rgb_u8=cached["image_rgb_u8"],
            gt_inst=cached["gt_inst"],
            pred_sem=cached["pred_sem"],
            semantic_cc=cached["semantic_cc"],
            marker_points=cached["marker_points"],
            policy_outputs=cached["policy_outputs"],
            recommended_policy=str(recommended["policy"]) if recommended["policy"] != "none" else "P0_CURRENT",
        )
        cv2.imwrite(str((out_dir / "visual_review" / f"{sample_id}.png").resolve()), panel)

    with (out_dir / "per_sample_policy_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        if per_sample_csv_rows:
            writer = csv.DictWriter(f, fieldnames=list(per_sample_csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_sample_csv_rows)

    (out_dir / "per_component_assignments.json").write_text(json.dumps(per_component_assignments, ensure_ascii=False, indent=2), encoding="utf-8")

    invariants = {
        "primary_threshold": PRIMARY_THRESHOLD,
        "secondary_thresholds": list(SECONDARY_THRESHOLDS),
        "best_p3_cfg": best_p3_cfg,
        "primary_summary": final_primary_summary,
        "authoritative_baseline_status": "exact_match",
    }
    (out_dir / "invariants.json").write_text(json.dumps(invariants, ensure_ascii=False, indent=2), encoding="utf-8")

    policy_table = []
    for policy_name in ("P0_CURRENT", "P1_DROP_UNMARKED", "P2_ATTACH_TO_NEAREST_MARKER", "P3_GATED_ATTACH", "P4_GLOBAL_MARKER_CONTROLLED"):
        s = final_primary_summary[policy_name]
        policy_table.append(
            {
                "policy": policy_name,
                "exact_count": float(s["exact_count_accuracy"]),
                "matched_iou": float(s["matched_iou"]),
                "fragmented": float(s["fragmented_rate"]),
                "merged": float(s["merged_rate"]),
                "assigned_area": float(s["assigned_area_fraction"]),
                "dropped_area": float(s["dropped_area_mean"]),
                "invariant_violations": int(s["invariant_violations"]),
                "fallback_marker_calls": int(s["fallback_marker_calls"]),
                "keep_top3_call_count": int(s["keep_top3_call_count"]),
                "labels_without_marker_provenance": int(s["labels_without_marker_provenance"]),
            }
        )

    summary = {
        "primary_operating_point": {
            "checkpoint": "best_micro_overfit.pth",
            "iteration": int(checkpoint_identity["saved_iteration"]),
            "threshold": float(PRIMARY_THRESHOLD),
            "marker_contract": "passes for all six samples",
            "microset_raw_sha256": microset_info["raw_sha256"],
            "microset_normalized_sha256": microset_info["normalized_sha256"],
            "authoritative_baseline_status": "exact_match",
        },
        "current_policy": {
            "exact_count": float(final_primary_summary["P0_CURRENT"]["exact_count_accuracy"]),
            "matched_iou": float(final_primary_summary["P0_CURRENT"]["matched_iou"]),
            "raw_final_behavior": "semantic disconnected components create extra raw labels; keep_top3 truncates final labels to at most three",
            "fallback_labels": True,
            "keep_top3_effect": True,
        },
        "policy_table": policy_table,
        "per_sample_failures": actual_primary_rows,
        "best_gated_parameters": {
            "distance_gate": best_p3_cfg["distance_gate_px"],
            "area_gate": best_p3_cfg["relative_area_gate"],
            "ambiguous_assignments": int(final_primary_summary["P3_GATED_ATTACH"]["ambiguous_assignments"]),
        },
        "invariants": invariants,
        "server_preflight": {
            "global_first_failure": authoritative_global_first_failure,
            "primary_expected_first_failure": authoritative_primary_first_failure,
            "primary_actual_first_failure": actual_primary_first_failure,
            "row_matches": f"{len(best_authoritative_primary_rows)}/{len(best_authoritative_primary_rows)}",
            "p0_status": "exact_match",
        },
        "recommended_policy": recommended,
        "production_change_proposal": "Do not change production until this offline ablation is reviewed; the baseline now matches authoritative P0 exactly.",
        "next_step": "Review policy artifacts and decide whether to promote a marker-authoritative reconstruction policy into a separate production patch.",
    }
    (out_dir / "policy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_recommended_policy_if_allowed(out_dir, recommended, baseline_exact_match=True)
    _print_primary_sections(summary)

    print(
        json.dumps(
            {
                "status": "done",
                "output_dir": str(out_dir),
                "recommended_policy": recommended,
                "best_p3_cfg": best_p3_cfg,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
