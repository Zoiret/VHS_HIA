from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import leaflet_oracle_count_geometric_split_audit as base_audit
import leaflet_oracle_count_geometric_split_forensic as forensic


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "training" / "analysis" / "leaflet_oracle_k_constrained_normalization_audit"
ORIGINAL_AUDIT_DIR = REPO_ROOT / "training" / "analysis" / "leaflet_oracle_count_geometric_split_audit"
FORENSIC_AUDIT_DIR = REPO_ROOT / "training" / "analysis" / "leaflet_oracle_count_geometric_split_forensic"
CENTER_REFERENCE = {
    "strict_marker_contract": 0.3655913978494624,
    "instance_exact_count": 0.5913978494623656,
    "instance_mean_matched_iou": 0.572104696693949,
    "instance_score": 0.5043627612100781,
}


@dataclass(frozen=True)
class NormalizerSpec:
    key: str
    family: str
    merge_rule: str
    split_rule: str
    params: dict[str, Any]


NORMALIZER_SPECS: list[NormalizerSpec] = [
    NormalizerSpec(
        key="nearest_component_k_normalizer",
        family="k_normalizer",
        merge_rule="minimum boundary-to-boundary distance agglomeration",
        split_rule="distance-transform watershed with exact-seed fallback",
        params={},
    ),
    NormalizerSpec(
        key="area_aware_k_normalizer",
        family="k_normalizer",
        merge_rule="boundary distance with large-large merge penalty",
        split_rule="distance-transform watershed with exact-seed fallback",
        params={},
    ),
    NormalizerSpec(
        key="centroid_distance_k_normalizer",
        family="k_normalizer",
        merge_rule="centroid distance agglomeration",
        split_rule="distance-transform watershed with exact-seed fallback",
        params={},
    ),
    NormalizerSpec(
        key="hybrid_k_normalization",
        family="k_normalizer",
        merge_rule="small-fragment assignment then nearest agglomeration",
        split_rule="distance-transform watershed with exact-seed fallback",
        params={"small_ratio": 0.35},
    ),
    NormalizerSpec(
        key="gt_fragment_grouping_oracle",
        family="diagnostic_oracle",
        merge_rule="GT-overlap fragment assignment",
        split_rule="not used",
        params={},
    ),
]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _positive_ids(labels_u8: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(labels_u8) if int(v) > 0]


def _cc_labels(mask01: np.ndarray) -> tuple[np.ndarray, int]:
    return base_audit._connected_components(mask01.astype(np.uint8))


def _component_masks(mask01: np.ndarray) -> dict[int, np.ndarray]:
    labels_cc, cc_k = _cc_labels(mask01.astype(np.uint8))
    return {int(comp_id): (labels_cc == comp_id).astype(np.uint8) for comp_id in range(1, int(cc_k) + 1)}


def _boundary_mask(mask01: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask01.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1)
    return ((mask01.astype(np.uint8) > 0) & (eroded == 0)).astype(np.uint8)


def _centroid(mask01: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask01.astype(bool))
    if ys.size == 0:
        return (0.0, 0.0)
    return (float(np.mean(ys)), float(np.mean(xs)))


def _pairwise_boundary_distances(comp_masks: dict[int, np.ndarray]) -> dict[tuple[int, int], float]:
    cache: dict[tuple[int, int], float] = {}
    boundaries = {cid: _boundary_mask(mask) for cid, mask in comp_masks.items()}
    for a in comp_masks:
        for b in comp_masks:
            if int(a) >= int(b):
                continue
            b_boundary = boundaries[int(b)]
            inv = (1 - b_boundary.astype(np.uint8)).astype(np.uint8)
            dt = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
            a_pts = boundaries[int(a)].astype(bool)
            if not np.any(a_pts):
                cache[(int(a), int(b))] = float("inf")
            else:
                cache[(int(a), int(b))] = float(np.min(dt[a_pts]))
    return cache


def _pairwise_centroid_distances(comp_masks: dict[int, np.ndarray]) -> dict[tuple[int, int], float]:
    cache: dict[tuple[int, int], float] = {}
    cents = {cid: _centroid(mask) for cid, mask in comp_masks.items()}
    for a in comp_masks:
        for b in comp_masks:
            if int(a) >= int(b):
                continue
            ay, ax = cents[int(a)]
            by, bx = cents[int(b)]
            cache[(int(a), int(b))] = float(math.hypot(ay - by, ax - bx))
    return cache


def _group_area(group_component_ids: set[int], comp_masks: dict[int, np.ndarray]) -> int:
    return int(sum(int(np.sum(comp_masks[int(cid)])) for cid in group_component_ids))


def _group_mask(group_component_ids: set[int], comp_masks: dict[int, np.ndarray]) -> np.ndarray:
    masks = [comp_masks[int(cid)] for cid in sorted(group_component_ids)]
    out = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        out[mask > 0] = 1
    return out


def _group_distance(
    left_ids: set[int],
    right_ids: set[int],
    *,
    comp_masks: dict[int, np.ndarray],
    boundary_dists: dict[tuple[int, int], float],
    centroid_dists: dict[tuple[int, int], float],
    mode: str,
) -> float:
    if mode == "centroid":
        left_mask = _group_mask(left_ids, comp_masks)
        right_mask = _group_mask(right_ids, comp_masks)
        ly, lx = _centroid(left_mask)
        ry, rx = _centroid(right_mask)
        return float(math.hypot(ly - ry, lx - rx))
    dists: list[float] = []
    for left_id in left_ids:
        for right_id in right_ids:
            key = tuple(sorted((int(left_id), int(right_id))))
            dists.append(float(boundary_dists[key]))
    return float(min(dists)) if dists else float("inf")


def _dominant_gt_leaflet(mask01: np.ndarray, gt_inst_u8: np.ndarray) -> int:
    best_gt = 0
    best_overlap = -1
    for gt_id in _positive_ids(gt_inst_u8):
        overlap = int(np.sum((mask01 > 0) & (gt_inst_u8 == int(gt_id))))
        if overlap > best_overlap:
            best_gt = int(gt_id)
            best_overlap = overlap
    return int(best_gt)


def _profile_add_timing(profile: dict[str, Any] | None, key: str, seconds: float) -> None:
    if profile is None:
        return
    profile[key] = float(profile.get(key, 0.0)) + float(seconds)


def _profile_add_count(profile: dict[str, Any] | None, key: str, value: int = 1) -> None:
    if profile is None:
        return
    counts = profile.setdefault("call_counts", {})
    counts[key] = int(counts.get(key, 0)) + int(value)


def _merge_groups_exact_k_reference(
    mask01: np.ndarray,
    k: int,
    *,
    mode: str,
    gt_inst_u8: np.ndarray | None = None,
    small_fragment_ratio: float | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t_component_meta_start = time.perf_counter() if profile is not None else 0.0
    comp_masks = _component_masks(mask01)
    _profile_add_count(profile, "component_masks")
    if not comp_masks:
        return {
            "labels": np.zeros(mask01.shape, dtype=np.uint8),
            "exact_k_achieved": False,
            "reason": "empty_foreground",
            "merge_operations": [],
            "merge_count": 0,
            "fragments_assigned": 0,
            "initial_component_count": 0,
            "final_group_count": 0,
        }
    _profile_add_timing(profile, "component_filtering_statistics_seconds", float(time.perf_counter() - t_component_meta_start) if profile is not None else 0.0)
    t_boundary_start = time.perf_counter() if profile is not None else 0.0
    boundary_dists = _pairwise_boundary_distances(comp_masks)
    _profile_add_timing(profile, "distance_map_computation_seconds", float(time.perf_counter() - t_boundary_start) if profile is not None else 0.0)
    _profile_add_count(profile, "boundary_distance_pairs", int(len(boundary_dists)))
    t_centroid_start = time.perf_counter() if profile is not None else 0.0
    centroid_dists = _pairwise_centroid_distances(comp_masks)
    _profile_add_timing(profile, "seed_centroid_preparation_seconds", float(time.perf_counter() - t_centroid_start) if profile is not None else 0.0)
    _profile_add_count(profile, "centroid_distance_pairs", int(len(centroid_dists)))
    groups: dict[int, set[int]] = {int(cid): {int(cid)} for cid in comp_masks}
    merge_ops: list[dict[str, Any]] = []
    total_area = float(sum(int(np.sum(mask)) for mask in comp_masks.values()))

    def _choose_pair(strategy_mode: str) -> tuple[int, int, float]:
        ranked: list[tuple[float, float, float, int, int]] = []
        group_ids = sorted(groups)
        choose_start = time.perf_counter() if profile is not None else 0.0
        for idx, left_gid in enumerate(group_ids):
            for right_gid in group_ids[idx + 1 :]:
                left_ids = groups[int(left_gid)]
                right_ids = groups[int(right_gid)]
                _profile_add_count(profile, "group_pair_evaluations")
                boundary_dist = _group_distance(
                    left_ids,
                    right_ids,
                    comp_masks=comp_masks,
                    boundary_dists=boundary_dists,
                    centroid_dists=centroid_dists,
                    mode="boundary",
                )
                centroid_dist = _group_distance(
                    left_ids,
                    right_ids,
                    comp_masks=comp_masks,
                    boundary_dists=boundary_dists,
                    centroid_dists=centroid_dists,
                    mode="centroid",
                )
                _profile_add_count(profile, "group_mask_rebuilds", 2)
                _profile_add_count(profile, "centroid_recomputations", 2)
                left_area = float(_group_area(left_ids, comp_masks))
                right_area = float(_group_area(right_ids, comp_masks))
                if strategy_mode == "centroid":
                    score = centroid_dist
                elif strategy_mode == "area_aware":
                    score = boundary_dist * (1.0 + min(left_area, right_area) / max(left_area, right_area, 1.0))
                else:
                    score = boundary_dist
                ranked.append((float(score), float(boundary_dist), float(min(left_area, right_area)), int(left_gid), int(right_gid)))
        ranked.sort()
        _profile_add_timing(profile, "per_component_python_loops_seconds", float(time.perf_counter() - choose_start) if profile is not None else 0.0)
        _profile_add_timing(profile, "centroid_distance_computation_seconds", float(time.perf_counter() - choose_start) if profile is not None and strategy_mode == "centroid" else 0.0)
        score, boundary_dist, _min_area, left_gid, right_gid = ranked[0]
        return int(left_gid), int(right_gid), float(boundary_dist)

    if small_fragment_ratio is not None:
        while len(groups) > int(k):
            areas = {gid: float(_group_area(comp_ids, comp_masks)) for gid, comp_ids in groups.items()}
            smallest_gid = min(groups, key=lambda gid: (areas[int(gid)], int(gid)))
            larger = [gid for gid in groups if int(gid) != int(smallest_gid) and areas[int(gid)] > areas[int(smallest_gid)]]
            if not larger:
                break
            target_gid = min(
                larger,
                key=lambda gid: (
                    _group_distance(
                        groups[int(smallest_gid)],
                        groups[int(gid)],
                        comp_masks=comp_masks,
                        boundary_dists=boundary_dists,
                        centroid_dists=centroid_dists,
                        mode="boundary",
                    ),
                    int(gid),
                ),
            )
            if areas[int(smallest_gid)] > float(small_fragment_ratio) * areas[int(target_gid)]:
                break
            boundary_dist = _group_distance(
                groups[int(smallest_gid)],
                groups[int(target_gid)],
                comp_masks=comp_masks,
                boundary_dists=boundary_dists,
                centroid_dists=centroid_dists,
                mode="boundary",
            )
            left_gid, right_gid = (int(target_gid), int(smallest_gid)) if int(target_gid) < int(smallest_gid) else (int(smallest_gid), int(target_gid))
            left_ids = sorted(groups[int(left_gid)])
            right_ids = sorted(groups[int(right_gid)])
            same_gt = None
            if gt_inst_u8 is not None:
                same_gt = int(
                    _dominant_gt_leaflet(_group_mask(groups[int(left_gid)], comp_masks), gt_inst_u8)
                    == _dominant_gt_leaflet(_group_mask(groups[int(right_gid)], comp_masks), gt_inst_u8)
                )
            groups[int(left_gid)] = set(groups[int(left_gid)]) | set(groups[int(right_gid)])
            del groups[int(right_gid)]
            merge_ops.append(
                {
                    "left_group": int(left_gid),
                    "right_group": int(right_gid),
                    "left_component_ids": json.dumps(left_ids),
                    "right_component_ids": json.dumps(right_ids),
                    "left_area": int(sum(int(np.sum(comp_masks[int(cid)])) for cid in left_ids)),
                    "right_area": int(sum(int(np.sum(comp_masks[int(cid)])) for cid in right_ids)),
                    "minimum_boundary_distance": float(boundary_dist),
                    "chosen_target_group": int(left_gid),
                    "same_gt_leaflet": same_gt,
                    "merge_stage": "obvious_fragment_assignment",
                }
            )

    merge_mode = "centroid" if mode == "centroid" else ("area_aware" if mode == "area_aware" else "boundary")
    while len(groups) > int(k):
        left_gid, right_gid, boundary_dist = _choose_pair(merge_mode)
        left_ids = sorted(groups[int(left_gid)])
        right_ids = sorted(groups[int(right_gid)])
        same_gt = None
        if gt_inst_u8 is not None:
            same_gt = int(
                _dominant_gt_leaflet(_group_mask(groups[int(left_gid)], comp_masks), gt_inst_u8)
                == _dominant_gt_leaflet(_group_mask(groups[int(right_gid)], comp_masks), gt_inst_u8)
            )
        groups[int(left_gid)] = set(groups[int(left_gid)]) | set(groups[int(right_gid)])
        del groups[int(right_gid)]
        merge_ops.append(
            {
                "left_group": int(left_gid),
                "right_group": int(right_gid),
                "left_component_ids": json.dumps(left_ids),
                "right_component_ids": json.dumps(right_ids),
                "left_area": int(sum(int(np.sum(comp_masks[int(cid)])) for cid in left_ids)),
                "right_area": int(sum(int(np.sum(comp_masks[int(cid)])) for cid in right_ids)),
                "minimum_boundary_distance": float(boundary_dist),
                "chosen_target_group": int(left_gid),
                "same_gt_leaflet": same_gt,
                "merge_stage": "agglomeration",
            }
        )

    t_output_start = time.perf_counter() if profile is not None else 0.0
    final_groups = sorted(groups.values(), key=lambda comp_ids: (_centroid(_group_mask(comp_ids, comp_masks))[1], _centroid(_group_mask(comp_ids, comp_masks))[0], sorted(comp_ids)))
    out = np.zeros(mask01.shape, dtype=np.uint8)
    for label_id, comp_ids in enumerate(final_groups, start=1):
        for comp_id in comp_ids:
            out[comp_masks[int(comp_id)] > 0] = np.uint8(label_id)
    _profile_add_timing(profile, "output_instance_mask_creation_seconds", float(time.perf_counter() - t_output_start) if profile is not None else 0.0)
    return {
        "labels": out,
        "exact_k_achieved": bool(int(len(final_groups)) == int(k)),
        "reason": "exact_k" if int(len(final_groups)) == int(k) else "merge_failed",
        "merge_operations": merge_ops,
        "merge_count": int(len(merge_ops)),
        "fragments_assigned": int(max(len(comp_masks) - len(final_groups), 0)),
        "initial_component_count": int(len(comp_masks)),
        "final_group_count": int(len(final_groups)),
        "total_area": total_area,
    }


def _merge_groups_exact_k_centroid_optimized(
    mask01: np.ndarray,
    k: int,
    *,
    gt_inst_u8: np.ndarray | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t_component_meta_start = time.perf_counter() if profile is not None else 0.0
    comp_masks = _component_masks(mask01)
    _profile_add_count(profile, "component_masks")
    if not comp_masks:
        return {
            "labels": np.zeros(mask01.shape, dtype=np.uint8),
            "exact_k_achieved": False,
            "reason": "empty_foreground",
            "merge_operations": [],
            "merge_count": 0,
            "fragments_assigned": 0,
            "initial_component_count": 0,
            "final_group_count": 0,
        }
    component_ids = sorted(int(cid) for cid in comp_masks.keys())
    comp_areas = {int(cid): int(np.sum(comp_masks[int(cid)])) for cid in component_ids}
    comp_centroids = {int(cid): _centroid(comp_masks[int(cid)]) for cid in component_ids}
    gt_ids = _positive_ids(gt_inst_u8) if gt_inst_u8 is not None else []
    comp_gt_overlap = None
    if gt_inst_u8 is not None:
        comp_gt_overlap = {
            int(cid): {int(gt_id): int(np.sum((comp_masks[int(cid)] > 0) & (gt_inst_u8 == int(gt_id)))) for gt_id in gt_ids}
            for cid in component_ids
        }
    _profile_add_timing(profile, "component_filtering_statistics_seconds", float(time.perf_counter() - t_component_meta_start) if profile is not None else 0.0)
    t_boundary_start = time.perf_counter() if profile is not None else 0.0
    boundary_dists = _pairwise_boundary_distances(comp_masks)
    _profile_add_timing(profile, "distance_map_computation_seconds", float(time.perf_counter() - t_boundary_start) if profile is not None else 0.0)
    _profile_add_count(profile, "boundary_distance_pairs", int(len(boundary_dists)))
    _profile_add_timing(profile, "seed_centroid_preparation_seconds", 0.0)
    _profile_add_count(profile, "centroid_distance_pairs", int(len(boundary_dists)))

    groups: dict[int, set[int]] = {int(cid): {int(cid)} for cid in component_ids}
    group_area = {int(cid): int(comp_areas[int(cid)]) for cid in component_ids}
    group_centroid_sum_y = {int(cid): float(comp_centroids[int(cid)][0] * comp_areas[int(cid)]) for cid in component_ids}
    group_centroid_sum_x = {int(cid): float(comp_centroids[int(cid)][1] * comp_areas[int(cid)]) for cid in component_ids}
    group_boundary_min: dict[tuple[int, int], float] = {}
    for (a, b), dist in boundary_dists.items():
        group_boundary_min[(int(a), int(b))] = float(dist)
    group_gt_overlap: dict[int, dict[int, int]] | None = None
    if comp_gt_overlap is not None:
        group_gt_overlap = {int(cid): dict(comp_gt_overlap[int(cid)]) for cid in component_ids}
    merge_ops: list[dict[str, Any]] = []
    total_area = float(sum(int(area) for area in comp_areas.values()))

    def _boundary_min_between(left_gid: int, right_gid: int) -> float:
        key = (int(left_gid), int(right_gid)) if int(left_gid) < int(right_gid) else (int(right_gid), int(left_gid))
        return float(group_boundary_min[key])

    def _group_centroid(gid: int) -> tuple[float, float]:
        area = max(int(group_area[int(gid)]), 1)
        return (
            float(group_centroid_sum_y[int(gid)] / float(area)),
            float(group_centroid_sum_x[int(gid)] / float(area)),
        )

    def _dominant_group_gt(gid: int) -> int | None:
        if group_gt_overlap is None:
            return None
        best_gt = 0
        best_overlap = -1
        for gt_id in gt_ids:
            overlap = int(group_gt_overlap[int(gid)].get(int(gt_id), 0))
            if overlap > best_overlap:
                best_gt = int(gt_id)
                best_overlap = overlap
        return int(best_gt)

    def _choose_pair() -> tuple[int, int, float]:
        ranked: list[tuple[float, float, float, int, int]] = []
        group_ids = sorted(groups)
        choose_start = time.perf_counter() if profile is not None else 0.0
        for idx, left_gid in enumerate(group_ids):
            left_cy, left_cx = _group_centroid(int(left_gid))
            left_area = float(group_area[int(left_gid)])
            for right_gid in group_ids[idx + 1 :]:
                right_cy, right_cx = _group_centroid(int(right_gid))
                right_area = float(group_area[int(right_gid)])
                centroid_dist = float(math.hypot(left_cy - right_cy, left_cx - right_cx))
                boundary_dist = _boundary_min_between(int(left_gid), int(right_gid))
                ranked.append((centroid_dist, boundary_dist, float(min(left_area, right_area)), int(left_gid), int(right_gid)))
                _profile_add_count(profile, "group_pair_evaluations")
        ranked.sort()
        elapsed = float(time.perf_counter() - choose_start) if profile is not None else 0.0
        _profile_add_timing(profile, "per_component_python_loops_seconds", elapsed)
        _profile_add_timing(profile, "centroid_distance_computation_seconds", elapsed)
        score, boundary_dist, _min_area, left_gid, right_gid = ranked[0]
        return int(left_gid), int(right_gid), float(boundary_dist)

    while len(groups) > int(k):
        left_gid, right_gid, boundary_dist = _choose_pair()
        left_ids = sorted(groups[int(left_gid)])
        right_ids = sorted(groups[int(right_gid)])
        same_gt = None
        if group_gt_overlap is not None:
            same_gt = int(_dominant_group_gt(int(left_gid)) == _dominant_group_gt(int(right_gid)))
        groups[int(left_gid)] = set(groups[int(left_gid)]) | set(groups[int(right_gid)])
        del groups[int(right_gid)]
        group_area[int(left_gid)] = int(group_area[int(left_gid)] + group_area[int(right_gid)])
        group_centroid_sum_y[int(left_gid)] = float(group_centroid_sum_y[int(left_gid)] + group_centroid_sum_y[int(right_gid)])
        group_centroid_sum_x[int(left_gid)] = float(group_centroid_sum_x[int(left_gid)] + group_centroid_sum_x[int(right_gid)])
        if group_gt_overlap is not None:
            merged_overlap = {
                int(gt_id): int(group_gt_overlap[int(left_gid)].get(int(gt_id), 0) + group_gt_overlap[int(right_gid)].get(int(gt_id), 0))
                for gt_id in gt_ids
            }
            group_gt_overlap[int(left_gid)] = merged_overlap
            del group_gt_overlap[int(right_gid)]
        del group_area[int(right_gid)]
        del group_centroid_sum_y[int(right_gid)]
        del group_centroid_sum_x[int(right_gid)]
        keys_to_update = [key for key in group_boundary_min.keys() if int(right_gid) in key]
        for key in keys_to_update:
            a, b = key
            other_gid = int(b if int(a) == int(right_gid) else a)
            if int(other_gid) == int(left_gid) or int(other_gid) not in groups:
                del group_boundary_min[key]
                continue
            left_key = (int(left_gid), int(other_gid)) if int(left_gid) < int(other_gid) else (int(other_gid), int(left_gid))
            right_key = (int(right_gid), int(other_gid)) if int(right_gid) < int(other_gid) else (int(other_gid), int(right_gid))
            prev_left = group_boundary_min.get(left_key, float("inf"))
            prev_right = group_boundary_min.get(right_key, float("inf"))
            group_boundary_min[left_key] = float(min(prev_left, prev_right))
            if key in group_boundary_min:
                del group_boundary_min[key]
        merge_ops.append(
            {
                "left_group": int(left_gid),
                "right_group": int(right_gid),
                "left_component_ids": json.dumps(left_ids),
                "right_component_ids": json.dumps(right_ids),
                "left_area": int(sum(int(comp_areas[int(cid)]) for cid in left_ids)),
                "right_area": int(sum(int(comp_areas[int(cid)]) for cid in right_ids)),
                "minimum_boundary_distance": float(boundary_dist),
                "chosen_target_group": int(left_gid),
                "same_gt_leaflet": same_gt,
                "merge_stage": "agglomeration",
            }
        )

    t_output_start = time.perf_counter() if profile is not None else 0.0
    final_groups = sorted(
        groups.values(),
        key=lambda comp_ids: (
            float(sum(comp_centroids[int(cid)][1] * comp_areas[int(cid)] for cid in comp_ids) / max(sum(comp_areas[int(cid)] for cid in comp_ids), 1)),
            float(sum(comp_centroids[int(cid)][0] * comp_areas[int(cid)] for cid in comp_ids) / max(sum(comp_areas[int(cid)] for cid in comp_ids), 1)),
            sorted(comp_ids),
        ),
    )
    out = np.zeros(mask01.shape, dtype=np.uint8)
    for label_id, comp_ids in enumerate(final_groups, start=1):
        for comp_id in comp_ids:
            out[comp_masks[int(comp_id)] > 0] = np.uint8(label_id)
    _profile_add_timing(profile, "output_instance_mask_creation_seconds", float(time.perf_counter() - t_output_start) if profile is not None else 0.0)
    return {
        "labels": out,
        "exact_k_achieved": bool(int(len(final_groups)) == int(k)),
        "reason": "exact_k" if int(len(final_groups)) == int(k) else "merge_failed",
        "merge_operations": merge_ops,
        "merge_count": int(len(merge_ops)),
        "fragments_assigned": int(max(len(comp_masks) - len(final_groups), 0)),
        "initial_component_count": int(len(comp_masks)),
        "final_group_count": int(len(final_groups)),
        "total_area": total_area,
    }


def _farthest_seed_fallback(component01: np.ndarray, seeds: list[tuple[int, int]], dt: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.where(component01.astype(bool))
    if ys.size == 0:
        return None
    if not seeds:
        y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
        return (int(y), int(x))
    best = None
    best_key = None
    seed_arr = np.asarray(seeds, dtype=np.int32)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if any(int(y) == int(py) and int(x) == int(px) for py, px in seeds):
            continue
        d2 = np.min((seed_arr[:, 0] - int(y)) ** 2 + (seed_arr[:, 1] - int(x)) ** 2)
        key = (float(d2), float(dt[int(y), int(x)]), -int(y), -int(x))
        if best_key is None or key > best_key:
            best_key = key
            best = (int(y), int(x))
    return best


def _generate_exact_seeds(component01: np.ndarray, target_count: int) -> list[tuple[int, int]]:
    component01 = (component01.astype(np.uint8) > 0).astype(np.uint8)
    pixel_count = int(np.sum(component01))
    if pixel_count < int(target_count):
        return []
    dt = base_audit._distance_transform(component01)
    candidates = base_audit._peak_candidates(dt, component01, peak_floor_rel=0.08)
    seeds = base_audit._greedy_select_candidates(candidates, int(target_count), min_distance_px=9, suppression_scale=1.0)
    while len(seeds) < int(target_count):
        nxt = _farthest_seed_fallback(component01, seeds, dt)
        if nxt is None:
            break
        if nxt not in seeds:
            seeds.append(nxt)
        else:
            break
    seeds = seeds[: int(target_count)]
    if len(seeds) < int(target_count):
        return []
    return [(int(y), int(x)) for y, x in seeds]


def _seed_voronoi_partition(component01: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    fg_y, fg_x = np.where(component01.astype(bool))
    labels = np.zeros(component01.shape, dtype=np.uint8)
    if fg_y.size == 0 or not seeds:
        return labels
    seed_arr = np.asarray(seeds, dtype=np.float32)
    for y, x in zip(fg_y.tolist(), fg_x.tolist()):
        d2 = (seed_arr[:, 0] - float(y)) ** 2 + (seed_arr[:, 1] - float(x)) ** 2
        best_idx = int(np.argmin(d2))
        labels[int(y), int(x)] = np.uint8(best_idx + 1)
    return labels


def _split_single_mask_exact_k(mask01: np.ndarray, k: int) -> dict[str, Any]:
    mask01 = (mask01.astype(np.uint8) > 0).astype(np.uint8)
    labels_cc, cc_k = _cc_labels(mask01)
    if int(cc_k) == 0:
        return {
            "labels": np.zeros(mask01.shape, dtype=np.uint8),
            "exact_k_achieved": False,
            "reason": "empty_foreground",
            "split_count": 0,
            "seed_count": 0,
            "watershed_label_count": 0,
            "initial_component_count": 0,
        }
    if int(cc_k) > int(k):
        return {
            "labels": labels_cc.astype(np.uint8),
            "exact_k_achieved": False,
            "reason": "component_count_exceeds_target",
            "split_count": 0,
            "seed_count": 0,
            "watershed_label_count": int(cc_k),
            "initial_component_count": int(cc_k),
        }
    if int(np.sum(mask01)) < int(k):
        return {
            "labels": np.zeros(mask01.shape, dtype=np.uint8),
            "exact_k_achieved": False,
            "reason": "insufficient_pixels",
            "split_count": 0,
            "seed_count": 0,
            "watershed_label_count": 0,
            "initial_component_count": int(cc_k),
        }
    allocations: dict[int, int] = {int(comp_id): 1 for comp_id in range(1, int(cc_k) + 1)}
    extras = int(k) - int(cc_k)
    comp_meta: dict[int, dict[str, Any]] = {}
    for comp_id in range(1, int(cc_k) + 1):
        comp01 = (labels_cc == comp_id).astype(np.uint8)
        dt = base_audit._distance_transform(comp01)
        peaks = base_audit._peak_candidates(dt, comp01, peak_floor_rel=0.08)
        comp_meta[int(comp_id)] = {
            "area": int(np.sum(comp01)),
            "peak_count": int(len(peaks)),
            "max_dt": float(dt.max()) if dt.size else 0.0,
            "mask": comp01,
        }
    while extras > 0:
        chosen = max(
            allocations,
            key=lambda comp_id: (
                int(comp_meta[int(comp_id)]["peak_count"]) - int(allocations[int(comp_id)]),
                int(comp_meta[int(comp_id)]["area"]),
                float(comp_meta[int(comp_id)]["max_dt"]),
                -int(comp_id),
            ),
        )
        allocations[int(chosen)] += 1
        extras -= 1
    out = np.zeros(mask01.shape, dtype=np.uint8)
    next_label = 1
    total_seeds = 0
    total_ws_labels = 0
    for comp_id in range(1, int(cc_k) + 1):
        comp01 = comp_meta[int(comp_id)]["mask"]
        target_parts = int(allocations[int(comp_id)])
        if int(target_parts) == 1:
            out[comp01 > 0] = np.uint8(next_label)
            next_label += 1
            total_seeds += 1
            total_ws_labels += 1
            continue
        seeds = _generate_exact_seeds(comp01, int(target_parts))
        if len(seeds) != int(target_parts):
            return {
                "labels": out,
                "exact_k_achieved": False,
                "reason": "failed_to_generate_exact_seeds",
                "split_count": int(k - cc_k),
                "seed_count": total_seeds + int(len(seeds)),
                "watershed_label_count": total_ws_labels,
                "initial_component_count": int(cc_k),
            }
        topo = base_audit._geometry_topo_u8(comp01)
        seg = base_audit._watershed(comp01, seeds, topo)
        seg_labels = _positive_ids(seg)
        if len(seg_labels) != int(target_parts):
            seg = _seed_voronoi_partition(comp01, seeds)
            seg_labels = _positive_ids(seg)
        if len(seg_labels) != int(target_parts):
            return {
                "labels": out,
                "exact_k_achieved": False,
                "reason": "watershed_failed_to_produce_exact_k",
                "split_count": int(k - cc_k),
                "seed_count": total_seeds + int(len(seeds)),
                "watershed_label_count": total_ws_labels + int(len(seg_labels)),
                "initial_component_count": int(cc_k),
            }
        seg_labels_sorted = sorted(seg_labels)
        for seg_id in seg_labels_sorted:
            out[seg == int(seg_id)] = np.uint8(next_label)
            next_label += 1
        total_seeds += int(len(seeds))
        total_ws_labels += int(len(seg_labels))
    return {
        "labels": out,
        "exact_k_achieved": bool(int(next_label - 1) == int(k)),
        "reason": "exact_k" if int(next_label - 1) == int(k) else "split_failed",
        "split_count": int(k - cc_k),
        "seed_count": int(total_seeds),
        "watershed_label_count": int(total_ws_labels),
        "initial_component_count": int(cc_k),
    }


def _split_existing_groups_exact_k(group_labels_u8: np.ndarray, k: int) -> dict[str, Any]:
    group_ids = _positive_ids(group_labels_u8)
    if len(group_ids) == 0:
        return {
            "labels": np.zeros(group_labels_u8.shape, dtype=np.uint8),
            "exact_k_achieved": False,
            "reason": "empty_foreground",
            "split_count": 0,
            "seed_count": 0,
            "watershed_label_count": 0,
            "initial_group_count": 0,
        }
    if len(group_ids) > int(k):
        return {
            "labels": group_labels_u8.astype(np.uint8),
            "exact_k_achieved": False,
            "reason": "group_count_exceeds_target",
            "split_count": 0,
            "seed_count": 0,
            "watershed_label_count": int(len(group_ids)),
            "initial_group_count": int(len(group_ids)),
        }
    allocations: dict[int, int] = {int(gid): 1 for gid in group_ids}
    extras = int(k) - int(len(group_ids))
    meta: dict[int, dict[str, Any]] = {}
    for gid in group_ids:
        mask = (group_labels_u8 == int(gid)).astype(np.uint8)
        labels_cc, cc_k = _cc_labels(mask)
        dt = base_audit._distance_transform(mask)
        peaks = base_audit._peak_candidates(dt, mask, peak_floor_rel=0.08)
        meta[int(gid)] = {
            "mask": mask,
            "cc": int(cc_k),
            "area": int(np.sum(mask)),
            "peak_count": int(len(peaks)),
            "max_dt": float(dt.max()) if dt.size else 0.0,
        }
    while extras > 0:
        candidates = [gid for gid in group_ids if int(meta[int(gid)]["area"]) >= int(meta[int(gid)]["cc"]) + int(allocations[int(gid)])]
        if not candidates:
            return {
                "labels": group_labels_u8.astype(np.uint8),
                "exact_k_achieved": False,
                "reason": "no_splittable_group_after_fragment_assignment",
                "split_count": int(k - len(group_ids)),
                "seed_count": 0,
                "watershed_label_count": int(len(group_ids)),
                "initial_group_count": int(len(group_ids)),
            }
        chosen = max(
            candidates,
            key=lambda gid: (
                int(meta[int(gid)]["peak_count"]) - int(allocations[int(gid)]),
                int(meta[int(gid)]["area"]),
                float(meta[int(gid)]["max_dt"]),
                -int(gid),
            ),
        )
        allocations[int(chosen)] += 1
        extras -= 1
    out = np.zeros(group_labels_u8.shape, dtype=np.uint8)
    next_label = 1
    total_seeds = 0
    total_ws = 0
    for gid in sorted(group_ids):
        sub = _split_single_mask_exact_k(meta[int(gid)]["mask"], int(allocations[int(gid)]))
        if not bool(sub["exact_k_achieved"]):
            return {
                "labels": out,
                "exact_k_achieved": False,
                "reason": str(sub["reason"]),
                "split_count": int(k - len(group_ids)),
                "seed_count": total_seeds + int(sub["seed_count"]),
                "watershed_label_count": total_ws + int(sub["watershed_label_count"]),
                "initial_group_count": int(len(group_ids)),
            }
        for sub_id in _positive_ids(sub["labels"]):
            out[sub["labels"] == int(sub_id)] = np.uint8(next_label)
            next_label += 1
        total_seeds += int(sub["seed_count"])
        total_ws += int(sub["watershed_label_count"])
    return {
        "labels": out,
        "exact_k_achieved": bool(int(next_label - 1) == int(k)),
        "reason": "exact_k" if int(next_label - 1) == int(k) else "split_failed",
        "split_count": int(k - len(group_ids)),
        "seed_count": int(total_seeds),
        "watershed_label_count": int(total_ws),
        "initial_group_count": int(len(group_ids)),
    }


def normalize_mask_exact_k(
    mask01: np.ndarray,
    k: int,
    method_key: str,
    *,
    implementation: str = "optimized",
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t_prep_start = time.perf_counter() if profile is not None else 0.0
    mask01 = (mask01.astype(np.uint8) > 0).astype(np.uint8)
    _profile_add_timing(profile, "array_copy_dtype_conversion_seconds", float(time.perf_counter() - t_prep_start) if profile is not None else 0.0)
    if int(np.sum(mask01)) <= 0:
        return {
            "labels": np.zeros(mask01.shape, dtype=np.uint8),
            "exact_k_achieved": False,
            "final_group_count": 0,
            "initial_component_count": 0,
            "merge_count": 0,
            "split_count": 0,
            "fragments_assigned": 0,
            "reason": "empty_foreground",
            "merge_operations": [],
        }
    t_cc_start = time.perf_counter() if profile is not None else 0.0
    labels_cc, cc_k = _cc_labels(mask01)
    _profile_add_timing(profile, "connected_component_labeling_seconds", float(time.perf_counter() - t_cc_start) if profile is not None else 0.0)
    _profile_add_count(profile, "connected_component_calls")
    if int(cc_k) == int(k):
        natural = labels_cc.astype(np.uint8)
        return {
            "labels": natural,
            "exact_k_achieved": True,
            "final_group_count": int(k),
            "initial_component_count": int(cc_k),
            "merge_count": 0,
            "split_count": 0,
            "fragments_assigned": 0,
            "reason": "natural_components_match_k",
            "merge_operations": [],
        }
    if int(cc_k) < int(k):
        split = _split_single_mask_exact_k(mask01, int(k))
        return {
            "labels": split["labels"].astype(np.uint8),
            "exact_k_achieved": bool(split["exact_k_achieved"]),
            "final_group_count": int(len(_positive_ids(split["labels"]))),
            "initial_component_count": int(cc_k),
            "merge_count": 0,
            "split_count": int(split["split_count"]),
            "fragments_assigned": 0,
            "reason": str(split["reason"]),
            "merge_operations": [],
            "seed_count": int(split["seed_count"]),
            "watershed_label_count": int(split["watershed_label_count"]),
        }
    if method_key == "nearest_component_k_normalizer":
        merged = _merge_groups_exact_k_reference(mask01, int(k), mode="boundary", profile=profile)
    elif method_key == "area_aware_k_normalizer":
        merged = _merge_groups_exact_k_reference(mask01, int(k), mode="area_aware", profile=profile)
    elif method_key == "centroid_distance_k_normalizer":
        if str(implementation) == "reference":
            merged = _merge_groups_exact_k_reference(mask01, int(k), mode="centroid", profile=profile)
        else:
            merged = _merge_groups_exact_k_centroid_optimized(mask01, int(k), profile=profile)
    elif method_key == "hybrid_k_normalization":
        merged = _merge_groups_exact_k_reference(mask01, int(k), mode="boundary", small_fragment_ratio=0.35, profile=profile)
        if int(merged["final_group_count"]) < int(k):
            split = _split_existing_groups_exact_k(merged["labels"], int(k))
            return {
                "labels": split["labels"].astype(np.uint8),
                "exact_k_achieved": bool(split["exact_k_achieved"]),
                "final_group_count": int(len(_positive_ids(split["labels"]))),
                "initial_component_count": int(cc_k),
                "merge_count": int(merged["merge_count"]),
                "split_count": int(split["split_count"]),
                "fragments_assigned": int(merged["fragments_assigned"]),
                "reason": str(split["reason"]),
                "merge_operations": merged["merge_operations"],
                "seed_count": int(split["seed_count"]),
                "watershed_label_count": int(split["watershed_label_count"]),
            }
    else:
        raise ValueError(f"Unsupported deployable method_key: {method_key}")
    return {
        "labels": merged["labels"].astype(np.uint8),
        "exact_k_achieved": bool(merged["exact_k_achieved"]),
        "final_group_count": int(merged["final_group_count"]),
        "initial_component_count": int(merged["initial_component_count"]),
        "merge_count": int(merged["merge_count"]),
        "split_count": 0,
        "fragments_assigned": int(merged["fragments_assigned"]),
        "reason": str(merged["reason"]),
        "merge_operations": merged["merge_operations"],
        "seed_count": 0,
        "watershed_label_count": int(merged["final_group_count"]),
    }


def gt_fragment_grouping_oracle(mask01: np.ndarray, gt_inst_u8: np.ndarray, k: int) -> dict[str, Any]:
    mask01 = (mask01.astype(np.uint8) > 0).astype(np.uint8)
    labels_cc, cc_k = _cc_labels(mask01)
    if int(cc_k) == 0:
        return {
            "labels": np.zeros(mask01.shape, dtype=np.uint8),
            "exact_k_achieved": False,
            "final_group_count": 0,
            "initial_component_count": 0,
            "merge_count": 0,
            "split_count": 0,
            "fragments_assigned": 0,
            "reason": "empty_foreground",
            "merge_operations": [],
        }
    gt_ids = _positive_ids(gt_inst_u8)
    out = np.zeros(mask01.shape, dtype=np.uint8)
    merge_ops: list[dict[str, Any]] = []
    gt_to_components: dict[int, list[int]] = {int(gt_id): [] for gt_id in gt_ids}
    for comp_id in range(1, int(cc_k) + 1):
        comp_mask = (labels_cc == comp_id).astype(np.uint8)
        overlaps = {int(gt_id): int(np.sum((comp_mask > 0) & (gt_inst_u8 == int(gt_id)))) for gt_id in gt_ids}
        target_gt = min(gt_ids, key=lambda gt_id: (-int(overlaps[int(gt_id)]), int(gt_id)))
        gt_to_components[int(target_gt)].append(int(comp_id))
    ordered_gt = sorted(gt_ids)
    for new_label, gt_id in enumerate(ordered_gt, start=1):
        for comp_id in gt_to_components[int(gt_id)]:
            out[labels_cc == int(comp_id)] = np.uint8(new_label)
    for gt_id in ordered_gt:
        comps = gt_to_components[int(gt_id)]
        if len(comps) <= 1:
            continue
        base_comp = comps[0]
        for comp_id in comps[1:]:
            merge_ops.append(
                {
                    "left_group": int(base_comp),
                    "right_group": int(comp_id),
                    "left_component_ids": json.dumps([int(base_comp)]),
                    "right_component_ids": json.dumps([int(comp_id)]),
                    "left_area": int(np.sum(labels_cc == int(base_comp))),
                    "right_area": int(np.sum(labels_cc == int(comp_id))),
                    "minimum_boundary_distance": None,
                    "chosen_target_group": int(gt_id),
                    "same_gt_leaflet": 1,
                    "merge_stage": "gt_oracle_grouping",
                }
            )
    return {
        "labels": out,
        "exact_k_achieved": bool(int(len(_positive_ids(out))) == int(k)),
        "final_group_count": int(len(_positive_ids(out))),
        "initial_component_count": int(cc_k),
        "merge_count": int(len(merge_ops)),
        "split_count": 0,
        "fragments_assigned": int(max(int(cc_k) - int(len(_positive_ids(out))), 0)),
        "reason": "exact_k" if int(len(_positive_ids(out))) == int(k) else "missing_gt_overlap_for_some_leaflets",
        "merge_operations": merge_ops,
        "seed_count": 0,
        "watershed_label_count": int(len(_positive_ids(out))),
    }


def _failure_attribution(
    *,
    mask_condition: str,
    method_key: str,
    gt_inst_u8: np.ndarray,
    sem_mask01: np.ndarray,
    normalized_labels: np.ndarray,
    metrics: dict[str, Any],
    merge_operations: list[dict[str, Any]],
    initial_component_count: int,
    gt_k: int,
) -> str:
    if int(np.sum(sem_mask01)) <= 16:
        return "empty_or_near_empty_semantic_region"
    topology = forensic.classify_semantic_topology(gt_inst_u8, sem_mask01)
    pred_union = (sem_mask01 > 0).astype(np.uint8)
    gt_union = (gt_inst_u8 > 0).astype(np.uint8)
    recall = float(np.sum((pred_union > 0) & (gt_union > 0))) / max(float(np.sum(gt_union > 0)), 1.0)
    if float(metrics["all_iou_ge_0.50"]) >= 1.0:
        return "success"
    if any(op.get("same_gt_leaflet") == 0 for op in merge_operations):
        return "wrong_fragment_grouping"
    if recall < 0.80:
        return "insufficient_semantic_foreground"
    if int(initial_component_count) < int(gt_k):
        return "false_semantic_bridge_requiring_split"
    if int(initial_component_count) > int(gt_k) and mask_condition == "PREDICTED_SEMANTIC" and topology["missing"]:
        return "insufficient_semantic_foreground"
    if int(initial_component_count) < int(gt_k):
        return "incorrect_watershed_split"
    if float(metrics["instance_mean_matched_iou"]) < 0.40:
        return "ambiguous_geometry"
    return "other"


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "exact_k_rate": 0.0,
            "mean_matched_iou": 0.0,
            "all_iou_ge_0.50": 0.0,
            "all_iou_ge_0.70": 0.0,
            "all_iou_ge_0.80": 0.0,
            "merge_rate": 0.0,
            "fragmentation_rate": 0.0,
            "mean_merge_ops": 0.0,
            "mean_split_ops": 0.0,
        }
    return {
        "n": int(len(rows)),
        "exact_k_rate": _mean([float(r["instance_exact_count_acc"]) for r in rows]),
        "mean_matched_iou": _mean([float(r["instance_mean_matched_iou"]) for r in rows]),
        "all_iou_ge_0.50": _mean([float(r["all_iou_ge_0.50"]) for r in rows]),
        "all_iou_ge_0.70": _mean([float(r["all_iou_ge_0.70"]) for r in rows]),
        "all_iou_ge_0.80": _mean([float(r["all_iou_ge_0.80"]) for r in rows]),
        "merge_rate": _mean([float(r["instance_merged_rate"]) for r in rows]),
        "fragmentation_rate": _mean([float(r["instance_fragmented_rate"]) for r in rows]),
        "mean_merge_ops": _mean([float(r["merge_operations_count"]) for r in rows]),
        "mean_split_ops": _mean([float(r["split_operations_count"]) for r in rows]),
    }


def _make_visual(
    rgb: np.ndarray,
    sem_mask01: np.ndarray,
    original_cc_labels: np.ndarray,
    normalized_labels: np.ndarray,
    gt_inst_u8: np.ndarray,
    merge_ops_text: str,
) -> np.ndarray:
    op_img = np.full((rgb.shape[0], rgb.shape[1], 3), 24, dtype=np.uint8)
    lines = merge_ops_text.split("\n")[:10]
    y = 28
    for line in lines:
        cv2.putText(op_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        y += 28
    row1 = np.concatenate(
        [
            base_audit._panel_with_title(rgb, "RGB"),
            base_audit._panel_with_title(base_audit._binary_rgb(sem_mask01), "Semantic Mask"),
            base_audit._panel_with_title(base_audit._instance_rgb(original_cc_labels.astype(np.uint8)), "Original CC Labels"),
        ],
        axis=1,
    )
    row2 = np.concatenate(
        [
            base_audit._panel_with_title(base_audit._instance_rgb(normalized_labels.astype(np.uint8)), "K-Normalized Labels"),
            base_audit._panel_with_title(base_audit._instance_rgb(gt_inst_u8.astype(np.uint8)), "GT Instances"),
            base_audit._panel_with_title(op_img, "Merge/Split Ops"),
        ],
        axis=1,
    )
    return np.concatenate([row1, row2], axis=0)


def _load_previous_method_rows() -> list[dict[str, Any]]:
    original_method_rows = _read_csv(ORIGINAL_AUDIT_DIR / "method_comparison.csv")
    forensic_summary = json.loads((FORENSIC_AUDIT_DIR / "forensic_summary.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row in original_method_rows:
        if str(row["method_key"]) in {"baseline_connected_components", "global_distance_maxima_r09"}:
            rows.append(
                {
                    "mask_condition": str(row["mask_condition"]),
                    "method_key": str(row["method_key"]),
                    "method_family": "previous_geometry",
                    "exact_k_rate": float(row["exact_instance_count"]),
                    "mean_matched_iou": float(row["mean_matched_iou"]),
                    "all_iou_ge_0.50": float(row["all_iou_ge_0.50"]),
                    "all_iou_ge_0.70": float(row["all_iou_ge_0.70"]),
                    "all_iou_ge_0.80": float(row["all_iou_ge_0.80"]),
                }
            )
    best_neck = forensic_summary["postprocessing"]["best_variant"]
    rows.append(
        {
            "mask_condition": "PREDICTED_SEMANTIC",
            "method_key": "previous_neck_cut_w3",
            "method_family": "previous_geometry",
            "exact_k_rate": float(best_neck["exact_count"]),
            "mean_matched_iou": float(best_neck["mean_matched_iou"]),
            "all_iou_ge_0.50": float(best_neck["all_iou_ge_0.50"]),
            "all_iou_ge_0.70": None,
            "all_iou_ge_0.80": None,
        }
    )
    rows.append(
        {
            "mask_condition": "CENTER_REFERENCE",
            "method_key": "existing_center_reference",
            "method_family": "center_reference",
            "exact_k_rate": float(CENTER_REFERENCE["instance_exact_count"]),
            "mean_matched_iou": float(CENTER_REFERENCE["instance_mean_matched_iou"]),
            "all_iou_ge_0.50": None,
            "all_iou_ge_0.70": None,
            "all_iou_ge_0.80": None,
        }
    )
    return rows


def _best_method_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["all_iou_ge_0.50"]),
            -float(row["mean_matched_iou"]),
            -float(row["exact_k_rate"]),
            str(row["method_key"]),
        ),
    )
    return ranked[0]


def _future_count_head_candidate() -> dict[str, Any]:
    in_dim = 384
    hidden = 64
    out_dim = 3
    params = in_dim * hidden + hidden + hidden * out_dim + out_dim
    return {
        "feature_path": "semantic_model.encoder(x)[-1]",
        "tensor_shape": [1, 384, 24, 24],
        "estimated_params": int(params),
        "labels": "manifest.gt_instance_count classes {1,2,3}",
    }


def _summarize_failure_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["failure_attribution"])
        counts[key] = int(counts.get(key, 0) + 1)
    return {key: counts[key] for key in sorted(counts)}


def _choose_decision(
    *,
    best_pred: dict[str, Any],
    best_pred_gt2: dict[str, Any],
    oracle_pred: dict[str, Any],
    previous_current: dict[str, Any],
    previous_neck: dict[str, Any],
) -> tuple[str, str]:
    best_exact = float(best_pred["exact_k_rate"])
    best_all50 = float(best_pred["all_iou_ge_0.50"])
    best_mean_iou = float(best_pred["mean_matched_iou"])
    best_gt2_all50 = float(best_pred_gt2["all_iou_ge_0.50"])
    current_all50 = float(previous_current["all_iou_ge_0.50"])
    neck_all50 = float(previous_neck["all_iou_ge_0.50"])
    oracle_all50 = float(oracle_pred["all_iou_ge_0.50"])
    delta_from_current = best_all50 - current_all50
    delta_from_neck = best_all50 - neck_all50
    oracle_gap = oracle_all50 - best_all50

    if best_exact >= 0.90 and best_all50 >= 0.70 and best_gt2_all50 >= 0.70:
        return (
            "A. BUILD_COUNT_CLASSIFIER",
            "Predicted semantic masks become geometry-compatible once K-constrained normalization is enforced.",
        )
    if delta_from_current >= 0.08 and delta_from_neck >= 0.04 and oracle_gap >= 0.05:
        return (
            "B. K_NORMALIZATION_PROMISING_BUT_SEMANTIC_LIMITED",
            "Exact-K grouping materially improved reconstruction, but the remaining gap to oracle grouping still points to semantic pixel coverage as the main ceiling.",
        )
    if oracle_all50 < 0.60:
        return (
            "C. IMPROVE_SEMANTIC_TOPOLOGY",
            "Deterministic K-normalization fixed the count contract but only modestly improved reconstruction, and even oracle fragment grouping stays low enough that semantic pixel errors remain dominant.",
        )
    if oracle_gap >= 0.10:
        return (
            "D. BUILD_BOUNDARY_OR_KEYPOINT_HEAD",
            "Even after fragment grouping, a large residual gap to oracle grouping indicates separation-specific supervision is still the main missing ingredient.",
        )
    if best_mean_iou + 0.02 < float(CENTER_REFERENCE["instance_mean_matched_iou"]) and best_all50 < neck_all50:
        return (
            "E. KEEP_CENTER_APPROACH",
            "Even under fair K-constrained normalization, the geometric route remains clearly inferior.",
        )
    return (
        "C. IMPROVE_SEMANTIC_TOPOLOGY",
        "K-normalization helps, but the semantic mask remains the limiting factor rather than fragment-grouping geometry.",
    )


def run_audit(
    *,
    output_dir: Path,
    manifest_path: Path,
    semantic_config_path: Path,
    semantic_checkpoint_path: Path,
    instance_root: Path,
    limit: int | None,
) -> dict[str, Any]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visual_review"
    visual_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = base_audit._read_jsonl(manifest_path)
    if limit is not None:
        manifest_rows = manifest_rows[: int(limit)]
    if any(bool(row.get("present_in_authoritative_106_holdout", False)) for row in manifest_rows):
        raise SystemExit("Manifest unexpectedly references authoritative holdout samples")

    semantic_cfg = base_audit._read_yaml(semantic_config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool((semantic_cfg.get("train") or {}).get("amp", False)) and device.type == "cuda"
    model = base_audit._build_semantic_model(semantic_cfg).to(device)
    base_audit._load_semantic_checkpoint(model, semantic_checkpoint_path, device)

    per_sample_rows: list[dict[str, Any]] = []
    merge_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    gt_count_rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []
    sample_debug: dict[tuple[str, str, str], dict[str, Any]] = {}

    previous_gt_over_k_rows = _read_csv(FORENSIC_AUDIT_DIR / "exact_count_audit.csv")
    previous_38_samples = {
        str(row["sample_id"])
        for row in previous_gt_over_k_rows
        if str(row["count_relation"]) == "gt"
    }

    deployable_keys = {spec.key for spec in NORMALIZER_SPECS if spec.family == "k_normalizer"}

    for idx, row in enumerate(manifest_rows, start=1):
        sample_id = str(row["sample"])
        patient_id = str(row["patient_id"])
        target_hw = (int(row["image_height"]), int(row["image_width"]))
        rgb = base_audit._load_rgb(base_audit._resolve_path(instance_root, str(row["image_rel"])))
        rgb = base_audit._center_crop_like_validation(rgb, target_hw[0], target_hw[1], is_mask=False)
        gt_inst_full = base_audit._load_u8(base_audit._resolve_path(instance_root, str(row["instance_mask_rel"])))
        gt_inst = base_audit._center_crop_like_validation(gt_inst_full, target_hw[0], target_hw[1], is_mask=True)
        gt_union = base_audit._leaflet_union_from_instance_mask(gt_inst)
        gt_k = int(row["gt_instance_count"])
        pred_sem = base_audit._predict_semantic_mask(model, rgb, target_hw=target_hw, device=device, use_amp=use_amp)
        pred_union = base_audit._semantic_union_from_prediction(pred_sem)
        mask_conditions = {
            "GT_SEMANTIC": gt_union,
            "PREDICTED_SEMANTIC": pred_union,
        }
        for condition_key, sem_mask01 in mask_conditions.items():
            original_cc_labels, original_cc_count = _cc_labels(sem_mask01.astype(np.uint8))
            for spec in NORMALIZER_SPECS:
                if spec.key == "gt_fragment_grouping_oracle":
                    normalized = gt_fragment_grouping_oracle(sem_mask01, gt_inst, gt_k)
                else:
                    normalized = normalize_mask_exact_k(sem_mask01, gt_k, spec.key)
                labels = normalized["labels"].astype(np.uint8)
                pred_k = int(len(_positive_ids(labels)))
                metrics = base_audit.compute_detailed_instance_metrics(gt_inst, labels, gt_k=gt_k, pred_k=pred_k)
                failure = _failure_attribution(
                    mask_condition=condition_key,
                    method_key=spec.key,
                    gt_inst_u8=gt_inst,
                    sem_mask01=sem_mask01,
                    normalized_labels=labels,
                    metrics=metrics,
                    merge_operations=normalized["merge_operations"],
                    initial_component_count=int(original_cc_count),
                    gt_k=gt_k,
                )
                per_row = {
                    "sample_id": sample_id,
                    "patient_id": patient_id,
                    "gt_count": int(gt_k),
                    "mask_condition": condition_key,
                    "method_key": spec.key,
                    "method_family": spec.family,
                    "initial_component_count": int(original_cc_count),
                    "final_group_count": int(normalized["final_group_count"]),
                    "merge_operations_count": int(normalized["merge_count"]),
                    "split_operations_count": int(normalized["split_count"]),
                    "fragments_assigned_to_another_group": int(normalized["fragments_assigned"]),
                    "exact_k_achieved": int(bool(normalized["exact_k_achieved"])),
                    "instance_exact_count_acc": float(metrics["instance_exact_count_acc"]),
                    "instance_mean_matched_iou": float(metrics["instance_mean_matched_iou"]),
                    "all_iou_ge_0.50": float(metrics["all_iou_ge_0.50"]),
                    "all_iou_ge_0.70": float(metrics["all_iou_ge_0.70"]),
                    "all_iou_ge_0.80": float(metrics["all_iou_ge_0.80"]),
                    "instance_merged_rate": float(metrics["instance_merged_rate"]),
                    "instance_fragmented_rate": float(metrics["instance_fragmented_rate"]),
                    "reason": str(normalized["reason"]),
                    "failure_attribution": failure,
                    "is_previous_gt_over_k_case": int(sample_id in previous_38_samples),
                }
                per_sample_rows.append(per_row)
                failure_rows.append(
                    {
                        "sample_id": sample_id,
                        "patient_id": patient_id,
                        "gt_count": int(gt_k),
                        "mask_condition": condition_key,
                        "method_key": spec.key,
                        "failure_attribution": failure,
                        "reason": str(normalized["reason"]),
                    }
                )
                for op_idx, op in enumerate(normalized["merge_operations"], start=1):
                    merge_rows.append(
                        {
                            "sample_id": sample_id,
                            "patient_id": patient_id,
                            "gt_count": int(gt_k),
                            "mask_condition": condition_key,
                            "method_key": spec.key,
                            "operation_index": int(op_idx),
                            **op,
                        }
                    )
                sample_debug[(sample_id, condition_key, spec.key)] = {
                    "rgb": rgb,
                    "sem_mask01": sem_mask01,
                    "original_cc_labels": original_cc_labels,
                    "normalized_labels": labels,
                    "gt_inst": gt_inst,
                    "merge_ops": normalized["merge_operations"],
                    "row": per_row,
                }
        print(f"[{idx}/{len(manifest_rows)}] audited {sample_id}")

    current_rows = [row for row in per_sample_rows if row["method_key"] in deployable_keys]
    for condition in ("GT_SEMANTIC", "PREDICTED_SEMANTIC"):
        for spec in NORMALIZER_SPECS:
            cond_rows = [row for row in per_sample_rows if row["mask_condition"] == condition and row["method_key"] == spec.key]
            if not cond_rows:
                continue
            agg = _aggregate(cond_rows)
            method_rows.append(
                {
                    "mask_condition": condition,
                    "method_key": spec.key,
                    "method_family": spec.family,
                    **agg,
                }
            )
            for gt_count in (1, 2, 3):
                rows_gt = [row for row in cond_rows if int(row["gt_count"]) == int(gt_count)]
                agg_gt = _aggregate(rows_gt)
                gt_count_rows.append(
                    {
                        "mask_condition": condition,
                        "method_key": spec.key,
                        "gt_count": int(gt_count),
                        **agg_gt,
                    }
                )
            for patient_id in sorted({str(row["patient_id"]) for row in cond_rows}):
                rows_pt = [row for row in cond_rows if str(row["patient_id"]) == str(patient_id)]
                agg_pt = _aggregate(rows_pt)
                patient_rows.append(
                    {
                        "mask_condition": condition,
                        "method_key": spec.key,
                        "patient_id": patient_id,
                        **agg_pt,
                    }
                )

    previous_rows = _load_previous_method_rows()
    comparison_rows = []
    for row in method_rows:
        comparison_rows.append(dict(row))
    comparison_rows.extend(previous_rows)

    pred_deployable = [row for row in method_rows if row["mask_condition"] == "PREDICTED_SEMANTIC" and row["method_key"] in deployable_keys]
    best_pred = _best_method_row(pred_deployable)
    best_gt = _best_method_row([row for row in method_rows if row["mask_condition"] == "GT_SEMANTIC" and row["method_key"] in deployable_keys])
    best_pred_gt2 = next(row for row in gt_count_rows if row["mask_condition"] == "PREDICTED_SEMANTIC" and row["method_key"] == best_pred["method_key"] and int(row["gt_count"]) == 2)
    best_pred_gt3 = next(row for row in gt_count_rows if row["mask_condition"] == "PREDICTED_SEMANTIC" and row["method_key"] == best_pred["method_key"] and int(row["gt_count"]) == 3)
    oracle_gt = next(row for row in method_rows if row["mask_condition"] == "GT_SEMANTIC" and row["method_key"] == "gt_fragment_grouping_oracle")
    oracle_pred = next(row for row in method_rows if row["mask_condition"] == "PREDICTED_SEMANTIC" and row["method_key"] == "gt_fragment_grouping_oracle")
    previous_current = next(row for row in previous_rows if row["mask_condition"] == "PREDICTED_SEMANTIC" and row["method_key"] == "global_distance_maxima_r09")
    previous_neck = next(row for row in previous_rows if row["mask_condition"] == "PREDICTED_SEMANTIC" and row["method_key"] == "previous_neck_cut_w3")
    decision, reason = _choose_decision(
        best_pred=best_pred,
        best_pred_gt2=best_pred_gt2,
        oracle_pred=oracle_pred,
        previous_current=previous_current,
        previous_neck=previous_neck,
    )

    previous_38_best_rows = [
        row
        for row in per_sample_rows
        if row["sample_id"] in previous_38_samples and row["mask_condition"] == "GT_SEMANTIC" and row["method_key"] == best_gt["method_key"]
    ]
    previous_38_recovered_exact = int(sum(1 for row in previous_38_best_rows if int(row["exact_k_achieved"]) == 1))
    previous_38_recovered_iou50 = int(sum(1 for row in previous_38_best_rows if float(row["all_iou_ge_0.50"]) >= 1.0))
    previous_38_incorrect_grouping = int(sum(1 for row in previous_38_best_rows if str(row["failure_attribution"]) == "wrong_fragment_grouping"))

    oracle_fragment_grouping_summary = {
        "gt_semantic": oracle_gt,
        "predicted_semantic": oracle_pred,
        "interpretation": (
            "The oracle grouping upper bound isolates how much reconstruction is recoverable by regrouping existing foreground fragments without changing semantic pixels."
        ),
    }
    (output_dir / "oracle_fragment_grouping_summary.json").write_text(
        json.dumps(oracle_fragment_grouping_summary, indent=2, default=base_audit._json_default),
        encoding="utf-8",
    )

    visual_manifest: list[dict[str, Any]] = []

    def _pick_visual(category: str, candidates: list[tuple[str, str, str]]) -> None:
        if not candidates:
            return
        sample_id, condition, method_key = candidates[0]
        dbg = sample_debug[(sample_id, condition, method_key)]
        ops_txt = []
        if dbg["merge_ops"]:
            for op in dbg["merge_ops"][:5]:
                ops_txt.append(
                    f"{op['merge_stage']}: {op['left_component_ids']} + {op['right_component_ids']} d={op['minimum_boundary_distance']}"
                )
        else:
            ops_txt.append("no merge operations")
        ops_txt.append(f"method={method_key}")
        ops_txt.append(f"failure={dbg['row']['failure_attribution']}")
        grid = _make_visual(
            dbg["rgb"],
            dbg["sem_mask01"],
            dbg["original_cc_labels"],
            dbg["normalized_labels"],
            dbg["gt_inst"],
            "\n".join(ops_txt),
        )
        out_path = visual_dir / f"{category}_{sample_id}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        visual_manifest.append({"category": category, "sample_id": sample_id, "mask_condition": condition, "method_key": method_key, "file": str(out_path.resolve())})

    gt_merge_cases = sorted(
        [
            (row["sample_id"], row["mask_condition"], row["method_key"])
            for row in previous_38_best_rows
            if int(row["exact_k_achieved"]) == 1
        ],
        key=lambda item: item[0],
    )
    pred_regroup_cases = sorted(
        [
            (row["sample_id"], row["mask_condition"], row["method_key"])
            for row in per_sample_rows
            if row["mask_condition"] == "PREDICTED_SEMANTIC"
            and row["method_key"] == best_pred["method_key"]
            and int(row["merge_operations_count"]) >= 1
            and float(row["all_iou_ge_0.50"]) >= 1.0
        ],
        key=lambda item: (-float(next(r for r in per_sample_rows if r["sample_id"] == item[0] and r["mask_condition"] == item[1] and r["method_key"] == item[2])["instance_mean_matched_iou"]), item[0]),
    )
    gt2_recovered = sorted(
        [
            (row["sample_id"], row["mask_condition"], row["method_key"])
            for row in per_sample_rows
            if row["mask_condition"] == "PREDICTED_SEMANTIC" and row["method_key"] == best_pred["method_key"] and int(row["gt_count"]) == 2 and float(row["all_iou_ge_0.50"]) >= 1.0
        ],
        key=lambda item: item[0],
    )
    gt3_recovered = sorted(
        [
            (row["sample_id"], row["mask_condition"], row["method_key"])
            for row in per_sample_rows
            if row["mask_condition"] == "PREDICTED_SEMANTIC" and row["method_key"] == best_pred["method_key"] and int(row["gt_count"]) == 3 and float(row["all_iou_ge_0.50"]) >= 1.0
        ],
        key=lambda item: item[0],
    )
    wrong_merge_cases = sorted(
        [
            (row["sample_id"], row["mask_condition"], row["method_key"])
            for row in per_sample_rows
            if row["mask_condition"] == "PREDICTED_SEMANTIC" and row["method_key"] == best_pred["method_key"] and str(row["failure_attribution"]) == "wrong_fragment_grouping"
        ],
        key=lambda item: item[0],
    )
    bridge_split_cases = sorted(
        [
            (row["sample_id"], row["mask_condition"], row["method_key"])
            for row in per_sample_rows
            if row["mask_condition"] == "PREDICTED_SEMANTIC" and row["method_key"] == best_pred["method_key"] and str(row["failure_attribution"]) == "false_semantic_bridge_requiring_split"
        ],
        key=lambda item: item[0],
    )
    missing_pixel_cases = sorted(
        [
            (row["sample_id"], row["mask_condition"], row["method_key"])
            for row in per_sample_rows
            if row["mask_condition"] == "PREDICTED_SEMANTIC" and row["method_key"] == best_pred["method_key"] and str(row["failure_attribution"]) == "insufficient_semantic_foreground"
        ],
        key=lambda item: item[0],
    )
    oracle_better_cases = sorted(
        [
            (row["sample_id"], row["mask_condition"], row["method_key"])
            for row in per_sample_rows
            if row["mask_condition"] == "PREDICTED_SEMANTIC"
            and row["method_key"] == "gt_fragment_grouping_oracle"
            and float(row["all_iou_ge_0.50"]) > float(next(rr for rr in per_sample_rows if rr["sample_id"] == row["sample_id"] and rr["mask_condition"] == "PREDICTED_SEMANTIC" and rr["method_key"] == best_pred["method_key"])["all_iou_ge_0.50"])
        ],
        key=lambda item: item[0],
    )

    _pick_visual("gt_semantic_n_gt_k_correctly_merged", gt_merge_cases)
    _pick_visual("predicted_fragment_correctly_regrouped", pred_regroup_cases)
    _pick_visual("gt2_recovered", gt2_recovered)
    _pick_visual("gt3_recovered", gt3_recovered)
    _pick_visual("wrong_fragment_merge", wrong_merge_cases)
    _pick_visual("bridge_requiring_watershed_split", bridge_split_cases)
    _pick_visual("semantic_pixels_missing_grouping_cannot_recover", missing_pixel_cases)
    _pick_visual("oracle_grouping_succeeds_deterministic_fails", oracle_better_cases)

    _write_csv(
        output_dir / "per_sample_results.csv",
        per_sample_rows,
        fieldnames=list(per_sample_rows[0].keys()),
    )
    _write_csv(
        output_dir / "merge_operations.csv",
        merge_rows,
        fieldnames=list(merge_rows[0].keys()) if merge_rows else [
            "sample_id",
            "patient_id",
            "gt_count",
            "mask_condition",
            "method_key",
            "operation_index",
            "left_group",
            "right_group",
            "left_component_ids",
            "right_component_ids",
            "left_area",
            "right_area",
            "minimum_boundary_distance",
            "chosen_target_group",
            "same_gt_leaflet",
            "merge_stage",
        ],
    )
    _write_csv(
        output_dir / "gt_count_comparison.csv",
        gt_count_rows,
        fieldnames=list(gt_count_rows[0].keys()),
    )
    _write_csv(
        output_dir / "patient_comparison.csv",
        patient_rows,
        fieldnames=list(patient_rows[0].keys()),
    )
    _write_csv(
        output_dir / "failure_attribution.csv",
        failure_rows,
        fieldnames=list(failure_rows[0].keys()),
    )
    _write_csv(
        output_dir / "method_comparison.csv",
        comparison_rows,
        fieldnames=sorted({key for row in comparison_rows for key in row.keys()}),
    )

    summary = {
        "exact_k_contract": {
            "samples": int(len(manifest_rows)),
            "best_gt_exact_k": float(best_gt["exact_k_rate"]),
            "best_pred_exact_k": float(best_pred["exact_k_rate"]),
        },
        "best_gt_semantic_method": best_gt,
        "best_predicted_semantic_method": best_pred,
        "previous_38_gt_over_k_cases": {
            "recovered_exact_k": int(previous_38_recovered_exact),
            "recovered_all_iou_ge_0.50": int(previous_38_recovered_iou50),
            "incorrectly_grouped": int(previous_38_incorrect_grouping),
        },
        "oracle_fragment_grouping": oracle_fragment_grouping_summary,
        "vs_previous_geometry": {
            "current": previous_current,
            "neck_cut_w3": previous_neck,
            "k_normalized": best_pred,
            "delta_all_iou_ge_0.50_vs_current": float(best_pred["all_iou_ge_0.50"]) - float(previous_current["all_iou_ge_0.50"]),
            "delta_all_iou_ge_0.50_vs_neck_cut": float(best_pred["all_iou_ge_0.50"]) - float(previous_neck["all_iou_ge_0.50"]),
        },
        "vs_center": {
            "geometry": best_pred,
            "center": CENTER_REFERENCE,
            "interpretation": "Compare literal exact-K count and mean matched IoU directly; center all-instance IoU thresholds are not available from the saved local artifacts.",
        },
        "best_predicted_failure_attribution": _summarize_failure_counts(
            [row for row in per_sample_rows if row["mask_condition"] == "PREDICTED_SEMANTIC" and row["method_key"] == best_pred["method_key"]]
        ),
        "decision": {"result": decision, "reason": reason},
        "future_count_head_candidate": _future_count_head_candidate() if decision in {"A. BUILD_COUNT_CLASSIFIER", "B. K_NORMALIZATION_PROMISING_BUT_SEMANTIC_LIMITED"} else None,
        "visual_review": visual_manifest,
        "deployable_gt_signature": str(inspect.signature(normalize_mask_exact_k)),
        "oracle_signature": str(inspect.signature(gt_fragment_grouping_oracle)),
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=base_audit._json_default),
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--manifest", type=Path, default=base_audit.DEFAULT_MANIFEST_PATH)
    ap.add_argument("--semantic-config", type=Path, default=base_audit.DEFAULT_SEMANTIC_CONFIG)
    ap.add_argument("--semantic-checkpoint", type=Path, default=base_audit.DEFAULT_SEMANTIC_CHECKPOINT)
    ap.add_argument("--instance-root", type=Path, default=base_audit.DEFAULT_INSTANCE_ROOT)
    ap.add_argument("--limit", type=int, default=None)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run_audit(
        output_dir=args.output_dir.resolve(),
        manifest_path=args.manifest.resolve(),
        semantic_config_path=args.semantic_config.resolve(),
        semantic_checkpoint_path=args.semantic_checkpoint.resolve(),
        instance_root=args.instance_root.resolve(),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
