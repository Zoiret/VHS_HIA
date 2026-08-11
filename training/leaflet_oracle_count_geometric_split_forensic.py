from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

import leaflet_oracle_count_geometric_split_audit as base_audit
from validate_centerhead import compute_instance_metrics_from_masks


REPO_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_AUDIT_DIR = REPO_ROOT / "training" / "analysis" / "leaflet_oracle_count_geometric_split_audit"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "training" / "analysis" / "leaflet_oracle_count_geometric_split_forensic"
GT_FORENSIC_METHOD_KEY = "global_distance_maxima_r09"
PRED_FORENSIC_METHOD_KEY = "global_distance_maxima_r09"


@dataclass(frozen=True)
class VariantSpec:
    key: str
    family: str
    params: dict[str, Any]


POSTPROCESS_VARIANTS: list[VariantSpec] = [
    VariantSpec("current", "identity", {}),
    VariantSpec("opening_k3", "opening", {"kernel_size": 3}),
    VariantSpec("opening_k5", "opening", {"kernel_size": 5}),
    VariantSpec("erode_restore_r1", "erode_restore", {"radius": 1}),
    VariantSpec("erode_restore_r2", "erode_restore", {"radius": 2}),
    VariantSpec("neck_cut_w2", "neck_cut", {"width_thr": 2.0, "min_lobe_area": 1200}),
    VariantSpec("neck_cut_w3", "neck_cut", {"width_thr": 3.0, "min_lobe_area": 1200}),
]


def _kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(size), int(size)))


def _cc_labels(mask01: np.ndarray) -> tuple[np.ndarray, int]:
    return base_audit._connected_components(mask01.astype(np.uint8))


def _positive_ids(labels_u8: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(labels_u8) if int(v) > 0]


def _union_iou(a01: np.ndarray, b01: np.ndarray) -> float:
    a = a01.astype(bool)
    b = b01.astype(bool)
    inter = float(np.sum(a & b))
    union = float(np.sum(a | b))
    return float(inter / union) if union > 0 else 0.0


def _union_dice(a01: np.ndarray, b01: np.ndarray) -> float:
    a = a01.astype(bool)
    b = b01.astype(bool)
    inter = float(np.sum(a & b))
    denom = float(np.sum(a) + np.sum(b))
    return float((2.0 * inter) / denom) if denom > 0 else 0.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _load_original_summary(audit_dir: Path) -> dict[str, Any]:
    return json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))


def _load_original_csv(audit_dir: Path, name: str) -> list[dict[str, Any]]:
    with (audit_dir / name).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def written_contract_classification(*, gt_summary: dict[str, Any], pred_summary: dict[str, Any], pred_gt2: dict[str, Any]) -> str:
    if (
        float(pred_summary["exact_instance_count"]) >= 0.85
        and float(pred_summary["all_iou_ge_0.50"]) >= 0.75
        and float(pred_gt2["all_iou_ge_0.50"]) >= 0.70
    ):
        return "STRONG_GEOMETRIC_SIGNAL"
    if (
        float(pred_summary["exact_instance_count"]) >= 0.70
        and float(pred_summary["all_iou_ge_0.50"]) >= 0.60
    ):
        return "PROMISING_GEOMETRIC_SIGNAL"
    if (
        float(gt_summary["all_iou_ge_0.50"]) >= 0.90
        and float(gt_summary["mean_matched_iou"]) >= 0.85
        and float(pred_summary["all_iou_ge_0.50"]) + 0.20 < float(gt_summary["all_iou_ge_0.50"])
    ):
        return "WEAK_GEOMETRIC_SIGNAL"
    return "GEOMETRY_INSUFFICIENT"


def written_contract_decision(classification: str) -> tuple[str, str]:
    if classification in {"STRONG_GEOMETRIC_SIGNAL", "PROMISING_GEOMETRIC_SIGNAL"}:
        return ("A. BUILD_COUNT_CLASSIFIER", "Predicted semantic masks are already geometry-compatible enough without retraining.")
    if classification == "WEAK_GEOMETRIC_SIGNAL":
        return ("B. IMPROVE_SEMANTIC_TOPOLOGY", "GT-semantic geometry is strong while predicted-semantic topology is the bottleneck.")
    return ("C. BUILD_BOUNDARY_OR_KEYPOINT_HEAD", "Even GT-semantic oracle-K geometry would be too weak.")


def _unique_seed_count(seeds: list[tuple[int, int]]) -> int:
    return int(len({(int(y), int(x)) for y, x in seeds}))


def _analyze_seeded_split(mask01: np.ndarray, k: int, spec: base_audit.SeedMethodSpec) -> dict[str, Any]:
    mask01 = (mask01.astype(np.uint8) > 0).astype(np.uint8)
    split = base_audit.split_mask_with_oracle_k(mask01, int(k), spec)
    seed_trace = dict(split.get("seed_trace", {}))
    labels = split["labels"].astype(np.uint8)
    seeds = [(int(y), int(x)) for y, x in split.get("seeds", [])]
    unique_seed_count = _unique_seed_count(seeds)
    component_count_before = int(seed_trace.get("component_count", 0))
    pred_count = int(split["pred_count"])
    reason = "exact_k"
    if component_count_before > int(k):
        reason = "disconnected_component_policy"
    elif len(seeds) < int(k):
        reason = "fewer_than_k_valid_seeds"
    elif unique_seed_count < int(k):
        reason = "seed_collision"
    elif pred_count < int(k):
        reason = "watershed_label_disappears"
    elif pred_count > int(k):
        reason = "disconnected_component_policy" if component_count_before > int(k) else "other"
    return {
        "labels": labels,
        "pred_count": pred_count,
        "seed_requested": int(k),
        "seed_created": int(len(seeds)),
        "seed_unique": unique_seed_count,
        "component_count_before": component_count_before,
        "watershed_label_count": int(labels.max()),
        "mismatch_reason": reason,
        "seed_trace": seed_trace,
        "distance_transform": split["distance_transform"],
        "seeds": seeds,
    }


def _component_overlap_sets(gt_inst_u8: np.ndarray, pred_union01: np.ndarray) -> dict[str, Any]:
    pred_labels, pred_cc = _cc_labels(pred_union01.astype(np.uint8))
    gt_ids = _positive_ids(gt_inst_u8)
    pred_component_gt_ids: dict[int, set[int]] = {}
    pred_extra_components = 0
    for comp_id in range(1, int(pred_cc) + 1):
        overlap = sorted({int(v) for v in np.unique(gt_inst_u8[pred_labels == comp_id]) if int(v) > 0})
        pred_component_gt_ids[int(comp_id)] = set(int(v) for v in overlap)
        if not overlap:
            pred_extra_components += 1
    gt_instance_pred_components: dict[int, set[int]] = {}
    for gt_id in gt_ids:
        overlap = sorted({int(v) for v in np.unique(pred_labels[gt_inst_u8 == gt_id]) if int(v) > 0})
        gt_instance_pred_components[int(gt_id)] = set(int(v) for v in overlap)
    return {
        "pred_labels": pred_labels,
        "pred_cc": int(pred_cc),
        "pred_component_gt_ids": pred_component_gt_ids,
        "gt_instance_pred_components": gt_instance_pred_components,
        "pred_extra_components": int(pred_extra_components),
    }


def classify_semantic_topology(gt_inst_u8: np.ndarray, pred_union01: np.ndarray) -> dict[str, Any]:
    gt_union = (gt_inst_u8 > 0).astype(np.uint8)
    overlaps = _component_overlap_sets(gt_inst_u8, pred_union01)
    gt_ids = _positive_ids(gt_inst_u8)
    bridge = any(len(ids) >= 2 for ids in overlaps["pred_component_gt_ids"].values())
    missing = False
    for gt_id in gt_ids:
        pred_comps = overlaps["gt_instance_pred_components"][int(gt_id)]
        if len(pred_comps) != 1:
            missing = True
            break
    if not missing:
        for gt_id in gt_ids:
            gt_mask = gt_inst_u8 == int(gt_id)
            recall = float(np.sum(gt_mask & pred_union01.astype(bool))) / max(float(np.sum(gt_mask)), 1.0)
            if recall < 0.90:
                missing = True
                break
    pred_extra = int(overlaps["pred_extra_components"])
    if bridge and missing:
        topo_class = "D"
    elif bridge:
        topo_class = "B"
    elif missing:
        topo_class = "C"
    else:
        one_to_one = (
            all(len(ids) == 1 for ids in overlaps["pred_component_gt_ids"].values() if ids)
            and all(len(ids) == 1 for ids in overlaps["gt_instance_pred_components"].values())
            and pred_extra == 0
        )
        topo_class = "A" if one_to_one else "E"
    return {
        "topology_class": topo_class,
        "bridge": bool(bridge),
        "missing": bool(missing),
        "pred_component_count": int(overlaps["pred_cc"]),
        "pred_extra_components": pred_extra,
        "pred_component_gt_ids": {str(k): sorted(v) for k, v in overlaps["pred_component_gt_ids"].items()},
        "gt_instance_pred_components": {str(k): sorted(v) for k, v in overlaps["gt_instance_pred_components"].items()},
        "leaflet_dice": _union_dice(gt_union, pred_union01),
        "leaflet_iou": _union_iou(gt_union, pred_union01),
    }


def _opening(mask01: np.ndarray, kernel_size: int) -> np.ndarray:
    return cv2.morphologyEx(mask01.astype(np.uint8), cv2.MORPH_OPEN, _kernel(int(kernel_size)))


def _geodesic_restore(eroded: np.ndarray, reference: np.ndarray) -> np.ndarray:
    marker = eroded.astype(np.uint8).copy()
    ref = reference.astype(np.uint8)
    while True:
        grown = cv2.dilate(marker, _kernel(3), iterations=1)
        grown = np.minimum(grown, ref).astype(np.uint8)
        if np.array_equal(grown, marker):
            return marker
        marker = grown


def _erode_restore(mask01: np.ndarray, radius: int) -> np.ndarray:
    size = int(2 * radius + 1)
    eroded = cv2.erode(mask01.astype(np.uint8), _kernel(size), iterations=1)
    return _geodesic_restore(eroded, mask01.astype(np.uint8))


def _neck_cut(mask01: np.ndarray, width_thr: float, min_lobe_area: int) -> np.ndarray:
    mask01 = mask01.astype(np.uint8)
    labels, cc_k = _cc_labels(mask01)
    out = np.zeros_like(mask01, dtype=np.uint8)
    for comp_id in range(1, int(cc_k) + 1):
        comp = (labels == comp_id).astype(np.uint8)
        area = int(np.sum(comp))
        if area < int(min_lobe_area):
            out[comp > 0] = 1
            continue
        dt = base_audit._distance_transform(comp)
        neck = ((comp > 0) & (dt <= float(width_thr))).astype(np.uint8)
        cut = comp.copy()
        cut[neck > 0] = 0
        cut_labels, cut_k = _cc_labels(cut)
        lobe_ids = [lab for lab in range(1, int(cut_k) + 1) if int(np.sum(cut_labels == lab)) >= int(min_lobe_area)]
        if len(lobe_ids) >= 2:
            for lab in lobe_ids:
                out[cut_labels == lab] = 1
        else:
            out[comp > 0] = 1
    return out


def apply_variant(mask01: np.ndarray, spec: VariantSpec) -> np.ndarray:
    mask01 = (mask01.astype(np.uint8) > 0).astype(np.uint8)
    if spec.family == "identity":
        return mask01
    if spec.family == "opening":
        return _opening(mask01, int(spec.params["kernel_size"]))
    if spec.family == "erode_restore":
        return _erode_restore(mask01, int(spec.params["radius"]))
    if spec.family == "neck_cut":
        return _neck_cut(mask01, float(spec.params["width_thr"]), int(spec.params["min_lobe_area"]))
    raise ValueError(f"Unknown variant family: {spec.family}")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _summary_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "n": float(len(rows)),
        "exact_count": _mean([float(r["instance_exact_count_acc"]) for r in rows]),
        "mean_matched_iou": _mean([float(r["instance_mean_matched_iou"]) for r in rows]),
        "all_iou_ge_0.50": _mean([float(r["all_iou_ge_0.50"]) for r in rows]),
        "all_iou_ge_0.70": _mean([float(r["all_iou_ge_0.70"]) for r in rows]),
        "merge_rate": _mean([float(r["instance_merged_rate"]) for r in rows]),
        "fragmentation_rate": _mean([float(r["instance_fragmented_rate"]) for r in rows]),
    }


def _save_visual(path: Path, image_rgb_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb_u8, cv2.COLOR_RGB2BGR))


def _panel(title: str, image_rgb_u8: np.ndarray) -> np.ndarray:
    return base_audit._panel_with_title(image_rgb_u8, title)


def _variant_visual(
    rgb: np.ndarray,
    gt_union01: np.ndarray,
    current_mask01: np.ndarray,
    variant_mask01: np.ndarray,
    current_split: dict[str, Any],
    variant_split: dict[str, Any],
    gt_inst: np.ndarray,
) -> np.ndarray:
    row1 = np.concatenate(
        [
            _panel("RGB", rgb),
            _panel("GT Union", base_audit._binary_rgb(gt_union01)),
            _panel("Current Mask", base_audit._binary_rgb(current_mask01)),
            _panel("Variant Mask", base_audit._binary_rgb(variant_mask01)),
        ],
        axis=1,
    )
    row2 = np.concatenate(
        [
            _panel("Current DT", base_audit._distance_rgb(current_split["distance_transform"])),
            _panel("Current Seeds", base_audit._draw_seeds(current_mask01, current_split["seeds"])),
            _panel("Current Inst", base_audit._instance_rgb(current_split["labels"])),
            _panel("GT Inst", base_audit._instance_rgb(gt_inst)),
        ],
        axis=1,
    )
    row3 = np.concatenate(
        [
            _panel("Variant DT", base_audit._distance_rgb(variant_split["distance_transform"])),
            _panel("Variant Seeds", base_audit._draw_seeds(variant_mask01, variant_split["seeds"])),
            _panel("Variant Inst", base_audit._instance_rgb(variant_split["labels"])),
            _panel("Variant Overlay", base_audit._overlay_comparison(rgb, variant_split["labels"], gt_inst)),
        ],
        axis=1,
    )
    return np.concatenate([row1, row2, row3], axis=0)


def run_forensic(
    *,
    output_dir: Path,
    manifest_path: Path,
    semantic_config_path: Path,
    semantic_checkpoint_path: Path,
    instance_root: Path,
    original_audit_dir: Path,
    limit: int | None,
) -> dict[str, Any]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visual_review"
    visual_dir.mkdir(parents=True, exist_ok=True)

    orig_summary = _load_original_summary(original_audit_dir)
    method_rows = _load_original_csv(original_audit_dir, "method_comparison.csv")
    gt_count_rows = _load_original_csv(original_audit_dir, "gt_count_comparison.csv")

    implemented_classification = str(orig_summary["classification"])
    previous_decision = str(orig_summary["next_step"]["decision"])
    implemented_gt = dict(orig_summary["best_gt_semantic_method"])
    implemented_pred = dict(orig_summary["best_predicted_semantic_method"])
    implemented_pred_gt2 = next(
        row for row in gt_count_rows if row["mask_condition"] == "PREDICTED_SEMANTIC" and row["method_key"] == implemented_pred["method_key"] and int(row["gt_count"]) == 2
    )
    mechanical_classification = written_contract_classification(
        gt_summary=implemented_gt,
        pred_summary=implemented_pred,
        pred_gt2=implemented_pred_gt2,
    )
    mechanical_decision, mechanical_reason = written_contract_decision(mechanical_classification)
    decision_logic_audit = {
        "previous_classification": implemented_classification,
        "mechanically_correct_classification": mechanical_classification,
        "previous_next_step": previous_decision,
        "mechanically_correct_next_step": mechanical_decision,
        "implemented_logic": {
            "classification_function": "original _classification requires gt_summary.exact_instance_count >= 0.70 before WEAK_GEOMETRIC_SIGNAL is allowed",
            "next_step_function": "original _next_step_decision prioritizes center comparison before honoring WEAK_GEOMETRIC_SIGNAL",
        },
        "root_cause": [
            "decision_logic_implementation_bug",
            "reporting_bug",
            "metric_interpretation_issue",
        ],
        "why_previous_report_returned_keep_center": [
            "The implemented WEAK_GEOMETRIC_SIGNAL path was blocked by an extra gt exact-count >= 0.70 requirement that was not in the written contract.",
            "The implemented next-step logic short-circuited to KEEP_CENTER whenever geometric exact-count and mean matched IoU were both below the center reference.",
            "The GT-semantic headline row came from the independently selected best GT method (baseline_connected_components), not from the oracle-K watershed method described in prose.",
        ],
    }
    (output_dir / "decision_logic_audit.json").write_text(json.dumps(decision_logic_audit, indent=2), encoding="utf-8")

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

    gt_spec = next(spec for spec in base_audit.SEED_METHOD_SPECS if spec.key == GT_FORENSIC_METHOD_KEY)
    pred_spec = next(spec for spec in base_audit.SEED_METHOD_SPECS if spec.key == PRED_FORENSIC_METHOD_KEY)

    exact_count_rows: list[dict[str, Any]] = []
    topology_rows: list[dict[str, Any]] = []
    pixel_vs_topology_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    sample_cache: dict[str, dict[str, Any]] = {}

    for row in manifest_rows:
        sample_id = str(row["sample"])
        target_hw = (int(row["image_height"]), int(row["image_width"]))
        rgb = base_audit._load_rgb(base_audit._resolve_path(instance_root, str(row["image_rel"])))
        rgb = base_audit._center_crop_like_validation(rgb, target_hw[0], target_hw[1], is_mask=False)
        gt_inst_full = base_audit._load_u8(base_audit._resolve_path(instance_root, str(row["instance_mask_rel"])))
        gt_inst = base_audit._center_crop_like_validation(gt_inst_full, target_hw[0], target_hw[1], is_mask=True)
        gt_union = base_audit._leaflet_union_from_instance_mask(gt_inst)
        gt_k = int(row["gt_instance_count"])
        pred_sem = base_audit._predict_semantic_mask(model, rgb, target_hw=target_hw, device=device, use_amp=use_amp)
        pred_union = base_audit._semantic_union_from_prediction(pred_sem)
        gt_split = _analyze_seeded_split(gt_union, gt_k, gt_spec)
        gt_metrics = base_audit.compute_detailed_instance_metrics(gt_inst, gt_split["labels"], gt_k=gt_k, pred_k=gt_split["pred_count"])
        pred_current_split = _analyze_seeded_split(pred_union, gt_k, pred_spec)
        pred_current_metrics = base_audit.compute_detailed_instance_metrics(
            gt_inst,
            pred_current_split["labels"],
            gt_k=gt_k,
            pred_k=pred_current_split["pred_count"],
        )
        topology = classify_semantic_topology(gt_inst, pred_union)

        exact_count_rows.append(
            {
                "sample_id": sample_id,
                "patient_id": str(row["patient_id"]),
                "gt_count": int(gt_k),
                "reported_gt_metric_method": str(orig_summary["best_gt_semantic_method"]["method_key"]),
                "forensic_gt_split_method": GT_FORENSIC_METHOD_KEY,
                "exact_count_definition": "literal pred_instance_count == gt_instance_count on reconstructed instance labels",
                "requested_k": int(gt_k),
                "component_count_before": int(gt_split["component_count_before"]),
                "seed_requested": int(gt_split["seed_requested"]),
                "seed_created": int(gt_split["seed_created"]),
                "seed_unique": int(gt_split["seed_unique"]),
                "watershed_label_count": int(gt_split["watershed_label_count"]),
                "pred_instance_count": int(gt_split["pred_count"]),
                "count_relation": "eq" if int(gt_split["pred_count"]) == int(gt_k) else ("lt" if int(gt_split["pred_count"]) < int(gt_k) else "gt"),
                "mismatch_reason": str(gt_split["mismatch_reason"]),
                "instance_exact_count_acc": float(gt_metrics["instance_exact_count_acc"]),
            }
        )

        topology_rows.append(
            {
                "sample_id": sample_id,
                "patient_id": str(row["patient_id"]),
                "gt_count": int(gt_k),
                "topology_class": str(topology["topology_class"]),
                "bridge_flag": int(bool(topology["bridge"])),
                "missing_flag": int(bool(topology["missing"])),
                "leaflet_dice": float(topology["leaflet_dice"]),
                "leaflet_iou": float(topology["leaflet_iou"]),
                "pred_component_count": int(topology["pred_component_count"]),
                "pred_extra_components": int(topology["pred_extra_components"]),
                "oracle_k_gt_semantic_success": float(gt_metrics["all_iou_ge_0.50"]),
                "oracle_k_predicted_success": float(pred_current_metrics["all_iou_ge_0.50"]),
            }
        )
        pixel_vs_topology_rows.append(
            {
                "sample_id": sample_id,
                "patient_id": str(row["patient_id"]),
                "gt_count": int(gt_k),
                "topology_class": str(topology["topology_class"]),
                "leaflet_dice": float(topology["leaflet_dice"]),
                "leaflet_iou": float(topology["leaflet_iou"]),
                "gt_semantic_oracle_success": float(gt_metrics["all_iou_ge_0.50"]),
                "predicted_semantic_oracle_success": float(pred_current_metrics["all_iou_ge_0.50"]),
            }
        )

        sample_cache[sample_id] = {
            "rgb": rgb,
            "gt_inst": gt_inst,
            "gt_union": gt_union,
            "pred_union": pred_union,
            "gt_split": gt_split,
            "pred_current_split": pred_current_split,
            "pred_current_metrics": pred_current_metrics,
            "topology": topology,
            "gt_metrics": gt_metrics,
        }

        for variant in POSTPROCESS_VARIANTS:
            variant_mask = apply_variant(pred_union, variant)
            variant_topology = classify_semantic_topology(gt_inst, variant_mask)
            variant_split = _analyze_seeded_split(variant_mask, gt_k, pred_spec)
            variant_metrics = base_audit.compute_detailed_instance_metrics(
                gt_inst,
                variant_split["labels"],
                gt_k=gt_k,
                pred_k=variant_split["pred_count"],
            )
            ablation_rows.append(
                {
                    "sample_id": sample_id,
                    "patient_id": str(row["patient_id"]),
                    "gt_count": int(gt_k),
                    "variant_key": variant.key,
                    "variant_family": variant.family,
                    "topology_class": str(variant_topology["topology_class"]),
                    "leaflet_dice": float(variant_topology["leaflet_dice"]),
                    "leaflet_iou": float(variant_topology["leaflet_iou"]),
                    "pred_instance_count": int(variant_split["pred_count"]),
                    "instance_exact_count_acc": float(variant_metrics["instance_exact_count_acc"]),
                    "instance_mean_matched_iou": float(variant_metrics["instance_mean_matched_iou"]),
                    "all_iou_ge_0.50": float(variant_metrics["all_iou_ge_0.50"]),
                    "all_iou_ge_0.70": float(variant_metrics["all_iou_ge_0.70"]),
                    "instance_merged_rate": float(variant_metrics["instance_merged_rate"]),
                    "instance_fragmented_rate": float(variant_metrics["instance_fragmented_rate"]),
                    "bridge_flag": int(bool(variant_topology["bridge"])),
                    "missing_flag": int(bool(variant_topology["missing"])),
                    "seed_count": int(variant_split["seed_created"]),
                    "watershed_count": int(variant_split["watershed_label_count"]),
                }
            )

            sample_cache[sample_id][f"variant_{variant.key}"] = {
                "mask": variant_mask,
                "topology": variant_topology,
                "split": variant_split,
                "metrics": variant_metrics,
            }

    exact_count_fieldnames = list(exact_count_rows[0].keys()) if exact_count_rows else []
    topology_fieldnames = list(topology_rows[0].keys()) if topology_rows else []
    pixel_fieldnames = list(pixel_vs_topology_rows[0].keys()) if pixel_vs_topology_rows else []
    ablation_fieldnames = list(ablation_rows[0].keys()) if ablation_rows else []
    _write_csv(output_dir / "exact_count_audit.csv", exact_count_rows, exact_count_fieldnames)
    _write_csv(output_dir / "semantic_topology_audit.csv", topology_rows, topology_fieldnames)
    _write_csv(output_dir / "pixel_vs_topology.csv", pixel_vs_topology_rows, pixel_fieldnames)
    _write_csv(output_dir / "postprocessing_ablation.csv", ablation_rows, ablation_fieldnames)

    eq_rows = exact_count_rows
    exact_eq = int(sum(1 for row in eq_rows if row["count_relation"] == "eq"))
    exact_lt = int(sum(1 for row in eq_rows if row["count_relation"] == "lt"))
    exact_gt = int(sum(1 for row in eq_rows if row["count_relation"] == "gt"))
    mismatch_reasons: dict[str, int] = {}
    for row in eq_rows:
        if row["count_relation"] == "eq":
            continue
        mismatch_reasons[str(row["mismatch_reason"])] = mismatch_reasons.get(str(row["mismatch_reason"]), 0) + 1

    topo_counts = {k: 0 for k in ["A", "B", "C", "D", "E"]}
    for row in topology_rows:
        topo_counts[str(row["topology_class"])] += 1

    correct_topo_rows = [row for row in topology_rows if row["topology_class"] == "A"]
    bridge_topo_rows = [row for row in topology_rows if row["topology_class"] in {"B", "D"}]
    missing_topo_rows = [row for row in topology_rows if row["topology_class"] in {"C", "D"}]

    variant_aggregates: list[dict[str, Any]] = []
    for variant in POSTPROCESS_VARIANTS:
        rows = [row for row in ablation_rows if row["variant_key"] == variant.key]
        gt2_rows = [row for row in rows if int(row["gt_count"]) == 2]
        gt3_rows = [row for row in rows if int(row["gt_count"]) == 3]
        current_topo = {str(r["sample_id"]): str(r["topology_class"]) for r in topology_rows}
        variant_aggregates.append(
            {
                "variant_key": variant.key,
                "all_iou_ge_0.50": _mean([float(r["all_iou_ge_0.50"]) for r in rows]),
                "mean_matched_iou": _mean([float(r["instance_mean_matched_iou"]) for r in rows]),
                "gt2_success": _mean([float(r["all_iou_ge_0.50"]) for r in gt2_rows]),
                "gt3_success": _mean([float(r["all_iou_ge_0.50"]) for r in gt3_rows]),
                "false_bridges_remaining": int(sum(1 for r in rows if str(r["topology_class"]) in {"B", "D"})),
                "missing_tissue_failures_introduced": int(
                    sum(
                        1
                        for r in rows
                        if current_topo[str(r["sample_id"])] not in {"C", "D"} and str(r["topology_class"]) in {"C", "D"}
                    )
                ),
                "exact_count": _mean([float(r["instance_exact_count_acc"]) for r in rows]),
            }
        )

    best_variant = sorted(
        variant_aggregates,
        key=lambda row: (-float(row["all_iou_ge_0.50"]), -float(row["mean_matched_iou"]), str(row["variant_key"])),
    )[0]

    center_definition = "literal pred_instance_count == gt_instance_count, where pred_instance_count is computed from reconstructed semantic+center instance labels after marker selection and watershed/connected-component reconstruction"
    geometric_definition = "literal pred_instance_count == gt_instance_count, where pred_instance_count is computed from reconstructed semantic-mask geometry after oracle-K seed selection and watershed/connected-component reconstruction"

    revised_result = "WEAK_GEOMETRIC_SIGNAL" if mechanical_classification == "WEAK_GEOMETRIC_SIGNAL" else mechanical_classification
    if best_variant["all_iou_ge_0.50"] >= 0.70 and best_variant["gt2_success"] >= 0.70 and best_variant["false_bridges_remaining"] <= 15:
        revised_decision = "A. BUILD_COUNT_CLASSIFIER"
        revised_reason = "Cheap deterministic postprocessing made predicted semantic masks sufficiently geometry-compatible."
    else:
        revised_decision = "B. IMPROVE_SEMANTIC_TOPOLOGY"
        revised_reason = "GT-semantic geometry is strong, but predicted-semantic topology remains the bottleneck even after cheap deterministic cleanup."

    visual_manifest: list[dict[str, Any]] = []

    def _pick_sample(candidates: list[str], category: str, variant_key: str | None = None) -> None:
        if not candidates:
            return
        sample_id = candidates[0]
        cache = sample_cache[sample_id]
        current = cache["pred_current_split"]
        if variant_key is None:
            variant_block = cache["variant_current"]
            title_key = "current"
        else:
            variant_block = cache[f"variant_{variant_key}"]
            title_key = variant_key
        grid = _variant_visual(
            cache["rgb"],
            cache["gt_union"],
            cache["pred_union"],
            variant_block["mask"],
            current,
            variant_block["split"],
            cache["gt_inst"],
        )
        out_path = visual_dir / f"{category}_{sample_id}.png"
        _save_visual(out_path, grid)
        visual_manifest.append({"category": category, "sample_id": sample_id, "file": str(out_path.resolve()), "variant": title_key})

    bridge_high_dice = sorted(
        [sid for sid, cache in sample_cache.items() if cache["topology"]["topology_class"] in {"B", "D"}],
        key=lambda sid: -float(sample_cache[sid]["topology"]["leaflet_dice"]),
    )
    high_dice_failure = sorted(
        [
            sid
            for sid, cache in sample_cache.items()
            if float(cache["topology"]["leaflet_dice"]) >= 0.80 and float(cache["pred_current_metrics"]["all_iou_ge_0.50"]) < 1.0
        ],
        key=lambda sid: -float(sample_cache[sid]["topology"]["leaflet_dice"]),
    )
    opening_fix = sorted(
        [
            sid
            for sid, cache in sample_cache.items()
            if cache["topology"]["topology_class"] in {"B", "D"}
            and float(cache["pred_current_metrics"]["all_iou_ge_0.50"]) < 1.0
            and float(cache["variant_opening_k3"]["metrics"]["all_iou_ge_0.50"]) >= 1.0
        ],
        key=lambda sid: -float(sample_cache[sid]["variant_opening_k3"]["metrics"]["instance_mean_matched_iou"]),
    )
    neck_fix = sorted(
        [
            sid
            for sid, cache in sample_cache.items()
            if cache["topology"]["topology_class"] in {"B", "D"}
            and float(cache["pred_current_metrics"]["all_iou_ge_0.50"]) < 1.0
            and float(cache["variant_neck_cut_w2"]["metrics"]["all_iou_ge_0.50"]) >= 1.0
        ],
        key=lambda sid: -float(sample_cache[sid]["variant_neck_cut_w2"]["metrics"]["instance_mean_matched_iou"]),
    )
    damaged_valid = sorted(
        [
            sid
            for sid, cache in sample_cache.items()
            if float(cache["pred_current_metrics"]["all_iou_ge_0.70"]) >= 1.0
            and (
                float(cache["variant_opening_k5"]["metrics"]["all_iou_ge_0.70"]) < 1.0
                or float(cache["variant_opening_k5"]["metrics"]["instance_exact_count_acc"]) < float(cache["pred_current_metrics"]["instance_exact_count_acc"])
            )
        ],
        key=lambda sid: (
            float(sample_cache[sid]["variant_opening_k5"]["metrics"]["all_iou_ge_0.70"]),
            float(sample_cache[sid]["variant_opening_k5"]["metrics"]["instance_mean_matched_iou"]),
        ),
    )
    every_variant_fail = sorted(
        [
            sid
            for sid, cache in sample_cache.items()
            if float(cache["gt_metrics"]["all_iou_ge_0.50"]) >= 1.0
            and all(float(cache[f"variant_{variant.key}"]["metrics"]["all_iou_ge_0.50"]) < 1.0 for variant in POSTPROCESS_VARIANTS)
        ],
        key=lambda sid: -float(sample_cache[sid]["gt_metrics"]["instance_mean_matched_iou"]),
    )

    _pick_sample(bridge_high_dice, "high_dice_false_bridge", "current")
    _pick_sample(high_dice_failure, "high_dice_watershed_failure", "current")
    _pick_sample(opening_fix, "mild_opening_fixes_bridge", "opening_k3")
    _pick_sample(neck_fix, "neck_cut_fixes_bridge", "neck_cut_w2")
    _pick_sample(damaged_valid, "postprocess_damages_valid_leaflet", "opening_k5")
    _pick_sample(every_variant_fail, "gt_semantic_succeeds_all_predicted_variants_fail", "neck_cut_w2")

    forensic_summary = {
        "decision_logic_audit": decision_logic_audit,
        "exact_count_audit": {
            "samples_total": len(eq_rows),
            "predicted_count_eq_k": exact_eq,
            "predicted_count_lt_k": exact_lt,
            "predicted_count_gt_k": exact_gt,
            "mismatch_reasons": mismatch_reasons,
            "reported_gt_metric_method": str(orig_summary["best_gt_semantic_method"]["method_key"]),
            "forensic_gt_split_method": GT_FORENSIC_METHOD_KEY,
        },
        "gt_semantic_validity": {
            "construction": "GT semantic union = (aligned 768x768 GT instance label map > 0), with no predicted semantic inputs",
            "mean_connected_components_before": _mean([float(r["component_count_before"]) for r in exact_count_rows]),
            "mean_connected_components_after": _mean([float(r["pred_instance_count"]) for r in exact_count_rows]),
            "mean_seed_requested": _mean([float(r["seed_requested"]) for r in exact_count_rows]),
            "mean_seed_created": _mean([float(r["seed_created"]) for r in exact_count_rows]),
            "mean_watershed_labels": _mean([float(r["watershed_label_count"]) for r in exact_count_rows]),
            "exact_k_seed_success_rate": _mean([1.0 if int(r["seed_created"]) == int(r["requested_k"]) else 0.0 for r in exact_count_rows]),
            "watershed_label_success_rate": _mean([1.0 if int(r["watershed_label_count"]) == int(r["requested_k"]) else 0.0 for r in exact_count_rows]),
        },
        "predicted_semantic_topology": {
            "correct": topo_counts["A"],
            "false_bridges": topo_counts["B"],
            "missing_tissue": topo_counts["C"],
            "both": topo_counts["D"],
            "other": topo_counts["E"],
        },
        "pixel_vs_topology": {
            "correct_topology_dice": _mean([float(r["leaflet_dice"]) for r in correct_topo_rows]),
            "correct_topology_iou": _mean([float(r["leaflet_iou"]) for r in correct_topo_rows]),
            "bridge_case_dice": _mean([float(r["leaflet_dice"]) for r in bridge_topo_rows]),
            "bridge_case_iou": _mean([float(r["leaflet_iou"]) for r in bridge_topo_rows]),
            "missing_case_dice": _mean([float(r["leaflet_dice"]) for r in missing_topo_rows]),
            "missing_case_iou": _mean([float(r["leaflet_iou"]) for r in missing_topo_rows]),
        },
        "postprocessing": {
            "variant_aggregates": variant_aggregates,
            "best_variant": best_variant,
        },
        "metric_equivalence": {
            "center_exact_count_definition": center_definition,
            "geometric_exact_count_definition": geometric_definition,
            "directly_comparable": True,
            "caveat": "The literal count metric is the same, but prior reporting mixed different mask conditions and different reconstruction methods in the headline comparison.",
        },
        "revised_diagnosis": revised_result,
        "next_step": {"decision": revised_decision, "reason": revised_reason},
        "visual_review": visual_manifest,
    }
    (output_dir / "forensic_summary.json").write_text(
        json.dumps(forensic_summary, indent=2, default=base_audit._json_default),
        encoding="utf-8",
    )
    return forensic_summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--manifest", type=Path, default=base_audit.DEFAULT_MANIFEST_PATH)
    ap.add_argument("--semantic-config", type=Path, default=base_audit.DEFAULT_SEMANTIC_CONFIG)
    ap.add_argument("--semantic-checkpoint", type=Path, default=base_audit.DEFAULT_SEMANTIC_CHECKPOINT)
    ap.add_argument("--instance-root", type=Path, default=base_audit.DEFAULT_INSTANCE_ROOT)
    ap.add_argument("--original-audit-dir", type=Path, default=ORIGINAL_AUDIT_DIR)
    ap.add_argument("--limit", type=int, default=None)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run_forensic(
        output_dir=args.output_dir.resolve(),
        manifest_path=args.manifest.resolve(),
        semantic_config_path=args.semantic_config.resolve(),
        semantic_checkpoint_path=args.semantic_checkpoint.resolve(),
        instance_root=args.instance_root.resolve(),
        original_audit_dir=args.original_audit_dir.resolve(),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
