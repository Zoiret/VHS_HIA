from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from augmentations import get_val_augmentations
from dataset_centerhead import SegmentationWithCenterDataset
from models_centerhead import UnetPlusPlusSemanticCenterHead, load_semantic_checkpoint_non_strict
from validate_centerhead import (
    _case_type,
    _connected_components,
    _extract_metadata_centers,
    _fallback_marker,
    _geometry_topo_u8,
    _keep_top3_by_area,
    _match_centers,
    _markers_from_center_map,
    _watershed,
    compute_instance_metrics_from_masks,
    reconstruct_instances_from_semantic_and_center,
)


def _read_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise SystemExit("pyyaml is not installed. Install training deps with:\n  py -m pip install -r requirements-train.txt") from e
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"Config root must be a dict: {path}")
    return obj


def _simple_preprocess_uint8_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    return (img_rgb_u8.astype(np.float32) / 255.0).astype(np.float32)


def _seed_all(seed: int) -> None:
    s = int(seed)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _make_device(cfg: dict, device_arg: str) -> torch.device:
    if str(device_arg).strip():
        return torch.device(str(device_arg).strip())
    train_dev = str((cfg.get("train") or {}).get("device", "")).strip()
    if train_dev:
        return torch.device(train_dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _center_feature_cfg_from_cfg(cfg: dict) -> dict | None:
    center_feature = (cfg.get("model") or {}).get("center_feature", None)
    return dict(center_feature) if isinstance(center_feature, dict) else None


def _build_model_from_cfg(cfg: dict) -> UnetPlusPlusSemanticCenterHead:
    encoder = cfg["model"].get("encoder") or cfg["model"].get("encoder_name")
    if not encoder:
        raise SystemExit("Config: model.encoder_name is required")
    center_head_type = str((cfg.get("model") or {}).get("center_head_type", "linear_1x1")).strip().lower() or "linear_1x1"
    model = UnetPlusPlusSemanticCenterHead(
        encoder_name=str(encoder),
        encoder_weights=cfg["model"].get("encoder_weights", None),
        in_channels=int(cfg["model"]["in_channels"]),
        classes=int(cfg["model"]["classes"]),
        center_head_type=center_head_type,
        center_feature=_center_feature_cfg_from_cfg(cfg),
    )
    init_path = (cfg.get("train") or {}).get("init_checkpoint", None)
    if init_path:
        load_semantic_checkpoint_non_strict(model, str(Path(init_path).resolve()))
    return model


def _build_loader(cfg: dict, split_txt: Path, device: torch.device) -> DataLoader:
    ds_root = Path(cfg["dataset"]["root"]).resolve()
    num_classes = int(cfg["model"]["classes"])
    input_size = int(cfg["model"]["input_size"])

    import segmentation_models_pytorch as smp

    encoder = cfg["model"].get("encoder") or cfg["model"].get("encoder_name")
    encoder_weights = cfg["model"].get("encoder_weights", None)
    if encoder_weights is None:
        preprocessing_fn = _simple_preprocess_uint8_rgb
    else:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(str(encoder), encoder_weights)

    ds = SegmentationWithCenterDataset(
        dataset_root=ds_root,
        split_txt=split_txt.resolve(),
        num_classes=num_classes,
        augment_fn=get_val_augmentations(input_size, input_size),
        preprocessing_fn=preprocessing_fn,
    )
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def _load_checkpoint(checkpoint_path: Path) -> dict:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise SystemExit(f"Unsupported checkpoint format: {checkpoint_path}")
    return ckpt


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


def _load_gt_instance(instance_root: Path, sample_id: str, target_hw: tuple[int, int]) -> np.ndarray:
    gt_inst_path = (instance_root / "instance_masks" / f"{sample_id}.png").resolve()
    gt_inst = cv2.imread(str(gt_inst_path), cv2.IMREAD_UNCHANGED)
    if gt_inst is None:
        raise FileNotFoundError(str(gt_inst_path))
    if gt_inst.ndim == 3:
        gt_inst = gt_inst[:, :, 0]
    gt_inst = gt_inst.astype(np.uint8)
    th, tw = target_hw
    if gt_inst.shape[:2] != (th, tw):
        gh, gw = gt_inst.shape[:2]
        y0 = (gh - th) // 2
        x0 = (gw - tw) // 2
        gt_inst = gt_inst[y0 : y0 + th, x0 : x0 + tw]
    return gt_inst


def _positive_label_ids(labels: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(labels) if int(v) > 0]


def _label_areas(labels: np.ndarray) -> dict[int, int]:
    return {lab: int(np.sum(labels == lab)) for lab in _positive_label_ids(labels)}


def _label_connected_component_counts(labels: np.ndarray) -> dict[int, int]:
    out = {}
    for lab in _positive_label_ids(labels):
        _cc, count = _connected_components((labels == lab).astype(np.uint8))
        out[lab] = int(count)
    return out


def _marker_label_info(labels: np.ndarray, marker_points: list[dict]) -> dict:
    marker_labels = []
    labels_with_markers = set()
    disappeared = []
    for idx, mp in enumerate(marker_points, start=1):
        y = int(mp["y"])
        x = int(mp["x"])
        lab = int(labels[y, x]) if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] else 0
        marker_labels.append({"marker_index": int(idx), "y": y, "x": x, "label_id": int(lab)})
        if lab > 0:
            labels_with_markers.add(int(lab))
        else:
            disappeared.append(int(idx))
    labels_without_markers = [lab for lab in _positive_label_ids(labels) if lab not in labels_with_markers]
    return {
        "marker_labels": marker_labels,
        "markers_that_disappeared": disappeared,
        "labels_without_markers": labels_without_markers,
    }


def _stage_stats(stage_name: str, labels: np.ndarray, marker_points: list[dict]) -> dict:
    ids = _positive_label_ids(labels)
    marker_info = _marker_label_info(labels, marker_points)
    return {
        "stage": stage_name,
        "count": int(len(ids)),
        "unique_label_ids": ids,
        "label_areas": _label_areas(labels),
        "connected_components_per_label": _label_connected_component_counts(labels),
        "labels_without_markers": marker_info["labels_without_markers"],
        "markers_that_disappeared": marker_info["markers_that_disappeared"],
        "marker_labels": marker_info["marker_labels"],
    }


def _compare_stage_labels(before: np.ndarray, after: np.ndarray) -> dict:
    before_ids = _positive_label_ids(before)
    after_ids = _positive_label_ids(after)
    removed = []
    splits = []
    for lab in before_ids:
        overlap = {int(v) for v in np.unique(after[before == lab]) if int(v) > 0}
        if not overlap:
            removed.append(int(lab))
        if len(overlap) > 1:
            splits.append({"before_label": int(lab), "after_labels": sorted(overlap)})
    merges = []
    for lab in after_ids:
        overlap = {int(v) for v in np.unique(before[after == lab]) if int(v) > 0}
        if len(overlap) > 1:
            merges.append({"after_label": int(lab), "before_labels": sorted(overlap)})
    return {"removed_labels": removed, "splits": splits, "merges": merges}


def _sample_center_metrics(pred_pts: list[tuple[int, int]], gt_pts: list[tuple[int, int]]) -> dict:
    tp, fp, fn, matches = _match_centers(pred_pts, gt_pts, max_dist_px=16.0)
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-7))
    loc_err = float(sum(float(m[4]) for m in matches) / max(len(matches), 1))
    return {
        "center_precision": precision,
        "center_recall": recall,
        "center_f1": f1,
        "center_count_accuracy": float(int(len(pred_pts) == len(gt_pts))),
        "center_loc_err_px": loc_err,
        "predicted_center_count": int(len(pred_pts)),
        "gt_center_count": int(len(gt_pts)),
        "center_matches": [
            {
                "pred_y": int(py),
                "pred_x": int(px),
                "gt_y": int(gy),
                "gt_x": int(gx),
                "distance_px": float(d),
            }
            for (py, px, gy, gx, d) in matches
        ],
    }


def _semantic_topology(gt_mask: np.ndarray, gt_inst: np.ndarray, pred_sem: np.ndarray, trace: dict) -> dict:
    leaf_union = trace["leaf_union"].astype(bool)
    labels_cc = trace["semantic_components"].astype(np.int32)
    comp_ids = list(range(1, int(trace["semantic_component_count"]) + 1))
    comp_areas = {comp_id: int(np.sum(labels_cc == comp_id)) for comp_id in comp_ids}
    gt_instance_ids = [inst_id for inst_id in [1, 2, 3] if int(np.sum(gt_inst == inst_id)) > 0]
    overlaps_per_gt = {}
    gt_split = {}
    holes = {}
    for inst_id in gt_instance_ids:
        overlap = sorted({int(v) for v in np.unique(labels_cc[gt_inst == inst_id]) if int(v) > 0})
        overlaps_per_gt[int(inst_id)] = overlap
        gt_split[int(inst_id)] = bool(len(overlap) > 1)
        holes[int(inst_id)] = {
            "missing_leaflet_pixels": int(np.sum((gt_inst == inst_id) & (~leaf_union))),
            "has_gap": bool(np.any((gt_inst == inst_id) & (~leaf_union))),
        }
    orphan_components = []
    for comp_id in comp_ids:
        overlap_gt = sorted({int(v) for v in np.unique(gt_inst[labels_cc == comp_id]) if int(v) > 0})
        if not overlap_gt:
            orphan_components.append(int(comp_id))
    return {
        "gt_semantic_foreground_class_counts": {
            "leaflet_pixels": int(np.sum(gt_mask == 1)),
            "fibrous_ring_pixels": int(np.sum(gt_mask == 2)),
        },
        "predicted_leaflet_connected_components": int(trace["semantic_component_count"]),
        "predicted_leaflet_component_areas": comp_areas,
        "components_overlapping_each_gt_instance": overlaps_per_gt,
        "gt_instance_split_across_semantic_components": gt_split,
        "semantic_components_overlapping_no_gt_instance": orphan_components,
        "holes_or_gaps_inside_gt_instance": holes,
    }


def _marker_contract(gt_inst: np.ndarray, marker_points: list[dict]) -> dict:
    gt_instance_ids = [inst_id for inst_id in [1, 2, 3] if int(np.sum(gt_inst == inst_id)) > 0]
    per_marker = []
    counts = {int(inst_id): 0 for inst_id in gt_instance_ids}
    outside = 0
    for idx, mp in enumerate(marker_points, start=1):
        y = int(mp["y"])
        x = int(mp["x"])
        inst_id = int(gt_inst[y, x]) if 0 <= y < gt_inst.shape[0] and 0 <= x < gt_inst.shape[1] else 0
        outside_flag = bool(inst_id == 0)
        if outside_flag:
            outside += 1
        elif inst_id in counts:
            counts[inst_id] += 1
        per_marker.append(
            {
                "marker_index": int(idx),
                "y": y,
                "x": x,
                "score": float(mp["score"]),
                "gt_instance_id": int(inst_id),
                "outside_all_gt_instances": outside_flag,
            }
        )
    zero = [int(inst_id) for inst_id, c in counts.items() if int(c) == 0]
    one = [int(inst_id) for inst_id, c in counts.items() if int(c) == 1]
    multi = [int(inst_id) for inst_id, c in counts.items() if int(c) > 1]
    gt_total = int(len(gt_instance_ids))
    return {
        "extracted_marker_count": int(len(marker_points)),
        "markers": per_marker,
        "markers_outside_all_gt_instances": int(outside),
        "gt_instances_with_zero_markers": zero,
        "gt_instances_with_one_marker": one,
        "gt_instances_with_multiple_markers": multi,
        "gt_instances_total": gt_total,
        "gt_instances_with_exactly_one_marker_count": int(len(one)),
        "missing_gt_instance_markers": int(len(zero)),
        "multiple_markers_inside_gt_instances": int(len(multi)),
        "one_marker_per_instance_rate": float(len(one) / max(gt_total, 1)),
        "marker_contract_pass": bool(
            int(len(marker_points)) == gt_total and len(zero) == 0 and len(multi) == 0 and int(outside) == 0
        ),
    }


def _classify_failure(
    marker_contract: dict,
    semantic_topology: dict,
    trace: dict,
    metrics: dict,
    stage2_vs_3: dict,
    stage3_vs_4: dict,
) -> list[str]:
    causes = []
    marker_contract_pass = bool(marker_contract["marker_contract_pass"])
    raw_count = int(trace["raw_reconstruction_count"])
    post_count = int(trace["postprocessed_count"])
    final_count = int(trace["final_count"])
    marker_count = int(marker_contract["extracted_marker_count"])
    fallback_used = any(bool(c["used_fallback"]) for c in trace["component_traces"])
    semantic_split = any(bool(v) for v in semantic_topology["gt_instance_split_across_semantic_components"].values())
    orphan_components = len(semantic_topology["semantic_components_overlapping_no_gt_instance"]) > 0

    if marker_contract_pass:
        if semantic_split:
            causes.append("A")
        if fallback_used and orphan_components:
            causes.append("B")
        if raw_count == marker_count and post_count != raw_count and (stage2_vs_3["splits"] or stage2_vs_3["merges"]):
            causes.append("C")
        if final_count == int(metrics["gt_instance_count"]) and bool(metrics["instance_exact_count"]) and str(metrics["case"]) != "correct":
            causes.append("F")
    if not causes and marker_contract_pass and raw_count != marker_count and fallback_used:
        causes.append("B")
    if not causes and marker_contract_pass and post_count != raw_count and (stage2_vs_3["splits"] or stage2_vs_3["merges"]):
        causes.append("C")
    if not causes and marker_contract_pass and stage3_vs_4["splits"]:
        causes.append("F")
    if not causes:
        causes.append("G")
    return sorted(set(causes))


def _stage_failure_summary(
    sample_id: str,
    marker_contract: dict,
    trace: dict,
    stage_marker: dict,
    stage_raw: dict,
    stage_post: dict,
    stage_final: dict,
) -> dict | None:
    if not marker_contract["marker_contract_pass"]:
        return None
    marker_count = int(stage_marker["count"])
    comparisons = [
        ("raw reconstruction/watershed", marker_count, int(stage_raw["count"]), "_fallback_marker"),
        ("postprocessed reconstruction", int(stage_raw["count"]), int(stage_post["count"]), "_keep_top3_by_area"),
        ("final labels passed to metrics", int(stage_post["count"]), int(stage_final["count"]), "compute_instance_metrics_from_masks"),
    ]
    for stage_name, before, after, default_fn in comparisons:
        if int(after) != int(marker_count) if stage_name == "raw reconstruction/watershed" else int(after) != int(before):
            if stage_name == "raw reconstruction/watershed":
                if any(bool(c["used_fallback"]) for c in trace["component_traces"]):
                    labels = stage_raw["labels_without_markers"]
                    fn_name = "_fallback_marker"
                else:
                    labels = stage_raw["unique_label_ids"]
                    fn_name = "_watershed"
            elif stage_name == "postprocessed reconstruction":
                labels = stage_post["markers_that_disappeared"]
                fn_name = default_fn
            else:
                labels = stage_final["markers_that_disappeared"]
                fn_name = default_fn
            return {
                "sample": sample_id,
                "stage": stage_name,
                "before": int(before),
                "after": int(after),
                "labels": labels,
                "function": fn_name,
            }
    return None


def _mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[mask == 1] = np.array((0, 220, 0), dtype=np.uint8)
    out[mask == 2] = np.array((220, 140, 0), dtype=np.uint8)
    return out


def _labels_to_bgr(labels: np.ndarray) -> np.ndarray:
    h, w = labels.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    palette = {
        1: (0, 255, 0),
        2: (255, 0, 0),
        3: (0, 0, 255),
        4: (255, 255, 0),
        5: (255, 0, 255),
        6: (0, 255, 255),
    }
    for lab in _positive_label_ids(labels):
        out[labels == lab] = np.array(palette.get(lab, (180, 180, 180)), dtype=np.uint8)
    return out


def _center_prob_to_bgr(center_prob: np.ndarray) -> np.ndarray:
    x8 = (np.clip(center_prob.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return cv2.applyColorMap(x8, cv2.COLORMAP_VIRIDIS)


def _semantic_cc_to_bgr(labels_cc: np.ndarray) -> np.ndarray:
    return _labels_to_bgr(labels_cc.astype(np.uint8))


def _matching_panel(gt_inst: np.ndarray, pred_inst: np.ndarray) -> np.ndarray:
    gt_rgb = _labels_to_bgr(gt_inst)
    pred_rgb = _labels_to_bgr(pred_inst)
    return cv2.addWeighted(gt_rgb, 0.5, pred_rgb, 0.5, 0.0)


def _draw_markers(base_bgr: np.ndarray, marker_points: list[dict]) -> np.ndarray:
    out = base_bgr.copy()
    for idx, mp in enumerate(marker_points, start=1):
        x = int(mp["x"])
        y = int(mp["y"])
        cv2.circle(out, (x, y), 7, (255, 0, 0), 2)
        cv2.putText(out, str(idx), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return out


def _resize_panel(img_bgr: np.ndarray, size: int = 256) -> np.ndarray:
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_NEAREST if img_bgr.ndim == 2 else cv2.INTER_AREA)


def _annotate_panel(img_bgr: np.ndarray, title: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (20, 20, 20), thickness=-1)
    cv2.putText(out, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _make_visual_panel(
    *,
    image_rgb_u8: np.ndarray,
    gt_mask: np.ndarray,
    gt_inst: np.ndarray,
    pred_sem: np.ndarray,
    labels_cc: np.ndarray,
    center_prob: np.ndarray,
    marker_points: list[dict],
    raw_inst: np.ndarray,
    final_inst: np.ndarray,
    matching: np.ndarray,
    summary_lines: list[str],
) -> np.ndarray:
    original = cv2.cvtColor(image_rgb_u8, cv2.COLOR_RGB2BGR)
    panels = [
        ("1. original image", original),
        ("2. GT semantic", _mask_to_bgr(gt_mask)),
        ("3. GT instances", _labels_to_bgr(gt_inst)),
        ("4. predicted semantic", _mask_to_bgr(pred_sem)),
        ("5. predicted semantic CC", _semantic_cc_to_bgr(labels_cc)),
        ("6. center probability", _center_prob_to_bgr(center_prob)),
        ("7. extracted markers", _draw_markers(original, marker_points)),
        ("8. raw reconstruction", _labels_to_bgr(raw_inst)),
        ("9. final labels", _labels_to_bgr(final_inst)),
        ("10. GT/pred matching", matching),
    ]
    tiles = [_annotate_panel(_resize_panel(img), title) for title, img in panels]
    top = np.concatenate(tiles[:5], axis=1)
    bottom = np.concatenate(tiles[5:], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    header_h = 26 + 22 * len(summary_lines)
    header = np.full((header_h, grid.shape[1], 3), 18, dtype=np.uint8)
    for i, line in enumerate(summary_lines, start=1):
        cv2.putText(header, line, (12, 18 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate([header, grid], axis=0)


def _format_thr(thr: float) -> str:
    return f"{float(thr):.3f}".replace(".", "p")


def _line_ref(func) -> dict:
    start = int(inspect.getsourcelines(func)[1])
    end = start + len(inspect.getsourcelines(func)[0]) - 1
    file_path = Path(inspect.getsourcefile(func) or "").resolve()
    return {
        "file": str(file_path),
        "start_line": int(start),
        "end_line": int(end),
        "ref": f"{file_path}#L{start}-L{end}",
    }


def _reconstruction_contract_answers() -> dict:
    return {
        "one_output_per_marker_guaranteed": {
            "answer": False,
            "because": "Компонента semantic leaflet без marker получает fallback marker и отдельный output label.",
            "refs": [_line_ref(reconstruct_instances_from_semantic_and_center), _line_ref(_fallback_marker)],
        },
        "behavior_for_unmarked_semantic_components": {
            "answer": "fallback marker inserted, component kept as its own label",
            "refs": [_line_ref(_fallback_marker), _line_ref(reconstruct_instances_from_semantic_and_center)],
        },
        "behavior_when_gt_leaflet_is_split_in_pred_semantic": {
            "answer": "Каждая disconnected semantic component обрабатывается независимо; merge между компонентами не выполняется.",
            "refs": [_line_ref(_connected_components), _line_ref(reconstruct_instances_from_semantic_and_center)],
        },
        "can_one_marker_label_split_into_multiple_output_labels": {
            "answer": False,
            "because": "При <=1 marker внутри semantic component компоненте сразу присваивается один output label; split возможен только при >1 markers через watershed.",
            "refs": [_line_ref(reconstruct_instances_from_semantic_and_center), _line_ref(_watershed)],
        },
        "connected_component_relabel_after_watershed": {
            "answer": False,
            "because": "После watershed выполняется только sequential relabeling и keep_top3_by_area, но не connected-component split/relabel.",
            "refs": [_line_ref(_watershed), _line_ref(_keep_top3_by_area), _line_ref(reconstruct_instances_from_semantic_and_center)],
        },
        "small_components_deleted": {
            "answer": "Only via keep_top3_by_area; no absolute min-area deletion in reconstruction path.",
            "refs": [_line_ref(_keep_top3_by_area)],
        },
        "fallback_labels_added_for_regions_without_marker": {
            "answer": True,
            "refs": [_line_ref(_fallback_marker), _line_ref(reconstruct_instances_from_semantic_and_center)],
        },
        "annulus_included_in_reconstructed_instance_count": {
            "answer": False,
            "because": "Reconstruction uses leaf_union = pred_sem == 1, so ring/class 2 is excluded.",
            "refs": [_line_ref(reconstruct_instances_from_semantic_and_center)],
        },
        "count_contract_consistent_between_reconstruction_and_instance_metrics": {
            "answer": True,
            "because": "instance_exact_count_acc is computed as pred_k == gt_k on the same reconstructed pred_inst labels.",
            "refs": [_line_ref(compute_instance_metrics_from_masks)],
        },
        "failure_mode_evaluator_contract": {
            "answer": "diagnostic evaluator uses the same count logic through _case_type(gt_k, pred_k)",
            "refs": [_line_ref(_case_type)],
        },
    }


def _checkpoint_metadata_entry(cfg: dict, checkpoint_path: Path) -> dict:
    ckpt = _load_checkpoint(checkpoint_path)
    extra = ckpt.get("extra", {}) if isinstance(ckpt.get("extra", {}), dict) else {}
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "saved_iteration": int(ckpt.get("step")) if ckpt.get("step") is not None else None,
        "saved_threshold": extra.get("best_threshold", None),
        "saved_f1": extra.get("best_center_f1", None),
        "best_step_in_extra": extra.get("best_step", None),
        "center_feature_path": ((cfg.get("model") or {}).get("center_feature") or {}).get("module_path", None),
        "adapter_configuration": {
            "expected_channels": ((cfg.get("model") or {}).get("center_feature") or {}).get("expected_channels", None),
            "adapter_out_channels": ((cfg.get("model") or {}).get("center_feature") or {}).get("adapter_out_channels", None),
            "native_stride": ((cfg.get("model") or {}).get("center_feature") or {}).get("native_stride", None),
            "upsample_logits_to_target": ((cfg.get("model") or {}).get("center_feature") or {}).get("upsample_logits_to_target", None),
        },
        "normalization_mode": ((cfg.get("center_loss") or {}).get("normalization_mode", None)),
        "center_fp32": ((cfg.get("train") or {}).get("center_fp32", None)),
    }


def _evaluate_combo(
    *,
    cfg: dict,
    device: torch.device,
    loader,
    checkpoint_tag: str,
    checkpoint_path: Path,
    checkpoint_metadata: dict,
    threshold: float,
    output_root: Path,
    instance_root: Path,
) -> dict:
    model = _build_model_from_cfg(cfg)
    ckpt = _load_checkpoint(checkpoint_path)
    state = ckpt.get("model", ckpt)
    incompat = model.load_state_dict(state, strict=False)
    missing = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
    unexpected = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
    if unexpected or missing:
        raise RuntimeError(f"{checkpoint_tag}: checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()

    combo_name = f"{checkpoint_tag}_thr_{_format_thr(threshold)}"
    combo_visual_root = (output_root / "visual_review" / combo_name).resolve()
    combo_visual_root.mkdir(parents=True, exist_ok=True)

    per_sample_rows = []
    per_sample_details = []
    first_failing_stage = None

    for sample_idx, batch in enumerate(loader):
        with torch.no_grad():
            images = batch["image"].to(device)
            out = model(images)

        image_path = Path(str(batch["image_path"][0])).resolve()
        sample_id = image_path.stem
        image_rgb_u8 = (np.clip(batch["image"].detach().cpu().numpy()[0].transpose(1, 2, 0), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        gt_mask = batch["mask"].detach().cpu().numpy()[0].astype(np.uint8)
        pred_sem = torch.argmax(out["semantic"], dim=1).detach().cpu().numpy()[0].astype(np.uint8)
        center_prob = torch.sigmoid(out["center"]).detach().cpu().numpy()[0, 0].astype(np.float32)
        trace_final, pred_k, pred_pts_scored, trace = reconstruct_instances_from_semantic_and_center(
            pred_sem,
            center_prob,
            float(threshold),
            max_markers=3,
            return_trace=True,
        )
        pred_inst = trace_final
        marker_points = trace["marker_points"]
        pred_pts = [(int(mp["y"]), int(mp["x"])) for mp in marker_points]
        gt_pts = _extract_metadata_centers(str(batch["metadata_path"][0]))
        gt_inst = _load_gt_instance(instance_root, sample_id, pred_sem.shape[:2])
        center_metrics = _sample_center_metrics(pred_pts, gt_pts)
        marker_contract = _marker_contract(gt_inst, marker_points)
        semantic_topology = _semantic_topology(gt_mask, gt_inst, pred_sem, trace)
        gt_k = int(len([k for k in [1, 2, 3] if int(np.sum(gt_inst == k)) > 0]))
        inst_metrics = compute_instance_metrics_from_masks(gt_inst, pred_inst, gt_k=gt_k, pred_k=pred_k)

        stage_marker = _stage_stats("extracted_marker_labels", trace["marker_labels"], marker_points)
        stage_raw = _stage_stats("raw_reconstruction", trace["raw_reconstruction_labels"], marker_points)
        stage_post = _stage_stats("postprocessed_reconstruction", trace["postprocessed_labels"], marker_points)
        stage_final = _stage_stats("final_labels_passed_to_metrics", trace["final_labels"], marker_points)
        stage2_vs_3 = _compare_stage_labels(trace["raw_reconstruction_labels"], trace["postprocessed_labels"])
        stage3_vs_4 = _compare_stage_labels(trace["postprocessed_labels"], trace["final_labels"])

        invariant = _stage_failure_summary(
            sample_id,
            marker_contract,
            trace,
            stage_marker,
            stage_raw,
            stage_post,
            stage_final,
        )
        if first_failing_stage is None and invariant is not None:
            first_failing_stage = invariant

        failure_classes = _classify_failure(marker_contract, semantic_topology, trace, inst_metrics, stage2_vs_3, stage3_vs_4)
        summary_row = {
            "checkpoint_tag": checkpoint_tag,
            "checkpoint_iteration": checkpoint_metadata.get("saved_iteration", None),
            "threshold": float(threshold),
            "sample": sample_id,
            "sample_index": int(sample_idx),
            "gt_instances": int(gt_k),
            "markers": int(marker_contract["extracted_marker_count"]),
            "marker_contract": bool(marker_contract["marker_contract_pass"]),
            "semantic_cc": int(trace["semantic_component_count"]),
            "raw_reconstructed": int(stage_raw["count"]),
            "final_reconstructed": int(stage_final["count"]),
            "exact_count": bool(inst_metrics["instance_exact_count"]),
            "failure_class": ",".join(failure_classes),
        }
        per_sample_rows.append(summary_row)

        matching = _matching_panel(gt_inst, pred_inst)
        summary_lines = [
            f"sample={sample_id} checkpoint={checkpoint_tag} step={checkpoint_metadata.get('saved_iteration')} thr={float(threshold):.3f}",
            f"GT instances={gt_k} markers={marker_contract['extracted_marker_count']} raw={stage_raw['count']} final={stage_final['count']}",
            f"exact_count={bool(inst_metrics['instance_exact_count'])} case={inst_metrics['case']} merged={bool(inst_metrics['instance_merged'])} fragmented={bool(inst_metrics['instance_fragmented'])}",
        ]
        panel = _make_visual_panel(
            image_rgb_u8=image_rgb_u8,
            gt_mask=gt_mask,
            gt_inst=gt_inst,
            pred_sem=pred_sem,
            labels_cc=trace["semantic_components"],
            center_prob=center_prob,
            marker_points=marker_points,
            raw_inst=trace["raw_reconstruction_labels"],
            final_inst=trace["final_labels"],
            matching=matching,
            summary_lines=summary_lines,
        )
        panel_path = (combo_visual_root / f"{sample_id}.png").resolve()
        cv2.imwrite(str(panel_path), panel)

        if bool(marker_contract["marker_contract_pass"]) and not bool(inst_metrics["instance_exact_count"]):
            fail_dir = (output_root / "visual_review" / "marker_pass_instance_fail").resolve()
            fail_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str((fail_dir / f"{combo_name}_{sample_id}.png").resolve()), panel)
        if bool(marker_contract["marker_contract_pass"]) and bool(inst_metrics["instance_exact_count"]):
            pass_dir = (output_root / "visual_review" / "marker_and_instance_pass").resolve()
            pass_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str((pass_dir / f"{combo_name}_{sample_id}.png").resolve()), panel)

        per_sample_details.append(
            {
                "checkpoint_tag": checkpoint_tag,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_iteration": checkpoint_metadata.get("saved_iteration", None),
                "threshold": float(threshold),
                "sample_index": int(sample_idx),
                "sample": sample_id,
                "sample_path": str(image_path),
                "identifiers": {
                    "gt_center_count": int(len(gt_pts)),
                    "gt_instance_count": int(gt_k),
                    "semantic_foreground_class_counts": semantic_topology["gt_semantic_foreground_class_counts"],
                },
                "center_metrics": center_metrics,
                "marker_contract": marker_contract,
                "semantic_topology": semantic_topology,
                "reconstruction_stages": {
                    "extracted_marker_labels": stage_marker,
                    "raw_reconstruction": stage_raw,
                    "postprocessed_reconstruction": stage_post,
                    "final_labels_passed_to_metrics": stage_final,
                    "raw_to_postprocess_delta": stage2_vs_3,
                    "postprocess_to_final_delta": stage3_vs_4,
                },
                "metrics": {
                    "reconstructed_instance_count": int(inst_metrics["pred_instance_count"]),
                    "instance_exact_count": bool(inst_metrics["instance_exact_count"]),
                    "matched_iou": float(inst_metrics["instance_mean_matched_iou"]),
                    "merged": bool(inst_metrics["instance_merged"]),
                    "fragmented": bool(inst_metrics["instance_fragmented"]),
                    "mixed": bool(inst_metrics["instance_mixed"]),
                    "perfect_recovery": bool(inst_metrics["instance_perfect"]),
                    "instance_score": float(inst_metrics["instance_mean_matched_iou"]) - 0.25 * float(inst_metrics["instance_merged_rate"]) - 0.15 * float(inst_metrics["instance_fragmented_rate"]),
                },
                "invariant": invariant,
                "failure_class": failure_classes,
                "visual_panel": str(panel_path),
                "trace": {
                    "component_traces": trace["component_traces"],
                },
            }
        )

    return {
        "checkpoint_tag": checkpoint_tag,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_iteration": checkpoint_metadata.get("saved_iteration", None),
        "threshold": float(threshold),
        "per_sample_rows": per_sample_rows,
        "per_sample_details": per_sample_details,
        "first_failing_stage": first_failing_stage,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="training/configs/unetpp_effb3_centerhead_spatial_x2_2_adapter_legacy_fp32_micro.yaml")
    ap.add_argument("--run-dir", type=str, default="training/runs/unetpp_effb3_centerhead_spatial_x2_2_adapter_legacy_fp32_micro")
    ap.add_argument("--output-dir", type=str, default="training/analysis/centerhead_spatial_x2_2_adapter_reconstruction_audit")
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = _read_yaml((repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    _seed_all(int(cfg.get("seed", 1337)))
    device = _make_device(cfg, device_arg=str(args.device))
    run_dir = (repo_root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    out_dir = (repo_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    microset_txt = (run_dir / "microset.txt").resolve()
    if not microset_txt.exists():
        raise SystemExit(f"Microset file not found: {microset_txt}")

    loader = _build_loader(cfg, microset_txt, device=device)
    instance_root = Path(cfg["dataset"]["instance_root"]).resolve()

    checkpoint_specs = [
        ("best", (run_dir / "best_micro_overfit.pth").resolve()),
        ("last", (run_dir / "last.pth").resolve()),
    ]
    checkpoint_metadata = {}
    for tag, path in checkpoint_specs:
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {path}")
        checkpoint_metadata[tag] = _checkpoint_metadata_entry(cfg, path)

    metadata_path = (out_dir / "checkpoint_metadata.json").resolve()
    metadata_path.write_text(json.dumps(checkpoint_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    best_thr = checkpoint_metadata["best"].get("saved_threshold", None)
    last_thr = checkpoint_metadata["last"].get("saved_threshold", None)
    thresholds_by_checkpoint = {
        "best": sorted({float(x) for x in [best_thr, 0.03, 0.02, 0.05] if x is not None}),
        "last": sorted({float(x) for x in [last_thr, 0.01, 0.005, 0.02] if x is not None}),
    }

    combo_results = []
    for tag, ckpt_path in checkpoint_specs:
        for thr in thresholds_by_checkpoint[tag]:
            combo_results.append(
                _evaluate_combo(
                    cfg=cfg,
                    device=device,
                    loader=loader,
                    checkpoint_tag=tag,
                    checkpoint_path=ckpt_path,
                    checkpoint_metadata=checkpoint_metadata[tag],
                    threshold=float(thr),
                    output_root=out_dir,
                    instance_root=instance_root,
                )
            )

    per_sample_rows = []
    per_sample_details = []
    first_failing_stage = None
    root_cause_counts = {k: 0 for k in ["A", "B", "C", "D", "E", "F", "G"]}
    for combo in combo_results:
        per_sample_rows.extend(combo["per_sample_rows"])
        per_sample_details.extend(combo["per_sample_details"])
        if first_failing_stage is None and combo["first_failing_stage"] is not None:
            first_failing_stage = combo["first_failing_stage"]
        for row in combo["per_sample_details"]:
            is_target_failure = bool(row["marker_contract"]["marker_contract_pass"]) and (
                row["invariant"] is not None or (not bool(row["metrics"]["instance_exact_count"]))
            )
            if is_target_failure:
                for cause in row["failure_class"]:
                    root_cause_counts[str(cause)] = int(root_cause_counts.get(str(cause), 0)) + 1

    csv_path = (out_dir / "per_sample_audit.csv").resolve()
    json_path = (out_dir / "per_sample_audit.json").resolve()
    if per_sample_rows:
        fieldnames = []
        for row in per_sample_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_sample_rows:
                writer.writerow(row)
    json_path.write_text(json.dumps(per_sample_details, ensure_ascii=False, indent=2), encoding="utf-8")

    reconstruction_contract = _reconstruction_contract_answers()
    invariants_payload = {
        "first_failing_stage": first_failing_stage,
        "per_sample_invariants": [
            {
                "checkpoint_tag": row["checkpoint_tag"],
                "threshold": row["threshold"],
                "sample": row["sample"],
                "marker_contract_pass": row["marker_contract"]["marker_contract_pass"],
                "invariant": row["invariant"],
            }
            for row in per_sample_details
        ],
        "reconstruction_contract": reconstruction_contract,
    }
    invariants_path = (out_dir / "reconstruction_invariants.json").resolve()
    invariants_path.write_text(json.dumps(invariants_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "checkpoints": {
            "best_checkpoint_iteration": checkpoint_metadata["best"].get("saved_iteration", None),
            "last_checkpoint_iteration": checkpoint_metadata["last"].get("saved_iteration", None),
            "thresholds_audited": thresholds_by_checkpoint,
        },
        "per_sample_table": per_sample_rows,
        "first_failing_stage": first_failing_stage,
        "reconstruction_contract": {
            "one_output_per_marker_guaranteed": reconstruction_contract["one_output_per_marker_guaranteed"]["answer"],
            "behavior_for_unmarked_semantic_components": reconstruction_contract["behavior_for_unmarked_semantic_components"]["answer"],
            "connected_component_relabel": reconstruction_contract["connected_component_relabel_after_watershed"]["answer"],
            "annulus_included": reconstruction_contract["annulus_included_in_reconstructed_instance_count"]["answer"],
            "count_contract": reconstruction_contract["count_contract_consistent_between_reconstruction_and_instance_metrics"]["answer"],
        },
        "root_cause_counts": {
            "semantic_split": int(root_cause_counts["A"]),
            "fallback_components": int(root_cause_counts["B"]),
            "postprocess_split": int(root_cause_counts["C"]),
            "annulus_leakage": int(root_cause_counts["D"]),
            "metric_mismatch": int(root_cause_counts["E"]),
            "instance_metric_bug": int(root_cause_counts["F"]),
            "marker_diagnostic_bug": int(root_cause_counts["F"]),
            "other": int(root_cause_counts["G"]),
        },
        "synthetic_test": {
            "normal_two_instance_case": "covered by unittest",
            "disconnected_semantic_case": "covered by unittest",
        },
    }

    decision = "A"
    if summary["root_cause_counts"]["metric_mismatch"] > 0:
        decision = "C"
    elif summary["root_cause_counts"]["postprocess_split"] > 0 and summary["root_cause_counts"]["semantic_split"] == 0 and summary["root_cause_counts"]["fallback_components"] == 0:
        decision = "B"
    elif summary["root_cause_counts"]["marker_diagnostic_bug"] > 0 and summary["root_cause_counts"]["semantic_split"] == 0 and summary["root_cause_counts"]["fallback_components"] == 0:
        decision = "D"
    summary["decision"] = decision
    summary["next_step"] = {
        "A": "Точечно диагностировать обработку semantic disconnected components в reconstruction contract, без изменения center branch.",
        "B": "Подготовить минимальный фикс reconstruction/postprocessing logic с отдельным regression test.",
        "C": "Унифицировать metric contract между reconstruction и evaluator и пересчитать существующие run metrics.",
        "D": "Исправить marker diagnostic contract и повторно пересчитать только offline diagnostics.",
    }[decision]

    summary_path = (out_dir / "audit_summary.json").resolve()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
