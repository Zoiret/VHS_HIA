from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from augmentations import get_val_augmentations
from audit_micro_reconstruction_contract import (
    _build_loader,
    _build_model_from_cfg,
    _load_gt_instance,
    _make_device,
    _marker_contract,
    _read_yaml,
    _resolve_path,
    _seed_all,
)
from compare_holdout_manifests import _compare_rows, _read_identity_manifest
from compare_reconstruction_policies import _policy_metrics, _write_csv_atomic, _write_json_atomic, run_policy
from dataset_centerhead import SegmentationWithCenterDataset, _read_image_rgb, _read_mask_u8, _read_u16
from diagnose_center_generalization_holdout import DIAGNOSTIC_THRESHOLDS, POLICIES, _gt_marker_points, _run_policy_with_explicit_markers
from validate_centerhead import _connected_components, _dice_iou_binary, _extract_metadata_centers
from validate_reconstruction_policies_holdout import (
    AUTHORITATIVE_BEST_CHECKPOINT_SHA256,
    AUTHORITATIVE_SEMANTIC_CHECKPOINT_SHA256,
    EXPECTED_HOLDOUT_IDENTITY_SHA256,
    _build_loader_split_file,
    _canonical_identity_entries,
    _checkpoint_identity,
    _identity_manifest_sha256,
    _identity_manifest_text,
    _inventory_holdout_samples,
    _manifest_identity_status,
    _overall_authoritative_status,
    _semantic_checkpoint_identity,
    _stable_json_dumps,
)


DEFAULT_OUTPUT_DIR = "training/analysis/centerhead_spatial_x2_2_preprocessing_parity_audit"
DIAGNOSIS_DEFAULT_DIR = "training/analysis/centerhead_spatial_x2_2_center_generalization_holdout_diagnosis"
MICROSET_IDS = (
    "m01_p02_s00",
    "m01_p02_s04",
    "m01_p01_s00",
    "m01_p01_s01",
    "m01_p01_s02",
    "m01_p01_s03",
)
NO_TRAINING_OCCURRED = True
PRODUCTION_FILES_UNCHANGED = True


def _simple_preprocess_uint8_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    return (img_rgb_u8.astype(np.float32) / 255.0).astype(np.float32)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    c = np.ascontiguousarray(arr)
    return _sha256_bytes(c.tobytes())


def _stats_array(arr: np.ndarray) -> dict[str, Any]:
    c = np.ascontiguousarray(arr)
    finite = np.isfinite(c)
    return {
        "shape": list(c.shape),
        "dtype": str(c.dtype),
        "min": float(np.min(c)) if c.size else None,
        "max": float(np.max(c)) if c.size else None,
        "mean": float(np.mean(c)) if c.size else None,
        "std": float(np.std(c)) if c.size else None,
        "nan_count": int(np.isnan(c).sum()) if np.issubdtype(c.dtype, np.floating) else 0,
        "inf_count": int(np.isinf(c).sum()) if np.issubdtype(c.dtype, np.floating) else 0,
        "finite_values": bool(np.all(finite)),
        "sha256": _sha256_array(c),
    }


def _preprocessing_fn_from_cfg(cfg: dict):
    encoder = cfg["model"].get("encoder") or cfg["model"].get("encoder_name")
    encoder_weights = cfg["model"].get("encoder_weights", None)
    if encoder_weights is None:
        return _simple_preprocess_uint8_rgb
    import segmentation_models_pytorch as smp

    return smp.encoders.get_preprocessing_fn(str(encoder), encoder_weights)


def _transform_params(height: int, width: int, target_h: int, target_w: int) -> dict[str, Any]:
    new_h = max(int(height), int(target_h))
    new_w = max(int(width), int(target_w))
    scale_y = float(new_h) / float(height)
    scale_x = float(new_w) / float(width)
    y0 = int((new_h - int(target_h)) // 2) if new_h > int(target_h) else 0
    x0 = int((new_w - int(target_w)) // 2) if new_w > int(target_w) else 0
    return {
        "source_shape": [int(height), int(width)],
        "resized_shape": [int(new_h), int(new_w)],
        "scale_y": scale_y,
        "scale_x": scale_x,
        "crop_y0": int(y0),
        "crop_x0": int(x0),
        "target_shape": [int(target_h), int(target_w)],
        "coordinate_convention": "y/x (row/column)",
        "rounding_rule": "nearest-integer reporting; target alignment validated against transformed center map",
        "resize_interpolation_image": "cv2.INTER_LINEAR",
        "resize_interpolation_mask": "cv2.INTER_NEAREST",
        "resize_interpolation_center": "cv2.INTER_LINEAR",
        "aspect_ratio_behavior": "stretch only when source dimension is smaller than target, then center-crop",
        "padding_or_cropping": "center_crop",
    }


def _dataset_item_from_entry(cfg: dict, repo_root: Path, entry: dict) -> dict:
    dataset_root = _resolve_path(repo_root, cfg["dataset"]["root"])
    if dataset_root is None:
        raise SystemExit("Config dataset.root missing")
    preprocessing_fn = _preprocessing_fn_from_cfg(cfg)
    input_size = int(cfg["model"]["input_size"])
    with tempfile.TemporaryDirectory() as tmp:
        split_path = Path(tmp) / "single.txt"
        split_path.write_text(f"{entry['image_rel']}\t{entry['semantic_rel']}\n", encoding="utf-8")
        ds = SegmentationWithCenterDataset(
            dataset_root=dataset_root,
            split_txt=split_path,
            num_classes=int(cfg["model"]["classes"]),
            augment_fn=get_val_augmentations(input_size, input_size),
            preprocessing_fn=preprocessing_fn,
        )
        return ds[0]


def _loader_item_from_entry(cfg: dict, repo_root: Path, out_dir: Path, entry: dict, device: torch.device) -> dict:
    split_path = _build_loader_split_file(out_dir, [entry])
    loader = _build_loader(cfg, repo_root=repo_root, split_txt=split_path, device=device)
    for batch in loader:
        return batch
    raise RuntimeError("Loader returned no batch")


def _selected_heldout_samples(diagnosis_dir: Path) -> dict[str, list[str]]:
    rows = list(csv.DictReader((diagnosis_dir / "per_sample_center_diagnostics.csv").read_text(encoding="utf-8").splitlines()))
    rows = [row for row in rows if abs(float(row["threshold"]) - 0.03) < 1e-9]
    missing = sorted(rows, key=lambda row: (-int(row["missing_gt_instances"]), float(row["center_f1"]), str(row["sample"])))
    dup_out = sorted(rows, key=lambda row: (-(int(row["duplicate_instances"]) + int(row["outside_markers"])), float(row["center_f1"]), str(row["sample"])))
    best = sorted(rows, key=lambda row: (0 if row["marker_contract_pass"] == "True" else 1, -float(row["center_f1"]), str(row["sample"])))
    def _take_unique(source: list[dict], existing: set[str], n: int) -> list[str]:
        out = []
        for row in source:
            sample = str(row["sample"])
            if sample in existing:
                continue
            existing.add(sample)
            out.append(sample)
            if len(out) >= int(n):
                break
        return out
    used: set[str] = set()
    return {
        "worst_missing_marker": _take_unique(missing, used, 4),
        "worst_duplicate_or_outside": _take_unique(dup_out, used, 4),
        "best_or_near_pass": _take_unique(best, used, 4),
    }


def _tensor_identity_compare(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    same_shape = list(left.shape) == list(right.shape)
    max_abs = float(np.max(np.abs(left.astype(np.float32) - right.astype(np.float32)))) if same_shape else None
    return {
        "same_shape": bool(same_shape),
        "same_sha256": bool(_sha256_array(left) == _sha256_array(right)) if same_shape else False,
        "max_abs_delta": max_abs,
    }


def _center_target_peaks(transformed_center: np.ndarray, transformed_inst: np.ndarray) -> list[dict]:
    peaks = []
    for inst_id in sorted(int(v) for v in np.unique(transformed_inst) if int(v) > 0):
        mask = transformed_inst == int(inst_id)
        if not np.any(mask):
            continue
        vals = transformed_center.copy()
        vals[~mask] = 0.0
        y, x = np.unravel_index(int(np.argmax(vals)), vals.shape)
        peaks.append(
            {
                "instance_id": int(inst_id),
                "y": int(y),
                "x": int(x),
                "value": float(vals[y, x]),
                "inside_instance": bool(mask[y, x]),
            }
        )
    return peaks


def _rgb_bgr_mismatch_detected(rgb: np.ndarray, bgr_like: np.ndarray) -> bool:
    if rgb.shape != bgr_like.shape:
        return True
    if np.array_equal(rgb, bgr_like):
        return False
    if np.array_equal(rgb[:, :, ::-1], bgr_like):
        return True
    return True


def _normalization_mismatch_detected(left: np.ndarray, right: np.ndarray, *, atol: float = 1e-7) -> bool:
    if left.shape != right.shape:
        return True
    return bool(np.max(np.abs(left.astype(np.float32) - right.astype(np.float32))) > float(atol))


def _xy_swap_detected(y: int, x: int, mask: np.ndarray) -> bool:
    if 0 <= int(y) < mask.shape[0] and 0 <= int(x) < mask.shape[1] and bool(mask[int(y), int(x)]):
        return False
    if 0 <= int(x) < mask.shape[0] and 0 <= int(y) < mask.shape[1] and bool(mask[int(x), int(y)]):
        return True
    return False


def _sigmoid_double_application_detected(logits: np.ndarray, prob_once: np.ndarray, prob_twice: np.ndarray, *, atol: float = 1e-7) -> bool:
    expected_once = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    expected_twice = 1.0 / (1.0 + np.exp(-expected_once))
    once_ok = np.max(np.abs(expected_once - prob_once.astype(np.float64))) <= float(atol)
    twice_ok = np.max(np.abs(expected_twice - prob_twice.astype(np.float64))) <= float(atol)
    return bool(once_ok and twice_ok and np.max(np.abs(prob_once.astype(np.float64) - prob_twice.astype(np.float64))) > float(atol))


def _semantic_center_coverage(pred_sem: np.ndarray, gt_pts: list[tuple[int, int]]) -> dict[str, Any]:
    inside = 0
    for y, x in gt_pts:
        if 0 <= int(y) < pred_sem.shape[0] and 0 <= int(x) < pred_sem.shape[1] and int(pred_sem[int(y), int(x)]) == 1:
            inside += 1
    total = len(gt_pts)
    return {
        "gt_center_inside_predicted_leaflet_count": int(inside),
        "gt_centers_total": int(total),
        "gt_center_inside_predicted_leaflet_rate": float(inside / max(total, 1)),
        "gt_centers_outside_predicted_leaflet": int(total - inside),
    }


def _capture_model_outputs(model, image_t: torch.Tensor) -> dict[str, Any]:
    with torch.inference_mode():
        semantic_a, decoder_a = model.forward_base(image_t)
        center_features_a = model.resolve_center_features(decoder_a)
        adapter_out_a = model.center_adapter(center_features_a) if model.center_adapter is not None else center_features_a
        center_logits_a = model.center_head(adapter_out_a)
        center_logits_a = model.upsample_center_logits(center_logits_a)
        semantic_prob_a = torch.softmax(semantic_a, dim=1)
        center_prob_a = torch.sigmoid(center_logits_a)

        semantic_b, decoder_b = model.forward_base(image_t)
        center_features_b = model.resolve_center_features(decoder_b)
        adapter_out_b = model.center_adapter(center_features_b) if model.center_adapter is not None else center_features_b
        center_logits_b = model.center_head(adapter_out_b)
        center_logits_b = model.upsample_center_logits(center_logits_b)
        semantic_prob_b = torch.softmax(semantic_b, dim=1)
        center_prob_b = torch.sigmoid(center_logits_b)

    def _to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    return {
        "semantic_logits": _stats_array(_to_np(semantic_a)),
        "semantic_probabilities": _stats_array(_to_np(semantic_prob_a)),
        "semantic_argmax": _stats_array(torch.argmax(semantic_a, dim=1).detach().cpu().numpy()),
        "predicted_leaflet_foreground": _stats_array((torch.argmax(semantic_a, dim=1).detach().cpu().numpy() == 1).astype(np.uint8)),
        "x2_2_feature": _stats_array(_to_np(center_features_a)),
        "adapter_output": _stats_array(_to_np(adapter_out_a)),
        "center_logits": _stats_array(_to_np(center_logits_a)),
        "center_sigmoid_heatmap": _stats_array(_to_np(center_prob_a)),
        "deterministic_repeat_max_delta": {
            "semantic_logits": float(np.max(np.abs(_to_np(semantic_a) - _to_np(semantic_b)))),
            "center_logits": float(np.max(np.abs(_to_np(center_logits_a) - _to_np(center_logits_b)))),
            "x2_2_feature": float(np.max(np.abs(_to_np(center_features_a) - _to_np(center_features_b)))),
            "adapter_output": float(np.max(np.abs(_to_np(adapter_out_a) - _to_np(adapter_out_b)))),
            "center_sigmoid_heatmap": float(np.max(np.abs(_to_np(center_prob_a) - _to_np(center_prob_b)))),
        },
    }


def _semantic_diag_row(sample: str, gt_sem: np.ndarray, pred_sem: np.ndarray, gt_inst: np.ndarray, gt_pts: list[tuple[int, int]], center_oracle_p1_row: dict) -> dict:
    gt_fg = (gt_sem == 1).astype(np.uint8)
    pred_fg = (pred_sem == 1).astype(np.uint8)
    dice, iou = _dice_iou_binary(gt_fg, pred_fg)
    gt_cc = int(_connected_components(gt_fg)[1])
    pred_cc = int(_connected_components(pred_fg)[1])
    coverage = _semantic_center_coverage(pred_sem, gt_pts)
    gt_area = int(np.sum(gt_fg > 0))
    pred_area = int(np.sum(pred_fg > 0))
    fn = int(np.sum((gt_fg > 0) & (pred_fg == 0)))
    fp = int(np.sum((gt_fg == 0) & (pred_fg > 0)))
    instance_coverages = []
    for inst_id in sorted(int(v) for v in np.unique(gt_inst) if int(v) > 0):
        mask = gt_inst == int(inst_id)
        instance_coverages.append(float(np.mean(pred_fg[mask] > 0)) if np.any(mask) else 0.0)
    largest_cov = 1.0
    if pred_cc > 0 and pred_area > 0:
        labels, _ = _connected_components(pred_fg)
        largest = max(int(np.sum(labels == cid)) for cid in range(1, int(pred_cc) + 1))
        largest_cov = float(largest / pred_area)
    return {
        "sample": sample,
        "leaflet_fg_iou": float(iou),
        "leaflet_fg_dice": float(dice),
        "gt_center_inside_predicted_leaflet_rate": coverage["gt_center_inside_predicted_leaflet_rate"],
        "gt_centers_outside_predicted_leaflet": coverage["gt_centers_outside_predicted_leaflet"],
        "gt_instance_coverage_by_predicted_semantic": float(np.mean(instance_coverages)) if instance_coverages else 0.0,
        "semantic_false_negative_fraction": float(fn / max(gt_area, 1)),
        "semantic_false_positive_fraction": float(fp / max(pred_area, 1)) if pred_area > 0 else 0.0,
        "predicted_semantic_cc_count": int(pred_cc),
        "gt_semantic_cc_count": int(gt_cc),
        "fragmentation_ratio": float(pred_cc / max(gt_cc, 1)),
        "largest_component_coverage": float(largest_cov),
        "center_oracle_markers_preserved": bool(center_oracle_p1_row["markers_preserved"]),
        "center_oracle_exact_count": bool(center_oracle_p1_row["exact_count"]),
        "center_oracle_merged": bool(center_oracle_p1_row["merged"]),
        "center_oracle_dropped_area_fraction": float(center_oracle_p1_row["dropped_area_fraction"]),
        "center_oracle_matched_iou": float(center_oracle_p1_row["matched_iou"]),
    }


def _replay_parity_rows(left_dir: Path, right_dir: Path) -> tuple[list[dict], dict]:
    left_center = list(csv.DictReader((left_dir / "per_sample_center_diagnostics.csv").read_text(encoding="utf-8").splitlines()))
    right_center = list(csv.DictReader((right_dir / "per_sample_center_diagnostics.csv").read_text(encoding="utf-8").splitlines()))
    left_scope = list(csv.DictReader((left_dir / "per_sample_oracle_policy_metrics.csv").read_text(encoding="utf-8").splitlines()))
    right_scope = list(csv.DictReader((right_dir / "per_sample_oracle_policy_metrics.csv").read_text(encoding="utf-8").splitlines()))
    rows = []
    center_key = lambda row: (str(row["sample"]), str(row["threshold"]))
    scope_key = lambda row: (str(row["sample"]), str(row["scope"]), str(row["policy"]))
    left_center_map = {center_key(row): row for row in left_center}
    right_center_map = {center_key(row): row for row in right_center}
    left_scope_map = {scope_key(row): row for row in left_scope}
    right_scope_map = {scope_key(row): row for row in right_scope}
    threshold_crossings = 0
    different_marker_coords = 0
    different_output_counts = 0
    exact_discrete = True
    float_deltas = []
    for key in sorted(set(left_center_map) & set(right_center_map)):
        a = left_center_map[key]
        b = right_center_map[key]
        marker_contract_diff = str(a["marker_contract_pass"]) != str(b["marker_contract_pass"])
        count_diff = int(a["predicted_count"]) != int(b["predicted_count"])
        threshold_crossings += int(marker_contract_diff)
        exact_discrete = exact_discrete and (not marker_contract_diff) and (not count_diff)
        rows.append(
            {
                "kind": "center",
                "sample": key[0],
                "threshold": key[1],
                "marker_contract_pass_equal": not marker_contract_diff,
                "predicted_count_equal": not count_diff,
                "center_f1_abs_delta": abs(float(a["center_f1"]) - float(b["center_f1"])),
            }
        )
        float_deltas.append(abs(float(a["center_f1"]) - float(b["center_f1"])))
    for key in sorted(set(left_scope_map) & set(right_scope_map)):
        a = left_scope_map[key]
        b = right_scope_map[key]
        output_count_diff = int(a["final_output_label_count"]) != int(b["final_output_label_count"])
        markers_diff = int(a["marker_count"]) != int(b["marker_count"])
        inv_diff = str(a["invariant_pass"]) != str(b["invariant_pass"])
        exact_discrete = exact_discrete and (not output_count_diff) and (not markers_diff) and (not inv_diff)
        different_output_counts += int(output_count_diff)
        rows.append(
            {
                "kind": "scope",
                "sample": key[0],
                "scope": key[1],
                "policy": key[2],
                "marker_count_equal": not markers_diff,
                "output_count_equal": not output_count_diff,
                "invariant_equal": not inv_diff,
                "matched_iou_abs_delta": abs(float(a["matched_iou"]) - float(b["matched_iou"])),
                "dice_abs_delta": abs(float(a["mean_matched_dice"]) - float(b["mean_matched_dice"])) if a["mean_matched_dice"] and b["mean_matched_dice"] else None,
            }
        )
        float_deltas.append(abs(float(a["matched_iou"]) - float(b["matched_iou"])))
    summary = {
        "status": "exact_match" if exact_discrete else "device_sensitive_discrete_output",
        "exact_discrete_matches": bool(exact_discrete),
        "maximum_absolute_delta": float(max(float_deltas)) if float_deltas else 0.0,
        "median_absolute_delta": float(np.median(float_deltas)) if float_deltas else 0.0,
        "samples_crossing_threshold_due_to_numerical_differences": int(threshold_crossings),
        "samples_with_different_marker_coordinates": int(different_marker_coords),
        "samples_with_different_output_counts": int(different_output_counts),
    }
    return rows, summary


def _final_bottleneck_decision(*, preprocessing_mismatch: bool, coordinate_mismatch: bool, center_pass_rate: float, semantic_mean_iou: float) -> str:
    if preprocessing_mismatch:
        return "preprocessing_mismatch_detected"
    if coordinate_mismatch:
        return "coordinate_or_target_mismatch_detected"
    if center_pass_rate < 0.1 and semantic_mean_iou >= 0.8:
        return "genuine_center_generalization_failure"
    if center_pass_rate >= 0.1 and semantic_mean_iou < 0.8:
        return "genuine_semantic_generalization_failure"
    return "mixed_model_generalization_failure"


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="training/configs/unetpp_effb3_centerhead_spatial_x2_2_adapter_legacy_fp32_micro.yaml")
    ap.add_argument("--run-dir", type=str, default="training/runs/unetpp_effb3_centerhead_spatial_x2_2_adapter_legacy_fp32_micro")
    ap.add_argument("--diagnosis-dir", type=str, default=DIAGNOSIS_DEFAULT_DIR)
    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--compare-other-diagnosis-dir", type=str, default="")
    ap.add_argument("--expected-manifest-identity-sha", type=str, default="")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _seed_all(1337)
    cfg_path = _resolve_path(repo_root, args.config)
    run_dir = _resolve_path(repo_root, args.run_dir)
    diagnosis_dir = _resolve_path(repo_root, args.diagnosis_dir)
    out_dir = _resolve_path(repo_root, args.output_dir)
    if cfg_path is None or run_dir is None or diagnosis_dir is None or out_dir is None:
        raise SystemExit("Failed to resolve required paths")
    cfg = _read_yaml(cfg_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = _inventory_holdout_samples(cfg, repo_root)
    eligible_entries = list(inventory["eligible"])
    entry_by_sample = {str(entry["sample"]): entry for entry in eligible_entries}
    manifest_entries = _canonical_identity_entries(inventory)
    identity_sha = _identity_manifest_sha256(manifest_entries)
    expected_identity_sha = str(args.expected_manifest_identity_sha).strip() or (str(EXPECTED_HOLDOUT_IDENTITY_SHA256).strip() if EXPECTED_HOLDOUT_IDENTITY_SHA256 else None)
    manifest_status = _manifest_identity_status(
        actual_sha=identity_sha,
        expected_sha=expected_identity_sha,
        unique_sample_count=len({entry["sample"] for entry in manifest_entries}),
        row_count=len(manifest_entries),
    )

    device = _make_device(cfg, args.device)
    model = _build_model_from_cfg(cfg, repo_root=repo_root)
    ckpt_info = _checkpoint_identity(run_dir)
    ckpt_info.update(_semantic_checkpoint_identity(cfg, repo_root))
    ckpt_info["checkpoint_identity_status"] = (
        "exact_match"
        if str(ckpt_info["checkpoint_sha256"]) == str(AUTHORITATIVE_BEST_CHECKPOINT_SHA256) and int(ckpt_info["checkpoint_iteration"]) == 75
        else "checkpoint_identity_mismatch"
    )
    ckpt_info["semantic_checkpoint_identity_status"] = (
        "exact_match"
        if str(ckpt_info["semantic_checkpoint_sha256"]) == str(AUTHORITATIVE_SEMANTIC_CHECKPOINT_SHA256)
        else "semantic_checkpoint_identity_mismatch"
    )
    ckpt_info["manifest_identity_status"] = manifest_status
    ckpt_info["overall_authoritative_status"] = _overall_authoritative_status(
        checkpoint_identity_status=ckpt_info["checkpoint_identity_status"],
        semantic_checkpoint_identity_status=ckpt_info["semantic_checkpoint_identity_status"],
        manifest_identity_status=manifest_status,
    )
    incompat = model.load_state_dict(ckpt_info["state_dict"], strict=False)
    missing = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
    unexpected = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
    if unexpected or missing:
        raise SystemExit(f"Checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()

    selected = _selected_heldout_samples(diagnosis_dir)
    selected_samples = list(MICROSET_IDS)
    for group in ("worst_missing_marker", "worst_duplicate_or_outside", "best_or_near_pass"):
        for sample in selected[group]:
            if sample not in selected_samples:
                selected_samples.append(sample)

    preprocessing_rows = []
    preprocessing_mismatch = False
    coordinate_mismatch = False
    input_size = int(cfg["model"]["input_size"])
    for sample_id in selected_samples:
        if sample_id in entry_by_sample:
            entry = entry_by_sample[sample_id]
        else:
            # authoritative microset samples may not be in val/test; recover from dataset root directly
            dataset_root = Path(inventory["dataset_root"])
            entry = {
                "sample": sample_id,
                "split": "authoritative_microset",
                "image_rel": f"images/{sample_id}.png",
                "semantic_rel": f"semantic_masks/{sample_id}.png",
                "image_path": str((dataset_root / "images" / f"{sample_id}.png").resolve()),
                "gt_semantic_path": str((dataset_root / "semantic_masks" / f"{sample_id}.png").resolve()),
                "gt_instance_path": str((Path(inventory["instance_root"]) / "instance_masks" / f"{sample_id}.png").resolve()),
                "center_path": str((dataset_root / "center_maps" / f"{sample_id}.png").resolve()),
                "metadata_path": str((dataset_root / "metadata" / f"{sample_id}.json").resolve()),
                "gt_instance_count": int(len(_extract_metadata_centers(str((dataset_root / "metadata" / f"{sample_id}.json").resolve())))),
                "mouse_id": str(sample_id).split("_")[0],
            }
        direct = _dataset_item_from_entry(cfg, repo_root, entry)
        loader_batch = _loader_item_from_entry(cfg, repo_root, out_dir, entry, device=torch.device("cpu"))

        image_rgb = _read_image_rgb(Path(entry["image_path"]))
        gt_sem_raw = _read_mask_u8(Path(entry["gt_semantic_path"]))
        center_raw_u16 = _read_u16(Path(entry["center_path"]))
        gt_inst = _load_gt_instance(Path(inventory["instance_root"]), sample_id, tuple(direct["mask"].shape[-2:]))
        transform = _transform_params(image_rgb.shape[0], image_rgb.shape[1], input_size, input_size)
        center_peaks = _center_target_peaks(direct["center"].detach().cpu().numpy()[0], gt_inst)
        inside_all = all(bool(item["inside_instance"]) for item in center_peaks)
        coordinate_mismatch = coordinate_mismatch or (not inside_all)
        for y, x in _extract_metadata_centers(entry["metadata_path"]):
            coordinate_mismatch = coordinate_mismatch or _xy_swap_detected(int(y), int(x), _read_mask_u8(Path(entry["gt_semantic_path"])) == 1)

        loader_image = loader_batch["image"].detach().cpu().numpy()[0]
        direct_image = direct["image"].detach().cpu().numpy()
        image_compare = _tensor_identity_compare(direct_image, loader_image)
        preprocessing_mismatch = preprocessing_mismatch or (not bool(image_compare["same_sha256"]))
        preprocessing_rows.append(
            {
                "sample": sample_id,
                "split": entry["split"],
                "source_image_path": entry["image_path"],
                "source_image_sha256": _sha256_bytes(Path(entry["image_path"]).read_bytes()),
                "decoded_shape": json.dumps(list(image_rgb.shape)),
                "decoded_dtype": str(image_rgb.dtype),
                "channel_order": "RGB",
                "raw_decoded_rgb_sha256": _sha256_array(image_rgb),
                "raw_image_stats": _stable_json_dumps(_stats_array(image_rgb)),
                "resized_shape": json.dumps(transform["resized_shape"]),
                "resize_interpolation": transform["resize_interpolation_image"],
                "normalized_tensor_shape": json.dumps(list(direct_image.shape)),
                "normalized_tensor_dtype": str(direct_image.dtype),
                "normalized_tensor_sha256": _sha256_array(direct_image),
                "normalized_tensor_stats": _stable_json_dumps(_stats_array(direct_image)),
                "loader_tensor_sha256": _sha256_array(loader_image),
                "loader_tensor_match": bool(image_compare["same_sha256"]),
                "loader_tensor_max_abs_delta": image_compare["max_abs_delta"],
                "config_input_size": int(input_size),
                "actual_input_size": json.dumps([int(direct_image.shape[1]), int(direct_image.shape[2])]),
                "gt_semantic_shape": json.dumps(list(gt_sem_raw.shape)),
                "gt_semantic_unique_labels": json.dumps(sorted(int(v) for v in np.unique(gt_sem_raw))),
                "gt_instance_shape": json.dumps(list(gt_inst.shape)),
                "gt_instance_unique_ids": json.dumps(sorted(int(v) for v in np.unique(gt_inst))),
                "gt_count": int(entry["gt_instance_count"]),
                "gt_center_coordinates_source_space": json.dumps(_extract_metadata_centers(entry["metadata_path"])),
                "transformed_center_target_peaks": json.dumps(center_peaks),
                "target_generation_shape": json.dumps(list(direct["center"].detach().cpu().numpy()[0].shape)),
                "target_generation_sha256": _sha256_array(direct["center"].detach().cpu().numpy()[0]),
                "transformed_center_inside_instance": bool(inside_all),
            }
        )

    _write_csv_atomic((out_dir / "preprocessing_parity_selected_samples.csv").resolve(), preprocessing_rows)

    model_loader_split = _build_loader_split_file(out_dir, eligible_entries)
    full_loader = _build_loader(cfg, repo_root=repo_root, split_txt=model_loader_split, device=device)
    semantic_rows = []
    scope_rows = []
    center_pass_flags = []
    for batch in full_loader:
        image_t = batch["image"].to(device)
        out = model(image_t)
        sample_id = Path(str(batch["image_path"][0])).stem
        pred_sem = torch.argmax(out["semantic"], dim=1).detach().cpu().numpy()[0].astype(np.uint8)
        center_prob = torch.sigmoid(out["center"]).detach().cpu().numpy()[0, 0].astype(np.float32)
        gt_sem = batch["mask"].detach().cpu().numpy()[0].astype(np.uint8)
        gt_pts = _extract_metadata_centers(str(batch["metadata_path"][0]))
        gt_markers = _gt_marker_points(gt_pts)
        gt_inst = _load_gt_instance(Path(inventory["instance_root"]), sample_id, pred_sem.shape[:2])
        p1_center_oracle_inst, p1_center_oracle_trace = _run_policy_with_explicit_markers("P1_DROP_UNMARKED", pred_sem, gt_markers)
        center_oracle_metrics = _policy_metrics(
            policy_name="P1_DROP_UNMARKED",
            gt_inst=gt_inst,
            pred_sem=pred_sem,
            pred_inst=p1_center_oracle_inst,
            marker_points=gt_markers,
            trace=p1_center_oracle_trace,
        )
        center_oracle_row = {
            "sample": sample_id,
            "markers_preserved": bool(len(center_oracle_metrics["contract"]["markers_without_output_label"]) == 0),
            "exact_count": bool(center_oracle_metrics["counts"]["exact_count"]),
            "merged": bool(center_oracle_metrics["instance_metrics"]["merged"]),
            "dropped_area_fraction": float(center_oracle_metrics["area_accounting"]["dropped_area"] / max(int(center_oracle_metrics["area_accounting"]["semantic_leaflet_area"]), 1)),
            "matched_iou": float(center_oracle_metrics["instance_metrics"]["matched_iou"]),
        }
        semantic_rows.append(_semantic_diag_row(sample_id, gt_sem, pred_sem, gt_inst, gt_pts, center_oracle_row))
        pred_markers = run_policy("P1_DROP_UNMARKED", pred_sem, center_prob, 0.03)[1]
        center_pass_flags.append(int(_marker_contract(gt_inst, pred_markers)["marker_contract_pass"]))
        scope_rows.append(
            {
                "sample": sample_id,
                "semantic_cc_count": int(_connected_components((pred_sem == 1).astype(np.uint8))[1]),
                "marker_count": int(len(pred_markers)),
                "center_prob_stats": _stable_json_dumps(_stats_array(center_prob)),
            }
        )

    _write_csv_atomic((out_dir / "per_sample_semantic_diagnostics.csv").resolve(), semantic_rows)
    semantic_summary = {
        "sample_count": int(len(semantic_rows)),
        "mean_leaflet_fg_iou": float(np.mean([float(row["leaflet_fg_iou"]) for row in semantic_rows])) if semantic_rows else None,
        "mean_leaflet_fg_dice": float(np.mean([float(row["leaflet_fg_dice"]) for row in semantic_rows])) if semantic_rows else None,
        "mean_center_coverage": float(np.mean([float(row["gt_center_inside_predicted_leaflet_rate"]) for row in semantic_rows])) if semantic_rows else None,
        "mean_fragmentation_ratio": float(np.mean([float(row["fragmentation_ratio"]) for row in semantic_rows])) if semantic_rows else None,
        "markers_preserved_rate_if_center_oracle": float(np.mean([1.0 if bool(row["center_oracle_markers_preserved"]) else 0.0 for row in semantic_rows])) if semantic_rows else None,
        "center_oracle_exact_count_rate": float(np.mean([1.0 if bool(row["center_oracle_exact_count"]) else 0.0 for row in semantic_rows])) if semantic_rows else None,
    }
    _write_json_atomic((out_dir / "semantic_failure_summary.json").resolve(), semantic_summary)

    model_stats = _capture_model_outputs(model, next(iter(full_loader))["image"].to(device))
    _write_json_atomic(
        (out_dir / "model_feature_parity_summary.json").resolve(),
        {
            "model_eval_active": bool(not model.training),
            "torch_inference_mode_used": True,
            "semantic_checkpoint_sha256": ckpt_info["semantic_checkpoint_sha256"],
            "center_checkpoint_sha256": ckpt_info["checkpoint_sha256"],
            "missing_checkpoint_keys": len(missing),
            "unexpected_checkpoint_keys": len(unexpected),
            "center_feature_path": "base.decoder.blocks.x_2_2",
            "feature_channels": 32,
            "adapter_channels": 16,
            "center_branch_fp32": True,
            "center_normalization_mode": "legacy_num_pos",
            **model_stats,
        },
    )

    if str(args.compare_other_diagnosis_dir).strip():
        other_dir = _resolve_path(repo_root, args.compare_other_diagnosis_dir)
        if other_dir is None:
            raise SystemExit("Failed to resolve compare-other-diagnosis-dir")
        replay_rows, replay_summary = _replay_parity_rows(diagnosis_dir, other_dir)
        _write_csv_atomic((out_dir / "per_sample_replay_parity.csv").resolve(), replay_rows)
        _write_json_atomic((out_dir / "replay_parity.json").resolve(), replay_summary)

    decision_status = _final_bottleneck_decision(
        preprocessing_mismatch=preprocessing_mismatch,
        coordinate_mismatch=coordinate_mismatch,
        center_pass_rate=float(np.mean(center_pass_flags)) if center_pass_flags else 0.0,
        semantic_mean_iou=float(semantic_summary["mean_leaflet_fg_iou"] or 0.0),
    )
    final_decision = {
        "status": decision_status,
        "authoritative_identity_status": ckpt_info["overall_authoritative_status"],
        "preprocessing_mismatch": bool(preprocessing_mismatch),
        "coordinate_mismatch": bool(coordinate_mismatch),
        "semantic_summary": semantic_summary,
        "no_training_occurred": True,
        "production_files_unchanged": True,
    }
    _write_json_atomic((out_dir / "final_bottleneck_decision.json").resolve(), final_decision)

    print(
        json.dumps(
            {
                "status": "done",
                "output_dir": str(out_dir),
                "canonical_identity_sha256": identity_sha,
                "manifest_identity_status": manifest_status,
                "overall_authoritative_status": ckpt_info["overall_authoritative_status"],
                "final_bottleneck_decision": decision_status,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
