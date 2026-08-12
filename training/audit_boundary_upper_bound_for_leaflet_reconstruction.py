from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e

import audit_semantic_soft_logit_recoverability as soft_audit
import evaluate_semantic_topology_aux_postrun as postrun
import leaflet_oracle_count_geometric_split_audit as base_audit
import leaflet_oracle_count_geometric_split_forensic as forensic
import semantic_topology_aux as topo_aux


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "training" / "analysis" / "boundary_oracle_upper_bound_audit"
VISUAL_DIR = OUTPUT_DIR / "visual_review"
AUDIT_SPLIT = REPO_ROOT / "datasets" / "converted_full_multiclass_curated" / "test.txt"
CHECKPOINT_PATH = REPO_ROOT / "training" / "runs" / "unetpp_effb3_a100_multiclass_curated_finetune_stage2_lr1e5_100ep" / "best_mean_fg.pth"
CHECKPOINT_SHA256 = "ea19846a35da02cc0cb6041d814f206719eb1926f3b02cfd6fbf448d39834c48"
NORMALIZER_METHOD = "centroid_distance_k_normalizer"
PROHIBITED_PATH_SUBSTRINGS = ("center_full_val_manifest.jsonl", "authoritative_106_holdout", "holdout")
STARTING_MASKS = ("hard_argmax", "p50_diagnostic")
ORACLE_METHODS = (
    "oracle_bridge_removal",
    "ideal_separator_target",
    "oracle_fp_removal",
    "oracle_fn_restoration",
    "perfect_gt_semantic",
)


def _assert_safe_path(path: Path) -> None:
    text = str(path).replace("\\", "/").lower()
    for token in PROHIBITED_PATH_SUBSTRINGS:
        if token.lower() in text:
            raise SystemExit(f"Prohibited path detected in boundary upper-bound audit: {path}")


def build_audit_contract() -> dict[str, Any]:
    return {
        "audit_split": str(AUDIT_SPLIT.resolve()),
        "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
        "expected_checkpoint_sha256": CHECKPOINT_SHA256,
        "normalizer_method": NORMALIZER_METHOD,
        "starting_masks": list(STARTING_MASKS),
        "oracle_methods": list(ORACLE_METHODS),
        "holdout_used": False,
        "center_full_val_manifest_used": False,
        "training_launched": False,
        "checkpoint_modified": False,
    }


def _bridge_oracle_mask(pred_mask01: np.ndarray, false_bridge_pixels01: np.ndarray) -> np.ndarray:
    return ((pred_mask01 > 0) & (false_bridge_pixels01 == 0)).astype(np.uint8)


def _fp_oracle_mask(pred_mask01: np.ndarray, gt_union01: np.ndarray) -> np.ndarray:
    return ((pred_mask01 > 0) & (gt_union01 > 0)).astype(np.uint8)


def _fn_oracle_mask(pred_mask01: np.ndarray, gt_union01: np.ndarray) -> np.ndarray:
    return ((pred_mask01 > 0) | (gt_union01 > 0)).astype(np.uint8)


def _gt_union_mask(gt_inst_u8: np.ndarray) -> np.ndarray:
    return (gt_inst_u8 > 0).astype(np.uint8)


def _separator_oracle_mask(pred_mask01: np.ndarray, separator01: np.ndarray) -> np.ndarray:
    return ((pred_mask01 > 0) & (separator01 == 0)).astype(np.uint8)


def _starting_masks_from_logits(logits: torch.Tensor) -> dict[str, np.ndarray]:
    soft = soft_audit._softmax_probs(logits)
    return {
        "hard_argmax": (soft["pred_semantic"] == 1).astype(np.uint8),
        "p50_diagnostic": soft_audit._threshold_mask(soft["p_leaf"], 0.50).astype(np.uint8),
        "p_leaf": soft["p_leaf"],
    }


def _sample_eval_row(
    *,
    sample_id: str,
    start_mask: str,
    method: str,
    pred_mask01: np.ndarray,
    gt_inst_u8: np.ndarray,
) -> dict[str, Any]:
    gt_k = int(len(topo_aux._positive_instance_ids(gt_inst_u8.astype(np.uint8))))
    normalized = postrun.run_locked_normalization(pred_mask01.astype(np.uint8), gt_k)
    pred_inst = normalized["labels"].astype(np.uint8)
    metrics = base_audit.compute_detailed_instance_metrics(gt_inst_u8.astype(np.uint8), pred_inst, gt_k=gt_k, pred_k=int(normalized["final_group_count"]))
    topo = forensic.classify_semantic_topology(gt_inst_u8.astype(np.uint8), pred_mask01.astype(np.uint8))
    return {
        "sample_id": str(sample_id),
        "starting_mask": start_mask,
        "method": method,
        "gt_count": gt_k,
        "exact_k_success": int(bool(metrics["instance_exact_count_acc"])),
        "mean_matched_iou": float(metrics["instance_mean_matched_iou"]),
        "median_matched_iou": float(metrics["median_matched_iou"]),
        "all_iou_ge_0.50_success": int(bool(metrics["all_iou_ge_0.50"])),
        "all_iou_ge_0.70_success": int(bool(metrics["all_iou_ge_0.70"])),
        "bridge_flag": int(bool(topo["bridge"])),
        "missing_flag": int(bool(topo["missing"])),
        "topology_class": str(topo["topology_class"]),
        "pred_mask_pixels": int(np.sum(pred_mask01 > 0)),
        "normalized_instances": pred_inst,
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = int(len(rows))
    if n == 0:
        return {
            "n": 0,
            "mean_matched_iou": 0.0,
            "all_iou_ge_0.50_count": 0,
            "all_iou_ge_0.50_rate": 0.0,
            "all_iou_ge_0.70_count": 0,
            "all_iou_ge_0.70_rate": 0.0,
            "gt1_success": "0/0",
            "gt2_success": "0/0",
            "gt3_success": "0/0",
            "gt1_rate": 0.0,
            "gt2_rate": 0.0,
            "gt3_rate": 0.0,
        }
    out = {
        "n": n,
        "mean_matched_iou": float(np.mean([float(row["mean_matched_iou"]) for row in rows])),
        "all_iou_ge_0.50_count": int(sum(int(row["all_iou_ge_0.50_success"]) for row in rows)),
        "all_iou_ge_0.70_count": int(sum(int(row["all_iou_ge_0.70_success"]) for row in rows)),
    }
    out["all_iou_ge_0.50_rate"] = float(out["all_iou_ge_0.50_count"] / max(n, 1))
    out["all_iou_ge_0.70_rate"] = float(out["all_iou_ge_0.70_count"] / max(n, 1))
    for gt_count in (1, 2, 3):
        subset = [row for row in rows if int(row["gt_count"]) == gt_count]
        success = int(sum(int(row["all_iou_ge_0.50_success"]) for row in subset))
        out[f"gt{gt_count}_success"] = f"{success}/{len(subset)}"
        out[f"gt{gt_count}_rate"] = float(success / len(subset)) if subset else 0.0
    return out


def _gt_count_rows(start_mask: str, method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for gt_count in (1, 2, 3):
        subset = [row for row in rows if int(row["gt_count"]) == gt_count]
        success50 = int(sum(int(row["all_iou_ge_0.50_success"]) for row in subset))
        success70 = int(sum(int(row["all_iou_ge_0.70_success"]) for row in subset))
        out.append(
            {
                "starting_mask": start_mask,
                "method": method,
                "gt_count": int(gt_count),
                "n": int(len(subset)),
                "all_iou_ge_0.50_count": success50,
                "all_iou_ge_0.50_rate": float(success50 / len(subset)) if subset else 0.0,
                "all_iou_ge_0.70_count": success70,
                "all_iou_ge_0.70_rate": float(success70 / len(subset)) if subset else 0.0,
                "mean_matched_iou": float(np.mean([float(row["mean_matched_iou"]) for row in subset])) if subset else 0.0,
            }
        )
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


def _mask_rgb(mask01: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(mask01.shape + (3,), dtype=np.uint8)
    out[mask01.astype(bool)] = np.asarray(color, dtype=np.uint8)
    return out


def _panel(title: str, image_rgb: np.ndarray) -> np.ndarray:
    canvas = np.full((image_rgb.shape[0] + 36, image_rgb.shape[1], 3), 18, dtype=np.uint8)
    canvas[36:, :, :] = image_rgb
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def _save_visual(
    *,
    category: str,
    sample_id: str,
    image_rgb: np.ndarray,
    gt_inst_u8: np.ndarray,
    start_mask01: np.ndarray,
    bridge_mask01: np.ndarray,
    fp_mask01: np.ndarray,
    fn_mask01: np.ndarray,
    bridge_out: np.ndarray,
    fp_out: np.ndarray,
    fn_out: np.ndarray,
) -> str:
    panels = [
        _panel("RGB", image_rgb),
        _panel("GT Instances", _instance_rgb(gt_inst_u8)),
        _panel("Starting Mask", _mask_rgb(start_mask01, (255, 255, 255))),
        _panel("Bridge Oracle Mask", _mask_rgb(bridge_mask01, (255, 255, 255))),
        _panel("FP Removal Mask", _mask_rgb(fp_mask01, (255, 255, 255))),
        _panel("FN Restoration Mask", _mask_rgb(fn_mask01, (255, 255, 255))),
        _panel("Bridge K-normalized", _instance_rgb(bridge_out)),
        _panel("FP Removal K-normalized", _instance_rgb(fp_out)),
        _panel("FN Restoration K-normalized", _instance_rgb(fn_out)),
    ]
    row1 = np.concatenate(panels[:3], axis=1)
    row2 = np.concatenate(panels[3:6], axis=1)
    row3 = np.concatenate(panels[6:9], axis=1)
    grid = np.concatenate([row1, row2, row3], axis=0)
    out_path = VISUAL_DIR / f"{category}_{sample_id}.png"
    cv2.imwrite(str(out_path), grid)
    return str(out_path.resolve())


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _classify_boundary_viability(p50_bridge: dict[str, Any], p50_separator: dict[str, Any], p50_start: dict[str, Any]) -> str:
    best_delta = max(
        float(p50_bridge["all_iou_ge_0.50_rate"]) - float(p50_start["all_iou_ge_0.50_rate"]),
        float(p50_separator["all_iou_ge_0.50_rate"]) - float(p50_start["all_iou_ge_0.50_rate"]),
    )
    best_gt3 = max(
        float(p50_bridge["gt3_rate"]) - float(p50_start["gt3_rate"]),
        float(p50_separator["gt3_rate"]) - float(p50_start["gt3_rate"]),
    )
    if best_delta >= 0.20:
        return "STRONG_BOUNDARY_UPPER_BOUND"
    if best_delta >= 0.10 and best_gt3 > 0.0:
        return "PROMISING_BOUNDARY_UPPER_BOUND"
    if best_delta > 0.0:
        return "WEAK_BOUNDARY_UPPER_BOUND"
    return "BOUNDARY_INSUFFICIENT"


def _next_step_decision(
    *,
    hard_start: dict[str, Any],
    p50_start: dict[str, Any],
    hard_bridge: dict[str, Any],
    p50_bridge: dict[str, Any],
    hard_separator: dict[str, Any],
    p50_separator: dict[str, Any],
    hard_fp: dict[str, Any],
    p50_fp: dict[str, Any],
    hard_fn: dict[str, Any],
    p50_fn: dict[str, Any],
) -> tuple[str, str]:
    boundary_class = _classify_boundary_viability(p50_bridge, p50_separator, p50_start)
    fp_gain = max(
        float(hard_fp["all_iou_ge_0.50_rate"]) - float(hard_start["all_iou_ge_0.50_rate"]),
        float(p50_fp["all_iou_ge_0.50_rate"]) - float(p50_start["all_iou_ge_0.50_rate"]),
    )
    fn_gain = max(
        float(hard_fn["all_iou_ge_0.50_rate"]) - float(hard_start["all_iou_ge_0.50_rate"]),
        float(p50_fn["all_iou_ge_0.50_rate"]) - float(p50_start["all_iou_ge_0.50_rate"]),
    )
    boundary_gain = max(
        float(hard_bridge["all_iou_ge_0.50_rate"]) - float(hard_start["all_iou_ge_0.50_rate"]),
        float(p50_bridge["all_iou_ge_0.50_rate"]) - float(p50_start["all_iou_ge_0.50_rate"]),
        float(hard_separator["all_iou_ge_0.50_rate"]) - float(hard_start["all_iou_ge_0.50_rate"]),
        float(p50_separator["all_iou_ge_0.50_rate"]) - float(p50_start["all_iou_ge_0.50_rate"]),
    )
    if boundary_class in {"STRONG_BOUNDARY_UPPER_BOUND", "PROMISING_BOUNDARY_UPPER_BOUND"} and fn_gain < boundary_gain + 0.10:
        return (
            "A. BUILD_DEDICATED_BOUNDARY_HEAD",
            "The realistic bridge/separator oracles show a meaningful reconstruction upper bound on the current best semantic mask, so a dedicated boundary head is justified.",
        )
    if boundary_gain > 0.0 and fn_gain >= 0.10 and fp_gain >= 0.10:
        return (
            "B. BUILD_BOUNDARY_PLUS_FOREGROUND_RECOVERY_MODEL",
            "Both false-positive suppression and missing-pixel restoration contribute materially, so the next model should address both rather than only one side.",
        )
    if boundary_gain > 0.0 or fn_gain > 0.0:
        return (
            "C. BUILD_INSTANCE_KEYPOINT_OR_INSTANCE_AFFINITY_MODEL",
            "Boundary-only gains are limited while some unrecovered cases still need stronger instance structure than simple semantic cleanup can provide.",
        )
    return (
        "D. REVISIT_SEMANTIC_MODEL",
        "Even these generous oracles do not move reconstruction enough, so the semantic backbone/task definition needs reconsideration before adding more geometry heads.",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    contract = build_audit_contract()
    _assert_safe_path(AUDIT_SPLIT)
    _assert_safe_path(CHECKPOINT_PATH)

    cfg = topo_aux._read_yaml(REPO_ROOT / "training" / "configs" / "unetpp_effb3_semantic_topology_aux_finetune_100ep.yaml")
    topology_contract = postrun._read_saved_topology_contract(REPO_ROOT / "training" / "runs" / "unetpp_effb3_semantic_topology_aux_finetune_100ep")
    dataset = postrun._build_dataset(cfg, AUDIT_SPLIT, topology_contract)
    device = postrun._resolve_device()
    use_amp = topo_aux._amp_enabled(cfg, device)

    checkpoint_sha = topo_aux._sha256_file(CHECKPOINT_PATH.resolve())
    if checkpoint_sha != CHECKPOINT_SHA256:
        raise SystemExit(f"Checkpoint SHA mismatch for baseline semantic checkpoint: {checkpoint_sha}")

    model = topo_aux.build_model_from_cfg(cfg).to(device)
    checkpoint_meta = postrun._load_checkpoint_into_wrapper(model, CHECKPOINT_PATH)
    model.eval()

    oracle_rows: list[dict[str, Any]] = []
    gt_count_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    gt3_failure_rows: list[dict[str, Any]] = []
    sample_cache: dict[str, Any] = {}
    samples_with_bridge_pixels: dict[str, int] = {name: 0 for name in STARTING_MASKS}
    samples_with_topology_change: dict[str, int] = {name: 0 for name in STARTING_MASKS}
    total_bridge_pixels_removed: dict[str, int] = {name: 0 for name in STARTING_MASKS}

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            sample_id = str(sample["sample_id"])
            image_t = sample["image"].unsqueeze(0).to(device)
            gt_sem_u8 = sample["mask"].numpy().astype(np.uint8)
            gt_inst_u8 = topo_aux._read_u8(Path(sample["instance_path"]))
            gt_inst_u8 = topo_aux._center_crop_like_validation(gt_inst_u8, gt_sem_u8.shape[0], gt_sem_u8.shape[1], is_mask=True)
            gt_union01 = _gt_union_mask(gt_inst_u8)
            with topo_aux._autocast_ctx(device, enabled=use_amp):
                outputs = model(image_t)
            starts = _starting_masks_from_logits(outputs["semantic_logits"])
            separator01 = topo_aux.generate_topology_target(gt_inst_u8.astype(np.uint8), topology_contract, return_parts=True)[1]["inter_instance_separation"].astype(np.uint8)

            sample_cache[sample_id] = {
                "image_path": str(sample["image_path"]),
                "gt_inst_u8": gt_inst_u8,
                "start_masks": {},
                "rows": {},
            }

            for start_name in STARTING_MASKS:
                start_mask01 = starts[start_name].astype(np.uint8)
                bridge_pixels01 = soft_audit._topology_pixel_categories(
                    gt_sem_u8=gt_sem_u8,
                    gt_inst_u8=gt_inst_u8,
                    pred_sem_u8=(start_mask01 > 0).astype(np.uint8),
                    pred_union01=start_mask01,
                    topology_contract=topology_contract,
                )["FALSE_BRIDGE_PIXELS"].astype(np.uint8)
                bridge_removed01 = _bridge_oracle_mask(start_mask01, bridge_pixels01)
                fp_removed01 = _fp_oracle_mask(start_mask01, gt_union01)
                fn_restored01 = _fn_oracle_mask(start_mask01, gt_union01)
                separator_removed01 = _separator_oracle_mask(start_mask01, separator01)
                gt_perfect01 = gt_union01.astype(np.uint8)

                masks = {
                    start_name: start_mask01,
                    "oracle_bridge_removal": bridge_removed01,
                    "ideal_separator_target": separator_removed01,
                    "oracle_fp_removal": fp_removed01,
                    "oracle_fn_restoration": fn_restored01,
                    "perfect_gt_semantic": gt_perfect01,
                }
                sample_cache[sample_id]["start_masks"][start_name] = start_mask01
                if int(np.sum(bridge_pixels01 > 0)) > 0:
                    samples_with_bridge_pixels[start_name] += 1
                total_bridge_pixels_removed[start_name] += int(np.sum(bridge_pixels01 > 0))

                start_topo = forensic.classify_semantic_topology(gt_inst_u8, start_mask01)
                bridge_topo = forensic.classify_semantic_topology(gt_inst_u8, bridge_removed01)
                if (
                    str(start_topo["topology_class"]) != str(bridge_topo["topology_class"])
                    or bool(start_topo["bridge"]) != bool(bridge_topo["bridge"])
                    or bool(start_topo["missing"]) != bool(bridge_topo["missing"])
                ):
                    samples_with_topology_change[start_name] += 1

                for method_name, mask01 in masks.items():
                    row = _sample_eval_row(
                        sample_id=sample_id,
                        start_mask=start_name,
                        method=method_name,
                        pred_mask01=mask01,
                        gt_inst_u8=gt_inst_u8,
                    )
                    row["bridge_pixels_removed"] = int(np.sum(bridge_pixels01 > 0)) if method_name == "oracle_bridge_removal" else 0
                    row["separator_pixels_removed"] = int(np.sum((start_mask01 > 0) & (separator01 > 0))) if method_name == "ideal_separator_target" else 0
                    row["false_positive_pixels_removed"] = int(np.sum((start_mask01 > 0) & (gt_union01 == 0))) if method_name == "oracle_fp_removal" else 0
                    row["missing_pixels_added"] = int(np.sum((gt_union01 > 0) & (start_mask01 == 0))) if method_name == "oracle_fn_restoration" else 0
                    oracle_rows.append({k: v for k, v in row.items() if k != "normalized_instances"})
                    per_sample_rows.append({k: v for k, v in row.items() if k != "normalized_instances"})
                    sample_cache[sample_id]["rows"][(start_name, method_name)] = row

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in oracle_rows:
        grouped.setdefault((str(row["starting_mask"]), str(row["method"])), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (start_name, method_name), rows in sorted(grouped.items()):
        agg = _aggregate_rows(rows)
        agg.update(
            {
                "starting_mask": start_name,
                "method": method_name,
            }
        )
        summary_rows.append(agg)
        gt_count_rows.extend(_gt_count_rows(start_name, method_name, rows))

    def _agg(start_name: str, method_name: str) -> dict[str, Any]:
        return next(row for row in summary_rows if row["starting_mask"] == start_name and row["method"] == method_name)

    hard_start = _agg("hard_argmax", "hard_argmax")
    p50_start = _agg("p50_diagnostic", "p50_diagnostic")
    hard_bridge = _agg("hard_argmax", "oracle_bridge_removal")
    p50_bridge = _agg("p50_diagnostic", "oracle_bridge_removal")
    hard_separator = _agg("hard_argmax", "ideal_separator_target")
    p50_separator = _agg("p50_diagnostic", "ideal_separator_target")
    hard_fp = _agg("hard_argmax", "oracle_fp_removal")
    p50_fp = _agg("p50_diagnostic", "oracle_fp_removal")
    hard_fn = _agg("hard_argmax", "oracle_fn_restoration")
    p50_fn = _agg("p50_diagnostic", "oracle_fn_restoration")
    gt_union = _agg("hard_argmax", "perfect_gt_semantic")

    for row in summary_rows:
        start = _agg(str(row["starting_mask"]), str(row["starting_mask"])) if str(row["method"]) == str(row["starting_mask"]) else _agg(str(row["starting_mask"]), str(row["starting_mask"]))
        row["delta_mean_matched_iou"] = float(row["mean_matched_iou"]) - float(start["mean_matched_iou"])
        row["delta_all_iou_ge_0.50_rate"] = float(row["all_iou_ge_0.50_rate"]) - float(start["all_iou_ge_0.50_rate"])
        row["delta_all_iou_ge_0.70_rate"] = float(row["all_iou_ge_0.70_rate"]) - float(start["all_iou_ge_0.70_rate"])
        if str(row["method"]) == "oracle_bridge_removal":
            row["samples_with_bridge_pixels"] = int(samples_with_bridge_pixels[str(row["starting_mask"])])
            row["samples_with_topology_change_after_bridge_removal"] = int(samples_with_topology_change[str(row["starting_mask"])])
            row["total_bridge_pixels_removed"] = int(total_bridge_pixels_removed[str(row["starting_mask"])])
        else:
            row["samples_with_bridge_pixels"] = 0
            row["samples_with_topology_change_after_bridge_removal"] = 0
            row["total_bridge_pixels_removed"] = 0

    hard_failures = {
        sid: sample_cache[sid]["rows"][("hard_argmax", "hard_argmax")]
        for sid in sample_cache
        if int(sample_cache[sid]["rows"][("hard_argmax", "hard_argmax")]["all_iou_ge_0.50_success"]) == 0
    }
    for sample_id, hard_row in sorted(hard_failures.items()):
        bridge_pass = bool(sample_cache[sample_id]["rows"][("hard_argmax", "oracle_bridge_removal")]["all_iou_ge_0.50_success"])
        fp_pass = bool(sample_cache[sample_id]["rows"][("hard_argmax", "oracle_fp_removal")]["all_iou_ge_0.50_success"])
        fn_pass = bool(sample_cache[sample_id]["rows"][("hard_argmax", "oracle_fn_restoration")]["all_iou_ge_0.50_success"])
        gt_pass = bool(sample_cache[sample_id]["rows"][("hard_argmax", "perfect_gt_semantic")]["all_iou_ge_0.50_success"])
        if bridge_pass and not fn_pass:
            recoverability = "bridge removal only"
        elif fp_pass and (not bridge_pass) and (not fn_pass):
            recoverability = "broader false-positive removal"
        elif fn_pass and (not bridge_pass) and (not fp_pass):
            recoverability = "missing-pixel restoration"
        elif (fp_pass or bridge_pass) and fn_pass:
            recoverability = "both FP and FN correction"
        else:
            recoverability = "neither under locked K-normalization"
        for row in per_sample_rows:
            if row["sample_id"] == sample_id and row["starting_mask"] == "hard_argmax" and row["method"] == "hard_argmax":
                row["hard_failure_recoverability"] = recoverability
                row["hard_bridge_oracle_pass"] = int(bridge_pass)
                row["hard_fp_oracle_pass"] = int(fp_pass)
                row["hard_fn_oracle_pass"] = int(fn_pass)
                row["hard_gt_union_pass"] = int(gt_pass)

    p50_gt3_failures = [
        sid
        for sid in sample_cache
        if int(sample_cache[sid]["rows"][("p50_diagnostic", "p50_diagnostic")]["gt_count"]) == 3
        and int(sample_cache[sid]["rows"][("p50_diagnostic", "p50_diagnostic")]["all_iou_ge_0.50_success"]) == 0
    ]
    for sample_id in sorted(p50_gt3_failures):
        gt3_failure_rows.append(
            {
                "sample_id": sample_id,
                "p50_fail": 1,
                "bridge_recoverable": int(bool(sample_cache[sample_id]["rows"][("p50_diagnostic", "oracle_bridge_removal")]["all_iou_ge_0.50_success"])),
                "fp_recoverable": int(bool(sample_cache[sample_id]["rows"][("p50_diagnostic", "oracle_fp_removal")]["all_iou_ge_0.50_success"])),
                "fn_recoverable": int(bool(sample_cache[sample_id]["rows"][("p50_diagnostic", "oracle_fn_restoration")]["all_iou_ge_0.50_success"])),
                "gt_union_recoverable": int(bool(sample_cache[sample_id]["rows"][("p50_diagnostic", "perfect_gt_semantic")]["all_iou_ge_0.50_success"])),
            }
        )

    boundary_class = _classify_boundary_viability(p50_bridge, p50_separator, p50_start)
    next_step, next_reason = _next_step_decision(
        hard_start=hard_start,
        p50_start=p50_start,
        hard_bridge=hard_bridge,
        p50_bridge=p50_bridge,
        hard_separator=hard_separator,
        p50_separator=p50_separator,
        hard_fp=hard_fp,
        p50_fp=p50_fp,
        hard_fn=hard_fn,
        p50_fn=p50_fn,
    )

    visual_examples: dict[str, str] = {}
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

    def _pick_sample(sample_ids: list[str]) -> str | None:
        return str(sample_ids[0]) if sample_ids else None

    hard_bridge_fp = [
        sid for sid in hard_failures
        if bool(sample_cache[sid]["rows"][("hard_argmax", "oracle_bridge_removal")]["all_iou_ge_0.50_success"])
    ]
    hard_fp_fp = [
        sid for sid in hard_failures
        if bool(sample_cache[sid]["rows"][("hard_argmax", "oracle_fp_removal")]["all_iou_ge_0.50_success"])
    ]
    hard_fn_fp = [
        sid for sid in hard_failures
        if bool(sample_cache[sid]["rows"][("hard_argmax", "oracle_fn_restoration")]["all_iou_ge_0.50_success"])
    ]
    hard_both = [
        sid for sid in hard_failures
        if not bool(sample_cache[sid]["rows"][("hard_argmax", "oracle_bridge_removal")]["all_iou_ge_0.50_success"])
        and not bool(sample_cache[sid]["rows"][("hard_argmax", "oracle_fp_removal")]["all_iou_ge_0.50_success"])
        and not bool(sample_cache[sid]["rows"][("hard_argmax", "oracle_fn_restoration")]["all_iou_ge_0.50_success"])
        and bool(sample_cache[sid]["rows"][("hard_argmax", "perfect_gt_semantic")]["all_iou_ge_0.50_success"])
    ]
    hard_neither = [
        sid for sid in hard_failures
        if not bool(sample_cache[sid]["rows"][("hard_argmax", "oracle_bridge_removal")]["all_iou_ge_0.50_success"])
        and not bool(sample_cache[sid]["rows"][("hard_argmax", "oracle_fp_removal")]["all_iou_ge_0.50_success"])
        and not bool(sample_cache[sid]["rows"][("hard_argmax", "oracle_fn_restoration")]["all_iou_ge_0.50_success"])
        and bool(sample_cache[sid]["rows"][("hard_argmax", "perfect_gt_semantic")]["all_iou_ge_0.50_success"])
    ]
    picks = {
        "bridge_oracle_converts_fail_to_pass": _pick_sample(hard_bridge_fp),
        "perfect_fp_removal_converts_fail_to_pass": _pick_sample(hard_fp_fp),
        "fn_restoration_converts_fail_to_pass": _pick_sample(hard_fn_fp),
        "gt3_boundary_recoverable_case": _pick_sample([row["sample_id"] for row in gt3_failure_rows if int(row["bridge_recoverable"]) == 1]),
        "gt3_fn_dominated_case": _pick_sample([row["sample_id"] for row in gt3_failure_rows if int(row["fn_recoverable"]) == 1 and int(row["bridge_recoverable"]) == 0]),
        "case_requiring_both": _pick_sample(hard_both),
        "case_not_recovered_until_gt_union": _pick_sample(hard_neither),
    }
    for category, sample_id in picks.items():
        if not sample_id:
            continue
        cache = sample_cache[sample_id]
        image_rgb = topo_aux._center_crop_like_validation(
            topo_aux._read_image_rgb(Path(cache["image_path"])),
            cache["gt_inst_u8"].shape[0],
            cache["gt_inst_u8"].shape[1],
            is_mask=False,
        )
        start_row = cache["rows"][("hard_argmax", "hard_argmax")]
        bridge_row = cache["rows"][("hard_argmax", "oracle_bridge_removal")]
        fp_row = cache["rows"][("hard_argmax", "oracle_fp_removal")]
        fn_row = cache["rows"][("hard_argmax", "oracle_fn_restoration")]
        visual_examples[category] = _save_visual(
            category=category,
            sample_id=sample_id,
            image_rgb=image_rgb,
            gt_inst_u8=cache["gt_inst_u8"],
            start_mask01=cache["start_masks"]["hard_argmax"],
            bridge_mask01=_bridge_oracle_mask(cache["start_masks"]["hard_argmax"], soft_audit._topology_pixel_categories(
                gt_sem_u8=_gt_union_mask(cache["gt_inst_u8"]),
                gt_inst_u8=cache["gt_inst_u8"],
                pred_sem_u8=cache["start_masks"]["hard_argmax"],
                pred_union01=cache["start_masks"]["hard_argmax"],
                topology_contract=topology_contract,
            )["FALSE_BRIDGE_PIXELS"].astype(np.uint8)),
            fp_mask01=_fp_oracle_mask(cache["start_masks"]["hard_argmax"], _gt_union_mask(cache["gt_inst_u8"])),
            fn_mask01=_fn_oracle_mask(cache["start_masks"]["hard_argmax"], _gt_union_mask(cache["gt_inst_u8"])),
            bridge_out=bridge_row["normalized_instances"].astype(np.uint8),
            fp_out=fp_row["normalized_instances"].astype(np.uint8),
            fn_out=fn_row["normalized_instances"].astype(np.uint8),
        )

    summary = {
        "contract": contract,
        "checkpoint": checkpoint_meta,
        "starting_masks": {
            "hard_argmax": hard_start,
            "p50_diagnostic": p50_start,
        },
        "oracle_summary": summary_rows,
        "boundary_upper_bound_classification": boundary_class,
        "next_step": {
            "decision": next_step,
            "reason": next_reason,
        },
        "bridge_oracle_diagnostics": {
            "hard_argmax": {
                "total_bridge_pixels_removed": int(total_bridge_pixels_removed["hard_argmax"]),
                "samples_with_bridge_pixels": int(samples_with_bridge_pixels["hard_argmax"]),
                "samples_with_topology_change_after_removal": int(samples_with_topology_change["hard_argmax"]),
            },
            "p50_diagnostic": {
                "total_bridge_pixels_removed": int(total_bridge_pixels_removed["p50_diagnostic"]),
                "samples_with_bridge_pixels": int(samples_with_bridge_pixels["p50_diagnostic"]),
                "samples_with_topology_change_after_removal": int(samples_with_topology_change["p50_diagnostic"]),
            },
        },
        "gt3_failure_analysis": gt3_failure_rows,
        "visual_examples": visual_examples,
    }

    _write_json(OUTPUT_DIR / "audit_summary.json", summary)
    _write_csv(
        OUTPUT_DIR / "oracle_comparison.csv",
        summary_rows,
        [
            "starting_mask",
            "method",
            "n",
            "mean_matched_iou",
            "all_iou_ge_0.50_count",
            "all_iou_ge_0.50_rate",
            "all_iou_ge_0.70_count",
            "all_iou_ge_0.70_rate",
            "gt1_success",
            "gt2_success",
            "gt3_success",
            "gt1_rate",
            "gt2_rate",
            "gt3_rate",
            "delta_mean_matched_iou",
            "delta_all_iou_ge_0.50_rate",
            "delta_all_iou_ge_0.70_rate",
            "samples_with_bridge_pixels",
            "samples_with_topology_change_after_bridge_removal",
            "total_bridge_pixels_removed",
        ],
    )
    _write_csv(
        OUTPUT_DIR / "gt_count_comparison.csv",
        gt_count_rows,
        [
            "starting_mask",
            "method",
            "gt_count",
            "n",
            "all_iou_ge_0.50_count",
            "all_iou_ge_0.50_rate",
            "all_iou_ge_0.70_count",
            "all_iou_ge_0.70_rate",
            "mean_matched_iou",
        ],
    )
    _write_csv(
        OUTPUT_DIR / "per_sample_recoverability.csv",
        per_sample_rows,
        [
            "sample_id",
            "starting_mask",
            "method",
            "gt_count",
            "exact_k_success",
            "mean_matched_iou",
            "median_matched_iou",
            "all_iou_ge_0.50_success",
            "all_iou_ge_0.70_success",
            "bridge_flag",
            "missing_flag",
            "topology_class",
            "bridge_pixels_removed",
            "separator_pixels_removed",
            "false_positive_pixels_removed",
            "missing_pixels_added",
            "hard_failure_recoverability",
            "hard_bridge_oracle_pass",
            "hard_fp_oracle_pass",
            "hard_fn_oracle_pass",
            "hard_gt_union_pass",
        ],
    )
    _write_csv(
        OUTPUT_DIR / "gt3_failure_analysis.csv",
        gt3_failure_rows,
        ["sample_id", "p50_fail", "bridge_recoverable", "fp_recoverable", "fn_recoverable", "gt_union_recoverable"],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
