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
    import torch.nn.functional as F
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e

import evaluate_semantic_topology_aux_postrun as postrun
import leaflet_oracle_count_geometric_split_audit as base_audit
import leaflet_oracle_count_geometric_split_forensic as forensic
import semantic_topology_aux as topo_aux


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "training" / "analysis" / "semantic_soft_logit_recoverability_audit"
VISUAL_DIR = OUTPUT_DIR / "visual_review"
AUDIT_SPLIT = REPO_ROOT / "datasets" / "converted_full_multiclass_curated" / "test.txt"
NORMALIZER_METHOD = "centroid_distance_k_normalizer"
THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60)
HIGH_SEED_THRESHOLD = 0.50
CANDIDATE_THRESHOLD = 0.30
MARGIN_MIN = -0.10
P_LEAF_MIN = 0.30
PROHIBITED_PATH_SUBSTRINGS = ("center_full_val_manifest.jsonl", "authoritative_106_holdout", "holdout")
CATEGORY_ORDER = (
    "TRUE_LEAFLET_CORRECT",
    "TRUE_LEAFLET_MISSED",
    "TRUE_BACKGROUND_CORRECT",
    "FALSE_LEAFLET",
    "MISSING_TOPOLOGY_CRITICAL",
    "FALSE_BRIDGE_PIXELS",
)


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path
    diagnostic_role: str


CHECKPOINTS: tuple[CheckpointSpec, ...] = (
    CheckpointSpec(
        label="baseline",
        path=REPO_ROOT / "training" / "runs" / "unetpp_effb3_a100_multiclass_curated_finetune_stage2_lr1e5_100ep" / "best_mean_fg.pth",
        diagnostic_role="primary",
    ),
    CheckpointSpec(
        label="topology_best_semantic",
        path=REPO_ROOT / "training" / "runs" / "unetpp_effb3_semantic_topology_aux_finetune_100ep" / "best_mean_fg.pth",
        diagnostic_role="secondary",
    ),
)


def _assert_safe_path(path: Path) -> None:
    text = str(path).replace("\\", "/").lower()
    for token in PROHIBITED_PATH_SUBSTRINGS:
        if token.lower() in text:
            raise SystemExit(f"Prohibited path detected in soft-logit audit: {path}")


def build_audit_contract() -> dict[str, Any]:
    return {
        "audit_split": str(AUDIT_SPLIT.resolve()),
        "normalizer_method": NORMALIZER_METHOD,
        "thresholds": list(THRESHOLDS),
        "hysteresis_seed_threshold": float(HIGH_SEED_THRESHOLD),
        "hysteresis_candidate_threshold": float(CANDIDATE_THRESHOLD),
        "optional_margin_rule": {
            "p_leaf_min": float(P_LEAF_MIN),
            "leaflet_margin_min": float(MARGIN_MIN),
        },
        "holdout_used": False,
        "center_full_val_manifest_used": False,
        "checkpoint_selection_modified": False,
        "training_launched": False,
    }


def _softmax_probs(logits: torch.Tensor) -> dict[str, np.ndarray]:
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)
    pred = np.argmax(probs, axis=0).astype(np.uint8)
    p_leaf = probs[1]
    p_competing = np.maximum(probs[0], probs[2])
    margin = p_leaf - p_competing
    return {
        "probs": probs,
        "pred_semantic": pred,
        "p_leaf": p_leaf.astype(np.float32),
        "p_competing": p_competing.astype(np.float32),
        "leaflet_margin": margin.astype(np.float32),
    }


def _threshold_mask(p_leaf: np.ndarray, threshold: float) -> np.ndarray:
    return (p_leaf >= float(threshold)).astype(np.uint8)


def _hysteresis_leaflet_mask(p_leaf: np.ndarray, *, high_threshold: float, candidate_threshold: float) -> np.ndarray:
    seeds = (p_leaf >= float(high_threshold)).astype(np.uint8)
    candidate = (p_leaf >= float(candidate_threshold)).astype(np.uint8)
    if int(np.sum(seeds)) == 0:
        return np.zeros_like(candidate, dtype=np.uint8)
    labels, comp_count = base_audit._connected_components(candidate.astype(np.uint8))
    out = np.zeros_like(candidate, dtype=np.uint8)
    for comp_id in range(1, int(comp_count) + 1):
        comp = labels == int(comp_id)
        if bool(np.any(seeds[comp] > 0)):
            out[comp] = 1
    return out.astype(np.uint8)


def _margin_mask(p_leaf: np.ndarray, competing: np.ndarray, *, p_leaf_min: float, margin_min: float) -> np.ndarray:
    margin = p_leaf - competing
    return ((p_leaf >= float(p_leaf_min)) & (margin >= float(margin_min))).astype(np.uint8)


def _connected_component_count(mask01: np.ndarray) -> int:
    _labels, count = base_audit._connected_components(mask01.astype(np.uint8))
    return int(count)


def _bridge_component_mask(gt_inst_u8: np.ndarray, pred_union01: np.ndarray) -> np.ndarray:
    overlaps = forensic._component_overlap_sets(gt_inst_u8.astype(np.uint8), pred_union01.astype(np.uint8))
    pred_labels = overlaps["pred_labels"].astype(np.uint8)
    out = np.zeros_like(pred_union01, dtype=np.uint8)
    for comp_id_str, gt_ids in overlaps["pred_component_gt_ids"].items():
        comp_id = int(comp_id_str)
        if len(gt_ids) >= 2:
            out[pred_labels == comp_id] = 1
    return out.astype(np.uint8)


def _bridge_component_count(gt_inst_u8: np.ndarray, pred_union01: np.ndarray) -> int:
    overlaps = forensic._component_overlap_sets(gt_inst_u8.astype(np.uint8), pred_union01.astype(np.uint8))
    return int(sum(1 for gt_ids in overlaps["pred_component_gt_ids"].values() if len(gt_ids) >= 2))


def _critical_foreground_mask(gt_inst_u8: np.ndarray, contract: topo_aux.TopologyTargetContract) -> np.ndarray:
    _target, parts = topo_aux.generate_topology_target(gt_inst_u8.astype(np.uint8), contract, return_parts=True)
    return parts["critical_foreground"].astype(np.uint8)


def _topology_pixel_categories(
    *,
    gt_sem_u8: np.ndarray,
    gt_inst_u8: np.ndarray,
    pred_sem_u8: np.ndarray,
    pred_union01: np.ndarray,
    topology_contract: topo_aux.TopologyTargetContract,
) -> dict[str, np.ndarray]:
    gt_leaf = (gt_sem_u8 == 1).astype(np.uint8)
    pred_leaf = (pred_union01 > 0).astype(np.uint8)
    critical_fg = _critical_foreground_mask(gt_inst_u8, topology_contract)
    bridge_pixels = _bridge_component_mask(gt_inst_u8, pred_leaf)
    false_leaflet = ((gt_leaf == 0) & (pred_leaf > 0)).astype(np.uint8)
    return {
        "TRUE_LEAFLET_CORRECT": ((gt_leaf > 0) & (pred_leaf > 0)).astype(np.uint8),
        "TRUE_LEAFLET_MISSED": ((gt_leaf > 0) & (pred_leaf == 0)).astype(np.uint8),
        "TRUE_BACKGROUND_CORRECT": ((gt_leaf == 0) & (pred_leaf == 0)).astype(np.uint8),
        "FALSE_LEAFLET": false_leaflet.astype(np.uint8),
        "MISSING_TOPOLOGY_CRITICAL": ((gt_leaf > 0) & (pred_leaf == 0) & (critical_fg > 0)).astype(np.uint8),
        "FALSE_BRIDGE_PIXELS": ((false_leaflet > 0) & (bridge_pixels > 0)).astype(np.uint8),
    }


def _masked_values(arr: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    return arr[mask01.astype(bool)].astype(np.float32)


def _distribution_summary(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _leaflet_union_metrics(gt_leaf01: np.ndarray, pred_leaf01: np.ndarray) -> dict[str, float]:
    gt = gt_leaf01.astype(bool)
    pred = pred_leaf01.astype(bool)
    tp = float(np.sum(gt & pred))
    fp = float(np.sum((~gt) & pred))
    fn = float(np.sum(gt & (~pred)))
    dice = (2.0 * tp) / max((2.0 * tp + fp + fn), 1.0)
    iou = tp / max((tp + fp + fn), 1.0)
    return {"leaflet_dice": float(dice), "leaflet_iou": float(iou), "tp": tp, "fp": fp, "fn": fn}


def _missing_topology_critical_recall(missing_critical01: np.ndarray, pred_leaf01: np.ndarray) -> dict[str, float]:
    total = int(np.sum(missing_critical01 > 0))
    recovered = int(np.sum((missing_critical01 > 0) & (pred_leaf01 > 0)))
    return {
        "critical_total": total,
        "critical_recovered": recovered,
        "critical_recall": float(recovered / total) if total > 0 else 0.0,
    }


def _sample_semantic_row(sample_id: str, method: str, checkpoint: str, gt_leaf01: np.ndarray, pred_leaf01: np.ndarray, gt_inst_u8: np.ndarray) -> dict[str, Any]:
    gt_k = int(len(topo_aux._positive_instance_ids(gt_inst_u8.astype(np.uint8))))
    normalized = postrun.run_locked_normalization(pred_leaf01.astype(np.uint8), gt_k)
    pred_inst = normalized["labels"].astype(np.uint8)
    metrics = base_audit.compute_detailed_instance_metrics(gt_inst_u8.astype(np.uint8), pred_inst, gt_k=gt_k, pred_k=int(normalized["final_group_count"]))
    topology = forensic.classify_semantic_topology(gt_inst_u8.astype(np.uint8), pred_leaf01.astype(np.uint8))
    union = _leaflet_union_metrics(gt_leaf01, pred_leaf01)
    bridge_count = _bridge_component_count(gt_inst_u8, pred_leaf01)
    return {
        "checkpoint": checkpoint,
        "method": method,
        "sample_id": str(sample_id),
        "gt_count": gt_k,
        "leaflet_dice": float(union["leaflet_dice"]),
        "leaflet_iou": float(union["leaflet_iou"]),
        "connected_components": int(_connected_component_count(pred_leaf01)),
        "false_positive_leaflet_area": int(union["fp"]),
        "bridge_component_count": int(bridge_count),
        "exact_k_success": int(bool(metrics["instance_exact_count_acc"])),
        "mean_matched_iou": float(metrics["instance_mean_matched_iou"]),
        "median_matched_iou": float(metrics["median_matched_iou"]),
        "all_iou_ge_0.50_success": int(bool(metrics["all_iou_ge_0.50"])),
        "all_iou_ge_0.70_success": int(bool(metrics["all_iou_ge_0.70"])),
        "topology_class": str(topology["topology_class"]),
        "bridge_flag": int(bool(topology["bridge"])),
        "missing_flag": int(bool(topology["missing"])),
        "normalized_instances": pred_inst,
    }


def _aggregate_method_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = int(len(rows))
    if n == 0:
        return {
            "n": 0,
            "leaflet_dice": 0.0,
            "leaflet_iou": 0.0,
            "mean_connected_components": 0.0,
            "median_connected_components": 0.0,
            "false_positive_leaflet_area_mean": 0.0,
            "false_bridge_component_count": 0,
            "exact_k_count": 0,
            "exact_k_rate": 0.0,
            "mean_matched_iou": 0.0,
            "all_iou_ge_0.50_count": 0,
            "all_iou_ge_0.50_rate": 0.0,
            "all_iou_ge_0.70_count": 0,
            "all_iou_ge_0.70_rate": 0.0,
            "gt1_success": "0/0",
            "gt2_success": "0/0",
            "gt3_success": "0/0",
        }
    gt_buckets = {}
    for gt_count in (1, 2, 3):
        subset = [row for row in rows if int(row["gt_count"]) == gt_count]
        success = int(sum(int(row["all_iou_ge_0.50_success"]) for row in subset))
        gt_buckets[gt_count] = {"count": len(subset), "success": success, "rate": float(success / len(subset)) if subset else 0.0}
    exact_k_count = int(sum(int(row["exact_k_success"]) for row in rows))
    success50 = int(sum(int(row["all_iou_ge_0.50_success"]) for row in rows))
    success70 = int(sum(int(row["all_iou_ge_0.70_success"]) for row in rows))
    return {
        "n": n,
        "leaflet_dice": float(np.mean([float(row["leaflet_dice"]) for row in rows])),
        "leaflet_iou": float(np.mean([float(row["leaflet_iou"]) for row in rows])),
        "mean_connected_components": float(np.mean([float(row["connected_components"]) for row in rows])),
        "median_connected_components": float(np.median([float(row["connected_components"]) for row in rows])),
        "false_positive_leaflet_area_mean": float(np.mean([float(row["false_positive_leaflet_area"]) for row in rows])),
        "false_bridge_component_count": int(sum(int(row["bridge_component_count"]) for row in rows)),
        "exact_k_count": exact_k_count,
        "exact_k_rate": float(exact_k_count / n),
        "mean_matched_iou": float(np.mean([float(row["mean_matched_iou"]) for row in rows])),
        "all_iou_ge_0.50_count": success50,
        "all_iou_ge_0.50_rate": float(success50 / n),
        "all_iou_ge_0.70_count": success70,
        "all_iou_ge_0.70_rate": float(success70 / n),
        "gt1_success": f"{gt_buckets[1]['success']}/{gt_buckets[1]['count']}",
        "gt2_success": f"{gt_buckets[2]['success']}/{gt_buckets[2]['count']}",
        "gt3_success": f"{gt_buckets[3]['success']}/{gt_buckets[3]['count']}",
        "gt1_rate": float(gt_buckets[1]["rate"]),
        "gt2_rate": float(gt_buckets[2]["rate"]),
        "gt3_rate": float(gt_buckets[3]["rate"]),
    }


def _category_distribution_rows(
    *,
    checkpoint: str,
    category_values: dict[str, dict[str, list[np.ndarray]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        p_leaf = np.concatenate(category_values[category]["p_leaf"], axis=0) if category_values[category]["p_leaf"] else np.zeros((0,), dtype=np.float32)
        p_comp = np.concatenate(category_values[category]["p_competing"], axis=0) if category_values[category]["p_competing"] else np.zeros((0,), dtype=np.float32)
        margin = np.concatenate(category_values[category]["leaflet_margin"], axis=0) if category_values[category]["leaflet_margin"] else np.zeros((0,), dtype=np.float32)
        rows.append(
            {
                "checkpoint": checkpoint,
                "category": category,
                "pixel_count": int(p_leaf.size),
                **{f"p_leaf_{k}": v for k, v in _distribution_summary(p_leaf).items()},
                **{f"p_competing_{k}": v for k, v in _distribution_summary(p_comp).items()},
                **{f"leaflet_margin_{k}": v for k, v in _distribution_summary(margin).items()},
            }
        )
    return rows


def _probability_heatmap_rgb(p_leaf: np.ndarray) -> np.ndarray:
    scaled = np.clip(np.round(p_leaf * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


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


def _panel(title: str, image_rgb: np.ndarray) -> np.ndarray:
    canvas = np.full((image_rgb.shape[0] + 36, image_rgb.shape[1], 3), 18, dtype=np.uint8)
    canvas[36:, :, :] = image_rgb
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def _save_visual(
    *,
    sample_id: str,
    category: str,
    image_rgb: np.ndarray,
    gt_leaf01: np.ndarray,
    hard_mask01: np.ndarray,
    p_leaf: np.ndarray,
    soft_mask01: np.ndarray,
    normalized_instances: np.ndarray,
    gt_instances: np.ndarray,
) -> str:
    panels = [
        _panel("RGB", image_rgb),
        _panel("GT Leaflet", _mask_rgb(gt_leaf01, (255, 255, 255))),
        _panel("Hard Argmax Leaflet", _mask_rgb(hard_mask01, (255, 255, 255))),
        _panel("P(leaflet)", _probability_heatmap_rgb(p_leaf)),
        _panel("Soft Mask", _mask_rgb(soft_mask01, (255, 255, 255))),
        _panel("K-normalized Output", _instance_rgb(normalized_instances)),
        _panel("GT Instances", _instance_rgb(gt_instances)),
    ]
    filler = np.full_like(panels[0], 18)
    row1 = np.concatenate(panels[:4], axis=1)
    row2 = np.concatenate([panels[4], panels[5], panels[6], filler], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
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


def _diagnosis_and_decision(
    *,
    baseline_hard: dict[str, Any],
    baseline_best_soft: dict[str, Any],
    baseline_prob_rows: dict[str, dict[str, Any]],
) -> tuple[str, str, str]:
    critical_missed = baseline_prob_rows["MISSING_TOPOLOGY_CRITICAL"]
    background = baseline_prob_rows["TRUE_BACKGROUND_CORRECT"]
    bridge = baseline_prob_rows["FALSE_BRIDGE_PIXELS"]
    improvement_50 = float(baseline_best_soft["all_iou_ge_0.50_rate"]) - float(baseline_hard["all_iou_ge_0.50_rate"])
    improvement_iou = float(baseline_best_soft["mean_matched_iou"]) - float(baseline_hard["mean_matched_iou"])
    critical_median = float(critical_missed["p_leaf_median"])
    critical_margin = float(critical_missed["leaflet_margin_median"])
    background_median = float(background["p_leaf_median"])
    bridge_margin = float(bridge["leaflet_margin_median"])
    bridge_leaf_median = float(bridge["p_leaf_median"])
    if improvement_50 >= 0.15 and improvement_iou > 0.05:
        return (
            "Soft semantic probabilities already recover a substantial part of the topology loss beyond hard argmax.",
            "A. SOFT_SEMANTIC_IS_SUFFICIENT",
            "The diagnostic masks materially improve locked reconstruction without any new learning, so future work should consume semantic probabilities directly.",
        )
    if critical_median >= 0.30 and critical_median > background_median + 0.10 and critical_margin > -0.20:
        return (
            "Many topology-critical missed leaflet pixels still carry usable leaflet probability, but bridge handling remains weak.",
            "B. BUILD_BOUNDARY_HEAD_WITH_SOFT_SEMANTIC",
            "Missing tissue looks downstream-recoverable from semantic logits, while false-bridge suppression still needs an explicit inference-time boundary signal.",
        )
    if critical_median >= 0.05 or critical_margin > -0.80:
        favored = "boundary prediction" if bridge_leaf_median >= 0.50 or bridge_margin > 0.20 else "endpoint/keypoint prediction"
        return (
            f"Hard-mask misses are only partially recoverable from logits and the evidence still points toward a dedicated {favored} route.",
            "C. BUILD_INSTANCE_KEYPOINT_OR_BOUNDARY_MODEL",
            f"The missed-topology pixels do not recover reliably enough from soft semantic probabilities alone, so the next model should add explicit geometric structure; current evidence favors {favored}.",
        )
    return (
        "Dominant missed topology pixels carry very weak leaflet probability and do not look downstream-recoverable.",
        "D. SEMANTIC_RETRAINING_REQUIRED",
        "The missing tissue is largely absent from semantic logits themselves, so downstream reconstruction alone is unlikely to fix it.",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    contract = build_audit_contract()
    _assert_safe_path(AUDIT_SPLIT)
    cfg = topo_aux._read_yaml(REPO_ROOT / "training" / "configs" / "unetpp_effb3_semantic_topology_aux_finetune_100ep.yaml")
    topology_contract = postrun._read_saved_topology_contract(REPO_ROOT / "training" / "runs" / "unetpp_effb3_semantic_topology_aux_finetune_100ep")
    dataset = postrun._build_dataset(cfg, AUDIT_SPLIT, topology_contract)
    device = postrun._resolve_device()
    use_amp = topo_aux._amp_enabled(cfg, device)

    checkpoint_meta: list[dict[str, Any]] = []
    probability_rows_all: list[dict[str, Any]] = []
    topology_category_rows: list[dict[str, Any]] = []
    threshold_curve_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    recovered_sample_rows: list[dict[str, Any]] = []
    per_checkpoint_probability_summary: dict[str, dict[str, dict[str, Any]]] = {}
    baseline_soft_cache: dict[str, Any] = {}

    for spec in CHECKPOINTS:
        model = topo_aux.build_model_from_cfg(cfg).to(device)
        meta = postrun._load_checkpoint_into_wrapper(model, spec.path)
        checkpoint_meta.append({"label": spec.label, "path": str(spec.path.resolve()), "sha256": str(meta["sha256"]), "epoch": meta["epoch"]})
        model.eval()

        category_values = {
            category: {"p_leaf": [], "p_competing": [], "leaflet_margin": []}
            for category in CATEGORY_ORDER
        }
        method_sample_rows: dict[str, list[dict[str, Any]]] = {
            "hard_argmax": [],
            **{f"threshold_{thr:.2f}": [] for thr in THRESHOLDS},
            "hysteresis_0.50_0.30": [],
            "margin_0.30_margin-0.10": [],
        }
        critical_recall_totals = {key: {"total": 0, "recovered": 0} for key in method_sample_rows}
        sample_cache: dict[str, Any] = {}

        with torch.no_grad():
            for idx in range(len(dataset)):
                sample = dataset[idx]
                sample_id = str(sample["sample_id"])
                image_t = sample["image"].unsqueeze(0).to(device)
                gt_sem_u8 = sample["mask"].numpy().astype(np.uint8)
                gt_leaf01 = (gt_sem_u8 == 1).astype(np.uint8)
                gt_inst_u8 = topo_aux._read_u8(Path(sample["instance_path"]))
                gt_inst_u8 = topo_aux._center_crop_like_validation(gt_inst_u8, gt_sem_u8.shape[0], gt_sem_u8.shape[1], is_mask=True)
                with topo_aux._autocast_ctx(device, enabled=use_amp):
                    outputs = model(image_t)
                soft = _softmax_probs(outputs["semantic_logits"])
                hard_union01 = (soft["pred_semantic"] == 1).astype(np.uint8)
                categories = _topology_pixel_categories(
                    gt_sem_u8=gt_sem_u8,
                    gt_inst_u8=gt_inst_u8,
                    pred_sem_u8=soft["pred_semantic"],
                    pred_union01=hard_union01,
                    topology_contract=topology_contract,
                )
                for category_name, mask01 in categories.items():
                    category_values[category_name]["p_leaf"].append(_masked_values(soft["p_leaf"], mask01))
                    category_values[category_name]["p_competing"].append(_masked_values(soft["p_competing"], mask01))
                    category_values[category_name]["leaflet_margin"].append(_masked_values(soft["leaflet_margin"], mask01))
                topology_category_rows.append(
                    {
                        "checkpoint": spec.label,
                        "sample_id": sample_id,
                        "total_pixels": int(gt_sem_u8.size),
                        **{f"{name.lower()}_count": int(np.sum(mask > 0)) for name, mask in categories.items()},
                    }
                )

                masks_by_method = {"hard_argmax": hard_union01}
                for thr in THRESHOLDS:
                    masks_by_method[f"threshold_{thr:.2f}"] = _threshold_mask(soft["p_leaf"], thr)
                masks_by_method["hysteresis_0.50_0.30"] = _hysteresis_leaflet_mask(
                    soft["p_leaf"],
                    high_threshold=HIGH_SEED_THRESHOLD,
                    candidate_threshold=CANDIDATE_THRESHOLD,
                )
                masks_by_method["margin_0.30_margin-0.10"] = _margin_mask(
                    soft["p_leaf"],
                    soft["p_competing"],
                    p_leaf_min=P_LEAF_MIN,
                    margin_min=MARGIN_MIN,
                )

                sample_cache[sample_id] = {
                    "image_path": str(sample["image_path"]),
                    "instance_path": str(sample["instance_path"]),
                    "gt_leaf01": gt_leaf01,
                    "gt_inst_u8": gt_inst_u8,
                    "hard_mask01": hard_union01,
                    "p_leaf": soft["p_leaf"],
                    "categories": categories,
                    "method_rows": {},
                    "method_masks": {},
                }

                for method, pred_mask01 in masks_by_method.items():
                    method_row = _sample_semantic_row(
                        sample_id=sample_id,
                        method=method,
                        checkpoint=spec.label,
                        gt_leaf01=gt_leaf01,
                        pred_leaf01=pred_mask01,
                        gt_inst_u8=gt_inst_u8,
                    )
                    crit = _missing_topology_critical_recall(categories["MISSING_TOPOLOGY_CRITICAL"], pred_mask01)
                    method_row.update(crit)
                    method_sample_rows[method].append(method_row)
                    per_sample_rows.append({k: v for k, v in method_row.items() if k != "normalized_instances"})
                    critical_recall_totals[method]["total"] += int(crit["critical_total"])
                    critical_recall_totals[method]["recovered"] += int(crit["critical_recovered"])
                    sample_cache[sample_id]["method_rows"][method] = method_row
                    sample_cache[sample_id]["method_masks"][method] = pred_mask01.astype(np.uint8)

        probability_rows = _category_distribution_rows(checkpoint=spec.label, category_values=category_values)
        probability_rows_all.extend(probability_rows)
        per_checkpoint_probability_summary[spec.label] = {row["category"]: row for row in probability_rows}

        for method, rows in method_sample_rows.items():
            agg = _aggregate_method_rows(rows)
            agg["checkpoint"] = spec.label
            agg["method"] = method
            total = int(critical_recall_totals[method]["total"])
            recovered = int(critical_recall_totals[method]["recovered"])
            agg["missing_topology_critical_recall"] = float(recovered / total) if total > 0 else 0.0
            agg["missing_topology_critical_total"] = total
            agg["missing_topology_critical_recovered"] = recovered
            threshold_curve_rows.append(agg)

        if spec.label == "baseline":
            baseline_soft_cache = sample_cache

    hard_baseline = next(row for row in threshold_curve_rows if row["checkpoint"] == "baseline" and row["method"] == "hard_argmax")
    candidate_soft_rows = [
        row for row in threshold_curve_rows
        if row["checkpoint"] == "baseline" and row["method"] != "hard_argmax"
    ]
    candidate_soft_rows.sort(
        key=lambda row: (
            float(row["all_iou_ge_0.50_rate"]),
            float(row["mean_matched_iou"]),
            -float(row["false_bridge_component_count"]),
        ),
        reverse=True,
    )
    best_soft = candidate_soft_rows[0]

    baseline_rows = [row for row in per_sample_rows if row["checkpoint"] == "baseline"]
    baseline_hard_failures = {
        str(row["sample_id"]): row
        for row in baseline_rows
        if row["method"] == "hard_argmax" and int(row["all_iou_ge_0.50_success"]) == 0
    }
    for sample_id, hard_row in sorted(baseline_hard_failures.items()):
        candidates = [
            row
            for row in baseline_rows
            if str(row["sample_id"]) == sample_id and row["method"] != "hard_argmax"
        ]
        recovered_candidates = [row for row in candidates if int(row["all_iou_ge_0.50_success"]) == 1]
        if not recovered_candidates:
            continue
        recovered_candidates.sort(
            key=lambda row: (
                float(row["mean_matched_iou"]),
                -float(row["bridge_component_count"]),
            ),
            reverse=True,
        )
        recovered_row = recovered_candidates[0]
        hard_method_row = baseline_soft_cache[sample_id]["method_rows"]["hard_argmax"]
        recovered_categories = baseline_soft_cache[sample_id]["categories"]
        hard_bridge = bool(hard_method_row["bridge_flag"])
        new_bridge = bool(recovered_row["bridge_flag"]) and (not hard_bridge)
        reason = "missing-foreground failure recovered"
        if hard_bridge and (not bool(recovered_row["bridge_flag"])):
            reason = "false-bridge failure recovered"
        elif new_bridge:
            reason = "new false-bridge failure introduced"
        recovered_sample_rows.append(
            {
                "sample_id": sample_id,
                "method": str(recovered_row["method"]),
                "gt_count": int(hard_row["gt_count"]),
                "reason": reason,
                "hard_mean_matched_iou": float(hard_row["mean_matched_iou"]),
                "soft_mean_matched_iou": float(recovered_row["mean_matched_iou"]),
                "hard_bridge_flag": int(hard_row["bridge_flag"]),
                "soft_bridge_flag": int(recovered_row["bridge_flag"]),
                "missing_topology_critical_pixels": int(np.sum(recovered_categories["MISSING_TOPOLOGY_CRITICAL"] > 0)),
            }
        )

    diagnosis, next_step, next_reason = _diagnosis_and_decision(
        baseline_hard=hard_baseline,
        baseline_best_soft=best_soft,
        baseline_prob_rows=per_checkpoint_probability_summary["baseline"],
    )

    visual_examples: dict[str, str] = {}
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

    def _pick_sample(predicate, sort_key=None) -> str | None:
        items = [sid for sid, cache in baseline_soft_cache.items() if predicate(sid, cache)]
        if not items:
            return None
        if sort_key is not None:
            items.sort(key=lambda sid: sort_key(sid, baseline_soft_cache[sid]), reverse=True)
        return str(items[0])

    selected_soft_method = str(best_soft["method"])
    picks: dict[str, dict[str, str] | None] = {
        "hard_missing_but_soft_high": (
            {"sample_id": sid, "method": selected_soft_method}
            if (sid := _pick_sample(
            lambda _sid, cache: int(np.sum((cache["categories"]["TRUE_LEAFLET_MISSED"] > 0) & (cache["p_leaf"] >= 0.30))) > 0,
            lambda _sid, cache: int(np.sum((cache["categories"]["TRUE_LEAFLET_MISSED"] > 0) & (cache["p_leaf"] >= 0.30))),
        )) else None
        ),
        "genuinely_unrecoverable_missing": (
            {"sample_id": sid, "method": selected_soft_method}
            if (sid := _pick_sample(
            lambda _sid, cache: int(np.sum((cache["categories"]["TRUE_LEAFLET_MISSED"] > 0) & (cache["p_leaf"] < 0.10))) > 0,
            lambda _sid, cache: int(np.sum((cache["categories"]["TRUE_LEAFLET_MISSED"] > 0) & (cache["p_leaf"] < 0.10))),
        )) else None
        ),
        "false_bridge_weak_leaflet_probability": (
            {"sample_id": sid, "method": selected_soft_method}
            if (sid := _pick_sample(
            lambda _sid, cache: int(np.sum((cache["categories"]["FALSE_BRIDGE_PIXELS"] > 0) & (cache["p_leaf"] < 0.30))) > 0,
            lambda _sid, cache: int(np.sum((cache["categories"]["FALSE_BRIDGE_PIXELS"] > 0) & (cache["p_leaf"] < 0.30))),
        )) else None
        ),
        "false_bridge_strong_leaflet_probability": (
            {"sample_id": sid, "method": selected_soft_method}
            if (sid := _pick_sample(
            lambda _sid, cache: int(np.sum((cache["categories"]["FALSE_BRIDGE_PIXELS"] > 0) & (cache["p_leaf"] >= 0.50))) > 0,
            lambda _sid, cache: int(np.sum((cache["categories"]["FALSE_BRIDGE_PIXELS"] > 0) & (cache["p_leaf"] >= 0.50))),
        )) else None
        ),
        "gt2_recovered_by_soft_mask": next(
            ({"sample_id": str(row["sample_id"]), "method": str(row["method"])} for row in recovered_sample_rows if int(row["gt_count"]) == 2),
            None,
        ),
        "gt3_recovered_by_soft_mask": next(
            ({"sample_id": str(row["sample_id"]), "method": str(row["method"])} for row in recovered_sample_rows if int(row["gt_count"]) == 3),
            None,
        ),
        "soft_mask_introduces_new_bridge": next(
            ({"sample_id": str(row["sample_id"]), "method": str(row["method"])} for row in recovered_sample_rows if row["reason"] == "new false-bridge failure introduced"),
            None,
        ),
    }
    for category, pick in picks.items():
        if not pick:
            continue
        sample_id = str(pick["sample_id"])
        method = str(pick["method"])
        cache = baseline_soft_cache[sample_id]
        method_row = cache["method_rows"][method]
        image_rgb = topo_aux._center_crop_like_validation(topo_aux._read_image_rgb(Path(cache["image_path"])), cache["gt_leaf01"].shape[0], cache["gt_leaf01"].shape[1], is_mask=False)
        visual_examples[category] = _save_visual(
            sample_id=str(sample_id),
            category=category,
            image_rgb=image_rgb,
            gt_leaf01=cache["gt_leaf01"],
            hard_mask01=cache["hard_mask01"],
            p_leaf=cache["p_leaf"],
            soft_mask01=cache["method_masks"][method],
            normalized_instances=method_row["normalized_instances"].astype(np.uint8),
            gt_instances=cache["gt_inst_u8"].astype(np.uint8),
        )

    summary = {
        "contract": contract,
        "checkpoints": checkpoint_meta,
        "probability_distributions": per_checkpoint_probability_summary,
        "hard_argmax_baseline": hard_baseline,
        "best_soft_baseline": best_soft,
        "diagnosis": diagnosis,
        "next_step": {
            "decision": next_step,
            "reason": next_reason,
        },
        "recovered_sample_count": len(recovered_sample_rows),
        "visual_examples": visual_examples,
    }

    _write_json(OUTPUT_DIR / "audit_summary.json", summary)
    _write_csv(
        OUTPUT_DIR / "pixel_probability_distributions.csv",
        probability_rows_all,
        [
            "checkpoint",
            "category",
            "pixel_count",
            "p_leaf_count",
            "p_leaf_mean",
            "p_leaf_median",
            "p_leaf_p10",
            "p_leaf_p25",
            "p_leaf_p75",
            "p_leaf_p90",
            "p_competing_count",
            "p_competing_mean",
            "p_competing_median",
            "p_competing_p10",
            "p_competing_p25",
            "p_competing_p75",
            "p_competing_p90",
            "leaflet_margin_count",
            "leaflet_margin_mean",
            "leaflet_margin_median",
            "leaflet_margin_p10",
            "leaflet_margin_p25",
            "leaflet_margin_p75",
            "leaflet_margin_p90",
        ],
    )
    _write_csv(
        OUTPUT_DIR / "threshold_curve.csv",
        threshold_curve_rows,
        [
            "checkpoint",
            "method",
            "n",
            "leaflet_dice",
            "leaflet_iou",
            "mean_connected_components",
            "median_connected_components",
            "missing_topology_critical_recall",
            "missing_topology_critical_total",
            "missing_topology_critical_recovered",
            "false_positive_leaflet_area_mean",
            "false_bridge_component_count",
            "exact_k_count",
            "exact_k_rate",
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
        ],
    )
    _write_csv(
        OUTPUT_DIR / "per_sample_results.csv",
        per_sample_rows,
        [
            "checkpoint",
            "method",
            "sample_id",
            "gt_count",
            "leaflet_dice",
            "leaflet_iou",
            "connected_components",
            "critical_total",
            "critical_recovered",
            "critical_recall",
            "false_positive_leaflet_area",
            "bridge_component_count",
            "exact_k_success",
            "mean_matched_iou",
            "median_matched_iou",
            "all_iou_ge_0.50_success",
            "all_iou_ge_0.70_success",
            "topology_class",
            "bridge_flag",
            "missing_flag",
        ],
    )
    _write_csv(
        OUTPUT_DIR / "topology_pixel_categories.csv",
        topology_category_rows,
        [
            "checkpoint",
            "sample_id",
            "total_pixels",
            "true_leaflet_correct_count",
            "true_leaflet_missed_count",
            "true_background_correct_count",
            "false_leaflet_count",
            "missing_topology_critical_count",
            "false_bridge_pixels_count",
        ],
    )
    _write_csv(
        OUTPUT_DIR / "recovered_samples.csv",
        recovered_sample_rows,
        [
            "sample_id",
            "method",
            "gt_count",
            "reason",
            "hard_mean_matched_iou",
            "soft_mean_matched_iou",
            "hard_bridge_flag",
            "soft_bridge_flag",
            "missing_topology_critical_pixels",
        ],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
