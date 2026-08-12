from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e

import leaflet_oracle_count_geometric_split_audit as base_audit
import leaflet_oracle_count_geometric_split_forensic as forensic
import leaflet_oracle_k_constrained_normalization_audit as k_audit
import semantic_topology_aux as topo_aux
from metrics import compute_per_class_metrics_from_logits


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "training" / "analysis" / "semantic_topology_aux_postrun_eval"
VISUAL_DIR = OUTPUT_DIR / "visual_review"
NORMALIZER_METHOD = "centroid_distance_k_normalizer"
RESEARCH_EVAL_NAME = "semantic_topology_research_eval"
RESEARCH_EVAL_SPLIT = REPO_ROOT / "datasets" / "converted_full_multiclass_curated" / "test.txt"
SEMANTIC_VAL_SPLIT = REPO_ROOT / "datasets" / "converted_full_multiclass_curated" / "val.txt"
PROHIBITED_PATH_SUBSTRINGS = ("center_full_val_manifest.jsonl", "authoritative_106_holdout", "holdout")
CHECKPOINT_ORDER = ("baseline", "topology_best_semantic", "topology_last_diagnostic")


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path
    semantic_only_base: bool
    diagnostic_only: bool


CHECKPOINTS: tuple[CheckpointSpec, ...] = (
    CheckpointSpec(
        label="baseline",
        path=REPO_ROOT / "training" / "runs" / "unetpp_effb3_a100_multiclass_curated_finetune_stage2_lr1e5_100ep" / "best_mean_fg.pth",
        semantic_only_base=True,
        diagnostic_only=False,
    ),
    CheckpointSpec(
        label="topology_best_semantic",
        path=REPO_ROOT / "training" / "runs" / "unetpp_effb3_semantic_topology_aux_finetune_100ep" / "best_mean_fg.pth",
        semantic_only_base=False,
        diagnostic_only=False,
    ),
    CheckpointSpec(
        label="topology_last_diagnostic",
        path=REPO_ROOT / "training" / "runs" / "unetpp_effb3_semantic_topology_aux_finetune_100ep" / "last.pth",
        semantic_only_base=False,
        diagnostic_only=True,
    ),
)


def _assert_safe_path(path: Path) -> None:
    text = str(path).replace("\\", "/").lower()
    for token in PROHIBITED_PATH_SUBSTRINGS:
        if token.lower() in text:
            raise SystemExit(f"Prohibited path detected in post-run evaluation: {path}")


def build_eval_contract() -> dict[str, Any]:
    return {
        "research_eval_name": RESEARCH_EVAL_NAME,
        "research_eval_split": str(RESEARCH_EVAL_SPLIT.resolve()),
        "semantic_val_split": str(SEMANTIC_VAL_SPLIT.resolve()),
        "normalizer_method": NORMALIZER_METHOD,
        "holdout_used": False,
        "checkpoint_selection_from_research_eval": False,
        "center_full_val_manifest_used": False,
        "checkpoint_order": list(CHECKPOINT_ORDER),
    }


def _read_saved_topology_contract(run_dir: Path) -> topo_aux.TopologyTargetContract:
    payload = json.loads((run_dir / "topology_target_contract.json").read_text(encoding="utf-8"))
    return topo_aux.TopologyTargetContract(
        separation_radius_px=int(payload["separation_radius_px"]),
        narrow_width_threshold_px=int(payload["narrow_width_threshold_px"]),
        include_foreground_boundary=bool(payload["include_foreground_boundary"]),
        source_split_txt=str(topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT.resolve()),
        source_instance_root=str(topo_aux.DEFAULT_INSTANCE_ROOT.resolve()),
        selection_rule=str(payload["selection_rule"]),
        train_only=bool(payload.get("train_only", True)),
    )


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_dataset(cfg: dict[str, Any], split_txt: Path, contract: topo_aux.TopologyTargetContract) -> topo_aux.SemanticTopologyAuxDataset:
    dataset_cfg = cfg.get("dataset") or {}
    dataset_root = topo_aux._resolve_repo_path(dataset_cfg.get("root", topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT), topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT)
    instance_root = topo_aux._resolve_repo_path(dataset_cfg.get("instance_root", topo_aux.DEFAULT_INSTANCE_ROOT), topo_aux.DEFAULT_INSTANCE_ROOT)
    _assert_safe_path(split_txt)
    _assert_safe_path(dataset_root)
    _assert_safe_path(instance_root)
    return topo_aux.SemanticTopologyAuxDataset(
        dataset_root=dataset_root,
        split_txt=split_txt.resolve(),
        instance_root=instance_root,
        contract=contract,
        num_classes=int((cfg.get("model") or {})["classes"]),
        input_size=int((cfg.get("model") or {})["input_size"]),
        augment_cfg=cfg.get("augment"),
        training=False,
    )


def _build_loader(dataset, batch_size: int, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=False,
        drop_last=False,
    )


def _load_checkpoint_into_wrapper(model: topo_aux.UnetPlusPlusSemanticTopologyAux, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise SystemExit(f"Unsupported checkpoint format: {checkpoint_path}")
    state_keys = list(state.keys())
    if any(key.startswith("base.") or key.startswith("topology_head.") for key in state_keys):
        incompat = model.load_state_dict(state, strict=True)
    else:
        incompat = model.base.load_state_dict(state, strict=True)
    missing = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
    unexpected = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint incompatibility for {checkpoint_path}: missing={missing[:5]} unexpected={unexpected[:5]}")
    extra = ckpt.get("extra", {}) if isinstance(ckpt, dict) else {}
    return {
        "path": str(checkpoint_path),
        "sha256": topo_aux._sha256_file(checkpoint_path),
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "extra": extra,
    }


def _safe_rate(numerator: float, denominator: float) -> float:
    if float(denominator) <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _binary_precision_recall_dice(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = _safe_rate(tp, tp + fp)
    recall = _safe_rate(tp, tp + fn)
    dice = _safe_rate(2 * tp, 2 * tp + fp + fn)
    return precision, recall, dice


def _semantic_summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "leaflet_dice": 0.0,
            "leaflet_iou": 0.0,
            "ring_dice": 0.0,
            "ring_iou": 0.0,
            "mean_dice_fg": 0.0,
            "mean_iou_fg": 0.0,
        }
    return {
        "n": int(len(rows)),
        "leaflet_dice": float(np.mean([float(r["leaflet_dice"]) for r in rows])),
        "leaflet_iou": float(np.mean([float(r["leaflet_iou"]) for r in rows])),
        "ring_dice": float(np.mean([float(r["ring_dice"]) for r in rows])),
        "ring_iou": float(np.mean([float(r["ring_iou"]) for r in rows])),
        "mean_dice_fg": float(np.mean([float(r["mean_dice_fg"]) for r in rows])),
        "mean_iou_fg": float(np.mean([float(r["mean_iou_fg"]) for r in rows])),
    }


def _evaluate_semantic_rows(
    model: topo_aux.UnetPlusPlusSemanticTopologyAux,
    loader: DataLoader,
    *,
    device: torch.device,
    use_amp: bool,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with topo_aux._autocast_ctx(device, enabled=use_amp):
                outputs = model(images)
            metrics = compute_per_class_metrics_from_logits(outputs["semantic_logits"], masks, num_classes=int(outputs["semantic_logits"].shape[1]))
            sample_ids = list(batch["sample_id"])
            for idx, sample_id in enumerate(sample_ids):
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "leaflet_dice": float(metrics.dice[1]),
                        "leaflet_iou": float(metrics.iou[1]),
                        "ring_dice": float(metrics.dice[2]),
                        "ring_iou": float(metrics.iou[2]),
                        "mean_dice_fg": float((float(metrics.dice[1]) + float(metrics.dice[2])) / 2.0),
                        "mean_iou_fg": float((float(metrics.iou[1]) + float(metrics.iou[2])) / 2.0),
                    }
                )
    return rows


def _evaluate_auxiliary_head(
    model: topo_aux.UnetPlusPlusSemanticTopologyAux,
    loader: DataLoader,
    *,
    device: torch.device,
    use_amp: bool,
) -> list[dict[str, Any]]:
    model.eval()
    names = ["critical_foreground", "inter_instance_separation"]
    total_pixels = [0, 0]
    total_positive = [0, 0]
    total_bce = [0.0, 0.0]
    total_tp = [0, 0]
    total_fp = [0, 0]
    total_fn = [0, 0]
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["topology_target"].to(device, non_blocking=True)
            with topo_aux._autocast_ctx(device, enabled=use_amp):
                outputs = model(images)
                logits = outputs["topology_logits"]
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).to(dtype=torch.int64)
            target_i64 = (targets >= 0.5).to(dtype=torch.int64)
            for channel in range(2):
                logit_ch = logits[:, channel : channel + 1]
                target_ch = targets[:, channel : channel + 1]
                pred_ch = preds[:, channel : channel + 1]
                target_bin = target_i64[:, channel : channel + 1]
                total_pixels[channel] += int(target_ch.numel())
                total_positive[channel] += int(torch.sum(target_bin).item())
                total_bce[channel] += float(F.binary_cross_entropy_with_logits(logit_ch, target_ch, reduction="sum").item())
                total_tp[channel] += int(torch.sum((pred_ch == 1) & (target_bin == 1)).item())
                total_fp[channel] += int(torch.sum((pred_ch == 1) & (target_bin == 0)).item())
                total_fn[channel] += int(torch.sum((pred_ch == 0) & (target_bin == 1)).item())
    rows: list[dict[str, Any]] = []
    for channel, name in enumerate(names):
        precision, recall, dice = _binary_precision_recall_dice(total_tp[channel], total_fp[channel], total_fn[channel])
        rows.append(
            {
                "channel": int(channel),
                "name": name,
                "target_positive_fraction": _safe_rate(total_positive[channel], total_pixels[channel]),
                "bce": _safe_rate(total_bce[channel], total_pixels[channel]),
                "dice": float(dice),
                "precision": float(precision),
                "recall": float(recall),
                "tp": int(total_tp[channel]),
                "fp": int(total_fp[channel]),
                "fn": int(total_fn[channel]),
            }
        )
    return rows


def run_locked_normalization(pred_union01: np.ndarray, gt_k: int) -> dict[str, Any]:
    return k_audit.normalize_mask_exact_k(pred_union01.astype(np.uint8), int(gt_k), NORMALIZER_METHOD)


def _sample_semantic_metrics(pred_sem_u8: np.ndarray, gt_sem_u8: np.ndarray) -> dict[str, float]:
    pred_t = torch.from_numpy(pred_sem_u8[None, ...].astype(np.int64))
    gt_t = torch.from_numpy(gt_sem_u8[None, ...].astype(np.int64))
    num_classes = int(max(int(pred_sem_u8.max()), int(gt_sem_u8.max()), 2) + 1)
    logits = F.one_hot(pred_t, num_classes=num_classes).permute(0, 3, 1, 2).float()
    metrics = compute_per_class_metrics_from_logits(logits, gt_t, num_classes=num_classes)
    return {
        "leaflet_dice": float(metrics.dice[1]),
        "leaflet_iou": float(metrics.iou[1]),
        "ring_dice": float(metrics.dice[2]),
        "ring_iou": float(metrics.iou[2]),
        "mean_dice_fg": float((float(metrics.dice[1]) + float(metrics.dice[2])) / 2.0),
        "mean_iou_fg": float((float(metrics.iou[1]) + float(metrics.iou[2])) / 2.0),
    }


def _sample_topology_failure(topology: dict[str, Any], reconstruction_success: bool) -> str:
    if reconstruction_success:
        return "none"
    topo_class = str(topology["topology_class"])
    if topo_class == "B":
        return "false_semantic_bridges"
    if topo_class == "C":
        return "missing_semantic_pixels"
    if topo_class == "D":
        return "both"
    if topo_class == "E":
        return "ambiguous_geometry"
    return "other"


def _aggregate_research_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "semantic": _semantic_summary_from_rows([]),
            "instance": {
                "exact_k_count": 0,
                "exact_k_rate": 0.0,
                "mean_matched_iou": 0.0,
                "median_matched_iou": 0.0,
                "all_iou_ge_0.50_count": 0,
                "all_iou_ge_0.50_rate": 0.0,
                "all_iou_ge_0.70_count": 0,
                "all_iou_ge_0.70_rate": 0.0,
                "all_iou_ge_0.80_count": 0,
                "all_iou_ge_0.80_rate": 0.0,
            },
            "topology_failures": {},
        }
    semantic = _semantic_summary_from_rows(rows)
    n = int(len(rows))
    exact_k_count = int(sum(int(r["exact_k_success"]) for r in rows))
    succ50 = int(sum(int(r["all_iou_ge_0.50_success"]) for r in rows))
    succ70 = int(sum(int(r["all_iou_ge_0.70_success"]) for r in rows))
    succ80 = int(sum(int(r["all_iou_ge_0.80_success"]) for r in rows))
    failure_names = ["missing_semantic_pixels", "false_semantic_bridges", "both", "ambiguous_geometry", "other"]
    failure_summary = {}
    for name in failure_names:
        count = int(sum(1 for r in rows if str(r["topology_failure"]) == name))
        failure_summary[name] = {"count": count, "rate": _safe_rate(count, n)}
    return {
        "n": n,
        "semantic": semantic,
        "instance": {
            "exact_k_count": exact_k_count,
            "exact_k_rate": _safe_rate(exact_k_count, n),
            "mean_matched_iou": float(np.mean([float(r["mean_matched_iou"]) for r in rows])),
            "median_matched_iou": float(np.mean([float(r["median_matched_iou"]) for r in rows])),
            "all_iou_ge_0.50_count": succ50,
            "all_iou_ge_0.50_rate": _safe_rate(succ50, n),
            "all_iou_ge_0.70_count": succ70,
            "all_iou_ge_0.70_rate": _safe_rate(succ70, n),
            "all_iou_ge_0.80_count": succ80,
            "all_iou_ge_0.80_rate": _safe_rate(succ80, n),
        },
        "topology_failures": failure_summary,
    }


def _gt_bucket_summary(rows: list[dict[str, Any]], gt_count: int) -> dict[str, Any]:
    subset = [row for row in rows if int(row["gt_count"]) == int(gt_count)]
    n = int(len(subset))
    success50 = int(sum(int(row["all_iou_ge_0.50_success"]) for row in subset))
    success70 = int(sum(int(row["all_iou_ge_0.70_success"]) for row in subset))
    success80 = int(sum(int(row["all_iou_ge_0.80_success"]) for row in subset))
    return {
        "gt_count": int(gt_count),
        "n": n,
        "all_iou_ge_0.50_count": success50,
        "all_iou_ge_0.50_rate": _safe_rate(success50, n),
        "all_iou_ge_0.70_count": success70,
        "all_iou_ge_0.70_rate": _safe_rate(success70, n),
        "all_iou_ge_0.80_count": success80,
        "all_iou_ge_0.80_rate": _safe_rate(success80, n),
        "mean_matched_iou": float(np.mean([float(r["mean_matched_iou"]) for r in subset])) if subset else 0.0,
    }


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


def _binary_mask_rgb(mask01: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(mask01.shape + (3,), dtype=np.uint8)
    out[mask01.astype(bool)] = np.asarray(color, dtype=np.uint8)
    return out


def _save_visual(sample_row_a: dict[str, Any], sample_row_b: dict[str, Any], category: str) -> str:
    image_rgb = topo_aux._center_crop_like_validation(topo_aux._read_image_rgb(Path(sample_row_a["image_path"])), 768, 768, is_mask=False)
    gt_inst = topo_aux._center_crop_like_validation(topo_aux._read_u8(Path(sample_row_a["instance_path"])), 768, 768, is_mask=True)
    base_sem = sample_row_a["pred_semantic"]
    topo_sem = sample_row_b["pred_semantic"]
    base_inst = sample_row_a["normalized_instances"]
    topo_inst = sample_row_b["normalized_instances"]
    panels = [
        ("RGB", image_rgb),
        ("GT Instances", _instance_rgb(gt_inst)),
        ("Baseline Semantic", _binary_mask_rgb(base_sem == 1, (255, 255, 255))),
        ("Topology-best Semantic", _binary_mask_rgb(topo_sem == 1, (255, 255, 255))),
        ("Baseline K-normalized", _instance_rgb(base_inst)),
        ("Topology-best K-normalized", _instance_rgb(topo_inst)),
    ]
    titled: list[np.ndarray] = []
    for title, panel in panels:
        canvas = np.full((panel.shape[0] + 36, panel.shape[1], 3), 18, dtype=np.uint8)
        canvas[36:, :, :] = panel
        cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        titled.append(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    row_top = np.concatenate(titled[:3], axis=1)
    row_bottom = np.concatenate(titled[3:], axis=1)
    grid = np.concatenate([row_top, row_bottom], axis=0)
    out_path = VISUAL_DIR / f"{category}_{sample_row_a['sample_id']}.png"
    cv2.imwrite(str(out_path), grid)
    return str(out_path.resolve())


def _change_category(before: dict[str, Any], after: dict[str, Any]) -> str:
    before_missing = bool(before["topology"]["missing"])
    before_bridge = bool(before["topology"]["bridge"])
    after_missing = bool(after["topology"]["missing"])
    after_bridge = bool(after["topology"]["bridge"])
    if before_missing and not after_missing and before_bridge and not after_bridge:
        return "both"
    if before_missing and not after_missing:
        return "missing-foreground recovery"
    if before_bridge and not after_bridge:
        return "false-bridge removal"
    if before["topology_failure"] == after["topology_failure"] and float(after["mean_matched_iou"]) > float(before["mean_matched_iou"]):
        return "changed split geometry"
    return "other"


def _summarize_sample_changes(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id_a = {str(row["sample_id"]): row for row in rows_a}
    by_id_b = {str(row["sample_id"]): row for row in rows_b}
    out_rows: list[dict[str, Any]] = []
    fail_to_pass = 0
    pass_to_fail = 0
    unchanged_pass = 0
    unchanged_fail = 0
    for sample_id in sorted(by_id_a):
        a = by_id_a[sample_id]
        b = by_id_b[sample_id]
        a_pass = bool(a["all_iou_ge_0.50_success"])
        b_pass = bool(b["all_iou_ge_0.50_success"])
        if (not a_pass) and b_pass:
            status = "fail_to_pass"
            fail_to_pass += 1
            reason = _change_category(a, b)
        elif a_pass and (not b_pass):
            status = "pass_to_fail"
            pass_to_fail += 1
            reason = _change_category(a, b)
        elif a_pass and b_pass:
            status = "unchanged_pass"
            unchanged_pass += 1
            reason = "other"
        else:
            status = "unchanged_fail"
            unchanged_fail += 1
            reason = _change_category(a, b) if float(b["mean_matched_iou"]) != float(a["mean_matched_iou"]) else "other"
        out_rows.append(
            {
                "sample_id": sample_id,
                "gt_count": int(a["gt_count"]),
                "status": status,
                "reason": reason,
                "baseline_all_iou_ge_0.50": int(a["all_iou_ge_0.50_success"]),
                "topology_best_all_iou_ge_0.50": int(b["all_iou_ge_0.50_success"]),
                "baseline_mean_matched_iou": float(a["mean_matched_iou"]),
                "topology_best_mean_matched_iou": float(b["mean_matched_iou"]),
                "baseline_failure": str(a["topology_failure"]),
                "topology_best_failure": str(b["topology_failure"]),
            }
        )
    return out_rows, {
        "fail_to_pass": fail_to_pass,
        "pass_to_fail": pass_to_fail,
        "unchanged_pass": unchanged_pass,
        "unchanged_fail": unchanged_fail,
    }


def _pick_visual_examples(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], sample_changes: list[dict[str, Any]]) -> dict[str, str]:
    by_id_a = {str(row["sample_id"]): row for row in rows_a}
    by_id_b = {str(row["sample_id"]): row for row in rows_b}
    saved: dict[str, str] = {}

    def _select(predicate) -> str | None:
        candidates = [change for change in sample_changes if predicate(change)]
        if not candidates:
            return None
        candidates.sort(key=lambda row: float(row["topology_best_mean_matched_iou"]) - float(row["baseline_mean_matched_iou"]), reverse=True)
        return str(candidates[0]["sample_id"])

    picks = {
        "baseline_fail_to_topology_pass": _select(lambda row: row["status"] == "fail_to_pass"),
        "gt2_improvement": _select(lambda row: int(row["gt_count"]) == 2 and float(row["topology_best_mean_matched_iou"]) > float(row["baseline_mean_matched_iou"])),
        "gt3_improvement": _select(lambda row: int(row["gt_count"]) == 3 and float(row["topology_best_mean_matched_iou"]) > float(row["baseline_mean_matched_iou"])),
        "missing_foreground_recovery": _select(lambda row: row["reason"] == "missing-foreground recovery"),
        "false_bridge_removal": _select(lambda row: row["reason"] == "false-bridge removal"),
        "baseline_pass_to_topology_fail": _select(lambda row: row["status"] == "pass_to_fail"),
        "no_meaningful_change": _select(lambda row: row["status"] in {"unchanged_pass", "unchanged_fail"} and abs(float(row["topology_best_mean_matched_iou"]) - float(row["baseline_mean_matched_iou"])) < 1e-6),
    }
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    for category, sample_id in picks.items():
        if sample_id is None:
            continue
        saved[category] = _save_visual(by_id_a[sample_id], by_id_b[sample_id], category)
    return saved


def _primary_classification(delta: dict[str, Any], baseline: dict[str, Any], topology_best: dict[str, Any]) -> str:
    recon_delta = float(delta["all_iou_ge_0.50_rate"])
    semantic_delta = float(delta["mean_dice_fg"])
    gt2_delta = float(delta["gt2_all_iou_ge_0.50_rate"])
    gt3_delta = float(delta["gt3_all_iou_ge_0.50_rate"])
    if recon_delta >= 0.15 and semantic_delta > -0.01:
        return "STRONG_SIGNAL"
    if recon_delta >= 0.08 and gt2_delta >= 0.0 and gt3_delta >= 0.0:
        return "PROMISING_SIGNAL"
    if (
        recon_delta > 0.0
        or float(delta["mean_matched_iou"]) > 0.0
        or int(delta["missing_failures"]) != 0
        or int(delta["bridge_failures"]) != 0
    ):
        return "WEAK_SIGNAL"
    return "NEGATIVE_SIGNAL"


def _next_step_decision(classification: str, aux_rows: list[dict[str, Any]]) -> tuple[str, str]:
    aux_learnable = bool(aux_rows) and all(
        (float(row["recall"]) >= 0.05) and (float(row["dice"]) >= 0.01)
        for row in aux_rows
    )
    if classification in {"STRONG_SIGNAL", "PROMISING_SIGNAL"}:
        return "A. BUILD_COUNT_CLASSIFIER", "Primary A vs B comparison shows topology supervision improved downstream reconstruction enough to justify the next count-head experiment."
    if classification == "WEAK_SIGNAL" and aux_learnable:
        return "B. REFINE_TOPOLOGY_SUPERVISION", "There is a small but reproducible signal and the auxiliary channels are not dead, so the next step is to refine the supervision rather than abandon the route."
    if classification == "WEAK_SIGNAL":
        return "C. BUILD_BOUNDARY_OR_KEYPOINT_HEAD", "The locked reconstruction pipeline shows only a weak gain, and the current auxiliary channels do not both look learnable enough to justify iterating this exact supervision design."
    if classification == "NEGATIVE_SIGNAL":
        return "D. RETURN_TO_CENTER_BASELINE", "The topology-semantic route did not improve the locked reconstruction benchmark enough to justify continuing it as the main line."
    return "C. BUILD_BOUNDARY_OR_KEYPOINT_HEAD", "The geometry route still looks attractive, but this auxiliary semantic head did not change topology enough on its own."


def _last_checkpoint_diagnostic(baseline: dict[str, Any], best: dict[str, Any], last: dict[str, Any]) -> tuple[str, str]:
    recon_gain_vs_best = float(last["instance"]["all_iou_ge_0.50_rate"]) - float(best["instance"]["all_iou_ge_0.50_rate"])
    semantic_cost_vs_best = float(last["semantic"]["mean_dice_fg"]) - float(best["semantic"]["mean_dice_fg"])
    mean_iou_gain = float(last["instance"]["mean_matched_iou"]) - float(best["instance"]["mean_matched_iou"])
    if recon_gain_vs_best <= 0.0 and mean_iou_gain <= 0.0 and semantic_cost_vs_best <= 0.0:
        return "late_regression", "Last checkpoint is worse than topology-best on both semantic quality and locked reconstruction."
    if recon_gain_vs_best <= 0.0 and mean_iou_gain <= 0.0:
        return "no_late_topology_gain", "Last checkpoint does not improve the locked reconstruction metrics over topology-best."
    if (recon_gain_vs_best > 0.0 or mean_iou_gain > 0.0) and semantic_cost_vs_best >= -0.01:
        return "late_topology_gain_with_small_semantic_cost", "Locked reconstruction improved after semantic peak while semantic mean_dice_fg drift stayed small."
    if (recon_gain_vs_best > 0.0 or mean_iou_gain > 0.0) and semantic_cost_vs_best < -0.01:
        return "late_topology_gain_with_large_semantic_cost", "Locked reconstruction improved after semantic peak, but the semantic validation drop is no longer small."
    return "mixed", "Late checkpoint movement is not cleanly one-sided across semantic and reconstruction metrics."


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


def _research_eval_rows(
    model: topo_aux.UnetPlusPlusSemanticTopologyAux,
    dataset: topo_aux.SemanticTopologyAuxDataset,
    *,
    device: torch.device,
    use_amp: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image_t = sample["image"].unsqueeze(0).to(device)
            gt_sem = sample["mask"].numpy().astype(np.uint8)
            gt_inst = topo_aux._read_u8(Path(sample["instance_path"]))
            gt_inst = topo_aux._center_crop_like_validation(gt_inst, gt_sem.shape[0], gt_sem.shape[1], is_mask=True)
            gt_k = int(len(topo_aux._positive_instance_ids(gt_inst)))
            with topo_aux._autocast_ctx(device, enabled=use_amp):
                outputs = model(image_t)
            pred_sem = torch.argmax(outputs["semantic_logits"], dim=1)[0].detach().cpu().numpy().astype(np.uint8)
            pred_union = (pred_sem == 1).astype(np.uint8)
            normalized = run_locked_normalization(pred_union, gt_k)
            pred_inst = normalized["labels"].astype(np.uint8)
            metrics = base_audit.compute_detailed_instance_metrics(gt_inst, pred_inst, gt_k=gt_k, pred_k=int(normalized["final_group_count"]))
            topology = forensic.classify_semantic_topology(gt_inst, pred_union)
            semantic = _sample_semantic_metrics(pred_sem, gt_sem)
            rows.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "image_path": str(sample["image_path"]),
                    "instance_path": str(sample["instance_path"]),
                    "gt_count": gt_k,
                    **semantic,
                    "exact_k_success": int(bool(metrics["instance_exact_count_acc"])),
                    "mean_matched_iou": float(metrics["instance_mean_matched_iou"]),
                    "median_matched_iou": float(metrics["median_matched_iou"]),
                    "all_iou_ge_0.50_success": int(bool(metrics["all_iou_ge_0.50"])),
                    "all_iou_ge_0.70_success": int(bool(metrics["all_iou_ge_0.70"])),
                    "all_iou_ge_0.80_success": int(bool(metrics["all_iou_ge_0.80"])),
                    "matched_iou_per_gt": [float(v) for v in metrics["matched_iou_per_gt"]],
                    "topology_failure": _sample_topology_failure(topology, bool(metrics["all_iou_ge_0.50"])),
                    "topology": topology,
                    "pred_semantic": pred_sem,
                    "normalized_instances": pred_inst,
                }
            )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    contract = build_eval_contract()
    _assert_safe_path(RESEARCH_EVAL_SPLIT)
    _assert_safe_path(SEMANTIC_VAL_SPLIT)
    cfg = topo_aux._read_yaml(REPO_ROOT / "training" / "configs" / "unetpp_effb3_semantic_topology_aux_finetune_100ep.yaml")
    topology_run_dir = REPO_ROOT / "training" / "runs" / "unetpp_effb3_semantic_topology_aux_finetune_100ep"
    topology_contract = _read_saved_topology_contract(topology_run_dir)
    device = _resolve_device()
    use_amp = topo_aux._amp_enabled(cfg, device)

    val_dataset = _build_dataset(cfg, SEMANTIC_VAL_SPLIT, topology_contract)
    test_dataset = _build_dataset(cfg, RESEARCH_EVAL_SPLIT, topology_contract)
    val_loader = _build_loader(val_dataset, batch_size=int((cfg.get("train") or {}).get("batch_size", 16)), num_workers=0)
    aux_loader = _build_loader(val_dataset, batch_size=1, num_workers=0)

    checkpoint_identities: list[dict[str, Any]] = []
    semantic_val_rows: list[dict[str, Any]] = []
    research_eval_rows_all: list[dict[str, Any]] = []
    gt_count_rows: list[dict[str, Any]] = []
    aux_rows_all: list[dict[str, Any]] = []
    research_eval_payloads: dict[str, dict[str, Any]] = {}
    per_checkpoint_sample_rows: dict[str, list[dict[str, Any]]] = {}

    for spec in CHECKPOINTS:
        if not spec.path.exists():
            raise SystemExit(f"Checkpoint not found: {spec.path}")
        model = topo_aux.build_model_from_cfg(cfg).to(device)
        ckpt_meta = _load_checkpoint_into_wrapper(model, spec.path)
        checkpoint_identities.append(
            {
                "label": spec.label,
                "path": str(spec.path.resolve()),
                "sha256": str(ckpt_meta["sha256"]),
                "epoch": ckpt_meta["epoch"],
                "diagnostic_only": bool(spec.diagnostic_only),
            }
        )
        semantic_rows = _evaluate_semantic_rows(model, val_loader, device=device, use_amp=use_amp)
        semantic_summary = _semantic_summary_from_rows(semantic_rows)
        semantic_val_rows.append({"checkpoint": spec.label, **semantic_summary})
        research_rows = _research_eval_rows(model, test_dataset, device=device, use_amp=use_amp)
        per_checkpoint_sample_rows[spec.label] = research_rows
        research_summary = _aggregate_research_rows(research_rows)
        research_eval_payloads[spec.label] = research_summary
        research_eval_rows_all.append(
            {
                "checkpoint": spec.label,
                "n": research_summary["n"],
                **research_summary["semantic"],
                **research_summary["instance"],
                "missing_semantic_pixels_count": research_summary["topology_failures"]["missing_semantic_pixels"]["count"],
                "false_semantic_bridges_count": research_summary["topology_failures"]["false_semantic_bridges"]["count"],
                "both_count": research_summary["topology_failures"]["both"]["count"],
                "ambiguous_geometry_count": research_summary["topology_failures"]["ambiguous_geometry"]["count"],
                "other_count": research_summary["topology_failures"]["other"]["count"],
            }
        )
        for gt_count in (1, 2, 3):
            gt_count_rows.append({"checkpoint": spec.label, **_gt_bucket_summary(research_rows, gt_count)})
        if spec.label in {"topology_best_semantic", "topology_last_diagnostic"}:
            aux_rows = _evaluate_auxiliary_head(model, aux_loader, device=device, use_amp=use_amp)
            for row in aux_rows:
                aux_rows_all.append({"checkpoint": spec.label, **row})

    sample_changes, sample_change_summary = _summarize_sample_changes(
        per_checkpoint_sample_rows["baseline"],
        per_checkpoint_sample_rows["topology_best_semantic"],
    )
    visual_examples = _pick_visual_examples(
        per_checkpoint_sample_rows["baseline"],
        per_checkpoint_sample_rows["topology_best_semantic"],
        sample_changes,
    )

    baseline_summary = research_eval_payloads["baseline"]
    best_summary = research_eval_payloads["topology_best_semantic"]
    last_summary = research_eval_payloads["topology_last_diagnostic"]
    baseline_sem_val = next(row for row in semantic_val_rows if row["checkpoint"] == "baseline")
    best_sem_val = next(row for row in semantic_val_rows if row["checkpoint"] == "topology_best_semantic")
    last_sem_val = next(row for row in semantic_val_rows if row["checkpoint"] == "topology_last_diagnostic")
    gt2_base = next(row for row in gt_count_rows if row["checkpoint"] == "baseline" and int(row["gt_count"]) == 2)
    gt2_best = next(row for row in gt_count_rows if row["checkpoint"] == "topology_best_semantic" and int(row["gt_count"]) == 2)
    gt3_base = next(row for row in gt_count_rows if row["checkpoint"] == "baseline" and int(row["gt_count"]) == 3)
    gt3_best = next(row for row in gt_count_rows if row["checkpoint"] == "topology_best_semantic" and int(row["gt_count"]) == 3)
    primary_delta = {
        "mean_dice_fg": float(best_summary["semantic"]["mean_dice_fg"]) - float(baseline_summary["semantic"]["mean_dice_fg"]),
        "leaflet_iou": float(best_summary["semantic"]["leaflet_iou"]) - float(baseline_summary["semantic"]["leaflet_iou"]),
        "mean_matched_iou": float(best_summary["instance"]["mean_matched_iou"]) - float(baseline_summary["instance"]["mean_matched_iou"]),
        "all_iou_ge_0.50_rate": float(best_summary["instance"]["all_iou_ge_0.50_rate"]) - float(baseline_summary["instance"]["all_iou_ge_0.50_rate"]),
        "all_iou_ge_0.70_rate": float(best_summary["instance"]["all_iou_ge_0.70_rate"]) - float(baseline_summary["instance"]["all_iou_ge_0.70_rate"]),
        "gt2_all_iou_ge_0.50_rate": float(gt2_best["all_iou_ge_0.50_rate"]) - float(gt2_base["all_iou_ge_0.50_rate"]),
        "gt3_all_iou_ge_0.50_rate": float(gt3_best["all_iou_ge_0.50_rate"]) - float(gt3_base["all_iou_ge_0.50_rate"]),
        "missing_failures": int(best_summary["topology_failures"]["missing_semantic_pixels"]["count"]) - int(baseline_summary["topology_failures"]["missing_semantic_pixels"]["count"]),
        "bridge_failures": int(best_summary["topology_failures"]["false_semantic_bridges"]["count"]) - int(baseline_summary["topology_failures"]["false_semantic_bridges"]["count"]),
    }
    primary_classification = _primary_classification(primary_delta, baseline_summary, best_summary)
    last_diag_result, last_diag_evidence = _last_checkpoint_diagnostic(baseline_summary, best_summary, last_summary)
    best_aux_rows = [row for row in aux_rows_all if row["checkpoint"] == "topology_best_semantic"]
    next_step_decision, next_step_reason = _next_step_decision(primary_classification, best_aux_rows)

    postrun_summary = {
        "contract": contract,
        "checkpoint_identities": checkpoint_identities,
        "semantic_validation": {
            "baseline": baseline_sem_val,
            "topology_best_semantic": best_sem_val,
            "topology_last_diagnostic": last_sem_val,
            "best_vs_baseline_delta": {
                "leaflet_dice": float(best_sem_val["leaflet_dice"]) - float(baseline_sem_val["leaflet_dice"]),
                "leaflet_iou": float(best_sem_val["leaflet_iou"]) - float(baseline_sem_val["leaflet_iou"]),
                "ring_dice": float(best_sem_val["ring_dice"]) - float(baseline_sem_val["ring_dice"]),
                "ring_iou": float(best_sem_val["ring_iou"]) - float(baseline_sem_val["ring_iou"]),
                "mean_dice_fg": float(best_sem_val["mean_dice_fg"]) - float(baseline_sem_val["mean_dice_fg"]),
                "mean_iou_fg": float(best_sem_val["mean_iou_fg"]) - float(baseline_sem_val["mean_iou_fg"]),
            },
        },
        "research_eval": research_eval_payloads,
        "primary_delta": primary_delta,
        "sample_change_summary": sample_change_summary,
        "auxiliary_head_metrics": aux_rows_all,
        "last_checkpoint_diagnostic": {
            "result": last_diag_result,
            "evidence": last_diag_evidence,
        },
        "primary_classification": primary_classification,
        "next_step": {
            "decision": next_step_decision,
            "reason": next_step_reason,
        },
        "visual_review_examples": visual_examples,
    }

    _write_json(OUTPUT_DIR / "checkpoint_identities.json", checkpoint_identities)
    _write_csv(
        OUTPUT_DIR / "semantic_val_comparison.csv",
        semantic_val_rows,
        ["checkpoint", "n", "leaflet_dice", "leaflet_iou", "ring_dice", "ring_iou", "mean_dice_fg", "mean_iou_fg"],
    )
    _write_csv(
        OUTPUT_DIR / "research_eval_comparison.csv",
        research_eval_rows_all,
        [
            "checkpoint",
            "n",
            "leaflet_dice",
            "leaflet_iou",
            "ring_dice",
            "ring_iou",
            "mean_dice_fg",
            "mean_iou_fg",
            "exact_k_count",
            "exact_k_rate",
            "mean_matched_iou",
            "median_matched_iou",
            "all_iou_ge_0.50_count",
            "all_iou_ge_0.50_rate",
            "all_iou_ge_0.70_count",
            "all_iou_ge_0.70_rate",
            "all_iou_ge_0.80_count",
            "all_iou_ge_0.80_rate",
            "missing_semantic_pixels_count",
            "false_semantic_bridges_count",
            "both_count",
            "ambiguous_geometry_count",
            "other_count",
        ],
    )
    _write_csv(
        OUTPUT_DIR / "gt_count_comparison.csv",
        gt_count_rows,
        [
            "checkpoint",
            "gt_count",
            "n",
            "all_iou_ge_0.50_count",
            "all_iou_ge_0.50_rate",
            "all_iou_ge_0.70_count",
            "all_iou_ge_0.70_rate",
            "all_iou_ge_0.80_count",
            "all_iou_ge_0.80_rate",
            "mean_matched_iou",
        ],
    )
    _write_csv(
        OUTPUT_DIR / "per_sample_changes.csv",
        sample_changes,
        [
            "sample_id",
            "gt_count",
            "status",
            "reason",
            "baseline_all_iou_ge_0.50",
            "topology_best_all_iou_ge_0.50",
            "baseline_mean_matched_iou",
            "topology_best_mean_matched_iou",
            "baseline_failure",
            "topology_best_failure",
        ],
    )
    _write_csv(
        OUTPUT_DIR / "auxiliary_head_metrics.csv",
        aux_rows_all,
        ["checkpoint", "channel", "name", "target_positive_fraction", "bce", "dice", "precision", "recall", "tp", "fp", "fn"],
    )
    _write_json(OUTPUT_DIR / "postrun_summary.json", postrun_summary)

    print(json.dumps(postrun_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
