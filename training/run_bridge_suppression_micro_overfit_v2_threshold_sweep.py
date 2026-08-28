from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import bridge_suppression_head as bridge


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml"
DEFAULT_RUN_DIR = REPO_ROOT / "training" / "runs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "training" / "analysis" / "bridge_suppression_micro_overfit_v2_threshold_sweep"
DEFAULT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
DEFAULT_CHECKPOINT_ORDER = ["best_reconstruction.pth", "best_pixel_f1.pth", "last.pth"]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise SystemExit(f"Unsupported bridge checkpoint format: {path}")
    model_state = payload["model"]
    if not isinstance(model_state, dict):
        raise SystemExit(f"Bridge checkpoint missing model state dict: {path}")
    return {
        "path": str(path),
        "name": path.name,
        "file_sha256": bridge._sha256_file(path),
        "model_state_sha256": bridge.canonical_model_state_sha256(model_state),
        "step": int(payload.get("step", -1)),
        "extra": payload.get("extra", {}),
    }


def resolve_checkpoint_to_evaluate(*, run_dir: Path, explicit_checkpoint: str | None = None) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    if explicit_checkpoint:
        path = bridge._resolve_repo_path(explicit_checkpoint, DEFAULT_RUN_DIR / "best_reconstruction.pth")
        if not path.exists():
            raise SystemExit(f"Explicit checkpoint does not exist: {path}")
        candidates.append(inspect_checkpoint(path))
    else:
        for name in DEFAULT_CHECKPOINT_ORDER:
            path = run_dir / name
            if path.exists():
                candidates.append(inspect_checkpoint(path))
    if not candidates:
        expected = ", ".join(str(run_dir / name) for name in DEFAULT_CHECKPOINT_ORDER)
        raise SystemExit(f"No V2 bridge checkpoint found. Expected one of: {expected}")
    preferred = None
    for item in candidates:
        if Path(item["path"]).name == "best_reconstruction.pth":
            preferred = item
            break
    if preferred is None:
        preferred = candidates[0]
    model_shas = {str(item["model_state_sha256"]) for item in candidates}
    steps = {int(item["step"]) for item in candidates}
    return {
        "evaluated": preferred,
        "available": candidates,
        "best_pixel_best_reconstruction_last_identical": bool(len(model_shas) == 1 and len(steps) == 1 and len(candidates) == 3),
    }


def load_trained_bridge_checkpoint(model: bridge.FrozenSemanticBridgeSuppressionModel, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    state = payload.get("model")
    if not isinstance(state, dict):
        raise SystemExit(f"Bridge checkpoint missing model state dict: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise SystemExit(f"Unexpected bridge checkpoint incompatibility: missing={missing[:5]} unexpected={unexpected[:5]}")
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": bridge._sha256_file(checkpoint_path),
        "step": int(payload.get("step", -1)),
        "extra": payload.get("extra", {}),
    }


def load_locked_v2_cached_records(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    dataset_cfg = cfg.get("dataset") or {}
    micro_cfg = cfg.get("micro_overfit") or {}
    train_split = bridge._resolve_repo_path(dataset_cfg.get("train_txt", bridge.DEFAULT_TRAIN_SPLIT), bridge.DEFAULT_TRAIN_SPLIT)
    val_split = bridge._resolve_repo_path(dataset_cfg.get("val_txt", bridge.DEFAULT_VAL_SPLIT), bridge.DEFAULT_VAL_SPLIT)
    test_split = bridge._resolve_repo_path(dataset_cfg.get("test_txt", bridge.DEFAULT_TEST_SPLIT), bridge.DEFAULT_TEST_SPLIT)
    bridge._assert_safe_path(train_split)
    bridge._assert_safe_path(val_split)
    bridge._assert_safe_path(test_split)

    model = bridge.build_model_from_cfg(cfg).to(device)
    semantic_checkpoint = bridge.load_semantic_checkpoint(
        model,
        bridge._resolve_repo_path((cfg.get("train") or {}).get("init_checkpoint"), bridge.DEFAULT_SEMANTIC_CHECKPOINT),
    )
    manifest_path = bridge._resolve_repo_path(micro_cfg.get("manifest_path"), bridge.MICRO_MANIFEST_V2_PATH)
    manifest_payload = bridge.read_locked_micro_manifest(manifest_path)
    manifest_payload["_manifest_path"] = str(manifest_path.resolve())
    split_validation = bridge.validate_locked_manifest_source_split(
        manifest_payload=manifest_payload,
        configured_train_split=train_split,
    )
    if str(split_validation.get("status")) != "pass":
        raise SystemExit(json.dumps({"split_validation": split_validation}, ensure_ascii=False, indent=2))
    records = bridge.mine_bridge_records_for_split(
        cfg=cfg,
        split_txt=train_split,
        model=model,
        device=device,
        cache_features=True,
        selected_sample_ids=[str(v) for v in manifest_payload["sample_ids"]],
    )
    record_validation = bridge.validate_locked_micro_records(
        manifest_payload=manifest_payload,
        records=records,
        split_txt=train_split,
    )
    if str(record_validation.get("status")) != "pass":
        raise SystemExit(json.dumps({"split_validation": split_validation, "record_validation": record_validation}, ensure_ascii=False, indent=2))
    return {
        "model": model,
        "semantic_checkpoint": semantic_checkpoint,
        "manifest": manifest_payload,
        "manifest_validation": {
            "split_validation": split_validation,
            "record_validation": record_validation,
        },
        "cached_records": bridge.cache_microset_features(records),
    }


def _split_ratio(text: str) -> tuple[int, int]:
    left, right = str(text).split("/", 1)
    return int(left), int(right)


def flatten_threshold_metrics(threshold: float, recon: dict[str, Any]) -> dict[str, Any]:
    full_recon = recon["reconstruction"]["p50_minus_predicted_bridge"]
    positive_recon = recon["positive_subset"]["reconstruction"]["p50_minus_predicted_bridge"]
    negative = recon["negative_subset"]
    calibration = recon["removal_calibration"]
    full_gt2_success, full_gt2_n = _split_ratio(full_recon["gt2_success"])
    full_gt3_success, full_gt3_n = _split_ratio(full_recon["gt3_success"])
    positive_gt2_success, positive_gt2_n = _split_ratio(positive_recon["gt2_success"])
    positive_gt3_success, positive_gt3_n = _split_ratio(positive_recon["gt3_success"])
    return {
        "threshold": float(threshold),
        "pixel_precision": float(recon["pixel"]["precision"]),
        "pixel_recall": float(recon["pixel"]["recall"]),
        "pixel_f1": float(recon["pixel"]["f1"]),
        "pixel_dice": float(recon["pixel"]["dice"]),
        "pixel_tp": int(recon["pixel"]["tp"]),
        "pixel_fp": int(recon["pixel"]["fp"]),
        "pixel_fn": int(recon["pixel"]["fn"]),
        "positive_pixel_precision": float(recon["positive_subset"]["pixel"]["precision"]),
        "positive_pixel_recall": float(recon["positive_subset"]["pixel"]["recall"]),
        "positive_pixel_f1": float(recon["positive_subset"]["pixel"]["f1"]),
        "positive_pixel_dice": float(recon["positive_subset"]["pixel"]["dice"]),
        "positive_pixel_tp": int(recon["positive_subset"]["pixel"]["tp"]),
        "positive_pixel_fp": int(recon["positive_subset"]["pixel"]["fp"]),
        "positive_pixel_fn": int(recon["positive_subset"]["pixel"]["fn"]),
        "full_mean_iou": float(full_recon["mean_matched_iou"]),
        "full_success50_count": int(full_recon["all_iou_ge_0.50_count"]),
        "full_success50_rate": float(full_recon["all_iou_ge_0.50_rate"]),
        "full_gt2_success": str(full_recon["gt2_success"]),
        "full_gt2_success_count": int(full_gt2_success),
        "full_gt2_n": int(full_gt2_n),
        "full_gt3_success": str(full_recon["gt3_success"]),
        "full_gt3_success_count": int(full_gt3_success),
        "full_gt3_n": int(full_gt3_n),
        "positive_mean_iou": float(positive_recon["mean_matched_iou"]),
        "positive_success50_count": int(positive_recon["all_iou_ge_0.50_count"]),
        "positive_success50_rate": float(positive_recon["all_iou_ge_0.50_rate"]),
        "positive_gt2_success": str(positive_recon["gt2_success"]),
        "positive_gt2_success_count": int(positive_gt2_success),
        "positive_gt2_n": int(positive_gt2_n),
        "positive_gt3_success": str(positive_recon["gt3_success"]),
        "positive_gt3_success_count": int(positive_gt3_success),
        "positive_gt3_n": int(positive_gt3_n),
        "negative_predicted_bridge_pixels": int(negative["predicted_bridge_pixels"]),
        "negative_removed_fraction": float(negative["fraction_of_candidate_pixels_removed"]),
        "negative_zero_removal_samples": int(negative["samples_with_zero_predicted_removal"]),
        "negative_mean_iou": float(negative["refined_mean_matched_iou"]),
        "negative_num_improves": int(negative["num_improves"]),
        "negative_num_unchanged": int(negative["num_unchanged"]),
        "negative_num_regresses": int(negative["num_regresses"]),
        "negative_num_component_topology_changes": int(negative["num_component_topology_changes"]),
        "all_removed_over_candidate": float(calibration["all_removed_over_candidate"]),
        "positive_removed_over_candidate": float(calibration["positive_removed_over_candidate"]),
        "negative_removed_over_candidate": float(calibration["negative_removed_over_candidate"]),
        "positive_gt_bridge_over_candidate": float(calibration["positive_gt_bridge_over_candidate"]),
    }


def threshold_rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["negative_num_component_topology_changes"]),
        int(row["negative_num_regresses"]),
        -int(row["positive_success50_count"]),
        -float(row["positive_mean_iou"]),
        -float(row["full_mean_iou"]),
        -float(row["pixel_f1"]),
        float(row["threshold"]),
    )


def select_best_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise SystemExit("Threshold sweep produced no rows.")
    return min(rows, key=threshold_rank_key)


def find_safe_threshold(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    safe_rows = [
        row for row in rows
        if int(row["positive_success50_count"]) >= 3
        and int(row["negative_num_regresses"]) == 0
        and int(row["negative_num_component_topology_changes"]) == 0
    ]
    if not safe_rows:
        return None
    return min(safe_rows, key=threshold_rank_key)


def run_threshold_sweep(
    *,
    model: bridge.FrozenSemanticBridgeSuppressionModel,
    cached_records: list[dict[str, Any]],
    device: torch.device,
    thresholds: list[float],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    per_threshold: dict[str, Any] = {}
    for threshold in thresholds:
        recon = bridge.evaluate_reconstruction_levels_on_cached(model, cached_records, device, threshold=float(threshold))
        row = flatten_threshold_metrics(float(threshold), recon)
        rows.append(row)
        per_threshold[f"{float(threshold):.2f}"] = {
            "threshold": float(threshold),
            "pixel": recon["pixel"],
            "positive_pixel": recon["positive_subset"]["pixel"],
            "reconstruction": recon["reconstruction"]["p50_minus_predicted_bridge"],
            "positive_reconstruction": recon["positive_subset"]["reconstruction"]["p50_minus_predicted_bridge"],
            "negative_preservation": recon["negative_subset"],
            "removal_calibration": recon["removal_calibration"],
            "per_sample": recon["per_sample"],
        }
    best_row = select_best_threshold(rows)
    safe_row = find_safe_threshold(rows)
    return {
        "rows": rows,
        "per_threshold": per_threshold,
        "safe_threshold": safe_row,
        "safe_separation_threshold_exists": bool(safe_row is not None),
        "best_preservation_aware_threshold": best_row,
    }


def current_checkpoint_policy_flaw() -> str:
    return (
        "Current best_reconstruction selection ranks checkpoints only by "
        "(predicted success50, predicted mean matched IoU). "
        "It ignores negative preservation, so a checkpoint can win while still "
        "regressing every bridge-negative sample and changing negative component topology."
    )


def proposed_preservation_aware_checkpoint_policy() -> list[str]:
    return [
        "1. minimize negative component topology changes",
        "2. minimize negative regressions",
        "3. maximize positive success50",
        "4. maximize positive mean matched IoU",
        "5. maximize overall success50",
        "6. maximize overall mean matched IoU",
        "7. maximize pixel F1",
        "8. earlier step tie-break",
    ]


def run_analysis(*, config_path: Path, run_dir: Path, output_dir: Path, checkpoint_arg: str | None = None) -> dict[str, Any]:
    cfg = bridge._read_yaml(config_path.resolve())
    cfg["_config_path"] = str(config_path.resolve())
    device = bridge._select_device()
    checkpoint_choice = resolve_checkpoint_to_evaluate(run_dir=run_dir.resolve(), explicit_checkpoint=checkpoint_arg)
    loaded = load_locked_v2_cached_records(cfg, device)
    model = loaded["model"]
    bridge_checkpoint = load_trained_bridge_checkpoint(model, Path(checkpoint_choice["evaluated"]["path"]))
    sweep = run_threshold_sweep(
        model=model,
        cached_records=loaded["cached_records"],
        device=device,
        thresholds=[float(v) for v in DEFAULT_THRESHOLDS],
    )
    fieldnames = [
        "threshold",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pixel_dice",
        "pixel_tp",
        "pixel_fp",
        "pixel_fn",
        "positive_pixel_precision",
        "positive_pixel_recall",
        "positive_pixel_f1",
        "positive_pixel_dice",
        "positive_pixel_tp",
        "positive_pixel_fp",
        "positive_pixel_fn",
        "full_mean_iou",
        "full_success50_count",
        "full_success50_rate",
        "full_gt2_success",
        "full_gt3_success",
        "positive_mean_iou",
        "positive_success50_count",
        "positive_success50_rate",
        "positive_gt2_success",
        "positive_gt3_success",
        "negative_predicted_bridge_pixels",
        "negative_removed_fraction",
        "negative_zero_removal_samples",
        "negative_mean_iou",
        "negative_num_improves",
        "negative_num_unchanged",
        "negative_num_regresses",
        "negative_num_component_topology_changes",
        "all_removed_over_candidate",
        "positive_removed_over_candidate",
        "negative_removed_over_candidate",
        "positive_gt_bridge_over_candidate",
    ]
    _write_json(output_dir / "threshold_sweep.json", {
        "config_path": str(config_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "checkpoint_choice": checkpoint_choice,
        "semantic_checkpoint": loaded["semantic_checkpoint"],
        "bridge_checkpoint": bridge_checkpoint,
        "manifest_validation": loaded["manifest_validation"],
        "thresholds": [float(v) for v in DEFAULT_THRESHOLDS],
        "safe_separation_threshold_exists": sweep["safe_separation_threshold_exists"],
        "safe_threshold": sweep["safe_threshold"],
        "best_preservation_aware_threshold": sweep["best_preservation_aware_threshold"],
        "checkpoint_selection_audit": {
            "current_policy_flaw": current_checkpoint_policy_flaw(),
            "proposed_policy": proposed_preservation_aware_checkpoint_policy(),
        },
        "rows": sweep["rows"],
        "per_threshold": sweep["per_threshold"],
    })
    _write_csv(output_dir / "threshold_sweep.csv", sweep["rows"], fieldnames)
    return {
        "config_path": str(config_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "checkpoint_choice": checkpoint_choice,
        "bridge_checkpoint": bridge_checkpoint,
        "safe_separation_threshold_exists": sweep["safe_separation_threshold_exists"],
        "safe_threshold": sweep["safe_threshold"],
        "best_preservation_aware_threshold": sweep["best_preservation_aware_threshold"],
        "rows": sweep["rows"],
        "checkpoint_selection_audit": {
            "current_policy_flaw": current_checkpoint_policy_flaw(),
            "proposed_policy": proposed_preservation_aware_checkpoint_policy(),
        },
        "output_json": str((output_dir / "threshold_sweep.json").resolve()),
        "output_csv": str((output_dir / "threshold_sweep.csv").resolve()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    ap.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--checkpoint", type=str, default=None)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_analysis(
        config_path=bridge._resolve_repo_path(args.config, DEFAULT_CONFIG),
        run_dir=bridge._resolve_repo_path(args.run_dir, DEFAULT_RUN_DIR),
        output_dir=bridge._resolve_repo_path(args.output_dir, DEFAULT_OUTPUT_DIR),
        checkpoint_arg=args.checkpoint,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
