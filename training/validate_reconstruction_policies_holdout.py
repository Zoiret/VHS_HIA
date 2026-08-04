from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
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
    _load_checkpoint,
    _load_gt_instance,
    _make_device,
    _marker_contract,
    _read_yaml,
    _resolve_path,
    _resize_panel,
    _sample_center_metrics,
    _seed_all,
    _sha256_file,
)
from compare_reconstruction_policies import (
    PRIMARY_THRESHOLD,
    SECONDARY_THRESHOLDS,
    _assert_policy_contract,
    _json_safe_trace,
    _labels_to_bgr,
    _none_or_float,
    _none_or_int,
    _overlay_component_ids,
    _policy_metrics,
    _validate_policy_artifact_integrity,
    _write_csv_atomic,
    _write_json_atomic,
    _write_text_atomic,
    run_policy,
)
from validate_centerhead import _extract_metadata_centers


DEFAULT_OUTPUT_DIR = "training/analysis/centerhead_spatial_x2_2_reconstruction_policy_holdout"
EXCLUDED_MICROSET_IDS = (
    "m01_p02_s00",
    "m01_p02_s04",
    "m01_p01_s00",
    "m01_p01_s01",
    "m01_p01_s02",
    "m01_p01_s03",
)
FIXED_P3_CFG = {"distance_gate_px": 8.0, "relative_area_gate": 0.01}
BOOTSTRAP_SEED = 0
BOOTSTRAP_SAMPLES = 10_000
MIN_TOTAL_HELDOUT = 20
MIN_PER_GT_COUNT = {1: 3, 2: 3, 3: 3}
REQUIRED_POLICIES = (
    "P0_CURRENT",
    "P1_DROP_UNMARKED",
    "P2_ATTACH_TO_NEAREST_MARKER",
    "P3_GATED_ATTACH",
    "P4_GLOBAL_MARKER_CONTROLLED",
)


def _read_split_entries(dataset_root: Path, split_txt: Path, split_name: str) -> list[dict]:
    entries = []
    for line in split_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) != 2:
            raise SystemExit(f"Invalid line in {split_txt}: {line!r}")
        image_rel, semantic_rel = parts
        sample_id = Path(image_rel).stem
        entries.append(
            {
                "sample": sample_id,
                "split": split_name,
                "image_rel": image_rel,
                "semantic_rel": semantic_rel,
                "image_path": (dataset_root / image_rel).resolve(),
                "gt_semantic_path": (dataset_root / semantic_rel).resolve(),
                "center_path": (dataset_root / "center_maps" / f"{sample_id}.png").resolve(),
                "metadata_path": (dataset_root / "metadata" / f"{sample_id}.json").resolve(),
            }
        )
    return entries


def _mouse_id(sample_id: str) -> str:
    return str(sample_id).split("_")[0]


def _semantic_cc_bucket(cc_count: int) -> str:
    if int(cc_count) <= 1:
        return "1"
    if int(cc_count) <= 3:
        return "2-3"
    if int(cc_count) <= 7:
        return "4-7"
    return "8+"


def _manifest_text(entries: list[dict]) -> str:
    header = "\t".join(
        [
            "sample_id",
            "image_path",
            "gt_semantic_path",
            "gt_instance_path",
            "source_dataset",
            "gt_instance_count",
        ]
    )
    lines = [header]
    for entry in entries:
        lines.append(
            "\t".join(
                [
                    str(entry["sample"]),
                    str(entry["image_path"]),
                    str(entry["gt_semantic_path"]),
                    str(entry["gt_instance_path"]),
                    str(entry["source_dataset"]),
                    str(entry["gt_instance_count"]),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _manifest_sha256(entries: list[dict]) -> str:
    return hashlib.sha256(_manifest_text(entries).encode("utf-8")).hexdigest()


def _inventory_holdout_samples(cfg: dict, repo_root: Path) -> dict:
    dataset_root = _resolve_path(repo_root, cfg["dataset"]["root"])
    instance_root = _resolve_path(repo_root, cfg["dataset"]["instance_root"])
    if dataset_root is None or instance_root is None:
        raise SystemExit("Config: dataset.root and dataset.instance_root are required")
    split_specs = [
        ("val", _resolve_path(repo_root, cfg["dataset"]["val_txt"])),
        ("test", _resolve_path(repo_root, cfg["dataset"]["test_txt"])),
    ]
    excluded = []
    missing = []
    eligible = []
    seen = set()
    for split_name, split_txt in split_specs:
        if split_txt is None or not split_txt.exists():
            missing.append({"split": split_name, "reason": "missing_split_txt", "path": str(split_txt) if split_txt is not None else None})
            continue
        for entry in _read_split_entries(dataset_root, split_txt, split_name):
            sample_id = str(entry["sample"])
            if sample_id in seen:
                continue
            seen.add(sample_id)
            if sample_id in EXCLUDED_MICROSET_IDS:
                excluded.append({"sample": sample_id, "split": split_name, "reason": "authoritative_microset"})
                continue
            gt_instance_path = (instance_root / "instance_masks" / f"{sample_id}.png").resolve()
            requirements = {
                "image_exists": bool(entry["image_path"].exists()),
                "gt_semantic_exists": bool(entry["gt_semantic_path"].exists()),
                "gt_instance_exists": bool(gt_instance_path.exists()),
                "center_exists": bool(entry["center_path"].exists()),
                "metadata_exists": bool(entry["metadata_path"].exists()),
            }
            if not all(requirements.values()):
                missing.append({"sample": sample_id, "split": split_name, "requirements": requirements})
                continue
            meta = json.loads(entry["metadata_path"].read_text(encoding="utf-8"))
            gt_count = int(meta.get("instance_count", 0))
            eligible.append(
                {
                    "sample": sample_id,
                    "split": split_name,
                    "image_rel": entry["image_rel"],
                    "semantic_rel": entry["semantic_rel"],
                    "image_path": str(entry["image_path"]),
                    "gt_semantic_path": str(entry["gt_semantic_path"]),
                    "gt_instance_path": str(gt_instance_path),
                    "metadata_path": str(entry["metadata_path"]),
                    "source_dataset": "converted_leaflet_distance",
                    "gt_instance_count": gt_count,
                    "mouse_id": _mouse_id(sample_id),
                }
            )
    eligible.sort(key=lambda item: (item["split"], item["sample"]))
    gt_counter = Counter(int(entry["gt_instance_count"]) for entry in eligible)
    evidence_min = len(eligible) >= MIN_TOTAL_HELDOUT and all(int(gt_counter.get(k, 0)) >= int(v) for k, v in MIN_PER_GT_COUNT.items())
    return {
        "dataset_root": str(dataset_root),
        "instance_root": str(instance_root),
        "eligible": eligible,
        "excluded_microset": excluded,
        "missing": missing,
        "evidence_level": "sufficient_evidence" if evidence_min else "insufficient_evidence",
        "meets_minimum": bool(evidence_min),
        "gt_instance_distribution": {str(k): int(v) for k, v in sorted(gt_counter.items())},
    }


def _write_holdout_manifest(out_dir: Path, inventory: dict) -> tuple[Path, Path]:
    manifest_path = (out_dir / "holdout_manifest.txt").resolve()
    manifest_text = _manifest_text(inventory["eligible"])
    _write_text_atomic(manifest_path, manifest_text, encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    metadata = {
        "manifest_sha256": manifest_sha,
        "eligible_count": len(inventory["eligible"]),
        "excluded_microset_count": len(inventory["excluded_microset"]),
        "missing_count": len(inventory["missing"]),
        "evidence_level": inventory["evidence_level"],
        "meets_minimum": inventory["meets_minimum"],
        "gt_instance_distribution": inventory["gt_instance_distribution"],
        "excluded_microset_ids": list(EXCLUDED_MICROSET_IDS),
        "dataset_root": inventory["dataset_root"],
        "instance_root": inventory["instance_root"],
        "missing": inventory["missing"],
    }
    metadata_path = (out_dir / "holdout_manifest_metadata.json").resolve()
    _write_json_atomic(metadata_path, metadata)
    return manifest_path, metadata_path


def _build_loader_split_file(out_dir: Path, eligible_entries: list[dict]) -> Path:
    split_path = (out_dir / "_holdout_loader_split.txt").resolve()
    lines = [f"{entry['image_rel']}\t{entry['semantic_rel']}" for entry in eligible_entries]
    _write_text_atomic(split_path, "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return split_path


def _checkpoint_identity(run_dir: Path) -> dict:
    best = (run_dir / "best_micro_overfit.pth").resolve()
    if not best.exists():
        raise SystemExit(f"Missing best checkpoint: {best}")
    ckpt = _load_checkpoint(best)
    saved_iteration = int(ckpt.get("step", -1))
    saved_best_threshold = ((ckpt.get("extra") or {}).get("best_threshold", None))
    return {
        "checkpoint_path": str(best),
        "checkpoint_sha256": _sha256_file(best),
        "checkpoint_iteration": saved_iteration,
        "saved_best_threshold": None if saved_best_threshold is None else float(saved_best_threshold),
        "state_dict": ckpt.get("model", ckpt),
    }


def _row_from_metrics(*, sample_entry: dict, threshold: float, policy: str, marker_contract: dict, center_metrics: dict, policy_metrics: dict, p3_cfg: dict | None) -> dict:
    return {
        "sample": sample_entry["sample"],
        "split": sample_entry["split"],
        "sample_index": int(sample_entry["sample_index"]),
        "mouse_id": sample_entry["mouse_id"],
        "threshold": float(threshold),
        "policy": policy,
        "gt_instance_count": int(policy_metrics["counts"]["gt_instance_count"]),
        "marker_count": int(policy_metrics["counts"]["marker_count"]),
        "marker_contract_pass": bool(marker_contract["marker_contract_pass"]),
        "markers_outside_all_gt_instances": int(marker_contract["markers_outside_all_gt_instances"]),
        "one_marker_per_instance_rate": float(marker_contract["one_marker_per_instance_rate"]),
        "semantic_cc_count": int(policy_metrics["counts"]["semantic_connected_component_count"]),
        "raw_output_label_count": int(policy_metrics["counts"]["raw_output_label_count"]),
        "final_output_label_count": int(policy_metrics["counts"]["final_output_label_count"]),
        "output_count_error": int(policy_metrics["counts"]["final_output_label_count"] - policy_metrics["counts"]["gt_instance_count"]),
        "exact_count": bool(policy_metrics["counts"]["exact_count"]),
        "matched_iou": float(policy_metrics["instance_metrics"]["matched_iou"]),
        "mean_matched_dice": policy_metrics["instance_metrics"]["mean_matched_dice"],
        "merged": bool(policy_metrics["instance_metrics"]["merged"]),
        "fragmented": bool(policy_metrics["instance_metrics"]["fragmented"]),
        "mixed": bool(policy_metrics["instance_metrics"]["mixed"]),
        "perfect_recovery": bool(policy_metrics["instance_metrics"]["perfect_recovery"]),
        "instance_score": float(policy_metrics["instance_metrics"]["instance_score"]),
        "assigned_area_fraction": float(policy_metrics["area_accounting"]["assigned_area_fraction"]),
        "dropped_area_fraction": float(policy_metrics["area_accounting"]["dropped_area"] / max(int(policy_metrics["area_accounting"]["semantic_leaflet_area"]), 1)),
        "invariant_pass": bool(policy_metrics["contract"]["pass"]),
        "fallback_marker_calls": int(policy_metrics["contract"]["fallback_marker_calls"]),
        "keep_top3_call_count": int(policy_metrics["contract"]["keep_top3_call_count"]),
        "raw_labels_without_marker_provenance": int(policy_metrics["contract"]["raw_labels_without_marker_provenance"]),
        "final_labels_without_marker_provenance": int(policy_metrics["contract"]["final_labels_without_marker_provenance"]),
        "ambiguous_assignments": _none_or_int(policy_metrics["component_assignment"]["ambiguous_assignments"]),
        "markers_preserved": bool(len(policy_metrics["contract"]["markers_without_output_label"]) == 0),
        "marked_components": _none_or_int(policy_metrics["component_assignment"]["marked_components"]),
        "unmarked_components": _none_or_int(policy_metrics["component_assignment"]["unmarked_components"]),
        "unmarked_component_area": _none_or_int(policy_metrics["area_accounting"]["unmarked_component_area"]),
        "unmarked_components_rejected": _none_or_int(policy_metrics["component_assignment"]["unmarked_components_rejected"]),
        "component_diagnostic_status": policy_metrics["component_assignment"]["diagnostic_status"],
        "center_precision": float(center_metrics["center_precision"]),
        "center_recall": float(center_metrics["center_recall"]),
        "center_f1": float(center_metrics["center_f1"]),
        "distance_gate_px": None if p3_cfg is None else float(p3_cfg["distance_gate_px"]),
        "relative_area_gate": None if p3_cfg is None or p3_cfg["relative_area_gate"] is None else float(p3_cfg["relative_area_gate"]),
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _aggregate_rows(rows: list[dict]) -> dict:
    dice_values = [float(r["mean_matched_dice"]) for r in rows if r["mean_matched_dice"] is not None]
    iou_values = [float(r["matched_iou"]) for r in rows]
    exact_vals = [1.0 if bool(r["exact_count"]) else 0.0 for r in rows]
    assigned_vals = [float(r["assigned_area_fraction"]) for r in rows]
    dropped_vals = [float(r["dropped_area_fraction"]) for r in rows]
    out_err = [float(r["output_count_error"]) for r in rows]
    ambiguous_vals = [int(r["ambiguous_assignments"]) for r in rows if r["ambiguous_assignments"] is not None]
    return {
        "sample_count": len(rows),
        "marker_contract_pass_count": int(sum(1 for r in rows if bool(r["marker_contract_pass"]))),
        "marker_contract_pass_rate": float(np.mean([1.0 if bool(r["marker_contract_pass"]) else 0.0 for r in rows])) if rows else None,
        "exact_count_accuracy": _mean(exact_vals),
        "mean_matched_iou": _mean(iou_values),
        "median_matched_iou": _median(iou_values),
        "mean_dice": _mean(dice_values),
        "median_dice": _median(dice_values),
        "fragmented_rate": float(np.mean([1.0 if bool(r["fragmented"]) else 0.0 for r in rows])) if rows else None,
        "merged_rate": float(np.mean([1.0 if bool(r["merged"]) else 0.0 for r in rows])) if rows else None,
        "mixed_rate": float(np.mean([1.0 if bool(r["mixed"]) else 0.0 for r in rows])) if rows else None,
        "assigned_area_fraction": _mean(assigned_vals),
        "dropped_area_fraction": _mean(dropped_vals),
        "invariant_violation_count": int(sum(1 for r in rows if not bool(r["invariant_pass"]))),
        "fallback_marker_calls": int(sum(int(r["fallback_marker_calls"]) for r in rows)),
        "keep_top3_calls": int(sum(int(r["keep_top3_call_count"]) for r in rows)),
        "raw_labels_without_marker_provenance": int(sum(int(r["raw_labels_without_marker_provenance"]) for r in rows)),
        "final_labels_without_marker_provenance": int(sum(int(r["final_labels_without_marker_provenance"]) for r in rows)),
        "ambiguous_assignments": int(sum(ambiguous_vals)) if ambiguous_vals else 0,
        "markers_preserved_rate": float(np.mean([1.0 if bool(r["markers_preserved"]) else 0.0 for r in rows])) if rows else None,
        "mean_output_count_error": _mean(out_err),
        "median_output_count_error": _median(out_err),
    }


def _condition_rows_for_marker_contract(rows: list[dict]) -> list[dict]:
    return [row for row in rows if bool(row["marker_contract_pass"])]


def _paired_p1_vs_p0(rows: list[dict], threshold: float) -> list[dict]:
    p0 = {row["sample"]: row for row in rows if row["policy"] == "P0_CURRENT" and abs(float(row["threshold"]) - float(threshold)) < 1e-9 and bool(row["marker_contract_pass"])}
    p1 = {row["sample"]: row for row in rows if row["policy"] == "P1_DROP_UNMARKED" and abs(float(row["threshold"]) - float(threshold)) < 1e-9 and bool(row["marker_contract_pass"])}
    out = []
    for sample in sorted(set(p0) & set(p1)):
        r0 = p0[sample]
        r1 = p1[sample]
        out.append(
            {
                "sample": sample,
                "split": r0["split"],
                "mouse_id": r0["mouse_id"],
                "gt_instance_count": int(r0["gt_instance_count"]),
                "semantic_cc_count": int(r0["semantic_cc_count"]),
                "p0_exact_count": bool(r0["exact_count"]),
                "p1_exact_count": bool(r1["exact_count"]),
                "exact_count_delta": int(bool(r1["exact_count"])) - int(bool(r0["exact_count"])),
                "p0_matched_iou": float(r0["matched_iou"]),
                "p1_matched_iou": float(r1["matched_iou"]),
                "matched_iou_delta": float(r1["matched_iou"]) - float(r0["matched_iou"]),
                "p0_dice": None if r0["mean_matched_dice"] is None else float(r0["mean_matched_dice"]),
                "p1_dice": None if r1["mean_matched_dice"] is None else float(r1["mean_matched_dice"]),
                "dice_delta": (None if r0["mean_matched_dice"] is None or r1["mean_matched_dice"] is None else float(r1["mean_matched_dice"]) - float(r0["mean_matched_dice"])),
                "p0_assigned_area_fraction": float(r0["assigned_area_fraction"]),
                "p1_assigned_area_fraction": float(r1["assigned_area_fraction"]),
                "assigned_area_delta": float(r1["assigned_area_fraction"]) - float(r0["assigned_area_fraction"]),
                "p0_dropped_area_fraction": float(r0["dropped_area_fraction"]),
                "p1_dropped_area_fraction": float(r1["dropped_area_fraction"]),
                "dropped_area_delta": float(r1["dropped_area_fraction"]) - float(r0["dropped_area_fraction"]),
            }
        )
    return out


def _bootstrap_mean_ci(values: list[float], *, seed: int = BOOTSTRAP_SEED, n_bootstrap: int = BOOTSTRAP_SAMPLES) -> dict:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "seed": seed, "n_bootstrap": n_bootstrap}
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))
    means = np.mean(arr[idx], axis=1)
    return {
        "mean": float(np.mean(arr)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "seed": int(seed),
        "n_bootstrap": int(n_bootstrap),
    }


def _promotion_decision(*, conditioned_primary_by_policy: dict, paired_rows: list[dict], evidence_sufficient: bool) -> dict:
    p0 = conditioned_primary_by_policy["P0_CURRENT"]
    p1 = conditioned_primary_by_policy["P1_DROP_UNMARKED"]
    iou_deltas = [float(row["matched_iou_delta"]) for row in paired_rows]
    exact_deltas = [float(row["exact_count_delta"]) for row in paired_rows]
    mean_iou_delta = float(np.mean(iou_deltas)) if iou_deltas else None
    median_iou_delta = float(np.median(iou_deltas)) if iou_deltas else None
    criteria = {
        "invariant_violations_zero": int(p1["invariant_violation_count"]) == 0,
        "fallback_calls_zero": int(p1["fallback_marker_calls"]) == 0,
        "keep_top3_zero": int(p1["keep_top3_calls"]) == 0,
        "labels_without_marker_provenance_zero": int(p1["final_labels_without_marker_provenance"]) == 0 and int(p1["raw_labels_without_marker_provenance"]) == 0,
        "exact_count_ge_p0": (p1["exact_count_accuracy"] or 0.0) >= (p0["exact_count_accuracy"] or 0.0),
        "mean_iou_delta_ge_minus_0p02": mean_iou_delta is not None and mean_iou_delta >= -0.02,
        "median_iou_delta_ge_minus_0p01": median_iou_delta is not None and median_iou_delta >= -0.01,
        "no_marker_disappears": (p1["markers_preserved_rate"] or 0.0) >= 1.0,
        "output_count_never_exceeds_marker_count": int(p1["invariant_violation_count"]) == 0,
        "heldout_evidence_minimum": bool(evidence_sufficient),
    }
    passed = sorted([k for k, v in criteria.items() if v])
    failed = sorted([k for k, v in criteria.items() if not v])
    if not evidence_sufficient:
        status = "insufficient_evidence"
    elif not failed:
        status = "candidate_for_production_patch"
    else:
        status = "reject"
    return {
        "status": status,
        "criteria_passed": passed,
        "criteria_failed": failed,
        "mean_matched_iou_delta": mean_iou_delta,
        "median_matched_iou_delta": median_iou_delta,
        "bootstrap_mean_iou_delta_ci95": _bootstrap_mean_ci(iou_deltas),
        "bootstrap_exact_count_delta_ci95": _bootstrap_mean_ci(exact_deltas),
    }


def _worst_case_panel(*, sample: str, image_rgb_u8: np.ndarray, gt_inst: np.ndarray, pred_sem: np.ndarray, semantic_cc: np.ndarray, center_prob: np.ndarray, marker_points: list[dict], p0_trace: dict, p0_final: np.ndarray, p1_final: np.ndarray, p0_iou: float, p1_iou: float, gt_count: int, marker_count: int) -> np.ndarray:
    original = cv2.cvtColor(image_rgb_u8, cv2.COLOR_RGB2BGR)
    center_vis = _center_prob_to_bgr(center_prob)
    center_vis = _draw_markers(center_vis, [{"y": int(mp["y"]), "x": int(mp["x"])} for mp in marker_points])
    dropped = ((p0_final > 0) & (p1_final == 0)).astype(np.uint8)
    dropped_vis = np.zeros((dropped.shape[0], dropped.shape[1], 3), dtype=np.uint8)
    dropped_vis[dropped.astype(bool)] = np.array([0, 0, 255], dtype=np.uint8)
    diff = np.zeros((p0_final.shape[0], p0_final.shape[1], 3), dtype=np.uint8)
    diff[(p0_final > 0) & (p1_final == 0)] = np.array([0, 0, 255], dtype=np.uint8)
    diff[(p0_final == 0) & (p1_final > 0)] = np.array([0, 255, 0], dtype=np.uint8)
    panels = [
        _annotate_panel(_resize_panel(original), f"{sample} image"),
        _annotate_panel(_resize_panel(_labels_to_bgr(gt_inst)), f"GT instances={gt_count}"),
        _annotate_panel(_resize_panel(_overlay_component_ids(pred_sem, semantic_cc)), "semantic / CC"),
        _annotate_panel(_resize_panel(center_vis), f"markers={marker_count}"),
        _annotate_panel(_resize_panel(_labels_to_bgr(p0_trace['raw_labels'])), "P0 raw"),
        _annotate_panel(_resize_panel(_labels_to_bgr(p0_final)), f"P0 final IoU={p0_iou:.4f}"),
        _annotate_panel(_resize_panel(_labels_to_bgr(p1_final)), f"P1 final IoU={p1_iou:.4f}"),
        _annotate_panel(_resize_panel(dropped_vis), "dropped by P1"),
        _annotate_panel(_resize_panel(diff), "P1 - P0 diff"),
    ]
    width = max(panel.shape[1] for panel in panels)
    height = max(panel.shape[0] for panel in panels)
    normalized = [cv2.resize(panel, (width, height), interpolation=cv2.INTER_NEAREST) for panel in panels]
    top = np.concatenate(normalized[:5], axis=1)
    bottom = np.concatenate(normalized[5:] + [np.zeros_like(normalized[0])], axis=1)
    return np.concatenate([top, bottom], axis=0)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="training/configs/unetpp_effb3_centerhead_spatial_x2_2_adapter_legacy_fp32_micro.yaml")
    ap.add_argument("--run-dir", type=str, default="training/runs/unetpp_effb3_centerhead_spatial_x2_2_adapter_legacy_fp32_micro")
    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _seed_all(int(1337))
    cfg_path = _resolve_path(repo_root, args.config)
    run_dir = _resolve_path(repo_root, args.run_dir)
    out_dir = _resolve_path(repo_root, args.output_dir)
    if cfg_path is None or run_dir is None or out_dir is None:
        raise SystemExit("Failed to resolve config/run-dir/output-dir")
    cfg = _read_yaml(cfg_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = _inventory_holdout_samples(cfg, repo_root)
    manifest_path, manifest_meta_path = _write_holdout_manifest(out_dir, inventory)
    eligible_entries = list(inventory["eligible"])
    if not eligible_entries:
        payload = {"status": "insufficient_holdout_data", "inventory": inventory}
        _write_json_atomic(out_dir / "promotion_decision.json", {"status": "insufficient_holdout_data"})
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    loader_split = _build_loader_split_file(out_dir, eligible_entries)
    device = _make_device(cfg, args.device)
    loader = _build_loader(cfg, repo_root=repo_root, split_txt=loader_split, device=device)
    model = _build_model_from_cfg(cfg, repo_root=repo_root)
    ckpt_info = _checkpoint_identity(run_dir)
    if int(ckpt_info["checkpoint_iteration"]) != 75:
        raise SystemExit(json.dumps({"status": "checkpoint_iteration_mismatch", **ckpt_info}, ensure_ascii=False, indent=2))
    incompat = model.load_state_dict(ckpt_info["state_dict"], strict=False)
    missing = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
    unexpected = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
    if unexpected or missing:
        raise SystemExit(f"Checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()

    entry_by_sample = {entry["sample"]: dict(entry, sample_index=idx) for idx, entry in enumerate(eligible_entries)}
    per_sample_rows = []
    per_component_assignments = {"primary_threshold": PRIMARY_THRESHOLD, "secondary_thresholds": list(SECONDARY_THRESHOLDS), "samples": []}

    for batch in loader:
        images = batch["image"].to(device)
        out = model(images)
        sample_id = Path(str(batch["image_path"][0])).stem
        entry = entry_by_sample[sample_id]
        pred_sem = torch.argmax(out["semantic"], dim=1).detach().cpu().numpy()[0].astype(np.uint8)
        center_prob = torch.sigmoid(out["center"]).detach().cpu().numpy()[0, 0].astype(np.float32)
        gt_inst = _load_gt_instance(Path(inventory["instance_root"]), sample_id, pred_sem.shape[:2])
        gt_pts = _extract_metadata_centers(str(batch["metadata_path"][0]))
        sample_pack = {"sample": sample_id, "sample_index": int(entry["sample_index"]), "thresholds": {}}
        for threshold in (PRIMARY_THRESHOLD,) + SECONDARY_THRESHOLDS:
            threshold_key = f"{float(threshold):.2f}"
            sample_pack["thresholds"][threshold_key] = {}
            shared_marker_contract = None
            for policy in ("P0_CURRENT", "P1_DROP_UNMARKED", "P2_ATTACH_TO_NEAREST_MARKER", "P4_GLOBAL_MARKER_CONTROLLED"):
                pred_inst, marker_points, trace = run_policy(policy, pred_sem, center_prob, float(threshold))
                center_metrics = _sample_center_metrics([(int(mp["y"]), int(mp["x"])) for mp in marker_points], gt_pts)
                marker_contract = _marker_contract(gt_inst, marker_points)
                policy_metrics = _policy_metrics(policy_name=policy, gt_inst=gt_inst, pred_sem=pred_sem, pred_inst=pred_inst, marker_points=marker_points, trace=trace)
                _assert_policy_contract(policy, policy_metrics)
                per_sample_rows.append(_row_from_metrics(sample_entry=entry, threshold=float(threshold), policy=policy, marker_contract=marker_contract, center_metrics=center_metrics, policy_metrics=policy_metrics, p3_cfg=None))
                sample_pack["thresholds"][threshold_key][policy] = {"metrics": policy_metrics, "trace": _json_safe_trace(trace)}
                shared_marker_contract = marker_contract
            pred_inst, marker_points, trace = run_policy("P3_GATED_ATTACH", pred_sem, center_prob, float(threshold), p3_cfg=FIXED_P3_CFG)
            center_metrics = _sample_center_metrics([(int(mp["y"]), int(mp["x"])) for mp in marker_points], gt_pts)
            marker_contract = shared_marker_contract or _marker_contract(gt_inst, marker_points)
            policy_metrics = _policy_metrics(policy_name="P3_GATED_ATTACH", gt_inst=gt_inst, pred_sem=pred_sem, pred_inst=pred_inst, marker_points=marker_points, trace=trace)
            _assert_policy_contract("P3_GATED_ATTACH", policy_metrics)
            per_sample_rows.append(_row_from_metrics(sample_entry=entry, threshold=float(threshold), policy="P3_GATED_ATTACH", marker_contract=marker_contract, center_metrics=center_metrics, policy_metrics=policy_metrics, p3_cfg=FIXED_P3_CFG))
            sample_pack["thresholds"][threshold_key]["P3_GATED_ATTACH"] = {"metrics": policy_metrics, "trace": _json_safe_trace(trace), "cfg": dict(FIXED_P3_CFG)}
        per_component_assignments["samples"].append(sample_pack)

    per_sample_rows.sort(key=lambda row: (int(row["sample_index"]), float(row["threshold"]), str(row["policy"])))
    _validate_policy_artifact_integrity(sample_entries=per_component_assignments["samples"], per_sample_csv_rows=per_sample_rows, thresholds=(PRIMARY_THRESHOLD,) + SECONDARY_THRESHOLDS, required_policies=REQUIRED_POLICIES)

    primary_rows = [row for row in per_sample_rows if abs(float(row["threshold"]) - float(PRIMARY_THRESHOLD)) < 1e-9]
    conditioned_primary_rows = _condition_rows_for_marker_contract(primary_rows)
    end_to_end_results = {}
    conditioned_results = {}
    for threshold in (PRIMARY_THRESHOLD,) + SECONDARY_THRESHOLDS:
        threshold_key = f"{float(threshold):.2f}"
        end_to_end_results[threshold_key] = {}
        conditioned_results[threshold_key] = {}
        threshold_rows = [row for row in per_sample_rows if abs(float(row["threshold"]) - float(threshold)) < 1e-9]
        conditioned_rows = _condition_rows_for_marker_contract(threshold_rows)
        for policy in REQUIRED_POLICIES:
            end_to_end_results[threshold_key][policy] = _aggregate_rows([row for row in threshold_rows if row["policy"] == policy])
            conditioned_results[threshold_key][policy] = _aggregate_rows([row for row in conditioned_rows if row["policy"] == policy])

    paired_rows = _paired_p1_vs_p0(primary_rows, PRIMARY_THRESHOLD)
    paired_path = (out_dir / "paired_p1_vs_p0.csv").resolve()
    _write_csv_atomic(paired_path, paired_rows)

    stratified_rows = []
    for scope_name, source_rows in (("end_to_end", primary_rows), ("marker_conditioned", conditioned_primary_rows)):
        for policy in REQUIRED_POLICIES:
            policy_rows = [row for row in source_rows if row["policy"] == policy]
            groups = defaultdict(list)
            for row in policy_rows:
                groups[("gt_instance_count", str(row["gt_instance_count"]))].append(row)
                groups[("semantic_cc_bucket", _semantic_cc_bucket(int(row["semantic_cc_count"])))].append(row)
                groups[("marker_contract", "pass" if bool(row["marker_contract_pass"]) else "fail")].append(row)
                groups[("mouse_id", str(row["mouse_id"]))].append(row)
            for (stratum_type, stratum_value), rows in groups.items():
                agg = _aggregate_rows(rows)
                stratified_rows.append(
                    {
                        "scope": scope_name,
                        "policy": policy,
                        "stratum_type": stratum_type,
                        "stratum_value": stratum_value,
                        "sample_count": agg["sample_count"],
                        "exact_count_accuracy": agg["exact_count_accuracy"],
                        "mean_matched_iou": agg["mean_matched_iou"],
                        "median_matched_iou": agg["median_matched_iou"],
                        "mean_dice": agg["mean_dice"],
                        "median_dice": agg["median_dice"],
                    }
                )
    _write_csv_atomic((out_dir / "stratified_summary.csv").resolve(), stratified_rows)

    worst_iou_rows = sorted(paired_rows, key=lambda row: float(row["matched_iou_delta"]))[:10]
    worst_drop_rows = sorted(paired_rows, key=lambda row: float(row["p1_dropped_area_fraction"]), reverse=True)[:10]
    _write_csv_atomic((out_dir / "worst_iou_regressions.csv").resolve(), worst_iou_rows)
    _write_csv_atomic((out_dir / "worst_dropped_area.csv").resolve(), worst_drop_rows)

    conditioned_primary_by_policy = {
        policy: conditioned_results[f"{PRIMARY_THRESHOLD:.2f}"][policy]
        for policy in REQUIRED_POLICIES
    }
    promotion = _promotion_decision(
        conditioned_primary_by_policy=conditioned_primary_by_policy,
        paired_rows=paired_rows,
        evidence_sufficient=bool(inventory["meets_minimum"]),
    )

    invariants = {
        "primary_threshold_locked": float(PRIMARY_THRESHOLD),
        "secondary_thresholds": list(SECONDARY_THRESHOLDS),
        "fixed_p3_cfg": dict(FIXED_P3_CFG),
        "component_artifact_unique_samples": len(per_component_assignments["samples"]) == len({(s["sample"], s["sample_index"]) for s in per_component_assignments["samples"]}),
    }

    summary = {
        "microset_conclusion": {
            "candidate": "P1_DROP_UNMARKED",
            "exact_count_delta_vs_p0_primary_conditioned": promotion["bootstrap_exact_count_delta_ci95"]["mean"],
            "iou_delta_vs_p0_primary_conditioned": promotion["bootstrap_mean_iou_delta_ci95"]["mean"],
            "known_trade_off": "P1 fixes count/provenance invariants by dropping unmarked fragments; held-out evaluation quantifies the IoU and dropped-area cost.",
        },
        "holdout_inventory": {
            "eligible": len(eligible_entries),
            "excluded_microset": len(inventory["excluded_microset"]),
            "missing": len(inventory["missing"]),
            "evidence_level": inventory["evidence_level"],
            "gt_instance_distribution": inventory["gt_instance_distribution"],
        },
        "locked_operating_point": {
            "checkpoint": ckpt_info["checkpoint_path"],
            "iteration": ckpt_info["checkpoint_iteration"],
            "primary_threshold": float(PRIMARY_THRESHOLD),
            "saved_threshold": ckpt_info["saved_best_threshold"],
            "saved_threshold_note": "checkpoint.saved_best_threshold reflects the micro-overfit center-F1 optimum; held-out reconstruction validation keeps the authoritative policy operating point locked at 0.03.",
        },
        "end_to_end_results": end_to_end_results,
        "marker_conditioned_results": conditioned_results,
        "paired_p1_vs_p0_primary": {
            "sample_count": len(paired_rows),
            "bootstrap_mean_iou_delta_ci95": promotion["bootstrap_mean_iou_delta_ci95"],
            "bootstrap_exact_count_delta_ci95": promotion["bootstrap_exact_count_delta_ci95"],
        },
        "promotion_decision": promotion,
    }

    _write_csv_atomic((out_dir / "per_sample_policy_metrics.csv").resolve(), per_sample_rows)
    _write_json_atomic((out_dir / "per_component_assignments.json").resolve(), per_component_assignments)
    _write_json_atomic((out_dir / "invariants.json").resolve(), invariants)
    _write_json_atomic((out_dir / "validation_summary.json").resolve(), summary)
    _write_json_atomic((out_dir / "promotion_decision.json").resolve(), promotion)

    worst_sample_ids = sorted({row["sample"] for row in worst_iou_rows} | {row["sample"] for row in worst_drop_rows})
    visual_dir = (out_dir / "visual_review").resolve()
    visual_dir.mkdir(parents=True, exist_ok=True)
    if worst_sample_ids:
        selected = set(worst_sample_ids)
        for batch in loader:
            sample_id = Path(str(batch["image_path"][0])).stem
            if sample_id not in selected:
                continue
            images = batch["image"].to(device)
            out = model(images)
            pred_sem = torch.argmax(out["semantic"], dim=1).detach().cpu().numpy()[0].astype(np.uint8)
            center_prob = torch.sigmoid(out["center"]).detach().cpu().numpy()[0, 0].astype(np.float32)
            gt_inst = _load_gt_instance(Path(inventory["instance_root"]), sample_id, pred_sem.shape[:2])
            image_rgb_u8 = (np.clip(batch["image"].detach().cpu().numpy()[0].transpose(1, 2, 0), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            p0_final, marker_points, p0_trace = run_policy("P0_CURRENT", pred_sem, center_prob, PRIMARY_THRESHOLD)
            p1_final, _marker_points, p1_trace = run_policy("P1_DROP_UNMARKED", pred_sem, center_prob, PRIMARY_THRESHOLD)
            p0_metrics = _policy_metrics(policy_name="P0_CURRENT", gt_inst=gt_inst, pred_sem=pred_sem, pred_inst=p0_final, marker_points=marker_points, trace=p0_trace)
            p1_metrics = _policy_metrics(policy_name="P1_DROP_UNMARKED", gt_inst=gt_inst, pred_sem=pred_sem, pred_inst=p1_final, marker_points=marker_points, trace=p1_trace)
            panel = _worst_case_panel(
                sample=sample_id,
                image_rgb_u8=image_rgb_u8,
                gt_inst=gt_inst,
                pred_sem=pred_sem,
                semantic_cc=p0_trace["semantic_components"],
                center_prob=center_prob,
                marker_points=marker_points,
                p0_trace=p0_trace,
                p0_final=p0_final,
                p1_final=p1_final,
                p0_iou=float(p0_metrics["instance_metrics"]["matched_iou"]),
                p1_iou=float(p1_metrics["instance_metrics"]["matched_iou"]),
                gt_count=int(p0_metrics["counts"]["gt_instance_count"]),
                marker_count=int(p0_metrics["counts"]["marker_count"]),
            )
            cv2.imwrite(str((visual_dir / f"{sample_id}.png").resolve()), panel)

    print(
        json.dumps(
            {
                "status": "done",
                "output_dir": str(out_dir),
                "holdout_manifest": str(manifest_path),
                "holdout_manifest_metadata": str(manifest_meta_path),
                "eligible": len(eligible_entries),
                "promotion_status": promotion["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
