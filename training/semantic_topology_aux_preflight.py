from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

import semantic_topology_aux as topo_aux


DEFAULT_PREFLIGHT_OUTPUT_DIR = topo_aux.REPO_ROOT / "training" / "analysis" / "semantic_topology_aux_preflight"
DEFAULT_SMOKE_CONFIG_PATH = topo_aux.REPO_ROOT / "training" / "configs" / "unetpp_effb3_semantic_topology_aux_finetune_cuda_smoke.yaml"


def _patient_id_from_sample(sample_id: str) -> str:
    parts = str(sample_id).split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else str(sample_id)


def _split_identity(split_txt: Path) -> dict[str, Any]:
    samples: list[str] = []
    with split_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_rel, _mask_rel = line.split("\t")
            samples.append(Path(image_rel).stem)
    patients = sorted({_patient_id_from_sample(sample) for sample in samples})
    return {
        "path": str(split_txt.resolve()),
        "sha256": topo_aux._sha256_file(split_txt.resolve()),
        "samples": sorted(samples),
        "patients": patients,
        "sample_count": int(len(samples)),
        "patient_count": int(len(patients)),
    }


def _manifest_identity(manifest_path: Path) -> dict[str, Any]:
    rows = topo_aux._read_jsonl(manifest_path.resolve())
    if any(bool(row.get("present_in_authoritative_106_holdout", False)) for row in rows):
        raise SystemExit("Authoritative holdout samples are not allowed in topology reconstruction validation")
    samples = sorted(str(row["sample"]) for row in rows)
    patients = sorted({str(row["patient_id"]) for row in rows})
    return {
        "path": str(manifest_path.resolve()),
        "sha256": topo_aux._sha256_file(manifest_path.resolve()),
        "samples": samples,
        "patients": patients,
        "sample_count": int(len(samples)),
        "patient_count": int(len(patients)),
    }


def _pair_overlap(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    sample_overlap = sorted(set(left["samples"]) & set(right["samples"]))
    patient_overlap = sorted(set(left["patients"]) & set(right["patients"]))
    return {
        "left_path": left["path"],
        "right_path": right["path"],
        "sample_overlap_count": int(len(sample_overlap)),
        "patient_overlap_count": int(len(patient_overlap)),
        "sample_overlap_ids": sample_overlap,
        "patient_overlap_ids": patient_overlap,
    }


def audit_data_isolation(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = cfg.get("dataset") or {}
    train_txt = topo_aux._resolve_repo_path(dataset_cfg.get("train_txt", topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT), topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT)
    val_txt = topo_aux._resolve_repo_path(dataset_cfg.get("val_txt", topo_aux.DEFAULT_SEMANTIC_VAL_SPLIT), topo_aux.DEFAULT_SEMANTIC_VAL_SPLIT)
    test_txt = topo_aux._resolve_repo_path(dataset_cfg.get("test_txt", topo_aux.REPO_ROOT / "datasets" / "converted_full_multiclass_curated" / "test.txt"), topo_aux.REPO_ROOT / "datasets" / "converted_full_multiclass_curated" / "test.txt")
    research_manifest = topo_aux._resolve_repo_path(dataset_cfg.get("research_val_manifest", topo_aux.DEFAULT_RESEARCH_MANIFEST), topo_aux.DEFAULT_RESEARCH_MANIFEST)

    semantic_train = _split_identity(train_txt)
    semantic_val = _split_identity(val_txt)
    semantic_test = _split_identity(test_txt)
    topology_reconstruction_val = _manifest_identity(research_manifest)

    train_val = _pair_overlap(semantic_train, semantic_val)
    train_topology = _pair_overlap(semantic_train, topology_reconstruction_val)
    val_topology = _pair_overlap(semantic_val, topology_reconstruction_val)

    center_val_rows = topo_aux._read_jsonl(research_manifest.resolve())
    center_val_minus_train = sorted(str(row["sample"]) for row in center_val_rows if str(row["sample"]) not in set(semantic_train["samples"]))
    center_val_minus_train_patients = sorted({_patient_id_from_sample(sample) for sample in center_val_minus_train})
    semantic_test_vs_train = _pair_overlap(semantic_train, semantic_test)
    semantic_test_vs_topology = _pair_overlap(semantic_test, topology_reconstruction_val)

    proposal = {
        "blocked_due_to_sample_overlap": bool(train_topology["sample_overlap_count"] > 0),
        "smallest_valid_adjustment": (
            "Do not use training/manifests/center_full_val_manifest.jsonl for checkpoint selection because it overlaps 43 train samples. "
            "Keep semantic checkpointing on semantic val, and if an oracle-K selection set is still required, build a new research-only manifest from the existing semantic test.txt IDs "
            "(20 samples, 0 sample overlap with semantic train) rather than inventing a new split silently."
        ),
        "existing_disjoint_repository_data": {
            "semantic_test": semantic_test,
            "semantic_test_vs_train": semantic_test_vs_train,
            "semantic_test_vs_current_topology_manifest": semantic_test_vs_topology,
            "current_topology_manifest_minus_semantic_train_samples": {
                "sample_count": int(len(center_val_minus_train)),
                "samples": center_val_minus_train,
                "patient_count": int(len(center_val_minus_train_patients)),
                "patients": center_val_minus_train_patients,
            },
        },
    }
    return {
        "semantic_train": semantic_train,
        "semantic_val": semantic_val,
        "topology_reconstruction_val": topology_reconstruction_val,
        "pairs": {
            "train_vs_val": train_val,
            "train_vs_topology_reconstruction_val": train_topology,
            "val_vs_topology_reconstruction_val": val_topology,
        },
        "proposal": proposal,
    }


def _target_component_row(sample_id: str, gt_count: int, instance_mask: np.ndarray, parts: dict[str, np.ndarray], image_shape: tuple[int, int]) -> dict[str, Any]:
    boundary = parts["boundary"] > 0
    separation = parts["separation"] > 0
    narrow = parts["narrow"] > 0
    total_px = int(image_shape[0] * image_shape[1])
    outer_boundary = int(np.count_nonzero(boundary & (instance_mask == 0)))
    separation_count = int(np.count_nonzero(separation))
    narrow_count = int(np.count_nonzero(narrow))
    boundary_sep = int(np.count_nonzero(boundary & separation))
    boundary_narrow = int(np.count_nonzero(boundary & narrow))
    separation_narrow = int(np.count_nonzero(separation & narrow))
    unique_separation = int(np.count_nonzero(separation & ~boundary & ~narrow))
    unique_narrow = int(np.count_nonzero(narrow & ~boundary & ~separation))
    union_count = int(np.count_nonzero(boundary | separation | narrow))

    def frac(value: int) -> float:
        return float(value / max(total_px, 1))

    return {
        "sample_id": sample_id,
        "gt_count": int(gt_count),
        "image_pixel_count": total_px,
        "outer_boundary_count": outer_boundary,
        "outer_boundary_fraction": frac(outer_boundary),
        "separation_count": separation_count,
        "separation_fraction": frac(separation_count),
        "narrow_count": narrow_count,
        "narrow_fraction": frac(narrow_count),
        "boundary_and_separation_count": boundary_sep,
        "boundary_and_separation_fraction": frac(boundary_sep),
        "boundary_and_narrow_count": boundary_narrow,
        "boundary_and_narrow_fraction": frac(boundary_narrow),
        "separation_and_narrow_count": separation_narrow,
        "separation_and_narrow_fraction": frac(separation_narrow),
        "unique_separation_count": unique_separation,
        "unique_separation_fraction": frac(unique_separation),
        "unique_narrow_count": unique_narrow,
        "unique_narrow_fraction": frac(unique_narrow),
        "union_count": union_count,
        "union_fraction": frac(union_count),
    }


def _aggregate_component_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    keys = [
        "outer_boundary_count",
        "separation_count",
        "narrow_count",
        "boundary_and_separation_count",
        "boundary_and_narrow_count",
        "separation_and_narrow_count",
        "unique_separation_count",
        "unique_narrow_count",
        "union_count",
    ]
    image_pixels = int(sum(int(row["image_pixel_count"]) for row in rows))
    out: dict[str, Any] = {
        "sample_count": int(len(rows)),
        "image_pixels_total": int(image_pixels),
    }
    for key in keys:
        total = int(sum(int(row[key]) for row in rows))
        out[key] = total
        out[key.replace("_count", "_fraction")] = float(total / max(image_pixels, 1))
    return out


def _select_target_examples(rows: list[dict[str, Any]]) -> dict[str, str]:
    candidates: dict[str, tuple[float, str]] = {
        "gt1": (float("-inf"), ""),
        "gt2": (float("-inf"), ""),
        "gt3": (float("-inf"), ""),
        "narrow_leaflet": (float("-inf"), ""),
        "close_neighbors": (float("-inf"), ""),
        "disconnected_same_leaflet": (float("-inf"), ""),
    }
    for row in rows:
        sample_id = str(row["sample_id"])
        gt_count = int(row["gt_count"])
        if gt_count == 1 and float(row["outer_boundary_fraction"]) > candidates["gt1"][0]:
            candidates["gt1"] = (float(row["outer_boundary_fraction"]), sample_id)
        if gt_count == 2 and float(row["separation_fraction"]) > candidates["gt2"][0]:
            candidates["gt2"] = (float(row["separation_fraction"]), sample_id)
        if gt_count == 3 and float(row["separation_fraction"]) > candidates["gt3"][0]:
            candidates["gt3"] = (float(row["separation_fraction"]), sample_id)
        if float(row["unique_narrow_fraction"]) > candidates["narrow_leaflet"][0]:
            candidates["narrow_leaflet"] = (float(row["unique_narrow_fraction"]), sample_id)
        if float(row["unique_separation_fraction"]) > candidates["close_neighbors"][0]:
            candidates["close_neighbors"] = (float(row["unique_separation_fraction"]), sample_id)
        if float(row.get("disconnected_same_leaflet", 0.0)) > candidates["disconnected_same_leaflet"][0]:
            candidates["disconnected_same_leaflet"] = (float(row.get("disconnected_same_leaflet", 0.0)), sample_id)
    return {key: sample_id for key, (_score, sample_id) in candidates.items() if sample_id}


def audit_target_composition(cfg: dict[str, Any], contract: topo_aux.TopologyTargetContract, output_dir: Path) -> dict[str, Any]:
    dataset_cfg = cfg.get("dataset") or {}
    dataset_root = topo_aux._resolve_repo_path(dataset_cfg.get("root", topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT), topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT)
    train_split_txt = topo_aux._resolve_repo_path(dataset_cfg.get("train_txt", topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT), topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT)
    instance_root = topo_aux._resolve_repo_path(dataset_cfg.get("instance_root", topo_aux.DEFAULT_INSTANCE_ROOT), topo_aux.DEFAULT_INSTANCE_ROOT)
    items = topo_aux.read_split_file(dataset_root.resolve(), train_split_txt.resolve())
    rows: list[dict[str, Any]] = []
    rows_by_sample: dict[str, dict[str, Any]] = {}
    for item in items:
        sample_id = Path(item.image_path).stem
        instance_mask = topo_aux._read_u8(instance_root / "instance_masks" / f"{sample_id}.png")
        gt_count = len(topo_aux._positive_instance_ids(instance_mask))
        target, parts = topo_aux.generate_topology_target(instance_mask, contract, return_parts=True)
        comp_counts = topo_aux._instance_component_counts(instance_mask)
        row = _target_component_row(sample_id, gt_count, instance_mask, parts, instance_mask.shape[:2])
        row["disconnected_same_leaflet"] = int(any(int(v) > 1 for v in comp_counts.values()))
        rows.append(row)
        rows_by_sample[sample_id] = row

    aggregate = _aggregate_component_rows(rows)
    gt1_rows = [row for row in rows if int(row["gt_count"]) == 1]
    gt2_rows = [row for row in rows if int(row["gt_count"]) == 2]
    gt3_rows = [row for row in rows if int(row["gt_count"]) == 3]

    examples_dir = output_dir / "target_composition_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_target_examples(rows)
    saved_files: list[str] = []
    for category, sample_id in selected.items():
        matching = next(item for item in items if Path(item.image_path).stem == sample_id)
        rgb = topo_aux._center_crop_like_validation(topo_aux._read_image_rgb(matching.image_path), 768, 768, is_mask=False)
        instance_mask = topo_aux._center_crop_like_validation(topo_aux._read_u8(instance_root / "instance_masks" / f"{sample_id}.png"), 768, 768, is_mask=True)
        target, parts = topo_aux.generate_topology_target(instance_mask, contract, return_parts=True)
        boundary = (parts["boundary"] > 0).astype(np.uint8)
        separation = (parts["separation"] > 0).astype(np.uint8)
        narrow = (parts["narrow"] > 0).astype(np.uint8)
        overlay = rgb.copy().astype(np.float32)
        overlay[boundary.astype(bool)] = overlay[boundary.astype(bool)] * 0.5 + np.asarray([255.0, 255.0, 0.0], dtype=np.float32) * 0.5
        overlay[separation.astype(bool)] = overlay[separation.astype(bool)] * 0.4 + np.asarray([255.0, 0.0, 255.0], dtype=np.float32) * 0.6
        overlay[narrow.astype(bool)] = overlay[narrow.astype(bool)] * 0.4 + np.asarray([255.0, 0.0, 0.0], dtype=np.float32) * 0.6
        overlay = overlay.astype(np.uint8)

        def _mask_rgb(mask01: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
            out = np.zeros((mask01.shape[0], mask01.shape[1], 3), dtype=np.uint8)
            out[mask01 > 0] = np.asarray(color, dtype=np.uint8)
            return out

        inst_vis = cv2.applyColorMap(((instance_mask.astype(np.float32) / max(float(instance_mask.max()), 1.0)) * 255.0 + 0.5).astype(np.uint8), cv2.COLORMAP_TURBO)
        panel_top = np.concatenate(
            [
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                inst_vis,
                cv2.cvtColor(_mask_rgb(boundary, (255, 255, 0)), cv2.COLOR_RGB2BGR),
            ],
            axis=1,
        )
        panel_bottom = np.concatenate(
            [
                cv2.cvtColor(_mask_rgb(separation, (255, 0, 255)), cv2.COLOR_RGB2BGR),
                cv2.cvtColor(_mask_rgb(narrow, (255, 0, 0)), cv2.COLOR_RGB2BGR),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            ],
            axis=1,
        )
        grid = np.concatenate([panel_top, panel_bottom], axis=0)
        out_path = examples_dir / f"{category}_{sample_id}.png"
        cv2.imwrite(str(out_path), grid)
        saved_files.append(str(out_path.resolve()))

    return {
        "aggregate": aggregate,
        "gt1": _aggregate_component_rows(gt1_rows),
        "gt2": _aggregate_component_rows(gt2_rows),
        "gt3": _aggregate_component_rows(gt3_rows),
        "per_sample_rows": rows,
        "selected_examples": selected,
        "saved_files": saved_files,
    }


def _median_range(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _grad_norm(named_params: list[tuple[str, torch.nn.Parameter]]) -> float:
    return float(topo_aux._named_grad_l2_norm(named_params))


def _build_gradient_audit_batches(cfg: dict[str, Any], contract: topo_aux.TopologyTargetContract) -> list[dict[str, torch.Tensor]]:
    dataset_cfg = cfg.get("dataset") or {}
    dataset_root = topo_aux._resolve_repo_path(dataset_cfg.get("root", topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT), topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT)
    train_split_txt = topo_aux._resolve_repo_path(dataset_cfg.get("train_txt", topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT), topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT)
    instance_root = topo_aux._resolve_repo_path(dataset_cfg.get("instance_root", topo_aux.DEFAULT_INSTANCE_ROOT), topo_aux.DEFAULT_INSTANCE_ROOT)
    ds = topo_aux.SemanticTopologyAuxDataset(
        dataset_root=dataset_root,
        split_txt=train_split_txt,
        instance_root=instance_root,
        contract=contract,
        num_classes=int(cfg["model"]["classes"]),
        input_size=int(cfg["model"]["input_size"]),
        augment_cfg=None,
        training=False,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, drop_last=False)
    batches: list[dict[str, torch.Tensor]] = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 4:
            break
        batches.append(batch)
    return batches


def _single_lambda_gradient_audit(cfg: dict[str, Any], contract: topo_aux.TopologyTargetContract, device: torch.device, lambda_topology: float) -> dict[str, Any]:
    torch.manual_seed(int(cfg.get("seed", 1337)))
    model = topo_aux.build_model_from_cfg(cfg).to(device)
    topo_aux.load_semantic_checkpoint(
        model,
        topo_aux._resolve_repo_path((cfg.get("train") or {}).get("init_checkpoint", topo_aux.DEFAULT_SEMANTIC_CHECKPOINT), topo_aux.DEFAULT_SEMANTIC_CHECKPOINT),
    )
    freeze_info = topo_aux.apply_training_policy(model, cfg)
    topo_aux.set_train_modes(model, freeze_info)
    semantic_loss = topo_aux.build_semantic_loss_from_cfg(cfg, device)
    raw_topology_loss = topo_aux.BinaryBCEDiceLoss(
        bce_weight=float((cfg.get("topology_aux") or {}).get("topology_bce_weight", 1.0)),
        dice_weight=float((cfg.get("topology_aux") or {}).get("topology_dice_weight", 1.0)),
    ).to(device)
    use_amp = topo_aux._amp_enabled(cfg, device)

    x04_named = [(name, p) for name, p in model.named_parameters() if name in set(freeze_info["selected_decoder_param_names"])]
    seg_named = [(name, p) for name, p in model.named_parameters() if name.startswith("base.segmentation_head.")]
    topo_named = [(name, p) for name, p in model.named_parameters() if name.startswith("topology_head.")]
    batches = _build_gradient_audit_batches(cfg, contract)

    rows: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(batches):
        images = batch["image"].to(device)
        semantic_target = batch["mask"].to(device)
        topology_target = batch["topology_target"].to(device)

        def _run(which: str) -> dict[str, float]:
            model.zero_grad(set_to_none=True)
            topo_aux.set_train_modes(model, freeze_info)
            with topo_aux._autocast_ctx(device, enabled=use_amp):
                outputs = model(images)
                sem = semantic_loss(outputs["semantic_logits"], semantic_target)
                topo = raw_topology_loss(outputs["topology_logits"], topology_target)
                if which == "semantic":
                    loss = sem
                elif which == "raw_topology":
                    loss = topo
                elif which == "weighted_topology":
                    loss = float(lambda_topology) * topo
                elif which == "combined":
                    loss = sem + float(lambda_topology) * topo
                else:
                    raise KeyError(which)
            loss.backward()
            return {
                "x0_4_grad_norm": _grad_norm(x04_named),
                "segmentation_head_grad_norm": _grad_norm(seg_named),
                "topology_head_grad_norm": _grad_norm(topo_named),
                "semantic_loss": float(sem.detach().cpu().item()),
                "raw_topology_loss": float(topo.detach().cpu().item()),
            }

        semantic_stats = _run("semantic")
        raw_topology_stats = _run("raw_topology")
        weighted_topology_stats = _run("weighted_topology")
        combined_stats = _run("combined")
        ratio = float(weighted_topology_stats["x0_4_grad_norm"] / max(semantic_stats["x0_4_grad_norm"], 1e-12))
        rows.append(
            {
                "batch_index": int(batch_idx),
                "semantic_x0_4_grad_norm": semantic_stats["x0_4_grad_norm"],
                "semantic_segmentation_head_grad_norm": semantic_stats["segmentation_head_grad_norm"],
                "raw_topology_x0_4_grad_norm": raw_topology_stats["x0_4_grad_norm"],
                "raw_topology_topology_head_grad_norm": raw_topology_stats["topology_head_grad_norm"],
                "weighted_topology_x0_4_grad_norm": weighted_topology_stats["x0_4_grad_norm"],
                "combined_x0_4_grad_norm": combined_stats["x0_4_grad_norm"],
                "combined_segmentation_head_grad_norm": combined_stats["segmentation_head_grad_norm"],
                "combined_topology_head_grad_norm": combined_stats["topology_head_grad_norm"],
                "semantic_loss": semantic_stats["semantic_loss"],
                "raw_topology_loss": semantic_stats["raw_topology_loss"],
                "weighted_topology_to_semantic_x0_4_ratio": ratio,
            }
        )

    ratio_stats = _median_range([float(row["weighted_topology_to_semantic_x0_4_ratio"]) for row in rows])
    return {
        "lambda_topology": float(lambda_topology),
        "batch_count": int(len(rows)),
        "batch_size": 4,
        "rows": rows,
        "semantic_x0_4_grad_norm": _median_range([float(row["semantic_x0_4_grad_norm"]) for row in rows]),
        "semantic_segmentation_head_grad_norm": _median_range([float(row["semantic_segmentation_head_grad_norm"]) for row in rows]),
        "raw_topology_x0_4_grad_norm": _median_range([float(row["raw_topology_x0_4_grad_norm"]) for row in rows]),
        "raw_topology_topology_head_grad_norm": _median_range([float(row["raw_topology_topology_head_grad_norm"]) for row in rows]),
        "weighted_topology_x0_4_grad_norm": _median_range([float(row["weighted_topology_x0_4_grad_norm"]) for row in rows]),
        "combined_x0_4_grad_norm": _median_range([float(row["combined_x0_4_grad_norm"]) for row in rows]),
        "combined_segmentation_head_grad_norm": _median_range([float(row["combined_segmentation_head_grad_norm"]) for row in rows]),
        "combined_topology_head_grad_norm": _median_range([float(row["combined_topology_head_grad_norm"]) for row in rows]),
        "weighted_topology_to_semantic_x0_4_ratio": ratio_stats,
    }


def audit_gradient_contribution(cfg: dict[str, Any], contract: topo_aux.TopologyTargetContract, device: torch.device) -> dict[str, Any]:
    primary_lambda = float((cfg.get("topology_aux") or {}).get("lambda_topology", 0.2))
    primary = _single_lambda_gradient_audit(cfg, contract, device, lambda_topology=primary_lambda)
    dominance_rule = "If median(weighted_topology_x0_4_grad_norm / semantic_x0_4_grad_norm) > 1.0, test exactly one conservative alternative lambda=0.1."
    alternative = None
    if float(primary["weighted_topology_to_semantic_x0_4_ratio"]["median"] or 0.0) > 1.0:
        alternative = _single_lambda_gradient_audit(cfg, contract, device, lambda_topology=0.1)
    selected_lambda, reason = select_lambda_from_gradient_summaries(primary_lambda, primary, alternative)
    return {
        "selection_rule": dominance_rule,
        "lambda_0p2": primary,
        "lambda_0p1": alternative,
        "selected_lambda": float(selected_lambda),
        "reason": reason,
    }


def select_lambda_from_gradient_summaries(
    primary_lambda: float,
    primary_summary: dict[str, Any],
    alternative_summary: dict[str, Any] | None,
) -> tuple[float, str]:
    primary_ratio = float(primary_summary["weighted_topology_to_semantic_x0_4_ratio"]["median"] or 0.0)
    if primary_ratio <= 1.0:
        return float(primary_lambda), "Retained configured lambda because weighted topology and semantic x_0_4 gradients remained comparable."
    if alternative_summary is None:
        return float(primary_lambda), "Retained configured lambda because no conservative alternative audit was available."
    alt_ratio = float(alternative_summary["weighted_topology_to_semantic_x0_4_ratio"]["median"] or 0.0)
    if alt_ratio <= 1.0:
        return 0.1, "Selected lambda=0.1 because lambda=0.2 made weighted topology dominate the shared x_0_4 median gradient, while 0.1 kept topology influential but not dominant."
    return float(primary_lambda), "Retained lambda=0.2 because lambda=0.1 did not restore a comparable shared-decoder gradient balance."


def audit_freeze_mode(cfg: dict[str, Any]) -> dict[str, Any]:
    model = topo_aux.build_model_from_cfg(cfg)
    freeze_info = topo_aux.apply_training_policy(model, cfg)
    sequence: list[dict[str, Any]] = []
    for label in ["after_set_train_modes", "after_eval_then_reset"]:
        if label == "after_eval_then_reset":
            model.eval()
        topo_aux.set_train_modes(model, freeze_info)
        modules = dict(model.named_modules())
        frozen_decoder_prefixes = [
            name for name, _module in model.named_modules()
            if name.startswith("base.decoder.blocks.") and len(name.split(".")) == 4 and name not in set(freeze_info["trainable_decoder_modules"])
        ]
        frozen_bn_train = []
        for name, module in model.named_modules():
            running_mean = getattr(module, "running_mean", None)
            running_var = getattr(module, "running_var", None)
            if running_mean is None or running_var is None:
                continue
            is_trainable_scope = (
                name.startswith("topology_head.")
                or name.startswith("base.segmentation_head.")
                or any(name == prefix or name.startswith(prefix + ".") for prefix in freeze_info["trainable_decoder_modules"])
            )
            if not is_trainable_scope and bool(module.training):
                frozen_bn_train.append(name)
        sequence.append(
            {
                "label": label,
                "encoder_eval": bool(not model.base.encoder.training),
                "frozen_decoder_eval": bool(all(not modules[prefix].training for prefix in frozen_decoder_prefixes)),
                "x_0_4_train": bool(modules["base.decoder.blocks.x_0_4"].training),
                "segmentation_head_train": bool(model.base.segmentation_head.training),
                "topology_head_train": bool(model.topology_head.training),
                "frozen_bn_train_count": int(len(frozen_bn_train)),
                "frozen_bn_train_modules": frozen_bn_train,
            }
        )
    return {
        "sequence": sequence,
        "status": bool(all(int(entry["frozen_bn_train_count"]) == 0 and entry["encoder_eval"] and entry["frozen_decoder_eval"] and entry["x_0_4_train"] and entry["segmentation_head_train"] and entry["topology_head_train"] for entry in sequence)),
    }


def trace_schedule_contract(baseline_cfg: dict[str, Any], topology_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": {
            "optimizer": "AdamW(model.parameters(), lr=train.lr, weight_decay=train.weight_decay)",
            "scheduler": baseline_cfg.get("scheduler"),
            "early_stopping": baseline_cfg.get("early_stopping"),
            "epochs": int((baseline_cfg.get("train") or {}).get("epochs", 0)),
        },
        "topology_config": {
            "optimizer": {
                "type": "AdamW(parameter_groups)",
                "groups": {
                    "decoder": float((topology_cfg.get("train") or {}).get("lr_decoder", (topology_cfg.get("train") or {}).get("lr", 1.0e-5))),
                    "segmentation_head": float((topology_cfg.get("train") or {}).get("lr_segmentation_head", (topology_cfg.get("train") or {}).get("lr", 1.0e-5))),
                    "topology_head": float((topology_cfg.get("train") or {}).get("lr_topology_head", 1.0e-4)),
                },
                "weight_decay": float((topology_cfg.get("train") or {}).get("weight_decay", 1.0e-5)),
            },
            "scheduler": topology_cfg.get("scheduler"),
            "early_stopping": topology_cfg.get("early_stopping"),
            "epochs": int((topology_cfg.get("train") or {}).get("epochs", 0)),
        },
    }


def write_smoke_config(cfg: dict[str, Any], selected_lambda: float, smoke_config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise SystemExit("PyYAML is required for smoke config generation") from e

    smoke_cfg = json.loads(json.dumps(cfg))
    smoke_cfg.setdefault("topology_aux", {})["lambda_topology"] = float(selected_lambda)
    smoke_cfg.setdefault("train", {})["save_dir"] = "training/runs/unetpp_effb3_semantic_topology_aux_finetune_100ep_cuda_smoke"
    smoke_cfg["train"]["dry_run_steps"] = 1
    smoke_cfg["train"]["epochs"] = 1
    smoke_config_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_config_path.write_text(yaml.safe_dump(smoke_cfg, sort_keys=False), encoding="utf-8")
    return {
        "path": str(smoke_config_path.resolve()),
        "sha256": topo_aux._sha256_file(smoke_config_path.resolve()),
        "save_dir": str(smoke_cfg["train"]["save_dir"]),
    }


def run_preflight(cfg: dict[str, Any], baseline_cfg: dict[str, Any], output_dir: Path, device: torch.device) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_isolation = audit_data_isolation(cfg)
    dataset_cfg = cfg.get("dataset") or {}
    contract, contract_audit = topo_aux.choose_topology_target_contract(
        dataset_root=topo_aux._resolve_repo_path(dataset_cfg.get("root", topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT), topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT),
        train_split_txt=topo_aux._resolve_repo_path(dataset_cfg.get("train_txt", topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT), topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT),
        instance_root=topo_aux._resolve_repo_path(dataset_cfg.get("instance_root", topo_aux.DEFAULT_INSTANCE_ROOT), topo_aux.DEFAULT_INSTANCE_ROOT),
    )
    target_composition = audit_target_composition(cfg, contract, output_dir)
    gradient_contribution = audit_gradient_contribution(cfg, contract, device)
    freeze_mode = audit_freeze_mode(cfg)
    schedule_contract = trace_schedule_contract(baseline_cfg, cfg)
    smoke_config = write_smoke_config(cfg, float(gradient_contribution["selected_lambda"]), DEFAULT_SMOKE_CONFIG_PATH)

    train_save_dir = topo_aux._resolve_repo_path((cfg.get("train") or {}).get("save_dir"), topo_aux.REPO_ROOT / "training" / "runs" / "semantic_topology_aux")
    full_run_dir_exists = bool(train_save_dir.exists())
    train_topology_overlap = int(dataset_isolation["pairs"]["train_vs_topology_reconstruction_val"]["sample_overlap_count"])
    readiness = "blocked" if train_topology_overlap > 0 else "ready_for_A100_smoke"

    summary = {
        "dataset_isolation": dataset_isolation,
        "target_contract": topo_aux.asdict(contract),
        "target_contract_audit": contract_audit,
        "target_composition": target_composition,
        "gradient_contribution": gradient_contribution,
        "schedule_contract": schedule_contract,
        "freeze_mode": freeze_mode,
        "semantic_checkpoint_sha256": topo_aux._sha256_file(topo_aux._resolve_repo_path((cfg.get("train") or {}).get("init_checkpoint", topo_aux.DEFAULT_SEMANTIC_CHECKPOINT), topo_aux.DEFAULT_SEMANTIC_CHECKPOINT)),
        "smoke_config": smoke_config,
        "full_run_save_dir": str((cfg.get("train") or {}).get("save_dir")),
        "full_run_dir_exists": full_run_dir_exists,
        "a100_smoke_command": f"py -3 training/train_semantic_topology_aux.py --config {Path(smoke_config['path']).relative_to(topo_aux.REPO_ROOT)} --smoke-test",
        "training_readiness": readiness,
    }
    topo_aux._write_json(output_dir / "dataset_isolation.json", dataset_isolation)
    topo_aux._write_json(output_dir / "target_contract.json", topo_aux.asdict(contract))
    topo_aux._write_json(output_dir / "target_contract_audit.json", contract_audit)
    topo_aux._write_json(output_dir / "target_composition.json", {k: v for k, v in target_composition.items() if k != "per_sample_rows"})
    topo_aux._write_csv(output_dir / "target_composition_rows.csv", target_composition["per_sample_rows"])
    topo_aux._write_json(output_dir / "gradient_contribution.json", gradient_contribution)
    topo_aux._write_json(output_dir / "freeze_mode.json", freeze_mode)
    topo_aux._write_json(output_dir / "schedule_contract.json", schedule_contract)
    topo_aux._write_json(output_dir / "preflight_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=topo_aux.REPO_ROOT / "training" / "configs" / "unetpp_effb3_semantic_topology_aux_finetune_100ep.yaml",
    )
    ap.add_argument(
        "--baseline-config",
        type=Path,
        default=topo_aux.DEFAULT_BASELINE_CONFIG,
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PREFLIGHT_OUTPUT_DIR,
    )
    args = ap.parse_args()

    cfg = topo_aux._read_yaml(args.config.resolve())
    baseline_cfg = topo_aux._read_yaml(args.baseline_config.resolve())
    device = torch.device("cpu")
    summary = run_preflight(cfg=cfg, baseline_cfg=baseline_cfg, output_dir=args.output_dir.resolve(), device=device)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
