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
            if not line.strip():
                continue
            image_rel, _mask_rel = line.strip().split("\t")
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


def audit_research_eval_isolation(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = cfg.get("dataset") or {}
    train_txt = topo_aux._resolve_repo_path(dataset_cfg.get("train_txt", topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT), topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT)
    val_txt = topo_aux._resolve_repo_path(dataset_cfg.get("val_txt", topo_aux.DEFAULT_SEMANTIC_VAL_SPLIT), topo_aux.DEFAULT_SEMANTIC_VAL_SPLIT)
    research_eval_split_txt = topo_aux._resolve_repo_path(dataset_cfg.get("research_eval_split_txt", topo_aux.DEFAULT_SEMANTIC_TEST_SPLIT), topo_aux.DEFAULT_SEMANTIC_TEST_SPLIT)
    instance_root = topo_aux._resolve_repo_path(dataset_cfg.get("instance_root", topo_aux.DEFAULT_INSTANCE_ROOT), topo_aux.DEFAULT_INSTANCE_ROOT)

    semantic_train = _split_identity(train_txt)
    semantic_val = _split_identity(val_txt)
    research_eval = _split_identity(research_eval_split_txt)

    instance_mask_available = 0
    gt_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
    gt_count_available = 0
    missing_instance_masks: list[str] = []
    for sample_id in research_eval["samples"]:
        instance_path = instance_root / "instance_masks" / f"{sample_id}.png"
        if not instance_path.exists():
            missing_instance_masks.append(sample_id)
            continue
        instance_mask_available += 1
        inst = topo_aux._read_u8(instance_path)
        gt_count = len(topo_aux._positive_instance_ids(inst))
        if gt_count in gt_counts:
            gt_counts[int(gt_count)] += 1
        gt_count_available += 1

    train_overlap = _pair_overlap(semantic_train, research_eval)
    val_overlap = _pair_overlap(semantic_val, research_eval)
    return {
        "research_eval_name": "semantic_topology_research_eval",
        "research_eval": research_eval,
        "train_overlap": train_overlap,
        "val_overlap": val_overlap,
        "instance_mask_available_count": int(instance_mask_available),
        "instance_mask_available_fraction": float(instance_mask_available / max(research_eval["sample_count"], 1)),
        "missing_instance_masks": missing_instance_masks,
        "gt_instance_count_available_count": int(gt_count_available),
        "gt_distribution": {
            "gt1": int(gt_counts[1]),
            "gt2": int(gt_counts[2]),
            "gt3": int(gt_counts[3]),
        },
        "analysis_only_contract": {
            "optimizer": False,
            "scheduler": False,
            "early_stopping": False,
            "checkpoint_selection": False,
            "lambda_selection": False,
            "target_parameter_selection": False,
        },
        "verdict": "blocked" if int(train_overlap["sample_overlap_count"]) > 0 else "analysis_only_frozen",
    }


def _target_component_row(sample_id: str, gt_count: int, parts: dict[str, np.ndarray], image_shape: tuple[int, int]) -> dict[str, Any]:
    critical_fg = parts["critical_foreground"] > 0
    separation = parts["inter_instance_separation"] > 0
    total_px = int(image_shape[0] * image_shape[1])

    def frac(value: int) -> float:
        return float(value / max(total_px, 1))

    critical_count = int(np.count_nonzero(critical_fg))
    separation_count = int(np.count_nonzero(separation))
    union_count = int(np.count_nonzero(critical_fg | separation))
    overlap_count = int(np.count_nonzero(critical_fg & separation))
    return {
        "sample_id": sample_id,
        "gt_count": int(gt_count),
        "image_pixel_count": total_px,
        "critical_foreground_count": critical_count,
        "critical_foreground_fraction": frac(critical_count),
        "inter_instance_separation_count": separation_count,
        "inter_instance_separation_fraction": frac(separation_count),
        "overlap_count": overlap_count,
        "overlap_fraction": frac(overlap_count),
        "union_count": union_count,
        "union_fraction": frac(union_count),
        "has_nonzero_separation": int(separation_count > 0),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    image_pixels = int(sum(int(row["image_pixel_count"]) for row in rows))
    out = {
        "sample_count": int(len(rows)),
        "image_pixels_total": image_pixels,
    }
    for key in ["critical_foreground_count", "inter_instance_separation_count", "overlap_count", "union_count"]:
        total = int(sum(int(row[key]) for row in rows))
        out[key] = total
        out[key.replace("_count", "_fraction")] = float(total / max(image_pixels, 1))
    return out


def _select_visual_examples(rows: list[dict[str, Any]]) -> dict[str, str]:
    candidates: dict[str, tuple[float, str]] = {
        "gt1": (float("-inf"), ""),
        "gt2_close_neighbors": (float("-inf"), ""),
        "gt3_close_neighbors": (float("-inf"), ""),
        "narrow_leaflet": (float("-inf"), ""),
        "disconnected_same_leaflet": (float("-inf"), ""),
        "no_separation_target": (float("-inf"), ""),
    }
    for row in rows:
        sample_id = str(row["sample_id"])
        gt_count = int(row["gt_count"])
        if gt_count == 1 and float(row["critical_foreground_fraction"]) > candidates["gt1"][0]:
            candidates["gt1"] = (float(row["critical_foreground_fraction"]), sample_id)
        if gt_count == 2 and float(row["inter_instance_separation_fraction"]) > candidates["gt2_close_neighbors"][0]:
            candidates["gt2_close_neighbors"] = (float(row["inter_instance_separation_fraction"]), sample_id)
        if gt_count == 3 and float(row["inter_instance_separation_fraction"]) > candidates["gt3_close_neighbors"][0]:
            candidates["gt3_close_neighbors"] = (float(row["inter_instance_separation_fraction"]), sample_id)
        if float(row["critical_foreground_fraction"]) > candidates["narrow_leaflet"][0]:
            candidates["narrow_leaflet"] = (float(row["critical_foreground_fraction"]), sample_id)
        if float(row.get("disconnected_same_leaflet", 0.0)) > candidates["disconnected_same_leaflet"][0]:
            candidates["disconnected_same_leaflet"] = (float(row.get("disconnected_same_leaflet", 0.0)), sample_id)
        if int(row["has_nonzero_separation"]) == 0 and float(row["critical_foreground_fraction"]) > candidates["no_separation_target"][0]:
            candidates["no_separation_target"] = (float(row["critical_foreground_fraction"]), sample_id)
    return {key: sample_id for key, (_score, sample_id) in candidates.items() if sample_id}


def audit_target_targets(cfg: dict[str, Any], contract: topo_aux.TopologyTargetContract, output_dir: Path) -> dict[str, Any]:
    dataset_cfg = cfg.get("dataset") or {}
    dataset_root = topo_aux._resolve_repo_path(dataset_cfg.get("root", topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT), topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT)
    train_split_txt = topo_aux._resolve_repo_path(dataset_cfg.get("train_txt", topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT), topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT)
    instance_root = topo_aux._resolve_repo_path(dataset_cfg.get("instance_root", topo_aux.DEFAULT_INSTANCE_ROOT), topo_aux.DEFAULT_INSTANCE_ROOT)
    items = topo_aux.read_split_file(dataset_root.resolve(), train_split_txt.resolve())
    rows: list[dict[str, Any]] = []
    for item in items:
        sample_id = Path(item.image_path).stem
        instance_mask = topo_aux._read_u8(instance_root / "instance_masks" / f"{sample_id}.png")
        gt_count = len(topo_aux._positive_instance_ids(instance_mask))
        _target, parts = topo_aux.generate_topology_target(instance_mask, contract, return_parts=True)
        comp_counts = topo_aux._instance_component_counts(instance_mask)
        row = _target_component_row(sample_id, gt_count, parts, instance_mask.shape[:2])
        row["disconnected_same_leaflet"] = int(any(int(v) > 1 for v in comp_counts.values()))
        rows.append(row)

    aggregate = _aggregate_rows(rows)
    gt1_rows = [row for row in rows if int(row["gt_count"]) == 1]
    gt2_rows = [row for row in rows if int(row["gt_count"]) == 2]
    gt3_rows = [row for row in rows if int(row["gt_count"]) == 3]
    gt2_nonzero = int(sum(int(row["has_nonzero_separation"]) for row in gt2_rows))
    gt3_nonzero = int(sum(int(row["has_nonzero_separation"]) for row in gt3_rows))

    examples_dir = output_dir / "target_visual_audit"
    examples_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_visual_examples(rows)
    saved_files: list[str] = []
    sample_lookup = {Path(item.image_path).stem: item for item in items}
    for category, sample_id in selected.items():
        item = sample_lookup[sample_id]
        rgb = topo_aux._center_crop_like_validation(topo_aux._read_image_rgb(item.image_path), 768, 768, is_mask=False)
        semantic_mask = topo_aux._center_crop_like_validation(topo_aux._read_u8(item.mask_path), 768, 768, is_mask=True)
        instance_mask = topo_aux._center_crop_like_validation(topo_aux._read_u8(instance_root / "instance_masks" / f"{sample_id}.png"), 768, 768, is_mask=True)
        _target, parts = topo_aux.generate_topology_target(instance_mask, contract, return_parts=True)
        critical_fg = parts["critical_foreground"].astype(np.uint8)
        separation = parts["inter_instance_separation"].astype(np.uint8)
        semantic_union = (semantic_mask == 1).astype(np.uint8)
        overlay = rgb.copy().astype(np.float32)
        overlay[critical_fg > 0] = overlay[critical_fg > 0] * 0.45 + np.asarray([255.0, 0.0, 0.0], dtype=np.float32) * 0.55
        overlay[separation > 0] = overlay[separation > 0] * 0.35 + np.asarray([0.0, 255.0, 255.0], dtype=np.float32) * 0.65
        overlay = overlay.astype(np.uint8)

        def _mask_rgb(mask01: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
            out = np.zeros((mask01.shape[0], mask01.shape[1], 3), dtype=np.uint8)
            out[mask01 > 0] = np.asarray(color, dtype=np.uint8)
            return out

        inst_vis = cv2.applyColorMap(((instance_mask.astype(np.float32) / max(float(instance_mask.max()), 1.0)) * 255.0 + 0.5).astype(np.uint8), cv2.COLORMAP_TURBO)
        semantic_vis = _mask_rgb(semantic_union, (0, 255, 0))
        critical_vis = _mask_rgb(critical_fg, (255, 0, 0))
        separation_vis = _mask_rgb(separation, (0, 255, 255))
        panel_top = np.concatenate(
            [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), inst_vis, cv2.cvtColor(semantic_vis, cv2.COLOR_RGB2BGR)],
            axis=1,
        )
        panel_bottom = np.concatenate(
            [cv2.cvtColor(critical_vis, cv2.COLOR_RGB2BGR), cv2.cvtColor(separation_vis, cv2.COLOR_RGB2BGR), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)],
            axis=1,
        )
        grid = np.concatenate([panel_top, panel_bottom], axis=0)
        out_path = examples_dir / f"{category}_{sample_id}.png"
        cv2.imwrite(str(out_path), grid)
        saved_files.append(str(out_path.resolve()))

    return {
        "aggregate": aggregate,
        "gt1": _aggregate_rows(gt1_rows),
        "gt2": _aggregate_rows(gt2_rows),
        "gt3": _aggregate_rows(gt3_rows),
        "gt2_nonzero_separation_count": gt2_nonzero,
        "gt2_nonzero_separation_fraction": float(gt2_nonzero / max(len(gt2_rows), 1)),
        "gt3_nonzero_separation_count": gt3_nonzero,
        "gt3_nonzero_separation_fraction": float(gt3_nonzero / max(len(gt3_rows), 1)),
        "rows": rows,
        "saved_files": saved_files,
        "selected_examples": selected,
    }


def _median_range(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max())}


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
    binary_loss = topo_aux.BinaryBCEDiceLoss(
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
                topo_fg = binary_loss(outputs["topology_logits"][:, 0:1], topology_target[:, 0:1])
                topo_sep = binary_loss(outputs["topology_logits"][:, 1:2], topology_target[:, 1:2])
                topo_mean = 0.5 * (topo_fg + topo_sep)
                if which == "semantic":
                    loss = sem
                elif which == "raw_topology":
                    loss = topo_mean
                elif which == "weighted_topology":
                    loss = float(lambda_topology) * topo_mean
                elif which == "combined":
                    loss = sem + float(lambda_topology) * topo_mean
                else:
                    raise KeyError(which)
            loss.backward()
            return {
                "x0_4_grad_norm": _grad_norm(x04_named),
                "segmentation_head_grad_norm": _grad_norm(seg_named),
                "topology_head_grad_norm": _grad_norm(topo_named),
                "semantic_loss": float(sem.detach().cpu().item()),
                "topology_fg_loss": float(topo_fg.detach().cpu().item()),
                "topology_separation_loss": float(topo_sep.detach().cpu().item()),
                "raw_topology_loss": float(topo_mean.detach().cpu().item()),
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
                "raw_topology_loss": raw_topology_stats["raw_topology_loss"],
                "topology_fg_loss": raw_topology_stats["topology_fg_loss"],
                "topology_separation_loss": raw_topology_stats["topology_separation_loss"],
                "weighted_topology_to_semantic_x0_4_ratio": ratio,
            }
        )

    return {
        "lambda_topology": float(lambda_topology),
        "batch_count": int(len(rows)),
        "rows": rows,
        "semantic_x0_4_grad_norm": _median_range([float(row["semantic_x0_4_grad_norm"]) for row in rows]),
        "semantic_segmentation_head_grad_norm": _median_range([float(row["semantic_segmentation_head_grad_norm"]) for row in rows]),
        "raw_topology_x0_4_grad_norm": _median_range([float(row["raw_topology_x0_4_grad_norm"]) for row in rows]),
        "raw_topology_topology_head_grad_norm": _median_range([float(row["raw_topology_topology_head_grad_norm"]) for row in rows]),
        "weighted_topology_x0_4_grad_norm": _median_range([float(row["weighted_topology_x0_4_grad_norm"]) for row in rows]),
        "combined_x0_4_grad_norm": _median_range([float(row["combined_x0_4_grad_norm"]) for row in rows]),
        "combined_segmentation_head_grad_norm": _median_range([float(row["combined_segmentation_head_grad_norm"]) for row in rows]),
        "combined_topology_head_grad_norm": _median_range([float(row["combined_topology_head_grad_norm"]) for row in rows]),
        "weighted_topology_to_semantic_x0_4_ratio": _median_range([float(row["weighted_topology_to_semantic_x0_4_ratio"]) for row in rows]),
    }


def select_lambda_from_gradient_summaries(primary_lambda: float, primary_summary: dict[str, Any], alternative_summary: dict[str, Any] | None) -> tuple[float, str]:
    primary_ratio = float(primary_summary["weighted_topology_to_semantic_x0_4_ratio"]["median"] or 0.0)
    if primary_ratio <= 0.30:
        return float(primary_lambda), "Retained lambda=0.2 because the two-channel auxiliary shared-decoder median ratio stayed <= 0.30."
    if alternative_summary is None:
        return float(primary_lambda), "Retained lambda because no conservative alternative summary was available."
    alt_ratio = float(alternative_summary["weighted_topology_to_semantic_x0_4_ratio"]["median"] or 0.0)
    if alt_ratio <= 0.30:
        return 0.1, "Selected lambda=0.1 because lambda=0.2 exceeded the auxiliary gradient budget, while 0.1 restored the shared-decoder median ratio to <= 0.30."
    return float(primary_lambda), "Retained lambda=0.2 because lambda=0.1 did not restore the requested gradient budget."


def audit_gradient_contribution(cfg: dict[str, Any], contract: topo_aux.TopologyTargetContract, device: torch.device) -> dict[str, Any]:
    primary_lambda = float((cfg.get("topology_aux") or {}).get("lambda_topology", 0.2))
    primary = _single_lambda_gradient_audit(cfg, contract, device, lambda_topology=primary_lambda)
    alternative = None
    if float(primary["weighted_topology_to_semantic_x0_4_ratio"]["median"] or 0.0) > 0.30:
        alternative = _single_lambda_gradient_audit(cfg, contract, device, lambda_topology=0.1)
    selected_lambda, reason = select_lambda_from_gradient_summaries(primary_lambda, primary, alternative)
    return {
        "selection_rule": "Keep lambda=0.2 if median(weighted_auxiliary_x0_4_grad / semantic_x0_4_grad) <= 0.30 and no pathological batch dominates; otherwise test exactly one alternative lambda=0.1.",
        "lambda_0p2": primary,
        "lambda_0p1": alternative,
        "selected_lambda": float(selected_lambda),
        "reason": reason,
    }


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
            "checkpoint_selection": "best_mean_fg.pth by highest semantic mean_dice_fg",
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
            "checkpoint_selection": "best_mean_fg.pth by highest semantic mean_dice_fg",
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
    smoke_cfg["train"]["epochs"] = 1
    smoke_cfg["train"]["dry_run_steps"] = 1
    smoke_config_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_config_path.write_text(yaml.safe_dump(smoke_cfg, sort_keys=False), encoding="utf-8")
    return {
        "path": str(smoke_config_path.resolve()),
        "sha256": topo_aux._sha256_file(smoke_config_path.resolve()),
        "save_dir": str(smoke_cfg["train"]["save_dir"]),
    }


def run_preflight(cfg: dict[str, Any], baseline_cfg: dict[str, Any], output_dir: Path, device: torch.device) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    research_eval = audit_research_eval_isolation(cfg)
    dataset_cfg = cfg.get("dataset") or {}
    train_split_txt = topo_aux._resolve_repo_path(dataset_cfg.get("train_txt", topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT), topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT)
    instance_root = topo_aux._resolve_repo_path(dataset_cfg.get("instance_root", topo_aux.DEFAULT_INSTANCE_ROOT), topo_aux.DEFAULT_INSTANCE_ROOT)
    dataset_root = topo_aux._resolve_repo_path(dataset_cfg.get("root", topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT), topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT)
    contract, contract_audit = topo_aux.choose_topology_target_contract(
        dataset_root=dataset_root,
        train_split_txt=train_split_txt,
        instance_root=instance_root,
    )
    target_audit = audit_target_targets(cfg, contract, output_dir)
    gradient = audit_gradient_contribution(cfg, contract, device)
    freeze_mode = audit_freeze_mode(cfg)
    schedule = trace_schedule_contract(baseline_cfg, cfg)
    smoke_config = write_smoke_config(cfg, float(gradient["selected_lambda"]), DEFAULT_SMOKE_CONFIG_PATH)
    validation_contract = topo_aux.build_validation_contract(
        topo_aux._resolve_repo_path(dataset_cfg.get("research_eval_split_txt", topo_aux.DEFAULT_SEMANTIC_TEST_SPLIT), topo_aux.DEFAULT_SEMANTIC_TEST_SPLIT)
    )

    full_run_dir = topo_aux._resolve_repo_path((cfg.get("train") or {}).get("save_dir"), topo_aux.REPO_ROOT / "training" / "runs" / "semantic_topology_aux")
    readiness = "blocked" if research_eval["verdict"] == "blocked" else "ready_for_A100_smoke"
    summary = {
        "research_eval_isolation": research_eval,
        "target_contract": topo_aux.asdict(contract),
        "target_contract_audit": contract_audit,
        "target_audit": target_audit,
        "gradient_contribution": gradient,
        "schedule_contract": schedule,
        "freeze_mode": freeze_mode,
        "validation_contract": validation_contract,
        "semantic_checkpoint_sha256": topo_aux._sha256_file(
            topo_aux._resolve_repo_path((cfg.get("train") or {}).get("init_checkpoint", topo_aux.DEFAULT_SEMANTIC_CHECKPOINT), topo_aux.DEFAULT_SEMANTIC_CHECKPOINT)
        ),
        "full_run_save_dir": str((cfg.get("train") or {}).get("save_dir")),
        "full_run_dir_exists": bool(full_run_dir.exists()),
        "smoke_config": smoke_config,
        "ubuntu_a100_smoke_command": f"python -u training/train_semantic_topology_aux.py --config {Path(smoke_config['path']).relative_to(topo_aux.REPO_ROOT).as_posix()} --smoke-test",
        "training_readiness": readiness,
    }

    topo_aux._write_json(output_dir / "research_eval_isolation.json", research_eval)
    topo_aux._write_json(output_dir / "target_contract.json", topo_aux.asdict(contract))
    topo_aux._write_json(output_dir / "target_contract_audit.json", contract_audit)
    topo_aux._write_json(output_dir / "target_audit.json", {k: v for k, v in target_audit.items() if k != "rows"})
    topo_aux._write_csv(output_dir / "target_audit_rows.csv", target_audit["rows"])
    topo_aux._write_json(output_dir / "gradient_contribution.json", gradient)
    topo_aux._write_json(output_dir / "freeze_mode.json", freeze_mode)
    topo_aux._write_json(output_dir / "schedule_contract.json", schedule)
    topo_aux._write_json(output_dir / "validation_contract.json", validation_contract)
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
    summary = run_preflight(cfg=cfg, baseline_cfg=baseline_cfg, output_dir=args.output_dir.resolve(), device=torch.device("cpu"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
