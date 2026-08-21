from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e

import bridge_suppression_head as bridge


def _save_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "step",
        "loss",
        "balanced_bce",
        "dice_loss",
        "positive_bce",
        "negative_bce",
        "bridge_precision",
        "bridge_recall",
        "bridge_f1",
        "bridge_dice",
        "tp",
        "fp",
        "fn",
        "p50_start_mean_iou",
        "pred_mean_iou",
        "oracle_mean_iou",
        "p50_start_success50",
        "pred_success50",
        "oracle_success50",
        "positive_precision",
        "positive_recall",
        "positive_f1",
        "positive_tp",
        "positive_fp",
        "positive_fn",
        "positive_start_mean_iou",
        "positive_pred_mean_iou",
        "positive_oracle_mean_iou",
        "positive_start_success50",
        "positive_pred_success50",
        "positive_oracle_success50",
        "negative_predicted_bridge_pixels",
        "negative_removed_fraction",
        "negative_zero_predicted_removal",
        "negative_start_mean_iou",
        "negative_pred_mean_iou",
        "negative_num_improves",
        "negative_num_unchanged",
        "negative_num_regresses",
        "negative_num_component_topology_changes",
        "all_removed_over_candidate",
        "positive_removed_over_candidate",
        "negative_removed_over_candidate",
        "positive_gt_bridge_over_candidate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _canonical_config_arg(cfg: dict[str, Any]) -> str:
    cfg_path = Path(str(cfg["_config_path"])).resolve()
    try:
        return cfg_path.relative_to(bridge.REPO_ROOT).as_posix()
    except ValueError:
        return str(cfg_path)


def _micro_augment_flags(cfg: dict[str, Any]) -> dict[str, bool]:
    aug = cfg.get("augment") or {}
    return {
        "rotate90": bool(aug.get("rotate90", False)),
        "hflip": bool(aug.get("hflip", False)),
        "vflip": bool(aug.get("vflip", False)),
        "brightness_contrast": bool(aug.get("brightness_contrast", False)),
        "gamma": bool(aug.get("gamma", False)),
    }


def _smoke_step(
    *,
    model: bridge.FrozenSemanticBridgeSuppressionModel,
    raw_record: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    loss_fn: bridge.CandidateBalancedBCEDiceLoss,
    device: torch.device,
    use_amp: bool,
    scaler,
) -> dict[str, Any]:
    image_rgb = bridge._center_crop_like_validation(
        bridge._load_image_rgb(Path(raw_record["image_path"])),
        raw_record["gt_semantic"].shape[0],
        raw_record["gt_semantic"].shape[1],
        is_mask=False,
    )
    image_t = torch.from_numpy(bridge._simple_preprocess_uint8_rgb(image_rgb).transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    bridge_target = torch.from_numpy(raw_record["bridge_target"][None, None, ...].astype(np.float32)).to(device)
    semantic_named = [(name, p) for name, p in model.named_parameters() if name.startswith("base.")]
    trainable_named = [(name, p) for name, p in model.named_parameters() if (name.startswith("context_projection.") or name.startswith("bridge_head."))]
    base_bn_ref = bridge._collect_batchnorm_stats(model.base)
    semantic_snap = bridge._snapshot_named_parameters(semantic_named)
    trainable_snap = bridge._snapshot_named_parameters(trainable_named)

    optimizer.zero_grad(set_to_none=True)
    with bridge._autocast_ctx(device, enabled=use_amp):
        outputs = model(image_t)
        loss_dict = loss_fn(
            bridge_logits=outputs["bridge_logits"],
            bridge_target=bridge_target,
            candidate_mask=outputs["candidate_mask"],
        )
    if scaler is not None:
        scaler.scale(loss_dict["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss_dict["loss"].backward()
        optimizer.step()

    grad_context = [(name, p) for name, p in trainable_named if name.startswith("context_projection.")]
    grad_head = [(name, p) for name, p in trainable_named if name.startswith("bridge_head.")]
    return {
        "status": "pass",
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_properties(0).name if device.type == "cuda" else None,
        "amp_enabled": bool(use_amp),
        "x_0_4_shape": list(outputs["x_0_4"].shape),
        "x_2_2_shape": list(outputs["x_2_2"].shape),
        "projected_context_shape": list(outputs["projected_context"].shape),
        "concatenated_shape": list(outputs["concatenated"].shape),
        "bridge_logits_shape": list(outputs["bridge_logits"].shape),
        "candidate_mask_shape": list(outputs["candidate_mask"].shape),
        "bridge_target_shape": list(bridge_target.shape),
        "dtypes": {
            "x_0_4": str(outputs["x_0_4"].dtype),
            "x_2_2": str(outputs["x_2_2"].dtype),
            "projected_context": str(outputs["projected_context"].dtype),
            "p_leaf": str(outputs["p_leaf"].dtype),
            "bridge_logits": str(outputs["bridge_logits"].dtype),
            "loss": str(loss_dict["loss"].dtype),
        },
        "losses": {
            "total": float(loss_dict["loss"].detach().cpu().item()),
            "balanced_bce": float(loss_dict["balanced_bce"].detach().cpu().item()),
            "dice_loss": float(loss_dict["dice_loss"].detach().cpu().item()),
            "positive_bce": float(loss_dict["positive_bce"].detach().cpu().item()),
            "negative_bce": float(loss_dict["negative_bce"].detach().cpu().item()),
        },
        "gradients": {
            "context_projection_grad_norm": float(bridge._named_grad_l2_norm(grad_context)),
            "bridge_head_grad_norm": float(bridge._named_grad_l2_norm(grad_head)),
            "context_projection_grad_present": int(bridge._count_present_grads(grad_context)),
            "bridge_head_grad_present": int(bridge._count_present_grads(grad_head)),
            "semantic_grad_count": int(bridge._count_present_grads(semantic_named)),
            "all_trainable_grads_finite": bool(bridge._all_grads_finite(trainable_named)),
        },
        "frozen_state_proof": {
            "semantic_parameter_max_delta": float(bridge._max_parameter_delta_from_snapshot(semantic_named, semantic_snap)),
            "semantic_bn_max_delta": float(bridge._max_bn_delta(model.base, base_bn_ref)),
            "trainable_parameter_max_delta": float(bridge._max_parameter_delta_from_snapshot(trainable_named, trainable_snap)),
            "semantic_eval": bool(not model.base.training),
        },
    }


def _run_micro_overfit(
    *,
    model: bridge.FrozenSemanticBridgeSuppressionModel,
    cached_records: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    loss_fn: bridge.CandidateBalancedBCEDiceLoss,
    device: torch.device,
    max_steps: int,
    log_every: int,
    save_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    batch = bridge.stack_cached_batch(cached_records, device)
    trainable_named = [(name, p) for name, p in model.named_parameters() if (name.startswith("context_projection.") or name.startswith("bridge_head."))]
    semantic_named = [(name, p) for name, p in model.named_parameters() if name.startswith("base.")]
    semantic_ref = bridge._snapshot_named_parameters(semantic_named)
    base_bn_ref = bridge._collect_batchnorm_stats(model.base)
    metrics_rows: list[dict[str, Any]] = []
    best_pixel_f1 = float("-inf")
    best_pixel_payload: dict[str, Any] | None = None
    best_reconstruction_key = (float("-inf"), float("-inf"))
    best_reconstruction_payload: dict[str, Any] | None = None

    for step in range(1, int(max_steps) + 1):
        model.train(True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model.bridge_forward_from_cached(
            x_0_4=batch["x_0_4"],
            x_2_2=batch["x_2_2"],
            p_leaf=batch["p_leaf"],
        )
        loss_dict = loss_fn(
            bridge_logits=outputs["bridge_logits"],
            bridge_target=batch["bridge_target"],
            candidate_mask=outputs["candidate_mask"],
        )
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        if step == 1 or step % int(log_every) == 0 or step == int(max_steps):
            model.eval()
            with torch.no_grad():
                eval_outputs = model.bridge_forward_from_cached(
                    x_0_4=batch["x_0_4"],
                    x_2_2=batch["x_2_2"],
                    p_leaf=batch["p_leaf"],
                )
                bridge_probs = torch.sigmoid(eval_outputs["bridge_logits"])
                pixel = bridge.compute_binary_metrics_from_domain(
                    bridge_probs=bridge_probs,
                    bridge_target=batch["bridge_target"],
                    candidate_mask=batch["candidate_mask"],
                )
                recon = bridge.evaluate_reconstruction_levels_on_cached(model, cached_records, device)
            positive_subset = recon["positive_subset"]
            negative_subset = recon["negative_subset"]
            removal_calibration = recon["removal_calibration"]
            row = {
                "step": int(step),
                "loss": float(loss.detach().cpu().item()),
                "balanced_bce": float(loss_dict["balanced_bce"].detach().cpu().item()),
                "dice_loss": float(loss_dict["dice_loss"].detach().cpu().item()),
                "positive_bce": float(loss_dict["positive_bce"].detach().cpu().item()),
                "negative_bce": float(loss_dict["negative_bce"].detach().cpu().item()),
                "bridge_precision": float(pixel["precision"]),
                "bridge_recall": float(pixel["recall"]),
                "bridge_f1": float(pixel["f1"]),
                "bridge_dice": float(pixel["dice"]),
                "tp": int(pixel["tp"]),
                "fp": int(pixel["fp"]),
                "fn": int(pixel["fn"]),
                "p50_start_mean_iou": float(recon["reconstruction"]["p50_start"]["mean_matched_iou"]),
                "pred_mean_iou": float(recon["reconstruction"]["p50_minus_predicted_bridge"]["mean_matched_iou"]),
                "oracle_mean_iou": float(recon["reconstruction"]["p50_minus_gt_oracle_bridge"]["mean_matched_iou"]),
                "p50_start_success50": int(recon["reconstruction"]["p50_start"]["all_iou_ge_0.50_count"]),
                "pred_success50": int(recon["reconstruction"]["p50_minus_predicted_bridge"]["all_iou_ge_0.50_count"]),
                "oracle_success50": int(recon["reconstruction"]["p50_minus_gt_oracle_bridge"]["all_iou_ge_0.50_count"]),
                "positive_precision": float(positive_subset["pixel"]["precision"]),
                "positive_recall": float(positive_subset["pixel"]["recall"]),
                "positive_f1": float(positive_subset["pixel"]["f1"]),
                "positive_tp": int(positive_subset["pixel"]["tp"]),
                "positive_fp": int(positive_subset["pixel"]["fp"]),
                "positive_fn": int(positive_subset["pixel"]["fn"]),
                "positive_start_mean_iou": float(positive_subset["reconstruction"]["p50_start"]["mean_matched_iou"]),
                "positive_pred_mean_iou": float(positive_subset["reconstruction"]["p50_minus_predicted_bridge"]["mean_matched_iou"]),
                "positive_oracle_mean_iou": float(positive_subset["reconstruction"]["p50_minus_gt_oracle_bridge"]["mean_matched_iou"]),
                "positive_start_success50": int(positive_subset["reconstruction"]["p50_start"]["all_iou_ge_0.50_count"]),
                "positive_pred_success50": int(positive_subset["reconstruction"]["p50_minus_predicted_bridge"]["all_iou_ge_0.50_count"]),
                "positive_oracle_success50": int(positive_subset["reconstruction"]["p50_minus_gt_oracle_bridge"]["all_iou_ge_0.50_count"]),
                "negative_predicted_bridge_pixels": int(negative_subset["predicted_bridge_pixels"]),
                "negative_removed_fraction": float(negative_subset["fraction_of_candidate_pixels_removed"]),
                "negative_zero_predicted_removal": int(negative_subset["samples_with_zero_predicted_removal"]),
                "negative_start_mean_iou": float(negative_subset["starting_mean_matched_iou"]),
                "negative_pred_mean_iou": float(negative_subset["refined_mean_matched_iou"]),
                "negative_num_improves": int(negative_subset["num_improves"]),
                "negative_num_unchanged": int(negative_subset["num_unchanged"]),
                "negative_num_regresses": int(negative_subset["num_regresses"]),
                "negative_num_component_topology_changes": int(negative_subset["num_component_topology_changes"]),
                "all_removed_over_candidate": float(removal_calibration["all_removed_over_candidate"]),
                "positive_removed_over_candidate": float(removal_calibration["positive_removed_over_candidate"]),
                "negative_removed_over_candidate": float(removal_calibration["negative_removed_over_candidate"]),
                "positive_gt_bridge_over_candidate": float(removal_calibration["positive_gt_bridge_over_candidate"]),
            }
            metrics_rows.append(row)
            if float(pixel["f1"]) > float(best_pixel_f1):
                best_pixel_f1 = float(pixel["f1"])
                best_pixel_payload = {
                    "step": int(step),
                    "selection_policy": "pixel_f1",
                    "selection_reason": {"bridge_f1": float(pixel["f1"])},
                    "pixel": pixel,
                    "positive_subset": positive_subset,
                    "negative_subset": negative_subset,
                    "removal_calibration": removal_calibration,
                    "reconstruction": recon["reconstruction"],
                }
                bridge.save_checkpoint(
                    save_dir / "best_pixel_f1.pth",
                    model,
                    optimizer,
                    step,
                    cfg,
                    extra={"best_payload": best_pixel_payload},
                )
            reconstruction_key = (
                int(recon["reconstruction"]["p50_minus_predicted_bridge"]["all_iou_ge_0.50_count"]),
                float(recon["reconstruction"]["p50_minus_predicted_bridge"]["mean_matched_iou"]),
            )
            if reconstruction_key > best_reconstruction_key:
                best_reconstruction_key = reconstruction_key
                best_reconstruction_payload = {
                    "step": int(step),
                    "selection_policy": "reconstruction",
                    "selection_reason": {
                        "predicted_success50": int(reconstruction_key[0]),
                        "predicted_mean_iou": float(reconstruction_key[1]),
                    },
                    "pixel": pixel,
                    "positive_subset": positive_subset,
                    "negative_subset": negative_subset,
                    "removal_calibration": removal_calibration,
                    "reconstruction": recon["reconstruction"],
                }
                bridge.save_checkpoint(
                    save_dir / "best_reconstruction.pth",
                    model,
                    optimizer,
                    step,
                    cfg,
                    extra={"best_payload": best_reconstruction_payload},
                )

    bridge.save_checkpoint(
        save_dir / "last.pth",
        model,
        optimizer,
        int(max_steps),
        cfg,
        extra={
            "best_pixel_payload": best_pixel_payload,
            "best_reconstruction_payload": best_reconstruction_payload,
            "selection_policy": "last",
            "selection_reason": {"step": int(max_steps)},
        },
    )
    _save_metrics_csv(save_dir / "micro_overfit_metrics.csv", metrics_rows)
    final_row = metrics_rows[-1]
    summary = {
        "initial": metrics_rows[0],
        "final": final_row,
        "best_pixel": best_pixel_payload,
        "best_reconstruction": best_reconstruction_payload,
        "semantic_parameter_max_delta": float(bridge._max_parameter_delta_from_snapshot(semantic_named, semantic_ref)),
        "semantic_bn_max_delta": float(bridge._max_bn_delta(model.base, base_bn_ref)),
        "trainable_grad_finite": bool(bridge._all_grads_finite(trainable_named)),
    }
    bridge._write_json(save_dir / "summary.json", summary)
    return summary


def _save_target_audit_and_split_contract(
    *,
    save_dir: Path,
    train_audit: dict[str, Any],
    val_audit: dict[str, Any],
    train_visuals: dict[str, str],
    micro_manifest: dict[str, Any],
    manifest_resolution: dict[str, Any],
) -> None:
    bridge._write_json(save_dir / "train_bridge_target_audit.json", train_audit)
    bridge._write_json(save_dir / "val_bridge_target_audit.json", val_audit)
    bridge._write_json(save_dir / "bridge_target_visuals.json", train_visuals)
    bridge._write_json(save_dir / "micro_manifest_summary.json", micro_manifest)
    bridge._write_json(save_dir / "micro_manifest_resolution.json", manifest_resolution)


def run_pipeline(cfg: dict[str, Any], *, smoke_only: bool = False, preflight_only: bool = False) -> dict[str, Any]:
    save_dir = bridge._resolve_repo_path((cfg.get("train") or {}).get("save_dir"), bridge.REPO_ROOT / "training" / "runs" / "bridge_suppression_micro")
    future_full_dir = bridge._resolve_repo_path((cfg.get("reserved_full_run") or {}).get("save_dir"), bridge.REPO_ROOT / "training" / "runs" / "bridge_suppression_full")
    save_dir.mkdir(parents=True, exist_ok=True)
    if future_full_dir.exists():
        raise SystemExit(f"Reserved future full run directory must remain nonexistent: {future_full_dir}")

    bridge._seed_everything(int(cfg.get("seed", 1337)))
    device = bridge._select_device()
    use_amp = bridge._amp_enabled(cfg, device)
    scaler = bridge._make_grad_scaler(device, enabled=use_amp)
    model = bridge.build_model_from_cfg(cfg).to(device)
    checkpoint_info = bridge.load_semantic_checkpoint(
        model,
        bridge._resolve_repo_path((cfg.get("train") or {}).get("init_checkpoint"), bridge.DEFAULT_SEMANTIC_CHECKPOINT),
    )
    loss_fn = bridge.CandidateBalancedBCEDiceLoss()

    dataset_cfg = cfg.get("dataset") or {}
    train_split = bridge._resolve_repo_path(dataset_cfg.get("train_txt", bridge.DEFAULT_TRAIN_SPLIT), bridge.DEFAULT_TRAIN_SPLIT)
    val_split = bridge._resolve_repo_path(dataset_cfg.get("val_txt", bridge.DEFAULT_VAL_SPLIT), bridge.DEFAULT_VAL_SPLIT)
    test_split = bridge._resolve_repo_path(dataset_cfg.get("test_txt", bridge.DEFAULT_TEST_SPLIT), bridge.DEFAULT_TEST_SPLIT)
    bridge._assert_safe_path(train_split)
    bridge._assert_safe_path(val_split)
    bridge._assert_safe_path(test_split)

    if smoke_only:
        optimizer, optimizer_meta = bridge.build_optimizer(model, cfg)
        first_split_item = bridge._build_split_items(cfg, train_split)[0]
        smoke_records = bridge.mine_bridge_records_for_split(
            cfg=cfg,
            split_txt=train_split,
            model=model,
            device=device,
            cache_features=False,
            selected_sample_ids={str(first_split_item["sample_id"])},
        )
        raw_smoke_record = smoke_records[0]
        smoke_summary = _smoke_step(
            model=model,
            raw_record=raw_smoke_record,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
        )
        bridge._write_json(save_dir / "smoke_test_summary.json", smoke_summary)
        return {
            "checkpoint": checkpoint_info,
            "semantic_inference_backend": bridge.semantic_inference_backend_summary(cfg, device),
            "bridge_training": {
                "amp_requested": bool((cfg.get("train") or {}).get("amp", False)),
                "amp_enabled": bool(use_amp),
            },
            "optimizer": optimizer_meta,
            "smoke": smoke_summary,
            "a100_smoke_command": f"python -u training/train_bridge_suppression_head.py --config {_canonical_config_arg(cfg)} --smoke-test" if device.type != "cuda" else None,
        }

    train_records = bridge.mine_bridge_records_for_split(
        cfg=cfg,
        split_txt=train_split,
        model=model,
        device=device,
        cache_features=False,
    )
    train_audit = bridge.summarize_bridge_records(train_records, train_split)
    train_visuals = bridge.save_train_target_visual_audit(train_records, save_dir)
    val_records = bridge.mine_bridge_records_for_split(
        cfg=cfg,
        split_txt=val_split,
        model=model,
        device=device,
        cache_features=False,
    )
    val_audit = bridge.build_validation_audit(train_records=train_records, val_records=val_records, val_split=val_split)
    micro_cfg = cfg.get("micro_overfit") or {}
    manifest_path = bridge._resolve_repo_path(micro_cfg.get("manifest_path"), bridge.MICRO_MANIFEST_V2_PATH)
    manifest_payload = bridge.read_locked_micro_manifest(manifest_path)
    manifest_payload["_manifest_path"] = str(manifest_path.resolve())
    manifest_split_validation = bridge.validate_locked_manifest_source_split(
        manifest_payload=manifest_payload,
        configured_train_split=train_split,
    )
    if str(manifest_split_validation.get("status")) != "pass":
        bridge._write_json(save_dir / "micro_manifest_resolution.json", {"split_validation": manifest_split_validation})
        raise SystemExit(str(manifest_split_validation.get("error")))
    micro_records = bridge.mine_bridge_records_for_split(
        cfg=cfg,
        split_txt=train_split,
        model=model,
        device=device,
        cache_features=True,
        selected_sample_ids=[str(v) for v in manifest_payload["sample_ids"]],
    )
    manifest_resolution = bridge.validate_locked_micro_records(
        manifest_payload=manifest_payload,
        records=micro_records,
        split_txt=train_split,
    )
    manifest_resolution_payload = {
        "split_validation": manifest_split_validation,
        "record_validation": manifest_resolution,
    }
    bridge._write_json(save_dir / "micro_manifest_resolution.json", manifest_resolution_payload)
    if str(manifest_resolution.get("status")) != "pass":
        raise SystemExit(json.dumps(manifest_resolution_payload, ensure_ascii=False, indent=2))
    if preflight_only:
        overall = {
            "checkpoint": checkpoint_info,
            "semantic_inference_backend": bridge.semantic_inference_backend_summary(cfg, device),
            "bridge_training": {
                "amp_requested": bool((cfg.get("train") or {}).get("amp", False)),
                "amp_enabled": bool(use_amp),
            },
            "train_audit": train_audit,
            "val_audit": val_audit,
            "micro_manifest": manifest_payload,
            "micro_manifest_resolution": manifest_resolution_payload,
            "micro_augment_flags": _micro_augment_flags(cfg),
            "future_full_run_exists": bool(future_full_dir.exists()),
            "future_full_run_dir": str(future_full_dir.resolve()),
        }
        bridge._write_json(save_dir / "preflight_summary.json", overall)
        return overall
    optimizer, optimizer_meta = bridge.build_optimizer(model, cfg)
    _save_target_audit_and_split_contract(
        save_dir=save_dir,
        train_audit=train_audit,
        val_audit=val_audit,
        train_visuals=train_visuals,
        micro_manifest=manifest_payload,
        manifest_resolution=manifest_resolution_payload,
    )

    raw_smoke_record = next((row for row in micro_records if int(row["bridge_positive"]) == 1), micro_records[0])
    smoke_summary = _smoke_step(
        model=model,
        raw_record=raw_smoke_record,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
    )
    bridge._write_json(save_dir / "smoke_test_summary.json", smoke_summary)
    cached_micro = bridge.cache_microset_features(micro_records)
    micro_summary = _run_micro_overfit(
        model=model,
        cached_records=cached_micro,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        max_steps=int(micro_cfg.get("max_steps", 120)),
        log_every=int(micro_cfg.get("log_every", 10)),
        save_dir=save_dir,
        cfg=cfg,
    )
    overall = {
        "checkpoint": checkpoint_info,
        "semantic_inference_backend": bridge.semantic_inference_backend_summary(cfg, device),
        "bridge_training": {
            "amp_requested": bool((cfg.get("train") or {}).get("amp", False)),
            "amp_enabled": bool(use_amp),
        },
        "optimizer": optimizer_meta,
        "train_audit": train_audit,
        "val_audit": val_audit,
        "smoke": smoke_summary,
        "micro_manifest": manifest_payload,
        "micro_manifest_resolution": manifest_resolution_payload,
        "micro_augment_flags": _micro_augment_flags(cfg),
        "micro_overfit": micro_summary,
        "future_full_run_exists": bool(future_full_dir.exists()),
        "future_full_run_dir": str(future_full_dir.resolve()),
        "a100_smoke_command": f"python -u training/train_bridge_suppression_head.py --config {_canonical_config_arg(cfg)} --smoke-test" if device.type != "cuda" else None,
    }
    bridge._write_json(save_dir / "readiness_summary.json", overall)
    return overall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    cfg_path = bridge._resolve_repo_path(args.config, bridge.DEFAULT_SEMANTIC_CONFIG)
    cfg = bridge._read_yaml(cfg_path)
    cfg["_config_path"] = str(cfg_path.resolve())
    summary = run_pipeline(cfg, smoke_only=bool(args.smoke_test), preflight_only=bool(args.preflight_only))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
