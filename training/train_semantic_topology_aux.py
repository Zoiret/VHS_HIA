from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e
from tqdm import tqdm

import semantic_topology_aux as topo_aux


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA is not available, using CPU.")
    print(f"Device: {device}")
    return device


def _make_grad_scaler(device: torch.device, enabled: bool):
    if device.type == "cuda" and bool(enabled):
        return torch.amp.GradScaler("cuda")
    return None


def _build_scheduler_from_cfg(cfg: dict[str, object], optimizer: torch.optim.Optimizer):
    sched_cfg = cfg.get("scheduler", None)
    if not isinstance(sched_cfg, dict) or not sched_cfg:
        return None
    t = str(sched_cfg.get("type", "")).strip().lower()
    if t not in {"reduce_on_plateau", "reduce_lr_on_plateau"}:
        raise SystemExit(f"Unsupported scheduler.type: {t!r}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode=str(sched_cfg.get("mode", "max")).strip().lower(),
        factor=float(sched_cfg.get("factor", 0.5)),
        patience=int(sched_cfg.get("patience", 8)),
        min_lr=float(sched_cfg.get("min_lr", 0.0)),
    )


def _get_monitor_value(
    monitor: str,
    *,
    val_loss: float | None,
    mean_dice_fg: float | None,
    mean_iou_fg: float | None,
) -> float | None:
    m = str(monitor).strip().lower()
    if m in {"mean_dice_fg", "dice_fg"}:
        return mean_dice_fg
    if m in {"mean_iou_fg", "iou_fg"}:
        return mean_iou_fg
    if m in {"val_loss", "loss"}:
        return val_loss
    raise SystemExit(f"Unsupported monitor value: {monitor!r}")


def _execute_training_step(
    *,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor | object],
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    scaler,
    freeze_info: dict[str, object],
    capture_debug: bool = False,
) -> dict[str, object]:
    images = batch["image"].to(device, non_blocking=True)
    semantic_target = batch["mask"].to(device, non_blocking=True)
    topology_target = batch["topology_target"].to(device, non_blocking=True)

    trainable_decoder_named = [(name, p) for name, p in model.named_parameters() if name in set(freeze_info["selected_decoder_param_names"])]
    segmentation_head_named = [(name, p) for name, p in model.named_parameters() if name.startswith("base.segmentation_head.")]
    topology_head_named = [(name, p) for name, p in model.named_parameters() if name.startswith("topology_head.")]
    frozen_named = [(name, p) for name, p in model.named_parameters() if not bool(p.requires_grad)]
    encoder_frozen_named = [(name, p) for name, p in model.named_parameters() if name.startswith("base.encoder.") and not bool(p.requires_grad)]

    frozen_snap = topo_aux._snapshot_named_parameters(frozen_named) if capture_debug else None
    trainable_snap = (
        {
            "decoder": topo_aux._snapshot_named_parameters(trainable_decoder_named),
            "segmentation_head": topo_aux._snapshot_named_parameters(segmentation_head_named),
            "topology_head": topo_aux._snapshot_named_parameters(topology_head_named),
        }
        if capture_debug
        else None
    )
    selected_prefixes = [str(path).replace("base.", "", 1) for path in freeze_info["trainable_decoder_modules"]]
    bn_ref = topo_aux._collect_batchnorm_stats(model.base) if capture_debug else None

    optimizer.zero_grad(set_to_none=True)
    with topo_aux._autocast_ctx(device, enabled=use_amp):
        outputs = model(images)
        loss_dict = loss_fn(
            semantic_logits=outputs["semantic_logits"],
            semantic_target=semantic_target,
            topology_logits=outputs["topology_logits"],
            topology_target=topology_target,
        )
    combined_loss = loss_dict["combined_loss"]
    if scaler is not None:
        scaler.scale(combined_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        combined_loss.backward()
        optimizer.step()

    summary: dict[str, object] = {
        "outputs": outputs,
        "loss_dict": loss_dict,
    }
    if not capture_debug:
        return summary

    summary.update(
        {
            "decoder_feature_dtype": str(outputs["decoder_feature"].dtype),
            "topology_head_input_dtype": str(outputs["decoder_feature"].dtype),
            "topology_head_weight_dtype": str(model.topology_head.block[0].weight.dtype),
            "semantic_logits_dtype": str(outputs["semantic_logits"].dtype),
            "topology_logits_dtype": str(outputs["topology_logits"].dtype),
            "semantic_loss_dtype": str(loss_dict["semantic_loss"].dtype),
            "topology_loss_dtype": str(loss_dict["topology_loss"].dtype),
            "combined_loss_dtype": str(loss_dict["combined_loss"].dtype),
            "topology_head_grad_norm": float(topo_aux._named_grad_l2_norm(topology_head_named)),
            "segmentation_head_grad_norm": float(topo_aux._named_grad_l2_norm(segmentation_head_named)),
            "selected_decoder_grad_norm": float(topo_aux._named_grad_l2_norm(trainable_decoder_named)),
            "topology_head_grad_present": int(topo_aux._count_present_grads(topology_head_named)),
            "segmentation_head_grad_present": int(topo_aux._count_present_grads(segmentation_head_named)),
            "selected_decoder_grad_present": int(topo_aux._count_present_grads(trainable_decoder_named)),
            "frozen_encoder_grad_count": int(topo_aux._count_present_grads(encoder_frozen_named)),
            "frozen_parameter_max_delta": float(topo_aux._max_parameter_delta_from_snapshot(frozen_named, frozen_snap or {})),
            "frozen_bn_max_delta": float(topo_aux._max_bn_delta_filtered(model.base, bn_ref or [], exclude_prefixes=selected_prefixes) or 0.0),
            "selected_bn_max_delta": float(topo_aux._max_bn_delta_filtered(model.base, bn_ref or [], include_prefixes=selected_prefixes) or 0.0),
            "selected_decoder_parameter_delta": float(topo_aux._max_parameter_delta_from_snapshot(trainable_decoder_named, (trainable_snap or {})["decoder"])),
            "segmentation_head_parameter_delta": float(topo_aux._max_parameter_delta_from_snapshot(segmentation_head_named, (trainable_snap or {})["segmentation_head"])),
            "topology_head_parameter_delta": float(topo_aux._max_parameter_delta_from_snapshot(topology_head_named, (trainable_snap or {})["topology_head"])),
            "backward_finite": bool(
                topo_aux._all_grads_finite(trainable_decoder_named)
                and topo_aux._all_grads_finite(segmentation_head_named)
                and topo_aux._all_grads_finite(topology_head_named)
            ),
            "any_nonfinite_gradients": bool(
                not topo_aux._all_grads_finite(trainable_decoder_named)
                or not topo_aux._all_grads_finite(segmentation_head_named)
                or not topo_aux._all_grads_finite(topology_head_named)
            ),
        }
    )
    return summary


def _build_loaders(cfg: dict[str, object], contract: topo_aux.TopologyTargetContract, device: torch.device):
    dataset_cfg = cfg.get("dataset") or {}
    train_cfg = cfg.get("train") or {}
    dataset_root = topo_aux._resolve_repo_path(dataset_cfg.get("root", topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT), topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT)
    train_split = topo_aux._resolve_repo_path(dataset_cfg.get("train_txt", topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT), topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT)
    val_split = topo_aux._resolve_repo_path(dataset_cfg.get("val_txt", topo_aux.DEFAULT_SEMANTIC_VAL_SPLIT), topo_aux.DEFAULT_SEMANTIC_VAL_SPLIT)
    instance_root = topo_aux._resolve_repo_path(dataset_cfg.get("instance_root", topo_aux.DEFAULT_INSTANCE_ROOT), topo_aux.DEFAULT_INSTANCE_ROOT)
    input_size = int((cfg.get("model") or {})["input_size"])
    num_classes = int((cfg.get("model") or {})["classes"])

    train_ds = topo_aux.SemanticTopologyAuxDataset(
        dataset_root=dataset_root,
        split_txt=train_split,
        instance_root=instance_root,
        contract=contract,
        num_classes=num_classes,
        input_size=input_size,
        augment_cfg=cfg.get("augment", None),
        training=True,
    )
    val_ds = topo_aux.SemanticTopologyAuxDataset(
        dataset_root=dataset_root,
        split_txt=val_split,
        instance_root=instance_root,
        contract=contract,
        num_classes=num_classes,
        input_size=input_size,
        augment_cfg=cfg.get("augment", None),
        training=False,
    )

    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory_cfg = train_cfg.get("pin_memory", None)
    pin_memory = bool(pin_memory_cfg) if pin_memory_cfg is not None else (device.type == "cuda")
    persistent_workers_cfg = train_cfg.get("persistent_workers", None)
    persistent_workers = bool(persistent_workers_cfg) if persistent_workers_cfg is not None else False
    prefetch_factor_cfg = train_cfg.get("prefetch_factor", None)
    prefetch_factor = int(prefetch_factor_cfg) if prefetch_factor_cfg is not None else 2
    if device.type != "cuda":
        num_workers = 0
        pin_memory = False
        persistent_workers = False

    dl_kwargs = {}
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = bool(persistent_workers)
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        **dl_kwargs,
    )
    return train_loader, val_loader


def smoke_test(cfg: dict[str, object], device: torch.device) -> dict[str, object]:
    dataset_cfg = cfg.get("dataset") or {}
    contract, _audit = topo_aux.choose_topology_target_contract(
        dataset_root=topo_aux._resolve_repo_path(dataset_cfg.get("root", topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT), topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT),
        train_split_txt=topo_aux._resolve_repo_path(dataset_cfg.get("train_txt", topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT), topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT),
        instance_root=topo_aux._resolve_repo_path(dataset_cfg.get("instance_root", topo_aux.DEFAULT_INSTANCE_ROOT), topo_aux.DEFAULT_INSTANCE_ROOT),
    )
    dataset_root = topo_aux._resolve_repo_path(dataset_cfg.get("root", topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT), topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT)
    train_split_txt = topo_aux._resolve_repo_path(dataset_cfg.get("train_txt", topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT), topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT)
    instance_root = topo_aux._resolve_repo_path(dataset_cfg.get("instance_root", topo_aux.DEFAULT_INSTANCE_ROOT), topo_aux.DEFAULT_INSTANCE_ROOT)
    model = topo_aux.build_model_from_cfg(cfg).to(device)
    checkpoint_info = topo_aux.load_semantic_checkpoint(
        model,
        topo_aux._resolve_repo_path((cfg.get("train") or {}).get("init_checkpoint", topo_aux.DEFAULT_SEMANTIC_CHECKPOINT), topo_aux.DEFAULT_SEMANTIC_CHECKPOINT),
    )
    freeze_info = topo_aux.apply_training_policy(model, cfg)
    topo_aux.set_train_modes(model, freeze_info)
    optimizer, optimizer_meta = topo_aux.build_optimizer_groups(model, cfg, freeze_info)
    loss_fn = topo_aux.build_combined_loss_from_cfg(cfg, device)
    use_amp = topo_aux._amp_enabled(cfg, device)
    scaler = _make_grad_scaler(device, enabled=use_amp)

    train_ds = topo_aux.SemanticTopologyAuxDataset(
        dataset_root=dataset_root,
        split_txt=train_split_txt,
        instance_root=instance_root,
        contract=contract,
        num_classes=int(cfg["model"]["classes"]),
        input_size=int(cfg["model"]["input_size"]),
        augment_cfg=cfg.get("augment", None),
        training=True,
    )
    batch = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)))
    step = _execute_training_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
        freeze_info=freeze_info,
        capture_debug=True,
    )
    outputs = step["outputs"]
    loss_dict = step["loss_dict"]
    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_properties(0).name if device.type == "cuda" else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_only": bool(device.type != "cuda"),
        "amp_enabled": bool(use_amp),
        "checkpoint": checkpoint_info,
        "semantic_forward": bool(outputs["semantic_logits"].shape[1] == int(cfg["model"]["classes"])),
        "semantic_logits_shape": list(outputs["semantic_logits"].shape),
        "topology_logits_shape": list(outputs["topology_logits"].shape),
        "decoder_feature_shape": list(outputs["decoder_feature"].shape),
        "topology_target_shape": list(batch["topology_target"].shape),
        "semantic_loss_finite": bool(torch.isfinite(loss_dict["semantic_loss"]).all().item()),
        "topology_loss_finite": bool(torch.isfinite(loss_dict["topology_loss"]).all().item()),
        "combined_loss_finite": bool(torch.isfinite(loss_dict["combined_loss"]).all().item()),
        "semantic_loss": float(loss_dict["semantic_loss"].detach().cpu().item()),
        "topology_loss": float(loss_dict["topology_loss"].detach().cpu().item()),
        "combined_loss": float(loss_dict["combined_loss"].detach().cpu().item()),
        "decoder_feature_dtype": step["decoder_feature_dtype"],
        "topology_head_input_dtype": step["topology_head_input_dtype"],
        "topology_head_weight_dtype": step["topology_head_weight_dtype"],
        "semantic_logits_dtype": step["semantic_logits_dtype"],
        "topology_logits_dtype": step["topology_logits_dtype"],
        "semantic_loss_dtype": step["semantic_loss_dtype"],
        "topology_loss_dtype": step["topology_loss_dtype"],
        "combined_loss_dtype": step["combined_loss_dtype"],
        "backward_finite": step["backward_finite"],
        "any_nonfinite_gradients": step["any_nonfinite_gradients"],
        "topology_head_grad_norm": step["topology_head_grad_norm"],
        "segmentation_head_grad_norm": step["segmentation_head_grad_norm"],
        "selected_decoder_grad_norm": step["selected_decoder_grad_norm"],
        "topology_head_grad_present": step["topology_head_grad_present"],
        "segmentation_head_grad_present": step["segmentation_head_grad_present"],
        "selected_decoder_grad_present": step["selected_decoder_grad_present"],
        "frozen_encoder_grad_count": step["frozen_encoder_grad_count"],
        "frozen_parameter_max_delta": step["frozen_parameter_max_delta"],
        "frozen_bn_max_delta": step["frozen_bn_max_delta"],
        "selected_bn_max_delta": step["selected_bn_max_delta"],
        "selected_decoder_parameter_delta": step["selected_decoder_parameter_delta"],
        "segmentation_head_parameter_delta": step["segmentation_head_parameter_delta"],
        "topology_head_parameter_delta": step["topology_head_parameter_delta"],
        "optimizer_groups": optimizer_meta,
        "trainable_parameter_names": freeze_info["trainable_names"],
    }


def train(cfg: dict[str, object], device: torch.device) -> None:
    out_dir = topo_aux._resolve_repo_path((cfg.get("train") or {}).get("save_dir"), topo_aux.REPO_ROOT / "training" / "runs" / "semantic_topology_aux")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = cfg.get("dataset") or {}
    contract, contract_audit = topo_aux.choose_topology_target_contract(
        dataset_root=topo_aux._resolve_repo_path(dataset_cfg.get("root", topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT), topo_aux.DEFAULT_SEMANTIC_DATASET_ROOT),
        train_split_txt=topo_aux._resolve_repo_path(dataset_cfg.get("train_txt", topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT), topo_aux.DEFAULT_SEMANTIC_TRAIN_SPLIT),
        instance_root=topo_aux._resolve_repo_path(dataset_cfg.get("instance_root", topo_aux.DEFAULT_INSTANCE_ROOT), topo_aux.DEFAULT_INSTANCE_ROOT),
    )
    topo_aux._write_json(out_dir / "topology_target_contract.json", topo_aux.asdict(contract))
    topo_aux._write_json(out_dir / "topology_target_contract_audit.json", contract_audit)

    train_loader, val_loader = _build_loaders(cfg, contract, device)
    model = topo_aux.build_model_from_cfg(cfg).to(device)
    checkpoint_info = topo_aux.load_semantic_checkpoint(
        model,
        topo_aux._resolve_repo_path((cfg.get("train") or {}).get("init_checkpoint", topo_aux.DEFAULT_SEMANTIC_CHECKPOINT), topo_aux.DEFAULT_SEMANTIC_CHECKPOINT),
    )
    freeze_info = topo_aux.apply_training_policy(model, cfg)
    topo_aux.set_train_modes(model, freeze_info)
    optimizer, optimizer_meta = topo_aux.build_optimizer_groups(model, cfg, freeze_info)
    loss_fn = topo_aux.build_combined_loss_from_cfg(cfg, device)
    scheduler = _build_scheduler_from_cfg(cfg, optimizer)
    use_amp = topo_aux._amp_enabled(cfg, device)
    scaler = _make_grad_scaler(device, enabled=use_amp)

    epochs = int((cfg.get("train") or {}).get("epochs", 100))
    log_every = int((cfg.get("train") or {}).get("log_every", 10))
    metrics_path = out_dir / "metrics.csv"
    if not metrics_path.exists():
        with metrics_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_combined_loss",
                    "val_combined_loss",
                    "mean_dice_fg",
                    "mean_iou_fg",
                    "leaflet_dice",
                    "leaflet_iou",
                    "ring_dice",
                    "ring_iou",
                    "epoch_time_sec",
                ]
            )

    best_mean_fg = None
    es_cfg = cfg.get("early_stopping", None)
    es_enabled = isinstance(es_cfg, dict) and bool(es_cfg)
    es_monitor = str(es_cfg.get("monitor", "mean_dice_fg")).strip() if es_enabled else None
    es_mode = str(es_cfg.get("mode", "max")).strip().lower() if es_enabled else "max"
    es_patience = int(es_cfg.get("patience", 20)) if es_enabled else 0
    es_best = None
    es_bad_epochs = 0
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()
        topo_aux.set_train_modes(model, freeze_info)
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar, start=1):
            step = _execute_training_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                use_amp=use_amp,
                scaler=scaler,
                freeze_info=freeze_info,
                capture_debug=False,
            )
            loss_dict = step["loss_dict"]
            combined_loss = loss_dict["combined_loss"]
            running_loss += float(combined_loss.item())
            n_batches += 1
            if batch_idx % log_every == 0:
                pbar.set_postfix(loss=f"{running_loss / max(n_batches, 1):.6f}")

        semantic_metrics = topo_aux.compute_semantic_metrics(model, val_loader, device, loss_fn, use_amp)
        epoch_time = time.perf_counter() - epoch_t0
        topo_aux.save_checkpoint(
            out_dir / "last.pth",
            model,
            optimizer,
            epoch,
            cfg,
            extra={
                "checkpoint_type": "last",
                "semantic_metrics": semantic_metrics,
                "checkpoint_init": checkpoint_info,
                "optimizer_groups": optimizer_meta,
            },
        )

        mean_dice_fg = float(semantic_metrics["mean_dice_fg"])
        if best_mean_fg is None or mean_dice_fg > float(best_mean_fg):
            best_mean_fg = float(mean_dice_fg)
            topo_aux.save_checkpoint(
                out_dir / "best_mean_fg.pth",
                model,
                optimizer,
                epoch,
                cfg,
                extra={
                    "checkpoint_type": "semantic",
                    "semantic_metrics": semantic_metrics,
                    "research_only": False,
                },
            )

        if scheduler is not None:
            sched_cfg = cfg.get("scheduler") or {}
            sched_monitor = sched_cfg.get("monitor", "mean_dice_fg")
            sched_value = _get_monitor_value(
                str(sched_monitor),
                val_loss=float(semantic_metrics["combined_loss"]),
                mean_dice_fg=float(semantic_metrics["mean_dice_fg"]),
                mean_iou_fg=float(semantic_metrics["mean_iou_fg"]),
            )
            if sched_value is not None:
                scheduler.step(sched_value)

        if es_enabled:
            es_value = _get_monitor_value(
                str(es_monitor),
                val_loss=float(semantic_metrics["combined_loss"]),
                mean_dice_fg=float(semantic_metrics["mean_dice_fg"]),
                mean_iou_fg=float(semantic_metrics["mean_iou_fg"]),
            )
            if es_value is not None:
                improved = False
                if es_best is None:
                    improved = True
                elif es_mode == "max":
                    improved = float(es_value) > float(es_best)
                elif es_mode == "min":
                    improved = float(es_value) < float(es_best)
                else:
                    raise SystemExit(f"Unsupported early_stopping.mode: {es_mode!r}")

                if improved:
                    es_best = float(es_value)
                    es_bad_epochs = 0
                else:
                    es_bad_epochs += 1
                    if es_bad_epochs >= es_patience:
                        print(
                            f"Early stopping: no improvement in {es_patience} epochs for {es_monitor} (best={es_best:.6f})"
                        )
                        break

        with metrics_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    float(running_loss / max(n_batches, 1)),
                    float(semantic_metrics["combined_loss"]),
                    float(semantic_metrics["mean_dice_fg"]),
                    float(semantic_metrics["mean_iou_fg"]),
                    float(semantic_metrics["leaflet_dice"]),
                    float(semantic_metrics["leaflet_iou"]),
                    float(semantic_metrics["ring_dice"]),
                    float(semantic_metrics["ring_iou"]),
                    float(epoch_time),
                ]
            )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "semantic_mean_dice_fg": semantic_metrics["mean_dice_fg"],
                    "leaflet_dice": semantic_metrics["leaflet_dice"],
                    "ring_dice": semantic_metrics["ring_dice"],
                },
                ensure_ascii=False,
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=topo_aux.REPO_ROOT / "training" / "configs" / "unetpp_effb3_semantic_topology_aux_finetune_100ep.yaml",
    )
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    cfg = topo_aux._read_yaml(args.config.resolve())
    _seed_everything(int(cfg.get("seed", 1337)))
    device = _select_device()
    if args.smoke_test:
        print(json.dumps(smoke_test(cfg, device=device), ensure_ascii=False, indent=2))
        return
    train(cfg, device=device)


if __name__ == "__main__":
    main()
