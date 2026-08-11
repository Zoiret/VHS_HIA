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
    return topo_aux.run_smoke_test(cfg=cfg, contract=contract, device=device)


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
                    "oracle_exact_k",
                    "oracle_mean_matched_iou",
                    "oracle_all_iou_ge_0.50",
                    "oracle_gt2_success",
                    "oracle_gt3_success",
                    "epoch_time_sec",
                ]
            )

    best_mean_fg = None
    best_topology = None
    research_manifest = topo_aux._resolve_repo_path(dataset_cfg.get("research_val_manifest", topo_aux.DEFAULT_RESEARCH_MANIFEST), topo_aux.DEFAULT_RESEARCH_MANIFEST)
    research_image_root = topo_aux._resolve_repo_path(dataset_cfg.get("research_image_root", topo_aux.DEFAULT_INSTANCE_ROOT), topo_aux.DEFAULT_INSTANCE_ROOT)
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()
        topo_aux.set_train_modes(model, freeze_info)
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar, start=1):
            images = batch["image"].to(device, non_blocking=True)
            semantic_target = batch["mask"].to(device, non_blocking=True)
            topology_target = batch["topology_target"].to(device, non_blocking=True)
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
            running_loss += float(combined_loss.item())
            n_batches += 1
            if batch_idx % log_every == 0:
                pbar.set_postfix(loss=f"{running_loss / max(n_batches, 1):.6f}")

        semantic_metrics = topo_aux.compute_semantic_metrics(model, val_loader, device, loss_fn, use_amp)
        topology_metrics = topo_aux.evaluate_oracle_k_reconstruction(
            model,
            manifest_path=research_manifest,
            image_root=research_image_root,
            device=device,
            use_amp=use_amp,
        )
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
                "topology_metrics": topology_metrics,
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
                    "topology_metrics": topology_metrics,
                    "research_only": False,
                },
            )

        candidate_topology = {
            "epoch": int(epoch),
            "all_iou_ge_0.50": float(topology_metrics["all_iou_ge_0.50"]),
            "mean_matched_iou": float(topology_metrics["mean_matched_iou"]),
            "gt2_success": float(topology_metrics["gt2_success"]),
            "semantic_mean_fg": float(semantic_metrics["mean_dice_fg"]),
        }
        if topo_aux.topology_reconstruction_better(candidate_topology, best_topology):
            best_topology = candidate_topology
            topo_aux.save_checkpoint(
                out_dir / "best_topology_reconstruction.pth",
                model,
                optimizer,
                epoch,
                cfg,
                extra={
                    "checkpoint_type": "experimental_topology_reconstruction",
                    "semantic_metrics": semantic_metrics,
                    "topology_metrics": topology_metrics,
                    "research_only": True,
                    "normalizer_method": "centroid_distance_k_normalizer",
                    "oracle_k_source": "manifest.gt_instance_count",
                },
            )

        if scheduler is not None:
            scheduler.step(float(semantic_metrics["mean_dice_fg"]))

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
                    float(topology_metrics["exact_k"]),
                    float(topology_metrics["mean_matched_iou"]),
                    float(topology_metrics["all_iou_ge_0.50"]),
                    float(topology_metrics["gt2_success"]),
                    float(topology_metrics["gt3_success"]),
                    float(epoch_time),
                ]
            )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "semantic_mean_dice_fg": semantic_metrics["mean_dice_fg"],
                    "oracle_all_iou_ge_0.50": topology_metrics["all_iou_ge_0.50"],
                    "oracle_gt2_success": topology_metrics["gt2_success"],
                    "oracle_gt3_success": topology_metrics["gt3_success"],
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
