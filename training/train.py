from __future__ import annotations

import argparse
import csv
import json
import os
import time
import contextlib
from pathlib import Path

import numpy as np
try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt\n"
        "Then re-run training/train.py"
    ) from e
from tqdm import tqdm

from augmentations import get_train_augmentations, get_val_augmentations
from dataset import SegmentationDataset
from losses import CombinedCrossEntropyDiceLoss
from validate import format_metrics, validate


def _simple_preprocess_uint8_rgb(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def _amp_enabled(cfg: dict, device: torch.device) -> bool:
    train_cfg = cfg.get("train") or {}
    if not isinstance(train_cfg, dict):
        return device.type == "cuda"
    v = train_cfg.get("amp", None)
    if v is None:
        return device.type == "cuda"
    return bool(v) and device.type == "cuda"


def _autocast_ctx(device: torch.device, enabled: bool):
    if device.type == "cuda" and bool(enabled):
        return torch.amp.autocast("cuda", enabled=True)
    return contextlib.nullcontext()


def _make_grad_scaler(device: torch.device, enabled: bool):
    if device.type == "cuda" and bool(enabled):
        return torch.amp.GradScaler("cuda")
    return None


def _build_loss_from_cfg(cfg: dict, device: torch.device) -> torch.nn.Module:
    num_classes = int(cfg["model"]["classes"])
    loss_cfg = cfg.get("loss") or {}
    dataset_cfg = cfg.get("dataset") or {}

    ce_coef = loss_cfg.get("ce_coef", None)
    dice_coef = loss_cfg.get("dice_coef", None)

    if ce_coef is None:
        ce_coef = loss_cfg.get("ce_weight", 1.0)
    if dice_coef is None:
        dice_coef = loss_cfg.get("dice_weight", 1.0)

    class_weights = None
    ce_weight_cfg = loss_cfg.get("ce_weight", None)
    if isinstance(ce_weight_cfg, dict):
        if num_classes == 2:
            w_bg = float(ce_weight_cfg.get("background", 1.0))
            w_leaflet = float(ce_weight_cfg.get("leaflet", 1.0))
            class_weights = torch.tensor([w_bg, w_leaflet], dtype=torch.float32, device=device)
        elif num_classes == 3:
            w_bg = float(ce_weight_cfg.get("background", 1.0))
            w_leaflet = float(ce_weight_cfg.get("leaflet", 1.0))
            w_ring = float(ce_weight_cfg.get("fibrous_ring", 1.0))
            class_weights = torch.tensor([w_bg, w_leaflet, w_ring], dtype=torch.float32, device=device)

    ce_class_weights_cfg = loss_cfg.get("ce_class_weights", None)
    if int(num_classes) == 3 and isinstance(ce_class_weights_cfg, list) and len(ce_class_weights_cfg) == 3:
        class_weights = torch.tensor([float(x) for x in ce_class_weights_cfg], dtype=torch.float32, device=device)

    boundary_cfg = loss_cfg.get("boundary", None)
    boundary_enabled = False
    boundary_coef = 0.0
    boundary_mode = "weight_map"
    if isinstance(boundary_cfg, dict):
        boundary_enabled = bool(boundary_cfg.get("enabled", False))
        boundary_coef = float(boundary_cfg.get("coef", 0.0))
        boundary_mode = str(boundary_cfg.get("mode", "weight_map"))

    target = str(dataset_cfg.get("target", "multiclass")).strip().lower() if isinstance(dataset_cfg, dict) else "multiclass"
    if not (num_classes == 2 and target == "leaflet_only"):
        boundary_enabled = False

    return CombinedCrossEntropyDiceLoss(
        num_classes=num_classes,
        ce_coef=float(ce_coef),
        dice_coef=float(dice_coef),
        class_weights=class_weights,
        boundary_enabled=boundary_enabled,
        boundary_coef=boundary_coef,
        boundary_mode=boundary_mode,
    ).to(device)


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


def _build_scheduler_from_cfg(cfg: dict, optimizer: torch.optim.Optimizer):
    sched_cfg = cfg.get("scheduler", None)
    if not isinstance(sched_cfg, dict) or not sched_cfg:
        return None

    t = str(sched_cfg.get("type", "")).strip().lower()
    if t not in {"reduce_on_plateau", "reduce_lr_on_plateau"}:
        raise SystemExit(f"Unsupported scheduler.type: {t!r}")

    mode = str(sched_cfg.get("mode", "max")).strip().lower()
    factor = float(sched_cfg.get("factor", 0.5))
    patience = int(sched_cfg.get("patience", 8))
    min_lr = float(sched_cfg.get("min_lr", 0.0))

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise SystemExit(
            "PyYAML is not installed. Install training deps with:\n"
            "  py -m pip install -r requirements-train.txt"
        ) from e

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a YAML dict: {path}")
    return data


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
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
        print(f"CUDA device: {props.name} ({total_gb:.1f} GB)")
    return device


def _class_names(num_classes: int) -> list[str]:
    if int(num_classes) == 3:
        return ["background", "leaflet", "fibrous_ring"]
    if int(num_classes) == 2:
        return ["background", "leaflet"]
    return [f"class_{i}" for i in range(int(num_classes))]


def _get_save_dir(cfg: dict) -> Path:
    train_cfg = cfg.get("train") or {}
    if not isinstance(train_cfg, dict):
        raise SystemExit("Config: train must be a dict")
    save_dir = train_cfg.get("save_dir") or train_cfg.get("output_dir")
    if not save_dir:
        raise SystemExit("Config: train.save_dir (or train.output_dir) is required")
    return Path(save_dir).resolve()


def _build_loaders(cfg: dict, device: torch.device):
    ds_root = Path(cfg["dataset"]["root"]).resolve()
    train_txt = Path(cfg["dataset"]["train_txt"]).resolve()
    val_txt = Path(cfg["dataset"]["val_txt"]).resolve()

    num_classes = int(cfg["model"]["classes"])
    input_size = int(cfg["model"]["input_size"])

    import segmentation_models_pytorch as smp

    encoder = cfg["model"].get("encoder") or cfg["model"].get("encoder_name")
    if not encoder:
        raise SystemExit("Config: model.encoder (or model.encoder_name) is required")
    encoder_weights = cfg["model"].get("encoder_weights", None)
    if encoder_weights is None:
        preprocessing_fn = _simple_preprocess_uint8_rgb
    else:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    dataset_cfg = cfg.get("dataset") or {}
    target = dataset_cfg.get("target", None)
    crop_mode = dataset_cfg.get("crop_mode", None)
    crop_padding = float(dataset_cfg.get("crop_padding", 0.0)) if isinstance(dataset_cfg, dict) else 0.0
    boundary_cfg = dataset_cfg.get("boundary", None) if isinstance(dataset_cfg, dict) else None

    train_ds = SegmentationDataset(
        dataset_root=ds_root,
        split_txt=train_txt,
        num_classes=num_classes,
        target=target,
        crop_mode=crop_mode,
        crop_padding=crop_padding,
        boundary_cfg=boundary_cfg,
        augment_fn=get_train_augmentations(input_size, input_size, cfg.get("augment", None)),
        preprocessing_fn=preprocessing_fn,
    )
    val_ds = SegmentationDataset(
        dataset_root=ds_root,
        split_txt=val_txt,
        num_classes=num_classes,
        target=target,
        crop_mode=crop_mode,
        crop_padding=crop_padding,
        boundary_cfg=boundary_cfg,
        augment_fn=get_val_augmentations(input_size, input_size),
        preprocessing_fn=preprocessing_fn,
    )

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    pin_memory_cfg = cfg.get("train", {}).get("pin_memory", None)
    pin_memory = bool(pin_memory_cfg) if pin_memory_cfg is not None else (device.type == "cuda")
    persistent_workers_cfg = cfg.get("train", {}).get("persistent_workers", None)
    persistent_workers = bool(persistent_workers_cfg) if persistent_workers_cfg is not None else False
    prefetch_factor_cfg = cfg.get("train", {}).get("prefetch_factor", None)
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


def _build_model(cfg: dict) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    encoder = cfg["model"].get("encoder") or cfg["model"].get("encoder_name")
    if not encoder:
        raise SystemExit("Config: model.encoder (or model.encoder_name) is required")

    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=cfg["model"].get("encoder_weights", None),
        in_channels=int(cfg["model"]["in_channels"]),
        classes=int(cfg["model"]["classes"]),
    )

    train_init_path = (cfg.get("train") or {}).get("init_checkpoint", None)
    init_path = train_init_path or cfg.get("model", {}).get("init_from_checkpoint", None)
    if init_path:
        init_path = str(init_path)
        ckpt = torch.load(init_path, map_location="cpu")
        state = ckpt.get("model") if isinstance(ckpt, dict) else None
        if state is None:
            state = ckpt
        if not isinstance(state, dict):
            raise SystemExit(f"Unsupported checkpoint format: {init_path}")

        if train_init_path:
            model_state = model.state_dict()
            filtered: dict = {}
            loaded_keys: list[str] = []
            skipped_keys: list[str] = []
            for k, v in state.items():
                if str(k).startswith("segmentation_head"):
                    skipped_keys.append(str(k))
                    continue
                if k in model_state and hasattr(v, "shape") and hasattr(model_state[k], "shape") and v.shape == model_state[k].shape:
                    filtered[k] = v
                    loaded_keys.append(str(k))
                else:
                    skipped_keys.append(str(k))

            incompat = model.load_state_dict(filtered, strict=False)
            missing_keys = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []

            print(f"Init from checkpoint (filtered): {init_path}")
            print("loaded keys:")
            for k in sorted(loaded_keys):
                print(f"- {k}")
            print("skipped keys:")
            for k in sorted(skipped_keys):
                print(f"- {k}")
            print("missing keys:")
            for k in sorted(missing_keys):
                print(f"- {k}")
        else:
            ignore_mismatched = bool(cfg.get("model", {}).get("init_ignore_mismatched", True))
            if ignore_mismatched:
                model_state = model.state_dict()
                filtered = {}
                loaded = 0
                skipped = 0
                for k, v in state.items():
                    if k in model_state and hasattr(v, "shape") and hasattr(model_state[k], "shape") and v.shape == model_state[k].shape:
                        filtered[k] = v
                        loaded += 1
                    else:
                        skipped += 1
                model.load_state_dict(filtered, strict=False)
                print(f"Init from checkpoint: {init_path}")
                print(f"  loaded: {loaded}  skipped: {skipped}")
            else:
                model.load_state_dict(state, strict=True)
                print(f"Init from checkpoint (strict): {init_path}")

    return model


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "config": cfg,
        },
        str(path),
    )


def dry_run(cfg: dict, device: torch.device) -> None:
    train_loader, _ = _build_loaders(cfg, device=device)
    model = _build_model(cfg).to(device)
    model.train()

    loss_fn = _build_loss_from_cfg(cfg, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))

    amp_enabled = _amp_enabled(cfg, device=device)
    scaler = _make_grad_scaler(device=device, enabled=amp_enabled)
    steps = int(cfg["train"].get("dry_run_steps", 2))
    print(f"Dry-run steps: {steps}")

    it = iter(train_loader)
    for step in range(steps):
        batch = next(it)
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        boundary = batch.get("boundary", None)
        if boundary is not None:
            boundary = boundary.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device, enabled=amp_enabled):
            logits = model(images)
            loss = loss_fn(logits, masks, boundary_target=boundary)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        print(f"step={step} images={tuple(images.shape)} logits={tuple(logits.shape)} loss={loss.item():.6f}")


def benchmark(cfg: dict, device: torch.device, steps: int) -> None:
    steps = int(steps)
    if steps <= 0:
        raise SystemExit("--benchmark-steps must be > 0")

    train_loader, _ = _build_loaders(cfg, device=device)
    model = _build_model(cfg).to(device)
    model.train()

    loss_fn = _build_loss_from_cfg(cfg, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    amp_enabled = _amp_enabled(cfg, device=device)
    scaler = _make_grad_scaler(device=device, enabled=amp_enabled)

    batch_size = int(cfg["train"]["batch_size"])
    input_size = int(cfg["model"]["input_size"])
    requested_num_workers = int(cfg["train"]["num_workers"])
    effective_num_workers = 0 if device.type != "cuda" else requested_num_workers
    pin_memory_cfg = cfg.get("train", {}).get("pin_memory", None)
    pin_memory = bool(pin_memory_cfg) if pin_memory_cfg is not None else (device.type == "cuda")
    if device.type != "cuda":
        pin_memory = False

    train_samples = len(getattr(train_loader, "dataset", []))
    steps_per_epoch = len(train_loader)

    print("Benchmark")
    print(f"  device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
        print(f"  gpu: {props.name} ({total_gb:.1f} GB)")
    print(f"  input_size: {input_size}")
    print(f"  batch_size: {batch_size}")
    print(f"  num_workers: {effective_num_workers}")
    print(f"  pin_memory: {pin_memory}")
    print(f"  train samples: {train_samples}")
    print(f"  steps_per_epoch: {steps_per_epoch}")
    print(f"  benchmark steps: {steps}")

    it = iter(train_loader)
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in tqdm(range(steps), desc="Benchmark", unit="step"):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        boundary = batch.get("boundary", None)
        if boundary is not None:
            boundary = boundary.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device, enabled=amp_enabled):
            logits = model(images)
            loss = loss_fn(logits, masks, boundary_target=boundary)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    sec_per_step = total_time / steps
    estimated_epoch_sec = sec_per_step * steps_per_epoch
    estimated_min_per_epoch = estimated_epoch_sec / 60.0

    print()
    print(f"total_time_sec: {total_time:.3f}")
    print(f"sec_per_step: {sec_per_step:.3f}")
    print(f"estimated_sec_per_epoch: {estimated_epoch_sec:.1f}")
    print(f"estimated_min_per_epoch: {estimated_min_per_epoch:.2f}")
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserv = torch.cuda.memory_reserved() / (1024**2)
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserv = torch.cuda.max_memory_reserved() / (1024**2)
        print(f"cuda_mem_allocated_mb: {alloc:.1f}")
        print(f"cuda_mem_reserved_mb: {reserv:.1f}")
        print(f"cuda_peak_allocated_mb: {peak_alloc:.1f}")
        print(f"cuda_peak_reserved_mb: {peak_reserv:.1f}")


def train(cfg: dict, device: torch.device) -> None:
    out_dir = _get_save_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = _build_loaders(cfg, device=device)
    model = _build_model(cfg).to(device)

    num_classes = int(cfg["model"]["classes"])
    dataset_target = str((cfg.get("dataset") or {}).get("target", "multiclass")).strip().lower()
    shape_cfg = (cfg.get("train") or {}).get("shape_diagnostics", None)
    if shape_cfg is None:
        shape_enabled = (num_classes == 2 and dataset_target == "leaflet_only") or (num_classes == 3)
    else:
        shape_enabled = bool(shape_cfg)
    loss_fn = _build_loss_from_cfg(cfg, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    amp_enabled = _amp_enabled(cfg, device=device)
    scaler = _make_grad_scaler(device=device, enabled=amp_enabled)
    scheduler = _build_scheduler_from_cfg(cfg, optimizer=optimizer)

    epochs = int(cfg["train"]["epochs"])
    log_every = int(cfg["train"]["log_every"])

    metrics_path = out_dir / "metrics.csv"
    num_classes = int(cfg["model"]["classes"])
    class_names = _class_names(num_classes)

    if not metrics_path.exists():
        with metrics_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "dice_background",
                    "dice_leaflet",
                    "dice_fibrous_ring",
                    "iou_background",
                    "iou_leaflet",
                    "iou_fibrous_ring",
                    "mean_dice_fg",
                    "mean_iou_fg",
                    "epoch_time_sec",
                ]
            )

    best_score = None
    es_cfg = cfg.get("early_stopping", None)
    es_enabled = isinstance(es_cfg, dict) and bool(es_cfg)
    es_monitor = str(es_cfg.get("monitor", "mean_dice_fg")).strip() if es_enabled else None
    es_mode = str(es_cfg.get("mode", "max")).strip().lower() if es_enabled else "max"
    es_patience = int(es_cfg.get("patience", 20)) if es_enabled else 0
    es_best = None
    es_bad_epochs = 0

    writer = None
    tb_enabled = bool((cfg.get("train") or {}).get("tensorboard", False))
    if tb_enabled:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            print("TensorBoard is not installed. Install with:\n  py -m pip install tensorboard")
        else:
            writer = SummaryWriter(log_dir=str((out_dir / "tensorboard").resolve()))

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            boundary = batch.get("boundary", None)
            if boundary is not None:
                boundary = boundary.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(device, enabled=amp_enabled):
                logits = model(images)
                loss = loss_fn(logits, masks, boundary_target=boundary)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

            if batch_idx % log_every == 0:
                pbar.set_postfix(loss=f"{running_loss / n_batches:.6f}")

        train_loss = (running_loss / n_batches) if n_batches else None
        val_metrics = validate(
            model=model,
            loader=val_loader,
            num_classes=num_classes,
            device=device,
            loss_fn=loss_fn,
            shape_diagnostics=shape_enabled,
            shape_out_dir=out_dir,
            epoch=epoch,
        )
        epoch_time_sec = time.perf_counter() - epoch_t0

        dice = val_metrics.get("dice") or []
        iou = val_metrics.get("iou") or []
        val_loss = val_metrics.get("loss")

        dice_fg = [float(dice[i]) for i in range(1, num_classes)] if dice else []
        iou_fg = [float(iou[i]) for i in range(1, num_classes)] if iou else []
        mean_dice_fg = float(sum(dice_fg) / len(dice_fg)) if dice_fg else None
        mean_iou_fg = float(sum(iou_fg) / len(iou_fg)) if iou_fg else None

        if train_loss is not None:
            print(f"train_loss={train_loss:.6f} epoch_time_sec={epoch_time_sec:.1f}")
        if val_loss is not None:
            print(f"val_loss={val_loss:.6f}")
        if dice and iou:
            for i in range(min(num_classes, len(dice), len(iou))):
                name = class_names[i] if i < len(class_names) else f"class_{i}"
                print(f"{name}: dice={dice[i]:.4f} iou={iou[i]:.4f}")
            if mean_dice_fg is not None:
                print(f"mean_dice_fg={mean_dice_fg:.4f} mean_iou_fg={mean_iou_fg:.4f}")
        else:
            print(format_metrics(val_metrics))

        shape = val_metrics.get("shape", None)
        if isinstance(shape, dict) and shape:
            print(
                "shape: "
                f"merged_suspect_count={shape.get('merged_suspect_count')} "
                f"extra_fragments_count={shape.get('extra_fragments_count')} "
                f"mean_pred_components={shape.get('mean_pred_components')} "
                f"mean_gt_components={shape.get('mean_gt_components')}"
            )

        ckpt_path = out_dir / "last.pt"
        _save_checkpoint(ckpt_path, model, optimizer, epoch, cfg)

        score = mean_dice_fg
        if score is not None and (best_score is None or score > best_score):
            best_score = float(score)
            _save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, cfg)
            _save_checkpoint(out_dir / "best_mean_fg.pt", model, optimizer, epoch, cfg)

        if scheduler is not None:
            sched_cfg = cfg.get("scheduler") or {}
            sched_monitor = sched_cfg.get("monitor", "mean_dice_fg")
            sched_value = _get_monitor_value(
                str(sched_monitor),
                val_loss=(float(val_loss) if val_loss is not None else None),
                mean_dice_fg=(float(mean_dice_fg) if mean_dice_fg is not None else None),
                mean_iou_fg=(float(mean_iou_fg) if mean_iou_fg is not None else None),
            )
            if sched_value is not None:
                scheduler.step(sched_value)

        if es_enabled:
            es_value = _get_monitor_value(
                str(es_monitor),
                val_loss=(float(val_loss) if val_loss is not None else None),
                mean_dice_fg=(float(mean_dice_fg) if mean_dice_fg is not None else None),
                mean_iou_fg=(float(mean_iou_fg) if mean_iou_fg is not None else None),
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
            w = csv.writer(f)
            def _pick(arr: list[float], idx: int) -> float | None:
                return float(arr[idx]) if idx < len(arr) else None

            w.writerow(
                [
                    epoch,
                    float(train_loss) if train_loss is not None else None,
                    float(val_loss) if val_loss is not None else None,
                    _pick(dice, 0),
                    _pick(dice, 1),
                    _pick(dice, 2),
                    _pick(iou, 0),
                    _pick(iou, 1),
                    _pick(iou, 2),
                    mean_dice_fg,
                    mean_iou_fg,
                    float(epoch_time_sec),
                ]
            )

        if writer is not None:
            if train_loss is not None:
                writer.add_scalar("train_loss", float(train_loss), int(epoch))
            if val_loss is not None:
                writer.add_scalar("val_loss", float(val_loss), int(epoch))
            if dice and iou:
                if len(dice) > 0:
                    writer.add_scalar("dice/background", float(dice[0]), int(epoch))
                if len(dice) > 1:
                    writer.add_scalar("dice/leaflet", float(dice[1]), int(epoch))
                if len(dice) > 2:
                    writer.add_scalar("dice/fibrous_ring", float(dice[2]), int(epoch))
                if len(iou) > 0:
                    writer.add_scalar("iou/background", float(iou[0]), int(epoch))
                if len(iou) > 1:
                    writer.add_scalar("iou/leaflet", float(iou[1]), int(epoch))
                if len(iou) > 2:
                    writer.add_scalar("iou/fibrous_ring", float(iou[2]), int(epoch))
            if mean_dice_fg is not None:
                writer.add_scalar("mean_dice_fg", float(mean_dice_fg), int(epoch))
            if mean_iou_fg is not None:
                writer.add_scalar("mean_iou_fg", float(mean_iou_fg), int(epoch))
            lr = float(optimizer.param_groups[0].get("lr", 0.0))
            writer.add_scalar("learning_rate", lr, int(epoch))

    if writer is not None:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Training script for semantic segmentation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    seed = int(cfg.get("seed", 1337))
    _seed_everything(seed)

    device = _select_device()

    print(f"Config: {args.config.resolve()}")
    if device.type == "cuda":
        train_cfg = cfg.get("train") or {}
        cudnn_benchmark = bool(train_cfg.get("cudnn_benchmark", False)) if isinstance(train_cfg, dict) else False
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if args.benchmark_steps is not None:
        benchmark(cfg, device=device, steps=int(args.benchmark_steps))
        return
    if args.dry_run:
        dry_run(cfg, device=device)
        return

    train(cfg, device=device)


if __name__ == "__main__":
    main()
