from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

from augmentations import get_train_augmentations, get_val_augmentations
from dataset_centerhead import SegmentationWithCenterDataset
from losses import CombinedCrossEntropyDiceLoss
from models_centerhead import UnetPlusPlusSemanticCenterHead, load_semantic_checkpoint_non_strict
from validate_centerhead import validate_centerhead


def _read_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise SystemExit("pyyaml is not installed. Install training deps with:\n  py -m pip install -r requirements-train.txt") from e
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"Config root must be a dict: {path}")
    return obj


def _simple_preprocess_uint8_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    return (img_rgb_u8.astype(np.float32) / 255.0).astype(np.float32)


def _get_save_dir(cfg: dict) -> Path:
    train_cfg = cfg.get("train") or {}
    if not isinstance(train_cfg, dict):
        raise SystemExit("Config: train must be a dict")
    save_dir = train_cfg.get("save_dir") or train_cfg.get("output_dir")
    if not save_dir:
        raise SystemExit("Config: train.save_dir is required")
    return Path(save_dir).resolve()


def _seed_all(seed: int) -> None:
    s = int(seed)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _make_device(cfg: dict) -> torch.device:
    dev = str((cfg.get("train") or {}).get("device", "")).strip().lower()
    if dev:
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_loaders(cfg: dict, device: torch.device):
    ds_root = Path(cfg["dataset"]["root"]).resolve()
    train_txt = Path(cfg["dataset"]["train_txt"]).resolve()
    val_txt = Path(cfg["dataset"]["val_txt"]).resolve()

    num_classes = int(cfg["model"]["classes"])
    input_size = int(cfg["model"]["input_size"])

    import segmentation_models_pytorch as smp

    encoder = cfg["model"].get("encoder") or cfg["model"].get("encoder_name")
    if not encoder:
        raise SystemExit("Config: model.encoder_name is required")
    encoder_weights = cfg["model"].get("encoder_weights", None)
    if encoder_weights is None:
        preprocessing_fn = _simple_preprocess_uint8_rgb
    else:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    train_ds = SegmentationWithCenterDataset(
        dataset_root=ds_root,
        split_txt=train_txt,
        num_classes=num_classes,
        augment_fn=get_train_augmentations(input_size, input_size, cfg.get("augment", None)),
        preprocessing_fn=preprocessing_fn,
    )
    val_ds = SegmentationWithCenterDataset(
        dataset_root=ds_root,
        split_txt=val_txt,
        num_classes=num_classes,
        augment_fn=get_val_augmentations(input_size, input_size),
        preprocessing_fn=preprocessing_fn,
    )

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    pin_memory = bool((cfg.get("train") or {}).get("pin_memory", device.type == "cuda"))
    persistent_workers = bool((cfg.get("train") or {}).get("persistent_workers", False))
    prefetch_factor = int((cfg.get("train") or {}).get("prefetch_factor", 2))
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


def _compute_center_pos_weight(dataset_root: Path, train_txt: Path, *, thr: float = 0.5, max_pos_weight: float = 1000.0) -> float:
    from dataset import read_split_file

    items = read_split_file(dataset_root, train_txt)
    pos = 0
    total = 0
    thr_u16 = int(float(thr) * 65535.0 + 0.5)
    for it in tqdm(items, desc="Compute pos_weight", unit="sample"):
        sid = Path(it.image_path).stem
        p = (dataset_root / "center_maps" / f"{sid}.png").resolve()
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = m[:, :, 0]
        if m.dtype != np.uint16:
            m = m.astype(np.uint16)
        total += int(m.size)
        pos += int(np.sum(m >= thr_u16))
    neg = max(0, total - pos)
    pw = float(neg / max(pos, 1))
    pw = float(min(pw, float(max_pos_weight)))
    return pw


def _build_model(cfg: dict) -> torch.nn.Module:
    encoder = cfg["model"].get("encoder") or cfg["model"].get("encoder_name")
    if not encoder:
        raise SystemExit("Config: model.encoder_name is required")
    model = UnetPlusPlusSemanticCenterHead(
        encoder_name=str(encoder),
        encoder_weights=cfg["model"].get("encoder_weights", None),
        in_channels=int(cfg["model"]["in_channels"]),
        classes=int(cfg["model"]["classes"]),
    )

    init_path = (cfg.get("train") or {}).get("init_checkpoint", None)
    if init_path:
        missing, unexpected = load_semantic_checkpoint_non_strict(model, str(init_path))
        print(f"Loaded init checkpoint: {init_path}")
        print(f"missing keys: {len(missing)}")
        for k in missing[:50]:
            print(f"- {k}")
        if len(missing) > 50:
            print(f"... ({len(missing) - 50} more)")
        print(f"unexpected keys: {len(unexpected)}")
        for k in unexpected[:50]:
            print(f"- {k}")
        if len(unexpected) > 50:
            print(f"... ({len(unexpected) - 50} more)")
    return model


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, cfg: dict, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "config": cfg,
            "extra": extra,
        },
        str(path),
    )


def _instance_score(metrics: dict) -> float | None:
    miou = metrics.get("instance_mean_matched_iou", None)
    mr = metrics.get("instance_merged_rate", None)
    fr = metrics.get("instance_fragmented_rate", None)
    if miou is None or mr is None or fr is None:
        return None
    return float(miou) - 0.25 * float(mr) - 0.15 * float(fr)


def _autocast_ctx(device: torch.device, enabled: bool):
    if not enabled:
        return torch.autocast(device_type=device.type, enabled=False)
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=True)
    return torch.autocast(device_type=device.type, enabled=False)


def _export_val_visuals(out_dir: Path, model: torch.nn.Module, loader, device: torch.device, *, max_samples: int = 20) -> None:
    out_vis = out_dir / "val_visuals"
    out_vis.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].detach().cpu().numpy().astype(np.uint8)
        centers = batch["center"].detach().cpu().numpy().astype(np.float32)
        paths = batch.get("image_path", None)
        if not isinstance(paths, list):
            paths = [None for _ in range(int(images.shape[0]))]
        with torch.no_grad():
            out = model(images)
            sem_logits = out["semantic"]
            center_logits = out["center"]
            sem_pred = torch.argmax(sem_logits, dim=1).detach().cpu().numpy().astype(np.uint8)
            center_prob = torch.sigmoid(center_logits).detach().cpu().numpy().astype(np.float32)
        imgs = images.detach().cpu().clamp(0.0, 1.0).numpy().transpose(0, 2, 3, 1)
        for i in range(int(imgs.shape[0])):
            if saved >= int(max_samples):
                return
            sid = Path(str(paths[i])).stem if isinstance(paths[i], str) else f"sample_{saved}"
            sd = out_vis / sid
            sd.mkdir(parents=True, exist_ok=True)
            img_u8 = (imgs[i] * 255.0 + 0.5).astype(np.uint8)
            cv2.imwrite(str(sd / "original.png"), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(sd / "gt_semantic.png"), masks[i].astype(np.uint8))
            cv2.imwrite(str(sd / "pred_semantic.png"), sem_pred[i].astype(np.uint8))
            gt_center_u16 = np.clip(centers[i, 0], 0.0, 1.0)
            gt_center_u16 = (gt_center_u16 * 65535.0 + 0.5).astype(np.uint16)
            pr_center_u16 = np.clip(center_prob[i, 0], 0.0, 1.0)
            pr_center_u16 = (pr_center_u16 * 65535.0 + 0.5).astype(np.uint16)
            cv2.imwrite(str(sd / "gt_center.png"), gt_center_u16)
            cv2.imwrite(str(sd / "pred_center.png"), pr_center_u16)
            saved += 1


def _markers_from_center_u16(center_u16: np.ndarray, thr: float, max_markers: int = 3) -> list[dict]:
    cm = center_u16.astype(np.float32) / 65535.0
    bin_m = (cm >= float(thr)).astype(np.uint8)
    n, lab = cv2.connectedComponents(bin_m, connectivity=8)
    out = []
    for li in range(1, int(n)):
        ys, xs = np.where(lab == li)
        if ys.size == 0:
            continue
        vals = cm[ys, xs]
        j = int(np.argmax(vals))
        y = int(ys[j])
        x = int(xs[j])
        out.append({"y": y, "x": x, "score": float(vals[j]), "area": int(ys.size)})
    out.sort(key=lambda d: float(d["score"]), reverse=True)
    return out[: int(max_markers)]


def _export_center_baseline(out_dir: Path, model: torch.nn.Module, loader, device: torch.device, *, max_samples: int, thr: float) -> None:
    out_base = out_dir / "center_baseline"
    out_base.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in loader:
        images = batch["image"].to(device)
        centers = batch["center"].detach().cpu().numpy().astype(np.float32)
        paths = batch.get("image_path", [])
        meta_paths = batch.get("metadata_path", [])
        with torch.no_grad():
            out = model(images)
            center_logits = out["center"]
            center_prob = torch.sigmoid(center_logits).detach().cpu().numpy().astype(np.float32)
        imgs = images.detach().cpu().clamp(0.0, 1.0).numpy().transpose(0, 2, 3, 1)
        for i in range(int(imgs.shape[0])):
            if saved >= int(max_samples):
                return
            sid = Path(str(paths[i])).stem if i < len(paths) and isinstance(paths[i], str) else f"sample_{saved}"
            sd = out_base / sid
            sd.mkdir(parents=True, exist_ok=True)
            img_u8 = (imgs[i] * 255.0 + 0.5).astype(np.uint8)
            cv2.imwrite(str(sd / "original.png"), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))

            gt_u16 = np.clip(centers[i, 0], 0.0, 1.0)
            gt_u16 = (gt_u16 * 65535.0 + 0.5).astype(np.uint16)
            pr_u16 = np.clip(center_prob[i, 0], 0.0, 1.0)
            pr_u16 = (pr_u16 * 65535.0 + 0.5).astype(np.uint16)
            cv2.imwrite(str(sd / "gt_center.png"), gt_u16)
            cv2.imwrite(str(sd / "pred_center.png"), pr_u16)

            pred_markers = _markers_from_center_u16(pr_u16, thr=float(thr), max_markers=3)
            gt_markers = _markers_from_center_u16(gt_u16, thr=float(thr), max_markers=3)
            gt_instance_count = None
            mp = meta_paths[i] if i < len(meta_paths) else None
            if isinstance(mp, str) and mp:
                try:
                    obj = json.loads(Path(mp).read_text(encoding="utf-8"))
                    gt_instance_count = int(obj.get("instance_count", len(gt_markers)))
                except Exception:
                    gt_instance_count = int(len(gt_markers))
            else:
                gt_instance_count = int(len(gt_markers))

            vis = img_u8.copy()
            for j, m in enumerate(pred_markers, start=1):
                cv2.circle(vis, (int(m["x"]), int(m["y"])), 6, (255, 0, 0), 2)
                cv2.putText(vis, str(j), (int(m["x"]) + 7, int(m["y"]) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            for j, m in enumerate(gt_markers, start=1):
                cv2.circle(vis, (int(m["x"]), int(m["y"])), 6, (0, 255, 255), 2)
            cv2.imwrite(str(sd / "markers.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            (sd / "metrics.json").write_text(
                json.dumps(
                    {
                        "sample": sid,
                        "thr": float(thr),
                        "pred_marker_count": int(len(pred_markers)),
                        "gt_marker_count_from_center_map": int(len(gt_markers)),
                        "gt_instance_count": int(gt_instance_count),
                        "pred_markers": pred_markers,
                        "gt_markers": gt_markers,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            saved += 1


def smoke_test(cfg: dict, device: torch.device) -> dict:
    out_dir = _get_save_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== GPU/ENV CHECK ===")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"device: {device.type}")
    if device.type == "cuda":
        print(f"torch.version.cuda: {torch.version.cuda}")
        idx = int(device.index) if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        print(f"GPU: {props.name}")
        print(f"VRAM: {props.total_memory / (1024**3):.2f} GB")
    amp_enabled = bool((cfg.get("train") or {}).get("amp", False)) and device.type == "cuda"
    print(f"AMP: {amp_enabled}")
    print(f"batch_size: {int((cfg.get('train') or {}).get('batch_size', 1))}")

    train_loader, val_loader = _build_loaders(cfg, device=device)
    model = _build_model(cfg).to(device)
    model.train()

    num_classes = int(cfg["model"]["classes"])
    class_weights_cfg = (cfg.get("loss") or {}).get("ce_class_weights", None)
    class_weights = None
    if class_weights_cfg is not None:
        class_weights = torch.tensor([float(x) for x in class_weights_cfg], dtype=torch.float32, device=device)

    semantic_loss_fn = CombinedCrossEntropyDiceLoss(
        num_classes=num_classes,
        ce_coef=float((cfg.get("loss") or {}).get("ce_coef", 1.0)),
        dice_coef=float((cfg.get("loss") or {}).get("dice_coef", 1.0)),
        class_weights=class_weights,
    ).to(device)

    ds_root = Path(cfg["dataset"]["root"]).resolve()
    train_txt = Path(cfg["dataset"]["train_txt"]).resolve()
    pw = float((cfg.get("center") or {}).get("pos_weight", 0.0) or 0.0)
    if pw <= 0.0:
        pw = _compute_center_pos_weight(ds_root, train_txt, thr=float((cfg.get("center") or {}).get("pos_weight_thr", 0.5)))
    pw = float(min(max(pw, 1.0), float((cfg.get("center") or {}).get("pos_weight_max", 1000.0))))
    center_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device)).to(device)
    lambda_center = float((cfg.get("center") or {}).get("lambda", 1.0))

    base_lr = float((cfg.get("train") or {}).get("lr_backbone", cfg["train"]["lr"]))
    head_lr = float((cfg.get("train") or {}).get("lr_center_head", base_lr * 10.0))
    params_base = []
    params_head = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("center_head.") or ".center_head." in n:
            params_head.append(p)
        else:
            params_base.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": params_base, "lr": base_lr}, {"params": params_head, "lr": head_lr}],
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    steps = int((cfg.get("train") or {}).get("smoke_steps", 2))
    train_it = iter(train_loader)
    last = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(int(steps)):
        batch = next(train_it)
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        centers = batch["center"].to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(images)
        sem_logits = out["semantic"]
        center_logits = out["center"]
        loss_sem = semantic_loss_fn(sem_logits, masks)
        loss_center = center_loss_fn(center_logits, centers)
        loss = loss_sem + float(lambda_center) * loss_center
        loss.backward()
        optimizer.step()
        grad = next(iter(model.center_head.parameters())).grad
        grad_norm = float(grad.detach().abs().mean().item()) if grad is not None else 0.0
        enc_grad_ok = None
        for p in model.base.encoder.parameters():
            if p.grad is not None:
                enc_grad_ok = bool(torch.isfinite(p.grad).all().item())
                break
        dec_grad_ok = None
        for p in model.base.decoder.parameters():
            if p.grad is not None:
                dec_grad_ok = bool(torch.isfinite(p.grad).all().item())
                break
        last = {
            "semantic_shape": tuple(sem_logits.shape),
            "center_shape": tuple(center_logits.shape),
            "loss_semantic": float(loss_sem.item()),
            "loss_center": float(loss_center.item()),
            "loss_total": float(loss.item()),
            "center_grad_mean_abs": grad_norm,
            "encoder_grad_finite": enc_grad_ok,
            "decoder_grad_finite": dec_grad_ok,
            "pos_weight": float(pw),
            "lambda_center": float(lambda_center),
        }

    model.eval()
    val_it = iter(val_loader)
    val_losses = []
    with torch.no_grad():
        for _ in range(2):
            vb = next(val_it)
            out = model(vb["image"].to(device))
            v_sem = out["semantic"]
            v_ctr = out["center"]
            v_loss_sem = semantic_loss_fn(v_sem, vb["mask"].to(device))
            v_loss_center = center_loss_fn(v_ctr, vb["center"].to(device))
            v_loss = v_loss_sem + float(lambda_center) * v_loss_center
            val_losses.append(
                {
                    "val_semantic_shape": tuple(v_sem.shape),
                    "val_center_shape": tuple(v_ctr.shape),
                    "val_loss_semantic": float(v_loss_sem.item()),
                    "val_loss_center": float(v_loss_center.item()),
                    "val_loss_total": float(v_loss.item()),
                }
            )
    last["val_batches"] = val_losses
    if device.type == "cuda":
        last["peak_vram_gb"] = float(torch.cuda.max_memory_allocated() / (1024**3))
    return last


def train(cfg: dict, device: torch.device) -> None:
    out_dir = _get_save_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    train_loader, val_loader = _build_loaders(cfg, device=device)
    model = _build_model(cfg).to(device)

    num_classes = int(cfg["model"]["classes"])
    class_weights_cfg = (cfg.get("loss") or {}).get("ce_class_weights", None)
    class_weights = None
    if class_weights_cfg is not None:
        class_weights = torch.tensor([float(x) for x in class_weights_cfg], dtype=torch.float32, device=device)
    semantic_loss_fn = CombinedCrossEntropyDiceLoss(
        num_classes=num_classes,
        ce_coef=float((cfg.get("loss") or {}).get("ce_coef", 1.0)),
        dice_coef=float((cfg.get("loss") or {}).get("dice_coef", 1.0)),
        class_weights=class_weights,
    ).to(device)

    ds_root = Path(cfg["dataset"]["root"]).resolve()
    train_txt = Path(cfg["dataset"]["train_txt"]).resolve()
    pw = float((cfg.get("center") or {}).get("pos_weight", 0.0) or 0.0)
    if pw <= 0.0:
        pw = _compute_center_pos_weight(ds_root, train_txt, thr=float((cfg.get("center") or {}).get("pos_weight_thr", 0.5)))
    pw = float(min(max(pw, 1.0), float((cfg.get("center") or {}).get("pos_weight_max", 1000.0))))
    center_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device)).to(device)
    lambda_center = float((cfg.get("center") or {}).get("lambda", 1.0))

    base_lr = float((cfg.get("train") or {}).get("lr_backbone", cfg["train"]["lr"]))
    head_lr = float((cfg.get("train") or {}).get("lr_center_head", base_lr * 10.0))
    params_base = []
    params_head = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("center_head.") or ".center_head." in n:
            params_head.append(p)
        else:
            params_base.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": params_base, "lr": base_lr}, {"params": params_head, "lr": head_lr}],
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    scheduler_cfg = cfg.get("scheduler") or {}
    scheduler = None
    if str(scheduler_cfg.get("type", "")).strip().lower() == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.get("mode", "max")),
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 5)),
            min_lr=float(scheduler_cfg.get("min_lr", 0.0)),
        )

    early_cfg = cfg.get("early_stopping") or {}
    early_patience = int(early_cfg.get("patience", 20)) if isinstance(early_cfg, dict) else 20
    early_monitor = str(early_cfg.get("monitor", "instance_score")) if isinstance(early_cfg, dict) else "instance_score"
    early_mode = str(early_cfg.get("mode", "max")) if isinstance(early_cfg, dict) else "max"

    epochs = int(cfg["train"]["epochs"])
    log_every = int(cfg["train"].get("log_every", 10))
    amp_enabled = bool((cfg.get("train") or {}).get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    metrics_csv = out_dir / "metrics.csv"
    if not metrics_csv.exists():
        with metrics_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_semantic_loss",
                    "val_center_loss",
                    "mean_dice_fg",
                    "dice_leaflet",
                    "dice_ring",
                    "center_f1",
                    "center_precision",
                    "center_recall",
                    "center_pos_frac",
                    "center_pred_count_mean",
                    "center_gt_count_mean",
                    "center_zero_cases",
                    "center_extra_cases",
                    "center_loc_err_px",
                    "center_count_acc",
                    "instance_score",
                    "instance_exact_count_acc",
                    "instance_mean_matched_iou",
                    "instance_median_matched_iou",
                    "instance_merged_rate",
                    "instance_fragmented_rate",
                    "instance_mixed_rate",
                    "instance_perfect_rate",
                    "lr_backbone",
                    "lr_center_head",
                ]
            )

    instance_root = Path((cfg.get("dataset") or {}).get("instance_root", "datasets/converted_leaflet_instances")).resolve()

    best_mean_fg = None
    best_center_f1 = None
    best_instance = None
    best_epoch_mean_fg = None
    best_epoch_center = None
    best_epoch_instance = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        t0 = time.perf_counter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for bi, batch in enumerate(pbar, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            centers = batch["center"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(device, enabled=amp_enabled):
                out = model(images)
                sem_logits = out["semantic"]
                center_logits = out["center"]
                loss_sem = semantic_loss_fn(sem_logits, masks)
                loss_center = center_loss_fn(center_logits, centers)
                loss = loss_sem + float(lambda_center) * loss_center

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running += float(loss.item())
            n_batches += 1
            if bi % log_every == 0:
                pbar.set_postfix(loss=f"{running / n_batches:.6f}")

        train_loss = float(running / max(n_batches, 1))
        val_metrics = validate_centerhead(
            model=model,
            loader=val_loader,
            num_classes=num_classes,
            device=device,
            semantic_loss_fn=semantic_loss_fn,
            center_loss_fn=center_loss_fn,
            instance_root=instance_root,
            center_thr=float((cfg.get("center") or {}).get("marker_thr", 0.3)),
        )

        mean_fg = val_metrics.get("mean_dice_fg", None)
        center_f1 = val_metrics.get("center_f1", None)
        inst_score = _instance_score(val_metrics)

        lr_backbone_now = float(optimizer.param_groups[0]["lr"])
        lr_center_now = float(optimizer.param_groups[1]["lr"])

        with metrics_csv.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    epoch,
                    train_loss,
                    float(val_metrics["semantic_loss"]),
                    float(val_metrics["center_loss"]),
                    float(mean_fg) if mean_fg is not None else "",
                    float((val_metrics.get("dice") or {}).get("leaflet", "")) if isinstance(val_metrics.get("dice"), dict) else "",
                    float((val_metrics.get("dice") or {}).get("fibrous_ring", "")) if isinstance(val_metrics.get("dice"), dict) else "",
                    float(center_f1) if center_f1 is not None else "",
                    float(val_metrics.get("center_precision")) if val_metrics.get("center_precision") is not None else "",
                    float(val_metrics.get("center_recall")) if val_metrics.get("center_recall") is not None else "",
                    float(val_metrics.get("center_pos_frac")) if val_metrics.get("center_pos_frac") is not None else "",
                    float(val_metrics.get("center_pred_count_mean")) if val_metrics.get("center_pred_count_mean") is not None else "",
                    float(val_metrics.get("center_gt_count_mean")) if val_metrics.get("center_gt_count_mean") is not None else "",
                    int(val_metrics.get("center_zero_cases")) if val_metrics.get("center_zero_cases") is not None else "",
                    int(val_metrics.get("center_extra_cases")) if val_metrics.get("center_extra_cases") is not None else "",
                    float(val_metrics.get("center_loc_err_px")) if val_metrics.get("center_loc_err_px") is not None else "",
                    float(val_metrics.get("center_count_acc")) if val_metrics.get("center_count_acc") is not None else "",
                    float(inst_score) if inst_score is not None else "",
                    float(val_metrics["instance_exact_count_acc"]),
                    float(val_metrics["instance_mean_matched_iou"]),
                    float(val_metrics.get("instance_median_matched_iou")) if val_metrics.get("instance_median_matched_iou") is not None else "",
                    float(val_metrics["instance_merged_rate"]),
                    float(val_metrics["instance_fragmented_rate"]),
                    float(val_metrics.get("instance_mixed_rate")) if val_metrics.get("instance_mixed_rate") is not None else "",
                    float(val_metrics.get("instance_perfect_rate")) if val_metrics.get("instance_perfect_rate") is not None else "",
                    lr_backbone_now,
                    lr_center_now,
                ]
            )

        _save_checkpoint(out_dir / "last.pth", model, optimizer, epoch, cfg, extra={"val": val_metrics})

        improved = False
        if mean_fg is not None and (best_mean_fg is None or float(mean_fg) > float(best_mean_fg)):
            best_mean_fg = float(mean_fg)
            best_epoch_mean_fg = int(epoch)
            _save_checkpoint(out_dir / "best_mean_fg.pth", model, optimizer, epoch, cfg, extra={"val": val_metrics})
            improved = True
        if center_f1 is not None and (best_center_f1 is None or float(center_f1) > float(best_center_f1)):
            best_center_f1 = float(center_f1)
            best_epoch_center = int(epoch)
            _save_checkpoint(out_dir / "best_center_f1.pth", model, optimizer, epoch, cfg, extra={"val": val_metrics})
            improved = True
        if inst_score is not None and (best_instance is None or float(inst_score) > float(best_instance)):
            best_instance = float(inst_score)
            best_epoch_instance = int(epoch)
            _save_checkpoint(out_dir / "best_instance_score.pth", model, optimizer, epoch, cfg, extra={"val": val_metrics})
            _export_val_visuals(out_dir, model, val_loader, device, max_samples=20)
            improved = True

        if scheduler is not None:
            monitor_key = str((scheduler_cfg or {}).get("monitor", early_monitor))
            monitor_val = val_metrics.get(monitor_key, None)
            if monitor_val is None and monitor_key == "instance_score":
                monitor_val = inst_score
            if monitor_val is not None:
                scheduler.step(float(monitor_val))

        monitor_val_es = val_metrics.get(early_monitor, None)
        if monitor_val_es is None and early_monitor == "instance_score":
            monitor_val_es = inst_score
        if monitor_val_es is None:
            monitor_val_es = inst_score

        if monitor_val_es is None:
            no_improve += 1
        else:
            if best_instance is None and early_monitor != "instance_score":
                pass
            if improved:
                no_improve = 0
            else:
                no_improve += 1

        dt = time.perf_counter() - t0
        print(
            f"epoch={epoch} time={dt:.1f}s train_loss={train_loss:.6f} "
            f"mean_fg={mean_fg} center_f1={center_f1} instance_score={inst_score} "
            f"lr_backbone={lr_backbone_now:.2e} lr_center={lr_center_now:.2e}"
        )

        if no_improve >= int(early_patience):
            print(f"Early stopping: no improvement for {no_improve} epochs (monitor={early_monitor})")
            break

    (out_dir / "best_summary.json").write_text(
        json.dumps(
            {
                "best_mean_fg": best_mean_fg,
                "best_epoch_mean_fg": best_epoch_mean_fg,
                "best_center_f1": best_center_f1,
                "best_epoch_center_f1": best_epoch_center,
                "best_instance_score": best_instance,
                "best_epoch_instance_score": best_epoch_instance,
                "pos_weight": pw,
                "lambda_center": lambda_center,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--export-center-baseline", type=int, default=0)
    args = ap.parse_args()

    cfg = _read_yaml(args.config.resolve())
    _seed_all(int(cfg.get("seed", 1337)))
    device = _make_device(cfg)
    print(f"Device: {device}")
    if args.smoke_test:
        res = smoke_test(cfg, device=device)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    if int(args.export_center_baseline) > 0:
        out_dir = _get_save_dir(cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        _, val_loader = _build_loaders(cfg, device=device)
        model = _build_model(cfg).to(device)
        _export_center_baseline(
            out_dir,
            model,
            val_loader,
            device,
            max_samples=int(args.export_center_baseline),
            thr=float((cfg.get("center") or {}).get("marker_thr", 0.3)),
        )
        return
    train(cfg, device=device)


if __name__ == "__main__":
    main()
