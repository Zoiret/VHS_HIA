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
from losses import CenterNetFocalHeatmapLoss, CombinedCrossEntropyDiceLoss
from models_centerhead import UnetPlusPlusSemanticCenterHead, load_semantic_checkpoint_non_strict
from validate_centerhead import (
    _best_perm_sum,
    _case_type,
    _connected_components,
    _extract_metadata_centers,
    _fallback_marker,
    _geometry_topo_u8,
    _iou_matrix,
    _keep_top3_by_area,
    _markers_from_center_map,
    _watershed,
    validate_centerhead,
)


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
    center_head_type = str((cfg.get("model") or {}).get("center_head_type", "linear_1x1")).strip().lower() or "linear_1x1"
    model = UnetPlusPlusSemanticCenterHead(
        encoder_name=str(encoder),
        encoder_weights=cfg["model"].get("encoder_weights", None),
        in_channels=int(cfg["model"]["in_channels"]),
        classes=int(cfg["model"]["classes"]),
        center_head_type=center_head_type,
    )

    init_path = (cfg.get("train") or {}).get("init_checkpoint", None)
    center_from_scratch = True
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
        center_missing_prefix = "center_head."
        center_missing = [k for k in missing if str(k).startswith(center_missing_prefix)]
        center_from_scratch = bool(len(center_missing) > 0)

    init_bias = (cfg.get("model") or {}).get("center_head_init_bias", None)
    applied_bias = None
    applied_sigmoid = None
    if init_bias is not None and bool(center_from_scratch):
        b = float(init_bias)
        layer0 = model.center_head_output_layer()
        if layer0 is None or not hasattr(layer0, "bias") or layer0.bias is None:
            raise RuntimeError("center head output bias not found for center_head_init_bias")
        with torch.no_grad():
            layer0.bias.fill_(b)
        applied_bias = float(b)
        applied_sigmoid = float(1.0 / (1.0 + np.exp(-b)))
        print(f"center head initialized from scratch: {center_from_scratch}")
        print(f"applied center bias: {applied_bias}")
        print(f"sigmoid(initial bias): {applied_sigmoid:.6f}")
    else:
        print(f"center head initialized from scratch: {center_from_scratch}")
        if init_bias is not None:
            b = float(init_bias)
            applied_sigmoid = float(1.0 / (1.0 + np.exp(-b)))
            print(f"applied center bias: (skipped)")
            print(f"sigmoid(initial bias): {applied_sigmoid:.6f}")
    return model


def _build_center_loss(cfg: dict, device: torch.device, *, dataset_root: Path, train_txt: Path):
    loss_cfg = cfg.get("center_loss") or {}
    if not isinstance(loss_cfg, dict):
        loss_cfg = {}
    loss_type = str(loss_cfg.get("type", "")).strip().lower()
    if not loss_type:
        loss_type = "bce"

    if loss_type == "centernet_focal":
        alpha = float(loss_cfg.get("alpha", 2.0))
        beta = float(loss_cfg.get("beta", 4.0))
        loss_fn = CenterNetFocalHeatmapLoss(alpha=alpha, beta=beta).to(device)
        return loss_fn, {"type": "centernet_focal", "alpha": alpha, "beta": beta}

    pw = float((cfg.get("center") or {}).get("pos_weight", 0.0) or 0.0)
    if pw <= 0.0:
        pw = _compute_center_pos_weight(dataset_root, train_txt, thr=float((cfg.get("center") or {}).get("pos_weight_thr", 0.5)))
    pw = float(min(max(pw, 1.0), float((cfg.get("center") or {}).get("pos_weight_max", 1000.0))))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device)).to(device)
    return loss_fn, {"type": "bce", "pos_weight": pw}


def _freeze_base_enabled(cfg: dict) -> bool:
    return bool((cfg.get("train") or {}).get("freeze_base", False))


def _apply_freeze_base(model: UnetPlusPlusSemanticCenterHead) -> dict:
    for p in model.base.parameters():
        p.requires_grad = False
    for p in model.center_head.parameters():
        p.requires_grad = True
    model.freeze_base = True
    total_params = int(sum(int(p.numel()) for p in model.parameters()))
    trainable_params = int(sum(int(p.numel()) for p in model.parameters() if bool(p.requires_grad)))
    trainable_names = [n for (n, p) in model.named_parameters() if bool(p.requires_grad)]
    assert all(n.startswith("center_head.") for n in trainable_names), f"Non-center_head trainable params found: {trainable_names[:10]}"
    return {"total_params": total_params, "trainable_params": trainable_params, "trainable_names": trainable_names}


def _set_train_modes(model: UnetPlusPlusSemanticCenterHead, *, freeze_base: bool) -> None:
    if freeze_base:
        model.center_head.train()
        model.base.eval()
    else:
        model.train()


def _collect_batchnorm_stats(model: torch.nn.Module) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    out = []
    for name, m in model.named_modules():
        rm = getattr(m, "running_mean", None)
        rv = getattr(m, "running_var", None)
        if rm is None or rv is None:
            continue
        if not torch.is_tensor(rm) or not torch.is_tensor(rv):
            continue
        out.append((name, rm.detach().clone(), rv.detach().clone()))
    return out


def _max_bn_delta(model: torch.nn.Module, ref: list[tuple[str, torch.Tensor, torch.Tensor]]) -> float:
    max_d = 0.0
    for name, rm0, rv0 in ref:
        m = dict(model.named_modules()).get(name, None)
        if m is None:
            continue
        rm = getattr(m, "running_mean", None)
        rv = getattr(m, "running_var", None)
        if rm is None or rv is None:
            continue
        d1 = float((rm.detach() - rm0).abs().max().item()) if rm.numel() else 0.0
        d2 = float((rv.detach() - rv0).abs().max().item()) if rv.numel() else 0.0
        max_d = max(max_d, d1, d2)
    return float(max_d)


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


def _colorize_instances_u8(inst_u8: np.ndarray) -> np.ndarray:
    h, w = inst_u8.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    colors = {
        0: (0, 0, 0),
        1: (0, 255, 0),
        2: (255, 0, 0),
        3: (0, 0, 255),
    }
    for k, c in colors.items():
        out[inst_u8 == int(k)] = np.asarray(c, dtype=np.uint8)
    return out


def _export_center_diagnostics(
    out_dir: Path,
    model: UnetPlusPlusSemanticCenterHead,
    loader,
    device: torch.device,
    *,
    instance_root: Path,
    center_thr: float,
    tag: str,
    max_samples: int = 20,
) -> None:
    out_root = (out_dir / "center_output_diagnostics" / str(tag)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0
    for batch in loader:
        if saved >= int(max_samples):
            break

        images = batch["image"].to(device)
        image_paths = batch.get("image_path", None)
        meta_paths = batch.get("metadata_path", None)
        if not isinstance(image_paths, list):
            image_paths = [None for _ in range(int(images.shape[0]))]
        if not isinstance(meta_paths, list):
            meta_paths = [None for _ in range(int(images.shape[0]))]

        with torch.no_grad():
            out = model(images)
            sem_logits = out["semantic"]
            ctr_logits = out["center"]
            pred_sem = torch.argmax(sem_logits, dim=1).detach().cpu().numpy().astype(np.uint8)
            ctr_prob = torch.sigmoid(ctr_logits).detach().cpu().numpy().astype(np.float32)

        imgs = images.detach().cpu().clamp(0.0, 1.0).numpy().transpose(0, 2, 3, 1)
        gt_center = batch["center"].detach().cpu().numpy().astype(np.float32)

        for i in range(int(pred_sem.shape[0])):
            if saved >= int(max_samples):
                break
            sid = Path(str(image_paths[i])).stem if isinstance(image_paths[i], str) else f"sample_{saved}"

            img_u8 = (imgs[i] * 255.0 + 0.5).astype(np.uint8)
            gt_center_u16 = (np.clip(gt_center[i, 0], 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
            pr_center_u16 = (np.clip(ctr_prob[i, 0], 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
            thr_u8_01 = (ctr_prob[i, 0] >= 0.1).astype(np.uint8) * 255
            thr_u8_03 = (ctr_prob[i, 0] >= 0.3).astype(np.uint8) * 255

            leaf_union = pred_sem[i] == 1
            pred_pts_scored = _markers_from_center_map(ctr_prob[i, 0], leaf_union, float(center_thr), max_markers=3)
            pred_pts = [(y, x) for (y, x, _) in pred_pts_scored]

            mp = meta_paths[i] if i < len(meta_paths) else None
            gt_pts = _extract_metadata_centers(str(mp)) if isinstance(mp, str) and mp else []

            if int(len(pred_pts)) == 0:
                center_bucket = "zero_centers"
            elif int(len(pred_pts)) == int(len(gt_pts)):
                center_bucket = "correct_center_count"
            else:
                center_bucket = "extra_centers"

            gt_inst_path = (instance_root / "instance_masks" / f"{sid}.png").resolve()
            gt_inst_src = cv2.imread(str(gt_inst_path), cv2.IMREAD_UNCHANGED)
            if gt_inst_src is None:
                saved += 1
                continue
            if gt_inst_src.ndim == 3:
                gt_inst_src = gt_inst_src[:, :, 0]
            gt_inst = gt_inst_src.astype(np.uint8)
            if gt_inst.shape[:2] != pred_sem[i].shape[:2]:
                h, w = pred_sem[i].shape[:2]
                gh, gw = gt_inst.shape[:2]
                y0 = (gh - h) // 2
                x0 = (gw - w) // 2
                gt_inst = gt_inst[y0 : y0 + h, x0 : x0 + w]

            gt_k = int(len([k for k in [1, 2, 3] if int(np.sum(gt_inst == k)) > 0]))
            labels_cc, cc_k = _connected_components(leaf_union.astype(np.uint8))
            pred_inst = np.zeros_like(gt_inst, dtype=np.uint8)
            next_lab = 1
            for comp_id in range(1, int(cc_k) + 1):
                comp01 = labels_cc == comp_id
                in_markers = [(y, x) for (y, x) in pred_pts if bool(comp01[int(y), int(x)])]
                if len(in_markers) == 0:
                    fb = _fallback_marker(comp01)
                    if fb is not None:
                        in_markers = [fb]
                if len(in_markers) <= 1:
                    pred_inst[comp01] = np.uint8(next_lab)
                    next_lab += 1
                    continue
                topo = _geometry_topo_u8(comp01.astype(np.uint8))
                seg = _watershed(comp01.astype(np.uint8), in_markers, topo)
                seg, seg_k = _keep_top3_by_area(seg)
                if seg_k <= 1:
                    pred_inst[comp01] = np.uint8(next_lab)
                    next_lab += 1
                    continue
                for local in range(1, int(seg_k) + 1):
                    pred_inst[seg == local] = np.uint8(next_lab)
                    next_lab += 1
            pred_inst, pred_k = _keep_top3_by_area(pred_inst)

            case = _case_type(gt_k, int(pred_k))
            inst_bucket = str(case)
            if inst_bucket == "correct":
                inst_bucket = "correct_instances"
            if inst_bucket not in {"merged", "fragmented", "mixed", "correct_instances"}:
                inst_bucket = "correct_instances"

            sd_center = (out_root / center_bucket / sid).resolve()
            sd_inst = (out_root / inst_bucket / sid).resolve()
            for sd in [sd_center, sd_inst]:
                sd.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(sd / "original.png"), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(sd / "gt_center.png"), gt_center_u16)
                cv2.imwrite(str(sd / "pred_center_prob.png"), pr_center_u16)
                cv2.imwrite(str(sd / "thresholded_0p1.png"), thr_u8_01)
                cv2.imwrite(str(sd / "thresholded_0p3.png"), thr_u8_03)

                markers_vis = cv2.cvtColor(img_u8.copy(), cv2.COLOR_RGB2BGR)
                for j, (y, x, s) in enumerate(pred_pts_scored, start=1):
                    cv2.circle(markers_vis, (int(x), int(y)), 6, (255, 0, 0), 2)
                    cv2.putText(markers_vis, str(j), (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(markers_vis, f"{float(s):.2f}", (int(x) + 7, int(y) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                for j, (y, x) in enumerate(gt_pts, start=1):
                    cv2.circle(markers_vis, (int(x), int(y)), 6, (0, 255, 255), 2)
                cv2.imwrite(str(sd / "markers.png"), markers_vis)

                cv2.imwrite(str(sd / "gt_instance_mask.png"), gt_inst.astype(np.uint8))
                cv2.imwrite(str(sd / "reconstructed_instances.png"), pred_inst.astype(np.uint8))

                iou_mat = _iou_matrix(gt_inst, pred_inst, gt_k, int(pred_k))
                sum_iou = _best_perm_sum(iou_mat)
                mean_iou = float(sum_iou / max(gt_k, 1))
                (sd / "metrics.json").write_text(
                    json.dumps(
                        {
                            "sample": sid,
                            "tag": str(tag),
                            "center_thr": float(center_thr),
                            "gt_center_count": int(len(gt_pts)),
                            "pred_center_count": int(len(pred_pts)),
                            "gt_instance_count": int(gt_k),
                            "pred_instance_count": int(pred_k),
                            "case": str(case),
                            "mean_matched_iou": float(mean_iou),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                a = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
                b = cv2.applyColorMap(((gt_center_u16.astype(np.float32) / 65535.0) * 255.0 + 0.5).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                c = cv2.applyColorMap(((pr_center_u16.astype(np.float32) / 65535.0) * 255.0 + 0.5).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                d = cv2.addWeighted(
                    cv2.cvtColor(_colorize_instances_u8(gt_inst), cv2.COLOR_RGB2BGR),
                    0.5,
                    cv2.cvtColor(_colorize_instances_u8(pred_inst), cv2.COLOR_RGB2BGR),
                    0.5,
                    0.0,
                )
                top = np.concatenate([a, b], axis=1)
                bot = np.concatenate([c, d], axis=1)
                grid = np.concatenate([top, bot], axis=0)
                cv2.imwrite(str(sd / "compare.png"), grid)

            saved += 1


def _threshold_sweep(
    *,
    model: torch.nn.Module,
    loader,
    num_classes: int,
    device: torch.device,
    semantic_loss_fn: torch.nn.Module,
    center_loss_fn: torch.nn.Module,
    instance_root: Path,
    thresholds: list[float],
) -> dict:
    rows = []
    best = None
    for thr in thresholds:
        m = validate_centerhead(
            model=model,
            loader=loader,
            num_classes=num_classes,
            device=device,
            semantic_loss_fn=semantic_loss_fn,
            center_loss_fn=center_loss_fn,
            instance_root=instance_root,
            center_thr=float(thr),
        )
        inst_score = _instance_score(m)
        row = {
            "threshold": float(thr),
            "center_precision": m.get("center_precision"),
            "center_recall": m.get("center_recall"),
            "center_f1": m.get("center_f1"),
            "center_count_acc": m.get("center_count_acc"),
            "center_zero_cases": m.get("center_zero_cases"),
            "center_extra_cases": m.get("center_extra_cases"),
            "instance_exact_count_acc": m.get("instance_exact_count_acc"),
            "instance_mean_matched_iou": m.get("instance_mean_matched_iou"),
            "instance_score": float(inst_score) if inst_score is not None else None,
        }
        rows.append(row)
        if best is None or float(row.get("center_f1") or 0.0) > float(best.get("center_f1") or 0.0):
            best = row
    return {"rows": rows, "best": best}


def _maybe_run_threshold_sweep(
    cfg: dict,
    *,
    out_dir: Path,
    tag: str,
    model: torch.nn.Module,
    val_loader,
    num_classes: int,
    device: torch.device,
    semantic_loss_fn: torch.nn.Module,
    center_loss_fn: torch.nn.Module,
    instance_root: Path,
) -> None:
    loss_cfg = cfg.get("center_loss") or {}
    if not isinstance(loss_cfg, dict):
        return
    thr_list = loss_cfg.get("threshold_sweep", None)
    if not isinstance(thr_list, list) or not thr_list:
        return
    thresholds = [float(x) for x in thr_list]
    res = _threshold_sweep(
        model=model,
        loader=val_loader,
        num_classes=num_classes,
        device=device,
        semantic_loss_fn=semantic_loss_fn,
        center_loss_fn=center_loss_fn,
        instance_root=instance_root,
        thresholds=thresholds,
    )
    out_p = (out_dir / "threshold_sweeps").resolve()
    out_p.mkdir(parents=True, exist_ok=True)
    (out_p / f"{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


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

    freeze_base = _freeze_base_enabled(cfg)
    train_loader, val_loader = _build_loaders(cfg, device=device)
    model = _build_model(cfg).to(device)
    if freeze_base:
        freeze_info = _apply_freeze_base(model)
        print("=== FREEZE BASE ===")
        print(f"total_params: {freeze_info['total_params']}")
        print(f"trainable_params: {freeze_info['trainable_params']}")
        for n in freeze_info["trainable_names"]:
            print(f"trainable: {n}")
    _set_train_modes(model, freeze_base=freeze_base)

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
    center_loss_fn, center_loss_info = _build_center_loss(cfg, device=device, dataset_root=ds_root, train_txt=train_txt)
    lambda_center = float((cfg.get("center") or {}).get("lambda", 1.0))

    base_lr = float((cfg.get("train") or {}).get("lr_backbone", cfg["train"]["lr"]))
    head_lr = float((cfg.get("train") or {}).get("lr_center_head", base_lr * 10.0))
    clip_norm = float((cfg.get("train") or {}).get("center_grad_clip_norm", 0.0) or 0.0)
    if freeze_base:
        optimizer = torch.optim.AdamW(
            [{"params": list(model.center_head.parameters()), "lr": head_lr}],
            weight_decay=float(cfg["train"]["weight_decay"]),
        )
    else:
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
    bn_ref = _collect_batchnorm_stats(model.base) if freeze_base else []
    for _ in range(int(steps)):
        batch = next(train_it)
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        centers = batch["center"].to(device)
        optimizer.zero_grad(set_to_none=True)
        _set_train_modes(model, freeze_base=freeze_base)
        with torch.no_grad():
            sem_before = model(images)["semantic"].detach().clone() if freeze_base else None
        out = model(images)
        sem_logits = out["semantic"]
        center_logits = out["center"]
        loss_sem = semantic_loss_fn(sem_logits, masks)
        if isinstance(center_loss_fn, CenterNetFocalHeatmapLoss):
            with torch.no_grad():
                pr0 = torch.sigmoid(center_logits.detach()).detach()
                gt0 = centers.detach()
                pos_exact = gt0 >= 0.9999
                near = gt0 >= 0.1
                far = gt0 < 0.1
                prob_pos_mean = float(pr0[pos_exact].mean().item()) if bool(pos_exact.any().item()) else None
                prob_near_mean = float(pr0[near].mean().item()) if bool(near.any().item()) else None
                prob_far_mean = float(pr0[far].mean().item()) if bool(far.any().item()) else None

            details = center_loss_fn(center_logits, centers, return_details=True)
            loss_center = details["loss"]
            center_pos_loss = float(details["pos_loss"].item())
            center_neg_loss = float(details["neg_loss"].item())
            center_num_pos = float(details["num_pos"].item())
            center_mean_pred = float(details["mean_pred"].item())
            if float(center_num_pos) <= 0.0:
                raise SystemExit("Freeze smoke test failed: focal num_pos == 0")
            with torch.no_grad():
                pr = torch.sigmoid(center_logits).detach()
                pos_frac_005 = float((pr >= 0.05).float().mean().item())
                pos_frac_01 = float((pr >= 0.1).float().mean().item())
                pos_frac_03 = float((pr >= 0.3).float().mean().item())
                pos_frac_05 = float((pr >= 0.5).float().mean().item())
        else:
            loss_center = center_loss_fn(center_logits, centers)
            center_pos_loss = None
            center_neg_loss = None
            center_num_pos = None
            center_mean_pred = float(torch.sigmoid(center_logits).detach().mean().item())
            with torch.no_grad():
                pr = torch.sigmoid(center_logits).detach()
                pos_frac_005 = float((pr >= 0.05).float().mean().item())
                pos_frac_01 = float((pr >= 0.1).float().mean().item())
                pos_frac_03 = float((pr >= 0.3).float().mean().item())
                pos_frac_05 = float((pr >= 0.5).float().mean().item())
        loss = loss_center if freeze_base else (loss_sem + float(lambda_center) * loss_center)

        if not bool(torch.isfinite(loss).all().item()):
            raise SystemExit("Smoke test failed: loss is not finite")

        loss.backward()

        grad_mean_abs = float(next(iter(model.center_head.parameters())).grad.detach().abs().mean().item())
        grad_max_abs = float(next(iter(model.center_head.parameters())).grad.detach().abs().max().item())
        if not np.isfinite(grad_mean_abs) or not np.isfinite(grad_max_abs):
            raise SystemExit("Smoke test failed: center grad is not finite")

        grad_norm_before = None
        grad_norm_after = None
        if float(clip_norm) > 0.0:
            params = list(model.center_head.parameters())
            with torch.no_grad():
                s = 0.0
                for p in params:
                    if p.grad is None:
                        continue
                    s += float(torch.sum(p.grad.detach().float() ** 2).item())
                grad_norm_before = float(np.sqrt(s))
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(clip_norm))
            with torch.no_grad():
                s = 0.0
                for p in params:
                    if p.grad is None:
                        continue
                    s += float(torch.sum(p.grad.detach().float() ** 2).item())
                grad_norm_after = float(np.sqrt(s))

        optimizer.step()
        with torch.no_grad():
            sem_after = model(images)["semantic"].detach().clone() if freeze_base else None
        sem_delta = None
        if freeze_base and sem_before is not None and sem_after is not None:
            sem_delta = float((sem_before - sem_after).abs().max().item())
        params_finite = True
        for p in model.center_head.parameters():
            if not bool(torch.isfinite(p.detach()).all().item()):
                params_finite = False
                break
        logits_finite = bool(torch.isfinite(center_logits.detach()).all().item())
        grad = next(iter(model.center_head.parameters())).grad
        grad_norm = float(grad.detach().abs().mean().item()) if grad is not None else 0.0
        base_grad_any = False
        for p in model.base.parameters():
            if p.grad is not None:
                base_grad_any = True
                break
        bn_delta = _max_bn_delta(model.base, bn_ref) if freeze_base else None
        last = {
            "semantic_shape": tuple(sem_logits.shape),
            "center_shape": tuple(center_logits.shape),
            "loss_semantic": float(loss_sem.item()),
            "loss_center": float(loss_center.item()),
            "loss_total": float(loss.item()),
            "center_grad_mean_abs": grad_norm,
            "center_grad_max_abs": float(grad_max_abs),
            "grad_norm_before_clip": grad_norm_before,
            "grad_norm_after_clip": grad_norm_after,
            "center_grad_all_finite": bool(np.isfinite(grad_mean_abs) and np.isfinite(grad_max_abs)),
            "base_grad_any": bool(base_grad_any),
            "base_eval_mode": bool(not model.base.training),
            "center_train_mode": bool(model.center_head.training),
            "semantic_logits_max_abs_delta_after_step": sem_delta,
            "bn_running_stats_max_abs_delta_after_step": bn_delta,
            "center_loss": center_loss_info,
            "lambda_center": float(lambda_center),
            "freeze_base": bool(freeze_base),
            "focal_pos_loss": center_pos_loss,
            "focal_neg_loss": center_neg_loss,
            "focal_num_pos": center_num_pos,
            "center_mean_pred_prob": center_mean_pred,
            "center_prob_mean_pos_exact": prob_pos_mean if isinstance(center_loss_fn, CenterNetFocalHeatmapLoss) else None,
            "center_prob_mean_near": prob_near_mean if isinstance(center_loss_fn, CenterNetFocalHeatmapLoss) else None,
            "center_prob_mean_far": prob_far_mean if isinstance(center_loss_fn, CenterNetFocalHeatmapLoss) else None,
            "center_pos_frac_thr_0p1": pos_frac_01,
            "center_pos_frac_thr_0p3": pos_frac_03,
            "center_pos_frac_thr_0p5": pos_frac_05,
            "center_pos_frac_thr_0p05": pos_frac_005,
            "parameters_finite_after_step": bool(params_finite),
            "logits_finite_after_step": bool(logits_finite),
        }

    if freeze_base:
        if bool(last.get("base_grad_any", False)):
            raise SystemExit("Freeze smoke test failed: base_grad_any=true")
        if not bool(last.get("base_eval_mode", False)):
            raise SystemExit("Freeze smoke test failed: base is not in eval mode")
        if not bool(last.get("center_train_mode", False)):
            raise SystemExit("Freeze smoke test failed: center_head is not in train mode")
        if (last.get("semantic_logits_max_abs_delta_after_step") is None) or float(last["semantic_logits_max_abs_delta_after_step"]) != 0.0:
            raise SystemExit(f"Freeze smoke test failed: semantic logits changed (delta={last.get('semantic_logits_max_abs_delta_after_step')})")
        if (last.get("bn_running_stats_max_abs_delta_after_step") is None) or float(last["bn_running_stats_max_abs_delta_after_step"]) != 0.0:
            raise SystemExit(f"Freeze smoke test failed: BatchNorm stats changed (delta={last.get('bn_running_stats_max_abs_delta_after_step')})")
        if float(last.get("center_grad_mean_abs", 0.0)) <= 0.0:
            raise SystemExit("Freeze smoke test failed: center grad is zero")

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

    freeze_base = _freeze_base_enabled(cfg)
    train_loader, val_loader = _build_loaders(cfg, device=device)
    model = _build_model(cfg).to(device)
    freeze_info = None
    if freeze_base:
        freeze_info = _apply_freeze_base(model)
        print("=== FREEZE BASE ===")
        print(f"total_params: {freeze_info['total_params']}")
        print(f"trainable_params: {freeze_info['trainable_params']}")
        print("trainable parameter groups:")
        for n in freeze_info["trainable_names"]:
            print(f"- {n}")

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
    center_loss_fn, center_loss_info = _build_center_loss(cfg, device=device, dataset_root=ds_root, train_txt=train_txt)
    lambda_center = float((cfg.get("center") or {}).get("lambda", 1.0))

    base_lr = float((cfg.get("train") or {}).get("lr_backbone", cfg["train"]["lr"]))
    head_lr = float((cfg.get("train") or {}).get("lr_center_head", base_lr * 10.0))
    if freeze_base:
        optimizer = torch.optim.AdamW(
            [{"params": list(model.center_head.parameters()), "lr": head_lr}],
            weight_decay=float(cfg["train"]["weight_decay"]),
        )
    else:
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
                    "center_prob_mean_pos",
                    "center_prob_mean_near",
                    "center_prob_mean_far",
                    "center_prob_mean_max",
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

    center_thr = float((cfg.get("center") or {}).get("marker_thr", 0.3))
    semantic_mean_fg0 = None

    _set_train_modes(model, freeze_base=freeze_base)
    val_metrics0 = validate_centerhead(
        model=model,
        loader=val_loader,
        num_classes=num_classes,
        device=device,
        semantic_loss_fn=semantic_loss_fn,
        center_loss_fn=center_loss_fn,
        instance_root=instance_root,
        center_thr=center_thr,
    )
    mean_fg0 = val_metrics0.get("mean_dice_fg", None)
    semantic_mean_fg0 = float(mean_fg0) if mean_fg0 is not None else None
    inst_score0 = _instance_score(val_metrics0)

    with metrics_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                0,
                "",
                float(val_metrics0["semantic_loss"]),
                float(val_metrics0["center_loss"]),
                float(mean_fg0) if mean_fg0 is not None else "",
                float(val_metrics0["dice"][1]) if isinstance(val_metrics0.get("dice"), list) and len(val_metrics0["dice"]) > 1 else "",
                float(val_metrics0["dice"][2]) if isinstance(val_metrics0.get("dice"), list) and len(val_metrics0["dice"]) > 2 else "",
                float(val_metrics0.get("center_f1")) if val_metrics0.get("center_f1") is not None else "",
                float(val_metrics0.get("center_precision")) if val_metrics0.get("center_precision") is not None else "",
                float(val_metrics0.get("center_recall")) if val_metrics0.get("center_recall") is not None else "",
                float(val_metrics0.get("center_pos_frac")) if val_metrics0.get("center_pos_frac") is not None else "",
                float(val_metrics0.get("center_pred_count_mean")) if val_metrics0.get("center_pred_count_mean") is not None else "",
                float(val_metrics0.get("center_gt_count_mean")) if val_metrics0.get("center_gt_count_mean") is not None else "",
                int(val_metrics0.get("center_zero_cases")) if val_metrics0.get("center_zero_cases") is not None else "",
                int(val_metrics0.get("center_extra_cases")) if val_metrics0.get("center_extra_cases") is not None else "",
                float(val_metrics0.get("center_loc_err_px")) if val_metrics0.get("center_loc_err_px") is not None else "",
                float(val_metrics0.get("center_count_acc")) if val_metrics0.get("center_count_acc") is not None else "",
                float(inst_score0) if inst_score0 is not None else "",
                float(val_metrics0["instance_exact_count_acc"]),
                float(val_metrics0["instance_mean_matched_iou"]),
                float(val_metrics0.get("instance_median_matched_iou")) if val_metrics0.get("instance_median_matched_iou") is not None else "",
                float(val_metrics0["instance_merged_rate"]),
                float(val_metrics0["instance_fragmented_rate"]),
                float(val_metrics0.get("instance_mixed_rate")) if val_metrics0.get("instance_mixed_rate") is not None else "",
                float(val_metrics0.get("instance_perfect_rate")) if val_metrics0.get("instance_perfect_rate") is not None else "",
                float(val_metrics0.get("center_prob_mean_pos")) if val_metrics0.get("center_prob_mean_pos") is not None else "",
                float(val_metrics0.get("center_prob_mean_near")) if val_metrics0.get("center_prob_mean_near") is not None else "",
                float(val_metrics0.get("center_prob_mean_far")) if val_metrics0.get("center_prob_mean_far") is not None else "",
                float(val_metrics0.get("center_prob_mean_max")) if val_metrics0.get("center_prob_mean_max") is not None else "",
                "" if freeze_base else float(optimizer.param_groups[0]["lr"]),
                float(optimizer.param_groups[0]["lr"]) if freeze_base else float(optimizer.param_groups[1]["lr"]),
            ]
        )

    if freeze_base:
        _export_center_diagnostics(
            out_dir,
            model,
            val_loader,
            device,
            instance_root=instance_root,
            center_thr=center_thr,
            tag="epoch0",
            max_samples=20,
        )
        _maybe_run_threshold_sweep(
            cfg,
            out_dir=out_dir,
            tag="epoch0",
            model=model,
            val_loader=val_loader,
            num_classes=num_classes,
            device=device,
            semantic_loss_fn=semantic_loss_fn,
            center_loss_fn=center_loss_fn,
            instance_root=instance_root,
        )

    for epoch in range(1, epochs + 1):
        _set_train_modes(model, freeze_base=freeze_base)
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
                loss = loss_center if freeze_base else (loss_sem + float(lambda_center) * loss_center)

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
            center_thr=center_thr,
        )

        mean_fg = val_metrics.get("mean_dice_fg", None)
        center_f1 = val_metrics.get("center_f1", None)
        inst_score = _instance_score(val_metrics)

        lr_backbone_now = 0.0 if freeze_base else float(optimizer.param_groups[0]["lr"])
        lr_center_now = float(optimizer.param_groups[0]["lr"]) if freeze_base else float(optimizer.param_groups[1]["lr"])

        with metrics_csv.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    epoch,
                    train_loss,
                    float(val_metrics["semantic_loss"]),
                    float(val_metrics["center_loss"]),
                    float(mean_fg) if mean_fg is not None else "",
                    float(val_metrics["dice"][1]) if isinstance(val_metrics.get("dice"), list) and len(val_metrics["dice"]) > 1 else "",
                    float(val_metrics["dice"][2]) if isinstance(val_metrics.get("dice"), list) and len(val_metrics["dice"]) > 2 else "",
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
                    float(val_metrics.get("center_prob_mean_pos")) if val_metrics.get("center_prob_mean_pos") is not None else "",
                    float(val_metrics.get("center_prob_mean_near")) if val_metrics.get("center_prob_mean_near") is not None else "",
                    float(val_metrics.get("center_prob_mean_far")) if val_metrics.get("center_prob_mean_far") is not None else "",
                    float(val_metrics.get("center_prob_mean_max")) if val_metrics.get("center_prob_mean_max") is not None else "",
                    "" if freeze_base else lr_backbone_now,
                    lr_center_now,
                ]
            )

        if freeze_base and semantic_mean_fg0 is not None and mean_fg is not None:
            dev = abs(float(mean_fg) - float(semantic_mean_fg0))
            if float(dev) > 0.002:
                raise SystemExit(f"Freeze stability check failed: |mean_fg - mean_fg0|={dev:.6f} > 0.002")

        _save_checkpoint(out_dir / "last.pth", model, optimizer, epoch, cfg, extra={"val": val_metrics})

        improved = False
        if (not freeze_base) and mean_fg is not None and (best_mean_fg is None or float(mean_fg) > float(best_mean_fg)):
            best_mean_fg = float(mean_fg)
            best_epoch_mean_fg = int(epoch)
            _save_checkpoint(out_dir / "best_mean_fg.pth", model, optimizer, epoch, cfg, extra={"val": val_metrics})
            improved = True
        if center_f1 is not None and (best_center_f1 is None or float(center_f1) > float(best_center_f1)):
            best_center_f1 = float(center_f1)
            best_epoch_center = int(epoch)
            _save_checkpoint(out_dir / "best_center_f1.pth", model, optimizer, epoch, cfg, extra={"val": val_metrics})
            if freeze_base:
                _export_center_diagnostics(
                    out_dir,
                    model,
                    val_loader,
                    device,
                    instance_root=instance_root,
                    center_thr=center_thr,
                    tag="best_center_f1",
                    max_samples=20,
                )
                _maybe_run_threshold_sweep(
                    cfg,
                    out_dir=out_dir,
                    tag="best_center_f1",
                    model=model,
                    val_loader=val_loader,
                    num_classes=num_classes,
                    device=device,
                    semantic_loss_fn=semantic_loss_fn,
                    center_loss_fn=center_loss_fn,
                    instance_root=instance_root,
                )
            improved = True
        if inst_score is not None and (best_instance is None or float(inst_score) > float(best_instance)):
            best_instance = float(inst_score)
            best_epoch_instance = int(epoch)
            _save_checkpoint(out_dir / "best_instance_score.pth", model, optimizer, epoch, cfg, extra={"val": val_metrics})
            if freeze_base:
                _export_center_diagnostics(
                    out_dir,
                    model,
                    val_loader,
                    device,
                    instance_root=instance_root,
                    center_thr=center_thr,
                    tag="best_instance_score",
                    max_samples=20,
                )
                _maybe_run_threshold_sweep(
                    cfg,
                    out_dir=out_dir,
                    tag="best_instance_score",
                    model=model,
                    val_loader=val_loader,
                    num_classes=num_classes,
                    device=device,
                    semantic_loss_fn=semantic_loss_fn,
                    center_loss_fn=center_loss_fn,
                    instance_root=instance_root,
                )
            else:
                _export_val_visuals(out_dir, model, val_loader, device, max_samples=20)
            improved = True

        if freeze_base and int(epoch) in {5, 10, 15, 20}:
            _export_center_diagnostics(
                out_dir,
                model,
                val_loader,
                device,
                instance_root=instance_root,
                center_thr=center_thr,
                tag=f"epoch{epoch}",
                max_samples=20,
            )
            _maybe_run_threshold_sweep(
                cfg,
                out_dir=out_dir,
                tag=f"epoch{epoch}",
                model=model,
                val_loader=val_loader,
                num_classes=num_classes,
                device=device,
                semantic_loss_fn=semantic_loss_fn,
                center_loss_fn=center_loss_fn,
                instance_root=instance_root,
            )

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
        if freeze_base:
            print(
                f"epoch={epoch} time={dt:.1f}s train_center_loss={train_loss:.6f} "
                f"mean_fg={mean_fg} center_f1={center_f1} instance_score={inst_score} "
                f"lr_center={lr_center_now:.2e}"
            )
        else:
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
                "center_loss": center_loss_info,
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
