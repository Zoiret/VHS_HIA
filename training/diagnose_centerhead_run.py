from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from augmentations import get_val_augmentations
from dataset_centerhead import SegmentationWithCenterDataset
from losses import CombinedCrossEntropyDiceLoss
from models_centerhead import UnetPlusPlusSemanticCenterHead, load_semantic_checkpoint_non_strict
from validate_centerhead import (
    _case_type,
    _connected_components,
    _extract_metadata_centers,
    _fallback_marker,
    _geometry_topo_u8,
    _iou_matrix,
    _keep_top3_by_area,
    _markers_from_center_map,
    _best_perm_sum,
    _watershed,
    validate_centerhead,
)


def _seed_all(seed: int) -> None:
    s = int(seed)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


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


def _make_device(cfg: dict, device_arg: str) -> torch.device:
    if str(device_arg).strip():
        return torch.device(str(device_arg).strip())
    dev = str((cfg.get("train") or {}).get("device", "")).strip()
    if dev:
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_save_dir(cfg: dict, override: str) -> Path:
    if str(override).strip():
        return Path(str(override)).resolve()
    train_cfg = cfg.get("train") or {}
    save_dir = train_cfg.get("save_dir") or train_cfg.get("output_dir")
    if not save_dir:
        raise SystemExit("Config: train.save_dir is required")
    return Path(save_dir).resolve()


def _build_val_loader(cfg: dict, device: torch.device, *, batch_size: int | None = None, num_workers: int | None = None):
    ds_root = Path(cfg["dataset"]["root"]).resolve()
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
        preprocessing_fn = smp.encoders.get_preprocessing_fn(str(encoder), encoder_weights)

    val_ds = SegmentationWithCenterDataset(
        dataset_root=ds_root,
        split_txt=val_txt,
        num_classes=num_classes,
        augment_fn=get_val_augmentations(input_size, input_size),
        preprocessing_fn=preprocessing_fn,
    )

    bs = int(batch_size if batch_size is not None else int((cfg.get("train") or {}).get("batch_size", 1)))
    nw = int(num_workers if num_workers is not None else int((cfg.get("train") or {}).get("num_workers", 0)))
    if device.type != "cuda":
        nw = 0

    dl_kwargs = {}
    if nw > 0:
        dl_kwargs["persistent_workers"] = bool((cfg.get("train") or {}).get("persistent_workers", False))
        dl_kwargs["prefetch_factor"] = int((cfg.get("train") or {}).get("prefetch_factor", 2))

    loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        **dl_kwargs,
    )
    return loader


def _build_losses(cfg: dict, device: torch.device):
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
        pw = 1.0
    pw = float(min(max(pw, 1.0), float((cfg.get("center") or {}).get("pos_weight_max", 1000.0))))
    center_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device)).to(device)
    return semantic_loss_fn, center_loss_fn


def _build_model_from_semantic_init(cfg: dict) -> torch.nn.Module:
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
        load_semantic_checkpoint_non_strict(model, str(init_path))
    return model


def _load_checkpoint_state(checkpoint_path: Path) -> tuple[dict, int | None]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        return ckpt["model"], int(ckpt.get("epoch")) if ckpt.get("epoch") is not None else None
    if isinstance(ckpt, dict):
        return ckpt, None
    raise SystemExit(f"Unsupported checkpoint format: {checkpoint_path}")


def _instance_score(metrics: dict) -> float | None:
    miou = metrics.get("instance_mean_matched_iou", None)
    mr = metrics.get("instance_merged_rate", None)
    fr = metrics.get("instance_fragmented_rate", None)
    if miou is None or mr is None or fr is None:
        return None
    return float(miou) - 0.25 * float(mr) - 0.15 * float(fr)


def epoch0(cfg: dict, *, save_dir: Path, device: torch.device) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)
    loader = _build_val_loader(cfg, device=device, batch_size=4, num_workers=0)
    model = _build_model_from_semantic_init(cfg).to(device)
    semantic_loss_fn, center_loss_fn = _build_losses(cfg, device=device)
    metrics = validate_centerhead(
        model=model,
        loader=loader,
        num_classes=int(cfg["model"]["classes"]),
        device=device,
        semantic_loss_fn=semantic_loss_fn,
        center_loss_fn=center_loss_fn,
        instance_root=Path(cfg["dataset"]["instance_root"]).resolve(),
        center_thr=float((cfg.get("center") or {}).get("marker_thr", 0.3)),
    )
    metrics["instance_score"] = _instance_score(metrics)
    out_path = save_dir / "epoch0_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"save_dir": str(save_dir), "epoch0_path": str(out_path), "metrics": metrics}


@dataclass(frozen=True)
class _CheckpointSpec:
    tag: str
    path: Path | None


def _collect_checkpoint_specs(save_dir: Path) -> list[_CheckpointSpec]:
    out = []
    for tag in ["best_mean_fg", "best_center_f1", "best_instance_score", "last"]:
        p = (save_dir / f"{tag}.pth").resolve()
        out.append(_CheckpointSpec(tag=tag, path=p if p.exists() else None))
    return out


def compare_checkpoints(cfg: dict, *, save_dir: Path, device: torch.device) -> dict:
    loader = _build_val_loader(cfg, device=device, batch_size=4, num_workers=0)
    semantic_loss_fn, center_loss_fn = _build_losses(cfg, device=device)
    instance_root = Path(cfg["dataset"]["instance_root"]).resolve()

    rows = []
    specs = [_CheckpointSpec(tag="init", path=None)] + _collect_checkpoint_specs(save_dir)
    for spec in specs:
        if spec.tag == "init":
            model = _build_model_from_semantic_init(cfg).to(device)
            epoch = 0
        else:
            if spec.path is None:
                rows.append({"tag": spec.tag, "exists": False})
                continue
            model = _build_model_from_semantic_init(cfg)
            state, epoch = _load_checkpoint_state(spec.path)
            incompat = model.load_state_dict(state, strict=False)
            missing = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
            unexpected = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
            if unexpected:
                raise RuntimeError(f"{spec.tag}: unexpected keys: {unexpected[:10]}")
            if missing:
                raise RuntimeError(f"{spec.tag}: missing keys: {missing[:10]}")
            model = model.to(device)

        m = validate_centerhead(
            model=model,
            loader=loader,
            num_classes=int(cfg["model"]["classes"]),
            device=device,
            semantic_loss_fn=semantic_loss_fn,
            center_loss_fn=center_loss_fn,
            instance_root=instance_root,
            center_thr=float((cfg.get("center") or {}).get("marker_thr", 0.3)),
        )
        m["instance_score"] = _instance_score(m)
        rows.append(
            {
                "tag": spec.tag,
                "exists": True,
                "epoch": int(epoch) if epoch is not None else None,
                "mean_fg": m.get("mean_dice_fg"),
                "dice_leaflet": (m.get("dice") or [None, None, None])[1],
                "dice_ring": (m.get("dice") or [None, None, None])[2],
                "center_precision": m.get("center_precision"),
                "center_recall": m.get("center_recall"),
                "center_f1": m.get("center_f1"),
                "center_pred_count_mean": m.get("center_pred_count_mean"),
                "center_gt_count_mean": m.get("center_gt_count_mean"),
                "center_zero_cases": m.get("center_zero_cases"),
                "center_extra_cases": m.get("center_extra_cases"),
                "instance_exact_count_acc": m.get("instance_exact_count_acc"),
                "instance_merged_rate": m.get("instance_merged_rate"),
                "instance_fragmented_rate": m.get("instance_fragmented_rate"),
                "instance_mean_matched_iou": m.get("instance_mean_matched_iou"),
                "instance_perfect_rate": m.get("instance_perfect_rate"),
                "instance_score": m.get("instance_score"),
            }
        )

    out_csv = save_dir / "checkpoint_comparison.csv"
    out_json = save_dir / "checkpoint_comparison.json"
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"save_dir": str(save_dir), "csv": str(out_csv), "json": str(out_json), "rows": rows}


def threshold_sweep(cfg: dict, *, save_dir: Path, device: torch.device, checkpoint_path: Path, thresholds: list[float]) -> dict:
    loader = _build_val_loader(cfg, device=device, batch_size=4, num_workers=0)
    semantic_loss_fn, center_loss_fn = _build_losses(cfg, device=device)
    instance_root = Path(cfg["dataset"]["instance_root"]).resolve()

    model = _build_model_from_semantic_init(cfg)
    state, epoch = _load_checkpoint_state(checkpoint_path)
    incompat = model.load_state_dict(state, strict=False)
    missing = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
    unexpected = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
    if unexpected or missing:
        raise RuntimeError(f"checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device)

    rows = []
    for thr in thresholds:
        m = validate_centerhead(
            model=model,
            loader=loader,
            num_classes=int(cfg["model"]["classes"]),
            device=device,
            semantic_loss_fn=semantic_loss_fn,
            center_loss_fn=center_loss_fn,
            instance_root=instance_root,
            center_thr=float(thr),
        )
        m["instance_score"] = _instance_score(m)
        rows.append(
            {
                "threshold": float(thr),
                "center_precision": m.get("center_precision"),
                "center_recall": m.get("center_recall"),
                "center_f1": m.get("center_f1"),
                "center_count_acc": m.get("center_count_acc"),
                "center_zero_cases": m.get("center_zero_cases"),
                "center_extra_cases": m.get("center_extra_cases"),
                "instance_exact_count_acc": m.get("instance_exact_count_acc"),
                "instance_merged_rate": m.get("instance_merged_rate"),
                "instance_fragmented_rate": m.get("instance_fragmented_rate"),
                "instance_mean_matched_iou": m.get("instance_mean_matched_iou"),
                "instance_score": m.get("instance_score"),
            }
        )

    out_csv = save_dir / "threshold_sweep.csv"
    out_json = save_dir / "threshold_sweep.json"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    out_json.write_text(json.dumps({"checkpoint": str(checkpoint_path), "epoch": epoch, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    best = max(rows, key=lambda r: float(r.get("center_f1") or 0.0)) if rows else None
    return {"save_dir": str(save_dir), "checkpoint": str(checkpoint_path), "csv": str(out_csv), "json": str(out_json), "best": best}


def _read_u8(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(str(path))
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def _colorize_instances(inst_u8: np.ndarray) -> np.ndarray:
    h, w = inst_u8.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    colors = {
        1: (0, 255, 0),
        2: (255, 0, 0),
        3: (0, 0, 255),
    }
    for k, c in colors.items():
        out[inst_u8 == k] = np.array(c, dtype=np.uint8)
    return out


def _save_compare(
    out_path: Path,
    *,
    original_rgb_u8: np.ndarray,
    gt_center_u16: np.ndarray,
    pred_center_u16: np.ndarray,
    gt_inst_u8: np.ndarray,
    pred_inst_u8: np.ndarray,
) -> None:
    h, w = original_rgb_u8.shape[:2]
    def _heat_u16(x: np.ndarray) -> np.ndarray:
        x8 = (np.clip(x.astype(np.float32) / 65535.0, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return cv2.applyColorMap(x8, cv2.COLORMAP_VIRIDIS)

    a = cv2.cvtColor(original_rgb_u8, cv2.COLOR_RGB2BGR)
    b = _heat_u16(gt_center_u16)
    c = _heat_u16(pred_center_u16)
    d1 = _colorize_instances(gt_inst_u8)
    d2 = _colorize_instances(pred_inst_u8)
    d = cv2.addWeighted(cv2.cvtColor(d1, cv2.COLOR_RGB2BGR), 0.5, cv2.cvtColor(d2, cv2.COLOR_RGB2BGR), 0.5, 0.0)

    top = np.concatenate([a, b], axis=1)
    bot = np.concatenate([c, d], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    if grid.shape[0] != 2 * h or grid.shape[1] != 2 * w:
        grid = cv2.resize(grid, (2 * w, 2 * h), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)


def _reconstruct_instances_from_pred(
    *,
    pred_sem_u8: np.ndarray,
    center_prob_f32: np.ndarray,
    center_thr: float,
) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
    leaf_union = pred_sem_u8 == 1
    pred_pts = [(y, x) for (y, x, _) in _markers_from_center_map(center_prob_f32, leaf_union, float(center_thr), max_markers=3)]

    labels_cc, cc_k = _connected_components(leaf_union.astype(np.uint8))
    pred_inst = np.zeros_like(pred_sem_u8, dtype=np.uint8)
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
    return pred_inst, int(pred_k), pred_pts


def export_center_outputs(
    cfg: dict,
    *,
    save_dir: Path,
    device: torch.device,
    tag: str,
    checkpoint_path: Path | None,
    center_thr: float,
    max_samples: int,
) -> dict:
    out_root = (save_dir / "center_output_diagnostics" / str(tag)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    loader = _build_val_loader(cfg, device=device, batch_size=1, num_workers=0)
    instance_root = Path(cfg["dataset"]["instance_root"]).resolve()

    model = _build_model_from_semantic_init(cfg)
    epoch = 0
    if checkpoint_path is not None:
        state, epoch = _load_checkpoint_state(checkpoint_path)
        incompat = model.load_state_dict(state, strict=False)
        missing = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
        unexpected = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
        if unexpected or missing:
            raise RuntimeError(f"{tag}: checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()

    saved = 0
    center_counters = {"zero_centers": 0, "extra_centers": 0, "correct_center_count": 0}
    instance_counters = {"merged_instances": 0, "fragmented_instances": 0, "mixed_instances": 0, "correct_instances": 0}
    for batch in loader:
        if saved >= int(max_samples):
            break
        sid = Path(str(batch["image_path"][0])).stem
        images = batch["image"].to(device)
        with torch.no_grad():
            out = model(images)
        sem_logits = out["semantic"]
        ctr_logits = out["center"]
        pred_sem = torch.argmax(sem_logits, dim=1).detach().cpu().numpy()[0].astype(np.uint8)
        ctr_prob = torch.sigmoid(ctr_logits).detach().cpu().numpy()[0, 0].astype(np.float32)

        img_f = batch["image"].detach().cpu().numpy()[0].transpose(1, 2, 0)
        img_u8 = (np.clip(img_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        gt_center_f = batch["center"].detach().cpu().numpy()[0, 0].astype(np.float32)
        gt_center_u16 = (np.clip(gt_center_f, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        pred_center_u16 = (np.clip(ctr_prob, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        bin_u8 = (ctr_prob >= float(center_thr)).astype(np.uint8) * 255

        meta_path = str(batch.get("metadata_path", [""])[0])
        gt_pts = _extract_metadata_centers(meta_path) if meta_path else []
        pred_pts_scored = _markers_from_center_map(ctr_prob, pred_sem == 1, float(center_thr), max_markers=3)
        pred_pts = [(y, x) for (y, x, _) in pred_pts_scored]

        gt_inst_path = (instance_root / "instance_masks" / f"{sid}.png").resolve()
        gt_inst = _read_u8(gt_inst_path)
        if gt_inst.shape[:2] != pred_sem.shape[:2]:
            h, w = pred_sem.shape[:2]
            gh, gw = gt_inst.shape[:2]
            y0 = (gh - h) // 2
            x0 = (gw - w) // 2
            gt_inst = gt_inst[y0 : y0 + h, x0 : x0 + w]

        gt_k = int(len([k for k in [1, 2, 3] if int(np.sum(gt_inst == k)) > 0]))
        pred_inst, pred_k, pred_pts_used = _reconstruct_instances_from_pred(pred_sem_u8=pred_sem, center_prob_f32=ctr_prob, center_thr=float(center_thr))

        if len(pred_pts) == 0:
            center_bucket = "zero_centers"
        elif int(len(pred_pts)) == int(len(gt_pts)):
            center_bucket = "correct_center_count"
        else:
            center_bucket = "extra_centers"

        case = _case_type(gt_k, pred_k)
        if case == "merged":
            inst_bucket = "merged_instances"
        elif case == "fragmented":
            inst_bucket = "fragmented_instances"
        elif case == "mixed":
            inst_bucket = "mixed_instances"
        else:
            inst_bucket = "correct_instances"

        center_counters[center_bucket] += 1
        instance_counters[inst_bucket] += 1

        def _save_to(sd: Path) -> None:
            sd.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sd / "original.png"), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(sd / "gt_center.png"), gt_center_u16)
            cv2.imwrite(str(sd / "pred_center_prob.png"), pred_center_u16)
            cv2.imwrite(str(sd / "binary_thr.png"), bin_u8)

            markers_vis = cv2.cvtColor(img_u8.copy(), cv2.COLOR_RGB2BGR)
            for j, (y, x, s) in enumerate(pred_pts_scored, start=1):
                cv2.circle(markers_vis, (int(x), int(y)), 6, (255, 0, 0), 2)
                cv2.putText(
                    markers_vis,
                    str(j),
                    (int(x) + 7, int(y) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    markers_vis,
                    f"{float(s):.2f}",
                    (int(x) + 7, int(y) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            for j, (y, x) in enumerate(gt_pts, start=1):
                cv2.circle(markers_vis, (int(x), int(y)), 6, (0, 255, 255), 2)
            cv2.imwrite(str(sd / "markers.png"), markers_vis)

            cv2.imwrite(str(sd / "gt_instances.png"), gt_inst.astype(np.uint8))
            cv2.imwrite(str(sd / "reconstructed_instances.png"), pred_inst.astype(np.uint8))
            _save_compare(
                sd / "compare.png",
                original_rgb_u8=img_u8,
                gt_center_u16=gt_center_u16,
                pred_center_u16=pred_center_u16,
                gt_inst_u8=gt_inst,
                pred_inst_u8=pred_inst,
            )

            iou_mat = _iou_matrix(gt_inst, pred_inst, gt_k, pred_k)
            sum_iou = _best_perm_sum(iou_mat)
            mean_iou = float(sum_iou / max(gt_k, 1))

            (sd / "metrics.json").write_text(
                json.dumps(
                    {
                        "sample": sid,
                        "tag": str(tag),
                        "checkpoint_epoch": int(epoch) if epoch is not None else None,
                        "center_thr": float(center_thr),
                        "gt_instance_count": int(gt_k),
                        "gt_center_count": int(len(gt_pts)),
                        "pred_center_count": int(len(pred_pts)),
                        "pred_centers": [{"y": int(y), "x": int(x), "score": float(s)} for (y, x, s) in pred_pts_scored],
                        "gt_centers": [{"y": int(y), "x": int(x)} for (y, x) in gt_pts],
                        "pred_pts_used_in_reconstruction": [{"y": int(y), "x": int(x)} for (y, x) in pred_pts_used],
                        "pred_instance_count": int(pred_k),
                        "case": str(case),
                        "mean_matched_iou": float(mean_iou),
                        "buckets": {"center": center_bucket, "instance": inst_bucket},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        _save_to((out_root / "center_count" / center_bucket / sid).resolve())
        _save_to((out_root / "instance" / inst_bucket / sid).resolve())

        saved += 1

    summary = {
        "tag": str(tag),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "saved": int(saved),
        "center_counters": center_counters,
        "instance_counters": instance_counters,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"out_dir": str(out_root), "summary": summary}


def pos_weight_diagnostic(cfg: dict, *, save_dir: Path, device: torch.device, checkpoint_path: Path, center_thr: float) -> dict:
    loader = _build_val_loader(cfg, device=device, batch_size=1, num_workers=0)
    model = _build_model_from_semantic_init(cfg)
    state, epoch = _load_checkpoint_state(checkpoint_path)
    incompat = model.load_state_dict(state, strict=False)
    missing = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
    unexpected = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
    if unexpected or missing:
        raise RuntimeError(f"checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()

    inside_prob_sum = 0.0
    inside_prob_n = 0
    outside_prob_sum = 0.0
    outside_prob_n = 0

    inside_logit_sum = 0.0
    inside_logit_n = 0
    outside_logit_sum = 0.0
    outside_logit_n = 0

    pos_frac_sum = 0.0
    pos_frac_n = 0

    for batch in loader:
        images = batch["image"].to(device)
        gt_center = batch["center"].detach().cpu().numpy()[0, 0].astype(np.float32)
        with torch.no_grad():
            out = model(images)
        ctr_logits = out["center"].detach().cpu().numpy()[0, 0].astype(np.float32)
        ctr_prob = 1.0 / (1.0 + np.exp(-ctr_logits))

        inside = gt_center >= 0.5
        outside = gt_center < 0.1

        if bool(np.any(inside)):
            inside_prob_sum += float(np.mean(ctr_prob[inside]))
            inside_prob_n += 1
            inside_logit_sum += float(np.mean(ctr_logits[inside]))
            inside_logit_n += 1
        if bool(np.any(outside)):
            outside_prob_sum += float(np.mean(ctr_prob[outside]))
            outside_prob_n += 1
            outside_logit_sum += float(np.mean(ctr_logits[outside]))
            outside_logit_n += 1

        pos_frac_sum += float(np.mean((ctr_prob >= float(center_thr)).astype(np.float32)))
        pos_frac_n += 1

    res = {
        "checkpoint": str(checkpoint_path),
        "epoch": int(epoch) if epoch is not None else None,
        "center_thr": float(center_thr),
        "inside_prob_mean": float(inside_prob_sum / max(inside_prob_n, 1)),
        "outside_prob_mean": float(outside_prob_sum / max(outside_prob_n, 1)),
        "inside_logit_mean": float(inside_logit_sum / max(inside_logit_n, 1)),
        "outside_logit_mean": float(outside_logit_sum / max(outside_logit_n, 1)),
        "predicted_positive_pixel_fraction_mean": float(pos_frac_sum / max(pos_frac_n, 1)),
    }
    out_path = save_dir / "pos_weight_diagnostic.json"
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"save_dir": str(save_dir), "json": str(out_path), "stats": res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["epoch0", "compare", "export", "threshold", "posstats"])
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--save-dir", type=str, default="")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--center-thr", type=float, default=0.3)
    ap.add_argument("--thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7")
    args = ap.parse_args()

    cfg = _read_yaml(Path(args.config))
    _seed_all(int(cfg.get("seed", 1337)))
    device = _make_device(cfg, device_arg=str(args.device))
    save_dir = _get_save_dir(cfg, override=str(args.save_dir))

    if args.mode == "epoch0":
        res = epoch0(cfg, save_dir=save_dir, device=device)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    if args.mode == "compare":
        res = compare_checkpoints(cfg, save_dir=save_dir, device=device)
        print(json.dumps({"csv": res["csv"], "json": res["json"]}, ensure_ascii=False, indent=2))
        return

    if args.mode == "export":
        tag = str(args.tag).strip() or "checkpoint"
        ckpt = str(args.checkpoint).strip()
        ckpt_path = Path(ckpt).resolve() if ckpt else None
        res = export_center_outputs(
            cfg,
            save_dir=save_dir,
            device=device,
            tag=tag,
            checkpoint_path=ckpt_path,
            center_thr=float(args.center_thr),
            max_samples=int(args.max_samples),
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    if args.mode == "threshold":
        ckpt = str(args.checkpoint).strip()
        if not ckpt:
            raise SystemExit("--checkpoint is required for threshold mode")
        thresholds = [float(x.strip()) for x in str(args.thresholds).split(",") if str(x).strip()]
        res = threshold_sweep(cfg, save_dir=save_dir, device=device, checkpoint_path=Path(ckpt).resolve(), thresholds=thresholds)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    if args.mode == "posstats":
        ckpt = str(args.checkpoint).strip()
        if not ckpt:
            raise SystemExit("--checkpoint is required for posstats mode")
        res = pos_weight_diagnostic(cfg, save_dir=save_dir, device=device, checkpoint_path=Path(ckpt).resolve(), center_thr=float(args.center_thr))
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
