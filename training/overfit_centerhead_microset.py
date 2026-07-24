from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from augmentations import get_val_augmentations
from dataset import read_split_file
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


def _seed_all(seed: int) -> None:
    s = int(seed)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _make_device(device: str) -> torch.device:
    d = str(device).strip()
    if d:
        return torch.device(d)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _center_bias_init(model: UnetPlusPlusSemanticCenterHead, bias: float) -> None:
    try:
        layer0 = model.center_head[0]
    except Exception:
        layer0 = None
    if layer0 is None or not hasattr(layer0, "bias") or layer0.bias is None:
        raise RuntimeError("center_head[0].bias not found for bias init")
    with torch.no_grad():
        layer0.bias.fill_(float(bias))


def _freeze_base(model: UnetPlusPlusSemanticCenterHead) -> None:
    for p in model.base.parameters():
        p.requires_grad = False
    for p in model.center_head.parameters():
        p.requires_grad = True
    model.freeze_base = True
    model.base.eval()
    model.center_head.train()


def _build_loader_for_split(
    cfg: dict,
    *,
    dataset_root: Path,
    split_txt: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
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

    ds = SegmentationWithCenterDataset(
        dataset_root=dataset_root,
        split_txt=split_txt,
        num_classes=num_classes,
        augment_fn=get_val_augmentations(input_size, input_size),
        preprocessing_fn=preprocessing_fn,
    )

    nw = int(num_workers)
    if device.type != "cuda":
        nw = 0

    dl_kwargs = {}
    if nw > 0:
        dl_kwargs["persistent_workers"] = False
        dl_kwargs["prefetch_factor"] = 2

    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        **dl_kwargs,
    )


@dataclass(frozen=True)
class Microset:
    split_txt: Path
    samples: list[str]
    distribution: dict[int, int]


def _select_microset(dataset_root: Path, train_txt: Path, out_dir: Path) -> Microset:
    items = read_split_file(dataset_root, train_txt)
    by_k: dict[int, list[str]] = {1: [], 2: [], 3: []}
    for it in items:
        sid = Path(it.image_path).stem
        meta = (dataset_root / "metadata" / f"{sid}.json").resolve()
        if not meta.exists():
            continue
        obj = json.loads(meta.read_text(encoding="utf-8"))
        k = int(obj.get("instance_count", 0) or 0)
        if k in by_k:
            by_k[k].append(sid)
    for k in by_k:
        by_k[k] = sorted(set(by_k[k]))
    sel = []
    for k in [1, 2, 3]:
        if len(by_k[k]) < 2:
            raise SystemExit(f"Not enough train samples for instance_count={k}: {len(by_k[k])}")
        sel.extend(by_k[k][:2])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = (out_dir / "microset.txt").resolve()
    lines = []
    for sid in sel:
        img_rel = f"images/{sid}.png"
        mask_rel = f"semantic_masks/{sid}.png"
        lines.append(f"{img_rel}\t{mask_rel}")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dist = {1: 2, 2: 2, 3: 2}
    return Microset(split_txt=out_txt, samples=sel, distribution=dist)


def _safe_sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(x))))


def _grad_l2_norm(params: list[torch.Tensor]) -> float:
    s = 0.0
    for p in params:
        if p.grad is None:
            continue
        s += float(torch.sum(p.grad.detach().float() ** 2).item())
    return float(math.sqrt(max(s, 0.0)))


def _params_finite(params: list[torch.Tensor]) -> bool:
    for p in params:
        if not bool(torch.isfinite(p.detach()).all().item()):
            return False
    return True


def _threshold_sweep_on_microset(
    *,
    model: UnetPlusPlusSemanticCenterHead,
    loader,
    device: torch.device,
    instance_root: Path,
    thresholds: list[float],
) -> dict:
    model.eval()
    best = None
    rows = []

    for thr in thresholds:
        tp = fp = fn = 0
        loc_err_sum = 0.0
        loc_err_n = 0
        count_ok = 0
        count_n = 0

        inst_exact = 0
        inst_n = 0
        inst_mean_iou_sum = 0.0
        inst_perfect = 0

        prob_pos_sum = 0.0
        prob_pos_n = 0
        prob_near_sum = 0.0
        prob_near_n = 0
        prob_far_sum = 0.0
        prob_far_n = 0
        prob_max_sum = 0.0
        prob_max_n = 0

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                centers = batch["center"].detach().cpu().numpy().astype(np.float32)
                meta_paths = batch.get("metadata_path", [])
                image_paths = batch.get("image_path", [])
                out = model(images)
                pred_sem = torch.argmax(out["semantic"], dim=1).detach().cpu().numpy().astype(np.uint8)
                pr_center = torch.sigmoid(out["center"]).detach().cpu().numpy().astype(np.float32)

                if not isinstance(meta_paths, list):
                    meta_paths = [None for _ in range(int(pred_sem.shape[0]))]
                if not isinstance(image_paths, list):
                    image_paths = [None for _ in range(int(pred_sem.shape[0]))]

                for i in range(int(pred_sem.shape[0])):
                    leaf_union = pred_sem[i] == 1
                    pred_pts_scored = _markers_from_center_map(pr_center[i, 0], leaf_union, float(thr), max_markers=3)
                    pred_pts = [(y, x) for (y, x, _) in pred_pts_scored]

                    mp = meta_paths[i] if i < len(meta_paths) else None
                    gt_pts = _extract_metadata_centers(str(mp)) if isinstance(mp, str) and mp else []

                    used_gt = set()
                    matches = []
                    for py, px in pred_pts:
                        best_j = None
                        best_d = None
                        for gi, (gy, gx) in enumerate(gt_pts):
                            if gi in used_gt:
                                continue
                            d = float(np.hypot(float(py - gy), float(px - gx)))
                            if best_d is None or d < best_d:
                                best_d = d
                                best_j = gi
                        if best_j is not None and best_d is not None and best_d <= 16.0:
                            used_gt.add(best_j)
                            matches.append(best_d)

                    tpi = int(len(matches))
                    fpi = int(max(0, len(pred_pts) - tpi))
                    fni = int(max(0, len(gt_pts) - tpi))
                    tp += tpi
                    fp += fpi
                    fn += fni
                    for d in matches:
                        loc_err_sum += float(d)
                        loc_err_n += 1
                    count_n += 1
                    count_ok += int(len(pred_pts) == len(gt_pts))

                    gt_map = centers[i, 0]
                    pr_map = pr_center[i, 0]
                    pos_exact = gt_map >= 0.9999
                    near = gt_map >= 0.1
                    far = gt_map < 0.1
                    if bool(np.any(pos_exact)):
                        prob_pos_sum += float(np.mean(pr_map[pos_exact]))
                        prob_pos_n += 1
                    if bool(np.any(near)):
                        prob_near_sum += float(np.mean(pr_map[near]))
                        prob_near_n += 1
                    if bool(np.any(far)):
                        prob_far_sum += float(np.mean(pr_map[far]))
                        prob_far_n += 1
                    prob_max_sum += float(np.max(pr_map))
                    prob_max_n += 1

                    sid = Path(str(image_paths[i])).stem if isinstance(image_paths[i], str) else None
                    if not sid:
                        continue
                    gt_inst_path = (instance_root / "instance_masks" / f"{sid}.png").resolve()
                    gt_inst = cv2.imread(str(gt_inst_path), cv2.IMREAD_UNCHANGED)
                    if gt_inst is None:
                        continue
                    if gt_inst.ndim == 3:
                        gt_inst = gt_inst[:, :, 0]
                    gt_inst = gt_inst.astype(np.uint8)
                    if gt_inst.shape[:2] != pred_sem[i].shape[:2]:
                        h, w = pred_sem[i].shape[:2]
                        gh, gw = gt_inst.shape[:2]
                        y0 = (gh - h) // 2
                        x0 = (gw - w) // 2
                        gt_inst = gt_inst[y0 : y0 + h, x0 : x0 + w]
                    gt_k = int(len([k for k in [1, 2, 3] if int(np.sum(gt_inst == k)) > 0]))
                    if gt_k <= 0:
                        continue

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

                    inst_n += 1
                    inst_exact += int(int(pred_k) == int(gt_k))
                    iou_mat = _iou_matrix(gt_inst, pred_inst, gt_k, int(pred_k))
                    sum_iou = _best_perm_sum(iou_mat)
                    mean_iou = float(sum_iou / max(gt_k, 1))
                    inst_mean_iou_sum += float(mean_iou)
                    inst_perfect += int((int(pred_k) == int(gt_k)) and (mean_iou >= 0.90))

        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float((2 * precision * recall) / max(precision + recall, 1e-7))
        loc_err = float(loc_err_sum / max(loc_err_n, 1))
        count_acc = float(count_ok / max(count_n, 1))
        inst_exact_acc = float(inst_exact / max(inst_n, 1))
        inst_mean_iou = float(inst_mean_iou_sum / max(inst_n, 1))
        inst_perfect_rate = float(inst_perfect / max(inst_n, 1))
        row = {
            "threshold": float(thr),
            "center_precision": precision,
            "center_recall": recall,
            "center_f1": f1,
            "center_count_acc": count_acc,
            "center_loc_err_px": loc_err,
            "center_prob_mean_pos": float(prob_pos_sum / max(prob_pos_n, 1)),
            "center_prob_mean_near": float(prob_near_sum / max(prob_near_n, 1)),
            "center_prob_mean_far": float(prob_far_sum / max(prob_far_n, 1)),
            "center_prob_mean_max": float(prob_max_sum / max(prob_max_n, 1)),
            "instance_exact_count_acc": inst_exact_acc,
            "instance_mean_matched_iou": inst_mean_iou,
            "instance_perfect_rate": inst_perfect_rate,
        }
        rows.append(row)
        if best is None or float(row["center_f1"]) > float(best["center_f1"]):
            best = row

    return {"rows": rows, "best": best}


def _export_visuals(
    *,
    out_dir: Path,
    model: UnetPlusPlusSemanticCenterHead,
    loader,
    device: torch.device,
    instance_root: Path,
    tag: str,
    best_threshold: float,
) -> None:
    out_root = (out_dir / "visuals" / str(tag)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            image_paths = batch.get("image_path", [])
            meta_paths = batch.get("metadata_path", [])
            if not isinstance(image_paths, list):
                image_paths = [None for _ in range(int(images.shape[0]))]
            if not isinstance(meta_paths, list):
                meta_paths = [None for _ in range(int(images.shape[0]))]

            out = model(images)
            pred_sem = torch.argmax(out["semantic"], dim=1).detach().cpu().numpy().astype(np.uint8)
            pr_center = torch.sigmoid(out["center"]).detach().cpu().numpy().astype(np.float32)
            imgs = images.detach().cpu().clamp(0.0, 1.0).numpy().transpose(0, 2, 3, 1)
            gt_center = batch["center"].detach().cpu().numpy().astype(np.float32)

            for i in range(int(pred_sem.shape[0])):
                sid = Path(str(image_paths[i])).stem if isinstance(image_paths[i], str) else f"sample_{i}"
                sd = (out_root / sid).resolve()
                sd.mkdir(parents=True, exist_ok=True)

                img_u8 = (imgs[i] * 255.0 + 0.5).astype(np.uint8)
                gt_u16 = (np.clip(gt_center[i, 0], 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
                pr_u16 = (np.clip(pr_center[i, 0], 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
                diff = np.abs(pr_center[i, 0] - gt_center[i, 0]).astype(np.float32)
                diff_u16 = (np.clip(diff, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)

                cv2.imwrite(str(sd / "original.png"), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(sd / "gt_center.png"), gt_u16)
                cv2.imwrite(str(sd / "pred_center_prob.png"), pr_u16)
                cv2.imwrite(str(sd / "diff_map.png"), diff_u16)

                leaf_union = pred_sem[i] == 1
                pred_pts_scored = _markers_from_center_map(pr_center[i, 0], leaf_union, float(best_threshold), max_markers=3)
                pred_pts = [(y, x) for (y, x, _) in pred_pts_scored]
                mp = meta_paths[i] if i < len(meta_paths) else None
                gt_pts = _extract_metadata_centers(str(mp)) if isinstance(mp, str) and mp else []

                markers_vis = cv2.cvtColor(img_u8.copy(), cv2.COLOR_RGB2BGR)
                for j, (y, x, s) in enumerate(pred_pts_scored, start=1):
                    cv2.circle(markers_vis, (int(x), int(y)), 6, (255, 0, 0), 2)
                    cv2.putText(markers_vis, str(j), (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(markers_vis, f"{float(s):.2f}", (int(x) + 7, int(y) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                for j, (y, x) in enumerate(gt_pts, start=1):
                    cv2.circle(markers_vis, (int(x), int(y)), 6, (0, 255, 255), 2)
                cv2.imwrite(str(sd / "markers.png"), markers_vis)

                gt_inst_path = (instance_root / "instance_masks" / f"{sid}.png").resolve()
                gt_inst = cv2.imread(str(gt_inst_path), cv2.IMREAD_UNCHANGED)
                if gt_inst is not None:
                    if gt_inst.ndim == 3:
                        gt_inst = gt_inst[:, :, 0]
                    gt_inst = gt_inst.astype(np.uint8)
                    if gt_inst.shape[:2] != pred_sem[i].shape[:2]:
                        h, w = pred_sem[i].shape[:2]
                        gh, gw = gt_inst.shape[:2]
                        y0 = (gh - h) // 2
                        x0 = (gw - w) // 2
                        gt_inst = gt_inst[y0 : y0 + h, x0 : x0 + w]
                    cv2.imwrite(str(sd / "gt_instances.png"), gt_inst)

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
                    cv2.imwrite(str(sd / "reconstructed_instances.png"), pred_inst)

                (sd / "metrics.json").write_text(
                    json.dumps(
                        {
                            "sample": sid,
                            "best_threshold": float(best_threshold),
                            "pred_centers": [{"y": int(y), "x": int(x), "score": float(s)} for (y, x, s) in pred_pts_scored],
                            "gt_centers": [{"y": int(y), "x": int(x)} for (y, x) in gt_pts],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("training/analysis/centerhead_micro_overfit"))
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--vis-iters", type=str, default="0,50,100,250,500,1000")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip-norm", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    cfg = _read_yaml(args.config.resolve())
    _seed_all(int(cfg.get("seed", 1337)))
    device = _make_device(args.device)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(cfg["dataset"]["root"]).resolve()
    train_txt = Path(cfg["dataset"]["train_txt"]).resolve()
    instance_root = Path((cfg.get("dataset") or {}).get("instance_root", "datasets/converted_leaflet_instances")).resolve()

    micro = _select_microset(dataset_root, train_txt, out_dir)
    (out_dir / "microset_manifest.json").write_text(
        json.dumps({"samples": micro.samples, "distribution": micro.distribution}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    loader = _build_loader_for_split(
        cfg,
        dataset_root=dataset_root,
        split_txt=micro.split_txt,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )

    encoder = cfg["model"].get("encoder") or cfg["model"].get("encoder_name")
    model = UnetPlusPlusSemanticCenterHead(
        encoder_name=str(encoder),
        encoder_weights=cfg["model"].get("encoder_weights", None),
        in_channels=int(cfg["model"]["in_channels"]),
        classes=int(cfg["model"]["classes"]),
    )
    init_path = (cfg.get("train") or {}).get("init_checkpoint", None)
    if not init_path:
        raise SystemExit("Config: train.init_checkpoint is required")
    missing, unexpected = load_semantic_checkpoint_non_strict(model, str(init_path))
    center_missing = {"center_head.0.weight", "center_head.0.bias"}
    center_from_scratch = bool(center_missing.issubset(set(missing)))
    if not center_from_scratch:
        raise SystemExit("Expected center_head to be from scratch in micro-overfit setup")

    bias = float((cfg.get("model") or {}).get("center_head_init_bias", -2.19))
    _center_bias_init(model, bias=bias)
    _freeze_base(model)
    model = model.to(device)

    focal_cfg = cfg.get("center_loss") or {}
    alpha = float((focal_cfg.get("alpha", 2.0) if isinstance(focal_cfg, dict) else 2.0))
    beta = float((focal_cfg.get("beta", 4.0) if isinstance(focal_cfg, dict) else 4.0))
    center_loss_fn = CenterNetFocalHeatmapLoss(alpha=alpha, beta=beta).to(device)

    num_classes = int(cfg["model"]["classes"])
    class_weights_cfg = (cfg.get("loss") or {}).get("ce_class_weights", None)
    class_weights = torch.tensor([float(x) for x in class_weights_cfg], dtype=torch.float32, device=device) if class_weights_cfg else None
    semantic_loss_fn = CombinedCrossEntropyDiceLoss(
        num_classes=num_classes,
        ce_coef=float((cfg.get("loss") or {}).get("ce_coef", 1.0)),
        dice_coef=float((cfg.get("loss") or {}).get("dice_coef", 1.0)),
        class_weights=class_weights,
    ).to(device)

    opt = torch.optim.AdamW(model.center_head.parameters(), lr=float(args.lr), weight_decay=0.0)
    clip_norm = float(args.grad_clip_norm)
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]

    vis_iters = sorted({int(x.strip()) for x in str(args.vis_iters).split(",") if str(x).strip()})
    metrics_csv = (out_dir / "micro_overfit_metrics.csv").resolve()
    if not metrics_csv.exists():
        with metrics_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "iter",
                    "loss",
                    "pos_loss_sum",
                    "neg_loss_sum",
                    "num_pos",
                    "center_prob_mean_pos",
                    "center_prob_mean_near",
                    "center_prob_mean_far",
                    "center_prob_mean_max",
                    "best_thr",
                    "best_f1",
                    "best_count_acc",
                    "best_loc_err_px",
                    "inst_exact_count_acc",
                    "inst_mean_matched_iou",
                    "inst_perfect_rate",
                    "grad_norm_before",
                    "grad_norm_after",
                    "clipped",
                    "center_weight_norm",
                    "center_bias",
                    "logits_min",
                    "logits_max",
                    "params_finite",
                ]
            )

    clipped_n = 0
    eval_every = int(args.eval_every)
    iters = int(args.iters)

    def _eval_and_log(step: int) -> None:
        sweep = _threshold_sweep_on_microset(model=model, loader=loader, device=device, instance_root=instance_root, thresholds=thresholds)
        best = sweep["best"] or {}
        (out_dir / "threshold_sweeps").mkdir(parents=True, exist_ok=True)
        (out_dir / "threshold_sweeps" / f"iter_{step:04d}.json").write_text(json.dumps(sweep, ensure_ascii=False, indent=2), encoding="utf-8")
        if step in vis_iters:
            _export_visuals(
                out_dir=out_dir,
                model=model,
                loader=loader,
                device=device,
                instance_root=instance_root,
                tag=f"iter_{step:04d}",
                best_threshold=float(best.get("threshold") or 0.1),
            )

    _eval_and_log(0)

    for step in range(1, iters + 1):
        model.base.eval()
        model.center_head.train()

        batch = next(iter(loader))
        images = batch["image"].to(device)
        centers = batch["center"].to(device)

        opt.zero_grad(set_to_none=True)
        out = model(images)
        logits = out["center"]
        details = center_loss_fn(logits, centers, return_details=True)
        loss = details["loss"]
        if not bool(torch.isfinite(loss).all().item()):
            raise SystemExit(f"Non-finite loss at iter {step}")
        loss.backward()

        params = list(model.center_head.parameters())
        grad_norm_before = _grad_l2_norm(params)
        clipped = False
        if float(clip_norm) > 0.0:
            clipped = bool(grad_norm_before > float(clip_norm))
            if clipped:
                clipped_n += 1
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(clip_norm))
        grad_norm_after = _grad_l2_norm(params)

        opt.step()

        with torch.no_grad():
            b = float(model.center_head[0].bias.detach().mean().item()) if hasattr(model.center_head[0], "bias") and model.center_head[0].bias is not None else None
            w_norm = float(model.center_head[0].weight.detach().float().norm().item()) if hasattr(model.center_head[0], "weight") else None
            logits_min = float(logits.detach().min().item())
            logits_max = float(logits.detach().max().item())
            params_finite = _params_finite(params)

        if not params_finite:
            raise SystemExit(f"Non-finite parameters at iter {step}")

        if step % eval_every == 0 or step == iters:
            sweep = _threshold_sweep_on_microset(model=model, loader=loader, device=device, instance_root=instance_root, thresholds=thresholds)
            best = sweep["best"] or {}
            (out_dir / "threshold_sweeps").mkdir(parents=True, exist_ok=True)
            (out_dir / "threshold_sweeps" / f"iter_{step:04d}.json").write_text(json.dumps(sweep, ensure_ascii=False, indent=2), encoding="utf-8")
            if step in vis_iters:
                _export_visuals(
                    out_dir=out_dir,
                    model=model,
                    loader=loader,
                    device=device,
                    instance_root=instance_root,
                    tag=f"iter_{step:04d}",
                    best_threshold=float(best.get("threshold") or 0.1),
                )

            with metrics_csv.open("a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        step,
                        float(loss.item()),
                        float(details["pos_loss"].item()),
                        float(details["neg_loss"].item()),
                        float(details["num_pos"].item()),
                        float(best.get("center_prob_mean_pos") or 0.0),
                        float(best.get("center_prob_mean_near") or 0.0),
                        float(best.get("center_prob_mean_far") or 0.0),
                        float(best.get("center_prob_mean_max") or 0.0),
                        float(best.get("threshold") or 0.0),
                        float(best.get("center_f1") or 0.0),
                        float(best.get("center_count_acc") or 0.0),
                        float(best.get("center_loc_err_px") or 0.0),
                        float(best.get("instance_exact_count_acc") or 0.0),
                        float(best.get("instance_mean_matched_iou") or 0.0),
                        float(best.get("instance_perfect_rate") or 0.0),
                        float(grad_norm_before),
                        float(grad_norm_after),
                        int(clipped),
                        float(w_norm) if w_norm is not None else "",
                        float(b) if b is not None else "",
                        float(logits_min),
                        float(logits_max),
                        int(params_finite),
                    ]
                )

            pct = 100.0 * float(clipped_n) / float(max(step, 1))
            print(
                f"iter={step} loss={loss.item():.6f} "
                f"best_thr={float(best.get('threshold') or 0.0):.3f} best_f1={float(best.get('center_f1') or 0.0):.4f} "
                f"clipped_pct={pct:.1f}%"
            )

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "samples": micro.samples,
                "distribution": micro.distribution,
                "iters": iters,
                "eval_every": eval_every,
                "alpha": alpha,
                "beta": beta,
                "init_bias": bias,
                "init_sigmoid": _safe_sigmoid(bias),
                "grad_clip_norm": clip_norm,
                "percent_iterations_clipped": float(100.0 * float(clipped_n) / float(max(iters, 1))),
                "metrics_csv": str(metrics_csv),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

