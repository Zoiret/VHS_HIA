from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from metrics import compute_per_class_metrics_from_logits


def _dice_iou_binary(gt01, pred01) -> tuple[float, float]:
    gt = gt01.astype(bool)
    pr = pred01.astype(bool)
    inter = int(np.logical_and(gt, pr).sum())
    gt_sum = int(gt.sum())
    pr_sum = int(pr.sum())
    denom_dice = gt_sum + pr_sum
    if denom_dice == 0:
        return 1.0, 1.0
    dice = float(2.0 * inter / denom_dice)
    union = gt_sum + pr_sum - inter
    iou = float(inter / union) if union > 0 else 1.0
    return dice, iou


def _extract_metadata_centers(meta_path: str) -> list[tuple[int, int]]:
    obj = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    centers = []
    for inst in obj.get("instances") or []:
        yx = inst.get("center_yx")
        if isinstance(yx, list) and len(yx) == 2 and isinstance(yx[0], int) and isinstance(yx[1], int):
            centers.append((int(yx[0]), int(yx[1])))
    return centers[:3]


def _markers_from_center_map(center_prob: np.ndarray, leaf_union: np.ndarray, thr: float, max_markers: int = 3) -> list[tuple[int, int, float]]:
    c = center_prob.astype(np.float32).copy()
    c[~leaf_union.astype(bool)] = 0.0
    m = (c >= float(thr)).astype(np.uint8)
    if int(m.sum()) == 0:
        return []
    n, labels = cv2.connectedComponents(m)
    pts: list[tuple[int, int, float]] = []
    for lab in range(1, int(n)):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        vals = c[ys, xs]
        k = int(np.argmax(vals))
        y = int(ys[k])
        x = int(xs[k])
        pts.append((y, x, float(c[y, x])))
    pts.sort(key=lambda t: t[2], reverse=True)
    return pts[: int(max_markers)]


def _match_centers(pred_yx: list[tuple[int, int]], gt_yx: list[tuple[int, int]], max_dist_px: float = 16.0):
    used_gt = set()
    matches = []
    for py, px in pred_yx:
        best = None
        best_d = None
        for gi, (gy, gx) in enumerate(gt_yx):
            if gi in used_gt:
                continue
            dy = float(py - gy)
            dx = float(px - gx)
            d = float(np.hypot(dy, dx))
            if best_d is None or d < best_d:
                best_d = d
                best = gi
        if best is not None and best_d is not None and best_d <= float(max_dist_px):
            used_gt.add(best)
            matches.append((py, px, gt_yx[best][0], gt_yx[best][1], float(best_d)))
    tp = int(len(matches))
    fp = int(max(0, len(pred_yx) - tp))
    fn = int(max(0, len(gt_yx) - tp))
    return tp, fp, fn, matches


def _connected_components(mask01: np.ndarray) -> tuple[np.ndarray, int]:
    m = (mask01.astype(np.uint8) > 0).astype(np.uint8) * 255
    n, labels = cv2.connectedComponents(m, connectivity=8)
    return labels.astype(np.int32), max(0, int(n) - 1)


def _geometry_topo_u8(component01: np.ndarray) -> np.ndarray:
    m = component01.astype(np.uint8)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3).astype(np.float32)
    if float(dt.max()) > 0.0:
        dt = dt / float(dt.max())
    topo = (1.0 - dt) * 255.0
    return topo.astype(np.uint8)


def _fallback_marker(component01: np.ndarray) -> tuple[int, int] | None:
    m = component01.astype(np.uint8)
    if int(m.sum()) == 0:
        return None
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3).astype(np.float32)
    if float(dt.max()) <= 0.0:
        ys, xs = np.where(component01)
        if ys.size == 0:
            return None
        return int(ys[0]), int(xs[0])
    y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return int(y), int(x)


def _watershed(component01: np.ndarray, markers_yx: list[tuple[int, int]], topo_u8: np.ndarray) -> np.ndarray:
    h, w = component01.shape[:2]
    mk = np.zeros((h, w), dtype=np.int32)
    mk[component01.astype(bool) == 0] = 1
    for idx, (y, x) in enumerate(markers_yx, start=2):
        if 0 <= y < h and 0 <= x < w and bool(component01[y, x]):
            mk[y, x] = int(idx)
    topo3 = cv2.cvtColor(topo_u8, cv2.COLOR_GRAY2BGR)
    cv2.watershed(topo3, mk)
    out = np.zeros((h, w), dtype=np.uint8)
    labs = sorted([int(v) for v in np.unique(mk) if int(v) > 1])
    for new_i, lab in enumerate(labs, start=1):
        out[(mk == lab) & component01.astype(bool)] = np.uint8(new_i)
    out[component01.astype(bool) == 0] = 0
    return out


def _keep_top3_by_area(labels_u8: np.ndarray) -> tuple[np.ndarray, int]:
    k = int(labels_u8.max())
    if k <= 3:
        return labels_u8, k
    areas = []
    for i in range(1, k + 1):
        areas.append((int(np.sum(labels_u8 == i)), i))
    areas.sort(reverse=True, key=lambda t: t[0])
    keep = [lab for _, lab in areas[:3]]
    out = np.zeros_like(labels_u8, dtype=np.uint8)
    for new_i, old_i in enumerate(keep, start=1):
        out[labels_u8 == old_i] = np.uint8(new_i)
    return out, 3


def _iou_matrix(gt_u8: np.ndarray, pred_u8: np.ndarray, gt_k: int, pred_k: int) -> np.ndarray:
    m = np.zeros((int(gt_k), int(pred_k)), dtype=np.float64)
    if gt_k == 0 or pred_k == 0:
        return m
    for gi in range(1, int(gt_k) + 1):
        g = gt_u8 == gi
        g_sum = float(np.sum(g))
        if g_sum <= 0:
            continue
        for pi in range(1, int(pred_k) + 1):
            p = pred_u8 == pi
            p_sum = float(np.sum(p))
            if p_sum <= 0:
                continue
            inter = float(np.sum(g & p))
            if inter <= 0:
                continue
            union = g_sum + p_sum - inter
            m[gi - 1, pi - 1] = inter / max(union, 1.0)
    return m


def _best_perm_sum(iou: np.ndarray) -> float:
    gt_k, pred_k = iou.shape[0], iou.shape[1]
    if gt_k == 0 or pred_k == 0:
        return 0.0
    k = min(int(gt_k), int(pred_k))
    best = -1.0
    import itertools

    for cols in itertools.permutations(range(int(pred_k)), k):
        s = 0.0
        for r, c in enumerate(cols):
            s += float(iou[r, c])
        if s > best:
            best = s
    return float(best if best >= 0 else 0.0)


def _case_type(gt_k: int, pred_k: int) -> str:
    merged = gt_k >= 2 and pred_k < gt_k
    fragmented = pred_k > gt_k
    if merged and fragmented:
        return "mixed"
    if merged:
        return "merged"
    if fragmented:
        return "fragmented"
    return "correct"


@torch.no_grad()
def validate_centerhead(
    *,
    model: torch.nn.Module,
    loader,
    num_classes: int,
    device: torch.device,
    semantic_loss_fn: torch.nn.Module,
    center_loss_fn: torch.nn.Module,
    instance_root: Path,
    center_thr: float = 0.3,
) -> dict:
    model.eval()
    total_sem_loss = 0.0
    total_center_loss = 0.0
    n_batches = 0

    dice_sum = [0.0 for _ in range(num_classes)]
    iou_sum = [0.0 for _ in range(num_classes)]

    tp = fp = fn = 0
    loc_err_sum = 0.0
    loc_err_n = 0
    count_acc_n = 0
    count_acc_ok = 0

    inst_exact = 0
    inst_n = 0
    inst_merged = 0
    inst_fragmented = 0
    inst_mixed = 0
    inst_mean_iou_sum = 0.0
    inst_median_iou_list = []
    inst_perfect = 0

    for batch in tqdm(loader, desc="Validate(centerhead)", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        centers = batch["center"].to(device, non_blocking=True)

        out = model(images)
        sem_logits = out["semantic"]
        center_logits = out["center"]

        sem_loss = semantic_loss_fn(sem_logits, masks)
        center_loss = center_loss_fn(center_logits, centers)
        total_sem_loss += float(sem_loss.item())
        total_center_loss += float(center_loss.item())
        n_batches += 1

        m = compute_per_class_metrics_from_logits(sem_logits, masks, num_classes=num_classes)
        for i in range(num_classes):
            dice_sum[i] += float(m.dice[i])
            iou_sum[i] += float(m.iou[i])

        pred_sem = torch.argmax(sem_logits, dim=1).detach().cpu().numpy().astype(np.uint8)
        pred_center = torch.sigmoid(center_logits).detach().cpu().numpy().astype(np.float32)
        gt_centers = centers.detach().cpu().numpy().astype(np.float32)
        gt_masks = masks.detach().cpu().numpy().astype(np.uint8)
        image_paths = batch.get("image_path", None)
        meta_paths = batch.get("metadata_path", None)
        if not isinstance(image_paths, list):
            image_paths = [None for _ in range(int(pred_sem.shape[0]))]
        if not isinstance(meta_paths, list):
            meta_paths = [None for _ in range(int(pred_sem.shape[0]))]

        for i in range(int(pred_sem.shape[0])):
            leaf_union = pred_sem[i] == 1
            gt_leaf_union = gt_masks[i] == 1
            pred_pts = [(y, x) for (y, x, _) in _markers_from_center_map(pred_center[i, 0], leaf_union, float(center_thr), max_markers=3)]
            meta_p = meta_paths[i] if i < len(meta_paths) else None
            gt_pts = _extract_metadata_centers(str(meta_p)) if isinstance(meta_p, str) and meta_p else []

            tpi, fpi, fni, matches = _match_centers(pred_pts, gt_pts, max_dist_px=16.0)
            tp += int(tpi)
            fp += int(fpi)
            fn += int(fni)
            for _, _, _, _, d in matches:
                loc_err_sum += float(d)
                loc_err_n += 1
            if len(gt_pts) > 0:
                count_acc_n += 1
                if int(len(pred_pts)) == int(len(gt_pts)):
                    count_acc_ok += 1

            if bool(np.any(gt_leaf_union)):
                pos_frac = float(np.mean((pred_center[i, 0][gt_leaf_union] >= float(center_thr)).astype(np.float32)))
            else:
                pos_frac = float(np.mean((pred_center[i, 0] >= float(center_thr)).astype(np.float32)))
            if "center_pos_frac_sum" not in locals():
                center_pos_frac_sum = 0.0
                center_pos_frac_n = 0
                pred_count_sum = 0.0
                gt_count_sum = 0.0
                zero_center_cases = 0
                extra_center_cases = 0
            center_pos_frac_sum += float(pos_frac)
            center_pos_frac_n += 1
            pred_count_sum += float(len(pred_pts))
            gt_count_sum += float(len(gt_pts))
            if int(len(pred_pts)) == 0:
                zero_center_cases += 1
            if int(len(pred_pts)) > 3:
                extra_center_cases += 1

            sid = Path(str(image_paths[i])).stem if isinstance(image_paths[i], str) else None
            if not sid:
                continue
            gt_inst_path = (instance_root / "instance_masks" / f"{sid}.png").resolve()
            gt_inst_src = cv2.imread(str(gt_inst_path), cv2.IMREAD_UNCHANGED)
            if gt_inst_src is None:
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

            case = _case_type(gt_k, pred_k)
            inst_n += 1
            inst_exact += int(pred_k == gt_k)
            inst_merged += int(case == "merged")
            inst_fragmented += int(case == "fragmented")
            inst_mixed += int(case == "mixed")

            iou_mat = _iou_matrix(gt_inst, pred_inst, gt_k, pred_k)
            sum_iou = _best_perm_sum(iou_mat)
            mean_iou = float(sum_iou / max(gt_k, 1))
            inst_mean_iou_sum += float(mean_iou)
            inst_median_iou_list.append(float(mean_iou))
            inst_perfect += int((pred_k == gt_k) and (mean_iou >= 0.90))

    n_samples = float(max(n_batches, 1))
    dice = [float(x / n_samples) for x in dice_sum]
    iou = [float(x / n_samples) for x in iou_sum]

    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float((2 * precision * recall / max(precision + recall, 1e-7)))
    loc_err = float(loc_err_sum / max(loc_err_n, 1))
    count_acc = float(count_acc_ok / max(count_acc_n, 1))
    center_pos_frac = float(center_pos_frac_sum / max(center_pos_frac_n, 1)) if "center_pos_frac_sum" in locals() else None
    pred_count_mean = float(pred_count_sum / max(center_pos_frac_n, 1)) if "pred_count_sum" in locals() else None
    gt_count_mean = float(gt_count_sum / max(center_pos_frac_n, 1)) if "gt_count_sum" in locals() else None
    zero_centers = int(zero_center_cases) if "zero_center_cases" in locals() else None
    extra_centers = int(extra_center_cases) if "extra_center_cases" in locals() else None

    mean_dice_fg = None
    if int(num_classes) == 3:
        mean_dice_fg = float((dice[1] + dice[2]) / 2.0)
    elif int(num_classes) == 2:
        mean_dice_fg = float(dice[1])

    inst_mean_iou = float(inst_mean_iou_sum / max(inst_n, 1))
    inst_median_iou = float(np.median(np.asarray(inst_median_iou_list, dtype=np.float32))) if inst_median_iou_list else None
    inst_perfect_rate = float(inst_perfect / max(inst_n, 1))
    inst_merged_rate = float(inst_merged / max(inst_n, 1))
    inst_fragmented_rate = float(inst_fragmented / max(inst_n, 1))
    inst_mixed_rate = float(inst_mixed / max(inst_n, 1))
    inst_exact_acc = float(inst_exact / max(inst_n, 1))

    return {
        "semantic_loss": float(total_sem_loss / max(n_batches, 1)),
        "center_loss": float(total_center_loss / max(n_batches, 1)),
        "dice": dice,
        "iou": iou,
        "mean_dice_fg": mean_dice_fg,
        "center_precision": precision,
        "center_recall": recall,
        "center_f1": f1,
        "center_loc_err_px": loc_err,
        "center_count_acc": count_acc,
        "center_pos_frac": center_pos_frac,
        "center_pred_count_mean": pred_count_mean,
        "center_gt_count_mean": gt_count_mean,
        "center_zero_cases": zero_centers,
        "center_extra_cases": extra_centers,
        "instance_exact_count_acc": inst_exact_acc,
        "instance_merged_rate": inst_merged_rate,
        "instance_fragmented_rate": inst_fragmented_rate,
        "instance_mixed_rate": inst_mixed_rate,
        "instance_mean_matched_iou": inst_mean_iou,
        "instance_median_matched_iou": inst_median_iou,
        "instance_perfect_rate": inst_perfect_rate,
    }
