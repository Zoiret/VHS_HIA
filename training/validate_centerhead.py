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


def _patient_id_from_sample(sample_id: str) -> str:
    if "_s" not in str(sample_id):
        return str(sample_id)
    return str(sample_id).rsplit("_s", 1)[0]


def _positive_label_ids(labels: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(labels) if int(v) > 0]


def _sample_center_metrics(pred_pts: list[tuple[int, int]], gt_pts: list[tuple[int, int]]) -> dict:
    tp, fp, fn, matches = _match_centers(pred_pts, gt_pts, max_dist_px=16.0)
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-7))
    loc_err = float(sum(float(m[4]) for m in matches) / max(len(matches), 1))
    return {
        "center_precision": precision,
        "center_recall": recall,
        "center_f1": f1,
        "center_count_accuracy": float(int(len(pred_pts) == len(gt_pts))),
        "center_loc_err_px": loc_err,
        "predicted_center_count": int(len(pred_pts)),
        "gt_center_count": int(len(gt_pts)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def _marker_contract(gt_inst: np.ndarray, pred_pts: list[tuple[int, int]]) -> dict:
    gt_instance_ids = _positive_label_ids(gt_inst)
    counts = {int(inst_id): 0 for inst_id in gt_instance_ids}
    outside = 0
    for y, x in pred_pts:
        inst_id = int(gt_inst[int(y), int(x)]) if 0 <= int(y) < gt_inst.shape[0] and 0 <= int(x) < gt_inst.shape[1] else 0
        if int(inst_id) == 0:
            outside += 1
        elif int(inst_id) in counts:
            counts[int(inst_id)] += 1
    zero = [int(inst_id) for inst_id, c in counts.items() if int(c) == 0]
    multi = [int(inst_id) for inst_id, c in counts.items() if int(c) > 1]
    one = [int(inst_id) for inst_id, c in counts.items() if int(c) == 1]
    gt_total = int(len(gt_instance_ids))
    return {
        "extracted_marker_count": int(len(pred_pts)),
        "markers_outside_all_gt_instances": int(outside),
        "missing_gt_instance_markers": int(len(zero)),
        "multiple_markers_inside_gt_instances": int(len(multi)),
        "gt_instances_total": gt_total,
        "gt_instances_with_exactly_one_marker_count": int(len(one)),
        "one_marker_per_instance_rate": float(len(one) / max(gt_total, 1)),
        "marker_contract_pass": bool(int(len(pred_pts)) == gt_total and len(zero) == 0 and len(multi) == 0 and int(outside) == 0),
    }


def _aggregate_center_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "sample_count": 0,
            "center_precision": None,
            "center_recall": None,
            "center_f1": None,
            "center_precision_mean_samples": None,
            "center_recall_mean_samples": None,
            "center_f1_mean_samples": None,
            "exact_center_count_accuracy": None,
            "strict_marker_contract_pass_count": 0,
            "strict_marker_contract_pass_rate": None,
            "localization_error_px": None,
        }
    tp = int(sum(int(row["tp"]) for row in rows))
    fp = int(sum(int(row["fp"]) for row in rows))
    fn = int(sum(int(row["fn"]) for row in rows))
    return {
        "sample_count": int(len(rows)),
        "center_precision": float(tp / max(tp + fp, 1)),
        "center_recall": float(tp / max(tp + fn, 1)),
        "center_f1": float((2.0 * tp) / max(2 * tp + fp + fn, 1)),
        "center_precision_mean_samples": float(np.mean([float(row["center_precision"]) for row in rows])),
        "center_recall_mean_samples": float(np.mean([float(row["center_recall"]) for row in rows])),
        "center_f1_mean_samples": float(np.mean([float(row["center_f1"]) for row in rows])),
        "exact_center_count_accuracy": float(np.mean([float(row["center_count_acc"]) for row in rows])),
        "strict_marker_contract_pass_count": int(sum(1 for row in rows if bool(row["marker_contract_pass"]))),
        "strict_marker_contract_pass_rate": float(np.mean([1.0 if bool(row["marker_contract_pass"]) else 0.0 for row in rows])),
        "localization_error_px": float(np.mean([float(row["center_loc_err_px"]) for row in rows])),
    }


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


def reconstruct_instances_from_semantic_and_center(
    pred_sem_u8: np.ndarray,
    center_prob_f32: np.ndarray,
    center_thr: float,
    *,
    max_markers: int = 3,
    return_trace: bool = False,
):
    leaf_union = pred_sem_u8 == 1
    pred_pts_scored = _markers_from_center_map(center_prob_f32, leaf_union, float(center_thr), max_markers=max_markers)
    pred_pts = [(int(y), int(x)) for (y, x, _) in pred_pts_scored]

    marker_labels = np.zeros_like(pred_sem_u8, dtype=np.uint8)
    for idx, (y, x, _score) in enumerate(pred_pts_scored, start=1):
        marker_labels[int(y), int(x)] = np.uint8(idx)

    labels_cc, cc_k = _connected_components(leaf_union.astype(np.uint8))
    pred_inst = np.zeros_like(pred_sem_u8, dtype=np.uint8)
    next_lab = 1
    component_traces = []

    for comp_id in range(1, int(cc_k) + 1):
        comp01 = labels_cc == comp_id
        in_markers_initial = [(y, x) for (y, x) in pred_pts if bool(comp01[int(y), int(x)])]
        in_markers = list(in_markers_initial)
        fallback_marker = None
        used_fallback = False
        watershed_local_count_before_keep = 0
        watershed_local_count_after_keep = 0
        output_labels = []
        path = "single_label"

        if len(in_markers) == 0:
            fb = _fallback_marker(comp01)
            if fb is not None:
                fallback_marker = (int(fb[0]), int(fb[1]))
                in_markers = [fallback_marker]
                used_fallback = True

        if len(in_markers) <= 1:
            pred_inst[comp01] = np.uint8(next_lab)
            output_labels = [int(next_lab)]
            next_lab += 1
        else:
            path = "watershed"
            topo = _geometry_topo_u8(comp01.astype(np.uint8))
            seg = _watershed(comp01.astype(np.uint8), in_markers, topo)
            watershed_local_count_before_keep = int(seg.max())
            seg, seg_k = _keep_top3_by_area(seg)
            watershed_local_count_after_keep = int(seg_k)
            if seg_k <= 1:
                path = "watershed_collapsed"
                pred_inst[comp01] = np.uint8(next_lab)
                output_labels = [int(next_lab)]
                next_lab += 1
            else:
                for local in range(1, int(seg_k) + 1):
                    pred_inst[seg == local] = np.uint8(next_lab)
                    output_labels.append(int(next_lab))
                    next_lab += 1

        component_traces.append(
            {
                "component_id": int(comp_id),
                "area": int(np.sum(comp01)),
                "marker_count_before_fallback": int(len(in_markers_initial)),
                "marker_count_after_fallback": int(len(in_markers)),
                "markers_before_fallback": [{"y": int(y), "x": int(x)} for (y, x) in in_markers_initial],
                "markers_used": [{"y": int(y), "x": int(x)} for (y, x) in in_markers],
                "used_fallback": bool(used_fallback),
                "fallback_marker": (
                    {"y": int(fallback_marker[0]), "x": int(fallback_marker[1])}
                    if fallback_marker is not None
                    else None
                ),
                "path": path,
                "watershed_local_count_before_keep": int(watershed_local_count_before_keep),
                "watershed_local_count_after_keep": int(watershed_local_count_after_keep),
                "output_labels": [int(v) for v in output_labels],
            }
        )

    raw_inst = pred_inst.copy()
    raw_k = int(raw_inst.max())
    final_inst, final_k = _keep_top3_by_area(pred_inst)

    if not return_trace:
        return final_inst, int(final_k), pred_pts_scored

    trace = {
        "leaf_union": leaf_union.astype(np.uint8),
        "semantic_components": labels_cc.astype(np.int32),
        "semantic_component_count": int(cc_k),
        "marker_labels": marker_labels,
        "marker_count": int(len(pred_pts_scored)),
        "marker_points": [{"y": int(y), "x": int(x), "score": float(s)} for (y, x, s) in pred_pts_scored],
        "raw_reconstruction_labels": raw_inst.astype(np.uint8),
        "raw_reconstruction_count": int(raw_k),
        "postprocessed_labels": final_inst.astype(np.uint8),
        "postprocessed_count": int(final_k),
        "final_labels": final_inst.astype(np.uint8),
        "final_count": int(final_k),
        "component_traces": component_traces,
    }
    return final_inst, int(final_k), pred_pts_scored, trace


def compute_instance_metrics_from_masks(
    gt_inst_u8: np.ndarray,
    pred_inst_u8: np.ndarray,
    *,
    gt_k: int | None = None,
    pred_k: int | None = None,
) -> dict:
    if gt_k is None:
        gt_k = int(len([int(v) for v in np.unique(gt_inst_u8) if int(v) > 0]))
    if pred_k is None:
        pred_k = int(len([int(v) for v in np.unique(pred_inst_u8) if int(v) > 0]))
    case = _case_type(int(gt_k), int(pred_k))
    iou_mat = _iou_matrix(gt_inst_u8, pred_inst_u8, int(gt_k), int(pred_k))
    sum_iou = _best_perm_sum(iou_mat)
    mean_iou = float(sum_iou / max(int(gt_k), 1))
    return {
        "gt_instance_count": int(gt_k),
        "pred_instance_count": int(pred_k),
        "case": str(case),
        "instance_exact_count": bool(int(pred_k) == int(gt_k)),
        "instance_exact_count_acc": float(int(int(pred_k) == int(gt_k))),
        "instance_mean_matched_iou": float(mean_iou),
        "instance_merged": bool(case == "merged"),
        "instance_fragmented": bool(case == "fragmented"),
        "instance_mixed": bool(case == "mixed"),
        "instance_merged_rate": float(int(case == "merged")),
        "instance_fragmented_rate": float(int(case == "fragmented")),
        "instance_mixed_rate": float(int(case == "mixed")),
        "instance_perfect": bool((int(pred_k) == int(gt_k)) and (mean_iou >= 0.90)),
        "instance_perfect_rate": float(int((int(pred_k) == int(gt_k)) and (mean_iou >= 0.90))),
        "iou_matrix": iou_mat,
    }


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
    prob_pos_sum = 0.0
    prob_pos_n = 0
    prob_near_sum = 0.0
    prob_near_n = 0
    prob_far_sum = 0.0
    prob_far_n = 0
    prob_max_sum = 0.0
    prob_max_n = 0
    center_rows: list[dict] = []

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
            center_metrics = _sample_center_metrics(pred_pts, gt_pts)

            tpi = int(center_metrics["tp"])
            fpi = int(center_metrics["fp"])
            fni = int(center_metrics["fn"])
            _tp2, _fp2, _fn2, matches = _match_centers(pred_pts, gt_pts, max_dist_px=16.0)
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

            gt_center_map = gt_centers[i, 0]
            pr_center_map = pred_center[i, 0]
            pos_exact = gt_center_map >= 0.9999
            near = gt_center_map >= 0.1
            far = gt_center_map < 0.1
            if bool(np.any(pos_exact)):
                prob_pos_sum += float(np.mean(pr_center_map[pos_exact]))
                prob_pos_n += 1
            if bool(np.any(near)):
                prob_near_sum += float(np.mean(pr_center_map[near]))
                prob_near_n += 1
            if bool(np.any(far)):
                prob_far_sum += float(np.mean(pr_center_map[far]))
                prob_far_n += 1
            prob_max_sum += float(np.max(pr_center_map))
            prob_max_n += 1

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

            marker_contract = _marker_contract(gt_inst, pred_pts)
            center_rows.append(
                {
                    "sample": str(sid),
                    "patient_id": _patient_id_from_sample(str(sid)),
                    "gt_instance_count": int(len(gt_pts)),
                    "tp": int(tpi),
                    "fp": int(fpi),
                    "fn": int(fni),
                    "center_precision": float(center_metrics["center_precision"]),
                    "center_recall": float(center_metrics["center_recall"]),
                    "center_f1": float(center_metrics["center_f1"]),
                    "center_count_acc": float(center_metrics["center_count_accuracy"]),
                    "center_loc_err_px": float(center_metrics["center_loc_err_px"]),
                    "predicted_center_count": int(center_metrics["predicted_center_count"]),
                    "marker_contract_pass": bool(marker_contract["marker_contract_pass"]),
                    "missing_gt_instance_markers": int(marker_contract["missing_gt_instance_markers"]),
                    "multiple_markers_inside_gt_instances": int(marker_contract["multiple_markers_inside_gt_instances"]),
                    "markers_outside_all_gt_instances": int(marker_contract["markers_outside_all_gt_instances"]),
                }
            )

            pred_inst, pred_k, _pred_pts_scored = reconstruct_instances_from_semantic_and_center(
                pred_sem[i],
                pred_center[i, 0],
                float(center_thr),
                max_markers=3,
                return_trace=False,
            )

            inst_metrics = compute_instance_metrics_from_masks(gt_inst, pred_inst, gt_k=gt_k, pred_k=pred_k)
            case = str(inst_metrics["case"])
            inst_n += 1
            inst_exact += int(bool(inst_metrics["instance_exact_count"]))
            inst_merged += int(case == "merged")
            inst_fragmented += int(case == "fragmented")
            inst_mixed += int(case == "mixed")
            mean_iou = float(inst_metrics["instance_mean_matched_iou"])
            inst_mean_iou_sum += float(mean_iou)
            inst_median_iou_list.append(float(mean_iou))
            inst_perfect += int(bool(inst_metrics["instance_perfect"]))

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
    prob_pos_mean = float(prob_pos_sum / max(prob_pos_n, 1))
    prob_near_mean = float(prob_near_sum / max(prob_near_n, 1))
    prob_far_mean = float(prob_far_sum / max(prob_far_n, 1))
    prob_max_mean = float(prob_max_sum / max(prob_max_n, 1))
    agg_all = _aggregate_center_rows(center_rows)
    per_gt_count = {
        str(gt_count): _aggregate_center_rows([row for row in center_rows if int(row["gt_instance_count"]) == int(gt_count)])
        for gt_count in (1, 2, 3)
    }
    per_patient_ids = sorted({str(row["patient_id"]) for row in center_rows})
    per_patient = {
        patient_id: _aggregate_center_rows([row for row in center_rows if str(row["patient_id"]) == str(patient_id)])
        for patient_id in per_patient_ids
    }

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
        "center_prob_mean_pos": prob_pos_mean,
        "center_prob_mean_near": prob_near_mean,
        "center_prob_mean_far": prob_far_mean,
        "center_prob_mean_max": prob_max_mean,
        "center_precision_mean_samples": agg_all["center_precision_mean_samples"],
        "center_recall_mean_samples": agg_all["center_recall_mean_samples"],
        "center_f1_mean_samples": agg_all["center_f1_mean_samples"],
        "strict_marker_contract_pass_count": agg_all["strict_marker_contract_pass_count"],
        "strict_marker_contract_pass_rate": agg_all["strict_marker_contract_pass_rate"],
        "per_gt_count_center_metrics": per_gt_count,
        "per_patient_center_metrics": per_patient,
        "per_sample_center_rows": center_rows,
    }
