from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from diagnose_centerhead_run import _build_model_from_semantic_init, _build_val_loader, _load_checkpoint_state, _read_yaml
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


def _to_float(x):
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v


def _to_int(x):
    if x is None or x == "":
        return None
    try:
        return int(x)
    except Exception:
        return None


def _read_metrics_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        row = dict(r)
        row["epoch"] = _to_int(r.get("epoch"))
        for k in list(r.keys()):
            if k == "epoch":
                continue
            if any(
                token in k
                for token in [
                    "loss",
                    "dice",
                    "f1",
                    "precision",
                    "recall",
                    "frac",
                    "count",
                    "loc_err",
                    "score",
                    "iou",
                    "rate",
                    "prob",
                    "grad_norm",
                    "weight_norm",
                    "bias",
                    "lr_",
                    "clipped_pct",
                ]
            ):
                row[k] = _to_float(r.get(k))
        out.append(row)
    return out


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_threshold_sweeps(threshold_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(threshold_dir.glob("*.json")):
        obj = _read_json(p)
        tag = p.stem
        if tag.startswith("epoch"):
            epoch = _to_int(tag.replace("epoch", ""))
        elif tag == "best_center_f1":
            epoch = None
        elif tag == "best_instance_score":
            epoch = None
        else:
            epoch = None
        for row in obj.get("rows") or []:
            out = dict(row)
            out["source_file"] = str(p)
            out["tag"] = tag
            out["epoch"] = epoch
            rows.append(out)
    return rows


def _argmax_row(rows: list[dict], key: str) -> dict | None:
    best = None
    for r in rows:
        v = _to_float(r.get(key))
        if v is None:
            continue
        if best is None or float(v) > float(_to_float(best.get(key)) or 0.0):
            best = r
    return dict(best) if best is not None else None


def _index_metrics_by_epoch(rows: list[dict]) -> dict[int, dict]:
    out = {}
    for r in rows:
        ep = _to_int(r.get("epoch"))
        if ep is not None:
            out[int(ep)] = r
    return out


def _best_by_epoch(rows: list[dict], metrics_by_epoch: dict[int, dict]) -> list[dict]:
    by_epoch: dict[int, list[dict]] = {}
    for r in rows:
        ep = _to_int(r.get("epoch"))
        if ep is None:
            continue
        by_epoch.setdefault(int(ep), []).append(r)
    out = []
    for epoch in sorted(by_epoch):
        rr = by_epoch[epoch]
        center_best = _argmax_row(rr, "center_f1")
        count_best = _argmax_row(rr, "center_count_acc")
        inst_best = _argmax_row(rr, "instance_score")
        miou_best = _argmax_row(rr, "instance_mean_matched_iou")
        m = metrics_by_epoch.get(epoch, {})
        out.append(
            {
                "epoch": epoch,
                "best_center_threshold": _to_float((center_best or {}).get("threshold")),
                "best_center_f1": _to_float((center_best or {}).get("center_f1")),
                "best_center_precision": _to_float((center_best or {}).get("center_precision")),
                "best_center_recall": _to_float((center_best or {}).get("center_recall")),
                "best_count_threshold": _to_float((count_best or {}).get("threshold")),
                "best_center_count_acc": _to_float((count_best or {}).get("center_count_acc")),
                "best_instance_threshold": _to_float((inst_best or {}).get("threshold")),
                "best_instance_score": _to_float((inst_best or {}).get("instance_score")),
                "best_instance_mean_matched_iou": _to_float((inst_best or {}).get("instance_mean_matched_iou")),
                "best_miou_threshold": _to_float((miou_best or {}).get("threshold")),
                "best_miou": _to_float((miou_best or {}).get("instance_mean_matched_iou")),
                "center_prob_mean_pos": _to_float(m.get("center_prob_mean_pos")),
                "center_prob_mean_near": _to_float(m.get("center_prob_mean_near")),
                "center_prob_mean_far": _to_float(m.get("center_prob_mean_far")),
                "center_prob_mean_max": _to_float(m.get("center_prob_mean_max")),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _safe_mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _checkpoint_audit(run_dir: Path) -> dict:
    out = {}
    for name in ["best_center_f1", "best_center_count_acc", "best_instance_score", "last"]:
        p = run_dir / f"{name}.pth"
        if not p.exists():
            out[name] = {"exists": False}
            continue
        ckpt = torch.load(str(p), map_location="cpu")
        extra = ckpt.get("extra") if isinstance(ckpt, dict) else {}
        best_thr = (extra or {}).get("best_threshold_metrics") or {}
        val = (extra or {}).get("val") or {}
        out[name] = {
            "exists": True,
            "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            "saved_threshold": _to_float(best_thr.get("threshold")),
            "center_f1": _to_float(best_thr.get("center_f1", val.get("center_f1"))),
            "center_count_acc": _to_float(best_thr.get("center_count_acc", val.get("center_count_acc"))),
            "instance_score": _to_float(best_thr.get("instance_score")),
            "instance_mean_matched_iou": _to_float(best_thr.get("instance_mean_matched_iou", val.get("instance_mean_matched_iou"))),
            "selection_uses_sweep_metrics": bool((extra or {}).get("threshold_sweep") is not None or (extra or {}).get("best_threshold_metrics") is not None),
        }
    return out


def _find_log_files(run_dir: Path) -> list[Path]:
    out = []
    for pattern in ["*.log", "*.txt", "stdout*", "nohup.out"]:
        out.extend(run_dir.glob(pattern))
    uniq = []
    seen = set()
    for p in out:
        rp = str(p.resolve())
        if rp not in seen and p.is_file():
            uniq.append(p)
            seen.add(rp)
    return uniq


def _optimization_audit(metrics_rows: list[dict], run_dir: Path) -> dict:
    grad_mean_vals = [_to_float(r.get("train_grad_norm_mean_before_clip")) for r in metrics_rows if _to_int(r.get("epoch")) not in [None, 0]]
    grad_max_vals = [_to_float(r.get("train_grad_norm_max_before_clip")) for r in metrics_rows if _to_int(r.get("epoch")) not in [None, 0]]
    clipped_vals = [_to_float(r.get("train_batches_clipped_pct")) for r in metrics_rows if _to_int(r.get("epoch")) not in [None, 0]]
    train_loss_vals = [_to_float(r.get("train_loss")) for r in metrics_rows if _to_int(r.get("epoch")) not in [None, 0]]
    val_center_loss_vals = [_to_float(r.get("val_center_loss")) for r in metrics_rows if _to_int(r.get("epoch")) is not None]
    val_sem_loss_vals = [_to_float(r.get("val_semantic_loss")) for r in metrics_rows if _to_int(r.get("epoch")) is not None]

    parameters_finite = True
    missing_fields = []
    nonfinite_grad_batch_count = None
    skipped_optimizer_step_count = None
    amp_scale_behavior = "not recoverable from saved run"
    log_hits = []
    for p in _find_log_files(run_dir):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "grad_mean=nan" in text or "grad_max=inf" in text:
            log_hits.append(str(p))
        if "GradScaler" in text or "scale" in text:
            amp_scale_behavior = f"log file present: {p.name}"

    if not any("train_grad_norm_mean_before_clip" in r for r in metrics_rows):
        missing_fields.append("train_grad_norm_mean_before_clip")
    if not any("train_batches_clipped_pct" in r for r in metrics_rows):
        missing_fields.append("train_batches_clipped_pct")

    if grad_mean_vals:
        finite_grad_norm_mean = _safe_mean([v for v in grad_mean_vals if v is not None and math.isfinite(float(v))])
    else:
        finite_grad_norm_mean = None

    return {
        "finite_grad_norm_mean": finite_grad_norm_mean,
        "nonfinite_grad_batch_count": nonfinite_grad_batch_count,
        "skipped_optimizer_step_count": skipped_optimizer_step_count,
        "clipped_batches_pct_mean": _safe_mean(clipped_vals),
        "clipped_batches_pct_max": max([float(v) for v in clipped_vals if v is not None], default=None),
        "grad_norm_max_before_clip_max": max([float(v) for v in grad_max_vals if v is not None], default=None),
        "parameters_finite": parameters_finite,
        "train_loss_all_finite": all(v is not None and math.isfinite(float(v)) for v in train_loss_vals),
        "val_center_loss_all_finite": all(v is not None and math.isfinite(float(v)) for v in val_center_loss_vals),
        "val_semantic_loss_all_finite": all(v is not None and math.isfinite(float(v)) for v in val_sem_loss_vals),
        "amp_behavior": amp_scale_behavior,
        "stdout_nonfinite_grad_hits": log_hits,
        "unrecoverable_from_saved_run": [
            "per-batch nonfinite grad counts",
            "GradScaler skipped step counts",
            "AMP scale history",
        ],
        "missing_logged_fields": missing_fields,
    }


def _read_u8(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(str(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def _colorize_instances(inst_u8: np.ndarray) -> np.ndarray:
    out = np.zeros((inst_u8.shape[0], inst_u8.shape[1], 3), dtype=np.uint8)
    colors = {1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255)}
    for k, c in colors.items():
        out[inst_u8 == k] = np.array(c, dtype=np.uint8)
    return out


def _reconstruct_instances(pred_sem_u8: np.ndarray, center_prob_f32: np.ndarray, center_thr: float) -> tuple[np.ndarray, int, list[tuple[int, int, float]]]:
    leaf_union = pred_sem_u8 == 1
    pred_pts = _markers_from_center_map(center_prob_f32, leaf_union, float(center_thr), max_markers=3)
    labels_cc, cc_k = _connected_components(leaf_union.astype(np.uint8))
    pred_inst = np.zeros_like(pred_sem_u8, dtype=np.uint8)
    next_lab = 1
    for comp_id in range(1, int(cc_k) + 1):
        comp01 = labels_cc == comp_id
        in_markers = [(y, x) for (y, x, _) in pred_pts if bool(comp01[int(y), int(x)])]
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


def _save_compare(out_path: Path, original_rgb_u8: np.ndarray, gt_center_u16: np.ndarray, pred_center_u16: np.ndarray, gt_inst_u8: np.ndarray, pred_inst_u8: np.ndarray, binary_u8: np.ndarray) -> None:
    def _heat_u16(x: np.ndarray) -> np.ndarray:
        x8 = (np.clip(x.astype(np.float32) / 65535.0, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return cv2.applyColorMap(x8, cv2.COLORMAP_VIRIDIS)

    a = cv2.cvtColor(original_rgb_u8, cv2.COLOR_RGB2BGR)
    b = _heat_u16(gt_center_u16)
    c = _heat_u16(pred_center_u16)
    d = cv2.cvtColor(binary_u8, cv2.COLOR_GRAY2BGR)
    e = cv2.cvtColor(_colorize_instances(gt_inst_u8), cv2.COLOR_RGB2BGR)
    f = cv2.cvtColor(_colorize_instances(pred_inst_u8), cv2.COLOR_RGB2BGR)
    top = np.concatenate([a, b, c], axis=1)
    bot = np.concatenate([d, e, f], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)


def _export_visual_review(cfg: dict, run_dir: Path, out_dir: Path, selection: dict) -> dict:
    device = torch.device("cpu")
    loader = _build_val_loader(cfg, device=device, batch_size=1, num_workers=0)
    instance_root = Path(cfg["dataset"]["instance_root"]).resolve()
    selected_tags = {
        "global_best_center": selection["global_best_center"],
        "global_best_instance": selection["global_best_instance"],
    }
    summary = {}
    for tag_name, meta in selected_tags.items():
        ckpt_name = meta.get("checkpoint_tag")
        ckpt_path = run_dir / f"{ckpt_name}.pth"
        if not ckpt_path.exists():
            summary[tag_name] = {"checkpoint_exists": False}
            continue
        threshold = float(meta["threshold"])
        model = _build_model_from_semantic_init(cfg).to(device)
        state, epoch = _load_checkpoint_state(ckpt_path)
        incompat = model.load_state_dict(state, strict=False)
        missing = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
        unexpected = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
        if missing or unexpected:
            raise RuntimeError(f"{tag_name}: checkpoint mismatch missing={len(missing)} unexpected={len(unexpected)}")
        model.eval()

        buckets = {"correct": [], "zero_centers": [], "extra_centers": [], "merged": [], "fragmented": []}
        for batch in loader:
            with torch.no_grad():
                out = model(batch["image"].to(device))
            sid = Path(str(batch["image_path"][0])).stem
            pred_sem = torch.argmax(out["semantic"], dim=1).detach().cpu().numpy()[0].astype(np.uint8)
            ctr_prob = torch.sigmoid(out["center"]).detach().cpu().numpy()[0, 0].astype(np.float32)
            pred_inst, pred_k, pred_pts = _reconstruct_instances(pred_sem, ctr_prob, threshold)

            meta_path = str(batch.get("metadata_path", [""])[0])
            gt_pts = _extract_metadata_centers(meta_path) if meta_path else []
            gt_k = len(gt_pts)
            case = _case_type(gt_k, pred_k)
            if len(pred_pts) == 0:
                center_bucket = "zero_centers"
            elif len(pred_pts) > len(gt_pts):
                center_bucket = "extra_centers"
            elif len(pred_pts) == len(gt_pts) and case == "correct":
                center_bucket = "correct"
            else:
                center_bucket = None

            if center_bucket in buckets and len(buckets[center_bucket]) < 10:
                buckets[center_bucket].append((sid, batch, ctr_prob, pred_sem, pred_inst, pred_pts, case))
            if case == "merged" and len(buckets["merged"]) < 10:
                buckets["merged"].append((sid, batch, ctr_prob, pred_sem, pred_inst, pred_pts, case))
            if case == "fragmented" and len(buckets["fragmented"]) < 10:
                buckets["fragmented"].append((sid, batch, ctr_prob, pred_sem, pred_inst, pred_pts, case))
            if all(len(v) >= 10 for v in buckets.values()):
                break

        tag_out = out_dir / tag_name
        counts = {}
        for bucket, items in buckets.items():
            counts[bucket] = len(items)
            for sid, batch, ctr_prob, pred_sem, pred_inst, pred_pts, case in items:
                sample_out = tag_out / bucket / sid
                sample_out.mkdir(parents=True, exist_ok=True)
                img_f = batch["image"].detach().cpu().numpy()[0].transpose(1, 2, 0)
                img_u8 = (np.clip(img_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                gt_center_f = batch["center"].detach().cpu().numpy()[0, 0].astype(np.float32)
                gt_center_u16 = (np.clip(gt_center_f, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
                pred_center_u16 = (np.clip(ctr_prob, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
                binary_u8 = ((ctr_prob >= threshold).astype(np.uint8) * 255)
                gt_inst_path = (instance_root / "instance_masks" / f"{sid}.png").resolve()
                gt_inst = _read_u8(gt_inst_path)
                if gt_inst.shape != pred_sem.shape:
                    gh, gw = gt_inst.shape[:2]
                    h, w = pred_sem.shape[:2]
                    y0 = (gh - h) // 2
                    x0 = (gw - w) // 2
                    gt_inst = gt_inst[y0 : y0 + h, x0 : x0 + w]
                cv2.imwrite(str(sample_out / "original.png"), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(sample_out / "gt_center.png"), gt_center_u16)
                cv2.imwrite(str(sample_out / "pred_center_prob.png"), pred_center_u16)
                cv2.imwrite(str(sample_out / "thresholded_map.png"), binary_u8)
                cv2.imwrite(str(sample_out / "gt_instances.png"), gt_inst.astype(np.uint8))
                cv2.imwrite(str(sample_out / "reconstructed_instances.png"), pred_inst.astype(np.uint8))
                markers_vis = cv2.cvtColor(img_u8.copy(), cv2.COLOR_RGB2BGR)
                for i, (y, x, score) in enumerate(pred_pts, start=1):
                    cv2.circle(markers_vis, (int(x), int(y)), 6, (255, 0, 0), 2)
                    cv2.putText(markers_vis, f"{i}:{score:.2f}", (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.imwrite(str(sample_out / "markers.png"), markers_vis)
                _save_compare(sample_out / "compare.png", img_u8, gt_center_u16, pred_center_u16, gt_inst, pred_inst, binary_u8)
                iou_mat = _iou_matrix(gt_inst, pred_inst, len(gt_pts), pred_k)
                mean_iou = float(_best_perm_sum(iou_mat) / max(len(gt_pts), 1))
                (sample_out / "metrics.json").write_text(
                    json.dumps(
                        {
                            "sample": sid,
                            "threshold": threshold,
                            "checkpoint_epoch": epoch,
                            "gt_center_count": len(gt_pts),
                            "pred_center_count": len(pred_pts),
                            "pred_instance_count": pred_k,
                            "case": case,
                            "mean_matched_iou": mean_iou,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        summary[tag_name] = {"checkpoint_exists": True, "checkpoint": ckpt_name, "threshold": threshold, "counts": counts}
    return summary


def _resolve_checkpoint_tag(checkpoint_audit: dict, epoch: int | None, metric_name: str) -> str | None:
    candidates = []
    for tag, meta in checkpoint_audit.items():
        if not meta.get("exists"):
            continue
        if _to_int(meta.get("epoch")) == _to_int(epoch):
            candidates.append(tag)
    if metric_name == "center_f1" and "best_center_f1" in candidates:
        return "best_center_f1"
    if metric_name == "instance_score" and "best_instance_score" in candidates:
        return "best_instance_score"
    if metric_name == "center_count_acc" and "best_center_count_acc" in candidates:
        return "best_center_count_acc"
    return candidates[0] if candidates else None


def analyze_run(config_path: Path, run_dir: Path, out_dir: Path) -> dict:
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    metrics_csv = run_dir / "metrics.csv"
    threshold_dir = run_dir / "threshold_sweeps"
    if not metrics_csv.exists():
        raise SystemExit(f"metrics.csv not found: {metrics_csv}")
    if not threshold_dir.exists():
        raise SystemExit(f"threshold_sweeps dir not found: {threshold_dir}")

    cfg = _read_yaml(config_path)
    metrics_rows = _read_metrics_csv(metrics_csv)
    metrics_by_epoch = _index_metrics_by_epoch(metrics_rows)
    sweep_rows = _flatten_threshold_sweeps(threshold_dir)
    checkpoint_audit = _checkpoint_audit(run_dir)

    for r in sweep_rows:
        r["checkpoint_tag"] = _resolve_checkpoint_tag(checkpoint_audit, _to_int(r.get("epoch")), "center_f1")

    best_center = _argmax_row(sweep_rows, "center_f1") or {}
    best_count = _argmax_row(sweep_rows, "center_count_acc") or {}
    best_instance = _argmax_row(sweep_rows, "instance_score") or {}
    best_miou = _argmax_row(sweep_rows, "instance_mean_matched_iou") or {}
    best_center["checkpoint_tag"] = _resolve_checkpoint_tag(checkpoint_audit, _to_int(best_center.get("epoch")), "center_f1")
    best_instance["checkpoint_tag"] = _resolve_checkpoint_tag(checkpoint_audit, _to_int(best_instance.get("epoch")), "instance_score")

    best_by_epoch_rows = _best_by_epoch(sweep_rows, metrics_by_epoch)

    epoch_candidates = []
    for ep in [0, 1, 2, 3, 5, 7, 10, 15, 20]:
        if ep in metrics_by_epoch:
            epoch_candidates.append(ep)
    for extra_ep in [_to_int(best_center.get("epoch")), _to_int(best_instance.get("epoch"))]:
        if extra_ep is not None and extra_ep not in epoch_candidates and extra_ep in metrics_by_epoch:
            epoch_candidates.append(extra_ep)
    epoch_candidates = sorted(epoch_candidates)

    prob_rows = []
    for ep in epoch_candidates:
        m = metrics_by_epoch[ep]
        pos = _to_float(m.get("center_prob_mean_pos"))
        near = _to_float(m.get("center_prob_mean_near"))
        far = _to_float(m.get("center_prob_mean_far"))
        prob_rows.append(
            {
                "epoch": ep,
                "center_prob_mean_pos": pos,
                "center_prob_mean_near": near,
                "center_prob_mean_far": far,
                "center_prob_mean_max": _to_float(m.get("center_prob_mean_max")),
                "pos_minus_far": (float(pos) - float(far)) if pos is not None and far is not None else None,
                "near_minus_far": (float(near) - float(far)) if near is not None and far is not None else None,
            }
        )

    optimization = _optimization_audit(metrics_rows, run_dir)

    global_best_metrics = {
        "best_center_f1": best_center,
        "best_center_count_acc": best_count,
        "best_instance_score": best_instance,
        "best_instance_mean_matched_iou": best_miou,
        "probability_separation": prob_rows,
        "optimization_audit": optimization,
        "checkpoint_audit": checkpoint_audit,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "all_epoch_threshold_metrics.csv", sweep_rows)
    _write_csv(out_dir / "best_by_epoch.csv", best_by_epoch_rows)
    (out_dir / "global_best_metrics.json").write_text(json.dumps(global_best_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    visual_summary = _export_visual_review(
        cfg,
        run_dir,
        out_dir / "visual_review",
        {
            "global_best_center": best_center,
            "global_best_instance": best_instance,
        },
    )

    analysis_summary = {
        "run_dir": str(run_dir),
        "metrics_csv": str(metrics_csv),
        "threshold_dir": str(threshold_dir),
        "global_best_center": best_center,
        "global_best_center_count_acc": best_count,
        "global_best_instance": best_instance,
        "global_best_instance_mean_matched_iou": best_miou,
        "probability_separation": prob_rows,
        "optimization_audit": optimization,
        "checkpoint_audit": checkpoint_audit,
        "visual_review": visual_summary,
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(analysis_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return analysis_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("training/configs/unetpp_effb3_centerhead_spatial_frozenbase_focal_pilot_20ep.yaml"))
    ap.add_argument("--run-dir", type=Path, default=Path("training/runs/unetpp_effb3_centerhead_spatial_frozenbase_focal_pilot_20ep"))
    ap.add_argument("--out-dir", type=Path, default=Path("training/analysis/spatial_frozenbase_pilot_analysis"))
    args = ap.parse_args()

    summary = analyze_run(config_path=args.config.resolve(), run_dir=args.run_dir.resolve(), out_dir=args.out_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
