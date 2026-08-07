from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from augmentations import get_val_augmentations
from dataset_centerhead import _read_image_rgb, _read_mask_u8, _read_u16
from train_centerhead import _build_model, _read_yaml, _simple_preprocess_uint8_rgb
from validate_centerhead import (
    _aggregate_center_rows,
    _extract_metadata_centers,
    _marker_contract,
    _markers_from_center_map,
    _patient_id_from_sample,
    _sample_center_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "training" / "runs" / "unetpp_effb3_centerhead_x2_2_adapter_full_dataset_baseline_100ep"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "training" / "analysis" / "center_full_dataset_baseline_diagnosis"
TRAIN_MANIFEST = REPO_ROOT / "training" / "manifests" / "center_full_train_manifest.jsonl"
VAL_MANIFEST = REPO_ROOT / "training" / "manifests" / "center_full_val_manifest.jsonl"
THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9)
MAX_MARKERS = 3
BEST_THRESHOLD_FOR_VISUALS = 0.01
LOCKED_REFERENCE_THRESHOLD = 0.03


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _sample_sort_key(sample: str) -> tuple[Any, ...]:
    parts: list[Any] = []
    buf = ""
    is_digit = False
    for ch in str(sample):
        if ch.isdigit() != is_digit and buf:
            parts.append(int(buf) if is_digit else buf)
            buf = ch
            is_digit = ch.isdigit()
        else:
            buf += ch
            is_digit = ch.isdigit()
    if buf:
        parts.append(int(buf) if is_digit else buf)
    return tuple(parts)


def _align_instance_mask(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read instance mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask.astype(np.int32)
    h, w = [int(v) for v in target_hw]
    if mask.shape[:2] != (h, w):
        gh, gw = mask.shape[:2]
        y0 = (gh - h) // 2
        x0 = (gw - w) // 2
        mask = mask[y0 : y0 + h, x0 : x0 + w]
    return mask


def _labels_to_bgr(labels: np.ndarray) -> np.ndarray:
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    palette = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    for lab in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        out[labels == lab] = palette[lab % len(palette)]
    return out


def _draw_points(base_bgr: np.ndarray, points_yx: list[tuple[int, int]], *, color: tuple[int, int, int]) -> np.ndarray:
    out = base_bgr.copy()
    for y, x in points_yx:
        cv2.circle(out, (int(x), int(y)), 5, color, thickness=2, lineType=cv2.LINE_AA)
    return out


class ManifestCenterDataset(Dataset):
    def __init__(self, manifest_rows: list[dict[str, Any]], cfg: dict) -> None:
        self.rows = sorted(manifest_rows, key=lambda row: int(row["sample_index"]))
        self.dataset_root = Path(cfg["dataset"]["root"]).resolve()
        self.instance_root = Path(cfg["dataset"]["instance_root"]).resolve()
        self.input_size = int(cfg["model"]["input_size"])
        self.augment_fn = get_val_augmentations(self.input_size, self.input_size)

        encoder = cfg["model"].get("encoder") or cfg["model"].get("encoder_name")
        encoder_weights = cfg["model"].get("encoder_weights", None)
        if encoder_weights is None:
            self.preprocessing_fn = _simple_preprocess_uint8_rgb
        else:
            import segmentation_models_pytorch as smp

            self.preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image_path = (self.dataset_root / str(row["image_rel"])).resolve()
        semantic_path = (self.dataset_root / str(row["semantic_mask_rel"])).resolve()
        center_path = (self.dataset_root / str(row["center_target_rel"])).resolve()
        metadata_path = (self.dataset_root / str(row["metadata_rel"])).resolve()
        instance_path = (self.instance_root / str(row["instance_mask_rel"])).resolve()

        image = _read_image_rgb(image_path)
        mask = _read_mask_u8(semantic_path)
        center_u16 = _read_u16(center_path)
        center = (center_u16.astype(np.float32) / 65535.0).astype(np.float32)
        image, mask, center = self.augment_fn(image, mask, center=center)
        image = self.preprocessing_fn(image)
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask).long()
        center_t = torch.from_numpy(center[None, :, :]).float()
        return {
            "image": image_t,
            "mask": mask_t,
            "center": center_t,
            "sample": str(row["sample"]),
            "patient_id": str(row["patient_id"]),
            "gt_instance_count": int(row["gt_instance_count"]),
            "image_path": str(image_path),
            "semantic_mask_path": str(semantic_path),
            "instance_mask_path": str(instance_path),
            "center_path": str(center_path),
            "metadata_path": str(metadata_path),
        }


def _build_loader(cfg: dict, manifest_path: Path) -> DataLoader:
    rows = _read_jsonl(manifest_path)
    ds = ManifestCenterDataset(rows, cfg)
    batch_size = int(cfg["train"]["batch_size"])
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)


def _load_checkpoint_model(cfg: dict, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = _build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    metadata = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "epoch": int(ckpt["epoch"]),
        "extra": ckpt.get("extra") or {},
    }
    return model, metadata


def _pred_count_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(int(row["predicted_count"]) for row in rows)
    total = int(len(rows))
    return {
        "predicted_count_0": int(counts.get(0, 0)),
        "predicted_count_1": int(counts.get(1, 0)),
        "predicted_count_2": int(counts.get(2, 0)),
        "predicted_count_3": int(counts.get(3, 0)),
        "fraction_predicted_count_3": float(counts.get(3, 0) / max(total, 1)),
    }


def _confusion_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = Counter((int(row["gt_instance_count"]), int(row["predicted_count"])) for row in rows)
    out = []
    total = int(len(rows))
    for gt_count in (1, 2, 3):
        for pred_count in (0, 1, 2, 3):
            count = int(table.get((gt_count, pred_count), 0))
            out.append(
                {
                    "gt_instance_count": int(gt_count),
                    "predicted_count": int(pred_count),
                    "sample_count": int(count),
                    "fraction": float(count / max(total, 1)),
                }
            )
    return out


def _heatmap_margin(center_prob: np.ndarray, gt_inst: np.ndarray, gt_pts: list[tuple[int, int]]) -> dict[str, Any]:
    gt_scores = [float(center_prob[int(y), int(x)]) for y, x in gt_pts if 0 <= int(y) < center_prob.shape[0] and 0 <= int(x) < center_prob.shape[1]]
    far_mask = gt_inst <= 0
    far_score = float(np.max(center_prob[far_mask])) if bool(np.any(far_mask)) else 0.0
    if gt_scores:
        min_gt = float(min(gt_scores))
        max_gt = float(max(gt_scores))
    else:
        min_gt = None
        max_gt = None
    return {
        "gt_center_scores": gt_scores,
        "max_gt_center_score": max_gt,
        "min_gt_center_score": min_gt,
        "maximum_far_background_score": far_score,
        "margin": None if min_gt is None else float(min_gt - far_score),
    }


def _aggregate_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agg = _aggregate_center_rows(rows)
    pred_counts = [int(row["predicted_count"]) for row in rows]
    agg.update(
        {
            "predicted_center_count_mean": float(np.mean(pred_counts)) if pred_counts else None,
            "predicted_center_count_median": float(np.median(np.asarray(pred_counts, dtype=np.float64))) if pred_counts else None,
            "missing_gt_instances": int(sum(int(row["missing_gt_instance_markers"]) for row in rows)),
            "duplicate_instances": int(sum(int(row["multiple_markers_inside_gt_instances"]) for row in rows)),
            "markers_outside_instances": int(sum(int(row["markers_outside_all_gt_instances"]) for row in rows)),
        }
    )
    agg.update(_pred_count_distribution(rows))
    return agg


def _aggregate_heatmap(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    margins = [float(row["margin"]) for row in rows if row["margin"] is not None]
    gt_scores = [float(row["min_gt_center_score"]) for row in rows if row["min_gt_center_score"] is not None]
    far_scores = [float(row["maximum_far_background_score"]) for row in rows if row["maximum_far_background_score"] is not None]
    gt_above_num = 0
    gt_above_den = 0
    for row in rows:
        scores = [float(v) for v in json.loads(row["gt_center_scores_json"])]
        gt_above_num += sum(1 for score in scores if score >= float(threshold))
        gt_above_den += len(scores)
    return {
        "median_gt_center_score": float(np.median(np.asarray(gt_scores, dtype=np.float64))) if gt_scores else None,
        "median_far_background_score": float(np.median(np.asarray(far_scores, dtype=np.float64))) if far_scores else None,
        "median_margin": float(np.median(np.asarray(margins, dtype=np.float64))) if margins else None,
        "fraction_samples_margin_gt_0": float(np.mean([1.0 if float(row["margin"]) > 0.0 else 0.0 for row in rows if row["margin"] is not None])) if margins else None,
        "fraction_gt_centers_above_threshold": float(gt_above_num / max(gt_above_den, 1)),
    }


def _choose_best_threshold(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = None
    best_key = None
    for row in summary_rows:
        key = (
            float(row["center_f1_mean_samples"]) if row["center_f1_mean_samples"] is not None else -float("inf"),
            float(row["strict_marker_contract_pass_rate"]) if row["strict_marker_contract_pass_rate"] is not None else -float("inf"),
            float(row["exact_center_count_accuracy"]) if row["exact_center_count_accuracy"] is not None else -float("inf"),
            -float(row["localization_error_px"]) if row["localization_error_px"] is not None else -float("inf"),
            -float(row["threshold"]),
        )
        if best_key is None or key > best_key:
            best = row
            best_key = key
    assert best is not None
    return best


def _make_visual_panel(sample_payload: dict[str, Any], *, out_path: Path) -> None:
    image = _read_image_rgb(Path(sample_payload["image_path"]))
    gt_inst = _align_instance_mask(Path(sample_payload["instance_mask_path"]), target_hw=image.shape[:2])
    heat = np.asarray(sample_payload["center_prob"], dtype=np.float32)
    heat_u8 = np.clip(heat * 255.0, 0.0, 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    gt_pts = [(int(y), int(x)) for y, x in sample_payload["gt_points"]]
    best_pts = [(int(y), int(x)) for y, x in sample_payload["markers_t01"]]
    ref_pts = [(int(y), int(x)) for y, x in sample_payload["markers_t03"]]

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_gt = _draw_points(img_bgr, gt_pts, color=(0, 255, 255))
    gt_inst_bgr = _draw_points(_labels_to_bgr(gt_inst.astype(np.int32)), gt_pts, color=(0, 255, 255))
    heat_gt = _draw_points(heat_bgr, gt_pts, color=(0, 255, 255))
    heat_best = _draw_points(heat_bgr, best_pts, color=(0, 255, 0))
    heat_ref = _draw_points(heat_bgr, ref_pts, color=(0, 0, 255))
    blank = np.full_like(img_bgr, 255)
    header = np.full((72, img_bgr.shape[1] * 3, 3), 255, dtype=np.uint8)
    cv2.putText(header, sample_payload["title"], (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(header, sample_payload["subtitle"], (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    top = np.concatenate([img_gt, gt_inst_bgr, heat_gt], axis=1)
    bottom = np.concatenate([heat_best, heat_ref, blank], axis=1)
    panel = np.concatenate([header, top, bottom], axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def _category_key(sample_row: dict[str, Any]) -> tuple[str, int]:
    gt_count = int(sample_row["gt_instance_count"])
    if bool(sample_row["marker_contract_pass"]):
        return (f"gt{gt_count}_strict_pass", gt_count)
    if int(sample_row["predicted_count"]) == MAX_MARKERS:
        return (f"gt{gt_count}_saturated3", gt_count)
    if int(sample_row["markers_outside_all_gt_instances"]) > 0:
        return (f"gt{gt_count}_outside", gt_count)
    if int(sample_row["missing_gt_instance_markers"]) > 0:
        return (f"gt{gt_count}_missed", gt_count)
    return (f"gt{gt_count}_other", gt_count)


def _select_visual_samples(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_category_key(row)[0]].append(row)
    for bucket in sorted(buckets.keys()):
        bucket_rows = sorted(buckets[bucket], key=lambda row: (_sample_sort_key(row["sample"]), float(row["threshold"])))
        for row in bucket_rows:
            sample = str(row["sample"])
            if sample in seen_samples:
                continue
            chosen.append(row)
            seen_samples.add(sample)
            break
    for row in sorted(rows, key=lambda row: (_sample_sort_key(row["sample"]), float(row["threshold"]))):
        sample = str(row["sample"])
        if sample in seen_samples:
            continue
        chosen.append(row)
        seen_samples.add(sample)
        if len(chosen) >= int(limit):
            break
    return chosen[: int(limit)]


def _scheduler_state_sequence(metrics_rows: list[dict[str, Any]], scheduler_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.AdamW([param], lr=0.001, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode=str(scheduler_cfg.get("mode", "max")),
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 5)),
        min_lr=float(scheduler_cfg.get("min_lr", 0.0)),
    )
    monitor = str(scheduler_cfg.get("monitor", "center_f1_mean_samples"))
    states = []
    for row in metrics_rows:
        if str(row["epoch"]) == "0":
            continue
        value = float(row[monitor])
        before = float(opt.param_groups[0]["lr"])
        sch.step(value)
        after = float(opt.param_groups[0]["lr"])
        states.append(
            {
                "epoch": int(row["epoch"]),
                "monitor_value": float(value),
                "lr_before_step": before,
                "lr_after_step": after,
                "num_bad_epochs": int(sch.num_bad_epochs),
                "best": float(sch.best),
                "cooldown_counter": int(sch.cooldown_counter),
                "lr_reduced": bool(after < before),
            }
        )
    return states


def _scheduler_audit(run_dir: Path) -> dict[str, Any]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics_rows = list(csv.DictReader((run_dir / "metrics.csv").open("r", encoding="utf-8")))
    scheduler_cfg = dict(config.get("scheduler") or {})
    states = _scheduler_state_sequence(metrics_rows, scheduler_cfg)
    reductions = [state for state in states if bool(state["lr_reduced"])]
    last_ckpt = torch.load(run_dir / "last.pth", map_location="cpu")
    optimizer_lr = float(last_ckpt["optimizer"]["param_groups"][0]["lr"])
    return {
        "instantiated": str(scheduler_cfg.get("type", "")).strip().lower() == "reduce_on_plateau",
        "type": scheduler_cfg.get("type"),
        "monitor": scheduler_cfg.get("monitor"),
        "monitor_value_by_epoch": [{"epoch": int(row["epoch"]), "value": float(row[scheduler_cfg["monitor"]])} for row in metrics_rows if str(row["epoch"]) != "0"],
        "logged_optimizer_lr_before_scheduler_step": [{"epoch": int(row["epoch"]), "lr": float(row["lr_center_head"])} for row in metrics_rows if str(row["epoch"]) not in {"", "0"} and row.get("lr_center_head")],
        "scheduler_state_by_epoch": states,
        "expected_lr_reductions": [{"epoch": int(state["epoch"]), "lr_before_step": float(state["lr_before_step"]), "lr_after_step": float(state["lr_after_step"])} for state in reductions],
        "actual_lr_reductions": [{"epoch": int(state["epoch"]), "lr_after_step": float(state["lr_after_step"])} for state in reductions],
        "final_optimizer_lr_from_last_checkpoint": optimizer_lr,
        "root_cause": "logging_before_scheduler_step",
    }


def _evaluate_checkpoint(
    *,
    cfg: dict,
    checkpoint_tag: str,
    checkpoint_path: Path,
    split_name: str,
    manifest_path: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    loader = _build_loader(cfg, manifest_path)
    model, ckpt_meta = _load_checkpoint_model(cfg, checkpoint_path, device)
    threshold_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    pred_dist_rows: list[dict[str, Any]] = []
    heat_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    visual_payloads: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            out = model(images)
            pred_sem = torch.argmax(out["semantic"], dim=1).detach().cpu().numpy().astype(np.uint8)
            pred_center = torch.sigmoid(out["center"]).detach().cpu().numpy().astype(np.float32)
            for i in range(int(pred_sem.shape[0])):
                sample = str(batch["sample"][i])
                patient_id = str(batch["patient_id"][i])
                gt_count = int(batch["gt_instance_count"][i])
                image_path = str(batch["image_path"][i])
                instance_mask_path = str(batch["instance_mask_path"][i])
                metadata_path = str(batch["metadata_path"][i])
                gt_pts = _extract_metadata_centers(metadata_path)
                gt_inst = _align_instance_mask(Path(instance_mask_path), target_hw=pred_sem[i].shape[:2])
                heat_stats = _heatmap_margin(pred_center[i, 0], gt_inst, gt_pts)
                gt_scores_json = json.dumps(heat_stats["gt_center_scores"], ensure_ascii=False)
                markers_t01 = [(int(y), int(x)) for (y, x, _s) in _markers_from_center_map(pred_center[i, 0], pred_sem[i] == 1, BEST_THRESHOLD_FOR_VISUALS, max_markers=MAX_MARKERS)]
                markers_t03 = [(int(y), int(x)) for (y, x, _s) in _markers_from_center_map(pred_center[i, 0], pred_sem[i] == 1, LOCKED_REFERENCE_THRESHOLD, max_markers=MAX_MARKERS)]
                visual_payloads.append(
                    {
                        "checkpoint_tag": checkpoint_tag,
                        "split": split_name,
                        "sample": sample,
                        "patient_id": patient_id,
                        "gt_instance_count": gt_count,
                        "image_path": image_path,
                        "instance_mask_path": instance_mask_path,
                        "center_prob": pred_center[i, 0].copy(),
                        "gt_points": list(gt_pts),
                        "markers_t01": markers_t01,
                        "markers_t03": markers_t03,
                        "title": f"{checkpoint_tag} / {split_name} / {sample}",
                        "subtitle": "",
                    }
                )
                for threshold in THRESHOLDS:
                    pred_pts = [(int(y), int(x)) for (y, x, _score) in _markers_from_center_map(pred_center[i, 0], pred_sem[i] == 1, float(threshold), max_markers=MAX_MARKERS)]
                    center_metrics = _sample_center_metrics(pred_pts, gt_pts)
                    contract = _marker_contract(gt_inst, pred_pts)
                    row = {
                        "checkpoint_tag": checkpoint_tag,
                        "checkpoint_sha256": ckpt_meta["checkpoint_sha256"],
                        "checkpoint_epoch": ckpt_meta["epoch"],
                        "split": split_name,
                        "sample": sample,
                        "patient_id": patient_id,
                        "gt_instance_count": gt_count,
                        "threshold": float(threshold),
                        "tp": int(center_metrics["tp"]),
                        "fp": int(center_metrics["fp"]),
                        "fn": int(center_metrics["fn"]),
                        "center_precision": float(center_metrics["center_precision"]),
                        "center_recall": float(center_metrics["center_recall"]),
                        "center_f1": float(center_metrics["center_f1"]),
                        "center_count_acc": float(center_metrics["center_count_accuracy"]),
                        "exact_center_count_accuracy": float(center_metrics["center_count_accuracy"]),
                        "center_loc_err_px": float(center_metrics["center_loc_err_px"]),
                        "localization_error_px": float(center_metrics["center_loc_err_px"]),
                        "predicted_count": int(center_metrics["predicted_center_count"]),
                        "marker_contract_pass": bool(contract["marker_contract_pass"]),
                        "missing_gt_instance_markers": int(contract["missing_gt_instance_markers"]),
                        "multiple_markers_inside_gt_instances": int(contract["multiple_markers_inside_gt_instances"]),
                        "markers_outside_all_gt_instances": int(contract["markers_outside_all_gt_instances"]),
                        "max_markers_cap": int(MAX_MARKERS),
                        "predicted_count_eq_3": bool(int(center_metrics["predicted_center_count"]) == int(MAX_MARKERS)),
                        "max_gt_center_score": heat_stats["max_gt_center_score"],
                        "min_gt_center_score": heat_stats["min_gt_center_score"],
                        "maximum_far_background_score": heat_stats["maximum_far_background_score"],
                        "margin": heat_stats["margin"],
                        "gt_center_scores_json": gt_scores_json,
                    }
                    per_sample_rows.append(row)

    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in per_sample_rows:
        grouped[(str(row["checkpoint_tag"]), str(row["split"]), float(row["threshold"]))].append(row)

    for (ckpt_tag, split, threshold), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        agg = _aggregate_threshold(rows)
        best_fields = {
            "checkpoint_tag": ckpt_tag,
            "split": split,
            "threshold": float(threshold),
            **agg,
        }
        threshold_rows.append(best_fields)
        heat_rows.append(
            {
                "checkpoint_tag": ckpt_tag,
                "split": split,
                "threshold": float(threshold),
                **_aggregate_heatmap(rows, float(threshold)),
            }
        )
        dist = _pred_count_distribution(rows)
        pred_dist_rows.append({"record_type": "distribution", "checkpoint_tag": ckpt_tag, "split": split, "threshold": float(threshold), **dist})
        for confusion in _confusion_rows(rows):
            pred_dist_rows.append({"record_type": "confusion", "checkpoint_tag": ckpt_tag, "split": split, "threshold": float(threshold), **confusion})
        for gt_count in (1, 2, 3):
            gt_subset = [row for row in rows if int(row["gt_instance_count"]) == int(gt_count)]
            gt_agg = _aggregate_threshold(gt_subset)
            gt_rows.append(
                {
                    "checkpoint_tag": ckpt_tag,
                    "split": split,
                    "threshold": float(threshold),
                    "gt_instance_count": int(gt_count),
                    **gt_agg,
                }
            )
    return threshold_rows, gt_rows, pred_dist_rows, heat_rows, per_sample_rows, visual_payloads


def _summary_rows(threshold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in threshold_rows:
        grouped[(str(row["checkpoint_tag"]), str(row["split"]))].append(row)
    for (checkpoint_tag, split), rows in sorted(grouped.items()):
        best = _choose_best_threshold(rows)
        locked = next(row for row in rows if abs(float(row["threshold"]) - float(LOCKED_REFERENCE_THRESHOLD)) < 1e-9)
        out.append(
            {
                "checkpoint_tag": checkpoint_tag,
                "split": split,
                "best_threshold": float(best["threshold"]),
                "best_center_f1_mean_samples": float(best["center_f1_mean_samples"]),
                "best_strict_marker_contract_pass_rate": float(best["strict_marker_contract_pass_rate"]),
                "best_exact_center_count_accuracy": float(best["exact_center_count_accuracy"]),
                "fraction_predicted_count_3_at_best": float(best["fraction_predicted_count_3"]),
                "locked_threshold": float(LOCKED_REFERENCE_THRESHOLD),
                "locked_center_f1_mean_samples": float(locked["center_f1_mean_samples"]),
                "locked_strict_marker_contract_pass_rate": float(locked["strict_marker_contract_pass_rate"]),
                "locked_exact_center_count_accuracy": float(locked["exact_center_count_accuracy"]),
                "fraction_predicted_count_3_at_locked": float(locked["fraction_predicted_count_3"]),
            }
        )
    return out


def _classify(summary_rows: list[dict[str, Any]], heat_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _summary(checkpoint_tag: str, split: str) -> dict[str, Any]:
        return next(row for row in summary_rows if row["checkpoint_tag"] == checkpoint_tag and row["split"] == split)

    def _heat(checkpoint_tag: str, split: str, threshold: float) -> dict[str, Any]:
        return next(
            row
            for row in heat_rows
            if row["checkpoint_tag"] == checkpoint_tag
            and row["split"] == split
            and abs(float(row["threshold"]) - float(threshold)) < 1e-9
        )

    best_train = _summary("best_primary", "train")
    best_val = _summary("best_primary", "val")
    last_train = _summary("last", "train")
    last_val = _summary("last", "val")
    best_heat_train = _heat("best_primary", "train", float(best_train["best_threshold"]))
    best_heat_val = _heat("best_primary", "val", float(best_val["best_threshold"]))
    last_heat_train = _heat("last", "train", float(last_train["best_threshold"]))
    last_heat_val = _heat("last", "val", float(last_val["best_threshold"]))

    best_train_f1 = float(best_train["best_center_f1_mean_samples"])
    best_val_f1 = float(best_val["best_center_f1_mean_samples"])
    best_train_margin = float(best_heat_train["median_margin"] or 0.0)
    best_val_margin = float(best_heat_val["median_margin"] or 0.0)
    best_train_margin_pos = float(best_heat_train["fraction_samples_margin_gt_0"] or 0.0)
    best_val_margin_pos = float(best_heat_val["fraction_samples_margin_gt_0"] or 0.0)

    last_train_f1 = float(last_train["best_center_f1_mean_samples"])
    last_val_f1 = float(last_val["best_center_f1_mean_samples"])
    last_train_contract = float(last_train["best_strict_marker_contract_pass_rate"])
    last_val_contract = float(last_val["best_strict_marker_contract_pass_rate"])
    last_train_margin = float(last_heat_train["median_margin"] or 0.0)
    last_val_margin = float(last_heat_val["median_margin"] or 0.0)
    last_train_margin_pos = float(last_heat_train["fraction_samples_margin_gt_0"] or 0.0)
    last_val_margin_pos = float(last_heat_val["fraction_samples_margin_gt_0"] or 0.0)

    if (
        last_train_f1 >= 0.80
        and last_train_contract >= 0.70
        and last_train_margin > 0.0
        and last_train_margin_pos >= 0.90
        and last_val_f1 <= 0.10
        and last_val_contract <= 0.15
        and last_val_margin < 0.0
        and last_val_margin_pos <= 0.10
    ):
        status = "center_head_overfitting"
    elif best_train_f1 < 0.20 and best_train_margin <= 0.0 and best_train_margin_pos < 0.5:
        status = "center_objective_extraction_mismatch"
    elif best_train_f1 < 0.20 and best_train_margin > 0.0:
        status = "insufficient_frozen_feature_representation"
    else:
        status = "mixed_center_training_failure"
    return {
        "result": status,
        "evidence": {
            "best_primary_train_center_f1_mean_samples": best_train_f1,
            "best_primary_val_center_f1_mean_samples": best_val_f1,
            "best_primary_train_median_margin": best_train_margin,
            "best_primary_val_median_margin": best_val_margin,
            "best_primary_train_fraction_samples_margin_gt_0": best_train_margin_pos,
            "best_primary_val_fraction_samples_margin_gt_0": best_val_margin_pos,
            "last_train_center_f1_mean_samples": last_train_f1,
            "last_val_center_f1_mean_samples": last_val_f1,
            "last_train_strict_marker_contract_pass_rate": last_train_contract,
            "last_val_strict_marker_contract_pass_rate": last_val_contract,
            "last_train_median_margin": last_train_margin,
            "last_val_median_margin": last_val_margin,
            "last_train_fraction_samples_margin_gt_0": last_train_margin_pos,
            "last_val_fraction_samples_margin_gt_0": last_val_margin_pos,
        },
    }


def run(run_dir: Path, train_manifest: Path, val_manifest: Path, output_dir: Path, device: torch.device) -> dict[str, Any]:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    scheduler_audit = _scheduler_audit(run_dir)
    _write_json((output_dir / "scheduler_audit.json").resolve(), scheduler_audit)

    eval_specs = [
        ("best_primary", run_dir / "best_primary.pth", "train", train_manifest),
        ("best_primary", run_dir / "best_primary.pth", "val", val_manifest),
        ("last", run_dir / "last.pth", "train", train_manifest),
        ("last", run_dir / "last.pth", "val", val_manifest),
    ]
    threshold_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    pred_dist_rows: list[dict[str, Any]] = []
    heat_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    visuals_by_eval: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for checkpoint_tag, checkpoint_path, split_name, manifest_path in eval_specs:
        t_rows, g_rows, p_rows, h_rows, s_rows, v_rows = _evaluate_checkpoint(
            cfg=cfg,
            checkpoint_tag=checkpoint_tag,
            checkpoint_path=checkpoint_path,
            split_name=split_name,
            manifest_path=manifest_path,
            device=device,
        )
        threshold_rows.extend(t_rows)
        gt_rows.extend(g_rows)
        pred_dist_rows.extend(p_rows)
        heat_rows.extend(h_rows)
        per_sample_rows.extend(s_rows)
        visuals_by_eval[(checkpoint_tag, split_name)].extend(v_rows)

    summary_rows = _summary_rows(threshold_rows)
    classification = _classify(summary_rows, heat_rows)

    threshold_fieldnames = [
        "checkpoint_tag",
        "split",
        "threshold",
        "sample_count",
        "center_precision",
        "center_recall",
        "center_f1",
        "center_precision_mean_samples",
        "center_recall_mean_samples",
        "center_f1_mean_samples",
        "exact_center_count_accuracy",
        "strict_marker_contract_pass_count",
        "strict_marker_contract_pass_rate",
        "localization_error_px",
        "predicted_center_count_mean",
        "predicted_center_count_median",
        "predicted_count_0",
        "predicted_count_1",
        "predicted_count_2",
        "predicted_count_3",
        "fraction_predicted_count_3",
        "missing_gt_instances",
        "duplicate_instances",
        "markers_outside_instances",
    ]
    _write_csv((output_dir / "threshold_summary.csv").resolve(), threshold_rows, threshold_fieldnames)
    _write_csv((output_dir / "gt_count_metrics.csv").resolve(), gt_rows, ["checkpoint_tag", "split", "threshold", "gt_instance_count"] + threshold_fieldnames[3:])
    _write_csv(
        (output_dir / "predicted_count_distribution.csv").resolve(),
        pred_dist_rows,
        ["record_type", "checkpoint_tag", "split", "threshold", "predicted_count_0", "predicted_count_1", "predicted_count_2", "predicted_count_3", "fraction_predicted_count_3", "gt_instance_count", "predicted_count", "sample_count", "fraction"],
    )
    _write_csv(
        (output_dir / "heatmap_separation_summary.csv").resolve(),
        heat_rows,
        ["checkpoint_tag", "split", "threshold", "median_gt_center_score", "median_far_background_score", "median_margin", "fraction_samples_margin_gt_0", "fraction_gt_centers_above_threshold"],
    )
    _write_csv(
        (output_dir / "per_sample_diagnostics.csv").resolve(),
        per_sample_rows,
        [
            "checkpoint_tag",
            "checkpoint_sha256",
            "checkpoint_epoch",
            "split",
            "sample",
            "patient_id",
            "gt_instance_count",
            "threshold",
            "tp",
            "fp",
            "fn",
            "center_precision",
            "center_recall",
            "center_f1",
            "center_count_acc",
            "exact_center_count_accuracy",
            "center_loc_err_px",
            "localization_error_px",
            "predicted_count",
            "marker_contract_pass",
            "missing_gt_instance_markers",
            "multiple_markers_inside_gt_instances",
            "markers_outside_all_gt_instances",
            "max_markers_cap",
            "predicted_count_eq_3",
            "max_gt_center_score",
            "min_gt_center_score",
            "maximum_far_background_score",
            "margin",
            "gt_center_scores_json",
        ],
    )
    _write_csv(
        (output_dir / "train_vs_val_metrics.csv").resolve(),
        summary_rows,
        [
            "checkpoint_tag",
            "split",
            "best_threshold",
            "best_center_f1_mean_samples",
            "best_strict_marker_contract_pass_rate",
            "best_exact_center_count_accuracy",
            "fraction_predicted_count_3_at_best",
            "locked_threshold",
            "locked_center_f1_mean_samples",
            "locked_strict_marker_contract_pass_rate",
            "locked_exact_center_count_accuracy",
            "fraction_predicted_count_3_at_locked",
        ],
    )

    visual_dir = (output_dir / "visual_review").resolve()
    best_val_rows = [row for row in per_sample_rows if row["checkpoint_tag"] == "best_primary" and row["split"] == "val" and abs(float(row["threshold"]) - float(BEST_THRESHOLD_FOR_VISUALS)) < 1e-9]
    best_train_rows = [row for row in per_sample_rows if row["checkpoint_tag"] == "best_primary" and row["split"] == "train" and abs(float(row["threshold"]) - float(BEST_THRESHOLD_FOR_VISUALS)) < 1e-9]
    selected_val = _select_visual_samples(best_val_rows, limit=20)
    selected_train = _select_visual_samples(best_train_rows, limit=10)
    visual_lookup = {(str(v["checkpoint_tag"]), str(v["split"]), str(v["sample"])): v for values in visuals_by_eval.values() for v in values}
    for split_name, rows in (("train", selected_train), ("val", selected_val)):
        for idx, row in enumerate(rows, start=1):
            payload = dict(visual_lookup[(str(row["checkpoint_tag"]), str(row["split"]), str(row["sample"]))])
            payload["subtitle"] = (
                f"gt={row['gt_instance_count']}  pred@0.01={row['predicted_count']}  "
                f"strict={int(bool(row['marker_contract_pass']))}  "
                f"margin={float(row['margin']) if row['margin'] is not None else 'na'}"
            )
            _make_visual_panel(payload, out_path=(visual_dir / split_name / f"{idx:02d}__{row['sample']}.png").resolve())

    diagnosis_summary = {
        "run_dir": str(run_dir.resolve()),
        "train_manifest": str(train_manifest.resolve()),
        "val_manifest": str(val_manifest.resolve()),
        "scheduler": scheduler_audit,
        "best_checkpoint": {
            "path": str((run_dir / "best_primary.pth").resolve()),
            "sha256": _sha256_file(run_dir / "best_primary.pth"),
        },
        "last_checkpoint": {
            "path": str((run_dir / "last.pth").resolve()),
            "sha256": _sha256_file(run_dir / "last.pth"),
        },
        "summary_rows": summary_rows,
        "classification": classification,
    }
    _write_json((output_dir / "diagnosis_summary.json").resolve(), diagnosis_summary)
    files_to_review = [
        output_dir / "diagnosis_summary.json",
        output_dir / "scheduler_audit.json",
        output_dir / "train_vs_val_metrics.csv",
        output_dir / "threshold_summary.csv",
        output_dir / "gt_count_metrics.csv",
        output_dir / "predicted_count_distribution.csv",
        output_dir / "heatmap_separation_summary.csv",
        output_dir / "per_sample_diagnostics.csv",
        output_dir / "visual_review",
    ]
    _write_text((output_dir / "files_to_review.txt").resolve(), "\n".join(str(path.resolve()) for path in files_to_review) + "\n")
    return diagnosis_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--train-manifest", type=str, default=str(TRAIN_MANIFEST))
    ap.add_argument("--val-manifest", type=str, default=str(VAL_MANIFEST))
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    device = torch.device(str(args.device))
    summary = run(
        run_dir=Path(args.run_dir).resolve(),
        train_manifest=Path(args.train_manifest).resolve(),
        val_manifest=Path(args.val_manifest).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        device=device,
    )
    print(json.dumps(summary["classification"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
