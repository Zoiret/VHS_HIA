from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from analyze_spatial_frozenbase_pilot import _read_u8, _reconstruct_instances, _save_compare
from augmentations import get_val_augmentations
from dataset_centerhead import SegmentationWithCenterDataset
from metrics import compute_per_class_metrics_from_logits
from train_centerhead import _autocast_ctx, _build_model, _instance_score, _make_device, _read_yaml, _seed_all
from validate_centerhead import (
    _best_perm_sum,
    _case_type,
    _extract_metadata_centers,
    _iou_matrix,
    _markers_from_center_map,
    _match_centers,
)


@dataclass
class TapSpec:
    role: str
    configured_path: str
    actual_path: str | None
    exists: bool
    participates_in_forward: bool
    output_shape: list[int] | None
    channels: int | None
    stride: int | None
    hook_calls: int
    replacement_reason: str | None = None


@dataclass
class ChannelStats:
    mean: torch.Tensor
    std: torch.Tensor
    zero_var_channels: int
    feature_norm_mean: float
    feature_norm_std: float
    feature_norm_min: float
    feature_norm_max: float


class LinearProbe(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.linear = torch.nn.Linear(int(in_channels), 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def _simple_preprocess_uint8_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    return (img_rgb_u8.astype(np.float32) / 255.0).astype(np.float32)


def _build_probe_loaders(cfg: dict, device: torch.device):
    ds_root = Path(cfg["dataset"]["root"]).resolve()
    train_txt = Path(cfg["dataset"]["train_txt"]).resolve()
    val_txt = Path(cfg["dataset"]["val_txt"]).resolve()
    num_classes = int(cfg["model"]["classes"])
    input_size = int(cfg["model"]["input_size"])
    encoder = cfg["model"].get("encoder") or cfg["model"].get("encoder_name")
    if not encoder:
        raise SystemExit("Config: model.encoder_name is required")
    encoder_weights = cfg["model"].get("encoder_weights", None)
    if encoder_weights is None:
        preprocessing_fn = _simple_preprocess_uint8_rgb
    else:
        import segmentation_models_pytorch as smp

        preprocessing_fn = smp.encoders.get_preprocessing_fn(str(encoder), encoder_weights)

    # Probe should observe the fixed dataset split without stochastic train augmentations.
    fixed_aug = get_val_augmentations(input_size, input_size)
    train_ds = SegmentationWithCenterDataset(
        dataset_root=ds_root,
        split_txt=train_txt,
        num_classes=num_classes,
        augment_fn=fixed_aug,
        preprocessing_fn=preprocessing_fn,
    )
    val_ds = SegmentationWithCenterDataset(
        dataset_root=ds_root,
        split_txt=val_txt,
        num_classes=num_classes,
        augment_fn=fixed_aug,
        preprocessing_fn=preprocessing_fn,
    )
    batch_size = int((cfg.get("train") or {}).get("batch_size", 1))
    num_workers = int((cfg.get("train") or {}).get("num_workers", 0))
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
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
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


class FeatureHookManager:
    def __init__(self, model: torch.nn.Module, module_paths: list[str]):
        self.model = model
        self.module_paths = list(module_paths)
        self.handles = []
        self.calls = {p: 0 for p in self.module_paths}
        self.outputs: dict[str, torch.Tensor] = {}
        mods = dict(model.named_modules())
        for path in self.module_paths:
            mod = mods.get(path)
            if mod is None:
                continue
            self.handles.append(mod.register_forward_hook(self._make_hook(path)))

    def _make_hook(self, path: str):
        def _hook(_module, _inputs, output):
            self.calls[path] = int(self.calls.get(path, 0)) + 1
            if torch.is_tensor(output):
                self.outputs[path] = output
            elif isinstance(output, (list, tuple)) and output and torch.is_tensor(output[0]):
                self.outputs[path] = output[0]

        return _hook

    def reset_outputs(self) -> None:
        self.outputs = {}

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def _json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _to_float(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _safe_mean(xs: list[float]) -> float | None:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _safe_std(xs: list[float]) -> float | None:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(vals) <= 1:
        return 0.0 if vals else None
    m = sum(vals) / len(vals)
    return float((sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5)


def _normalize_name(s: str) -> str:
    return str(s).replace("\\", "/")


def _self_check_binary_metrics() -> None:
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    s = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    auc = _binary_roc_auc(y, s)
    ap = _binary_average_precision(y, s)
    if abs(auc - 1.0) > 1e-7:
        raise SystemExit(f"ROC AUC self-check failed: got {auc}")
    if abs(ap - 1.0) > 1e-7:
        raise SystemExit(f"Average precision self-check failed: got {ap}")


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and y_score[order[j]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = float(np.sum(ranks[y_true == 1]))
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / max(pos * neg, 1)
    return float(auc)


def _binary_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = int(np.sum(y_true == 1))
    if pos == 0:
        return None
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for yi in y_sorted:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / pos
        precision = tp / max(tp + fp, 1)
        if yi == 1:
            ap += precision * max(recall - prev_recall, 0.0)
            prev_recall = recall
    return float(ap)


def _binary_balanced_accuracy(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> float | None:
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float64)
    pred = (y_prob >= float(thr)).astype(np.int64)
    pos = y_true == 1
    neg = y_true == 0
    if int(np.sum(pos)) == 0 or int(np.sum(neg)) == 0:
        return None
    tpr = float(np.mean(pred[pos] == 1))
    tnr = float(np.mean(pred[neg] == 0))
    return float((tpr + tnr) / 2.0)


def _pooled_std(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return None
    va = float(np.var(a, ddof=1)) if a.size > 1 else 0.0
    vb = float(np.var(b, ddof=1)) if b.size > 1 else 0.0
    denom = max((a.size - 1) + (b.size - 1), 1)
    pooled = (((a.size - 1) * va) + ((b.size - 1) * vb)) / denom
    return float(max(pooled, 0.0) ** 0.5)


def _grid_from_yx(coords_yx: np.ndarray, h: int, w: int, device: torch.device) -> torch.Tensor:
    if coords_yx.size == 0:
        return torch.zeros((1, 0, 1, 2), dtype=torch.float32, device=device)
    y = torch.from_numpy(coords_yx[:, 0].astype(np.float32)).to(device)
    x = torch.from_numpy(coords_yx[:, 1].astype(np.float32)).to(device)
    x_norm = ((x + 0.5) / float(w)) * 2.0 - 1.0
    y_norm = ((y + 0.5) / float(h)) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
    return grid


def _extract_vectors(feature_map: torch.Tensor, coords_yx: np.ndarray, out_h: int = 768, out_w: int = 768) -> torch.Tensor:
    if coords_yx.size == 0:
        return torch.zeros((0, int(feature_map.shape[1])), dtype=torch.float32)
    grid = _grid_from_yx(coords_yx, out_h, out_w, feature_map.device)
    sampled = F.grid_sample(feature_map.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return sampled[0, :, :, 0].transpose(0, 1).detach().cpu()


def _sample_coords(
    *,
    center_map: np.ndarray,
    rng: np.random.Generator,
    split: str,
) -> dict[str, np.ndarray]:
    pos = np.argwhere(center_map >= 0.9999)
    near = np.argwhere((center_map >= 0.1) & (center_map < 0.9999))
    far = np.argwhere(center_map < 0.1)
    n_pos = int(pos.shape[0])
    if n_pos <= 0:
        return {
            "pos": pos.astype(np.int32),
            "near": np.zeros((0, 2), dtype=np.int32),
            "far": np.zeros((0, 2), dtype=np.int32),
            "near_all_count": int(near.shape[0]),
            "far_all_count": int(far.shape[0]),
        }
    far_n = min(int(far.shape[0]), n_pos)
    far_idx = rng.choice(int(far.shape[0]), size=far_n, replace=False) if far_n > 0 else np.asarray([], dtype=np.int64)
    far_sample = far[far_idx] if far_n > 0 else np.zeros((0, 2), dtype=np.int32)
    if split == "train":
        near_sample = np.zeros((0, 2), dtype=np.int32)
    else:
        near_n = min(int(near.shape[0]), n_pos)
        near_idx = rng.choice(int(near.shape[0]), size=near_n, replace=False) if near_n > 0 else np.asarray([], dtype=np.int64)
        near_sample = near[near_idx] if near_n > 0 else np.zeros((0, 2), dtype=np.int32)
    return {
        "pos": pos.astype(np.int32),
        "near": near_sample.astype(np.int32),
        "far": far_sample.astype(np.int32),
        "near_all_count": int(near.shape[0]),
        "far_all_count": int(far.shape[0]),
    }


def _collect_bn_stats(model: torch.nn.Module) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    out = []
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            rm = getattr(m, "running_mean", None)
            rv = getattr(m, "running_var", None)
            if rm is not None and rv is not None:
                out.append((name, rm.detach().clone(), rv.detach().clone()))
    return out


def _max_bn_delta(model: torch.nn.Module, ref: list[tuple[str, torch.Tensor, torch.Tensor]]) -> float:
    mods = dict(model.named_modules())
    max_d = 0.0
    for name, rm0, rv0 in ref:
        mod = mods.get(name)
        if mod is None:
            continue
        rm = getattr(mod, "running_mean", None)
        rv = getattr(mod, "running_var", None)
        if rm is None or rv is None:
            continue
        d = 0.0
        if rm.numel():
            d = max(d, float((rm.detach() - rm0).abs().max().item()))
        if rv.numel():
            d = max(d, float((rv.detach() - rv0).abs().max().item()))
        max_d = max(max_d, d)
    return float(max_d)


def _max_parameter_delta(model: torch.nn.Module, ref: dict[str, torch.Tensor]) -> float:
    max_d = 0.0
    for name, p in model.named_parameters():
        old = ref.get(name)
        if old is None:
            continue
        if p.numel():
            max_d = max(max_d, float((p.detach().cpu() - old).abs().max().item()))
    return float(max_d)


def _probe_output_dir(cfg: dict, smoke_test: bool) -> Path:
    base = Path((cfg.get("probe") or {}).get("output_dir", "training/analysis/decoder_center_feature_probe")).resolve()
    return (base / "smoke_test") if smoke_test else base


def _candidate_taps(cfg: dict) -> list[dict]:
    taps = (cfg.get("probe") or {}).get("candidate_taps", [])
    if not isinstance(taps, list) or not taps:
        raise SystemExit("Config: probe.candidate_taps must be a non-empty list")
    out = []
    for item in taps:
        if not isinstance(item, dict):
            raise SystemExit("Config: each probe.candidate_taps item must be a dict")
        role = str(item.get("role", "")).strip().lower()
        path = str(item.get("path", "")).strip()
        if not role or not path:
            raise SystemExit("Config: each candidate tap requires role and path")
        out.append({"role": role, "path": path})
    return out


def _thresholds(cfg: dict) -> list[float]:
    vals = (cfg.get("probe") or {}).get("threshold_sweep", [])
    if not isinstance(vals, list) or not vals:
        raise SystemExit("Config: probe.threshold_sweep must be a list")
    return [float(x) for x in vals]


def _parse_block_ij(path: str) -> tuple[int, int] | None:
    m = re.search(r"\.x_(\d+)_(\d+)$", str(path))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _resolve_candidate_path(model: torch.nn.Module, configured_path: str) -> tuple[str | None, str | None]:
    mods = dict(model.named_modules())
    if configured_path in mods:
        return configured_path, None
    target = _parse_block_ij(configured_path)
    block_paths = [p for p in mods.keys() if p.startswith("base.decoder.blocks.x_")]
    if target is None or not block_paths:
        return None, "missing and no compatible decoder block found"
    best = None
    best_key = None
    for path in block_paths:
        ij = _parse_block_ij(path)
        if ij is None:
            continue
        key = (abs(ij[0] - target[0]) + abs(ij[1] - target[1]), abs(ij[1] - target[1]), abs(ij[0] - target[0]), path)
        if best_key is None or key < best_key:
            best_key = key
            best = path
    if best is None:
        return None, "missing and no compatible decoder block found"
    return best, f"configured path missing; replaced with nearest existing block {best}"


def _infer_dependencies(actual_paths: list[str], all_block_paths: list[str]) -> dict[str, dict]:
    existing = set(all_block_paths)
    out = {}
    for path in actual_paths:
        ij = _parse_block_ij(path)
        if ij is None:
            out[path] = {"parents": [], "children": []}
            continue
        i, j = ij
        parents = []
        for cand in [f"base.decoder.blocks.x_{i}_{j-1}", f"base.decoder.blocks.x_{i+1}_{j-1}"]:
            if cand in existing:
                parents.append(cand)
        children = []
        for cand in [f"base.decoder.blocks.x_{i}_{j+1}", f"base.decoder.blocks.x_{i-1}_{j+1}"]:
            if cand in existing:
                children.append(cand)
        out[path] = {"parents": parents, "children": children}
    return out


def _run_decoder_audit(model: torch.nn.Module, batch: dict, device: torch.device, amp_enabled: bool, candidates: list[dict]) -> tuple[dict, list[TapSpec]]:
    decoder = model.decoder
    decoder_named_modules = [name for name, _ in decoder.named_modules()]
    top_blocks = []
    if hasattr(decoder, "blocks") and isinstance(decoder.blocks, torch.nn.ModuleDict):
        top_blocks = [f"base.decoder.blocks.{name}" for name in decoder.blocks.keys()]
    hook_mgr = FeatureHookManager(model, top_blocks)
    with torch.no_grad():
        hook_mgr.reset_outputs()
        with _autocast_ctx(device, enabled=amp_enabled):
            sem_logits, decoder_output = model.forward_base(batch["image"].to(device))
    top_rows = []
    for path in top_blocks:
        feat = hook_mgr.outputs.get(path)
        shape = list(feat.shape) if feat is not None else None
        h = int(shape[-2]) if shape is not None else None
        stride = int(round(768 / max(h, 1))) if h is not None else None
        top_rows.append(
            {
                "path": path,
                "output_shape": shape,
                "channels": int(shape[1]) if shape is not None else None,
                "spatial_stride_vs_768": stride,
                "hook_calls": int(hook_mgr.calls.get(path, 0)),
                "participates_in_forward": bool(hook_mgr.calls.get(path, 0) > 0),
            }
        )
    taps: list[TapSpec] = []
    for cand in candidates:
        actual, reason = _resolve_candidate_path(model, cand["path"])
        feat = hook_mgr.outputs.get(actual) if actual else None
        shape = list(feat.shape) if feat is not None else None
        h = int(shape[-2]) if shape is not None else None
        taps.append(
            TapSpec(
                role=cand["role"],
                configured_path=cand["path"],
                actual_path=actual,
                exists=bool(actual is not None),
                participates_in_forward=bool(actual is not None and hook_mgr.calls.get(actual, 0) > 0),
                output_shape=shape,
                channels=int(shape[1]) if shape is not None else None,
                stride=int(round(768 / max(h, 1))) if h is not None else None,
                hook_calls=int(hook_mgr.calls.get(actual, 0)) if actual is not None else 0,
                replacement_reason=reason,
            )
        )
    deps = _infer_dependencies([t.actual_path for t in taps if t.actual_path], top_blocks)
    audit = {
        "decoder_class": decoder.__class__.__name__,
        "decoder_named_modules": decoder_named_modules,
        "top_level_decoder_blocks": top_rows,
        "candidate_taps": candidates,
        "actual_taps": [t.__dict__ for t in taps],
        "dependency_relation": deps,
        "final_decoder_output_shape": list(decoder_output.shape),
        "semantic_output_shape": list(sem_logits.shape),
    }
    hook_mgr.close()
    return audit, taps


def _collect_sampled_vectors(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    amp_enabled: bool,
    taps: list[TapSpec],
    seeds: list[int],
    split: str,
) -> tuple[dict, dict]:
    actual_taps = [t.actual_path for t in taps if t.actual_path and t.participates_in_forward]
    hook_mgr = FeatureHookManager(model, actual_taps)
    data = {
        "by_tap_seed": {
            t.actual_path: {
                int(seed): {"train_pos": [], "train_far": [], "val_pos": [], "val_near": [], "val_far": []} for seed in seeds
            }
            for t in taps
            if t.actual_path and t.participates_in_forward
        }
    }
    stats = {
        "split": split,
        "images_with_no_exact_positives": {int(seed): 0 for seed in seeds},
        "sample_counts": {int(seed): {"positive": 0, "far": 0, "near": 0} for seed in seeds},
        "raw_region_counts": {int(seed): {"near": 0, "far": 0} for seed in seeds},
    }
    t0 = time.perf_counter()
    sample_counter = 0
    for batch in loader:
        with torch.no_grad():
            hook_mgr.reset_outputs()
            with _autocast_ctx(device, enabled=amp_enabled):
                _sem_logits, _decoder_output = model.forward_base(batch["image"].to(device, non_blocking=True))
        centers = batch["center"].detach().cpu().numpy().astype(np.float32)
        for bi in range(int(centers.shape[0])):
            center_map = centers[bi, 0]
            for seed in seeds:
                rng = np.random.default_rng(int(seed) * 1000003 + int(sample_counter))
                sampled = _sample_coords(center_map=center_map, rng=rng, split=split)
                if int(sampled["pos"].shape[0]) == 0:
                    stats["images_with_no_exact_positives"][int(seed)] += 1
                stats["sample_counts"][int(seed)]["positive"] += int(sampled["pos"].shape[0])
                stats["sample_counts"][int(seed)]["far"] += int(sampled["far"].shape[0])
                stats["sample_counts"][int(seed)]["near"] += int(sampled["near"].shape[0])
                stats["raw_region_counts"][int(seed)]["near"] += int(sampled["near_all_count"])
                stats["raw_region_counts"][int(seed)]["far"] += int(sampled["far_all_count"])
                for tap in taps:
                    if not tap.actual_path or not tap.participates_in_forward:
                        continue
                    fmap = hook_mgr.outputs[tap.actual_path][bi : bi + 1]
                    tap_seed_store = data["by_tap_seed"][tap.actual_path][int(seed)]
                    if int(sampled["pos"].shape[0]) > 0:
                        tap_seed_store[f"{split}_pos"].append(_extract_vectors(fmap, sampled["pos"]))
                    if int(sampled["far"].shape[0]) > 0:
                        tap_seed_store[f"{split}_far"].append(_extract_vectors(fmap, sampled["far"]))
                    if split == "val" and int(sampled["near"].shape[0]) > 0:
                        tap_seed_store["val_near"].append(_extract_vectors(fmap, sampled["near"]))
            sample_counter += 1
    stats["wall_time_sec"] = float(time.perf_counter() - t0)
    hook_mgr.close()
    return data, stats


def _cat_vectors(parts: list[torch.Tensor], channels: int) -> torch.Tensor:
    if not parts:
        return torch.zeros((0, int(channels)), dtype=torch.float32)
    return torch.cat(parts, dim=0).float()


def _compute_channel_stats(train_pos: torch.Tensor, train_far: torch.Tensor) -> ChannelStats:
    x = torch.cat([train_pos, train_far], dim=0).float()
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    norms = torch.linalg.norm(x, dim=1)
    return ChannelStats(
        mean=mean,
        std=std,
        zero_var_channels=int(torch.sum(std <= 1e-6).item()),
        feature_norm_mean=float(norms.mean().item()) if norms.numel() else 0.0,
        feature_norm_std=float(norms.std(unbiased=False).item()) if norms.numel() else 0.0,
        feature_norm_min=float(norms.min().item()) if norms.numel() else 0.0,
        feature_norm_max=float(norms.max().item()) if norms.numel() else 0.0,
    )


def _standardize(x: torch.Tensor, stats: ChannelStats) -> torch.Tensor:
    denom = torch.clamp(stats.std, min=1e-6)
    return (x.float() - stats.mean) / denom


def _train_linear_probe(
    *,
    train_pos: torch.Tensor,
    train_far: torch.Tensor,
    stats: ChannelStats,
    seed: int,
    lr: float,
    weight_decay: float,
    steps: int,
    smoke_steps: int | None = None,
) -> tuple[LinearProbe, dict]:
    x_pos = _standardize(train_pos, stats)
    x_far = _standardize(train_far, stats)
    x = torch.cat([x_pos, x_far], dim=0)
    y = torch.cat([torch.ones(x_pos.shape[0]), torch.zeros(x_far.shape[0])], dim=0)
    probe = LinearProbe(int(x.shape[1]))
    _seed_all(int(seed))
    for p in probe.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.02)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    run_steps = int(smoke_steps if smoke_steps is not None else steps)
    grad_finite_all = True
    loss_last = None
    for _ in range(int(run_steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(x)
        loss = loss_fn(logits, y)
        loss.backward()
        grads = [p.grad.detach() for p in probe.parameters() if p.grad is not None]
        grad_finite = all(bool(torch.isfinite(g).all().item()) for g in grads)
        grad_finite_all = bool(grad_finite_all and grad_finite)
        optimizer.step()
        loss_last = float(loss.item())
    with torch.no_grad():
        pos_logits = probe(x_pos)
        far_logits = probe(x_far)
        w = probe.linear.weight.detach().float()
        b = probe.linear.bias.detach().float()
    info = {
        "train_loss": float(loss_last) if loss_last is not None else None,
        "positive_logit_mean": float(pos_logits.mean().item()) if pos_logits.numel() else None,
        "far_logit_mean": float(far_logits.mean().item()) if far_logits.numel() else None,
        "weight_norm": float(w.norm().item()),
        "bias": float(b.mean().item()),
        "gradient_finite": bool(grad_finite_all),
        "probe_parameter_count": int(sum(int(p.numel()) for p in probe.parameters())),
    }
    return probe, info


def _classification_metrics(
    *,
    probe: LinearProbe,
    val_pos: torch.Tensor,
    val_near: torch.Tensor,
    val_far: torch.Tensor,
    stats: ChannelStats,
) -> dict:
    x_pos = _standardize(val_pos, stats)
    x_near = _standardize(val_near, stats)
    x_far = _standardize(val_far, stats)
    with torch.no_grad():
        pos_prob = torch.sigmoid(probe(x_pos)).cpu().numpy().astype(np.float64) if x_pos.numel() else np.zeros((0,), dtype=np.float64)
        near_prob = torch.sigmoid(probe(x_near)).cpu().numpy().astype(np.float64) if x_near.numel() else np.zeros((0,), dtype=np.float64)
        far_prob = torch.sigmoid(probe(x_far)).cpu().numpy().astype(np.float64) if x_far.numel() else np.zeros((0,), dtype=np.float64)
    labels = np.concatenate([np.ones_like(pos_prob, dtype=np.int64), np.zeros_like(far_prob, dtype=np.int64)])
    scores = np.concatenate([pos_prob, far_prob]) if pos_prob.size or far_prob.size else np.zeros((0,), dtype=np.float64)
    pooled = _pooled_std(pos_prob, far_prob)
    return {
        "roc_auc": _binary_roc_auc(labels, scores) if scores.size else None,
        "average_precision": _binary_average_precision(labels, scores) if scores.size else None,
        "balanced_accuracy": _binary_balanced_accuracy(labels, scores) if scores.size else None,
        "pos_probability_mean": float(np.mean(pos_prob)) if pos_prob.size else None,
        "near_probability_mean": float(np.mean(near_prob)) if near_prob.size else None,
        "far_probability_mean": float(np.mean(far_prob)) if far_prob.size else None,
        "pos_probability_std": float(np.std(pos_prob)) if pos_prob.size else None,
        "near_probability_std": float(np.std(near_prob)) if near_prob.size else None,
        "far_probability_std": float(np.std(far_prob)) if far_prob.size else None,
        "pos_minus_far": (float(np.mean(pos_prob)) - float(np.mean(far_prob))) if pos_prob.size and far_prob.size else None,
        "near_minus_far": (float(np.mean(near_prob)) - float(np.mean(far_prob))) if near_prob.size and far_prob.size else None,
        "pos_minus_near": (float(np.mean(pos_prob)) - float(np.mean(near_prob))) if pos_prob.size and near_prob.size else None,
        "standardized_separation": ((float(np.mean(pos_prob)) - float(np.mean(far_prob))) / max(float(pooled), 1e-6)) if pooled is not None and pos_prob.size and far_prob.size else None,
    }


def _standardize_feature_map(feature_map: torch.Tensor, stats: ChannelStats) -> torch.Tensor:
    mean = stats.mean.to(feature_map.device).view(1, -1, 1, 1)
    std = torch.clamp(stats.std.to(feature_map.device), min=1e-6).view(1, -1, 1, 1)
    return (feature_map.float() - mean) / std


def _probe_logits_single(feature_map: torch.Tensor, probe: LinearProbe, stats: ChannelStats, out_hw: tuple[int, int]) -> torch.Tensor:
    x = _standardize_feature_map(feature_map, stats)
    weight = probe.linear.weight.detach().float().view(1, -1, 1, 1).to(feature_map.device)
    bias = probe.linear.bias.detach().float().view(1).to(feature_map.device)
    native = F.conv2d(x, weight=weight, bias=bias)
    if tuple(native.shape[-2:]) != tuple(out_hw):
        native = F.interpolate(native, size=out_hw, mode="bilinear", align_corners=False)
    return native


def _probe_logits_fusion(
    feature_a: torch.Tensor,
    feature_b: torch.Tensor,
    probe: LinearProbe,
    stats_a: ChannelStats,
    stats_b: ChannelStats,
    out_hw: tuple[int, int],
) -> tuple[torch.Tensor, dict]:
    xa = _standardize_feature_map(feature_a, stats_a)
    xb = _standardize_feature_map(feature_b, stats_b)
    ca = int(xa.shape[1])
    wa = probe.linear.weight.detach().float()[:, :ca].contiguous().view(1, ca, 1, 1).to(feature_a.device)
    wb = probe.linear.weight.detach().float()[:, ca:].contiguous().view(1, int(xb.shape[1]), 1, 1).to(feature_b.device)
    bias = probe.linear.bias.detach().float().view(1).to(feature_a.device)
    la = F.conv2d(xa, weight=wa, bias=None)
    lb = F.conv2d(xb, weight=wb, bias=None)
    if tuple(la.shape[-2:]) != tuple(out_hw):
        la = F.interpolate(la, size=out_hw, mode="bilinear", align_corners=False)
    if tuple(lb.shape[-2:]) != tuple(out_hw):
        lb = F.interpolate(lb, size=out_hw, mode="bilinear", align_corners=False)
    return la + lb + bias.view(1, 1, 1, 1), {
        "tap_a_weight_norm": float(wa.norm().item()),
        "tap_b_weight_norm": float(wb.norm().item()),
    }


def _sample_failure_modes(
    *,
    gt_inst: np.ndarray,
    pred_pts: list[tuple[int, int, float]],
    gt_pts: list[tuple[int, int]],
    pred_k: int,
    mean_iou: float,
    tp: int,
) -> dict[str, int]:
    several_peaks = 0
    peak_between = 0
    per_leaflet = Counter()
    for y, x, _score in pred_pts:
        lab = int(gt_inst[int(y), int(x)]) if 0 <= int(y) < gt_inst.shape[0] and 0 <= int(x) < gt_inst.shape[1] else 0
        if lab in (1, 2, 3):
            per_leaflet[lab] += 1
        else:
            peak_between = 1
    if any(v > 1 for v in per_leaflet.values()):
        several_peaks = 1
    case = _case_type(len(gt_pts), pred_k)
    marker_correct_but_reconstruction_wrong = int(len(pred_pts) == len(gt_pts) and tp == len(gt_pts) and (case != "correct" or mean_iou < 0.90))
    displaced = int(len(pred_pts) > 0 and tp < min(len(pred_pts), len(gt_pts)) and len(pred_pts) <= len(gt_pts))
    return {
        "missing_center": int(len(pred_pts) == 0 and len(gt_pts) > 0),
        "displaced_center": displaced,
        "extra_center": int(len(pred_pts) > len(gt_pts)),
        "several_peaks_inside_one_leaflet": several_peaks,
        "peak_between_gt_leaflets": peak_between,
        "marker_correct_but_reconstruction_wrong": marker_correct_but_reconstruction_wrong,
        "merged_reconstruction": int(case == "merged"),
        "fragmented_reconstruction": int(case == "fragmented"),
    }


def _init_eval_acc() -> dict:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "loc_err_sum": 0.0,
        "loc_err_n": 0,
        "count_acc_ok": 0,
        "count_acc_n": 0,
        "center_pos_frac_sum": 0.0,
        "center_pos_frac_n": 0,
        "pred_count_sum": 0.0,
        "gt_count_sum": 0.0,
        "zero_center_cases": 0,
        "extra_center_cases": 0,
        "prob_pos_sum": 0.0,
        "prob_pos_n": 0,
        "prob_near_sum": 0.0,
        "prob_near_n": 0,
        "prob_far_sum": 0.0,
        "prob_far_n": 0,
        "prob_max_sum": 0.0,
        "prob_max_n": 0,
        "inst_exact": 0,
        "inst_n": 0,
        "inst_merged": 0,
        "inst_fragmented": 0,
        "inst_mixed": 0,
        "inst_mean_iou_sum": 0.0,
        "inst_median_iou_list": [],
        "inst_perfect": 0,
        "failure_modes": Counter(),
    }


def _finalize_eval_acc(acc: dict) -> dict:
    tp = int(acc["tp"])
    fp = int(acc["fp"])
    fn = int(acc["fn"])
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float((2 * precision * recall) / max(precision + recall, 1e-7))
    metrics = {
        "center_precision": precision,
        "center_recall": recall,
        "center_f1": f1,
        "center_count_acc": float(acc["count_acc_ok"] / max(acc["count_acc_n"], 1)),
        "center_loc_err_px": float(acc["loc_err_sum"] / max(acc["loc_err_n"], 1)),
        "center_pos_frac": float(acc["center_pos_frac_sum"] / max(acc["center_pos_frac_n"], 1)) if acc["center_pos_frac_n"] > 0 else None,
        "center_pred_count_mean": float(acc["pred_count_sum"] / max(acc["center_pos_frac_n"], 1)) if acc["center_pos_frac_n"] > 0 else None,
        "center_gt_count_mean": float(acc["gt_count_sum"] / max(acc["center_pos_frac_n"], 1)) if acc["center_pos_frac_n"] > 0 else None,
        "center_zero_cases": int(acc["zero_center_cases"]),
        "center_extra_cases": int(acc["extra_center_cases"]),
        "instance_exact_count_acc": float(acc["inst_exact"] / max(acc["inst_n"], 1)),
        "instance_merged_rate": float(acc["inst_merged"] / max(acc["inst_n"], 1)),
        "instance_fragmented_rate": float(acc["inst_fragmented"] / max(acc["inst_n"], 1)),
        "instance_mixed_rate": float(acc["inst_mixed"] / max(acc["inst_n"], 1)),
        "instance_mean_matched_iou": float(acc["inst_mean_iou_sum"] / max(acc["inst_n"], 1)),
        "instance_median_matched_iou": float(np.median(np.asarray(acc["inst_median_iou_list"], dtype=np.float32))) if acc["inst_median_iou_list"] else None,
        "instance_perfect_rate": float(acc["inst_perfect"] / max(acc["inst_n"], 1)),
        "center_prob_mean_pos": float(acc["prob_pos_sum"] / max(acc["prob_pos_n"], 1)),
        "center_prob_mean_near": float(acc["prob_near_sum"] / max(acc["prob_near_n"], 1)),
        "center_prob_mean_far": float(acc["prob_far_sum"] / max(acc["prob_far_n"], 1)),
        "center_prob_mean_max": float(acc["prob_max_sum"] / max(acc["prob_max_n"], 1)),
        "failure_modes": dict(acc["failure_modes"]),
    }
    metrics["instance_score"] = _instance_score(metrics)
    return metrics


def _evaluate_dense_entries(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    amp_enabled: bool,
    entries: list[dict],
    tap_lookup: dict[str, TapSpec],
    thresholds: list[float],
    instance_root: Path,
) -> dict[str, dict]:
    needed_taps = sorted(
        {
            path
            for e in entries
            for path in ([e["tap_path"]] if e["kind"] == "single" else [e["tap_path_a"], e["tap_path_b"]])
            if path
        }
    )
    hook_mgr = FeatureHookManager(model, needed_taps)
    out = {}
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for entry in entries:
        out[entry["name"]] = {
            "rows": {float(thr): _init_eval_acc() for thr in thresholds},
            "metadata": {},
        }
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].detach().cpu().numpy().astype(np.uint8)
        centers = batch["center"].detach().cpu().numpy().astype(np.float32)
        image_paths = batch.get("image_path", [])
        meta_paths = batch.get("metadata_path", [])
        hook_mgr.reset_outputs()
        with torch.no_grad():
            with _autocast_ctx(device, enabled=amp_enabled):
                sem_logits, _decoder_output = model.forward_base(images)
        pred_sem = torch.argmax(sem_logits, dim=1).detach().cpu().numpy().astype(np.uint8)
        dense_probs = {}
        for entry in entries:
            if entry["kind"] == "single":
                fmap = hook_mgr.outputs[entry["tap_path"]]
                logits = _probe_logits_single(fmap, entry["probe"], entry["channel_stats"], out_hw=(images.shape[-2], images.shape[-1]))
                dense_probs[entry["name"]] = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)[:, 0]
            else:
                fmap_a = hook_mgr.outputs[entry["tap_path_a"]]
                fmap_b = hook_mgr.outputs[entry["tap_path_b"]]
                logits, coef_info = _probe_logits_fusion(
                    fmap_a,
                    fmap_b,
                    entry["probe"],
                    entry["channel_stats_a"],
                    entry["channel_stats_b"],
                    out_hw=(images.shape[-2], images.shape[-1]),
                )
                dense_probs[entry["name"]] = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)[:, 0]
                out[entry["name"]]["metadata"] = coef_info
        for bi in range(int(images.shape[0])):
            sid = Path(str(image_paths[bi])).stem if image_paths else f"sample_{bi}"
            meta_p = str(meta_paths[bi]) if meta_paths else ""
            gt_pts = _extract_metadata_centers(meta_p) if meta_p else []
            gt_mask = masks[bi]
            gt_center_map = centers[bi, 0]
            gt_inst_path = (instance_root / "instance_masks" / f"{sid}.png").resolve()
            gt_inst = _read_u8(gt_inst_path)
            if gt_inst.shape != pred_sem[bi].shape:
                gh, gw = gt_inst.shape[:2]
                h, w = pred_sem[bi].shape[:2]
                y0 = (gh - h) // 2
                x0 = (gw - w) // 2
                gt_inst = gt_inst[y0 : y0 + h, x0 : x0 + w]
            gt_k = int(len([k for k in [1, 2, 3] if int(np.sum(gt_inst == k)) > 0]))
            for entry in entries:
                pr_center_map = dense_probs[entry["name"]][bi]
                leaf_union = pred_sem[bi] == 1
                pos_exact = gt_center_map >= 0.9999
                near = gt_center_map >= 0.1
                far = gt_center_map < 0.1
                for thr in thresholds:
                    acc = out[entry["name"]]["rows"][float(thr)]
                    pred_pts = _markers_from_center_map(pr_center_map, leaf_union, float(thr), max_markers=3)
                    pred_yx = [(int(y), int(x)) for y, x, _score in pred_pts]
                    tpi, fpi, fni, matches = _match_centers(pred_yx, gt_pts, max_dist_px=16.0)
                    acc["tp"] += int(tpi)
                    acc["fp"] += int(fpi)
                    acc["fn"] += int(fni)
                    for _py, _px, _gy, _gx, d in matches:
                        acc["loc_err_sum"] += float(d)
                        acc["loc_err_n"] += 1
                    if len(gt_pts) > 0:
                        acc["count_acc_n"] += 1
                        if len(pred_pts) == len(gt_pts):
                            acc["count_acc_ok"] += 1
                    gt_leaf_union = gt_mask == 1
                    if bool(np.any(gt_leaf_union)):
                        pos_frac = float(np.mean((pr_center_map[gt_leaf_union] >= float(thr)).astype(np.float32)))
                    else:
                        pos_frac = float(np.mean((pr_center_map >= float(thr)).astype(np.float32)))
                    acc["center_pos_frac_sum"] += float(pos_frac)
                    acc["center_pos_frac_n"] += 1
                    acc["pred_count_sum"] += float(len(pred_pts))
                    acc["gt_count_sum"] += float(len(gt_pts))
                    if len(pred_pts) == 0:
                        acc["zero_center_cases"] += 1
                    if len(pred_pts) > 3:
                        acc["extra_center_cases"] += 1
                    if bool(np.any(pos_exact)):
                        acc["prob_pos_sum"] += float(np.mean(pr_center_map[pos_exact]))
                        acc["prob_pos_n"] += 1
                    if bool(np.any(near)):
                        acc["prob_near_sum"] += float(np.mean(pr_center_map[near]))
                        acc["prob_near_n"] += 1
                    if bool(np.any(far)):
                        acc["prob_far_sum"] += float(np.mean(pr_center_map[far]))
                        acc["prob_far_n"] += 1
                    acc["prob_max_sum"] += float(np.max(pr_center_map))
                    acc["prob_max_n"] += 1

                    if gt_k <= 0:
                        continue
                    pred_inst, pred_k, pred_pts_recon = _reconstruct_instances(pred_sem[bi], pr_center_map, float(thr))
                    case = _case_type(gt_k, pred_k)
                    acc["inst_n"] += 1
                    acc["inst_exact"] += int(pred_k == gt_k)
                    acc["inst_merged"] += int(case == "merged")
                    acc["inst_fragmented"] += int(case == "fragmented")
                    acc["inst_mixed"] += int(case == "mixed")
                    iou_mat = _iou_matrix(gt_inst, pred_inst, gt_k, pred_k)
                    sum_iou = _best_perm_sum(iou_mat)
                    mean_iou = float(sum_iou / max(gt_k, 1))
                    acc["inst_mean_iou_sum"] += float(mean_iou)
                    acc["inst_median_iou_list"].append(float(mean_iou))
                    acc["inst_perfect"] += int((pred_k == gt_k) and (mean_iou >= 0.90))
                    acc["failure_modes"].update(
                        _sample_failure_modes(
                            gt_inst=gt_inst,
                            pred_pts=pred_pts_recon,
                            gt_pts=gt_pts,
                            pred_k=pred_k,
                            mean_iou=mean_iou,
                            tp=tpi,
                        )
                    )
    hook_mgr.close()
    peak_vram_gb = None
    if device.type == "cuda":
        peak_vram_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    final = {}
    for name, payload in out.items():
        rows = []
        for thr in thresholds:
            row = {"threshold": float(thr), **_finalize_eval_acc(payload["rows"][float(thr)])}
            rows.append(row)
        best_center = max(rows, key=lambda r: float(r.get("center_f1") or 0.0))
        best_instance = max(rows, key=lambda r: float(r.get("instance_score") or 0.0))
        final[name] = {
            "rows": rows,
            "best_center": best_center,
            "best_instance": best_instance,
            "metadata": payload["metadata"],
            "wall_time_sec": float(time.perf_counter() - t0),
            "peak_vram_gb": peak_vram_gb,
        }
    return final


def _threshold_robustness(rows: list[dict]) -> dict:
    if not rows:
        return {"best_threshold": None, "thresholds_within_90pct": [], "stable": False}
    best = max(rows, key=lambda r: float(r.get("center_f1") or 0.0))
    best_f1 = float(best.get("center_f1") or 0.0)
    selected = [r for r in rows if float(r.get("center_f1") or 0.0) >= 0.9 * best_f1] if best_f1 > 0 else []
    selected = sorted(selected, key=lambda r: float(r.get("threshold") or 0.0))
    adjacent_pairs = []
    for i in range(len(selected) - 1):
        a = selected[i]
        b = selected[i + 1]
        ca = float(a.get("center_pred_count_mean") or 0.0)
        cb = float(b.get("center_pred_count_mean") or 0.0)
        collapse_ok = True if ca <= 1e-9 else (abs(cb - ca) / max(ca, 1e-9) <= 0.30)
        adjacent_pairs.append(
            {
                "threshold_a": float(a.get("threshold") or 0.0),
                "threshold_b": float(b.get("threshold") or 0.0),
                "count_change_frac": (abs(cb - ca) / max(ca, 1e-9)) if ca > 1e-9 else None,
                "instance_score_change": float((b.get("instance_score") or 0.0) - (a.get("instance_score") or 0.0)),
                "collapse_ok": bool(collapse_ok),
            }
        )
    stable = any(bool(p["collapse_ok"]) for p in adjacent_pairs)
    return {
        "best_threshold": float(best.get("threshold") or 0.0),
        "best_f1": best_f1,
        "thresholds_within_90pct": [float(r.get("threshold") or 0.0) for r in selected],
        "count_within_90pct": int(len(selected)),
        "adjacent_pairs": adjacent_pairs,
        "stable": bool(stable and len(selected) >= 2),
    }


def _aggregate_single_tap_results(
    *,
    tap_specs: list[TapSpec],
    classification_results: list[dict],
    dense_results: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    by_tap = defaultdict(list)
    for row in classification_results:
        name = str(row["tap_path"])
        dense = dense_results[row["entry_name"]]
        best_center = dense["best_center"]
        best_instance = dense["best_instance"]
        robustness = _threshold_robustness(dense["rows"])
        by_tap[name].append({**row, "best_center": best_center, "best_instance": best_instance, "threshold_robustness": robustness})
    per_seed_rows = []
    aggregate_rows = []
    tap_meta = {t.actual_path: t for t in tap_specs if t.actual_path}
    for tap_path, rows in by_tap.items():
        for r in rows:
            per_seed_rows.append(
                {
                    "tap_path": tap_path,
                    "seed": int(r["seed"]),
                    "role": tap_meta[tap_path].role if tap_path in tap_meta else "",
                    "probe_parameters": int(r["probe_train"]["probe_parameter_count"]),
                    "roc_auc": r["classification"]["roc_auc"],
                    "average_precision": r["classification"]["average_precision"],
                    "pos_probability": r["classification"]["pos_probability_mean"],
                    "near_probability": r["classification"]["near_probability_mean"],
                    "far_probability": r["classification"]["far_probability_mean"],
                    "pos_minus_far": r["classification"]["pos_minus_far"],
                    "best_threshold": r["best_center"]["threshold"],
                    "precision": r["best_center"]["center_precision"],
                    "recall": r["best_center"]["center_recall"],
                    "center_f1": r["best_center"]["center_f1"],
                    "count_accuracy": r["best_center"]["center_count_acc"],
                    "localization_error": r["best_center"]["center_loc_err_px"],
                    "instance_score": r["best_instance"]["instance_score"],
                    "matched_iou": r["best_instance"]["instance_mean_matched_iou"],
                    "several_peaks_inside_one_leaflet": (r["best_center"].get("failure_modes") or {}).get("several_peaks_inside_one_leaflet"),
                    "fragmented_rate": r["best_instance"]["instance_fragmented_rate"],
                    "thresholds_within_90pct": len(r["threshold_robustness"]["thresholds_within_90pct"]),
                    "stable": r["threshold_robustness"]["stable"],
                }
            )
        aucs = [r["classification"]["roc_auc"] for r in rows]
        aps = [r["classification"]["average_precision"] for r in rows]
        f1s = [r["best_center"]["center_f1"] for r in rows]
        rank_row = {
            "tap_path": tap_path,
            "role": tap_meta[tap_path].role if tap_path in tap_meta else "",
            "feature_shape": tap_meta[tap_path].output_shape if tap_path in tap_meta else None,
            "channels": tap_meta[tap_path].channels if tap_path in tap_meta else None,
            "probe_parameters": int(rows[0]["probe_train"]["probe_parameter_count"]) if rows else None,
            "roc_auc_mean": _safe_mean(aucs),
            "roc_auc_std": _safe_std(aucs),
            "average_precision_mean": _safe_mean(aps),
            "average_precision_std": _safe_std(aps),
            "pos_probability_mean": _safe_mean([r["classification"]["pos_probability_mean"] for r in rows]),
            "near_probability_mean": _safe_mean([r["classification"]["near_probability_mean"] for r in rows]),
            "far_probability_mean": _safe_mean([r["classification"]["far_probability_mean"] for r in rows]),
            "pos_minus_far_mean": _safe_mean([r["classification"]["pos_minus_far"] for r in rows]),
            "best_threshold_mean": _safe_mean([r["best_center"]["threshold"] for r in rows]),
            "precision_mean": _safe_mean([r["best_center"]["center_precision"] for r in rows]),
            "recall_mean": _safe_mean([r["best_center"]["center_recall"] for r in rows]),
            "center_f1_mean": _safe_mean(f1s),
            "center_f1_std": _safe_std(f1s),
            "count_accuracy_mean": _safe_mean([r["best_center"]["center_count_acc"] for r in rows]),
            "localization_error_mean": _safe_mean([r["best_center"]["center_loc_err_px"] for r in rows]),
            "instance_score_mean": _safe_mean([r["best_instance"]["instance_score"] for r in rows]),
            "matched_iou_mean": _safe_mean([r["best_instance"]["instance_mean_matched_iou"] for r in rows]),
            "several_peaks_inside_one_leaflet_mean": _safe_mean([(r["best_center"].get("failure_modes") or {}).get("several_peaks_inside_one_leaflet") for r in rows]),
            "fragmented_rate_mean": _safe_mean([r["best_instance"]["instance_fragmented_rate"] for r in rows]),
            "thresholds_within_90pct_mean": _safe_mean([len(r["threshold_robustness"]["thresholds_within_90pct"]) for r in rows]),
            "stable_any": any(bool(r["threshold_robustness"]["stable"]) for r in rows),
            "best_seed": int(max(rows, key=lambda r: float(r["best_center"]["center_f1"] or 0.0))["seed"]),
            "worst_seed": int(min(rows, key=lambda r: float(r["best_center"]["center_f1"] or 0.0))["seed"]),
        }
        aggregate_rows.append(rank_row)
    aggregate_rows.sort(
        key=lambda r: (
            float(r.get("center_f1_mean") or -1.0),
            float(r.get("average_precision_mean") or -1.0),
            float(r.get("instance_score_mean") or -1.0),
            float(r.get("thresholds_within_90pct_mean") or -1.0),
        ),
        reverse=True,
    )
    return per_seed_rows, aggregate_rows


def _pick_top_two_taps(aggregate_rows: list[dict]) -> list[str]:
    ranked = sorted(
        aggregate_rows,
        key=lambda r: (
            float(r.get("center_f1_mean") or -1.0),
            float(r.get("average_precision_mean") or -1.0),
            float(r.get("instance_score_mean") or -1.0),
            float(r.get("thresholds_within_90pct_mean") or -1.0),
        ),
        reverse=True,
    )
    return [str(r["tap_path"]) for r in ranked[:2]]


def _prepare_visual_review(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    amp_enabled: bool,
    representative_entries: dict[str, dict],
    tap_specs: dict[str, TapSpec],
    out_dir: Path,
    visual_samples: int,
    instance_root: Path,
) -> dict:
    needed_taps = sorted(
        {
            path
            for entry in representative_entries.values()
            if entry is not None
            for path in ([entry["tap_path"]] if entry["kind"] == "single" else [entry["tap_path_a"], entry["tap_path_b"]])
        }
    )
    hook_mgr = FeatureHookManager(model, needed_taps)
    sample_summaries = []
    special_examples = {}
    saved = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        hook_mgr.reset_outputs()
        with torch.no_grad():
            with _autocast_ctx(device, enabled=amp_enabled):
                sem_logits, _decoder_output = model.forward_base(images)
        for bi in range(int(images.shape[0])):
            if saved >= int(visual_samples):
                break
            sid = Path(str(batch["image_path"][bi])).stem
            img_f = batch["image"].detach().cpu().numpy()[bi].transpose(1, 2, 0)
            img_u8 = (np.clip(img_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            gt_center = batch["center"].detach().cpu().numpy()[bi, 0].astype(np.float32)
            gt_center_u16 = (np.clip(gt_center, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
            gt_inst = _read_u8((instance_root / "instance_masks" / f"{sid}.png").resolve())
            pred_sem = torch.argmax(sem_logits[bi : bi + 1], dim=1).detach().cpu().numpy()[0].astype(np.uint8)
            if gt_inst.shape != pred_sem.shape:
                gh, gw = gt_inst.shape[:2]
                h, w = pred_sem.shape[:2]
                y0 = (gh - h) // 2
                x0 = (gw - w) // 2
                gt_inst = gt_inst[y0 : y0 + h, x0 : x0 + w]
            sample_dir = out_dir / sid
            sample_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sample_dir / "original.png"), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(sample_dir / "gt_center.png"), gt_center_u16)
            cv2.imwrite(str(sample_dir / "gt_instances.png"), gt_inst.astype(np.uint8))

            per_view = {}
            for label, entry in representative_entries.items():
                if entry is None:
                    continue
                if entry["kind"] == "single":
                    fmap = hook_mgr.outputs[entry["tap_path"]][bi : bi + 1]
                    logits = _probe_logits_single(fmap, entry["probe"], entry["channel_stats"], out_hw=(images.shape[-2], images.shape[-1]))
                else:
                    fmap_a = hook_mgr.outputs[entry["tap_path_a"]][bi : bi + 1]
                    fmap_b = hook_mgr.outputs[entry["tap_path_b"]][bi : bi + 1]
                    logits, _coef = _probe_logits_fusion(
                        fmap_a,
                        fmap_b,
                        entry["probe"],
                        entry["channel_stats_a"],
                        entry["channel_stats_b"],
                        out_hw=(images.shape[-2], images.shape[-1]),
                    )
                prob = torch.sigmoid(logits).detach().cpu().numpy()[0, 0].astype(np.float32)
                pred_inst, pred_k, pred_pts = _reconstruct_instances(pred_sem, prob, float(entry["best_threshold"]))
                pred_u16 = (np.clip(prob, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
                binary_u8 = ((prob >= float(entry["best_threshold"])).astype(np.uint8) * 255)
                markers_vis = cv2.cvtColor(img_u8.copy(), cv2.COLOR_RGB2BGR)
                for idx, (y, x, score) in enumerate(pred_pts, start=1):
                    cv2.circle(markers_vis, (int(x), int(y)), 6, (255, 0, 0), 2)
                    cv2.putText(markers_vis, f"{idx}:{score:.2f}", (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.imwrite(str(sample_dir / f"{label}_prob.png"), pred_u16)
                cv2.imwrite(str(sample_dir / f"{label}_markers.png"), markers_vis)
                cv2.imwrite(str(sample_dir / f"{label}_reconstructed_instances.png"), pred_inst.astype(np.uint8))
                _save_compare(sample_dir / f"{label}_compare.png", img_u8, gt_center_u16, pred_u16, gt_inst, pred_inst, binary_u8)
                per_view[label] = {
                    "pred_center_count": len(pred_pts),
                    "pred_instance_count": int(pred_k),
                }
            panel_parts = [cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR), cv2.applyColorMap(((gt_center.astype(np.float32) * 255.0) + 0.5).astype(np.uint8), cv2.COLORMAP_VIRIDIS)]
            for label in ["final_tap", "best_intermediate_tap", "best_deeper_tap", "fusion"]:
                p = sample_dir / f"{label}_prob.png"
                if p.exists():
                    arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    if arr is not None and arr.ndim == 2:
                        arr = cv2.applyColorMap(((arr.astype(np.float32) / 65535.0) * 255.0 + 0.5).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                        panel_parts.append(arr)
            grid = np.concatenate(panel_parts, axis=1)
            cv2.imwrite(str(sample_dir / "comparison_panel.png"), grid)
            sample_summaries.append({"sample": sid, "views": per_view})
            saved += 1
            final_v = per_view.get("final_tap", {})
            deeper_v = per_view.get("best_deeper_tap", {})
            fusion_v = per_view.get("fusion", {})
            if "both_extra_peaks" not in special_examples and final_v.get("pred_center_count", 0) > 2 and deeper_v.get("pred_center_count", 0) > 2:
                special_examples["both_extra_peaks"] = sid
            if "fusion_improves" not in special_examples and fusion_v.get("pred_instance_count", 0) > 0 and final_v.get("pred_instance_count", 0) > 0 and fusion_v["pred_instance_count"] != final_v["pred_instance_count"]:
                special_examples["fusion_improves"] = sid
        if saved >= int(visual_samples):
            break
    hook_mgr.close()
    return {"saved_samples": saved, "special_examples": special_examples, "samples": sample_summaries}


def _save_probe_weight(path: Path, *, probe: LinearProbe, stats: ChannelStats, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "probe_state_dict": probe.state_dict(),
            "channel_mean": stats.mean,
            "channel_std": stats.std,
            "meta": meta,
        },
        path,
    )


def _run_smoke(
    *,
    cfg: dict,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    amp_enabled: bool,
    taps: list[TapSpec],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    base_param_ref = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    bn_ref = _collect_bn_stats(model)
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    hook_mgr = FeatureHookManager(model, [t.actual_path for t in taps if t.actual_path and t.participates_in_forward])
    smoke = {"taps": [t.__dict__ for t in taps]}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        hook_mgr.reset_outputs()
        with _autocast_ctx(device, enabled=amp_enabled):
            sem_logits_train, _ = model.forward_base(train_batch["image"].to(device))
            sem_logits_val, _ = model.forward_base(val_batch["image"].to(device))
    smoke["hook_calls"] = dict(hook_mgr.calls)
    seeds = [1337]
    train_data, train_stats = _collect_sampled_vectors(model=model, loader=[train_batch], device=device, amp_enabled=amp_enabled, taps=taps, seeds=seeds, split="train")
    val_data, val_stats = _collect_sampled_vectors(model=model, loader=[val_batch], device=device, amp_enabled=amp_enabled, taps=taps, seeds=seeds, split="val")
    smoke["train_sampling"] = train_stats
    smoke["val_sampling"] = val_stats
    tap_results = []
    dense_entries = []
    for tap in taps:
        if not tap.actual_path or not tap.participates_in_forward:
            continue
        store_tr = train_data["by_tap_seed"][tap.actual_path][1337]
        store_va = val_data["by_tap_seed"][tap.actual_path][1337]
        train_pos = _cat_vectors(store_tr["train_pos"], int(tap.channels or 0))
        train_far = _cat_vectors(store_tr["train_far"], int(tap.channels or 0))
        val_pos = _cat_vectors(store_va["val_pos"], int(tap.channels or 0))
        val_near = _cat_vectors(store_va["val_near"], int(tap.channels or 0))
        val_far = _cat_vectors(store_va["val_far"], int(tap.channels or 0))
        if train_pos.shape[0] == 0 or train_far.shape[0] == 0:
            continue
        stats = _compute_channel_stats(train_pos, train_far)
        probe, train_info = _train_linear_probe(
            train_pos=train_pos,
            train_far=train_far,
            stats=stats,
            seed=1337,
            lr=float((cfg.get("probe") or {}).get("probe_lr", 1e-2)),
            weight_decay=float((cfg.get("probe") or {}).get("probe_weight_decay", 1e-4)),
            steps=int((cfg.get("probe") or {}).get("probe_steps", 1000)),
            smoke_steps=int((cfg.get("probe") or {}).get("smoke_probe_steps", 5)),
        )
        cls = _classification_metrics(probe=probe, val_pos=val_pos, val_near=val_near, val_far=val_far, stats=stats)
        dense_entries.append(
            {
                "name": tap.actual_path,
                "kind": "single",
                "tap_path": tap.actual_path,
                "probe": probe,
                "channel_stats": stats,
                "best_threshold": 0.03,
            }
        )
        tap_results.append({"tap": tap.actual_path, "train": train_info, "classification": cls})
    dense = _evaluate_dense_entries(
        model=model,
        loader=[val_batch],
        device=device,
        amp_enabled=amp_enabled,
        entries=dense_entries,
        tap_lookup={t.actual_path: t for t in taps if t.actual_path},
        thresholds=[0.03],
        instance_root=Path(cfg["dataset"]["instance_root"]).resolve(),
    )
    peak_vram = None
    if device.type == "cuda":
        peak_vram = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    smoke["tap_results"] = tap_results
    smoke["dense"] = dense
    smoke["finite_semantic_logits"] = bool(torch.isfinite(sem_logits_train.detach()).all().item()) and bool(torch.isfinite(sem_logits_val.detach()).all().item())
    smoke["base_parameter_max_delta"] = _max_parameter_delta(model, base_param_ref)
    smoke["bn_stats_max_delta"] = _max_bn_delta(model, bn_ref)
    smoke["peak_vram_gb"] = peak_vram
    hook_mgr.close()
    _json_dump(out_dir / "smoke_summary.json", smoke)
    print(json.dumps(smoke, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("training/configs/unetpp_effb3_decoder_center_feature_probe.yaml"))
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    _self_check_binary_metrics()
    cfg = _read_yaml(args.config.resolve())
    out_dir = _probe_output_dir(cfg, smoke_test=bool(args.smoke_test))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _make_device(cfg)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Configured CUDA device is unavailable; falling back to CPU for this run.")
        device = torch.device("cpu")
    amp_enabled = bool((cfg.get("train") or {}).get("amp", False)) and device.type == "cuda"
    _seed_all(int((cfg.get("probe") or {}).get("seeds", [1337])[0]))

    train_loader, val_loader = _build_probe_loaders(cfg, device=device)
    model = _build_model(cfg).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            m.eval()

    candidates = _candidate_taps(cfg)
    audit_batch = next(iter(val_loader))
    decoder_audit, tap_specs = _run_decoder_audit(model, audit_batch, device=device, amp_enabled=amp_enabled, candidates=candidates)
    _json_dump(out_dir / "decoder_audit.json", decoder_audit)

    actual_taps = [t for t in tap_specs if t.actual_path and t.participates_in_forward]
    roles = {t.role for t in actual_taps}
    if "final" not in roles or "intermediate" not in roles or "deeper" not in roles:
        raise SystemExit(f"Probe requires final/intermediate/deeper taps, got roles={sorted(roles)}")

    if args.smoke_test:
        _run_smoke(
            cfg=cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
            taps=actual_taps,
            out_dir=out_dir,
        )
        return

    seeds = [int(x) for x in (cfg.get("probe") or {}).get("seeds", [1337, 2025, 2026])]
    t_feat0 = time.perf_counter()
    train_data, train_stats = _collect_sampled_vectors(model=model, loader=train_loader, device=device, amp_enabled=amp_enabled, taps=actual_taps, seeds=seeds, split="train")
    val_data, val_stats = _collect_sampled_vectors(model=model, loader=val_loader, device=device, amp_enabled=amp_enabled, taps=actual_taps, seeds=seeds, split="val")
    feature_wall_time = float(time.perf_counter() - t_feat0)

    sampled_stats = {
        "train": train_stats,
        "val": val_stats,
        "feature_extraction_wall_time_sec": feature_wall_time,
        "taps": {},
    }

    thresholds = _thresholds(cfg)
    probe_lr = float((cfg.get("probe") or {}).get("probe_lr", 1e-2))
    probe_wd = float((cfg.get("probe") or {}).get("probe_weight_decay", 1e-4))
    probe_steps = int((cfg.get("probe") or {}).get("probe_steps", 1000))
    classification_results = []
    dense_single_entries = []
    t_train0 = time.perf_counter()
    for tap in actual_taps:
        sampled_stats["taps"][tap.actual_path] = {}
        for seed in seeds:
            store_tr = train_data["by_tap_seed"][tap.actual_path][int(seed)]
            store_va = val_data["by_tap_seed"][tap.actual_path][int(seed)]
            train_pos = _cat_vectors(store_tr["train_pos"], int(tap.channels or 0))
            train_far = _cat_vectors(store_tr["train_far"], int(tap.channels or 0))
            val_pos = _cat_vectors(store_va["val_pos"], int(tap.channels or 0))
            val_near = _cat_vectors(store_va["val_near"], int(tap.channels or 0))
            val_far = _cat_vectors(store_va["val_far"], int(tap.channels or 0))
            if train_pos.shape[0] == 0 or train_far.shape[0] == 0:
                raise SystemExit(f"No train samples for tap={tap.actual_path} seed={seed}")
            stats = _compute_channel_stats(train_pos, train_far)
            sampled_stats["taps"][tap.actual_path][str(seed)] = {
                "channels": int(tap.channels or 0),
                "train_positive_vectors": int(train_pos.shape[0]),
                "train_far_vectors": int(train_far.shape[0]),
                "val_positive_vectors": int(val_pos.shape[0]),
                "val_near_vectors": int(val_near.shape[0]),
                "val_far_vectors": int(val_far.shape[0]),
                "channel_mean": stats.mean.tolist(),
                "channel_std": stats.std.tolist(),
                "zero_near_zero_variance_channels": int(stats.zero_var_channels),
                "feature_norm_mean": float(stats.feature_norm_mean),
                "feature_norm_std": float(stats.feature_norm_std),
                "feature_norm_min": float(stats.feature_norm_min),
                "feature_norm_max": float(stats.feature_norm_max),
            }
            probe, train_info = _train_linear_probe(
                train_pos=train_pos,
                train_far=train_far,
                stats=stats,
                seed=int(seed),
                lr=probe_lr,
                weight_decay=probe_wd,
                steps=probe_steps,
            )
            class_info = _classification_metrics(probe=probe, val_pos=val_pos, val_near=val_near, val_far=val_far, stats=stats)
            entry_name = f"{tap.actual_path}__seed{seed}"
            dense_single_entries.append(
                {
                    "name": entry_name,
                    "kind": "single",
                    "tap_path": tap.actual_path,
                    "probe": probe,
                    "channel_stats": stats,
                    "best_threshold": None,
                }
            )
            _save_probe_weight(
                out_dir / "probe_weights" / f"{entry_name}.pt",
                probe=probe,
                stats=stats,
                meta={"tap_path": tap.actual_path, "seed": int(seed), "role": tap.role},
            )
            classification_results.append(
                {
                    "entry_name": entry_name,
                    "tap_path": tap.actual_path,
                    "role": tap.role,
                    "seed": int(seed),
                    "probe_train": train_info,
                    "classification": class_info,
                }
            )
    train_wall_time = float(time.perf_counter() - t_train0)
    _json_dump(out_dir / "sampled_feature_statistics.json", sampled_stats)

    dense_single = _evaluate_dense_entries(
        model=model,
        loader=val_loader,
        device=device,
        amp_enabled=amp_enabled,
        entries=dense_single_entries,
        tap_lookup={t.actual_path: t for t in actual_taps if t.actual_path},
        thresholds=thresholds,
        instance_root=Path(cfg["dataset"]["instance_root"]).resolve(),
    )

    for entry in dense_single_entries:
        _json_dump(out_dir / "threshold_sweeps" / f"{entry['name']}.json", dense_single[entry["name"]])

    single_seed_rows, single_aggregate_rows = _aggregate_single_tap_results(
        tap_specs=actual_taps,
        classification_results=classification_results,
        dense_results=dense_single,
    )
    _write_csv(out_dir / "single_tap_probe_results.csv", single_seed_rows)
    _json_dump(
        out_dir / "single_tap_probe_results.json",
        {
            "per_seed": single_seed_rows,
            "aggregate": single_aggregate_rows,
            "training_wall_time_sec": train_wall_time,
            "validation_wall_time_sec": _safe_mean([dense_single[n]["wall_time_sec"] for n in dense_single]),
        },
    )

    fusion_rows = []
    fusion_aggregate_rows = []
    fusion_dense = {}
    fusion_entries = []
    top_two = _pick_top_two_taps(single_aggregate_rows)
    if len(top_two) >= 2:
        tap_a, tap_b = top_two[:2]
        for seed in seeds:
            tap_a_meta = next(t for t in actual_taps if t.actual_path == tap_a)
            tap_b_meta = next(t for t in actual_taps if t.actual_path == tap_b)
            tr_a = train_data["by_tap_seed"][tap_a][int(seed)]
            tr_b = train_data["by_tap_seed"][tap_b][int(seed)]
            va_a = val_data["by_tap_seed"][tap_a][int(seed)]
            va_b = val_data["by_tap_seed"][tap_b][int(seed)]
            xpa = _cat_vectors(tr_a["train_pos"], int(tap_a_meta.channels or 0))
            xfa = _cat_vectors(tr_a["train_far"], int(tap_a_meta.channels or 0))
            xpb = _cat_vectors(tr_b["train_pos"], int(tap_b_meta.channels or 0))
            xfb = _cat_vectors(tr_b["train_far"], int(tap_b_meta.channels or 0))
            stats_a = _compute_channel_stats(xpa, xfa)
            stats_b = _compute_channel_stats(xpb, xfb)
            train_pos = torch.cat([_standardize(xpa, stats_a), _standardize(xpb, stats_b)], dim=1)
            train_far = torch.cat([_standardize(xfa, stats_a), _standardize(xfb, stats_b)], dim=1)
            # Re-train directly on concatenated standardized vectors with a dedicated probe.
            _seed_all(int(seed))
            fusion_probe = LinearProbe(int(train_pos.shape[1]))
            for p in fusion_probe.parameters():
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
            optimizer = torch.optim.AdamW(fusion_probe.parameters(), lr=probe_lr, weight_decay=probe_wd)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            y = torch.cat([torch.ones(train_pos.shape[0]), torch.zeros(train_far.shape[0])], dim=0)
            x = torch.cat([train_pos, train_far], dim=0)
            grad_finite_all = True
            loss_last = None
            for _ in range(probe_steps):
                optimizer.zero_grad(set_to_none=True)
                logits = fusion_probe(x)
                loss = loss_fn(logits, y)
                loss.backward()
                grads = [p.grad.detach() for p in fusion_probe.parameters() if p.grad is not None]
                grad_finite_all = bool(grad_finite_all and all(bool(torch.isfinite(g).all().item()) for g in grads))
                optimizer.step()
                loss_last = float(loss.item())
            val_pos = torch.cat(
                [
                    _standardize(_cat_vectors(va_a["val_pos"], int(tap_a_meta.channels or 0)), stats_a),
                    _standardize(_cat_vectors(va_b["val_pos"], int(tap_b_meta.channels or 0)), stats_b),
                ],
                dim=1,
            )
            val_near = torch.cat(
                [
                    _standardize(_cat_vectors(va_a["val_near"], int(tap_a_meta.channels or 0)), stats_a),
                    _standardize(_cat_vectors(va_b["val_near"], int(tap_b_meta.channels or 0)), stats_b),
                ],
                dim=1,
            )
            val_far = torch.cat(
                [
                    _standardize(_cat_vectors(va_a["val_far"], int(tap_a_meta.channels or 0)), stats_a),
                    _standardize(_cat_vectors(va_b["val_far"], int(tap_b_meta.channels or 0)), stats_b),
                ],
                dim=1,
            )
            with torch.no_grad():
                pos_prob = torch.sigmoid(fusion_probe(val_pos)).cpu().numpy().astype(np.float64)
                near_prob = torch.sigmoid(fusion_probe(val_near)).cpu().numpy().astype(np.float64)
                far_prob = torch.sigmoid(fusion_probe(val_far)).cpu().numpy().astype(np.float64)
            labels = np.concatenate([np.ones_like(pos_prob, dtype=np.int64), np.zeros_like(far_prob, dtype=np.int64)])
            scores = np.concatenate([pos_prob, far_prob])
            class_info = {
                "roc_auc": _binary_roc_auc(labels, scores) if scores.size else None,
                "average_precision": _binary_average_precision(labels, scores) if scores.size else None,
                "balanced_accuracy": _binary_balanced_accuracy(labels, scores) if scores.size else None,
                "pos_probability_mean": float(np.mean(pos_prob)) if pos_prob.size else None,
                "near_probability_mean": float(np.mean(near_prob)) if near_prob.size else None,
                "far_probability_mean": float(np.mean(far_prob)) if far_prob.size else None,
                "pos_minus_far": (float(np.mean(pos_prob)) - float(np.mean(far_prob))) if pos_prob.size and far_prob.size else None,
            }
            entry_name = f"fusion__{tap_a}__{tap_b}__seed{seed}"
            fusion_entries.append(
                {
                    "name": entry_name,
                    "kind": "fusion",
                    "tap_path_a": tap_a,
                    "tap_path_b": tap_b,
                    "probe": fusion_probe,
                    "channel_stats_a": stats_a,
                    "channel_stats_b": stats_b,
                    "best_threshold": None,
                }
            )
            _save_probe_weight(
                out_dir / "probe_weights" / f"{entry_name}.pt",
                probe=fusion_probe,
                stats=ChannelStats(
                    mean=torch.cat([stats_a.mean, stats_b.mean]),
                    std=torch.cat([stats_a.std, stats_b.std]),
                    zero_var_channels=int(stats_a.zero_var_channels + stats_b.zero_var_channels),
                    feature_norm_mean=0.0,
                    feature_norm_std=0.0,
                    feature_norm_min=0.0,
                    feature_norm_max=0.0,
                ),
                meta={"tap_path_a": tap_a, "tap_path_b": tap_b, "seed": int(seed), "gradient_finite": bool(grad_finite_all), "train_loss": loss_last},
            )
            fusion_rows.append(
                {
                    "entry_name": entry_name,
                    "seed": int(seed),
                    "taps": [tap_a, tap_b],
                    "probe_parameter_count": int(sum(int(p.numel()) for p in fusion_probe.parameters())),
                    "classification": class_info,
                    "train_loss": loss_last,
                    "gradient_finite": bool(grad_finite_all),
                }
            )
        fusion_dense = _evaluate_dense_entries(
            model=model,
            loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
            entries=fusion_entries,
            tap_lookup={t.actual_path: t for t in actual_taps if t.actual_path},
            thresholds=thresholds,
            instance_root=Path(cfg["dataset"]["instance_root"]).resolve(),
        )
        for entry in fusion_entries:
            _json_dump(out_dir / "threshold_sweeps" / f"{entry['name']}.json", fusion_dense[entry["name"]])
        for row in fusion_rows:
            dense = fusion_dense[row["entry_name"]]
            row["best_center"] = dense["best_center"]
            row["best_instance"] = dense["best_instance"]
            row["threshold_robustness"] = _threshold_robustness(dense["rows"])
            row["coefficient_norm_per_tap"] = dense.get("metadata", {})
        _write_csv(
            out_dir / "fusion_probe_results.csv",
            [
                {
                    "entry_name": r["entry_name"],
                    "seed": r["seed"],
                    "tap_a": r["taps"][0],
                    "tap_b": r["taps"][1],
                    "roc_auc": r["classification"]["roc_auc"],
                    "average_precision": r["classification"]["average_precision"],
                    "best_threshold": r["best_center"]["threshold"],
                    "center_f1": r["best_center"]["center_f1"],
                    "instance_score": r["best_instance"]["instance_score"],
                    "matched_iou": r["best_instance"]["instance_mean_matched_iou"],
                    "stable": r["threshold_robustness"]["stable"],
                }
                for r in fusion_rows
            ],
        )
        if fusion_rows:
            fusion_aggregate_rows.append(
                {
                    "taps": top_two,
                    "center_f1_mean": _safe_mean([r["best_center"]["center_f1"] for r in fusion_rows]),
                    "center_f1_std": _safe_std([r["best_center"]["center_f1"] for r in fusion_rows]),
                    "instance_score_mean": _safe_mean([r["best_instance"]["instance_score"] for r in fusion_rows]),
                    "matched_iou_mean": _safe_mean([r["best_instance"]["instance_mean_matched_iou"] for r in fusion_rows]),
                    "gain_over_best_single_f1": _safe_mean([r["best_center"]["center_f1"] for r in fusion_rows]) - float(single_aggregate_rows[0]["center_f1_mean"]),
                    "gain_over_best_single_instance": _safe_mean([r["best_instance"]["instance_score"] for r in fusion_rows]) - float(single_aggregate_rows[0]["instance_score_mean"]),
                    "threshold_robustness": {
                        "stable_any": any(bool(r["threshold_robustness"]["stable"]) for r in fusion_rows),
                        "thresholds_within_90pct_mean": _safe_mean([len(r["threshold_robustness"]["thresholds_within_90pct"]) for r in fusion_rows]),
                    },
                    "coefficient_norm_per_tap": {
                        "tap_a_mean": _safe_mean([r["coefficient_norm_per_tap"].get("tap_a_weight_norm") for r in fusion_rows]),
                        "tap_b_mean": _safe_mean([r["coefficient_norm_per_tap"].get("tap_b_weight_norm") for r in fusion_rows]),
                    },
                }
            )
        _json_dump(out_dir / "fusion_probe_results.json", {"per_seed": fusion_rows, "aggregate": fusion_aggregate_rows})
    else:
        _write_csv(out_dir / "fusion_probe_results.csv", [])
        _json_dump(out_dir / "fusion_probe_results.json", {"per_seed": [], "aggregate": []})

    best_single = single_aggregate_rows[0] if single_aggregate_rows else None
    decision = "D"
    next_step = "Пересмотр center target representation и training objective."
    if best_single is not None:
        if str(best_single["tap_path"]).endswith("x_0_4"):
            decision = "A"
            next_step = "Пересмотр center target representation или peak suppression."
        else:
            f1_gain = float(best_single.get("center_f1_mean") or 0.0) - 0.1413428
            inst_gain = float(best_single.get("instance_score_mean") or 0.0) - 0.4052636
            peaks = float(best_single.get("several_peaks_inside_one_leaflet_mean") or 0.0)
            final_row = next((r for r in single_aggregate_rows if str(r["tap_path"]).endswith("x_0_4")), None)
            final_peaks = float(final_row.get("several_peaks_inside_one_leaflet_mean") or 0.0) if final_row is not None else None
            if (f1_gain >= 0.03 or inst_gain >= 0.005) and final_peaks not in [None, 0.0] and peaks <= 0.75 * final_peaks:
                decision = "B"
                next_step = "Подключить spatial center head к deeper tap через минимальный adapter."
            else:
                decision = "D"
                next_step = "Пересмотр center target representation и training objective."
    if fusion_aggregate_rows:
        fus = fusion_aggregate_rows[0]
        if float(fus.get("gain_over_best_single_f1") or 0.0) >= 0.02 or float(fus.get("gain_over_best_single_instance") or 0.0) >= 0.005:
            decision = "C"
            next_step = "Минимальный two-tap fusion adapter."

    representative_entries = {
        "final_tap": None,
        "best_intermediate_tap": None,
        "best_deeper_tap": None,
        "fusion": None,
    }
    dense_entry_lookup = {e["name"]: e for e in dense_single_entries}
    for row in single_aggregate_rows:
        tap_path = str(row["tap_path"])
        role = str(row["role"])
        best_seed = int(row["best_seed"])
        entry_name = f"{tap_path}__seed{best_seed}"
        entry = dense_entry_lookup.get(entry_name)
        if entry is None:
            continue
        meta = dense_single[entry_name]
        prepared = dict(entry)
        prepared["best_threshold"] = float(meta["best_center"]["threshold"])
        if role == "final" and representative_entries["final_tap"] is None:
            representative_entries["final_tap"] = prepared
        if role == "intermediate" and representative_entries["best_intermediate_tap"] is None:
            representative_entries["best_intermediate_tap"] = prepared
        if role == "deeper" and representative_entries["best_deeper_tap"] is None:
            representative_entries["best_deeper_tap"] = prepared
    if fusion_entries and fusion_rows:
        best_fusion_seed = int(max(fusion_rows, key=lambda r: float(r["best_center"]["center_f1"] or 0.0))["seed"])
        fusion_name = f"fusion__{top_two[0]}__{top_two[1]}__seed{best_fusion_seed}"
        if fusion_name in [e["name"] for e in fusion_entries]:
            fusion_entry = next(e for e in fusion_entries if e["name"] == fusion_name)
            representative_entries["fusion"] = {**fusion_entry, "best_threshold": float(fusion_dense[fusion_name]["best_center"]["threshold"])}

    visual_summary = _prepare_visual_review(
        model=model,
        loader=val_loader,
        device=device,
        amp_enabled=amp_enabled,
        representative_entries=representative_entries,
        tap_specs={t.actual_path: t for t in actual_taps if t.actual_path},
        out_dir=out_dir / "visual_review",
        visual_samples=int((cfg.get("probe") or {}).get("visual_samples", 20)),
        instance_root=Path(cfg["dataset"]["instance_root"]).resolve(),
    )

    summary = {
        "decoder_audit": decoder_audit,
        "data": {
            "train_positive_samples": train_stats["sample_counts"],
            "val_sample_counts": val_stats["sample_counts"],
            "leakage_checks": {
                "train_split_only_for_probe_fit": True,
                "val_split_not_used_for_probe_training": True,
                "test_split_not_used": True,
            },
        },
        "single_tap_ranking": single_aggregate_rows,
        "fusion_result": fusion_aggregate_rows,
        "visual_review": visual_summary,
        "baseline_context": {
            "frozen_spatial_fp32": {"center_f1": 0.1376812, "instance_score": 0.4043244, "matched_iou": 0.4826262, "pos_minus_far": 0.0323306},
            "partial_unfreeze_spatial": {"center_f1": 0.1413428, "instance_score": 0.4052636, "matched_iou": 0.4854523, "pos_minus_far": 0.0327889},
            "amp_spatial_frozen": {"center_f1": 0.1588448},
            "epoch0_fallback": {"instance_score": 0.4233875, "matched_iou": 0.4941423},
        },
        "decision": decision,
        "next_step": next_step,
    }
    _json_dump(out_dir / "probe_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
