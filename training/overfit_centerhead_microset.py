from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from augmentations import get_val_augmentations
from dataset import read_split_file
from dataset_centerhead import SegmentationWithCenterDataset
from losses import CenterNetFocalHeatmapLoss
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


def _autocast_ctx(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type="cuda", enabled=True)


def _amp_enabled(cfg: dict, device: torch.device) -> bool:
    return bool((cfg.get("train") or {}).get("amp", False)) and device.type == "cuda"


def _center_fp32_enabled(cfg: dict) -> bool:
    return bool((cfg.get("train") or {}).get("center_fp32", False))


def _dtype_name(x: torch.Tensor | None) -> str | None:
    if x is None:
        return None
    return str(x.dtype).replace("torch.", "")


def _center_bias_init(model: UnetPlusPlusSemanticCenterHead, bias: float) -> None:
    layer0 = model.center_head_output_layer()
    if layer0 is None or not hasattr(layer0, "bias") or layer0.bias is None:
        raise RuntimeError("center head output bias not found for bias init")
    with torch.no_grad():
        layer0.bias.fill_(float(bias))


def _freeze_base(model: UnetPlusPlusSemanticCenterHead) -> None:
    for p in model.base.parameters():
        p.requires_grad = False
    if getattr(model, "center_adapter", None) is not None:
        for p in model.center_adapter.parameters():
            p.requires_grad = True
    for p in model.center_head.parameters():
        p.requires_grad = True
    model.freeze_base = True
    model.base.eval()
    if getattr(model, "center_adapter", None) is not None:
        model.center_adapter.train()
    model.center_head.train()


def _center_feature_cfg_from_cfg(cfg: dict) -> dict | None:
    center_feature = (cfg.get("model") or {}).get("center_feature", None)
    return dict(center_feature) if isinstance(center_feature, dict) else None


def _center_branch_named_params(model: UnetPlusPlusSemanticCenterHead) -> list[tuple[str, torch.nn.Parameter]]:
    out = []
    for n, p in model.named_parameters():
        if not bool(p.requires_grad):
            continue
        if str(n).startswith("center_head.") or str(n).startswith("center_adapter."):
            out.append((n, p))
    return out


def _center_branch_params(model: UnetPlusPlusSemanticCenterHead) -> list[torch.nn.Parameter]:
    return [p for _n, p in _center_branch_named_params(model)]


def _center_adapter_params(model: UnetPlusPlusSemanticCenterHead) -> list[torch.nn.Parameter]:
    if getattr(model, "center_adapter", None) is None:
        return []
    return list(model.center_adapter.parameters())


def _center_head_params(model: UnetPlusPlusSemanticCenterHead) -> list[torch.nn.Parameter]:
    return list(model.center_head.parameters())


def _parameter_count(params: list[torch.nn.Parameter]) -> int:
    return int(sum(int(p.numel()) for p in params))


def _snapshot_named_parameters(named_params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    return {str(n): p.detach().clone() for n, p in named_params}


def _max_parameter_delta(named_params: list[tuple[str, torch.nn.Parameter]], ref: dict[str, torch.Tensor]) -> float:
    max_d = 0.0
    for n, p in named_params:
        old = ref.get(str(n), None)
        if old is None:
            continue
        if p.numel():
            max_d = max(max_d, float((p.detach() - old).abs().max().item()))
    return float(max_d)


def _count_present_grads(params: list[torch.nn.Parameter]) -> int:
    return int(sum(1 for p in params if p.grad is not None))


def _center_branch_prefix_ok(trainable_names: list[str]) -> bool:
    return all(str(n).startswith("center_head.") or str(n).startswith("center_adapter.") for n in trainable_names)


def _build_center_branch_optimizer(model: UnetPlusPlusSemanticCenterHead, *, center_lr: float, adapter_lr: float, weight_decay: float = 0.0) -> tuple[torch.optim.Optimizer, list[dict]]:
    group_specs = []
    adapter_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and str(n).startswith("center_adapter.")]
    head_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and str(n).startswith("center_head.")]
    if adapter_named:
        group_specs.append({"name": "center_adapter", "lr": float(adapter_lr), "named_params": adapter_named})
    if head_named:
        group_specs.append({"name": "center_head", "lr": float(center_lr), "named_params": head_named})
    if not group_specs:
        raise SystemExit("No trainable center-branch parameters found for optimizer")
    seen = set()
    for g in group_specs:
        for n, _p in g["named_params"]:
            if n in seen:
                raise SystemExit(f"Optimizer overlap detected for parameter: {n}")
            seen.add(n)
    frozen_in_optimizer = [n for g in group_specs for n, p in g["named_params"] if not p.requires_grad]
    if frozen_in_optimizer:
        raise SystemExit(f"Frozen parameters included in optimizer: {frozen_in_optimizer[:10]}")
    opt = torch.optim.AdamW(
        [{"params": [p for _n, p in g["named_params"]], "lr": float(g["lr"])} for g in group_specs],
        weight_decay=float(weight_decay),
    )
    meta = [
        {
            "name": g["name"],
            "lr": float(g["lr"]),
            "parameter_count": int(sum(int(p.numel()) for _n, p in g["named_params"])),
            "parameter_names": [n for n, _p in g["named_params"]],
        }
        for g in group_specs
    ]
    return opt, meta


def _upsample_alignment_test(model: UnetPlusPlusSemanticCenterHead, device: torch.device) -> dict:
    stride = max(int(getattr(model, "center_feature_native_stride", 1) or 1), 1)
    if stride <= 1:
        return {
            "native_peak": None,
            "expected_full_coordinate": None,
            "actual_full_coordinate": None,
            "error_px": 0.0,
            "batch_consistent": True,
        }
    native_h = 192
    native_w = 192
    peak_y = 41
    peak_x = 73
    expected_y = int(peak_y * stride + stride // 2)
    expected_x = int(peak_x * stride + stride // 2)
    native = torch.zeros((1, 1, native_h, native_w), dtype=torch.float32, device=device)
    native[0, 0, peak_y, peak_x] = 10.0
    up1 = model.upsample_center_logits(native)
    max_idx1 = int(torch.argmax(up1[0, 0]).item())
    y1 = int(max_idx1 // int(up1.shape[-1]))
    x1 = int(max_idx1 % int(up1.shape[-1]))
    native_b = native.repeat(2, 1, 1, 1)
    upb = model.upsample_center_logits(native_b)
    max_idxb = int(torch.argmax(upb[1, 0]).item())
    yb = int(max_idxb // int(upb.shape[-1]))
    xb = int(max_idxb % int(upb.shape[-1]))
    err = float(np.hypot(float(y1 - expected_y), float(x1 - expected_x)))
    return {
        "native_peak": [int(peak_y), int(peak_x)],
        "expected_full_coordinate": [int(expected_y), int(expected_x)],
        "actual_full_coordinate": [int(y1), int(x1)],
        "error_px": float(err),
        "batch_consistent": bool((y1 == yb) and (x1 == xb)),
    }


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


def _load_existing_microset(dataset_root: Path, microset_txt: Path, out_dir: Path) -> Microset:
    src = microset_txt.resolve()
    if not src.exists():
        raise SystemExit(f"Microset file not found: {src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = (out_dir / "microset.txt").resolve()
    shutil.copyfile(str(src), str(dst))

    items = read_split_file(dataset_root, dst)
    samples = []
    dist: dict[int, int] = {1: 0, 2: 0, 3: 0}
    for it in items:
        sid = Path(it.image_path).stem
        samples.append(sid)
        meta = (dataset_root / "metadata" / f"{sid}.json").resolve()
        if not meta.exists():
            raise SystemExit(f"Metadata not found for microset sample: {sid}")
        obj = json.loads(meta.read_text(encoding="utf-8"))
        k = int(obj.get("instance_count", 0) or 0)
        if k in dist:
            dist[k] += 1
    if len(samples) != 6:
        raise SystemExit(f"Expected exactly 6 microset samples, got {len(samples)}")
    return Microset(split_txt=dst, samples=samples, distribution=dist)


def _safe_sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(x))))


def _grad_l2_norm(params: list[torch.Tensor]) -> float:
    s = 0.0
    for p in params:
        if p.grad is None:
            continue
        s += float(torch.sum(p.grad.detach().float() ** 2).item())
    return float(math.sqrt(max(s, 0.0)))


def _flatten_center_head_grads(params: list[torch.Tensor]) -> torch.Tensor:
    flat = []
    dev = None
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach().float().reshape(-1)
        flat.append(g)
        dev = g.device
    if flat:
        return torch.cat(flat, dim=0)
    return torch.zeros((0,), dtype=torch.float32, device=(dev if dev is not None else torch.device("cpu")))


def _instance_score_from_row(row: dict) -> float | None:
    miou = row.get("instance_mean_matched_iou", None)
    mr = row.get("instance_merged_rate", None)
    fr = row.get("instance_fragmented_rate", None)
    if miou is None or mr is None or fr is None:
        return None
    return float(miou) - 0.25 * float(mr) - 0.15 * float(fr)


def _grad_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if int(a.numel()) == 0 or int(b.numel()) == 0 or int(a.numel()) != int(b.numel()):
        return None
    an = float(torch.norm(a).item())
    bn = float(torch.norm(b).item())
    if an <= 0.0 or bn <= 0.0:
        return None
    return float(torch.dot(a, b).item() / max(an * bn, 1e-12))


def _params_finite(params: list[torch.Tensor]) -> bool:
    for p in params:
        if not bool(torch.isfinite(p.detach()).all().item()):
            return False
    return True


def _count_center_head_batchnorms(model: UnetPlusPlusSemanticCenterHead) -> int:
    n = 0
    for m in model.center_head.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            n += 1
    return int(n)


def _copy_bn_stats(model: torch.nn.Module) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
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
    mods = dict(model.named_modules())
    for name, rm0, rv0 in ref:
        m = mods.get(name, None)
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


def _center_head_output_bias(model: UnetPlusPlusSemanticCenterHead) -> float | None:
    layer = model.center_head_output_layer()
    if layer is None or not hasattr(layer, "bias") or layer.bias is None:
        return None
    return float(layer.bias.detach().mean().item())


def _center_head_weight_norm(model: UnetPlusPlusSemanticCenterHead) -> float | None:
    layer = model.center_head_output_layer()
    if layer is None or not hasattr(layer, "weight") or layer.weight is None:
        return None
    return float(layer.weight.detach().float().norm().item())


def _make_model_from_cfg(cfg: dict, device: torch.device) -> tuple[UnetPlusPlusSemanticCenterHead, float]:
    encoder = cfg["model"].get("encoder") or cfg["model"].get("encoder_name")
    center_head_type = str((cfg.get("model") or {}).get("center_head_type", "linear_1x1")).strip().lower() or "linear_1x1"
    model = UnetPlusPlusSemanticCenterHead(
        encoder_name=str(encoder),
        encoder_weights=cfg["model"].get("encoder_weights", None),
        in_channels=int(cfg["model"]["in_channels"]),
        classes=int(cfg["model"]["classes"]),
        center_head_type=center_head_type,
        center_feature=_center_feature_cfg_from_cfg(cfg),
    )
    init_path = (cfg.get("train") or {}).get("init_checkpoint", None)
    if not init_path:
        raise SystemExit("Config: train.init_checkpoint is required")
    missing, _unexpected = load_semantic_checkpoint_non_strict(model, str(init_path))
    center_from_scratch = bool(any(str(k).startswith("center_head.") or str(k).startswith("center_adapter.") for k in missing))
    if not center_from_scratch:
        raise SystemExit("Expected center_head to be from scratch in micro-overfit setup")
    bias = float((cfg.get("model") or {}).get("center_head_init_bias", -2.19))
    _center_bias_init(model, bias=bias)
    _freeze_base(model)
    model = model.to(device)
    return model, bias


def _make_center_loss_from_cfg(cfg: dict, device: torch.device, *, normalization_mode: str | None = None) -> CenterNetFocalHeatmapLoss:
    focal_cfg = cfg.get("center_loss") or {}
    alpha = float((focal_cfg.get("alpha", 2.0) if isinstance(focal_cfg, dict) else 2.0))
    beta = float((focal_cfg.get("beta", 4.0) if isinstance(focal_cfg, dict) else 4.0))
    norm_mode = normalization_mode
    if norm_mode is None:
        norm_mode = str((focal_cfg.get("normalization_mode", "legacy_num_pos") if isinstance(focal_cfg, dict) else "legacy_num_pos")).strip().lower() or "legacy_num_pos"
    return CenterNetFocalHeatmapLoss(alpha=alpha, beta=beta, normalization_mode=norm_mode).to(device)


def _forward_frozen_base(
    *,
    model: UnetPlusPlusSemanticCenterHead,
    images: torch.Tensor,
    device: torch.device,
    amp_enabled_global: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        with _autocast_ctx(device, enabled=amp_enabled_global):
            semantic_logits, decoder_output = model.forward_base(images)
    return semantic_logits, decoder_output.detach()


def _forward_center_with_precision(
    *,
    model: UnetPlusPlusSemanticCenterHead,
    decoder_output: torch.Tensor,
    centers: torch.Tensor,
    center_loss_fn: CenterNetFocalHeatmapLoss,
    device: torch.device,
    amp_enabled_global: bool,
    center_fp32: bool,
    return_details: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | dict, dict]:
    decoder_features = model.resolve_center_features(decoder_output.detach())
    captured_info = model.center_feature_capture_info()
    captured_dtype_before_cast = captured_info.get("captured_dtype", _dtype_name(decoder_features))
    if center_fp32:
        decoder_features = decoder_features.float()
    center_autocast_enabled = bool(amp_enabled_global and (not center_fp32))
    with _autocast_ctx(device, enabled=center_autocast_enabled):
        native_center_logits = model.center_head(decoder_features if getattr(model, "center_adapter", None) is None else model.center_adapter(decoder_features))
        center_logits = model.upsample_center_logits(native_center_logits)
        if return_details and isinstance(center_loss_fn, CenterNetFocalHeatmapLoss):
            details = center_loss_fn(center_logits.float(), centers.float(), return_details=True)
            center_loss = details["loss"]
            payload: torch.Tensor | dict = details
        else:
            center_loss = center_loss_fn(center_logits.float(), centers.float())
            payload = center_loss
    precision_info = {
        "amp_enabled_global": bool(amp_enabled_global),
        "center_fp32": bool(center_fp32),
        "center_autocast_enabled": bool(center_autocast_enabled),
        "center_grad_scaler_enabled": bool(center_autocast_enabled),
        "decoder_features_dtype": _dtype_name(decoder_features),
        "captured_center_features_dtype_before_cast": captured_dtype_before_cast,
        "center_logits_dtype": _dtype_name(center_logits),
        "native_center_logits_dtype": _dtype_name(native_center_logits),
        "native_center_logits_shape": list(native_center_logits.shape),
        "final_center_logits_shape": list(center_logits.shape),
        "center_loss_dtype": _dtype_name(center_loss),
    }
    return decoder_features, center_logits, payload, precision_info


def _read_legacy_sqrt_precheck() -> dict:
    path = Path("training/analysis/centerhead_spatial_legacy_sqrt_hw_micro_overfit/same_batch_comparison.json").resolve()
    if not path.exists():
        return {"path": str(path), "exists": False, "gradient_cosine_legacy_vs_legacy_sqrt_hw": None}
    obj = json.loads(path.read_text(encoding="utf-8"))
    cosine = obj.get("gradient_cosine_legacy_vs_legacy_sqrt_hw")
    return {"path": str(path), "exists": True, "gradient_cosine_legacy_vs_legacy_sqrt_hw": cosine}


def _run_single_mode_one_batch(
    *,
    cfg: dict,
    batch: dict,
    device: torch.device,
    lr: float,
    clip_norm: float,
    normalization_mode: str,
) -> dict:
    model, _bias = _make_model_from_cfg(cfg, device)
    center_loss_fn = _make_center_loss_from_cfg(cfg, device, normalization_mode=normalization_mode)
    optimizer, _optimizer_meta = _build_center_branch_optimizer(model, center_lr=float(lr), adapter_lr=float(lr), weight_decay=0.0)
    amp_enabled = bool((cfg.get("train") or {}).get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    images = batch["image"].to(device)
    centers = batch["center"].to(device)
    bn_ref = _copy_bn_stats(model.base)
    with torch.no_grad():
        sem_before = model(images)["semantic"].detach().clone()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", enabled=amp_enabled):
        out = model(images)
        center_logits = out["center"]
        details = center_loss_fn(center_logits, centers, return_details=True)
        loss = details["loss"]
    if not bool(torch.isfinite(loss).all().item()):
        raise SystemExit(f"Same-batch comparison failed: non-finite loss for mode={normalization_mode}")

    if amp_enabled:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()

    params = _center_branch_params(model)
    grad_norm_before = _grad_l2_norm(params)
    grad_finite = all(bool(torch.isfinite(p.grad).all().item()) for p in params if p.grad is not None)
    grad_max_abs = max([float(p.grad.detach().abs().max().item()) for p in params if p.grad is not None], default=0.0)
    if float(clip_norm) > 0.0 and math.isfinite(grad_norm_before):
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(clip_norm))
    grad_norm_after = _grad_l2_norm(params)

    prev_scale = float(scaler.get_scale()) if amp_enabled else None
    if amp_enabled:
        scaler.step(optimizer)
        scaler.update()
        new_scale = float(scaler.get_scale())
        skipped = bool(new_scale < prev_scale)
        amp_scale = new_scale
    else:
        optimizer.step()
        skipped = False
        amp_scale = None

    with torch.no_grad():
        sem_after = model(images)["semantic"].detach().clone()
        sem_delta = float((sem_before - sem_after).abs().max().item())
        bn_delta = _max_bn_delta(model.base, bn_ref)
        params_finite = _params_finite(params)
        logits_finite = bool(torch.isfinite(center_logits.detach()).all().item())

    peak_vram = None
    if device.type == "cuda":
        peak_vram = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    pos_exact = centers >= 0.9999
    near = centers >= 0.1
    far = centers < 0.1
    prob = torch.sigmoid(center_logits.detach())
    prob_pos = float(prob[pos_exact].mean().item()) if bool(pos_exact.any().item()) else None
    prob_near = float(prob[near].mean().item()) if bool(near.any().item()) else None
    prob_far = float(prob[far].mean().item()) if bool(far.any().item()) else None

    norm_pos = float(details["normalized_pos_loss"].item())
    norm_neg = float(details["normalized_neg_loss"].item())
    ratio = float(norm_neg / max(norm_pos, 1e-12)) if norm_pos > 0.0 else None
    return {
        "normalization_mode": normalization_mode,
        "total_loss": float(loss.item()),
        "positive_loss_sum": float(details["pos_loss"].item()),
        "negative_loss_sum": float(details["neg_loss"].item()),
        "positive_normalizer": float(details["pos_normalizer"].item()),
        "negative_normalizer": float(details["neg_normalizer"].item()),
        "normalized_positive_loss": norm_pos,
        "normalized_negative_loss": norm_neg,
        "negative_positive_ratio": ratio,
        "num_positive_pixels": float(details["num_pos"].item()),
        "effective_negative_weight_sum": float(details["neg_normalizer"].item()),
        "mean_pred_probability": float(details["mean_pred"].item()),
        "prob_mean_pos": prob_pos,
        "prob_mean_near": prob_near,
        "prob_mean_far": prob_far,
        "grad_norm_before": float(grad_norm_before),
        "grad_norm_after": float(grad_norm_after),
        "grad_max_abs": float(grad_max_abs),
        "grad_finite": bool(grad_finite),
        "parameters_finite_after_step": bool(params_finite),
        "logits_finite_after_step": bool(logits_finite),
        "skipped_amp_step": bool(skipped),
        "amp_scale_last": amp_scale,
        "peak_vram_mb": peak_vram,
        "semantic_delta": float(sem_delta),
        "bn_delta": float(bn_delta),
    }


def _run_same_batch_comparison(
    *,
    cfg: dict,
    loader,
    device: torch.device,
    lr: float,
    clip_norm: float,
) -> dict:
    batch = next(iter(loader))
    model, _bias = _make_model_from_cfg(cfg, device)
    amp_enabled = bool((cfg.get("train") or {}).get("amp", False)) and device.type == "cuda"
    images = batch["image"].to(device)
    centers = batch["center"].to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.autocast(device_type="cuda", enabled=amp_enabled):
        out = model(images)
        center_logits = out["center"]

    params = _center_branch_params(model)
    results = {}
    flat_grads: dict[str, torch.Tensor] = {}
    modes = ["legacy_num_pos", "balanced_resolution", "legacy_sqrt_hw"]
    for idx, mode in enumerate(modes):
        model.zero_grad(set_to_none=True)
        loss_fn = _make_center_loss_from_cfg(cfg, device, normalization_mode=mode)
        details = loss_fn(center_logits, centers, return_details=True)
        loss = details["loss"]
        if not bool(torch.isfinite(loss).all().item()):
            raise SystemExit(f"Same-batch comparison failed: non-finite loss for mode={mode}")
        loss.backward(retain_graph=(idx < len(modes) - 1))
        grad_norm_before = _grad_l2_norm(params)
        grad_finite = all(bool(torch.isfinite(p.grad).all().item()) for p in params if p.grad is not None)
        grad_flat = _flatten_center_head_grads(params)
        flat_grads[mode] = grad_flat.detach().clone()
        clipped_required = bool(math.isfinite(grad_norm_before) and float(grad_norm_before) > float(clip_norm))
        if float(clip_norm) > 0.0 and math.isfinite(grad_norm_before):
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(clip_norm))
        grad_norm_after = _grad_l2_norm(params)
        peak_vram = None
        if device.type == "cuda":
            peak_vram = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        results[mode] = {
            "normalization_mode": mode,
            "total_loss": float(loss.item()),
            "positive_loss_sum": float(details["pos_loss"].item()),
            "negative_loss_sum": float(details["neg_loss"].item()),
            "num_positive_pixels": float(details["num_pos"].item()),
            "legacy_unscaled_loss": float(details["legacy_unscaled_loss"].item()),
            "resolution_height": float(details["resolution_height"]),
            "resolution_width": float(details["resolution_width"]),
            "resolution_scale": float(details["resolution_scale"]),
            "scaled_total_loss": float(details["scaled_total_loss"].item()),
            "negative_to_positive_sum_ratio": float(details["negative_to_positive_sum_ratio"].item()),
            "normalized_positive_loss": float(details["normalized_pos_loss"].item()),
            "normalized_negative_loss": float(details["normalized_neg_loss"].item()),
            "grad_norm_before": float(grad_norm_before),
            "grad_norm_after": float(grad_norm_after),
            "gradients_finite": bool(grad_finite),
            "clipping_required": bool(clipped_required),
            "peak_vram_mb": peak_vram,
        }

    legacy = results["legacy_num_pos"]
    balanced = results["balanced_resolution"]
    scaled = results["legacy_sqrt_hw"]
    return {
        "legacy_num_pos": legacy,
        "balanced_resolution": balanced,
        "legacy_sqrt_hw": scaled,
        "loss_reduction_balanced_factor": float(legacy["total_loss"] / max(balanced["total_loss"], 1e-12)),
        "grad_reduction_balanced_factor": float(legacy["grad_norm_before"] / max(balanced["grad_norm_before"], 1e-12)),
        "loss_reduction_legacy_sqrt_hw_factor": float(legacy["total_loss"] / max(scaled["total_loss"], 1e-12)),
        "grad_reduction_legacy_sqrt_hw_factor": float(legacy["grad_norm_before"] / max(scaled["grad_norm_before"], 1e-12)),
        "gradient_cosine_legacy_vs_legacy_sqrt_hw": _grad_cosine_similarity(flat_grads["legacy_num_pos"], flat_grads["legacy_sqrt_hw"]),
        "gradient_cosine_legacy_vs_balanced_resolution": _grad_cosine_similarity(flat_grads["legacy_num_pos"], flat_grads["balanced_resolution"]),
    }


def _run_same_batch_amp_vs_fp32_comparison(
    *,
    cfg: dict,
    loader,
    device: torch.device,
    lr: float,
    clip_norm: float,
) -> dict:
    batch = next(iter(loader))
    model, _bias = _make_model_from_cfg(cfg, device)
    images = batch["image"].to(device)
    centers = batch["center"].to(device)
    amp_enabled_global = _amp_enabled(cfg, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    _semantic_logits, decoder_output = _forward_frozen_base(
        model=model,
        images=images,
        device=device,
        amp_enabled_global=amp_enabled_global,
    )
    params = _center_branch_params(model)
    optimizer, _optimizer_meta = _build_center_branch_optimizer(model, center_lr=float(lr), adapter_lr=float(lr), weight_decay=0.0)
    loss_fn = _make_center_loss_from_cfg(cfg, device, normalization_mode="legacy_num_pos")

    def _run_mode(center_fp32: bool) -> tuple[dict, torch.Tensor]:
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        _decoder_features, center_logits, details, precision_info = _forward_center_with_precision(
            model=model,
            decoder_output=decoder_output,
            centers=centers,
            center_loss_fn=loss_fn,
            device=device,
            amp_enabled_global=amp_enabled_global,
            center_fp32=center_fp32,
            return_details=True,
        )
        assert isinstance(details, dict)
        loss = details["loss"]
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_enabled_global and (not center_fp32)))
        if bool(scaler.is_enabled()):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        grad_flat = _flatten_center_head_grads(params).detach().clone()
        grad_norm = _grad_l2_norm(params)
        grad_max_abs = max([float(p.grad.detach().abs().max().item()) for p in params if p.grad is not None], default=0.0)
        nonfinite_grad_tensors = int(sum(0 if (p.grad is None or bool(torch.isfinite(p.grad).all().item())) else 1 for p in params))
        clipped_required = bool(math.isfinite(grad_norm) and float(grad_norm) > float(clip_norm))
        peak_vram = None
        if device.type == "cuda":
            peak_vram = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        result = {
            "loss": float(loss.item()),
            "grad_norm_after_unscale": float(grad_norm),
            "nonfinite_grad_tensors": int(nonfinite_grad_tensors),
            "maximum_absolute_gradient": float(grad_max_abs),
            "clipping_required": bool(clipped_required),
            "peak_vram_mb": peak_vram,
            **precision_info,
        }
        return result, grad_flat

    amp_result, amp_grad = _run_mode(center_fp32=False)
    fp32_result, fp32_grad = _run_mode(center_fp32=True)
    amp_loss = float(amp_result["loss"])
    fp32_loss = float(fp32_result["loss"])
    rel_diff = abs(amp_loss - fp32_loss) / max(abs(fp32_loss), 1e-12)
    return {
        "legacy_num_pos_amp_center": amp_result,
        "legacy_num_pos_fp32_center": fp32_result,
        "relative_loss_difference": float(rel_diff),
        "gradient_cosine_similarity": _grad_cosine_similarity(amp_grad, fp32_grad),
    }


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "extra": extra,
        },
        str(path),
    )


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
        pred_count_sum = 0
        gt_count_sum = 0
        zero_center_cases = 0
        extra_center_cases = 0

        inst_exact = 0
        inst_n = 0
        inst_mean_iou_sum = 0.0
        inst_perfect = 0
        inst_merged = 0
        inst_fragmented = 0
        inst_mixed = 0

        gt_instance_total = 0
        one_marker_gt_instances = 0
        missing_gt_instance_markers = 0
        several_markers_inside_one_gt_instance = 0
        markers_outside_gt_instances = 0

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
                    pred_count_sum += int(len(pred_pts))
                    gt_count_sum += int(len(gt_pts))
                    zero_center_cases += int(len(pred_pts) == 0)
                    extra_center_cases += int(len(pred_pts) > len(gt_pts))

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
                    case = _case_type(gt_k, int(pred_k))
                    inst_merged += int(case == "merged")
                    inst_fragmented += int(case == "fragmented")
                    inst_mixed += int(case == "mixed")
                    iou_mat = _iou_matrix(gt_inst, pred_inst, gt_k, int(pred_k))
                    sum_iou = _best_perm_sum(iou_mat)
                    mean_iou = float(sum_iou / max(gt_k, 1))
                    inst_mean_iou_sum += float(mean_iou)
                    inst_perfect += int((int(pred_k) == int(gt_k)) and (mean_iou >= 0.90))

                    for inst_id in [1, 2, 3]:
                        if int(np.sum(gt_inst == inst_id)) <= 0:
                            continue
                        gt_instance_total += 1
                        marker_count = int(sum(1 for (y, x) in pred_pts if int(gt_inst[int(y), int(x)]) == inst_id))
                        if marker_count == 1:
                            one_marker_gt_instances += 1
                        elif marker_count == 0:
                            missing_gt_instance_markers += 1
                        else:
                            several_markers_inside_one_gt_instance += 1
                    markers_outside_gt_instances += int(sum(1 for (y, x) in pred_pts if int(gt_inst[int(y), int(x)]) == 0))

        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float((2 * precision * recall) / max(precision + recall, 1e-7))
        loc_err = float(loc_err_sum / max(loc_err_n, 1))
        count_acc = float(count_ok / max(count_n, 1))
        inst_exact_acc = float(inst_exact / max(inst_n, 1))
        inst_mean_iou = float(inst_mean_iou_sum / max(inst_n, 1))
        inst_perfect_rate = float(inst_perfect / max(inst_n, 1))
        inst_merged_rate = float(inst_merged / max(inst_n, 1))
        inst_fragmented_rate = float(inst_fragmented / max(inst_n, 1))
        inst_mixed_rate = float(inst_mixed / max(inst_n, 1))
        row = {
            "threshold": float(thr),
            "center_precision": precision,
            "center_recall": recall,
            "center_f1": f1,
            "center_count_acc": count_acc,
            "center_loc_err_px": loc_err,
            "center_pred_count_mean": float(pred_count_sum / max(count_n, 1)),
            "center_gt_count_mean": float(gt_count_sum / max(count_n, 1)),
            "center_zero_cases": int(zero_center_cases),
            "center_extra_cases": int(extra_center_cases),
            "center_prob_mean_pos": float(prob_pos_sum / max(prob_pos_n, 1)),
            "center_prob_mean_near": float(prob_near_sum / max(prob_near_n, 1)),
            "center_prob_mean_far": float(prob_far_sum / max(prob_far_n, 1)),
            "center_prob_mean_max": float(prob_max_sum / max(prob_max_n, 1)),
            "center_pos_minus_far": float((prob_pos_sum / max(prob_pos_n, 1)) - (prob_far_sum / max(prob_far_n, 1))),
            "center_pos_minus_near": float((prob_pos_sum / max(prob_pos_n, 1)) - (prob_near_sum / max(prob_near_n, 1))),
            "one_marker_per_gt_instance_rate": float(one_marker_gt_instances / max(gt_instance_total, 1)),
            "missing_gt_instance_marker_count": int(missing_gt_instance_markers),
            "several_markers_inside_one_gt_instance_count": int(several_markers_inside_one_gt_instance),
            "markers_outside_gt_instances": int(markers_outside_gt_instances),
            "instance_exact_count_acc": inst_exact_acc,
            "instance_merged_rate": inst_merged_rate,
            "instance_fragmented_rate": inst_fragmented_rate,
            "instance_mixed_rate": inst_mixed_rate,
            "instance_mean_matched_iou": inst_mean_iou,
            "instance_perfect_rate": inst_perfect_rate,
        }
        row["instance_score"] = _instance_score_from_row(row)
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
                bin_best = (pr_center[i, 0] >= float(best_threshold)).astype(np.uint8) * 255
                cv2.imwrite(str(sd / "binary_best_thr.png"), bin_best)

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

                    compare = np.concatenate(
                        [
                            cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR),
                            cv2.applyColorMap((gt_u16 / 256).astype(np.uint8), cv2.COLORMAP_JET),
                            cv2.applyColorMap((pr_u16 / 256).astype(np.uint8), cv2.COLORMAP_JET),
                            cv2.applyColorMap((diff_u16 / 256).astype(np.uint8), cv2.COLORMAP_MAGMA),
                            markers_vis,
                        ],
                        axis=1,
                    )
                    cv2.imwrite(str(sd / "compare.png"), compare)

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


def _run_smoke_test(
    *,
    model: UnetPlusPlusSemanticCenterHead,
    loader,
    device: torch.device,
    center_loss_fn: CenterNetFocalHeatmapLoss,
    optimizer: torch.optim.Optimizer,
    optimizer_meta: list[dict],
    clip_norm: float,
    amp_enabled_global: bool,
    center_fp32: bool,
) -> dict:
    batch = next(iter(loader))
    images = batch["image"].to(device)
    centers = batch["center"].to(device)
    model.base.eval()
    if getattr(model, "center_adapter", None) is not None:
        model.center_adapter.train()
    model.center_head.train()
    bn_ref = _copy_bn_stats(model.base)
    adapter_named = [(n, p) for n, p in model.named_parameters() if str(n).startswith("center_adapter.")]
    head_named = [(n, p) for n, p in model.named_parameters() if str(n).startswith("center_head.")]
    frozen_named = [(n, p) for n, p in model.named_parameters() if not bool(p.requires_grad)]
    adapter_ref = _snapshot_named_parameters(adapter_named)
    head_ref = _snapshot_named_parameters(head_named)
    frozen_ref = _snapshot_named_parameters(frozen_named)
    align_t0 = time.perf_counter()
    alignment = _upsample_alignment_test(model, device)
    alignment["runtime_sec"] = float(time.perf_counter() - align_t0)
    if alignment.get("error_px", 0.0) > 3.0:
        raise SystemExit(f"Smoke test failed: upsample alignment error too large ({alignment['error_px']:.3f}px)")
    if not bool(alignment.get("batch_consistent", False)):
        raise SystemExit("Smoke test failed: upsample alignment differs across batch size")
    with torch.no_grad():
        sem_before, _decoder_before = _forward_frozen_base(
            model=model,
            images=images,
            device=device,
            amp_enabled_global=amp_enabled_global,
        )
    optimizer.zero_grad(set_to_none=True)
    forward_t0 = time.perf_counter()
    _sem_logits, decoder_output = _forward_frozen_base(
        model=model,
        images=images,
        device=device,
        amp_enabled_global=amp_enabled_global,
    )
    capture_info = model.center_feature_capture_info()
    decoder_features, center_logits, details, precision_info = _forward_center_with_precision(
        model=model,
        decoder_output=decoder_output,
        centers=centers,
        center_loss_fn=center_loss_fn,
        device=device,
        amp_enabled_global=amp_enabled_global,
        center_fp32=center_fp32,
        return_details=True,
    )
    forward_time = float(time.perf_counter() - forward_t0)
    assert isinstance(details, dict)
    loss = details["loss"]
    if not bool(torch.isfinite(loss).all().item()):
        raise SystemExit("Smoke test failed: non-finite loss")
    expected_native_h = int(centers.shape[-2]) // max(int(capture_info.get("native_stride") or 1), 1)
    expected_native_w = int(centers.shape[-1]) // max(int(capture_info.get("native_stride") or 1), 1)
    captured_shape = capture_info.get("captured_shape") or []
    if int(capture_info.get("hook_call_count") or 0) != 1:
        raise SystemExit(f"Smoke test failed: expected hook_call_count=1, got {capture_info.get('hook_call_count')}")
    if captured_shape and len(captured_shape) >= 4:
        if int(captured_shape[-2]) != int(expected_native_h) or int(captured_shape[-1]) != int(expected_native_w):
            raise SystemExit(
                "Smoke test failed: captured feature shape does not match configured stride "
                f"(expected spatial {expected_native_h}x{expected_native_w}, got {captured_shape[-2]}x{captured_shape[-1]})"
            )
    native_shape = precision_info.get("native_center_logits_shape") or []
    final_shape = precision_info.get("final_center_logits_shape") or []
    if native_shape and len(native_shape) >= 4:
        if int(native_shape[-2]) != int(expected_native_h) or int(native_shape[-1]) != int(expected_native_w):
            raise SystemExit(
                "Smoke test failed: native center logits shape mismatch "
                f"(expected {expected_native_h}x{expected_native_w}, got {native_shape[-2]}x{native_shape[-1]})"
            )
    if final_shape and len(final_shape) >= 4:
        if int(final_shape[-2]) != int(centers.shape[-2]) or int(final_shape[-1]) != int(centers.shape[-1]):
            raise SystemExit(
                "Smoke test failed: upsampled center logits shape mismatch "
                f"(expected {int(centers.shape[-2])}x{int(centers.shape[-1])}, got {final_shape[-2]}x{final_shape[-1]})"
            )
    backward_t0 = time.perf_counter()
    loss.backward()
    backward_time = float(time.perf_counter() - backward_t0)

    trainable_names = [n for (n, p) in model.named_parameters() if bool(p.requires_grad)]
    assert _center_branch_prefix_ok(trainable_names), f"Unexpected trainable params found: {trainable_names[:10]}"
    frozen_base_grad_count = _count_present_grads(list(model.base.parameters()))
    center_branch_params = _center_branch_params(model)
    adapter_params = _center_adapter_params(model)
    head_params = _center_head_params(model)
    adapter_grad_before = _grad_l2_norm(adapter_params)
    head_grad_before = _grad_l2_norm(head_params)
    combined_grad_before = _grad_l2_norm(center_branch_params)
    grad_nonzero = bool(combined_grad_before > 0.0)
    if not grad_nonzero:
        raise SystemExit("Smoke test failed: center-branch gradients are zero")
    torch.nn.utils.clip_grad_norm_(center_branch_params, max_norm=float(clip_norm))
    adapter_grad_after = _grad_l2_norm(adapter_params)
    head_grad_after = _grad_l2_norm(head_params)
    combined_grad_after = _grad_l2_norm(center_branch_params)
    optimizer.step()

    with torch.no_grad():
        sem_after, _decoder_after = _forward_frozen_base(
            model=model,
            images=images,
            device=device,
            amp_enabled_global=amp_enabled_global,
        )
        sem_delta = float((sem_before - sem_after).abs().max().item())
        bn_delta = _max_bn_delta(model.base, bn_ref)
        params_finite = _params_finite(center_branch_params)
        logits_finite = bool(torch.isfinite(center_logits.detach()).all().item())
        adapter_delta = _max_parameter_delta(adapter_named, adapter_ref)
        head_delta = _max_parameter_delta(head_named, head_ref)
        frozen_delta = _max_parameter_delta(frozen_named, frozen_ref)

    peak_vram = None
    if device.type == "cuda":
        peak_vram = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    return {
        "passed": True,
        "feature_capture": {
            "configured_module_path": capture_info.get("configured_module_path"),
            "actual_module_path": capture_info.get("actual_module_path"),
            "hook_call_count": int(capture_info.get("hook_call_count") or 0),
            "captured_shape": capture_info.get("captured_shape"),
            "captured_dtype_before_cast": precision_info.get("captured_center_features_dtype_before_cast"),
            "captured_dtype_after_cast": precision_info.get("decoder_features_dtype"),
            "expected_channels": capture_info.get("expected_channels"),
            "actual_channels": capture_info.get("actual_channels"),
            "native_stride": capture_info.get("native_stride"),
        },
        "model": {
            "adapter_parameters": int(_parameter_count(adapter_params)),
            "center_head_parameters": int(_parameter_count(head_params)),
            "total_trainable_parameters": int(_parameter_count(center_branch_params)),
            "exact_trainable_parameter_names": trainable_names,
            "frozen_parameter_count": int(sum(int(p.numel()) for _n, p in frozen_named)),
            "optimizer_groups": optimizer_meta,
            "optimizer_overlap": False,
            "frozen_parameters_in_optimizer": 0,
        },
        "forward": {
            "semantic_logits_shape": list(_sem_logits.shape),
            "semantic_logits_dtype": _dtype_name(_sem_logits),
            "native_center_logits_shape": precision_info.get("native_center_logits_shape"),
            "native_center_logits_dtype": precision_info.get("native_center_logits_dtype"),
            "upsampled_center_logits_shape": precision_info.get("final_center_logits_shape"),
            "upsampled_center_logits_dtype": precision_info.get("center_logits_dtype"),
            "center_target_shape": list(centers.shape),
            "center_target_dtype": _dtype_name(centers),
            "center_loss_dtype": precision_info.get("center_loss_dtype"),
            "loss": float(loss.item()),
            "loss_finite": bool(torch.isfinite(loss).all().item()),
            "logits_finite": bool(logits_finite),
        },
        "gradients": {
            "adapter_grad_norm_before_clip": float(adapter_grad_before),
            "center_head_grad_norm_before_clip": float(head_grad_before),
            "combined_grad_norm_before_clip": float(combined_grad_before),
            "adapter_grad_norm_after_clip": float(adapter_grad_after),
            "center_head_grad_norm_after_clip": float(head_grad_after),
            "combined_grad_norm_after_clip": float(combined_grad_after),
            "nonfinite_gradient_tensors": int(sum(0 if (p.grad is None or bool(torch.isfinite(p.grad).all().item())) else 1 for p in center_branch_params)),
            "frozen_base_gradient_count": int(frozen_base_grad_count),
        },
        "parameter_deltas": {
            "adapter_parameter_delta": float(adapter_delta),
            "center_head_parameter_delta": float(head_delta),
            "frozen_parameter_max_delta": float(frozen_delta),
            "frozen_bn_stats_max_delta": float(bn_delta),
            "semantic_logits_max_abs_delta_after_step": float(sem_delta),
        },
        "runtime": {
            "peak_vram_mb": peak_vram,
            "forward_time_sec": float(forward_time),
            "backward_time_sec": float(backward_time),
        },
        "alignment_test": alignment,
        "batchnorm_in_center_head": int(_count_center_head_batchnorms(model)),
        "groupnorm_present": bool(any(isinstance(m, torch.nn.GroupNorm) for m in model.center_head.modules())),
        "parameters_finite_after_step": bool(params_finite),
        "final_bias": _center_head_output_bias(model),
        **precision_info,
        "decoder_features_dtype_runtime": _dtype_name(decoder_features),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--microset-txt", type=Path, default=Path("training/analysis/centerhead_micro_overfit/microset.txt"))
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--vis-iters", type=str, default="0,25,50,100,250,500,750,1000")
    ap.add_argument("--lr", type=float, default=-1.0)
    ap.add_argument("--grad-clip-norm", type=float, default=-1.0)
    ap.add_argument("--batch-size", type=int, default=-1)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    cfg = _read_yaml(args.config.resolve())
    _seed_all(int(cfg.get("seed", 1337)))
    device = _make_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Configured CUDA device is unavailable; falling back to CPU for this run.")
        device = torch.device("cpu")
    amp_enabled_global = _amp_enabled(cfg, device)
    center_fp32 = _center_fp32_enabled(cfg)

    focal_cfg = cfg.get("center_loss") or {}
    normalization_mode = str((focal_cfg.get("normalization_mode", "legacy_num_pos") if isinstance(focal_cfg, dict) else "legacy_num_pos")).strip().lower() or "legacy_num_pos"
    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
    elif normalization_mode == "legacy_num_pos" and center_fp32 and str(((cfg.get("model") or {}).get("center_feature") or {}).get("module_path", "")).strip() == "base.decoder.blocks.x_2_2":
        out_dir = Path("training/analysis/centerhead_spatial_x2_2_adapter_legacy_fp32_micro_overfit").resolve()
    elif normalization_mode == "legacy_num_pos" and center_fp32:
        out_dir = Path("training/analysis/centerhead_spatial_legacy_fp32_micro_overfit").resolve()
    elif normalization_mode == "legacy_sqrt_hw":
        out_dir = Path("training/analysis/centerhead_spatial_legacy_sqrt_hw_micro_overfit").resolve()
    elif normalization_mode == "balanced_resolution":
        out_dir = Path("training/analysis/centerhead_spatial_balancednorm_micro_overfit").resolve()
    else:
        out_dir = Path("training/analysis/centerhead_spatial_micro_overfit").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(cfg["dataset"]["root"]).resolve()
    instance_root = Path((cfg.get("dataset") or {}).get("instance_root", "datasets/converted_leaflet_instances")).resolve()

    micro = _load_existing_microset(dataset_root, args.microset_txt, out_dir)
    (out_dir / "microset_manifest.json").write_text(
        json.dumps({"samples": micro.samples, "distribution": micro.distribution}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    loader = _build_loader_for_split(
        cfg,
        dataset_root=dataset_root,
        split_txt=micro.split_txt,
        device=device,
        batch_size=int(args.batch_size if int(args.batch_size) > 0 else int((cfg.get("train") or {}).get("batch_size", 4))),
        num_workers=int(args.num_workers),
    )

    center_head_type = str((cfg.get("model") or {}).get("center_head_type", "linear_1x1")).strip().lower() or "linear_1x1"
    model, bias = _make_model_from_cfg(cfg, device)

    alpha = float((focal_cfg.get("alpha", 2.0) if isinstance(focal_cfg, dict) else 2.0))
    beta = float((focal_cfg.get("beta", 4.0) if isinstance(focal_cfg, dict) else 4.0))
    center_loss_fn = _make_center_loss_from_cfg(cfg, device, normalization_mode=normalization_mode)

    lr = float(args.lr if float(args.lr) > 0 else float((cfg.get("train") or {}).get("lr_center_head", 1e-4)))
    adapter_lr = float((cfg.get("train") or {}).get("lr_center_adapter", lr))
    clip_norm = float(args.grad_clip_norm if float(args.grad_clip_norm) > 0 else float((cfg.get("train") or {}).get("center_grad_clip_norm", 5.0)))
    opt, opt_meta = _build_center_branch_optimizer(model, center_lr=float(lr), adapter_lr=float(adapter_lr), weight_decay=0.0)
    thresholds = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90]

    layer_out = model.center_head_output_layer()
    trainable_names = [n for (n, p) in model.named_parameters() if bool(p.requires_grad)]
    architecture = {
        "head_type": center_head_type,
        "layers": "3x3 stem -> 4 residual dilated blocks -> 3x3 refine -> 1x1 out" if center_head_type == "spatial_dilated" else "single segmentation head",
        "dilation_sequence": [1, 2, 4, 8] if center_head_type == "spatial_dilated" else [],
        "trainable_parameters": int(sum(int(p.numel()) for p in model.parameters() if bool(p.requires_grad))),
        "center_adapter_parameters": int(_parameter_count(_center_adapter_params(model))),
        "center_head_parameters": int(sum(int(p.numel()) for p in model.center_head.parameters())),
        "total_parameters": int(sum(int(p.numel()) for p in model.parameters())),
        "receptive_field": "approx 35x35 from center head alone" if center_head_type == "spatial_dilated" else "pointwise/near-local",
        "final_bias": _center_head_output_bias(model),
        "output_layer": layer_out.__class__.__name__,
        "normalization_mode": normalization_mode,
        "amp_enabled_global": bool(amp_enabled_global),
        "center_fp32": bool(center_fp32),
        "trainable_names": trainable_names,
        "center_feature": model.center_feature_cfg,
    }
    (out_dir / "architecture.json").write_text(json.dumps(architecture, ensure_ascii=False, indent=2), encoding="utf-8")

    legacy_sqrt_precheck = _read_legacy_sqrt_precheck()
    cosine_precheck = legacy_sqrt_precheck.get("gradient_cosine_legacy_vs_legacy_sqrt_hw")
    if normalization_mode == "legacy_sqrt_hw" and not legacy_sqrt_precheck.get("exists"):
        raise SystemExit(f"legacy_sqrt_hw precheck file not found: {legacy_sqrt_precheck.get('path')}")
    if normalization_mode == "legacy_sqrt_hw" and cosine_precheck is None:
        raise SystemExit("legacy_sqrt_hw precheck missing gradient_cosine_legacy_vs_legacy_sqrt_hw")
    if normalization_mode == "legacy_sqrt_hw" and legacy_sqrt_precheck.get("exists") and cosine_precheck is not None and float(cosine_precheck) < 0.99999:
        raise SystemExit(f"legacy_sqrt_hw precheck failed: cosine={float(cosine_precheck):.8f} < 0.99999")
    (out_dir / "legacy_sqrt_hw_precheck.json").write_text(json.dumps(legacy_sqrt_precheck, ensure_ascii=False, indent=2), encoding="utf-8")

    same_batch = _run_same_batch_comparison(
        cfg=cfg,
        loader=loader,
        device=device,
        lr=float(lr),
        clip_norm=clip_norm,
    )
    (out_dir / "same_batch_comparison.json").write_text(json.dumps(same_batch, ensure_ascii=False, indent=2), encoding="utf-8")

    same_batch_amp_vs_fp32 = _run_same_batch_amp_vs_fp32_comparison(
        cfg=cfg,
        loader=loader,
        device=device,
        lr=float(lr),
        clip_norm=clip_norm,
    )
    (out_dir / "same_batch_amp_vs_fp32.json").write_text(json.dumps(same_batch_amp_vs_fp32, ensure_ascii=False, indent=2), encoding="utf-8")

    smoke = _run_smoke_test(
        model=model,
        loader=loader,
        device=device,
        center_loss_fn=center_loss_fn,
        optimizer=opt,
        optimizer_meta=opt_meta,
        clip_norm=clip_norm,
        amp_enabled_global=amp_enabled_global,
        center_fp32=center_fp32,
    )
    (out_dir / "smoke_test.json").write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")

    # Rebuild a fresh model after smoke test so iter=0 truly starts before any optimizer step.
    model, _bias2 = _make_model_from_cfg(cfg, device)
    opt, opt_meta = _build_center_branch_optimizer(model, center_lr=float(lr), adapter_lr=float(adapter_lr), weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled_global and (not center_fp32)))

    vis_iters = sorted({int(x.strip()) for x in str(args.vis_iters).split(",") if str(x).strip()})
    metrics_csv = (out_dir / "micro_overfit_metrics.csv").resolve()
    if not metrics_csv.exists():
        with metrics_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "iter",
                    "loss",
                    "legacy_unscaled_loss",
                    "resolution_scale",
                    "pos_loss_sum",
                    "neg_loss_sum",
                    "num_pos",
                    "negative_to_positive_sum_ratio",
                    "neg_pos_ratio",
                    "center_prob_mean_pos",
                    "center_prob_mean_near",
                    "center_prob_mean_far",
                    "center_pos_minus_far",
                    "center_pos_minus_near",
                    "center_prob_mean_max",
                    "best_thr",
                    "best_precision",
                    "best_recall",
                    "best_f1",
                    "best_count_acc",
                    "best_loc_err_px",
                    "best_pred_count_mean",
                    "best_gt_count_mean",
                    "one_marker_per_gt_instance_rate",
                    "missing_gt_instance_marker_count",
                    "several_markers_inside_one_gt_instance_count",
                    "markers_outside_gt_instances",
                    "inst_exact_count_acc",
                    "inst_merged_rate",
                    "inst_fragmented_rate",
                    "inst_mixed_rate",
                    "inst_mean_matched_iou",
                    "inst_perfect_rate",
                    "instance_score",
                    "adapter_grad_norm_before",
                    "center_head_grad_norm_before",
                    "combined_grad_norm_before",
                    "adapter_grad_norm_after",
                    "center_head_grad_norm_after",
                    "combined_grad_norm_after",
                    "clipped",
                    "clipped_pct",
                    "nonfinite_gradient_tensors",
                    "nonfinite_grad_count",
                    "skipped_step_count",
                    "center_weight_norm",
                    "center_bias",
                    "logits_min",
                    "logits_max",
                    "params_finite",
                    "nan_or_inf",
                ]
            )

    clipped_n = 0
    nonfinite_grad_n = 0
    skipped_steps_n = 0
    eval_every = int(args.eval_every)
    iters = int(args.iters)
    best_f1 = None
    best_step = 0

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
    best_ckpt = (out_dir / "best_micro_overfit.pth").resolve()
    last_ckpt = (out_dir / "last.pth").resolve()

    for step in range(1, iters + 1):
        model.base.eval()
        if getattr(model, "center_adapter", None) is not None:
            model.center_adapter.train()
        model.center_head.train()

        batch = next(iter(loader))
        images = batch["image"].to(device)
        centers = batch["center"].to(device)

        opt.zero_grad(set_to_none=True)
        _sem_logits, decoder_output = _forward_frozen_base(
            model=model,
            images=images,
            device=device,
            amp_enabled_global=amp_enabled_global,
        )
        decoder_features, logits, details, precision_info = _forward_center_with_precision(
            model=model,
            decoder_output=decoder_output,
            centers=centers,
            center_loss_fn=center_loss_fn,
            device=device,
            amp_enabled_global=amp_enabled_global,
            center_fp32=center_fp32,
            return_details=True,
        )
        assert isinstance(details, dict)
        loss = details["loss"]
        if not bool(torch.isfinite(loss).all().item()):
            raise SystemExit(f"Non-finite loss at iter {step}")
        if bool(scaler.is_enabled()):
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()

        params = _center_branch_params(model)
        adapter_params = _center_adapter_params(model)
        head_params = _center_head_params(model)
        adapter_grad_norm_before = _grad_l2_norm(adapter_params)
        head_grad_norm_before = _grad_l2_norm(head_params)
        grad_norm_before = _grad_l2_norm(params)
        nonfinite_gradient_tensors = int(sum(0 if (p.grad is None or bool(torch.isfinite(p.grad).all().item())) else 1 for p in params))
        grad_nonfinite = bool(nonfinite_gradient_tensors > 0) or (not math.isfinite(grad_norm_before))
        if grad_nonfinite:
            nonfinite_grad_n += 1
        clipped = False
        if float(clip_norm) > 0.0 and math.isfinite(grad_norm_before):
            clipped = bool(grad_norm_before > float(clip_norm))
            if clipped:
                clipped_n += 1
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(clip_norm))
        adapter_grad_norm_after = _grad_l2_norm(adapter_params)
        head_grad_norm_after = _grad_l2_norm(head_params)
        grad_norm_after = _grad_l2_norm(params)
        skipped_steps = 0
        if bool(scaler.is_enabled()):
            prev_scale = float(scaler.get_scale())
            scaler.step(opt)
            scaler.update()
            new_scale = float(scaler.get_scale())
            skipped_steps = int(new_scale < prev_scale)
            skipped_steps_n += int(skipped_steps)
        else:
            opt.step()

        with torch.no_grad():
            b = _center_head_output_bias(model)
            w_norm = _center_head_weight_norm(model)
            logits_min = float(logits.detach().min().item())
            logits_max = float(logits.detach().max().item())
            params_finite = _params_finite(params)
            nan_or_inf = bool((not params_finite) or (not bool(torch.isfinite(logits.detach()).all().item())))

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
            if best_f1 is None or float(best.get("center_f1") or 0.0) > float(best_f1):
                best_f1 = float(best.get("center_f1") or 0.0)
                best_step = int(step)
                _save_checkpoint(
                    best_ckpt,
                    model,
                    opt,
                    step,
                    {"best_threshold": float(best.get("threshold") or 0.0), "best_center_f1": float(best_f1)},
                )

            with metrics_csv.open("a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        step,
                        float(loss.item()),
                        float(details["legacy_unscaled_loss"].item()),
                        float(details["resolution_scale"]),
                        float(details["pos_loss"].item()),
                        float(details["neg_loss"].item()),
                        float(details["num_pos"].item()),
                        float(details["negative_to_positive_sum_ratio"].item()),
                        float(details["normalized_neg_loss"].item() / max(float(details["normalized_pos_loss"].item()), 1e-12)) if float(details["normalized_pos_loss"].item()) > 0 else "",
                        float(best.get("center_prob_mean_pos") or 0.0),
                        float(best.get("center_prob_mean_near") or 0.0),
                        float(best.get("center_prob_mean_far") or 0.0),
                        float(best.get("center_pos_minus_far") or 0.0),
                        float(best.get("center_pos_minus_near") or 0.0),
                        float(best.get("center_prob_mean_max") or 0.0),
                        float(best.get("threshold") or 0.0),
                        float(best.get("center_precision") or 0.0),
                        float(best.get("center_recall") or 0.0),
                        float(best.get("center_f1") or 0.0),
                        float(best.get("center_count_acc") or 0.0),
                        float(best.get("center_loc_err_px") or 0.0),
                        float(best.get("center_pred_count_mean") or 0.0),
                        float(best.get("center_gt_count_mean") or 0.0),
                        float(best.get("one_marker_per_gt_instance_rate") or 0.0),
                        int(best.get("missing_gt_instance_marker_count") or 0),
                        int(best.get("several_markers_inside_one_gt_instance_count") or 0),
                        int(best.get("markers_outside_gt_instances") or 0),
                        float(best.get("instance_exact_count_acc") or 0.0),
                        float(best.get("instance_merged_rate") or 0.0),
                        float(best.get("instance_fragmented_rate") or 0.0),
                        float(best.get("instance_mixed_rate") or 0.0),
                        float(best.get("instance_mean_matched_iou") or 0.0),
                        float(best.get("instance_perfect_rate") or 0.0),
                        float(best.get("instance_score") or 0.0),
                        float(adapter_grad_norm_before),
                        float(head_grad_norm_before),
                        float(grad_norm_before),
                        float(adapter_grad_norm_after),
                        float(head_grad_norm_after),
                        float(grad_norm_after),
                        int(clipped),
                        float(100.0 * float(clipped_n) / float(max(step, 1))),
                        int(nonfinite_gradient_tensors),
                        int(nonfinite_grad_n),
                        int(skipped_steps_n),
                        float(w_norm) if w_norm is not None else "",
                        float(b) if b is not None else "",
                        float(logits_min),
                        float(logits_max),
                        int(params_finite),
                        int(nan_or_inf),
                    ]
                )

            pct = 100.0 * float(clipped_n) / float(max(step, 1))
            print(
                f"iter={step} loss={loss.item():.6f} "
                f"best_thr={float(best.get('threshold') or 0.0):.3f} best_f1={float(best.get('center_f1') or 0.0):.4f} "
                f"clipped_pct={pct:.1f}% "
                f"center_dtype={precision_info.get('center_logits_dtype')}"
            )
            _save_checkpoint(
                last_ckpt,
                model,
                opt,
                step,
                {"best_step": int(best_step), "best_center_f1": float(best_f1 or 0.0), "best_threshold": float(best.get("threshold") or 0.0)},
            )

    if iters not in vis_iters and best_step > 0:
        sweep_best = json.loads((out_dir / "threshold_sweeps" / f"iter_{best_step:04d}.json").read_text(encoding="utf-8"))
        best_row = sweep_best.get("best") or {}
        _export_visuals(
            out_dir=out_dir,
            model=model,
            loader=loader,
            device=device,
            instance_root=instance_root,
            tag="best",
            best_threshold=float(best_row.get("threshold") or 0.1),
        )

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "architecture": architecture,
                "legacy_sqrt_hw_precheck": legacy_sqrt_precheck,
                "smoke_test": smoke,
                "samples": micro.samples,
                "distribution": micro.distribution,
                "iters": iters,
                "eval_every": eval_every,
                "alpha": alpha,
                "beta": beta,
                "normalization_mode": normalization_mode,
                "amp_enabled_global": bool(amp_enabled_global),
                "center_fp32": bool(center_fp32),
                "init_bias": bias,
                "init_sigmoid": _safe_sigmoid(bias),
                "grad_clip_norm": clip_norm,
                "same_batch_comparison": same_batch,
                "same_batch_amp_vs_fp32": same_batch_amp_vs_fp32,
                "percent_iterations_clipped": float(100.0 * float(clipped_n) / float(max(iters, 1))),
                "best_step": int(best_step),
                "best_center_f1": float(best_f1 or 0.0),
                "metrics_csv": str(metrics_csv),
                "best_checkpoint": str(best_ckpt),
                "last_checkpoint": str(last_ckpt),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
