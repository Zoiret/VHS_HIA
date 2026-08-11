from __future__ import annotations

import argparse
import csv
import hashlib
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _append_dict_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _configs_equal_for_run(existing: dict, current: dict) -> bool:
    return json.dumps(existing, sort_keys=True, ensure_ascii=False) == json.dumps(current, sort_keys=True, ensure_ascii=False)


def _ensure_save_dir_compatible(out_dir: Path, cfg: dict) -> None:
    if not out_dir.exists():
        return
    existing_entries = [p.name for p in out_dir.iterdir()]
    if not existing_entries:
        return
    config_path = out_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"Save directory already contains files without config.json: {out_dir}")
    existing_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if not _configs_equal_for_run(existing_cfg, cfg):
        raise SystemExit(f"Save directory already contains an incompatible run: {out_dir}")


def _float_or_none(x):
    return None if x is None else float(x)


def _locked_threshold_context_label(locked_threshold: float) -> str:
    return f"locked_reference_threshold@{float(locked_threshold):.6f}"


def _best_sweep_threshold_context_label() -> str:
    return "best_validation_threshold_from_sweep"


def _center_branch_prefixes(model: UnetPlusPlusSemanticCenterHead) -> list[str]:
    prefixes = list(getattr(model, "center_branch_parameter_prefixes", lambda: ["center_head.", "center_adapter."])())
    return [str(prefix) for prefix in prefixes]


def _is_center_branch_param_name(model: UnetPlusPlusSemanticCenterHead, name: str) -> bool:
    return any(str(name).startswith(prefix) for prefix in _center_branch_prefixes(model))


def _strict_checkpoint_metric_key(cfg: dict) -> str:
    return str(((cfg.get("train") or {}).get("strict_checkpoint_metric")) or "strict_marker_contract_pass_rate")


def _candidate_key(row: dict | None, *, primary_metric: str, epoch: int, include_threshold: bool) -> tuple:
    if not isinstance(row, dict):
        return (-float("inf"), -float("inf"), -float("inf"), -float("inf"), float("inf"), float("inf"))
    loc_err = row.get("localization_error_px_pooled_matches", row.get("localization_error_px", row.get("center_loc_err_px", None)))
    threshold = row.get("threshold", None)
    secondary_metric = "center_f1_mean_samples" if str(primary_metric) == "strict_marker_contract_pass_rate" else "strict_marker_contract_pass_rate"
    return (
        float(row.get(primary_metric) if row.get(primary_metric) is not None else -float("inf")),
        float(row.get(secondary_metric) if row.get(secondary_metric) is not None else -float("inf")),
        float(row.get("exact_center_count_accuracy", row.get("center_count_acc", None)) if (row.get("exact_center_count_accuracy", row.get("center_count_acc", None)) is not None) else -float("inf")),
        -float(loc_err if loc_err is not None else float("inf")),
        -float(epoch),
        -float(threshold) if include_threshold and threshold is not None else 0.0,
    )


def _select_best_threshold_row(rows: list[dict], *, primary_metric: str = "center_f1_mean_samples") -> dict | None:
    best = None
    best_key = None
    for row in rows:
        key = _candidate_key(row, primary_metric=primary_metric, epoch=0, include_threshold=True)
        if best_key is None or key > best_key:
            best = row
            best_key = key
    return best


def _is_better_epoch_candidate(candidate: dict | None, incumbent: dict | None, *, epoch: int, incumbent_epoch: int | None, primary_metric: str) -> bool:
    if not isinstance(candidate, dict):
        return False
    if incumbent is None or incumbent_epoch is None:
        return True
    return _candidate_key(candidate, primary_metric=primary_metric, epoch=epoch, include_threshold=False) > _candidate_key(incumbent, primary_metric=primary_metric, epoch=incumbent_epoch, include_threshold=False)


def _write_validation_reports(out_dir: Path, *, epoch: int, val_metrics: dict, sweep_res: dict | None, locked_threshold: float) -> None:
    per_patient_rows = []
    for patient_id, metrics in sorted((val_metrics.get("per_patient_center_metrics") or {}).items()):
        per_patient_rows.append(
            {
                "epoch": int(epoch),
                "patient_id": str(patient_id),
                **dict(metrics),
            }
        )
    _append_dict_rows(
        out_dir / "validation_per_patient_metrics.csv",
        per_patient_rows,
        [
            "epoch",
            "patient_id",
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
            "localization_error_px_pooled_matches",
        ],
    )
    per_gt_rows = []
    for gt_count, metrics in sorted((val_metrics.get("per_gt_count_center_metrics") or {}).items(), key=lambda kv: int(kv[0])):
        per_gt_rows.append(
            {
                "epoch": int(epoch),
                "gt_instance_count": int(gt_count),
                **dict(metrics),
            }
        )
    _append_dict_rows(
        out_dir / "validation_gt_count_metrics.csv",
        per_gt_rows,
        [
            "epoch",
            "gt_instance_count",
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
            "localization_error_px_pooled_matches",
        ],
    )
    threshold_rows = []
    rows = list((sweep_res or {}).get("rows") or [])
    primary = _select_best_threshold_row(rows)
    strict_best = _select_best_threshold_row(rows, primary_metric="strict_marker_contract_pass_rate")
    for row in rows:
        threshold_rows.append(
            {
                "epoch": int(epoch),
                **dict(row),
                "is_locked_reference_threshold": bool(abs(float(row["threshold"]) - float(locked_threshold)) < 1e-9),
                "is_best_primary_threshold": bool(primary is not None and abs(float(row["threshold"]) - float(primary["threshold"])) < 1e-9),
                "is_best_strict_threshold": bool(strict_best is not None and abs(float(row["threshold"]) - float(strict_best["threshold"])) < 1e-9),
            }
        )
    _append_dict_rows(
        out_dir / "validation_threshold_summary.csv",
        threshold_rows,
        [
            "epoch",
            "threshold",
            "center_precision",
            "center_recall",
            "center_f1",
            "center_precision_mean_samples",
            "center_recall_mean_samples",
            "center_f1_mean_samples",
            "predicted_center_count_mean",
            "predicted_center_count_median",
            "exact_center_count_accuracy",
            "strict_marker_contract_pass_count",
            "strict_marker_contract_pass_rate",
            "missing_gt_instances",
            "gt_instances_with_multiple_markers",
            "markers_outside_all_gt_instances",
            "localization_error_px",
            "localization_error_px_pooled_matches",
            "raw_component_count_mean",
            "raw_component_count_median",
            "fraction_raw_component_count_gt_3",
            "fraction_predicted_count_eq_3",
            "duplicate_markers_total",
            "markers_outside_all_gt_instances_total",
            "missing_gt_markers_total",
            "median_heatmap_margin",
            "fraction_heatmap_margin_gt_0",
            "predicted_count_distribution",
            "sample_count_gt1",
            "sample_count_gt2",
            "sample_count_gt3",
            "pass_count_gt1",
            "pass_count_gt2",
            "pass_count_gt3",
            "is_locked_reference_threshold",
            "is_best_primary_threshold",
            "is_best_strict_threshold",
        ],
    )


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
        center_feature=(cfg.get("model") or {}).get("center_feature", None),
    )

    init_path = (cfg.get("train") or {}).get("init_checkpoint", None)
    center_from_scratch = True
    model.semantic_init_report = {
        "checkpoint_path": str(init_path) if init_path else None,
        "missing_keys": [],
        "unexpected_keys": [],
        "allowed_missing_keys": [],
        "disallowed_missing_keys": [],
        "status": "not_loaded" if not init_path else "pending",
    }
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
        allowed_missing = [k for k in missing if _is_center_branch_param_name(model, str(k))]
        disallowed_missing = [k for k in missing if k not in allowed_missing]
        center_missing = [k for k in missing if _is_center_branch_param_name(model, str(k))]
        center_from_scratch = bool(len(center_missing) > 0)
        model.semantic_init_report = {
            "checkpoint_path": str(Path(init_path).resolve()),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "allowed_missing_keys": list(allowed_missing),
            "disallowed_missing_keys": list(disallowed_missing),
            "status": "exact" if (not disallowed_missing and not unexpected) else "mismatch",
        }
        if bool((cfg.get("train") or {}).get("require_exact_semantic_checkpoint_load", False)) and (disallowed_missing or unexpected):
            raise SystemExit(
                "Semantic checkpoint load is not exact: "
                f"disallowed_missing={len(disallowed_missing)} unexpected={len(unexpected)}"
            )

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
        normalization_mode = str(loss_cfg.get("normalization_mode", "legacy_num_pos")).strip().lower() or "legacy_num_pos"
        loss_fn = CenterNetFocalHeatmapLoss(alpha=alpha, beta=beta, normalization_mode=normalization_mode).to(device)
        return loss_fn, {"type": "centernet_focal", "alpha": alpha, "beta": beta, "normalization_mode": normalization_mode}

    pw = float((cfg.get("center") or {}).get("pos_weight", 0.0) or 0.0)
    if pw <= 0.0:
        pw = _compute_center_pos_weight(dataset_root, train_txt, thr=float((cfg.get("center") or {}).get("pos_weight_thr", 0.5)))
    pw = float(min(max(pw, 1.0), float((cfg.get("center") or {}).get("pos_weight_max", 1000.0))))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device)).to(device)
    return loss_fn, {"type": "bce", "pos_weight": pw}


def _freeze_base_enabled(cfg: dict) -> bool:
    return bool((cfg.get("train") or {}).get("freeze_base", False))


def _trainable_base_module_paths(cfg: dict) -> list[str]:
    vals = (cfg.get("train") or {}).get("trainable_base_modules", [])
    if vals in [None, ""]:
        return []
    if not isinstance(vals, list):
        raise SystemExit("Config: train.trainable_base_modules must be a list of module paths")
    out = []
    for x in vals:
        sx = str(x).strip()
        if sx:
            out.append(sx)
    return out


def _resolve_named_module(model: torch.nn.Module, module_path: str) -> torch.nn.Module:
    mods = dict(model.named_modules())
    mod = mods.get(str(module_path), None)
    if mod is None:
        raise SystemExit(f"Unknown module path in trainable_base_modules: {module_path}")
    return mod


def _param_names_for_module_path(model: torch.nn.Module, module_path: str) -> list[str]:
    prefix = str(module_path).strip() + "."
    return [n for (n, _p) in model.named_parameters() if n.startswith(prefix)]


def _parameter_count_by_names(model: torch.nn.Module, names: list[str]) -> int:
    want = set(str(n) for n in names)
    return int(sum(int(p.numel()) for n, p in model.named_parameters() if n in want))


def _partial_unfreeze_enabled(cfg: dict) -> bool:
    return _freeze_base_enabled(cfg) and len(_trainable_base_module_paths(cfg)) > 0


def _apply_training_policy(model: UnetPlusPlusSemanticCenterHead, cfg: dict) -> dict:
    freeze_base = _freeze_base_enabled(cfg)
    trainable_base_modules = _trainable_base_module_paths(cfg) if freeze_base else []

    for p in model.parameters():
        p.requires_grad = False
    for module in model.center_branch_modules():
        for p in module.parameters():
            p.requires_grad = True

    selected_param_names: list[str] = []
    for module_path in trainable_base_modules:
        _resolve_named_module(model, module_path)
        names = _param_names_for_module_path(model, module_path)
        if not names:
            raise SystemExit(f"Selected trainable module has no parameters: {module_path}")
        for n, p in model.named_parameters():
            if n in names:
                p.requires_grad = True
        selected_param_names.extend(names)

    model.freeze_base = bool(freeze_base)
    model.trainable_base_module_paths = list(trainable_base_modules)
    model.partial_unfreeze = bool(len(trainable_base_modules) > 0)

    total_params = int(sum(int(p.numel()) for p in model.parameters()))
    trainable_params = int(sum(int(p.numel()) for p in model.parameters() if bool(p.requires_grad)))
    trainable_names = [n for (n, p) in model.named_parameters() if bool(p.requires_grad)]
    allowed_prefixes = _center_branch_prefixes(model) + [f"{p}." for p in trainable_base_modules]
    assert all(any(n.startswith(pref) for pref in allowed_prefixes) for n in trainable_names), f"Unexpected trainable params found: {trainable_names[:10]}"

    center_param_names = [n for n, _p in model.named_parameters() if _is_center_branch_param_name(model, n)]
    decoder_param_names = sorted(set(selected_param_names))
    return {
        "freeze_base": bool(freeze_base),
        "partial_unfreeze": bool(len(trainable_base_modules) > 0),
        "trainable_base_modules": list(trainable_base_modules),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_names": trainable_names,
        "center_param_names": center_param_names,
        "decoder_param_names": decoder_param_names,
        "center_trainable_params": _parameter_count_by_names(model, center_param_names),
        "decoder_trainable_params": _parameter_count_by_names(model, decoder_param_names),
    }


def _set_train_modes(model: UnetPlusPlusSemanticCenterHead, *, freeze_base: bool) -> None:
    if freeze_base:
        model.train()
        model.base.eval()
        model.encoder.eval()
        model.segmentation_head.eval()
        model.base.decoder.eval()
        for module in model.center_branch_modules():
            module.train()
        for module_path in list(getattr(model, "trainable_base_module_paths", []) or []):
            _resolve_named_module(model, module_path).train()
    else:
        model.train()


def _apply_freeze_base(model: UnetPlusPlusSemanticCenterHead) -> dict:
    return _apply_training_policy(model, {"train": {"freeze_base": True}})


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


def _grad_l2_norm(params: list[torch.nn.Parameter]) -> float:
    s = 0.0
    for p in params:
        if p.grad is None:
            continue
        s += float(torch.sum(p.grad.detach().float() ** 2).item())
    return float(np.sqrt(max(s, 0.0)))


def _named_grad_l2_norm(named_params: list[tuple[str, torch.nn.Parameter]]) -> float:
    s = 0.0
    for _n, p in named_params:
        if p.grad is None:
            continue
        s += float(torch.sum(p.grad.detach().float() ** 2).item())
    return float(np.sqrt(max(s, 0.0)))


def _nonfinite_grad_tensor_count(named_params: list[tuple[str, torch.nn.Parameter]]) -> int:
    out = 0
    for _n, p in named_params:
        if p.grad is None:
            continue
        if not bool(torch.isfinite(p.grad.detach()).all().item()):
            out += 1
    return int(out)


def _all_parameters_finite(params: list[torch.nn.Parameter]) -> bool:
    for p in params:
        if not bool(torch.isfinite(p.detach()).all().item()):
            return False
    return True


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


def _name_matches_prefixes(name: str, prefixes: list[str]) -> bool:
    return any(str(name).startswith(str(p)) for p in prefixes)


def _max_bn_delta_filtered(model: torch.nn.Module, ref: list[tuple[str, torch.Tensor, torch.Tensor]], *, include_prefixes: list[str] | None = None, exclude_prefixes: list[str] | None = None) -> float | None:
    max_d = None
    for name, rm0, rv0 in ref:
        if include_prefixes is not None and not _name_matches_prefixes(name, include_prefixes):
            continue
        if exclude_prefixes is not None and _name_matches_prefixes(name, exclude_prefixes):
            continue
        m = dict(model.named_modules()).get(name, None)
        if m is None:
            continue
        rm = getattr(m, "running_mean", None)
        rv = getattr(m, "running_var", None)
        if rm is None or rv is None:
            continue
        d1 = float((rm.detach() - rm0).abs().max().item()) if rm.numel() else 0.0
        d2 = float((rv.detach() - rv0).abs().max().item()) if rv.numel() else 0.0
        cur = max(d1, d2)
        max_d = cur if max_d is None else max(float(max_d), float(cur))
    return float(max_d) if max_d is not None else None


def _snapshot_named_parameters(named_params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    return {str(n): p.detach().clone() for n, p in named_params}


def _max_parameter_delta_from_snapshot(named_params: list[tuple[str, torch.nn.Parameter]], snap: dict[str, torch.Tensor]) -> float:
    max_d = 0.0
    for n, p in named_params:
        ref = snap.get(str(n), None)
        if ref is None:
            continue
        d = float((p.detach() - ref).abs().max().item()) if p.numel() else 0.0
        max_d = max(max_d, d)
    return float(max_d)


def _count_present_grads(named_params: list[tuple[str, torch.nn.Parameter]]) -> int:
    return int(sum(1 for _n, p in named_params if p.grad is not None))


def _build_optimizer_groups(model: UnetPlusPlusSemanticCenterHead, cfg: dict, freeze_info: dict | None, *, freeze_base: bool) -> tuple[torch.optim.Optimizer, list[dict]]:
    base_lr = float((cfg.get("train") or {}).get("lr_backbone", cfg["train"]["lr"]))
    head_lr = float((cfg.get("train") or {}).get("lr_center_head", base_lr * 10.0))
    decoder_lr = float((cfg.get("train") or {}).get("lr_unfrozen_decoder", base_lr))
    wd = float(cfg["train"]["weight_decay"])

    group_specs: list[dict] = []
    if freeze_base:
        center_named = [(n, p) for n, p in model.named_parameters() if _is_center_branch_param_name(model, n) and p.requires_grad]
        decoder_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and (not _is_center_branch_param_name(model, n))]
        group_specs.append({"name": "center_head", "named_params": center_named, "lr": head_lr})
        if decoder_named:
            group_specs.append({"name": "unfrozen_decoder", "named_params": decoder_named, "lr": decoder_lr})
    else:
        params_base = [(n, p) for n, p in model.named_parameters() if p.requires_grad and not _is_center_branch_param_name(model, n)]
        params_head = [(n, p) for n, p in model.named_parameters() if p.requires_grad and _is_center_branch_param_name(model, n)]
        group_specs = [
            {"name": "base", "named_params": params_base, "lr": base_lr},
            {"name": "center_head", "named_params": params_head, "lr": head_lr},
        ]

    seen = set()
    for g in group_specs:
        for n, _p in g["named_params"]:
            if n in seen:
                raise SystemExit(f"Optimizer group overlap detected for parameter: {n}")
            seen.add(n)
    frozen_in_optimizer = [n for g in group_specs for n, p in g["named_params"] if not p.requires_grad]
    if frozen_in_optimizer:
        raise SystemExit(f"Frozen parameters included in optimizer: {frozen_in_optimizer[:10]}")

    optimizer = torch.optim.AdamW(
        [{"params": [p for _n, p in g["named_params"]], "lr": float(g["lr"])} for g in group_specs if g["named_params"]],
        weight_decay=wd,
    )
    meta = []
    for g in group_specs:
        meta.append(
            {
                "name": g["name"],
                "lr": float(g["lr"]),
                "param_count": int(sum(int(p.numel()) for _n, p in g["named_params"])),
                "parameter_names": [n for n, _p in g["named_params"]],
            }
        )
    return optimizer, meta


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


def _center_head_output_bias(model: torch.nn.Module) -> float | None:
    layer = model.center_head_output_layer()
    if layer is None or not hasattr(layer, "bias") or layer.bias is None:
        return None
    return float(layer.bias.detach().mean().item())


def _center_head_weight_norm(model: torch.nn.Module) -> float | None:
    layer = model.center_head_output_layer()
    if layer is None or not hasattr(layer, "weight") or layer.weight is None:
        return None
    return float(layer.weight.detach().float().norm().item())


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


def _center_fp32_enabled(cfg: dict) -> bool:
    return bool((cfg.get("train") or {}).get("center_fp32", False))


def _dtype_name(x: torch.Tensor | None) -> str | None:
    if x is None:
        return None
    return str(x.dtype).replace("torch.", "")


def _forward_base_for_center_training(
    *,
    model: UnetPlusPlusSemanticCenterHead,
    images: torch.Tensor,
    device: torch.device,
    amp_enabled_global: bool,
    detach_output: bool,
    no_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    ctx = torch.no_grad() if bool(no_grad) else torch.enable_grad()
    with ctx:
        with _autocast_ctx(device, enabled=amp_enabled_global):
            semantic_logits, decoder_output = model.forward_base(images)
    return semantic_logits, decoder_output.detach() if bool(detach_output) else decoder_output


def _forward_center_with_precision(
    *,
    model: UnetPlusPlusSemanticCenterHead,
    decoder_output: torch.Tensor,
    centers: torch.Tensor,
    center_loss_fn,
    device: torch.device,
    amp_enabled_global: bool,
    center_fp32: bool,
    detach_decoder_output: bool = True,
    return_details: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | dict, dict]:
    decoder_features = decoder_output.detach() if bool(detach_decoder_output) else decoder_output
    requested_center_feature_dtype = torch.float32 if bool(center_fp32) else None
    decoder_features = model.resolve_center_features(decoder_features, feature_dtype=requested_center_feature_dtype)
    center_autocast_enabled = bool(amp_enabled_global and (not center_fp32))
    with _autocast_ctx(device, enabled=center_autocast_enabled):
        center_logits = model.forward_center_from_features(decoder_features)
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
        "decoder_output_dtype_before_center_boundary": _dtype_name(decoder_output),
        "decoder_features_dtype": _dtype_name(decoder_features),
        "decoder_features_shape": list(decoder_features.shape) if torch.is_tensor(decoder_features) else None,
        "center_logits_dtype": _dtype_name(center_logits),
        "center_logits_shape": list(center_logits.shape) if torch.is_tensor(center_logits) else None,
        "center_loss_dtype": _dtype_name(center_loss),
        "center_feature_capture_info": dict(getattr(model, "center_feature_capture_info", lambda: {})() or {}),
        "center_feature_resolve_info": dict(getattr(model, "_last_center_resolve_info", {}) or {}),
        "center_primary_projection_weight_dtype": _dtype_name(getattr(getattr(model, "center_primary_projection", None), "weight", None)),
        "center_context_projection_weight_dtype": _dtype_name(getattr(getattr(model, "center_context_projection", None), "weight", None)),
        "center_fusion_adapter_weight_dtype": _dtype_name(getattr(getattr(getattr(model, "center_fusion_adapter", None), "__getitem__", lambda *_args, **_kwargs: None)(0) if getattr(model, "center_fusion_adapter", None) is not None else None, "weight", None)),
        "center_adapter_weight_dtype": _dtype_name(getattr(getattr(model, "center_adapter", None), "weight", None)),
        "center_head_output_weight_dtype": _dtype_name(getattr(model.center_head_output_layer(), "weight", None)),
    }
    return decoder_features, center_logits, payload, precision_info


def _forward_frozen_base(
    *,
    model: UnetPlusPlusSemanticCenterHead,
    images: torch.Tensor,
    device: torch.device,
    amp_enabled_global: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _forward_base_for_center_training(
        model=model,
        images=images,
        device=device,
        amp_enabled_global=amp_enabled_global,
        detach_output=True,
        no_grad=True,
    )


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
            thr_u8_001 = (ctr_prob[i, 0] >= 0.01).astype(np.uint8) * 255
            thr_u8_003 = (ctr_prob[i, 0] >= 0.03).astype(np.uint8) * 255
            thr_u8_01 = (ctr_prob[i, 0] >= 0.1).astype(np.uint8) * 255
            thr_u8_03 = (ctr_prob[i, 0] >= 0.3).astype(np.uint8) * 255

            leaf_union = pred_sem[i] == 1
            pred_pts_scored = _markers_from_center_map(ctr_prob[i, 0], leaf_union, float(center_thr), max_markers=3)
            pred_pts = [(y, x) for (y, x, _) in pred_pts_scored]

            mp = meta_paths[i] if i < len(meta_paths) else None
            gt_pts = _extract_metadata_centers(str(mp)) if isinstance(mp, str) and mp else []

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
                inst_bucket = "correct"
            if inst_bucket not in {"merged", "fragmented", "mixed", "correct"}:
                inst_bucket = "correct"

            buckets = [inst_bucket]
            if int(len(pred_pts)) == 0:
                buckets.append("zero_centers")
            if int(len(pred_pts)) > int(len(gt_pts)):
                buckets.append("extra_centers")
            if int(len(pred_pts)) != int(len(gt_pts)):
                buckets.append("incorrect_count")
            else:
                buckets.append("correct")

            for bucket in sorted(set(buckets)):
                sd = (out_root / bucket / sid).resolve()
                sd.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(sd / "original.png"), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(sd / "gt_center.png"), gt_center_u16)
                cv2.imwrite(str(sd / "pred_center_prob.png"), pr_center_u16)
                cv2.imwrite(str(sd / "thresholded_0p01.png"), thr_u8_001)
                cv2.imwrite(str(sd / "thresholded_0p03.png"), thr_u8_003)
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
            "center_precision_mean_samples": m.get("center_precision_mean_samples"),
            "center_recall_mean_samples": m.get("center_recall_mean_samples"),
            "center_f1_mean_samples": m.get("center_f1_mean_samples"),
            "center_count_acc": m.get("center_count_acc"),
            "exact_center_count_accuracy": m.get("center_count_acc"),
            "strict_marker_contract_pass_count": m.get("strict_marker_contract_pass_count"),
            "strict_marker_contract_pass_rate": m.get("strict_marker_contract_pass_rate"),
            "center_loc_err_px": m.get("center_loc_err_px"),
            "localization_error_px": m.get("center_loc_err_px"),
            "localization_error_px_pooled_matches": m.get("localization_error_px_pooled_matches", m.get("center_loc_err_px")),
            "center_zero_cases": m.get("center_zero_cases"),
            "center_extra_cases": m.get("center_extra_cases"),
            "raw_component_count_mean": m.get("raw_component_count_mean"),
            "raw_component_count_median": m.get("raw_component_count_median"),
            "fraction_raw_component_count_gt_3": m.get("fraction_raw_component_count_gt_3"),
            "fraction_predicted_count_eq_3": m.get("fraction_predicted_count_eq_3"),
            "duplicate_markers_total": m.get("duplicate_markers_total"),
            "markers_outside_all_gt_instances_total": m.get("markers_outside_all_gt_instances_total"),
            "missing_gt_markers_total": m.get("missing_gt_markers_total"),
            "median_heatmap_margin": m.get("median_heatmap_margin"),
            "fraction_heatmap_margin_gt_0": m.get("fraction_heatmap_margin_gt_0"),
            "predicted_count_distribution": m.get("predicted_count_distribution"),
            "instance_exact_count_acc": m.get("instance_exact_count_acc"),
            "instance_merged_rate": m.get("instance_merged_rate"),
            "instance_fragmented_rate": m.get("instance_fragmented_rate"),
            "instance_mean_matched_iou": m.get("instance_mean_matched_iou"),
            "instance_perfect_rate": m.get("instance_perfect_rate"),
            "instance_score": float(inst_score) if inst_score is not None else None,
            "sample_count_gt1": (m.get("per_gt_count_center_metrics") or {}).get("1", {}).get("sample_count"),
            "sample_count_gt2": (m.get("per_gt_count_center_metrics") or {}).get("2", {}).get("sample_count"),
            "sample_count_gt3": (m.get("per_gt_count_center_metrics") or {}).get("3", {}).get("sample_count"),
            "pass_count_gt1": (m.get("per_gt_count_center_metrics") or {}).get("1", {}).get("strict_marker_contract_pass_count"),
            "pass_count_gt2": (m.get("per_gt_count_center_metrics") or {}).get("2", {}).get("strict_marker_contract_pass_count"),
            "pass_count_gt3": (m.get("per_gt_count_center_metrics") or {}).get("3", {}).get("strict_marker_contract_pass_count"),
        }
        rows.append(row)
    return {
        "rows": rows,
        "best": _select_best_threshold_row(rows, primary_metric="center_f1_mean_samples"),
        "best_center_f1": _select_best_threshold_row(rows, primary_metric="center_f1_mean_samples"),
        "best_strict_marker_contract": _select_best_threshold_row(rows, primary_metric="strict_marker_contract_pass_rate"),
        "best_center_count_acc": _select_best_threshold_row(rows, primary_metric="exact_center_count_accuracy"),
        "best_instance_score": _select_best_threshold_row(rows, primary_metric="instance_score"),
    }


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
) -> dict | None:
    loss_cfg = cfg.get("center_loss") or {}
    if not isinstance(loss_cfg, dict):
        return None
    thr_list = loss_cfg.get("threshold_sweep", None)
    if not isinstance(thr_list, list) or not thr_list:
        return None
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
    return res


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
    center_fp32 = _center_fp32_enabled(cfg)
    print(f"AMP: {amp_enabled}")
    print(f"center_fp32: {center_fp32}")
    print(f"center_autocast_enabled: {bool(amp_enabled and (not center_fp32))}")
    print(f"center_grad_scaler_enabled: {bool(amp_enabled and (not center_fp32))}")
    print(f"batch_size: {int((cfg.get('train') or {}).get('batch_size', 1))}")

    freeze_base = _freeze_base_enabled(cfg)
    partial_unfreeze = _partial_unfreeze_enabled(cfg)
    train_loader, val_loader = _build_loaders(cfg, device=device)
    model = _build_model(cfg).to(device)
    if freeze_base:
        freeze_info = _apply_training_policy(model, cfg)
        print("=== FREEZE BASE ===")
        print(f"total_params: {freeze_info['total_params']}")
        print(f"trainable_params: {freeze_info['trainable_params']}")
        print(f"center_trainable_params: {freeze_info['center_trainable_params']}")
        print(f"decoder_trainable_params: {freeze_info['decoder_trainable_params']}")
        print(f"partial_unfreeze: {freeze_info['partial_unfreeze']}")
        for mp in freeze_info["trainable_base_modules"]:
            print(f"trainable_base_module: {mp}")
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

    clip_norm = float((cfg.get("train") or {}).get("center_grad_clip_norm", 0.0) or 0.0)
    optimizer, optimizer_meta = _build_optimizer_groups(model, cfg, freeze_info if freeze_base else None, freeze_base=freeze_base)

    steps = int((cfg.get("train") or {}).get("smoke_steps", 2))
    train_it = iter(train_loader)
    last = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    bn_ref = _collect_batchnorm_stats(model.base) if freeze_base else []
    center_named_params = [(n, p) for n, p in model.named_parameters() if _is_center_branch_param_name(model, n) and p.requires_grad]
    decoder_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and (not _is_center_branch_param_name(model, n))]
    frozen_named_params = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    encoder_frozen_named = [(n, p) for n, p in model.named_parameters() if n.startswith("base.encoder.") and not p.requires_grad]
    decoder_frozen_named = [(n, p) for n, p in model.named_parameters() if n.startswith("base.decoder.") and not p.requires_grad]
    semantic_head_frozen_named = [(n, p) for n, p in model.named_parameters() if n.startswith("base.segmentation_head.") and not p.requires_grad]
    context_module_path = str((((cfg.get("model") or {}).get("center_feature") or {}).get("context_module_path", "")) or "").strip()
    context_frozen_named = [(n, p) for n, p in model.named_parameters() if context_module_path and n.startswith(context_module_path + ".") and not p.requires_grad]
    for _ in range(int(steps)):
        batch = next(train_it)
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        centers = batch["center"].to(device)
        optimizer.zero_grad(set_to_none=True)
        _set_train_modes(model, freeze_base=freeze_base)
        with torch.no_grad():
            if freeze_base:
                sem_before, _decoder_before = _forward_base_for_center_training(
                    model=model,
                    images=images,
                    device=device,
                    amp_enabled_global=amp_enabled,
                    detach_output=False,
                    no_grad=True,
                )
            else:
                sem_before = None
        center_snapshot = _snapshot_named_parameters(center_named_params)
        decoder_snapshot = _snapshot_named_parameters(decoder_named_params)
        frozen_snapshot = _snapshot_named_parameters(frozen_named_params)
        if freeze_base:
            sem_logits, decoder_output = _forward_base_for_center_training(
                model=model,
                images=images,
                device=device,
                amp_enabled_global=amp_enabled,
                detach_output=not partial_unfreeze,
                no_grad=not partial_unfreeze,
            )
            _decoder_features, center_logits, center_payload, precision_info = _forward_center_with_precision(
                model=model,
                decoder_output=decoder_output,
                centers=centers,
                center_loss_fn=center_loss_fn,
                device=device,
                amp_enabled_global=amp_enabled,
                center_fp32=center_fp32,
                detach_decoder_output=not partial_unfreeze,
                return_details=isinstance(center_loss_fn, CenterNetFocalHeatmapLoss),
            )
            if isinstance(center_loss_fn, CenterNetFocalHeatmapLoss):
                assert isinstance(center_payload, dict)
                details = center_payload
                loss_center = details["loss"]
            else:
                details = None
                assert torch.is_tensor(center_payload)
                loss_center = center_payload
        else:
            out = model(images)
            sem_logits = out["semantic"]
            center_logits = out["center"]
            precision_info = {
                "amp_enabled_global": bool(amp_enabled),
                "center_fp32": bool(center_fp32),
                "center_autocast_enabled": bool(amp_enabled),
                "center_grad_scaler_enabled": bool(amp_enabled),
                "decoder_features_dtype": None,
                "center_logits_dtype": _dtype_name(center_logits),
                "center_loss_dtype": None,
            }
        semantic_logits_for_loss = sem_logits.float()
        semantic_targets_for_loss = masks.long()
        loss_sem = semantic_loss_fn(semantic_logits_for_loss, semantic_targets_for_loss)
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

            if not freeze_base:
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
            if not freeze_base:
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
        center_grad_norm_before = _named_grad_l2_norm(center_named_params)
        decoder_grad_norm_before = _named_grad_l2_norm(decoder_named_params)
        combined_grad_norm_before = _grad_l2_norm([p for _n, p in center_named_params + decoder_named_params])
        nonfinite_grad_tensors = _nonfinite_grad_tensor_count(center_named_params + decoder_named_params)
        if nonfinite_grad_tensors > 0:
            raise SystemExit("Smoke test failed: non-finite trainable gradients")
        if (center_grad_norm_before <= 0.0) or (partial_unfreeze and decoder_grad_norm_before <= 0.0):
            raise SystemExit("Smoke test failed: missing trainable gradients")

        if float(clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_([p for _n, p in center_named_params + decoder_named_params], max_norm=float(clip_norm))
        center_grad_norm_after = _named_grad_l2_norm(center_named_params)
        decoder_grad_norm_after = _named_grad_l2_norm(decoder_named_params)
        combined_grad_norm_after = _grad_l2_norm([p for _n, p in center_named_params + decoder_named_params])

        optimizer.step()
        with torch.no_grad():
            if freeze_base:
                sem_after, _decoder_after = _forward_base_for_center_training(
                    model=model,
                    images=images,
                    device=device,
                    amp_enabled_global=amp_enabled,
                    detach_output=False,
                    no_grad=True,
                )
            else:
                sem_after = None
        sem_delta = None
        if freeze_base and sem_before is not None and sem_after is not None:
            sem_delta = float((sem_before - sem_after).abs().max().item())
        params_finite = bool(_all_parameters_finite([p for _n, p in center_named_params + decoder_named_params]))
        logits_finite = bool(torch.isfinite(center_logits.detach()).all().item())
        base_grad_any = bool(any(p.grad is not None for p in model.base.parameters()))
        frozen_bn_delta = _max_bn_delta_filtered(model.base, bn_ref, exclude_prefixes=[p.replace("base.", "", 1) for p in freeze_info["trainable_base_modules"]]) if freeze_base else None
        selected_bn_delta = _max_bn_delta_filtered(model.base, bn_ref, include_prefixes=[p.replace("base.", "", 1) for p in freeze_info["trainable_base_modules"]]) if freeze_base and partial_unfreeze else None
        selected_decoder_delta = _max_parameter_delta_from_snapshot(decoder_named_params, decoder_snapshot)
        center_head_delta = _max_parameter_delta_from_snapshot(center_named_params, center_snapshot)
        frozen_param_max_delta = _max_parameter_delta_from_snapshot(frozen_named_params, frozen_snapshot)
        last = {
            "semantic_shape": tuple(sem_logits.shape),
            "center_shape": tuple(center_logits.shape),
            "optimizer_center_lr": float(optimizer.param_groups[0]["lr"]) if freeze_base else float(optimizer.param_groups[1]["lr"]),
            "optimizer_decoder_lr": float(next((g["lr"] for g in optimizer_meta if g["name"] == "unfrozen_decoder"), 0.0)) if freeze_base else float(optimizer.param_groups[0]["lr"]),
            "loss_semantic": float(loss_sem.item()),
            "loss_center": float(loss_center.item()),
            "loss_total": float(loss.item()),
            "center_grad_norm_before_clip": float(center_grad_norm_before),
            "decoder_grad_norm_before_clip": float(decoder_grad_norm_before),
            "combined_grad_norm_before_clip": float(combined_grad_norm_before),
            "center_grad_norm_after_clip": float(center_grad_norm_after),
            "decoder_grad_norm_after_clip": float(decoder_grad_norm_after),
            "combined_grad_norm_after_clip": float(combined_grad_norm_after),
            "center_grad_all_finite": True,
            "base_grad_any": bool(base_grad_any),
            "base_eval_mode": bool(not model.base.training),
            "encoder_eval_mode": bool(not model.encoder.training),
            "decoder_eval_mode": bool(not model.base.decoder.training),
            "segmentation_head_eval_mode": bool(not model.segmentation_head.training),
            "center_train_mode": bool(all(module.training for module in model.center_branch_modules())),
            "selected_decoder_train_mode": bool(all(_resolve_named_module(model, mp).training for mp in freeze_info["trainable_base_modules"])) if freeze_base and partial_unfreeze else None,
            "semantic_logits_max_abs_delta_after_step": sem_delta,
            "frozen_bn_running_stats_max_abs_delta_after_step": frozen_bn_delta,
            "selected_block_bn_running_stats_max_abs_delta_after_step": selected_bn_delta,
            "center_loss": center_loss_info,
            "lambda_center": float(lambda_center),
            "freeze_base": bool(freeze_base),
            "partial_unfreeze": bool(partial_unfreeze),
            "trainable_parameter_names": list(freeze_info["trainable_names"]) if freeze_base and freeze_info is not None else [n for n, _p in center_named_params + decoder_named_params],
            "total_trainable_parameter_count": int(freeze_info["trainable_params"]) if freeze_base and freeze_info is not None else int(sum(int(p.numel()) for _n, p in center_named_params + decoder_named_params)),
            "center_head_trainable_parameter_count": int(freeze_info["center_trainable_params"]) if freeze_base and freeze_info is not None else int(sum(int(p.numel()) for _n, p in center_named_params)),
            "decoder_block_trainable_parameter_count": int(freeze_info["decoder_trainable_params"]) if freeze_base and freeze_info is not None else int(sum(int(p.numel()) for _n, p in decoder_named_params)),
            "trainable_base_modules": list(freeze_info["trainable_base_modules"]) if freeze_base and freeze_info is not None else [],
            "trainable_parameter_groups": optimizer_meta,
            "optimizer_group_overlap": False,
            "optimizer_frozen_parameter_count": 0,
            "nonfinite_gradient_tensors": int(nonfinite_grad_tensors),
            "frozen_encoder_grad_count": int(_count_present_grads(encoder_frozen_named)),
            "frozen_decoder_grad_count": int(_count_present_grads(decoder_frozen_named)),
            "semantic_head_grad_count": int(_count_present_grads(semantic_head_frozen_named)),
            "context_feature_frozen_grad_count": int(_count_present_grads(context_frozen_named)) if context_module_path else None,
            "selected_decoder_parameter_delta": float(selected_decoder_delta),
            "center_head_parameter_delta": float(center_head_delta),
            "frozen_parameter_max_delta": float(frozen_param_max_delta),
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
            "semantic_logits_original_dtype": _dtype_name(sem_logits),
            "semantic_logits_loss_dtype": _dtype_name(semantic_logits_for_loss),
            "semantic_target_dtype": _dtype_name(semantic_targets_for_loss),
            "semantic_loss_dtype": _dtype_name(loss_sem),
            "semantic_loss_finite": bool(torch.isfinite(loss_sem.detach()).all().item()),
            **precision_info,
        }

    if freeze_base:
        if (not partial_unfreeze) and bool(last.get("base_grad_any", False)):
            raise SystemExit("Freeze smoke test failed: base_grad_any=true")
        if not bool(last.get("base_eval_mode", False)):
            raise SystemExit("Freeze smoke test failed: base is not in eval mode")
        if not bool(last.get("encoder_eval_mode", False)):
            raise SystemExit("Freeze smoke test failed: encoder is not in eval mode")
        if not bool(last.get("decoder_eval_mode", False)):
            raise SystemExit("Freeze smoke test failed: decoder is not in eval mode")
        if not bool(last.get("segmentation_head_eval_mode", False)):
            raise SystemExit("Freeze smoke test failed: segmentation head is not in eval mode")
        if not bool(last.get("center_train_mode", False)):
            raise SystemExit("Freeze smoke test failed: center_head is not in train mode")
        capture_info = dict(last.get("center_feature_capture_info") or {})
        primary_capture_shape = capture_info.get("captured_shape")
        context_capture_shape = ((capture_info.get("context") or {}).get("captured_shape") if isinstance(capture_info.get("context"), dict) else None)
        if primary_capture_shape is None:
            raise SystemExit("Freeze smoke test failed: primary center feature was not captured")
        if context_module_path and context_capture_shape is None:
            raise SystemExit("Freeze smoke test failed: contextual center feature was not captured")
        if partial_unfreeze and not bool(last.get("selected_decoder_train_mode", False)):
            raise SystemExit("Partial unfreeze smoke failed: selected decoder block is not in train mode")
        if (not partial_unfreeze) and ((last.get("semantic_logits_max_abs_delta_after_step") is None) or float(last["semantic_logits_max_abs_delta_after_step"]) != 0.0):
            raise SystemExit(f"Freeze smoke test failed: semantic logits changed (delta={last.get('semantic_logits_max_abs_delta_after_step')})")
        if (last.get("frozen_bn_running_stats_max_abs_delta_after_step") is None) or float(last["frozen_bn_running_stats_max_abs_delta_after_step"]) != 0.0:
            raise SystemExit(f"Freeze smoke test failed: frozen BatchNorm stats changed (delta={last.get('frozen_bn_running_stats_max_abs_delta_after_step')})")
        if float(last.get("center_grad_norm_before_clip", 0.0)) <= 0.0:
            raise SystemExit("Freeze smoke test failed: center grad is zero")
        if partial_unfreeze and float(last.get("selected_decoder_parameter_delta", 0.0)) <= 0.0:
            raise SystemExit("Partial unfreeze smoke failed: selected decoder did not update")
        if partial_unfreeze and float(last.get("center_head_parameter_delta", 0.0)) <= 0.0:
            raise SystemExit("Partial unfreeze smoke failed: center head did not update")
        if context_module_path and int(last.get("context_feature_frozen_grad_count", -1)) != 0:
            raise SystemExit("Partial unfreeze smoke failed: frozen contextual feature block received gradients")
        if float(last.get("frozen_parameter_max_delta", 0.0)) != 0.0:
            raise SystemExit(f"Partial unfreeze smoke failed: frozen parameter changed (delta={last.get('frozen_parameter_max_delta')})")

    model.eval()
    val_it = iter(val_loader)
    val_losses = []
    with torch.no_grad():
        for _ in range(2):
            vb = next(val_it)
            out = model(vb["image"].to(device))
            v_sem = out["semantic"]
            v_ctr = out["center"]
            v_sem_loss_logits = v_sem.float()
            v_sem_loss_target = vb["mask"].to(device).long()
            v_loss_sem = semantic_loss_fn(v_sem_loss_logits, v_sem_loss_target)
            v_loss_center = center_loss_fn(v_ctr, vb["center"].to(device))
            v_loss = v_loss_sem + float(lambda_center) * v_loss_center
            val_losses.append(
                {
                    "val_semantic_shape": tuple(v_sem.shape),
                    "val_center_shape": tuple(v_ctr.shape),
                    "val_loss_semantic": float(v_loss_sem.item()),
                    "val_loss_center": float(v_loss_center.item()),
                    "val_loss_total": float(v_loss.item()),
                    "val_semantic_logits_original_dtype": _dtype_name(v_sem),
                    "val_semantic_logits_loss_dtype": _dtype_name(v_sem_loss_logits),
                    "val_semantic_target_dtype": _dtype_name(v_sem_loss_target),
                    "val_semantic_loss_dtype": _dtype_name(v_loss_sem),
                    "val_semantic_loss_finite": bool(torch.isfinite(v_loss_sem).all().item()),
                }
            )
    last["val_batches"] = val_losses
    if device.type == "cuda":
        last["peak_vram_gb"] = float(torch.cuda.max_memory_allocated() / (1024**3))
    return last


def train(cfg: dict, device: torch.device) -> None:
    out_dir = _get_save_dir(cfg)
    _ensure_save_dir_compatible(out_dir, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(out_dir / "config.json", cfg)

    semantic_ckpt = Path((cfg.get("train") or {}).get("init_checkpoint", "")).resolve()
    train_manifest = Path((cfg.get("dataset") or {}).get("train_manifest", "")).resolve() if (cfg.get("dataset") or {}).get("train_manifest") else None
    val_manifest = Path((cfg.get("dataset") or {}).get("val_manifest", "")).resolve() if (cfg.get("dataset") or {}).get("val_manifest") else None
    identities = {
        "semantic_checkpoint_path": str(semantic_ckpt),
        "semantic_checkpoint_sha256": _sha256_file(semantic_ckpt) if semantic_ckpt.exists() else None,
        "expected_semantic_checkpoint_sha256": (cfg.get("train") or {}).get("init_checkpoint_sha256"),
        "train_manifest_path": str(train_manifest) if train_manifest else None,
        "train_manifest_sha256": _sha256_file(train_manifest) if train_manifest and train_manifest.exists() else None,
        "expected_train_manifest_sha256": (cfg.get("dataset") or {}).get("train_manifest_sha256"),
        "val_manifest_path": str(val_manifest) if val_manifest else None,
        "val_manifest_sha256": _sha256_file(val_manifest) if val_manifest and val_manifest.exists() else None,
        "expected_val_manifest_sha256": (cfg.get("dataset") or {}).get("val_manifest_sha256"),
        "device": str(device),
    }
    if identities["expected_semantic_checkpoint_sha256"] and identities["semantic_checkpoint_sha256"] != identities["expected_semantic_checkpoint_sha256"]:
        raise SystemExit("Semantic checkpoint identity mismatch")
    if identities["expected_train_manifest_sha256"] and identities["train_manifest_sha256"] != identities["expected_train_manifest_sha256"]:
        raise SystemExit("Train manifest identity mismatch")
    if identities["expected_val_manifest_sha256"] and identities["val_manifest_sha256"] != identities["expected_val_manifest_sha256"]:
        raise SystemExit("Validation manifest identity mismatch")
    _write_json_atomic(out_dir / "preflight_identities.json", identities)

    freeze_base = _freeze_base_enabled(cfg)
    partial_unfreeze = _partial_unfreeze_enabled(cfg)
    train_loader, val_loader = _build_loaders(cfg, device=device)
    model = _build_model(cfg).to(device)
    freeze_info = None
    if freeze_base:
        freeze_info = _apply_training_policy(model, cfg)
        print("=== FREEZE BASE ===")
        print(f"total_params: {freeze_info['total_params']}")
        print(f"trainable_params: {freeze_info['trainable_params']}")
        print(f"center_trainable_params: {freeze_info['center_trainable_params']}")
        print(f"decoder_trainable_params: {freeze_info['decoder_trainable_params']}")
        print(f"partial_unfreeze: {freeze_info['partial_unfreeze']}")
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

    optimizer, optimizer_meta = _build_optimizer_groups(model, cfg, freeze_info if freeze_base else None, freeze_base=freeze_base)
    for g in optimizer_meta:
        print(f"optimizer_group: {g['name']} lr={g['lr']} param_count={g['param_count']}")

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
    center_fp32 = _center_fp32_enabled(cfg)
    print(f"amp_enabled_global: {amp_enabled}")
    print(f"center_fp32: {center_fp32}")
    print(f"center_autocast_enabled: {bool(amp_enabled and (not center_fp32))}")
    print(f"center_grad_scaler_enabled: {bool(amp_enabled and (not center_fp32))}")
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and (not center_fp32)))

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
                    "semantic_mean_dice_fg_delta_from_init",
                    "semantic_mean_dice_fg_abs_delta_from_init",
                    "semantic_dice_leaflet_delta_from_init",
                    "semantic_dice_ring_delta_from_init",
                    "center_f1",
                    "center_f1_mean_samples",
                    "center_precision",
                    "center_recall",
                    "strict_marker_contract_pass_rate",
                    "strict_marker_contract_pass_count",
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
                    "sweep_best_threshold_center_f1",
                    "sweep_best_center_f1",
                    "sweep_best_threshold_center_count_acc",
                    "sweep_best_center_count_acc",
                    "sweep_best_threshold_instance_score",
                    "sweep_best_instance_score",
                    "sweep_best_instance_mean_matched_iou",
                    "scheduler_monitor_name",
                    "scheduler_monitor_threshold_context",
                    "scheduler_monitor_value",
                    "checkpoint_selection_metric_name",
                    "checkpoint_selection_threshold_context",
                    "checkpoint_selection_metric_value",
                    "early_stop_metric_name",
                    "early_stop_threshold_context",
                    "early_stop_reset_policy",
                    "early_stop_metric_value",
                    "lr_before_scheduler_step",
                    "lr_after_scheduler_step",
                    "scheduler_best",
                    "scheduler_num_bad_epochs",
                    "scheduler_cooldown_counter",
                    "train_grad_norm_mean_before_clip",
                    "train_grad_norm_max_before_clip",
                    "train_center_grad_norm_mean_before_clip",
                    "train_decoder_grad_norm_mean_before_clip",
                    "train_combined_grad_norm_mean_before_clip",
                    "train_finite_grad_norm_mean_before_clip",
                    "train_finite_grad_norm_max_before_clip",
                    "train_nonfinite_grad_batch_count",
                    "train_skipped_optimizer_step_count",
                    "train_clipped_batch_count",
                    "train_batches_clipped_pct",
                    "train_grad_norm_mean_after_clip",
                    "train_center_grad_norm_mean_after_clip",
                    "train_decoder_grad_norm_mean_after_clip",
                    "train_combined_grad_norm_mean_after_clip",
                    "amp_scale_min",
                    "amp_scale_max",
                    "amp_scale_last",
                    "train_loss_is_finite",
                    "parameters_finite",
                    "train_logits_finite",
                    "train_decoder_features_dtype",
                    "train_center_logits_dtype",
                    "train_center_loss_dtype",
                    "train_center_grad_scaler_enabled",
                    "center_head_weight_norm",
                    "center_head_output_bias",
                    "lr_unfrozen_decoder",
                    "lr_backbone",
                    "lr_center_head",
                ]
            )

    instance_root = Path((cfg.get("dataset") or {}).get("instance_root", "datasets/converted_leaflet_instances")).resolve()

    best_mean_fg = None
    best_center_f1 = None
    best_center_count_acc = None
    best_instance = None
    best_primary_candidate = None
    best_strict_candidate = None
    best_epoch_mean_fg = None
    best_epoch_center = None
    best_epoch_center_count_acc = None
    best_epoch_instance = None
    best_epoch_primary = None
    best_epoch_strict = None
    no_improve = 0

    center_thr = float((cfg.get("center") or {}).get("marker_thr", 0.3))
    primary_metric_key = str((cfg.get("train") or {}).get("checkpoint_selection_metric", "center_f1_mean_samples"))
    strict_metric_key = _strict_checkpoint_metric_key(cfg)
    semantic_mean_fg0 = None
    semantic_degradation_streak = 0

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
    sweep0 = _maybe_run_threshold_sweep(
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
    ) if freeze_base else None
    sweep0_center = (sweep0 or {}).get("best_center_f1") if isinstance(sweep0, dict) else None
    sweep0_count = (sweep0 or {}).get("best_center_count_acc") if isinstance(sweep0, dict) else None
    sweep0_inst = (sweep0 or {}).get("best_instance_score") if isinstance(sweep0, dict) else None

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
                0.0 if mean_fg0 is not None else "",
                0.0 if mean_fg0 is not None else "",
                0.0 if isinstance(val_metrics0.get("dice"), list) and len(val_metrics0["dice"]) > 1 else "",
                0.0 if isinstance(val_metrics0.get("dice"), list) and len(val_metrics0["dice"]) > 2 else "",
                float(val_metrics0.get("center_f1")) if val_metrics0.get("center_f1") is not None else "",
                float(val_metrics0.get("center_f1_mean_samples")) if val_metrics0.get("center_f1_mean_samples") is not None else "",
                float(val_metrics0.get("center_precision")) if val_metrics0.get("center_precision") is not None else "",
                float(val_metrics0.get("center_recall")) if val_metrics0.get("center_recall") is not None else "",
                float(val_metrics0.get("strict_marker_contract_pass_rate")) if val_metrics0.get("strict_marker_contract_pass_rate") is not None else "",
                int(val_metrics0.get("strict_marker_contract_pass_count")) if val_metrics0.get("strict_marker_contract_pass_count") is not None else "",
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
                float(sweep0_center.get("threshold")) if isinstance(sweep0_center, dict) and sweep0_center.get("threshold") is not None else "",
                float(sweep0_center.get("center_f1")) if isinstance(sweep0_center, dict) and sweep0_center.get("center_f1") is not None else "",
                float(sweep0_count.get("threshold")) if isinstance(sweep0_count, dict) and sweep0_count.get("threshold") is not None else "",
                float(sweep0_count.get("center_count_acc")) if isinstance(sweep0_count, dict) and sweep0_count.get("center_count_acc") is not None else "",
                float(sweep0_inst.get("threshold")) if isinstance(sweep0_inst, dict) and sweep0_inst.get("threshold") is not None else "",
                float(sweep0_inst.get("instance_score")) if isinstance(sweep0_inst, dict) and sweep0_inst.get("instance_score") is not None else "",
                float(sweep0_inst.get("instance_mean_matched_iou")) if isinstance(sweep0_inst, dict) and sweep0_inst.get("instance_mean_matched_iou") is not None else "",
                str((scheduler_cfg or {}).get("monitor", early_monitor)) if scheduler is not None else "",
                _locked_threshold_context_label(center_thr) if scheduler is not None else "",
                "",
                str(primary_metric_key),
                _best_sweep_threshold_context_label() if freeze_base else _locked_threshold_context_label(center_thr),
                float(sweep0_center.get(primary_metric_key)) if isinstance(sweep0_center, dict) and sweep0_center.get(primary_metric_key) is not None else (float(val_metrics0.get(primary_metric_key)) if val_metrics0.get(primary_metric_key) is not None else ""),
                str(early_monitor),
                _locked_threshold_context_label(center_thr),
                "any_checkpoint_improvement_resets_patience",
                float(val_metrics0.get(early_monitor)) if val_metrics0.get(early_monitor) is not None else (float(inst_score0) if early_monitor == "instance_score" and inst_score0 is not None else ""),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                float(scaler.get_scale()) if amp_enabled else "",
                float(scaler.get_scale()) if amp_enabled else "",
                float(scaler.get_scale()) if amp_enabled else "",
                True,
                bool(_all_parameters_finite(list(model.parameters()))),
                True,
                "",
                "",
                "",
                bool(amp_enabled and (not center_fp32)),
                float(_center_head_weight_norm(model)) if _center_head_weight_norm(model) is not None else "",
                float(_center_head_output_bias(model)) if _center_head_output_bias(model) is not None else "",
                float(next((g["lr"] for g in optimizer_meta if g["name"] == "unfrozen_decoder"), 0.0)) if partial_unfreeze else "",
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
            center_thr=float(sweep0_center.get("threshold")) if isinstance(sweep0_center, dict) and sweep0_center.get("threshold") is not None else center_thr,
            tag="epoch0",
            max_samples=20,
        )
        (out_dir / "epoch0_metrics.json").write_text(json.dumps({"val_metrics": val_metrics0, "instance_score": inst_score0, "threshold_sweep": sweep0}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_validation_reports(out_dir, epoch=0, val_metrics=val_metrics0, sweep_res=sweep0, locked_threshold=center_thr)

    for epoch in range(1, epochs + 1):
        _set_train_modes(model, freeze_base=freeze_base)
        running = 0.0
        n_batches = 0
        grad_norm_before_sum = 0.0
        grad_norm_before_max = 0.0
        grad_norm_after_sum = 0.0
        finite_grad_norm_before_sum = 0.0
        finite_grad_norm_before_max = 0.0
        nonfinite_grad_batches = 0
        skipped_optimizer_steps = 0
        clipped_batches = 0
        amp_scale_min = None
        amp_scale_max = None
        amp_scale_last = None
        train_loss_is_finite = True
        train_logits_finite = True
        last_precision_info = None
        t0 = time.perf_counter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        center_named_params = [(n, p) for n, p in model.named_parameters() if _is_center_branch_param_name(model, n) and p.requires_grad]
        decoder_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and (not _is_center_branch_param_name(model, n))]
        trainable_named_params = center_named_params + decoder_named_params
        center_grad_before_sum = 0.0
        decoder_grad_before_sum = 0.0
        combined_grad_before_sum = 0.0
        center_grad_after_sum = 0.0
        decoder_grad_after_sum = 0.0
        combined_grad_after_sum = 0.0
        clip_norm = float((cfg.get("train") or {}).get("center_grad_clip_norm", 0.0) or 0.0)
        for bi, batch in enumerate(pbar, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            centers = batch["center"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if freeze_base:
                sem_logits, decoder_output = _forward_base_for_center_training(
                    model=model,
                    images=images,
                    device=device,
                    amp_enabled_global=amp_enabled,
                    detach_output=not partial_unfreeze,
                    no_grad=not partial_unfreeze,
                )
                _decoder_features, center_logits, center_payload, precision_info = _forward_center_with_precision(
                    model=model,
                    decoder_output=decoder_output,
                    centers=centers,
                    center_loss_fn=center_loss_fn,
                    device=device,
                    amp_enabled_global=amp_enabled,
                    center_fp32=center_fp32,
                    detach_decoder_output=not partial_unfreeze,
                    return_details=False,
                )
                assert torch.is_tensor(center_payload)
                loss_center = center_payload
                loss_sem = semantic_loss_fn(sem_logits.float(), masks.long())
                loss = loss_center
            else:
                with _autocast_ctx(device, enabled=amp_enabled):
                    out = model(images)
                    sem_logits = out["semantic"]
                    center_logits = out["center"]
                    loss_sem = semantic_loss_fn(sem_logits.float(), masks.long())
                    loss_center = center_loss_fn(center_logits, centers)
                    loss = loss_sem + float(lambda_center) * loss_center
                precision_info = {
                    "amp_enabled_global": bool(amp_enabled),
                    "center_fp32": bool(center_fp32),
                    "center_autocast_enabled": bool(amp_enabled),
                    "center_grad_scaler_enabled": bool(amp_enabled),
                    "decoder_features_dtype": None,
                    "decoder_features_shape": None,
                    "center_logits_dtype": _dtype_name(center_logits),
                    "center_logits_shape": list(center_logits.shape) if torch.is_tensor(center_logits) else None,
                    "center_loss_dtype": _dtype_name(loss_center),
                    "center_feature_capture_info": dict(getattr(model, "center_feature_capture_info", lambda: {})() or {}),
                }
            last_precision_info = precision_info
            if not bool(torch.isfinite(loss.detach()).all().item()):
                train_loss_is_finite = False
            if not bool(torch.isfinite(center_logits.detach()).all().item()):
                train_logits_finite = False

            if bool(scaler.is_enabled()):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                center_before = _named_grad_l2_norm(center_named_params)
                decoder_before = _named_grad_l2_norm(decoder_named_params)
                grad_before = _grad_l2_norm([p for _n, p in trainable_named_params])
                grad_norm_before_sum += float(grad_before)
                center_grad_before_sum += float(center_before)
                decoder_grad_before_sum += float(decoder_before)
                combined_grad_before_sum += float(grad_before)
                if np.isfinite(grad_before):
                    finite_grad_norm_before_sum += float(grad_before)
                    finite_grad_norm_before_max = max(float(finite_grad_norm_before_max), float(grad_before))
                    grad_norm_before_max = max(float(grad_norm_before_max), float(grad_before))
                else:
                    nonfinite_grad_batches += 1
                    grad_norm_before_max = float("inf")
                if float(clip_norm) > 0.0 and np.isfinite(grad_before):
                    if float(grad_before) > float(clip_norm):
                        clipped_batches += 1
                    torch.nn.utils.clip_grad_norm_([p for _n, p in trainable_named_params], max_norm=float(clip_norm))
                grad_after = _grad_l2_norm([p for _n, p in trainable_named_params])
                center_after = _named_grad_l2_norm(center_named_params)
                decoder_after = _named_grad_l2_norm(decoder_named_params)
                grad_norm_after_sum += float(grad_after)
                center_grad_after_sum += float(center_after)
                decoder_grad_after_sum += float(decoder_after)
                combined_grad_after_sum += float(grad_after)
                prev_scale = float(scaler.get_scale())
                amp_scale_min = prev_scale if amp_scale_min is None else min(float(amp_scale_min), prev_scale)
                amp_scale_max = prev_scale if amp_scale_max is None else max(float(amp_scale_max), prev_scale)
                scaler.step(optimizer)
                scaler.update()
                new_scale = float(scaler.get_scale())
                amp_scale_min = new_scale if amp_scale_min is None else min(float(amp_scale_min), new_scale)
                amp_scale_max = new_scale if amp_scale_max is None else max(float(amp_scale_max), new_scale)
                amp_scale_last = new_scale
                if float(new_scale) < float(prev_scale):
                    skipped_optimizer_steps += 1
            else:
                loss.backward()
                center_before = _named_grad_l2_norm(center_named_params)
                decoder_before = _named_grad_l2_norm(decoder_named_params)
                grad_before = _grad_l2_norm([p for _n, p in trainable_named_params])
                grad_norm_before_sum += float(grad_before)
                center_grad_before_sum += float(center_before)
                decoder_grad_before_sum += float(decoder_before)
                combined_grad_before_sum += float(grad_before)
                if np.isfinite(grad_before):
                    finite_grad_norm_before_sum += float(grad_before)
                    finite_grad_norm_before_max = max(float(finite_grad_norm_before_max), float(grad_before))
                    grad_norm_before_max = max(float(grad_norm_before_max), float(grad_before))
                else:
                    nonfinite_grad_batches += 1
                    grad_norm_before_max = float("inf")
                if float(clip_norm) > 0.0 and np.isfinite(grad_before):
                    if float(grad_before) > float(clip_norm):
                        clipped_batches += 1
                    torch.nn.utils.clip_grad_norm_([p for _n, p in trainable_named_params], max_norm=float(clip_norm))
                grad_after = _grad_l2_norm([p for _n, p in trainable_named_params])
                center_after = _named_grad_l2_norm(center_named_params)
                decoder_after = _named_grad_l2_norm(decoder_named_params)
                grad_norm_after_sum += float(grad_after)
                center_grad_after_sum += float(center_after)
                decoder_grad_after_sum += float(decoder_after)
                combined_grad_after_sum += float(grad_after)
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
        dice_leaflet = float(val_metrics["dice"][1]) if isinstance(val_metrics.get("dice"), list) and len(val_metrics["dice"]) > 1 else None
        dice_ring = float(val_metrics["dice"][2]) if isinstance(val_metrics.get("dice"), list) and len(val_metrics["dice"]) > 2 else None
        semantic_mean_fg_delta = (float(mean_fg) - float(semantic_mean_fg0)) if (semantic_mean_fg0 is not None and mean_fg is not None) else None
        semantic_mean_fg_abs_delta = abs(float(semantic_mean_fg_delta)) if semantic_mean_fg_delta is not None else None
        semantic_dice_leaflet_delta = (float(dice_leaflet) - float(val_metrics0["dice"][1])) if (dice_leaflet is not None and isinstance(val_metrics0.get("dice"), list) and len(val_metrics0["dice"]) > 1) else None
        semantic_dice_ring_delta = (float(dice_ring) - float(val_metrics0["dice"][2])) if (dice_ring is not None and isinstance(val_metrics0.get("dice"), list) and len(val_metrics0["dice"]) > 2) else None
        center_f1 = val_metrics.get("center_f1", None)
        inst_score = _instance_score(val_metrics)
        sweep_res = _maybe_run_threshold_sweep(
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
        ) if freeze_base else None
        sweep_best_center = (sweep_res or {}).get("best_center_f1") if isinstance(sweep_res, dict) else None
        sweep_best_strict = (sweep_res or {}).get("best_strict_marker_contract") if isinstance(sweep_res, dict) else None
        sweep_best_count = (sweep_res or {}).get("best_center_count_acc") if isinstance(sweep_res, dict) else None
        sweep_best_inst = (sweep_res or {}).get("best_instance_score") if isinstance(sweep_res, dict) else None
        primary_candidate = sweep_best_center if isinstance(sweep_best_center, dict) else {
            "threshold": float(center_thr),
            "center_f1_mean_samples": _float_or_none(val_metrics.get("center_f1_mean_samples")),
            "strict_marker_contract_pass_rate": _float_or_none(val_metrics.get("strict_marker_contract_pass_rate")),
            "exact_center_count_accuracy": _float_or_none(val_metrics.get("center_count_acc")),
            "localization_error_px": _float_or_none(val_metrics.get("center_loc_err_px")),
        }
        strict_candidate = sweep_best_strict if isinstance(sweep_best_strict, dict) else {
            "threshold": float(center_thr),
            "center_f1_mean_samples": _float_or_none(val_metrics.get("center_f1_mean_samples")),
            "strict_marker_contract_pass_rate": _float_or_none(val_metrics.get("strict_marker_contract_pass_rate")),
            "exact_center_count_accuracy": _float_or_none(val_metrics.get("center_count_acc")),
            "localization_error_px": _float_or_none(val_metrics.get("center_loc_err_px")),
        }
        center_f1_for_ckpt = float(primary_candidate.get(primary_metric_key)) if isinstance(primary_candidate, dict) and primary_candidate.get(primary_metric_key) is not None else (float(center_f1) if center_f1 is not None else None)
        center_count_acc_for_ckpt = float(sweep_best_count.get("center_count_acc")) if isinstance(sweep_best_count, dict) and sweep_best_count.get("center_count_acc") is not None else (float(val_metrics.get("center_count_acc")) if val_metrics.get("center_count_acc") is not None else None)
        inst_score_for_ckpt = float(sweep_best_inst.get("instance_score")) if isinstance(sweep_best_inst, dict) and sweep_best_inst.get("instance_score") is not None else (float(inst_score) if inst_score is not None else None)

        lr_unfrozen_decoder_now = float(next((g["lr"] for g in optimizer_meta if g["name"] == "unfrozen_decoder"), 0.0)) if partial_unfreeze else 0.0
        lr_backbone_now = 0.0 if freeze_base else float(optimizer.param_groups[0]["lr"])
        lr_center_now = float(next((g["lr"] for g in optimizer_meta if g["name"] == "center_head"), 0.0)) if freeze_base else float(optimizer.param_groups[1]["lr"])
        finite_batch_count = int(max(n_batches - nonfinite_grad_batches, 0))
        grad_norm_mean_before = float(finite_grad_norm_before_sum / max(finite_batch_count, 1)) if finite_batch_count > 0 else float("nan")
        grad_norm_mean_after = float(grad_norm_after_sum / max(n_batches, 1))
        finite_grad_norm_mean_before = float(finite_grad_norm_before_sum / max(finite_batch_count, 1)) if finite_batch_count > 0 else None
        clipped_pct = float(100.0 * float(clipped_batches) / float(max(n_batches, 1)))
        center_w_norm = _center_head_weight_norm(model)
        center_bias = _center_head_output_bias(model)
        params_finite = bool(_all_parameters_finite(list(model.parameters())))
        center_grad_mean_before = float(center_grad_before_sum / max(n_batches, 1))
        decoder_grad_mean_before = float(decoder_grad_before_sum / max(n_batches, 1))
        combined_grad_mean_before = float(combined_grad_before_sum / max(n_batches, 1))
        center_grad_mean_after = float(center_grad_after_sum / max(n_batches, 1))
        decoder_grad_mean_after = float(decoder_grad_after_sum / max(n_batches, 1))
        combined_grad_mean_after = float(combined_grad_after_sum / max(n_batches, 1))
        _write_validation_reports(out_dir, epoch=epoch, val_metrics=val_metrics, sweep_res=sweep_res, locked_threshold=center_thr)

        if freeze_base and (not partial_unfreeze) and semantic_mean_fg0 is not None and mean_fg is not None:
            dev = abs(float(mean_fg) - float(semantic_mean_fg0))
            if float(dev) > 0.002:
                raise SystemExit(f"Freeze stability check failed: |mean_fg - mean_fg0|={dev:.6f} > 0.002")
        if freeze_base and partial_unfreeze and semantic_mean_fg0 is not None and mean_fg is not None:
            dev = float(semantic_mean_fg0) - float(mean_fg)
            if dev > 0.02:
                semantic_degradation_streak += 1
            else:
                semantic_degradation_streak = 0
            if semantic_degradation_streak >= 2:
                raise SystemExit(f"Semantic degradation stop: mean_fg dropped by {dev:.6f} for two consecutive epochs")

        _save_checkpoint(out_dir / "last.pth", model, optimizer, epoch, cfg, extra={"val": val_metrics, "threshold_sweep": sweep_res})

        improved = False
        if (not freeze_base) and mean_fg is not None and (best_mean_fg is None or float(mean_fg) > float(best_mean_fg)):
            best_mean_fg = float(mean_fg)
            best_epoch_mean_fg = int(epoch)
            _save_checkpoint(out_dir / "best_mean_fg.pth", model, optimizer, epoch, cfg, extra={"val": val_metrics})
            improved = True
        if center_f1_for_ckpt is not None and (best_center_f1 is None or float(center_f1_for_ckpt) > float(best_center_f1)):
            best_center_f1 = float(center_f1_for_ckpt)
            best_epoch_center = int(epoch)
            _save_checkpoint(out_dir / "best_center_f1.pth", model, optimizer, epoch, cfg, extra={"val": val_metrics, "threshold_sweep": sweep_res, "best_threshold_metrics": sweep_best_center})
            if freeze_base:
                _export_center_diagnostics(
                    out_dir,
                    model,
                    val_loader,
                    device,
                    instance_root=instance_root,
                    center_thr=float(sweep_best_center.get("threshold")) if isinstance(sweep_best_center, dict) and sweep_best_center.get("threshold") is not None else center_thr,
                    tag="best_center_f1",
                    max_samples=20,
                )
            improved = True
        if _is_better_epoch_candidate(primary_candidate, best_primary_candidate, epoch=epoch, incumbent_epoch=best_epoch_primary, primary_metric=primary_metric_key):
            best_primary_candidate = dict(primary_candidate)
            best_epoch_primary = int(epoch)
            best_primary_path = out_dir / "best_primary.pth"
            _save_checkpoint(best_primary_path, model, optimizer, epoch, cfg, extra={"val": val_metrics, "threshold_sweep": sweep_res, "best_threshold_metrics": primary_candidate})
            best_primary_sha = _sha256_file(best_primary_path)
            _write_json_atomic(
                out_dir / "best_checkpoint_metadata.json",
                {
                    "checkpoint_path": str(best_primary_path.resolve()),
                    "checkpoint_sha256": best_primary_sha,
                    "epoch": int(epoch),
                    "selection_metric": primary_metric_key,
                    "tie_break_rule": "higher_strict_marker_contract_pass_rate_then_higher_exact_center_count_accuracy_then_lower_localization_error_then_earlier_epoch",
                    "tie_break_rule_canonical": "higher_strict_marker_contract_pass_rate_then_higher_exact_center_count_accuracy_then_lower_localization_error_px_pooled_matches_then_earlier_epoch",
                    "best_threshold_metrics": primary_candidate,
                    "locked_reference_threshold": float(center_thr),
                    "locked_reference_metrics": next((row for row in list((sweep_res or {}).get("rows") or []) if abs(float(row["threshold"]) - float(center_thr)) < 1e-9), None),
                },
            )
            improved = True
        if _is_better_epoch_candidate(strict_candidate, best_strict_candidate, epoch=epoch, incumbent_epoch=best_epoch_strict, primary_metric=strict_metric_key):
            best_strict_candidate = dict(strict_candidate)
            best_epoch_strict = int(epoch)
            best_strict_path = out_dir / "best_strict_marker_contract.pth"
            _save_checkpoint(best_strict_path, model, optimizer, epoch, cfg, extra={"val": val_metrics, "threshold_sweep": sweep_res, "best_threshold_metrics": strict_candidate})
            best_strict_sha = _sha256_file(best_strict_path)
            _write_json_atomic(
                out_dir / "best_strict_checkpoint_metadata.json",
                {
                    "checkpoint_path": str(best_strict_path.resolve()),
                    "checkpoint_sha256": best_strict_sha,
                    "epoch": int(epoch),
                    "selection_metric": strict_metric_key,
                    "tie_break_rule": "higher_center_f1_mean_samples_then_higher_exact_center_count_accuracy_then_lower_localization_error_px_pooled_matches_then_earlier_epoch",
                    "best_threshold_metrics": strict_candidate,
                    "locked_reference_threshold": float(center_thr),
                    "locked_reference_metrics": next((row for row in list((sweep_res or {}).get("rows") or []) if abs(float(row["threshold"]) - float(center_thr)) < 1e-9), None),
                },
            )
            improved = True
        if center_count_acc_for_ckpt is not None and (best_center_count_acc is None or float(center_count_acc_for_ckpt) > float(best_center_count_acc)):
            best_center_count_acc = float(center_count_acc_for_ckpt)
            best_epoch_center_count_acc = int(epoch)
            _save_checkpoint(out_dir / "best_center_count_acc.pth", model, optimizer, epoch, cfg, extra={"val": val_metrics, "threshold_sweep": sweep_res, "best_threshold_metrics": sweep_best_count})
            improved = True
        if inst_score_for_ckpt is not None and (best_instance is None or float(inst_score_for_ckpt) > float(best_instance)):
            best_instance = float(inst_score_for_ckpt)
            best_epoch_instance = int(epoch)
            _save_checkpoint(out_dir / "best_instance_score.pth", model, optimizer, epoch, cfg, extra={"val": val_metrics, "threshold_sweep": sweep_res, "best_threshold_metrics": sweep_best_inst})
            if freeze_base:
                _export_center_diagnostics(
                    out_dir,
                    model,
                    val_loader,
                    device,
                    instance_root=instance_root,
                    center_thr=float(sweep_best_inst.get("threshold")) if isinstance(sweep_best_inst, dict) and sweep_best_inst.get("threshold") is not None else center_thr,
                    tag="best_instance_score",
                    max_samples=20,
                )
            else:
                _export_val_visuals(out_dir, model, val_loader, device, max_samples=20)
            improved = True

        if freeze_base and int(epoch) in {1, 5, 10, 15, 20}:
            _export_center_diagnostics(
                out_dir,
                model,
                val_loader,
                device,
                instance_root=instance_root,
                center_thr=float(sweep_best_center.get("threshold")) if isinstance(sweep_best_center, dict) and sweep_best_center.get("threshold") is not None else center_thr,
                tag=f"epoch{epoch}",
                max_samples=20,
            )

        scheduler_monitor_name = str((scheduler_cfg or {}).get("monitor", early_monitor)) if scheduler is not None else ""
        scheduler_monitor_threshold_context = _locked_threshold_context_label(center_thr) if scheduler is not None else ""
        checkpoint_selection_metric_name = str(primary_metric_key)
        checkpoint_selection_threshold_context = _best_sweep_threshold_context_label() if freeze_base else _locked_threshold_context_label(center_thr)
        checkpoint_selection_metric_value = center_f1_for_ckpt
        early_stop_metric_name = str(early_monitor)
        early_stop_threshold_context = _locked_threshold_context_label(center_thr)
        early_stop_reset_policy = "any_checkpoint_improvement_resets_patience"
        lr_before_scheduler_step = lr_center_now
        lr_after_scheduler_step = lr_center_now
        scheduler_best_value = None
        scheduler_num_bad_epochs = None
        scheduler_cooldown_counter = None

        if scheduler is not None:
            monitor_key = scheduler_monitor_name
            monitor_val = val_metrics.get(monitor_key, None)
            if monitor_val is None and monitor_key == "instance_score":
                monitor_val = inst_score
            if monitor_val is not None:
                scheduler.step(float(monitor_val))
                lr_after_scheduler_step = float(next((g["lr"] for g in optimizer_meta if g["name"] == "center_head"), 0.0)) if freeze_base else float(optimizer.param_groups[1]["lr"])
                scheduler_best_value = float(scheduler.best)
                scheduler_num_bad_epochs = int(scheduler.num_bad_epochs)
                scheduler_cooldown_counter = int(getattr(scheduler, "cooldown_counter", 0))

        monitor_val_es = val_metrics.get(early_monitor, None)
        if monitor_val_es is None and early_monitor == "instance_score":
            monitor_val_es = inst_score
        if monitor_val_es is None:
            monitor_val_es = inst_score

        with metrics_csv.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    epoch,
                    train_loss,
                    float(val_metrics["semantic_loss"]),
                    float(val_metrics["center_loss"]),
                    float(mean_fg) if mean_fg is not None else "",
                    float(dice_leaflet) if dice_leaflet is not None else "",
                    float(dice_ring) if dice_ring is not None else "",
                    float(semantic_mean_fg_delta) if semantic_mean_fg_delta is not None else "",
                    float(semantic_mean_fg_abs_delta) if semantic_mean_fg_abs_delta is not None else "",
                    float(semantic_dice_leaflet_delta) if semantic_dice_leaflet_delta is not None else "",
                    float(semantic_dice_ring_delta) if semantic_dice_ring_delta is not None else "",
                    float(center_f1) if center_f1 is not None else "",
                    float(val_metrics.get("center_f1_mean_samples")) if val_metrics.get("center_f1_mean_samples") is not None else "",
                    float(val_metrics.get("center_precision")) if val_metrics.get("center_precision") is not None else "",
                    float(val_metrics.get("center_recall")) if val_metrics.get("center_recall") is not None else "",
                    float(val_metrics.get("strict_marker_contract_pass_rate")) if val_metrics.get("strict_marker_contract_pass_rate") is not None else "",
                    int(val_metrics.get("strict_marker_contract_pass_count")) if val_metrics.get("strict_marker_contract_pass_count") is not None else "",
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
                    float(sweep_best_center.get("threshold")) if isinstance(sweep_best_center, dict) and sweep_best_center.get("threshold") is not None else "",
                    float(sweep_best_center.get("center_f1")) if isinstance(sweep_best_center, dict) and sweep_best_center.get("center_f1") is not None else "",
                    float(sweep_best_count.get("threshold")) if isinstance(sweep_best_count, dict) and sweep_best_count.get("threshold") is not None else "",
                    float(sweep_best_count.get("center_count_acc")) if isinstance(sweep_best_count, dict) and sweep_best_count.get("center_count_acc") is not None else "",
                    float(sweep_best_inst.get("threshold")) if isinstance(sweep_best_inst, dict) and sweep_best_inst.get("threshold") is not None else "",
                    float(sweep_best_inst.get("instance_score")) if isinstance(sweep_best_inst, dict) and sweep_best_inst.get("instance_score") is not None else "",
                    float(sweep_best_inst.get("instance_mean_matched_iou")) if isinstance(sweep_best_inst, dict) and sweep_best_inst.get("instance_mean_matched_iou") is not None else "",
                    scheduler_monitor_name,
                    scheduler_monitor_threshold_context,
                    float(monitor_val) if scheduler is not None and monitor_val is not None else "",
                    checkpoint_selection_metric_name,
                    checkpoint_selection_threshold_context,
                    float(checkpoint_selection_metric_value) if checkpoint_selection_metric_value is not None else "",
                    early_stop_metric_name,
                    early_stop_threshold_context,
                    early_stop_reset_policy,
                    float(monitor_val_es) if monitor_val_es is not None else "",
                    float(lr_before_scheduler_step),
                    float(lr_after_scheduler_step),
                    float(scheduler_best_value) if scheduler_best_value is not None else "",
                    int(scheduler_num_bad_epochs) if scheduler_num_bad_epochs is not None else "",
                    int(scheduler_cooldown_counter) if scheduler_cooldown_counter is not None else "",
                    grad_norm_mean_before,
                    grad_norm_before_max,
                    center_grad_mean_before,
                    decoder_grad_mean_before,
                    combined_grad_mean_before,
                    float(finite_grad_norm_mean_before) if finite_grad_norm_mean_before is not None else "",
                    float(finite_grad_norm_before_max) if finite_batch_count > 0 else "",
                    int(nonfinite_grad_batches),
                    int(skipped_optimizer_steps),
                    int(clipped_batches),
                    clipped_pct,
                    grad_norm_mean_after,
                    center_grad_mean_after,
                    decoder_grad_mean_after,
                    combined_grad_mean_after,
                    float(amp_scale_min) if amp_scale_min is not None else "",
                    float(amp_scale_max) if amp_scale_max is not None else "",
                    float(amp_scale_last) if amp_scale_last is not None else "",
                    bool(train_loss_is_finite),
                    bool(params_finite),
                    bool(train_logits_finite),
                    (last_precision_info or {}).get("decoder_features_dtype", ""),
                    (last_precision_info or {}).get("center_logits_dtype", ""),
                    (last_precision_info or {}).get("center_loss_dtype", ""),
                    (last_precision_info or {}).get("center_grad_scaler_enabled", ""),
                    float(center_w_norm) if center_w_norm is not None else "",
                    float(center_bias) if center_bias is not None else "",
                    lr_unfrozen_decoder_now if partial_unfreeze else "",
                    "" if freeze_base else lr_backbone_now,
                    lr_before_scheduler_step,
                ]
            )

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
                f"mean_fg={mean_fg} center_metric={center_f1_for_ckpt} instance_score={inst_score_for_ckpt} "
                f"grad_mean={grad_norm_mean_before:.4f} grad_max={grad_norm_before_max:.4f} clipped={clipped_pct:.1f}% "
                f"nonfinite_grad_batches={nonfinite_grad_batches} skipped_steps={skipped_optimizer_steps} "
                f"lr_decoder={lr_unfrozen_decoder_now:.2e} lr_center={lr_before_scheduler_step:.2e}->{lr_after_scheduler_step:.2e}"
            )
        else:
            print(
                f"epoch={epoch} time={dt:.1f}s train_loss={train_loss:.6f} "
                f"mean_fg={mean_fg} center_metric={center_f1_for_ckpt} instance_score={inst_score} "
                f"lr_backbone={lr_backbone_now:.2e} lr_center={lr_before_scheduler_step:.2e}->{lr_after_scheduler_step:.2e}"
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
                "best_primary_metric": best_primary_candidate,
                "best_epoch_primary_metric": best_epoch_primary,
                "best_strict_marker_contract_metric": best_strict_candidate,
                "best_epoch_strict_marker_contract_metric": best_epoch_strict,
                "best_center_count_acc": best_center_count_acc,
                "best_epoch_center_count_acc": best_epoch_center_count_acc,
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
    _write_json_atomic(
        out_dir / "final_training_summary.json",
        {
            "best_mean_fg": best_mean_fg,
            "best_epoch_mean_fg": best_epoch_mean_fg,
            "best_center_f1": best_center_f1,
            "best_epoch_center_f1": best_epoch_center,
            "best_primary_metric": best_primary_candidate,
            "best_epoch_primary_metric": best_epoch_primary,
            "best_strict_marker_contract_metric": best_strict_candidate,
            "best_epoch_strict_marker_contract_metric": best_epoch_strict,
            "best_center_count_acc": best_center_count_acc,
            "best_epoch_center_count_acc": best_epoch_center_count_acc,
            "best_instance_score": best_instance,
            "best_epoch_instance_score": best_epoch_instance,
            "selection_metric": primary_metric_key,
            "strict_selection_metric": strict_metric_key,
            "locked_reference_threshold": float(center_thr),
            "center_loss": center_loss_info,
            "lambda_center": lambda_center,
        },
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
