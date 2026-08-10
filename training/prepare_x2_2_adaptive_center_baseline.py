from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from train_centerhead import (
    _apply_training_policy,
    _build_model,
    _build_optimizer_groups,
    _read_yaml,
    smoke_test,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
AUG_BASELINE_CONFIG_PATH = REPO_ROOT / "training" / "configs" / "unetpp_effb3_centerhead_x2_2_adapter_full_dataset_aug_baseline_100ep.yaml"
NEW_CONFIG_PATH = REPO_ROOT / "training" / "configs" / "unetpp_effb3_centerhead_x2_2_adapter_full_dataset_aug_x2_2_unfreeze_100ep.yaml"
OUTPUT_DIR = REPO_ROOT / "training" / "analysis" / "center_full_dataset_aug_x2_2_unfreeze_readiness"
SEMANTIC_BASELINE_RUN_DIR = REPO_ROOT / "training" / "runs" / "unetpp_effb3_centerhead_x2_2_adapter_full_dataset_baseline_100ep"
TARGET_MODULE = "base.decoder.blocks.x_2_2"
NEW_SAVE_DIR = "training/runs/unetpp_effb3_centerhead_x2_2_adapter_full_dataset_aug_x2_2_unfreeze_100ep"
X2_2_LR = 1.0e-5
CENTER_LR = 1.0e-3


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


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _config_diff_paths(base: Any, other: Any, prefix: str = "") -> list[str]:
    if isinstance(base, dict) and isinstance(other, dict):
        keys = sorted(set(base.keys()) | set(other.keys()))
        out: list[str] = []
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in base or key not in other:
                out.append(path)
                continue
            out.extend(_config_diff_paths(base[key], other[key], path))
        return out
    if isinstance(base, list) and isinstance(other, list):
        if len(base) != len(other):
            return [prefix]
        out: list[str] = []
        for idx, (b, o) in enumerate(zip(base, other)):
            out.extend(_config_diff_paths(b, o, f"{prefix}[{idx}]"))
        return out
    if base != other:
        return [prefix]
    return []


def _x2_2_module_audit(cfg: dict) -> dict[str, Any]:
    model = _build_model(cfg)
    freeze_info = _apply_training_policy(model, cfg)
    module = dict(model.named_modules())[TARGET_MODULE]
    param_names = [n for n, _p in model.named_parameters() if n.startswith(TARGET_MODULE + ".")]
    stateful_layers = []
    batchnorm_layers = []
    for name, mod in module.named_modules():
        if hasattr(mod, "running_mean") or hasattr(mod, "running_var"):
            stateful_layers.append({"name": name or TARGET_MODULE, "type": type(mod).__name__})
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            batchnorm_layers.append({"name": name or TARGET_MODULE, "type": type(mod).__name__})
    optimizer, optimizer_meta = _build_optimizer_groups(model, cfg, freeze_info if freeze_info.get("freeze_base") else None, freeze_base=True)
    del optimizer
    center_group = next(group for group in optimizer_meta if group["name"] == "center_head")
    x2_2_group = next(group for group in optimizer_meta if group["name"] != "center_head")
    forbidden = [
        name
        for name in x2_2_group["parameter_names"]
        if not str(name).startswith(TARGET_MODULE + ".")
    ]
    trainable_names = list(freeze_info["trainable_names"])
    forbidden_trainable = [
        name
        for name in trainable_names
        if not (str(name).startswith("center_head.") or str(name).startswith("center_adapter.") or str(name).startswith(TARGET_MODULE + "."))
    ]
    return {
        "module": TARGET_MODULE,
        "parameter_names": param_names,
        "parameter_count": int(sum(int(p.numel()) for n, p in model.named_parameters() if n in set(param_names))),
        "trainable_parameter_names": trainable_names,
        "center_parameter_names": list(center_group["parameter_names"]),
        "center_parameter_count": int(center_group["param_count"]),
        "x2_2_group_name": str(x2_2_group["name"]),
        "x2_2_parameter_names": list(x2_2_group["parameter_names"]),
        "x2_2_parameter_count": int(x2_2_group["param_count"]),
        "x2_2_lr": float(x2_2_group["lr"]),
        "center_lr": float(center_group["lr"]),
        "optimizer_group_count": int(len(optimizer_meta)),
        "optimizer_groups": optimizer_meta,
        "forbidden_optimizer_parameters": forbidden,
        "forbidden_trainable_parameters": forbidden_trainable,
        "semantic_init_report": dict(getattr(model, "semantic_init_report", {}) or {}),
        "batchnorm_stateful_layers": batchnorm_layers,
        "all_stateful_layers": stateful_layers,
        "isolated_unfreeze_safe": bool(not forbidden and not forbidden_trainable and len(optimizer_meta) == 2),
        "dependencies": {
            "upstream_decoder_blocks_frozen": True,
            "encoder_frozen": True,
            "segmentation_head_frozen": True,
            "autograd_through_upstream_activations_required": True,
            "frozen_upstream_parameters_receive_no_grad_updates": True,
            "selected_block_train_mode_required": True,
            "all_other_semantic_modules_eval_mode_required": True,
        },
    }


def _semantic_baseline_from_epoch0(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.csv"
    rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8")))
    row0 = next(row for row in rows if str(row["epoch"]) == "0")
    return {
        "source_run_dir": str(run_dir.resolve()),
        "epoch": 0,
        "mean_foreground_iou": float(row0["mean_dice_fg"]),
        "per_class": {
            "leaflet_dice": float(row0["dice_leaflet"]),
            "ring_dice": float(row0["dice_ring"]),
        },
    }


def _build_config(aug_cfg: dict, x2_2_audit: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    cfg = copy.deepcopy(aug_cfg)
    cfg["train"]["save_dir"] = NEW_SAVE_DIR
    cfg["train"]["trainable_base_modules"] = [TARGET_MODULE]
    cfg["train"]["lr_unfrozen_decoder"] = float(X2_2_LR)
    cfg.setdefault("experiment_metadata", {})
    cfg["experiment_metadata"]["semantic_checkpoint_sha256"] = cfg["train"]["init_checkpoint_sha256"]
    cfg["experiment_metadata"]["train_manifest_sha256"] = cfg["dataset"]["train_manifest_sha256"]
    cfg["experiment_metadata"]["val_manifest_sha256"] = cfg["dataset"]["val_manifest_sha256"]
    cfg["experiment_metadata"]["augmentation"] = copy.deepcopy(cfg["augment"])
    cfg["experiment_metadata"]["trainable_modules"] = [TARGET_MODULE, "center_adapter", "center_head"]
    cfg["experiment_metadata"]["frozen_modules"] = [
        "base.encoder",
        "base.decoder.blocks(except x_2_2)",
        "base.segmentation_head",
        "all_other_semantic_parameters",
    ]
    cfg["experiment_metadata"]["parameter_counts"] = {
        "center": int(x2_2_audit["center_parameter_count"]),
        "x2_2": int(x2_2_audit["x2_2_parameter_count"]),
    }
    cfg["experiment_metadata"]["optimizer_groups"] = {
        "center_head": {"lr": float(CENTER_LR), "parameter_count": int(x2_2_audit["center_parameter_count"])},
        "x2_2": {"lr": float(X2_2_LR), "parameter_count": int(x2_2_audit["x2_2_parameter_count"])},
    }
    diff_paths = _config_diff_paths(aug_cfg, cfg)
    return cfg, diff_paths


def _summarize_smoke(smoke: dict[str, Any]) -> dict[str, Any]:
    center_grad_ok = bool(smoke.get("center_grad_all_finite", False)) and float(smoke.get("center_grad_norm_before_clip", 0.0)) > 0.0
    x2_2_grad_ok = float(smoke.get("decoder_grad_norm_before_clip", 0.0)) > 0.0 and float(smoke.get("selected_decoder_parameter_delta", 0.0)) > 0.0
    forbidden_grad_ok = (
        int(smoke.get("frozen_encoder_grad_count", -1)) == 0
        and int(smoke.get("frozen_decoder_grad_count", -1)) == 0
        and int(smoke.get("semantic_head_grad_count", -1)) == 0
        and float(smoke.get("frozen_parameter_max_delta", -1.0)) == 0.0
    )
    return {
        "forward": "passed" if smoke.get("semantic_shape") and smoke.get("center_shape") else "failed",
        "loss": "passed" if bool(smoke.get("semantic_loss_finite", False)) and float(smoke.get("loss_total", float("nan"))) == float(smoke.get("loss_total", float("nan"))) else "failed",
        "backward": "passed" if center_grad_ok and x2_2_grad_ok else "failed",
        "center_gradients": "passed" if center_grad_ok else "failed",
        "x2_2_gradients": "passed" if x2_2_grad_ok else "failed",
        "forbidden_gradients": "passed" if forbidden_grad_ok else "failed",
        "raw": smoke,
    }


def run(*, output_dir: Path, config_path: Path) -> dict[str, Any]:
    aug_cfg = _read_yaml(AUG_BASELINE_CONFIG_PATH)
    x2_2_audit = _x2_2_module_audit(copy.deepcopy(aug_cfg) | {"train": dict((aug_cfg.get("train") or {}), trainable_base_modules=[TARGET_MODULE], lr_unfrozen_decoder=float(X2_2_LR))})
    cfg, diff_paths = _build_config(aug_cfg, x2_2_audit)
    _write_yaml(config_path, cfg)

    semantic_baseline = _semantic_baseline_from_epoch0(SEMANTIC_BASELINE_RUN_DIR)
    smoke = smoke_test(cfg, device=torch.device("cpu"))
    smoke_summary = _summarize_smoke(smoke)

    readiness = {
        "status": "ready_for_training" if all(smoke_summary[key] == "passed" for key in ["forward", "loss", "backward", "center_gradients", "x2_2_gradients", "forbidden_gradients"]) and bool(x2_2_audit["isolated_unfreeze_safe"]) else "blocked",
        "config": str(config_path.resolve()),
        "changed_fields_vs_augmented_baseline": diff_paths,
        "unchanged_contract_fields": [
            "dataset.train_manifest",
            "dataset.val_manifest",
            "train.init_checkpoint",
            "augment",
            "model.center_feature",
            "center_loss",
            "center.marker_thr",
            "center_loss.threshold_sweep",
            "train.batch_size",
            "train.center_grad_clip_norm",
            "train.epochs",
            "seed",
        ],
        "x2_2_audit": x2_2_audit,
        "semantic_baseline": semantic_baseline,
        "smoke_test": {k: v for k, v in smoke_summary.items() if k != "raw"},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json((output_dir / "x2_2_audit.json").resolve(), x2_2_audit)
    _write_json((output_dir / "semantic_baseline.json").resolve(), semantic_baseline)
    _write_json((output_dir / "smoke_test_summary.json").resolve(), smoke_summary)
    _write_json((output_dir / "readiness_summary.json").resolve(), readiness)
    _write_text(
        (output_dir / "files_to_review.txt").resolve(),
        "\n".join(
            [
                str((output_dir / "x2_2_audit.json").resolve()),
                str((output_dir / "semantic_baseline.json").resolve()),
                str((output_dir / "smoke_test_summary.json").resolve()),
                str((output_dir / "readiness_summary.json").resolve()),
                str(config_path.resolve()),
            ]
        )
        + "\n",
    )
    return readiness


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    ap.add_argument("--config-out", type=str, default=str(NEW_CONFIG_PATH))
    args = ap.parse_args()
    summary = run(output_dir=Path(args.output_dir).resolve(), config_path=Path(args.config_out).resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
