from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from train_centerhead import (
    _apply_training_policy,
    _build_center_loss,
    _build_loaders,
    _build_model,
    _build_optimizer_groups,
    _center_fp32_enabled,
    _forward_base_for_center_training,
    _forward_center_with_precision,
    _read_yaml,
    _set_train_modes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_CONFIG_PATH = REPO_ROOT / "training" / "configs" / "unetpp_effb3_centerhead_x2_2_adapter_full_dataset_aug_x2_2_unfreeze_100ep.yaml"
NEW_CONFIG_PATH = REPO_ROOT / "training" / "configs" / "unetpp_effb3_centerhead_multiscale_x2_2_x1_1_full_dataset_aug_100ep.yaml"
OUTPUT_DIR = REPO_ROOT / "training" / "analysis" / "center_multiscale_x2_2_x1_1_readiness"

PRIMARY_MODULE = "base.decoder.blocks.x_2_2"
SELECTED_CONTEXT_MODULE = "base.decoder.blocks.x_1_1"
NEW_SAVE_DIR = "training/runs/unetpp_effb3_centerhead_multiscale_x2_2_x1_1_full_dataset_aug_100ep"

PRIMARY_PROJECTION_OUT_CHANNELS = 16
CONTEXT_PROJECTION_OUT_CHANNELS = 16
FUSION_OUT_CHANNELS = 32
STRICT_CHECKPOINT_METRIC = "strict_marker_contract_pass_rate"


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
        for idx, (base_item, other_item) in enumerate(zip(base, other)):
            out.extend(_config_diff_paths(base_item, other_item, f"{prefix}[{idx}]"))
        return out
    if base != other:
        return [prefix]
    return []


def _parameter_names_for_module(model: torch.nn.Module, module_path: str) -> list[str]:
    prefix = str(module_path).strip() + "."
    return [name for name, _param in model.named_parameters() if name.startswith(prefix)]


def _parameter_count(model: torch.nn.Module, parameter_names: list[str]) -> int:
    wanted = set(str(name) for name in parameter_names)
    return int(sum(int(param.numel()) for name, param in model.named_parameters() if name in wanted))


def _decoder_dependency_graph(model: torch.nn.Module) -> dict[str, list[str]]:
    decoder = model.base.decoder
    graph: dict[str, list[str]] = {}
    for layer_idx in range(len(decoder.in_channels) - 1):
        for depth_idx in range(int(decoder.depth) - int(layer_idx)):
            if layer_idx == 0:
                graph[f"x_{depth_idx}_{depth_idx}"] = []
            else:
                dense_level = depth_idx + layer_idx
                graph[f"x_{depth_idx}_{dense_level}"] = [f"x_{depth_idx}_{dense_level - 1}"] + [
                    f"x_{idx}_{dense_level}" for idx in range(depth_idx + 1, dense_level + 1)
                ]
    graph[f"x_{0}_{decoder.depth}"] = [f"x_{0}_{decoder.depth - 1}"]
    return graph


def _has_path(graph: dict[str, list[str]], start: str, target: str, seen: set[str] | None = None) -> bool:
    if start == target:
        return True
    if seen is None:
        seen = set()
    if start in seen:
        return False
    seen.add(start)
    for child in graph.get(start, []):
        if _has_path(graph, child, target, seen):
            return True
    return False


def _relation_to_x2_2(graph: dict[str, list[str]], block_name: str) -> str:
    if block_name == "x_2_2":
        return "primary_current_tap"
    if _has_path(graph, "x_2_2", block_name):
        return "upstream_of_x_2_2"
    if _has_path(graph, block_name, "x_2_2"):
        return "downstream_of_x_2_2"
    return "parallel_to_x_2_2"


def _trace_decoder_blocks(model: torch.nn.Module, *, input_size: int) -> dict[str, dict[str, Any]]:
    x = torch.zeros(1, int(model.base.encoder._in_channels), int(input_size), int(input_size))
    decoder = model.base.decoder
    graph = _decoder_dependency_graph(model)
    with torch.no_grad():
        encoder_features = model.encoder(x)
        features = encoder_features[1:]
        features = features[::-1]
        dense_x: dict[str, torch.Tensor] = {}
        topology: dict[str, dict[str, Any]] = {}
        for layer_idx in range(len(decoder.in_channels) - 1):
            for depth_idx in range(int(decoder.depth) - int(layer_idx)):
                block_name = f"x_{depth_idx}_{depth_idx}" if layer_idx == 0 else f"x_{depth_idx}_{depth_idx + layer_idx}"
                block = decoder.blocks[block_name]
                if layer_idx == 0:
                    output = block(features[depth_idx], features[depth_idx + 1])
                else:
                    dense_level = depth_idx + layer_idx
                    cat_features = [dense_x[f"x_{idx}_{dense_level}"] for idx in range(depth_idx + 1, dense_level + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_level + 1]], dim=1)
                    output = block(dense_x[f"x_{depth_idx}_{dense_level - 1}"], cat_features)
                dense_x[block_name] = output
        final_name = f"x_{0}_{decoder.depth}"
        dense_x[final_name] = decoder.blocks[final_name](dense_x[f"x_{0}_{decoder.depth - 1}"])

    for block_name, tensor in dense_x.items():
        module_path = f"base.decoder.blocks.{block_name}"
        parameter_names = _parameter_names_for_module(model, module_path)
        stride = int(input_size // int(tensor.shape[-1]))
        batchnorm_layers = [
            {"name": (f"{module_path}.{name}" if name else module_path), "type": type(module).__name__}
            for name, module in dict(model.named_modules())[module_path].named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        ]
        if block_name == "x_0_0":
            rationale = "Deepest and coarsest decoder feature. Strongest object context, but 48x48 is likely too coarse for precise leaflet-center placement."
        elif block_name == "x_0_1":
            rationale = "Stride-8 decoder feature that already mixes x_0_0 and x_1_1 context, but at 128 channels it is substantially heavier than needed for a controlled first multiscale test."
        elif block_name == "x_1_1":
            rationale = "Stride-8 decoder feature with materially larger context than x_2_2 while keeping a usable 96x96 grid and a modest 48-channel footprint."
        elif block_name == "x_2_2":
            rationale = "Current stride-4 primary center tap. Highest spatial detail among the contextual candidates already used by the center branch."
        else:
            rationale = "Materialized in the same decoder forward pass, but not a primary candidate for the first deeper-context experiment."
        topology[module_path] = {
            "module_path": module_path,
            "block_name": block_name,
            "shape": [int(v) for v in tensor.shape],
            "channels": int(tensor.shape[1]),
            "native_stride": stride,
            "spatial_resolution": [int(tensor.shape[-2]), int(tensor.shape[-1])],
            "already_materialized_in_same_forward": True,
            "requires_hooks": False,
            "relation_to_x2_2": _relation_to_x2_2(graph, block_name),
            "receptive_field_context_rationale": rationale,
            "changing_parameters_affects_semantic_output": True,
            "parameter_names": parameter_names,
            "parameter_count": _parameter_count(model, parameter_names),
            "parameter_owner_module": module_path,
            "batchnorm_layers": batchnorm_layers,
        }
    return topology


def _candidate_nodes(topology: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for module_path, info in sorted(topology.items()):
        if int(info["native_stride"]) <= 4:
            continue
        if str(info["relation_to_x2_2"]).startswith("downstream"):
            continue
        candidates.append(info)
    return candidates


def _select_context_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    selected = next(info for info in candidates if str(info["module_path"]) == SELECTED_CONTEXT_MODULE)
    selected["selection_reason"] = (
        "Selected x_1_1 as the first deeper contextual tap because it doubles spatial stride from 4 to 8, stays at a usable 96x96 grid, "
        "is already materialized in the same decoder pass, remains parallel to x_2_2, and adds less branch weight than x_0_1 while keeping more context than x_2_2 alone."
    )
    return selected


def _center_group_count(optimizer_meta: list[dict[str, Any]]) -> int:
    return int(next(group["param_count"] for group in optimizer_meta if str(group["name"]) == "center_head"))


def _x2_2_group_count(optimizer_meta: list[dict[str, Any]]) -> int:
    return int(next(group["param_count"] for group in optimizer_meta if str(group["name"]) != "center_head"))


def _build_config(control_cfg: dict, *, selected_context: dict[str, Any], control_center_param_count: int) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    cfg = copy.deepcopy(control_cfg)
    cfg["train"]["save_dir"] = NEW_SAVE_DIR
    cfg["train"]["strict_checkpoint_metric"] = STRICT_CHECKPOINT_METRIC
    cfg["model"]["center_feature"] = dict(cfg["model"]["center_feature"])
    cfg["model"]["center_feature"]["context_module_path"] = str(selected_context["module_path"])
    cfg["model"]["center_feature"]["context_expected_channels"] = int(selected_context["channels"])
    cfg["model"]["center_feature"]["context_native_stride"] = int(selected_context["native_stride"])
    cfg["model"]["center_feature"]["primary_projection_out_channels"] = int(PRIMARY_PROJECTION_OUT_CHANNELS)
    cfg["model"]["center_feature"]["context_projection_out_channels"] = int(CONTEXT_PROJECTION_OUT_CHANNELS)
    cfg["model"]["center_feature"]["fusion_method"] = "concat"
    cfg["model"]["center_feature"]["fusion_out_channels"] = int(FUSION_OUT_CHANNELS)

    model = _build_model(cfg)
    freeze_info = _apply_training_policy(model, cfg)
    _optimizer, optimizer_meta = _build_optimizer_groups(model, cfg, freeze_info, freeze_base=True)
    center_param_count = _center_group_count(optimizer_meta)
    x2_2_param_count = _x2_2_group_count(optimizer_meta)
    context_param_names = _parameter_names_for_module(model, str(selected_context["module_path"]))
    context_param_count = _parameter_count(model, context_param_names)
    added_param_count = int(center_param_count - int(control_center_param_count))

    cfg.setdefault("experiment_metadata", {})
    cfg["experiment_metadata"]["primary_center_tap"] = {
        "module_path": PRIMARY_MODULE,
        "shape": [1, 32, 192, 192],
        "channels": 32,
        "native_stride": 4,
    }
    cfg["experiment_metadata"]["context_center_tap"] = {
        "module_path": str(selected_context["module_path"]),
        "shape": list(selected_context["shape"]),
        "channels": int(selected_context["channels"]),
        "native_stride": int(selected_context["native_stride"]),
        "relation_to_x2_2": str(selected_context["relation_to_x2_2"]),
    }
    cfg["experiment_metadata"]["fusion"] = {
        "primary_projection_out_channels": int(PRIMARY_PROJECTION_OUT_CHANNELS),
        "context_projection_out_channels": int(CONTEXT_PROJECTION_OUT_CHANNELS),
        "fusion_method": "concat",
        "fusion_out_channels": int(FUSION_OUT_CHANNELS),
        "resulting_center_input_channels": int(FUSION_OUT_CHANNELS),
    }
    cfg["experiment_metadata"]["trainable_modules"] = [
        PRIMARY_MODULE,
        "center_primary_projection",
        "center_context_projection",
        "center_fusion_adapter",
        "center_adapter",
        "center_head",
    ]
    cfg["experiment_metadata"]["frozen_modules"] = [
        "base.encoder",
        f"{selected_context['module_path']}",
        "base.decoder.blocks(except x_2_2 and the selected frozen context tap remains frozen)",
        "base.segmentation_head",
        "all_other_semantic_parameters",
    ]
    cfg["experiment_metadata"]["parameter_counts"] = {
        "center_fusion_and_head": int(center_param_count),
        "added_multiscale_parameters": int(added_param_count),
        "x2_2": int(x2_2_param_count),
        "frozen_context_block": int(context_param_count),
    }
    cfg["experiment_metadata"]["optimizer_groups"] = {
        "center_head": {
            "lr": float(cfg["train"]["lr_center_head"]),
            "parameter_count": int(center_param_count),
        },
        "x2_2": {
            "lr": float(cfg["train"]["lr_unfrozen_decoder"]),
            "parameter_count": int(x2_2_param_count),
        },
    }
    cfg["experiment_metadata"]["checkpoint_files"] = {
        "best_primary": "best_primary.pth",
        "best_strict_marker_contract": "best_strict_marker_contract.pth",
    }
    diff_paths = _config_diff_paths(control_cfg, cfg)
    audit = {
        "center_param_count": int(center_param_count),
        "x2_2_param_count": int(x2_2_param_count),
        "context_param_count": int(context_param_count),
        "added_multiscale_parameters": int(added_param_count),
        "optimizer_groups": optimizer_meta,
        "forbidden_trainable_parameters": [
            name
            for name in freeze_info["trainable_names"]
            if not (
                name.startswith("center_primary_projection.")
                or name.startswith("center_context_projection.")
                or name.startswith("center_fusion_adapter.")
                or name.startswith("center_adapter.")
                or name.startswith("center_head.")
                or name.startswith(PRIMARY_MODULE + ".")
            )
        ],
        "frozen_context_parameter_names": context_param_names,
        "x2_2_parameter_names": _parameter_names_for_module(model, PRIMARY_MODULE),
    }
    return cfg, diff_paths, audit


def _grad_stats(named_params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, Any]:
    present = [(name, param) for name, param in named_params if param.grad is not None]
    finite = bool(all(torch.isfinite(param.grad.detach()).all().item() for _name, param in present))
    nonzero = bool(any(float(param.grad.detach().abs().sum().item()) > 0.0 for _name, param in present))
    return {
        "parameter_count": int(len(named_params)),
        "gradient_tensor_count": int(len(present)),
        "all_finite": finite,
        "any_nonzero": nonzero,
    }


def _run_smoke(cfg: dict, *, device: torch.device) -> dict[str, Any]:
    train_loader, _val_loader = _build_loaders(cfg, device=device)
    model = _build_model(cfg).to(device)
    freeze_info = _apply_training_policy(model, cfg)
    optimizer, optimizer_meta = _build_optimizer_groups(model, cfg, freeze_info, freeze_base=True)
    optimizer.zero_grad(set_to_none=True)
    _set_train_modes(model, freeze_base=True)

    batch = next(iter(train_loader))
    images = batch["image"].to(device)
    centers = batch["center"].to(device)
    amp_enabled = bool((cfg.get("train") or {}).get("amp", False)) and device.type == "cuda"
    center_fp32 = _center_fp32_enabled(cfg)
    dataset_root = Path(cfg["dataset"]["root"]).resolve()
    train_txt = Path(cfg["dataset"]["train_txt"]).resolve()
    center_loss_fn, _center_loss_info = _build_center_loss(cfg, device=device, dataset_root=dataset_root, train_txt=train_txt)

    semantic_logits, decoder_output = _forward_base_for_center_training(
        model=model,
        images=images,
        device=device,
        amp_enabled_global=amp_enabled,
        detach_output=False,
        no_grad=False,
    )
    fused_features, center_logits, center_payload, precision_info = _forward_center_with_precision(
        model=model,
        decoder_output=decoder_output,
        centers=centers,
        center_loss_fn=center_loss_fn,
        device=device,
        amp_enabled_global=amp_enabled,
        center_fp32=center_fp32,
        detach_decoder_output=False,
        return_details=True,
    )
    loss_center = center_payload["loss"] if isinstance(center_payload, dict) else center_payload
    if not bool(torch.isfinite(loss_center).all().item()):
        raise SystemExit("Smoke test failed: center loss is not finite")
    loss_center.backward()

    named_parameters = list(model.named_parameters())
    x2_2_named = [(name, param) for name, param in named_parameters if name.startswith(PRIMARY_MODULE + ".")]
    context_named = [(name, param) for name, param in named_parameters if name.startswith(SELECTED_CONTEXT_MODULE + ".")]
    encoder_named = [(name, param) for name, param in named_parameters if name.startswith("base.encoder.")]
    decoder_other_named = [
        (name, param)
        for name, param in named_parameters
        if name.startswith("base.decoder.")
        and not name.startswith(PRIMARY_MODULE + ".")
        and not name.startswith(SELECTED_CONTEXT_MODULE + ".")
    ]
    segmentation_head_named = [(name, param) for name, param in named_parameters if name.startswith("base.segmentation_head.")]
    center_branch_stats = {
        prefix.rstrip("."): _grad_stats([(name, param) for name, param in named_parameters if name.startswith(prefix)])
        for prefix in model.center_branch_parameter_prefixes()
    }

    capture_info = dict(precision_info.get("center_feature_capture_info") or {})
    context_capture = dict(capture_info.get("context") or {})
    return {
        "semantic_init_report": dict(getattr(model, "semantic_init_report", {}) or {}),
        "optimizer_groups": optimizer_meta,
        "primary_capture_shape": capture_info.get("captured_shape"),
        "context_capture_shape": context_capture.get("captured_shape"),
        "resolved_center_feature_shape": precision_info.get("decoder_features_shape"),
        "center_logits_shape": precision_info.get("center_logits_shape"),
        "semantic_logits_shape": list(semantic_logits.shape),
        "feature_extraction_requires_hooks": bool(capture_info.get("access_requires_hooks", True)),
        "semantic_logits_finite": bool(torch.isfinite(semantic_logits.detach()).all().item()),
        "resolved_center_features_finite": bool(torch.isfinite(fused_features.detach()).all().item()),
        "center_logits_finite": bool(torch.isfinite(center_logits.detach()).all().item()),
        "center_loss_finite": bool(torch.isfinite(loss_center.detach()).all().item()),
        "center_branch_stats": center_branch_stats,
        "x2_2_grad_stats": _grad_stats(x2_2_named),
        "context_grad_stats": _grad_stats(context_named),
        "encoder_grad_stats": _grad_stats(encoder_named),
        "decoder_other_grad_stats": _grad_stats(decoder_other_named),
        "segmentation_head_grad_stats": _grad_stats(segmentation_head_named),
        "frozen_context_module_training": bool(dict(model.named_modules())[SELECTED_CONTEXT_MODULE].training),
        "x2_2_module_training": bool(dict(model.named_modules())[PRIMARY_MODULE].training),
        "encoder_training": bool(model.encoder.training),
        "decoder_training": bool(model.decoder.training),
        "segmentation_head_training": bool(model.segmentation_head.training),
        "center_branch_training": bool(all(module.training for module in model.center_branch_modules())),
    }


def _summarize_smoke(smoke: dict[str, Any]) -> dict[str, Any]:
    center_prefixes_ok = all(
        bool(stats["gradient_tensor_count"] > 0 and stats["all_finite"] and stats["any_nonzero"])
        for stats in smoke["center_branch_stats"].values()
    )
    return {
        "feature_extraction": "passed"
        if (
            list((smoke["primary_capture_shape"] or [None, None, None, None])[1:]) == [32, 192, 192]
            and list((smoke["context_capture_shape"] or [None, None, None, None])[1:]) == [48, 96, 96]
            and list((smoke["resolved_center_feature_shape"] or [None, None, None, None])[1:]) == [32, 192, 192]
            and not bool(smoke["feature_extraction_requires_hooks"])
        )
        else "failed",
        "forward": "passed" if smoke["semantic_logits_finite"] and smoke["resolved_center_features_finite"] and smoke["center_logits_finite"] else "failed",
        "loss": "passed" if smoke["center_loss_finite"] else "failed",
        "backward": "passed" if center_prefixes_ok and smoke["x2_2_grad_stats"]["any_nonzero"] else "failed",
        "x2_2_gradients": "passed"
        if (
            smoke["x2_2_grad_stats"]["gradient_tensor_count"] > 0
            and smoke["x2_2_grad_stats"]["all_finite"]
            and smoke["x2_2_grad_stats"]["any_nonzero"]
        )
        else "failed",
        "fusion_gradients": "passed" if center_prefixes_ok else "failed",
        "deeper_feature_gradients": "passed" if smoke["context_grad_stats"]["gradient_tensor_count"] == 0 else "failed",
        "forbidden_gradients": "passed"
        if (
            smoke["encoder_grad_stats"]["gradient_tensor_count"] == 0
            and smoke["decoder_other_grad_stats"]["gradient_tensor_count"] == 0
            and smoke["segmentation_head_grad_stats"]["gradient_tensor_count"] == 0
        )
        else "failed",
        "semantic_forward": "passed" if smoke["semantic_logits_shape"][1:] == [3, 768, 768] else "failed",
        "raw": smoke,
    }


def run(*, output_dir: Path, config_out: Path, device: torch.device) -> dict[str, Any]:
    control_cfg = _read_yaml(CONTROL_CONFIG_PATH)
    control_model = _build_model(control_cfg)
    control_freeze_info = _apply_training_policy(control_model, control_cfg)
    _optimizer, control_optimizer_meta = _build_optimizer_groups(control_model, control_cfg, control_freeze_info, freeze_base=True)
    topology = _trace_decoder_blocks(control_model, input_size=int(control_cfg["model"]["input_size"]))
    candidates = _candidate_nodes(topology)
    selected_context = _select_context_candidate(copy.deepcopy(candidates))
    new_cfg, diff_paths, contract_audit = _build_config(
        control_cfg,
        selected_context=selected_context,
        control_center_param_count=_center_group_count(control_optimizer_meta),
    )
    _write_yaml(config_out, new_cfg)

    smoke = _run_smoke(new_cfg, device=device)
    smoke_summary = _summarize_smoke(smoke)
    readiness = {
        "status": "ready_for_training"
        if all(smoke_summary[key] == "passed" for key in smoke_summary if key != "raw") and not contract_audit["forbidden_trainable_parameters"]
        else "blocked",
        "config_path": str(config_out.resolve()),
        "save_dir": str(Path(new_cfg["train"]["save_dir"]).resolve()),
        "selected_context_module": str(selected_context["module_path"]),
        "changed_fields_vs_control": diff_paths,
        "topology_candidates": [candidate["module_path"] for candidate in candidates],
        "trainable_contract": contract_audit,
        "smoke_summary": {key: value for key, value in smoke_summary.items() if key != "raw"},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "feature_topology.json", {"x2_2": topology[PRIMARY_MODULE], "candidate_nodes": candidates, "selected_context": selected_context, "all_blocks": topology})
    _write_json(output_dir / "trainable_contract.json", contract_audit)
    _write_json(output_dir / "smoke_test_summary.json", smoke_summary)
    _write_json(output_dir / "readiness_summary.json", readiness)
    _write_text(
        output_dir / "files_to_review.txt",
        "\n".join(
            [
                str((output_dir / "feature_topology.json").resolve()),
                str((output_dir / "trainable_contract.json").resolve()),
                str((output_dir / "smoke_test_summary.json").resolve()),
                str((output_dir / "readiness_summary.json").resolve()),
                str(config_out.resolve()),
            ]
        )
        + "\n",
    )
    return readiness


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    ap.add_argument("--config-out", type=str, default=str(NEW_CONFIG_PATH))
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    result = run(output_dir=Path(args.output_dir).resolve(), config_out=Path(args.config_out).resolve(), device=torch.device(args.device))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
