from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import torch


_THIS_DIR = Path(__file__).resolve().parent
_ACTUAL_MULTISCALE_CONFIG_PATH = (
    _THIS_DIR / "configs" / "unetpp_effb3_centerhead_multiscale_x2_2_x1_1_full_dataset_aug_100ep.yaml"
)


def _synthetic_centers(device: torch.device) -> torch.Tensor:
    centers = torch.zeros((1, 1, 768, 768), dtype=torch.float32, device=device)
    for y, x in [(128, 128), (384, 352), (640, 512)]:
        centers[0, 0, int(y), int(x)] = 1.0
    return centers


def _grad_stats(named_params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, object]:
    grads = [p.grad.detach() for _name, p in named_params if p.grad is not None]
    return {
        "gradient_tensor_count": int(len(grads)),
        "all_finite": bool(all(bool(torch.isfinite(g).all().item()) for g in grads)),
        "any_nonzero": bool(any(bool(torch.count_nonzero(g).item()) for g in grads)),
    }


def _max_delta(named_tensors: list[tuple[str, torch.Tensor]], snap: dict[str, torch.Tensor]) -> float:
    max_delta = 0.0
    for name, tensor in named_tensors:
        ref = snap[str(name)]
        delta = float((tensor.detach() - ref).abs().max().item()) if tensor.numel() else 0.0
        max_delta = max(max_delta, delta)
    return float(max_delta)


class TestPrepareMultiscaleCenterRepresentationBaseline(unittest.TestCase):
    @staticmethod
    def _prep():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import prepare_multiscale_center_representation_baseline as mod

        return mod

    @staticmethod
    def _train():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import train_centerhead as mod

        return mod

    @classmethod
    def setUpClass(cls):
        cls.prep = cls._prep()
        cls.train = cls._train()
        cls.control_cfg = cls.prep._read_yaml(cls.prep.CONTROL_CONFIG_PATH)
        cls.control_model = cls.train._build_model(cls.control_cfg)
        cls.control_freeze = cls.train._apply_training_policy(cls.control_model, cls.control_cfg)
        cls._control_optimizer, cls.control_optimizer_meta = cls.train._build_optimizer_groups(cls.control_model, cls.control_cfg, cls.control_freeze, freeze_base=True)
        cls.topology = cls.prep._trace_decoder_blocks(cls.control_model, input_size=int(cls.control_cfg["model"]["input_size"]))
        cls.candidates = cls.prep._candidate_nodes(cls.topology)
        cls.selected = cls.prep._select_context_candidate(copy.deepcopy(cls.candidates))
        cls.new_cfg, cls.diff_paths, cls.contract_audit = cls.prep._build_config(
            cls.control_cfg,
            selected_context=cls.selected,
            control_center_param_count=cls.prep._center_group_count(cls.control_optimizer_meta),
        )
        cls.actual_multiscale_cfg = cls.train._read_yaml(_ACTUAL_MULTISCALE_CONFIG_PATH)

    def _build_actual_multiscale_model(self, device: torch.device) -> tuple[torch.nn.Module, dict, list[dict]]:
        model = self.train._build_model(self.actual_multiscale_cfg).to(device)
        freeze_info = self.train._apply_training_policy(model, self.actual_multiscale_cfg)
        self.train._set_train_modes(model, freeze_base=True)
        _optimizer, optimizer_meta = self.train._build_optimizer_groups(
            model,
            self.actual_multiscale_cfg,
            freeze_info,
            freeze_base=True,
        )
        return model, freeze_info, optimizer_meta

    def test_exact_deeper_feature_extraction(self):
        self.assertEqual(self.selected["module_path"], self.prep.SELECTED_CONTEXT_MODULE)
        self.assertFalse(self.selected["requires_hooks"])
        self.assertEqual(self.selected["relation_to_x2_2"], "parallel_to_x_2_2")
        self.assertEqual(self.selected["shape"], [1, 48, 96, 96])

    def test_tensor_shape_stride_contract(self):
        primary = self.topology[self.prep.PRIMARY_MODULE]
        context = self.topology[self.prep.SELECTED_CONTEXT_MODULE]
        self.assertEqual(primary["shape"], [1, 32, 192, 192])
        self.assertEqual(primary["native_stride"], 4)
        self.assertEqual(context["shape"], [1, 48, 96, 96])
        self.assertEqual(context["native_stride"], 8)

    def test_projection_and_fusion_dimensions(self):
        model = self.train._build_model(self.new_cfg)
        x = torch.zeros(1, 3, 768, 768)
        semantic, decoder_output = model.forward_base(x)
        fused = model.resolve_center_features(decoder_output)
        center_logits = model.forward_center_from_features(fused)
        capture = model.center_feature_capture_info()
        self.assertEqual(capture["captured_shape"], [1, 32, 192, 192])
        self.assertEqual(capture["context"]["captured_shape"], [1, 48, 96, 96])
        self.assertEqual(fused.shape, torch.Size([1, 32, 192, 192]))
        self.assertEqual(semantic.shape, torch.Size([1, 3, 768, 768]))
        self.assertEqual(center_logits.shape, torch.Size([1, 1, 768, 768]))

    def test_exact_trainable_parameter_membership(self):
        model = self.train._build_model(self.new_cfg)
        freeze_info = self.train._apply_training_policy(model, self.new_cfg)
        trainable = set(freeze_info["trainable_names"])
        self.assertTrue(any(name.startswith(self.prep.PRIMARY_MODULE + ".") for name in trainable))
        self.assertTrue(any(name.startswith("center_primary_projection.") for name in trainable))
        self.assertTrue(any(name.startswith("center_context_projection.") for name in trainable))
        self.assertTrue(any(name.startswith("center_fusion_adapter.") for name in trainable))
        self.assertEqual(self.contract_audit["forbidden_trainable_parameters"], [])

    def test_no_forbidden_optimizer_parameters(self):
        self.assertEqual(self.contract_audit["forbidden_trainable_parameters"], [])
        groups = self.contract_audit["optimizer_groups"]
        self.assertEqual(len(groups), 2)
        group_names = {group["name"] for group in groups}
        self.assertEqual(group_names, {"center_head", "unfrozen_decoder"})

    def test_optimizer_lr_groups(self):
        groups = self.contract_audit["optimizer_groups"]
        center = next(group for group in groups if group["name"] == "center_head")
        x2_2 = next(group for group in groups if group["name"] == "unfrozen_decoder")
        self.assertAlmostEqual(float(center["lr"]), 1.0e-3, places=12)
        self.assertAlmostEqual(float(x2_2["lr"]), 1.0e-5, places=12)
        self.assertEqual(set(x2_2["parameter_names"]), set(self.contract_audit["x2_2_parameter_names"]))

    def test_frozen_batchnorm_eval_behavior(self):
        model = self.train._build_model(self.new_cfg)
        self.train._apply_training_policy(model, self.new_cfg)
        self.train._set_train_modes(model, freeze_base=True)
        context_module = dict(model.named_modules())[self.prep.SELECTED_CONTEXT_MODULE]
        x2_2_module = dict(model.named_modules())[self.prep.PRIMARY_MODULE]
        self.assertFalse(model.base.training)
        self.assertFalse(context_module.training)
        self.assertTrue(x2_2_module.training)
        self.assertTrue(all(not module.training for module in context_module.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)))
        self.assertTrue(all(module.training for module in x2_2_module.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)))

    def test_repeated_model_train_cannot_reactivate_frozen_base(self):
        model = self.train._build_model(self.new_cfg)
        self.train._apply_training_policy(model, self.new_cfg)
        model.train()
        self.train._set_train_modes(model, freeze_base=True)
        model.train()
        self.train._set_train_modes(model, freeze_base=True)
        self.assertFalse(model.base.training)
        self.assertFalse(dict(model.named_modules())[self.prep.SELECTED_CONTEXT_MODULE].training)
        self.assertTrue(dict(model.named_modules())[self.prep.PRIMARY_MODULE].training)

    def test_best_primary_checkpoint_selection_unchanged(self):
        rows = [
            {
                "threshold": 0.01,
                "center_f1_mean_samples": 0.30,
                "strict_marker_contract_pass_rate": 0.20,
                "exact_center_count_accuracy": 0.40,
                "localization_error_px_pooled_matches": 4.0,
            },
            {
                "threshold": 0.02,
                "center_f1_mean_samples": 0.30,
                "strict_marker_contract_pass_rate": 0.25,
                "exact_center_count_accuracy": 0.35,
                "localization_error_px_pooled_matches": 3.0,
            },
        ]
        best = self.train._select_best_threshold_row(rows, primary_metric="center_f1_mean_samples")
        self.assertEqual(float(best["threshold"]), 0.02)

    def test_best_strict_checkpoint_selection(self):
        rows = [
            {
                "threshold": 0.01,
                "center_f1_mean_samples": 0.40,
                "strict_marker_contract_pass_rate": 0.20,
                "exact_center_count_accuracy": 0.50,
                "localization_error_px_pooled_matches": 3.0,
            },
            {
                "threshold": 0.02,
                "center_f1_mean_samples": 0.30,
                "strict_marker_contract_pass_rate": 0.35,
                "exact_center_count_accuracy": 0.40,
                "localization_error_px_pooled_matches": 5.0,
            },
        ]
        best = self.train._select_best_threshold_row(rows, primary_metric="strict_marker_contract_pass_rate")
        self.assertEqual(float(best["threshold"]), 0.02)

    def test_strict_checkpoint_tie_break(self):
        rows = [
            {
                "threshold": 0.01,
                "center_f1_mean_samples": 0.20,
                "strict_marker_contract_pass_rate": 0.30,
                "exact_center_count_accuracy": 0.60,
                "localization_error_px_pooled_matches": 2.0,
            },
            {
                "threshold": 0.02,
                "center_f1_mean_samples": 0.25,
                "strict_marker_contract_pass_rate": 0.30,
                "exact_center_count_accuracy": 0.55,
                "localization_error_px_pooled_matches": 1.0,
            },
            {
                "threshold": 0.03,
                "center_f1_mean_samples": 0.25,
                "strict_marker_contract_pass_rate": 0.30,
                "exact_center_count_accuracy": 0.55,
                "localization_error_px_pooled_matches": 1.5,
            },
        ]
        best = self.train._select_best_threshold_row(rows, primary_metric="strict_marker_contract_pass_rate")
        self.assertEqual(float(best["threshold"]), 0.02)

    def test_augmentation_identical_to_control(self):
        self.assertEqual(self.new_cfg["augment"], self.control_cfg["augment"])

    def test_loss_threshold_and_max_markers_unchanged(self):
        self.assertEqual(self.new_cfg["center_loss"], self.control_cfg["center_loss"])
        self.assertEqual(self.new_cfg["center"], self.control_cfg["center"])
        self.assertEqual(self.new_cfg["validation"], self.control_cfg["validation"])

    def test_no_authoritative_holdout_path_referenced(self):
        dumped = json.dumps(self.new_cfg, sort_keys=True).lower()
        self.assertNotIn("holdout", dumped)

    def test_center_fp32_precision_boundary_handles_multiscale_half_features(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        amp_enabled = device.type == "cuda"
        model, freeze_info, optimizer_meta = self._build_actual_multiscale_model(device)
        self.assertEqual(freeze_info["trainable_base_modules"], ["base.decoder.blocks.x_2_2"])
        self.assertEqual({group["name"] for group in optimizer_meta}, {"center_head", "unfrozen_decoder"})

        images = torch.randn(1, 3, 768, 768, device=device)
        centers = _synthetic_centers(device)
        center_loss_fn = self.train.CenterNetFocalHeatmapLoss(alpha=2.0, beta=4.0, normalization_mode="legacy_num_pos").to(device)

        model.zero_grad(set_to_none=True)
        _semantic_logits, decoder_output = self.train._forward_base_for_center_training(
            model=model,
            images=images,
            device=device,
            amp_enabled_global=amp_enabled,
            detach_output=False,
            no_grad=False,
        )
        if device.type == "cuda":
            expected_input_dtype = str(decoder_output.dtype).replace("torch.", "")
            self.assertNotEqual(expected_input_dtype, "float32")
        else:
            decoder_output = decoder_output.to(dtype=torch.float16)
            for path, feat in list(model._captured_center_features.items()):
                if torch.is_tensor(feat):
                    model._captured_center_features[str(path)] = feat.to(dtype=torch.float16)
            expected_input_dtype = "float16"

        _decoder_features, center_logits, center_loss, precision_info = self.train._forward_center_with_precision(
            model=model,
            decoder_output=decoder_output,
            centers=centers,
            center_loss_fn=center_loss_fn,
            device=device,
            amp_enabled_global=amp_enabled,
            center_fp32=True,
            detach_decoder_output=False,
            return_details=False,
        )
        self.assertTrue(torch.is_tensor(center_loss))
        self.assertTrue(bool(torch.isfinite(center_loss).all().item()))
        resolve_info = precision_info["center_feature_resolve_info"]
        self.assertEqual(precision_info["decoder_output_dtype_before_center_boundary"], expected_input_dtype)
        self.assertEqual(resolve_info["primary_before_dtype"], expected_input_dtype)
        self.assertEqual(resolve_info["context_before_dtype"], expected_input_dtype)
        self.assertEqual(resolve_info["primary_after_dtype"], "float32")
        self.assertEqual(resolve_info["context_after_dtype"], "float32")
        self.assertEqual(precision_info["center_primary_projection_weight_dtype"], "float32")
        self.assertEqual(precision_info["center_context_projection_weight_dtype"], "float32")
        self.assertEqual(precision_info["center_fusion_adapter_weight_dtype"], "float32")
        self.assertEqual(precision_info["center_adapter_weight_dtype"], "float32")
        self.assertEqual(precision_info["center_head_output_weight_dtype"], "float32")
        self.assertEqual(precision_info["center_logits_dtype"], "float32")

        center_loss.backward()

        x2_2_named = [(n, p) for n, p in model.named_parameters() if n.startswith("base.decoder.blocks.x_2_2.")]
        context_named = [(n, p) for n, p in model.named_parameters() if n.startswith("base.decoder.blocks.x_1_1.")]
        center_named = [
            (n, p)
            for n, p in model.named_parameters()
            if any(
                n.startswith(prefix)
                for prefix in [
                    "center_primary_projection.",
                    "center_context_projection.",
                    "center_fusion_adapter.",
                    "center_adapter.",
                    "center_head.",
                ]
            )
        ]
        forbidden_named = [
            (n, p)
            for n, p in model.named_parameters()
            if (
                n.startswith("base.encoder.")
                or n.startswith("base.segmentation_head.")
                or (n.startswith("base.decoder.") and not n.startswith("base.decoder.blocks.x_2_2.") and not n.startswith("base.decoder.blocks.x_1_1."))
            )
        ]
        x2_2_grad = _grad_stats(x2_2_named)
        context_grad = _grad_stats(context_named)
        center_grad = _grad_stats(center_named)
        forbidden_grad = _grad_stats(forbidden_named)
        self.assertGreater(x2_2_grad["gradient_tensor_count"], 0)
        self.assertTrue(x2_2_grad["all_finite"])
        self.assertTrue(x2_2_grad["any_nonzero"])
        self.assertEqual(context_grad["gradient_tensor_count"], 0)
        self.assertGreater(center_grad["gradient_tensor_count"], 0)
        self.assertTrue(center_grad["all_finite"])
        self.assertTrue(center_grad["any_nonzero"])
        self.assertEqual(forbidden_grad["gradient_tensor_count"], 0)

    def test_freeze_mode_immutability_for_exact_multiscale_config(self):
        device = torch.device("cpu")
        model, freeze_info, _optimizer_meta = self._build_actual_multiscale_model(device)
        optimizer, _ = self.train._build_optimizer_groups(model, self.actual_multiscale_cfg, freeze_info, freeze_base=True)
        center_loss_fn = self.train.CenterNetFocalHeatmapLoss(alpha=2.0, beta=4.0, normalization_mode="legacy_num_pos").to(device)

        named_modules = dict(model.named_modules())
        self.assertFalse(model.base.training)
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.base.decoder.training)
        self.assertFalse(model.segmentation_head.training)
        self.assertFalse(named_modules["base.decoder.blocks.x_1_1"].training)
        self.assertTrue(named_modules["base.decoder.blocks.x_2_2"].training)
        for module in model.center_branch_modules():
            self.assertTrue(module.training)

        frozen_bn_training_true = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and (not name.startswith("base.decoder.blocks.x_2_2"))
            and bool(module.training)
        ]
        self.assertEqual(frozen_bn_training_true, [])

        frozen_named_params = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
        x2_2_named_params = [(n, p) for n, p in model.named_parameters() if n.startswith("base.decoder.blocks.x_2_2.") and p.requires_grad]
        center_named_params = [
            (n, p)
            for n, p in model.named_parameters()
            if p.requires_grad
            and any(
                n.startswith(prefix)
                for prefix in [
                    "center_primary_projection.",
                    "center_context_projection.",
                    "center_fusion_adapter.",
                    "center_adapter.",
                    "center_head.",
                ]
            )
        ]
        forbidden_grad_named = [
            (n, p)
            for n, p in model.named_parameters()
            if (
                n.startswith("base.encoder.")
                or n.startswith("base.segmentation_head.")
                or n.startswith("base.decoder.blocks.x_1_1.")
                or (n.startswith("base.decoder.") and not n.startswith("base.decoder.blocks.x_2_2.") and not n.startswith("base.decoder.blocks.x_1_1."))
            )
        ]
        frozen_buffer_names = [
            (n, b)
            for n, b in model.named_buffers()
            if not n.startswith("base.decoder.blocks.x_2_2.")
        ]
        frozen_param_snapshot = {n: p.detach().clone() for n, p in frozen_named_params}
        x2_2_snapshot = {n: p.detach().clone() for n, p in x2_2_named_params}
        center_snapshot = {n: p.detach().clone() for n, p in center_named_params}
        frozen_buffer_snapshot = {n: b.detach().clone() for n, b in frozen_buffer_names}

        images = torch.randn(1, 3, 768, 768, device=device)
        centers = _synthetic_centers(device)
        optimizer.zero_grad(set_to_none=True)
        _semantic_logits, decoder_output = self.train._forward_base_for_center_training(
            model=model,
            images=images,
            device=device,
            amp_enabled_global=False,
            detach_output=False,
            no_grad=False,
        )
        _decoder_features, _center_logits, center_loss, precision_info = self.train._forward_center_with_precision(
            model=model,
            decoder_output=decoder_output,
            centers=centers,
            center_loss_fn=center_loss_fn,
            device=device,
            amp_enabled_global=False,
            center_fp32=True,
            detach_decoder_output=False,
            return_details=False,
        )
        self.assertEqual(precision_info["center_feature_resolve_info"]["primary_after_dtype"], "float32")
        self.assertEqual(precision_info["center_feature_resolve_info"]["context_after_dtype"], "float32")
        center_loss.backward()

        forbidden_grad = _grad_stats(forbidden_grad_named)
        x2_2_grad = _grad_stats(x2_2_named_params)
        center_grad = _grad_stats(center_named_params)
        self.assertEqual(forbidden_grad["gradient_tensor_count"], 0)
        self.assertGreater(x2_2_grad["gradient_tensor_count"], 0)
        self.assertTrue(x2_2_grad["all_finite"])
        self.assertTrue(x2_2_grad["any_nonzero"])
        self.assertGreater(center_grad["gradient_tensor_count"], 0)
        self.assertTrue(center_grad["all_finite"])
        self.assertTrue(center_grad["any_nonzero"])

        optimizer.step()

        self.assertEqual(_max_delta(frozen_named_params, frozen_param_snapshot), 0.0)
        self.assertEqual(_max_delta(frozen_buffer_names, frozen_buffer_snapshot), 0.0)
        self.assertGreater(_max_delta(x2_2_named_params, x2_2_snapshot), 0.0)
        self.assertGreater(_max_delta(center_named_params, center_snapshot), 0.0)


if __name__ == "__main__":
    unittest.main()
