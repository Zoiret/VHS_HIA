from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import torch


_THIS_DIR = Path(__file__).resolve().parent


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


if __name__ == "__main__":
    unittest.main()
