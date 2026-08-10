from __future__ import annotations

import sys
import unittest
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent


class TestPrepareX22AdaptiveCenterBaseline(unittest.TestCase):
    @staticmethod
    def _prep():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import prepare_x2_2_adaptive_center_baseline as mod

        return mod

    @classmethod
    def setUpClass(cls):
        cls.mod = cls._prep()
        cls.aug_cfg = cls.mod._read_yaml(cls.mod.AUG_BASELINE_CONFIG_PATH)
        cfg = dict(cls.aug_cfg)
        cfg["train"] = dict((cls.aug_cfg.get("train") or {}), trainable_base_modules=[cls.mod.TARGET_MODULE], lr_unfrozen_decoder=float(cls.mod.X2_2_LR))
        cls.audit = cls.mod._x2_2_module_audit(cfg)
        cls.new_cfg, cls.diff_paths = cls.mod._build_config(cls.aug_cfg, cls.audit)

    def test_exact_x2_2_only_unfreeze(self):
        self.assertEqual(self.audit["module"], self.mod.TARGET_MODULE)
        self.assertEqual(self.audit["forbidden_trainable_parameters"], [])
        self.assertTrue(all(name.startswith(self.mod.TARGET_MODULE + ".") for name in self.audit["x2_2_parameter_names"]))
        self.assertTrue(all(name.startswith("center_head.") or name.startswith("center_adapter.") for name in self.audit["center_parameter_names"]))

    def test_no_other_decoder_parameters_trainable(self):
        self.assertEqual(self.audit["forbidden_optimizer_parameters"], [])
        self.assertTrue(self.audit["isolated_unfreeze_safe"])

    def test_optimizer_group_membership_and_lr_exact(self):
        groups = self.audit["optimizer_groups"]
        self.assertEqual(len(groups), 2)
        center = next(group for group in groups if group["name"] == "center_head")
        other = next(group for group in groups if group["name"] != "center_head")
        self.assertAlmostEqual(float(center["lr"]), 1.0e-3, places=12)
        self.assertAlmostEqual(float(other["lr"]), 1.0e-5, places=12)
        self.assertEqual(set(other["parameter_names"]), set(self.audit["x2_2_parameter_names"]))

    def test_semantic_checkpoint_initialization_exact(self):
        report = self.audit["semantic_init_report"]
        self.assertEqual(report["status"], "exact")
        self.assertEqual(report["unexpected_keys"], [])
        self.assertEqual(report["disallowed_missing_keys"], [])

    def test_augmentation_config_unchanged_versus_augmented_baseline(self):
        self.assertEqual(self.new_cfg["augment"], self.aug_cfg["augment"])

    def test_loss_max_markers_and_threshold_sweep_unchanged(self):
        self.assertEqual(self.new_cfg["center_loss"], self.aug_cfg["center_loss"])
        self.assertEqual(self.new_cfg["center"], self.aug_cfg["center"])
        self.assertEqual(self.new_cfg["validation"], self.aug_cfg["validation"])

    def test_config_changes_are_controlled(self):
        self.assertIn("train.trainable_base_modules", set(self.diff_paths))
        self.assertIn("train.lr_unfrozen_decoder", set(self.diff_paths))
        self.assertIn("train.save_dir", set(self.diff_paths))
        self.assertEqual(self.new_cfg["train"]["save_dir"], self.mod.NEW_SAVE_DIR)

    def test_summarize_smoke_contract(self):
        smoke = {
            "semantic_shape": (1, 3, 768, 768),
            "center_shape": (1, 1, 768, 768),
            "semantic_loss_finite": True,
            "loss_total": 1.0,
            "center_grad_all_finite": True,
            "center_grad_norm_before_clip": 2.0,
            "decoder_grad_norm_before_clip": 1.0,
            "selected_decoder_parameter_delta": 0.01,
            "frozen_encoder_grad_count": 0,
            "frozen_decoder_grad_count": 0,
            "semantic_head_grad_count": 0,
            "frozen_parameter_max_delta": 0.0,
        }
        summary = self.mod._summarize_smoke(smoke)
        self.assertEqual(summary["forward"], "passed")
        self.assertEqual(summary["center_gradients"], "passed")
        self.assertEqual(summary["x2_2_gradients"], "passed")
        self.assertEqual(summary["forbidden_gradients"], "passed")


if __name__ == "__main__":
    unittest.main()
