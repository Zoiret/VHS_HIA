from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bridge_presence_gate_v5_utility_aligned_dev as v5
import bridge_suppression_head as bridge
import train_bridge_presence_gate_v5_utility_aligned_dev as runner


class TestTrainBridgePresenceGateV5UtilityAlignedDev(unittest.TestCase):
    def _fake_prepared(self):
        gate_model = torch.nn.Sequential(torch.nn.Linear(105, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
        return {
            "manifest_stage": {"manifest": {"contract": {"train_summary": {"sample_count": 121, "patient_count": 17}}}},
            "device": torch.device("cpu"),
            "frozen_model": mock.Mock(base=torch.nn.Linear(2, 2), bridge_head=torch.nn.Linear(2, 2)),
            "frozen_v2_checkpoint": {"checkpoint_file_sha256": "abc", "checkpoint_model_state_sha256": "def"},
            "train_prepared": {
                "cached_records": [{"sample_id": "t1"}],
                "frozen_logits": torch.ones((1, 1, 2, 2)),
                "frozen_logit_diagnostics": {"input_devices": {"x_0_4": "cpu", "x_2_2": "cpu", "p_leaf": "cpu"}},
                "gate_model": gate_model,
                "features_t": torch.ones((1, 105)),
                "targets_t": torch.ones((1, 1)),
                "feature_rows": [{"sample_id": "t1", "bridge_positive_target": 1, "utility_gate_target": 1, "candidate_fraction": 0.1}],
                "hard_gate_state_cache": [{"sample_id": "t1", "gate_target": 1, "closed": {"predicted_success50": 0, "predicted_mean_iou": 0.1}, "open": {"predicted_success50": 1, "predicted_mean_iou": 0.2}}],
                "utility_target_rows": [{"sample_id": "t1", "utility_gate_target": 1}],
            },
            "val_prepared": {
                "features_t": torch.ones((1, 105)),
                "feature_rows": [{"sample_id": "v1", "bridge_positive_target": 1, "candidate_fraction": 0.1}],
                "hard_gate_state_cache": [{"sample_id": "v1", "gate_target": 1, "bridge_positive": 1, "closed": {"predicted_success50": 0, "predicted_mean_iou": 0.1, "start_mean_iou": 0.1, "component_topology_changed": 0, "predicted_removed_pixels": 0, "candidate_pixels": 10}, "open": {"predicted_success50": 1, "predicted_mean_iou": 0.2, "start_mean_iou": 0.1, "component_topology_changed": 0, "predicted_removed_pixels": 1, "candidate_pixels": 10}}],
                "state_summary": {
                    "always_closed": {"positive_success50": 1, "positive_mean_matched_iou": 0.4234538944043546},
                    "always_open": {"positive_success50": 2, "positive_mean_matched_iou": 0.5061690111512952, "negative_regressions": 20, "negative_topology_changes": 24},
                    "two_state_positive_success50_union_upper_bound": 3,
                },
            },
            "success_criteria_v2": {
                "utility": {"positive_success50_min": 2, "positive_mean_matched_iou_min": 0.47342348408224233},
                "safety": {"negative_regressions": 0, "negative_topology_changes": 0},
            },
            "v5_utility_target_summary": {"utility_open_count": 1, "utility_closed_count": 0},
            "v5_bridge_positive_agreement": {"agreement_fraction": 1.0},
            "v5_feature_feasibility": {"class_balance": {"utility_open_count": 1}},
        }

    def test_prepare_training_inputs_overrides_train_targets_only(self):
        base = {
            "train_prepared": {
                "feature_rows": [{"sample_id": "t1", "bridge_positive_target": 1, "candidate_fraction": 0.1, "candidate_component_count": 1.0, "candidate_pixels": 10, "candidate_fraction_denominator": 100, "bridge_score_mean": 0.1, "bridge_score_max": 0.1, "bridge_score_top1pct_mean": 0.1, "bridge_score_top5pct_mean": 0.1, "bridge_score_frac_ge_0p50": 0.0, "bridge_score_frac_ge_0p75": 0.0, "bridge_score_frac_ge_0p90": 0.0}],
                "hard_gate_state_cache": [{"sample_id": "t1", "gate_target": 1, "bridge_positive": 1, "closed": {"predicted_success50": 0, "predicted_mean_iou": 0.1}, "open": {"predicted_success50": 1, "predicted_mean_iou": 0.2}}],
                "features_t": torch.ones((1, 105)),
                "targets_t": torch.zeros((1, 1)),
            },
            "val_prepared": {"feature_rows": [{"sample_id": "v1"}], "hard_gate_state_cache": [{"sample_id": "v1", "gate_target": 1}]},
            "success_criteria_v2": {},
        }
        with mock.patch.object(runner.v4_runner, "_prepare_training_inputs_core", return_value=base):
            prepared = runner._prepare_training_inputs({})
        self.assertTrue(torch.equal(prepared["train_prepared"]["targets_t"], torch.tensor([[1.0]], dtype=torch.float32)))
        self.assertEqual(prepared["train_prepared"]["hard_gate_state_cache"][0]["gate_target"], 1)
        self.assertEqual(prepared["val_prepared"]["hard_gate_state_cache"][0]["gate_target"], 1)

    def test_validation_not_used_during_optimization_or_checkpoint_selection(self):
        prepared = self._fake_prepared()
        with mock.patch.object(runner, "_prepare_training_inputs", return_value=prepared), \
             mock.patch.object(runner.dev, "assert_locked_val_references"), \
             mock.patch.object(runner.dev, "assert_locked_active_success_criterion_v2"), \
             mock.patch.object(runner.dev, "snapshot_frozen_backbone_state", return_value={"named": [], "params": {}, "bn": {}}), \
             mock.patch.object(runner.dev, "frozen_backbone_invariant_deltas", return_value={"semantic_parameter_max_delta": 0.0, "semantic_bn_state_max_delta": 0.0, "v2_pixel_head_parameter_max_delta": 0.0}), \
             mock.patch.object(runner.v4_runner, "_train_only_run", return_value={"best_train_loss": 0.1, "max_steps": 300}), \
             mock.patch.object(runner.v4_runner, "_load_gate_checkpoint", return_value={"step": 1}), \
             mock.patch.object(runner.v4_runner, "_gate_probabilities", side_effect=[np.array([1.0]), np.array([1.0])]), \
             mock.patch.object(runner.v4_runner, "_evaluate_split", side_effect=[{"trained_v4": {"per_sample_detailed": [], "patient_level_exploratory": {}}, "generalization_pass": {"status_text": "NO"}}, {"trained_v4": {"per_sample_detailed": [], "patient_level_exploratory": {}}, "generalization_pass": {"status_text": "NO"}}]):
            with tempfile.TemporaryDirectory() as td:
                cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v5_utility_aligned_dev_v1.yaml")
                cfg = dict(cfg)
                cfg["train"] = dict(cfg["train"])
                cfg["train"]["save_dir"] = str(Path(td) / "run")
                summary = runner.run_pipeline(cfg)
        self.assertTrue(summary["reused_development_validation"])
        self.assertFalse(summary["fresh_unseen_development_validation"])
        self.assertEqual(summary["training_contract"]["threshold"], 0.50)


if __name__ == "__main__":
    unittest.main()
