from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bridge_presence_gate_v4_patient_disjoint_dev as dev
import bridge_presence_gate_v5_utility_aligned_dev as v5
import bridge_suppression_head as bridge


class TestBridgePresenceGateV5UtilityAlignedDev(unittest.TestCase):
    def _feature_rows(self):
        return [
            {
                "sample_id": "p1",
                "patient_id": "pat1",
                "gt_count": 2,
                "bridge_positive_target": 1,
                "candidate_pixels": 10,
                "candidate_fraction_denominator": 100,
                "candidate_fraction": 0.10,
                "candidate_component_count": 1.0,
                "bridge_score_mean": 0.1,
                "bridge_score_max": 0.2,
                "bridge_score_top1pct_mean": 0.2,
                "bridge_score_top5pct_mean": 0.2,
                "bridge_score_frac_ge_0p50": 0.0,
                "bridge_score_frac_ge_0p75": 0.0,
                "bridge_score_frac_ge_0p90": 0.0,
            },
            {
                "sample_id": "p2",
                "patient_id": "pat2",
                "gt_count": 3,
                "bridge_positive_target": 1,
                "candidate_pixels": 20,
                "candidate_fraction_denominator": 100,
                "candidate_fraction": 0.20,
                "candidate_component_count": 2.0,
                "bridge_score_mean": 0.3,
                "bridge_score_max": 0.4,
                "bridge_score_top1pct_mean": 0.4,
                "bridge_score_top5pct_mean": 0.4,
                "bridge_score_frac_ge_0p50": 0.0,
                "bridge_score_frac_ge_0p75": 0.0,
                "bridge_score_frac_ge_0p90": 0.0,
            },
            {
                "sample_id": "n1",
                "patient_id": "pat3",
                "gt_count": 1,
                "bridge_positive_target": 0,
                "candidate_pixels": 30,
                "candidate_fraction_denominator": 100,
                "candidate_fraction": 0.30,
                "candidate_component_count": 3.0,
                "bridge_score_mean": 0.5,
                "bridge_score_max": 0.6,
                "bridge_score_top1pct_mean": 0.6,
                "bridge_score_top5pct_mean": 0.6,
                "bridge_score_frac_ge_0p50": 1.0,
                "bridge_score_frac_ge_0p75": 0.0,
                "bridge_score_frac_ge_0p90": 0.0,
            },
        ]

    def _state_cache(self):
        return [
            {
                "sample_id": "p1",
                "gate_target": 1,
                "bridge_positive": 1,
                "closed": {"predicted_success50": 0, "predicted_mean_iou": 0.20},
                "open": {"predicted_success50": 1, "predicted_mean_iou": 0.21},
            },
            {
                "sample_id": "p2",
                "gate_target": 1,
                "bridge_positive": 1,
                "closed": {"predicted_success50": 1, "predicted_mean_iou": 0.30},
                "open": {"predicted_success50": 1, "predicted_mean_iou": 0.35},
            },
            {
                "sample_id": "n1",
                "gate_target": 0,
                "bridge_positive": 0,
                "closed": {"predicted_success50": 1, "predicted_mean_iou": 0.90},
                "open": {"predicted_success50": 1, "predicted_mean_iou": 0.95},
            },
        ]

    def test_utility_targets_derived_from_train_only_rows(self):
        rows = v5.build_utility_target_rows(feature_rows=self._feature_rows(), hard_gate_state_cache=self._state_cache())
        self.assertEqual([row["sample_id"] for row in rows], ["p1", "p2", "n1"])

    def test_negatives_always_closed(self):
        rows = v5.build_utility_target_rows(feature_rows=self._feature_rows(), hard_gate_state_cache=self._state_cache())
        neg = [row for row in rows if row["sample_id"] == "n1"][0]
        self.assertEqual(neg["utility_gate_target"], 0)
        self.assertEqual(neg["decision_reason"], "negative_forced_closed")

    def test_lexicographic_success_then_iou_then_closed_tie_rule(self):
        state_success = {"closed": {"predicted_success50": 0, "predicted_mean_iou": 0.4}, "open": {"predicted_success50": 1, "predicted_mean_iou": 0.1}}
        state_iou = {"closed": {"predicted_success50": 1, "predicted_mean_iou": 0.4}, "open": {"predicted_success50": 1, "predicted_mean_iou": 0.5}}
        state_tie = {"closed": {"predicted_success50": 1, "predicted_mean_iou": 0.4}, "open": {"predicted_success50": 1, "predicted_mean_iou": 0.4}}
        self.assertEqual(v5._utility_choice_for_positive(state_success), (1, "success_improvement"))
        self.assertEqual(v5._utility_choice_for_positive(state_iou), (1, "iou_only_improvement"))
        self.assertEqual(v5._utility_choice_for_positive(state_tie), (0, "exact_tie_closed"))

    def test_bridge_positive_agreement_quantifies_target_change(self):
        rows = v5.build_utility_target_rows(feature_rows=self._feature_rows(), hard_gate_state_cache=self._state_cache())
        agreement = v5.bridge_positive_vs_utility_target_agreement(rows)
        self.assertEqual(agreement["tp"], 2)
        self.assertEqual(agreement["tn"], 1)
        self.assertEqual(agreement["fp"], 0)
        self.assertEqual(agreement["fn"], 0)

    def test_feature_dimension_and_contract_unchanged(self):
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v5_utility_aligned_dev_v1.yaml")
        contract = v5.build_v5_training_contract(cfg)
        self.assertEqual(contract["features"], dev.FEATURE_DIMENSION)
        self.assertEqual(contract["trainable_parameters"], dev.TRAINABLE_GATE_PARAMS)
        self.assertEqual(contract["optimizer"], "AdamW")
        self.assertEqual(contract["learning_rate"], 0.001)
        self.assertEqual(contract["steps"], 300)
        self.assertEqual(contract["seed"], 1337)
        self.assertEqual(contract["threshold"], 0.50)
        self.assertEqual(contract["checkpoint_selection"]["metric"], "gate_train_loss")

    def test_targets_tensor_matches_rows(self):
        rows = v5.build_utility_target_rows(feature_rows=self._feature_rows(), hard_gate_state_cache=self._state_cache())
        targets_t = v5.build_utility_targets_tensor(rows)
        self.assertTrue(torch.equal(targets_t, torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float32)))

    def test_override_gate_targets_in_state_cache(self):
        rows = v5.build_utility_target_rows(feature_rows=self._feature_rows(), hard_gate_state_cache=self._state_cache())
        overridden = v5.override_gate_targets_in_state_cache(self._state_cache(), rows)
        self.assertEqual([int(row["gate_target"]) for row in overridden], [1, 1, 0])

    def test_conflicting_duplicate_feature_vectors_detected(self):
        rows = [
            {"sample_id": "a", "patient_id": "p1", "utility_gate_target": 0},
            {"sample_id": "b", "patient_id": "p2", "utility_gate_target": 1},
        ]
        features_t = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float32)
        conflicts = v5.find_conflicting_duplicate_feature_vectors(features_t=features_t, utility_target_rows=rows)
        self.assertEqual(len(conflicts), 1)


if __name__ == "__main__":
    unittest.main()
