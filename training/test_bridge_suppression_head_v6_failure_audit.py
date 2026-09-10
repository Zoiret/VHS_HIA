from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bridge_suppression_head_v6_failure_audit as audit


class TestBridgeSuppressionHeadV6FailureAudit(unittest.TestCase):
    def _cached_row(self, *, sample_id: str = "s1", patient_id: str = "p1", bridge_positive: int = 0) -> dict[str, object]:
        candidate_mask_np = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        target_np = np.array([[0, 0], [0, 0]], dtype=np.uint8)
        if int(bridge_positive) == 1:
            target_np = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        return {
            "sample_id": sample_id,
            "patient_id": patient_id,
            "gt_count": 1,
            "bridge_positive": int(bridge_positive),
            "candidate_mask_np": candidate_mask_np,
            "bridge_target": torch.from_numpy(target_np[None, ...].astype(np.float32)),
            "gt_instances": np.array([[1, 1], [0, 0]], dtype=np.uint8),
            "component_count_start": 1,
            "start_reconstruction": {
                "pred_k": 1,
                "metrics": {
                    "all_iou_ge_0.50": True,
                    "instance_mean_matched_iou": 0.8,
                },
            },
        }

    def _topology_change_row(self) -> dict[str, object]:
        return {
            "sample_id": "neg_topo",
            "patient_id": "p_topo",
            "gt_count": 1,
            "bridge_positive": 0,
            "candidate_mask_np": np.array([[1, 0], [0, 1]], dtype=np.uint8),
            "bridge_target": torch.from_numpy(np.zeros((1, 2, 2), dtype=np.float32)),
            "gt_instances": np.array([[1, 0], [0, 1]], dtype=np.uint8),
            "component_count_start": 2,
            "start_reconstruction": {
                "pred_k": 1,
                "metrics": {
                    "all_iou_ge_0.50": True,
                    "instance_mean_matched_iou": 0.8,
                },
            },
        }

    def test_spatial_damage_partition_is_exhaustive_and_non_overlapping(self):
        row = self._cached_row()
        pred_remove = np.array([[1, 1], [1, 0]], dtype=bool)
        out = audit.spatial_damage_partition(row, pred_remove)
        self.assertEqual(int(np.sum(out["pred_remove_candidate"])), 2)
        self.assertEqual(int(np.sum(out["inside_gt"])), 2)
        self.assertEqual(int(np.sum(out["outside_gt"])), 0)
        self.assertEqual(int(np.sum(out["inside_gt"] & out["outside_gt"])), 0)
        self.assertEqual(
            int(np.sum(out["pred_remove_candidate"])),
            int(np.sum(out["inside_gt"]) + np.sum(out["outside_gt"])),
        )

    def test_exact_target_oracle_leaves_zero_target_samples_identical_to_closed(self):
        row = self._cached_row(bridge_positive=0)
        fake_pred = {
            "pred_k": 1,
            "metrics": {
                "all_iou_ge_0.50": True,
                "instance_mean_matched_iou": 0.8,
            },
        }
        with mock.patch.object(audit.bridge, "run_locked_reconstruction", return_value=fake_pred):
            out = audit.exact_target_oracle([row])
        self.assertTrue(out["zero_target_identical_to_closed"])
        self.assertEqual(out["negative_regressions"], 0)
        self.assertEqual(out["negative_topology_changes"], 0)

    def test_exact_target_oracle_blocks_when_zero_target_changes(self):
        row = self._cached_row(bridge_positive=0)
        bad_pred = {
            "pred_k": 2,
            "metrics": {
                "all_iou_ge_0.50": False,
                "instance_mean_matched_iou": 0.3,
            },
        }
        with mock.patch.object(audit.bridge, "run_locked_reconstruction", return_value=bad_pred):
            with self.assertRaises(SystemExit) as cm:
                audit.exact_target_oracle([row])
        self.assertIn("exact_target_oracle_zero_target_not_identical_to_closed", str(cm.exception))

    def test_authoritative_epoch_remains_63(self):
        self.assertEqual(audit.AUTHORITATIVE_V6_EPOCH, 63)
        self.assertEqual(audit.FIXED_REMOVE_THRESHOLD, 0.50)

    def test_nonzero_topology_changes_aggregate_from_component_topology_field(self):
        row = self._topology_change_row()
        probs = np.array([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=np.float32)
        fake_pred = {
            "pred_k": 1,
            "metrics": {
                "all_iou_ge_0.50": True,
                "instance_mean_matched_iou": 0.8,
            },
        }
        with mock.patch.object(audit.bridge, "run_locked_reconstruction", return_value=fake_pred):
            out = audit.evaluate_prob_set_on_cached_records([row], probs)
        self.assertEqual(out["reconstruction"]["negative_topology_changes"], 1)
        self.assertEqual(out["per_sample"][0]["component_topology_changed"], 1)

    def test_missing_topology_field_fails_closed(self):
        eval_payload = {
            "reconstruction": {"negative_topology_changes": 1},
            "per_sample": [{"sample_id": "s1", "bridge_positive": 0}],
        }
        with self.assertRaises(SystemExit) as cm:
            audit.validate_topology_consistency(
                eval_payload=eval_payload,
                patient_payload=None,
                expected_negative_topology_changes=None,
                context="missing_field_test",
            )
        self.assertIn("missing_required_fields", str(cm.exception))

    def test_patient_topology_sum_must_equal_global(self):
        eval_payload = {
            "reconstruction": {"negative_topology_changes": 2},
            "per_sample": [
                {"sample_id": "a", "bridge_positive": 0, "component_topology_changed": 1},
                {"sample_id": "b", "bridge_positive": 0, "component_topology_changed": 1},
            ],
        }
        patient_payload = {
            "rows": [
                {"patient_id": "p1", "negative_topology_changes": 1},
                {"patient_id": "p2", "negative_topology_changes": 1},
            ]
        }
        out = audit.validate_topology_consistency(
            eval_payload=eval_payload,
            patient_payload=patient_payload,
            expected_negative_topology_changes=2,
            context="patient_sum_test",
        )
        self.assertTrue(out["patient_sum_parity"])

    def test_exact_target_oracle_compact_summary_exposes_positive_metrics(self):
        compact = audit.compact_exact_target_oracle_summary(
            {
                "overall": {"all_iou_ge_0.50_count": 4, "mean_matched_iou": 0.75},
                "positive": {"all_iou_ge_0.50_count": 3, "mean_matched_iou": 0.8},
                "negative_regressions": 0,
                "negative_topology_changes": 0,
                "zero_target_identical_to_closed": True,
            }
        )
        self.assertEqual(compact["positive_success50"], 3)
        self.assertAlmostEqual(compact["positive_mean_matched_iou"], 0.8)

    def test_score_partition_summary_exposes_required_quantiles(self):
        row = self._cached_row(bridge_positive=1)
        probs = np.array([[[[0.1, 0.9], [0.4, 0.8]]]], dtype=np.float32)
        out = audit.summarize_score_partition([row], probs)
        self.assertIn("p75", out["TRUE_BRIDGE"])
        self.assertIn("p90", out["TRUE_BRIDGE"])
        self.assertIn("p95", out["TRUE_BRIDGE"])
        self.assertIn("p99", out["TRUE_BRIDGE"])
        self.assertIn("fraction_ge_0p50", out["TRUE_BRIDGE"])


if __name__ == "__main__":
    unittest.main()
