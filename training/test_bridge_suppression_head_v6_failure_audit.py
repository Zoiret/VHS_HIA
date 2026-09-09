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


if __name__ == "__main__":
    unittest.main()
