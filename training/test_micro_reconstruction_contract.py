from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent


class TestMicroReconstructionContract(unittest.TestCase):
    def _import_helpers(self):
        import sys

        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        from validate_centerhead import compute_instance_metrics_from_masks, reconstruct_instances_from_semantic_and_center

        return reconstruct_instances_from_semantic_and_center, compute_instance_metrics_from_masks

    def test_two_instance_contract_holds(self):
        reconstruct, compute_metrics = self._import_helpers()

        pred_sem = np.zeros((32, 32), dtype=np.uint8)
        pred_sem[4:12, 4:12] = 1
        pred_sem[20:28, 20:28] = 1
        center_prob = np.zeros((32, 32), dtype=np.float32)
        center_prob[8, 8] = 0.95
        center_prob[24, 24] = 0.93
        gt_inst = np.zeros((32, 32), dtype=np.uint8)
        gt_inst[4:12, 4:12] = 1
        gt_inst[20:28, 20:28] = 2

        pred_inst, pred_k, pred_pts_scored, trace = reconstruct(
            pred_sem,
            center_prob,
            0.5,
            max_markers=3,
            return_trace=True,
        )
        metrics = compute_metrics(gt_inst, pred_inst, gt_k=2, pred_k=pred_k)

        self.assertEqual(len(pred_pts_scored), 2)
        self.assertEqual(pred_k, 2)
        self.assertTrue(bool(metrics["instance_exact_count"]))
        self.assertFalse(bool(metrics["instance_fragmented"]))
        self.assertFalse(bool(metrics["instance_merged"]))
        self.assertEqual(int(trace["raw_reconstruction_count"]), 2)
        self.assertEqual(int(trace["final_count"]), 2)

    def test_disconnected_semantic_component_current_behavior(self):
        reconstruct, compute_metrics = self._import_helpers()

        pred_sem = np.zeros((40, 40), dtype=np.uint8)
        pred_sem[8:16, 6:14] = 1
        pred_sem[8:16, 24:32] = 1
        center_prob = np.zeros((40, 40), dtype=np.float32)
        center_prob[12, 10] = 0.98
        gt_inst = np.zeros((40, 40), dtype=np.uint8)
        gt_inst[8:16, 6:14] = 1
        gt_inst[8:16, 24:32] = 1

        pred_inst, pred_k, pred_pts_scored, trace = reconstruct(
            pred_sem,
            center_prob,
            0.5,
            max_markers=3,
            return_trace=True,
        )
        metrics = compute_metrics(gt_inst, pred_inst, gt_k=1, pred_k=pred_k)

        self.assertEqual(len(pred_pts_scored), 1)
        self.assertEqual(int(trace["semantic_component_count"]), 2)
        self.assertTrue(any(bool(comp["used_fallback"]) for comp in trace["component_traces"]))
        self.assertEqual(pred_k, 2)
        self.assertFalse(bool(metrics["instance_exact_count"]))
        self.assertTrue(bool(metrics["instance_fragmented"]))
        self.assertFalse(bool(metrics["instance_merged"]))


if __name__ == "__main__":
    unittest.main()
