from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import numpy as np
import torch

import bridge_presence_gate_v4 as gate_v4


class TestBridgePresenceGateV4(unittest.TestCase):
    def test_gate_target_map_from_manifest_uses_locked_bridge_pixels(self):
        payload = {
            "rows": [
                {"sample_id": "a", "bridge_pixels": 10},
                {"sample_id": "b", "bridge_pixels": 0},
            ]
        }
        out = gate_v4.gate_target_map_from_manifest(payload)
        self.assertEqual(out, {"a": 1, "b": 0})

    def test_extract_gate_features_has_no_gt_scalar_leakage(self):
        records = [
            {
                "sample_id": "a",
                "gate_target": 1,
                "bridge_positive": 1,
                "candidate_pixels": 4,
                "component_count_start": 2,
                "x_0_4": torch.ones((16, 2, 2), dtype=torch.float32),
                "x_2_2": torch.ones((32, 1, 1), dtype=torch.float32),
                "candidate_mask_np": np.ones((2, 2), dtype=np.uint8),
            }
        ]
        logits = torch.ones((1, 1, 2, 2), dtype=torch.float32)
        rows, features_t, targets_t, pixel_remove_masks = gate_v4.extract_gate_feature_rows(records, logits)
        self.assertEqual(int(targets_t[0].item()), 1)
        self.assertEqual(features_t.shape[1], 96 + len(gate_v4.SCALAR_FEATURE_NAMES))
        self.assertEqual(int(np.sum(pixel_remove_masks[0])), 4)
        self.assertEqual(set(gate_v4.SCALAR_FEATURE_NAMES), {
            "bridge_score_mean",
            "bridge_score_max",
            "bridge_score_top1pct_mean",
            "bridge_score_top5pct_mean",
            "bridge_score_frac_ge_0p50",
            "bridge_score_frac_ge_0p75",
            "bridge_score_frac_ge_0p90",
            "candidate_fraction",
            "candidate_component_count",
        })
        self.assertFalse(any(key.startswith("gt_") for key in rows[0].keys() if key not in {"bridge_positive_target"}))

    def test_hard_gate_mask_semantics(self):
        pixel_remove = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        closed = gate_v4.apply_hard_sample_gate(pixel_remove, False)
        opened = gate_v4.apply_hard_sample_gate(pixel_remove, True)
        self.assertTrue(np.array_equal(closed, np.zeros_like(pixel_remove)))
        self.assertTrue(np.array_equal(opened, pixel_remove))

    def test_threshold_sweep_and_trivial_closed_gate_not_safe_useful(self):
        records = [
            {
                "sample_id": "p",
                "gate_target": 1,
                "bridge_positive": 1,
                "candidate_pixels": 4,
                "component_count_start": 1,
                "candidate_mask_np": np.ones((2, 2), dtype=np.uint8),
                "gt_instances": np.ones((2, 2), dtype=np.uint8),
                "start_reconstruction": {"metrics": {"instance_mean_matched_iou": 0.3, "all_iou_ge_0.50": False}},
            },
            {
                "sample_id": "n",
                "gate_target": 0,
                "bridge_positive": 0,
                "candidate_pixels": 4,
                "component_count_start": 1,
                "candidate_mask_np": np.ones((2, 2), dtype=np.uint8),
                "gt_instances": np.ones((2, 2), dtype=np.uint8),
                "start_reconstruction": {"metrics": {"instance_mean_matched_iou": 0.8, "all_iou_ge_0.50": True}},
            },
        ]
        pixel_remove_masks = [np.ones((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8)]

        def fake_reconstruction(pred_leaf01, gt_inst_u8):
            fg = int(np.sum(pred_leaf01))
            mean = 0.7 if fg == 0 else (0.3 if fg == 4 else 0.8)
            return {
                "result": {"metrics": {"instance_mean_matched_iou": float(mean), "all_iou_ge_0.50": bool(mean >= 0.50)}},
                "timing": {
                    "total_seconds": 0.25,
                    "normalization_seconds": 0.10,
                    "metrics_seconds": 0.10,
                    "topology_seconds": 0.05,
                },
            }

        with mock.patch("bridge_presence_gate_v4.bridge.run_locked_reconstruction_with_timing", side_effect=fake_reconstruction), \
             mock.patch("bridge_presence_gate_v4.bridge._connected_components", return_value=(np.ones((2, 2), dtype=np.int32), 1)):
            closed_eval = gate_v4.evaluate_gate_threshold_on_cached(records, pixel_remove_masks, np.array([0.1, 0.1]), gate_threshold=0.5)
            self.assertFalse(closed_eval["safe_useful"])
            self.assertIn("timing", closed_eval)
            self.assertGreaterEqual(float(closed_eval["timing"]["cpu_reconstruction_seconds"]), 0.5)
            sweep = gate_v4.gate_threshold_sweep(records, pixel_remove_masks, np.array([0.1, 0.9]), [0.05, 0.95])
        self.assertEqual(len(sweep), 2)
        self.assertEqual([row["gate_threshold"] for row in sweep], [0.05, 0.95])

    def test_safe_useful_key_prefers_positive_success_then_iou(self):
        a = {"gated_reconstruction": {"positive_success50": 3, "positive_mean_matched_iou": 0.6, "overall_mean_matched_iou": 0.5}, "classification": {"balanced_accuracy": 0.8}}
        b = {"gated_reconstruction": {"positive_success50": 2, "positive_mean_matched_iou": 0.9, "overall_mean_matched_iou": 0.9}, "classification": {"balanced_accuracy": 1.0}}
        self.assertGreater(gate_v4.safe_useful_key(a), gate_v4.safe_useful_key(b))


if __name__ == "__main__":
    unittest.main()
