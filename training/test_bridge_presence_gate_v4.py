from __future__ import annotations

import sys
import tempfile
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
    def _toy_eval_fixture(self):
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
        gate_probs = np.array([0.9, 0.9], dtype=np.float64)
        return records, pixel_remove_masks, gate_probs

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
        records, pixel_remove_masks, _gate_probs = self._toy_eval_fixture()

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
            state_cache, cache_timing = gate_v4.build_hard_gate_state_cache(records, pixel_remove_masks)
            self.assertGreaterEqual(float(cache_timing["cpu_reconstruction_seconds"]), 0.5)
            closed_eval = gate_v4.evaluate_gate_threshold_on_cached(state_cache, np.array([0.1, 0.1]), gate_threshold=0.5)
            self.assertFalse(closed_eval["safe_useful"])
            self.assertIn("timing", closed_eval)
            self.assertEqual(float(closed_eval["timing"]["cpu_reconstruction_seconds"]), 0.0)
            sweep = gate_v4.gate_threshold_sweep(state_cache, np.array([0.1, 0.9]), [0.05, 0.95])
        self.assertEqual(len(sweep), 2)
        self.assertEqual([row["gate_threshold"] for row in sweep], [0.05, 0.95])

    def test_hard_gate_state_cache_matches_reference_evaluator_exactly(self):
        records, pixel_remove_masks, gate_probs = self._toy_eval_fixture()

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
            reference = gate_v4._evaluate_gate_threshold_reference(records, pixel_remove_masks, gate_probs, gate_threshold=0.5)
            state_cache, _cache_timing = gate_v4.build_hard_gate_state_cache(records, pixel_remove_masks)
            cached = gate_v4.evaluate_gate_threshold_on_cached(state_cache, gate_probs, gate_threshold=0.5)

        for key in ("tp", "tn", "fp", "fn", "sensitivity", "specificity", "balanced_accuracy"):
            self.assertEqual(reference["classification"][key], cached["classification"][key])
        self.assertEqual(reference["gated_reconstruction"], cached["gated_reconstruction"])
        self.assertEqual(reference["gate_open_samples"], cached["gate_open_samples"])
        self.assertEqual(reference["gate_closed_samples"], cached["gate_closed_samples"])
        self.assertEqual(reference["safe_useful"], cached["safe_useful"])
        self.assertEqual(reference["per_sample"], cached["per_sample"])

    def test_cached_evaluator_and_sweep_do_not_rerun_reconstruction(self):
        records, pixel_remove_masks, gate_probs = self._toy_eval_fixture()

        def fake_reconstruction(pred_leaf01, gt_inst_u8):
            return {
                "result": {"metrics": {"instance_mean_matched_iou": 0.5, "all_iou_ge_0.50": True}},
                "timing": {
                    "total_seconds": 0.25,
                    "normalization_seconds": 0.10,
                    "metrics_seconds": 0.10,
                    "topology_seconds": 0.05,
                },
            }

        with mock.patch("bridge_presence_gate_v4.bridge.run_locked_reconstruction_with_timing", side_effect=fake_reconstruction) as recon_mock, \
             mock.patch("bridge_presence_gate_v4.bridge._connected_components", return_value=(np.ones((2, 2), dtype=np.int32), 1)) as cc_mock:
            state_cache, _cache_timing = gate_v4.build_hard_gate_state_cache(records, pixel_remove_masks)
            self.assertEqual(recon_mock.call_count, 2)
            self.assertEqual(cc_mock.call_count, 2)
            gate_v4.evaluate_gate_threshold_on_cached(state_cache, gate_probs, gate_threshold=0.5)
            gate_v4.gate_threshold_sweep(state_cache, gate_probs, [0.05, 0.50, 0.95])
            self.assertEqual(recon_mock.call_count, 2)
            self.assertEqual(cc_mock.call_count, 2)

    def test_safe_useful_policy_unchanged_with_cached_states(self):
        state_cache = [
            {
                "sample_id": "p1",
                "gate_target": 1,
                "bridge_positive": 1,
                "candidate_pixels": 4,
                "original_v2_remove_pixels": 2,
                "closed": {
                    "candidate_pixels": 4,
                    "predicted_removed_pixels": 0,
                    "predicted_removed_fraction": 0.0,
                    "start_mean_iou": 0.3,
                    "predicted_mean_iou": 0.3,
                    "start_success50": 0,
                    "predicted_success50": 0,
                    "component_count_start": 1,
                    "component_count_predicted": 1,
                    "component_topology_changed": 0,
                    "predicted_reconstruction_runtime_seconds": 0.0,
                    "predicted_normalization_runtime_seconds": 0.0,
                    "predicted_metric_runtime_seconds": 0.0,
                    "predicted_topology_runtime_seconds": 0.0,
                },
                "open": {
                    "candidate_pixels": 4,
                    "predicted_removed_pixels": 2,
                    "predicted_removed_fraction": 0.5,
                    "start_mean_iou": 0.3,
                    "predicted_mean_iou": 0.7,
                    "start_success50": 0,
                    "predicted_success50": 1,
                    "component_count_start": 1,
                    "component_count_predicted": 1,
                    "component_topology_changed": 0,
                    "predicted_reconstruction_runtime_seconds": 0.1,
                    "predicted_normalization_runtime_seconds": 0.05,
                    "predicted_metric_runtime_seconds": 0.03,
                    "predicted_topology_runtime_seconds": 0.02,
                },
            },
            {
                "sample_id": "p2",
                "gate_target": 1,
                "bridge_positive": 1,
                "candidate_pixels": 4,
                "original_v2_remove_pixels": 2,
                "closed": {
                    "candidate_pixels": 4,
                    "predicted_removed_pixels": 0,
                    "predicted_removed_fraction": 0.0,
                    "start_mean_iou": 0.3,
                    "predicted_mean_iou": 0.3,
                    "start_success50": 0,
                    "predicted_success50": 0,
                    "component_count_start": 1,
                    "component_count_predicted": 1,
                    "component_topology_changed": 0,
                    "predicted_reconstruction_runtime_seconds": 0.0,
                    "predicted_normalization_runtime_seconds": 0.0,
                    "predicted_metric_runtime_seconds": 0.0,
                    "predicted_topology_runtime_seconds": 0.0,
                },
                "open": {
                    "candidate_pixels": 4,
                    "predicted_removed_pixels": 2,
                    "predicted_removed_fraction": 0.5,
                    "start_mean_iou": 0.3,
                    "predicted_mean_iou": 0.6,
                    "start_success50": 0,
                    "predicted_success50": 1,
                    "component_count_start": 1,
                    "component_count_predicted": 1,
                    "component_topology_changed": 0,
                    "predicted_reconstruction_runtime_seconds": 0.1,
                    "predicted_normalization_runtime_seconds": 0.05,
                    "predicted_metric_runtime_seconds": 0.03,
                    "predicted_topology_runtime_seconds": 0.02,
                },
            },
            {
                "sample_id": "p3",
                "gate_target": 1,
                "bridge_positive": 1,
                "candidate_pixels": 4,
                "original_v2_remove_pixels": 2,
                "closed": {
                    "candidate_pixels": 4,
                    "predicted_removed_pixels": 0,
                    "predicted_removed_fraction": 0.0,
                    "start_mean_iou": 0.3,
                    "predicted_mean_iou": 0.3,
                    "start_success50": 0,
                    "predicted_success50": 0,
                    "component_count_start": 1,
                    "component_count_predicted": 1,
                    "component_topology_changed": 0,
                    "predicted_reconstruction_runtime_seconds": 0.0,
                    "predicted_normalization_runtime_seconds": 0.0,
                    "predicted_metric_runtime_seconds": 0.0,
                    "predicted_topology_runtime_seconds": 0.0,
                },
                "open": {
                    "candidate_pixels": 4,
                    "predicted_removed_pixels": 2,
                    "predicted_removed_fraction": 0.5,
                    "start_mean_iou": 0.3,
                    "predicted_mean_iou": 0.65,
                    "start_success50": 0,
                    "predicted_success50": 1,
                    "component_count_start": 1,
                    "component_count_predicted": 1,
                    "component_topology_changed": 0,
                    "predicted_reconstruction_runtime_seconds": 0.1,
                    "predicted_normalization_runtime_seconds": 0.05,
                    "predicted_metric_runtime_seconds": 0.03,
                    "predicted_topology_runtime_seconds": 0.02,
                },
            },
            {
                "sample_id": "n1",
                "gate_target": 0,
                "bridge_positive": 0,
                "candidate_pixels": 4,
                "original_v2_remove_pixels": 2,
                "closed": {
                    "candidate_pixels": 4,
                    "predicted_removed_pixels": 0,
                    "predicted_removed_fraction": 0.0,
                    "start_mean_iou": 0.8,
                    "predicted_mean_iou": 0.8,
                    "start_success50": 1,
                    "predicted_success50": 1,
                    "component_count_start": 1,
                    "component_count_predicted": 1,
                    "component_topology_changed": 0,
                    "predicted_reconstruction_runtime_seconds": 0.0,
                    "predicted_normalization_runtime_seconds": 0.0,
                    "predicted_metric_runtime_seconds": 0.0,
                    "predicted_topology_runtime_seconds": 0.0,
                },
                "open": {
                    "candidate_pixels": 4,
                    "predicted_removed_pixels": 2,
                    "predicted_removed_fraction": 0.5,
                    "start_mean_iou": 0.8,
                    "predicted_mean_iou": 0.6,
                    "start_success50": 1,
                    "predicted_success50": 1,
                    "component_count_start": 1,
                    "component_count_predicted": 2,
                    "component_topology_changed": 1,
                    "predicted_reconstruction_runtime_seconds": 0.1,
                    "predicted_normalization_runtime_seconds": 0.05,
                    "predicted_metric_runtime_seconds": 0.03,
                    "predicted_topology_runtime_seconds": 0.02,
                },
            },
        ]
        safe = gate_v4.evaluate_gate_threshold_on_cached(state_cache, np.array([0.9, 0.9, 0.9, 0.1]), gate_threshold=0.5)
        unsafe = gate_v4.evaluate_gate_threshold_on_cached(state_cache, np.array([0.9, 0.9, 0.9, 0.9]), gate_threshold=0.5)
        self.assertTrue(safe["safe_useful"])
        self.assertFalse(unsafe["safe_useful"])

    def test_compare_nested_payloads_accepts_equal_nested_numpy_payload(self):
        ref = {
            "metrics": {
                "iou_matrix": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
                "matched_ious": [np.float64(1.0), np.float64(0.5)],
                "flags": (True, False),
            }
        }
        opt = {
            "metrics": {
                "iou_matrix": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
                "matched_ious": [np.float64(1.0), np.float64(0.5)],
                "flags": (True, False),
            }
        }
        self.assertIsNone(gate_v4.compare_nested_payloads(ref, opt, path="metrics"))

    def test_compare_nested_payloads_reports_array_value_mismatch_path(self):
        mismatch = gate_v4.compare_nested_payloads(
            {"iou_matrix": np.array([[1, 2]], dtype=np.int32)},
            {"iou_matrix": np.array([[1, 3]], dtype=np.int32)},
            path="metrics",
        )
        self.assertIsNotNone(mismatch)
        self.assertEqual(mismatch["path"], "metrics.iou_matrix")
        self.assertEqual(mismatch["reason"], "value_mismatch")

    def test_compare_nested_payloads_reports_array_shape_mismatch(self):
        mismatch = gate_v4.compare_nested_payloads(
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 3), dtype=np.uint8),
            path="result.instance_mask",
        )
        self.assertIsNotNone(mismatch)
        self.assertEqual(mismatch["path"], "result.instance_mask")
        self.assertEqual(mismatch["reason"], "shape_mismatch")

    def test_compare_nested_payloads_handles_numpy_scalars(self):
        self.assertIsNone(
            gate_v4.compare_nested_payloads(np.int64(3), np.int64(3), path="metrics.pred_k")
        )
        mismatch = gate_v4.compare_nested_payloads(np.float64(1.0), np.float32(1.0), path="metrics.score")
        self.assertIsNotNone(mismatch)
        self.assertEqual(mismatch["reason"], "dtype_mismatch")

    def test_compare_nested_payloads_reports_nested_list_path(self):
        mismatch = gate_v4.compare_nested_payloads(
            {"matched_ious": [0.1, 0.2, 0.3]},
            {"matched_ious": [0.1, 0.25, 0.3]},
            path="metrics",
        )
        self.assertIsNotNone(mismatch)
        self.assertEqual(mismatch["path"], "metrics.matched_ious[1]")

    def test_format_payload_mismatch_includes_diagnostic_path(self):
        message = gate_v4.format_payload_mismatch(
            sample_id="s1",
            state_name="OPEN",
            mismatch={"path": "metrics.iou_matrix", "reason": "value_mismatch"},
            category="metric_parity",
        )
        self.assertIn("\"path\": \"metrics.iou_matrix\"", message)
        self.assertIn("\"sample_id\": \"s1\"", message)

    def test_profile_hard_gate_reconstruction_states_reports_exact_instance_mask_path(self):
        cached_records = [
            {
                "sample_id": "s1",
                "candidate_mask_np": np.ones((2, 2), dtype=np.uint8),
                "gt_instances": np.ones((2, 2), dtype=np.uint8),
                "candidate_pixels": 4,
            }
        ]
        pixel_remove_masks = [np.ones((2, 2), dtype=np.uint8)]
        reference_payload = {
            "result": {
                "labels": np.array([[1, 0], [0, 1]], dtype=np.uint8),
                "metrics": {"iou_matrix": np.array([[1.0]], dtype=np.float64)},
                "topology": {"topology_class": "ok"},
            },
            "profile": {
                "foreground_pixels_entering_normalizer": 2,
                "input_component_count": 1,
                "expected_k": 1,
                "output_component_count": 1,
                "total_reconstruction_seconds": 1.0,
                "input_mask_preparation_seconds": 0.0,
                "connected_component_labeling_seconds": 0.0,
                "component_filtering_statistics_seconds": 0.0,
                "seed_centroid_preparation_seconds": 0.0,
                "distance_map_computation_seconds": 0.0,
                "centroid_distance_computation_seconds": 0.0,
                "pixel_to_instance_assignment_seconds": 0.0,
                "per_component_python_loops_seconds": 0.0,
                "morphology_seconds": 0.0,
                "output_instance_mask_creation_seconds": 0.0,
                "gt_matching_seconds": 0.0,
                "iou_matrix_construction_seconds": 0.0,
                "success50_aggregate_seconds": 0.0,
                "array_copy_dtype_conversion_seconds": 0.0,
                "call_counts": {},
            },
        }
        optimized_payload = {
            "result": {
                "labels": np.array([[1, 1], [0, 1]], dtype=np.uint8),
                "metrics": {"iou_matrix": np.array([[1.0]], dtype=np.float64)},
                "topology": {"topology_class": "ok"},
            },
            "profile": dict(reference_payload["profile"]),
        }
        with tempfile.TemporaryDirectory() as td:
            with mock.patch("bridge_presence_gate_v4.bridge.run_locked_reconstruction_profiled", side_effect=[reference_payload, optimized_payload]):
                with self.assertRaises(SystemExit) as ctx:
                    gate_v4.profile_hard_gate_reconstruction_states(
                        cached_records,
                        pixel_remove_masks,
                        output_dir=Path(td),
                    )
        self.assertIn("result.instance_mask", str(ctx.exception))

    def test_safe_useful_key_prefers_positive_success_then_iou(self):
        a = {"gated_reconstruction": {"positive_success50": 3, "positive_mean_matched_iou": 0.6, "overall_mean_matched_iou": 0.5}, "classification": {"balanced_accuracy": 0.8}}
        b = {"gated_reconstruction": {"positive_success50": 2, "positive_mean_matched_iou": 0.9, "overall_mean_matched_iou": 0.9}, "classification": {"balanced_accuracy": 1.0}}
        self.assertGreater(gate_v4.safe_useful_key(a), gate_v4.safe_useful_key(b))


if __name__ == "__main__":
    unittest.main()
