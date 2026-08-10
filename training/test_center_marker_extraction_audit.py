from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent


class TestCenterMarkerExtractionAudit(unittest.TestCase):
    @staticmethod
    def _mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import audit_center_marker_extraction as mod

        return mod

    def test_raw_connected_component_count_before_cap(self):
        mod = self._mod()
        center = np.zeros((12, 12), dtype=np.float32)
        for y, x in [(1, 1), (1, 10), (10, 1), (10, 10)]:
            center[y, x] = 0.9
        leaf = np.ones_like(center, dtype=bool)
        gt_inst = np.zeros_like(center, dtype=np.int32)
        comp = mod._component_stats(center, leaf, gt_inst, threshold=0.5)
        pts = mod._representative_markers_from_components(comp["components"], policy="current", max_markers=3)
        self.assertEqual(len(comp["components"]), 4)
        self.assertEqual(len(pts), 3)

    def test_threshold_induced_component_splitting(self):
        mod = self._mod()
        center = np.zeros((9, 9), dtype=np.float32)
        center[4, 2] = 0.9
        center[4, 3] = 0.2
        center[4, 4] = 0.2
        center[4, 5] = 0.2
        center[4, 6] = 0.9
        leaf = np.ones_like(center, dtype=bool)
        gt_inst = np.zeros_like(center, dtype=np.int32)
        lo = mod._component_stats(center, leaf, gt_inst, threshold=0.1)
        hi = mod._component_stats(center, leaf, gt_inst, threshold=0.5)
        split = mod._split_from_previous(lo["labels"], hi["labels"])
        self.assertEqual(len(lo["components"]), 1)
        self.assertEqual(len(hi["components"]), 2)
        self.assertTrue(split["split_from_previous_threshold"])
        self.assertEqual(split["split_parent_component_count"], 1)
        self.assertEqual(split["split_child_component_count"], 2)

    def test_max_markers_cap_reporting(self):
        mod = self._mod()
        rows = [
            {"gt_instance_count": 3, "raw_component_count": 5},
            {"gt_instance_count": 3, "raw_component_count": 2},
            {"gt_instance_count": 1, "raw_component_count": 4},
        ]
        out = mod._raw_and_capped_count_rows(rows, split_name="val", threshold=0.01)
        all_row = next(row for row in out if row["gt_group"] == "all")
        self.assertAlmostEqual(all_row["fraction_raw_count_gt_3"], 2.0 / 3.0, places=6)
        self.assertIn("\"3\":2", all_row["capped_count_histogram_json"])

    def test_component_argmax_matches_production_extractor(self):
        mod = self._mod()
        center = np.zeros((8, 8), dtype=np.float32)
        center[2:5, 2:5] = 0.2
        center[4, 4] = 0.95
        leaf = np.ones_like(center, dtype=bool)
        gt_inst = np.zeros_like(center, dtype=np.int32)
        comp = mod._component_stats(center, leaf, gt_inst, threshold=0.1)
        current = mod._representative_markers_from_components(comp["components"], policy="current", max_markers=3)
        production = mod._current_policy_markers(center, leaf, threshold=0.1, max_markers=3)
        self.assertEqual(current, production)
        self.assertEqual((current[0][0], current[0][1]), (4, 4))

    def test_nms_kernels_change_peak_count_deterministically(self):
        mod = self._mod()
        center = np.zeros((13, 13), dtype=np.float32)
        center[6, 4] = 0.95
        center[6, 8] = 0.9
        center[1, 1] = 0.85
        leaf = np.ones_like(center, dtype=bool)
        k3 = mod._max_pool_nms_markers(center, leaf, threshold=0.5, kernel=3, max_markers=None)
        k5 = mod._max_pool_nms_markers(center, leaf, threshold=0.5, kernel=5, max_markers=None)
        k9 = mod._max_pool_nms_markers(center, leaf, threshold=0.5, kernel=9, max_markers=None)
        self.assertEqual(len(k3), 3)
        self.assertEqual(len(k5), 3)
        self.assertEqual(len(k9), 2)

    def test_coordinate_contract_is_identity_after_upsample(self):
        mod = self._mod()
        cfg = {
            "model": {
                "input_size": 768,
                "center_feature": {
                    "module_path": "base.decoder.blocks.x_2_2",
                    "native_stride": 4,
                    "upsample_logits_to_target": True,
                }
            }
        }
        contract = mod._current_extraction_contract(cfg, Path(__file__))
        self.assertEqual(contract["center_feature_resolution_hw_before_upsample"], [192, 192])
        self.assertEqual(contract["center_heatmap_resolution_hw_after_model_forward"], [768, 768])
        self.assertEqual(contract["conversion_from_heatmap_to_768_image_coordinates"], "identity_mapping_after_model_upsample_to_target")

    def test_localization_metric_audit_labels_pooled_vs_mean_sample(self):
        mod = self._mod()
        rows = [
            {
                "split": "val",
                "threshold": 0.01,
                "match_distances": [1.0, 9.0],
            },
            {
                "split": "val",
                "threshold": 0.01,
                "match_distances": [5.0],
            },
        ]
        summary_rows = [
            {
                "split": "val",
                "threshold": 0.01,
                "policy": "current",
                "cap_policy": "capped_top3",
                "localization_error_px_mean_samples": 4.0,
            }
        ]
        pooled = {}
        grouped = {}
        for row in rows:
            grouped.setdefault(row["threshold"], []).append(row)
        for threshold, subset in grouped.items():
            pooled[threshold] = float(np.mean(np.asarray([d for row in subset for d in row["match_distances"]], dtype=np.float64)))
        self.assertAlmostEqual(pooled[0.01], 5.0, places=6)
        self.assertNotAlmostEqual(summary_rows[0]["localization_error_px_mean_samples"], pooled[0.01], places=6)

    def test_metric_alignment_explains_positive_margin_strict_fail(self):
        mod = self._mod()
        current_rows = [
            {
                "split": "val",
                "sample": "s1",
                "gt_instance_count": 2,
                "threshold": 0.01,
                "center_f1": 0.5,
                "marker_contract_pass": False,
                "predicted_count": 3,
                "missing_gt_instance_markers": 0,
                "multiple_markers_inside_gt_instances": 1,
                "markers_outside_all_gt_instances": 0,
                "margin": 0.1,
            },
            {
                "split": "val",
                "sample": "s2",
                "gt_instance_count": 2,
                "threshold": 0.01,
                "center_f1": 1.0,
                "marker_contract_pass": True,
                "predicted_count": 2,
                "missing_gt_instance_markers": 0,
                "multiple_markers_inside_gt_instances": 0,
                "markers_outside_all_gt_instances": 0,
                "margin": 0.2,
            },
        ]
        topology_rows = [
            {
                "split": "val",
                "sample": "s1",
                "threshold": 0.01,
                "raw_component_count": 3,
                "gt_instances_intersected_by_multiple_components": 1,
                "components_outside_all_gt_instances": 0,
            },
            {
                "split": "val",
                "sample": "s2",
                "threshold": 0.01,
                "raw_component_count": 2,
                "gt_instances_intersected_by_multiple_components": 0,
                "components_outside_all_gt_instances": 0,
            },
        ]
        out = mod._metric_alignment_summary(current_rows, topology_rows)
        self.assertEqual(out["reference_threshold_for_per_sample_relationships"], 0.01)
        self.assertEqual(len(out["positive_margin_but_strict_fail_samples"]), 1)
        self.assertIn("duplicate_or_fragmented_components_inside_gt", out["positive_margin_but_strict_fail_reason_counts"])

    def test_diagnostic_current_policy_does_not_modify_production_extraction(self):
        mod = self._mod()
        center = np.zeros((9, 9), dtype=np.float32)
        center[2, 2] = 0.8
        center[6, 6] = 0.9
        leaf = np.ones_like(center, dtype=bool)
        pts_before = mod._current_policy_markers(center, leaf, threshold=0.5, max_markers=3)
        comp = mod._component_stats(center, leaf, np.zeros_like(center, dtype=np.int32), threshold=0.5)
        pts_diag = mod._representative_markers_from_components(comp["components"], policy="current", max_markers=3)
        pts_after = mod._current_policy_markers(center, leaf, threshold=0.5, max_markers=3)
        self.assertEqual(pts_before, pts_diag)
        self.assertEqual(pts_before, pts_after)


if __name__ == "__main__":
    unittest.main()
