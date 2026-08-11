from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import leaflet_oracle_count_geometric_split_audit as audit


def _spec(key: str) -> audit.SeedMethodSpec:
    for spec in audit.SEED_METHOD_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(key)


class TestLeafletOracleCountGeometricSplitAudit(unittest.TestCase):
    def test_extract_gt_count_from_metadata(self):
        meta = {
            "instance_count": 3,
            "source_instance_ids": [1, 2, 4],
            "instances": [{"instance_id": 1}, {"instance_id": 2}, {"instance_id": 4}],
        }
        self.assertEqual(audit.extract_gt_count_from_metadata(meta), 3)

    def test_select_exact_k_seeds_returns_k_for_three_blobs(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv = __import__("cv2")
        cv.circle(mask, (14, 14), 8, 1, thickness=-1)
        cv.circle(mask, (48, 14), 8, 1, thickness=-1)
        cv.circle(mask, (32, 48), 8, 1, thickness=-1)
        trace = audit.select_exact_k_seeds(mask, 3, _spec("component_aware_maxima"))
        self.assertEqual(len(trace["seeds"]), 3)
        self.assertEqual(int(trace["component_count"]), 3)

    def test_k1_behavior_single_component(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv = __import__("cv2")
        cv.ellipse(mask, (32, 32), (10, 20), 0, 0, 360, 1, thickness=-1)
        result = audit.split_mask_with_oracle_k(mask, 1, _spec("global_distance_maxima_r09"))
        self.assertEqual(int(result["pred_count"]), 1)
        self.assertEqual(int(result["labels"].max()), 1)

    def test_k2_behavior_connected_dumbbell(self):
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv = __import__("cv2")
        cv.circle(mask, (28, 48), 16, 1, thickness=-1)
        cv.circle(mask, (68, 48), 16, 1, thickness=-1)
        cv.rectangle(mask, (28, 42), (68, 54), 1, thickness=-1)
        result = audit.split_mask_with_oracle_k(mask, 2, _spec("global_distance_maxima_r09"))
        self.assertEqual(int(result["pred_count"]), 2)

    def test_k3_behavior_three_lobes(self):
        mask = np.zeros((120, 120), dtype=np.uint8)
        cv = __import__("cv2")
        cv.circle(mask, (35, 40), 18, 1, thickness=-1)
        cv.circle(mask, (82, 40), 18, 1, thickness=-1)
        cv.circle(mask, (58, 82), 18, 1, thickness=-1)
        cv.rectangle(mask, (35, 36), (82, 48), 1, thickness=-1)
        cv.rectangle(mask, (48, 48), (68, 82), 1, thickness=-1)
        result = audit.split_mask_with_oracle_k(mask, 3, _spec("component_aware_maxima"))
        self.assertEqual(int(result["pred_count"]), 3)

    def test_disconnected_components_exceed_k(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv = __import__("cv2")
        cv.circle(mask, (12, 12), 6, 1, thickness=-1)
        cv.circle(mask, (48, 12), 6, 1, thickness=-1)
        cv.circle(mask, (32, 48), 6, 1, thickness=-1)
        result = audit.split_mask_with_oracle_k(mask, 2, _spec("component_aware_maxima"))
        self.assertEqual(int(result["pred_count"]), 3)
        self.assertEqual(result["seed_trace"]["impossible_reason"], "connected_components_exceed_k")

    def test_watershed_is_constrained_to_foreground(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv = __import__("cv2")
        cv.circle(mask, (24, 40), 14, 1, thickness=-1)
        cv.circle(mask, (56, 40), 14, 1, thickness=-1)
        cv.rectangle(mask, (24, 35), (56, 45), 1, thickness=-1)
        result = audit.split_mask_with_oracle_k(mask, 2, _spec("prominent_maxima_rel20"))
        self.assertTrue(np.all(result["labels"][mask == 0] == 0))

    def test_deterministic_output(self):
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv = __import__("cv2")
        cv.circle(mask, (28, 48), 16, 1, thickness=-1)
        cv.circle(mask, (68, 48), 16, 1, thickness=-1)
        cv.rectangle(mask, (28, 42), (68, 54), 1, thickness=-1)
        a = audit.split_mask_with_oracle_k(mask, 2, _spec("global_distance_maxima_r15"))
        b = audit.split_mask_with_oracle_k(mask, 2, _spec("global_distance_maxima_r15"))
        self.assertTrue(np.array_equal(a["labels"], b["labels"]))
        self.assertEqual(a["seeds"], b["seeds"])

    def test_detailed_instance_matching(self):
        gt = np.zeros((40, 40), dtype=np.uint8)
        pred = np.zeros((40, 40), dtype=np.uint8)
        gt[5:15, 5:15] = 1
        gt[20:30, 20:30] = 2
        pred[5:15, 5:15] = 1
        pred[20:30, 20:30] = 2
        metrics = audit.compute_detailed_instance_metrics(gt, pred, gt_k=2, pred_k=2)
        self.assertEqual(metrics["matched_iou_per_gt"], [1.0, 1.0])
        self.assertEqual(int(metrics["unmatched_gt_instances"]), 0)
        self.assertEqual(int(metrics["unmatched_pred_instances"]), 0)
        self.assertEqual(float(metrics["all_iou_ge_0.80"]), 1.0)

    def test_no_holdout_referenced_by_default_manifest(self):
        rows = audit._read_jsonl(audit.DEFAULT_MANIFEST_PATH)
        self.assertTrue(rows)
        self.assertTrue(all(not bool(row.get("present_in_authoritative_106_holdout", False)) for row in rows))

    def test_output_dir_is_analysis_only(self):
        out = audit.DEFAULT_OUTPUT_DIR.resolve()
        parts = {part.lower() for part in out.parts}
        self.assertIn("analysis", parts)
        self.assertNotIn("production", parts)
        self.assertNotIn("p1", parts)

    def test_semantic_contract_summary_matches_expected_ids(self):
        rows = audit._read_jsonl(audit.DEFAULT_MANIFEST_PATH)
        cfg = audit._read_yaml(audit.DEFAULT_SEMANTIC_CONFIG)
        summary = audit._semantic_contract_summary(rows[:1], cfg, audit.DEFAULT_INSTANCE_ROOT, audit.DEFAULT_SEMANTIC_ROOT)
        self.assertEqual(summary["leaflet_class"], 1)
        self.assertEqual(summary["ring_class"], 2)
        self.assertEqual(summary["semantic_class_ids"], [0, 1, 2])
        self.assertIn("768x768", json.dumps(summary))


if __name__ == "__main__":
    unittest.main()
