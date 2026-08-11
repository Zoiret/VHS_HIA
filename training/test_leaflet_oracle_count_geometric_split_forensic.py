from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import leaflet_oracle_count_geometric_split_audit as base_audit
import leaflet_oracle_count_geometric_split_forensic as forensic


class TestLeafletOracleCountGeometricSplitForensic(unittest.TestCase):
    def test_written_contract_classification_returns_weak_when_gt_good_and_pred_limited(self):
        gt_summary = {
            "all_iou_ge_0.50": 0.9785,
            "mean_matched_iou": 0.9301,
            "exact_instance_count": 0.5914,
        }
        pred_summary = {
            "exact_instance_count": 0.3548,
            "all_iou_ge_0.50": 0.4194,
        }
        pred_gt2 = {"all_iou_ge_0.50": 0.5161}
        self.assertEqual(
            forensic.written_contract_classification(gt_summary=gt_summary, pred_summary=pred_summary, pred_gt2=pred_gt2),
            "WEAK_GEOMETRIC_SIGNAL",
        )

    def test_literal_oracle_k_count_behavior_can_exceed_k_when_components_exceed_k(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.circle(mask, (15, 15), 6, 1, thickness=-1)
        cv2.circle(mask, (40, 15), 6, 1, thickness=-1)
        cv2.circle(mask, (65, 15), 6, 1, thickness=-1)
        spec = next(spec for spec in base_audit.SEED_METHOD_SPECS if spec.key == "global_distance_maxima_r09")
        result = forensic._analyze_seeded_split(mask, 2, spec)
        self.assertEqual(result["pred_count"], 3)
        self.assertEqual(result["mismatch_reason"], "disconnected_component_policy")

    def test_false_bridge_detection(self):
        gt = np.zeros((64, 64), dtype=np.uint8)
        gt[10:30, 8:24] = 1
        gt[10:30, 40:56] = 2
        pred = np.zeros((64, 64), dtype=np.uint8)
        pred[10:30, 8:56] = 1
        topo = forensic.classify_semantic_topology(gt, pred)
        self.assertEqual(topo["topology_class"], "B")
        self.assertTrue(topo["bridge"])
        self.assertFalse(topo["missing"])

    def test_topology_attribution_both_bridge_and_missing(self):
        gt = np.zeros((80, 80), dtype=np.uint8)
        gt[10:35, 10:30] = 1
        gt[10:35, 50:70] = 2
        gt[45:70, 30:50] = 3
        pred = np.zeros((80, 80), dtype=np.uint8)
        pred[10:35, 10:70] = 1
        pred[45:55, 30:50] = 1
        pred[60:70, 30:50] = 1
        topo = forensic.classify_semantic_topology(gt, pred)
        self.assertEqual(topo["topology_class"], "D")
        self.assertTrue(topo["bridge"])
        self.assertTrue(topo["missing"])

    def test_postprocessing_is_deterministic(self):
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.circle(mask, (28, 48), 16, 1, thickness=-1)
        cv2.circle(mask, (68, 48), 16, 1, thickness=-1)
        cv2.rectangle(mask, (28, 43), (68, 53), 1, thickness=-1)
        variant = next(v for v in forensic.POSTPROCESS_VARIANTS if v.key == "neck_cut_w2")
        a = forensic.apply_variant(mask, variant)
        b = forensic.apply_variant(mask, variant)
        self.assertTrue(np.array_equal(a, b))

    def test_no_holdout_reference_in_default_manifest(self):
        rows = base_audit._read_jsonl(base_audit.DEFAULT_MANIFEST_PATH)
        self.assertTrue(rows)
        self.assertTrue(all(not bool(row.get("present_in_authoritative_106_holdout", False)) for row in rows))

    def test_no_production_modification_in_default_output_dir(self):
        out = forensic.DEFAULT_OUTPUT_DIR.resolve()
        parts = {part.lower() for part in out.parts}
        self.assertIn("analysis", parts)
        self.assertNotIn("production", parts)


if __name__ == "__main__":
    unittest.main()
