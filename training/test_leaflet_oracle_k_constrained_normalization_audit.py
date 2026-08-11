from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import leaflet_oracle_k_constrained_normalization_audit as audit
import leaflet_oracle_count_geometric_split_audit as base_audit


class TestLeafletOracleKConstrainedNormalizationAudit(unittest.TestCase):
    def test_n_eq_k_keeps_natural_components(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(mask, (16, 16), 6, 1, thickness=-1)
        cv2.circle(mask, (48, 48), 6, 1, thickness=-1)
        out = audit.normalize_mask_exact_k(mask, 2, "nearest_component_k_normalizer")
        self.assertTrue(out["exact_k_achieved"])
        self.assertEqual(out["final_group_count"], 2)
        self.assertEqual(out["merge_count"], 0)
        self.assertEqual(out["split_count"], 0)

    def test_n_gt_k_merges_to_exact_k(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.circle(mask, (14, 40), 6, 1, thickness=-1)
        cv2.circle(mask, (32, 40), 6, 1, thickness=-1)
        cv2.circle(mask, (66, 40), 6, 1, thickness=-1)
        out = audit.normalize_mask_exact_k(mask, 2, "nearest_component_k_normalizer")
        self.assertTrue(out["exact_k_achieved"])
        self.assertEqual(out["final_group_count"], 2)
        self.assertEqual(out["merge_count"], 1)

    def test_n_lt_k_splits_to_exact_k(self):
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.circle(mask, (28, 48), 16, 1, thickness=-1)
        cv2.circle(mask, (68, 48), 16, 1, thickness=-1)
        cv2.rectangle(mask, (28, 42), (68, 54), 1, thickness=-1)
        out = audit.normalize_mask_exact_k(mask, 2, "nearest_component_k_normalizer")
        self.assertTrue(out["exact_k_achieved"])
        self.assertEqual(out["final_group_count"], 2)
        self.assertGreaterEqual(out["split_count"], 1)

    def test_disconnected_pieces_can_share_same_label(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.circle(mask, (12, 40), 5, 1, thickness=-1)
        cv2.circle(mask, (28, 40), 5, 1, thickness=-1)
        cv2.circle(mask, (68, 40), 5, 1, thickness=-1)
        out = audit.normalize_mask_exact_k(mask, 2, "nearest_component_k_normalizer")
        labels = out["labels"]
        left_labels = sorted(int(v) for v in np.unique(labels[:, :40]) if int(v) > 0)
        self.assertEqual(len(left_labels), 1)
        self.assertEqual(out["final_group_count"], 2)

    def test_exact_k_output_contract(self):
        mask = np.zeros((120, 120), dtype=np.uint8)
        cv2.circle(mask, (30, 30), 10, 1, thickness=-1)
        cv2.circle(mask, (90, 30), 10, 1, thickness=-1)
        cv2.circle(mask, (60, 90), 10, 1, thickness=-1)
        out = audit.normalize_mask_exact_k(mask, 2, "area_aware_k_normalizer")
        self.assertEqual(len(audit._positive_ids(out["labels"])), 2)

    def test_deterministic_merge_order(self):
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.circle(mask, (16, 48), 7, 1, thickness=-1)
        cv2.circle(mask, (34, 48), 7, 1, thickness=-1)
        cv2.circle(mask, (78, 48), 7, 1, thickness=-1)
        a = audit.normalize_mask_exact_k(mask, 2, "nearest_component_k_normalizer")
        b = audit.normalize_mask_exact_k(mask, 2, "nearest_component_k_normalizer")
        self.assertEqual(a["merge_operations"], b["merge_operations"])
        self.assertTrue(np.array_equal(a["labels"], b["labels"]))

    def test_deployable_algorithm_does_not_take_gt_labels(self):
        sig = inspect.signature(audit.normalize_mask_exact_k)
        self.assertNotIn("gt_inst_u8", sig.parameters)

    def test_oracle_grouping_is_isolated(self):
        sig = inspect.signature(audit.gt_fragment_grouping_oracle)
        self.assertIn("gt_inst_u8", sig.parameters)
        self.assertNotIn("gt_inst_u8", inspect.signature(audit.normalize_mask_exact_k).parameters)

    def test_decision_prefers_semantic_topology_when_oracle_ceiling_is_low(self):
        decision, reason = audit._choose_decision(
            best_pred={
                "exact_k_rate": 1.0,
                "all_iou_ge_0.50": 0.4838709677,
                "mean_matched_iou": 0.5745997306,
            },
            best_pred_gt2={"all_iou_ge_0.50": 0.4193548387},
            oracle_pred={"all_iou_ge_0.50": 0.5376344086},
            previous_current={"all_iou_ge_0.50": 0.4193548387},
            previous_neck={"all_iou_ge_0.50": 0.4623655914},
        )
        self.assertEqual(decision, "C. IMPROVE_SEMANTIC_TOPOLOGY")
        self.assertIn("semantic pixel", reason.lower())

    def test_no_holdout_reference(self):
        rows = base_audit._read_jsonl(base_audit.DEFAULT_MANIFEST_PATH)
        self.assertTrue(rows)
        self.assertTrue(all(not bool(row.get("present_in_authoritative_106_holdout", False)) for row in rows))

    def test_no_production_modification_path(self):
        out = audit.DEFAULT_OUTPUT_DIR.resolve()
        parts = {part.lower() for part in out.parts}
        self.assertIn("analysis", parts)
        self.assertNotIn("production", parts)


if __name__ == "__main__":
    unittest.main()
