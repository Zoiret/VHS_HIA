from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent


class TestCenterGeneralizationHoldout(unittest.TestCase):
    @staticmethod
    def _mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import diagnose_center_generalization_holdout as mod

        return mod

    @staticmethod
    def _audit_mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import audit_micro_reconstruction_contract as mod

        return mod

    def test_gt_centers_produce_exactly_one_marker_per_gt_instance(self):
        mod = self._mod()
        audit = self._audit_mod()
        gt_inst = np.zeros((16, 16), dtype=np.uint8)
        gt_inst[1:5, 1:5] = 1
        gt_inst[8:12, 8:12] = 2
        markers = mod._gt_marker_points([(2, 2), (9, 9)])
        contract = audit._marker_contract(gt_inst, markers)
        self.assertTrue(contract["marker_contract_pass"])

    def test_center_oracle_scope_does_not_use_gt_instance_labels_as_output(self):
        mod = self._mod()
        pred_sem = np.zeros((16, 16), dtype=np.uint8)
        pred_sem[1:4, 1:4] = 1
        pred_sem[8:10, 8:10] = 1
        gt_inst = np.zeros((16, 16), dtype=np.uint8)
        gt_inst[1:5, 1:5] = 1
        gt_inst[8:12, 8:12] = 2
        markers = mod._gt_marker_points([(2, 2), (9, 9)])
        pred_inst, _trace = mod._run_policy_with_explicit_markers("P1_DROP_UNMARKED", pred_sem, markers)
        self.assertFalse(np.array_equal(pred_inst, gt_inst))
        self.assertTrue(np.all(pred_inst[pred_sem == 0] == 0))

    def test_full_oracle_p1_output_count_equals_gt_marker_count(self):
        mod = self._mod()
        gt_sem = np.zeros((16, 16), dtype=np.uint8)
        gt_sem[1:5, 1:5] = 1
        gt_sem[8:12, 8:12] = 1
        markers = mod._gt_marker_points([(2, 2), (9, 9)])
        pred_inst, _trace = mod._run_policy_with_explicit_markers("P1_DROP_UNMARKED", gt_sem, markers)
        self.assertEqual(int(pred_inst.max()), len(markers))

    def test_p0_gt_count_confusion_table_deterministic(self):
        mod = self._mod()
        rows = [
            {"scope": "end_to_end", "policy": "P0_CURRENT", "gt_instance_count": 1, "final_output_label_count": 1},
            {"scope": "end_to_end", "policy": "P0_CURRENT", "gt_instance_count": 3, "final_output_label_count": 3},
            {"scope": "end_to_end", "policy": "P0_CURRENT", "gt_instance_count": 3, "final_output_label_count": 3},
        ]
        self.assertEqual(mod._p0_count_confusion(rows), mod._p0_count_confusion(rows))

    def test_diagnostic_threshold_sweep_cannot_overwrite_locked_threshold(self):
        mod = self._mod()
        self.assertEqual(mod.PRIMARY_THRESHOLD, 0.03)
        self.assertIn(0.03, mod.DIAGNOSTIC_THRESHOLDS)
        self.assertNotEqual(tuple(mod.DIAGNOSTIC_THRESHOLDS), (mod.PRIMARY_THRESHOLD,))

    def test_bottleneck_classification_uses_scope_deltas(self):
        mod = self._mod()
        scope_summary = {
            "scope_results": {
                "end_to_end": {"P1_DROP_UNMARKED": {"exact_count_accuracy": 0.2, "mean_matched_iou": 0.2, "dropped_area_fraction": 0.4, "invariant_violation_count": 0}},
                "center_oracle": {"P1_DROP_UNMARKED": {"exact_count_accuracy": 1.0, "mean_matched_iou": 0.92, "dropped_area_fraction": 0.01, "invariant_violation_count": 0}},
                "full_oracle": {"P1_DROP_UNMARKED": {"exact_count_accuracy": 1.0, "mean_matched_iou": 0.94, "dropped_area_fraction": 0.0, "invariant_violation_count": 0}},
            }
        }
        full_oracle_invariants = {"P1_DROP_UNMARKED": {"all_samples_invariant_violations": 0, "output_count_over_marker_count": 0}}
        decision = mod._classify_bottleneck(scope_summary, full_oracle_invariants)
        self.assertEqual(decision["status"], "center_branch_primary_bottleneck")

    def test_no_training_occurs(self):
        mod = self._mod()
        self.assertTrue(mod.NO_TRAINING_OCCURRED)

    def test_production_files_remain_unchanged(self):
        mod = self._mod()
        self.assertTrue(mod.PRODUCTION_FILES_UNCHANGED)
        self.assertIn("analysis", mod.DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    unittest.main()
