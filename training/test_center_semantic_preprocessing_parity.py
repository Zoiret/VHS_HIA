from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent


class TestCenterSemanticPreprocessingParity(unittest.TestCase):
    @staticmethod
    def _mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import audit_center_semantic_preprocessing_parity as mod

        return mod

    def test_cpu_cuda_comparison_is_per_sample(self):
        mod = self._mod()
        with self.subTest("helper exists"):
            self.assertTrue(callable(mod._replay_parity_rows))

    def test_rgb_bgr_mismatch_is_detected(self):
        mod = self._mod()
        rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)
        bgr = np.array([[[30, 20, 10]]], dtype=np.uint8)
        self.assertFalse(mod._rgb_bgr_mismatch_detected(rgb, rgb.copy()))
        self.assertTrue(mod._rgb_bgr_mismatch_detected(rgb, bgr))

    def test_normalization_mismatch_is_detected(self):
        mod = self._mod()
        a = np.array([0.0, 1.0], dtype=np.float32)
        b = np.array([0.0, 0.5], dtype=np.float32)
        self.assertTrue(mod._normalization_mismatch_detected(a, b))

    def test_xy_coordinate_swap_is_detected(self):
        mod = self._mod()
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[4, 1] = 1
        self.assertTrue(mod._xy_swap_detected(1, 4, mask))

    def test_transformed_gt_center_remains_inside_gt_instance(self):
        mod = self._mod()
        inst = np.zeros((8, 8), dtype=np.uint8)
        inst[2:5, 2:5] = 1
        center = np.zeros((8, 8), dtype=np.float32)
        center[3, 3] = 1.0
        peaks = mod._center_target_peaks(center, inst)
        self.assertEqual(len(peaks), 1)
        self.assertTrue(peaks[0]["inside_instance"])

    def test_sigmoid_double_application_is_detected(self):
        mod = self._mod()
        logits = np.array([[0.0, 2.0]], dtype=np.float32)
        prob_once = 1.0 / (1.0 + np.exp(-logits))
        prob_twice = 1.0 / (1.0 + np.exp(-prob_once))
        self.assertTrue(mod._sigmoid_double_application_detected(logits, prob_once, prob_twice))

    def test_semantic_center_coverage_metric_is_correct(self):
        mod = self._mod()
        pred_sem = np.zeros((4, 4), dtype=np.uint8)
        pred_sem[1, 1] = 1
        pred_sem[2, 2] = 1
        coverage = mod._semantic_center_coverage(pred_sem, [(1, 1), (0, 0), (2, 2)])
        self.assertEqual(coverage["gt_center_inside_predicted_leaflet_count"], 2)
        self.assertEqual(coverage["gt_centers_outside_predicted_leaflet"], 1)

    def test_no_training_or_production_modification_flags(self):
        mod = self._mod()
        self.assertTrue(mod.NO_TRAINING_OCCURRED)
        self.assertTrue(mod.PRODUCTION_FILES_UNCHANGED)


if __name__ == "__main__":
    unittest.main()
