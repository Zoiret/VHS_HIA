from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent


class TestAuditBoundaryUpperBoundForLeafletReconstruction(unittest.TestCase):
    @staticmethod
    def _mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import audit_boundary_upper_bound_for_leaflet_reconstruction as mod

        return mod

    def test_fp_oracle_never_adds_foreground(self):
        mod = self._mod()
        pred = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        gt_union = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        got = mod._fp_oracle_mask(pred, gt_union)
        self.assertTrue(np.all(got <= pred))
        self.assertTrue(np.array_equal(got, np.array([[1, 0], [0, 1]], dtype=np.uint8)))

    def test_fn_oracle_never_removes_foreground(self):
        mod = self._mod()
        pred = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        gt_union = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        got = mod._fn_oracle_mask(pred, gt_union)
        self.assertTrue(np.all(got >= pred))
        self.assertTrue(np.array_equal(got, np.array([[1, 1], [0, 1]], dtype=np.uint8)))

    def test_bridge_oracle_only_removes_identified_bridge_pixels(self):
        mod = self._mod()
        pred = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        bridge = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        got = mod._bridge_oracle_mask(pred, bridge)
        self.assertTrue(np.all(got <= pred))
        self.assertEqual(int(np.sum((pred > 0) & (got == 0))), int(np.sum(bridge > 0)))
        self.assertTrue(np.array_equal(got, np.array([[1, 0], [0, 1]], dtype=np.uint8)))

    def test_gt_union_uses_instance_positive_only(self):
        mod = self._mod()
        inst = np.array([[0, 2], [1, 0]], dtype=np.uint8)
        got = mod._gt_union_mask(inst)
        self.assertTrue(np.array_equal(got, np.array([[0, 1], [1, 0]], dtype=np.uint8)))

    def test_locked_normalizer_constant_and_contract(self):
        mod = self._mod()
        self.assertEqual(mod.NORMALIZER_METHOD, "centroid_distance_k_normalizer")
        contract = mod.build_audit_contract()
        self.assertFalse(contract["holdout_used"])
        self.assertFalse(contract["center_full_val_manifest_used"])
        self.assertFalse(contract["training_launched"])
        self.assertTrue(contract["audit_split"].endswith("datasets\\converted_full_multiclass_curated\\test.txt") or contract["audit_split"].endswith("datasets/converted_full_multiclass_curated/test.txt"))

    def test_safe_path_rejects_center_full_val_and_holdout(self):
        mod = self._mod()
        with self.assertRaises(SystemExit):
            mod._assert_safe_path(Path("training/manifests/center_full_val_manifest.jsonl"))
        with self.assertRaises(SystemExit):
            mod._assert_safe_path(Path("authoritative_106_holdout.txt"))


if __name__ == "__main__":
    unittest.main()
