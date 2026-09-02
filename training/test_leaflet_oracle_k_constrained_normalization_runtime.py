from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import leaflet_oracle_k_constrained_normalization_audit as mod


class TestLeafletOracleKConstrainedNormalizationRuntime(unittest.TestCase):
    def _assert_reference_parity(self, mask01: np.ndarray, k: int) -> None:
        ref_profile: dict[str, object] = {}
        opt_profile: dict[str, object] = {}
        ref = mod.normalize_mask_exact_k(mask01, k, "centroid_distance_k_normalizer", implementation="reference", profile=ref_profile)
        opt = mod.normalize_mask_exact_k(mask01, k, "centroid_distance_k_normalizer", implementation="optimized", profile=opt_profile)
        self.assertTrue(np.array_equal(ref["labels"], opt["labels"]))
        self.assertEqual(ref["final_group_count"], opt["final_group_count"])
        self.assertEqual(ref["merge_operations"], opt["merge_operations"])
        self.assertEqual(ref["reason"], opt["reason"])
        self.assertIn("connected_component_calls", dict(opt_profile.get("call_counts") or {}))

    def test_gt2_fragmented_mask_reference_parity(self):
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        mask[1:3, 6:8] = 1
        mask[6:8, 2:4] = 1
        self._assert_reference_parity(mask, 2)

    def test_gt3_fragmented_mask_reference_parity(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        mask[1:3, 7:9] = 1
        mask[7:9, 4:6] = 1
        mask[4:5, 4:5] = 1
        self._assert_reference_parity(mask, 3)

    def test_already_separated_mask_reference_parity(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        mask[1:3, 5:7] = 1
        self._assert_reference_parity(mask, 2)

    def test_tie_breaking_reference_parity(self):
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        mask[1:3, 6:8] = 1
        mask[6:8, 1:3] = 1
        mask[6:8, 6:8] = 1
        self._assert_reference_parity(mask, 2)

    def test_fragmented_open_closed_style_masks_reference_parity(self):
        closed_mask = np.zeros((8, 8), dtype=np.uint8)
        closed_mask[2:6, 2:6] = 1
        open_mask = closed_mask.copy()
        open_mask[3:5, 3:5] = 0
        self._assert_reference_parity(closed_mask, 2)
        self._assert_reference_parity(open_mask, 2)


if __name__ == "__main__":
    unittest.main()
