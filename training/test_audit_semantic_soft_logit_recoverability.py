from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


_THIS_DIR = Path(__file__).resolve().parent


class TestAuditSemanticSoftLogitRecoverability(unittest.TestCase):
    @staticmethod
    def _mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import audit_semantic_soft_logit_recoverability as mod

        return mod

    def test_probability_extraction_and_hard_argmax_reproduction(self):
        mod = self._mod()
        logits = torch.tensor([[[[0.0, 1.0], [0.5, -1.0]], [[2.0, 0.0], [0.6, 0.0]], [[-1.0, 0.5], [0.4, 2.0]]]], dtype=torch.float32)
        out = mod._softmax_probs(logits)
        self.assertEqual(out["probs"].shape, (3, 2, 2))
        expected = torch.argmax(logits, dim=1)[0].numpy().astype(np.uint8)
        self.assertTrue(np.array_equal(out["pred_semantic"], expected))
        self.assertTrue(np.allclose(out["leaflet_margin"], out["p_leaf"] - out["p_competing"]))

    def test_pixel_category_exclusivity_for_base_partition(self):
        mod = self._mod()
        gt_sem = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        gt_inst = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        pred_sem = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        pred_union = (pred_sem == 1).astype(np.uint8)
        contract = mock.Mock()
        with mock.patch.object(mod, "_critical_foreground_mask", return_value=np.array([[1, 1], [0, 0]], dtype=np.uint8)), \
             mock.patch.object(mod, "_bridge_component_mask", return_value=np.array([[0, 0], [0, 1]], dtype=np.uint8)):
            cats = mod._topology_pixel_categories(
                gt_sem_u8=gt_sem,
                gt_inst_u8=gt_inst,
                pred_sem_u8=pred_sem,
                pred_union01=pred_union,
                topology_contract=contract,
            )
        base_sum = cats["TRUE_LEAFLET_CORRECT"] + cats["TRUE_LEAFLET_MISSED"] + cats["TRUE_BACKGROUND_CORRECT"] + cats["FALSE_LEAFLET"]
        self.assertTrue(np.array_equal(base_sum, np.ones_like(gt_sem, dtype=np.uint8)))
        self.assertEqual(int(np.sum(cats["MISSING_TOPOLOGY_CRITICAL"])), 1)
        self.assertEqual(int(np.sum(cats["FALSE_BRIDGE_PIXELS"])), 1)

    def test_threshold_and_hysteresis_are_deterministic(self):
        mod = self._mod()
        p_leaf = np.array(
            [
                [0.2, 0.31, 0.29],
                [0.51, 0.45, 0.1],
                [0.52, 0.33, 0.05],
            ],
            dtype=np.float32,
        )
        t1 = mod._threshold_mask(p_leaf, 0.30)
        t2 = mod._threshold_mask(p_leaf, 0.30)
        self.assertTrue(np.array_equal(t1, t2))
        h1 = mod._hysteresis_leaflet_mask(p_leaf, high_threshold=0.50, candidate_threshold=0.30)
        h2 = mod._hysteresis_leaflet_mask(p_leaf, high_threshold=0.50, candidate_threshold=0.30)
        self.assertTrue(np.array_equal(h1, h2))
        self.assertLessEqual(int(np.sum(h1)), int(np.sum(t1)))

    def test_locked_k_normalizer_is_unchanged(self):
        mod = self._mod()
        self.assertEqual(mod.NORMALIZER_METHOD, "centroid_distance_k_normalizer")
        with mock.patch.object(mod.postrun, "run_locked_normalization", return_value={"labels": np.zeros((2, 2), dtype=np.uint8), "final_group_count": 1}) as patched:
            out = mod.postrun.run_locked_normalization(np.zeros((2, 2), dtype=np.uint8), 1)
        self.assertEqual(out["final_group_count"], 1)
        patched.assert_called_once()

    def test_contract_uses_test_txt_only_and_no_holdout(self):
        mod = self._mod()
        contract = mod.build_audit_contract()
        self.assertTrue(contract["audit_split"].endswith("datasets\\converted_full_multiclass_curated\\test.txt") or contract["audit_split"].endswith("datasets/converted_full_multiclass_curated/test.txt"))
        self.assertFalse(contract["holdout_used"])
        self.assertFalse(contract["center_full_val_manifest_used"])
        with self.assertRaises(SystemExit):
            mod._assert_safe_path(Path("training/manifests/center_full_val_manifest.jsonl"))
        with self.assertRaises(SystemExit):
            mod._assert_safe_path(Path("authoritative_106_holdout.txt"))

    def test_margin_mask_rule(self):
        mod = self._mod()
        p_leaf = np.array([[0.35, 0.29], [0.31, 0.8]], dtype=np.float32)
        p_comp = np.array([[0.40, 0.20], [0.50, 0.1]], dtype=np.float32)
        got = mod._margin_mask(p_leaf, p_comp, p_leaf_min=0.30, margin_min=-0.10)
        expected = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(got, expected))


if __name__ == "__main__":
    unittest.main()
