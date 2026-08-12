from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent


class TestEvaluateSemanticTopologyAuxPostrun(unittest.TestCase):
    @staticmethod
    def _mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import evaluate_semantic_topology_aux_postrun as mod

        return mod

    def test_research_eval_contract_uses_test_txt_only(self):
        mod = self._mod()
        contract = mod.build_eval_contract()
        self.assertTrue(contract["research_eval_split"].endswith("datasets\\converted_full_multiclass_curated\\test.txt") or contract["research_eval_split"].endswith("datasets/converted_full_multiclass_curated/test.txt"))
        self.assertFalse(contract["checkpoint_selection_from_research_eval"])
        self.assertFalse(contract["holdout_used"])

    def test_no_center_full_val_manifest_in_contract_or_constants(self):
        mod = self._mod()
        serialized = str(mod.build_eval_contract()) + str(mod.PROHIBITED_PATH_SUBSTRINGS)
        self.assertNotIn("center_full_val_manifest.jsonl", str(mod.RESEARCH_EVAL_SPLIT))
        self.assertIn("center_full_val_manifest.jsonl", serialized)

    def test_locked_k_normalizer_is_unchanged(self):
        mod = self._mod()
        self.assertEqual(mod.NORMALIZER_METHOD, "centroid_distance_k_normalizer")
        with mock.patch.object(mod.k_audit, "normalize_mask_exact_k", return_value={"labels": np.zeros((4, 4), dtype=np.uint8), "final_group_count": 1}) as patched:
            mod.run_locked_normalization(np.zeros((4, 4), dtype=np.uint8), 1)
        patched.assert_called_once()
        args = patched.call_args.args
        self.assertEqual(args[2], "centroid_distance_k_normalizer")

    def test_prohibited_path_guard_rejects_holdout_and_center_manifest(self):
        mod = self._mod()
        with self.assertRaises(SystemExit):
            mod._assert_safe_path(Path("training/manifests/center_full_val_manifest.jsonl"))
        with self.assertRaises(SystemExit):
            mod._assert_safe_path(Path("authoritative_106_holdout.txt"))

    def test_checkpoint_order_is_fixed_and_not_metric_selected(self):
        mod = self._mod()
        self.assertEqual(mod.CHECKPOINT_ORDER, ("baseline", "topology_best_semantic", "topology_last_diagnostic"))
        self.assertTrue(any(spec.diagnostic_only for spec in mod.CHECKPOINTS if spec.label == "topology_last_diagnostic"))


if __name__ == "__main__":
    unittest.main()
