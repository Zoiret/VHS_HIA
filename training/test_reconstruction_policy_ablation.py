from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


_THIS_DIR = Path(__file__).resolve().parent


def _marker(marker_id: int, y: int, x: int, score: float = 1.0) -> dict:
    return {"marker_id": int(marker_id), "y": int(y), "x": int(x), "score": float(score)}


class TestReconstructionPolicyAblation(unittest.TestCase):
    @staticmethod
    def _mod():
        import sys

        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import compare_reconstruction_policies as mod

        return mod

    def test_authoritative_p0_baseline_fixture_pass(self):
        mod = self._mod()
        expected_rows = [
            {"sample": "m01_p02_s00", "markers": 1, "semantic_cc": 1, "raw_reconstructed": 1, "final_reconstructed": 1},
            {"sample": "m01_p02_s04", "markers": 1, "semantic_cc": 5, "raw_reconstructed": 5, "final_reconstructed": 3},
            {"sample": "m01_p01_s00", "markers": 2, "semantic_cc": 4, "raw_reconstructed": 4, "final_reconstructed": 3},
            {"sample": "m01_p01_s01", "markers": 2, "semantic_cc": 11, "raw_reconstructed": 11, "final_reconstructed": 3},
            {"sample": "m01_p01_s02", "markers": 3, "semantic_cc": 4, "raw_reconstructed": 5, "final_reconstructed": 3},
            {"sample": "m01_p01_s03", "markers": 3, "semantic_cc": 5, "raw_reconstructed": 7, "final_reconstructed": 3},
        ]
        authoritative_first = {
            "sample": "m01_p01_s02",
            "stage": "raw reconstruction/watershed",
            "before": 3,
            "after": 5,
            "function": "_fallback_marker",
        }
        actual_rows = []
        for idx, row in enumerate(expected_rows):
            actual_rows.append(
                {
                    "sample": row["sample"],
                    "sample_index": idx,
                    "markers": row["markers"],
                    "marker_contract": True,
                    "semantic_cc": row["semantic_cc"],
                    "raw_reconstructed": row["raw_reconstructed"],
                    "final_reconstructed": row["final_reconstructed"],
                    "exact_count": row["sample"] in {"m01_p02_s00", "m01_p01_s02", "m01_p01_s03"},
                    "first_failing_invariant": authoritative_first if row["sample"] == "m01_p01_s02" else None,
                }
            )
        self.assertTrue(
            mod._authoritative_baseline_matches(
                expected_rows=expected_rows,
                actual_rows=actual_rows,
                p0_summary={"exact_count_accuracy": 0.5},
                authoritative_invariants={"first_failing_stage": authoritative_first},
            )
        )

    def test_one_changed_p0_count_causes_hard_failure(self):
        mod = self._mod()
        expected_rows = [{"sample": "m01_p02_s00", "markers": 1, "semantic_cc": 1, "raw_reconstructed": 1, "final_reconstructed": 1}]
        actual_rows = [
            {
                "sample": "m01_p02_s00",
                "sample_index": 0,
                "markers": 1,
                "marker_contract": True,
                "semantic_cc": 1,
                "raw_reconstructed": 1,
                "final_reconstructed": 3,
                "exact_count": False,
                "first_failing_invariant": None,
            }
        ]
        self.assertFalse(
            mod._authoritative_baseline_matches(
                expected_rows=expected_rows,
                actual_rows=actual_rows,
                p0_summary={"exact_count_accuracy": 0.5},
                authoritative_invariants={"first_failing_stage": None},
            )
        )

    def test_checkpoint_iteration_mismatch_1_vs_75_causes_hard_failure(self):
        mod = self._mod()
        cfg = {
            "model": {"center_feature": {"module_path": "base.decoder.blocks.x_2_2", "expected_channels": 32, "adapter_out_channels": 16}},
            "center_loss": {"normalization_mode": "legacy_num_pos"},
            "train": {"center_fp32": True},
        }
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "best_micro_overfit.pth"
            torch.save({"model": {}, "step": 1, "extra": {"best_threshold": 0.03}}, ckpt_path)
            with self.assertRaises(mod.BaselineMismatchError):
                mod._verify_checkpoint_metadata(
                    cfg=cfg,
                    checkpoint_path=ckpt_path,
                    semantic_checkpoint_path=None,
                    authoritative_checkpoint_meta={"checkpoint_path": str(ckpt_path), "saved_iteration": 75, "saved_threshold": 0.05},
                    authoritative_resolved_source={},
                )

    def test_p1_with_one_marker_cannot_output_three_labels(self):
        mod = self._mod()
        leaf = np.zeros((48, 48), dtype=np.uint8)
        leaf[6:18, 6:18] = 1
        leaf[28:34, 28:34] = 1
        leaf[38:42, 8:12] = 1
        markers = [_marker(1, 12, 12)]
        pred, _trace = mod.reconstruct_policy_componentwise(leaf_union=leaf, marker_points=markers, drop_unmarked=True, attach_unmarked=False)
        self.assertEqual(int(len(mod._positive_ids(pred))), 1)

    def test_p1_never_invokes_fallback(self):
        mod = self._mod()
        leaf = np.zeros((48, 48), dtype=np.uint8)
        leaf[6:18, 6:18] = 1
        leaf[28:34, 28:34] = 1
        markers = [_marker(1, 12, 12)]
        _pred, trace = mod.reconstruct_policy_componentwise(leaf_union=leaf, marker_points=markers, drop_unmarked=True, attach_unmarked=False)
        self.assertEqual(int(trace["fallback_marker_calls"]), 0)
        self.assertEqual(int(trace["keep_top3_call_count"]), 0)

    def test_p2_never_creates_new_label_id(self):
        mod = self._mod()
        leaf = np.zeros((48, 48), dtype=np.uint8)
        leaf[6:18, 6:18] = 1
        leaf[28:34, 28:34] = 1
        markers = [_marker(1, 12, 12)]
        pred, trace = mod.reconstruct_policy_componentwise(leaf_union=leaf, marker_points=markers, drop_unmarked=False, attach_unmarked=True)
        self.assertEqual(int(trace["new_non_marker_label_count"]), 0)
        self.assertEqual(sorted(mod._positive_ids(pred)), [1])

    def test_p3_rejected_fragment_remains_unassigned(self):
        mod = self._mod()
        leaf = np.zeros((64, 64), dtype=np.uint8)
        leaf[20:32, 20:32] = 1
        leaf[16:18, 24:28] = 1
        leaf[50:54, 50:54] = 1
        markers = [_marker(1, 26, 26)]
        pred, trace = mod.reconstruct_policy_componentwise(
            leaf_union=leaf,
            marker_points=markers,
            drop_unmarked=False,
            attach_unmarked=True,
            boundary_gate_px=8.0,
            relative_area_gate=0.10,
        )
        self.assertEqual(int(pred[51, 51]), 0)
        rejected = [c for c in trace["component_assignments"] if c.get("mode") == "unmarked_component" and not bool(c.get("assigned", False))]
        self.assertTrue(len(rejected) >= 1)

    def test_p4_zero_markers_returns_zero_labels(self):
        mod = self._mod()
        leaf = np.zeros((32, 32), dtype=np.uint8)
        leaf[4:12, 4:12] = 1
        leaf[20:28, 20:28] = 1
        pred, _trace = mod.reconstruct_policy_global(leaf_union=leaf, marker_points=[], attach_unmarked=False)
        self.assertEqual(int(len(mod._positive_ids(pred))), 0)

    def test_recommended_policy_not_written_after_baseline_mismatch(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            target = out_dir / "recommended_policy.json"
            target.write_text("stale", encoding="utf-8")
            written = mod._write_recommended_policy_if_allowed(out_dir, {"policy": "P1_DROP_UNMARKED"}, baseline_exact_match=False)
            self.assertFalse(written)
            self.assertFalse(target.exists())

    def test_lf_crlf_microset_equivalence(self):
        mod = self._mod()
        content_lf = "images/m01_p02_s00.png\tsemantic_masks/m01_p02_s00.png\nimages/m01_p02_s04.png\tsemantic_masks/m01_p02_s04.png\n"
        content_crlf = "images/m01_p02_s00.png\tsemantic_masks/m01_p02_s00.png\r\nimages/m01_p02_s04.png\tsemantic_masks/m01_p02_s04.png\r\n"
        with tempfile.TemporaryDirectory() as tmp:
            lf = Path(tmp) / "lf.txt"
            crlf = Path(tmp) / "crlf.txt"
            lf.write_text(content_lf, encoding="utf-8")
            crlf.write_text(content_crlf, encoding="utf-8")
            self.assertEqual(mod._normalized_microset_sha256(lf), mod._normalized_microset_sha256(crlf))


if __name__ == "__main__":
    unittest.main()
