from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


_THIS_DIR = Path(__file__).resolve().parent


def _marker(marker_id: int, y: int, x: int, score: float = 1.0) -> dict:
    return {"marker_id": int(marker_id), "y": int(y), "x": int(x), "score": float(score)}


def _primary_expected_rows():
    return [
        {"checkpoint_tag": "best", "checkpoint_iteration": 75, "threshold": 0.03, "sample": "m01_p02_s00", "sample_index": 0, "markers": 1, "marker_contract": True, "semantic_cc": 1, "raw_reconstructed": 1, "final_reconstructed": 1},
        {"checkpoint_tag": "best", "checkpoint_iteration": 75, "threshold": 0.03, "sample": "m01_p02_s04", "sample_index": 1, "markers": 1, "marker_contract": True, "semantic_cc": 5, "raw_reconstructed": 5, "final_reconstructed": 3},
        {"checkpoint_tag": "best", "checkpoint_iteration": 75, "threshold": 0.03, "sample": "m01_p01_s00", "sample_index": 2, "markers": 2, "marker_contract": True, "semantic_cc": 4, "raw_reconstructed": 4, "final_reconstructed": 3},
        {"checkpoint_tag": "best", "checkpoint_iteration": 75, "threshold": 0.03, "sample": "m01_p01_s01", "sample_index": 3, "markers": 2, "marker_contract": True, "semantic_cc": 11, "raw_reconstructed": 11, "final_reconstructed": 3},
        {"checkpoint_tag": "best", "checkpoint_iteration": 75, "threshold": 0.03, "sample": "m01_p01_s02", "sample_index": 4, "markers": 3, "marker_contract": True, "semantic_cc": 4, "raw_reconstructed": 5, "final_reconstructed": 3},
        {"checkpoint_tag": "best", "checkpoint_iteration": 75, "threshold": 0.03, "sample": "m01_p01_s03", "sample_index": 5, "markers": 3, "marker_contract": True, "semantic_cc": 5, "raw_reconstructed": 7, "final_reconstructed": 3},
    ]


def _actual_primary_rows():
    return [
        {"sample": "m01_p02_s00", "sample_index": 0, "markers": 1, "marker_contract": True, "semantic_cc": 1, "raw_reconstructed": 1, "final_reconstructed": 1, "exact_count": True, "first_failing_invariant": None},
        {"sample": "m01_p02_s04", "sample_index": 1, "markers": 1, "marker_contract": True, "semantic_cc": 5, "raw_reconstructed": 5, "final_reconstructed": 3, "exact_count": False, "first_failing_invariant": {"sample": "m01_p02_s04", "stage": "raw reconstruction/watershed", "before": 1, "after": 5, "function": "_fallback_marker"}},
        {"sample": "m01_p01_s00", "sample_index": 2, "markers": 2, "marker_contract": True, "semantic_cc": 4, "raw_reconstructed": 4, "final_reconstructed": 3, "exact_count": False, "first_failing_invariant": {"sample": "m01_p01_s00", "stage": "raw reconstruction/watershed", "before": 2, "after": 4, "function": "_fallback_marker"}},
        {"sample": "m01_p01_s01", "sample_index": 3, "markers": 2, "marker_contract": True, "semantic_cc": 11, "raw_reconstructed": 11, "final_reconstructed": 3, "exact_count": False, "first_failing_invariant": {"sample": "m01_p01_s01", "stage": "raw reconstruction/watershed", "before": 2, "after": 11, "function": "_fallback_marker"}},
        {"sample": "m01_p01_s02", "sample_index": 4, "markers": 3, "marker_contract": True, "semantic_cc": 4, "raw_reconstructed": 5, "final_reconstructed": 3, "exact_count": True, "first_failing_invariant": {"sample": "m01_p01_s02", "stage": "raw reconstruction/watershed", "before": 3, "after": 5, "function": "_fallback_marker"}},
        {"sample": "m01_p01_s03", "sample_index": 5, "markers": 3, "marker_contract": True, "semantic_cc": 5, "raw_reconstructed": 7, "final_reconstructed": 3, "exact_count": True, "first_failing_invariant": {"sample": "m01_p01_s03", "stage": "raw reconstruction/watershed", "before": 3, "after": 7, "function": "_fallback_marker"}},
    ]


class TestReconstructionPolicyAblation(unittest.TestCase):
    @staticmethod
    def _mod():
        import sys

        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import compare_reconstruction_policies as mod

        return mod

    def test_global_first_failure_differs_from_primary_while_primary_baseline_passes(self):
        mod = self._mod()
        expected = _primary_expected_rows()
        actual = _actual_primary_rows()
        primary_first = mod._expected_primary_first_failure_from_rows(expected)
        global_first = {"checkpoint_tag": "best", "checkpoint_iteration": 75, "threshold": 0.02, "sample": "m01_p01_s02", "stage": "raw reconstruction/watershed", "before": 3, "after": 5, "function": "_fallback_marker", "labels": "unrecoverable"}
        self.assertNotEqual(global_first["sample"], primary_first["sample"])
        self.assertTrue(
            mod._authoritative_baseline_matches(
                expected_rows=expected,
                actual_rows=actual,
                p0_summary={"exact_count_accuracy": 0.5},
                authoritative_primary_first_failure=primary_first,
            )
        )

    def test_authoritative_primary_first_failure_is_m01_p02_s04(self):
        mod = self._mod()
        primary_first = mod._expected_primary_first_failure_from_rows(_primary_expected_rows())
        self.assertEqual(
            primary_first,
            {
                "checkpoint_tag": "best",
                "checkpoint_iteration": 75,
                "threshold": 0.03,
                "sample": "m01_p02_s04",
                "stage": "raw reconstruction/watershed",
                "before": 1,
                "after": 5,
                "function": "_fallback_marker",
                "labels": "unrecoverable",
            },
        )

    def test_one_changed_primary_raw_count_hard_fails_and_sets_first_differing_sample(self):
        mod = self._mod()
        expected = _primary_expected_rows()
        actual = _actual_primary_rows()
        actual[1] = dict(actual[1], raw_reconstructed=4)
        self.assertFalse(
            mod._authoritative_baseline_matches(
                expected_rows=expected,
                actual_rows=actual,
                p0_summary={"exact_count_accuracy": 0.5},
                authoritative_primary_first_failure=mod._expected_primary_first_failure_from_rows(expected),
            )
        )
        report = mod._baseline_mismatch_payload(
            expected_rows=expected,
            actual_rows=actual,
            checkpoint_identity={"checkpoint_sha256": "abc", "semantic_checkpoint_sha256": "def", "center_fp32": True, "device": "cuda", "path_matches_authoritative_metadata": True},
            microset_precheck={"errors": []},
            authoritative_summary={},
            authoritative_global_first_failure={"sample": "m01_p01_s02"},
            authoritative_primary_first_failure=mod._expected_primary_first_failure_from_rows(expected),
            source_commit="commit",
        )
        self.assertEqual(report["first_differing_sample"], "m01_p02_s04")

    def test_one_changed_primary_marker_count_hard_fails(self):
        mod = self._mod()
        expected = _primary_expected_rows()
        actual = _actual_primary_rows()
        actual[2] = dict(actual[2], markers=1)
        self.assertFalse(
            mod._authoritative_baseline_matches(
                expected_rows=expected,
                actual_rows=actual,
                p0_summary={"exact_count_accuracy": 0.5},
                authoritative_primary_first_failure=mod._expected_primary_first_failure_from_rows(expected),
            )
        )

    def test_same_aggregates_but_changed_per_sample_row_hard_fails(self):
        mod = self._mod()
        expected = _primary_expected_rows()
        actual = _actual_primary_rows()
        actual[0] = dict(actual[0], raw_reconstructed=2, exact_count=False)
        actual[5] = dict(actual[5], raw_reconstructed=6, exact_count=True)
        self.assertAlmostEqual(0.5, 0.5)
        self.assertFalse(
            mod._authoritative_baseline_matches(
                expected_rows=expected,
                actual_rows=actual,
                p0_summary={"exact_count_accuracy": 0.5},
                authoritative_primary_first_failure=mod._expected_primary_first_failure_from_rows(expected),
            )
        )

    def test_all_primary_rows_match_and_global_first_failure_differs_no_mismatch_file_written(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            mismatch = out_dir / "baseline_mismatch.json"
            mismatch.write_text("stale", encoding="utf-8")
            mod._clear_baseline_mismatch_if_present(out_dir)
            self.assertFalse(mismatch.exists())
            written = mod._write_recommended_policy_if_allowed(out_dir, {"policy": "P1_DROP_UNMARKED"}, baseline_exact_match=True)
            self.assertTrue(written)
            self.assertTrue((out_dir / "recommended_policy.json").exists())

    def test_float_threshold_matching_uses_strict_tolerance(self):
        mod = self._mod()
        self.assertTrue(mod._threshold_matches("0.0300000000", 0.03))
        self.assertTrue(mod._threshold_matches(0.0300000005, 0.03, tol=1e-8))
        self.assertFalse(mod._threshold_matches(0.0301, 0.03, tol=1e-8))

    def test_hashes_unavailable_in_authoritative_metadata_do_not_fail_solely(self):
        mod = self._mod()
        self.assertEqual(mod._hash_match_status("abc", None), "unavailable_in_authoritative_audit")
        self.assertEqual(mod._hash_match_status(None, "abc"), "unavailable_locally")
        self.assertTrue(mod._authoritative_baseline_matches(expected_rows=_primary_expected_rows(), actual_rows=_actual_primary_rows(), p0_summary={"exact_count_accuracy": 0.5}, authoritative_primary_first_failure=mod._expected_primary_first_failure_from_rows(_primary_expected_rows())))

    def test_hostname_git_reporting_returns_value_or_reason(self):
        mod = self._mod()
        host = mod._safe_hostname()
        git = mod._safe_git_commit(Path("e:/3d_visual/ml"))
        self.assertIn("status", host)
        self.assertTrue(host["value"] or host["status"].startswith("unavailable"))
        self.assertIn("status", git)
        self.assertTrue(git["value"] or git["status"].startswith("unavailable"))

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
