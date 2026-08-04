from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent


def _marker(marker_id: int, y: int, x: int, score: float = 1.0) -> dict:
    return {"marker_id": int(marker_id), "y": int(y), "x": int(x), "score": float(score)}


def _json_fixture_row() -> dict:
    return {
        "checkpoint_tag": "best",
        "checkpoint_iteration": 75,
        "threshold": 0.03,
        "sample": "m01_p02_s04",
        "sample_index": 1,
        "identifiers": {"gt_instance_count": 1},
        "marker_contract": {
            "extracted_marker_count": 1,
            "marker_contract_pass": True,
        },
        "semantic_topology": {
            "predicted_leaflet_connected_components": 5,
        },
        "reconstruction_stages": {
            "raw_reconstruction": {"count": 5},
            "final_labels_passed_to_metrics": {"count": 3},
        },
        "metrics": {
            "reconstructed_instance_count": 3,
            "instance_exact_count": False,
        },
        "failure_class": ["B"],
        "invariant": {
            "sample": "m01_p02_s04",
            "stage": "raw reconstruction/watershed",
            "before": 1,
            "after": 5,
            "labels": [2, 3, 4, 5],
            "function": "_fallback_marker",
        },
    }


def _csv_fixture_row() -> dict:
    return {
        "checkpoint_tag": "best",
        "checkpoint_iteration": "75",
        "threshold": "0.03",
        "sample": "m01_p02_s04",
        "sample_index": "1",
        "gt_instances": "1",
        "markers": "1",
        "marker_contract": "True",
        "semantic_cc": "5",
        "raw_reconstructed": "5",
        "final_reconstructed": "3",
        "exact_count": "False",
        "failure_class": "B",
    }


def _primary_expected_rows():
    return [
        {"checkpoint_tag": "best", "checkpoint_iteration": 75, "threshold": 0.03, "sample": "m01_p02_s00", "sample_index": 0, "gt_instances": 1, "markers": 1, "marker_contract": True, "semantic_cc": 1, "raw_reconstructed": 1, "final_reconstructed": 1, "exact_count": True, "failure_class": ("G",), "invariant": None},
        {"checkpoint_tag": "best", "checkpoint_iteration": 75, "threshold": 0.03, "sample": "m01_p02_s04", "sample_index": 1, "gt_instances": 1, "markers": 1, "marker_contract": True, "semantic_cc": 5, "raw_reconstructed": 5, "final_reconstructed": 3, "exact_count": False, "failure_class": ("B",), "invariant": _json_fixture_row()["invariant"]},
        {"checkpoint_tag": "best", "checkpoint_iteration": 75, "threshold": 0.03, "sample": "m01_p01_s00", "sample_index": 2, "gt_instances": 2, "markers": 2, "marker_contract": True, "semantic_cc": 4, "raw_reconstructed": 4, "final_reconstructed": 3, "exact_count": False, "failure_class": ("B",), "invariant": {"sample": "m01_p01_s00", "stage": "raw reconstruction/watershed", "before": 2, "after": 4, "function": "_fallback_marker"}},
        {"checkpoint_tag": "best", "checkpoint_iteration": 75, "threshold": 0.03, "sample": "m01_p01_s01", "sample_index": 3, "gt_instances": 2, "markers": 2, "marker_contract": True, "semantic_cc": 11, "raw_reconstructed": 11, "final_reconstructed": 3, "exact_count": False, "failure_class": ("A", "B"), "invariant": {"sample": "m01_p01_s01", "stage": "raw reconstruction/watershed", "before": 2, "after": 11, "function": "_fallback_marker"}},
        {"checkpoint_tag": "best", "checkpoint_iteration": 75, "threshold": 0.03, "sample": "m01_p01_s02", "sample_index": 4, "gt_instances": 3, "markers": 3, "marker_contract": True, "semantic_cc": 4, "raw_reconstructed": 5, "final_reconstructed": 3, "exact_count": True, "failure_class": ("B",), "invariant": {"sample": "m01_p01_s02", "stage": "raw reconstruction/watershed", "before": 3, "after": 5, "function": "_fallback_marker"}},
        {"checkpoint_tag": "best", "checkpoint_iteration": 75, "threshold": 0.03, "sample": "m01_p01_s03", "sample_index": 5, "gt_instances": 3, "markers": 3, "marker_contract": True, "semantic_cc": 5, "raw_reconstructed": 7, "final_reconstructed": 3, "exact_count": True, "failure_class": ("A", "B"), "invariant": {"sample": "m01_p01_s03", "stage": "raw reconstruction/watershed", "before": 3, "after": 7, "function": "_fallback_marker"}},
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
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import compare_reconstruction_policies as mod

        return mod

    def test_real_nested_json_row_normalizes_correctly(self):
        mod = self._mod()
        row = mod._normalize_authoritative_json_rows([_json_fixture_row()])[0]
        self.assertEqual(
            row,
            {
                "checkpoint_tag": "best",
                "checkpoint_iteration": 75,
                "threshold": 0.03,
                "sample": "m01_p02_s04",
                "sample_index": 1,
                "gt_instances": 1,
                "markers": 1,
                "marker_contract": True,
                "semantic_cc": 5,
                "raw_reconstructed": 5,
                "final_reconstructed": 3,
                "exact_count": False,
                "failure_class": ("B",),
                "invariant": _json_fixture_row()["invariant"],
            },
        )

    def test_real_flat_csv_row_normalizes_correctly(self):
        mod = self._mod()
        row = mod._normalize_authoritative_csv_rows([_csv_fixture_row()])[0]
        self.assertEqual(
            row,
            {
                "checkpoint_tag": "best",
                "checkpoint_iteration": 75,
                "threshold": 0.03,
                "sample": "m01_p02_s04",
                "sample_index": 1,
                "gt_instances": 1,
                "markers": 1,
                "marker_contract": True,
                "semantic_cc": 5,
                "raw_reconstructed": 5,
                "final_reconstructed": 3,
                "exact_count": False,
                "failure_class": ("B",),
                "invariant": None,
            },
        )

    def test_json_and_csv_normalize_identically(self):
        mod = self._mod()
        j = mod._normalize_authoritative_json_rows([_json_fixture_row()])[0]
        c = mod._normalize_authoritative_csv_rows([_csv_fixture_row()])[0]
        self.assertEqual(mod._canonical_row_projection(j), mod._canonical_row_projection(c))

    def test_nested_marker_field_missing_gives_controlled_error(self):
        mod = self._mod()
        row = _json_fixture_row()
        del row["marker_contract"]["extracted_marker_count"]
        with self.assertRaises(mod.AuthoritativeSchemaError) as ctx:
            mod._normalize_authoritative_json_rows([row])
        payload = ctx.exception.payload
        self.assertEqual(payload["status"], "authoritative_schema_error")
        self.assertEqual(payload["source"], "per_sample_audit.json")
        self.assertEqual(payload["canonical_field"], "markers")
        self.assertEqual(payload["missing_path"], ["marker_contract", "extracted_marker_count"])

    def test_nested_raw_reconstruction_field_missing_gives_controlled_error(self):
        mod = self._mod()
        row = _json_fixture_row()
        del row["reconstruction_stages"]["raw_reconstruction"]["count"]
        with self.assertRaises(mod.AuthoritativeSchemaError) as ctx:
            mod._normalize_authoritative_json_rows([row])
        payload = ctx.exception.payload
        self.assertEqual(payload["status"], "authoritative_schema_error")
        self.assertEqual(payload["canonical_field"], "raw_reconstructed")
        self.assertEqual(payload["missing_path"], ["reconstruction_stages", "raw_reconstruction", "count"])

    def test_final_reconstructed_count_differs_from_metrics_count_hard_fails(self):
        mod = self._mod()
        row = _json_fixture_row()
        row["metrics"]["reconstructed_instance_count"] = 4
        with self.assertRaises(mod.AuthoritativeSchemaError) as ctx:
            mod._normalize_authoritative_json_rows([row])
        self.assertIn("metrics.reconstructed_instance_count", json.dumps(ctx.exception.payload, ensure_ascii=False))

    def test_csv_false_parses_as_false(self):
        mod = self._mod()
        row = _csv_fixture_row()
        row["marker_contract"] = "False"
        row["exact_count"] = "0"
        normalized = mod._normalize_authoritative_csv_rows([row])[0]
        self.assertFalse(normalized["marker_contract"])
        self.assertFalse(normalized["exact_count"])

    def test_json_failure_class_list_and_csv_string_become_same_tuple(self):
        mod = self._mod()
        row_json = _json_fixture_row()
        row_json["failure_class"] = ["A", "B"]
        row_csv = _csv_fixture_row()
        row_csv["failure_class"] = "A,B"
        self.assertEqual(mod._normalize_authoritative_json_rows([row_json])[0]["failure_class"], ("A", "B"))
        self.assertEqual(mod._normalize_authoritative_csv_rows([row_csv])[0]["failure_class"], ("A", "B"))

    def test_primary_filtering_returns_six_rows(self):
        mod = self._mod()
        rows = _primary_expected_rows() + [
            {"checkpoint_tag": "best", "checkpoint_iteration": 75, "threshold": 0.02, "sample": "m01_p02_s00", "sample_index": 0, "gt_instances": 1, "markers": 3, "marker_contract": False, "semantic_cc": 1, "raw_reconstructed": 3, "final_reconstructed": 3, "exact_count": False, "failure_class": ("G",), "invariant": None},
            {"checkpoint_tag": "last", "checkpoint_iteration": 1000, "threshold": 0.03, "sample": "m01_p02_s00", "sample_index": 0, "gt_instances": 1, "markers": 1, "marker_contract": True, "semantic_cc": 1, "raw_reconstructed": 1, "final_reconstructed": 1, "exact_count": True, "failure_class": ("G",), "invariant": None},
        ]
        filtered = mod._policy_rows_for_primary(rows)
        self.assertEqual(len(filtered), 6)

    def test_primary_first_failure_is_m01_p02_s04_1_to_5(self):
        mod = self._mod()
        primary_first = mod._expected_primary_first_failure_from_rows(_primary_expected_rows())
        self.assertEqual(primary_first["sample"], "m01_p02_s04")
        self.assertEqual(primary_first["before"], 1)
        self.assertEqual(primary_first["after"], 5)
        self.assertEqual(primary_first["function"], "_fallback_marker")

    def test_no_keyerror_escapes_on_broken_json_schema(self):
        mod = self._mod()
        row = _json_fixture_row()
        del row["semantic_topology"]
        with self.assertRaises(mod.AuthoritativeSchemaError) as ctx:
            mod._normalize_authoritative_json_rows([row])
        self.assertNotIn("KeyError", repr(ctx.exception))

    def test_p1_p4_do_not_execute_after_schema_source_mismatch(self):
        mod = self._mod()
        mismatch = mod.AuthoritativeSourceMismatchError({"status": "authoritative_source_mismatch", "reason": "test"})
        argv = [
            "compare_reconstruction_policies.py",
            "--config",
            "training/configs/unetpp_effb3_centerhead_spatial_x2_2_adapter_legacy_fp32_micro.yaml",
            "--run-dir",
            "training/analysis/centerhead_spatial_x2_2_adapter_legacy_fp32_micro_overfit",
            "--microset-file",
            "training/analysis/centerhead_spatial_x2_2_adapter_legacy_fp32_micro_overfit/microset.txt",
            "--authoritative-audit-dir",
            "training/analysis/centerhead_spatial_x2_2_adapter_reconstruction_audit",
            "--output-dir",
            str((_REPO_ROOT / "tmp_schema_stop").resolve()),
            "--device",
            "cpu",
        ]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(mod, "_load_authoritative_sources", side_effect=mismatch):
                with mock.patch.object(mod, "run_policy", side_effect=AssertionError("run_policy must not execute")):
                    with self.assertRaises(SystemExit):
                        mod.main()

    def test_load_authoritative_sources_cross_checks_json_and_csv(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "per_sample_audit.json").write_text(json.dumps([_json_fixture_row()], ensure_ascii=False, indent=2), encoding="utf-8")
            (root / "per_sample_audit.csv").write_text(
                "checkpoint_tag,checkpoint_iteration,threshold,sample,sample_index,gt_instances,markers,marker_contract,semantic_cc,raw_reconstructed,final_reconstructed,exact_count,failure_class\n"
                "best,75,0.03,m01_p02_s04,1,1,1,True,5,5,3,False,B\n",
                encoding="utf-8",
            )
            bundle = mod._load_authoritative_sources(root)
            self.assertEqual(bundle["json_row_count"], 1)
            self.assertEqual(bundle["csv_row_count"], 1)
            self.assertTrue(bundle["source_consistency"])
            self.assertEqual(bundle["canonical_rows"][0]["invariant"]["function"], "_fallback_marker")

    def test_load_authoritative_sources_mismatch_hard_fails(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "per_sample_audit.json").write_text(json.dumps([_json_fixture_row()], ensure_ascii=False, indent=2), encoding="utf-8")
            (root / "per_sample_audit.csv").write_text(
                "checkpoint_tag,checkpoint_iteration,threshold,sample,sample_index,gt_instances,markers,marker_contract,semantic_cc,raw_reconstructed,final_reconstructed,exact_count,failure_class\n"
                "best,75,0.03,m01_p02_s04,1,1,1,True,5,4,3,False,B\n",
                encoding="utf-8",
            )
            with self.assertRaises(mod.AuthoritativeSourceMismatchError):
                mod._load_authoritative_sources(root)

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
        self.assertEqual(primary_first["sample"], "m01_p02_s04")

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
        self.assertTrue(
            mod._authoritative_baseline_matches(
                expected_rows=_primary_expected_rows(),
                actual_rows=_actual_primary_rows(),
                p0_summary={"exact_count_accuracy": 0.5},
                authoritative_primary_first_failure=mod._expected_primary_first_failure_from_rows(_primary_expected_rows()),
            )
        )

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

    def test_labels_to_bgr_all_zero_map(self):
        mod = self._mod()
        labels = np.zeros((8, 9), dtype=np.int32)
        out = mod._labels_to_bgr(labels)
        self.assertEqual(out.shape, (8, 9, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.array_equal(out, np.zeros((8, 9, 3), dtype=np.uint8)))

    def test_labels_to_bgr_one_positive_label(self):
        mod = self._mod()
        labels = np.zeros((6, 6), dtype=np.int32)
        labels[1:4, 2:5] = 3
        out = mod._labels_to_bgr(labels)
        self.assertTrue(np.all(out[0, 0] == 0))
        self.assertTrue(np.any(out[2, 3] != 0))

    def test_labels_to_bgr_non_contiguous_labels_get_distinct_colors(self):
        mod = self._mod()
        labels = np.zeros((5, 6), dtype=np.int32)
        labels[0:2, 0:2] = 2
        labels[2:4, 2:4] = 5
        labels[1:3, 4:6] = 11
        out = mod._labels_to_bgr(labels)
        self.assertFalse(np.array_equal(out[0, 0], out[2, 2]))
        self.assertFalse(np.array_equal(out[0, 0], out[1, 4]))
        self.assertFalse(np.array_equal(out[2, 2], out[1, 4]))

    def test_labels_to_bgr_repeated_calls_are_byte_identical(self):
        mod = self._mod()
        labels = np.zeros((7, 7), dtype=np.int32)
        labels[1:3, 1:3] = 2
        labels[4:6, 4:6] = 5
        self.assertTrue(np.array_equal(mod._labels_to_bgr(labels), mod._labels_to_bgr(labels)))

    def test_labels_to_bgr_does_not_mutate_input(self):
        mod = self._mod()
        labels = np.zeros((6, 6), dtype=np.int32)
        labels[2:4, 2:4] = 7
        original = labels.copy()
        _ = mod._labels_to_bgr(labels)
        self.assertTrue(np.array_equal(labels, original))

    def test_labels_to_bgr_int16_int32_int64_equivalent(self):
        mod = self._mod()
        base = np.zeros((6, 6), dtype=np.int64)
        base[1:3, 1:3] = 2
        base[3:5, 3:5] = 11
        out16 = mod._labels_to_bgr(base.astype(np.int16))
        out32 = mod._labels_to_bgr(base.astype(np.int32))
        out64 = mod._labels_to_bgr(base.astype(np.int64))
        self.assertTrue(np.array_equal(out16, out32))
        self.assertTrue(np.array_equal(out32, out64))

    def test_make_policy_comparison_panel_synthetic_smoke(self):
        mod = self._mod()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        gt_inst = np.zeros((32, 32), dtype=np.int32)
        gt_inst[4:12, 4:12] = 1
        gt_inst[18:28, 18:28] = 2
        pred_sem = np.zeros((32, 32), dtype=np.uint8)
        pred_sem[4:12, 4:12] = 1
        pred_sem[18:28, 18:28] = 1
        semantic_cc = np.zeros((32, 32), dtype=np.int32)
        semantic_cc[4:12, 4:12] = 1
        semantic_cc[18:28, 18:28] = 2
        labels = gt_inst.astype(np.uint8)
        metrics = {
            "counts": {"final_output_label_count": 2},
            "instance_metrics": {"matched_iou": 0.9},
            "area_accounting": {"assigned_area_fraction": 1.0},
        }
        policy_outputs = {
            "P0_CURRENT": {"labels": labels, "metrics": metrics},
            "P1_DROP_UNMARKED": {"labels": labels, "metrics": metrics},
            "P2_ATTACH_TO_NEAREST_MARKER": {"labels": labels, "metrics": metrics},
            "P3_GATED_ATTACH": {"labels": labels, "metrics": metrics},
            "P4_GLOBAL_MARKER_CONTROLLED": {"labels": labels, "metrics": metrics},
        }
        panel = mod._make_policy_comparison_panel(
            sample="synthetic_sample",
            image_rgb_u8=image,
            gt_inst=gt_inst,
            pred_sem=pred_sem,
            semantic_cc=semantic_cc,
            marker_points=[{"marker_id": 1, "y": 8, "x": 8}, {"marker_id": 2, "y": 22, "x": 22}],
            policy_outputs=policy_outputs,
            recommended_policy="P0_CURRENT",
        )
        self.assertEqual(panel.dtype, np.uint8)
        self.assertEqual(panel.ndim, 3)
        self.assertEqual(panel.shape[2], 3)
        self.assertGreater(panel.size, 0)

    def test_panel_invalid_shape_raises_controlled_visualization_error(self):
        mod = self._mod()
        labels = np.zeros((16, 16), dtype=np.int32)
        metrics = {
            "counts": {"final_output_label_count": 0},
            "instance_metrics": {"matched_iou": 0.0},
            "area_accounting": {"assigned_area_fraction": 0.0},
        }
        policy_outputs = {
            "P0_CURRENT": {"labels": labels, "metrics": metrics},
            "P1_DROP_UNMARKED": {"labels": labels, "metrics": metrics},
            "P2_ATTACH_TO_NEAREST_MARKER": {"labels": labels, "metrics": metrics},
            "P3_GATED_ATTACH": {"labels": labels, "metrics": metrics},
            "P4_GLOBAL_MARKER_CONTROLLED": {"labels": labels, "metrics": metrics},
        }
        with self.assertRaises(mod.VisualizationError) as ctx:
            mod._make_policy_comparison_panel(
                sample="bad_sample",
                image_rgb_u8=np.zeros((16, 16, 3), dtype=np.uint8),
                gt_inst=np.zeros((8, 8), dtype=np.int32),
                pred_sem=np.zeros((16, 16), dtype=np.uint8),
                semantic_cc=np.zeros((16, 16), dtype=np.int32),
                marker_points=[],
                policy_outputs=policy_outputs,
                recommended_policy="P0_CURRENT",
            )
        self.assertEqual(ctx.exception.payload["sample"], "bad_sample")

    def test_write_core_policy_artifacts_preserves_results_before_visual_failure(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            summary = {"recommended_policy": {"policy": "P1_DROP_UNMARKED"}}
            rows = [{"sample": "s1", "policy": "P0_CURRENT"}]
            per_component_assignments = {"samples": []}
            invariants = {"primary_threshold": 0.03}
            recommended = {"policy": "P1_DROP_UNMARKED"}
            mod._write_core_policy_artifacts(
                out_dir=out_dir,
                summary=summary,
                per_sample_csv_rows=rows,
                per_component_assignments=per_component_assignments,
                invariants=invariants,
                recommended=recommended,
            )
            payload = {"status": "visualization_error", "sample": "s1"}
            mod._write_json_atomic(out_dir / "visualization_error.json", payload)
            self.assertTrue((out_dir / "policy_summary.json").exists())
            self.assertTrue((out_dir / "per_sample_policy_metrics.csv").exists())
            self.assertTrue((out_dir / "per_component_assignments.json").exists())
            self.assertTrue((out_dir / "invariants.json").exists())
            self.assertTrue((out_dir / "recommended_policy.json").exists())
            self.assertTrue((out_dir / "visualization_error.json").exists())

    def test_policy_artifact_integrity_rejects_duplicate_sample_entries(self):
        mod = self._mod()
        rows = [
            {"sample_index": 0, "sample": "a", "policy": "P0_CURRENT"},
            {"sample_index": 0, "sample": "a", "policy": "P1_DROP_UNMARKED"},
            {"sample_index": 1, "sample": "b", "policy": "P0_CURRENT"},
            {"sample_index": 1, "sample": "b", "policy": "P1_DROP_UNMARKED"},
        ]
        samples = [
            {"sample": "a", "sample_index": 0},
            {"sample": "a", "sample_index": 0},
        ]
        with self.assertRaises(RuntimeError):
            mod._validate_policy_artifact_integrity(
                sample_entries=samples,
                per_sample_csv_rows=rows,
                thresholds=(0.03,),
                required_policies=("P0_CURRENT", "P1_DROP_UNMARKED"),
            )

    def test_policy_artifact_integrity_accepts_complete_five_policy_grid(self):
        mod = self._mod()
        samples = [{"sample": f"s{i}", "sample_index": i} for i in range(2)]
        rows = []
        for sample in samples:
            for threshold in (0.02, 0.03, 0.05):
                for policy in ("P0_CURRENT", "P1_DROP_UNMARKED", "P2_ATTACH_TO_NEAREST_MARKER", "P3_GATED_ATTACH", "P4_GLOBAL_MARKER_CONTROLLED"):
                    rows.append({"sample": sample["sample"], "sample_index": sample["sample_index"], "threshold": threshold, "policy": policy})
        mod._validate_policy_artifact_integrity(
            sample_entries=samples,
            per_sample_csv_rows=rows,
            thresholds=(0.03, 0.02, 0.05),
            required_policies=("P0_CURRENT", "P1_DROP_UNMARKED", "P2_ATTACH_TO_NEAREST_MARKER", "P3_GATED_ATTACH", "P4_GLOBAL_MARKER_CONTROLLED"),
        )

    def test_aggregate_policy_rows_splits_raw_and_final_provenance(self):
        mod = self._mod()
        rows = [
            {
                "counts": {"exact_count": True},
                "instance_metrics": {"matched_iou": 0.7, "fragmented": False, "merged": False},
                "area_accounting": {"assigned_area_fraction": 1.0, "dropped_area": 0},
                "contract": {
                    "pass": True,
                    "markers_preserved_count": 1,
                    "fallback_marker_calls": 2,
                    "keep_top3_call_count": 1,
                    "raw_labels_without_marker_provenance": 4,
                    "final_labels_without_marker_provenance": 0,
                },
                "component_assignment": {"ambiguous_assignments": None},
            }
        ]
        summary = mod._aggregate_policy_rows(rows)
        self.assertEqual(summary["raw_labels_without_marker_provenance"], 4)
        self.assertEqual(summary["final_labels_without_marker_provenance"], 0)

    def test_p0_component_assignment_semantics_can_be_unavailable(self):
        mod = self._mod()
        gt_inst = np.zeros((16, 16), dtype=np.uint8)
        gt_inst[2:8, 2:8] = 1
        pred_sem = np.zeros((16, 16), dtype=np.uint8)
        pred_sem[2:8, 2:8] = 1
        pred_inst = gt_inst.copy()
        marker_points = [_marker(1, 4, 4)]
        trace = {
            "semantic_component_count": 1,
            "raw_labels": pred_inst.copy(),
            "raw_count": 1,
            "final_labels": pred_inst.copy(),
            "component_assignments": [{"component_id": 1, "marker_count_before_fallback": 1}],
            "merged_markers": [],
            "fallback_marker_calls": 0,
            "keep_top3_call_count": 0,
            "new_non_marker_label_count": 0,
        }
        metrics = mod._policy_metrics(
            policy_name="P0_CURRENT",
            gt_inst=gt_inst,
            pred_sem=pred_sem,
            pred_inst=pred_inst,
            marker_points=marker_points,
            trace=trace,
        )
        self.assertIsNone(metrics["component_assignment"]["marked_components"])
        self.assertEqual(metrics["component_assignment"]["diagnostic_status"], "unavailable_for_p0")


if __name__ == "__main__":
    unittest.main()
