from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bridge_presence_gate_v4_scalar_baseline_provenance_audit as audit
import bridge_presence_gate_v4_patient_disjoint_dev as dev


class TestBridgePresenceGateV4ScalarBaselineProvenanceAudit(unittest.TestCase):
    def test_canonical_row_hash_is_order_sensitive_to_row_content_but_not_builder_order(self):
        rows_a = [
            {
                "sample_id": "a",
                "patient_id": "p1",
                "gt_count": 1,
                "bridge_target": 0,
                "candidate_pixels": 10,
                "candidate_fraction": "0.125",
                "candidate_fraction_denominator": 80,
                "candidate_mask_shape": "8x10",
                "scalar_source": "fn",
                "dtype": "float64",
            },
            {
                "sample_id": "b",
                "patient_id": "p2",
                "gt_count": 2,
                "bridge_target": 1,
                "candidate_pixels": 20,
                "candidate_fraction": "0.25",
                "candidate_fraction_denominator": 80,
                "candidate_mask_shape": "8x10",
                "scalar_source": "fn",
                "dtype": "float64",
            },
        ]
        rows_b = [dict(rows_a[0]), dict(rows_a[1])]
        self.assertEqual(audit.canonical_row_sha256(rows_a), audit.canonical_row_sha256(rows_b))

    def test_build_forensic_selector_rows_captures_requested_fields(self):
        records = [
            {
                "sample_id": "a",
                "patient_id": "p1",
                "gt_count": 2,
                "candidate_pixels": 12,
                "candidate_mask_np": [[1, 0, 0], [1, 1, 0]],
            }
        ]
        feature_rows = [
            {
                "sample_id": "a",
                "bridge_positive_target": 1,
                "candidate_fraction": 0.5,
            }
        ]
        rows = audit.build_forensic_selector_rows(
            records=records,
            feature_rows=feature_rows,
            scalar_source_function="fn",
            candidate_fraction_source="fraction",
            target_source="target",
        )
        self.assertEqual(rows[0]["sample_id"], "a")
        self.assertEqual(rows[0]["candidate_fraction_denominator"], 6)
        self.assertEqual(rows[0]["candidate_mask_shape"], "2x3")
        self.assertEqual(rows[0]["bridge_target"], 1)

    def test_diff_forensic_row_sets_reports_subset_diff(self):
        reference = [
            {
                "sample_id": "a",
                "patient_id": "p1",
                "gt_count": 1,
                "bridge_target": 0,
                "candidate_pixels": 10,
                "candidate_fraction": "0.1",
                "candidate_fraction_denominator": 100,
                "candidate_mask_shape": "10x10",
                "scalar_source": "fn",
                "dtype": "float64",
            }
        ]
        current = []
        diff = audit.diff_forensic_row_sets(reference, current)
        self.assertEqual(diff["missing_sample_ids"], ["a"])
        self.assertEqual(diff["extra_sample_ids"], [])

    def test_diff_forensic_row_sets_reports_target_diff(self):
        reference = [
            {
                "sample_id": "a",
                "patient_id": "p1",
                "gt_count": 2,
                "bridge_target": 0,
                "candidate_pixels": 10,
                "candidate_fraction": "0.1",
                "candidate_fraction_denominator": 100,
                "candidate_mask_shape": "10x10",
                "scalar_source": "fn",
                "dtype": "float64",
            }
        ]
        current = [dict(reference[0], bridge_target=1)]
        diff = audit.diff_forensic_row_sets(reference, current)
        self.assertEqual(diff["target_differences"][0]["sample_id"], "a")
        self.assertEqual(diff["target_differences"][0]["reference_target"], 0)
        self.assertEqual(diff["target_differences"][0]["current_target"], 1)

    def test_diff_forensic_row_sets_reports_candidate_fraction_diff(self):
        reference = [
            {
                "sample_id": "a",
                "patient_id": "p1",
                "gt_count": 3,
                "bridge_target": 1,
                "candidate_pixels": 10,
                "candidate_fraction": "0.1",
                "candidate_fraction_denominator": 100,
                "candidate_mask_shape": "10x10",
                "scalar_source": "fn",
                "dtype": "float64",
            }
        ]
        current = [dict(reference[0], candidate_pixels=12, candidate_fraction="0.12")]
        diff = audit.diff_forensic_row_sets(reference, current)
        self.assertEqual(diff["candidate_fraction_differences"][0]["sample_id"], "a")
        self.assertAlmostEqual(diff["candidate_fraction_differences"][0]["absolute_difference"], 0.02)

    def test_selector_result_payload_records_mismatch_without_raising(self):
        feature_rows = [
            {
                "sample_id": "a",
                "bridge_positive_target": 1,
                "candidate_fraction": 0.1544308066368103,
            },
            {
                "sample_id": "b",
                "bridge_positive_target": 0,
                "candidate_fraction": 0.0461595430970192,
            },
        ]
        for row in feature_rows:
            row.setdefault("bridge_score_mean", 0.0)
            row.setdefault("bridge_score_max", 0.0)
            row.setdefault("bridge_score_top1pct_mean", 0.0)
            row.setdefault("bridge_score_top5pct_mean", 0.0)
            row.setdefault("bridge_score_frac_ge_0p50", 0.0)
            row.setdefault("bridge_score_frac_ge_0p75", 0.0)
            row.setdefault("bridge_score_frac_ge_0p90", 0.0)
            row.setdefault("candidate_component_count", 1.0)
        out = audit._selector_result_payload(feature_rows, selector_function="historical")
        self.assertFalse(out["matches_frozen_threshold"])
        self.assertFalse(out["frozen_rule_reproduced"])
        self.assertAlmostEqual(out["expected_frozen_threshold"], dev.FROZEN_SIMPLE_SCALAR_RULE["threshold"])

    def test_audit_writes_artifacts_and_returns_success_on_threshold_mismatch(self):
        historical_rows = [
            {
                "sample_id": "a",
                "patient_id": "p1",
                "gt_count": 1,
                "bridge_target": 0,
                "candidate_pixels": 10,
                "candidate_fraction": "0.1",
                "candidate_fraction_denominator": 100,
                "candidate_mask_shape": "10x10",
                "scalar_source": "historical",
                "dtype": "float64",
            }
        ]
        current_rows = [dict(historical_rows[0], scalar_source="current")]
        package = {
            "rows": historical_rows,
            "summary": audit.summarize_forensic_rows(historical_rows),
            "selector_result": {
                "scalar": "candidate_fraction",
                "direction": "ge",
                "threshold": 0.10029517486691475,
                "tp": 1,
                "tn": 1,
                "fp": 0,
                "fn": 0,
                "sensitivity": 1.0,
                "specificity": 1.0,
                "balanced_accuracy": 1.0,
                "selector_function": "historical",
                "matches_frozen_threshold": False,
                "frozen_rule_reproduced": False,
                "expected_frozen_threshold": dev.FROZEN_SIMPLE_SCALAR_RULE["threshold"],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            fake_cfg = {"analysis": {"scalar_baseline_provenance_audit_dir": str(Path(td))}, "seed": 1337}
            manifest_stage = {
                "manifest": {"contract": {"train_sample_ids": ["a"]}},
                "device": mock.Mock(type="cpu"),
            }
            with mock.patch.object(audit.preflight_runner, "_prepare_manifest", return_value=manifest_stage), \
                 mock.patch.object(audit.gate_v4, "load_frozen_v2_pixel_model_from_cfg", return_value=(mock.Mock(), {"checkpoint_file_sha256": "a"})), \
                 mock.patch.object(audit, "_direct_prepare_rows", return_value={"cached_records": [], "feature_rows": []}), \
                 mock.patch.object(audit.dev, "_prepare_split_preflight_core", return_value={"cached_records": [], "feature_rows": []}), \
                 mock.patch.object(audit.train_runner, "_prepare_training_inputs_core", return_value={"train_prepared": {"cached_records": [], "feature_rows": []}}), \
                 mock.patch.object(audit, "build_forensic_selector_rows", side_effect=[historical_rows, current_rows, current_rows]), \
                 mock.patch.object(audit, "_package_path_audit", side_effect=[package, dict(package, rows=current_rows, summary=audit.summarize_forensic_rows(current_rows)), dict(package, rows=current_rows, summary=audit.summarize_forensic_rows(current_rows))]), \
                 mock.patch.object(audit, "_selector_result_payload", return_value=package["selector_result"]), \
                 mock.patch.object(audit, "find_original_artifacts", return_value=[]):
                result = audit.run_pipeline(fake_cfg)
            self.assertTrue(result["audit_execution_success"])
            self.assertTrue((Path(td) / "historical_preflight_rows.csv").exists())
            self.assertTrue((Path(td) / "current_preflight_rows.csv").exists())
            self.assertTrue((Path(td) / "current_training_rows.csv").exists())
            self.assertTrue((Path(td) / "historical_vs_current_preflight_diff.json").exists())
            self.assertTrue((Path(td) / "selector_results.json").exists())
            self.assertTrue((Path(td) / "provenance_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
