from __future__ import annotations

import sys
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bridge_presence_gate_v4_scalar_baseline_provenance_audit as audit


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
                "scalar_source_function": "fn",
                "relevant_dtype": "float64",
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
                "scalar_source_function": "fn",
                "relevant_dtype": "float64",
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
                "scalar_source_function": "fn",
                "relevant_dtype": "float64",
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
                "scalar_source_function": "fn",
                "relevant_dtype": "float64",
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
                "scalar_source_function": "fn",
                "relevant_dtype": "float64",
            }
        ]
        current = [dict(reference[0], candidate_pixels=12, candidate_fraction="0.12")]
        diff = audit.diff_forensic_row_sets(reference, current)
        self.assertEqual(diff["candidate_fraction_differences"][0]["sample_id"], "a")
        self.assertAlmostEqual(diff["candidate_fraction_differences"][0]["absolute_difference"], 0.02)


if __name__ == "__main__":
    unittest.main()
