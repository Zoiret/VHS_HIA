from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent


class TestReconstructionPolicyHoldout(unittest.TestCase):
    @staticmethod
    def _mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import validate_reconstruction_policies_holdout as mod

        return mod

    @staticmethod
    def _compare_mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import compare_reconstruction_policies as mod

        return mod

    def _policy_metrics_fixture(self, *, raw_no_marker: int = 0, final_no_marker: int = 0, fallback: int = 0, keep_top3: int = 0, diagnostic_status: str = "ok") -> dict:
        return {
            "counts": {
                "gt_instance_count": 2,
                "marker_count": 2,
                "semantic_connected_component_count": 4,
                "raw_output_label_count": 2,
                "final_output_label_count": 2,
                "exact_count": True,
            },
            "instance_metrics": {
                "matched_iou": 0.75,
                "mean_matched_dice": 0.80,
                "merged": False,
                "fragmented": False,
                "mixed": False,
                "perfect_recovery": True,
                "instance_score": 0.75,
            },
            "area_accounting": {
                "assigned_area_fraction": 0.9,
                "dropped_area": 10,
                "semantic_leaflet_area": 100,
                "unmarked_component_area": None if diagnostic_status != "ok" else 0,
            },
            "component_assignment": {
                "ambiguous_assignments": 0 if diagnostic_status == "ok" else None,
                "marked_components": 2 if diagnostic_status == "ok" else None,
                "unmarked_components": 0 if diagnostic_status == "ok" else None,
                "unmarked_components_rejected": 0 if diagnostic_status == "ok" else None,
                "diagnostic_status": diagnostic_status,
            },
            "contract": {
                "pass": True,
                "marker_count_preservation": True,
                "markers_without_output_label": [],
                "raw_labels_without_marker_provenance": raw_no_marker,
                "final_labels_without_marker_provenance": final_no_marker,
                "fallback_marker_calls": fallback,
                "keep_top3_call_count": keep_top3,
            },
        }

    def test_microset_ids_excluded(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            ds = repo / "datasets/converted_leaflet_distance"
            inst = repo / "datasets/converted_leaflet_instances/instance_masks"
            for rel in ["images", "semantic_masks", "center_maps", "metadata"]:
                (ds / rel).mkdir(parents=True, exist_ok=True)
            inst.mkdir(parents=True, exist_ok=True)
            val = ds / "val.txt"
            test = ds / "test.txt"
            val.write_text("images/m01_p02_s00.png\tsemantic_masks/m01_p02_s00.png\nimages/m99_p01_s01.png\tsemantic_masks/m99_p01_s01.png\n", encoding="utf-8")
            test.write_text("", encoding="utf-8")
            for sid in ("m01_p02_s00", "m99_p01_s01"):
                for rel in ["images", "semantic_masks", "center_maps"]:
                    (ds / rel / f"{sid}.png").write_bytes(b"x")
                (ds / "metadata" / f"{sid}.json").write_text(json.dumps({"instance_count": 1}), encoding="utf-8")
                (inst / f"{sid}.png").write_bytes(b"x")
            cfg = {
                "dataset": {
                    "root": "datasets/converted_leaflet_distance",
                    "instance_root": "datasets/converted_leaflet_instances",
                    "val_txt": "datasets/converted_leaflet_distance/val.txt",
                    "test_txt": "datasets/converted_leaflet_distance/test.txt",
                }
            }
            inv = mod._inventory_holdout_samples(cfg, repo)
            self.assertEqual([e["sample"] for e in inv["eligible"]], ["m99_p01_s01"])
            self.assertEqual([e["sample"] for e in inv["excluded_microset"]], ["m01_p02_s00"])

    def test_one_sample_only_once_in_component_artifact(self):
        compare = self._compare_mod()
        with self.assertRaises(RuntimeError):
            compare._validate_policy_artifact_integrity(
                sample_entries=[{"sample": "a", "sample_index": 0}, {"sample": "a", "sample_index": 0}],
                per_sample_csv_rows=[],
                thresholds=(0.03,),
                required_policies=("P0_CURRENT",),
            )

    def test_all_five_policies_present_in_csv(self):
        compare = self._compare_mod()
        rows = []
        samples = [{"sample": "s0", "sample_index": 0}]
        for threshold in (0.02, 0.03, 0.05):
            for policy in self._mod().REQUIRED_POLICIES:
                rows.append({"sample": "s0", "sample_index": 0, "threshold": threshold, "policy": policy})
        compare._validate_policy_artifact_integrity(
            sample_entries=samples,
            per_sample_csv_rows=rows,
            thresholds=(0.03, 0.02, 0.05),
            required_policies=self._mod().REQUIRED_POLICIES,
        )

    def test_expected_csv_row_count_formula(self):
        compare = self._compare_mod()
        rows = []
        samples = [{"sample": "s0", "sample_index": 0}, {"sample": "s1", "sample_index": 1}]
        for sample in samples:
            for threshold in (0.02, 0.03, 0.05):
                for policy in self._mod().REQUIRED_POLICIES:
                    rows.append({"sample": sample["sample"], "sample_index": sample["sample_index"], "threshold": threshold, "policy": policy})
        compare._validate_policy_artifact_integrity(
            sample_entries=samples,
            per_sample_csv_rows=rows,
            thresholds=(0.03, 0.02, 0.05),
            required_policies=self._mod().REQUIRED_POLICIES,
        )
        self.assertEqual(len(rows), 2 * 3 * 5)

    def test_locked_threshold_remains_0p03(self):
        mod = self._mod()
        self.assertEqual(mod.PRIMARY_THRESHOLD, 0.03)

    def test_checkpoint_saved_threshold_cannot_override_primary_threshold(self):
        mod = self._mod()
        saved = 0.05
        self.assertNotEqual(saved, mod.PRIMARY_THRESHOLD)
        self.assertEqual(mod.PRIMARY_THRESHOLD, 0.03)

    def test_marker_contract_conditioned_subset_correct(self):
        mod = self._mod()
        rows = [
            {"sample": "a", "marker_contract_pass": True},
            {"sample": "b", "marker_contract_pass": False},
            {"sample": "c", "marker_contract_pass": True},
        ]
        conditioned = mod._condition_rows_for_marker_contract(rows)
        self.assertEqual([r["sample"] for r in conditioned], ["a", "c"])

    def test_p1_output_count_le_marker_count(self):
        mod = self._mod()
        entry = {"sample": "s", "split": "val", "sample_index": 0, "mouse_id": "m01"}
        marker_contract = {"marker_contract_pass": True, "markers_outside_all_gt_instances": 0, "one_marker_per_instance_rate": 1.0}
        center_metrics = {"center_precision": 1.0, "center_recall": 1.0, "center_f1": 1.0}
        metrics = self._policy_metrics_fixture()
        row = mod._row_from_metrics(sample_entry=entry, threshold=0.03, policy="P1_DROP_UNMARKED", marker_contract=marker_contract, center_metrics=center_metrics, policy_metrics=metrics, p3_cfg=None)
        self.assertLessEqual(int(row["final_output_label_count"]), int(row["marker_count"]))

    def test_p1_has_no_fallback(self):
        mod = self._mod()
        entry = {"sample": "s", "split": "val", "sample_index": 0, "mouse_id": "m01"}
        marker_contract = {"marker_contract_pass": True, "markers_outside_all_gt_instances": 0, "one_marker_per_instance_rate": 1.0}
        center_metrics = {"center_precision": 1.0, "center_recall": 1.0, "center_f1": 1.0}
        row = mod._row_from_metrics(sample_entry=entry, threshold=0.03, policy="P1_DROP_UNMARKED", marker_contract=marker_contract, center_metrics=center_metrics, policy_metrics=self._policy_metrics_fixture(fallback=0), p3_cfg=None)
        self.assertEqual(row["fallback_marker_calls"], 0)

    def test_p1_has_no_keep_top3(self):
        mod = self._mod()
        entry = {"sample": "s", "split": "val", "sample_index": 0, "mouse_id": "m01"}
        marker_contract = {"marker_contract_pass": True, "markers_outside_all_gt_instances": 0, "one_marker_per_instance_rate": 1.0}
        center_metrics = {"center_precision": 1.0, "center_recall": 1.0, "center_f1": 1.0}
        row = mod._row_from_metrics(sample_entry=entry, threshold=0.03, policy="P1_DROP_UNMARKED", marker_contract=marker_contract, center_metrics=center_metrics, policy_metrics=self._policy_metrics_fixture(keep_top3=0), p3_cfg=None)
        self.assertEqual(row["keep_top3_call_count"], 0)

    def test_raw_final_provenance_fields_separated(self):
        mod = self._mod()
        entry = {"sample": "s", "split": "val", "sample_index": 0, "mouse_id": "m01"}
        marker_contract = {"marker_contract_pass": True, "markers_outside_all_gt_instances": 0, "one_marker_per_instance_rate": 1.0}
        center_metrics = {"center_precision": 1.0, "center_recall": 1.0, "center_f1": 1.0}
        row = mod._row_from_metrics(sample_entry=entry, threshold=0.03, policy="P0_CURRENT", marker_contract=marker_contract, center_metrics=center_metrics, policy_metrics=self._policy_metrics_fixture(raw_no_marker=5, final_no_marker=0), p3_cfg=None)
        self.assertEqual(row["raw_labels_without_marker_provenance"], 5)
        self.assertEqual(row["final_labels_without_marker_provenance"], 0)

    def test_p0_unavailable_component_metrics_represented_as_null(self):
        mod = self._mod()
        entry = {"sample": "s", "split": "val", "sample_index": 0, "mouse_id": "m01"}
        marker_contract = {"marker_contract_pass": True, "markers_outside_all_gt_instances": 0, "one_marker_per_instance_rate": 1.0}
        center_metrics = {"center_precision": 1.0, "center_recall": 1.0, "center_f1": 1.0}
        row = mod._row_from_metrics(sample_entry=entry, threshold=0.03, policy="P0_CURRENT", marker_contract=marker_contract, center_metrics=center_metrics, policy_metrics=self._policy_metrics_fixture(diagnostic_status="unavailable_for_p0"), p3_cfg=None)
        self.assertIsNone(row["marked_components"])
        self.assertEqual(row["component_diagnostic_status"], "unavailable_for_p0")

    def test_bootstrap_deterministic(self):
        mod = self._mod()
        a = mod._bootstrap_mean_ci([0.1, -0.2, 0.3], seed=0, n_bootstrap=1000)
        b = mod._bootstrap_mean_ci([0.1, -0.2, 0.3], seed=0, n_bootstrap=1000)
        self.assertEqual(a, b)

    def test_insufficient_dataset_blocks_production_recommendation(self):
        mod = self._mod()
        conditioned = {
            "P0_CURRENT": {"exact_count_accuracy": 0.5},
            "P1_DROP_UNMARKED": {
                "invariant_violation_count": 0,
                "fallback_marker_calls": 0,
                "keep_top3_calls": 0,
                "final_labels_without_marker_provenance": 0,
                "raw_labels_without_marker_provenance": 0,
                "exact_count_accuracy": 1.0,
                "markers_preserved_rate": 1.0,
            },
        }
        decision = mod._promotion_decision(conditioned_primary_by_policy=conditioned, paired_rows=[{"matched_iou_delta": -0.001, "exact_count_delta": 1.0}], evidence_sufficient=False)
        self.assertEqual(decision["status"], "insufficient_evidence")

    def test_heldout_manifest_hash_deterministic(self):
        mod = self._mod()
        entries = [
            {"sample": "a", "image_path": "/x/a.png", "gt_semantic_path": "/x/a_mask.png", "gt_instance_path": "/y/a.png", "source_dataset": "converted_leaflet_distance", "gt_instance_count": 1},
            {"sample": "b", "image_path": "/x/b.png", "gt_semantic_path": "/x/b_mask.png", "gt_instance_path": "/y/b.png", "source_dataset": "converted_leaflet_distance", "gt_instance_count": 2},
        ]
        self.assertEqual(mod._manifest_sha256(entries), mod._manifest_sha256(entries))


if __name__ == "__main__":
    unittest.main()
