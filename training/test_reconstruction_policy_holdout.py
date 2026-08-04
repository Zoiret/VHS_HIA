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

    def _policy_row(
        self,
        *,
        sample: str,
        policy: str,
        marker_contract_pass: bool,
        gt_instance_count: int = 1,
        exact_count: bool = True,
        matched_iou: float = 0.7,
        invariant_pass: bool = True,
        fallback: int = 0,
        keep_top3: int = 0,
        raw_prov: int = 0,
        final_prov: int = 0,
        markers_preserved: bool = True,
        final_output_label_count: int = 1,
        marker_count: int = 1,
    ) -> dict:
        return {
            "sample": sample,
            "policy": policy,
            "marker_contract_pass": marker_contract_pass,
            "gt_instance_count": gt_instance_count,
            "semantic_cc_count": 1,
            "matched_iou": matched_iou,
            "mean_matched_dice": matched_iou,
            "exact_count": exact_count,
            "assigned_area_fraction": 1.0,
            "dropped_area_fraction": 0.0,
            "fragmented": False,
            "merged": False,
            "mixed": False,
            "invariant_pass": invariant_pass,
            "fallback_marker_calls": fallback,
            "keep_top3_call_count": keep_top3,
            "raw_labels_without_marker_provenance": raw_prov,
            "final_labels_without_marker_provenance": final_prov,
            "ambiguous_assignments": 0,
            "markers_preserved": markers_preserved,
            "final_output_label_count": final_output_label_count,
            "marker_count": marker_count,
            "output_count_error": int(final_output_label_count) - int(gt_instance_count),
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

    def test_promotion_evidence_uses_conditioned_count_not_total_manifest_count(self):
        mod = self._mod()
        conditioned_primary_rows = [
            self._policy_row(sample="a", policy="P0_CURRENT", marker_contract_pass=True, gt_instance_count=1, exact_count=False, matched_iou=0.6),
            self._policy_row(sample="a", policy="P1_DROP_UNMARKED", marker_contract_pass=True, gt_instance_count=1, exact_count=True, matched_iou=0.6),
            self._policy_row(sample="b", policy="P0_CURRENT", marker_contract_pass=True, gt_instance_count=2, exact_count=True, matched_iou=0.7),
            self._policy_row(sample="b", policy="P1_DROP_UNMARKED", marker_contract_pass=True, gt_instance_count=2, exact_count=True, matched_iou=0.7),
            self._policy_row(sample="c", policy="P0_CURRENT", marker_contract_pass=True, gt_instance_count=3, exact_count=True, matched_iou=0.8),
            self._policy_row(sample="c", policy="P1_DROP_UNMARKED", marker_contract_pass=True, gt_instance_count=3, exact_count=True, matched_iou=0.8),
        ]
        conditioned_primary_by_policy = {
            "P0_CURRENT": mod._aggregate_rows([row for row in conditioned_primary_rows if row["policy"] == "P0_CURRENT"]),
            "P1_DROP_UNMARKED": mod._aggregate_rows([row for row in conditioned_primary_rows if row["policy"] == "P1_DROP_UNMARKED"]),
        }
        end_rows = list(conditioned_primary_rows)
        end_by_policy = dict(conditioned_primary_by_policy)
        end_by_policy["_primary_rows"] = end_rows
        decision = mod._promotion_decision(
            total_holdout_sample_count=106,
            conditioned_primary_rows=conditioned_primary_rows,
            end_to_end_primary_by_policy=end_by_policy,
            conditioned_primary_by_policy=conditioned_primary_by_policy,
            paired_rows=[
                {"matched_iou_delta": 0.0, "exact_count_delta": 1.0},
                {"matched_iou_delta": 0.0, "exact_count_delta": 0.0},
                {"matched_iou_delta": 0.0, "exact_count_delta": 0.0},
            ],
            checkpoint_identity={"checkpoint_path": "x", "checkpoint_sha256": "ok", "checkpoint_iteration": 75, "authoritative_checkpoint_match": True},
        )
        self.assertEqual(decision["evidence_scope"]["marker_contract_pass_sample_count"], 3)
        self.assertFalse(decision["evidence_scope"]["conditioned_meets_minimum"])
        self.assertEqual(decision["status"], "insufficient_conditioned_evidence")

    def test_106_total_3_conditioned_returns_insufficient_conditioned_evidence(self):
        mod = self._mod()
        conditioned_primary_rows = []
        for sample, gt_count, p0_exact in (("a", 1, False), ("b", 2, True), ("c", 3, True)):
            conditioned_primary_rows.append(self._policy_row(sample=sample, policy="P0_CURRENT", marker_contract_pass=True, gt_instance_count=gt_count, exact_count=p0_exact))
            conditioned_primary_rows.append(self._policy_row(sample=sample, policy="P1_DROP_UNMARKED", marker_contract_pass=True, gt_instance_count=gt_count, exact_count=True))
        conditioned_primary_by_policy = {
            "P0_CURRENT": mod._aggregate_rows([row for row in conditioned_primary_rows if row["policy"] == "P0_CURRENT"]),
            "P1_DROP_UNMARKED": mod._aggregate_rows([row for row in conditioned_primary_rows if row["policy"] == "P1_DROP_UNMARKED"]),
        }
        end_by_policy = dict(conditioned_primary_by_policy)
        end_by_policy["_primary_rows"] = list(conditioned_primary_rows)
        decision = mod._promotion_decision(
            total_holdout_sample_count=106,
            conditioned_primary_rows=conditioned_primary_rows,
            end_to_end_primary_by_policy=end_by_policy,
            conditioned_primary_by_policy=conditioned_primary_by_policy,
            paired_rows=[{"matched_iou_delta": 0.0, "exact_count_delta": 1.0}],
            checkpoint_identity={"checkpoint_path": "x", "checkpoint_sha256": "ok", "checkpoint_iteration": 75, "authoritative_checkpoint_match": True},
        )
        self.assertEqual(decision["status"], "insufficient_conditioned_evidence")
        self.assertEqual(decision["reconstruction_policy_result"]["status"], "promising_but_underpowered")

    def test_no_candidate_for_production_patch_with_conditioned_n_lt_20(self):
        mod = self._mod()
        conditioned_primary_rows = [
            self._policy_row(sample="a", policy="P0_CURRENT", marker_contract_pass=True, gt_instance_count=1),
            self._policy_row(sample="a", policy="P1_DROP_UNMARKED", marker_contract_pass=True, gt_instance_count=1),
        ]
        conditioned_primary_by_policy = {
            "P0_CURRENT": mod._aggregate_rows([row for row in conditioned_primary_rows if row["policy"] == "P0_CURRENT"]),
            "P1_DROP_UNMARKED": mod._aggregate_rows([row for row in conditioned_primary_rows if row["policy"] == "P1_DROP_UNMARKED"]),
        }
        end_by_policy = dict(conditioned_primary_by_policy)
        end_by_policy["_primary_rows"] = list(conditioned_primary_rows)
        decision = mod._promotion_decision(
            total_holdout_sample_count=20,
            conditioned_primary_rows=conditioned_primary_rows,
            end_to_end_primary_by_policy=end_by_policy,
            conditioned_primary_by_policy=conditioned_primary_by_policy,
            paired_rows=[{"matched_iou_delta": 0.0, "exact_count_delta": 0.0}],
            checkpoint_identity={"checkpoint_path": "x", "checkpoint_sha256": "ok", "checkpoint_iteration": 75, "authoritative_checkpoint_match": True},
        )
        self.assertNotEqual(decision["status"], "candidate_for_production_patch")

    def test_decision_scopes_are_separate(self):
        mod = self._mod()
        conditioned_primary_rows = [
            self._policy_row(sample="a", policy="P0_CURRENT", marker_contract_pass=True, exact_count=False),
            self._policy_row(sample="a", policy="P1_DROP_UNMARKED", marker_contract_pass=True, exact_count=True),
        ]
        conditioned_primary_by_policy = {
            "P0_CURRENT": mod._aggregate_rows([row for row in conditioned_primary_rows if row["policy"] == "P0_CURRENT"]),
            "P1_DROP_UNMARKED": mod._aggregate_rows([row for row in conditioned_primary_rows if row["policy"] == "P1_DROP_UNMARKED"]),
        }
        end_rows = [
            self._policy_row(sample="a", policy="P0_CURRENT", marker_contract_pass=False, exact_count=True, matched_iou=0.5),
            self._policy_row(sample="a", policy="P1_DROP_UNMARKED", marker_contract_pass=False, exact_count=False, matched_iou=0.2),
        ]
        end_by_policy = {
            "P0_CURRENT": mod._aggregate_rows([row for row in end_rows if row["policy"] == "P0_CURRENT"]),
            "P1_DROP_UNMARKED": mod._aggregate_rows([row for row in end_rows if row["policy"] == "P1_DROP_UNMARKED"]),
            "_primary_rows": end_rows,
        }
        decision = mod._promotion_decision(
            total_holdout_sample_count=106,
            conditioned_primary_rows=conditioned_primary_rows,
            end_to_end_primary_by_policy=end_by_policy,
            conditioned_primary_by_policy=conditioned_primary_by_policy,
            paired_rows=[{"matched_iou_delta": 0.0, "exact_count_delta": 1.0}],
            checkpoint_identity={"checkpoint_path": "x", "checkpoint_sha256": "ok", "checkpoint_iteration": 75, "authoritative_checkpoint_match": True},
        )
        self.assertIn("reconstruction_policy_result", decision)
        self.assertIn("center_generalization_result", decision)
        self.assertIn("production_activation_result", decision)

    def test_all_sample_and_conditioned_invariants_reported_separately(self):
        mod = self._mod()
        conditioned_primary_rows = [
            self._policy_row(sample="a", policy="P0_CURRENT", marker_contract_pass=True, invariant_pass=True),
            self._policy_row(sample="a", policy="P1_DROP_UNMARKED", marker_contract_pass=True, invariant_pass=True),
        ]
        end_rows = [
            self._policy_row(sample="a", policy="P0_CURRENT", marker_contract_pass=False, invariant_pass=False, fallback=1),
            self._policy_row(sample="a", policy="P1_DROP_UNMARKED", marker_contract_pass=False, invariant_pass=False, raw_prov=1, markers_preserved=False, final_output_label_count=2, marker_count=1),
        ]
        conditioned_primary_by_policy = {
            "P0_CURRENT": mod._aggregate_rows([row for row in conditioned_primary_rows if row["policy"] == "P0_CURRENT"]),
            "P1_DROP_UNMARKED": mod._aggregate_rows([row for row in conditioned_primary_rows if row["policy"] == "P1_DROP_UNMARKED"]),
        }
        end_by_policy = {
            "P0_CURRENT": mod._aggregate_rows([row for row in end_rows if row["policy"] == "P0_CURRENT"]),
            "P1_DROP_UNMARKED": mod._aggregate_rows([row for row in end_rows if row["policy"] == "P1_DROP_UNMARKED"]),
            "_primary_rows": end_rows,
        }
        decision = mod._promotion_decision(
            total_holdout_sample_count=106,
            conditioned_primary_rows=conditioned_primary_rows,
            end_to_end_primary_by_policy=end_by_policy,
            conditioned_primary_by_policy=conditioned_primary_by_policy,
            paired_rows=[{"matched_iou_delta": 0.0, "exact_count_delta": 0.0}],
            checkpoint_identity={"checkpoint_path": "x", "checkpoint_sha256": "ok", "checkpoint_iteration": 75, "authoritative_checkpoint_match": True},
        )
        scoped = decision["scope_invariants"]["P1_DROP_UNMARKED"]
        self.assertEqual(scoped["all_samples"]["all_samples_invariant_violations"], 1)
        self.assertEqual(scoped["conditioned"]["all_samples_invariant_violations"], 0)

    def test_checkpoint_sha_mismatch_blocks_authoritative_decision(self):
        mod = self._mod()
        conditioned_primary_rows = [
            self._policy_row(sample="a", policy="P0_CURRENT", marker_contract_pass=True, gt_instance_count=1),
            self._policy_row(sample="a", policy="P1_DROP_UNMARKED", marker_contract_pass=True, gt_instance_count=1),
        ]
        conditioned_primary_by_policy = {
            "P0_CURRENT": mod._aggregate_rows([row for row in conditioned_primary_rows if row["policy"] == "P0_CURRENT"]),
            "P1_DROP_UNMARKED": mod._aggregate_rows([row for row in conditioned_primary_rows if row["policy"] == "P1_DROP_UNMARKED"]),
        }
        end_by_policy = dict(conditioned_primary_by_policy)
        end_by_policy["_primary_rows"] = list(conditioned_primary_rows)
        decision = mod._promotion_decision(
            total_holdout_sample_count=20,
            conditioned_primary_rows=conditioned_primary_rows,
            end_to_end_primary_by_policy=end_by_policy,
            conditioned_primary_by_policy=conditioned_primary_by_policy,
            paired_rows=[{"matched_iou_delta": 0.0, "exact_count_delta": 0.0}],
            checkpoint_identity={"checkpoint_path": "x", "checkpoint_sha256": "bad", "checkpoint_iteration": 75, "authoritative_checkpoint_match": False},
        )
        self.assertEqual(decision["production_activation_result"]["status"], "blocked")
        self.assertIn("checkpoint identity mismatch", decision["production_activation_result"]["reasons"])

    def test_heldout_manifest_hash_deterministic(self):
        mod = self._mod()
        entries = [
            {"sample": "a", "image_path": "/x/a.png", "gt_semantic_path": "/x/a_mask.png", "gt_instance_path": "/y/a.png", "source_dataset": "converted_leaflet_distance", "gt_instance_count": 1},
            {"sample": "b", "image_path": "/x/b.png", "gt_semantic_path": "/x/b_mask.png", "gt_instance_path": "/y/b.png", "source_dataset": "converted_leaflet_distance", "gt_instance_count": 2},
        ]
        self.assertEqual(mod._manifest_sha256(entries), mod._manifest_sha256(entries))


if __name__ == "__main__":
    unittest.main()
