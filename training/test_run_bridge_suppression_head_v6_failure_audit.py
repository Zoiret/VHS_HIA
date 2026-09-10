from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bridge_suppression_head as bridge
import run_bridge_suppression_head_v6_failure_audit as runner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestRunBridgeSuppressionHeadV6FailureAudit(unittest.TestCase):
    def _cached_record(self, sample_id: str, patient_id: str, bridge_positive: int) -> dict[str, object]:
        candidate_mask_np = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        bridge_target_np = np.array([[0, 0], [0, 0]], dtype=np.uint8)
        if int(bridge_positive) == 1:
            bridge_target_np = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        return {
            "sample_id": sample_id,
            "patient_id": patient_id,
            "gt_count": 1,
            "bridge_positive": int(bridge_positive),
            "candidate_mask_np": candidate_mask_np,
            "bridge_target": torch.from_numpy(bridge_target_np[None, ...].astype(np.float32)),
            "gt_instances": np.array([[1, 1], [0, 0]], dtype=np.uint8),
            "component_count_start": 1,
            "start_reconstruction": {
                "pred_k": 1,
                "metrics": {
                    "all_iou_ge_0.50": True,
                    "instance_mean_matched_iou": 0.8,
                },
            },
        }

    def test_authoritative_topology_reference_constants_match_expected_values(self):
        self.assertEqual(runner.audit.AUTHORITATIVE_TOPOLOGY_REFERENCES["v6_train_epoch63"], 62)
        self.assertEqual(runner.audit.AUTHORITATIVE_TOPOLOGY_REFERENCES["v6_val"], 21)
        self.assertEqual(runner.audit.AUTHORITATIVE_TOPOLOGY_REFERENCES["historical_v2_val"], 24)

    def test_runner_is_read_only_and_never_uses_test_holdout_or_threshold_sweep(self):
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_patient_disjoint_v6_dev.yaml")
        train_record = self._cached_record("train_1", "p1", 1)
        val_record = self._cached_record("val_1", "p2", 0)
        train_eval = {
            "reconstruction": {
                "positive_success50": 2,
                "positive_mean_matched_iou": 0.6,
                "negative_regressions": 0,
                "negative_topology_changes": 0,
            },
            "per_sample": [
                {
                    "sample_id": "train_1",
                    "component_topology_changed": 0,
                    "predicted_success50": 1,
                    "predicted_mean_iou": 0.6,
                }
            ],
        }
        val_eval = {
            "reconstruction": {
                "positive_success50": 5,
                "positive_mean_matched_iou": 0.6395,
                "negative_regressions": 19,
                "negative_topology_changes": 21,
            },
            "per_sample": [
                {
                    "sample_id": "val_1",
                    "component_topology_changed": 1,
                    "predicted_success50": 0,
                    "predicted_mean_iou": 0.2,
                }
            ],
        }
        compare_payload = {
            "historical_v2_open": {
                "true_bridge_precision": 0.5,
                "true_bridge_recall": 0.5,
                "true_bridge_f1": 0.5,
                "off_target_inside_gt_removal": 2,
                "off_target_outside_gt_removal": 3,
                "positive_success50": 1,
                "positive_total": 1,
                "negative_regressions": 1,
                "negative_regression_rate": 1.0,
                "negative_topology_changes": 1,
                "negative_topology_change_rate": 1.0,
            },
            "v6_open": {
                "true_bridge_precision": 0.4,
                "true_bridge_recall": 0.9,
                "true_bridge_f1": 0.55,
                "off_target_inside_gt_removal": 4,
                "off_target_outside_gt_removal": 4,
                "positive_success50": 1,
                "positive_total": 1,
                "negative_regressions": 1,
                "negative_regression_rate": 1.0,
                "negative_topology_changes": 1,
                "negative_topology_change_rate": 1.0,
            },
        }
        exact_oracle = {
            "overall": {"n": 1, "mean_matched_iou": 0.8, "all_iou_ge_0.50_count": 1, "all_iou_ge_0.50_rate": 1.0, "gt2_success": "0/0", "gt3_success": "0/0"},
            "positive": {"n": 0, "mean_matched_iou": 0.0, "all_iou_ge_0.50_count": 0, "all_iou_ge_0.50_rate": 0.0, "gt2_success": "0/0", "gt3_success": "0/0"},
            "overall_success50": 1,
            "overall_mean_matched_iou": 0.8,
            "positive_success50": 0,
            "positive_mean_matched_iou": 0.0,
            "negative_regressions": 0,
            "negative_topology_changes": 0,
            "zero_target_identical_to_closed": True,
        }
        pareto = {
            "all_epochs": [],
            "pareto_frontier": [],
            "any_positive_improved_epoch": True,
            "any_safe_negative_epoch": False,
            "safe_useful_epoch_exists": False,
            "interpretation": "destructive_removal_present_across_the_useful_region_of_the_training_trajectory",
        }
        score_summary = {
            "TRUE_BRIDGE": {"pixel_count": 1, "mean": 0.9, "median": 0.9, "p75": 0.9, "p90": 0.9, "p95": 0.9, "p99": 0.9, "fraction_ge_0p50": 1.0},
            "POSITIVE_SAMPLE_NON_BRIDGE": {"pixel_count": 1, "mean": 0.4, "median": 0.4, "p75": 0.4, "p90": 0.4, "p95": 0.4, "p99": 0.4, "fraction_ge_0p50": 0.0},
            "ZERO_TARGET_SAMPLE": {"pixel_count": 1, "mean": 0.8, "median": 0.8, "p75": 0.8, "p90": 0.8, "p95": 0.8, "p99": 0.8, "fraction_ge_0p50": 1.0},
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            run_dir = td_path / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = run_dir / "best_train_reconstruction.pth"
            torch.save({"model": {}, "step": 63, "extra": {}}, checkpoint_path)
            (run_dir / "train_history.json").write_text(json.dumps([{"epoch": 63, "train_positive_success50": 5, "train_positive_mean_matched_iou": 0.63, "train_negative_regressions": 19, "train_negative_topology_changes": 21}]), encoding="utf-8")
            before_sha = _sha256(checkpoint_path)
            cfg = dict(cfg)
            cfg["analysis"] = {"failure_audit_dir": str(td_path / "analysis")}
            mine_mock = mock.Mock(side_effect=[[train_record], [val_record]])
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(runner.split_runner, "_prepare_manifest", return_value={"device": torch.device("cpu"), "manifest": {"contract": {"train_sample_ids": ["train_1"], "val_sample_ids": ["val_1"]}}}))
                stack.enter_context(mock.patch.object(runner.v6, "build_v6_model_with_fresh_bridge_head", return_value=(mock.Mock(), {}, {})))
                stack.enter_context(mock.patch.object(runner.bridge, "mine_bridge_records_for_split", mine_mock))
                stack.enter_context(mock.patch.object(runner.v6, "build_v6_cached_records", side_effect=lambda rows: rows))
                stack.enter_context(mock.patch.object(runner.audit, "AUTHORITATIVE_V6_CHECKPOINT", checkpoint_path))
                stack.enter_context(mock.patch.object(runner.v6, "DEFAULT_RUN_DIR", run_dir))
                stack.enter_context(mock.patch.object(runner.audit, "load_v6_model_and_verify_invariants", return_value=(mock.Mock(), {"semantic_checkpoint": {"checkpoint_sha256": "semantic"}, "semantic_parameter_max_delta": 0.0, "semantic_bn_state_max_delta": 0.0})))
                stack.enter_context(mock.patch.object(runner.v6, "evaluate_open_on_cached_records", side_effect=[train_eval, val_eval]))
                stack.enter_context(mock.patch.object(runner.audit, "_bridge_probs_for_cached_records", side_effect=[np.zeros((1, 1, 2, 2), dtype=np.float32), np.zeros((1, 1, 2, 2), dtype=np.float32), np.zeros((1, 1, 2, 2), dtype=np.float32), np.zeros((1, 1, 2, 2), dtype=np.float32)]))
                stack.enter_context(mock.patch.object(runner.audit, "load_historical_v2_model", return_value=(mock.Mock(), {"checkpoint_sha256": "hist_v2"})))
                stack.enter_context(mock.patch.object(runner.audit, "evaluate_prob_set_on_cached_records", side_effect=[
                    {
                        "reconstruction": {"positive_success50": 2, "positive_mean_matched_iou": 0.5, "negative_regressions": 5, "negative_topology_changes": 10},
                        "per_sample": [{"sample_id": "train_1", "bridge_positive": 1, "predicted_success50": 1, "predicted_mean_iou": 0.5, "start_mean_iou": 0.4, "component_topology_changed": 0}],
                    },
                    {
                        "reconstruction": {"positive_success50": 2, "positive_mean_matched_iou": 0.5061, "negative_regressions": 20, "negative_topology_changes": 24},
                        "per_sample": [{"sample_id": "val_1", "bridge_positive": 0, "predicted_success50": 0, "predicted_mean_iou": 0.2, "start_mean_iou": 0.8, "component_topology_changed": 24}],
                    },
                ]))
                stack.enter_context(mock.patch.object(runner.audit, "validate_topology_consistency", side_effect=[
                    {"global_negative_topology_changes": 62, "per_sample_negative_topology_changes": 62, "patient_negative_topology_changes": None, "patient_sum_parity": None},
                    {"global_negative_topology_changes": 21, "per_sample_negative_topology_changes": 21, "patient_negative_topology_changes": 21, "patient_sum_parity": True},
                    {"global_negative_topology_changes": 24, "per_sample_negative_topology_changes": 24, "patient_negative_topology_changes": 24, "patient_sum_parity": True},
                ]))
                stack.enter_context(mock.patch.object(runner.audit, "summarize_score_partition", return_value=score_summary))
                stack.enter_context(mock.patch.object(runner.audit, "damage_decomposition", return_value={"aggregate": {"train_positive_target_samples": {"inside_gt_removed_pixels": 1, "outside_gt_removed_pixels": 0, "boundary_distance_bins": {"0_2": 1, "3_5": 0, "6_10": 0, "gt_10": 0}}, "train_zero_target_samples": {"inside_gt_removed_pixels": 0, "outside_gt_removed_pixels": 0, "boundary_distance_bins": {"0_2": 0, "3_5": 0, "6_10": 0, "gt_10": 0}}, "val_positive_target_samples": {"inside_gt_removed_pixels": 1, "outside_gt_removed_pixels": 2, "boundary_distance_bins": {"0_2": 1, "3_5": 1, "6_10": 0, "gt_10": 0}}, "val_zero_target_samples": {"inside_gt_removed_pixels": 3, "outside_gt_removed_pixels": 4, "boundary_distance_bins": {"0_2": 0, "3_5": 1, "6_10": 1, "gt_10": 2}}}, "per_sample_rows": []}))
                stack.enter_context(mock.patch.object(runner.audit, "removal_component_morphology", return_value={"aggregate": {"total_component_count": 1}, "component_area_distribution": {}, "per_sample_rows": [], "per_component_rows": []}))
                stack.enter_context(mock.patch.object(runner.audit, "topology_failure_cases", return_value=[{"sample_id": "val_1", "failure_class": "A_leaflet_interior_cut"}]))
                stack.enter_context(mock.patch.object(runner.audit, "exact_target_oracle", side_effect=[exact_oracle, exact_oracle]))
                stack.enter_context(mock.patch.object(runner.audit, "pareto_frontier", return_value=pareto))
                stack.enter_context(mock.patch.object(runner.audit, "compare_historical_v2_vs_v6", return_value=compare_payload))
                stack.enter_context(mock.patch.object(runner.audit, "interpret_historical_v2_vs_v6", return_value={"main_improvement": "bridge recall with higher removal aggressiveness", "main_unchanged_failure": "inside-GT destructive negative removal"}))
                stack.enter_context(mock.patch.object(runner.audit, "patient_level_failure_concentration", side_effect=[
                    {"rows": [{"patient_id": "p2", "negative_topology_changes": 21}], "global_negative_topology_changes": 21, "patient_negative_topology_changes_sum": 21, "patient_sum_parity": True},
                    {"rows": [{"patient_id": "p2", "negative_topology_changes": 24}], "global_negative_topology_changes": 24, "patient_negative_topology_changes_sum": 24, "patient_sum_parity": True},
                ]))
                stack.enter_context(mock.patch.object(runner.audit, "classify_dominant_failure", return_value={"dominant_failure_class": ["A_SCORE_SEPARABILITY_FAILURE"], "recommended_representation_family": "context-conditioned separation head"}))
                out = runner.run_pipeline(cfg)
            after_sha = _sha256(checkpoint_path)
        self.assertEqual(before_sha, after_sha)
        self.assertFalse(out["audit_contract"]["threshold_sweep_performed"])
        self.assertFalse(out["audit_contract"]["checkpoint_reselected"])
        self.assertEqual(out["audit_contract"]["authoritative_v6_epoch"], 63)
        self.assertTrue(out["audit_contract"]["no_test_access"])
        self.assertFalse(out["audit_contract"]["authoritative_holdout_touched"])
        self.assertEqual(out["topology_consistency"]["v6_train_epoch63"]["global_negative_topology_changes"], 62)
        self.assertEqual(out["topology_consistency"]["v6_reused_val"]["global_negative_topology_changes"], 21)
        self.assertEqual(out["topology_consistency"]["historical_v2_reused_val"]["global_negative_topology_changes"], 24)
        self.assertTrue(out["patient_level"]["v6_open"]["patient_sum_parity"])
        self.assertTrue(out["patient_level"]["historical_v2_open"]["patient_sum_parity"])
        self.assertEqual(mine_mock.call_count, 2)
        for call in mine_mock.call_args_list:
            self.assertTrue(str(call.kwargs["split_txt"]).endswith("datasets\\converted_full_multiclass_curated\\train.txt"))


if __name__ == "__main__":
    unittest.main()
