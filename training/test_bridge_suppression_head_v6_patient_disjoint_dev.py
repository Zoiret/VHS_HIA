from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bridge_presence_gate_v4_patient_disjoint_dev as gate_dev
import bridge_suppression_head as bridge
import bridge_suppression_head_v6_patient_disjoint_dev as v6


class TestBridgeSuppressionHeadV6PatientDisjointDev(unittest.TestCase):
    def _raw_record(self):
        return {
            "sample_id": "s1",
            "patient_id": "p1",
            "gt_count": 2,
            "bridge_positive": 1,
            "candidate_pixels": 4,
            "bridge_pixels": 1,
            "x_0_4": torch.arange(32, dtype=torch.float32).reshape(2, 4, 4),
            "x_2_2": torch.arange(8, dtype=torch.float32).reshape(2, 2, 2),
            "p_leaf": np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32),
            "candidate_mask": np.array([[0, 1], [1, 1]], dtype=np.uint8),
            "bridge_target": np.array([[0, 1], [0, 0]], dtype=np.uint8),
            "oracle_removed_mask": np.array([[0, 0], [1, 1]], dtype=np.uint8),
            "gt_instances": np.array([[1, 1], [0, 2]], dtype=np.uint8),
            "image_path": "dummy.png",
        }

    def test_microset_overlap_audit(self):
        contract = {
            "train_sample_ids": ["a", "b"],
            "val_sample_ids": ["c"],
            "train_patient_ids": ["p1", "p2"],
            "val_patient_ids": ["p3"],
        }
        micro_manifest = {
            "sample_ids": ["b", "c", "x"],
            "rows": [
                {"sample_id": "b", "patient_id": "p2"},
                {"sample_id": "c", "patient_id": "p3"},
                {"sample_id": "x", "patient_id": "px"},
            ],
        }
        overlap = v6.compute_microset_overlap(contract, micro_manifest)
        self.assertEqual(overlap["train_overlap_count"], 1)
        self.assertEqual(overlap["val_overlap_count"], 1)
        self.assertEqual(overlap["train_overlap_samples"], ["b"])
        self.assertEqual(overlap["val_overlap_samples"], ["c"])
        self.assertEqual(overlap["overlapping_patients"], ["p2", "p3"])

    def test_false_bridge_target_audit_counts(self):
        records = [
            {"sample_id": "a", "patient_id": "p1", "gt_count": 1, "bridge_positive": 0, "candidate_pixels": 10, "bridge_pixels": 0},
            {"sample_id": "b", "patient_id": "p1", "gt_count": 2, "bridge_positive": 1, "candidate_pixels": 20, "bridge_pixels": 4},
            {"sample_id": "c", "patient_id": "p2", "gt_count": 3, "bridge_positive": 1, "candidate_pixels": 30, "bridge_pixels": 6},
        ]
        summary = v6.summarize_false_bridge_targets(records)
        self.assertEqual(summary["positive_target_samples"], 2)
        self.assertEqual(summary["zero_target_samples"], 1)
        self.assertAlmostEqual(summary["false_bridge_over_candidate"], 10.0 / 60.0)
        self.assertEqual(summary["by_gt_count"]["GT2"]["false_bridge_pixels"], 4)
        self.assertEqual(summary["by_gt_count"]["GT3"]["false_bridge_pixels"], 6)

    def test_loss_hparam_parity_report_matches_historical_contract(self):
        hist = bridge._read_yaml(v6.HISTORICAL_FULL_CONFIG)
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_patient_disjoint_v6_dev.yaml")
        report = v6.build_loss_hparam_parity_report(cfg, hist)
        self.assertEqual(report["optimizer"], "AdamW")
        self.assertTrue(report["loss"]["dice_term"])
        self.assertEqual(report["loss"]["lambda_negative_mean"], 0.0)
        self.assertEqual(report["loss"]["lambda_negative_hard"], 0.0)
        self.assertEqual(report["learning_rate"], 0.001)
        self.assertEqual(report["batch_size"], 16)
        self.assertEqual(report["epoch_count"], 100)

    def test_development_criterion_is_predeclared(self):
        self.assertEqual(v6.V6_SAFETY_MAX_NEGATIVE_REGRESSIONS, 10)
        self.assertEqual(v6.V6_SAFETY_MAX_NEGATIVE_TOPOLOGY_CHANGES, 12)
        self.assertEqual(v6.DEVELOPMENT_REFERENCES["historical_v2_open"]["negative_regressions"], 20)
        self.assertEqual(v6.DEVELOPMENT_REFERENCES["historical_v2_open"]["negative_topology_changes"], 24)

    def test_v6_decision_rule_requires_utility_and_safety(self):
        val_eval = {
            "reconstruction": {
                "positive_success50": 2,
                "positive_mean_matched_iou": 0.5,
                "negative_regressions": 9,
                "negative_topology_changes": 11,
            }
        }
        out = v6.evaluate_v6_development_decision(val_eval)
        self.assertEqual(out["full_data_v2_representation_useful_on_reused_dev_val"], "YES")
        fail = v6.evaluate_v6_development_decision(
            {"reconstruction": {"positive_success50": 1, "positive_mean_matched_iou": 0.5, "negative_regressions": 9, "negative_topology_changes": 11}}
        )
        self.assertEqual(fail["local_v2_bridge_representation_insufficient"], "YES")

    def test_fresh_bridge_head_initialization_never_loads_old_v2_weights(self):
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_patient_disjoint_v6_dev.yaml")
        fake_model = mock.Mock()
        fake_model.context_projection.state_dict.return_value = {"w": np.zeros((1,), dtype=np.float32)}  # type: ignore[attr-defined]
        fake_model.bridge_head.state_dict.return_value = {"w": np.zeros((1,), dtype=np.float32)}  # type: ignore[attr-defined]
        with mock.patch.object(bridge, "build_model_from_cfg", return_value=fake_model), \
             mock.patch.object(bridge, "load_semantic_checkpoint", return_value={"checkpoint_sha256": "semantic"}), \
             mock.patch.object(bridge, "build_optimizer", return_value=(mock.Mock(), {"total_trainable_params": 1713})), \
             mock.patch.object(bridge, "canonical_model_state_sha256", return_value="sha") as sha_mock:
            _model, semantic_info, meta = v6.build_v6_model_with_fresh_bridge_head(cfg, device=mock.Mock())
        self.assertEqual(semantic_info["checkpoint_sha256"], "semantic")
        self.assertEqual(meta["total_trainable_params"], 1713)
        self.assertEqual(sha_mock.call_count, 2)

    def test_v6_cached_records_match_historical_tensor_contract(self):
        raw = self._raw_record()
        fake_reconstruction = {"labels": np.zeros((2, 2), dtype=np.uint8), "metrics": {"all_iou_ge_0.50": False, "instance_mean_matched_iou": 0.0}}
        with mock.patch.object(bridge, "run_locked_reconstruction", return_value=fake_reconstruction):
            historical = bridge.cache_microset_features([raw])[0]
            current = v6.build_v6_cached_records([raw])[0]
        for key in ("x_0_4", "x_2_2", "p_leaf", "candidate_mask", "bridge_target"):
            self.assertTrue(torch.is_tensor(current[key]))
            self.assertEqual(str(current[key].dtype), str(historical[key].dtype))
            self.assertEqual(list(current[key].shape), list(historical[key].shape))
            self.assertTrue(torch.equal(current[key], historical[key]))

    def test_stack_cached_batch_succeeds_on_corrected_v6_records(self):
        raw = self._raw_record()
        fake_reconstruction = {"labels": np.zeros((2, 2), dtype=np.uint8), "metrics": {"all_iou_ge_0.50": False, "instance_mean_matched_iou": 0.0}}
        with mock.patch.object(bridge, "run_locked_reconstruction", return_value=fake_reconstruction):
            records = v6.build_v6_cached_records([raw])
        batch = bridge.stack_cached_batch(records, torch.device("cpu"))
        self.assertEqual(list(batch["p_leaf"].shape), [1, 1, 2, 2])
        self.assertEqual(str(batch["candidate_mask"].dtype), "torch.float32")
        self.assertEqual(str(batch["bridge_target"].dtype), "torch.float32")

    def test_validate_train_cache_contract_blocks_ndarray_fields(self):
        bad = {
            "sample_id": "s1",
            "x_0_4": torch.ones((2, 4, 4), dtype=torch.float32),
            "x_2_2": torch.ones((2, 2, 2), dtype=torch.float32),
            "p_leaf": np.ones((2, 2), dtype=np.float32),
            "candidate_mask": np.ones((2, 2), dtype=np.uint8),
            "bridge_target": np.zeros((2, 2), dtype=np.uint8),
        }
        with self.assertRaises(SystemExit) as cm:
            v6.validate_train_cache_contract([bad])
        self.assertIn("v6_train_cache_contract_mismatch", str(cm.exception))

    def test_target_tensor_values_unchanged_after_tensorization(self):
        raw = self._raw_record()
        fake_reconstruction = {"labels": np.zeros((2, 2), dtype=np.uint8), "metrics": {"all_iou_ge_0.50": False, "instance_mean_matched_iou": 0.0}}
        with mock.patch.object(bridge, "run_locked_reconstruction", return_value=fake_reconstruction):
            records = v6.build_v6_cached_records([raw])
        target = records[0]["bridge_target"]
        self.assertTrue(torch.equal(target, torch.tensor([[[0.0, 1.0], [0.0, 0.0]]], dtype=torch.float32)))

    def test_metadata_preserved_after_v6_cache_construction(self):
        raw = self._raw_record()
        fake_reconstruction = {"labels": np.zeros((2, 2), dtype=np.uint8), "metrics": {"all_iou_ge_0.50": False, "instance_mean_matched_iou": 0.0}}
        with mock.patch.object(bridge, "run_locked_reconstruction", return_value=fake_reconstruction):
            records = v6.build_v6_cached_records([raw])
        self.assertEqual(records[0]["sample_id"], "s1")
        self.assertEqual(records[0]["patient_id"], "p1")
        self.assertEqual(records[0]["gt_count"], 2)
        self.assertEqual(records[0]["bridge_positive"], 1)
        self.assertEqual(records[0]["candidate_pixels"], 4)
        self.assertEqual(records[0]["bridge_pixels"], 1)

    def test_validate_train_cache_contract_checks_manifest_sample_set(self):
        raw = self._raw_record()
        fake_reconstruction = {"labels": np.zeros((2, 2), dtype=np.uint8), "metrics": {"all_iou_ge_0.50": False, "instance_mean_matched_iou": 0.0}}
        with mock.patch.object(bridge, "run_locked_reconstruction", return_value=fake_reconstruction):
            records = v6.build_v6_cached_records([raw])
        with self.assertRaises(SystemExit) as cm:
            v6.validate_train_cache_contract(records, expected_sample_ids=["other"])
        self.assertIn("manifest_sample_set_mismatch", str(cm.exception))

    def test_full_train_evaluate_open_on_cached_records_completes_with_patient_metadata(self):
        raw = self._raw_record()
        fake_reconstruction = {"labels": np.zeros((2, 2), dtype=np.uint8), "metrics": {"all_iou_ge_0.50": False, "instance_mean_matched_iou": 0.0}}
        with mock.patch.object(bridge, "run_locked_reconstruction", return_value=fake_reconstruction):
            records = v6.build_v6_cached_records([raw])

        class FakeModel(torch.nn.Module):
            def eval(self):
                return self

            def bridge_forward_from_cached(self, x_0_4, x_2_2, p_leaf):
                return {"bridge_logits": torch.zeros((x_0_4.shape[0], 1, p_leaf.shape[-2], p_leaf.shape[-1]), dtype=torch.float32)}

        out = v6.evaluate_open_on_cached_records(model=FakeModel(), cached_records=records, device=torch.device("cpu"))
        self.assertEqual(out["per_sample"][0]["patient_id"], "p1")
        self.assertEqual(out["per_sample"][0]["gt_count"], 2)
        patient_report = v6.patient_level_report(out["per_sample"])
        self.assertEqual(patient_report["rows"][0]["patient_id"], "p1")


if __name__ == "__main__":
    unittest.main()
