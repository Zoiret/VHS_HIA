from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bridge_suppression_head as bridge
import train_bridge_suppression_head_v6_patient_disjoint_dev as runner


class TestTrainBridgeSuppressionHeadV6PatientDisjointDev(unittest.TestCase):
    def _fake_prepared(self):
        model = mock.Mock()
        model.base = torch.nn.Sequential(torch.nn.BatchNorm2d(1))
        model.named_parameters = lambda: []
        optimizer = mock.Mock()
        return {
            "manifest_stage": {"manifest": {"contract": {"train_summary": {"sample_count": 121, "patient_count": 17}}}},
            "contract": {
                "train_summary": {"sample_count": 121, "patient_count": 17},
            },
            "device": torch.device("cpu"),
            "model": model,
            "semantic_info": {"checkpoint_sha256": "semantic"},
            "model_meta": {"total_trainable_params": 1713},
            "optimizer": optimizer,
            "optimizer_meta": {"total_trainable_params": 1713},
            "loss_fn": mock.Mock(),
            "train_records": [{"sample_id": "a", "patient_id": "p1", "bridge_positive": 1, "gt_bridge_pixels": 1, "candidate_pixels": 10}],
            "val_records": [{"sample_id": "b", "patient_id": "p2", "bridge_positive": 1, "gt_bridge_pixels": 1, "candidate_pixels": 10}],
            "train_cache_contract": {
                "sample_count": 1,
                "unique_sample_ids": 1,
                "patient_count": 1,
                "fields": {
                    "p_leaf": {"dtype": "torch.float32", "shape": [1, 2, 2]},
                    "candidate_mask": {"dtype": "torch.float32", "shape": [1, 2, 2]},
                    "bridge_target": {"dtype": "torch.float32", "shape": [1, 2, 2]},
                },
                "metadata_fields": {
                    "sample_id": "str",
                    "patient_id": "str",
                    "gt_count": "int",
                    "bridge_positive": "int",
                    "candidate_pixels": "int",
                    "bridge_pixels": "int",
                },
            },
        }

    def test_no_validation_before_checkpoint_freeze(self):
        prepared = self._fake_prepared()
        with mock.patch.object(runner, "_prepare_inputs", return_value=prepared), \
             mock.patch.object(runner.bridge, "_snapshot_named_parameters", return_value={}), \
             mock.patch.object(runner.bridge, "_collect_batchnorm_stats", return_value=[]), \
             mock.patch.object(runner, "_train_v6", return_value={"best_payload": {"epoch": 1}, "optimizer_updates": 8, "batch_count": 8, "epochs": 100}), \
             mock.patch.object(runner, "_load_checkpoint", return_value={"step": 1}), \
             mock.patch.object(runner.v6, "evaluate_open_on_cached_records", side_effect=[
                 {"reconstruction": {"positive_success50": 2, "positive_mean_matched_iou": 0.5, "negative_regressions": 0, "negative_topology_changes": 0}, "per_sample": []},
                 {"reconstruction": {"positive_success50": 2, "positive_mean_matched_iou": 0.5, "negative_regressions": 0, "negative_topology_changes": 0}, "per_sample": []},
             ]) as eval_mock, \
             mock.patch.object(runner.v6, "patient_level_report", return_value={"rows": [], "macro_mean": {}}), \
             mock.patch.object(runner.v6, "build_loss_hparam_parity_report", return_value={"loss": {}}):
            with tempfile.TemporaryDirectory() as td:
                cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_patient_disjoint_v6_dev.yaml")
                cfg = dict(cfg)
                cfg["train"] = dict(cfg["train"])
                cfg["train"]["save_dir"] = str(Path(td) / "run")
                summary = runner.run_pipeline(cfg)
        self.assertEqual(eval_mock.call_count, 2)
        self.assertTrue(summary["validation_status"]["reused_development_validation"])
        self.assertEqual(summary["training_progress"]["optimizer_updates"], 8)

    def test_v6_config_never_points_init_to_old_v2_checkpoint(self):
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_patient_disjoint_v6_dev.yaml")
        init_ckpt = str((cfg.get("train") or {}).get("init_checkpoint", ""))
        self.assertNotIn("bridge_suppression_frozen_semantic_micro_overfit_v2", init_ckpt)

    def test_prepare_inputs_tensorizes_records_and_validates_train_cache(self):
        raw = {
            "sample_id": "s1",
            "patient_id": "p1",
            "gt_count": 2,
            "bridge_positive": 1,
            "candidate_pixels": 4,
            "bridge_pixels": 1,
            "x_0_4": torch.ones((2, 4, 4), dtype=torch.float32),
            "x_2_2": torch.ones((2, 2, 2), dtype=torch.float32),
            "p_leaf": __import__("numpy").ones((2, 2), dtype="float32"),
            "candidate_mask": __import__("numpy").ones((2, 2), dtype="uint8"),
            "bridge_target": __import__("numpy").zeros((2, 2), dtype="uint8"),
            "oracle_removed_mask": __import__("numpy").ones((2, 2), dtype="uint8"),
            "gt_instances": __import__("numpy").ones((2, 2), dtype="uint8"),
            "image_path": "dummy.png",
        }
        cfg = {"dataset": {"train_txt": "datasets/converted_full_multiclass_curated/train.txt"}, "train": {"init_checkpoint": "x"}}
        fake_manifest = {"manifest": {"contract": {"train_sample_ids": ["s1"], "val_sample_ids": ["v1"]}}, "device": torch.device("cpu")}
        fake_model = mock.Mock()
        fake_reconstruction = {"labels": __import__("numpy").zeros((2, 2), dtype="uint8"), "metrics": {"all_iou_ge_0.50": False, "instance_mean_matched_iou": 0.0}}
        with mock.patch.object(runner.split_runner, "_prepare_manifest", return_value=fake_manifest), \
             mock.patch.object(runner.v6, "build_v6_model_with_fresh_bridge_head", return_value=(fake_model, {}, {})), \
             mock.patch.object(runner.bridge, "build_optimizer", return_value=(mock.Mock(), {})), \
             mock.patch.object(runner.bridge, "build_bridge_loss_from_cfg", return_value=mock.Mock()), \
             mock.patch.object(runner.bridge, "mine_bridge_records_for_split", side_effect=[[raw], [raw]]), \
             mock.patch.object(runner.bridge, "run_locked_reconstruction", return_value=fake_reconstruction):
            prepared = runner._prepare_inputs(cfg)
        self.assertTrue(torch.is_tensor(prepared["train_records"][0]["p_leaf"]))
        self.assertEqual(str(prepared["train_records"][0]["p_leaf"].dtype), "torch.float32")
        self.assertIn("train_cache_contract", prepared)
        self.assertEqual(prepared["train_records"][0]["patient_id"], "p1")

    def test_no_optimizer_step_occurs_if_cache_validation_fails(self):
        cfg = {"dataset": {"train_txt": "datasets/converted_full_multiclass_curated/train.txt"}, "train": {"init_checkpoint": "x"}}
        fake_manifest = {"manifest": {"contract": {"train_sample_ids": ["s1"], "val_sample_ids": ["v1"]}}, "device": torch.device("cpu")}
        fake_model = mock.Mock()
        fake_opt = mock.Mock()
        with mock.patch.object(runner.split_runner, "_prepare_manifest", return_value=fake_manifest), \
             mock.patch.object(runner.v6, "build_v6_model_with_fresh_bridge_head", return_value=(fake_model, {}, {})), \
             mock.patch.object(runner.bridge, "build_optimizer", return_value=(fake_opt, {})), \
             mock.patch.object(runner.bridge, "build_bridge_loss_from_cfg", return_value=mock.Mock()), \
             mock.patch.object(runner.bridge, "mine_bridge_records_for_split", side_effect=[[{"sample_id": "s1"}], [{"sample_id": "v1"}]]), \
             mock.patch.object(runner.v6, "build_v6_cached_records", side_effect=lambda rows: rows), \
             mock.patch.object(runner.v6, "validate_train_cache_contract", side_effect=SystemExit("bad contract")):
            with self.assertRaises(SystemExit):
                runner._prepare_inputs(cfg)
        fake_opt.step.assert_not_called()

    def test_metadata_attachment_does_not_change_tensor_values(self):
        raw = {
            "sample_id": "s1",
            "patient_id": "p1",
            "gt_count": 2,
            "bridge_positive": 1,
            "candidate_pixels": 4,
            "bridge_pixels": 1,
            "x_0_4": torch.ones((2, 4, 4), dtype=torch.float32),
            "x_2_2": torch.ones((2, 2, 2), dtype=torch.float32),
            "p_leaf": __import__("numpy").ones((2, 2), dtype="float32"),
            "candidate_mask": __import__("numpy").ones((2, 2), dtype="uint8"),
            "bridge_target": __import__("numpy").zeros((2, 2), dtype="uint8"),
            "oracle_removed_mask": __import__("numpy").ones((2, 2), dtype="uint8"),
            "gt_instances": __import__("numpy").ones((2, 2), dtype="uint8"),
            "image_path": "dummy.png",
        }
        fake_reconstruction = {"labels": __import__("numpy").zeros((2, 2), dtype="uint8"), "metrics": {"all_iou_ge_0.50": False, "instance_mean_matched_iou": 0.0}}
        with mock.patch.object(runner.bridge, "run_locked_reconstruction", return_value=fake_reconstruction):
            historical = runner.bridge.cache_microset_features([raw])[0]
            merged = runner.v6.build_v6_cached_records([raw])[0]
        for key in ("p_leaf", "x_0_4", "x_2_2", "candidate_mask", "bridge_target"):
            self.assertTrue(torch.equal(historical[key], merged[key]))


if __name__ == "__main__":
    unittest.main()
