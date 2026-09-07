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
        }

    def test_no_validation_before_checkpoint_freeze(self):
        prepared = self._fake_prepared()
        with mock.patch.object(runner, "_prepare_inputs", return_value=prepared), \
             mock.patch.object(runner.bridge, "_snapshot_named_parameters", return_value={}), \
             mock.patch.object(runner.bridge, "_collect_batchnorm_stats", return_value=[]), \
             mock.patch.object(runner, "_train_v6", return_value={"best_payload": {"epoch": 1}}), \
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

    def test_v6_config_never_points_init_to_old_v2_checkpoint(self):
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_patient_disjoint_v6_dev.yaml")
        init_ckpt = str((cfg.get("train") or {}).get("init_checkpoint", ""))
        self.assertNotIn("bridge_suppression_frozen_semantic_micro_overfit_v2", init_ckpt)


if __name__ == "__main__":
    unittest.main()
