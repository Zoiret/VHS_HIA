from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import numpy as np
import torch

import bridge_presence_gate_v4 as gate_v4
import bridge_suppression_head as bridge
import train_bridge_presence_gate_v4 as runner


class TestTrainBridgePresenceGateV4(unittest.TestCase):
    def test_v4_config_no_test_or_holdout_and_locked_manifest(self):
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v4_micro_overfit.yaml")
        self.assertIn("bridge_suppression_micro_overfit_v2_manifest.json", str(cfg["micro_overfit"]["manifest_path"]))
        self.assertEqual(float((cfg.get("frozen_v2_pixel_head") or {}).get("pixel_remove_threshold", 0.0)), 0.50)
        self.assertTrue(bool((cfg.get("experiment_notes") or {}).get("no_test_usage", False)))
        self.assertTrue(bool((cfg.get("experiment_notes") or {}).get("no_authoritative_holdout", False)))

    def test_gate_only_gradients(self):
        gate = gate_v4.SampleLevelBridgePresenceGate(input_dim=4, hidden_dim=4, dropout_p=0.0)
        x = torch.ones((2, 4), dtype=torch.float32)
        y = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
        loss = torch.nn.BCEWithLogitsLoss()(gate(x), y)
        loss.backward()
        self.assertTrue(all(param.grad is not None for param in gate.parameters()))

    def test_progress_eta_behavior_retained(self):
        eta = runner._progress_eta_seconds(
            current_step=100,
            max_steps=300,
            log_every=10,
            mean_step_seconds=1.0,
            recent_eval_seconds=[120.0],
        )
        self.assertGreaterEqual(eta, 200.0 + 20 * 120.0)

    def test_preflight_only_does_not_launch_training(self):
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v4_micro_overfit.yaml")
        with tempfile.TemporaryDirectory() as td:
            cfg = dict(cfg)
            cfg["train"] = dict(cfg["train"])
            cfg["train"]["save_dir"] = str(Path(td) / "run")
            cfg["reserved_full_run"] = dict(cfg["reserved_full_run"])
            cfg["reserved_full_run"]["save_dir"] = str(Path(td) / "future")
            cfg["analysis"] = dict(cfg["analysis"])
            cfg["analysis"]["feature_audit_dir"] = str(Path(td) / "analysis")
            fake_prepared = {
                "frozen_checkpoint": {"checkpoint_path": "x", "checkpoint_sha256": "y", "step": 300},
                "manifest_payload": {"sample_ids": ["a"]},
                "manifest_resolution": {"split_validation": {"status": "pass"}, "record_validation": {"status": "pass"}},
                "simple_threshold": {"simple_gate_threshold_exists": False, "best_scalar": None},
                "gate_model": gate_v4.SampleLevelBridgePresenceGate(input_dim=4, hidden_dim=4),
                "feature_rows": [{"sample_id": "a", "bridge_positive_target": 1}],
            }
            with mock.patch.object(runner, "_prepare_v4_inputs", return_value=fake_prepared), \
                 mock.patch.object(runner, "_run_gate_micro_overfit") as train_mock:
                out = runner.run_pipeline(cfg, preflight_only=True)
            train_mock.assert_not_called()
            self.assertEqual(out["frozen_v2_pixel_head"]["step"], 300)

    def test_missing_frozen_v2_checkpoint_blocks(self):
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v4_micro_overfit.yaml")
        with tempfile.TemporaryDirectory() as td:
            cfg = dict(cfg)
            cfg["frozen_v2_pixel_head"] = dict(cfg["frozen_v2_pixel_head"])
            cfg["frozen_v2_pixel_head"]["checkpoint_path"] = str(Path(td) / "missing.pth")
            with self.assertRaises(SystemExit):
                gate_v4.load_frozen_v2_pixel_model_from_cfg(cfg, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
