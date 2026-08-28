from __future__ import annotations

import sys
import tempfile
import unittest
import hashlib
from io import BytesIO
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
        self.assertTrue(str((cfg.get("frozen_v2_pixel_head") or {}).get("checkpoint_path", "")).endswith("best_reconstruction.pth"))
        self.assertEqual(str((cfg.get("frozen_v2_pixel_head") or {}).get("expected_file_sha256", "")), "e1e7a31a078f6baf1da05b79541e5818430583deb1f4329b3e99d47cdb055573")
        self.assertEqual(str((cfg.get("frozen_v2_pixel_head") or {}).get("expected_model_state_sha256", "")), "f21b80e1295271deaa48fb1cf0a7d26669e2821763ee115e146a02d3588cea6f")
        self.assertEqual(int((cfg.get("frozen_v2_pixel_head") or {}).get("expected_step", -1)), 300)
        self.assertEqual(str((cfg.get("frozen_v2_pixel_head") or {}).get("state_dict_key", "")), "model")
        self.assertTrue(bool((cfg.get("experiment_notes") or {}).get("no_test_usage", False)))
        self.assertTrue(bool((cfg.get("experiment_notes") or {}).get("no_authoritative_holdout", False)))

    def test_validate_expected_checkpoint_provenance_pass_and_fail_modes(self):
        good = {
            "checkpoint_path": "x",
            "checkpoint_file_sha256": "abc",
            "checkpoint_model_state_sha256": "def",
            "step": 300,
            "state_dict_key": "model",
        }
        cfg = {
            "frozen_v2_pixel_head": {
                "expected_file_sha256": "abc",
                "expected_model_state_sha256": "def",
                "expected_step": 300,
                "state_dict_key": "model",
            }
        }
        passed = gate_v4.validate_expected_bridge_checkpoint_provenance(cfg, good)
        self.assertEqual(passed["status"], "pass")
        self.assertTrue(passed["semantic_frozen"])
        self.assertTrue(passed["v2_pixel_head_frozen"])
        blocked_sha = gate_v4.validate_expected_bridge_checkpoint_provenance(
            {"frozen_v2_pixel_head": {**cfg["frozen_v2_pixel_head"], "expected_file_sha256": "zzz"}},
            good,
        )
        self.assertEqual(blocked_sha["status"], "blocked")
        self.assertTrue(any("file SHA256 mismatch" in err for err in blocked_sha["errors"]))
        blocked_model = gate_v4.validate_expected_bridge_checkpoint_provenance(
            {"frozen_v2_pixel_head": {**cfg["frozen_v2_pixel_head"], "expected_model_state_sha256": "zzz"}},
            good,
        )
        self.assertEqual(blocked_model["status"], "blocked")
        self.assertTrue(any("model-state SHA256 mismatch" in err for err in blocked_model["errors"]))
        blocked_step = gate_v4.validate_expected_bridge_checkpoint_provenance(
            {"frozen_v2_pixel_head": {**cfg["frozen_v2_pixel_head"], "expected_step": 299}},
            good,
        )
        self.assertEqual(blocked_step["status"], "blocked")
        self.assertTrue(any("step mismatch" in err for err in blocked_step["errors"]))

    def test_inspect_bridge_checkpoint_reports_locked_fields(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "best_reconstruction.pth"
            state = {"layer.weight": torch.ones((2, 2), dtype=torch.float32)}
            torch.save({"model": state, "step": 300}, str(path))
            info = gate_v4.inspect_bridge_checkpoint(path)
            buf = BytesIO()
            torch.save(state, buf)
            self.assertEqual(info["checkpoint_path"], str(path.resolve()))
            self.assertEqual(info["checkpoint_file_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(info["checkpoint_model_state_sha256"], hashlib.sha256(buf.getvalue()).hexdigest())
            self.assertEqual(info["step"], 300)
            self.assertEqual(info["state_dict_key"], "model")

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

    def test_wrong_checkpoint_provenance_fails_closed_before_model_load(self):
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v4_micro_overfit.yaml")
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "best_reconstruction.pth"
            torch.save({"model": {"w": torch.ones((1,), dtype=torch.float32)}, "step": 123}, str(ckpt_path))
            cfg = dict(cfg)
            cfg["frozen_v2_pixel_head"] = dict(cfg["frozen_v2_pixel_head"])
            cfg["frozen_v2_pixel_head"]["checkpoint_path"] = str(ckpt_path)
            with mock.patch.object(bridge, "build_model_from_cfg") as build_mock, \
                 mock.patch.object(bridge, "load_semantic_checkpoint", return_value={"checkpoint_path": "x", "checkpoint_sha256": "y"}):
                with self.assertRaises(SystemExit):
                    gate_v4.load_frozen_v2_pixel_model_from_cfg(cfg, torch.device("cpu"))
            build_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
