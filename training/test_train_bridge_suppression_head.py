from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


_THIS_DIR = Path(__file__).resolve().parent


class TestTrainBridgeSuppressionHead(unittest.TestCase):
    @staticmethod
    def _mods():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import bridge_suppression_head as bridge
        import train_bridge_suppression_head as runner

        return bridge, runner

    def test_smoke_only_does_not_require_full_train_val_audit_outputs(self):
        bridge, runner = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit.yaml")
        with tempfile.TemporaryDirectory() as td:
            cfg["train"]["save_dir"] = str(Path(td) / "micro")
            cfg["reserved_full_run"]["save_dir"] = str(Path(td) / "future_full")
            cfg["_config_path"] = str((bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit.yaml").resolve())
            fake_record = {
                "sample_id": "m00_p00_s00",
                "image_path": str((bridge.REPO_ROOT / "datasets" / "converted_full_multiclass" / "dummy.png").resolve()),
                "gt_semantic": __import__("numpy").zeros((16, 16), dtype="uint8"),
                "bridge_target": __import__("numpy").zeros((16, 16), dtype="uint8"),
            }
            with mock.patch.object(bridge, "_build_split_items", return_value=[{"sample_id": "m00_p00_s00"}]), \
                 mock.patch.object(bridge, "build_model_from_cfg"), \
                 mock.patch.object(bridge, "load_semantic_checkpoint", return_value={"checkpoint_sha256": bridge.CHECKPOINT_SHA256 if hasattr(bridge, "CHECKPOINT_SHA256") else ""}), \
                 mock.patch.object(bridge, "build_optimizer", return_value=(mock.Mock(), {"total_trainable_params": 1})), \
                 mock.patch.object(bridge, "mine_bridge_records_for_split", return_value=[fake_record]), \
                 mock.patch.object(runner, "_smoke_step", return_value={"status": "pass"}):
                summary = runner.run_pipeline(cfg, smoke_only=True)
            self.assertEqual(summary["smoke"]["status"], "pass")
            self.assertIn("a100_smoke_command", summary)

    def test_reserved_full_run_dir_blocker(self):
        bridge, runner = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit.yaml")
        with tempfile.TemporaryDirectory() as td:
            micro = Path(td) / "micro"
            future = Path(td) / "future"
            future.mkdir(parents=True, exist_ok=True)
            cfg["train"]["save_dir"] = str(micro)
            cfg["reserved_full_run"]["save_dir"] = str(future)
            cfg["_config_path"] = str((bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit.yaml").resolve())
            with self.assertRaises(SystemExit):
                runner.run_pipeline(cfg, smoke_only=True)


if __name__ == "__main__":
    unittest.main()
