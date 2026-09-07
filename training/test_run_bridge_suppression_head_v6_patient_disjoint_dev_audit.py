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
import run_bridge_suppression_head_v6_patient_disjoint_dev_audit as runner


class TestRunBridgeSuppressionHeadV6PatientDisjointDevAudit(unittest.TestCase):
    def _fake_manifest_stage(self):
        return {
            "device": torch.device("cpu"),
            "manifest": {
                "contract": {
                    "train_sample_ids": ["a"],
                    "val_sample_ids": ["b"],
                    "train_patient_ids": ["p1"],
                    "val_patient_ids": ["p2"],
                }
            },
        }

    def test_audit_reports_reused_dev_validation_only(self):
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_patient_disjoint_v6_dev.yaml")
        fake_records = [{"sample_id": "a", "patient_id": "p1", "gt_count": 2, "bridge_positive": 1, "candidate_pixels": 10, "bridge_pixels": 3}]
        with tempfile.TemporaryDirectory() as td:
            cfg = dict(cfg)
            cfg["analysis"] = {"feature_audit_dir": str(Path(td) / "analysis")}
            with mock.patch.object(runner.split_runner, "_prepare_manifest", return_value=self._fake_manifest_stage()), \
                 mock.patch.object(runner.v6, "load_historical_micro_manifest", return_value={"sample_ids": [], "rows": []}), \
                 mock.patch.object(runner.v6, "build_v6_model_with_fresh_bridge_head", return_value=(mock.Mock(), {"checkpoint_sha256": "semantic"}, {"total_trainable_params": 1713})), \
                 mock.patch.object(runner.bridge, "mine_bridge_records_for_split", side_effect=[fake_records, fake_records]), \
                 mock.patch.object(runner.v6, "build_loss_hparam_parity_report", return_value={"optimizer": "AdamW"}):
                out = runner.run_pipeline(cfg)
        self.assertTrue(out["validation_status"]["reused_development_validation"])
        self.assertFalse(out["validation_status"]["authoritative_holdout_touched"])


if __name__ == "__main__":
    unittest.main()
