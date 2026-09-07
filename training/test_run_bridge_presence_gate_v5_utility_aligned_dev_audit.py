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

import bridge_presence_gate_v5_utility_aligned_dev as v5
import bridge_suppression_head as bridge
import run_bridge_presence_gate_v5_utility_aligned_dev_audit as runner


class TestRunBridgePresenceGateV5UtilityAlignedDevAudit(unittest.TestCase):
    def _fake_manifest_stage(self):
        return {
            "device": torch.device("cpu"),
            "manifest": {"contract": {"train_sample_ids": ["a"], "val_sample_ids": ["v1"]}},
        }

    def _fake_prepared(self):
        return {
            "feature_rows": [
                {
                    "sample_id": "a",
                    "patient_id": "p1",
                    "gt_count": 2,
                    "bridge_positive_target": 1,
                    "candidate_pixels": 10,
                    "candidate_fraction_denominator": 100,
                    "candidate_fraction": 0.1,
                    "candidate_component_count": 1.0,
                    "bridge_score_mean": 0.1,
                    "bridge_score_max": 0.2,
                    "bridge_score_top1pct_mean": 0.2,
                    "bridge_score_top5pct_mean": 0.2,
                    "bridge_score_frac_ge_0p50": 0.0,
                    "bridge_score_frac_ge_0p75": 0.0,
                    "bridge_score_frac_ge_0p90": 0.0,
                }
            ],
            "hard_gate_state_cache": [
                {
                    "sample_id": "a",
                    "gate_target": 1,
                    "bridge_positive": 1,
                    "closed": {"predicted_success50": 0, "predicted_mean_iou": 0.1},
                    "open": {"predicted_success50": 1, "predicted_mean_iou": 0.2},
                }
            ],
            "features_t": torch.ones((1, 105), dtype=torch.float32),
            "selector_audit": {},
        }

    def test_audit_marks_validation_as_reused_dev(self):
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v5_utility_aligned_dev_v1.yaml")
        with tempfile.TemporaryDirectory() as td:
            cfg = dict(cfg)
            cfg["analysis"] = {"feature_audit_dir": str(Path(td) / "analysis")}
            with mock.patch.object(runner.preflight_runner, "_prepare_manifest", return_value=self._fake_manifest_stage()), \
                 mock.patch.object(runner.gate_v4, "load_frozen_v2_pixel_model_from_cfg", return_value=(mock.Mock(), {})), \
                 mock.patch.object(runner.dev, "_prepare_split_preflight_core", return_value=self._fake_prepared()):
                out = runner.run_pipeline(cfg)
        self.assertFalse(out["validation_status"]["fresh_unseen_development_validation"])
        self.assertTrue(out["validation_status"]["reused_development_validation"])
        self.assertFalse(out["validation_status"]["authoritative_holdout_touched"])


if __name__ == "__main__":
    unittest.main()
