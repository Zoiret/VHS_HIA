from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bridge_suppression_head as bridge
import run_bridge_presence_gate_v4_patient_disjoint_dev_preflight as runner


class TestRunBridgePresenceGateV4PatientDisjointDevPreflight(unittest.TestCase):
    def _fake_manifest_stage(self):
        return {
            "device": mock.Mock(type="cpu"),
            "semantic_model": mock.Mock(),
            "semantic_checkpoint": {"checkpoint_sha256": "semantic"},
            "records": [],
            "metadata_rows": [],
            "manifest": {
                "contract_path": "contract.json",
                "train_path": "train.txt",
                "val_path": "val.txt",
                "created": True,
                "contract": {
                    "train_sample_ids": ["a", "b"],
                    "val_sample_ids": ["c"],
                    "train_summary": {"patient_ids": ["p1"]},
                    "val_summary": {"patient_ids": ["p2"]},
                },
            },
            "split_payload": {
                "algorithm": {"seed": 1337},
                "patient_overlap": 0,
                "sample_overlap": 0,
            },
            "source_split": bridge.DEFAULT_TRAIN_SPLIT,
            "source_contract": {
                "source_path": "datasets/converted_full_multiclass_curated/train.txt",
                "source_canonical_sha256": "f5e920ffaf54c0a0034c457cf3c951f71e186a9f35e3fe67a5eee95737b2ee82",
                "source_sample_count": 157,
                "source_patient_count": 22,
                "record_mining_seconds": 1.0,
            },
        }

    def test_manifest_only_does_not_require_v2_checkpoint(self):
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v4_patient_disjoint_dev_v1.yaml")
        with tempfile.TemporaryDirectory() as td:
            cfg = dict(cfg)
            cfg["analysis"] = {"feature_audit_dir": str(Path(td) / "analysis")}
            with mock.patch.object(runner, "_prepare_manifest", return_value=self._fake_manifest_stage()):
                out = runner.run_pipeline(cfg, manifest_only=True)
        self.assertEqual(out["status"], "manifest_ready")
        self.assertEqual(out["feature_contract"]["dimensions"], 105)
        self.assertEqual(out["feature_contract"]["trainable_parameters"], 1713)

    def test_preflight_blocks_when_v2_checkpoint_missing_after_manifest(self):
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v4_patient_disjoint_dev_v1.yaml")
        with tempfile.TemporaryDirectory() as td:
            cfg = dict(cfg)
            cfg["analysis"] = {"feature_audit_dir": str(Path(td) / "analysis")}
            cfg["frozen_v2_pixel_head"] = dict(cfg["frozen_v2_pixel_head"])
            cfg["frozen_v2_pixel_head"]["checkpoint_path"] = str(Path(td) / "missing.pth")
            with mock.patch.object(runner, "_prepare_manifest", return_value=self._fake_manifest_stage()):
                with self.assertRaises(SystemExit):
                    runner.run_pipeline(cfg, manifest_only=False)


if __name__ == "__main__":
    unittest.main()
