from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent


class TestSemanticTopologyAuxPreflight(unittest.TestCase):
    @staticmethod
    def _mods():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import semantic_topology_aux as topo_aux
        import semantic_topology_aux_preflight as pre

        return topo_aux, pre

    @classmethod
    def setUpClass(cls):
        cls.topo_aux, cls.pre = cls._mods()
        cls.cfg = cls.topo_aux._read_yaml(cls.topo_aux.REPO_ROOT / "training" / "configs" / "unetpp_effb3_semantic_topology_aux_finetune_100ep.yaml")
        cls.contract = cls.topo_aux.TopologyTargetContract(
            separation_radius_px=8,
            narrow_width_threshold_px=12,
            include_foreground_boundary=False,
            source_split_txt="train-only",
            source_instance_root="instances",
            selection_rule="unit-test",
            train_only=True,
        )

    def test_pair_overlap_reports_exact_samples_and_patients(self):
        left = {"path": "left.txt", "samples": ["m01_p01_s00", "m02_p01_s00"], "patients": ["m01_p01", "m02_p01"]}
        right = {"path": "right.txt", "samples": ["m02_p01_s00", "m03_p01_s00"], "patients": ["m02_p01", "m03_p01"]}
        overlap = self.pre._pair_overlap(left, right)
        self.assertEqual(overlap["sample_overlap_ids"], ["m02_p01_s00"])
        self.assertEqual(overlap["patient_overlap_ids"], ["m02_p01"])

    def test_target_component_row_tracks_two_channels(self):
        inst = np.zeros((64, 64), dtype=np.uint8)
        inst[18:42, 8:20] = 1
        inst[18:42, 23:35] = 2
        _target, parts = self.topo_aux.generate_topology_target(inst, self.contract, return_parts=True)
        row = self.pre._target_component_row("m01_p01_s00", 2, parts, inst.shape)
        self.assertGreater(row["critical_foreground_count"], 0)
        self.assertGreater(row["inter_instance_separation_count"], 0)
        self.assertGreater(row["union_count"], 0)

    def test_select_lambda_keeps_primary_when_ratio_within_budget(self):
        selected, reason = self.pre.select_lambda_from_gradient_summaries(
            0.2,
            {"weighted_topology_to_semantic_x0_4_ratio": {"median": 0.18}},
            None,
        )
        self.assertEqual(selected, 0.2)
        self.assertIn("Retained lambda=0.2", reason)

    def test_select_lambda_prefers_0p1_when_primary_exceeds_budget(self):
        selected, reason = self.pre.select_lambda_from_gradient_summaries(
            0.2,
            {"weighted_topology_to_semantic_x0_4_ratio": {"median": 0.42}},
            {"weighted_topology_to_semantic_x0_4_ratio": {"median": 0.19}},
        )
        self.assertEqual(selected, 0.1)
        self.assertIn("Selected lambda=0.1", reason)

    def test_freeze_mode_audit_passes_for_current_config(self):
        audit = self.pre.audit_freeze_mode(self.cfg)
        self.assertTrue(audit["status"])
        for entry in audit["sequence"]:
            self.assertTrue(entry["encoder_eval"])
            self.assertTrue(entry["frozen_decoder_eval"])
            self.assertTrue(entry["x_0_4_train"])
            self.assertTrue(entry["segmentation_head_train"])
            self.assertTrue(entry["topology_head_train"])
            self.assertEqual(entry["frozen_bn_train_count"], 0)

    def test_write_smoke_config_uses_isolated_save_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "smoke.yaml"
            summary = self.pre.write_smoke_config(self.cfg, 0.2, out_path)
            smoke_cfg = self.topo_aux._read_yaml(out_path)
        self.assertEqual(summary["save_dir"], "training/runs/unetpp_effb3_semantic_topology_aux_finetune_100ep_cuda_smoke")
        self.assertEqual(smoke_cfg["topology_aux"]["lambda_topology"], 0.2)
        self.assertEqual(smoke_cfg["train"]["epochs"], 1)

    def test_research_eval_isolation_contract_uses_test_split_only(self):
        audit = self.pre.audit_research_eval_isolation(self.cfg)
        self.assertEqual(audit["research_eval"]["sha256"], "6936aeb7736a7a6b82f11dfcf59aa1d819efee8f5c405325af247858c81d79fd")
        self.assertEqual(audit["train_overlap"]["sample_overlap_count"], 0)
        self.assertEqual(audit["val_overlap"]["sample_overlap_count"], 0)
        self.assertEqual(audit["instance_mask_available_count"], audit["research_eval"]["sample_count"])
        self.assertEqual(audit["gt_distribution"], {"gt1": 4, "gt2": 9, "gt3": 7})


if __name__ == "__main__":
    unittest.main()
