from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


_THIS_DIR = Path(__file__).resolve().parent


class TestBridgeSuppressionHead(unittest.TestCase):
    @staticmethod
    def _mods():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import bridge_suppression_head as bridge
        import audit_semantic_soft_logit_recoverability as soft_audit

        return bridge, soft_audit

    def test_exact_false_bridge_equivalence(self):
        bridge, soft_audit = self._mods()
        gt_sem = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        gt_inst = np.array([[1, 0], [2, 0]], dtype=np.uint8)
        candidate = np.array([[1, 1], [1, 0]], dtype=np.uint8)
        contract = mock.Mock()
        with mock.patch.object(soft_audit, "_critical_foreground_mask", return_value=np.zeros_like(gt_sem, dtype=np.uint8)):
            ref = soft_audit._topology_pixel_categories(
                gt_sem_u8=gt_sem,
                gt_inst_u8=gt_inst,
                pred_sem_u8=candidate,
                pred_union01=candidate,
                topology_contract=contract,
            )["FALSE_BRIDGE_PIXELS"]
        got = bridge.false_bridge_pixels_from_candidate(gt_sem_u8=gt_sem, gt_inst_u8=gt_inst, candidate_mask01=candidate)
        self.assertTrue(np.array_equal(got, ref))

    def test_target_positives_subset_of_candidate_and_gt_non_leaflet(self):
        bridge, _soft = self._mods()
        gt_sem = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        gt_inst = np.array([[1, 0], [2, 0]], dtype=np.uint8)
        candidate = np.array([[1, 1], [1, 0]], dtype=np.uint8)
        got = bridge.false_bridge_pixels_from_candidate(gt_sem_u8=gt_sem, gt_inst_u8=gt_inst, candidate_mask01=candidate)
        self.assertTrue(np.all(got <= candidate))
        self.assertEqual(int(np.sum((got > 0) & (gt_sem == 1))), 0)

    def test_bridge_negative_target_is_all_zero(self):
        bridge, _soft = self._mods()
        gt_sem = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        gt_inst = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        candidate = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        got = bridge.false_bridge_pixels_from_candidate(gt_sem_u8=gt_sem, gt_inst_u8=gt_inst, candidate_mask01=candidate)
        self.assertEqual(int(np.sum(got)), 0)

    def test_candidate_masked_loss_and_zero_positive_safety(self):
        bridge, _soft = self._mods()
        loss_fn = bridge.CandidateBalancedBCEDiceLoss()
        logits = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        target = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        candidate = torch.tensor([[[[1, 0], [1, 0]]]], dtype=torch.float32)
        out = loss_fn(bridge_logits=logits, bridge_target=target, candidate_mask=candidate)
        self.assertTrue(torch.isfinite(out["loss"]).all().item())
        self.assertEqual(float(out["positive_count"].item()), 0.0)
        self.assertEqual(float(out["candidate_count"].item()), 2.0)

    def test_no_gt_labels_enter_inference_and_removal_only(self):
        bridge, _soft = self._mods()
        candidate = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        probs = np.array([[0.6, 0.4], [0.9, 0.8]], dtype=np.float32)
        refined = bridge.refine_candidate_with_bridge_probs(candidate, probs, threshold=0.50)
        self.assertTrue(np.all(refined <= candidate))
        self.assertTrue(np.array_equal(refined, np.array([[0, 1], [0, 0]], dtype=np.uint8)))

    def test_locked_k_normalizer_unchanged(self):
        bridge, _soft = self._mods()
        self.assertEqual(bridge.run_locked_reconstruction.__defaults__, None)
        self.assertEqual(bridge.postrun.NORMALIZER_METHOD, "centroid_distance_k_normalizer")

    def test_safe_paths_reject_center_full_val_and_holdout(self):
        bridge, _soft = self._mods()
        with self.assertRaises(SystemExit):
            bridge._assert_safe_path(Path("training/manifests/center_full_val_manifest.jsonl"))
        with self.assertRaises(SystemExit):
            bridge._assert_safe_path(Path("authoritative_106_holdout.txt"))

    def test_frozen_semantic_model_and_feature_taps(self):
        bridge, _soft = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit.yaml")
        model = bridge.build_model_from_cfg(cfg)
        self.assertTrue(all(not p.requires_grad for n, p in model.named_parameters() if n.startswith("base.")))
        self.assertTrue(any(p.requires_grad for n, p in model.named_parameters() if n.startswith("context_projection.")))
        self.assertTrue(any(p.requires_grad for n, p in model.named_parameters() if n.startswith("bridge_head.")))
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out["bridge_logits"].shape[:2]), (1, 1))
        self.assertEqual(tuple(out["candidate_mask"].shape[:2]), (1, 1))
        self.assertEqual(int(out["x_0_4"].shape[2]), 128)
        self.assertGreater(int(out["x_2_2"].shape[1]), 0)

    def test_validation_contract_uses_val_not_test(self):
        bridge, _soft = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit.yaml")
        dataset_cfg = cfg.get("dataset") or {}
        self.assertIn("val.txt", str(dataset_cfg.get("val_txt")))
        self.assertIn("test.txt", str(dataset_cfg.get("test_txt")))
        self.assertNotEqual(str(dataset_cfg.get("val_txt")), str(dataset_cfg.get("test_txt")))

    def test_future_full_run_not_created(self):
        bridge, _soft = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit.yaml")
        future_full = bridge._resolve_repo_path((cfg.get("reserved_full_run") or {}).get("save_dir"), bridge.REPO_ROOT / "training" / "runs" / "bridge_suppression_full")
        self.assertFalse(future_full.exists())


if __name__ == "__main__":
    unittest.main()
