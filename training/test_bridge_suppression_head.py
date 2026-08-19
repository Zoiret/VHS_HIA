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

    def test_bridge_contract_input_size_is_fixed_768(self):
        bridge, _soft = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit.yaml")
        self.assertEqual(bridge._bridge_input_hw_from_cfg(cfg), (768, 768))

    def test_bridge_target_is_generated_in_final_crop_coordinates(self):
        bridge, _soft = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit.yaml")

        full_rgb = np.zeros((1024, 1024, 3), dtype=np.uint8)
        full_sem = np.zeros((1024, 1024), dtype=np.uint8)
        full_inst = np.zeros((1024, 1024), dtype=np.uint8)
        full_sem[128:896, 128:896] = 7
        full_inst[128:896, 128:896] = 9
        cropped_sem = full_sem[128:896, 128:896]
        cropped_inst = full_inst[128:896, 128:896]

        class FakeModel:
            def eval(self):
                return self

            def __call__(self, image_batch):
                self.last_image_shape = tuple(image_batch.shape)
                bsz = int(image_batch.shape[0])
                return {
                    "p_leaf": torch.ones((bsz, 1, 768, 768), dtype=torch.float32),
                    "x_0_4": torch.zeros((bsz, 16, 768, 768), dtype=torch.float32),
                    "x_2_2": torch.zeros((bsz, 32, 192, 192), dtype=torch.float32),
                }

        model = FakeModel()
        fake_item = {
            "sample_id": "m00_p00_s00",
            "patient_id": "m00_p00",
            "image_path": "fake_image.png",
            "mask_path": "fake_mask.png",
            "instance_path": "fake_inst.png",
        }

        def fake_load_u8(path: Path):
            path_s = str(path)
            if "inst" in path_s:
                return full_inst.copy()
            return full_sem.copy()

        def fake_false_bridge(*, gt_sem_u8, gt_inst_u8, candidate_mask01):
            self.assertEqual(gt_sem_u8.shape, (768, 768))
            self.assertEqual(gt_inst_u8.shape, (768, 768))
            self.assertTrue(np.array_equal(gt_sem_u8, cropped_sem))
            self.assertTrue(np.array_equal(gt_inst_u8, cropped_inst))
            self.assertEqual(candidate_mask01.shape, (768, 768))
            return np.zeros_like(candidate_mask01, dtype=np.uint8)

        with mock.patch.object(bridge, "_build_split_items", return_value=[fake_item]), \
             mock.patch.object(bridge, "_load_image_rgb", return_value=full_rgb.copy()), \
             mock.patch.object(bridge, "_load_u8", side_effect=fake_load_u8), \
             mock.patch.object(bridge, "false_bridge_pixels_from_candidate", side_effect=fake_false_bridge), \
             mock.patch.object(bridge, "run_locked_reconstruction", return_value={"labels": np.zeros((768, 768), dtype=np.uint8)}):
            records = bridge.mine_bridge_records_for_split(
                cfg=cfg,
                split_txt=Path("datasets/converted_full_multiclass_curated/train.txt"),
                model=model,
                device=torch.device("cpu"),
                use_amp=False,
                cache_features=True,
                selected_sample_ids={"m00_p00_s00"},
            )
        self.assertEqual(len(records), 1)
        row = records[0]
        self.assertEqual(tuple(model.last_image_shape), (1, 3, 768, 768))
        self.assertEqual(row["gt_semantic"].shape, (768, 768))
        self.assertEqual(row["gt_instances"].shape, (768, 768))
        self.assertEqual(row["candidate_mask"].shape, (768, 768))
        self.assertEqual(row["bridge_target"].shape, (768, 768))
        self.assertEqual(tuple(row["x_0_4"].shape), (16, 768, 768))
        self.assertEqual(tuple(row["x_2_2"].shape), (32, 192, 192))
        self.assertEqual(row["bridge_contract_input_hw"], [768, 768])

    def test_locked_v2_manifest_counts(self):
        bridge, _soft = self._mods()
        payload = bridge.read_locked_micro_manifest(bridge.MICRO_MANIFEST_V2_PATH)
        summary = bridge.summarize_manifest_expectations(payload)
        self.assertEqual(summary["expected_sample_count"], 10)
        self.assertEqual(summary["expected_positive_count"], 6)
        self.assertEqual(summary["expected_negative_count"], 4)
        rows = payload["rows"]
        self.assertEqual(sum(1 for row in rows if int(row["bridge_positive"]) == 1 and int(row["gt_count"]) == 2), 3)
        self.assertEqual(sum(1 for row in rows if int(row["bridge_positive"]) == 1 and int(row["gt_count"]) == 3), 3)
        self.assertEqual(sum(1 for row in rows if int(row["bridge_positive"]) == 0 and int(row["gt_count"]) == 2), 2)
        self.assertEqual(sum(1 for row in rows if int(row["bridge_positive"]) == 0 and int(row["gt_count"]) == 3), 2)

    def test_missing_selected_sample_causes_hard_failure(self):
        bridge, _soft = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml")

        class FakeModel:
            def eval(self):
                return self

        with mock.patch.object(bridge, "_build_split_items", return_value=[{"sample_id": "present"}]):
            with self.assertRaises(SystemExit):
                bridge.mine_bridge_records_for_split(
                    cfg=cfg,
                    split_txt=bridge.DEFAULT_TRAIN_SPLIT,
                    model=FakeModel(),
                    device=torch.device("cpu"),
                    use_amp=False,
                    cache_features=False,
                    selected_sample_ids={"present", "missing"},
                )

    def test_validate_locked_micro_records_blocks_on_mismatch(self):
        bridge, _soft = self._mods()
        payload = {
            "_manifest_path": str(bridge.MICRO_MANIFEST_V2_PATH),
            "sample_ids": ["a", "b"],
            "rows": [
                {"sample_id": "a", "patient_id": "p1", "gt_count": 2, "bridge_positive": 1, "bridge_pixels": 5, "candidate_pixels": 10, "topology_changes_if_oracle_removed": 1},
                {"sample_id": "b", "patient_id": "p2", "gt_count": 3, "bridge_positive": 0, "bridge_pixels": 0, "candidate_pixels": 9, "topology_changes_if_oracle_removed": 0},
            ],
        }
        records = [
            {"sample_id": "a", "patient_id": "p1", "gt_count": 2, "bridge_positive": 1, "bridge_pixels": 5, "candidate_pixels": 10, "topology_changes_if_oracle_removed": 1},
        ]
        summary = bridge.validate_locked_micro_records(manifest_payload=payload, records=records, split_txt=bridge.DEFAULT_TRAIN_SPLIT)
        self.assertEqual(summary["status"], "blocked")
        self.assertEqual(summary["missing_ids"], ["b"])

    def test_negative_preservation_metrics_and_removal_calibration(self):
        bridge, _soft = self._mods()

        class FakeModel:
            def eval(self):
                return self

            def bridge_forward_from_cached(self, *, x_0_4, x_2_2, p_leaf):
                logits = torch.tensor(
                    [
                        [[[2.0, -2.0], [-2.0, -2.0]]],
                        [[[-2.0, -2.0], [-2.0, -2.0]]],
                    ],
                    dtype=torch.float32,
                )
                return {"bridge_logits": logits}

        records = [
            {
                "sample_id": "pos",
                "patient_id": "p1",
                "gt_count": 2,
                "bridge_positive": 1,
                "candidate_pixels": 4,
                "bridge_pixels": 2,
                "x_0_4": torch.zeros((16, 2, 2), dtype=torch.float32),
                "x_2_2": torch.zeros((32, 1, 1), dtype=torch.float32),
                "p_leaf": torch.ones((1, 2, 2), dtype=torch.float32),
                "candidate_mask": torch.ones((1, 2, 2), dtype=torch.float32),
                "bridge_target": torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32),
                "gt_instances": np.ones((2, 2), dtype=np.uint8),
                "candidate_mask_np": np.ones((2, 2), dtype=np.uint8),
                "oracle_removed_mask": np.array([[0, 0], [1, 1]], dtype=np.uint8),
                "image_path": "x",
            },
            {
                "sample_id": "neg",
                "patient_id": "p2",
                "gt_count": 3,
                "bridge_positive": 0,
                "candidate_pixels": 4,
                "bridge_pixels": 0,
                "x_0_4": torch.zeros((16, 2, 2), dtype=torch.float32),
                "x_2_2": torch.zeros((32, 1, 1), dtype=torch.float32),
                "p_leaf": torch.ones((1, 2, 2), dtype=torch.float32),
                "candidate_mask": torch.ones((1, 2, 2), dtype=torch.float32),
                "bridge_target": torch.zeros((1, 2, 2), dtype=torch.float32),
                "gt_instances": np.full((2, 2), 2, dtype=np.uint8),
                "candidate_mask_np": np.ones((2, 2), dtype=np.uint8),
                "oracle_removed_mask": np.ones((2, 2), dtype=np.uint8),
                "image_path": "y",
            },
        ]

        def fake_reconstruction(pred_leaf01, gt_inst_u8):
            marker = int(gt_inst_u8[0, 0])
            fg = int(np.sum(pred_leaf01))
            if marker == 1:
                mean_iou = {4: 0.30, 3: 0.55, 2: 0.90}[fg]
            else:
                mean_iou = {4: 0.80, 3: 0.60}.get(fg, 0.80)
            return {
                "metrics": {
                    "instance_mean_matched_iou": float(mean_iou),
                    "all_iou_ge_0.50": bool(mean_iou >= 0.50),
                }
            }

        with mock.patch.object(bridge, "run_locked_reconstruction", side_effect=fake_reconstruction):
            out = bridge.evaluate_reconstruction_levels_on_cached(FakeModel(), records, torch.device("cpu"))
        self.assertEqual(out["positive_subset"]["pixel"]["tp"], 1)
        self.assertEqual(out["negative_subset"]["predicted_bridge_pixels"], 0)
        self.assertEqual(out["negative_subset"]["samples_with_zero_predicted_removal"], 1)
        self.assertEqual(out["negative_subset"]["num_unchanged"], 1)
        self.assertEqual(out["negative_subset"]["num_component_topology_changes"], 0)
        self.assertAlmostEqual(out["removal_calibration"]["negative_removed_over_candidate"], 0.0)


if __name__ == "__main__":
    unittest.main()
