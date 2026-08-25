from __future__ import annotations

import json
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

    def test_bridge_target_depends_on_semantic_gt_mask(self):
        bridge, _soft = self._mods()
        candidate = np.array([[1, 1], [1, 0]], dtype=np.uint8)
        gt_inst = np.array([[1, 0], [2, 0]], dtype=np.uint8)
        gt_sem_a = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        gt_sem_b = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        target_a = bridge.false_bridge_pixels_from_candidate(gt_sem_u8=gt_sem_a, gt_inst_u8=gt_inst, candidate_mask01=candidate)
        target_b = bridge.false_bridge_pixels_from_candidate(gt_sem_u8=gt_sem_b, gt_inst_u8=gt_inst, candidate_mask01=candidate)
        self.assertFalse(np.array_equal(target_a, target_b))
        self.assertGreater(int(np.sum(target_a)), int(np.sum(target_b)))

    def test_semantic_inference_amp_is_disabled_for_v2(self):
        bridge, _soft = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml")
        self.assertFalse(bridge._semantic_inference_amp_enabled(cfg, torch.device("cuda")))
        self.assertFalse(bridge._semantic_inference_amp_enabled(cfg, torch.device("cpu")))
        backend = bridge.semantic_inference_backend_summary(cfg, torch.device("cpu"))
        self.assertFalse(backend["amp_requested"])
        self.assertFalse(backend["amp_enabled"])
        self.assertFalse(backend["matmul_allow_tf32"])
        self.assertFalse(backend["cudnn_allow_tf32"])
        self.assertFalse(backend["cudnn_benchmark"])
        self.assertTrue(backend["cudnn_deterministic"])

    def test_bridge_training_amp_can_remain_enabled_independently(self):
        bridge, _soft = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml")
        self.assertTrue(bridge._amp_enabled(cfg, torch.device("cuda")))
        self.assertFalse(bridge._semantic_inference_amp_enabled(cfg, torch.device("cuda")))

    def test_semantic_inference_backend_ctx_applies_locked_flags(self):
        bridge, _soft = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        with bridge._semantic_inference_backend_ctx(cfg, torch.device("cpu")):
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cudnn.deterministic)

    def test_semantic_inference_backend_ctx_restores_prior_flags(self):
        bridge, _soft = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml")
        original = (
            bool(torch.backends.cuda.matmul.allow_tf32),
            bool(torch.backends.cudnn.allow_tf32),
            bool(torch.backends.cudnn.benchmark),
            bool(torch.backends.cudnn.deterministic),
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        with bridge._semantic_inference_backend_ctx(cfg, torch.device("cpu")):
            pass
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(torch.backends.cudnn.allow_tf32)
        self.assertTrue(torch.backends.cudnn.benchmark)
        self.assertFalse(torch.backends.cudnn.deterministic)
        torch.backends.cuda.matmul.allow_tf32 = original[0]
        torch.backends.cudnn.allow_tf32 = original[1]
        torch.backends.cudnn.benchmark = original[2]
        torch.backends.cudnn.deterministic = original[3]

    def test_semantic_inference_backend_ctx_restores_flags_on_error(self):
        bridge, _soft = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml")
        original = (
            bool(torch.backends.cuda.matmul.allow_tf32),
            bool(torch.backends.cudnn.allow_tf32),
            bool(torch.backends.cudnn.benchmark),
            bool(torch.backends.cudnn.deterministic),
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        with self.assertRaises(RuntimeError):
            with bridge._semantic_inference_backend_ctx(cfg, torch.device("cpu")):
                raise RuntimeError("boom")
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(torch.backends.cudnn.allow_tf32)
        self.assertTrue(torch.backends.cudnn.benchmark)
        self.assertFalse(torch.backends.cudnn.deterministic)
        torch.backends.cuda.matmul.allow_tf32 = original[0]
        torch.backends.cudnn.allow_tf32 = original[1]
        torch.backends.cudnn.benchmark = original[2]
        torch.backends.cudnn.deterministic = original[3]

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

    def test_negative_preservation_loss_uses_only_bridge_negative_samples(self):
        bridge, _soft = self._mods()
        loss_fn = bridge.CandidateBalancedBCEDiceLoss(lambda_negative_mean=1.0, lambda_negative_hard=0.0, negative_hard_topk_fraction=0.5)
        logits = torch.tensor(
            [
                [[[5.0, -5.0]]],
                [[[1.0, -1.0]]],
            ],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [
                [[[1.0, 0.0]]],
                [[[0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        candidate = torch.ones_like(target)
        out = loss_fn(
            bridge_logits=logits,
            bridge_target=target,
            candidate_mask=candidate,
            bridge_positive=torch.tensor([1.0, 0.0], dtype=torch.float32),
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[1, 0], torch.zeros_like(logits[1, 0]), reduction="mean")
        self.assertAlmostEqual(float(out["negative_candidate_mean_bce"].item()), float(expected.item()), places=6)

    def test_negative_preservation_loss_is_normalized_per_sample(self):
        bridge, _soft = self._mods()
        loss_fn = bridge.CandidateBalancedBCEDiceLoss(lambda_negative_mean=1.0, lambda_negative_hard=0.0, negative_hard_topk_fraction=0.5)
        logits = torch.tensor(
            [
                [[[2.0, 2.0, 2.0, 2.0]]],
                [[[0.0, 0.0, 0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        target = torch.zeros_like(logits)
        candidate = torch.tensor(
            [
                [[[1.0, 1.0, 1.0, 1.0]]],
                [[[1.0, 0.0, 0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        out = loss_fn(
            bridge_logits=logits,
            bridge_target=target,
            candidate_mask=candidate,
            bridge_positive=torch.tensor([0.0, 0.0], dtype=torch.float32),
        )
        loss_a = torch.nn.functional.binary_cross_entropy_with_logits(logits[0, 0, 0, :4], torch.zeros((4,), dtype=torch.float32), reduction="mean")
        loss_b = torch.nn.functional.binary_cross_entropy_with_logits(logits[1, 0, 0, :1], torch.zeros((1,), dtype=torch.float32), reduction="mean")
        expected = 0.5 * (loss_a + loss_b)
        self.assertAlmostEqual(float(out["negative_candidate_mean_bce"].item()), float(expected.item()), places=6)

    def test_negative_hard_topk_selects_highest_scoring_candidate_logits(self):
        bridge, _soft = self._mods()
        loss_fn = bridge.CandidateBalancedBCEDiceLoss(lambda_negative_mean=0.0, lambda_negative_hard=1.0, negative_hard_topk_fraction=0.25)
        logits = torch.tensor([[[[-3.0, 2.0, 5.0, 1.0]]]], dtype=torch.float32)
        target = torch.zeros_like(logits)
        candidate = torch.ones_like(logits)
        out = loss_fn(
            bridge_logits=logits,
            bridge_target=target,
            candidate_mask=candidate,
            bridge_positive=torch.tensor([0.0], dtype=torch.float32),
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([5.0]), torch.tensor([0.0]), reduction="mean")
        self.assertAlmostEqual(float(out["negative_candidate_hard_bce"].item()), float(expected.item()), places=6)

    def test_all_zero_bridge_target_has_finite_gradients(self):
        bridge, _soft = self._mods()
        loss_fn = bridge.CandidateBalancedBCEDiceLoss(lambda_negative_mean=2.0, lambda_negative_hard=1.0, negative_hard_topk_fraction=0.5)
        logits = torch.zeros((2, 1, 2, 2), dtype=torch.float32, requires_grad=True)
        target = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
        candidate = torch.ones((2, 1, 2, 2), dtype=torch.float32)
        out = loss_fn(
            bridge_logits=logits,
            bridge_target=target,
            candidate_mask=candidate,
            bridge_positive=torch.tensor([0.0, 0.0], dtype=torch.float32),
        )
        out["loss"].backward()
        self.assertTrue(torch.isfinite(out["loss"]).all().item())
        self.assertTrue(torch.isfinite(logits.grad).all().item())

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

    def test_target_construction_ignores_train_amp_flag(self):
        bridge, _soft = self._mods()
        base_cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml")
        cfg_true = json.loads(json.dumps(base_cfg))
        cfg_false = json.loads(json.dumps(base_cfg))
        cfg_true["train"]["amp"] = True
        cfg_false["train"]["amp"] = False

        fake_item = {
            "sample_id": "m00_p00_s00",
            "patient_id": "m00_p00",
            "image_path": "fake_image.png",
            "mask_path": "fake_mask.png",
            "instance_path": "fake_inst.png",
        }
        full_rgb = np.zeros((768, 768, 3), dtype=np.uint8)
        full_sem = np.zeros((768, 768), dtype=np.uint8)
        full_inst = np.zeros((768, 768), dtype=np.uint8)
        full_inst[:, :384] = 1
        full_inst[:, 384:] = 2
        full_sem[:, :384] = 1
        p_leaf = np.full((1, 1, 768, 768), 0.25, dtype=np.float32)
        p_leaf[:, :, 100:140, 370:398] = 0.75
        x_0_4 = torch.zeros((1, 16, 768, 768), dtype=torch.float32)
        x_2_2 = torch.zeros((1, 32, 192, 192), dtype=torch.float32)

        class FakeModel:
            def eval(self):
                return self

            def __call__(self, image_batch):
                return {
                    "p_leaf": torch.from_numpy(p_leaf).to(image_batch.device),
                    "x_0_4": x_0_4.to(image_batch.device),
                    "x_2_2": x_2_2.to(image_batch.device),
                }

        with mock.patch.object(bridge, "_build_split_items", return_value=[fake_item]), \
             mock.patch.object(bridge, "_load_image_rgb", return_value=full_rgb.copy()), \
             mock.patch.object(bridge, "_load_u8", side_effect=[full_sem.copy(), full_inst.copy(), full_sem.copy(), full_inst.copy()]), \
             mock.patch.object(bridge, "run_locked_reconstruction", return_value={"labels": np.zeros((768, 768), dtype=np.uint8)}):
            rec_true = bridge.mine_bridge_records_for_split(
                cfg=cfg_true,
                split_txt=bridge.DEFAULT_TRAIN_SPLIT,
                model=FakeModel(),
                device=torch.device("cpu"),
                cache_features=True,
                selected_sample_ids={"m00_p00_s00"},
            )[0]
            rec_false = bridge.mine_bridge_records_for_split(
                cfg=cfg_false,
                split_txt=bridge.DEFAULT_TRAIN_SPLIT,
                model=FakeModel(),
                device=torch.device("cpu"),
                cache_features=True,
                selected_sample_ids={"m00_p00_s00"},
            )[0]
        self.assertEqual(rec_true["candidate_pixels"], rec_false["candidate_pixels"])
        self.assertEqual(rec_true["bridge_pixels"], rec_false["bridge_pixels"])
        self.assertTrue(np.array_equal(rec_true["candidate_mask"], rec_false["candidate_mask"]))
        self.assertTrue(np.array_equal(rec_true["bridge_target"], rec_false["bridge_target"]))

    def test_target_construction_uses_semantic_inference_amp_not_train_amp(self):
        bridge, _soft = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml")
        fake_item = {
            "sample_id": "m00_p00_s00",
            "patient_id": "m00_p00",
            "image_path": "fake_image.png",
            "mask_path": "fake_mask.png",
            "instance_path": "fake_inst.png",
        }

        class FakeModel:
            def eval(self):
                return self

            def __call__(self, image_batch):
                return {
                    "p_leaf": torch.zeros((1, 1, 768, 768), dtype=torch.float32, device=image_batch.device),
                    "x_0_4": torch.zeros((1, 16, 768, 768), dtype=torch.float32, device=image_batch.device),
                    "x_2_2": torch.zeros((1, 32, 192, 192), dtype=torch.float32, device=image_batch.device),
                }

        calls: list[bool] = []

        def fake_autocast(device, enabled):
            calls.append(bool(enabled))
            return bridge.contextlib.nullcontext()

        with mock.patch.object(bridge, "_build_split_items", return_value=[fake_item]), \
             mock.patch.object(bridge, "_load_image_rgb", return_value=np.zeros((768, 768, 3), dtype=np.uint8)), \
             mock.patch.object(bridge, "_load_u8", side_effect=[np.zeros((768, 768), dtype=np.uint8), np.zeros((768, 768), dtype=np.uint8)]), \
             mock.patch.object(bridge, "run_locked_reconstruction", return_value={"labels": np.zeros((768, 768), dtype=np.uint8)}), \
             mock.patch.object(bridge, "_autocast_ctx", side_effect=fake_autocast):
            bridge.mine_bridge_records_for_split(
                cfg=cfg,
                split_txt=bridge.DEFAULT_TRAIN_SPLIT,
                model=FakeModel(),
                device=torch.device("cpu"),
                cache_features=False,
                selected_sample_ids={"m00_p00_s00"},
            )
        self.assertEqual(calls, [False])

    def test_target_construction_uses_locked_backend_contract_not_ambient_flags(self):
        bridge, _soft = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml")
        fake_item = {
            "sample_id": "m00_p00_s00",
            "patient_id": "m00_p00",
            "image_path": "fake_image.png",
            "mask_path": "fake_mask.png",
            "instance_path": "fake_inst.png",
        }
        seen: list[tuple[bool, bool, bool, bool]] = []

        class FakeModel:
            def eval(self):
                return self

            def __call__(self, image_batch):
                seen.append((
                    bool(torch.backends.cuda.matmul.allow_tf32),
                    bool(torch.backends.cudnn.allow_tf32),
                    bool(torch.backends.cudnn.benchmark),
                    bool(torch.backends.cudnn.deterministic),
                ))
                return {
                    "p_leaf": torch.zeros((1, 1, 768, 768), dtype=torch.float32, device=image_batch.device),
                    "x_0_4": torch.zeros((1, 16, 768, 768), dtype=torch.float32, device=image_batch.device),
                    "x_2_2": torch.zeros((1, 32, 192, 192), dtype=torch.float32, device=image_batch.device),
                }

        original = (
            bool(torch.backends.cuda.matmul.allow_tf32),
            bool(torch.backends.cudnn.allow_tf32),
            bool(torch.backends.cudnn.benchmark),
            bool(torch.backends.cudnn.deterministic),
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        try:
            with mock.patch.object(bridge, "_build_split_items", return_value=[fake_item]), \
                 mock.patch.object(bridge, "_load_image_rgb", return_value=np.zeros((768, 768, 3), dtype=np.uint8)), \
                 mock.patch.object(bridge, "_load_u8", side_effect=[np.zeros((768, 768), dtype=np.uint8), np.zeros((768, 768), dtype=np.uint8)]), \
                 mock.patch.object(bridge, "run_locked_reconstruction", return_value={"labels": np.zeros((768, 768), dtype=np.uint8)}):
                bridge.mine_bridge_records_for_split(
                    cfg=cfg,
                    split_txt=bridge.DEFAULT_TRAIN_SPLIT,
                    model=FakeModel(),
                    device=torch.device("cpu"),
                    cache_features=False,
                    selected_sample_ids={"m00_p00_s00"},
                )
            self.assertEqual(seen, [(False, False, False, True)])
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.backends.cudnn.benchmark)
            self.assertFalse(torch.backends.cudnn.deterministic)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original[0]
            torch.backends.cudnn.allow_tf32 = original[1]
            torch.backends.cudnn.benchmark = original[2]
            torch.backends.cudnn.deterministic = original[3]

    def test_selected_sample_order_is_preserved_for_locked_manifest_resolution(self):
        bridge, _soft = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml")
        items = [
            {"sample_id": "b", "patient_id": "pb", "image_path": "b_img.png", "mask_path": "b_mask.png", "instance_path": "b_inst.png"},
            {"sample_id": "a", "patient_id": "pa", "image_path": "a_img.png", "mask_path": "a_mask.png", "instance_path": "a_inst.png"},
        ]

        class FakeModel:
            def eval(self):
                return self

            def __call__(self, image_batch):
                bsz = int(image_batch.shape[0])
                return {
                    "p_leaf": torch.zeros((bsz, 1, 768, 768), dtype=torch.float32, device=image_batch.device),
                    "x_0_4": torch.zeros((bsz, 16, 768, 768), dtype=torch.float32, device=image_batch.device),
                    "x_2_2": torch.zeros((bsz, 32, 192, 192), dtype=torch.float32, device=image_batch.device),
                }

        zeros = np.zeros((768, 768), dtype=np.uint8)
        with mock.patch.object(bridge, "_build_split_items", return_value=items), \
             mock.patch.object(bridge, "_load_image_rgb", return_value=np.zeros((768, 768, 3), dtype=np.uint8)), \
             mock.patch.object(bridge, "_load_u8", side_effect=[zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()]), \
             mock.patch.object(bridge, "run_locked_reconstruction", return_value={"labels": zeros.copy()}):
            records = bridge.mine_bridge_records_for_split(
                cfg=cfg,
                split_txt=bridge.DEFAULT_TRAIN_SPLIT,
                model=FakeModel(),
                device=torch.device("cpu"),
                cache_features=False,
                selected_sample_ids=["a", "b"],
            )
        self.assertEqual([row["sample_id"] for row in records], ["a", "b"])

    def test_locked_v2_manifest_counts(self):
        bridge, _soft = self._mods()
        payload = bridge.read_locked_micro_manifest(bridge.MICRO_MANIFEST_V2_PATH)
        self.assertEqual(payload["source_split"], "datasets/converted_full_multiclass_curated/train.txt")
        self.assertEqual(payload["source_split_canonical_sha256"], "f5e920ffaf54c0a0034c457cf3c951f71e186a9f35e3fe67a5eee95737b2ee82")
        summary = bridge.summarize_manifest_expectations(payload)
        self.assertEqual(summary["expected_sample_count"], 10)
        self.assertEqual(summary["expected_positive_count"], 6)
        self.assertEqual(summary["expected_negative_count"], 4)
        rows = payload["rows"]
        self.assertEqual(sum(1 for row in rows if int(row["bridge_positive"]) == 1 and int(row["gt_count"]) == 2), 3)
        self.assertEqual(sum(1 for row in rows if int(row["bridge_positive"]) == 1 and int(row["gt_count"]) == 3), 3)
        self.assertEqual(sum(1 for row in rows if int(row["bridge_positive"]) == 0 and int(row["gt_count"]) == 2), 2)
        self.assertEqual(sum(1 for row in rows if int(row["bridge_positive"]) == 0 and int(row["gt_count"]) == 3), 2)

    def test_portable_repo_relative_manifest_path_resolves_on_posix_and_windows_logic(self):
        bridge, _soft = self._mods()
        rel = "datasets/converted_full_multiclass_curated/train.txt"
        posix_text = bridge._portable_repo_path_text(rel, repo_root="/repo/project", platform_name="posix")
        windows_text = bridge._portable_repo_path_text(rel, repo_root=r"E:\repo\project", platform_name="nt")
        self.assertEqual(posix_text, "/repo/project/datasets/converted_full_multiclass_curated/train.txt")
        self.assertEqual(windows_text, r"E:\repo\project\datasets\converted_full_multiclass_curated\train.txt")

    def test_validate_locked_manifest_source_split_passes_for_relative_train_path_and_sha(self):
        bridge, _soft = self._mods()
        payload = {
            "_manifest_path": str(bridge.MICRO_MANIFEST_V2_PATH),
            "source_split": "datasets/converted_full_multiclass_curated/train.txt",
            "source_split_canonical_sha256": "f5e920ffaf54c0a0034c457cf3c951f71e186a9f35e3fe67a5eee95737b2ee82",
            "sample_ids": ["a"],
            "rows": [{"sample_id": "a"}],
        }
        summary = bridge.validate_locked_manifest_source_split(
            manifest_payload=payload,
            configured_train_split=bridge.DEFAULT_TRAIN_SPLIT,
        )
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["resolved_source_split"], str(bridge.DEFAULT_TRAIN_SPLIT.resolve()))
        self.assertEqual(summary["actual_source_split_canonical_sha256"], payload["source_split_canonical_sha256"])

    def test_validate_locked_manifest_source_split_blocks_wrong_relative_split(self):
        bridge, _soft = self._mods()
        payload = {
            "_manifest_path": str(bridge.MICRO_MANIFEST_V2_PATH),
            "source_split": "datasets/converted_full_multiclass_curated/val.txt",
            "source_split_canonical_sha256": "b9b4151c48fe7824be1a3e25c4f519ebadf6353a58f717f9679fde2ca8ff376c",
            "sample_ids": ["a"],
            "rows": [{"sample_id": "a"}],
        }
        summary = bridge.validate_locked_manifest_source_split(
            manifest_payload=payload,
            configured_train_split=bridge.DEFAULT_TRAIN_SPLIT,
        )
        self.assertEqual(summary["status"], "blocked")
        self.assertIn("Locked micro manifest must use train split only", str(summary["error"]))

    def test_validate_locked_manifest_source_split_blocks_wrong_absolute_path(self):
        bridge, _soft = self._mods()
        payload = {
            "_manifest_path": str(bridge.MICRO_MANIFEST_V2_PATH),
            "source_split": str(bridge.DEFAULT_VAL_SPLIT.resolve()),
            "source_split_canonical_sha256": "b9b4151c48fe7824be1a3e25c4f519ebadf6353a58f717f9679fde2ca8ff376c",
            "sample_ids": ["a"],
            "rows": [{"sample_id": "a"}],
        }
        summary = bridge.validate_locked_manifest_source_split(
            manifest_payload=payload,
            configured_train_split=bridge.DEFAULT_TRAIN_SPLIT,
        )
        self.assertEqual(summary["status"], "blocked")
        self.assertIn(str(bridge.DEFAULT_VAL_SPLIT.resolve()), str(summary["error"]))

    def test_windows_absolute_manifest_path_is_not_reinterpreted_on_posix(self):
        bridge, _soft = self._mods()
        with self.assertRaises(ValueError):
            bridge._portable_repo_path_text(
                r"E:\3d_visual\ml\datasets\converted_full_multiclass_curated\train.txt",
                repo_root="/home/user/repo",
                platform_name="posix",
            )

    def test_validate_locked_manifest_source_split_blocks_sha_mismatch(self):
        bridge, _soft = self._mods()
        payload = {
            "_manifest_path": str(bridge.MICRO_MANIFEST_V2_PATH),
            "source_split": "datasets/converted_full_multiclass_curated/train.txt",
            "source_split_canonical_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "sample_ids": ["a"],
            "rows": [{"sample_id": "a"}],
        }
        summary = bridge.validate_locked_manifest_source_split(
            manifest_payload=payload,
            configured_train_split=bridge.DEFAULT_TRAIN_SPLIT,
        )
        self.assertEqual(summary["status"], "blocked")
        self.assertIn("TRAIN canonical SHA256 mismatch", str(summary["error"]))

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
                    cache_features=False,
                    selected_sample_ids={"present", "missing"},
                )

    def test_validate_locked_micro_records_blocks_on_mismatch(self):
        bridge, _soft = self._mods()
        payload = {
            "_manifest_path": str(bridge.MICRO_MANIFEST_V2_PATH),
            "source_split": "datasets/converted_full_multiclass_curated/train.txt",
            "source_split_canonical_sha256": "f5e920ffaf54c0a0034c457cf3c951f71e186a9f35e3fe67a5eee95737b2ee82",
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
