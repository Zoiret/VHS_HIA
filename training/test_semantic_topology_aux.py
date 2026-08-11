from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch


_THIS_DIR = Path(__file__).resolve().parent


class TestSemanticTopologyAux(unittest.TestCase):
    @staticmethod
    def _mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import semantic_topology_aux as mod

        return mod

    @classmethod
    def setUpClass(cls):
        cls.mod = cls._mod()
        cls.cfg = cls.mod._read_yaml(cls.mod.REPO_ROOT / "training" / "configs" / "unetpp_effb3_semantic_topology_aux_finetune_100ep.yaml")
        cls.contract = cls.mod.TopologyTargetContract(
            separation_radius_px=8,
            narrow_width_threshold_px=12,
            include_foreground_boundary=False,
            source_split_txt="train-only",
            source_instance_root="instances",
            selection_rule="unit-test",
            train_only=True,
        )

    def _generate(self, instance_mask: np.ndarray):
        return self.mod.generate_topology_target(instance_mask.astype(np.uint8), self.contract, return_parts=True)

    def _write_sample(self, root: Path, sample_id: str, instance_mask: np.ndarray) -> None:
        image = np.full((64, 64, 3), 127, dtype=np.uint8)
        semantic = np.zeros((64, 64), dtype=np.uint8)
        semantic[instance_mask > 0] = 1
        cv2.imwrite(str((root / "images" / f"{sample_id}.png").resolve()), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str((root / "semantic_masks" / f"{sample_id}.png").resolve()), semantic)
        cv2.imwrite(str((root / "instance_root" / "instance_masks" / f"{sample_id}.png").resolve()), instance_mask.astype(np.uint8))

    def test_two_channel_target_shape(self):
        inst = np.zeros((64, 64), dtype=np.uint8)
        inst[16:36, 10:22] = 1
        target, _parts = self._generate(inst)
        self.assertEqual(tuple(target.shape), (64, 64, 2))

    def test_gt1_separation_channel_is_zero(self):
        inst = np.zeros((64, 64), dtype=np.uint8)
        inst[16:36, 8:22] = 1
        inst[16:36, 30:44] = 1
        inst[24:28, 22:30] = 1
        target, parts = self._generate(inst)
        self.assertEqual(int(target[..., 1].sum()), 0)
        self.assertEqual(int(parts["inter_instance_separation"].sum()), 0)

    def test_narrow_foreground_channel_marks_critical_bridge(self):
        inst = np.zeros((64, 64), dtype=np.uint8)
        inst[20:44, 10:18] = 1
        inst[20:44, 30:38] = 1
        inst[30:34, 18:30] = 1
        _target, parts = self._generate(inst)
        self.assertGreater(int(parts["critical_foreground"][30:34, 19:29].sum()), 0)

    def test_distinct_instance_separation_marks_background_only(self):
        inst = np.zeros((64, 64), dtype=np.uint8)
        inst[18:42, 10:24] = 1
        inst[18:42, 27:41] = 2
        _target, parts = self._generate(inst)
        sep = parts["inter_instance_separation"]
        self.assertGreater(int(sep.sum()), 0)
        self.assertEqual(int(np.count_nonzero(sep[inst > 0])), 0)
        self.assertGreater(int(sep[18:42, 24:27].sum()), 0)

    def test_same_instance_disconnected_fragments_do_not_create_inter_instance_separation(self):
        inst = np.zeros((64, 64), dtype=np.uint8)
        inst[16:28, 10:20] = 1
        inst[16:28, 25:35] = 1
        _target, parts = self._generate(inst)
        self.assertEqual(int(parts["inter_instance_separation"].sum()), 0)

    def test_choose_contract_is_train_only_and_derives_radius_from_train_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ("images", "semantic_masks", "instance_root/instance_masks"):
                (root / rel).mkdir(parents=True, exist_ok=True)
            train_split = root / "train.txt"

            train_a = np.zeros((64, 64), dtype=np.uint8)
            train_a[14:34, 8:20] = 1
            train_a[14:34, 29:41] = 2
            self._write_sample(root, "train_a", train_a)

            train_b = np.zeros((64, 64), dtype=np.uint8)
            train_b[18:46, 12:18] = 1
            self._write_sample(root, "train_b", train_b)

            train_split.write_text(
                "images/train_a.png\tsemantic_masks/train_a.png\n"
                "images/train_b.png\tsemantic_masks/train_b.png\n",
                encoding="utf-8",
            )
            contract, audit = self.mod.choose_topology_target_contract(
                dataset_root=root,
                train_split_txt=train_split,
                instance_root=root / "instance_root",
            )
        self.assertTrue(contract.train_only)
        self.assertEqual(contract.narrow_width_threshold_px, 12)
        self.assertIn(contract.separation_radius_px, {6, 8})
        self.assertIn("radius_8", audit["separation_radius_candidates"])
        self.assertIn("radius_6", audit["separation_radius_candidates"])
        self.assertNotIn("val", inspect.signature(self.mod.choose_topology_target_contract).parameters)

    def test_topology_head_output_shape_is_two_channel(self):
        torch.manual_seed(0)
        model = self.mod.build_model_from_cfg(self.cfg).eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 128, 128))
        self.assertEqual(tuple(outputs["semantic_logits"].shape), (1, 3, 128, 128))
        self.assertEqual(tuple(outputs["topology_logits"].shape), (1, 2, 128, 128))

    def test_frozen_encoder_contract(self):
        model = self.mod.build_model_from_cfg(self.cfg)
        freeze_info = self.mod.apply_training_policy(model, self.cfg)
        self.mod.set_train_modes(model, freeze_info)
        self.assertTrue(all(not param.requires_grad for name, param in model.named_parameters() if name.startswith("base.encoder.")))
        self.assertTrue(all(param.requires_grad for name, param in model.named_parameters() if name.startswith("base.segmentation_head.")))
        self.assertTrue(all(param.requires_grad for name, param in model.named_parameters() if name.startswith("topology_head.")))
        self.assertTrue(all(param.requires_grad for name, param in model.named_parameters() if name.startswith("base.decoder.blocks.x_0_4.")))
        self.assertTrue(model.topology_head.training)
        self.assertTrue(model.base.segmentation_head.training)
        self.assertFalse(model.base.encoder.training)

    def test_validation_contract_is_post_training_only_and_not_checkpoint_selection(self):
        contract = self.mod.build_validation_contract(self.mod.DEFAULT_SEMANTIC_TEST_SPLIT)
        self.assertTrue(contract["post_training_only"])
        self.assertTrue(contract["oracle_k_analysis_only"])
        self.assertEqual(contract["checkpoint_rules"], {"best_mean_fg.pth": "highest semantic mean_dice_fg"})
        self.assertIn("checkpoint_selection", contract["prohibited_training_uses"])

    def test_authoritative_holdout_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample": "s0",
                        "present_in_authoritative_106_holdout": True,
                        "image_rel": "images/s0.png",
                        "instance_mask_rel": "instance_masks/s0.png",
                        "image_height": 64,
                        "image_width": 64,
                        "gt_instance_count": 1,
                        "patient_id": "p0",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                self.mod.evaluate_oracle_k_reconstruction(
                    object(),
                    manifest_path=manifest,
                    image_root=Path(tmp),
                    device=torch.device("cpu"),
                    use_amp=False,
                )

    def test_config_isolation_and_full_run_dir_untouched(self):
        train_cfg = self.cfg["train"]
        notes = self.cfg["experiment_notes"]
        full_run_dir = self.mod.REPO_ROOT / train_cfg["save_dir"]
        self.assertEqual(train_cfg["save_dir"], "training/runs/unetpp_effb3_semantic_topology_aux_finetune_100ep")
        self.assertFalse(full_run_dir.exists())
        self.assertTrue(notes["no_authoritative_holdout"])
        self.assertTrue(notes["no_centerhead_weights"])
        self.assertFalse(notes["full_training_launched_in_prep"])
        self.assertTrue(notes["post_training_research_eval_only"])


if __name__ == "__main__":
    unittest.main()
