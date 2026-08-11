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
            boundary_width_px=3,
            separation_width_px=3,
            narrow_width_threshold_px=12,
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

    def test_gt1_target_has_boundary_and_narrow_without_separation(self):
        inst = np.zeros((64, 64), dtype=np.uint8)
        inst[16:36, 8:22] = 1
        inst[16:36, 30:44] = 1
        inst[24:28, 22:30] = 1
        target, parts = self._generate(inst)
        self.assertGreater(int(parts["boundary"].sum()), 0)
        self.assertEqual(int(parts["separation"].sum()), 0)
        self.assertGreater(int(parts["narrow"][24:28, 23:29].sum()), 0)
        self.assertGreater(int(target.sum()), 0)

    def test_gt2_target_marks_separation_band(self):
        inst = np.zeros((64, 64), dtype=np.uint8)
        inst[18:42, 10:24] = 1
        inst[18:42, 27:41] = 2
        target, parts = self._generate(inst)
        self.assertGreater(int(parts["separation"].sum()), 0)
        self.assertGreater(int(parts["separation"][18:42, 24:27].sum()), 0)
        self.assertGreater(int(target.sum()), 0)

    def test_gt3_target_supports_three_instances(self):
        inst = np.zeros((64, 64), dtype=np.uint8)
        inst[14:34, 6:18] = 1
        inst[14:34, 21:33] = 2
        inst[14:34, 36:48] = 3
        target, parts = self._generate(inst)
        self.assertEqual(sorted(self.mod._positive_instance_ids(inst)), [1, 2, 3])
        self.assertGreater(int(parts["boundary"].sum()), 0)
        self.assertGreater(int(parts["separation"].sum()), 0)
        self.assertGreater(int(target.sum()), 0)

    def test_narrow_foreground_rule_marks_critical_bridge(self):
        inst = np.zeros((64, 64), dtype=np.uint8)
        inst[20:44, 10:18] = 1
        inst[20:44, 30:38] = 1
        inst[30:34, 18:30] = 1
        _target, parts = self._generate(inst)
        self.assertGreater(int(parts["narrow"][30:34, 19:29].sum()), 0)

    def test_choose_contract_is_train_only_and_has_no_validation_parameter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ("images", "semantic_masks", "instance_root/instance_masks"):
                (root / rel).mkdir(parents=True, exist_ok=True)
            train_split = root / "train.txt"
            val_split = root / "val.txt"

            train_a = np.zeros((64, 64), dtype=np.uint8)
            train_a[10:26, 8:20] = 1
            train_a[10:26, 24:36] = 2
            self._write_sample(root, "train_a", train_a)

            train_b = np.zeros((64, 64), dtype=np.uint8)
            train_b[20:52, 12:18] = 1
            self._write_sample(root, "train_b", train_b)

            val_a = np.zeros((64, 64), dtype=np.uint8)
            val_a[8:56, 8:10] = 1
            val_a[8:56, 54:56] = 2
            self._write_sample(root, "val_a", val_a)

            train_split.write_text(
                "images/train_a.png\tsemantic_masks/train_a.png\n"
                "images/train_b.png\tsemantic_masks/train_b.png\n",
                encoding="utf-8",
            )
            val_split.write_text("images/val_a.png\tsemantic_masks/val_a.png\n", encoding="utf-8")

            contract, audit = self.mod.choose_topology_target_contract(
                dataset_root=root,
                train_split_txt=train_split,
                instance_root=root / "instance_root",
            )

        self.assertTrue(contract.train_only)
        self.assertEqual(contract.source_split_txt, str(train_split.resolve()))
        self.assertEqual(audit["sample_count"], 2)
        self.assertNotIn("val", inspect.signature(self.mod.choose_topology_target_contract).parameters)

    def test_topology_head_output_shape(self):
        torch.manual_seed(0)
        model = self.mod.build_model_from_cfg(self.cfg).eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 128, 128))
        self.assertEqual(tuple(outputs["semantic_logits"].shape), (1, 3, 128, 128))
        self.assertEqual(tuple(outputs["topology_logits"].shape), (1, 1, 128, 128))

    def test_frozen_encoder_contract(self):
        model = self.mod.build_model_from_cfg(self.cfg)
        freeze_info = self.mod.apply_training_policy(model, self.cfg)
        self.mod.set_train_modes(model, freeze_info)

        self.assertTrue(all(not param.requires_grad for name, param in model.named_parameters() if name.startswith("base.encoder.")))
        self.assertTrue(all(param.requires_grad for name, param in model.named_parameters() if name.startswith("base.segmentation_head.")))
        self.assertTrue(all(param.requires_grad for name, param in model.named_parameters() if name.startswith("topology_head.")))
        self.assertTrue(all(param.requires_grad for name, param in model.named_parameters() if name.startswith("base.decoder.blocks.x_0_4.")))
        self.assertTrue(all(not param.requires_grad for name, param in model.named_parameters() if name.startswith("base.decoder.blocks.x_0_3.")))
        self.assertTrue(model.topology_head.training)
        self.assertTrue(model.base.segmentation_head.training)
        self.assertFalse(model.base.encoder.training)

    def test_validation_contract_marks_oracle_k_as_analysis_only(self):
        contract = self.mod.build_validation_contract(self.mod.DEFAULT_RESEARCH_MANIFEST)
        self.assertTrue(contract["oracle_k_analysis_only"])
        self.assertFalse(contract["holdout_used"])
        self.assertEqual(contract["normalizer_method"], "centroid_distance_k_normalizer")
        self.assertIn("best_topology_reconstruction.pth", contract["checkpoint_rules"])

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

    def test_isolated_config_does_not_reuse_production_save_dir(self):
        train_cfg = self.cfg["train"]
        notes = self.cfg["experiment_notes"]
        self.assertEqual(train_cfg["save_dir"], "training/runs/unetpp_effb3_semantic_topology_aux_finetune_100ep")
        self.assertNotEqual(train_cfg["save_dir"], "training/runs/unetpp_effb3_a100_multiclass_curated_finetune_stage2_lr1e5_100ep")
        self.assertTrue(notes["no_authoritative_holdout"])
        self.assertTrue(notes["no_centerhead_weights"])
        self.assertFalse(notes["full_training_launched_in_prep"])


if __name__ == "__main__":
    unittest.main()
