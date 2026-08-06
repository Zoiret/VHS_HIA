from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


_THIS_DIR = Path(__file__).resolve().parent


class TestFullDatasetCenterTraining(unittest.TestCase):
    @staticmethod
    def _mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import prepare_full_dataset_center_training as mod

        return mod

    @staticmethod
    def _record(mod, *, sample: str, patient_id: str, gt_count: int, foreground_area: int = 1000, image_sha: str | None = None) -> object:
        return mod.SampleRecord(
            sample=sample,
            patient_id=patient_id,
            source_split="train",
            image_path=Path(f"/tmp/{sample}.png"),
            semantic_mask_path=Path(f"/tmp/{sample}_sem.png"),
            instance_mask_path=Path(f"/tmp/{sample}_inst.png"),
            center_target_path=Path(f"/tmp/{sample}_ctr.png"),
            metadata_path=Path(f"/tmp/{sample}.json"),
            gt_instance_count=int(gt_count),
            image_height=768,
            image_width=768,
            foreground_area=int(foreground_area),
            quality="clean",
            in_microset=False,
            in_authoritative_holdout=False,
            image_sha256=image_sha or f"img_{sample}",
            semantic_sha256=f"sem_{sample}",
            instance_sha256=f"inst_{sample}",
            center_sha256=f"ctr_{sample}",
            source_instance_ids=tuple(range(1, int(gt_count) + 1)),
            instance_areas=tuple([100] * int(gt_count)),
            instance_center_yx=tuple((100 + i * 20, 120 + i * 20) for i in range(int(gt_count))),
            max_dt_per_instance=tuple([5.0] * int(gt_count)),
            semantic_cc_count=int(gt_count),
            border_touching_instances=0,
            fragmented_semantic=False,
        )

    def test_patient_level_split_is_deterministic(self):
        mod = self._mod()
        records = [
            self._record(mod, sample="m01_p01_s00", patient_id="m01_p01", gt_count=1),
            self._record(mod, sample="m01_p01_s01", patient_id="m01_p01", gt_count=2),
            self._record(mod, sample="m02_p01_s00", patient_id="m02_p01", gt_count=3),
            self._record(mod, sample="m02_p01_s01", patient_id="m02_p01", gt_count=3),
            self._record(mod, sample="m03_p01_s00", patient_id="m03_p01", gt_count=2),
            self._record(mod, sample="m04_p01_s00", patient_id="m04_p01", gt_count=1),
            self._record(mod, sample="m05_p01_s00", patient_id="m05_p01", gt_count=2),
        ]
        train_a, val_a, _summary_a = mod._patient_level_split(records, seed=1337, val_ratio=0.3)
        train_b, val_b, _summary_b = mod._patient_level_split(records, seed=1337, val_ratio=0.3)
        self.assertEqual([r.sample for r in train_a], [r.sample for r in train_b])
        self.assertEqual([r.sample for r in val_a], [r.sample for r in val_b])

    def test_patient_level_split_has_no_patient_leakage(self):
        mod = self._mod()
        records = [
            self._record(mod, sample="m01_p01_s00", patient_id="m01_p01", gt_count=1),
            self._record(mod, sample="m01_p01_s01", patient_id="m01_p01", gt_count=2),
            self._record(mod, sample="m02_p01_s00", patient_id="m02_p01", gt_count=3),
            self._record(mod, sample="m03_p01_s00", patient_id="m03_p01", gt_count=2),
            self._record(mod, sample="m04_p01_s00", patient_id="m04_p01", gt_count=1),
        ]
        train_records, val_records, _summary = mod._patient_level_split(records, seed=1337, val_ratio=0.4)
        self.assertFalse({r.patient_id for r in train_records} & {r.patient_id for r in val_records})

    def test_holdout_exclusion_raises_on_leaked_sample(self):
        mod = self._mod()
        rec = self._record(mod, sample="m01_p01_s00", patient_id="m01_p01", gt_count=1)
        leaked = type(rec)(**{**rec.__dict__, "in_authoritative_holdout": True})
        with self.assertRaises(mod.ReadinessError) as cm:
            mod._assert_no_holdout_overlap([leaked])
        self.assertIn("holdout", str(cm.exception).lower())
        self.assertEqual(cm.exception.samples, ["m01_p01_s00"])

    def test_duplicate_sha_detection(self):
        mod = self._mod()
        train_records = [self._record(mod, sample="m01_p01_s00", patient_id="m01_p01", gt_count=1, image_sha="same")]
        val_records = [self._record(mod, sample="m02_p01_s00", patient_id="m02_p01", gt_count=1, image_sha="same")]
        dups = mod._duplicate_content_across_splits(train_records, val_records)
        self.assertEqual(len(dups), 1)
        self.assertEqual(dups[0]["artifact_type"], "image")

    def test_load_aligned_instance_center_crop_matches_target(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instance.png"
            raw = np.zeros((1024, 1024), dtype=np.uint8)
            raw[128 + 300, 128 + 400] = 1
            cv2.imwrite(str(path), raw)
            aligned, raw_hw = mod._load_aligned_instance(path, target_hw=(768, 768))
            self.assertEqual(raw_hw, (1024, 1024))
            self.assertEqual(aligned.shape, (768, 768))
            self.assertEqual(int(aligned[300, 400]), 1)

    def test_target_audit_accepts_synthetic_fixture(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "img.png"
            semantic_path = root / "sem.png"
            instance_path = root / "inst.png"
            center_path = root / "ctr.png"
            metadata_path = root / "meta.json"

            image = np.zeros((768, 768, 3), dtype=np.uint8)
            semantic = np.zeros((768, 768), dtype=np.uint8)
            semantic[200:320, 150:260] = 1
            semantic[420:540, 500:620] = 1
            raw_instance = np.zeros((1024, 1024), dtype=np.uint8)
            raw_instance[128 + 200 : 128 + 320, 128 + 150 : 128 + 260] = 1
            raw_instance[128 + 420 : 128 + 540, 128 + 500 : 128 + 620] = 2
            center = np.zeros((768, 768), dtype=np.uint16)
            center[250, 200] = 65535
            center[480, 560] = 65535
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(semantic_path), semantic)
            cv2.imwrite(str(instance_path), raw_instance)
            cv2.imwrite(str(center_path), center)
            metadata_path.write_text(
                json.dumps(
                    {
                        "sample": "m01_p01_s00",
                        "instance_count": 2,
                        "instances": [
                            {"instance_id": 1, "area": 13200, "max_dt": 7.0, "center_yx": [250, 200]},
                            {"instance_id": 2, "area": 13200, "max_dt": 7.0, "center_yx": [480, 560]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rec = mod.SampleRecord(
                sample="m01_p01_s00",
                patient_id="m01_p01",
                source_split="train",
                image_path=image_path,
                semantic_mask_path=semantic_path,
                instance_mask_path=instance_path,
                center_target_path=center_path,
                metadata_path=metadata_path,
                gt_instance_count=2,
                image_height=768,
                image_width=768,
                foreground_area=int(np.sum(semantic > 0)),
                quality="clean",
                in_microset=True,
                in_authoritative_holdout=False,
                image_sha256="img",
                semantic_sha256="sem",
                instance_sha256="inst",
                center_sha256="ctr",
                source_instance_ids=(1, 2),
                instance_areas=(13200, 13200),
                instance_center_yx=((250, 200), (480, 560)),
                max_dt_per_instance=(7.0, 7.0),
                semantic_cc_count=2,
                border_touching_instances=0,
                fragmented_semantic=False,
            )
            summary = mod._target_audit([rec], output_dir=root / "out")
            self.assertEqual(summary["invalid_samples"], 0)
            csv_path = root / "out" / "per_sample_target_audit.csv"
            self.assertTrue(csv_path.exists())

    def test_manifest_canonical_sha_is_deterministic(self):
        mod = self._mod()
        rows_a = [{"b": 2, "a": 1}, {"sample": "x", "gt": 1}]
        rows_b = [{"a": 1, "b": 2}, {"gt": 1, "sample": "x"}]
        self.assertEqual(mod._jsonl_sha256(rows_a), mod._jsonl_sha256(rows_b))

    def test_config_payload_contains_manifest_paths_and_threshold_policy(self):
        mod = self._mod()
        cfg = mod._config_payload()
        self.assertIn("train_manifest", cfg["dataset"])
        self.assertIn("val_manifest", cfg["dataset"])
        self.assertEqual(cfg["center"]["threshold_policy"]["locked_reference_threshold"], 0.03)
        self.assertEqual(cfg["train"]["checkpoint_selection_metric"], "center_f1_mean_samples")
        self.assertEqual(cfg["scheduler"]["monitor"], "center_f1_mean_samples")
        self.assertTrue(cfg["train"]["require_exact_semantic_checkpoint_load"])

    def test_config_yaml_parses(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cfg.yaml"
            with mock.patch.object(mod, "DEFAULT_CONFIG_PATH", out):
                mod._write_config(out)
            cfg = mod._read_yaml(out)
            self.assertEqual(cfg["model"]["center_feature"]["module_path"], "base.decoder.blocks.x_2_2")

    def test_smoke_summary_uses_synthetic_fixture(self):
        mod = self._mod()
        synthetic = {
            "semantic_shape": (1, 3, 768, 768),
            "center_shape": (1, 1, 768, 768),
            "semantic_loss_finite": True,
            "loss_total": 1.25,
            "combined_grad_norm_before_clip": 0.5,
            "center_grad_all_finite": True,
            "nonfinite_gradient_tensors": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with mock.patch.object(mod, "_read_yaml", return_value=mod._config_payload()), mock.patch.object(mod, "smoke_test", return_value=synthetic):
                summary = mod._smoke_summary(out_dir / "cfg.yaml", output_dir=out_dir)
            self.assertEqual(summary["forward"], "passed")
            self.assertEqual(summary["gradients"], "passed")
            self.assertTrue((out_dir / "smoke_test_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
