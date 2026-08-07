from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent


class TestFullDatasetCenterBaselineDiagnosis(unittest.TestCase):
    @staticmethod
    def _mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import diagnose_full_dataset_center_baseline as mod

        return mod

    def test_manifest_dataset_order_is_deterministic(self):
        mod = self._mod()
        rows = [
            {"sample_index": 2, "sample": "m01_p01_s02", "patient_id": "m01_p01", "gt_instance_count": 3, "image_rel": "images/a.png", "semantic_mask_rel": "semantic_masks/a.png", "center_target_rel": "center_maps/a.png", "metadata_rel": "metadata/a.json", "instance_mask_rel": "instance_masks/a.png"},
            {"sample_index": 0, "sample": "m01_p01_s00", "patient_id": "m01_p01", "gt_instance_count": 1, "image_rel": "images/b.png", "semantic_mask_rel": "semantic_masks/b.png", "center_target_rel": "center_maps/b.png", "metadata_rel": "metadata/b.json", "instance_mask_rel": "instance_masks/b.png"},
            {"sample_index": 1, "sample": "m01_p01_s01", "patient_id": "m01_p01", "gt_instance_count": 2, "image_rel": "images/c.png", "semantic_mask_rel": "semantic_masks/c.png", "center_target_rel": "center_maps/c.png", "metadata_rel": "metadata/c.json", "instance_mask_rel": "instance_masks/c.png"},
        ]
        cfg = {"dataset": {"root": ".", "instance_root": "."}, "model": {"input_size": 768, "encoder_name": "efficientnet-b3", "encoder_weights": None}, "train": {"batch_size": 2}}
        ds = mod.ManifestCenterDataset(rows, cfg)
        self.assertEqual([int(row["sample_index"]) for row in ds.rows], [0, 1, 2])

    def test_checkpoint_identity_sha_matches_file(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.bin"
            path.write_bytes(b"abc")
            self.assertEqual(mod._sha256_file(path), hashlib.sha256(b"abc").hexdigest())

    def test_predicted_count_distribution_and_saturation(self):
        mod = self._mod()
        rows = [
            {"predicted_count": 0},
            {"predicted_count": 1},
            {"predicted_count": 3},
            {"predicted_count": 3},
        ]
        dist = mod._pred_count_distribution(rows)
        self.assertEqual(dist["predicted_count_3"], 2)
        self.assertEqual(dist["fraction_predicted_count_3"], 0.5)

    def test_heatmap_margin_calculation(self):
        mod = self._mod()
        center_prob = np.zeros((8, 8), dtype=np.float32)
        center_prob[2, 2] = 0.8
        center_prob[5, 5] = 0.7
        center_prob[0, 7] = 0.4
        gt_inst = np.zeros((8, 8), dtype=np.int32)
        gt_inst[1:4, 1:4] = 1
        gt_inst[4:7, 4:7] = 2
        stats = mod._heatmap_margin(center_prob, gt_inst, [(2, 2), (5, 5)])
        self.assertAlmostEqual(stats["min_gt_center_score"], 0.7, places=6)
        self.assertAlmostEqual(stats["maximum_far_background_score"], 0.4, places=6)
        self.assertAlmostEqual(stats["margin"], 0.3, places=6)

    def test_scheduler_runtime_state_extraction(self):
        mod = self._mod()
        metrics = [
            {"epoch": "0", "center_f1_mean_samples": "0.0", "lr_center_head": ""},
            {"epoch": "1", "center_f1_mean_samples": "0.1", "lr_center_head": "0.001"},
            {"epoch": "2", "center_f1_mean_samples": "0.09", "lr_center_head": "0.001"},
            {"epoch": "3", "center_f1_mean_samples": "0.08", "lr_center_head": "0.001"},
            {"epoch": "4", "center_f1_mean_samples": "0.07", "lr_center_head": "0.001"},
            {"epoch": "5", "center_f1_mean_samples": "0.06", "lr_center_head": "0.001"},
            {"epoch": "6", "center_f1_mean_samples": "0.05", "lr_center_head": "0.001"},
            {"epoch": "7", "center_f1_mean_samples": "0.04", "lr_center_head": "0.001"},
        ]
        states = mod._scheduler_state_sequence(metrics, {"type": "reduce_on_plateau", "mode": "max", "monitor": "center_f1_mean_samples", "factor": 0.5, "patience": 5, "min_lr": 1e-6})
        self.assertTrue(any(bool(state["lr_reduced"]) for state in states))
        reduced = [state for state in states if bool(state["lr_reduced"])]
        self.assertEqual(reduced[0]["epoch"], 7)
        self.assertAlmostEqual(reduced[0]["lr_after_step"], 0.0005, places=12)

    def test_confusion_rows_cover_gt_and_prediction_grid(self):
        mod = self._mod()
        rows = [{"gt_instance_count": 1, "predicted_count": 1}, {"gt_instance_count": 3, "predicted_count": 3}]
        out = mod._confusion_rows(rows)
        self.assertEqual(len(out), 12)
        hit = next(row for row in out if int(row["gt_instance_count"]) == 3 and int(row["predicted_count"]) == 3)
        self.assertEqual(hit["sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
