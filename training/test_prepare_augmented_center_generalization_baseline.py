from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent


class TestPrepareAugmentedCenterGeneralizationBaseline(unittest.TestCase):
    @staticmethod
    def _aug():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import augmentations as mod

        return mod

    @staticmethod
    def _prep():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import prepare_augmented_center_generalization_baseline as mod

        return mod

    def _toy_arrays(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=np.uint8)
        center = np.zeros((4, 4), dtype=np.uint16)
        mask[1:3, 1:3] = 1
        mask[0:2, 2:4] = 2
        center[1, 1] = 65535
        center[0, 2] = 65535
        points = [(1, 1), (0, 2)]
        return image, mask, center, points

    def test_image_mask_center_horizontal_flip_alignment(self):
        aug = self._aug()
        image, mask, center, points = self._toy_arrays()
        _img, mask_tf, center_tf = aug.apply_exact_geometric_transform(image, mask, center=center, hflip=True, vflip=False, rot90_k=0)
        pts_tf = aug.transform_points_row_col_yx(points, mask.shape, hflip=True, vflip=False, rot90_k=0)
        peak_points = sorted((int(y), int(x)) for y, x in zip(*np.where(center_tf == int(center_tf.max()))))
        self.assertEqual(sorted(pts_tf), peak_points)
        self.assertTrue(all(int(mask_tf[y, x]) > 0 for y, x in pts_tf))

    def test_rotation90_alignment(self):
        aug = self._aug()
        image, mask, center, points = self._toy_arrays()
        _img, mask_tf, center_tf = aug.apply_exact_geometric_transform(image, mask, center=center, rot90_k=1)
        pts_tf = aug.transform_points_row_col_yx(points, mask.shape, rot90_k=1)
        peak_points = sorted((int(y), int(x)) for y, x in zip(*np.where(center_tf == int(center_tf.max()))))
        self.assertEqual(sorted(pts_tf), peak_points)
        self.assertTrue(all(int(mask_tf[y, x]) > 0 for y, x in pts_tf))

    def test_row_col_yx_preservation(self):
        aug = self._aug()
        pts = [(1, 2)]
        transformed = aug.transform_points_row_col_yx(pts, (4, 5), hflip=True, vflip=False, rot90_k=0)
        self.assertEqual(transformed, [(1, 2)])

    def test_center_remains_inside_instance_after_transform(self):
        aug = self._aug()
        image, mask, center, points = self._toy_arrays()
        _img, mask_tf, center_tf = aug.apply_exact_geometric_transform(image, mask, center=center, hflip=True, vflip=True, rot90_k=3)
        pts_tf = aug.transform_points_row_col_yx(points, mask.shape, hflip=True, vflip=True, rot90_k=3)
        self.assertTrue(all(int(mask_tf[y, x]) > 0 for y, x in pts_tf))
        peak_points = sorted((int(y), int(x)) for y, x in zip(*np.where(center_tf == int(center_tf.max()))))
        self.assertEqual(sorted(pts_tf), peak_points)

    def test_deterministic_augmentation_with_fixed_seed(self):
        aug = self._aug()
        cfg = {"rotate90": True, "hflip": True, "vflip": True, "brightness_contrast": True, "brightness_limit": 12.0, "contrast_limit": 0.1}
        p1 = aug.sample_train_augmentation_params(cfg, rng=random.Random(1337))
        p2 = aug.sample_train_augmentation_params(cfg, rng=random.Random(1337))
        self.assertEqual(p1, p2)

    def test_validation_augmentation_disabled(self):
        aug = self._aug()
        image, mask, center, _points = self._toy_arrays()
        val_aug = aug.get_val_augmentations(4, 4)
        out1 = val_aug(image.copy(), mask.copy(), center=center.copy())
        out2 = val_aug(image.copy(), mask.copy(), center=center.copy())
        self.assertTrue(np.array_equal(out1[0], out2[0]))
        self.assertTrue(np.array_equal(out1[1], out2[1]))
        self.assertTrue(np.array_equal(out1[2], out2[2]))

    def test_baseline_config_unchanged(self):
        prep = self._prep()
        baseline = prep._read_yaml(prep.BASELINE_CONFIG_PATH)
        self.assertEqual(baseline["augment"]["rotate90"], False)
        self.assertEqual(baseline["augment"]["hflip"], False)
        self.assertEqual(baseline["augment"]["vflip"], False)
        self.assertEqual(baseline["augment"]["brightness_contrast"], False)

    def test_new_config_changes_augmentation_only(self):
        prep = self._prep()
        baseline = prep._read_yaml(prep.BASELINE_CONFIG_PATH)
        new_cfg, diff_paths = prep._build_augmented_config(baseline)
        self.assertEqual(
            set(diff_paths),
            {
                "augment.rotate90",
                "augment.hflip",
                "augment.vflip",
                "augment.brightness_contrast",
                "augment.brightness_limit",
                "augment.contrast_limit",
                "augment.random_crop",
                "train.save_dir",
            },
        )
        self.assertEqual(new_cfg["train"]["save_dir"], prep.NEW_SAVE_DIR)

    def test_scheduler_logging_fields_present(self):
        import train_centerhead as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertIn("scheduler_monitor_name", source)
        self.assertIn("scheduler_monitor_threshold_context", source)
        self.assertIn("scheduler_monitor_value", source)
        self.assertIn("lr_before_scheduler_step", source)
        self.assertIn("lr_after_scheduler_step", source)
        self.assertIn("scheduler_best", source)
        self.assertIn("scheduler_num_bad_epochs", source)
        self.assertIn("early_stop_reset_policy", source)


if __name__ == "__main__":
    unittest.main()
