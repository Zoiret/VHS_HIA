from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent


class TestPostprocessMulticlassMask(unittest.TestCase):
    def test_radius0_is_noop(self):
        import sys

        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import postprocess

        mask = np.array(
            [
                [0, 0, 2, 2, 0],
                [0, 1, 2, 2, 0],
                [0, 1, 1, 0, 0],
            ],
            dtype=np.uint8,
        )
        out = postprocess.postprocess_multiclass_mask(mask, ring_erosion_radius=0)
        self.assertTrue(np.array_equal(out, mask))

    def test_leaflet_unchanged(self):
        import sys

        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import postprocess

        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        mask[6:9, 6:9] = 2
        out = postprocess.postprocess_multiclass_mask(mask, ring_erosion_radius=2)
        self.assertTrue(np.array_equal(out == 1, mask == 1))

    def test_ring_area_does_not_increase(self):
        import sys

        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import postprocess

        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[8:16, 8:16] = 2
        out = postprocess.postprocess_multiclass_mask(mask, ring_erosion_radius=1)
        self.assertLessEqual(int(np.sum(out == 2)), int(np.sum(mask == 2)))

    def test_no_unknown_class_ids(self):
        import sys

        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import postprocess

        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        mask[10:12, 10:12] = 2
        out = postprocess.postprocess_multiclass_mask(mask, ring_erosion_radius=3)
        uniq = set(np.unique(out).tolist())
        self.assertTrue(uniq.issubset({0, 1, 2}))

    def test_small_ring_can_disappear(self):
        import sys

        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import postprocess

        mask = np.zeros((15, 15), dtype=np.uint8)
        mask[7, 7] = 2
        out = postprocess.postprocess_multiclass_mask(mask, ring_erosion_radius=2)
        self.assertEqual(int(np.sum(out == 2)), 0)

    def test_empty_ring_ok(self):
        import sys

        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import postprocess

        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        out = postprocess.postprocess_multiclass_mask(mask, ring_erosion_radius=2)
        self.assertTrue(np.array_equal(out == 1, mask == 1))
        self.assertEqual(int(np.sum(out == 2)), 0)

    def test_dtype_and_shape_preserved(self):
        import sys

        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import postprocess

        mask = np.zeros((11, 7), dtype=np.uint8)
        mask[4:9, 2:6] = 2
        out = postprocess.postprocess_multiclass_mask(mask, ring_erosion_radius=1)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, mask.shape)


if __name__ == "__main__":
    unittest.main()
