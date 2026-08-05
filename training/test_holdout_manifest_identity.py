from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


_THIS_DIR = Path(__file__).resolve().parent


class TestHoldoutManifestIdentity(unittest.TestCase):
    @staticmethod
    def _holdout_mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import validate_reconstruction_policies_holdout as mod

        return mod

    @staticmethod
    def _compare_mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import compare_holdout_manifests as mod

        return mod

    def _inventory(self, root_name: str = "dataset_root") -> dict:
        return {
            "dataset_root": str(Path(root_name)),
            "instance_root": str(Path(root_name) / "instances"),
            "eligible": [
                {
                    "sample": "m01_p01_s18",
                    "split": "test",
                    "gt_instance_count": 3,
                    "image_path": str(Path(root_name) / "images" / "m01_p01_s18.png"),
                    "gt_semantic_path": str(Path(root_name) / "semantic_masks" / "m01_p01_s18.png"),
                    "gt_instance_path": str(Path(root_name) / "instances" / "instance_masks" / "m01_p01_s18.png"),
                    "center_path": str(Path(root_name) / "center_maps" / "m01_p01_s18.png"),
                },
                {
                    "sample": "m01_p01_s19",
                    "split": "val",
                    "gt_instance_count": 2,
                    "image_path": str(Path(root_name) / "images" / "m01_p01_s19.png"),
                    "gt_semantic_path": str(Path(root_name) / "semantic_masks" / "m01_p01_s19.png"),
                    "gt_instance_path": str(Path(root_name) / "instances" / "instance_masks" / "m01_p01_s19.png"),
                    "center_path": str(Path(root_name) / "center_maps" / "m01_p01_s19.png"),
                },
            ],
        }

    def test_canonical_manifest_ignores_absolute_dataset_root(self):
        mod = self._holdout_mod()
        with tempfile.TemporaryDirectory() as tmp:
            root_a = Path(tmp) / "a"
            root_b = Path(tmp) / "b"
            for root in (root_a, root_b):
                for rel in (
                    "images/m01_p01_s18.png",
                    "semantic_masks/m01_p01_s18.png",
                    "center_maps/m01_p01_s18.png",
                    "instances/instance_masks/m01_p01_s18.png",
                    "images/m01_p01_s19.png",
                    "semantic_masks/m01_p01_s19.png",
                    "center_maps/m01_p01_s19.png",
                    "instances/instance_masks/m01_p01_s19.png",
                ):
                    path = root / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"same")
            inv_a = self._inventory(str(root_a))
            inv_b = self._inventory(str(root_b))
            sha_a = mod._identity_manifest_sha256(mod._canonical_identity_entries(inv_a))
            sha_b = mod._identity_manifest_sha256(mod._canonical_identity_entries(inv_b))
            self.assertEqual(sha_a, sha_b)

    def test_windows_and_linux_separators_produce_same_canonical_sha(self):
        mod = self._holdout_mod()
        self.assertEqual(
            mod._path_to_posix_relative(r"C:\data\images\foo.png", r"C:\data"),
            "images/foo.png",
        )

    def test_windows_forward_slash_path_produces_same_result(self):
        mod = self._holdout_mod()
        self.assertEqual(
            mod._path_to_posix_relative("C:/data/images/foo.png", "C:/data"),
            "images/foo.png",
        )

    def test_mixed_windows_separators_produce_same_result(self):
        mod = self._holdout_mod()
        self.assertEqual(
            mod._path_to_posix_relative(r"C:\data/images\foo.png", "C:/data"),
            "images/foo.png",
        )

    def test_posix_path_produces_same_result(self):
        mod = self._holdout_mod()
        self.assertEqual(
            mod._path_to_posix_relative("/datasets/data/images/foo.png", "/datasets/data"),
            "images/foo.png",
        )

    def test_unc_path_produces_expected_result(self):
        mod = self._holdout_mod()
        self.assertEqual(
            mod._path_to_posix_relative(r"\\server\share\data\images\foo.png", r"\\server\share\data"),
            "images/foo.png",
        )

    def test_different_windows_drives_produce_controlled_error(self):
        mod = self._holdout_mod()
        with self.assertRaises(mod.CanonicalPathError) as cm:
            mod._path_to_posix_relative(r"D:\data\images\foo.png", r"C:\data")
        self.assertEqual(cm.exception.flavour, "windows")

    def test_different_unc_shares_produce_controlled_error(self):
        mod = self._holdout_mod()
        with self.assertRaises(mod.CanonicalPathError) as cm:
            mod._path_to_posix_relative(r"\\server\share2\data\images\foo.png", r"\\server\share\data")
        self.assertEqual(cm.exception.flavour, "windows")

    def test_outside_root_path_produces_controlled_error(self):
        mod = self._holdout_mod()
        with self.assertRaises(mod.CanonicalPathError) as cm:
            mod._path_to_posix_relative("/datasets/other/images/foo.png", "/datasets/data")
        self.assertEqual(cm.exception.flavour, "posix")

    def test_cwd_is_never_prepended_to_foreign_windows_paths(self):
        mod = self._holdout_mod()
        result = mod._path_to_posix_relative(r"C:\data\images\foo.png", r"C:\data")
        self.assertEqual(result, "images/foo.png")
        self.assertNotIn("Users", result)

    def test_canonical_sha_equal_for_equivalent_windows_linux_representations(self):
        mod = self._holdout_mod()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "images" / "foo.png"
            sem = root / "semantic_masks" / "foo.png"
            inst = root / "instances" / "instance_masks" / "foo.png"
            ctr = root / "center_maps" / "foo.png"
            for path in (img, sem, inst, ctr):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"same")
            inv_posix = {
                "dataset_root": "/datasets/data",
                "instance_root": "/datasets/data/instances",
                "eligible": [
                    {
                        "sample": "foo",
                        "split": "test",
                        "gt_instance_count": 1,
                        "image_path": "/datasets/data/images/foo.png",
                        "gt_semantic_path": "/datasets/data/semantic_masks/foo.png",
                        "gt_instance_path": "/datasets/data/instances/instance_masks/foo.png",
                        "center_path": "/datasets/data/center_maps/foo.png",
                    }
                ],
            }
            inv_windows = {
                "dataset_root": r"C:\datasets\data",
                "instance_root": r"C:\datasets\data\instances",
                "eligible": [
                    {
                        "sample": "foo",
                        "split": "test",
                        "gt_instance_count": 1,
                        "image_path": r"C:\datasets\data\images\foo.png",
                        "gt_semantic_path": r"C:\datasets\data\semantic_masks\foo.png",
                        "gt_instance_path": r"C:\datasets\data\instances\instance_masks\foo.png",
                        "center_path": r"C:\datasets\data\center_maps\foo.png",
                    }
                ],
            }
            with mock.patch.object(mod, "_identity_hash", return_value="samehash"):
                sha_posix = mod._identity_manifest_sha256(mod._canonical_identity_entries(inv_posix))
                sha_windows = mod._identity_manifest_sha256(mod._canonical_identity_entries(inv_windows))
            self.assertEqual(sha_posix, sha_windows)

    def test_row_ordering_does_not_change_canonical_sha(self):
        mod = self._holdout_mod()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in (
                "images/m01_p01_s18.png",
                "semantic_masks/m01_p01_s18.png",
                "center_maps/m01_p01_s18.png",
                "instances/instance_masks/m01_p01_s18.png",
                "images/m01_p01_s19.png",
                "semantic_masks/m01_p01_s19.png",
                "center_maps/m01_p01_s19.png",
                "instances/instance_masks/m01_p01_s19.png",
            ):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(rel.encode("utf-8"))
            inv = self._inventory(str(root))
            sha_a = mod._identity_manifest_sha256(mod._canonical_identity_entries(inv))
            inv["eligible"] = list(reversed(inv["eligible"]))
            sha_b = mod._identity_manifest_sha256(mod._canonical_identity_entries(inv))
            self.assertEqual(sha_a, sha_b)

    def test_content_hash_difference_changes_canonical_sha(self):
        mod = self._holdout_mod()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in (
                "images/m01_p01_s18.png",
                "semantic_masks/m01_p01_s18.png",
                "center_maps/m01_p01_s18.png",
                "instances/instance_masks/m01_p01_s18.png",
                "images/m01_p01_s19.png",
                "semantic_masks/m01_p01_s19.png",
                "center_maps/m01_p01_s19.png",
                "instances/instance_masks/m01_p01_s19.png",
            ):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"same")
            inv = self._inventory(str(root))
            sha_a = mod._identity_manifest_sha256(mod._canonical_identity_entries(inv))
            (root / "images" / "m01_p01_s19.png").write_bytes(b"different")
            sha_b = mod._identity_manifest_sha256(mod._canonical_identity_entries(inv))
            self.assertNotEqual(sha_a, sha_b)

    def test_expected_manifest_mismatch_produces_non_exact_status(self):
        mod = self._holdout_mod()
        status = mod._manifest_identity_status(actual_sha="abc", expected_sha="def", unique_sample_count=2, row_count=2)
        self.assertEqual(status, "manifest_identity_mismatch")

    def test_overall_status_cannot_be_ok_when_manifest_mismatch_exists(self):
        mod = self._holdout_mod()
        overall = mod._overall_authoritative_status(
            checkpoint_identity_status="exact_match",
            semantic_checkpoint_identity_status="exact_match",
            manifest_identity_status="manifest_identity_mismatch",
        )
        self.assertEqual(overall, "manifest_identity_mismatch")

    def test_local_server_manifest_diff_lists_exact_differences(self):
        mod = self._compare_mod()
        local_rows = [
            {
                "sample": "a",
                "sample_index": 0,
                "split": "test",
                "gt_instance_count": 3,
                "image_relative_path": "images/a.png",
                "semantic_gt_relative_path": "semantic_masks/a.png",
                "instance_gt_relative_path": "instance_masks/a.png",
                "center_gt_relative_path": "center_maps/a.png",
                "image_sha256": "1",
                "semantic_gt_sha256": "2",
                "instance_gt_sha256": "3",
                "center_gt_sha256": "4",
            }
        ]
        server_rows = [dict(local_rows[0], gt_instance_count=2)]
        diff = mod._compare_rows(local_rows, server_rows)
        self.assertFalse(diff["same_canonical_identity"])
        self.assertEqual(len(diff["gt_count_differences"]), 1)

    def test_compare_utility_detects_identical_manifests_as_exact_content_match(self):
        mod = self._compare_mod()
        rows = [
            {
                "sample": "a",
                "sample_index": 0,
                "split": "test",
                "gt_instance_count": 3,
                "image_relative_path": "images/a.png",
                "semantic_gt_relative_path": "semantic_masks/a.png",
                "instance_gt_relative_path": "instance_masks/a.png",
                "center_gt_relative_path": "center_maps/a.png",
                "image_sha256": "1",
                "semantic_gt_sha256": "2",
                "instance_gt_sha256": "3",
                "center_gt_sha256": "4",
            }
        ]
        diff = mod._compare_rows(rows, [dict(rows[0])])
        self.assertEqual(diff["status"], "exact_content_match")
        self.assertTrue(diff["same_canonical_identity"])

    def test_expected_sha_unset_cannot_produce_authoritative_exact_match(self):
        mod = self._holdout_mod()
        status = mod._manifest_identity_status(actual_sha="abc", expected_sha=None, unique_sample_count=2, row_count=2)
        self.assertEqual(status, "expected_manifest_identity_sha_unset")


if __name__ == "__main__":
    unittest.main()
