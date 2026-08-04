from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
