from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from training import audit_external_dataset_fingerprint as audit


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), arr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


class ExternalDatasetFingerprintTests(unittest.TestCase):
    def _make_dataset(
        self,
        root: Path,
        *,
        train_rows: list[str] | None = None,
        val_rows: list[str] | None = None,
        image_variant: int = 0,
        semantic_variant: int = 0,
        instance_variant: int = 0,
        missing_instance_sample: str | None = None,
    ) -> tuple[Path, Path, dict[str, Path]]:
        dataset_root = root / "repo_copy" / "datasets" / "converted_full_multiclass"
        instance_root = root / "repo_copy" / "datasets" / "converted_leaflet_instances"
        sample_ids = ["m01_p01_s00", "m02_p01_s00", "m03_p01_s00"]
        rgb_base = {
            "m01_p01_s00": np.full((8, 8, 3), 10, dtype=np.uint8),
            "m02_p01_s00": np.full((8, 8, 3), 20, dtype=np.uint8),
            "m03_p01_s00": np.full((8, 8, 3), 30, dtype=np.uint8),
        }
        sem_base = {
            "m01_p01_s00": np.zeros((8, 8), dtype=np.uint8),
            "m02_p01_s00": np.ones((8, 8), dtype=np.uint8),
            "m03_p01_s00": np.full((8, 8), 2, dtype=np.uint8),
        }
        inst_base = {
            "m01_p01_s00": np.pad(np.ones((4, 4), dtype=np.uint8), 2),
            "m02_p01_s00": np.pad(np.array([[1, 1], [0, 2]], dtype=np.uint8), ((3, 3), (3, 3))),
            "m03_p01_s00": np.pad(np.array([[1, 0], [0, 0]], dtype=np.uint8), ((3, 3), (3, 3))),
        }
        if image_variant:
            rgb_base["m02_p01_s00"][0, 0, 0] = np.uint8(99)
        if semantic_variant:
            sem_base["m02_p01_s00"][0, 0] = np.uint8(7)
        if instance_variant:
            inst_base["m02_p01_s00"][0, 0] = np.uint8(5)

        for sample_id in sample_ids:
            _write_png(dataset_root / "images" / f"{sample_id}.png", rgb_base[sample_id])
            _write_png(dataset_root / "masks" / f"{sample_id}.png", sem_base[sample_id])
            if sample_id != missing_instance_sample:
                _write_png(instance_root / "instance_masks" / f"{sample_id}.png", inst_base[sample_id])

        train_rows = train_rows or [
            "images/m01_p01_s00.png\tmasks/m01_p01_s00.png",
            "images/m02_p01_s00.png\tmasks/m02_p01_s00.png",
        ]
        val_rows = val_rows or [
            "images/m03_p01_s00.png\tmasks/m03_p01_s00.png",
        ]
        test_rows = [
            "images/m02_p01_s00.png\tmasks/m02_p01_s00.png",
        ]
        curated_root = root / "repo_copy" / "datasets" / "converted_full_multiclass_curated"
        curated_root.mkdir(parents=True, exist_ok=True)
        (curated_root / "train.txt").write_text("\r\n".join(train_rows) + "\r\n", encoding="utf-8")
        (curated_root / "val.txt").write_text("\n".join(val_rows) + "\n", encoding="utf-8")
        (curated_root / "test.txt").write_text("\n".join(test_rows) + "\n", encoding="utf-8")
        return dataset_root, instance_root, {
            "train": curated_root / "train.txt",
            "val": curated_root / "val.txt",
            "test": curated_root / "test.txt",
        }

    def test_canonical_split_hash_ignores_eol_bom_and_empty_lines(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.txt"
            p2 = Path(td) / "b.txt"
            p1.write_text("images/a.png\tmasks/a.png\r\n\r\nimages/b.png\tmasks/b.png\r\n", encoding="utf-8")
            p2.write_bytes(("\ufeffimages/a.png\tmasks/a.png\nimages/b.png\tmasks/b.png\n\n").encode("utf-8"))
            rows1, sha1 = audit.canonical_split_sha256(p1)
            rows2, sha2 = audit.canonical_split_sha256(p2)
            self.assertEqual(rows1, rows2)
            self.assertEqual(sha1, sha2)

    def test_order_changes_split_identity(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.txt"
            p2 = Path(td) / "b.txt"
            p1.write_text("images/a.png\tmasks/a.png\nimages/b.png\tmasks/b.png\n", encoding="utf-8")
            p2.write_text("images/b.png\tmasks/b.png\nimages/a.png\tmasks/a.png\n", encoding="utf-8")
            _rows1, sha1 = audit.canonical_split_sha256(p1)
            _rows2, sha2 = audit.canonical_split_sha256(p2)
            self.assertNotEqual(sha1, sha2)

    def test_asset_hash_and_contract_hash_are_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset_root, instance_root, splits = self._make_dataset(root)
            fp1 = audit.build_fingerprint(dataset_root=dataset_root, instance_root=instance_root, splits={"train": splits["train"], "val": splits["val"]})
            fp2 = audit.build_fingerprint(dataset_root=dataset_root, instance_root=instance_root, splits={"train": splits["train"], "val": splits["val"]})
            self.assertEqual(fp1["dataset_contract_sha256"], fp2["dataset_contract_sha256"])
            self.assertEqual(fp1["dataset_contract"]["assets"]["m01_p01_s00"]["image"]["sha256"], fp2["dataset_contract"]["assets"]["m01_p01_s00"]["image"]["sha256"])

    def test_absolute_location_does_not_change_dataset_contract_sha(self):
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            dataset_root_1, instance_root_1, splits_1 = self._make_dataset(Path(td1))
            dataset_root_2, instance_root_2, splits_2 = self._make_dataset(Path(td2))
            fp1 = audit.build_fingerprint(dataset_root=dataset_root_1, instance_root=instance_root_1, splits={"train": splits_1["train"], "val": splits_1["val"]})
            fp2 = audit.build_fingerprint(dataset_root=dataset_root_2, instance_root=instance_root_2, splits={"train": splits_2["train"], "val": splits_2["val"]})
            self.assertEqual(fp1["dataset_contract_sha256"], fp2["dataset_contract_sha256"])

    def test_changed_image_bytes_are_detected(self):
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            dataset_root_1, instance_root_1, splits_1 = self._make_dataset(Path(td1))
            dataset_root_2, instance_root_2, splits_2 = self._make_dataset(Path(td2), image_variant=1)
            fp1 = audit.build_fingerprint(dataset_root=dataset_root_1, instance_root=instance_root_1, splits={"train": splits_1["train"]})
            fp2 = audit.build_fingerprint(dataset_root=dataset_root_2, instance_root=instance_root_2, splits={"train": splits_2["train"]})
            cmp = audit.compare_fingerprints(fp1, fp2)
            self.assertIn("m02_p01_s00", cmp["asset_differences"]["image_sha_mismatch"])
            self.assertEqual(cmp["classification"], "DIFFERENT_ASSETS")

    def test_changed_semantic_mask_bytes_are_detected(self):
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            dataset_root_1, instance_root_1, splits_1 = self._make_dataset(Path(td1))
            dataset_root_2, instance_root_2, splits_2 = self._make_dataset(Path(td2), semantic_variant=1)
            fp1 = audit.build_fingerprint(dataset_root=dataset_root_1, instance_root=instance_root_1, splits={"train": splits_1["train"]})
            fp2 = audit.build_fingerprint(dataset_root=dataset_root_2, instance_root=instance_root_2, splits={"train": splits_2["train"]})
            cmp = audit.compare_fingerprints(fp1, fp2)
            self.assertIn("m02_p01_s00", cmp["asset_differences"]["semantic_mask_sha_mismatch"])
            self.assertEqual(cmp["classification"], "DIFFERENT_ASSETS")

    def test_changed_instance_mask_bytes_are_detected(self):
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            dataset_root_1, instance_root_1, splits_1 = self._make_dataset(Path(td1))
            dataset_root_2, instance_root_2, splits_2 = self._make_dataset(Path(td2), instance_variant=1)
            fp1 = audit.build_fingerprint(dataset_root=dataset_root_1, instance_root=instance_root_1, splits={"train": splits_1["train"]})
            fp2 = audit.build_fingerprint(dataset_root=dataset_root_2, instance_root=instance_root_2, splits={"train": splits_2["train"]})
            cmp = audit.compare_fingerprints(fp1, fp2)
            self.assertIn("m02_p01_s00", cmp["asset_differences"]["instance_mask_sha_mismatch"])
            self.assertEqual(cmp["classification"], "DIFFERENT_ASSETS")

    def test_missing_asset_detected(self):
        with tempfile.TemporaryDirectory() as td:
            dataset_root, instance_root, splits = self._make_dataset(Path(td), missing_instance_sample="m02_p01_s00")
            fp = audit.build_fingerprint(dataset_root=dataset_root, instance_root=instance_root, splits={"train": splits["train"]})
            asset = fp["dataset_contract"]["assets"]["m02_p01_s00"]["instance_mask"]
            self.assertFalse(asset["exists"])
            self.assertEqual(fp["summary"]["total_missing_assets"], 1)

    def test_compare_classifications_are_correct(self):
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2, tempfile.TemporaryDirectory() as td3:
            dataset_root_1, instance_root_1, splits_1 = self._make_dataset(Path(td1))
            dataset_root_2, instance_root_2, splits_2 = self._make_dataset(
                Path(td2),
                train_rows=[
                    "images/m02_p01_s00.png\tmasks/m02_p01_s00.png",
                    "images/m01_p01_s00.png\tmasks/m01_p01_s00.png",
                ],
            )
            dataset_root_3, instance_root_3, splits_3 = self._make_dataset(Path(td3), missing_instance_sample="m02_p01_s00")
            fp1 = audit.build_fingerprint(dataset_root=dataset_root_1, instance_root=instance_root_1, splits={"train": splits_1["train"]})
            fp2 = audit.build_fingerprint(dataset_root=dataset_root_2, instance_root=instance_root_2, splits={"train": splits_2["train"]})
            fp3 = audit.build_fingerprint(dataset_root=dataset_root_3, instance_root=instance_root_3, splits={"train": splits_3["train"]})
            self.assertEqual(audit.compare_fingerprints(fp1, copy_payload(fp1))["classification"], "IDENTICAL_DATASET")
            self.assertEqual(audit.compare_fingerprints(fp1, fp2)["classification"], "SAME_ASSETS_DIFFERENT_SPLIT")
            self.assertEqual(audit.compare_fingerprints(fp1, fp3)["classification"], "INCOMPLETE_COMPARISON")

    def test_same_pixels_different_png_encoding_have_same_decoded_pixel_sha(self):
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.png"
            b = Path(td) / "b.png"
            mask = np.array([[0, 1], [2, 2]], dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(a), mask, [cv2.IMWRITE_PNG_COMPRESSION, 0]))
            self.assertTrue(cv2.imwrite(str(b), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9]))
            fa = audit._semantic_mask_fingerprint(a, relative_path="masks/a.png")
            fb = audit._semantic_mask_fingerprint(b, relative_path="masks/b.png")
            self.assertNotEqual(fa["file_sha256"], fb["file_sha256"])
            self.assertEqual(fa["decoded_pixel_sha256"], fb["decoded_pixel_sha256"])
            self.assertEqual(fa["decoded_class_counts"], fb["decoded_class_counts"])

    def test_same_semantic_pixels_different_png_encoding_compare_as_identical_dataset(self):
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            dataset_root_1, instance_root_1, splits_1 = self._make_dataset(Path(td1))
            dataset_root_2, instance_root_2, splits_2 = self._make_dataset(Path(td2))
            mask = np.array([[0, 1], [2, 2]], dtype=np.uint8)
            png_a = dataset_root_1 / "masks" / "m02_p01_s00.png"
            png_b = dataset_root_2 / "masks" / "m02_p01_s00.png"
            self.assertTrue(cv2.imwrite(str(png_a), mask, [cv2.IMWRITE_PNG_COMPRESSION, 0]))
            self.assertTrue(cv2.imwrite(str(png_b), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9]))
            fp1 = audit.build_fingerprint(dataset_root=dataset_root_1, instance_root=instance_root_1, splits={"train": splits_1["train"]})
            fp2 = audit.build_fingerprint(dataset_root=dataset_root_2, instance_root=instance_root_2, splits={"train": splits_2["train"]})
            cmp = audit.compare_fingerprints(fp1, fp2)
            self.assertIn("m02_p01_s00", cmp["asset_differences"]["semantic_mask_sha_mismatch"])
            self.assertIn("m02_p01_s00", cmp["asset_differences"]["semantic_file_bytes_differ_pixels_identical"])
            self.assertNotIn("m02_p01_s00", cmp["asset_differences"]["semantic_pixels_differ"])
            self.assertEqual(cmp["classification"], "IDENTICAL_DATASET")

    def test_semantic_pixel_difference_detection(self):
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            dataset_root_1, instance_root_1, splits_1 = self._make_dataset(Path(td1))
            dataset_root_2, instance_root_2, splits_2 = self._make_dataset(Path(td2), semantic_variant=1)
            fp1 = audit.build_fingerprint(dataset_root=dataset_root_1, instance_root=instance_root_1, splits={"train": splits_1["train"]})
            fp2 = audit.build_fingerprint(dataset_root=dataset_root_2, instance_root=instance_root_2, splits={"train": splits_2["train"]})
            cmp = audit.compare_fingerprints(fp1, fp2)
            self.assertIn("m02_p01_s00", cmp["asset_differences"]["semantic_pixels_differ"])
            self.assertEqual(cmp["classification"], "DIFFERENT_ASSETS")

    def test_script_is_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            dataset_root, instance_root, splits = self._make_dataset(Path(td))
            tracked = sorted([
                dataset_root / "images" / "m01_p01_s00.png",
                dataset_root / "masks" / "m01_p01_s00.png",
                instance_root / "instance_masks" / "m01_p01_s00.png",
                splits["train"],
                splits["val"],
            ])
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
            _fp = audit.build_fingerprint(dataset_root=dataset_root, instance_root=instance_root, splits={"train": splits["train"], "val": splits["val"]})
            after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
            self.assertEqual(before, after)

    def test_compare_split_dirs_and_sync(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            canonical_dir = root / "canonical"
            external_dir = root / "external"
            canonical_dir.mkdir(parents=True, exist_ok=True)
            external_dir.mkdir(parents=True, exist_ok=True)
            for split_name, rows in {
                "train.txt": ["images/a.png\tmasks/a.png", "images/b.png\tmasks/b.png"],
                "val.txt": ["images/c.png\tmasks/c.png"],
                "test.txt": ["images/d.png\tmasks/d.png"],
            }.items():
                (canonical_dir / split_name).write_text("\n".join(rows) + "\n", encoding="utf-8")
            (external_dir / "train.txt").write_text("images/b.png\tmasks/b.png\nimages/a.png\tmasks/a.png\n", encoding="utf-8")
            (external_dir / "val.txt").write_text("images/c.png\tmasks/c.png\n", encoding="utf-8")
            (external_dir / "test.txt").write_text("images/d.png\tmasks/d.png\n", encoding="utf-8")
            drift = audit.compare_split_dirs(canonical_dir, external_dir)
            self.assertEqual(drift["status"], "DRIFT")
            self.assertFalse(drift["splits"]["train"]["match"])
            sync = audit.sync_split_dirs(canonical_dir, external_dir)
            self.assertTrue(sync["performed"])
            self.assertEqual(sync["verify"]["status"], "MATCH")
            self.assertTrue((external_dir / "_backup_before_canonical_sync" / "train.txt").exists())
            after = audit.compare_split_dirs(canonical_dir, external_dir)
            self.assertEqual(after["status"], "MATCH")


class WindowsResearchLineageTests(unittest.TestCase):
    def test_current_windows_test_matches_preserved_research_eval_contract(self):
        split_path = Path("datasets/converted_full_multiclass_curated/test.txt").resolve()
        artifact_path = Path("training/analysis/semantic_topology_aux_preflight/research_eval_isolation.json").resolve()
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        raw_sha = audit.raw_split_sha256(split_path)
        rows, _canonical_sha = audit.canonical_split_sha256(split_path)
        sample_ids = [Path(row.split("\t", 1)[0]).stem for row in rows]
        patient_ids = sorted({sample_id.rsplit("_", 1)[0] for sample_id in sample_ids})
        self.assertEqual(raw_sha, artifact["research_eval"]["sha256"])
        self.assertEqual(len(rows), 20)
        self.assertEqual(sorted(sample_ids), sorted(artifact["research_eval"]["samples"]))
        self.assertEqual(len(patient_ids), 15)

    def test_tracked_v1_manifests_match_current_external_windows_split(self):
        tracked_dir = Path("training/manifests/converted_full_multiclass_curated_v1").resolve()
        external_dir = Path("datasets/converted_full_multiclass_curated").resolve()
        report = audit.compare_split_dirs(tracked_dir, external_dir)
        self.assertEqual(report["status"], "MATCH")
        self.assertEqual(report["splits"]["train"]["canonical_sha256"], "f5e920ffaf54c0a0034c457cf3c951f71e186a9f35e3fe67a5eee95737b2ee82")
        self.assertEqual(report["splits"]["val"]["canonical_sha256"], "b9b4151c48fe7824be1a3e25c4f519ebadf6353a58f717f9679fde2ca8ff376c")
        self.assertEqual(report["splits"]["test"]["canonical_sha256"], "60e48e7c6416f75adbfbc8b5024bbb2b4b864052a53e97c2c6ad1b3b1256a568")


class CuratedSplitGeneratorDeterminismTests(unittest.TestCase):
    def test_make_curated_split_is_deterministic_across_subprocesses(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            converted_root = root / "datasets" / "converted_full_multiclass"
            curated_dir = root / "datasets" / "curated_full_multiclass"
            for split_name, rows in {
                "train.txt": [
                    "images/m01_p01_s00.png\tmasks/m01_p01_s00.png",
                    "images/m01_p01_s01.png\tmasks/m01_p01_s01.png",
                    "images/m01_p01_s02.png\tmasks/m01_p01_s02.png",
                ],
                "val.txt": [
                    "images/m02_p01_s00.png\tmasks/m02_p01_s00.png",
                    "images/m02_p01_s01.png\tmasks/m02_p01_s01.png",
                ],
                "test.txt": [
                    "images/m03_p01_s00.png\tmasks/m03_p01_s00.png",
                ],
            }.items():
                (converted_root / split_name).parent.mkdir(parents=True, exist_ok=True)
                (converted_root / split_name).write_text("\n".join(rows) + "\n", encoding="utf-8")
            curated_dir.mkdir(parents=True, exist_ok=True)
            (curated_dir / "clean.txt").write_text("m01_p01_s00\nm01_p01_s01\n", encoding="utf-8")
            (curated_dir / "medium.txt").write_text("m01_p01_s02\nm02_p01_s00\nm02_p01_s01\nm03_p01_s00\n", encoding="utf-8")
            (curated_dir / "bad.txt").write_text("", encoding="utf-8")
            (curated_dir / "unset.txt").write_text("", encoding="utf-8")
            shas: list[str] = []
            for idx in range(2):
                out_dir = root / f"out_{idx}"
                cmd = [
                    sys.executable,
                    str(Path("scripts/make_curated_split.py").resolve()),
                    "--converted-root",
                    str(converted_root),
                    "--curated-dir",
                    str(curated_dir),
                    "--out-dir",
                    str(out_dir),
                    "--seed",
                    "1337",
                ]
                subprocess.run(cmd, check=True, cwd=str(Path(".").resolve()), capture_output=True, text=True)
                rows = [x.strip() for x in (out_dir / "train.txt").read_text(encoding="utf-8-sig").splitlines() if x.strip()]
                shas.append(hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest())
            self.assertEqual(shas[0], shas[1])


def copy_payload(payload: dict) -> dict:
    return json.loads(json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
