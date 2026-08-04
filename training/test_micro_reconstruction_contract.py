from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch


_THIS_DIR = Path(__file__).resolve().parent


class TestMicroReconstructionContract(unittest.TestCase):
    def _import_helpers(self):
        import sys

        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        from validate_centerhead import compute_instance_metrics_from_masks, reconstruct_instances_from_semantic_and_center

        return reconstruct_instances_from_semantic_and_center, compute_instance_metrics_from_masks

    def test_two_instance_contract_holds(self):
        reconstruct, compute_metrics = self._import_helpers()

        pred_sem = np.zeros((32, 32), dtype=np.uint8)
        pred_sem[4:12, 4:12] = 1
        pred_sem[20:28, 20:28] = 1
        center_prob = np.zeros((32, 32), dtype=np.float32)
        center_prob[8, 8] = 0.95
        center_prob[24, 24] = 0.93
        gt_inst = np.zeros((32, 32), dtype=np.uint8)
        gt_inst[4:12, 4:12] = 1
        gt_inst[20:28, 20:28] = 2

        pred_inst, pred_k, pred_pts_scored, trace = reconstruct(
            pred_sem,
            center_prob,
            0.5,
            max_markers=3,
            return_trace=True,
        )
        metrics = compute_metrics(gt_inst, pred_inst, gt_k=2, pred_k=pred_k)

        self.assertEqual(len(pred_pts_scored), 2)
        self.assertEqual(pred_k, 2)
        self.assertTrue(bool(metrics["instance_exact_count"]))
        self.assertFalse(bool(metrics["instance_fragmented"]))
        self.assertFalse(bool(metrics["instance_merged"]))
        self.assertEqual(int(trace["raw_reconstruction_count"]), 2)
        self.assertEqual(int(trace["final_count"]), 2)

    def test_disconnected_semantic_component_current_behavior(self):
        reconstruct, compute_metrics = self._import_helpers()

        pred_sem = np.zeros((40, 40), dtype=np.uint8)
        pred_sem[8:16, 6:14] = 1
        pred_sem[8:16, 24:32] = 1
        center_prob = np.zeros((40, 40), dtype=np.float32)
        center_prob[12, 10] = 0.98
        gt_inst = np.zeros((40, 40), dtype=np.uint8)
        gt_inst[8:16, 6:14] = 1
        gt_inst[8:16, 24:32] = 1

        pred_inst, pred_k, pred_pts_scored, trace = reconstruct(
            pred_sem,
            center_prob,
            0.5,
            max_markers=3,
            return_trace=True,
        )
        metrics = compute_metrics(gt_inst, pred_inst, gt_k=1, pred_k=pred_k)

        self.assertEqual(len(pred_pts_scored), 1)
        self.assertEqual(int(trace["semantic_component_count"]), 2)
        self.assertTrue(any(bool(comp["used_fallback"]) for comp in trace["component_traces"]))
        self.assertEqual(pred_k, 2)
        self.assertFalse(bool(metrics["instance_exact_count"]))
        self.assertTrue(bool(metrics["instance_fragmented"]))
        self.assertFalse(bool(metrics["instance_merged"]))


class TestAuditMicroReconstructionCli(unittest.TestCase):
    def _import_audit_helpers(self):
        import sys

        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        from audit_micro_reconstruction_contract import (
            ArtifactResolutionError,
            _normalized_microset_sha256,
            _parse_microset_file,
            _resolve_output_dir_arg,
            _resolve_microset_path,
            _verify_artifacts,
            build_arg_parser,
        )

        return (
            ArtifactResolutionError,
            _normalized_microset_sha256,
            _parse_microset_file,
            _resolve_output_dir_arg,
            _resolve_microset_path,
            _verify_artifacts,
            build_arg_parser,
        )

    def _make_cfg(self, repo_root: Path, extra_dataset: dict | None = None) -> dict:
        dataset = {
            "root": "datasets/converted_leaflet_distance",
            "instance_root": "datasets/converted_leaflet_instances",
        }
        if extra_dataset:
            dataset.update(extra_dataset)
        return {
            "seed": 1337,
            "dataset": dataset,
            "model": {
                "encoder_name": "efficientnet-b3",
                "encoder_weights": None,
                "in_channels": 3,
                "classes": 3,
                "input_size": 768,
                "center_feature": {
                    "module_path": "base.decoder.blocks.x_2_2",
                    "expected_channels": 32,
                    "adapter_out_channels": 16,
                    "native_stride": 4,
                    "upsample_logits_to_target": True,
                },
            },
            "center_loss": {"normalization_mode": "legacy_num_pos"},
            "train": {
                "center_fp32": True,
                "init_checkpoint": str((repo_root / "dummy_init.pth").resolve()),
            },
        }

    def _write_microset(self, path: Path, sample_ids: list[str], newline: str = "\n") -> None:
        rows = [f"images/{sid}.png\tsemantic_masks/{sid}.png" for sid in sample_ids]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(newline.join(rows) + newline, encoding="utf-8", newline="")

    def _write_dataset(self, repo_root: Path, sample_ids: list[str]) -> None:
        for sid in sample_ids:
            img = repo_root / "datasets/converted_leaflet_distance/images" / f"{sid}.png"
            mask = repo_root / "datasets/converted_leaflet_distance/semantic_masks" / f"{sid}.png"
            inst = repo_root / "datasets/converted_leaflet_instances/instance_masks" / f"{sid}.png"
            img.parent.mkdir(parents=True, exist_ok=True)
            mask.parent.mkdir(parents=True, exist_ok=True)
            inst.parent.mkdir(parents=True, exist_ok=True)
            img.write_bytes(b"\x89PNG\r\n")
            mask.write_bytes(b"\x89PNG\r\n")
            inst.write_bytes(b"\x89PNG\r\n")

    def _write_dummy_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {}, "optimizer": {}, "step": 75, "extra": {"best_threshold": 0.03, "best_center_f1": 0.9}}, path)

    def _make_required_artifacts(self, repo_root: Path, sample_ids: list[str], microset_path: Path) -> tuple[dict, dict, dict]:
        self._write_dataset(repo_root, sample_ids)
        self._write_microset(microset_path, sample_ids)
        run_dir = repo_root / "training/runs/unetpp_effb3_centerhead_spatial_x2_2_adapter_legacy_fp32_micro"
        best_ckpt = run_dir / "best_micro_overfit.pth"
        last_ckpt = run_dir / "last.pth"
        self._write_dummy_checkpoint(best_ckpt)
        self._write_dummy_checkpoint(last_ckpt)
        (run_dir / "micro_overfit_metrics.csv").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "micro_overfit_metrics.csv").write_text("iter,best_f1\n75,0.9\n", encoding="utf-8")
        sweep_dir = run_dir / "threshold_sweeps"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        for it in (75, 100, 500, 525, 1000):
            (sweep_dir / f"iter_{it:04d}.json").write_text("{}", encoding="utf-8")
        (run_dir / "microset_manifest.json").write_text(
            '{"samples": ["' + '", "'.join(sample_ids) + '"]}',
            encoding="utf-8",
        )
        resolved_paths = {
            "run_dir": str(run_dir.resolve()),
            "microset": str(microset_path.resolve()),
            "best_checkpoint": str(best_ckpt.resolve()),
            "last_checkpoint": str(last_ckpt.resolve()),
            "metrics_csv": str((run_dir / "micro_overfit_metrics.csv").resolve()),
            "summary_json": str((run_dir / "summary.json").resolve()),
            "threshold_sweep_dir": str(sweep_dir.resolve()),
            "microset_manifest": str((run_dir / "microset_manifest.json").resolve()),
        }
        (run_dir / "summary.json").write_text("{}", encoding="utf-8")
        checkpoint_metadata = {
            "best": {"checkpoint_path": str(best_ckpt.resolve())},
            "last": {"checkpoint_path": str(last_ckpt.resolve())},
        }
        return resolved_paths, checkpoint_metadata, {"run_dir": run_dir}

    def test_explicit_microset_path_wins(self):
        (
            ArtifactResolutionError,
            _normalized_microset_sha256,
            _parse_microset_file,
            _resolve_output_dir_arg,
            _resolve_microset_path,
            _verify_artifacts,
            build_arg_parser,
        ) = self._import_audit_helpers()
        with TemporaryDirectory() as td:
            repo_root = Path(td)
            run_dir = repo_root / "training/runs/unetpp_effb3_centerhead_spatial_x2_2_adapter_legacy_fp32_micro"
            explicit = repo_root / "verified/microset.txt"
            self._write_microset(run_dir / "microset.txt", ["a", "b", "c", "d", "e", "f"])
            self._write_microset(explicit, ["m", "n", "o", "p", "q", "r"])
            chosen, candidates = _resolve_microset_path(run_dir=run_dir, explicit_microset=explicit)
            self.assertEqual(chosen.resolve(), explicit.resolve())
            self.assertEqual(len(candidates), 1)

    def test_missing_microset_raises(self):
        (
            ArtifactResolutionError,
            _normalized_microset_sha256,
            _parse_microset_file,
            _resolve_output_dir_arg,
            _resolve_microset_path,
            _verify_artifacts,
            build_arg_parser,
        ) = self._import_audit_helpers()
        with TemporaryDirectory() as td:
            repo_root = Path(td)
            run_dir = repo_root / "training/runs/unetpp_effb3_centerhead_spatial_x2_2_adapter_legacy_fp32_micro"
            with self.assertRaises(ArtifactResolutionError):
                _resolve_microset_path(run_dir=run_dir, explicit_microset=None)

    def test_valid_run_dir_microset_is_used(self):
        (
            ArtifactResolutionError,
            _normalized_microset_sha256,
            _parse_microset_file,
            _resolve_output_dir_arg,
            _resolve_microset_path,
            _verify_artifacts,
            build_arg_parser,
        ) = self._import_audit_helpers()
        with TemporaryDirectory() as td:
            repo_root = Path(td)
            run_dir = repo_root / "training/runs/unetpp_effb3_centerhead_spatial_x2_2_adapter_legacy_fp32_micro"
            run_microset = run_dir / "microset.txt"
            self._write_microset(run_microset, ["a", "b", "c", "d", "e", "f"])
            chosen, _candidates = _resolve_microset_path(run_dir=run_dir, explicit_microset=None)
            self.assertEqual(chosen.resolve(), run_microset.resolve())

    def test_wrong_number_of_samples_fails_precheck(self):
        (
            ArtifactResolutionError,
            _normalized_microset_sha256,
            _parse_microset_file,
            _resolve_output_dir_arg,
            _resolve_microset_path,
            _verify_artifacts,
            build_arg_parser,
        ) = self._import_audit_helpers()
        with TemporaryDirectory() as td:
            repo_root = Path(td)
            cfg = self._make_cfg(repo_root)
            sample_ids = ["a", "b", "c", "d", "e"]
            microset_path = repo_root / "training/runs/unetpp_effb3_centerhead_spatial_x2_2_adapter_legacy_fp32_micro/microset.txt"
            resolved_paths, checkpoint_metadata, _extra = self._make_required_artifacts(repo_root, sample_ids, microset_path)
            microset_info = _parse_microset_file(microset_path, (repo_root / "datasets/converted_leaflet_distance").resolve())
            missing = _verify_artifacts(
                cfg=cfg,
                resolved_paths=resolved_paths,
                microset_info=microset_info,
                checkpoint_metadata=checkpoint_metadata,
            )
            self.assertTrue(any("exactly 6 non-empty lines" in item for item in missing))

    def test_normalized_crlf_lf_equivalence(self):
        (
            ArtifactResolutionError,
            _normalized_microset_sha256,
            _parse_microset_file,
            _resolve_output_dir_arg,
            _resolve_microset_path,
            _verify_artifacts,
            build_arg_parser,
        ) = self._import_audit_helpers()
        with TemporaryDirectory() as td:
            root = Path(td)
            a = root / "a.txt"
            b = root / "b.txt"
            self._write_microset(a, ["a", "b", "c", "d", "e", "f"], newline="\n")
            self._write_microset(b, ["a", "b", "c", "d", "e", "f"], newline="\r\n")
            self.assertEqual(_normalized_microset_sha256(a), _normalized_microset_sha256(b))

    def test_genuinely_different_normalized_microsets(self):
        (
            ArtifactResolutionError,
            _normalized_microset_sha256,
            _parse_microset_file,
            _resolve_output_dir_arg,
            _resolve_microset_path,
            _verify_artifacts,
            build_arg_parser,
        ) = self._import_audit_helpers()
        with TemporaryDirectory() as td:
            root = Path(td)
            a = root / "a.txt"
            b = root / "b.txt"
            self._write_microset(a, ["a", "b", "c", "d", "e", "f"], newline="\n")
            self._write_microset(b, ["a", "b", "c", "d", "e", "g"], newline="\n")
            self.assertNotEqual(_normalized_microset_sha256(a), _normalized_microset_sha256(b))

    def test_out_dir_alias(self):
        (
            ArtifactResolutionError,
            _normalized_microset_sha256,
            _parse_microset_file,
            _resolve_output_dir_arg,
            _resolve_microset_path,
            _verify_artifacts,
            build_arg_parser,
        ) = self._import_audit_helpers()
        parser = build_arg_parser()
        args = parser.parse_args(["--out-dir", "foo/bar"])
        self.assertEqual(_resolve_output_dir_arg(args.output_dir, args.out_dir), "foo/bar")

    def test_output_dir_backward_compatibility(self):
        (
            ArtifactResolutionError,
            _normalized_microset_sha256,
            _parse_microset_file,
            _resolve_output_dir_arg,
            _resolve_microset_path,
            _verify_artifacts,
            build_arg_parser,
        ) = self._import_audit_helpers()
        parser = build_arg_parser()
        args = parser.parse_args(["--output-dir", "legacy/out"])
        self.assertEqual(_resolve_output_dir_arg(args.output_dir, args.out_dir), "legacy/out")

    def test_conflicting_out_dir_and_output_dir(self):
        (
            ArtifactResolutionError,
            _normalized_microset_sha256,
            _parse_microset_file,
            _resolve_output_dir_arg,
            _resolve_microset_path,
            _verify_artifacts,
            build_arg_parser,
        ) = self._import_audit_helpers()
        parser = build_arg_parser()
        args = parser.parse_args(["--out-dir", "foo/bar", "--output-dir", "other/bar"])
        with self.assertRaises(ArtifactResolutionError):
            _resolve_output_dir_arg(args.output_dir, args.out_dir)


if __name__ == "__main__":
    unittest.main()
