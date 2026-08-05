from __future__ import annotations

import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


_THIS_DIR = Path(__file__).resolve().parent


class TestAuthoritativeHoldoutPipeline(unittest.TestCase):
    @staticmethod
    def _mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import run_authoritative_holdout_pipeline as mod

        return mod

    def _populate_required_artifacts(self, mod, out: Path) -> None:
        for rel, spec in mod.REQUIRED_ARTIFACTS.items():
            path = out / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if spec["type"] == "json":
                path.write_text("{}", encoding="utf-8")
            elif spec["type"] == "csv":
                path.write_text("h\n1\n", encoding="utf-8")
            elif spec["type"] == "jsonl":
                path.write_text(''.join(json.dumps({"sample": f"s{i}"}) + "\n" for i in range(106)), encoding="utf-8")
            else:
                path.write_text("x", encoding="utf-8")

    def test_stage_order(self):
        mod = self._mod()
        order = []
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            out = repo / "training" / "analysis" / "out"
            out.mkdir(parents=True, exist_ok=True)
            args = SimpleNamespace(
                config="cfg",
                run_dir="training/analysis/run",
                output_dir=str(out),
                expected_manifest_identity_sha="sha",
                expected_center_checkpoint_sha="center",
                expected_semantic_checkpoint_sha="semantic",
                device="cpu",
                clean_output=False,
                skip_tests=False,
                include_visual_review=False,
            )
            with mock.patch.object(mod, "_parse_args", return_value=args), \
                mock.patch.object(mod, "_safe_output_dir"), \
                mock.patch.object(mod, "_repo_preflight", side_effect=lambda *a, **k: order.append("preflight") or {"git": {"commit": "abc", "branch": "main", "tracked_tree_clean": True}, "environment": {"hostname": "host", "device": "cpu", "python": "3.12", "torch": "2.x", "cuda_available": False}}), \
                mock.patch.object(mod, "_run_tests", side_effect=lambda *a, **k: order.append("tests") or mod.StageResult("tests", 0, 0.0, {})), \
                mock.patch.object(mod, "_checkpoint_identity", side_effect=lambda *a, **k: order.append("checkpoint") or mod.StageResult("checkpoint", 0, 0.0, {"center_checkpoint_sha": "c", "semantic_checkpoint_sha": "s"})), \
                mock.patch.object(mod, "_run_manifest_stage", side_effect=lambda *a, **k: order.append("manifest") or mod.StageResult("manifest", 0, 0.0, {"samples": 106, "unique_samples": 106, "split_counts": {"test": 53, "val": 53}, "gt_count_distribution": {"1": 15, "2": 37, "3": 54}, "manifest_identity_sha": "sha"})), \
                mock.patch.object(mod, "_run_diagnosis_stage", side_effect=lambda *a, **k: order.append("diagnosis") or mod.StageResult("diagnosis", 0, 0.0, {"bottleneck_status": "mixed"})), \
                mock.patch.object(mod, "_read_json", side_effect=[{"checkpoint_identity_status": "exact_match", "semantic_checkpoint_identity_status": "exact_match", "manifest_identity_status": "exact_match", "diagnosis_execution_status": "completed", "overall_authoritative_status": "exact_match"}, {"production_activation_result": {"status": "blocked"}}]), \
                mock.patch.object(mod, "_inspect_artifacts", side_effect=lambda *a, **k: order.append("artifacts") or ({"status": "passed", "required": [], "optional": [], "generated_only_by_another_pipeline": [], "missing_required": [], "malformed": []}, [], [], [])), \
                mock.patch.object(mod, "_bundle_review", side_effect=lambda *a, **k: order.append("bundle") or (out / "bundle.tar.gz", out / "bundle.tar.gz.sha256")), \
                mock.patch.object(mod, "_write_files_to_provide"), \
                mock.patch.object(mod, "_atomic_write_json"), \
                mock.patch.object(mod, "TeeLogger") as mock_logger:
                mock_logger.return_value = mock.MagicMock()
                with self.assertRaises(SystemExit) as cm:
                    mod.main()
                self.assertEqual(cm.exception.code, 0)
        self.assertEqual(order, ["preflight", "tests", "checkpoint", "manifest", "diagnosis", "artifacts", "bundle"])

    def test_stop_after_test_failure(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            args = SimpleNamespace(
                config="cfg",
                run_dir="training/analysis/run",
                output_dir=str(repo / "training" / "analysis" / "out"),
                expected_manifest_identity_sha="sha",
                expected_center_checkpoint_sha="center",
                expected_semantic_checkpoint_sha="semantic",
                device="cpu",
                clean_output=False,
                skip_tests=False,
                include_visual_review=False,
            )
            with mock.patch.object(mod, "_parse_args", return_value=args), \
                mock.patch.object(mod, "_safe_output_dir"), \
                mock.patch.object(mod, "_repo_preflight", return_value={"git": {}, "environment": {}}), \
                mock.patch.object(mod, "_run_tests", side_effect=mod.PipelineFailure(stage="tests", reason="boom", exit_code=mod.EXIT_TESTS_FAILED)), \
                mock.patch.object(mod, "_checkpoint_identity") as checkpoint_mock:
                with self.assertRaises(SystemExit) as cm:
                    mod.main()
                self.assertEqual(cm.exception.code, mod.EXIT_TESTS_FAILED)
                checkpoint_mock.assert_not_called()

    def test_stop_after_checkpoint_mismatch(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                config="cfg",
                run_dir="training/analysis/run",
                output_dir=str(Path(tmp) / "training" / "analysis" / "out"),
                expected_manifest_identity_sha="sha",
                expected_center_checkpoint_sha="center",
                expected_semantic_checkpoint_sha="semantic",
                device="cpu",
                clean_output=False,
                skip_tests=True,
                include_visual_review=False,
            )
            with mock.patch.object(mod, "_parse_args", return_value=args), \
                mock.patch.object(mod, "_safe_output_dir"), \
                mock.patch.object(mod, "_repo_preflight", return_value={"git": {}, "environment": {}}), \
                mock.patch.object(mod, "_checkpoint_identity", side_effect=mod.PipelineFailure(stage="checkpoint_identity", reason="bad", exit_code=mod.EXIT_CHECKPOINT_IDENTITY_FAILED)), \
                mock.patch.object(mod, "_run_manifest_stage") as manifest_mock:
                with self.assertRaises(SystemExit) as cm:
                    mod.main()
                self.assertEqual(cm.exception.code, mod.EXIT_CHECKPOINT_IDENTITY_FAILED)
                manifest_mock.assert_not_called()

    def test_stop_after_manifest_mismatch(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                config="cfg",
                run_dir="training/analysis/run",
                output_dir=str(Path(tmp) / "training" / "analysis" / "out"),
                expected_manifest_identity_sha="sha",
                expected_center_checkpoint_sha="center",
                expected_semantic_checkpoint_sha="semantic",
                device="cpu",
                clean_output=False,
                skip_tests=True,
                include_visual_review=False,
            )
            with mock.patch.object(mod, "_parse_args", return_value=args), \
                mock.patch.object(mod, "_safe_output_dir"), \
                mock.patch.object(mod, "_repo_preflight", return_value={"git": {}, "environment": {}}), \
                mock.patch.object(mod, "_checkpoint_identity", return_value=mod.StageResult("checkpoint", 0, 0.0, {"center_checkpoint_sha": "c", "semantic_checkpoint_sha": "s"})), \
                mock.patch.object(mod, "_run_manifest_stage", side_effect=mod.PipelineFailure(stage="manifest_generation", reason="bad sha", exit_code=mod.EXIT_MANIFEST_IDENTITY_MISMATCH)), \
                mock.patch.object(mod, "_run_diagnosis_stage") as diagnosis_mock:
                with self.assertRaises(SystemExit) as cm:
                    mod.main()
                self.assertEqual(cm.exception.code, mod.EXIT_MANIFEST_IDENTITY_MISMATCH)
                diagnosis_mock.assert_not_called()

    def test_exact_authoritative_statuses_required(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            out.mkdir(exist_ok=True)
            (out / "checkpoint_identity.json").write_text(json.dumps({
                "checkpoint_identity_status": "exact_match",
                "semantic_checkpoint_identity_status": "exact_match",
                "manifest_identity_status": "exact_match",
                "diagnosis_execution_status": "running_diagnostics",
                "overall_authoritative_status": "exact_match",
            }), encoding="utf-8")
            (out / "bottleneck_decision.json").write_text(json.dumps({"status": "mixed"}), encoding="utf-8")
            with mock.patch.object(mod, "_run_command", return_value=(0, 0.0)):
                with self.assertRaises(mod.PipelineFailure) as cm:
                    mod._run_diagnosis_stage(Path("."), config="cfg", run_dir="run", output_dir=str(out), device="cpu", expected_sha="sha", logger=mock.MagicMock())
            self.assertEqual(cm.exception.exit_code, mod.EXIT_AUTHORITATIVE_STATUS_MISMATCH)

    def test_unsafe_output_path_rejected(self):
        mod = self._mod()
        repo = Path("E:/3d_visual/ml").resolve()
        with self.assertRaises(mod.PipelineFailure):
            mod._safe_output_dir(repo, repo / "training")

    def test_summary_created_after_failure(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            out = repo / "training" / "analysis" / "out"
            args = SimpleNamespace(
                config="cfg",
                run_dir="training/analysis/run",
                output_dir=str(out),
                expected_manifest_identity_sha="sha",
                expected_center_checkpoint_sha="center",
                expected_semantic_checkpoint_sha="semantic",
                device="cpu",
                clean_output=False,
                skip_tests=False,
                include_visual_review=False,
            )
            with mock.patch.object(mod, "_parse_args", return_value=args), \
                mock.patch.object(mod, "_safe_output_dir"), \
                mock.patch.object(mod, "_repo_preflight", return_value={"git": {}, "environment": {}}), \
                mock.patch.object(mod, "_run_tests", side_effect=mod.PipelineFailure(stage="tests", reason="boom", exit_code=mod.EXIT_TESTS_FAILED)):
                with self.assertRaises(SystemExit):
                    mod.main()
            self.assertTrue((out / "pipeline_run_summary.json").exists())

    def test_artifact_integrity_catches_missing_json(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaises(mod.PipelineFailure):
                report, _index, _files, _large = mod._inspect_artifacts(out, include_visual_review=False)
                if report["status"] != "passed":
                    raise mod.PipelineFailure(stage="artifact_integrity", reason="Artifact integrity failed", exit_code=mod.EXIT_ARTIFACT_INTEGRITY_FAILED)

    def test_artifact_integrity_catches_malformed_json(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for rel in mod.REQUIRED_ARTIFACTS:
                path = out / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                if rel.endswith(".json"):
                    path.write_text("{bad", encoding="utf-8")
                elif rel.endswith(".csv"):
                    path.write_text("a\n1\n", encoding="utf-8")
                elif rel.endswith(".jsonl"):
                    path.write_text('{"sample":"x"}\n' * 106, encoding="utf-8")
                else:
                    path.write_text("x", encoding="utf-8")
            report, _index, _files, _large = mod._inspect_artifacts(out, include_visual_review=False)
            self.assertEqual(report["status"], "failed")

    def test_artifact_integrity_catches_empty_csv(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            path.write_text("header\n", encoding="utf-8")
            with self.assertRaises(mod.PipelineFailure):
                mod._validate_csv_file(path)

    def test_bundle_excludes_checkpoints(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for name in ("pipeline_run_summary.json", "pipeline.log", "artifact_index.json", "files_to_provide.txt"):
                (out / name).write_text("x", encoding="utf-8")
            (out / "bad_checkpoint.pth").write_text("secret", encoding="utf-8")
            artifact_index = [
                {"relative_path": "pipeline_run_summary.json"},
                {"relative_path": "pipeline.log"},
                {"relative_path": "artifact_index.json"},
                {"relative_path": "files_to_provide.txt"},
                {"relative_path": "bad_checkpoint.pth"},
            ]
            bundle, _ = mod._bundle_review(out, artifact_index=artifact_index, include_visual_review=False)
            with tarfile.open(bundle, "r:gz") as tar:
                self.assertNotIn("bad_checkpoint.pth", tar.getnames())

    def test_bundle_contains_required_review_artifacts(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for name in ("pipeline_run_summary.json", "pipeline.log", "artifact_index.json", "files_to_provide.txt"):
                (out / name).write_text("x", encoding="utf-8")
            artifact_index = [
                {"relative_path": "pipeline_run_summary.json"},
                {"relative_path": "pipeline.log"},
                {"relative_path": "artifact_index.json"},
                {"relative_path": "files_to_provide.txt"},
            ]
            bundle, _ = mod._bundle_review(out, artifact_index=artifact_index, include_visual_review=False)
            with tarfile.open(bundle, "r:gz") as tar:
                names = set(tar.getnames())
            self.assertIn("pipeline_run_summary.json", names)
            self.assertIn("pipeline.log", names)

    def test_files_to_provide_printed_on_success(self):
        mod = self._mod()
        block = mod._final_success_block(
            {
                "git": {"commit": "abc"},
                "environment": {"hostname": "host", "device": "cuda"},
                "dataset": {"samples": 106},
                "identity": {"manifest_identity_sha": "sha"},
                "authoritative_status": {"overall": "exact_match"},
                "bottleneck_status": "mixed",
                "production_activation": "blocked",
                "training_launched": False,
            },
            Path("/tmp/bundle.tar.gz"),
            Path("/tmp/bundle.tar.gz.sha256"),
            [],
        )
        self.assertIn("FILES TO PROVIDE:", block)

    def test_files_to_provide_printed_on_failure(self):
        mod = self._mod()
        block = mod._final_failure_block({"failed_stage": "tests", "failure_reason": "bad"}, Path("/tmp/out"), mod.EXIT_TESTS_FAILED)
        self.assertIn("FILES TO PROVIDE:", block)

    def test_stable_exit_codes(self):
        mod = self._mod()
        self.assertEqual(mod.EXIT_SUCCESS, 0)
        self.assertEqual(mod.EXIT_REPOSITORY_PREFLIGHT_FAILED, 10)
        self.assertEqual(mod.EXIT_TESTS_FAILED, 20)
        self.assertEqual(mod.EXIT_CHECKPOINT_IDENTITY_FAILED, 30)
        self.assertEqual(mod.EXIT_MANIFEST_GENERATION_FAILED, 40)
        self.assertEqual(mod.EXIT_MANIFEST_IDENTITY_MISMATCH, 41)
        self.assertEqual(mod.EXIT_DIAGNOSIS_SUBPROCESS_FAILED, 50)
        self.assertEqual(mod.EXIT_AUTHORITATIVE_STATUS_MISMATCH, 51)
        self.assertEqual(mod.EXIT_ARTIFACT_INTEGRITY_FAILED, 60)
        self.assertEqual(mod.EXIT_BUNDLE_CREATION_FAILED, 70)

    def test_stage1_module_list_includes_both_runner_test_modules(self):
        mod = self._mod()
        self.assertIn("training.test_authoritative_holdout_pipeline", mod.REQUIRED_TEST_MODULES)
        self.assertIn("training.test_cpu_cuda_replay_parity_runner", mod.REQUIRED_TEST_MODULES)

    def test_failure_summary_records_passed_stage0_and_failed_stage1(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            out = repo / "training" / "analysis" / "out"
            args = SimpleNamespace(
                config="cfg",
                run_dir="training/analysis/run",
                output_dir=str(out),
                expected_manifest_identity_sha="sha",
                expected_center_checkpoint_sha="center",
                expected_semantic_checkpoint_sha="semantic",
                device="cpu",
                clean_output=False,
                skip_tests=False,
                include_visual_review=False,
            )
            with mock.patch.object(mod, "_parse_args", return_value=args), \
                mock.patch.object(mod, "_safe_output_dir"), \
                mock.patch.object(mod, "_repo_preflight", return_value={"git": {"commit": "abc", "branch": "main", "tracked_tree_clean": True}, "environment": {"hostname": "host", "device": "cpu", "python": "3.12", "torch": "2.x", "cuda_available": False}}), \
                mock.patch.object(mod, "_run_tests", side_effect=mod.PipelineFailure(stage="tests", reason="boom", exit_code=mod.EXIT_TESTS_FAILED)):
                with self.assertRaises(SystemExit):
                    mod.main()
            summary = json.loads((out / "pipeline_run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["stages"][0]["stage"], "repository_preflight")
            self.assertEqual(summary["stages"][0]["status"], "passed")
            self.assertEqual(summary["stages"][1]["stage"], "tests")
            self.assertEqual(summary["stages"][1]["status"], "failed")

    def test_failure_summary_files_to_provide_is_non_empty(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            out = repo / "training" / "analysis" / "out"
            args = SimpleNamespace(
                config="cfg",
                run_dir="training/analysis/run",
                output_dir=str(out),
                expected_manifest_identity_sha="sha",
                expected_center_checkpoint_sha="center",
                expected_semantic_checkpoint_sha="semantic",
                device="cpu",
                clean_output=False,
                skip_tests=False,
                include_visual_review=False,
            )
            with mock.patch.object(mod, "_parse_args", return_value=args), \
                mock.patch.object(mod, "_safe_output_dir"), \
                mock.patch.object(mod, "_repo_preflight", return_value={"git": {"commit": "abc", "branch": "main", "tracked_tree_clean": True}, "environment": {"hostname": "host", "device": "cpu", "python": "3.12", "torch": "2.x", "cuda_available": False}}), \
                mock.patch.object(mod, "_run_tests", side_effect=mod.PipelineFailure(stage="tests", reason="boom", exit_code=mod.EXIT_TESTS_FAILED)):
                with self.assertRaises(SystemExit):
                    mod.main()
            summary = json.loads((out / "pipeline_run_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["files_to_provide"])
            self.assertIn(str((out / "pipeline_run_summary.json").resolve()), summary["files_to_provide"])
            self.assertIn(str((out / "pipeline.log").resolve()), summary["files_to_provide"])

    def test_json_files_to_provide_equals_console_files_list(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = {
                "failed_stage": "tests",
                "failure_reason": "boom",
                "files_to_provide": [
                    str((out / "pipeline_run_summary.json").resolve()),
                    str((out / "pipeline.log").resolve()),
                ],
            }
            block = mod._final_failure_block(summary, out, mod.EXIT_TESTS_FAILED)
            for item in summary["files_to_provide"]:
                self.assertIn(item, block)

    def test_inference_manifest_and_diagnosis_not_called_after_test_failure(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            args = SimpleNamespace(
                config="cfg",
                run_dir="training/analysis/run",
                output_dir=str(repo / "training" / "analysis" / "out"),
                expected_manifest_identity_sha="sha",
                expected_center_checkpoint_sha="center",
                expected_semantic_checkpoint_sha="semantic",
                device="cpu",
                clean_output=False,
                skip_tests=False,
                include_visual_review=False,
            )
            with mock.patch.object(mod, "_parse_args", return_value=args), \
                mock.patch.object(mod, "_safe_output_dir"), \
                mock.patch.object(mod, "_repo_preflight", return_value={"git": {"commit": "abc", "branch": "main", "tracked_tree_clean": True}, "environment": {"hostname": "host", "device": "cpu", "python": "3.12", "torch": "2.x", "cuda_available": False}}), \
                mock.patch.object(mod, "_run_tests", side_effect=mod.PipelineFailure(stage="tests", reason="boom", exit_code=mod.EXIT_TESTS_FAILED)), \
                mock.patch.object(mod, "_run_manifest_stage") as manifest_mock, \
                mock.patch.object(mod, "_run_diagnosis_stage") as diagnosis_mock:
                with self.assertRaises(SystemExit):
                    mod.main()
                manifest_mock.assert_not_called()
                diagnosis_mock.assert_not_called()

    def test_successful_diagnosis_data_persisted_before_artifact_integrity(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            out = repo / "training" / "analysis" / "out"
            args = SimpleNamespace(
                config="cfg",
                run_dir="training/analysis/run",
                output_dir=str(out),
                expected_manifest_identity_sha="sha",
                expected_center_checkpoint_sha="center",
                expected_semantic_checkpoint_sha="semantic",
                device="cpu",
                clean_output=False,
                skip_tests=True,
                include_visual_review=False,
            )
            with mock.patch.object(mod, "_parse_args", return_value=args), \
                mock.patch.object(mod, "_safe_output_dir"), \
                mock.patch.object(mod, "_repo_preflight", return_value={"git": {"commit": "abc", "branch": "main", "tracked_tree_clean": True}, "environment": {"hostname": "host", "device": "cpu", "python": "3.12", "torch": "2.x", "cuda_available": False}}), \
                mock.patch.object(mod, "_checkpoint_identity", return_value=mod.StageResult("checkpoint", 0, 0.0, {"center_checkpoint_sha": "csha", "semantic_checkpoint_sha": "ssha"})), \
                mock.patch.object(mod, "_run_manifest_stage", return_value=mod.StageResult("manifest", 0, 0.0, {"samples": 106, "unique_samples": 106, "split_counts": {"test": 53, "val": 53}, "gt_count_distribution": {"1": 15, "2": 37, "3": 54}, "manifest_identity_sha": "sha"})), \
                mock.patch.object(mod, "_run_diagnosis_stage", return_value=mod.StageResult("diagnosis", 0, 0.0, {"bottleneck_status": "mixed_center_and_semantic_failure"})), \
                mock.patch.object(mod, "_read_json", side_effect=[{"checkpoint_identity_status": "exact_match", "semantic_checkpoint_identity_status": "exact_match", "manifest_identity_status": "exact_match", "diagnosis_execution_status": "completed", "overall_authoritative_status": "exact_match"}]), \
                mock.patch.object(mod, "_inspect_artifacts", return_value=({"status": "failed", "required": [], "optional": [], "generated_only_by_another_pipeline": [], "missing_required": ["missing.json"], "malformed": []}, [], [], [])):
                with self.assertRaises(SystemExit) as cm:
                    mod.main()
                self.assertEqual(cm.exception.code, mod.EXIT_ARTIFACT_INTEGRITY_FAILED)
            summary = json.loads((out / "pipeline_run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["authoritative_status"]["overall"], "exact_match")
            self.assertEqual(summary["bottleneck_status"], "mixed_center_and_semantic_failure")
            self.assertEqual(summary["identity"]["manifest_identity_sha"], "sha")

    def test_missing_required_artifact_returns_exit_code_60_not_99(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            out = repo / "training" / "analysis" / "out"
            args = SimpleNamespace(
                config="cfg",
                run_dir="training/analysis/run",
                output_dir=str(out),
                expected_manifest_identity_sha="sha",
                expected_center_checkpoint_sha="center",
                expected_semantic_checkpoint_sha="semantic",
                device="cpu",
                clean_output=False,
                skip_tests=True,
                include_visual_review=False,
            )
            with mock.patch.object(mod, "_parse_args", return_value=args), \
                mock.patch.object(mod, "_safe_output_dir"), \
                mock.patch.object(mod, "_repo_preflight", return_value={"git": {"commit": "abc", "branch": "main", "tracked_tree_clean": True}, "environment": {"hostname": "host", "device": "cpu", "python": "3.12", "torch": "2.x", "cuda_available": False}}), \
                mock.patch.object(mod, "_checkpoint_identity", return_value=mod.StageResult("checkpoint", 0, 0.0, {"center_checkpoint_sha": "csha", "semantic_checkpoint_sha": "ssha"})), \
                mock.patch.object(mod, "_run_manifest_stage", return_value=mod.StageResult("manifest", 0, 0.0, {"samples": 106, "unique_samples": 106, "split_counts": {"test": 53, "val": 53}, "gt_count_distribution": {"1": 15, "2": 37, "3": 54}, "manifest_identity_sha": "sha"})), \
                mock.patch.object(mod, "_run_diagnosis_stage", return_value=mod.StageResult("diagnosis", 0, 0.0, {"bottleneck_status": "mixed"})), \
                mock.patch.object(mod, "_read_json", side_effect=[{"checkpoint_identity_status": "exact_match", "semantic_checkpoint_identity_status": "exact_match", "manifest_identity_status": "exact_match", "diagnosis_execution_status": "completed", "overall_authoritative_status": "exact_match"}]), \
                mock.patch.object(mod, "_inspect_artifacts", return_value=({"status": "failed", "required": [], "optional": [], "generated_only_by_another_pipeline": [], "missing_required": ["x"], "malformed": []}, [], [], [])):
                with self.assertRaises(SystemExit) as cm:
                    mod.main()
                self.assertEqual(cm.exception.code, mod.EXIT_ARTIFACT_INTEGRITY_FAILED)

    def test_missing_optional_artifact_does_not_fail(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._populate_required_artifacts(mod, out)
            report, _index, _files, _large = mod._inspect_artifacts(out, include_visual_review=False)
            self.assertEqual(report["status"], "passed")
            optional_statuses = {row["relative_path"]: row["status"] for row in report["optional"]}
            self.assertEqual(optional_statuses["end_to_end_vs_center_oracle.csv"], "absent_optional")

    def test_artifact_integrity_failed_stage_is_recorded(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            out = repo / "training" / "analysis" / "out"
            args = SimpleNamespace(
                config="cfg",
                run_dir="training/analysis/run",
                output_dir=str(out),
                expected_manifest_identity_sha="sha",
                expected_center_checkpoint_sha="center",
                expected_semantic_checkpoint_sha="semantic",
                device="cpu",
                clean_output=False,
                skip_tests=True,
                include_visual_review=False,
            )
            with mock.patch.object(mod, "_parse_args", return_value=args), \
                mock.patch.object(mod, "_safe_output_dir"), \
                mock.patch.object(mod, "_repo_preflight", return_value={"git": {"commit": "abc", "branch": "main", "tracked_tree_clean": True}, "environment": {"hostname": "host", "device": "cpu", "python": "3.12", "torch": "2.x", "cuda_available": False}}), \
                mock.patch.object(mod, "_checkpoint_identity", return_value=mod.StageResult("checkpoint", 0, 0.0, {"center_checkpoint_sha": "csha", "semantic_checkpoint_sha": "ssha"})), \
                mock.patch.object(mod, "_run_manifest_stage", return_value=mod.StageResult("manifest", 0, 0.0, {"samples": 106, "unique_samples": 106, "split_counts": {"test": 53, "val": 53}, "gt_count_distribution": {"1": 15, "2": 37, "3": 54}, "manifest_identity_sha": "sha"})), \
                mock.patch.object(mod, "_run_diagnosis_stage", return_value=mod.StageResult("diagnosis", 0, 0.0, {"bottleneck_status": "mixed"})), \
                mock.patch.object(mod, "_read_json", side_effect=[{"checkpoint_identity_status": "exact_match", "semantic_checkpoint_identity_status": "exact_match", "manifest_identity_status": "exact_match", "diagnosis_execution_status": "completed", "overall_authoritative_status": "exact_match"}]), \
                mock.patch.object(mod, "_inspect_artifacts", return_value=({"status": "failed", "required": [], "optional": [], "generated_only_by_another_pipeline": [], "missing_required": ["x"], "malformed": []}, [], [], [])):
                with self.assertRaises(SystemExit):
                    mod.main()
            summary = json.loads((out / "pipeline_run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["stages"][-1]["stage"], "artifact_integrity")
            self.assertEqual(summary["stages"][-1]["status"], "failed")

    def test_files_to_provide_contains_only_existing_files(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "pipeline.log").write_text("x", encoding="utf-8")
            (out / "pipeline_run_summary.json").write_text("{}", encoding="utf-8")
            files = mod._failure_files_to_provide(out)
            self.assertEqual(set(files), {str((out / "pipeline.log").resolve()), str((out / "pipeline_run_summary.json").resolve())})

    def test_actual_diagnosis_artifact_contract_matches_required_list(self):
        mod = self._mod()
        self.assertNotIn("corrected_promotion_decision.json", mod.REQUIRED_ARTIFACTS)
        self.assertIn("bottleneck_decision.json", mod.REQUIRED_ARTIFACTS)
        self.assertIn("oracle_scope_summary.json", mod.REQUIRED_ARTIFACTS)

    def test_runner_does_not_create_fake_promotion_decision_artifact(self):
        mod = self._mod()
        self.assertNotIn("corrected_promotion_decision.json", mod.REQUIRED_ARTIFACTS)
        self.assertIn("corrected_promotion_decision.json", mod.GENERATED_BY_OTHER_PIPELINES)

    def test_successful_integrity_proceeds_to_bundle_creation(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            out = repo / "training" / "analysis" / "out"
            args = SimpleNamespace(
                config="cfg",
                run_dir="training/analysis/run",
                output_dir=str(out),
                expected_manifest_identity_sha="sha",
                expected_center_checkpoint_sha="center",
                expected_semantic_checkpoint_sha="semantic",
                device="cpu",
                clean_output=False,
                skip_tests=True,
                include_visual_review=False,
            )
            with mock.patch.object(mod, "_parse_args", return_value=args), \
                mock.patch.object(mod, "_safe_output_dir"), \
                mock.patch.object(mod, "_repo_preflight", return_value={"git": {"commit": "abc", "branch": "main", "tracked_tree_clean": True}, "environment": {"hostname": "host", "device": "cpu", "python": "3.12", "torch": "2.x", "cuda_available": False}}), \
                mock.patch.object(mod, "_checkpoint_identity", return_value=mod.StageResult("checkpoint", 0, 0.0, {"center_checkpoint_sha": "csha", "semantic_checkpoint_sha": "ssha"})), \
                mock.patch.object(mod, "_run_manifest_stage", return_value=mod.StageResult("manifest", 0, 0.0, {"samples": 106, "unique_samples": 106, "split_counts": {"test": 53, "val": 53}, "gt_count_distribution": {"1": 15, "2": 37, "3": 54}, "manifest_identity_sha": "sha"})), \
                mock.patch.object(mod, "_run_diagnosis_stage", return_value=mod.StageResult("diagnosis", 0, 0.0, {"bottleneck_status": "mixed"})), \
                mock.patch.object(mod, "_read_json", side_effect=[{"checkpoint_identity_status": "exact_match", "semantic_checkpoint_identity_status": "exact_match", "manifest_identity_status": "exact_match", "diagnosis_execution_status": "completed", "overall_authoritative_status": "exact_match"}]), \
                mock.patch.object(mod, "_inspect_artifacts", return_value=({"status": "passed", "required": [], "optional": [], "generated_only_by_another_pipeline": [], "missing_required": [], "malformed": []}, [], [], [])), \
                mock.patch.object(mod, "_bundle_review", return_value=(out / "bundle.tar.gz", out / "bundle.tar.gz.sha256")) as bundle_mock, \
                mock.patch.object(mod, "_write_files_to_provide"), \
                mock.patch.object(mod, "_atomic_write_json"):
                with self.assertRaises(SystemExit) as cm:
                    mod.main()
                self.assertEqual(cm.exception.code, 0)
                bundle_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
