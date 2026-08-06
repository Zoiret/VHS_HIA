from __future__ import annotations

import csv
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


_THIS_DIR = Path(__file__).resolve().parent


class TestCpuCudaReplayParityRunner(unittest.TestCase):
    @staticmethod
    def _mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import run_cpu_cuda_replay_parity as mod

        return mod

    def _write_diag_dir(self, root: Path, *, device: str, manifest_sha: str = "sha", sample_suffix: str = "") -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "checkpoint_identity.json").write_text(
            json.dumps(
                {
                    "manifest_identity_sha256": manifest_sha,
                    "checkpoint_sha256": "center",
                    "semantic_checkpoint_sha256": "semantic",
                    "device": device,
                }
            ),
            encoding="utf-8",
        )
        (root / "holdout_manifest_metadata.json").write_text(
            json.dumps({"manifest_row_count": 106, "unique_sample_count": 106}),
            encoding="utf-8",
        )
        (root / "holdout_manifest_identity.jsonl").write_text(
            "".join(
                json.dumps({"sample": f"s{i}{sample_suffix}"}) + "\n"
                for i in range(106)
            ),
            encoding="utf-8",
        )
        return root

    @staticmethod
    def _heterogeneous_rows() -> list[dict]:
        return [
            {
                "kind": "center",
                "sample": "s02",
                "threshold": "0.03",
                "marker_contract_pass_equal": True,
                "predicted_count_equal": True,
                "center_f1_abs_delta": 0.0,
            },
            {
                "kind": "scope",
                "sample": "s01",
                "scope": "center_oracle",
                "policy": "P1_DROP_UNMARKED",
                "marker_count_equal": True,
                "output_count_equal": True,
                "invariant_equal": True,
                "matched_iou_abs_delta": 0.001,
                "dice_abs_delta": 0.002,
            },
        ]

    def test_parity_runner_rejects_different_canonical_sha(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            local = self._write_diag_dir(base / "local", device="cpu", manifest_sha="a")
            server = self._write_diag_dir(base / "server", device="cuda", manifest_sha="b")
            with self.assertRaises(mod.ParityFailure):
                mod._validate_identity(local, server)

    def test_parity_runner_rejects_different_sample_ids(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            local = self._write_diag_dir(base / "local", device="cpu", manifest_sha="a")
            server = self._write_diag_dir(base / "server", device="cuda", manifest_sha="a", sample_suffix="_x")
            with self.assertRaises(mod.ParityFailure):
                mod._validate_identity(local, server)

    def test_parity_runner_detects_discrete_mismatch(self):
        mod = self._mod()
        self.assertEqual(mod._classify({"exact_discrete_matches": False, "maximum_absolute_delta": 0.0}), "device_sensitive_discrete_output")

    def test_parity_bundle_generated_on_allowed_success_classification(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for name in ("replay_parity.json", "per_sample_replay_parity.csv", "parity_run_summary.json", "parity.log", "parity_artifact_index.json", "files_to_provide.txt"):
                (out / name).write_text("x", encoding="utf-8")
            bundle, _ = mod._bundle(out)
            with tarfile.open(bundle, "r:gz") as tar:
                names = set(tar.getnames())
            self.assertIn("replay_parity.json", names)
            self.assertIn("parity_run_summary.json", names)

    def test_main_exits_nonzero_on_discrete_mismatch(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            local = self._write_diag_dir(base / "local", device="cpu", manifest_sha="a")
            server = self._write_diag_dir(base / "server", device="cuda", manifest_sha="a")
            out = base / "out"
            args = SimpleNamespace(local_cpu_dir=str(local), server_cuda_dir=str(server), output_dir=str(out), clean_output=False)
            with mock.patch.object(mod, "_parse_args", return_value=args), \
                mock.patch.object(mod, "_replay_parity_rows", return_value=([{"a": 1}], {"exact_discrete_matches": False, "maximum_absolute_delta": 0.0, "median_absolute_delta": 0.0, "samples_crossing_threshold_due_to_numerical_differences": 1, "samples_with_different_marker_coordinates": 1, "samples_with_different_output_counts": 1})):
                with self.assertRaises(SystemExit) as cm:
                    mod.main()
                self.assertEqual(cm.exception.code, mod.EXIT_DISCRETE_MISMATCH)

    def test_identity_validation_accepts_matching_inputs(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            local = self._write_diag_dir(base / "local", device="cpu", manifest_sha="same")
            server = self._write_diag_dir(base / "server", device="cuda", manifest_sha="same")
            ident = mod._validate_identity(local, server)
            self.assertTrue(ident["canonical_manifest_match"])
            self.assertEqual(ident["samples"], 106)

    def test_heterogeneous_parity_rows_serialize_without_error(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parity.csv"
            fieldnames, row_count = mod._write_parity_csv(path, self._heterogeneous_rows())
            self.assertTrue(path.exists())
            self.assertGreater(len(fieldnames), 0)
            self.assertEqual(row_count, 2)

    def test_all_row_keys_are_represented_in_fieldnames(self):
        mod = self._mod()
        rows = self._heterogeneous_rows()
        fieldnames = mod._fieldnames_for_parity_rows(rows)
        self.assertEqual(mod._missing_schema_keys(mod._prepare_parity_rows(rows), fieldnames), [])

    def test_record_type_is_populated(self):
        mod = self._mod()
        prepared = mod._prepare_parity_rows(self._heterogeneous_rows())
        self.assertEqual({row["record_type"] for row in prepared}, {"center_diagnostic", "oracle_policy"})

    def test_deterministic_field_order(self):
        mod = self._mod()
        rows = list(reversed(self._heterogeneous_rows()))
        fieldnames = mod._fieldnames_for_parity_rows(rows)
        self.assertEqual(
            fieldnames,
            [
                "sample",
                "threshold",
                "record_type",
                "scope",
                "policy",
                "marker_contract_pass_equal",
                "predicted_count_equal",
                "marker_count_equal",
                "output_count_equal",
                "invariant_equal",
                "center_f1_abs_delta",
                "matched_iou_abs_delta",
                "dice_abs_delta",
                "kind",
            ],
        )

    def test_deterministic_row_order(self):
        mod = self._mod()
        prepared = mod._prepare_parity_rows(self._heterogeneous_rows())
        self.assertEqual([row["sample"] for row in prepared], ["s01", "s02"])

    def test_missing_type_specific_values_become_empty_cells(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parity.csv"
            mod._write_parity_csv(path, self._heterogeneous_rows())
            rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
            center = next(row for row in rows if row["record_type"] == "center_diagnostic")
            self.assertEqual(center["scope"], "")
            self.assertEqual(center["policy"], "")
            self.assertEqual(center["matched_iou_abs_delta"], "")
            self.assertEqual(center["dice_abs_delta"], "")

    def test_row_count_preserved(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parity.csv"
            _fieldnames, row_count = mod._write_parity_csv(path, self._heterogeneous_rows())
            rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
            self.assertEqual(len(rows), row_count)

    def test_oracle_policy_fields_preserved(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parity.csv"
            mod._write_parity_csv(path, self._heterogeneous_rows())
            rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
            oracle = next(row for row in rows if row["record_type"] == "oracle_policy")
            self.assertEqual(oracle["scope"], "center_oracle")
            self.assertEqual(oracle["policy"], "P1_DROP_UNMARKED")
            self.assertEqual(oracle["marker_count_equal"], "True")
            self.assertEqual(oracle["output_count_equal"], "True")
            self.assertEqual(oracle["invariant_equal"], "True")
            self.assertEqual(oracle["matched_iou_abs_delta"], "0.001")
            self.assertEqual(oracle["dice_abs_delta"], "0.002")

    def test_unexpected_new_key_triggers_controlled_schema_error_not_generic_99(self):
        mod = self._mod()
        rows = self._heterogeneous_rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parity.csv"
            with self.assertRaises(mod.ParityFailure) as ctx:
                mod._write_parity_csv(path, rows, fieldnames=["sample", "record_type"])
            self.assertEqual(ctx.exception.exit_code, mod.EXIT_OUTPUT_SERIALIZATION_FAILED)

    def test_successful_serialization_proceeds_to_bundle_creation(self):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            local = self._write_diag_dir(base / "local", device="cpu", manifest_sha="a")
            server = self._write_diag_dir(base / "server", device="cuda", manifest_sha="a")
            out = base / "out"
            args = SimpleNamespace(local_cpu_dir=str(local), server_cuda_dir=str(server), output_dir=str(out), clean_output=False)
            fake_rows = self._heterogeneous_rows()
            fake_summary = {
                "exact_discrete_matches": True,
                "maximum_absolute_delta": 0.001,
                "median_absolute_delta": 0.001,
                "samples_crossing_threshold_due_to_numerical_differences": 0,
                "samples_with_different_marker_coordinates": 0,
                "samples_with_different_output_counts": 0,
            }
            with mock.patch.object(mod, "_parse_args", return_value=args), \
                mock.patch.object(mod, "_replay_parity_rows", return_value=(fake_rows, fake_summary)):
                with self.assertRaises(SystemExit) as cm:
                    mod.main()
                self.assertEqual(cm.exception.code, mod.EXIT_SUCCESS)
            self.assertTrue((out / "cpu_cuda_parity_review_bundle.tar.gz").exists())
            self.assertTrue((out / "per_sample_replay_parity.csv").exists())


if __name__ == "__main__":
    unittest.main()
