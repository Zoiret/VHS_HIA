from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
