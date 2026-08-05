from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit_center_semantic_preprocessing_parity import _replay_parity_rows


DEFAULT_OUTPUT_DIR = "training/analysis/centerhead_spatial_x2_2_cpu_cuda_replay_parity"
EXIT_SUCCESS = 0
EXIT_IDENTITY_MISMATCH = 30
EXIT_DISCRETE_MISMATCH = 31
EXIT_BUNDLE_FAILURE = 70
EXIT_UNEXPECTED_EXCEPTION = 99


class ParityFailure(RuntimeError):
    def __init__(self, reason: str, exit_code: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.exit_code = int(exit_code)


class TeeLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")

    def write(self, text: str) -> None:
        self._fh.write(text)
        self._fh.flush()
        sys.stdout.write(text)
        sys.stdout.flush()

    def log(self, message: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self.write(f"[{ts}] {message}\n")

    def close(self) -> None:
        self._fh.close()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8", newline="\n") as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ParityFailure("Parity CSV has no rows", EXIT_DISCRETE_MISMATCH)
    fieldnames = list(rows[0].keys())
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8", newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(dir_path: Path) -> dict[str, Any]:
    chk = _read_json(dir_path / "checkpoint_identity.json")
    manifest = _read_json(dir_path / "holdout_manifest_metadata.json")
    return {
        "canonical_manifest_sha": chk.get("manifest_identity_sha256") or manifest.get("canonical_identity_sha256"),
        "checkpoint_sha": chk.get("checkpoint_sha256"),
        "semantic_checkpoint_sha": chk.get("semantic_checkpoint_sha256"),
        "device": chk.get("device"),
        "sample_count": int(manifest.get("manifest_row_count", manifest.get("eligible_count", 0))),
        "unique_sample_count": int(manifest.get("unique_sample_count", 0)),
    }


def _sample_ids(dir_path: Path) -> set[str]:
    rows = [json.loads(line) for line in (dir_path / "holdout_manifest_identity.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["sample"]) for row in rows}


def _validate_identity(local_dir: Path, server_dir: Path) -> dict[str, Any]:
    local = _identity(local_dir)
    server = _identity(server_dir)
    reasons = []
    if str(local["canonical_manifest_sha"]) != str(server["canonical_manifest_sha"]):
        reasons.append("canonical manifest SHA differs")
    if str(local["checkpoint_sha"]) != str(server["checkpoint_sha"]):
        reasons.append("center checkpoint SHA differs")
    if str(local["semantic_checkpoint_sha"]) != str(server["semantic_checkpoint_sha"]):
        reasons.append("semantic checkpoint SHA differs")
    if int(local["sample_count"]) != int(server["sample_count"]):
        reasons.append("sample count differs")
    if int(local["unique_sample_count"]) != int(server["unique_sample_count"]):
        reasons.append("unique sample count differs")
    if str(local["device"]).lower() != "cpu":
        reasons.append("local diagnosis device is not cpu")
    if str(server["device"]).lower() != "cuda":
        reasons.append("server diagnosis device is not cuda")
    local_samples = _sample_ids(local_dir)
    server_samples = _sample_ids(server_dir)
    if local_samples != server_samples:
        reasons.append("sample IDs differ")
    if reasons:
        raise ParityFailure("; ".join(reasons), EXIT_IDENTITY_MISMATCH)
    return {
        "canonical_manifest_match": True,
        "checkpoint_match": True,
        "semantic_checkpoint_match": True,
        "sample_ids_match": True,
        "samples": int(local["sample_count"]),
        "canonical_manifest_sha": str(local["canonical_manifest_sha"]),
    }


def _classify(summary: dict[str, Any]) -> str:
    if not bool(summary["exact_discrete_matches"]):
        return "device_sensitive_discrete_output"
    if float(summary["maximum_absolute_delta"]) > 0.0:
        return "floating_only_device_drift"
    return "exact_or_tolerance_match"


def _artifact_index(output_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    files = [
        ("replay_parity.json", "Parity aggregate summary"),
        ("per_sample_replay_parity.csv", "Per-sample parity comparison"),
        ("parity_run_summary.json", "Parity runner summary"),
        ("parity.log", "Parity runner log"),
    ]
    index = []
    provide = []
    for rel, desc in files:
        path = (output_dir / rel).resolve()
        if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
            raise ParityFailure(f"Missing or empty required artifact: {rel}", EXIT_BUNDLE_FAILURE)
        index.append(
            {
                "relative_path": rel,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
                "required_for_review": True,
                "description": desc,
            }
        )
        provide.append(str(path))
    return index, provide


def _bundle(output_dir: Path) -> tuple[Path, Path]:
    bundle = (output_dir / "cpu_cuda_parity_review_bundle.tar.gz").resolve()
    checksum = (output_dir / "cpu_cuda_parity_review_bundle.tar.gz.sha256").resolve()
    include_paths = [
        output_dir / "replay_parity.json",
        output_dir / "per_sample_replay_parity.csv",
        output_dir / "parity_run_summary.json",
        output_dir / "parity.log",
        output_dir / "parity_artifact_index.json",
        output_dir / "files_to_provide.txt",
    ]
    with tarfile.open(bundle, "w:gz") as tar:
        for path in sorted(include_paths, key=lambda p: p.name):
            info = tar.gettarinfo(str(path), arcname=path.name)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            with path.open("rb") as fh:
                tar.addfile(info, fh)
    checksum.write_text(f"{_sha256_file(bundle)}  {bundle.name}\n", encoding="utf-8", newline="\n")
    return bundle, checksum


def _write_files_to_provide(path: Path, files: list[str]) -> None:
    lines = ["FILES TO PROVIDE:"]
    for idx, item in enumerate(files, start=1):
        lines.append(f"{idx}. {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _final_block(summary: dict[str, Any], bundle: Path, checksum: Path) -> str:
    return (
        "==================================================\n"
        "CPU/CUDA REPLAY PARITY: SUCCESS\n"
        "==================================================\n"
        f"samples_compared: {summary['samples_compared']}\n"
        f"canonical_manifest_match: {str(summary['canonical_manifest_match']).lower()}\n"
        f"checkpoint_match: {str(summary['checkpoint_match']).lower()}\n"
        f"discrete_output_match: {str(summary['discrete_output_match']).lower()}\n"
        f"classification: {summary['classification']}\n\n"
        "FILES TO PROVIDE:\n"
        f"1. {bundle}\n"
        f"2. {checksum}\n"
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-cpu-dir", type=str, required=True)
    ap.add_argument("--server-cuda-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--clean-output", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    local_dir = (repo_root / args.local_cpu_dir).resolve() if not Path(args.local_cpu_dir).is_absolute() else Path(args.local_cpu_dir).resolve()
    server_dir = (repo_root / args.server_cuda_dir).resolve() if not Path(args.server_cuda_dir).is_absolute() else Path(args.server_cuda_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = TeeLogger((output_dir / "parity.log").resolve())
    summary: dict[str, Any] = {
        "status": "failed",
        "failure_reason": None,
        "samples_compared": 0,
        "canonical_manifest_match": False,
        "checkpoint_match": False,
        "discrete_output_match": False,
        "classification": None,
        "files_to_provide": [],
    }
    exit_code = EXIT_SUCCESS
    try:
        logger.log("Validating local/server diagnosis identity")
        ident = _validate_identity(local_dir, server_dir)
        logger.log("Running per-sample replay parity comparison")
        rows, replay_summary = _replay_parity_rows(local_dir, server_dir)
        classification = _classify(replay_summary)
        _write_csv((output_dir / "per_sample_replay_parity.csv").resolve(), rows)
        replay_payload = dict(replay_summary)
        replay_payload["classification"] = classification
        _atomic_write_json((output_dir / "replay_parity.json").resolve(), replay_payload)
        summary.update(
            {
                "status": "success" if classification != "device_sensitive_discrete_output" else "failed",
                "failure_reason": None if classification != "device_sensitive_discrete_output" else "Discrete CPU/CUDA output mismatch",
                "samples_compared": ident["samples"],
                "canonical_manifest_match": True,
                "checkpoint_match": True,
                "discrete_output_match": bool(replay_summary["exact_discrete_matches"]),
                "classification": classification,
                "replay_summary": replay_payload,
            }
        )
        _atomic_write_json((output_dir / "parity_run_summary.json").resolve(), summary)
        index, provide = _artifact_index(output_dir)
        _atomic_write_json((output_dir / "parity_artifact_index.json").resolve(), {"artifacts": index})
        _write_files_to_provide((output_dir / "files_to_provide.txt").resolve(), provide)
        index, provide = _artifact_index(output_dir)
        _atomic_write_json((output_dir / "parity_artifact_index.json").resolve(), {"artifacts": index})
        bundle, checksum = _bundle(output_dir)
        provide.extend([str(bundle), str(checksum)])
        _write_files_to_provide((output_dir / "files_to_provide.txt").resolve(), provide)
        summary["files_to_provide"] = provide
        _atomic_write_json((output_dir / "parity_run_summary.json").resolve(), summary)
        if classification == "device_sensitive_discrete_output":
            raise ParityFailure("Discrete CPU/CUDA mismatch detected", EXIT_DISCRETE_MISMATCH)
        logger.write(_final_block(summary, bundle, checksum))
        exit_code = EXIT_SUCCESS
    except ParityFailure as exc:
        exit_code = exc.exit_code
        summary["status"] = "failed"
        summary["failure_reason"] = exc.reason
        _atomic_write_json((output_dir / "parity_run_summary.json").resolve(), summary)
        logger.write(
            "==================================================\n"
            "CPU/CUDA REPLAY PARITY: FAILED\n"
            "==================================================\n"
            f"reason: {exc.reason}\n"
            f"exit_code: {exc.exit_code}\n"
        )
    except Exception as exc:  # noqa: BLE001
        exit_code = EXIT_UNEXPECTED_EXCEPTION
        summary["status"] = "failed"
        summary["failure_reason"] = str(exc)
        _atomic_write_json((output_dir / "parity_run_summary.json").resolve(), summary)
        logger.write(
            "==================================================\n"
            "CPU/CUDA REPLAY PARITY: FAILED\n"
            "==================================================\n"
            f"reason: {exc}\n"
            f"exit_code: {EXIT_UNEXPECTED_EXCEPTION}\n"
        )
    finally:
        logger.close()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
