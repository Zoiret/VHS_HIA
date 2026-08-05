from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "training/configs/unetpp_effb3_centerhead_spatial_x2_2_adapter_legacy_fp32_micro.yaml"
DEFAULT_RUN_DIR = "training/analysis/centerhead_spatial_x2_2_adapter_legacy_fp32_micro_overfit"
DEFAULT_OUTPUT_DIR = "training/analysis/centerhead_spatial_x2_2_center_generalization_holdout_diagnosis_authoritative"
DEFAULT_CENTER_SHA = "d582743fc28d39b10fb412443638404fe86bb2c2d1b7125f8287feae2766540b"
DEFAULT_SEMANTIC_SHA = "ea19846a35da02cc0cb6041d814f206719eb1926f3b02cfd6fbf448d39834c48"
SEMANTIC_CHECKPOINT_REL = "training/runs/unetpp_effb3_a100_multiclass_curated_finetune_stage2_lr1e5_100ep/best_mean_fg.pth"
VISUAL_REVIEW_SIZE_LIMIT_BYTES = 50 * 1024 * 1024
MICROSET_IDS = {
    "m01_p02_s00",
    "m01_p02_s04",
    "m01_p01_s00",
    "m01_p01_s01",
    "m01_p01_s02",
    "m01_p01_s03",
}
REQUIRED_TEST_MODULES = (
    "training.test_reconstruction_policy_ablation",
    "training.test_reconstruction_policy_holdout",
    "training.test_center_generalization_holdout",
    "training.test_holdout_manifest_identity",
    "training.test_center_semantic_preprocessing_parity",
    "training.test_authoritative_holdout_pipeline",
    "training.test_cpu_cuda_replay_parity_runner",
)
REQUIRED_ARTIFACTS: dict[str, dict[str, Any]] = {
    "pipeline_run_summary.json": {"type": "json", "required_for_review": True, "description": "Top-level orchestrator summary"},
    "pipeline.log": {"type": "text", "required_for_review": True, "description": "Full orchestrator and subprocess log"},
    "holdout_manifest.txt": {"type": "text", "required_for_review": True, "description": "Execution manifest"},
    "holdout_manifest_identity.jsonl": {"type": "jsonl", "required_for_review": True, "description": "Canonical identity manifest"},
    "holdout_manifest_metadata.json": {"type": "json", "required_for_review": True, "description": "Manifest metadata and counts"},
    "checkpoint_identity.json": {"type": "json", "required_for_review": True, "description": "Checkpoint and manifest identity statuses"},
    "corrected_promotion_decision.json": {"type": "json", "required_for_review": True, "description": "Corrected promotion gate result"},
    "bottleneck_decision.json": {"type": "json", "required_for_review": True, "description": "Bottleneck classification"},
    "oracle_scope_summary.json": {"type": "json", "required_for_review": True, "description": "End-to-end and oracle scope summary"},
    "full_oracle_invariants.json": {"type": "json", "required_for_review": True, "description": "Full-oracle invariant summary"},
    "center_threshold_summary.csv": {"type": "csv", "required_for_review": True, "description": "Center threshold sweep summary"},
    "per_sample_center_diagnostics.csv": {"type": "csv", "required_for_review": True, "description": "Per-sample center diagnostics"},
    "p0_gt_count_confusion.csv": {"type": "csv", "required_for_review": True, "description": "P0 count confusion table"},
    "semantic_failure_summary.json": {"type": "json", "required_for_review": True, "description": "Semantic failure aggregate summary"},
    "per_sample_semantic_diagnostics.csv": {"type": "csv", "required_for_review": True, "description": "Per-sample semantic diagnostics"},
}
OPTIONAL_ARTIFACTS: dict[str, dict[str, Any]] = {
    "end_to_end_vs_center_oracle.csv": {"type": "csv", "required_for_review": True, "description": "End-to-end vs center-oracle deltas"},
    "per_sample_oracle_policy_metrics.csv": {"type": "csv", "required_for_review": True, "description": "Per-sample oracle policy metrics"},
    "worst_center_failures.csv": {"type": "csv", "required_for_review": True, "description": "Worst center failures"},
    "visual_review": {"type": "dir", "required_for_review": False, "description": "Diagnostic visual panels"},
}
EXIT_SUCCESS = 0
EXIT_REPOSITORY_PREFLIGHT_FAILED = 10
EXIT_TESTS_FAILED = 20
EXIT_CHECKPOINT_IDENTITY_FAILED = 30
EXIT_MANIFEST_GENERATION_FAILED = 40
EXIT_MANIFEST_IDENTITY_MISMATCH = 41
EXIT_DIAGNOSIS_SUBPROCESS_FAILED = 50
EXIT_AUTHORITATIVE_STATUS_MISMATCH = 51
EXIT_ARTIFACT_INTEGRITY_FAILED = 60
EXIT_BUNDLE_CREATION_FAILED = 70
EXIT_UNEXPECTED_EXCEPTION = 99


@dataclass
class StageResult:
    name: str
    exit_code: int
    duration_sec: float
    details: dict[str, Any]


class PipelineFailure(RuntimeError):
    def __init__(self, *, stage: str, reason: str, exit_code: int) -> None:
        super().__init__(reason)
        self.stage = stage
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_command(command: list[str], *, cwd: Path, logger: TeeLogger, env: dict[str, str] | None = None) -> tuple[int, float]:
    logger.log(f"COMMAND: {' '.join(command)}")
    start = time.perf_counter()
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        logger.write(line)
    proc.stdout.close()
    rc = proc.wait()
    duration = time.perf_counter() - start
    logger.log(f"EXIT CODE: {rc} duration_sec={duration:.3f}")
    return int(rc), float(duration)


def _safe_output_dir(repo_root: Path, output_dir: Path) -> None:
    resolved_repo = repo_root.resolve()
    resolved_out = output_dir.resolve()
    analysis_root = (resolved_repo / "training" / "analysis").resolve()
    forbidden = {
        resolved_repo,
        (resolved_repo / "training").resolve(),
        analysis_root,
    }
    if resolved_out in forbidden:
        raise PipelineFailure(stage="output_preparation", reason=f"Unsafe output path: {resolved_out}", exit_code=EXIT_REPOSITORY_PREFLIGHT_FAILED)
    if analysis_root not in resolved_out.parents:
        raise PipelineFailure(stage="output_preparation", reason=f"Output path must be inside training/analysis: {resolved_out}", exit_code=EXIT_REPOSITORY_PREFLIGHT_FAILED)


def _git_output(repo_root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=str(repo_root), capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _tracked_training_changes(status_lines: list[str]) -> list[str]:
    bad = []
    for line in status_lines:
        if not line.strip():
            continue
        status = line[:2]
        path = line[3:]
        if status == "??":
            continue
        if path.startswith("training/") and not path.startswith("training/analysis/"):
            bad.append(path)
    return sorted(bad)


def _repo_preflight(repo_root: Path, requested_device: str) -> dict[str, Any]:
    commit = _git_output(repo_root, "rev-parse", "HEAD")
    branch = _git_output(repo_root, "branch", "--show-current")
    status_lines = _git_output(repo_root, "status", "--porcelain").splitlines()
    tracked_dirty = _tracked_training_changes(status_lines)
    if tracked_dirty:
        raise PipelineFailure(
            stage="repository_preflight",
            reason=f"Tracked pipeline files modified: {tracked_dirty}",
            exit_code=EXIT_REPOSITORY_PREFLIGHT_FAILED,
        )
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise PipelineFailure(stage="repository_preflight", reason=f"PyTorch import failed: {exc}", exit_code=EXIT_REPOSITORY_PREFLIGHT_FAILED)
    cuda_available = bool(torch.cuda.is_available())
    if requested_device == "cuda" and not cuda_available:
        raise PipelineFailure(stage="repository_preflight", reason="CUDA requested but not available", exit_code=EXIT_REPOSITORY_PREFLIGHT_FAILED)
    return {
        "git": {
            "commit": commit,
            "branch": branch,
            "tracked_tree_clean": len(tracked_dirty) == 0,
            "status_porcelain": status_lines,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "device": requested_device,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": cuda_available,
            "platform": platform.platform(),
        },
    }


def _run_tests(repo_root: Path, logger: TeeLogger) -> StageResult:
    command = [sys.executable, "-m", "unittest", *REQUIRED_TEST_MODULES]
    rc, duration = _run_command(command, cwd=repo_root, logger=logger)
    if rc != 0:
        raise PipelineFailure(stage="tests", reason="Unit tests failed", exit_code=EXIT_TESTS_FAILED)
    return StageResult(name="tests", exit_code=rc, duration_sec=duration, details={"modules": list(REQUIRED_TEST_MODULES)})


def _append_stage(summary: dict[str, Any], *, stage: str, status: str, exit_code: int, duration_sec: float, details: dict[str, Any] | None = None) -> None:
    summary.setdefault("stages", []).append(
        {
            "stage": str(stage),
            "status": str(status),
            "exit_code": int(exit_code),
            "duration_sec": float(duration_sec),
            "details": dict(details or {}),
        }
    )


def _checkpoint_identity(repo_root: Path, run_dir: Path, expected_center_sha: str, expected_semantic_sha: str) -> StageResult:
    center_ckpt = (run_dir / "best_micro_overfit.pth").resolve()
    semantic_ckpt = (repo_root / SEMANTIC_CHECKPOINT_REL).resolve()
    if not center_ckpt.exists():
        raise PipelineFailure(stage="checkpoint_identity", reason=f"Missing center checkpoint: {center_ckpt}", exit_code=EXIT_CHECKPOINT_IDENTITY_FAILED)
    if not semantic_ckpt.exists():
        raise PipelineFailure(stage="checkpoint_identity", reason=f"Missing semantic checkpoint: {semantic_ckpt}", exit_code=EXIT_CHECKPOINT_IDENTITY_FAILED)
    center_sha = _sha256_file(center_ckpt)
    semantic_sha = _sha256_file(semantic_ckpt)
    if center_sha != expected_center_sha:
        raise PipelineFailure(stage="checkpoint_identity", reason=f"Center checkpoint SHA mismatch: {center_sha}", exit_code=EXIT_CHECKPOINT_IDENTITY_FAILED)
    if semantic_sha != expected_semantic_sha:
        raise PipelineFailure(stage="checkpoint_identity", reason=f"Semantic checkpoint SHA mismatch: {semantic_sha}", exit_code=EXIT_CHECKPOINT_IDENTITY_FAILED)
    return StageResult(
        name="checkpoint_identity",
        exit_code=0,
        duration_sec=0.0,
        details={
            "center_checkpoint": str(center_ckpt),
            "center_checkpoint_sha": center_sha,
            "semantic_checkpoint": str(semantic_ckpt),
            "semantic_checkpoint_sha": semantic_sha,
        },
    )


def _clean_output_dir(repo_root: Path, output_dir: Path) -> None:
    _safe_output_dir(repo_root, output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _validate_manifest_outputs(output_dir: Path, expected_sha: str) -> dict[str, Any]:
    metadata = _read_json(output_dir / "holdout_manifest_metadata.json")
    checkpoint_identity = _read_json(output_dir / "checkpoint_identity.json")
    if int(metadata["manifest_row_count"]) != 106:
        raise PipelineFailure(stage="manifest_generation", reason=f"Manifest row count mismatch: {metadata['manifest_row_count']}", exit_code=EXIT_MANIFEST_GENERATION_FAILED)
    if int(metadata["unique_sample_count"]) != 106:
        raise PipelineFailure(stage="manifest_generation", reason=f"Manifest unique sample count mismatch: {metadata['unique_sample_count']}", exit_code=EXIT_MANIFEST_GENERATION_FAILED)
    split_counts = metadata["split_counts"]
    if int(split_counts.get("test", 0)) != 53 or int(split_counts.get("val", 0)) != 53:
        raise PipelineFailure(stage="manifest_generation", reason=f"Split counts mismatch: {split_counts}", exit_code=EXIT_MANIFEST_GENERATION_FAILED)
    gt_dist = metadata["gt_instance_distribution"]
    if {str(k): int(v) for k, v in gt_dist.items()} != {"1": 15, "2": 37, "3": 54}:
        raise PipelineFailure(stage="manifest_generation", reason=f"GT distribution mismatch: {gt_dist}", exit_code=EXIT_MANIFEST_GENERATION_FAILED)
    identity_lines = (output_dir / "holdout_manifest_identity.jsonl").read_text(encoding="utf-8").splitlines()
    if len(identity_lines) != 106:
        raise PipelineFailure(stage="manifest_generation", reason=f"Identity JSONL row count mismatch: {len(identity_lines)}", exit_code=EXIT_MANIFEST_GENERATION_FAILED)
    identity_rows = [json.loads(line) for line in identity_lines if line.strip()]
    unique_samples = {str(row["sample"]) for row in identity_rows}
    if len(unique_samples) != 106:
        raise PipelineFailure(stage="manifest_generation", reason="Canonical manifest contains duplicate samples", exit_code=EXIT_MANIFEST_GENERATION_FAILED)
    if MICROSET_IDS & unique_samples:
        raise PipelineFailure(stage="manifest_generation", reason=f"Microset IDs leaked into holdout manifest: {sorted(MICROSET_IDS & unique_samples)}", exit_code=EXIT_MANIFEST_GENERATION_FAILED)
    if str(metadata["canonical_identity_sha256"]) != str(expected_sha):
        raise PipelineFailure(stage="manifest_generation", reason=f"Canonical manifest SHA mismatch: {metadata['canonical_identity_sha256']}", exit_code=EXIT_MANIFEST_IDENTITY_MISMATCH)
    if str(checkpoint_identity.get("manifest_identity_status")) != "exact_match":
        raise PipelineFailure(stage="manifest_generation", reason=f"Manifest identity status mismatch: {checkpoint_identity.get('manifest_identity_status')}", exit_code=EXIT_MANIFEST_IDENTITY_MISMATCH)
    return {
        "samples": int(metadata["manifest_row_count"]),
        "unique_samples": int(metadata["unique_sample_count"]),
        "split_counts": split_counts,
        "gt_count_distribution": gt_dist,
        "manifest_identity_sha": str(metadata["canonical_identity_sha256"]),
    }


def _run_manifest_stage(repo_root: Path, *, config: str, run_dir: str, output_dir: str, device: str, expected_sha: str, logger: TeeLogger) -> StageResult:
    command = [
        sys.executable,
        "training/diagnose_center_generalization_holdout.py",
        "--config",
        config,
        "--run-dir",
        run_dir,
        "--output-dir",
        output_dir,
        "--device",
        device,
        "--manifest-only",
        "--expected-manifest-identity-sha",
        expected_sha,
    ]
    rc, duration = _run_command(command, cwd=repo_root, logger=logger)
    if rc != 0:
        raise PipelineFailure(stage="manifest_generation", reason=f"Manifest subprocess failed with exit code {rc}", exit_code=EXIT_MANIFEST_GENERATION_FAILED)
    details = _validate_manifest_outputs((repo_root / output_dir).resolve(), expected_sha)
    return StageResult(name="manifest_generation", exit_code=rc, duration_sec=duration, details=details)


def _run_diagnosis_stage(repo_root: Path, *, config: str, run_dir: str, output_dir: str, device: str, expected_sha: str, logger: TeeLogger) -> StageResult:
    command = [
        sys.executable,
        "training/diagnose_center_generalization_holdout.py",
        "--config",
        config,
        "--run-dir",
        run_dir,
        "--output-dir",
        output_dir,
        "--device",
        device,
        "--expected-manifest-identity-sha",
        expected_sha,
    ]
    rc, duration = _run_command(command, cwd=repo_root, logger=logger)
    out_dir = (repo_root / output_dir).resolve()
    checkpoint_identity = _read_json(out_dir / "checkpoint_identity.json")
    if rc != 0:
        raise PipelineFailure(stage="authoritative_diagnosis", reason=f"Diagnosis subprocess failed with exit code {rc}", exit_code=EXIT_DIAGNOSIS_SUBPROCESS_FAILED)
    required_statuses = {
        "checkpoint_identity_status": "exact_match",
        "semantic_checkpoint_identity_status": "exact_match",
        "manifest_identity_status": "exact_match",
        "diagnosis_execution_status": "completed",
        "overall_authoritative_status": "exact_match",
    }
    for key, expected in required_statuses.items():
        if str(checkpoint_identity.get(key)) != expected:
            raise PipelineFailure(stage="authoritative_diagnosis", reason=f"Authoritative status mismatch: {key}={checkpoint_identity.get(key)}", exit_code=EXIT_AUTHORITATIVE_STATUS_MISMATCH)
    bottleneck = _read_json(out_dir / "bottleneck_decision.json")
    return StageResult(name="authoritative_diagnosis", exit_code=rc, duration_sec=duration, details={"bottleneck_status": bottleneck["status"]})


def _validate_csv_file(path: Path) -> None:
    rows = path.read_text(encoding="utf-8").splitlines()
    if len(rows) < 2:
        raise PipelineFailure(stage="artifact_integrity", reason=f"CSV missing data rows: {path}", exit_code=EXIT_ARTIFACT_INTEGRITY_FAILED)
    reader = csv.reader(rows)
    header = next(reader)
    if not header:
        raise PipelineFailure(stage="artifact_integrity", reason=f"CSV missing header: {path}", exit_code=EXIT_ARTIFACT_INTEGRITY_FAILED)


def _validate_jsonl_file(path: Path) -> list[dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 106:
        raise PipelineFailure(stage="artifact_integrity", reason=f"JSONL row count mismatch: {path}", exit_code=EXIT_ARTIFACT_INTEGRITY_FAILED)
    rows = [json.loads(line) for line in lines]
    if len({str(row["sample"]) for row in rows}) != 106:
        raise PipelineFailure(stage="artifact_integrity", reason=f"JSONL unique sample count mismatch: {path}", exit_code=EXIT_ARTIFACT_INTEGRITY_FAILED)
    return rows


def _artifact_index(output_dir: Path, *, include_visual_review: bool) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    index: list[dict[str, Any]] = []
    files_to_provide: list[str] = []
    optional_large: list[str] = []
    for rel_path, spec in REQUIRED_ARTIFACTS.items():
        path = (output_dir / rel_path).resolve()
        if not path.exists():
            raise PipelineFailure(stage="artifact_integrity", reason=f"Missing required artifact: {rel_path}", exit_code=EXIT_ARTIFACT_INTEGRITY_FAILED)
        if not path.is_file():
            raise PipelineFailure(stage="artifact_integrity", reason=f"Required artifact is not a file: {rel_path}", exit_code=EXIT_ARTIFACT_INTEGRITY_FAILED)
        if path.stat().st_size <= 0:
            raise PipelineFailure(stage="artifact_integrity", reason=f"Required artifact is empty: {rel_path}", exit_code=EXIT_ARTIFACT_INTEGRITY_FAILED)
        if spec["type"] == "json":
            _read_json(path)
        elif spec["type"] == "csv":
            _validate_csv_file(path)
        elif spec["type"] == "jsonl":
            _validate_jsonl_file(path)
        entry = {
            "relative_path": rel_path,
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
            "required_for_review": bool(spec["required_for_review"]),
            "description": str(spec["description"]),
        }
        index.append(entry)
        if entry["required_for_review"]:
            files_to_provide.append(str(path))
    visual_index_lines: list[str] = []
    for rel_path, spec in OPTIONAL_ARTIFACTS.items():
        path = (output_dir / rel_path).resolve()
        if not path.exists():
            continue
        if path.is_file():
            if path.stat().st_size <= 0:
                raise PipelineFailure(stage="artifact_integrity", reason=f"Optional file is empty: {rel_path}", exit_code=EXIT_ARTIFACT_INTEGRITY_FAILED)
            if spec["type"] == "json":
                _read_json(path)
            elif spec["type"] == "csv":
                _validate_csv_file(path)
            entry = {
                "relative_path": rel_path,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
                "required_for_review": bool(spec["required_for_review"]),
                "description": str(spec["description"]),
            }
            index.append(entry)
            if entry["required_for_review"]:
                files_to_provide.append(str(path))
        elif path.is_dir():
            total_size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            index.append(
                {
                    "relative_path": rel_path,
                    "size_bytes": int(total_size),
                    "sha256": None,
                    "required_for_review": False,
                    "description": str(spec["description"]),
                }
            )
            if include_visual_review and total_size <= VISUAL_REVIEW_SIZE_LIMIT_BYTES:
                optional_large.append(str(path))
            else:
                visual_index_lines.append(f"visual_review_dir={path}")
                visual_index_lines.append(f"total_size_bytes={total_size}")
                for panel in sorted(path.rglob("*.png")):
                    visual_index_lines.append(str(panel.relative_to(output_dir)))
    if visual_index_lines:
        (output_dir / "visual_review_index.txt").write_text("\n".join(visual_index_lines) + "\n", encoding="utf-8", newline="\n")
        path = (output_dir / "visual_review_index.txt").resolve()
        index.append(
            {
                "relative_path": "visual_review_index.txt",
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
                "required_for_review": True,
                "description": "Visual review index when panels are excluded from bundle",
            }
        )
        files_to_provide.append(str(path))
    return index, files_to_provide, optional_large


def _write_files_to_provide(path: Path, files: list[str], optional_large: list[str]) -> None:
    lines = ["FILES TO PROVIDE:"]
    for idx, item in enumerate(files, start=1):
        lines.append(f"{idx}. {item}")
    if optional_large:
        lines.append("")
        lines.append("OPTIONAL LARGE FILES:")
        for idx, item in enumerate(optional_large, start=1):
            lines.append(f"{idx}. {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _bundle_review(output_dir: Path, *, artifact_index: list[dict[str, Any]], include_visual_review: bool) -> tuple[Path, Path]:
    bundle_path = (output_dir / "authoritative_holdout_review_bundle.tar.gz").resolve()
    checksum_path = (output_dir / "authoritative_holdout_review_bundle.tar.gz.sha256").resolve()
    excluded_suffixes = {".pth", ".pt", ".tar"}
    include_paths = []
    for entry in artifact_index:
        rel = entry["relative_path"]
        if rel == "visual_review":
            continue
        include_paths.append((output_dir / rel).resolve())
    visual_dir = (output_dir / "visual_review").resolve()
    if include_visual_review and visual_dir.exists():
        include_paths.extend(sorted(p.resolve() for p in visual_dir.rglob("*") if p.is_file()))
    with tarfile.open(bundle_path, "w:gz") as tar:
        for path in sorted(include_paths, key=lambda p: str(p.relative_to(output_dir)).replace("\\", "/")):
            if path.suffix in excluded_suffixes:
                continue
            if not path.is_file():
                continue
            arcname = str(path.relative_to(output_dir)).replace("\\", "/")
            info = tar.gettarinfo(str(path), arcname=arcname)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            with path.open("rb") as fh:
                tar.addfile(info, fh)
    checksum_path.write_text(f"{_sha256_file(bundle_path)}  {bundle_path.name}\n", encoding="utf-8", newline="\n")
    return bundle_path, checksum_path


def _final_success_block(summary: dict[str, Any], bundle_path: Path, checksum_path: Path, optional_large: list[str]) -> str:
    lines = [
        "==================================================",
        "AUTHORITATIVE HOLDOUT PIPELINE: SUCCESS",
        "==================================================",
        f"git_commit: {summary['git']['commit']}",
        f"hostname: {summary['environment']['hostname']}",
        f"device: {summary['environment']['device']}",
        f"samples: {summary['dataset']['samples']}",
        "canonical_manifest_sha:",
        f"{summary['identity']['manifest_identity_sha']}",
        f"overall_authoritative_status: {summary['authoritative_status']['overall']}",
        f"bottleneck_status: {summary['bottleneck_status']}",
        f"production_activation: {summary['production_activation']}",
        f"training_launched: {str(summary['training_launched']).lower()}",
        "",
        "FILES TO PROVIDE:",
        f"1. {bundle_path}",
        f"2. {checksum_path}",
    ]
    if optional_large:
        lines.extend(["", "OPTIONAL LARGE FILES:"])
        for idx, item in enumerate(optional_large, start=1):
            lines.append(f"{idx}. {item}")
    return "\n".join(lines) + "\n"


def _final_failure_block(summary: dict[str, Any], output_dir: Path, exit_code: int) -> str:
    relevant = list(summary.get("files_to_provide") or [])
    if not relevant:
        relevant = [str((output_dir / "pipeline_run_summary.json").resolve()), str((output_dir / "pipeline.log").resolve())]
    chk = (output_dir / "checkpoint_identity.json").resolve()
    if chk.exists() and str(chk) not in relevant:
        relevant.append(str(chk))
    lines = [
        "==================================================",
        "AUTHORITATIVE HOLDOUT PIPELINE: FAILED",
        "==================================================",
        f"failed_stage: {summary['failed_stage']}",
        f"reason: {summary['failure_reason']}",
        f"exit_code: {exit_code}",
        "diagnostics_preserved: true",
        "",
        "FILES TO PROVIDE:",
    ]
    for idx, item in enumerate(relevant, start=1):
        lines.append(f"{idx}. {item}")
    return "\n".join(lines) + "\n"


def _failure_files_to_provide(output_dir: Path) -> list[str]:
    files = [
        str((output_dir / "pipeline_run_summary.json").resolve()),
        str((output_dir / "pipeline.log").resolve()),
    ]
    chk = (output_dir / "checkpoint_identity.json").resolve()
    if chk.exists():
        files.append(str(chk))
    return files


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    ap.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR)
    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--expected-manifest-identity-sha", type=str, required=True)
    ap.add_argument("--expected-center-checkpoint-sha", type=str, default=DEFAULT_CENTER_SHA)
    ap.add_argument("--expected-semantic-checkpoint-sha", type=str, default=DEFAULT_SEMANTIC_SHA)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--clean-output", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--include-visual-review", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    _safe_output_dir(repo_root, output_dir)
    if args.clean_output:
        _clean_output_dir(repo_root, output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    logger = TeeLogger((output_dir / "pipeline.log").resolve())
    summary: dict[str, Any] = {
        "status": "failed",
        "failed_stage": None,
        "failure_reason": None,
        "git": {},
        "environment": {},
        "identity": {},
        "dataset": {},
        "authoritative_status": {},
        "bottleneck_status": None,
        "production_activation": "blocked",
        "training_launched": False,
        "production_changed": False,
        "artifact_bundle": None,
        "files_to_provide": [],
        "stages": [],
    }
    exit_code = EXIT_SUCCESS
    optional_large: list[str] = []
    try:
        logger.log("Stage 0: repository preflight")
        stage_start = time.perf_counter()
        preflight = _repo_preflight(repo_root, args.device)
        summary["git"] = {k: v for k, v in preflight["git"].items() if k != "status_porcelain"}
        summary["environment"] = preflight["environment"]
        _append_stage(summary, stage="repository_preflight", status="passed", exit_code=0, duration_sec=time.perf_counter() - stage_start, details={"requested_device": args.device})
        if not args.skip_tests:
            logger.log("Stage 1: tests")
            stage = _run_tests(repo_root, logger)
            _append_stage(summary, stage="tests", status="passed", exit_code=stage.exit_code, duration_sec=stage.duration_sec, details=stage.details)
        else:
            logger.log("Stage 1: tests skipped")
            _append_stage(summary, stage="tests", status="passed", exit_code=0, duration_sec=0.0, details={"skipped": True})
        logger.log("Stage 2: checkpoint identity")
        stage = _checkpoint_identity(repo_root, (repo_root / args.run_dir).resolve(), args.expected_center_checkpoint_sha, args.expected_semantic_checkpoint_sha)
        _append_stage(summary, stage="checkpoint_identity", status="passed", exit_code=stage.exit_code, duration_sec=stage.duration_sec, details=stage.details)
        summary["identity"]["center_checkpoint_sha"] = stage.details["center_checkpoint_sha"]
        summary["identity"]["semantic_checkpoint_sha"] = stage.details["semantic_checkpoint_sha"]
        logger.log("Stage 3: output preparation")
        _append_stage(summary, stage="output_preparation", status="passed", exit_code=0, duration_sec=0.0, details={"clean_output": bool(args.clean_output)})
        logger.log("Stage 4: canonical manifest generation")
        stage = _run_manifest_stage(
            repo_root,
            config=args.config,
            run_dir=args.run_dir,
            output_dir=args.output_dir,
            device=args.device,
            expected_sha=args.expected_manifest_identity_sha,
            logger=logger,
        )
        _append_stage(summary, stage="manifest_generation", status="passed", exit_code=stage.exit_code, duration_sec=stage.duration_sec, details=stage.details)
        summary["identity"]["manifest_identity_sha"] = stage.details["manifest_identity_sha"]
        summary["identity"]["manifest_identity_match"] = True
        summary["dataset"] = {
            "samples": stage.details["samples"],
            "unique_samples": stage.details["unique_samples"],
            "split_counts": stage.details["split_counts"],
            "gt_count_distribution": stage.details["gt_count_distribution"],
        }
        logger.log("Stage 5: authoritative CUDA diagnosis")
        stage = _run_diagnosis_stage(
            repo_root,
            config=args.config,
            run_dir=args.run_dir,
            output_dir=args.output_dir,
            device=args.device,
            expected_sha=args.expected_manifest_identity_sha,
            logger=logger,
        )
        _append_stage(summary, stage="authoritative_diagnosis", status="passed", exit_code=stage.exit_code, duration_sec=stage.duration_sec, details=stage.details)
        out_dir = (repo_root / args.output_dir).resolve()
        checkpoint_identity = _read_json(out_dir / "checkpoint_identity.json")
        promotion = _read_json(out_dir / "corrected_promotion_decision.json")
        summary["authoritative_status"] = {
            "checkpoint": checkpoint_identity["checkpoint_identity_status"],
            "semantic_checkpoint": checkpoint_identity["semantic_checkpoint_identity_status"],
            "manifest": checkpoint_identity["manifest_identity_status"],
            "diagnosis": checkpoint_identity["diagnosis_execution_status"],
            "overall": checkpoint_identity["overall_authoritative_status"],
        }
        summary["bottleneck_status"] = stage.details["bottleneck_status"]
        summary["production_activation"] = str(promotion["production_activation_result"]["status"])
        logger.log("Stage 6: artifact integrity")
        artifact_index, files_to_provide, optional_large = _artifact_index(out_dir, include_visual_review=args.include_visual_review)
        _atomic_write_json((out_dir / "artifact_index.json").resolve(), {"artifacts": artifact_index})
        _write_files_to_provide((out_dir / "files_to_provide.txt").resolve(), files_to_provide, optional_large)
        summary["files_to_provide"] = files_to_provide + optional_large
        bundle_path, checksum_path = _bundle_review(out_dir, artifact_index=artifact_index, include_visual_review=args.include_visual_review)
        summary["artifact_bundle"] = str(bundle_path)
        summary["status"] = "success"
        summary["failed_stage"] = None
        summary["failure_reason"] = None
        _atomic_write_json((out_dir / "pipeline_run_summary.json").resolve(), summary)
        logger.write(_final_success_block(summary, bundle_path, checksum_path, optional_large))
        exit_code = EXIT_SUCCESS
    except PipelineFailure as exc:
        exit_code = exc.exit_code
        summary["status"] = "failed"
        summary["failed_stage"] = exc.stage
        summary["failure_reason"] = exc.reason
        if exc.stage == "tests" and not any(stage["stage"] == "tests" for stage in summary.get("stages", [])):
            _append_stage(summary, stage="tests", status="failed", exit_code=1, duration_sec=0.0, details={"reason": exc.reason, "modules": list(REQUIRED_TEST_MODULES)})
        elif exc.stage not in {stage["stage"] for stage in summary.get("stages", [])}:
            _append_stage(summary, stage=exc.stage, status="failed", exit_code=exc.exit_code, duration_sec=0.0, details={"reason": exc.reason})
        summary["files_to_provide"] = _failure_files_to_provide(output_dir)
        _atomic_write_json((output_dir / "pipeline_run_summary.json").resolve(), summary)
        logger.write(_final_failure_block(summary, output_dir, exit_code))
    except Exception as exc:  # noqa: BLE001
        exit_code = EXIT_UNEXPECTED_EXCEPTION
        summary["status"] = "failed"
        summary["failed_stage"] = "unexpected_exception"
        summary["failure_reason"] = str(exc)
        if "repository_preflight" not in {stage["stage"] for stage in summary.get("stages", [])} and summary.get("environment"):
            _append_stage(summary, stage="repository_preflight", status="passed", exit_code=0, duration_sec=0.0, details={"requested_device": args.device})
        summary["files_to_provide"] = _failure_files_to_provide(output_dir)
        _atomic_write_json((output_dir / "pipeline_run_summary.json").resolve(), summary)
        logger.write(_final_failure_block(summary, output_dir, exit_code))
    finally:
        logger.close()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
