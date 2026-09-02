from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bridge_presence_gate_v4 as gate_v4
import bridge_suppression_head as bridge
import train_bridge_presence_gate_v4 as runner


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v4_micro_overfit.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "training" / "analysis" / "bridge_presence_gate_v4_reconstruction_profile"


def runtime_threading_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "cpu_affinity": None,
        "os_cpu_count": int(os.cpu_count() or 0),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "numpy_blas": None,
        "threadpool_info": None,
    }
    try:
        if hasattr(os, "sched_getaffinity"):
            snapshot["cpu_affinity"] = sorted(int(v) for v in os.sched_getaffinity(0))
    except Exception:
        snapshot["cpu_affinity"] = None
    try:
        import numpy as np

        snapshot["numpy_blas"] = np.__config__.get_info("blas_opt_info")
    except Exception:
        snapshot["numpy_blas"] = None
    try:
        from threadpoolctl import threadpool_info  # type: ignore

        snapshot["threadpool_info"] = threadpool_info()
    except Exception:
        snapshot["threadpool_info"] = None
    return snapshot


def build_profile(cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    bridge._seed_everything(int(cfg.get("seed", 1337)))
    prepared = runner._prepare_v4_inputs(cfg)
    profile = gate_v4.profile_hard_gate_reconstruction_states(
        prepared["cached_micro"],
        prepared["pixel_remove_masks"],
        output_dir=output_dir,
    )
    rows = profile["rows"]
    hotspot_seconds = float(sum(float(row["reference_centroid_distance_computation_seconds"]) for row in rows))
    total_ref = float(profile["reference_total_reconstruction_seconds"])
    profile_summary = {
        "profile_artifact_dir": str(output_dir.resolve()),
        "threading_snapshot": runtime_threading_snapshot(),
        "reference_total_reconstruction_seconds": total_ref,
        "optimized_total_reconstruction_seconds": float(profile["optimized_total_reconstruction_seconds"]),
        "reference_mean_per_state_seconds": float(profile["reference_mean_per_state_seconds"]),
        "optimized_mean_per_state_seconds": float(profile["optimized_mean_per_state_seconds"]),
        "speedup_factor": float(profile["speedup_factor"]),
        "hotspot": {
            "dominant_function": "_merge_groups_exact_k / centroid_distance_k_normalizer",
            "dominant_operation": "repeated group-centroid recomputation during pair ranking",
            "reference_centroid_distance_computation_seconds": hotspot_seconds,
            "reference_centroid_distance_fraction": float(hotspot_seconds / max(total_ref, 1.0e-12)),
            "reference_group_pair_evaluations": int(sum(int((row.get("reference_call_counts") or {}).get("group_pair_evaluations", 0)) for row in rows)),
        },
        "slowest_reference_state": profile.get("slowest_reference_state"),
        "slowest_optimized_state": profile.get("slowest_optimized_state"),
        "projected_157_sample_two_state_cache_seconds": float(profile["optimized_mean_per_state_seconds"] * 314.0),
    }
    bridge._write_json(output_dir / "reconstruction_profile_summary.json", profile_summary)
    return profile_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile V4 CLOSED/OPEN reconstruction states.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    cfg = bridge._read_yaml(Path(args.config))
    summary = build_profile(cfg, Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
