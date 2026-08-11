from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import semantic_topology_aux as topo_aux


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: CUDA is not available, using CPU for preparation smoke.")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=topo_aux.REPO_ROOT / "training" / "configs" / "unetpp_effb3_semantic_topology_aux_finetune_100ep.yaml",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=topo_aux.DEFAULT_PREP_OUTPUT_DIR,
    )
    args = ap.parse_args()

    cfg = topo_aux._read_yaml(args.config.resolve())
    summary = topo_aux.prepare_experiment(
        cfg=cfg,
        output_dir=args.output_dir.resolve(),
        device=_select_device(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
