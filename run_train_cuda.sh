#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export TORCH_HOME=".cache/torch"
export HF_HOME=".cache/huggingface"
export XDG_CACHE_HOME=".cache"
export MPLCONFIGDIR=".cache/matplotlib"

python training/train.py --config training/configs/unetpp_effb3_cuda_multiclass_full_aug_100ep.yaml
