#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"

python training/train.py --config training/configs/unetpp_effb3_a100_multiclass_full_100ep.yaml
