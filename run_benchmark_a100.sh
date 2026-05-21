#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0

python training/train.py \
  --config training/configs/unetpp_effb3_a100_multiclass_full_100ep.yaml \
  --benchmark-steps 50
