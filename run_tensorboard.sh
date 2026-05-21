#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

tensorboard --logdir training/runs/unetpp_effb3_a100_multiclass_full_100ep/tensorboard --host 0.0.0.0 --port 6006
