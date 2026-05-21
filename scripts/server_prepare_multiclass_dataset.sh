#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force)
      FORCE=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: bash scripts/server_prepare_multiclass_dataset.sh [--force]" >&2
      exit 2
      ;;
  esac
done

INPUT_DIR="Dataset_HIA/supervisely_sdk"
OUT_DIR="datasets/converted_full_multiclass"

if [ ! -d "$INPUT_DIR" ]; then
  echo "Input dataset not found: $ROOT_DIR/$INPUT_DIR" >&2
  exit 2
fi

if [ -d "$OUT_DIR" ] && [ "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  if [ "$FORCE" -ne 1 ]; then
    echo "Dataset already exists and is not empty: $ROOT_DIR/$OUT_DIR" >&2
    echo "Re-run with --force to recreate it." >&2
    exit 2
  fi
  echo "Recreating dataset (force): $ROOT_DIR/$OUT_DIR"
  rm -rf "$OUT_DIR"
fi

python scripts/convert_supervisely_to_segmentation.py \
  --input "$INPUT_DIR" \
  --output "$OUT_DIR" \
  --target multiclass

python scripts/validate_dataset_files.py --dataset-root "$OUT_DIR"
python scripts/dataset_stats.py --dataset-root "$OUT_DIR"
