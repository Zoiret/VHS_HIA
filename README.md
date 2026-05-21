# VHS_HIA — ML (Ubuntu 22.04 + NVIDIA)

This repo contains:
- Supervisely → segmentation dataset conversion (multiclass + leaflet-only)
- Dataset validation / stats / preview tools
- Training pipeline (PyTorch + segmentation_models_pytorch)
- Manual curation workflow (local HTML gallery) + curated split generation
- Offline (pre-generated) augmentation pipeline for CUDA servers

The repository does not include datasets/exports/checkpoints (they are ignored). You must place them locally under `datasets/` and `exports/`.

## System requirements (Ubuntu 22.04)

- Ubuntu 22.04
- NVIDIA driver installed (verify with `nvidia-smi`)
- Python (recommended: system Python 3.10 on Ubuntu 22.04)
- Git

This project currently trains on a single GPU per process. On multi-GPU machines (e.g. 2×A100 + RTX 4060), select a GPU via `CUDA_VISIBLE_DEVICES`.

## Quick start (Ubuntu 22.04 + CUDA)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install PyTorch with CUDA wheels (pick the right CUDA build for your driver; example uses CUDA 12.4 wheels):

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
python -m pip install -r requirements-train.txt
```

Verify CUDA:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

## Datasets and classes

Mask values (uint8):
- 0: background
- 1: leaflet
- 2: fibrous_ring (Aortic valve base)

Split file format (`train.txt`, `val.txt`, `test.txt`): one line per sample, tab-separated:
`images/<file>.png<TAB>masks/<file>.png`

Recommended datasets used in this repo:
- `datasets/converted_full_multiclass` (multiclass full dataset)
- `datasets/converted_full_leaflet_only` (leaflet-only full dataset)

## Validate a dataset (recommended before training)

```bash
python scripts/validate_dataset_files.py --dataset-root datasets/converted_full_multiclass
python scripts/dataset_stats.py --dataset-root datasets/converted_full_multiclass
```

If validation reports broken samples, remove them from split files (does not delete the image/mask files, only updates `*.txt` and creates `*.bak`):

```bash
python scripts/validate_dataset_files.py --dataset-root datasets/converted_full_multiclass --drop-broken
```

## Offline (pre-generated) augmentation for CUDA training

This generates a new dataset on disk so training can run without runtime spatial augmentations.

Command (train split only; val/test are copied as-is):

```bash
python scripts/build_augmented_dataset.py \
  --dataset-root datasets/converted_full_multiclass \
  --split train \
  --out-root datasets/converted_full_multiclass_aug \
  --num-variants 4 \
  --seed 42
```

Output structure:

```
datasets/converted_full_multiclass_aug/
  images/
  masks/
  train.txt
  val.txt
  test.txt
  augment_meta.csv
```

Validate the augmented dataset:

```bash
python scripts/validate_dataset_files.py --dataset-root datasets/converted_full_multiclass_aug
python scripts/dataset_stats.py --dataset-root datasets/converted_full_multiclass_aug
```

## Training (CUDA)

CUDA config (offline-augmented dataset; runtime aug disabled):
- `training/configs/unetpp_effb3_cuda_multiclass_full_aug_100ep.yaml`

Run on a specific GPU (example: use GPU 0):

```bash
export CUDA_VISIBLE_DEVICES=0
python training/train.py --config training/configs/unetpp_effb3_cuda_multiclass_full_aug_100ep.yaml
```

If you want TensorBoard logs, set in the config:

```yaml
train:
  tensorboard: true
```

Logs will be written to:
`training/runs/<run_name>/tensorboard`

Start TensorBoard:

```bash
python -m pip install tensorboard
tensorboard --logdir training/runs
```

## Portable server run script

Linux helper script (sets cache paths to local `.cache/` before training):

```bash
bash run_train_cuda.sh
```

It exports:
- `TORCH_HOME=.cache/torch`
- `HF_HOME=.cache/huggingface`
- `XDG_CACHE_HOME=.cache`
- `MPLCONFIGDIR=.cache/matplotlib`

## Training (CPU)

Example CPU config:
- `training/configs/unetpp_effb3_cpu_multiclass_full_100ep.yaml`

Dry-run (safe check: forward/loss/backward; does not start full training):

```bash
python training/train.py --config training/configs/unetpp_effb3_cpu_multiclass_full_100ep.yaml --dry-run
```

## Supervisely conversion

Put a Supervisely export under:

```
exports/supervisely/
```

Convert to multiclass dataset:

```bash
python scripts/convert_supervisely_to_segmentation.py --input exports/supervisely/328010_HIA --output datasets/converted_full_multiclass --target multiclass
```

Convert to leaflet-only dataset:

```bash
python scripts/convert_supervisely_to_segmentation.py --input exports/supervisely/328010_HIA --output datasets/converted_full_leaflet_only --target leaflet_only
```

## Curation workflow (multiclass / leaflet-only)

Build a local HTML gallery (no server):

```bash
python scripts/build_curation_gallery.py --source dataset --dataset-root datasets/converted_full_multiclass --out datasets/curated_full_multiclass/curation_gallery.html
```

After manual labeling (clean/medium/bad) and saving `curation_result.json`, apply it:

```bash
python scripts/apply_curation_result.py --input datasets/curated_full_multiclass/curation_result.json --curated-dir datasets/curated_full_multiclass
python scripts/make_curated_split.py --converted-root datasets/converted_full_multiclass --curated-dir datasets/curated_full_multiclass --out-dir datasets/converted_full_multiclass_curated
```
