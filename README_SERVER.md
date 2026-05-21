# Сервер (Ubuntu 22.04 + 2× A100 40GB) — быстрый старт

## 1) Обновить код

```bash
cd ~/Projects/VHS_HIA
git pull
```

## 2) Сделать скрипты исполняемыми

```bash
chmod +x scripts/server_prepare_multiclass_dataset.sh
chmod +x run_benchmark_a100.sh
chmod +x run_train_a100.sh
chmod +x run_tensorboard.sh
```

## 2) Проверить CUDA

```bash
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

## 3) Установить зависимости (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-train.txt
```

Если на сервере PyTorch ставится отдельно (под нужный CUDA build), сначала поставь `torch/torchvision` по официальной инструкции, а затем:

```bash
python -m pip install -r requirements-train.txt
```

## 4) Проверить датасет

Ожидаемый датасет:
- `datasets/converted_full_multiclass`

Если нужно конвертировать raw Supervisely export на сервере (источник: `Dataset_HIA/supervisely_sdk`):

```bash
bash scripts/server_prepare_multiclass_dataset.sh
```

Пересоздать датасет только явно:

```bash
bash scripts/server_prepare_multiclass_dataset.sh --force
```

Проверка целостности:

```bash
python scripts/validate_dataset_files.py --dataset-root datasets/converted_full_multiclass
python scripts/dataset_stats.py --dataset-root datasets/converted_full_multiclass
```

Если есть битые записи в split-файлах (не удаляет файлы, только обновляет `*.txt` и создаёт `*.bak`):

```bash
python scripts/validate_dataset_files.py --dataset-root datasets/converted_full_multiclass --drop-broken
```

## 5) Dry-run (быстрая проверка пайплайна)

```bash
python training/train.py --config training/configs/unetpp_effb3_a100_multiclass_full_100ep.yaml --dry-run
```

## 6) Benchmark (CUDA)

```bash
bash run_benchmark_a100.sh
```

Ожидаемый вывод включает:
- device + GPU name
- batch_size / input_size / num_workers
- sec_per_step + estimated_min_per_epoch
- CUDA memory allocated/reserved (+ peak)

## 7) Train (НЕ запускать автоматически)

Запуск вручную:

```bash
bash run_train_a100.sh
```

## 8) TensorBoard

```bash
bash run_tensorboard.sh
```

Открой:
- `http://<server-ip>:6006/`
