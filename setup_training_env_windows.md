# Setup ML Training Environment (Windows)

Цель: подготовить окружение для `training/` на Python 3.12 x64 (PyTorch-friendly), не ломая dataset pipeline.

## 1) Проверить установленные Python через py launcher

```bash
py -0p
```

Убедись, что есть Python 3.12 x64 (обычно будет строка вида `-3.12-64 ...`).

## 2) Создать venv на Python 3.12 внутри проекта

Запускать из `E:\3d_visual\ml`:

```bash
py -3.12 -m venv .venv
```

## 3) Активировать venv

```bash
.venv\Scripts\activate
```

Проверка, что активировался нужный Python:

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
```

## 4) Установить зависимости

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-train.txt
```

Если установка `torch` не проходит (часто из-за несовпадения версии Python/CUDA), ставь PyTorch по официальной инструкции под свою конфигурацию CPU/CUDA, затем повтори установку `requirements-train.txt` (или установи `segmentation-models-pytorch` отдельно):

- https://pytorch.org/get-started/locally/

## 5) Проверить torch

```bash
python -c "import torch; print('torch', torch.__version__)"
```

## 6) Проверить CUDA (если ожидается GPU)

```bash
python -c "import torch; print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda)"
```

Если `cuda_available` = `False`, training всё равно должен работать на CPU (будет предупреждение в консоли).

## 7) Запустить dry-run

Из `E:\3d_visual\ml`:

Для offline/VM режима (без интернета) убедись, что в конфиге стоит:

```yaml
model:
  encoder_weights: null
```

```bash
python training/train.py --config training/configs/unetpp_effb3.yaml --dry-run
```

Dry-run должен:
- загрузить несколько sample’ов,
- сделать forward pass,
- посчитать loss,
- сделать короткий backward (без долгого обучения).
