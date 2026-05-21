# Histological Image Analyzer — ML (Dataset Pipeline)

Этот подпроект содержит подготовку датасета и ML training pipeline.

## Структура

```
ml/
  datasets/
    raw/
    converted/
      images/
      masks/
      meta/
      train.txt
      val.txt
      test.txt
    previews/
  exports/
    supervisely/
  scripts/
    convert_supervisely_to_segmentation.py
    preview_dataset.py
    dataset_stats.py
  training/
    configs/
  models/
  notebooks/
  requirements.txt
  requirements-train.txt
```

## Установка зависимостей

```bash
python -m pip install -r requirements.txt
```

## Training окружение (Windows)

Training pipeline требует Python 3.12 x64 (чтобы корректно ставились wheels для PyTorch).

Создание venv внутри `E:\3d_visual\ml`:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-train.txt
```

Если установка `torch` падает, ставь PyTorch по официальной инструкции под свою конфигурацию (CPU/CUDA), затем докидывай остальное:

- https://pytorch.org/get-started/locally/

Offline/VM режим (без интернета):
- используй `encoder_weights: null` в training config, чтобы не было попыток скачать pretrained веса.
- `imagenet` веса требуют интернет или заранее прогретый cache.

Подробная инструкция: `setup_training_env_windows.md`.

Benchmark на CPU (оценка скорости без долгого обучения):

```bash
python training/train.py --config training/configs/unetpp_effb3.yaml --benchmark-steps 10
```

Короткий CPU training run (10 эпох):

```bash
python training/train.py --config training/configs/unetpp_effb3_cpu_10ep.yaml
```

Overnight CPU training run (~40 эпох):

```bash
python training/train.py --config training/configs/unetpp_effb3_cpu_overnight.yaml
```

## Конвертация Supervisely → segmentation dataset

Положи экспорт Supervisely в:

```
exports/supervisely/
```

Запуск:

```bash
python scripts/convert_supervisely_to_segmentation.py --input exports/supervisely/328010_HIA --output datasets/converted --target multiclass
```

Классы (итоговая маска uint8):
- 0 — background
- 1 — leaflet
- 2 — fibrous_ring

Формат split-файлов:
- `train.txt / val.txt / test.txt`: по строке на объект, 2 колонки через таб: `images/<file>\tmasks/<file>.png`

## Быстрый preview (overlay)

```bash
python scripts/preview_dataset.py --dataset-root datasets/converted --out datasets/previews
```

## Статистика датасета

```bash
python scripts/dataset_stats.py --dataset-root datasets/converted
```

## Проверка целостности converted dataset

Проверить, что все записи из `train.txt/val.txt/test.txt` указывают на читаемые image/mask файлы:

```bash
python scripts/validate_dataset_files.py --dataset-root datasets/converted
```

Удалить битые записи из split-файлов (файлы на диске не трогаются, создаются `*.bak`):

```bash
python scripts/validate_dataset_files.py --dataset-root datasets/converted --drop-broken
```

## Audit Supervisely export

```bash
python scripts/audit_supervisely_export.py --input exports/supervisely/328010_HIA
```

## Full conversion (без удаления старых датасетов)

Полный multiclass датасет (0 background, 1 Leaf, 2 Aortic valve base):

```bash
python scripts/convert_supervisely_to_segmentation.py --input exports/supervisely/328010_HIA --output datasets/converted_full_multiclass --target multiclass
python scripts/validate_dataset_files.py --dataset-root datasets/converted_full_multiclass
python scripts/dataset_stats.py --dataset-root datasets/converted_full_multiclass
```

Полный leaflet-only датасет (0 background, 1 Leaf; Aortic valve base → background):

```bash
python scripts/convert_supervisely_to_segmentation.py --input exports/supervisely/328010_HIA --output datasets/converted_full_leaflet_only --target leaflet_only
python scripts/validate_dataset_files.py --dataset-root datasets/converted_full_leaflet_only
python scripts/dataset_stats.py --dataset-root datasets/converted_full_leaflet_only
```

Разница:
- `datasets/converted` — старый датасет, созданный предыдущей логикой matching.
- `datasets/converted_full_*` — новые датасеты с устойчивым matching ann→img и опцией `--target`.

## Curation gallery для полного датасета

```bash
python scripts/build_curation_gallery.py --source dataset --dataset-root datasets/converted_full_leaflet_only --out datasets/curated_full_leaflet/curation_gallery.html
```

Применение результатов и генерация curated split:

```bash
python scripts/apply_curation_result.py --input datasets/curated_full_leaflet/curation_result.json --curated-dir datasets/curated_full_leaflet
python scripts/make_curated_split.py --converted-root datasets/converted_full_leaflet_only --curated-dir datasets/curated_full_leaflet --out-dir datasets/converted_full_leaflet_curated
```

## Curation gallery для полного multiclass датасета

Галерея:

```bash
python scripts/build_curation_gallery.py --source dataset --dataset-root datasets/converted_full_multiclass --out datasets/curated_full_multiclass/curation_gallery.html
```

Критерии multiclass curation:
- clean: хорошие leaflet, хорошо видимое fibrous ring, корректные границы, нормальная морфология
- medium: небольшие дефекты, слабый контраст, частично рваное кольцо, допустимо для train
- bad: сильные артефакты, плохая анатомия, сомнительная разметка

Применение результатов и генерация curated split:

```bash
python scripts/apply_curation_result.py --input datasets/curated_full_multiclass/curation_result.json --curated-dir datasets/curated_full_multiclass
python scripts/make_curated_split.py --converted-root datasets/converted_full_multiclass --curated-dir datasets/curated_full_multiclass --out-dir datasets/converted_full_multiclass_curated
```

## Curated subset (ручная сортировка)

Пайплайн:
- Сделать inference preview на `val.txt` (генерирует `training/inference_preview/compare/*.png`)
- Сгенерировать HTML-галерею для ручной сортировки:

```bash
python scripts/build_curation_gallery.py --source dataset
```

- Сгенерировать галерею только по inference previews:

```bash
python scripts/build_curation_gallery.py --source inference
```

- Тестовая маленькая галерея:

```bash
python scripts/build_curation_gallery.py --source dataset --limit 20
```

- Открыть файл в браузере (без сервера):
  - `datasets/curated/curation_gallery.html`
- Разметить samples как `clean / medium / bad`, нажать Export, сохранить JSON в:
  - `datasets/curated/curation_result.json`
- Применить JSON к спискам:

```bash
python scripts/apply_curation_result.py --input datasets/curated/curation_result.json
```

- Сгенерировать curated split (без копирования файлов, только новые `train/val/test`):

```bash
python scripts/make_curated_split.py
```
