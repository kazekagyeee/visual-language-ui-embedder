# Visual Language UI Embedder

Проект теперь состоит из двух независимых частей:

1. `legacy`-пайплайн на базе Qwen2.5-VL для тяжёлых контекстных UI-эмбеддингов и генерации описаний.
2. новый `lightweight retriever` на базе `google/siglip2-so400m-patch16-naflex` для `text-to-region retrieval` и экспорта коротких region embeddings.

Новый retriever не заменяет старый pipeline. Он живёт рядом, использует свой `src/`-стек и предназначен для индексируемого слоя поиска по UI-регионам.

## Что изменилось

В проект добавлен новый модуль обучения и инференса dual encoder:

- текстовый encoder: `text -> z_text`
- encoder региона: `bbox crop + bbox features -> z_image`
- пространство эмбеддингов: общее, `256-dim`, `L2-normalized`
- обучение: `contrastive + lambda_triplet * triplet`
- экспорт: positive region embeddings для загрузки в БД

Под это добавлены новые директории:

- [src/config](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\config)
- [src/data](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\data)
- [src/models](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\models)
- [src/training](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\training)
- [src/evaluation](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\evaluation)
- [src/inference](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\inference)
- [src/utils](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\utils)
- [scripts](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\scripts)

## Структура проекта

### Legacy stack

Старый стек остаётся рабочим и не тронут концептуально:

- [main.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\main.py): основной Qwen-based pipeline
- [box_aware_visual_encoder.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\box_aware_visual_encoder.py): box-aware visual encoder
- [uied_detector.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\uied_detector.py): UIED detector
- [training/train_lora_triplet.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\training\train_lora_triplet.py): LoRA triplet training для legacy pipeline

### New SigLIP2 retriever

Новый стек начинается с [src/models/ui_dual_encoder.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\models\ui_dual_encoder.py).

Ключевые файлы:

- [src/data/triplet_dataset.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\data\triplet_dataset.py): загрузка triplet dataset, crop логика, bbox features
- [src/data/collate.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\data\collate.py): батчинг через `AutoProcessor`
- [src/models/siglip2_text_tower.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\models\siglip2_text_tower.py): текстовая ветка
- [src/models/siglip2_image_tower.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\models\siglip2_image_tower.py): image ветка
- [src/models/bbox_mlp.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\models\bbox_mlp.py): MLP для bbox features
- [src/models/fusion_head.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\models\fusion_head.py): fusion head
- [src/training/losses.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\training\losses.py): contrastive/triplet losses
- [src/training/trainer.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\training\trainer.py): train/val loop, ранняя остановка, сохранение checkpoint
- [src/evaluation/retrieval_metrics.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\evaluation\retrieval_metrics.py): `Recall@K`, `MRR`, `Median Rank`
- [src/evaluation/pairwise_metrics.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\evaluation\pairwise_metrics.py): `pos_vs_neg_accuracy`, margin stats
- [src/inference/export_embeddings.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\inference\export_embeddings.py): экспорт positive corpus embeddings

## Архитектура нового retriever

Схема:

```text
query_text --> SigLIP2 text tower --> projection --> z_text

pos_crop --> SigLIP2 image tower --> \
pos_bbox_features --> bbox MLP --> fusion --> z_pos

neg_crop --> SigLIP2 image tower --> \
neg_bbox_features --> bbox MLP --> fusion --> z_neg
```

Свойства MVP:

- backbone: `google/siglip2-so400m-patch16-naflex`
- размерность эмбеддинга: `256`
- similarity: cosine
- все выходы L2-normalized
- по умолчанию backbone frozen
- опционально можно размораживать последние блоки

## Формат данных

Ожидаемый sample в JSON:

```json
{
  "image_path": "path/to/image.png",
  "text": "text query",
  "pos_bbox": [0.1, 0.2, 0.3, 0.4],
  "neg_bbox": [0.5, 0.6, 0.7, 0.8]
}
```

Требования:

- `bbox` нормализованы в `[0, 1]`
- `text` не пустой
- `x2 > x1`, `y2 > y1`
- изображения читаемы

Dataset слой сам:

- отбрасывает битые sample’ы
- умеет работать и с абсолютными, и с относительными путями
- режет `crop` с `crop_pad_ratio=0.05`
- делает `safe expand`, если crop слишком маленький
- вычисляет bbox features:

```text
[x1, y1, x2, y2, cx, cy, w, h, area]
```

## Подготовка датасета

Если JSON ссылается на изображения вне репозитория, сначала перенесите датасет внутрь проекта.

Для этого добавлен [scripts/relocate_triplet_dataset.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\scripts\relocate_triplet_dataset.py):

```bash
python scripts/relocate_triplet_dataset.py \
  --json-path training/triplet_dataset.json \
  --output-json-path training/triplet_dataset_local.json \
  --images-dir training/dataset_images
```

Что делает скрипт:

- копирует все используемые изображения в указанную папку внутри проекта
- переписывает `image_path` на относительные пути
- пропускает отсутствующие файлы
- одинаковые исходные изображения копирует один раз

После этого рекомендуется построить split-файл:

```bash
python scripts/build_splits.py \
  --json-path training/triplet_dataset_local.json \
  --output-path training/dual_encoder_splits.json
```

Разбиение идёт по `image_path`, а не по строкам JSON. Это важно, чтобы один и тот же экран не попал в train и val/test одновременно.

## Audit датасета

Проверка качества входных данных:

```bash
python scripts/audit_dataset.py --json-path training/triplet_dataset_local.json
```

Скрипт считает:

- количество валидных и невалидных sample’ов
- проблемы с bbox
- unreadable image count
- статистику размеров bbox

## Обучение нового retriever

Основной entrypoint: [scripts/train_dual_encoder.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\scripts\train_dual_encoder.py)

Минимальный запуск:

```bash
python scripts/train_dual_encoder.py \
  --json-path training/triplet_dataset_local.json \
  --split-path training/dual_encoder_splits.json \
  --output-dir artifacts/dual_encoder
```

Что делает training script:

- загружает `AutoProcessor` и `UIDualEncoder`
- создаёт train/val dataset
- запускает training loop
- считает val метрики
- сохраняет:
  - `last.ckpt`
  - `best_recall_at_1.ckpt`
  - `train_config.json`
  - `metrics_history.jsonl`

Основные training параметры находятся в [src/config/train_config.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\config\train_config.py).

Ключевые дефолты:

- `embedding_dim=256`
- `temperature=0.07`
- `triplet_margin=0.2`
- `lambda_triplet=0.3`
- `micro_batch_size=8`
- `grad_accum_steps=8`
- `effective_batch_size=64`
- `freeze_backbone=True`
- `mixed_precision="bf16"`

## Evaluation

Entry point: [scripts/eval_dual_encoder.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\scripts\eval_dual_encoder.py)

Пример:

```bash
python scripts/eval_dual_encoder.py \
  --json-path training/triplet_dataset_local.json \
  --split-path training/dual_encoder_splits.json \
  --checkpoint-path artifacts/dual_encoder/best_recall_at_1.ckpt \
  --config-path artifacts/dual_encoder/train_config.json \
  --split test
```

Считаются:

- `loss`
- `contrastive_loss`
- `triplet_loss`
- `pos_vs_neg_accuracy`
- `recall@1`
- `recall@5`
- `recall@10`
- `mrr`
- `median_rank`
- margin statistics

Дополнительно сохраняется qualitative report с best/worst/false positive/false negative кейсами.

## Экспорт эмбеддингов

Entry point: [scripts/export_region_embeddings.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\scripts\export_region_embeddings.py)

Пример:

```bash
python scripts/export_region_embeddings.py \
  --json-path training/triplet_dataset_local.json \
  --split-path training/dual_encoder_splits.json \
  --checkpoint-path artifacts/dual_encoder/best_recall_at_1.ckpt \
  --config-path artifacts/dual_encoder/train_config.json \
  --split test \
  --output-prefix artifacts/dual_encoder/test_regions
```

На выходе:

- `test_regions.npy`
- `test_regions.jsonl`
- `test_regions.parquet` если доступен parquet backend

Формат записи:

```json
{
  "sample_id": "...",
  "image_path": "...",
  "bbox": [x1, y1, x2, y2],
  "embedding": [...],
  "text": "...",
  "split": "train|val|test"
}
```

## Важные инженерные решения

### 1. Почему новый retriever отделён от legacy stack

Legacy pipeline дорогой и завязан на heavy teacher-like multimodal path. Для retrieval по БД нужен отдельный дешёвый индексируемый слой. Поэтому новый код лежит в `src/` и не зависит от `main.py` как от training dependency.

### 2. Почему используется `AutoProcessor`

Для SigLIP2 текстовая предобработка должна идти через штатный processor. В проекте это зафиксировано в collate-функции:

- `padding="max_length"`
- `truncation=True`
- `max_length=64`

Ручную токенизацию или кастомный lowercasing тут лучше не добавлять.

### 3. Почему image path теперь можно хранить относительным

Изначальный dataset ссылался на внешние абсолютные пути. Это плохо переносится между машинами и workspace. Поэтому:

- добавлен `relocate_triplet_dataset.py`
- dataset loader теперь умеет резолвить относительный путь относительно самого JSON

### 4. Shared backbone

В `UIDualEncoder` text tower и image tower используют общий SigLIP2 backbone, а не две независимые загруженные модели. Это уменьшает память и держит архитектуру ближе к dual encoder.

## Ограничения текущей версии

Сейчас это именно MVP:

- нет teacher distillation
- нет reranker / cross-encoder
- нет второй image branch с full screenshot thumbnail
- нет Matryoshka loss
- нет production API

Также стоит учитывать:

- `SigLIP2 NaFlex` весит заметно меньше legacy Qwen pipeline, но всё равно требует аккуратного обращения с VRAM
- качество сильно зависит от чистоты triplet dataset
- qualitative report пока текстовый, без визуальных thumbnail

## Быстрый рабочий сценарий

Если заходите в проект впервые, рабочая последовательность такая:

1. Установить зависимости:

```bash
pip install -r requirements.txt
```

2. Если dataset с внешними абсолютными путями, перенести его в проект:

```bash
python scripts/relocate_triplet_dataset.py \
  --json-path training/triplet_dataset.json \
  --output-json-path training/triplet_dataset_local.json \
  --images-dir training/dataset_images
```

3. Проверить dataset:

```bash
python scripts/audit_dataset.py --json-path training/triplet_dataset_local.json
```

4. Построить split:

```bash
python scripts/build_splits.py \
  --json-path training/triplet_dataset_local.json \
  --output-path training/dual_encoder_splits.json
```

5. Запустить обучение:

```bash
python scripts/train_dual_encoder.py \
  --json-path training/triplet_dataset_local.json \
  --split-path training/dual_encoder_splits.json \
  --output-dir artifacts/dual_encoder
```

6. Оценить checkpoint:

```bash
python scripts/eval_dual_encoder.py \
  --json-path training/triplet_dataset_local.json \
  --split-path training/dual_encoder_splits.json \
  --checkpoint-path artifacts/dual_encoder/best_recall_at_1.ckpt \
  --config-path artifacts/dual_encoder/train_config.json \
  --split test
```

7. Экспортировать эмбеддинги:

```bash
python scripts/export_region_embeddings.py \
  --json-path training/triplet_dataset_local.json \
  --split-path training/dual_encoder_splits.json \
  --checkpoint-path artifacts/dual_encoder/best_recall_at_1.ckpt \
  --config-path artifacts/dual_encoder/train_config.json \
  --split test \
  --output-prefix artifacts/dual_encoder/test_regions
```

## Что смотреть новому разработчику в первую очередь

Если нужно быстро разобраться в коде, порядок чтения такой:

1. [README.md](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\README.md)
2. [src/models/ui_dual_encoder.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\models\ui_dual_encoder.py)
3. [src/data/triplet_dataset.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\data\triplet_dataset.py)
4. [src/training/trainer.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\src\training\trainer.py)
5. [scripts/train_dual_encoder.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\scripts\train_dual_encoder.py)
6. [scripts/relocate_triplet_dataset.py](C:\Users\kazekagyee\Documents\CodeProjects\visual-language-ui-embedder\scripts\relocate_triplet_dataset.py)

## Лицензия

MIT
