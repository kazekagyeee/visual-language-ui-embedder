# Training 3B Projection Adapter

Этот пайплайн обучает последний слой-адаптер для retrieval:

`Qwen/UI encoder 3B vector -> BERT/SentenceTransformer vector`

Для пресета `3B` размерность UI-вектора равна `2048`, а для
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` размерность BERT-вектора равна `384`.
Обучается только маленький adapter, сам UI-энкодер и BERT-энкодер заморожены.

## Что используется из соседнего проекта

- `build_ui_index_from_pdf_pages.py`: логика извлечения скриншотов из PDF и OCR UI-элементов.
- `build_ui_training_pairs.py`: fuzzy matching OCR-текста с ожидаемыми UI-элементами.
- `user_scenario_eval.py`: `USER_TEST_CASES` как control queries для сценарных триплетов.
- `ui_siamese_ranker.py`: идея отдельного projection/ranker слоя поверх frozen embeddings.

## Структура

- `pdf/` - PDF-файлы для генерации датасета.
- `build_ui_index_from_pdfs.py` - строит OCR-индекс UI-элементов из PDF.
- `build_triplet_dataset.py` - строит триплеты `query, positive UI, negative UI`.
- `adapter_model.py` - adapter `input_dim -> BERT_dim`.
- `train_3b_projection_adapter.py` - обучает adapter для 3B модели.
- `evaluate_projection_adapter.py` - считает retrieval-метрики после обучения.
- `run_training.ps1` - полный запуск пайплайна.

## Установка зависимостей

```powershell
.\.venv\Scripts\python.exe -m pip install -r new_training\requirements.txt
```

## Запуск по шагам

```powershell
.\.venv\Scripts\python.exe new_training\build_ui_index_from_pdfs.py --pdf-dir new_training\pdf --out-dir new_training\generated\ui_index --resume
.\.venv\Scripts\python.exe new_training\build_triplet_dataset.py --ui-items new_training\generated\ui_index\ui_items.jsonl --out new_training\generated\triplets.jsonl
.\.venv\Scripts\python.exe new_training\train_3b_projection_adapter.py --triplets new_training\generated\triplets.jsonl --model-size 3B --device cuda
.\.venv\Scripts\python.exe new_training\evaluate_projection_adapter.py --model-size 3B --device cuda
```

По умолчанию 3B UI-энкодер работает на `--device`, а BERT/SentenceTransformer для запросов держится на CPU, чтобы не занимать лишнюю VRAM. Если нужно иначе, передайте `--bert-device cuda`.

Или одним скриптом:

```powershell
.\new_training\run_training.ps1 -Device cuda -Epochs 5 -BatchSize 64
```

Для OCR на GPU:

```powershell
.\new_training\run_training.ps1 -Device cuda -OcrDevice cuda
```

Если видите предупреждение `Using CPU. Note: This module is much faster with a GPU.`, оно относится к EasyOCR. По умолчанию скрипт использует `--ocr-device auto` и включает CUDA, когда `torch.cuda.is_available()` возвращает `True`; принудительно можно задать `--ocr-device cuda` или `-OcrDevice cuda`.

## Основные артефакты

- `generated/ui_index/ui_items.jsonl` - OCR UI-элементы из PDF.
- `generated/triplets.jsonl` - обучающие триплеты.
- `generated/qwen3b_visual_cache.pt` - кэш frozen 3B UI-векторов.
- `output/projection_adapter_3b/best_adapter.pt` - лучший adapter.
- `output/projection_adapter_3b/adapter.pt` - финальный adapter.
- `output/projection_adapter_eval.json` - подробный retrieval-отчет.
