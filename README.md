# Visual Language UI Embedder

Пайплайн для генерации контекстных эмбеддингов UI компонентов на основе Qwen2.5-VL для RAG-систем.

## 🎯 Назначение

Система извлекает семантические эмбеддинги отдельных UI компонентов с учетом контекста всего изображения и текстового описания. Эмбеддинги используются для поиска релевантных элементов интерфейса по текстовым запросам.

## 📊 Архитектура

```
Input Image (dynamic resolution)
    ↓ Smart Resize (align to 28px)
    ↓ UIED Detection (CV-based)
    ↓ Qwen2_5_BoxEncoder (ViT 32L, 1280-dim)
    ↓ ROI Pooling (extract per bbox)
    ↓ Qwen2VLSpatialMerge (2×2 grouping + MLP → 3584-dim)
    ↓ VisualToTextProjector (align with text)
    ↓ HeadlessQwen2_5 (28L LLM)
    ↓ Final Embeddings (N, 3584) → JSON
```

**Визуализация:** Откройте [pipeline_diagram.html](pipeline_diagram.html) для интерактивной диаграммы.

## 🔑 Ключевые возможности

### ✅ Реализовано

- **Native Dynamic Resolution**: Изображения обрабатываются без жёсткого resize к фиксированному размеру
- **Qwen2.5-VL Spatial Merge**: Правильная 2×2 группировка патчей + MLP проекция
- **ROI Pooling**: Извлечение эмбеддингов для каждого UI компонента из merged grid
- **Контекстные эмбеддинги**: Учет глобального изображения + текстового описания
- **Mean Pooling для текста**: Семантическое представление (лучше для retrieval)
- **Debug визуализация**: Сохранение bbox на изображении
- **Timing logs**: Замеры времени выполнения этапов

### 🔧 Опционально (TODO)

- **Learned Tokens**: Обучаемые токены вместо ROI pooling (требует файнтюнинга)
- **M-RoPE**: 3D позиционные эмбеддинги (temporal + spatial) вместо 2D

## 🚀 Использование

### Установка

```bash
pip install -r requirements.txt
```

### Запуск как standalone скрипт

```bash
python main.py
```

**Входные данные (по умолчанию):**
- `input_images/image_20_2.png` - изображение UI
- `input_images/image_20_2.txt` - текстовое описание

**Выходные данные:**
- `output/embeddings.json` - словарь с форматом `[tuple(bbox)] = embedding`
- `debug/uied_bbox_debug.png` - визуализация детекций
- `debug/embedding_similarities.json` - анализ косинусного сходства компонентов

### Использование как модели в других проектах

Вы можете легко импортировать пайплайн и использовать его программно в своих приложениях.

```python
from main import UIEmbedderPipeline
from config import UIEmbedderConfig
from PIL import Image

# 1. (Опционально) Переопределение конфигурации
config = UIEmbedderConfig(
    device="cuda", # если хватает VRAM
    debug_decode_embeddings=False # отключить дебаг для скорости
)

# 2. Инициализация пайплайна (загружает веса)
pipeline = UIEmbedderPipeline(config)

# 3. Подготовка входных данных
image = Image.open("path/to/my/ui_screenshot.png").convert("RGB")
text_context = "Текст описывающий контекст экрана"

# 4. Запуск генерации эмбеддингов
# Получаем словарь: ключи - координаты (x1, y1, x2, y2), значения - эмбеддинги (list of floats)
embeddings_dict = pipeline.process(image, text_context)

for bbox, embedding in embeddings_dict.items():
    print(f"Component at {bbox}: {len(embedding)}-dimensional vector")
```

### Конфигурация (`config.py`)

Все настраиваемые параметры вынесены в отдельный dataclass `UIEmbedderConfig`. Основные параметры:

- `device`: `"cpu"` или `"cuda"`
- `system_prompt` и `context_prompt`: промпты для LLM
- `llm_dim`, `heads_vis`, `depth_vis`: параметры архитектуры (под Qwen2.5-VL-7B)
- `model_name`: `"Qwen/Qwen2.5-VL-7B-Instruct"` (откуда скачивать веса)
- `img_size`, `patch_size_encoder`, `patch_size_resize`: настройки обработки изображений
- `max_dist`: параметр uied детектора для склеивания близких боксов
- `debug_decode_embeddings`: включение/выключение анализа сходства (анализ замедляет работу)

## 📁 Структура проекта

```
.
├── main.py                          # Основной скрипт
├── box_aware_visual_encoder.py      # ViT + Spatial Merge + ROI Pooling
├── vision_to_text_projector.py      # Проектор visual → LLM space
├── headless_qwen_llm.py             # LLM без generation head
├── uied_detector.py                 # UIED детектор UI компонентов
├── load_qwen_weights.py             # Загрузчик весов Qwen2-VL
├── pipeline_diagram.html            # Интерактивная диаграмма
├── input_images/                    # Входные данные
├── output/                          # Эмбеддинги
└── debug/                           # Debug артефакты
```

## 🔬 Использование для RAG

### 1. Индексация

```python
# Сохраняем эмбеддинги в векторную БД
import json

with open("output/embeddings.json") as f:
    segments = json.load(f)

for seg in segments:
    bbox = seg["bbox"]
    embedding = seg["embedding"]  # (3584,)
    # Сохранить в Pinecone/Weaviate/Qdrant
```

### 2. Поиск

```python
# Подготовка эмбеддинга запроса (используем токенизатор и эмбеддинги из pipeline)
query = "кнопка войти"
query_ids = pipeline.tokenizer(query, return_tensors="pt").input_ids.to(pipeline.device)
query_emb = pipeline.token_embedding(query_ids).mean(dim=1)  # Mean pooling

# Поиск по cosine similarity
# results = vector_db.search(query_emb.cpu().numpy(), top_k=5)
```

## ⚠️ Важные замечания

### Debug Decoding (LM Head)

**LM Head НЕ подходит для RAG!** Он обучен для автогрессивной генерации текста, а не для retrieval. Если видите пробелы/garbage в `decoded_embeddings.json` - это нормально.

**Для RAG используйте** `output_embeddings` напрямую из HeadlessLLM.

### Размерности

- **ViT Patches**: (H/14, W/14, 1280)
- **Spatial Merge**: (H/28, W/28, 3584) - 2×2 grouping + MLP
- **ROI Pooling**: (N, 3584) - mean pool per bbox
- **Final Embeddings**: (N, 3584) - after LLM processing

### Веса

Используются предобученные веса `Qwen/Qwen2.5-VL-7B-Instruct`:
- Vision Encoder (ViT): 32 слоя, 1280-dim
- Spatial Merge MLP: 5120 → 3584
- LLM: 28 слоёв, 3584-dim

## 📝 License

MIT