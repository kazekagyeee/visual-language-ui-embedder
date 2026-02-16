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

### Запуск

```python
python main.py
```

**Входные данные:**
- `input_images/image_20_2.png` - изображение UI
- `input_images/image_20_2.txt` - текстовое описание

**Выходные данные:**
- `output/embeddings.json` - финальные эмбеддинги (N, 3584)
- `debug/uied_bbox_debug.png` - визуализация детекций
- `debug/decoded_embeddings.json` - debug декодинг (опционально)

### Конфигурация

В `main.py`:

```python
# Системный промпт
SYSTEM_PROMPT = "You are a UI context describer assistant."

# Контекстный промпт
CONTEXT_PROMPT = "Описание задачи..."

# Режим работы
use_learned_tokens = False  # True для обучаемых токенов
use_mrope = False           # True для M-RoPE (3D)

# Debug
DEBUG_DECODE_EMBEDDINGS = True  # Декодирование через LM Head
```

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
# Энкодим запрос
query = "кнопка войти"
query_ids = tokenizer(query, return_tensors="pt").input_ids
query_emb = token_embedding(query_ids).mean(dim=1)  # Mean pooling

# Поиск по cosine similarity
# results = vector_db.search(query_emb, top_k=5)
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