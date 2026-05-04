# Short Siamese Cross-Modal RAG для visual-language-ui-embedder

Новая ветка расширяет текущий проект `visual-language-ui-embedder`: базовый Qwen2.5-VL pipeline оставляется как тяжёлый teacher / reranker, а поверх него добавляется лёгкий Siamese / Dual Encoder для быстрого сопоставления текста и UI-компонента/картинки.

## Идея

Текущий репозиторий генерирует контекстные эмбеддинги UI-компонентов через Qwen2.5-VL, UIED, ROI Pooling и Headless LLM. Новая часть решает задачу:

> по паре `текстовый запрос — изображение/компонент` получить короткие векторы и score схожести.

Qwen long embeddings используются как эталонные длинные признаки, а Siamese student учится сжимать их в короткое пространство.

## Что добавляется

```text
configs/short_siamese.yaml          # конфиг модели и путей
models/short_siamese_encoder.py     # лёгкий dual encoder + similarity head
training/pair_dataset.py            # dataset для пар text/image vector + label
training/train_short_siamese.py     # обучение BCE + contrastive loss
retrieval/build_short_index.py      # сборка FAISS/NumPy индекса эталонов
retrieval/search_short_index.py     # быстрый поиск по коротким векторам
retrieval/qwen_rerank_stub.py       # место для rerank через Qwen long vectors
rag/streamlit_app.py                # Streamlit RAG интерфейс
scripts/prepare_pairs_from_qwen.py  # генерация pair dataset из embeddings.json
scripts/run_train.sh                # запуск обучения
scripts/run_build_index.sh          # сборка индекса
scripts/run_streamlit.sh            # запуск RAG
requirements.short.txt              # дополнительные зависимости
```

## Формат данных

### `data/pairs.jsonl`

Каждая строка:

```json
{
  "id": "pair_001",
  "text": "кнопка входа",
  "text_vec": [0.1, 0.2],
  "image_vec": [0.3, 0.4],
  "qwen_long_vec": [0.01, 0.02],
  "label": 1,
  "meta": {"screen": "login.png", "bbox": [10, 20, 100, 60]}
}
```

`text_vec` и `image_vec` могут быть:

1. эмбеддингами из текущего Qwen pipeline;
2. заранее сохранёнными признаками из другой модели;
3. временными sentence/image embeddings для MVP.

### `data/reference_items.jsonl`

Эталонная база для RAG:

```json
{
  "id": "button_login_001",
  "title": "Login button",
  "text": "кнопка входа",
  "short_vec": [0.1, 0.2],
  "qwen_long_vec": [0.01, 0.02],
  "meta": {"screen": "login.png", "bbox": [10, 20, 100, 60]}
}
```

## Запуск

```bash
git checkout -b feature/siamese-short-crossmodal-embeddings
pip install -r requirements.txt
pip install -r requirements.short.txt
```

Обучение:

```bash
bash scripts/run_train.sh
```

Сборка индекса:

```bash
bash scripts/run_build_index.sh
```

RAG на Streamlit:

```bash
bash scripts/run_streamlit.sh
```

## Как это встраивается в старый проект

1. `main.py` генерирует Qwen embeddings в `output/embeddings.json`.
2. `scripts/prepare_pairs_from_qwen.py` превращает их в пары для обучения.
3. `training/train_short_siamese.py` обучает лёгкий encoder.
4. `retrieval/build_short_index.py` строит индекс коротких векторов.
5. `rag/streamlit_app.py` ищет релевантные UI-компоненты по тексту.
6. При необходимости найденные top-k кандидаты rerank-ятся Qwen long vectors.
