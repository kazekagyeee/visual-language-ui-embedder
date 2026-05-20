# Vision-Language Semantic UI Retrieval

Проект реализует систему семантического поиска элементов графического интерфейса
по текстовому запросу пользователя.

Тема диплома:

Использование Vision-Language моделей для семантического поиска
запрошенного фрагмента графического интерфейса по запросу.

---

# Идея проекта

Система анализирует PDF-документы с интерфейсами 1С,
выделяет UI-элементы и позволяет находить:

- кнопки,
- ссылки,
- пункты меню,
- поля,
- вкладки,
- надписи интерфейса

по естественному запросу пользователя.

Пример:

Запрос:
"где находятся ГОСТы и показатели контроля"

Результат:
- текстовый ответ,
- cropped interface view,
- выделенные UI-элементы.

---

# Архитектура

PDF
│
├── OCR (EasyOCR)
│
├── UI element extraction
│
├── UI crops
│
├── Siamese embeddings
│
├── Vector database
│
├── Hybrid retrieval
│
└── Streamlit visualization

---

# Pipeline

1. PDF indexing

Файл:
rag/build_pdf_rag.py

Создает:
- страницы PDF;
- текстовые блоки;
- embeddings.

---

2. UI extraction

Файл:
rag/build_ui_elements.py

EasyOCR выделяет:
- кнопки;
- гиперссылки;
- menu items;
- labels.

Создаются crop-изображения UI-элементов.

---

3. Siamese dataset

Файл:
training/make_ui_element_siamese_dataset.py

Создаются пары:

Positive:
"ГОСТы" ↔ crop ГОСТы

Negative:
"ГОСТы" ↔ crop другого элемента

---

4. Siamese training

Файл:
models/siamese_ui_encoder.py

Два encoder:

text encoder
image encoder

Они переводят:
- текст;
- изображения интерфейса

в общее embedding-пространство.

---

5. Vector DB

Файлы:
rag/ui_vector_db.py
retrieval/build_ui_vector_db.py

Создается локальная vector database:

vector_db/ui_elements/

Содержит:
- metadata;
- UI embeddings;
- OCR text;
- bbox.

---

6. Hybrid retrieval

Файл:
rag/ui_element_searcher.py

Комбинируются:
- semantic similarity;
- OCR similarity;
- siamese similarity.

---

7. Multi-query retrieval

Файл:
rag/multi_query.py

Запрос:
"ГОСТы и показатели контроля"

разбивается на:
- ГОСТы
- Показатели контроля

---

8. Visualization

Файл:
rag/streamlit_pdf_rag.py

Показывает:
- текстовый ответ;
- cropped interface;
- highlighted UI elements;
- debug info.

---

# Dataset

Каждый элемент датасета содержит:

- OCR text;
- UI crop;
- bbox;
- page;
- ui_type;
- label.

Пример:

{
  "text": "ГОСТы",
  "ui_type": "hyperlink",
  "page": 3,
  "bbox": [x0, y0, x1, y1]
}

---

# Benchmark

Файл:
evaluation/benchmark_ui_retrieval.py

Метрики:
- Top-1 accuracy
- Top-3 accuracy
- Top-k accuracy

---

# First run

python scripts/first_run_ui_pipeline.py

---

# Run Streamlit

python -m streamlit run rag/streamlit_pdf_rag.py

---

# Основная идея

Система переводит:
- текстовый запрос;
- изображение интерфейса

в общее embedding-пространство.

После этого поиск выполняется по близости векторов.

---

# Что умеет система

- semantic UI retrieval;
- OCR GUI parsing;
- multimodal retrieval;
- siamese retrieval;
- vector search;
- interface highlighting;
- cropped interface visualization.

---

# Соответствие теме диплома

Проект реализует:

- Vision-Language retrieval;
- semantic GUI search;
- OCR-based UI parsing;
- Siamese neural network;
- vector database;
- multimodal embeddings;
- semantic interface retrieval.
