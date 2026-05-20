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

Архитектура

Общий pipeline:

PDF documents
        ↓
Text and image extraction
        ↓
OCR and native image extraction
        ↓
UI element indexing
        ↓
Text and UI embeddings
        ↓
Semantic retrieval
        ↓
Domain dictionary for 1C terminology
        ↓
Siamese reranking
        ↓
Final UI filtering
        ↓
Visual grounding in Streamlit
Основные компоненты
Компонент	Назначение
PDF RAG index	Индекс текстовых фрагментов PDF
UI index	Индекс элементов интерфейса
OCR pipeline	Извлечение текстовых областей из скриншотов
Domain dictionary	Нормализация терминов 1С и OCR-ошибок
Siamese ranker	Семантическое сопоставление запроса и GUI-элемента
Final UI filter	Удаление OCR-мусора и дубликатов
Streamlit app	Пользовательский интерфейс системы
Структура проекта
visual-language-ui-embedder/
├── configs/
├── data/
│   ├── all_pdf_rag/
│   ├── ui_index/
│   ├── ui_trained_index/
│   └── ui_training_pairs.jsonl
├── evaluation/
│   ├── evaluate_ui_retrieval.py
│   ├── evaluate_ablation.py
│   ├── dataset_statistics.py
│   ├── user_scenario_eval.py
│   └── plot_final_reports.py
├── rag/
│   ├── streamlit_pdf_rag.py
│   ├── answer_engine.py
│   ├── domain_1c_dictionary.py
│   ├── domain_response.py
│   └── final_ui_filter.py
├── retrieval/
│   └── build_trained_ui_index.py
├── scripts/
│   ├── build_ui_training_pairs.py
│   ├── debug_ui_query.py
│   ├── final_smoke_test.py
│   └── inspect_ui_index.py
├── training/
│   ├── train_ui_siamese.py
│   └── ui_pair_dataset.py
├── checkpoints/
└── reports/
Используемые данные

В качестве источников используются PDF-документы с инструкциями и методическими материалами по 1С. Из документов извлекаются:

текстовые фрагменты;
страницы PDF;
скриншоты интерфейсов;
OCR-блоки;
GUI-элементы;
bounding boxes;
embedding-представления;
обучающие пары для Siamese-модели.
Статистика датасета
Показатель	Значение
PDF documents	3
Text chunks	1380
UI elements	50104
Screenshots	1443
Pages with text	881
Pages with UI	696
Распределение UI-элементов
Тип	Количество
text	30691
merged_text	13443
menu_item	3618
button	1128
native_ocr	1131
small_text	93
Обучение Siamese-модели

Для улучшения качества semantic matching обучается Siamese-ranker на парах:

query ↔ positive UI element
query ↔ negative UI element

Пример запуска:

py -m training.train_ui_siamese --pairs data/ui_training_pairs.jsonl --out checkpoints/ui_siamese_ranker.pt --epochs 25

Полученное качество обучения:

Метрика	Значение
Best validation accuracy	0.8864
Final train loss	0.1748
Final validation accuracy	0.8604
Epochs	25
Запуск системы
1. Построение PDF RAG индекса
py -m rag.build_pdf_rag --pdf data_source/instruction.pdf --out data/all_pdf_rag
2. Построение обучающих пар
py -m scripts.build_ui_training_pairs --ui-index-dir data/ui_index --out data/ui_training_pairs.jsonl
3. Обучение Siamese-модели
py -m training.train_ui_siamese --pairs data/ui_training_pairs.jsonl --out checkpoints/ui_siamese_ranker.pt --epochs 25
4. Построение trained UI index
py -m retrieval.build_trained_ui_index --ui-index-dir data/ui_index --checkpoint checkpoints/ui_siamese_ranker.pt --out-dir data/ui_trained_index
5. Запуск Streamlit
py -m streamlit run rag/streamlit_pdf_rag.py
Отладка запросов
py -m scripts.debug_ui_query --query "как создать нового контрагента"
py -m scripts.debug_ui_query --query "как открыть показатели контроля"
py -m scripts.debug_ui_query --query "где найти монитор интернет поддержки"
Оценка качества

Основная оценка retrieval:

py -m evaluation.evaluate_ui_retrieval

Пользовательские сценарии:

py -m evaluation.user_scenario_eval

Ablation study:

py -m evaluation.evaluate_ablation

Статистика датасета:

py -m evaluation.dataset_statistics

Построение графиков:

py -m evaluation.plot_final_reports
Актуальные результаты
UI Retrieval
Метрика	Значение
Success Rate	0.9167
Mean Precision	0.6361
Mean Recall	0.9792
Mean F1	0.7185
Mean MRR	0.6250
Hit@1	0.4167
Hit@3	0.8333
Hit@5	1.0000
PDF Accuracy	1.0000
User Scenario Evaluation
Метрика	Значение
Success Rate	0.8571
Mean Precision	0.6750
Mean Recall	0.9583
Mean F1	0.7417
Вывод

Разработанная система выполняет мультимодальный семантический поиск элементов графического интерфейса по пользовательскому текстовому запросу. В отличие от обычного поиска по PDF, система не только находит релевантный текстовый фрагмент, но и выполняет visual grounding результата в интерфейсе.

Основным преимуществом является высокая полнота поиска: система в большинстве случаев находит релевантные GUI-элементы даже при наличии OCR-искажений. Основным ограничением остается снижение precision из-за шумных OCR-фрагментов и дублирующихся merged_text элементов.

Проект соответствует задаче разработки алгоритмов интеллектуальной системы визуально-лингвистического поиска в графических интерфейсах.