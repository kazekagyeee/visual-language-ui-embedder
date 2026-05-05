# Multimodal PDF RAG для инструкции 1С/ERP

Проект реализует прототип системы поиска и ответа по большой PDF-инструкции с визуальной привязкой к страницам документа.

Система умеет:

- извлекать текстовые блоки из PDF;
- рендерить страницы PDF в изображения;
- сохранять кропы найденных текстовых блоков;
- строить semantic-индекс по тексту;
- искать по смыслу через sentence-transformers;
- искать по точным словам через BM25;
- объединять semantic search и BM25 в hybrid search;
- подсвечивать найденный блок на полной странице PDF;
- генерировать ответ по найденному контексту через LLM;
- дополнительно искать визуально похожие блоки через CLIP.

---

## Структура проекта

```text
visual-language-ui-embedder/
│
├── data_source/
│   └── instruction.pdf
│
├── data/
│   └── pdf_rag/
│       ├── items.jsonl
│       ├── embeddings.npy
│       ├── clip_items.jsonl
│       ├── clip_embeddings.npy
│       ├── pages/
│       │   ├── page_0001.png
│       │   └── ...
│       └── crops/
│           ├── page_0001_block_000.png
│           └── ...
│
├── rag/
│   ├── build_pdf_rag.py
│   ├── search_pdf_rag.py
│   ├── hybrid_search.py
│   ├── answer_engine.py
│   ├── clip_search.py
│   ├── build_clip_image_index.py
│   ├── search_clip_image_index.py
│   └── streamlit_pdf_rag.py
│
├── requirements.final.txt
└── README_RAG_PROJECT.md
