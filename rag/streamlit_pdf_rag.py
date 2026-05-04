# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer


RAG_DIR = Path("data/pdf_rag")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_data
def load_index():
    items = []
    with open(RAG_DIR / "items.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    embeddings = np.load(RAG_DIR / "embeddings.npy")
    return items, embeddings


def search(query, top_k):
    model = load_model()
    items, embeddings = load_index()

    query_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_vec

    top_ids = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[idx]), items[int(idx)]) for idx in top_ids]


st.set_page_config(page_title="PDF RAG + Crops", layout="wide")

st.title("PDF RAG + проверка кропов")
st.caption("Поиск идет по текстовым блокам PDF. Для каждого найденного блока показывается кроп изображения.")

query = st.text_input("Вопрос", "как создать заявку на контроль")
top_k = st.slider("Сколько результатов показать", 1, 10, 5)

if st.button("Искать"):
    results = search(query, top_k)

    context_parts = []

    for score, item in results:
        st.divider()
        st.subheader(f"Страница {item['page']} · блок {item['block_id']} · score={score:.4f}")

        col1, col2 = st.columns([1, 2])

        with col1:
            crop_path = Path(item["crop_image"])
            page_path = Path(item["page_image"])

            if crop_path.exists():
                st.image(Image.open(crop_path), caption="Кроп найденного блока", use_column_width=True)
            else:
                st.warning(f"Кроп не найден: {crop_path}")

            if page_path.exists():
                with st.expander("Показать всю страницу"):
                    st.image(Image.open(page_path), caption=f"Страница {item['page']}", use_column_width=True)

        with col2:
            st.write(item["text"])

        context_parts.append(f"[Страница {item['page']}]\n{item['text']}")

    st.divider()
    st.subheader("Контекст для RAG")
    st.text_area("Найденный контекст", "\n\n---\n\n".join(context_parts), height=350)
