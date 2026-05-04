# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from sentence_transformers import SentenceTransformer


RAG_DIR = Path("data/pdf_rag")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ZOOM = 2.0


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


def draw_bbox_on_page(page_path, bbox, zoom=ZOOM):
    img = Image.open(page_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    x0, y0, x1, y1 = bbox
    box = [x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom]

    for i in range(6):
        draw.rectangle(
            [box[0] - i, box[1] - i, box[2] + i, box[3] + i],
            outline="red",
        )

    return img


def search(query, top_k):
    model = load_model()
    items, embeddings = load_index()

    query_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_vec

    top_ids = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[idx]), items[int(idx)]) for idx in top_ids]


st.set_page_config(page_title="PDF RAG + Highlight", layout="wide")

st.title("PDF RAG + выделение найденного блока")
st.caption("Поиск идет по текстовым блокам PDF. Для каждого результата показывается вся страница с выделенным найденным блоком.")

query = st.text_input("Вопрос", "как создать заявку на контроль")
top_k = st.slider("Сколько результатов показать", 1, 10, 5)

if st.button("Искать"):
    results = search(query, top_k)
    context_parts = []

    for score, item in results:
        st.divider()
        st.subheader(f"Страница {item['page']} · блок {item['block_id']} · score={score:.4f}")

        col1, col2 = st.columns([1.4, 1])

        with col1:
            page_path = Path(item["page_image"])

            if page_path.exists():
                highlighted = draw_bbox_on_page(page_path, item["bbox"])
                st.image(
                    highlighted,
                    caption=f"Страница {item['page']} с выделенным найденным блоком",
                    use_column_width=True,
                )
            else:
                st.warning(f"Страница не найдена: {page_path}")

        with col2:
            st.markdown("### Найденный текст")
            st.write(item["text"])

            with st.expander("Технические данные"):
                st.json({
                    "page": item["page"],
                    "block_id": item["block_id"],
                    "score": score,
                    "bbox": item["bbox"],
                    "page_image": item["page_image"],
                })

        context_parts.append(f"[Страница {item['page']}]\n{item['text']}")

    st.divider()
    st.subheader("Контекст для RAG")
    st.text_area("Найденный контекст", "\n\n---\n\n".join(context_parts), height=350)
