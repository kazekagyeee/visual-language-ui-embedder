# -*- coding: utf-8 -*-

from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from rag.answer_engine import AnswerEngine
from rag.hybrid_search import HybridSearcher


ZOOM = 2.0


@st.cache_resource
def load_searcher():
    return HybridSearcher()


@st.cache_resource
def load_llm():
    return AnswerEngine()


def draw_bbox_on_page(page_path, bbox):
    img = Image.open(page_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    x0, y0, x1, y1 = bbox
    box = [x0 * ZOOM, y0 * ZOOM, x1 * ZOOM, y1 * ZOOM]

    for i in range(6):
        draw.rectangle(
            [box[0] - i, box[1] - i, box[2] + i, box[3] + i],
            outline="red",
        )

    return img


def build_context(results):
    parts = []

    for result in results:
        item = result["item"]
        parts.append(
            f"[Страница {item['page']}, блок {item['block_id']}]\n"
            f"{item['text']}"
        )

    return "\n\n---\n\n".join(parts)


st.set_page_config(page_title="PDF Hybrid RAG", layout="wide")

st.title("PDF Hybrid RAG: ответ + страницы с выделением")
st.caption("Hybrid search = semantic embeddings + BM25 по точным словам. Найденный блок подсвечивается на полной странице.")

query = st.text_input("Вопрос", "как создать заявку на контроль")

col_settings_1, col_settings_2, col_settings_3 = st.columns(3)

with col_settings_1:
    top_k = st.slider("Сколько блоков брать", 1, 12, 6)

with col_settings_2:
    alpha = st.slider(
        "Вес semantic search",
        0.0,
        1.0,
        0.65,
        0.05,
        help="0 = только BM25, 1 = только embeddings",
    )

with col_settings_3:
    generate_answer = st.checkbox("Генерировать ответ LLM", value=True)

if st.button("Искать"):
    searcher = load_searcher()
    results = searcher.search(query, top_k=top_k, alpha=alpha)

    context = build_context(results)

    if generate_answer:
        st.subheader("Ответ по инструкции")

        with st.spinner("Генерирую ответ по найденным страницам..."):
            llm = load_llm()
            answer = llm.generate(query, context)

        st.markdown(answer)

    st.subheader("Найденные источники")

    pages = sorted(set(result["item"]["page"] for result in results))
    st.info("Использованные страницы: " + ", ".join(map(str, pages)))

    for result in results:
        item = result["item"]

        st.divider()
        st.markdown(
            f"### Страница {item['page']} · блок {item['block_id']} · "
            f"hybrid={result['score']:.4f}"
        )

        st.caption(
            f"semantic={result['dense_score']:.4f} · "
            f"bm25={result['bm25_score']:.4f}"
        )

        col1, col2 = st.columns([1.5, 1])

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
                st.warning(f"Файл страницы не найден: {page_path}")

        with col2:
            st.markdown("#### Текст блока")
            st.write(item["text"])

            with st.expander("Технические данные"):
                st.json({
                    "page": item["page"],
                    "block_id": item["block_id"],
                    "hybrid_score": result["score"],
                    "semantic_score": result["dense_score"],
                    "bm25_score": result["bm25_score"],
                    "bbox": item["bbox"],
                    "page_image": item["page_image"],
                    "crop_image": item.get("crop_image"),
                })

    st.divider()
    st.subheader("Контекст, переданный в LLM")
    st.text_area("Контекст", context, height=320)
