# -*- coding: utf-8 -*-

from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from rag.answer_engine import AnswerEngine
from rag.clip_search import ClipImageSearcher
from rag.hybrid_search import HybridSearcher


@st.cache_resource
def load_text_searcher():
    return HybridSearcher()


@st.cache_resource
def load_clip_searcher():
    return ClipImageSearcher()


@st.cache_resource
def load_answer_engine():
    return AnswerEngine()


def draw_ui_targets(page_path, red_boxes_px):
    img = Image.open(page_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    if red_boxes_px:
        for box in red_boxes_px:
            x0, y0, x1, y1 = box

            for i in range(5):
                draw.rectangle(
                    [x0 - i, y0 - i, x1 + i, y1 + i],
                    outline="blue",
                )

    return img


def build_context(results):
    parts = []
    seen = set()

    for result in results:
        item = result["item"]
        key = item["text"].strip().lower()

        if key in seen:
            continue

        seen.add(key)

        parts.append(
            f"[Страница {item['page']}, блок {item['block_id']}]\n"
            f"{item['text']}"
        )

    return "\n\n---\n\n".join(parts)


def show_result_card(result, mode_name):
    item = result["item"]

    st.divider()
    st.markdown(
        f"### {mode_name}: страница {item['page']} · блок {item['block_id']} · score={result['score']:.4f}"
    )

    if "dense_score" in result:
        st.caption(
            f"semantic={result['dense_score']:.4f} · bm25={result['bm25_score']:.4f}"
        )

    col1, col2 = st.columns([1.5, 1])

    with col1:
        page_path = Path(item["page_image"])

        if page_path.exists():
            highlighted = draw_ui_targets(page_path, item.get("red_bboxes_px", []))
            st.image(
                highlighted,
                caption=f"Страница {item['page']}: найденные UI-элементы выделены синим",
            )
        else:
            st.warning(f"Файл страницы не найден: {page_path}")

    with col2:
        st.markdown("#### Текст блока")
        st.write(item["text"])

        target_crop = item.get("target_crop_image")
        if target_crop and Path(target_crop).exists():
            with st.expander("Показать найденный UI-элемент"):
                st.image(Image.open(target_crop))

        with st.expander("Технические данные"):
            st.json({
                "page": item["page"],
                "block_id": item["block_id"],
                "score": result["score"],
                "red_bboxes_px": item.get("red_bboxes_px", []),
                "target_crop_image": item.get("target_crop_image"),
                "page_image": item["page_image"],
            })


st.set_page_config(page_title="Multimodal PDF RAG", layout="wide")

st.title("Multimodal PDF RAG: текст + визуал")
st.caption("Текст ищется через Hybrid Search. Визуальные элементы берутся из красных аннотаций PDF и подсвечиваются синим.")

query = st.text_input("Вопрос", "как создать заявку на контроль")

col1, col2, col3, col4 = st.columns(4)

with col1:
    top_k_text = st.slider("Текстовых блоков", 1, 12, 6)

with col2:
    top_k_image = st.slider("Визуальных блоков", 0, 10, 4)

with col3:
    alpha = st.slider("Вес semantic", 0.0, 1.0, 0.35, 0.05)

with col4:
    generate_answer = st.checkbox("Ответ", value=True)

run_clip = st.checkbox("Добавить CLIP-поиск по изображениям", value=False)

if st.button("Искать"):
    text_searcher = load_text_searcher()
    text_results = text_searcher.search(query, top_k=top_k_text, alpha=alpha)

    context = build_context(text_results)

    if generate_answer:
        st.subheader("Ответ по инструкции")
        answer_engine = load_answer_engine()
        st.markdown(answer_engine.generate(query, context))

    st.subheader("Текстовые источники")

    pages = sorted(set(result["item"]["page"] for result in text_results))
    st.info("Использованные страницы: " + ", ".join(map(str, pages)))

    for result in text_results:
        show_result_card(result, "Text search")

    if run_clip and top_k_image > 0:
        st.subheader("Визуальные совпадения CLIP")

        try:
            clip_searcher = load_clip_searcher()

            if clip_searcher.embeddings is None:
                with st.spinner("Собираю CLIP-индекс..."):
                    count = clip_searcher.build_index()
                st.success(f"CLIP-индекс собран. Кропов: {count}")

            image_results = clip_searcher.search(query, top_k=top_k_image)

            for result in image_results:
                show_result_card(result, "CLIP image search")

        except Exception as exc:
            st.warning(f"CLIP-поиск не запустился: {exc}")

    st.divider()
    st.subheader("Контекст")
    st.text_area("Контекст", context, height=320)
