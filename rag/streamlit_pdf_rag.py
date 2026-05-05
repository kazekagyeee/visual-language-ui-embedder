# -*- coding: utf-8 -*-

from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from rag.answer_engine import AnswerEngine
from rag.clip_search import ClipImageSearcher
from rag.hybrid_search import HybridSearcher
from rag.visual_search import VisualDescriptionSearcher


@st.cache_resource
def load_text_searcher(rag_dir):
    return HybridSearcher(rag_dir=rag_dir)


@st.cache_resource
def load_clip_searcher(rag_dir):
    return ClipImageSearcher(rag_dir=rag_dir)


@st.cache_resource
def load_visual_searcher(rag_dir):
    return VisualDescriptionSearcher(rag_dir=rag_dir)


@st.cache_resource
def load_answer_engine():
    return AnswerEngine()


def normalize_token(text):
    import re
    text = text.lower().replace("ё", "е")
    return re.sub(r"[^а-яa-z0-9]+", "", text)


def find_query_word_boxes(query, page_words):
    import re

    query_tokens = [
        normalize_token(t)
        for t in re.findall(r"[а-яёa-z0-9]+", query.lower())
    ]

    query_tokens = [t for t in query_tokens if len(t) >= 3]

    matched = []

    for word in page_words:
        word_text = normalize_token(word.get("text", ""))

        if len(word_text) < 3:
            continue

        for token in query_tokens:
            if token in word_text or word_text in token:
                matched.append(word["bbox_px"])
                break

    return matched


def draw_targets(page_path, boxes_px, query=None, page_words=None):
    img = Image.open(page_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Синим — заранее найденные/размеченные визуальные области
    for box in boxes_px:
        x0, y0, x1, y1 = box

        for i in range(3):
            draw.rectangle(
                [x0 - i, y0 - i, x1 + i, y1 + i],
                outline="blue",
            )

    # Зеленым — конкретные слова интерфейса, совпавшие с запросом
    if query and page_words:
        word_boxes = find_query_word_boxes(query, page_words)

        for box in word_boxes:
            x0, y0, x1, y1 = box

            pad = 4
            for i in range(4):
                draw.rectangle(
                    [x0 - pad - i, y0 - pad - i, x1 + pad + i, y1 + pad + i],
                    outline="green",
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
            f"[Страница {item['page']}, блок {item.get('block_id', '-') }]\n"
            f"{item['text']}"
        )

    return "\n\n---\n\n".join(parts)


def show_text_result_card(result, mode_name):
    item = result["item"]

    st.divider()
    st.markdown(
        f"### {mode_name}: страница {item['page']} · блок {item.get('block_id', '-')} · score={result['score']:.4f}"
    )

    if "dense_score" in result:
        st.caption(
            f"semantic={result['dense_score']:.4f} · bm25={result['bm25_score']:.4f}"
        )

    col1, col2 = st.columns([1.5, 1])

    with col1:
        page_path = Path(item["page_image"])

        if page_path.exists():
            highlighted = draw_targets(page_path, item.get("target_bboxes_px", []), query=st.session_state.get("current_query"), page_words=item.get("page_words", []))
            st.image(
                highlighted,
                caption=f"Страница {item['page']}: найденные визуальные области выделены синим",
            )
        else:
            st.warning(f"Файл страницы не найден: {page_path}")

    with col2:
        st.markdown("#### Текст блока")
        st.write(item["text"])

        target_crops = item.get("target_crop_images", [])

        if target_crops:
            with st.expander("Показать найденные визуальные элементы"):
                for crop in target_crops[:5]:
                    crop_path = Path(crop)
                    if crop_path.exists():
                        st.image(Image.open(crop_path))

        with st.expander("Технические данные"):
            st.json(item)


def show_visual_result_card(result):
    item = result["item"]

    st.divider()
    st.markdown(
        f"### Visual description: страница {item['page']} · type={item['type']} · score={result['score']:.4f}"
    )

    col1, col2 = st.columns([1.3, 1])

    with col1:
        crop = item.get("target_crop_image")
        page_image = item.get("page_image")

        if crop and Path(crop).exists():
            st.image(Image.open(crop), caption="Найденный визуальный фрагмент")
        elif page_image and Path(page_image).exists():
            st.image(Image.open(page_image), caption="Найденная страница")
        else:
            st.warning("Изображение не найдено")

    with col2:
        st.markdown("#### Описание")
        st.write(item["text"])

        with st.expander("Технические данные"):
            st.json(item)


st.set_page_config(page_title="Multimodal PDF RAG", layout="wide")

st.title("Multimodal PDF RAG: текст + визуальные описания")
st.caption("Текстовый поиск + поиск по описаниям страниц и UI-кропов.")

rag_choice = st.selectbox(
    "Индекс PDF",
    [
        "data/pdf_rag",
        "data/pdf_rag_reports",
    ],
)

query = st.text_input("Вопрос", "где находятся ГОСТы и показатели контроля")

col1, col2, col3, col4 = st.columns(4)

with col1:
    top_k_text = st.slider("Текстовых блоков", 1, 12, 6)

with col2:
    top_k_visual = st.slider("Визуальных описаний", 0, 10, 5)

with col3:
    alpha = st.slider("Вес semantic", 0.0, 1.0, 0.35, 0.05)

with col4:
    generate_answer = st.checkbox("Ответ", value=True)

run_visual = st.checkbox("Искать по описаниям картинок и кропов", value=True)
run_clip = st.checkbox("Добавить CLIP-поиск по изображениям", value=False)

if st.button("Искать"):
    st.session_state["current_query"] = query
    rag_dir = Path(rag_choice)

    if not (rag_dir / "items.jsonl").exists():
        st.error(f"Индекс не найден: {rag_dir}")
        st.stop()

    text_searcher = load_text_searcher(str(rag_dir))
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
        show_text_result_card(result, "Text search")

    if run_visual and top_k_visual > 0:
        st.subheader("Поиск по визуальным описаниям")

        try:
            visual_searcher = load_visual_searcher(str(rag_dir))

            if visual_searcher.embeddings is None:
                with st.spinner("Собираю индекс визуальных описаний..."):
                    count = visual_searcher.build_index()
                st.success(f"Индекс визуальных описаний собран. Элементов: {count}")

            visual_results = visual_searcher.search(query, top_k=top_k_visual)

            for result in visual_results:
                show_visual_result_card(result)

        except Exception as exc:
            st.warning(f"Поиск по визуальным описаниям не запустился: {exc}")

    if run_clip:
        st.subheader("CLIP image search")

        try:
            clip_searcher = load_clip_searcher(str(rag_dir))

            if clip_searcher.embeddings is None:
                with st.spinner("Собираю CLIP-индекс..."):
                    count = clip_searcher.build_index()
                st.success(f"CLIP-индекс собран. Кропов: {count}")

            image_results = clip_searcher.search(query, top_k=top_k_visual)

            for result in image_results:
                show_text_result_card(result, "CLIP image search")

        except Exception as exc:
            st.warning(f"CLIP-поиск не запустился: {exc}")

    st.divider()
    st.subheader("Контекст")
    st.text_area("Контекст", context, height=320)



