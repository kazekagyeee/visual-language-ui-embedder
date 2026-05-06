# -*- coding: utf-8 -*-

from pathlib import Path
import re

import streamlit as st
from PIL import Image, ImageDraw

from rag.answer_engine import AnswerEngine
from rag.clip_search import ClipImageSearcher
from rag.hybrid_search import HybridSearcher
from rag.visual_search import VisualDescriptionSearcher
from rag.word_ui_siamese_search import WordUISiameseSearcher


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
def load_word_ui_siamese_searcher():
    return WordUISiameseSearcher()


@st.cache_resource
def load_answer_engine():
    return AnswerEngine()


def normalize_token(text):
    text = text.lower().replace("ё", "е")
    return re.sub(r"[^а-яa-z0-9]+", "", text)


def box_center(box):
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2, (y0 + y1) / 2


def is_inside_interface_zone(word_box, target_boxes):
    x0, y0, x1, y1 = word_box

    if y0 > 950:
        return False

    if not target_boxes:
        return True

    for target_box in target_boxes:
        tx0, ty0, tx1, ty1 = target_box

        zone = [
            max(0, tx0 - 450),
            max(0, ty0 - 180),
            tx1 + 520,
            ty1 + 180,
        ]

        zx0, zy0, zx1, zy1 = zone

        if x0 >= zx0 and y0 >= zy0 and x1 <= zx1 and y1 <= zy1:
            return True

    return False


def find_query_word_boxes(query, page_words, target_boxes):
    query_tokens = [
        normalize_token(t)
        for t in re.findall(r"[а-яёa-z0-9]+", query.lower())
    ]
    query_tokens = [t for t in query_tokens if len(t) >= 3]

    matched = []

    for word in page_words:
        word_text = normalize_token(word.get("text", ""))
        word_box = word.get("bbox_px")

        if not word_text or not word_box:
            continue

        if not is_inside_interface_zone(word_box, target_boxes):
            continue

        for token in query_tokens:
            if token in word_text or word_text in token:
                matched.append(word_box)
                break

    return matched


def draw_targets(page_path, boxes_px, query=None, page_words=None):
    img = Image.open(page_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # old red annotation boxes are ignored

    if query and page_words:
        word_boxes = find_query_word_boxes(query, page_words, boxes_px)

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
            f"[Страница {item['page']}, блок {item.get('block_id', '-')}]\n"
            f"{item['text']}"
        )

    return "\n\n---\n\n".join(parts)


def show_text_result_card(result, mode_name, query):
    item = result["item"]

    st.divider()
    st.markdown(
        f"### {mode_name}: страница {item['page']} · "
        f"блок {item.get('block_id', '-')} · score={result['score']:.4f}"
    )

    if "dense_score" in result:
        st.caption(
            f"semantic={result['dense_score']:.4f} · bm25={result['bm25_score']:.4f}"
        )

    col1, col2 = st.columns([1.5, 1])

    with col1:
        page_path = Path(item["page_image"])

        if page_path.exists():
            highlighted = draw_targets(
                page_path=page_path,
                boxes_px=[],
                query=query,
                page_words=item.get("page_words", []),
            )

            st.image(
                highlighted,
                caption=f"Страница {item['page']}: зелёным выделены найденные UI-слова",
            )
        else:
            st.warning(f"Файл страницы не найден: {page_path}")

    with col2:
        st.markdown("#### Текст блока")
        st.write(item["text"])

        with st.expander("Технические данные"):
            st.json(item)


def show_visual_result_card(result):
    item = result["item"]

    st.divider()
    st.markdown(
        f"### Visual description: страница {item['page']} · "
        f"type={item['type']} · score={result['score']:.4f}"
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


def show_siamese_ui_results(query):
    st.subheader("Siamese UI search")

    try:
        word_searcher = load_word_ui_siamese_searcher()

        if word_searcher.model is None:
            st.warning(
                "Siamese UI модель или индекс не найдены. "
                "Сначала обучи модель и собери индекс."
            )
            return

        word_results = word_searcher.search(query, top_k=5)

        for result in word_results:
            item = result["item"]

            st.divider()
            st.markdown(
                f"### UI element: {item['text']} · "
                f"page={item['page']} · score={result['score']:.4f}"
            )

            col_a, col_b = st.columns([1, 2])

            with col_a:
                if Path(item["image"]).exists():
                    st.image(Image.open(item["image"]), caption="Найденный UI-кроп")

            with col_b:
                st.json(
                    {
                        "text": item["text"],
                        "page": item["page"],
                        "bbox_px": item["bbox_px"],
                        "image": item["image"],
                    }
                )

    except Exception as exc:
        st.warning(f"Siamese UI search не запустился: {exc}")


st.set_page_config(page_title="Multimodal PDF RAG", layout="wide")

st.title("Multimodal PDF RAG: текст + визуал + Siamese UI")
st.caption("Hybrid search + visual descriptions + word-level Siamese для UI-элементов.")

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
run_siamese_ui = st.checkbox("Добавить Siamese UI search", value=True)

if st.button("Искать"):
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
        show_text_result_card(result, "Text search", query)

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
                show_text_result_card(result, "CLIP image search", query)

        except Exception as exc:
            st.warning(f"CLIP-поиск не запустился: {exc}")

    if run_siamese_ui:
        show_siamese_ui_results(query)

    st.divider()
    st.subheader("Контекст")
    st.text_area("Контекст", context, height=320)



