# -*- coding: utf-8 -*-

from pathlib import Path
import re

import streamlit as st
from PIL import Image, ImageDraw

from rag.answer_engine import AnswerEngine
from rag.hybrid_search import HybridSearcher
from rag.ui_element_searcher import UIElementSearcher


UI_PHRASES = [
    "ГОСТы",
    "Показатели контроля",
    "Виды контроля",
    "Группы прочности",
    "Входной контроль",
    "Заявки на контроль",
    "Выполнения входного контроля",
    "Акты входного контроля",
    "Создать",
    "Добавить",
    "Записать",
    "Записать и закрыть",
]


@st.cache_resource
def load_text_searcher(rag_dir):
    return HybridSearcher(rag_dir=rag_dir)


@st.cache_resource
def load_answer_engine():
    return AnswerEngine()


@st.cache_resource
def load_ui_searcher():
    return UIElementSearcher()


def norm(text):
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_ui_queries(query):
    q = norm(query)
    found = []

    for phrase in UI_PHRASES:
        p = norm(phrase)
        if p in q:
            found.append(phrase)

    if found:
        return found

    return [query]


def draw_bboxes(page_image, boxes):
    img = Image.open(page_image).convert("RGB")
    draw = ImageDraw.Draw(img)

    for bbox in boxes:
        x0, y0, x1, y1 = bbox
        for i in range(7):
            draw.rectangle(
                [x0 - i, y0 - i, x1 + i, y1 + i],
                outline="green",
            )

    return img


def build_context(results):
    parts = []
    seen = set()

    for result in results:
        item = result["item"]
        text = item["text"].strip()
        key = text.lower()

        if key in seen:
            continue

        seen.add(key)
        parts.append(f"[Страница {item['page']}, блок {item.get('block_id', '-')}]\n{text}")

    return "\n\n---\n\n".join(parts)


def filter_ui_results_by_text_pages(ui_results, text_results):
    text_pages = {r["item"]["page"] for r in text_results}
    preferred = [r for r in ui_results if r["item"]["page"] in text_pages]
    return preferred or ui_results


def get_best_ui_results_for_query(query, text_results, top_k_ui):
    ui_searcher = load_ui_searcher()

    if ui_searcher.model is None:
        return []

    subqueries = extract_ui_queries(query)
    collected = []
    seen = set()

    for subquery in subqueries:
        results = ui_searcher.search(subquery, top_k=top_k_ui * 5)
        results = filter_ui_results_by_text_pages(results, text_results)

        if not results:
            continue

        best = results[0]
        item = best["item"]

        key = (item["page"], tuple(item["bbox"]), norm(item["text"]))
        if key in seen:
            continue

        seen.add(key)
        best["matched_query"] = subquery
        collected.append(best)

    if len(collected) < top_k_ui and len(subqueries) == 1:
        extra = ui_searcher.search(query, top_k=top_k_ui * 3)
        extra = filter_ui_results_by_text_pages(extra, text_results)

        for result in extra:
            item = result["item"]
            key = (item["page"], tuple(item["bbox"]), norm(item["text"]))

            if key in seen:
                continue

            seen.add(key)
            result["matched_query"] = query
            collected.append(result)

            if len(collected) >= top_k_ui:
                break

    return collected[:top_k_ui]


def show_answer_with_ui(query, text_results, top_k_ui):
    context = build_context(text_results)

    st.subheader("Ответ по инструкции")
    answer_engine = load_answer_engine()
    st.markdown(answer_engine.generate(query, context))

    st.subheader("Что нажать / где смотреть в интерфейсе")

    try:
        ui_results = get_best_ui_results_for_query(query, text_results, top_k_ui)

        if not ui_results:
            st.warning("UI-элементы по этому вопросу не найдены.")
            return context

        grouped = {}

        for result in ui_results:
            item = result["item"]
            grouped.setdefault(item["page"], []).append(result)

        for page, results in grouped.items():
            first_item = results[0]["item"]
            boxes = [r["item"]["bbox"] for r in results]

            st.markdown(f"### Страница {page}: найдено элементов {len(results)}")

            col1, col2 = st.columns([1.5, 1])

            with col1:
                page_image = Path(first_item["page_image"])
                if page_image.exists():
                    highlighted = draw_bboxes(page_image, boxes)
                    st.image(highlighted, caption="Страница с выделенными UI-элементами")

            with col2:
                for result in results:
                    item = result["item"]

                    st.markdown(
                        f"**{result.get('matched_query', query)} → {item['text']}**  \n"
                        f"score={result['score']:.4f}, siamese={result['siamese_score']:.4f}"
                    )

                    crop = Path(item["crop_image"])
                    if crop.exists():
                        st.image(crop, caption="Кроп найденного элемента")

                    with st.expander(f"Технические данные: {item['text']}"):
                        st.json(
                            {
                                "matched_query": result.get("matched_query", query),
                                "text": item["text"],
                                "page": item["page"],
                                "bbox": item["bbox"],
                                "crop": item["crop_image"],
                                "score": result["score"],
                                "siamese_score": result["siamese_score"],
                            }
                        )

    except Exception as exc:
        st.warning(f"UI Element Search не запустился: {exc}")

    return context


def show_text_source(result):
    item = result["item"]

    st.divider()
    st.markdown(
        f"### Text source: страница {item['page']} · "
        f"блок {item.get('block_id', '-')} · score={result['score']:.4f}"
    )

    if "dense_score" in result:
        st.caption(f"semantic={result['dense_score']:.4f} · bm25={result['bm25_score']:.4f}")

    col1, col2 = st.columns([1.4, 1])

    with col1:
        page_image = Path(item["page_image"])
        if page_image.exists():
            st.image(Image.open(page_image), caption="Страница-источник")

    with col2:
        st.markdown("#### Текст блока")
        st.write(item["text"])

        with st.expander("Технические данные"):
            st.json(item)


st.set_page_config(page_title="PDF RAG + UI Element Search", layout="wide")

st.title("PDF RAG + UI Element Search")
st.caption("Отвечает по инструкции и сразу показывает конкретные кнопки/ссылки/надписи интерфейса 1С.")

rag_dir = st.selectbox("Индекс PDF", ["data/pdf_rag", "data/pdf_rag_reports"])

query = st.text_input("Вопрос", "где находятся ГОСТы и показатели контроля")

col1, col2, col3, col4 = st.columns(4)

with col1:
    top_k_text = st.slider("Текстовых источников", 1, 12, 5)

with col2:
    top_k_ui = st.slider("UI-элементов", 1, 20, 5)

with col3:
    alpha = st.slider("Вес semantic", 0.0, 1.0, 0.35, 0.05)

with col4:
    show_sources = st.checkbox("Показать текстовые источники", value=True)

if st.button("Искать"):
    rag_path = Path(rag_dir)

    if not (rag_path / "items.jsonl").exists():
        st.error(f"Не найден текстовый индекс: {rag_path / 'items.jsonl'}")
        st.stop()

    text_searcher = load_text_searcher(str(rag_path))
    text_results = text_searcher.search(query, top_k=top_k_text, alpha=alpha)

    context = show_answer_with_ui(query=query, text_results=text_results, top_k_ui=top_k_ui)

    if show_sources:
        st.subheader("Текстовые источники")

        pages = sorted(set(r["item"]["page"] for r in text_results))
        st.info("Использованные страницы: " + ", ".join(map(str, pages)))

        for result in text_results:
            show_text_source(result)

    st.divider()
    st.subheader("Контекст для ответа")
    st.text_area("Контекст", context, height=320)
