# -*- coding: utf-8 -*-

from pathlib import Path

import streamlit as st
from PIL import Image

from rag.answer_engine import AnswerEngine
from rag.hybrid_search import HybridSearcher
from rag.interface_crop import make_interface_crop, draw_boxes
from rag.multi_query import split_query_to_ui_phrases, group_results_by_best_page
from rag.ui_element_searcher import UIElementSearcher
from rag.explainability import draw_similarity_boxes


@st.cache_resource
def load_text_searcher(rag_dir):
    return HybridSearcher(rag_dir=rag_dir)


@st.cache_resource
def load_answer_engine():
    return AnswerEngine()


@st.cache_resource
def load_ui_searcher():
    return UIElementSearcher()


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


def get_known_ui_phrases(ui_searcher):
    if not ui_searcher.items:
        return []

    return sorted({item["text"] for item in ui_searcher.items})


def search_ui_many(query, text_results, max_total=12):
    """
    Ищем не несколько случайных кандидатов на одну фразу,
    а один лучший UI-элемент на каждую найденную UI-фразу.

    Пример:
    "где находятся ГОСТы и показатели контроля"
    ->
    "ГОСТы"
    "Показатели контроля"

    На выходе:
    максимум по одному лучшему элементу на каждую фразу.
    """
    ui_searcher = load_ui_searcher()

    if ui_searcher.model is None:
        return []

    known_phrases = get_known_ui_phrases(ui_searcher)
    subqueries = split_query_to_ui_phrases(query, known_phrases)

    text_pages = {r["item"]["page"] for r in text_results}

    all_results = []
    seen_phrases = set()

    for subquery in subqueries:
        results = ui_searcher.search(subquery, top_k=30)

        # Сначала предпочитаем страницы, которые нашел текстовый RAG.
        preferred = [r for r in results if r["item"]["page"] in text_pages]

        if preferred:
            results = preferred

        if not results:
            continue

        best = results[0]
        best["matched_query"] = subquery

        phrase_key = subquery.lower().replace("ё", "е").strip()

        if phrase_key in seen_phrases:
            continue

        seen_phrases.add(phrase_key)
        all_results.append(best)

    # Если нашли несколько элементов на разных страницах,
    # оставляем лучшую страницу, где покрыто максимум запрошенных фраз.
    best_page_results = group_results_by_best_page(all_results)

    return best_page_results[:max_total]


def show_answer_and_ui(query, text_results, top_k_ui):
    context = build_context(text_results)

    st.subheader("Ответ по инструкции")

    answer_engine = load_answer_engine()
    st.markdown(answer_engine.generate(query, context))

    st.subheader("Что нажать / где смотреть в интерфейсе")

    ui_results = search_ui_many(
        query=query,
        text_results=text_results,
        max_total=top_k_ui,
    )

    if not ui_results:
        st.warning("UI-элементы по этому вопросу не найдены.")
        return context, []

    first_item = ui_results[0]["item"]
    page_image = Path(first_item["page_image"])
    boxes = [r["item"]["bbox"] for r in ui_results]

    col1, col2 = st.columns([1.5, 1])

    with col1:
        if page_image.exists():
            interface_crop, shifted_boxes = make_interface_crop(
                page_image,
                boxes,
                pad=280,
            )

            highlighted = draw_boxes(interface_crop, shifted_boxes)

            st.image(
                highlighted,
                caption="Обрезанный интерфейс с выделенными найденными элементами",
            )
        else:
            st.warning(f"Не найдена картинка страницы: {page_image}")

    with col2:
        for result in ui_results:
            item = result["item"]

            st.markdown(
                f"**[{result.get('matched_query', query)}] → {item['text']}**  \n"
                f"type={item.get('ui_type', 'unknown')}  \n"
                f"score={result.get('final_score', result.get('score', 0.0)):.4f}, "
                f"siamese={result.get('siamese_score', 0.0):.4f}"
            )

            crop = Path(item["crop_image"])
            if crop.exists():
                st.image(crop, caption="Кроп найденного элемента")

            with st.expander(f"Технические данные: {item['text']}"):
                st.json(
                    {
                        "matched_query": result.get("matched_query", query),
                        "text": item.get("text"),
                        "ui_type": item.get("ui_type"),
                        "page": item.get("page"),
                        "bbox": item.get("bbox"),
                        "crop": item.get("crop_image"),
                        "score": result.get("score"),
                        "raw_score": result.get("raw_score"),
                        "final_score": result.get("final_score"),
                        "siamese_score": result.get("siamese_score"),
                        "exact_bonus": result.get("exact_bonus"),
                        "token_score": result.get("token_score"),
                        "type_bonus": result.get("type_bonus"),
                        "length_penalty": result.get("length_penalty"),
                    }
                )

    with st.expander("Debug: полная страница с выделением"):
        if page_image.exists():
            full = draw_boxes(Image.open(page_image), boxes)
            st.image(full, caption="Полная страница для проверки bbox")

    with st.expander("Debug: attention / similarity map"):
        if page_image.exists():
            attention = draw_similarity_boxes(
                page_image,
                ui_results,
                max_items=12,
            )
            st.image(attention, caption="Similarity map по найденным UI-кандидатам")

    with st.expander("Debug: raw UI results"):
        st.json(
            [
                {
                    "matched_query": r.get("matched_query", query),
                    "text": r["item"].get("text"),
                    "ui_type": r["item"].get("ui_type"),
                    "page": r["item"].get("page"),
                    "bbox": r["item"].get("bbox"),
                    "score": r.get("score"),
                    "raw_score": r.get("raw_score"),
                    "final_score": r.get("final_score"),
                    "siamese_score": r.get("siamese_score"),
                    "exact_bonus": r.get("exact_bonus"),
                    "token_score": r.get("token_score"),
                    "type_bonus": r.get("type_bonus"),
                    "length_penalty": r.get("length_penalty"),
                    "crop": r["item"].get("crop_image"),
                }
                for r in ui_results
            ]
        )

    return context, ui_results


def show_text_source(result):
    item = result["item"]

    st.divider()

    st.markdown(
        f"### Text source: страница {item['page']} · "
        f"блок {item.get('block_id', '-')} · score={result['score']:.4f}"
    )

    if "dense_score" in result:
        st.caption(
            f"semantic={result['dense_score']:.4f} · "
            f"bm25={result['bm25_score']:.4f}"
        )

    col1, col2 = st.columns([1.4, 1])

    with col1:
        page_image = Path(item["page_image"])

        if page_image.exists():
            st.image(Image.open(page_image), caption="Страница-источник")
        else:
            st.warning(f"Не найдена страница: {page_image}")

    with col2:
        st.markdown("#### Текст блока")
        st.write(item["text"])

        with st.expander("Технические данные"):
            st.json(item)


st.set_page_config(page_title="PDF RAG + UI Element Search", layout="wide")

st.title("PDF RAG + UI Element Search")
st.caption("Ответ по инструкции + поиск конкретных кнопок, ссылок и надписей интерфейса 1С.")

rag_dir = st.selectbox(
    "Индекс PDF",
    [
        "data/pdf_rag",
        "data/pdf_rag_reports",
    ],
)

query = st.text_input(
    "Вопрос",
    "где находятся ГОСТы и показатели контроля",
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    top_k_text = st.slider("Текстовых источников", 1, 12, 5)

with col2:
    top_k_ui = st.slider("UI-элементов", 1, 20, 8)

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
    text_results = text_searcher.search(
        query,
        top_k=top_k_text,
        alpha=alpha,
    )

    context, ui_results = show_answer_and_ui(
        query=query,
        text_results=text_results,
        top_k_ui=top_k_ui,
    )

    if show_sources:
        st.subheader("Текстовые источники")

        pages = sorted(set(r["item"]["page"] for r in text_results))
        st.info("Использованные страницы: " + ", ".join(map(str, pages)))

        for result in text_results:
            show_text_source(result)

    st.divider()
    st.subheader("Контекст для ответа")
    st.text_area("Контекст", context, height=320)
