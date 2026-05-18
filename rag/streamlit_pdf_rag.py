# -*- coding: utf-8 -*-

from pathlib import Path

import streamlit as st
from PIL import Image

from rag.answer_engine import AnswerEngine
from rag.hybrid_search import HybridSearcher
from rag.multi_query import split_query_to_ui_phrases
from rag.ui_element_searcher import UIElementSearcher
from rag.explainability import draw_similarity_boxes
from rag.ocr_cleaning import normalize_ocr_text
from rag.ui_visualization import make_ui_focus_image, make_full_debug_image


@st.cache_resource
def load_text_searcher(rag_dir):
    return HybridSearcher(rag_dir=rag_dir)


@st.cache_resource
def load_answer_engine():
    return AnswerEngine()


@st.cache_resource
def load_ui_searcher(checkpoint, index_dir):
    return UIElementSearcher(
        checkpoint=checkpoint,
        index_dir=index_dir,
    )


def build_context(results):
    parts = []
    seen = set()

    for result in results:
        item = result["item"]
        text = str(item.get("text", "")).strip()
        key = (item.get("page"), text.lower())

        if key in seen:
            continue

        seen.add(key)
        parts.append(
            f"[Страница {item.get('page')}, блок {item.get('block_id', item.get('block', '-'))}]\n{text}"
        )

    return "\n\n---\n\n".join(parts)


def get_known_ui_phrases(ui_searcher):
    if not getattr(ui_searcher, "items", None):
        return []

    return sorted({item.get("text", "") for item in ui_searcher.items if item.get("text")})


def item_norm(item):
    return item.get("normalized_text") or normalize_ocr_text(item.get("text", ""))


def clean_target_phrases(phrases):
    clean = []
    seen = set()

    for phrase in phrases:
        norm = normalize_ocr_text(phrase)

        if not norm:
            continue

        if norm in seen:
            continue

        seen.add(norm)
        clean.append(phrase)

    return clean


def score_candidate_for_phrase(phrase, result):
    item = result["item"]

    phrase_norm = normalize_ocr_text(phrase)
    item_text_norm = item_norm(item)

    base = float(result.get("final_score", result.get("score", 0.0)))

    phrase_tokens = set(phrase_norm.split())
    item_tokens = set(item_text_norm.split())

    score = base

    if item_text_norm == phrase_norm:
        score += 100.0

    elif phrase_norm in item_text_norm:
        score += 40.0

    elif item_text_norm in phrase_norm:
        score += 15.0

    overlap = len(phrase_tokens & item_tokens)
    score += overlap * 5.0

    if phrase_tokens and phrase_tokens.issubset(item_tokens):
        score += 30.0

    if item.get("ui_type") == "button":
        score += 4.0
    elif item.get("ui_type") == "hyperlink":
        score += 3.0
    elif item.get("ui_type") in {"tab", "sidebar_item"}:
        score += 2.0

    # штраф за обрезки типа "интернет-поддержки"
    if len(item_tokens) < len(phrase_tokens):
        score -= 20.0

    if len(item_text_norm.split()) > 7:
        score -= 10.0

    return score


def get_candidates_for_phrase(ui_searcher, phrase, top_k=180):
    results = ui_searcher.search(phrase, top_k=top_k)

    phrase_norm = normalize_ocr_text(phrase)

    filtered = []

    for result in results:
        item = result["item"]
        item_text_norm = item_norm(item)
        zone = item.get("zone_bbox")
        bbox = item.get("bbox")

        if not bbox:
            continue

        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]

        # 1) отбрасываем большие OCR-абзацы
        if bw > 900 and bh > 80:
            continue

        # 2) отбрасываем длинные строки обычного текста
        if bw / max(1, bh) > 12:
            continue

        # 3) отбрасываем мусорные куски текста
        norm = item_text_norm

        if len(norm) < 4:
            continue

        if norm.endswith(":"):
            continue

        if len(norm.split()) == 1 and norm in {
            "пользователей",
            "пользователь",
            "интернет",
            "поддержки",
            "поддержка",
            "под",
        }:
            continue

        if zone and bbox:
            # если элемент слишком близко к верхней границе зоны,
            # это часто текст абзаца над скриншотом, а не UI
            if bbox[1] < zone[1] + 25:
                continue

            zone_h = max(1, zone[3] - zone[1])
            rel_y = (bbox[1] - zone[1]) / zone_h

            if rel_y < 0.08 and item.get("ui_type") not in {"button", "input", "tab", "header"}:
                continue

        phrase_tokens = set(phrase_norm.split())
        item_tokens = set(item_text_norm.split())

        if not (
                item_text_norm == phrase_norm
                or phrase_norm in item_text_norm
                or phrase_tokens.issubset(item_tokens)
        ):
            continue
            result = dict(result)
            result["matched_query"] = phrase
            result["_phrase_score"] = score_candidate_for_phrase(phrase, result)
            filtered.append(result)

    if filtered:
        filtered.sort(key=lambda r: r["_phrase_score"], reverse=True)
        return filtered

    fallback = []

    for result in results[:30]:
        result = dict(result)
        result["matched_query"] = phrase
        result["_phrase_score"] = score_candidate_for_phrase(phrase, result)
        fallback.append(result)

    fallback.sort(key=lambda r: r["_phrase_score"], reverse=True)

    return fallback


def choose_one_common_page(phrase_to_candidates, text_pages):
    page_map = {}

    for phrase, candidates in phrase_to_candidates.items():
        for result in candidates:
            item = result["item"]
            page = item.get("page")

            page_map.setdefault(page, {})
            page_map[page].setdefault(phrase, [])
            page_map[page][phrase].append(result)

    best_page = None
    best_key = None

    for page, phrase_results in page_map.items():
        coverage = len(phrase_results)
        text_page_bonus = 1 if page in text_pages else 0

        score_sum = 0.0

        for phrase, results in phrase_results.items():
            results.sort(key=lambda r: r["_phrase_score"], reverse=True)
            score_sum += float(results[0]["_phrase_score"])

        key = (coverage, text_page_bonus, score_sum)

        if best_key is None or key > best_key:
            best_key = key
            best_page = page

    if best_page is None:
        return []

    chosen = []

    for phrase, candidates in phrase_to_candidates.items():
        same_page = [
            r for r in candidates
            if r["item"].get("page") == best_page
        ]

        if same_page:
            same_page.sort(key=lambda r: r["_phrase_score"], reverse=True)
            chosen.append(same_page[0])

    return chosen


def search_ui_exactly_by_query(query, text_results, checkpoint, index_dir):
    ui_searcher = load_ui_searcher(
        checkpoint=checkpoint,
        index_dir=index_dir,
    )

    if getattr(ui_searcher, "model", None) is None:
        return [], []

    known_phrases = get_known_ui_phrases(ui_searcher)
    raw_targets = split_query_to_ui_phrases(query, known_phrases)
    target_phrases = clean_target_phrases(raw_targets)

    text_pages = {
        r["item"].get("page")
        for r in text_results
        if r.get("item")
    }

    phrase_to_candidates = {}

    for phrase in target_phrases:
        candidates = get_candidates_for_phrase(ui_searcher, phrase, top_k=180)

        if candidates:
            phrase_to_candidates[phrase] = candidates

    ui_results = choose_one_common_page(
        phrase_to_candidates=phrase_to_candidates,
        text_pages=text_pages,
    )

    return target_phrases, ui_results


def filter_text_results_by_ui(text_results, ui_results):
    if not ui_results:
        return text_results

    ui_pages = {
        r["item"].get("page")
        for r in ui_results
        if r.get("item")
    }

    filtered = [
        r for r in text_results
        if r["item"].get("page") in ui_pages
    ]

    return filtered if filtered else text_results[:2]


def show_answer_and_ui(query, text_results, top_k_ui, checkpoint, index_dir):
    target_phrases, ui_results = search_ui_exactly_by_query(
        query=query,
        text_results=text_results,
        checkpoint=checkpoint,
        index_dir=index_dir,
    )

    filtered_text_results = filter_text_results_by_ui(text_results, ui_results)
    context = build_context(filtered_text_results)

    st.subheader("Ответ по инструкции")

    answer_engine = load_answer_engine()
    raw_answer = answer_engine.generate(query, context)

    if ui_results:
        unique_steps = []
        seen = set()

        # сортируем:
        # кнопки > ссылки > вкладки > прочее
        type_priority = {
            "button": 0,
            "input": 1,
            "tab": 2,
            "sidebar_item": 3,
            "hyperlink": 4,
        }

        sorted_results = sorted(
            ui_results,
            key=lambda r: (
                type_priority.get(
                    r["item"].get("ui_type"),
                    99
                ),
                -float(r.get("_phrase_score", 0.0))
            )
        )

        for r in sorted_results:
            item = r["item"]

            text = str(item.get("text", "")).strip()
            norm = normalize_ocr_text(text)

            # мусор OCR
            if len(norm) < 4:
                continue

            # слишком короткое слово
            if len(norm.split()) == 1 and len(norm) < 10:
                continue

            # мусорные обрезки
            if norm.endswith(":"):
                continue

            page = item.get("page")
            ui_type = item.get("ui_type")

            key = (page, norm)

            if key in seen:
                continue

            seen.add(key)

            unique_steps.append({
                "page": page,
                "text": text,
                "ui_type": ui_type,
            })

        st.markdown(raw_answer)

        if unique_steps:
            st.markdown("### Мини-инструкция")

            lines = []

            for idx, step in enumerate(unique_steps, start=1):
                text = step["text"]
                page = step["page"]
                ui_type = step["ui_type"]

                action = "откройте"

                if ui_type == "button":
                    action = "нажмите кнопку"

                elif ui_type == "hyperlink":
                    action = "перейдите по ссылке"

                elif ui_type in {"tab", "sidebar_item"}:
                    action = "выберите пункт"

                elif ui_type == "input":
                    action = "заполните поле"

                lines.append(
                    f"{idx}. На странице {page} {action} «{text}»."
                )

            st.markdown("\n".join(lines))

    else:
        st.markdown(raw_answer)

    st.subheader("Что нажать / где смотреть в интерфейсе")

    st.info(
        "Запрошено элементов: "
        + str(len(target_phrases))
        + " — "
        + ", ".join(target_phrases)
    )

    if not ui_results:
        st.warning("UI-элементы по этому вопросу не найдены.")
        return context, []

    ui_results = ui_results[:top_k_ui]

    st.success(
        "Найдено элементов на одной странице: "
        + str(len(ui_results))
        + " — "
        + ", ".join([r["item"].get("text", "") for r in ui_results])
    )

    first_item = ui_results[0]["item"]
    page_image = Path(first_item["page_image"])
    matched_elements = [r["item"] for r in ui_results]

    col1, col2 = st.columns([1.45, 1])

    with col1:
        if page_image.exists():
            focus_path = make_ui_focus_image(
                page_image_path=page_image,
                matched_elements=matched_elements,
                out_path="temp/ui_focus.png",
            )

            st.image(
                focus_path,
                caption="Обрезанный интерфейс: выделены только элементы из запроса",
                use_container_width=True,
            )
        else:
            st.warning(f"Не найдена картинка страницы: {page_image}")

    with col2:
        for result in ui_results:
            item = result["item"]

            st.markdown(
                f"### {result.get('matched_query')} → {item.get('text')}\n"
                f"type={item.get('ui_type', 'unknown')}  \n"
                f"score={result.get('final_score', result.get('score', 0.0)):.4f}, "
                f"siamese={result.get('siamese_score', 0.0):.4f}"
            )

            crop = Path(item.get("crop_image", ""))

            if crop.exists():
                st.image(crop, caption="Кроп найденного элемента")

            with st.expander(f"Технические данные: {item.get('text')}"):
                st.json(
                    {
                        "matched_query": result.get("matched_query"),
                        "text": item.get("text"),
                        "normalized_text": item.get("normalized_text"),
                        "ui_type": item.get("ui_type"),
                        "page": item.get("page"),
                        "bbox": item.get("bbox"),
                        "crop": item.get("crop_image"),
                        "score": result.get("score"),
                        "raw_score": result.get("raw_score"),
                        "final_score": result.get("final_score"),
                        "siamese_score": result.get("siamese_score"),
                        "phrase_score": result.get("_phrase_score"),
                    }
                )

    with st.expander("Debug: текстовые источники, использованные для ответа"):
        for result in filtered_text_results:
            item = result["item"]
            st.markdown(f"**Страница {item.get('page')} · score={result.get('score', 0):.4f}**")
            st.write(item.get("text", ""))

    with st.expander("Debug: полная страница с выделением"):
        if page_image.exists():
            full_path = make_full_debug_image(
                page_image_path=page_image,
                matched_elements=matched_elements,
                out_path="temp/ui_debug_full.png",
            )

            st.image(
                full_path,
                caption="Полная страница для проверки bbox",
                use_container_width=True,
            )

    with st.expander("Debug: similarity map"):
        if page_image.exists():
            attention = draw_similarity_boxes(
                page_image,
                ui_results,
                max_items=len(ui_results),
            )

            st.image(
                attention,
                caption="Similarity map только по выбранным элементам",
                use_container_width=True,
            )

    with st.expander("Debug: raw UI results"):
        st.json(
            [
                {
                    "matched_query": r.get("matched_query"),
                    "text": r["item"].get("text"),
                    "normalized_text": r["item"].get("normalized_text"),
                    "ui_type": r["item"].get("ui_type"),
                    "page": r["item"].get("page"),
                    "bbox": r["item"].get("bbox"),
                    "score": r.get("score"),
                    "final_score": r.get("final_score"),
                    "siamese_score": r.get("siamese_score"),
                    "phrase_score": r.get("_phrase_score"),
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
        f"### Text source: страница {item.get('page')} · "
        f"блок {item.get('block_id', item.get('block', '-'))} · score={result.get('score', 0):.4f}"
    )

    if "dense_score" in result:
        st.caption(
            f"semantic={result.get('dense_score', 0):.4f} · "
            f"bm25={result.get('bm25_score', 0):.4f}"
        )

    col1, col2 = st.columns([1.4, 1])

    with col1:
        page_image = Path(item.get("page_image", ""))

        if page_image.exists():
            st.image(Image.open(page_image), caption="Страница-источник")
        else:
            st.warning(f"Не найдена страница: {page_image}")

    with col2:
        st.markdown("#### Текст блока")
        st.write(item.get("text", ""))

        with st.expander("Технические данные"):
            st.json(item)


st.set_page_config(page_title="PDF RAG + UI Element Search", layout="wide")

st.title("PDF RAG + UI Element Search")
st.caption("Ответ по инструкции + поиск конкретных кнопок, ссылок и надписей интерфейса 1С.")

rag_dir = st.text_input(
    "Индекс PDF / RAG dir",
    value="data/services_1c_test_rag",
)

checkpoint = st.text_input(
    "Checkpoint Siamese модели",
    value="checkpoints/ui_elements_siamese_services/best.pt",
)

index_dir = st.text_input(
    "Индекс UI-элементов",
    value="indexes/services_1c_test_ui",
)

query = st.text_input(
    "Вопрос",
    "где открыть интернет-поддержку пользователей",
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
    checkpoint_path = Path(checkpoint)
    index_path = Path(index_dir)

    if not rag_path.exists():
        st.error(f"Не найден RAG dir: {rag_path}")
        st.stop()

    possible_text_indexes = [
        rag_path / "items.jsonl",
        rag_path / "text_blocks.jsonl",
        rag_path / "blocks.jsonl",
        rag_path / "chunks.jsonl",
    ]

    if not any(p.exists() for p in possible_text_indexes):
        st.error(
            "Не найден текстовый индекс. Ожидался один из файлов: "
            + ", ".join(str(p) for p in possible_text_indexes)
        )
        st.stop()

    if not checkpoint_path.exists():
        st.error(f"Не найден checkpoint: {checkpoint_path}")
        st.stop()

    if not index_path.exists():
        st.error(f"Не найден index dir: {index_path}")
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
        checkpoint=str(checkpoint_path),
        index_dir=str(index_path),
    )

    if show_sources:
        st.subheader("Текстовые источники")

        filtered_sources = filter_text_results_by_ui(text_results, ui_results)

        pages = sorted(
            {
                r["item"].get("page")
                for r in filtered_sources
                if r.get("item")
            }
        )

        st.info("Использованные страницы: " + ", ".join(map(str, pages)))

        for result in filtered_sources:
            show_text_source(result)

    st.divider()
    st.subheader("Контекст для ответа")
    st.text_area("Контекст", context, height=320)
