# -*- coding: utf-8 -*-

from pathlib import Path
import subprocess
import sys

import streamlit as st

from rag.answer_engine import AnswerEngine
from rag.hybrid_search import HybridSearcher
from rag.trained_ui_searcher import TrainedUIElementSearcher
from rag.ui_reranker import build_ui_semantic_results
from rag.ui_visualization import draw_ui_results, show_page_screenshots_from_ui_index
from rag.ocr_cleanup import cleanup_ocr_text
from rag.vlm_verifier import VLMVerifier, apply_vlm_verification


PDF_DIR = "data_source"
RAG_DIR = "data/all_pdf_rag"
UI_INDEX_DIR = "data/ui_trained_index"
UI_CHECKPOINT = "checkpoints/ui_siamese_ranker.pt"


st.set_page_config(
    page_title="Поиск по PDF-инструкциям",
    page_icon="🔎",
    layout="centered",
)


@st.cache_resource
def load_text_searcher():
    return HybridSearcher(rag_dir=RAG_DIR)


@st.cache_resource
def load_ui_searcher():
    return TrainedUIElementSearcher(
        index_dir=UI_INDEX_DIR,
        checkpoint=UI_CHECKPOINT,
    )


@st.cache_resource
def load_answer_engine():
    return AnswerEngine()


@st.cache_resource
def load_vlm_verifier(enabled):
    return VLMVerifier(enabled=enabled)


def pdf_index_exists():
    return (
        Path(RAG_DIR, "items.jsonl").exists()
        and Path(RAG_DIR, "embeddings.npy").exists()
    )


def ui_index_exists():
    return (
        Path(UI_INDEX_DIR, "ui_items.jsonl").exists()
        and Path(UI_INDEX_DIR, "ui_embeddings.npy").exists()
    )


def build_pdf_index():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.build_pdf_rag_multi",
            "--pdf-dir",
            PDF_DIR,
            "--out-dir",
            RAG_DIR,
        ],
        check=True,
    )


def build_ui_index():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "retrieval.build_trained_ui_index",
            "--ui-index-dir",
            "data/ui_index",
            "--checkpoint",
            UI_CHECKPOINT,
            "--out-dir",
            UI_INDEX_DIR,
        ],
        check=True,
    )


def save_uploaded_files(uploaded_files):
    pdf_dir = Path(PDF_DIR)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    for file in uploaded_files:
        out_path = pdf_dir / file.name

        with open(out_path, "wb") as f:
            f.write(file.getbuffer())

        saved.append(out_path.name)

    return saved


def rebuild_buttons():
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Пересобрать PDF-индекс", use_container_width=True):
            with st.spinner("Пересобираю PDF-индекс..."):
                build_pdf_index()
                st.cache_resource.clear()

            st.success("PDF-индекс пересобран.")
            st.rerun()

    with col2:
        if st.button("Пересобрать UI-индекс", use_container_width=True):
            with st.spinner("Пересобираю обученный UI-индекс..."):
                build_ui_index()
                st.cache_resource.clear()

            st.success("UI-индекс пересобран.")
            st.rerun()


def render_answer(response):
    st.markdown("## Ответ")
    st.markdown(f"**Источник:** {response['source']}")
    st.markdown(response["short_answer"])

    if response["steps"]:
        st.markdown("### Что сделать")

        for i, step in enumerate(response["steps"], start=1):
            st.markdown(f"{i}. {step}")

    with st.expander("Показать исходный текст найденного фрагмента"):
        st.write(response["raw_text"])


def get_page_window(response, window_before=3, window_after=5):
    page = response.get("page")

    if page is None or page == "":
        return None

    page = int(page)
    start = max(1, page - window_before)
    end = page + window_after

    return list(range(start, end + 1))


def render_ui_results(response, query, use_vlm=False):
    if not ui_index_exists():
        st.info("UI-индекс еще не собран. Нажмите «Пересобрать UI-индекс».")
        return

    searcher = load_ui_searcher()

    pages = get_page_window(
        response,
        window_before=3,
        window_after=5,
    )

    raw_results = searcher.search(
        query=query,
        targets=response.get("targets", []),
        page_filter=pages,
        pdf_filter=response.get("pdf_name"),
        top_k=100,
    )

    results = build_ui_semantic_results(
        query=query,
        response=response,
        results=raw_results,
        limit=8,
    )

    if use_vlm and results:
        with st.spinner("Проверяю найденные элементы через VLM..."):
            verifier = load_vlm_verifier(enabled=True)

            results = apply_vlm_verification(
                query=query,
                results=results,
                verifier=verifier,
                weight=0.25,
            )

    st.markdown("### Интерфейс 1С")

    if results:
        st.markdown("#### Найденные элементы")

        for idx, result in enumerate(results, start=1):
            item = result["item"]

            vlm_text = ""

            if use_vlm:
                vlm_score = result.get("vlm_score", "-")
                vlm_text = f", vlm={vlm_score}"

            st.markdown(
                f"{idx}. **{cleanup_ocr_text(item.get('text'))}** "
                f"<span style='color: #777;'>"
                f"(стр. {item.get('page')}, экран {item.get('screenshot_idx')}{vlm_text})"
                f"</span>",
                unsafe_allow_html=True,
            )

        images = draw_ui_results(results)

        if images:
            st.markdown("#### Скриншоты с найденными элементами")

            for image in images:
                st.image(image["path"], use_container_width=True)

    missing_count = max(
        0,
        len(response.get("targets", [])) - len(results),
    )

    if missing_count > 0:
        st.markdown("#### Дополнительные скриншоты по шагам")

        st.info(
            "Часть элементов не удалось точно распознать. "
            "Ниже показаны ближайшие скриншоты инструкции."
        )

        shown = set()

        for page in pages:
            fallback_images = show_page_screenshots_from_ui_index(
                ui_searcher=searcher,
                pdf_name=response.get("pdf_name"),
                page=page,
            )

            for image in fallback_images:
                if image["path"] in shown:
                    continue

                shown.add(image["path"])
                st.image(image["path"], use_container_width=True)

        return

    if not results:
        fallback_images = show_page_screenshots_from_ui_index(
            ui_searcher=searcher,
            pdf_name=response.get("pdf_name"),
            page=response.get("page"),
        )

        if fallback_images:
            st.info("Точные элементы не найдены, показываю скриншоты найденной страницы.")

            for image in fallback_images:
                st.image(image["path"], use_container_width=True)
        else:
            st.info("На ближайших скриншотах интерфейса не удалось найти подходящие элементы.")


def show_pdf_page(result):
    if not result:
        return

    item = result["item"]

    pdf_name = item.get("pdf_name")
    page = item.get("page")

    if not pdf_name or not page:
        return

    page_path = Path(RAG_DIR) / "pages" / f"{Path(pdf_name).stem}_page_{int(page):04d}.png"

    if page_path.exists():
        with st.expander("Показать страницу PDF целиком"):
            st.image(str(page_path), use_container_width=True)


def show_sources(results):
    if not results:
        return

    with st.expander("Показать дополнительные найденные фрагменты"):
        for result in results:
            item = result["item"]

            st.markdown(
                f"**{item.get('pdf_name', 'PDF')} — страница {item.get('page', '?')}**"
            )

            st.write(item.get("text", ""))

            st.caption(
                f"score={result.get('score', 0):.4f}, "
                f"bm25={result.get('bm25_score', 0):.4f}, "
                f"dense={result.get('dense_score', 0):.4f}"
            )


def main():
    st.title("Поиск по PDF-инструкциям")

    st.caption(
        "Система ищет ответ по всем PDF и показывает, куда нажать в интерфейсе 1С."
    )

    uploaded_files = st.file_uploader(
        "Добавить PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        saved = save_uploaded_files(uploaded_files)
        st.success("PDF добавлены: " + ", ".join(saved))
        st.info("После добавления PDF пересоберите индексы.")

    rebuild_buttons()

    use_vlm = st.checkbox(
        "Использовать VLM-проверку скриншотов",
        value=False,
        help=(
            "Опциональный Qwen2.5-VL reranker. "
            "Может работать медленно и требовать много памяти."
        ),
    )

    if not pdf_index_exists():
        st.warning("PDF-индекс еще не собран.")
        st.stop()

    query = st.text_input(
        "Ваш вопрос",
        placeholder="Например: как создать заявку на контроль",
    )

    clicked = st.button("Найти ответ", use_container_width=True)

    if not clicked:
        return

    if not query.strip():
        st.warning("Введите вопрос.")
        return

    with st.spinner("Ищу ответ и элементы интерфейса..."):
        text_searcher = load_text_searcher()

        results = text_searcher.search(
            query=query,
            top_k=7,
            alpha=0.15,
        )

        response = load_answer_engine().build_response(
            query=query,
            results=results,
        )

    render_answer(response)
    render_ui_results(response, query, use_vlm=use_vlm)

    if results:
        show_pdf_page(results[0])

    show_sources(results[1:])


if __name__ == "__main__":
    main()
