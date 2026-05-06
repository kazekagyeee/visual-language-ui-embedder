# -*- coding: utf-8 -*-

from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from rag.ui_element_searcher import UIElementSearcher


@st.cache_resource
def load_searcher():
    return UIElementSearcher()


def draw_bbox(page_image, bbox):
    img = Image.open(page_image).convert("RGB")
    draw = ImageDraw.Draw(img)

    x0, y0, x1, y1 = bbox

    for i in range(6):
        draw.rectangle(
            [x0 - i, y0 - i, x1 + i, y1 + i],
            outline="green",
        )

    return img


st.set_page_config(page_title="UI Element Search", layout="wide")

st.title("UI Element Search: OCR + Siamese")
st.caption("Ищет конкретные кнопки, ссылки и надписи внутри интерфейсов 1С на страницах PDF.")

query = st.text_input("Что найти на интерфейсе?", "ГОСТы")
top_k = st.slider("Сколько элементов показать", 1, 20, 8)

if st.button("Искать UI-элемент"):
    searcher = load_searcher()

    if searcher.model is None:
        st.error("Модель или индекс не найдены. Сначала собери OCR elements, обучи siamese и построй индекс.")
        st.stop()

    results = searcher.search(query, top_k=top_k)

    for result in results:
        item = result["item"]

        st.divider()
        st.markdown(
            f"### {item['text']} · page={item['page']} · "
            f"score={result['score']:.4f} · siamese={result['siamese_score']:.4f}"
        )

        col1, col2 = st.columns([1.4, 1])

        with col1:
            page_image = Path(item["page_image"])

            if page_image.exists():
                highlighted = draw_bbox(page_image, item["bbox"])
                st.image(highlighted, caption="Страница с выделенным UI-элементом")

        with col2:
            crop = Path(item["crop_image"])

            if crop.exists():
                st.image(Image.open(crop), caption="Кроп найденного UI-элемента")

            st.json({
                "text": item["text"],
                "page": item["page"],
                "bbox": item["bbox"],
                "crop": item["crop_image"],
                "confidence": item.get("confidence"),
            })
