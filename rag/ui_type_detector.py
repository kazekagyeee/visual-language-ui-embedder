# -*- coding: utf-8 -*-

import re


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_ui_type(text: str, bbox, page_width=None):
    text_norm = normalize_text(text)

    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0

    button_words = {
        "создать",
        "добавить",
        "записать",
        "закрыть",
        "провести",
        "ок",
        "отмена",
        "еще",
        "печать",
        "отчеты",
        "заполнить",
        "рассчитать",
    }

    menu_words = {
        "входной контроль",
        "склад",
        "производство",
        "закупки",
        "продажи",
        "справочники",
        "документы",
    }

    if text_norm in button_words or any(word in text_norm for word in button_words):
        return "button"

    if text_norm in menu_words:
        return "sidebar_item"

    if page_width and x1 < page_width * 0.35 and len(text_norm.split()) <= 3:
        return "sidebar_item"

    if len(text_norm.split()) <= 3 and h < 45:
        return "hyperlink"

    if ":" in text or text.endswith(":"):
        return "label"

    if w > 180 and h < 40:
        return "table_cell"

    return "label"
