# -*- coding: utf-8 -*-

import re


def norm(text):
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_pdf_heading(text, bbox, page_w, page_h):
    t = norm(text)
    x0, y0, x1, y1 = bbox
    w = x1 - x0

    if y0 < page_h * 0.16 and w > page_w * 0.35:
        return True

    bad = [
        "раздел",
        "глава",
        "инструкция",
        "пример",
        "рис",
        "таблица",
        "листинг",
    ]

    if y0 < page_h * 0.25 and any(b in t for b in bad):
        return True

    return False


def looks_like_paragraph(text, bbox, page_w, page_h):
    t = norm(text)
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    words = t.split()

    if len(words) >= 7:
        return True

    if w > page_w * 0.55 and len(words) >= 4:
        return True

    return False


def is_existing_markup(text, bbox):
    t = norm(text)
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0

    if t.isdigit() and w < 80 and h < 80:
        return True

    return False


def is_likely_ui_candidate(text, bbox, page_w, page_h):
    t = norm(text)

    if len(t) < 2:
        return False

    if is_existing_markup(text, bbox):
        return False

    if is_pdf_heading(text, bbox, page_w, page_h):
        return False

    if looks_like_paragraph(text, bbox, page_w, page_h):
        return False

    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0

    if h < 8 or w < 8:
        return False

    # UI обычно находится внутри скриншотов, не в самом верху PDF
    if y0 < page_h * 0.10:
        return False

    # Очень низ страницы часто содержит подписи/текст инструкции
    if y0 > page_h * 0.90:
        return False

    return True
