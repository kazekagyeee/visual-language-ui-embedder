# -*- coding: utf-8 -*-

import re


OCR_REPLACEMENTS = {
    "г0сты": "госты",
    "гостъ": "госты",
    "гост": "госты",
    "контроля-": "контроля",
    "показатепи": "показатели",
    "показатсли": "показатели",
    "входнои": "входной",
    "входнои контроль": "входной контроль",
}


def normalize_ocr_text(text: str) -> str:
    text = text.strip()
    text = text.replace("ё", "е")
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    text = re.sub(r"\s+", " ", text)

    lower = text.lower()

    for bad, good in OCR_REPLACEMENTS.items():
        lower = lower.replace(bad, good)

    lower = re.sub(r"([а-яa-z])-$", r"\1", lower)
    lower = re.sub(r"[^а-яa-z0-9\s\-]+", "", lower)
    lower = re.sub(r"\s+", " ", lower)

    return lower.strip()


def display_text(text: str) -> str:
    cleaned = normalize_ocr_text(text)
    return cleaned[:1].upper() + cleaned[1:] if cleaned else text
