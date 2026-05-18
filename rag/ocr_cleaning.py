# -*- coding: utf-8 -*-

import re


OCR_REPLACEMENTS = {
    "г0сты": "госты",
    "гостъ": "госты",
    "показатепи": "показатели",
    "показатсли": "показатели",
    "входнои": "входной",
}


def normalize_ocr_text(text: str) -> str:
    text = str(text).strip().lower().replace("ё", "е")
    text = text.replace("—", "-").replace("–", "-")
    text = text.strip("[]{}()«»\"'")
    text = re.sub(r"[:;,.]+$", "", text)

    for bad, good in OCR_REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(bad)}\b", good, text)

    text = re.sub(r"([а-яa-z])-$", r"\1", text)
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()
