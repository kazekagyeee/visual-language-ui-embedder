# -*- coding: utf-8 -*-

import re


OCR_REPLACEMENTS = {
    "контропь": "контроль",
    "контроя": "контроль",
    "контроЛя": "контроля",
    "докуненты": "документы",
    "выпопнения": "выполнения",
    "выпопнения": "выполнения",
    "выпопнення": "выполнения",
    "пходного": "входного",
    "входнога": "входного",
    "показатепи": "показатели",
    "говаров": "товаров",
    "скпад": "склад",
    "ппанирование": "планирование",
    "резупьтат": "результат",
    "интернетполлерскм": "интернет-поддержки",
    "интернетполлерски": "интернет-поддержки",
    "интернетподлерскм": "интернет-поддержки",
    "пользоватепей": "пользователей",
}


def cleanup_ocr_text(text):
    if not text:
        return ""

    text = str(text)

    text = text.replace("{", "")
    text = text.replace("}", "")
    text = text.replace("|", "")
    text = text.replace("[", "")
    text = text.replace("]", "")

    for bad, good in OCR_REPLACEMENTS.items():
        text = re.sub(
            rf"\b{re.escape(bad)}\b",
            good,
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(r"\s+", " ", text)

    return text.strip()