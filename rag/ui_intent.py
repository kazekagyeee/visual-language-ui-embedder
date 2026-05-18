# -*- coding: utf-8 -*-

import re


def normalize(text):
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def infer_ui_type(item):
    text = normalize(item.get("text", ""))

    if "ссылка" in text:
        return "hyperlink"

    if any(x in text for x in [
        "подключить",
        "войти",
        "сохранить",
        "ок",
        "далее",
    ]):
        return "button"

    return "text"


def action_word(ui_type):
    if ui_type == "button":
        return "нажмите кнопку"

    if ui_type == "hyperlink":
        return "перейдите по ссылке"

    return "найдите элемент"


def chain_priority(item):
    text = normalize(item.get("text", ""))

    if "интернет-поддержка пользователей" in text:
        return 0

    if "форма настройки" in text:
        return 1

    if "монитор интернет-поддержки" in text:
        return 2

    if "подключить интернет-поддержку" in text:
        return 3

    return 10