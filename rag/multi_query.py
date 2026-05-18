# -*- coding: utf-8 -*-

import re


DOMAIN_EXPANSIONS = {
    "монитор интернет-поддержки": [
        "интернет-поддержка пользователей",
        "монитор интернет-поддержки",
        "подключить интернет-поддержку",
        "форма настройки интернет-поддержки",
    ],
    "интернет-поддержк": [
        "интернет-поддержка пользователей",
        "монитор интернет-поддержки",
        "подключить интернет-поддержку",
        "если у вас еще нет логина",
    ],
}


def normalize_text(text: str):
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_query_to_ui_phrases(query: str):
    q = normalize_text(query)

    results = []

    for trigger, phrases in DOMAIN_EXPANSIONS.items():
        if trigger in q:
            results.extend(phrases)

    results.append(query)

    # remove duplicates
    uniq = []
    seen = set()

    for item in results:
        n = normalize_text(item)
        if n not in seen:
            seen.add(n)
            uniq.append(item)

    return uniq[:6]