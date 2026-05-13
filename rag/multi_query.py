# -*- coding: utf-8 -*-

import re


def normalize_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_query_to_ui_phrases(query, known_phrases):
    q = normalize_text(query)

    found = []

    for phrase in sorted(known_phrases, key=lambda x: len(normalize_text(x)), reverse=True):
        p = normalize_text(phrase)

        if len(p) < 3:
            continue

        if p in q:
            found.append(phrase)

    cleaned = []

    for phrase in found:
        p = normalize_text(phrase)

        nested = False
        for other in cleaned:
            o = normalize_text(other)
            if p in o and p != o:
                nested = True
                break

        if not nested:
            cleaned.append(phrase)

    return cleaned or [query]


def group_results_by_best_page(results):
    by_page = {}

    for result in results:
        item = result["item"]
        page = item["page"]
        matched = result.get("matched_query", item.get("text", ""))

        by_page.setdefault(page, {
            "results": [],
            "matched_queries": set(),
            "score_sum": 0.0,
        })

        by_page[page]["results"].append(result)
        by_page[page]["matched_queries"].add(matched)
        by_page[page]["score_sum"] += result["score"]

    if not by_page:
        return []

    best_page, best_group = max(
        by_page.items(),
        key=lambda x: (
            len(x[1]["matched_queries"]),
            x[1]["score_sum"],
        ),
    )

    selected = []
    seen = set()

    for result in sorted(best_group["results"], key=lambda r: r["score"], reverse=True):
        item = result["item"]
        key = (tuple(item["bbox"]), normalize_text(item["text"]))

        if key in seen:
            continue

        seen.add(key)
        selected.append(result)

    return selected
