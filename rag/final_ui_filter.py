# -*- coding: utf-8 -*-
from __future__ import annotations

import re

from rag.domain_1c_dictionary import normalize_1c_text, similarity, canonicalize_1c_term


NOISE_PATTERNS = [
    r"^[\W_]+$",
    r"^\d+$",
    r"^[a-zа-я]\s*$",
    r"контрагент\s*\(сотдан",
    r"контрагент\s*\(создан",
    r"контрагенты\s+контрагенты",
    r"организация\s+создание\)",
    r"проверить\s+заполнение",
]


def clean_ui_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_noise_text(text: str) -> bool:
    norm = normalize_1c_text(text)

    if not norm:
        return True

    if len(norm) <= 1:
        return True

    for pattern in NOISE_PATTERNS:
        if re.search(pattern, norm):
            return True

    words = norm.split()

    if len(words) >= 4 and len(set(words)) <= 2:
        return True

    bad_ratio = sum(1 for ch in norm if ch in "{}[]|\\/=_~") / max(1, len(norm))
    if bad_ratio > 0.25:
        return True

    return False


def target_match_score(text: str, targets: list[str]) -> float:
    if not targets:
        return 0.0

    text_norm = normalize_1c_text(text)

    best = 0.0
    for target in targets:
        score = similarity(text_norm, target)
        best = max(best, score)

    return best


def deduplicate_results(results: list[dict]) -> list[dict]:
    kept = []

    for result in results:
        item = result.get("item", {})
        text = clean_ui_text(item.get("text", ""))

        canonical = canonicalize_1c_term(text)

        duplicate_idx = None

        for i, old in enumerate(kept):
            old_item = old.get("item", {})
            old_text = clean_ui_text(old_item.get("text", ""))
            old_canonical = canonicalize_1c_term(old_text)

            same_text = similarity(canonical, old_canonical) >= 0.88
            same_page = item.get("page") == old_item.get("page")
            same_screen = item.get("screenshot_idx") == old_item.get("screenshot_idx")

            if same_text and same_page and same_screen:
                duplicate_idx = i
                break

        if duplicate_idx is None:
            kept.append(result)
        else:
            old_score = float(kept[duplicate_idx].get("semantic_score", kept[duplicate_idx].get("score", 0)))
            new_score = float(result.get("semantic_score", result.get("score", 0)))

            if new_score > old_score:
                kept[duplicate_idx] = result

    return kept


def final_filter_ui_results(
    results: list[dict],
    targets: list[str] | None = None,
    min_score: float = 0.18,
    min_target_score: float = 0.42,
    limit: int = 6,
) -> list[dict]:
    targets = targets or []

    filtered = []

    for result in results or []:
        item = result.get("item", {})
        text = clean_ui_text(item.get("text", ""))

        if is_noise_text(text):
            continue

        base_score = float(result.get("semantic_score", result.get("score", 0.0)))

        tm = target_match_score(text, targets)

        if targets and tm < min_target_score and base_score < 0.55:
            continue

        if base_score < min_score and tm < min_target_score:
            continue

        result = dict(result)
        result["target_match_score"] = tm

        filtered.append(result)

    filtered = deduplicate_results(filtered)

    filtered.sort(
        key=lambda r: (
            r.get("item", {}).get("page", 9999),
            r.get("item", {}).get("screenshot_idx", 9999),
            -float(r.get("target_match_score", 0.0)),
            -float(r.get("semantic_score", r.get("score", 0.0))),
        )
    )

    return filtered[:limit]
