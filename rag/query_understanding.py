# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from rag.domain_1c_dictionary import query_to_targets, canonicalize_1c_term


@dataclass
class QueryUnderstanding:
    query: str
    targets: list[str]
    steps: list[str]


def _unique(values: list[str]) -> list[str]:
    result = []
    for value in values:
        value = (value or "").strip()
        if value and value not in result:
            result.append(value)
    return result


def extract_quoted_targets(text: str) -> list[str]:
    if not text:
        return []
    targets = []
    for pattern in [r"«([^»]+)»", r'"([^"]+)"', r"'([^']+)'"]:
        for match in re.findall(pattern, text):
            targets.append(canonicalize_1c_term(match))
    return _unique(targets)


def extract_targets_from_text(text: str) -> list[str]:
    targets = []
    targets.extend(extract_quoted_targets(text))
    targets.extend(query_to_targets(text))
    return _unique(targets)


def extract_targets_from_query(query: str) -> list[str]:
    return extract_targets_from_text(query)


def get_query_targets(query: str) -> list[str]:
    return extract_targets_from_query(query)


def build_steps_from_targets(targets: list[str], query: str = "") -> list[str]:
    targets = _unique(targets)
    q = (query or "").lower()
    steps: list[str] = []

    if targets == ["Входной контроль"]:
        return ["Найдите раздел «Входной контроль»."]

    for target in targets:
        if target in {"Входной контроль", "Закупки", "Склад и доставка"}:
            steps.append(f"Откройте раздел «{target}».")
        elif target.startswith("АРМ "):
            steps.append(f"Перейдите в раздел «{target}».")
        elif target in {"Заявки на контроль", "Показатели контроля", "Контрагенты", "Организации"}:
            steps.append(f"Откройте пункт «{target}».")
        elif target in {"Создать", "Заполнить", "Записать", "Сформировать"}:
            steps.append(f"Нажмите «{target}».")
        elif target in {"ИНН", "Начните отсюда"}:
            steps.append(f"Введите данные в поле «{target}».")
        elif target.startswith("Подключить"):
            steps.append(f"Нажмите кнопку «{target}».")
        else:
            if "где" in q or "найти" in q:
                steps.append(f"Найдите элемент «{target}».")
            else:
                steps.append(f"Выберите «{target}».")

    return _unique(steps)


def understand_query(query: str, source_text: str | None = None) -> QueryUnderstanding:
    targets = []

    if source_text:
        targets.extend(extract_targets_from_text(source_text))

    targets.extend(extract_targets_from_query(query))
    targets = _unique(targets)

    steps = build_steps_from_targets(targets, query=query)

    return QueryUnderstanding(query=query, targets=targets, steps=steps)


def query_understanding(query: str, source_text: str | None = None) -> dict:
    return asdict(understand_query(query=query, source_text=source_text))


def enrich_targets_with_domain_dictionary(query: str, targets: list[str] | None = None) -> list[str]:
    result = list(targets or [])
    result.extend(query_to_targets(query))
    return _unique(result)


def extract_action_targets(query: str, source_text: str | None = None) -> list[str]:
    return understand_query(query=query, source_text=source_text).targets