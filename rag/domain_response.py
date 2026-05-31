# -*- coding: utf-8 -*-
from __future__ import annotations

from copy import deepcopy
from rag.domain_1c_dictionary import query_to_targets


def _q(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def _unique(values):
    result = []
    for value in values or []:
        if value and value not in result:
            result.append(value)
    return result


def infer_primary_targets(query: str, targets: list[str]) -> list[str]:
    q = _q(query)

    if "заяв" in q and "контрол" in q and ("создат" in q or "создать" in q):
        return ["Входной контроль", "АРМ Входной контроль", "Заявки на контроль", "Создать"]

    if "маршрут" in q and "карт" in q:
        return ["Маршрутная карта", "Печать"]

    if "ресурс" in q and "спецификац" in q:
        return ["Ресурсные спецификации", "Создать"]

    if "показател" in q and "контрол" in q:
        return ["Показатели контроля"]

    if "арм" in q and "входн" in q and "контрол" in q:
        return ["АРМ Входной контроль"]

    if "заяв" in q and "контрол" in q and "создат" not in q:
        return ["Заявки на контроль"]

    if "входн" in q and "контрол" in q and "арм" not in q and "заяв" not in q:
        return ["Входной контроль"]

    if "монитор" in q and "интернет" in q:
        return ["Монитор Интернет-поддержки"]

    if "подключ" in q and "интернет" in q:
        return ["Интернет-поддержка пользователей", "Подключить Интернет-поддержку"]

    if "контрагент" in q:
        if "инн" in q or "реквизит" in q or "создат" in q:
            return ["Контрагенты", "Создать", "ИНН"]
        return ["Контрагенты"]

    if "организац" in q:
        if "создат" in q:
            return ["Организации", "Создать"]
        return ["Организации"]

    if "перемещен" in q and "заказ" in q:
        return ["Заказы на перемещение", "Создать заказ на перемещение"]

    return targets[:1] if targets else []


def enrich_response_with_domain(query: str, response: dict) -> dict:
    response = deepcopy(response)
    q = _q(query)

    domain_targets = query_to_targets(query)
    old_targets = response.get("targets") or []
    targets = _unique(old_targets + domain_targets)

    if "маршрут" in q and "карт" in q:
        response["pdf_name"] = "instruction.pdf"
        response["page"] = 51
        response["source"] = "instruction.pdf, страница 51"
        targets = _unique(["Маршрутная карта", "Маршрутная карта (СБМ)", "Печать"] + targets)
        response["steps"] = [
            "Откройте документ или этап, для которого требуется маршрутная карта.",
            "Нажмите «Печать».",
            "Выберите пункт «Маршрутная карта».",
        ]

    elif "ресурс" in q and "спецификац" in q:
        response["pdf_name"] = "instruction.pdf"
        response["page"] = 48
        response["source"] = "instruction.pdf, страница 48"
        targets = _unique([
            "Ресурсные спецификации",
            "Ресурсная спецификация",
            "Создать",
            "Создать группу",
        ] + targets)
        response["steps"] = [
            "Откройте раздел «Производство».",
            "Перейдите в «Ресурсные спецификации».",
            "Нажмите «Создать».",
        ]

    elif "заяв" in q and "контрол" in q and ("создат" in q or "создать" in q):
        response["pdf_name"] = "instruction.pdf"
        response["page"] = 14
        response["source"] = "instruction.pdf, страница 14"
        targets = _unique(["Входной контроль", "АРМ Входной контроль", "Заявки на контроль", "Создать"] + targets)
        response["steps"] = [
            "Откройте вкладку «Входной контроль».",
            "Перейдите в раздел «АРМ Входной контроль».",
            "Откройте пункт «Заявки на контроль».",
            "Нажмите «Создать».",
        ]

    elif "показател" in q and "контрол" in q:
        response["pdf_name"] = "instruction.pdf"
        response["page"] = 3
        response["source"] = "instruction.pdf, страница 3"
        targets = _unique(["Входной контроль", "АРМ Входной контроль", "Показатели контроля", "Создать"] + targets)
        response["steps"] = [
            "Откройте раздел «Входной контроль».",
            "Перейдите в «АРМ Входной контроль».",
            "Откройте пункт «Показатели контроля».",
        ]

    elif "организац" in q:
        response["pdf_name"] = "services_1c.pdf"
        response["page"] = 20
        response["source"] = "services_1c.pdf, страница 20"
        targets = _unique(["Организации", "Создать", "ИНН", "Начните отсюда", "Заполнить"] + targets)
        response["steps"] = [
            "Откройте справочник «Организации».",
            "Нажмите «Создать».",
            "Введите ИНН в поле «Начните отсюда».",
            "Нажмите «Заполнить».",
        ]

    elif "контрагент" in q:
        response["pdf_name"] = "services_1c.pdf"
        response["page"] = 22
        response["source"] = "services_1c.pdf, страница 22"
        targets = _unique(["Контрагенты", "Создать", "ИНН", "Начните отсюда", "Заполнить"] + targets)
        response["steps"] = [
            "Откройте справочник «Контрагенты».",
            "Нажмите «Создать».",
            "Введите ИНН в поле «ИНН» или «Начните отсюда».",
            "Нажмите кнопку «Заполнить».",
        ]

    elif "монитор" in q and "интернет" in q:
        response["pdf_name"] = "services_1c.pdf"
        response["page"] = 188
        response["source"] = "services_1c.pdf, страница 188"
        targets = _unique(["Монитор Интернет-поддержки"] + targets)

    response["targets"] = targets
    response["primary_targets"] = infer_primary_targets(query, targets)
    response["context_targets"] = [t for t in targets if t not in response["primary_targets"]]

    return response
