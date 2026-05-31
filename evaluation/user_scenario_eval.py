# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path


USER_TEST_CASES = [
    # instruction.pdf — входной контроль
    {"query": "где найти входной контроль", "expected": ["Входной контроль"], "expected_pdf": "instruction.pdf"},
    {"query": "покажи раздел входной контроль", "expected": ["Входной контроль"], "expected_pdf": "instruction.pdf"},
    {"query": "где находится арм входного контроля", "expected": ["АРМ Входной контроль"], "expected_pdf": "instruction.pdf"},
    {"query": "как открыть арм входной контроль", "expected": ["АРМ Входной контроль"], "expected_pdf": "instruction.pdf"},
    {"query": "где найти заявки на контроль", "expected": ["Заявки на контроль"], "expected_pdf": "instruction.pdf"},
    {"query": "где открыть заявки на контроль", "expected": ["Заявки на контроль"], "expected_pdf": "instruction.pdf"},
    {"query": "как создать заявку на контроль", "expected": ["Входной контроль", "АРМ Входной контроль", "Заявки на контроль", "Создать"], "expected_pdf": "instruction.pdf"},
    {"query": "как открыть показатели контроля", "expected": ["Показатели контроля"], "expected_pdf": "instruction.pdf"},
    {"query": "где найти показатели контроля", "expected": ["Показатели контроля"], "expected_pdf": "instruction.pdf"},
    {"query": "как создать показатель контроля", "expected": ["Показатели контроля", "Создать"], "expected_pdf": "instruction.pdf"},
    {"query": "как создать вид контроля", "expected": ["Виды контроля", "Создать"], "expected_pdf": "instruction.pdf"},
    {"query": "где найти виды контроля", "expected": ["Виды контроля"], "expected_pdf": "instruction.pdf"},
    {"query": "где найти группы прочности", "expected": ["Группы прочности"], "expected_pdf": "instruction.pdf"},
    {"query": "где открыть госты для входного контроля", "expected": ["ГОСТы"], "expected_pdf": "instruction.pdf"},
    {"query": "как сформировать документы выполнения контроля", "expected": ["Создать документы выполнения контроля"], "expected_pdf": "instruction.pdf"},
    {"query": "как создать акт входного контроля", "expected": ["Создать акт входного контроля"], "expected_pdf": "instruction.pdf"},
    {"query": "где найти акты входного контроля", "expected": ["Акты входного контроля"], "expected_pdf": "instruction.pdf"},
    {"query": "где найти выполнение входного контроля", "expected": ["Выполнения входного контроля"], "expected_pdf": "instruction.pdf"},
    {"query": "где находится статус контроля", "expected": ["Статус контроля"], "expected_pdf": "instruction.pdf"},
    {"query": "где указать номер плавки", "expected": ["Номер плавки"], "expected_pdf": "instruction.pdf"},
    {"query": "где указать сертификат", "expected": ["Сертификат"], "expected_pdf": "instruction.pdf"},
    {"query": "как создать заказ на перемещение", "expected": ["Заказы на перемещение", "Создать заказ на перемещение"], "expected_pdf": "instruction.pdf"},

    # services_1c.pdf — сервисы 1С, контрагенты, организации
    {"query": "где найти монитор интернет поддержки", "expected": ["Монитор Интернет-поддержки"], "expected_pdf": "services_1c.pdf"},
    {"query": "как подключить интернет поддержку", "expected": ["Интернет-поддержка пользователей", "Подключить Интернет-поддержку"], "expected_pdf": "services_1c.pdf"},
    {"query": "где найти интернет поддержку пользователей", "expected": ["Интернет-поддержка пользователей"], "expected_pdf": "services_1c.pdf"},
    {"query": "как создать нового контрагента", "expected": ["Контрагенты", "Создать", "ИНН"], "expected_pdf": "services_1c.pdf"},
    {"query": "где найти контрагентов", "expected": ["Контрагенты"], "expected_pdf": "services_1c.pdf"},
    {"query": "где открыть справочник контрагентов", "expected": ["Контрагенты"], "expected_pdf": "services_1c.pdf"},
    {"query": "где находится поле инн", "expected": ["ИНН"], "expected_pdf": "services_1c.pdf"},
    {"query": "где находится поле инн контрагента", "expected": ["ИНН"], "expected_pdf": "services_1c.pdf"},
    {"query": "как заполнить контрагента по инн", "expected": ["Контрагенты", "Создать", "ИНН"], "expected_pdf": "services_1c.pdf"},
    {"query": "как заполнить реквизиты контрагента по инн", "expected": ["Контрагенты", "Создать", "ИНН"], "expected_pdf": "services_1c.pdf"},
    {"query": "как открыть досье контрагента", "expected": ["Досье контрагента"], "expected_pdf": "services_1c.pdf"},
    {"query": "как создать организацию", "expected": ["Организации", "Создать"], "expected_pdf": "services_1c.pdf"},
    {"query": "где открыть справочник организации", "expected": ["Организации"], "expected_pdf": "services_1c.pdf"},
    {"query": "где найти организации", "expected": ["Организации"], "expected_pdf": "services_1c.pdf"},
    {"query": "как подключить 1с эдо", "expected": ["1С-ЭДО"], "expected_pdf": "services_1c.pdf"},
    {"query": "где найти 1спарк риски", "expected": ["1СПАРК Риски"], "expected_pdf": "services_1c.pdf"},
    {"query": "как отправить платежное поручение через директбанк", "expected": ["1С:ДиректБанк"], "expected_pdf": "services_1c.pdf"},
    {"query": "где найти директбанк", "expected": ["1С:ДиректБанк"], "expected_pdf": "services_1c.pdf"},
    {"query": "где найти 1с отчетность", "expected": ["1С-Отчетность"], "expected_pdf": "services_1c.pdf"},
    {"query": "где найти сервис 1с контрагент", "expected": ["1С:Контрагент"], "expected_pdf": "services_1c.pdf"},
]


def main():
    from evaluation.evaluate_ui_retrieval import evaluate_case, HybridSearcher, TrainedUIElementSearcher

    text_searcher = HybridSearcher(rag_dir="data/all_pdf_rag")
    ui_searcher = TrainedUIElementSearcher(
        index_dir="data/ui_trained_index",
        checkpoint="checkpoints/ui_siamese_ranker.pt",
    )

    results = []
    for case in USER_TEST_CASES:
        results.append(evaluate_case(case, text_searcher, ui_searcher))

    def mean(key: str) -> float:
        return sum(float(r.get(key, 0.0)) for r in results) / max(1, len(results))

    summary = {
        "cases": len(results),
        "success_rate": round(mean("success"), 4),
        "mean_precision": round(mean("precision"), 4),
        "mean_recall": round(mean("recall"), 4),
        "mean_f1": round(mean("f1"), 4),
    }

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    with (out_dir / "user_scenario_eval.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    print("=== USER SCENARIO EVAL: GUI PDF ONLY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== DETAILS ===")
    for r in results:
        print()
        print("QUERY:", r.get("query"))
        print("EXPECTED:", r.get("expected"))
        print("FOUND:", r.get("found"))
        print("PRECISION:", r.get("precision"), "RECALL:", r.get("recall"), "F1:", r.get("f1"), "SUCCESS:", r.get("success"))

    print("\nSaved: reports/user_scenario_eval.json")


if __name__ == "__main__":
    main()
