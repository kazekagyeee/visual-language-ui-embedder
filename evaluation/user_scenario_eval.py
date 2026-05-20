# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path


USER_TEST_CASES = [
    {"query": "где найти входной контроль", "expected": ["Входной контроль"], "expected_pdf": "instruction.pdf"},
    {"query": "покажи раздел входной контроль", "expected": ["Входной контроль"], "expected_pdf": "instruction.pdf"},
    {"query": "где находится арм входного контроля", "expected": ["АРМ Входной контроль"], "expected_pdf": "instruction.pdf"},
    {"query": "как создать заявку на контроль", "expected": ["Входной контроль", "АРМ Входной контроль", "Заявки на контроль", "Создать"], "expected_pdf": "instruction.pdf"},
    {"query": "где открыть заявки на контроль", "expected": ["Заявки на контроль"], "expected_pdf": "instruction.pdf"},
    {"query": "как открыть показатели контроля", "expected": ["Показатели контроля"], "expected_pdf": "instruction.pdf"},
    {"query": "где найти монитор интернет поддержки", "expected": ["Монитор Интернет-поддержки"], "expected_pdf": "services_1c.pdf"},
    {"query": "как подключить интернет поддержку", "expected": ["Интернет-поддержка пользователей", "Подключить Интернет-поддержку"], "expected_pdf": "services_1c.pdf"},
    {"query": "как создать нового контрагента", "expected": ["Контрагенты", "Создать", "ИНН"], "expected_pdf": "services_1c.pdf"},
    {"query": "где находится поле инн", "expected": ["ИНН"], "expected_pdf": "services_1c.pdf"},
    {"query": "как заполнить контрагента по инн", "expected": ["Контрагенты", "Создать", "ИНН"], "expected_pdf": "services_1c.pdf"},
    {"query": "как создать организацию", "expected": ["Организации", "Создать"], "expected_pdf": "services_1c.pdf"},
    {"query": "где открыть справочник организации", "expected": ["Организации"], "expected_pdf": "services_1c.pdf"},
    {"query": "как создать заказ на перемещение", "expected": ["Склад и доставка", "Заказы на перемещение", "Создать заказ на перемещение"], "expected_pdf": "instruction.pdf"},
]


def _norm(x: str) -> str:
    return (x or "").replace("ё", "е").lower().strip()


def precision_recall_f1(expected: list[str], found: list[str]) -> tuple[float, float, float, bool]:
    exp = {_norm(x) for x in expected}
    got = {_norm(x) for x in found}

    if not exp:
        return 0.0, 0.0, 0.0, False

    matched = 0
    for e in exp:
        if any(e in g or g in e for g in got):
            matched += 1

    precision = matched / max(1, len(got))
    recall = matched / max(1, len(exp))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    success = recall > 0.0

    return precision, recall, f1, success


def main():
    from evaluation.evaluate_ui_retrieval import evaluate_case, HybridSearcher, TrainedUIElementSearcher

    text_searcher = HybridSearcher(rag_dir="data/all_pdf_rag")
    ui_searcher = TrainedUIElementSearcher(
        index_dir="data/ui_trained_index",
        checkpoint="checkpoints/ui_siamese_ranker.pt",
    )

    results = []
    for case in USER_TEST_CASES:
        result = evaluate_case(case, text_searcher, ui_searcher)
        results.append(result)

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

    print("=== USER SCENARIO EVAL ===")
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