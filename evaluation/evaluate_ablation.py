# -*- coding: utf-8 -*-

import json
from pathlib import Path

from rag.answer_engine import AnswerEngine
from rag.hybrid_search import HybridSearcher
from rag.ui_element_searcher import UIElementSearcher
from rag.trained_ui_searcher import TrainedUIElementSearcher
from rag.ui_reranker import build_ui_semantic_results
from rag.ocr_cleanup import cleanup_ocr_text


TEST_CASES = [
    {"query": "где найти входной контроль", "expected": ["Входной контроль"]},
    {"query": "как создать заявку на контроль", "expected": ["Входной контроль", "АРМ Входной контроль", "Заявки на контроль", "Создать"]},
    {"query": "где найти монитор интернет поддержки", "expected": ["Монитор Интернет-поддержки"]},
    {"query": "как создать нового контрагента", "expected": ["Контрагенты", "Создать", "ИНН"]},
    {"query": "как заполнить реквизиты контрагента по инн", "expected": ["Контрагенты", "Создать", "ИНН"]},
]


def normalize(text):
    return cleanup_ocr_text(text).lower().replace("ё", "е").strip()


def is_match(expected, found):
    e = normalize(expected)
    f = normalize(found)

    if e == f or e in f or f in e:
        return True

    ew = set(e.split())
    fw = set(f.split())

    if not ew or not fw:
        return False

    return len(ew & fw) / max(1, len(ew)) >= 0.65


def page_window(page):
    page = int(page)
    return list(range(max(1, page - 3), page + 6))


def score_case(expected, found):
    precision = sum(any(is_match(e, f) for e in expected) for f in found) / max(1, len(found))
    recall = sum(any(is_match(e, f) for f in found) for e in expected) / max(1, len(expected))

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def run_config(name, ui_searcher):
    text_searcher = HybridSearcher(rag_dir="data/all_pdf_rag")
    answer_engine = AnswerEngine()

    rows = []

    for case in TEST_CASES:
        query = case["query"]

        text_results = text_searcher.search(query=query, top_k=7, alpha=0.15)
        response = answer_engine.build_response(query, text_results)

        raw = ui_searcher.search(
            query=query,
            targets=response.get("targets", []),
            page_filter=page_window(response["page"]),
            pdf_filter=response["pdf_name"],
            top_k=100,
        )

        final = build_ui_semantic_results(
            query=query,
            response=response,
            results=raw,
            limit=8,
        )

        found = [
            cleanup_ocr_text(r["item"].get("text", ""))
            for r in final
        ]

        p, r, f1 = score_case(case["expected"], found)

        rows.append(
            {
                "query": query,
                "expected": case["expected"],
                "found": found,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "success": r == 1.0,
            }
        )

    return {
        "name": name,
        "precision": round(sum(x["precision"] for x in rows) / len(rows), 4),
        "recall": round(sum(x["recall"] for x in rows) / len(rows), 4),
        "f1": round(sum(x["f1"] for x in rows) / len(rows), 4),
        "success_rate": round(sum(x["success"] for x in rows) / len(rows), 4),
        "cases": rows,
    }


def main():
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    configs = []

    configs.append(
        run_config(
            "base_ui_index",
            UIElementSearcher(index_dir="data/ui_index"),
        )
    )

    configs.append(
        run_config(
            "trained_siamese_index",
            TrainedUIElementSearcher(
                index_dir="data/ui_trained_index",
                checkpoint="checkpoints/ui_siamese_ranker.pt",
            ),
        )
    )

    summary = {
        "configs": configs
    }

    with open(out_dir / "ablation_eval.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== ABLATION ===")
    for cfg in configs:
        print()
        print(cfg["name"])
        print("precision:", cfg["precision"])
        print("recall:", cfg["recall"])
        print("f1:", cfg["f1"])
        print("success_rate:", cfg["success_rate"])

    print("\nSaved: reports/ablation_eval.json")


if __name__ == "__main__":
    main()
