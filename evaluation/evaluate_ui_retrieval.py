# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path

from rag.answer_engine import AnswerEngine
from rag.hybrid_search import HybridSearcher
from rag.trained_ui_searcher import TrainedUIElementSearcher
from rag.ui_reranker import build_ui_semantic_results
from rag.ocr_cleanup import cleanup_ocr_text


TEST_CASES = [
    {
        "query": "где найти входной контроль",
        "expected": ["Входной контроль"],
        "expected_pdf": "instruction.pdf",
    },
    {
        "query": "как создать заявку на контроль",
        "expected": ["Входной контроль", "АРМ Входной контроль", "Заявки на контроль", "Создать"],
        "expected_pdf": "instruction.pdf",
    },
    {
        "query": "где находится арм входной контроль",
        "expected": ["АРМ Входной контроль"],
        "expected_pdf": "instruction.pdf",
    },
    {
        "query": "где найти заявки на контроль",
        "expected": ["Заявки на контроль"],
        "expected_pdf": "instruction.pdf",
    },
    {
        "query": "как открыть показатели контроля",
        "expected": ["Показатели контроля"],
        "expected_pdf": "instruction.pdf",
    },
    {
        "query": "где найти монитор интернет поддержки",
        "expected": ["Монитор Интернет-поддержки"],
        "expected_pdf": "services_1c.pdf",
    },
    {
        "query": "как подключить интернет поддержку",
        "expected": ["Интернет-поддержка пользователей", "Подключить Интернет-поддержку"],
        "expected_pdf": "services_1c.pdf",
    },
    {
        "query": "как создать нового контрагента",
        "expected": ["Контрагенты", "Создать", "ИНН"],
        "expected_pdf": "services_1c.pdf",
    },
    {
        "query": "как заполнить реквизиты контрагента по инн",
        "expected": ["Контрагенты", "Создать", "ИНН"],
        "expected_pdf": "services_1c.pdf",
    },
    {
        "query": "где найти контрагентов",
        "expected": ["Контрагенты"],
        "expected_pdf": "services_1c.pdf",
    },
    {
        "query": "где находится поле инн контрагента",
        "expected": ["ИНН"],
        "expected_pdf": "services_1c.pdf",
    },
    {
        "query": "как создать организацию",
        "expected": ["Организации", "Создать"],
        "expected_pdf": "services_1c.pdf",
    },
]


def normalize(text):
    return cleanup_ocr_text(text).lower().replace("ё", "е").strip()


def page_window(page, before=3, after=5):
    page = int(page)
    return list(range(max(1, page - before), page + after + 1))


def is_match(expected, found):
    e = normalize(expected)
    f = normalize(found)

    if not e or not f:
        return False

    if e == f:
        return True

    if e in f or f in e:
        return True

    e_words = set(e.split())
    f_words = set(f.split())

    if not e_words or not f_words:
        return False

    return len(e_words & f_words) / max(1, len(e_words)) >= 0.65


def precision_recall_f1(expected, found):
    if not found:
        precision = 0.0
    else:
        precision = sum(any(is_match(exp, f) for exp in expected) for f in found) / len(found)

    if not expected:
        recall = 0.0
    else:
        recall = sum(any(is_match(exp, f) for f in found) for exp in expected) / len(expected)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def mrr(expected, found):
    for idx, item in enumerate(found, start=1):
        if any(is_match(exp, item) for exp in expected):
            return 1.0 / idx
    return 0.0


def hit_at_k(expected, found, k):
    top = found[:k]
    return any(any(is_match(exp, item) for item in top) for exp in expected)


def evaluate_case(case, text_searcher, ui_searcher):
    query = case["query"]

    text_results = text_searcher.search(
        query=query,
        top_k=7,
        alpha=0.15,
    )

    response = AnswerEngine().build_response(query, text_results)

    raw_ui = ui_searcher.search(
        query=query,
        targets=response.get("targets", []),
        page_filter=page_window(response["page"]),
        pdf_filter=response["pdf_name"],
        top_k=100,
    )

    final_ui = build_ui_semantic_results(
        query=query,
        response=response,
        results=raw_ui,
        limit=8,
    )

    found = [
        cleanup_ocr_text(r["item"].get("text", ""))
        for r in final_ui
    ]

    expected = case["expected"]

    precision, recall, f1 = precision_recall_f1(expected, found)

    source_pdf = response.get("pdf_name")
    expected_pdf = case.get("expected_pdf")
    pdf_ok = source_pdf == expected_pdf if expected_pdf else None

    return {
        "query": query,
        "source": response.get("source"),
        "source_pdf": source_pdf,
        "expected_pdf": expected_pdf,
        "pdf_ok": pdf_ok,
        "source_page": response.get("page"),
        "targets": response.get("targets", []),
        "expected": expected,
        "found": found,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mrr": round(mrr(expected, found), 4),
        "hit_at_1": hit_at_k(expected, found, 1),
        "hit_at_3": hit_at_k(expected, found, 3),
        "hit_at_5": hit_at_k(expected, found, 5),
        "success": recall == 1.0,
    }


def write_csv(path, rows):
    fieldnames = [
        "query",
        "source",
        "source_pdf",
        "expected_pdf",
        "pdf_ok",
        "expected",
        "found",
        "precision",
        "recall",
        "f1",
        "mrr",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "success",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            r = dict(row)
            r["expected"] = " | ".join(row["expected"])
            r["found"] = " | ".join(row["found"])
            writer.writerow({k: r.get(k) for k in fieldnames})


def write_markdown(path, summary):
    lines = []

    lines.append("# UI Retrieval Evaluation")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Cases | {summary['cases']} |")
    lines.append(f"| Success Rate | {summary['success_rate']} |")
    lines.append(f"| Mean Precision | {summary['mean_precision']} |")
    lines.append(f"| Mean Recall | {summary['mean_recall']} |")
    lines.append(f"| Mean F1 | {summary['mean_f1']} |")
    lines.append(f"| Mean MRR | {summary['mean_mrr']} |")
    lines.append(f"| Hit@1 | {summary['hit_at_1']} |")
    lines.append(f"| Hit@3 | {summary['hit_at_3']} |")
    lines.append(f"| Hit@5 | {summary['hit_at_5']} |")
    lines.append(f"| PDF Accuracy | {summary['pdf_accuracy']} |")

    lines.append("")
    lines.append("## Details")
    lines.append("")
    lines.append("| Query | Expected | Found | Precision | Recall | Success |")
    lines.append("|---|---|---|---:|---:|---|")

    for r in summary["results"]:
        lines.append(
            f"| {r['query']} | "
            f"{', '.join(r['expected'])} | "
            f"{', '.join(r['found'])} | "
            f"{r['precision']} | "
            f"{r['recall']} | "
            f"{r['success']} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def mean(values):
    values = list(values)
    return sum(values) / max(1, len(values))


def main():
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    text_searcher = HybridSearcher(rag_dir="data/all_pdf_rag")
    ui_searcher = TrainedUIElementSearcher(
        index_dir="data/ui_trained_index",
        checkpoint="checkpoints/ui_siamese_ranker.pt",
    )

    results = []

    for case in TEST_CASES:
        result = evaluate_case(case, text_searcher, ui_searcher)
        results.append(result)

    summary = {
        "cases": len(results),
        "success_rate": round(mean(r["success"] for r in results), 4),
        "mean_precision": round(mean(r["precision"] for r in results), 4),
        "mean_recall": round(mean(r["recall"] for r in results), 4),
        "mean_f1": round(mean(r["f1"] for r in results), 4),
        "mean_mrr": round(mean(r["mrr"] for r in results), 4),
        "hit_at_1": round(mean(r["hit_at_1"] for r in results), 4),
        "hit_at_3": round(mean(r["hit_at_3"] for r in results), 4),
        "hit_at_5": round(mean(r["hit_at_5"] for r in results), 4),
        "pdf_accuracy": round(mean(r["pdf_ok"] for r in results if r["pdf_ok"] is not None), 4),
        "results": results,
    }

    with open(out_dir / "ui_retrieval_eval.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_csv(out_dir / "ui_retrieval_eval.csv", results)
    write_markdown(out_dir / "ui_retrieval_eval.md", summary)

    print("=== SUMMARY ===")
    for k, v in summary.items():
        if k != "results":
            print(f"{k}: {v}")

    print("\n=== DETAILS ===")
    for r in results:
        print()
        print("QUERY:", r["query"])
        print("EXPECTED:", r["expected"])
        print("FOUND:", r["found"])
        print("PRECISION:", r["precision"], "RECALL:", r["recall"], "F1:", r["f1"], "SUCCESS:", r["success"])

    print("\nSaved:")
    print("reports/ui_retrieval_eval.json")
    print("reports/ui_retrieval_eval.csv")
    print("reports/ui_retrieval_eval.md")


if __name__ == "__main__":
    main()
