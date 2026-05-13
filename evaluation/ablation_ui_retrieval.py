# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

from rag.ui_element_searcher import UIElementSearcher
from rag.ocr_cleaning import normalize_ocr_text


def text_match(pred, target):
    pred = normalize_ocr_text(pred)
    target = normalize_ocr_text(target)
    return target in pred or pred in target


def is_correct(result, target_text):
    return text_match(result["item"].get("text", ""), target_text)


def ocr_only_search(query, items, top_k):
    q = normalize_ocr_text(query)
    q_tokens = set(q.split())

    results = []

    for item in items:
        t = normalize_ocr_text(item.get("text", ""))
        t_tokens = set(t.split())

        if not q_tokens:
            score = 0.0
        else:
            score = len(q_tokens & t_tokens) / len(q_tokens)

        if q in t or t in q:
            score += 1.0

        results.append({
            "score": score,
            "item": item,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def evaluate_method(name, queries, search_fn, top_k):
    top1 = top3 = topk = 0
    ranks = []

    details = []

    for q in queries:
        results = search_fn(q["query"], top_k)
        rank = None

        for i, r in enumerate(results, start=1):
            if is_correct(r, q["target_text"]):
                rank = i
                break

        top1 += int(rank is not None and rank <= 1)
        top3 += int(rank is not None and rank <= 3)
        topk += int(rank is not None and rank <= top_k)

        if rank is not None:
            ranks.append(rank)

        details.append({
            "query": q["query"],
            "target": q["target_text"],
            "rank": rank,
            "best": results[0]["item"].get("text") if results else None,
            "best_page": results[0]["item"].get("page") if results else None,
        })

    n = max(len(queries), 1)

    return {
        "method": name,
        "top1": top1 / n,
        "top3": top3 / n,
        f"top{top_k}": topk / n,
        "mrr": sum(1 / r for r in ranks) / n if ranks else 0.0,
        "mean_rank_found_only": sum(ranks) / len(ranks) if ranks else None,
        "found": len(ranks),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/test_queries.json")
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", default="reports/ablation_ui_retrieval.json")
    args = parser.parse_args()

    with open(args.queries, "r", encoding="utf-8-sig") as f:
        queries = json.load(f)

    items = []
    with open(Path(args.rag_dir) / "ui_elements.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    siamese_searcher = UIElementSearcher()

    reports = []

    reports.append(evaluate_method(
        "ocr_only",
        queries,
        lambda query, top_k: ocr_only_search(query, items, top_k),
        args.top_k,
    ))

    reports.append(evaluate_method(
        "siamese_hybrid",
        queries,
        lambda query, top_k: siamese_searcher.search(query, top_k=top_k),
        args.top_k,
    ))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("ABLATION STUDY")
    print("=" * 80)

    for report in reports:
        print(report["method"])
        print(f"Top-1: {report['top1']:.4f}")
        print(f"Top-3: {report['top3']:.4f}")
        print(f"Top-{args.top_k}: {report[f'top{args.top_k}']:.4f}")
        print(f"MRR: {report['mrr']:.4f}")
        print()

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
