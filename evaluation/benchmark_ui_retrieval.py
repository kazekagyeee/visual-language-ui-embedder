# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from pathlib import Path

from rag.ui_element_searcher import UIElementSearcher


def normalize_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_match(pred_text, target_text):
    pred = normalize_text(pred_text)
    target = normalize_text(target_text)

    return target in pred or pred in target


def is_correct(result, target_text, target_pages=None):
    item = result["item"]

    if not text_match(item.get("text", ""), target_text):
        return False

    if target_pages:
        return item.get("page") in target_pages

    return True


def find_rank(results, target_text, target_pages=None):
    for idx, result in enumerate(results, start=1):
        if is_correct(result, target_text, target_pages):
            return idx

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/test_queries.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-json", default="reports/ui_retrieval_metrics.json")
    parser.add_argument("--out-csv", default="reports/ui_retrieval_metrics.csv")
    args = parser.parse_args()

    with open(args.queries, "r", encoding="utf-8-sig") as f:
        queries = json.load(f)

    searcher = UIElementSearcher()

    rows = []

    top1 = 0
    top3 = 0
    top5 = 0
    topk = 0
    reciprocal_ranks = []
    ranks = []

    for query_item in queries:
        query = query_item["query"]
        target_text = query_item["target_text"]

        target_pages = query_item.get("target_pages")

        if target_pages is None and query_item.get("target_page") is not None:
            target_pages = [query_item["target_page"]]

        results = searcher.search(query, top_k=args.top_k)

        rank = find_rank(results, target_text, target_pages)

        ok_top1 = rank is not None and rank <= 1
        ok_top3 = rank is not None and rank <= 3
        ok_top5 = rank is not None and rank <= 5
        ok_topk = rank is not None and rank <= args.top_k

        top1 += int(ok_top1)
        top3 += int(ok_top3)
        top5 += int(ok_top5)
        topk += int(ok_topk)

        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
            ranks.append(rank)

        best = results[0]["item"] if results else {}

        row = {
            "query": query,
            "target_text": target_text,
            "target_pages": target_pages,
            "rank": rank,
            "top1": ok_top1,
            "top3": ok_top3,
            "top5": ok_top5,
            f"top{args.top_k}": ok_topk,
            "best_prediction": best.get("text"),
            "best_page": best.get("page"),
            "best_bbox": best.get("bbox"),
            "best_score": results[0]["score"] if results else None,
            "candidates": [
                {
                    "rank": i,
                    "text": r["item"].get("text"),
                    "page": r["item"].get("page"),
                    "bbox": r["item"].get("bbox"),
                    "score": r.get("score"),
                    "siamese_score": r.get("siamese_score"),
                    "ui_type": r["item"].get("ui_type"),
                }
                for i, r in enumerate(results, start=1)
            ],
        }

        rows.append(row)

    n = max(len(queries), 1)

    metrics = {
        "num_queries": len(queries),
        "top1_accuracy": top1 / n,
        "top3_accuracy": top3 / n,
        "top5_accuracy": top5 / n,
        f"top{args.top_k}_accuracy": topk / n,
        "mrr": sum(reciprocal_ranks) / n,
        "mean_rank_found_only": sum(ranks) / max(len(ranks), 1),
        "found_queries": len(ranks),
        "not_found_queries": len(queries) - len(ranks),
    }

    report = {
        "metrics": metrics,
        "rows": rows,
    }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query",
                "target_text",
                "target_pages",
                "rank",
                "top1",
                "top3",
                "top5",
                f"top{args.top_k}",
                "best_prediction",
                "best_page",
                "best_score",
            ],
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                "query": row["query"],
                "target_text": row["target_text"],
                "target_pages": row["target_pages"],
                "rank": row["rank"],
                "top1": row["top1"],
                "top3": row["top3"],
                "top5": row["top5"],
                f"top{args.top_k}": row[f"top{args.top_k}"],
                "best_prediction": row["best_prediction"],
                "best_page": row["best_page"],
                "best_score": row["best_score"],
            })

    print("=" * 80)
    print("UI RETRIEVAL METRICS")
    print("=" * 80)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nSaved:")
    print(out_json)
    print(out_csv)

    print("\nDetails:")
    for row in rows:
        print("-" * 80)
        print(f"query: {row['query']}")
        print(f"target: {row['target_text']}")
        print(f"rank: {row['rank']}")
        print(f"best: {row['best_prediction']} / page={row['best_page']} / score={row['best_score']}")


if __name__ == "__main__":
    main()
