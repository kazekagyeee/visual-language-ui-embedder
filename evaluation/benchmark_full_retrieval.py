# -*- coding: utf-8 -*-

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from rag.hybrid_search import HybridSearcher
from rag.multi_query import split_query_to_ui_phrases
from rag.ocr_cleaning import normalize_ocr_text
from rag.ui_element_searcher import UIElementSearcher


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def item_norm(item):
    return item.get("normalized_text") or normalize_ocr_text(item.get("text", ""))


def get_known_ui_phrases(ui_searcher):
    return sorted({item["text"] for item in ui_searcher.items}) if ui_searcher.items else []


def score_candidate(phrase, result):
    phrase_norm = normalize_ocr_text(phrase)
    item = result["item"]
    text_norm = item_norm(item)

    score = float(result.get("final_score", result.get("score", 0.0)))

    if text_norm == phrase_norm:
        score += 10.0
    elif phrase_norm in text_norm or text_norm in phrase_norm:
        score += 4.0

    if item.get("ui_type") in {"button", "hyperlink", "sidebar_item", "tab", "input"}:
        score += 1.0

    return score


def search_for_query(query, text_searcher, ui_searcher, top_k_text, top_k_ui, alpha):
    text_results = text_searcher.search(query, top_k=top_k_text, alpha=alpha)
    text_pages = {r["item"]["page"] for r in text_results}

    known = get_known_ui_phrases(ui_searcher)
    phrases = split_query_to_ui_phrases(query, known)

    phrase_to_candidates = {}

    for phrase in phrases:
        phrase_norm = normalize_ocr_text(phrase)

        if not phrase_norm:
            continue

        results = ui_searcher.search(phrase, top_k=120)

        filtered = []
        for r in results:
            text_norm = item_norm(r["item"])

            if text_norm == phrase_norm or phrase_norm in text_norm or text_norm in phrase_norm:
                rr = dict(r)
                rr["matched_query"] = phrase
                rr["_eval_score"] = score_candidate(phrase, rr)
                filtered.append(rr)

        if not filtered:
            filtered = []
            for r in results[:top_k_ui]:
                rr = dict(r)
                rr["matched_query"] = phrase
                rr["_eval_score"] = score_candidate(phrase, rr)
                filtered.append(rr)

        phrase_to_candidates[phrase] = filtered

    page_map = {}

    for phrase, candidates in phrase_to_candidates.items():
        for r in candidates:
            page = r["item"]["page"]
            page_map.setdefault(page, {})
            page_map[page].setdefault(phrase, [])
            page_map[page][phrase].append(r)

    if not page_map:
        return text_results, [], phrases

    best_page = None
    best_key = None

    for page, by_phrase in page_map.items():
        coverage = len(by_phrase)
        text_page_bonus = 1 if page in text_pages else 0
        score_sum = 0.0

        for phrase, candidates in by_phrase.items():
            candidates.sort(key=lambda r: r["_eval_score"], reverse=True)
            score_sum += candidates[0]["_eval_score"]

        key = (coverage, text_page_bonus, score_sum)

        if best_key is None or key > best_key:
            best_key = key
            best_page = page

    chosen = []

    for phrase, candidates in phrase_to_candidates.items():
        same_page = [r for r in candidates if r["item"]["page"] == best_page]

        if same_page:
            same_page.sort(key=lambda r: r["_eval_score"], reverse=True)
            chosen.append(same_page[0])

    chosen.sort(key=lambda r: r["_eval_score"], reverse=True)

    return text_results, chosen[:top_k_ui], phrases


def evaluate_prediction(targets, predictions):
    target_hits = []

    for target in targets:
        t_norm = target.get("normalized_text") or normalize_ocr_text(target["text"])
        target_pages = target.get("target_pages")

        if target_pages is None:
            if target.get("page") is not None:
                target_pages = [target["page"]]
            else:
                target_pages = []

        hit_text = False
        hit_page = False
        hit_exact = False
        rank = None

        for idx, r in enumerate(predictions, start=1):
            p_norm = item_norm(r["item"])
            p_page = r["item"]["page"]

            text_match = p_norm == t_norm or t_norm in p_norm or p_norm in t_norm
            page_match = not target_pages or p_page in target_pages

            if text_match:
                hit_text = True

            if page_match:
                hit_page = True

            if text_match and page_match:
                hit_exact = True
                rank = idx
                break

        target_hits.append({
            "target_text": target["text"],
            "target_norm": t_norm,
            "target_pages": target_pages,
            "hit_text": hit_text,
            "hit_page": hit_page,
            "hit_exact": hit_exact,
            "rank": rank,
        })

    all_exact = all(x["hit_exact"] for x in target_hits)
    any_exact = any(x["hit_exact"] for x in target_hits)
    recall = sum(1 for x in target_hits if x["hit_exact"]) / max(1, len(target_hits))

    reciprocal_ranks = [1.0 / x["rank"] for x in target_hits if x["rank"]]
    mrr = sum(reciprocal_ranks) / len(targets) if targets else 0.0

    return {
        "all_targets_exact": all_exact,
        "any_target_exact": any_exact,
        "target_recall": recall,
        "mrr": mrr,
        "target_hits": target_hits,
        "predictions": [
            {
                "text": r["item"].get("text"),
                "normalized_text": item_norm(r["item"]),
                "page": r["item"].get("page"),
                "ui_type": r["item"].get("ui_type"),
                "score": r.get("score"),
                "final_score": r.get("final_score"),
                "siamese_score": r.get("siamese_score"),
                "eval_score": r.get("_eval_score"),
            }
            for r in predictions
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--queries", default="data/test_queries.json")
    parser.add_argument("--checkpoint", default="checkpoints/ui_elements_siamese/best.pt")
    parser.add_argument("--index-dir", default="indexes/ui_elements_siamese")
    parser.add_argument("--top-k-text", type=int, default=5)
    parser.add_argument("--top-k-ui", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = load_json(args.queries)

    text_searcher = HybridSearcher(rag_dir=args.rag_dir)
    ui_searcher = UIElementSearcher(
        checkpoint=args.checkpoint,
        index_dir=args.index_dir,
    )

    details = []
    times = []

    for q in queries:
        start = time.perf_counter()

        text_results, ui_results, phrases = search_for_query(
            query=q["query"],
            text_searcher=text_searcher,
            ui_searcher=ui_searcher,
            top_k_text=args.top_k_text,
            top_k_ui=args.top_k_ui,
            alpha=args.alpha,
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        ev = evaluate_prediction(q["targets"], ui_results)

        details.append({
            "query": q["query"],
            "type": q.get("type", "unknown"),
            "targets": q["targets"],
            "phrases_detected": phrases,
            "time_sec": elapsed,
            **ev,
        })

    n = len(details)

    metrics = {
        "queries": n,
        "single_queries": sum(1 for x in details if x["type"] == "single"),
        "multi_queries": sum(1 for x in details if x["type"] == "multi"),
        "all_targets_accuracy": sum(1 for x in details if x["all_targets_exact"]) / max(1, n),
        "any_target_accuracy": sum(1 for x in details if x["any_target_exact"]) / max(1, n),
        "mean_target_recall": statistics.mean([x["target_recall"] for x in details]) if details else 0.0,
        "mean_mrr": statistics.mean([x["mrr"] for x in details]) if details else 0.0,
        "avg_time_sec": statistics.mean(times) if times else 0.0,
        "median_time_sec": statistics.median(times) if times else 0.0,
        "top_k_text": args.top_k_text,
        "top_k_ui": args.top_k_ui,
        "alpha": args.alpha,
    }

    with open(out_dir / "ui_retrieval_metrics.json", "w", encoding="utf-8-sig") as f:
        json.dump(            {
                "metrics": metrics,
                "details": details,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(out_dir / "ui_retrieval_metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "query",
            "type",
            "all_targets_exact",
            "target_recall",
            "mrr",
            "time_sec",
            "targets",
            "predictions",
        ])

        for row in details:
            writer.writerow([
                row["query"],
                row["type"],
                row["all_targets_exact"],
                row["target_recall"],
                row["mrr"],
                f"{row['time_sec']:.4f}",
                json.dumps(row["targets"], ensure_ascii=False),
                json.dumps(row["predictions"], ensure_ascii=False),
            ])

    with open(out_dir / "error_analysis.md", "w", encoding="utf-8") as f:
        f.write("# Error analysis\n\n")

        for row in details:
            if row["all_targets_exact"]:
                continue

            f.write(f"## Query: {row['query']}\n\n")
            f.write(f"- type: {row['type']}\n")
            f.write(f"- recall: {row['target_recall']:.4f}\n")
            f.write(f"- mrr: {row['mrr']:.4f}\n")
            f.write(f"- time_sec: {row['time_sec']:.4f}\n\n")
            f.write("### Targets\n\n")
            f.write("```json\n")
            f.write(json.dumps(row["targets"], ensure_ascii=False, indent=2))
            f.write("\n```\n\n")
            f.write("### Predictions\n\n")
            f.write("```json\n")
            f.write(json.dumps(row["predictions"], ensure_ascii=False, indent=2))
            f.write("\n```\n\n")

    print("=" * 80)
    print("FULL UI RETRIEVAL BENCHMARK")
    print("=" * 80)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("=" * 80)
    print(f"Saved: {out_dir / 'ui_retrieval_metrics.json'}")
    print(f"Saved: {out_dir / 'ui_retrieval_metrics.csv'}")
    print(f"Saved: {out_dir / 'error_analysis.md'}")


if __name__ == "__main__":
    main()

