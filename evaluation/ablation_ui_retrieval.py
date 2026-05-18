# -*- coding: utf-8 -*-

import argparse
import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from rag.ocr_cleaning import normalize_ocr_text
from rag.ui_element_searcher import UIElementSearcher


SEMANTIC_ALIASES = {
    "добавить": [
        "добавить",
        "новый элемент",
        "создать новую запись",
        "добавление",
    ],
    "госты": [
        "госты",
        "нормативы",
        "нормативные документы",
        "стандарты",
        "требования",
    ],
    "показатели контроля": [
        "показатели контроля",
        "параметры проверки",
        "требования контроля",
        "характеристики контроля",
    ],
    "заявки на контроль": [
        "заявки на контроль",
        "созданные проверки",
        "заявки проверки",
        "проверки качества",
    ],
    "виды контроля": [
        "виды контроля",
        "типы проверок",
        "список типов контроля",
    ],
    "группы прочности": [
        "группы прочности",
        "прочность материалов",
        "справочник прочности",
    ],
    "входной контроль": [
        "входной контроль",
        "контроль качества",
        "раздел контроля",
    ],
    "выполнения входного контроля": [
        "выполнения входного контроля",
        "выполненные проверки",
        "результаты проверок",
    ],
    "акты входного контроля": [
        "акты входного контроля",
        "акты проверки",
        "документы проверки",
    ],
}


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def item_norm(item):
    return item.get("normalized_text") or normalize_ocr_text(item.get("text", ""))


def target_match(item, target):
    target_norm = target.get("normalized_text") or normalize_ocr_text(target["text"])
    target_pages = target.get("target_pages") or []

    p_norm = item_norm(item)

    text_ok = (
        p_norm == target_norm
        or target_norm in p_norm
        or p_norm in target_norm
    )

    page_ok = not target_pages or item.get("page") in target_pages

    return text_ok and page_ok


def evaluate_predictions(targets, predictions):
    hits = 0
    rr_sum = 0.0

    for target in targets:
        rank = None

        for i, pred in enumerate(predictions, start=1):
            if target_match(pred["item"], target):
                rank = i
                break

        if rank is not None:
            hits += 1
            rr_sum += 1.0 / rank

    recall = hits / max(1, len(targets))

    return {
        "all_targets_exact": hits == len(targets),
        "target_recall": recall,
        "mrr": rr_sum / max(1, len(targets)),
    }


def ocr_only_full_query(query, items, top_k):
    q_norm = normalize_ocr_text(query)
    q_tokens = set(q_norm.split())

    results = []

    for item in items:
        t_norm = item_norm(item)
        t_tokens = set(t_norm.split())

        score = 0.0

        if q_norm == t_norm:
            score += 10.0
        elif q_norm in t_norm or t_norm in q_norm:
            score += 5.0

        if q_tokens:
            score += len(q_tokens & t_tokens) / len(q_tokens)

        results.append({
            "score": score,
            "item": item,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def semantic_alias_search(query, items, top_k):
    q_norm = normalize_ocr_text(query)
    q_tokens = set(q_norm.split())

    results = []

    for item in items:
        t_norm = item_norm(item)

        score = 0.0

        aliases = SEMANTIC_ALIASES.get(t_norm, [])

        for alias in aliases:
            alias_norm = normalize_ocr_text(alias)
            alias_tokens = set(alias_norm.split())

            if alias_norm in q_norm or q_norm in alias_norm:
                score += 5.0

            if q_tokens:
                score += len(q_tokens & alias_tokens) / len(q_tokens)

        results.append({
            "score": score,
            "item": item,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def sentence_transformer_search(query, items, model, top_k):
    texts = [
        item.get("text", "")
        for item in items
    ]

    query_vec = model.encode([query], normalize_embeddings=True)[0]
    text_vecs = model.encode(texts, normalize_embeddings=True)

    results = []

    for item, vec in zip(items, text_vecs):
        score = float(query_vec @ vec)

        results.append({
            "score": score,
            "item": item,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def siamese_search(query, searcher, top_k):
    return searcher.search(query, top_k=top_k)


def evaluate_method(name, queries, search_fn, top_k):
    started = time.perf_counter()

    rows = []
    all_acc = 0
    recalls = []
    mrrs = []

    for q in queries:
        predictions = search_fn(q, top_k)
        ev = evaluate_predictions(q["targets"], predictions)

        all_acc += int(ev["all_targets_exact"])
        recalls.append(ev["target_recall"])
        mrrs.append(ev["mrr"])

        rows.append({
            "query": q["query"],
            "type": q.get("type"),
            "targets": q["targets"],
            "predictions": [
                {
                    "text": p["item"].get("text"),
                    "normalized_text": item_norm(p["item"]),
                    "page": p["item"].get("page"),
                    "score": p.get("score"),
                    "final_score": p.get("final_score"),
                    "siamese_score": p.get("siamese_score"),
                }
                for p in predictions
            ],
            **ev,
        })

    elapsed = time.perf_counter() - started
    n = max(1, len(queries))

    return {
        "method": name,
        "queries": len(queries),
        "all_targets_accuracy": all_acc / n,
        "mean_target_recall": sum(recalls) / n,
        "mean_mrr": sum(mrrs) / n,
        "avg_time_sec_per_query": elapsed / n,
        "details": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/semantic_test_queries.json")
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", default="reports/ablation_semantic_ui_retrieval.json")
    args = parser.parse_args()

    queries = load_json(args.queries)
    items = load_jsonl(Path(args.rag_dir) / "ui_elements.jsonl")

    searcher = UIElementSearcher()
    st_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    reports = []

    reports.append(evaluate_method(
        "ocr_only_full_query",
        queries,
        lambda q, top_k: ocr_only_full_query(q["query"], items, top_k),
        args.top_k,
    ))

    reports.append(evaluate_method(
        "semantic_alias_baseline",
        queries,
        lambda q, top_k: semantic_alias_search(q["query"], items, top_k),
        args.top_k,
    ))

    reports.append(evaluate_method(
        "sentence_transformer_text",
        queries,
        lambda q, top_k: sentence_transformer_search(q["query"], items, st_model, top_k),
        args.top_k,
    ))

    reports.append(evaluate_method(
        "siamese_hybrid",
        queries,
        lambda q, top_k: siamese_search(q["query"], searcher, top_k),
        args.top_k,
    ))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8-sig") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("SEMANTIC ABLATION STUDY")
    print("=" * 80)

    for r in reports:
        print(r["method"])
        print(f"  all_targets_accuracy: {r['all_targets_accuracy']:.4f}")
        print(f"  mean_target_recall:   {r['mean_target_recall']:.4f}")
        print(f"  mean_mrr:             {r['mean_mrr']:.4f}")
        print(f"  avg_time_sec/query:   {r['avg_time_sec_per_query']:.4f}")
        print()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
