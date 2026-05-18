# -*- coding: utf-8 -*-

import argparse

from rag.answer_engine import AnswerEngine
from rag.hybrid_search import HybridSearcher
from rag.ui_element_searcher import UIElementSearcher
from rag.ui_reranker import build_ui_semantic_results


def page_window(page, before=2, after=4):
    page = int(page)
    return list(range(max(1, page - before), page + after + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--rag-dir", default="data/all_pdf_rag")
    parser.add_argument("--ui-index-dir", default="data/ui_index")
    args = parser.parse_args()

    text_searcher = HybridSearcher(rag_dir=args.rag_dir)
    text_results = text_searcher.search(args.query, top_k=5, alpha=0.15)

    response = AnswerEngine().build_response(args.query, text_results)

    print("\n=== RESPONSE ===")
    print("source:", response["source"])
    print("pdf:", response["pdf_name"])
    print("page:", response["page"])
    print("targets:", response["targets"])
    print("steps:")
    for s in response["steps"]:
        print(" -", s)

    pages = page_window(response["page"])

    ui_searcher = UIElementSearcher(index_dir=args.ui_index_dir)

    raw = ui_searcher.search(
        query=args.query,
        targets=response.get("targets", []),
        page_filter=pages,
        pdf_filter=response.get("pdf_name"),
        top_k=80,
    )

    final = build_ui_semantic_results(
        query=args.query,
        response=response,
        results=raw,
        limit=8,
    )

    print("\n=== RAW UI TOP 20 ===")
    for i, r in enumerate(raw[:20], start=1):
        item = r["item"]
        print(
            f"{i}. {item.get('text')} | page={item.get('page')} "
            f"s={item.get('screenshot_idx')} type={item.get('ui_type')} "
            f"score={r.get('score'):.3f} target={r.get('target_score'):.3f}"
        )

    print("\n=== FINAL UI ===")
    for i, r in enumerate(final, start=1):
        item = r["item"]
        print(
            f"{i}. {item.get('text')} | matched={r.get('matched_target')} "
            f"page={item.get('page')} s={item.get('screenshot_idx')} "
            f"semantic={r.get('semantic_score'):.3f}"
        )


if __name__ == "__main__":
    main()
