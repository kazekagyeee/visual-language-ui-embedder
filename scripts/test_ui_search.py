# -*- coding: utf-8 -*-

import argparse

from rag.ui_element_searcher import UIElementSearcher
from rag.ui_reranker import build_ui_semantic_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/ui_index_test")
    parser.add_argument("--query", default="как создать заявку на контроль")
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--pdf", default="instruction.pdf")
    parser.add_argument("--window", type=int, default=2)
    args = parser.parse_args()

    response = {
        "targets": [
            "Входной контроль",
            "АРМ Входной контроль",
            "Заявки на контроль",
            "Создать",
        ],
        "steps": [
            "Откройте вкладку «Входной контроль».",
            "Перейдите в раздел «АРМ Входной контроль».",
            "Откройте пункт «Заявки на контроль».",
            "Нажмите «Создать».",
        ],
    }

    pages = list(range(args.page, args.page + args.window + 1))

    searcher = UIElementSearcher(index_dir=args.index_dir)

    results = searcher.search(
        query=args.query,
        targets=response["targets"],
        page_filter=pages,
        pdf_filter=args.pdf,
        top_k=40,
    )

    results = build_ui_semantic_results(
        query=args.query,
        response=response,
        results=results,
        limit=8,
    )

    print(f"QUERY: {args.query}")
    print(f"PAGES: {pages}")
    print()

    for i, r in enumerate(results, start=1):
        item = r["item"]
        print(
            f"{i}. {item.get('text')} | "
            f"type={item.get('ui_type')} | "
            f"page={item.get('page')} | "
            f"screenshot={item.get('screenshot_idx')} | "
            f"score={r.get('score', 0):.3f} | "
            f"semantic={r.get('semantic_score', 0):.3f}"
        )


if __name__ == "__main__":
    main()
