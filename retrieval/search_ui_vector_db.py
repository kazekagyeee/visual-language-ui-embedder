# -*- coding: utf-8 -*-

import argparse

from rag.ui_vector_searcher import UIVectorSearcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--checkpoint", default="checkpoints/ui_elements_siamese/best.pt")
    parser.add_argument("--db-dir", default="vector_db/ui_elements")
    args = parser.parse_args()

    searcher = UIVectorSearcher(
        checkpoint=args.checkpoint,
        db_dir=args.db_dir,
    )

    results = searcher.search(args.query, top_k=args.top_k)

    for i, result in enumerate(results, start=1):
        item = result["item"]

        print("=" * 100)
        print(f"{i}. score={result['score']:.4f}")
        print(f"vector_score={result['vector_score']:.4f}")
        print(f"text={item['text']}")
        print(f"page={item['page']}")
        print(f"bbox={item['bbox']}")
        print(f"crop={item['crop_image']}")


if __name__ == "__main__":
    main()
