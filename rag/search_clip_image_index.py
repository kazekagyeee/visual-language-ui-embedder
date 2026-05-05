# -*- coding: utf-8 -*-

import argparse

from rag.clip_search import ClipImageSearcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--model", default="clip-ViT-B-32")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    searcher = ClipImageSearcher(
        rag_dir=args.rag_dir,
        model_name=args.model,
    )

    results = searcher.search(args.query, args.top_k)

    for result in results:
        item = result["item"]

        print("=" * 100)
        print(f"score={result['score']:.4f}")
        print(f"page={item['page']} block={item['block_id']}")
        print(f"crop={item['crop_image']}")
        print(item["text"])


if __name__ == "__main__":
    main()
