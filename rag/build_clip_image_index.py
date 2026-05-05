# -*- coding: utf-8 -*-

import argparse

from rag.clip_search import ClipImageSearcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--model", default="clip-ViT-B-32")
    args = parser.parse_args()

    searcher = ClipImageSearcher(
        rag_dir=args.rag_dir,
        model_name=args.model,
    )

    count = searcher.build_index()
    print(f"CLIP image index built. Items: {count}")


if __name__ == "__main__":
    main()
