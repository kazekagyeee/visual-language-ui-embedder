# -*- coding: utf-8 -*-

import argparse

from rag.visual_search import VisualDescriptionSearcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    args = parser.parse_args()

    searcher = VisualDescriptionSearcher(rag_dir=args.rag_dir)
    count = searcher.build_index()

    print(f"Visual description index built. Items: {count}")


if __name__ == "__main__":
    main()
