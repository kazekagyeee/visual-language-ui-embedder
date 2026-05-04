# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def load_items(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()

    rag_dir = Path(args.rag_dir)

    items = load_items(rag_dir / "items.jsonl")
    embeddings = np.load(rag_dir / "embeddings.npy")

    model = SentenceTransformer(args.model)
    query_vec = model.encode([args.query], normalize_embeddings=True)[0]

    scores = embeddings @ query_vec
    top_ids = np.argsort(scores)[::-1][:args.top_k]

    for idx in top_ids:
        item = items[int(idx)]
        print("=" * 100)
        print(f"score={scores[idx]:.4f}")
        print(f"page={item['page']} block={item['block_id']}")
        print(f"crop={item['crop_image']}")
        print(item["text"])


if __name__ == "__main__":
    main()
