# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/ui_index")
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    items_path = index_dir / "ui_items.jsonl"

    if not items_path.exists():
        raise FileNotFoundError(items_path)

    items = load_jsonl(items_path)

    print(f"UI items: {len(items)}")

    model = SentenceTransformer(args.model)

    texts = [
        f"{item.get('text', '')} {item.get('ui_type', '')} страница {item.get('page', '')}"
        for item in items
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    np.save(index_dir / "ui_embeddings.npy", np.asarray(embeddings, dtype=np.float32))

    print("[DONE] ui_embeddings.npy")


if __name__ == "__main__":
    main()
