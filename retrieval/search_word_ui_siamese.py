# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from models.siamese_ui_encoder import SiameseUIEncoder


def load_index(index_dir):
    index_dir = Path(index_dir)

    items = []
    with open(index_dir / "items.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    embeddings = np.load(index_dir / "embeddings.npy")
    return items, embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/ui_siamese_words/best.pt")
    parser.add_argument("--index-dir", default="indexes/word_ui_siamese")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--text-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiameseUIEncoder.load(args.checkpoint, map_location=device).to(device)
    model.eval()

    text_model = SentenceTransformer(args.text_model)

    items, embeddings = load_index(args.index_dir)

    query_vec = text_model.encode(
        [args.query],
        normalize_embeddings=True,
    ).astype("float32")

    query_vec = torch.tensor(query_vec, dtype=torch.float32).to(device)

    with torch.no_grad():
        text_emb = model.encode_text(query_vec).cpu().numpy()[0]

    scores = embeddings @ text_emb
    top_ids = np.argsort(scores)[::-1][:args.top_k]

    for idx in top_ids:
        item = items[int(idx)]

        print("=" * 100)
        print(f"score={scores[idx]:.4f}")
        print(f"text={item['text']}")
        print(f"page={item['page']}")
        print(f"crop={item['image']}")
        print(f"bbox={item['bbox_px']}")


if __name__ == "__main__":
    main()
