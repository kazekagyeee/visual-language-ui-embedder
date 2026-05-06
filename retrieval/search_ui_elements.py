# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from models.siamese_ui_encoder import SiameseUIEncoder


def normalize_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_index(index_dir):
    index_dir = Path(index_dir)

    items = []
    with open(index_dir / "items.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    embeddings = np.load(index_dir / "embeddings.npy")
    return items, embeddings


def lexical_bonus(query, text):
    q = normalize_text(query)
    t = normalize_text(text)

    if q == t:
        return 1.0

    if q in t or t in q:
        return 0.6

    q_tokens = set(q.split())
    t_tokens = set(t.split())

    if not q_tokens:
        return 0.0

    return len(q_tokens & t_tokens) / len(q_tokens) * 0.4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/ui_elements_siamese/best.pt")
    parser.add_argument("--index-dir", default="indexes/ui_elements_siamese")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--text-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiameseUIEncoder.load(args.checkpoint, map_location=device).to(device)
    model.eval()

    text_model = SentenceTransformer(args.text_model)
    items, embeddings = load_index(args.index_dir)

    query_vec = text_model.encode([args.query], normalize_embeddings=True).astype("float32")
    query_vec = torch.tensor(query_vec, dtype=torch.float32).to(device)

    with torch.no_grad():
        text_emb = model.encode_text(query_vec).cpu().numpy()[0]

    siamese_scores = embeddings @ text_emb

    final_scores = []

    for score, item in zip(siamese_scores, items):
        bonus = lexical_bonus(args.query, item["text"])
        final_scores.append(float(score) + bonus)

    final_scores = np.array(final_scores)
    top_ids = np.argsort(final_scores)[::-1][:args.top_k]

    for idx in top_ids:
        item = items[int(idx)]

        print("=" * 100)
        print(f"final_score={final_scores[idx]:.4f}")
        print(f"siamese_score={siamese_scores[idx]:.4f}")
        print(f"text={item['text']}")
        print(f"page={item['page']}")
        print(f"bbox={item['bbox']}")
        print(f"crop={item['crop_image']}")


if __name__ == "__main__":
    main()
