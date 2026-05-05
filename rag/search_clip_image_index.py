# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import clip


def load_items(path: Path):
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
    args = parser.parse_args()

    rag_dir = Path(args.rag_dir)

    items = load_items(rag_dir / "clip_items.jsonl")
    embeddings = np.load(rag_dir / "clip_embeddings.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    text_tokens = clip.tokenize([args.query]).to(device)

    with torch.no_grad():
        text_emb = model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    text_emb = text_emb.cpu().numpy()[0]

    scores = embeddings @ text_emb
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
