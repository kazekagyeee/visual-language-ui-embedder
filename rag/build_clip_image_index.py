# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import clip


def load_items(rag_dir: Path):
    items = []
    with open(rag_dir / "items.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--out", default="clip_embeddings.npy")
    args = parser.parse_args()

    rag_dir = Path(args.rag_dir)
    out_path = rag_dir / args.out

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    items = load_items(rag_dir)

    embeddings = []
    kept_items = []

    for item in items:
        crop_path = Path(item["crop_image"])

        if not crop_path.exists():
            continue

        image = Image.open(crop_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.encode_image(image_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        embeddings.append(emb.cpu().numpy()[0])
        kept_items.append(item)

    np.save(out_path, np.array(embeddings, dtype=np.float32))

    with open(rag_dir / "clip_items.jsonl", "w", encoding="utf-8") as f:
        for item in kept_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved CLIP embeddings: {out_path}")
    print(f"Items: {len(kept_items)}")


if __name__ == "__main__":
    main()
