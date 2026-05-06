# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.siamese_ui_encoder import SiameseUIEncoder


def load_positive_items(pairs_path):
    items = []
    seen = set()

    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            if int(row["label"]) != 1:
                continue

            key = row["image"]
            if key in seen:
                continue

            seen.add(key)
            items.append(row)

    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/word_siamese_pairs.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/ui_siamese_words/best.pt")
    parser.add_argument("--out-dir", default="indexes/word_ui_siamese")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SiameseUIEncoder.load(args.checkpoint, map_location=device).to(device)
    model.eval()

    items = load_positive_items(args.pairs)

    transform = transforms.Compose([
        transforms.Resize((model.config.image_size, model.config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    embeddings = []

    with torch.no_grad():
        for item in items:
            image = Image.open(item["image"]).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            emb = model.encode_image(image)
            embeddings.append(emb.cpu().numpy()[0])

    np.save(out_dir / "embeddings.npy", np.array(embeddings, dtype=np.float32))

    with open(out_dir / "items.jsonl", "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved word UI index: {out_dir}")
    print(f"Items: {len(items)}")


if __name__ == "__main__":
    main()
