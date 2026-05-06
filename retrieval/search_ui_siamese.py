# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import transforms

from models.siamese_ui_encoder import SiameseUIEncoder


def load_items(rag_dir):
    items = []

    with open(Path(rag_dir) / "items.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            crops = item.get("target_crop_images") or []
            if not crops:
                crops = [item.get("crop_image")]

            crops = [c for c in crops if c and Path(c).exists()]

            for crop in crops:
                new_item = dict(item)
                new_item["search_crop"] = crop
                items.append(new_item)

    return items


def build_image_index(model, items, image_size, device):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    embeddings = []

    model.eval()

    with torch.no_grad():
        for item in items:
            image = Image.open(item["search_crop"]).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            emb = model.encode_image(image)
            embeddings.append(emb.cpu().numpy()[0])

    return np.array(embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--checkpoint", default="checkpoints/ui_siamese/best.pt")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--text-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiameseUIEncoder.load(args.checkpoint, map_location=device).to(device)
    model.eval()

    text_model = SentenceTransformer(args.text_model)

    items = load_items(args.rag_dir)
    image_embeddings = build_image_index(
        model=model,
        items=items,
        image_size=model.config.image_size,
        device=device,
    )

    query_vec = text_model.encode(
        [args.query],
        normalize_embeddings=True,
    ).astype("float32")

    query_vec = torch.tensor(query_vec, dtype=torch.float32).to(device)

    with torch.no_grad():
        text_emb = model.encode_text(query_vec).cpu().numpy()[0]

    scores = image_embeddings @ text_emb
    top_ids = np.argsort(scores)[::-1][:args.top_k]

    for idx in top_ids:
        item = items[int(idx)]

        print("=" * 100)
        print(f"score={scores[idx]:.4f}")
        print(f"page={item['page']} block={item.get('block_id')}")
        print(f"crop={item['search_crop']}")
        print(item["text"])


if __name__ == "__main__":
    main()
