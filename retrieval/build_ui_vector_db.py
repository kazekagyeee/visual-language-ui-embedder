# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.siamese_ui_encoder import SiameseUIEncoder
from rag.ui_vector_db import LocalUIVectorDB


def load_elements(path):
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            if Path(item["crop_image"]).exists():
                items.append(item)

    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--checkpoint", default="checkpoints/ui_elements_siamese/best.pt")
    parser.add_argument("--db-dir", default="vector_db/ui_elements")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiameseUIEncoder.load(args.checkpoint, map_location=device).to(device)
    model.eval()

    items = load_elements(Path(args.rag_dir) / "ui_elements.jsonl")

    transform = transforms.Compose([
        transforms.Resize((model.config.image_size, model.config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    vectors = []

    with torch.no_grad():
        for item in items:
            image = Image.open(item["crop_image"]).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            emb = model.encode_image(image)
            vectors.append(emb.cpu().numpy()[0])

    db = LocalUIVectorDB(args.db_dir)
    db.save(items, vectors)

    print(f"Saved vector DB: {args.db_dir}")
    print(f"Items: {len(items)}")


if __name__ == "__main__":
    main()
