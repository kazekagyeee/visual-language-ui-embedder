# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.siamese_ui_encoder import SiameseUIEncoder


def load_elements(path):
    elements = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            el = json.loads(line)

            if Path(el["crop_image"]).exists():
                elements.append(el)

    return elements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--checkpoint", default="checkpoints/ui_elements_siamese/best.pt")
    parser.add_argument("--out-dir", default="indexes/ui_elements_siamese")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rag_dir = Path(args.rag_dir)
    elements = load_elements(rag_dir / "ui_elements.jsonl")

    model = SiameseUIEncoder.load(args.checkpoint, map_location=device).to(device)
    model.eval()

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
        for el in elements:
            img = Image.open(el["crop_image"]).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            emb = model.encode_image(img)
            embeddings.append(emb.cpu().numpy()[0])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "embeddings.npy", np.array(embeddings, dtype=np.float32))

    with open(out_dir / "items.jsonl", "w", encoding="utf-8") as f:
        for el in elements:
            f.write(json.dumps(el, ensure_ascii=False) + "\n")

    print(f"Saved UI element index: {out_dir}")
    print(f"Items: {len(elements)}")


if __name__ == "__main__":
    main()
