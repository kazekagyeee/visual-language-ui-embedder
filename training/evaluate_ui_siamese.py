# -*- coding: utf-8 -*-

import argparse
import json

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import transforms

from models.siamese_ui_encoder import SiameseUIEncoder


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/ui_element_pairs.test.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/ui_elements_siamese/best.pt")
    parser.add_argument("--text-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = load_rows(args.pairs)

    model = SiameseUIEncoder.load(args.checkpoint, map_location=device).to(device)
    model.eval()

    text_model = SentenceTransformer(args.text_model)

    transform = transforms.Compose([
        transforms.Resize((model.config.image_size, model.config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    correct = 0
    total = 0

    with torch.no_grad():
        for row in rows:
            text_vec = text_model.encode(
                [row["text"]],
                normalize_embeddings=True,
            )

            text_vec = torch.tensor(text_vec, dtype=torch.float32).to(device)

            image = Image.open(row["image"]).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            out = model(text_vec, image)
            prob = torch.sigmoid(out["logits"])[0].item()
            pred = 1 if prob >= 0.5 else 0

            label = int(row["label"])

            correct += int(pred == label)
            total += 1

    acc = correct / max(total, 1)

    print(f"Test pairs: {total}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
