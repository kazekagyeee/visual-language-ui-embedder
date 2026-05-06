# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.siamese_ui_encoder import SiameseUIConfig, SiameseUIEncoder


class SiamesePairsDataset(Dataset):
    def __init__(self, pairs_path, text_model_name, image_size=128):
        self.rows = []

        with open(pairs_path, "r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

        self.text_model = SentenceTransformer(text_model_name)

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        texts = [row["text"] for row in self.rows]
        self.text_embeddings = self.text_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        image = Image.open(row["image"]).convert("RGB")
        image = self.image_transform(image)

        text_vec = torch.tensor(self.text_embeddings[idx], dtype=torch.float32)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return {
            "text_vec": text_vec,
            "image": image,
            "label": label,
        }


def train_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        text_vec = batch["text_vec"].to(device)
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        out = model(text_vec, image)

        bce_loss = F.binary_cross_entropy_with_logits(out["logits"], label)

        # contrastive часть:
        # positive должны иметь similarity ближе к 1
        # negative — ближе к 0 / ниже
        target_similarity = label * 2 - 1
        cosine_loss = F.mse_loss(out["similarity"], target_similarity)

        loss = bce_loss + 0.25 * cosine_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(label)

        preds = (torch.sigmoid(out["logits"]) >= 0.5).float()
        correct += int((preds == label).sum().item())
        total += len(label)

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/siamese_pairs.jsonl")
    parser.add_argument("--out", default="checkpoints/ui_siamese/best.pt")
    parser.add_argument("--text-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ds = SiamesePairsDataset(
        pairs_path=args.pairs,
        text_model_name=args.text_model,
        image_size=args.image_size,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    text_dim = ds.text_embeddings.shape[1]

    config = SiameseUIConfig(
        text_dim=text_dim,
        image_size=args.image_size,
        embedding_dim=args.embedding_dim,
    )

    model = SiameseUIEncoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = -1.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, loader, optimizer, device)

        print(f"epoch={epoch:03d} loss={loss:.4f} acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            model.save(out_path)

    print(f"Saved best model: {out_path}")
    print(f"Best acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
