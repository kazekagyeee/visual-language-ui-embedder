# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sentence_transformers import SentenceTransformer

from models.ui_siamese_ranker import UISiameseRanker
from training.ui_pair_dataset import UIPairDataset


def collate(batch):
    return {
        "query_vec": torch.stack([x["query_vec"] for x in batch]),
        "ui_vec": torch.stack([x["ui_vec"] for x in batch]),
        "label": torch.stack([x["label"] for x in batch]),
    }


def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            q = batch["query_vec"].to(device)
            u = batch["ui_vec"].to(device)
            y = batch["label"].to(device)

            logits = model(q, u) * 8.0
            loss = F.binary_cross_entropy_with_logits(logits, y)

            pred = (torch.sigmoid(logits) >= 0.5).float()

            correct += (pred == y).sum().item()
            total += y.numel()
            loss_sum += loss.item()

    return loss_sum / max(1, len(loader)), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/ui_training_pairs.jsonl")
    parser.add_argument("--out", default="checkpoints/ui_siamese_ranker.pt")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedder = SentenceTransformer(args.model_name)
    dataset = UIPairDataset(args.pairs, embedder)

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    model = UISiameseRanker(input_dim=384).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        loss_sum = 0.0

        for batch in train_loader:
            q = batch["query_vec"].to(device)
            u = batch["ui_vec"].to(device)
            y = batch["label"].to(device)

            logits = model(q, u) * 8.0
            loss = F.binary_cross_entropy_with_logits(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={loss_sum / max(1, len(train_loader)):.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model_name,
                    "input_dim": 384,
                },
                out,
            )

    print(f"[DONE] best_acc={best_acc:.4f}")
    print(f"[SAVED] {args.out}")


if __name__ == "__main__":
    main()
