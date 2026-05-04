import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from models.short_siamese_encoder import ShortSiameseConfig, ShortSiameseEncoder
from training.pair_dataset import PairJsonlDataset


def run_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    criterion = torch.nn.BCEWithLogitsLoss()

    for batch in loader:
        text_vec = batch["text_vec"].to(device)
        image_vec = batch["image_vec"].to(device)
        label = batch["label"].to(device)

        out = model(text_vec, image_vec)
        loss = criterion(out["logit"], label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = (out["score"] >= 0.5).float()
        total_correct += (pred == label).sum().item()
        total_count += label.numel()
        total_loss += loss.item() * label.numel()

    return total_loss / total_count, total_correct / total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/short_siamese.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PairJsonlDataset(cfg["data"]["train_pairs"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )

    model_cfg = ShortSiameseConfig(**cfg["model"])
    model = ShortSiameseEncoder(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
    )

    checkpoint = Path(cfg["paths"]["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    best_loss = 10**9

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        loss, acc = run_epoch(model, loader, optimizer, device)
        print(f"epoch={epoch:03d} loss={loss:.4f} acc={acc:.4f}")

        if loss < best_loss:
            best_loss = loss
            model.save(checkpoint)

    print(f"Saved best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
