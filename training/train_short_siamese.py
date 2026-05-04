from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from models.short_siamese_encoder import ShortSiameseConfig, ShortSiameseEncoder
from training.pair_dataset import PairJsonlDataset


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def contrastive_loss(cosine: torch.Tensor, label: torch.Tensor, margin: float) -> torch.Tensor:
    positive = label * (1.0 - cosine).pow(2)
    negative = (1.0 - label) * F.relu(cosine - margin).pow(2)
    return (positive + negative).mean()


def run_epoch(model, loader, optimizer, device, cfg, train: bool):
    model.train(train)
    total_loss = 0.0
    labels_all, scores_all = [], []
    for batch in loader:
        text_vec = batch["text_vec"].to(device)
        image_vec = batch["image_vec"].to(device)
        label = batch["label"].to(device)

        with torch.set_grad_enabled(train):
            out = model(text_vec, image_vec)
            bce = F.binary_cross_entropy_with_logits(out["logits"], label)
            con = contrastive_loss(out["cosine"], label, cfg["training"]["margin"])
            loss = cfg["training"]["bce_weight"] * bce + cfg["training"]["contrastive_weight"] * con
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * text_vec.size(0)
        labels_all.extend(label.detach().cpu().tolist())
        scores_all.extend(out["score"].detach().cpu().tolist())

    preds = [1 if s >= 0.5 else 0 for s in scores_all]
    acc = accuracy_score(labels_all, preds)
    try:
        auc = roc_auc_score(labels_all, scores_all)
    except ValueError:
        auc = 0.0
    return total_loss / len(loader.dataset), acc, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/short_siamese.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))
    device = resolve_device(cfg["training"]["device"])
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PairJsonlDataset(cfg["data"]["train_pairs"])
    val_path = Path(cfg["data"]["val_pairs"])
    val_ds = PairJsonlDataset(val_path) if val_path.exists() else train_ds

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    model_cfg = ShortSiameseConfig(**cfg["model"])
    model = ShortSiameseEncoder(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )

    best_auc = -1.0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss, train_acc, train_auc = run_epoch(model, train_loader, optimizer, device, cfg, True)
        val_loss, val_acc, val_auc = run_epoch(model, val_loader, optimizer, device, cfg, False)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"train_auc={train_auc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_auc={val_auc:.3f}"
        )
        if val_auc >= best_auc:
            best_auc = val_auc
            model.save(str(ckpt_dir / "best.pt"))

    print(f"Saved best checkpoint to {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
