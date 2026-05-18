# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from models.ui_siamese_ranker import UISiameseRanker


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui-index-dir", default="data/ui_index")
    parser.add_argument("--checkpoint", default="checkpoints/ui_siamese_ranker.pt")
    parser.add_argument("--out-dir", default="data/ui_trained_index")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    embedder = SentenceTransformer(ckpt["model_name"])

    model = UISiameseRanker(input_dim=ckpt.get("input_dim", 384)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    items = load_jsonl(Path(args.ui_index_dir) / "ui_items.jsonl")

    texts = [
        f"{item.get('text', '')} {item.get('ui_type', '')}"
        for item in items
    ]

    base_vecs = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    with torch.no_grad():
        x = torch.tensor(base_vecs, dtype=torch.float32).to(device)
        ui_vecs = model.encode_ui(x).cpu().numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "ui_items.jsonl", "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    np.save(out_dir / "ui_embeddings.npy", ui_vecs.astype("float32"))

    meta = {
        "source_ui_index": args.ui_index_dir,
        "checkpoint": args.checkpoint,
        "items": len(items),
    }

    with open(out_dir / "ui_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[DONE]")
    print(f"items={len(items)}")
    print(f"out={out_dir}")


if __name__ == "__main__":
    main()
