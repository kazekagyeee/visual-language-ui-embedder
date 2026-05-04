from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from models.short_siamese_encoder import ShortSiameseEncoder


def load_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/short_siamese.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/short_siamese/best.pt")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShortSiameseEncoder.load(args.checkpoint, map_location=device).to(device)
    model.eval()

    vectors = []
    metadata = []
    for item in load_jsonl(cfg["data"]["reference_items"]):
        if "short_vec" in item:
            vec = np.asarray(item["short_vec"], dtype="float32")
        else:
            image_vec = torch.tensor(item["image_vec"], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                vec = model.encode_image(image_vec).squeeze(0).cpu().numpy().astype("float32")
        vec = vec / max(np.linalg.norm(vec), 1e-12)
        vectors.append(vec)
        metadata.append({k: v for k, v in item.items() if k not in {"short_vec", "image_vec", "qwen_long_vec"}})

    matrix = np.stack(vectors).astype("float32")
    out_np = Path(cfg["retrieval"]["numpy_index_path"])
    out_meta = Path(cfg["retrieval"]["metadata_path"])
    out_np.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_np, matrix)

    with out_meta.open("w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        import faiss

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, cfg["retrieval"]["index_path"])
        print(f"FAISS index saved: {cfg['retrieval']['index_path']}")
    except Exception as exc:
        print(f"FAISS skipped, NumPy index saved instead. Reason: {exc}")

    print(f"NumPy index saved: {out_np}")
    print(f"Metadata saved: {out_meta}")


if __name__ == "__main__":
    main()
