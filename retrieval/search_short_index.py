from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import yaml

from models.short_siamese_encoder import ShortSiameseEncoder


def load_metadata(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def search_numpy(query_vec: np.ndarray, matrix: np.ndarray, metadata: list, top_k: int):
    query_vec = query_vec.astype("float32")
    query_vec = query_vec / max(np.linalg.norm(query_vec), 1e-12)
    scores = matrix @ query_vec
    idx = np.argsort(-scores)[:top_k]
    return [{"score": float(scores[i]), **metadata[i]} for i in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/short_siamese.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/short_siamese/best.pt")
    parser.add_argument("--query-vec-json", required=True, help="JSON list with teacher text vector")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    top_k = args.top_k or cfg["retrieval"]["top_k"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShortSiameseEncoder.load(args.checkpoint, map_location=device).to(device)
    model.eval()

    text_vec = torch.tensor(json.loads(args.query_vec_json), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        query_short = model.encode_text(text_vec).squeeze(0).cpu().numpy()

    matrix = np.load(cfg["retrieval"]["numpy_index_path"])
    metadata = load_metadata(cfg["retrieval"]["metadata_path"])
    results = search_numpy(query_short, matrix, metadata, top_k)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
