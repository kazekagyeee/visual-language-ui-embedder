from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.collate import build_triplet_collate_fn


@torch.inference_mode()
def export_positive_embeddings(model, dataset, processor, device: str, output_prefix: str | Path, batch_size: int = 8) -> dict:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=build_triplet_collate_fn(processor))
    all_embeddings = []
    metadata = []

    for batch in loader:
        image_inputs = {k: v.to(device) for k, v in batch["pos_image_inputs"].items()}
        bbox_features = batch["pos_bbox_features"].to(device)
        embeddings = model.encode_image(image_inputs, bbox_features).detach().cpu().float().numpy()
        all_embeddings.append(embeddings)
        for idx, emb in enumerate(embeddings):
            metadata.append(
                {
                    "sample_id": batch["sample_id"][idx],
                    "image_path": batch["image_path"][idx],
                    "bbox": batch["pos_bbox"][idx],
                    "embedding": emb.tolist(),
                    "text": batch["text"][idx],
                    "split": getattr(dataset, "split", "unknown"),
                }
            )

    matrix = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, 256), dtype=np.float32)
    np.save(str(output_prefix) + ".npy", matrix.astype(np.float32))
    with open(str(output_prefix) + ".jsonl", "w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        pd.DataFrame(metadata).to_parquet(str(output_prefix) + ".parquet", index=False)
        parquet_path = str(output_prefix) + ".parquet"
    except Exception:
        parquet_path = None

    return {
        "count": len(metadata),
        "npy_path": str(output_prefix) + ".npy",
        "jsonl_path": str(output_prefix) + ".jsonl",
        "parquet_path": parquet_path,
    }
