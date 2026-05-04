from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class PairJsonlDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        self.items: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))
        if not self.items:
            raise ValueError(f"Dataset is empty: {self.path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        return {
            "id": item.get("id", str(idx)),
            "text_vec": torch.tensor(item["text_vec"], dtype=torch.float32),
            "image_vec": torch.tensor(item["image_vec"], dtype=torch.float32),
            "label": torch.tensor(float(item["label"]), dtype=torch.float32),
            "meta": item.get("meta", {}),
        }
