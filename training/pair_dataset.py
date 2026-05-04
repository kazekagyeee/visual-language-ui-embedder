import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PairJsonlDataset(Dataset):
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        self.items = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))

        if not self.items:
            raise ValueError(f"Dataset is empty: {self.path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "text_vec": torch.tensor(item["text_vec"], dtype=torch.float32),
            "image_vec": torch.tensor(item["image_vec"], dtype=torch.float32),
            "label": torch.tensor(float(item["label"]), dtype=torch.float32),
        }
