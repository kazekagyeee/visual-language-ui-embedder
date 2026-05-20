# -*- coding: utf-8 -*-

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


class UIPairDataset(Dataset):
    def __init__(self, pairs_path, embedder):
        self.rows = load_jsonl(pairs_path)
        self.embedder = embedder

        self.query_cache = {}
        self.ui_cache = {}

    def __len__(self):
        return len(self.rows)

    def encode(self, text, cache):
        if text not in cache:
            vec = self.embedder.encode(
                [text],
                normalize_embeddings=True,
            )[0]
            cache[text] = torch.tensor(vec, dtype=torch.float32)

        return cache[text]

    def __getitem__(self, idx):
        row = self.rows[idx]

        query = row["query"]
        ui_text = f"{row.get('ui_text', '')} {row.get('ui_type', '')}"

        return {
            "query_vec": self.encode(query, self.query_cache),
            "ui_vec": self.encode(ui_text, self.ui_cache),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "row": row,
        }
