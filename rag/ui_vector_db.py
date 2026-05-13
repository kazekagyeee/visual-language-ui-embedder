# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np


class LocalUIVectorDB:
    def __init__(self, db_dir="vector_db/ui_elements"):
        self.db_dir = Path(db_dir)
        self.items_path = self.db_dir / "items.jsonl"
        self.vectors_path = self.db_dir / "vectors.npy"

        self.items = []
        self.vectors = None

        if self.items_path.exists() and self.vectors_path.exists():
            self.load()

    def save(self, items, vectors):
        self.db_dir.mkdir(parents=True, exist_ok=True)

        vectors = np.array(vectors, dtype=np.float32)

        with open(self.items_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        np.save(self.vectors_path, vectors)

        self.items = items
        self.vectors = vectors

    def load(self):
        self.items = []

        with open(self.items_path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

        self.vectors = np.load(self.vectors_path)

    def search(self, query_vector, top_k=10):
        if self.vectors is None:
            raise FileNotFoundError("Vector DB is empty.")

        query_vector = np.array(query_vector, dtype=np.float32)
        query_vector = query_vector / max(np.linalg.norm(query_vector), 1e-8)

        scores = self.vectors @ query_vector

        top_ids = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_ids:
            results.append({
                "score": float(scores[idx]),
                "item": self.items[int(idx)],
            })

        return results
