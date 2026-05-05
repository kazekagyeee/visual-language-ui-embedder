# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class VisualDescriptionSearcher:
    def __init__(
        self,
        rag_dir="data/pdf_rag",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.rag_dir = Path(rag_dir)
        self.model = SentenceTransformer(model_name)

        self.desc_path = self.rag_dir / "visual_descriptions.jsonl"
        self.emb_path = self.rag_dir / "visual_descriptions_embeddings.npy"

        self.items = []
        self.embeddings = None

        if self.desc_path.exists() and self.emb_path.exists():
            self.load_index()

    def load_index(self):
        self.items = []

        with open(self.desc_path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

        self.embeddings = np.load(self.emb_path)

    def build_index(self):
        if not self.desc_path.exists():
            raise FileNotFoundError(
                f"visual_descriptions.jsonl not found: {self.desc_path}"
            )

        self.items = []

        with open(self.desc_path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

        texts = [item["text"] for item in self.items]

        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        np.save(self.emb_path, np.array(self.embeddings, dtype=np.float32))

        return len(self.items)

    def search(self, query, top_k=5):
        if self.embeddings is None or not self.items:
            raise FileNotFoundError("Visual descriptions index not found.")

        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec

        top_ids = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_ids:
            results.append({
                "score": float(scores[idx]),
                "item": self.items[int(idx)],
            })

        return results
