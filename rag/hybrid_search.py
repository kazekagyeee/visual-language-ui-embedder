# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


RAG_DIR = Path("data/pdf_rag")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def tokenize(text: str):
    text = text.lower()
    return re.findall(r"[а-яёa-z0-9\-]+", text)


class HybridSearcher:
    def __init__(self, rag_dir=RAG_DIR, model_name=MODEL_NAME):
        self.rag_dir = Path(rag_dir)
        self.model = SentenceTransformer(model_name)

        self.items = []
        with open(self.rag_dir / "items.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

        self.embeddings = np.load(self.rag_dir / "embeddings.npy")

        corpus_tokens = [tokenize(item["text"]) for item in self.items]
        self.bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, top_k: int = 6, alpha: float = 0.65):
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        dense_scores = self.embeddings @ query_vec

        sparse_scores = np.array(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

        dense_norm = self._normalize(dense_scores)
        sparse_norm = self._normalize(sparse_scores)

        final_scores = alpha * dense_norm + (1 - alpha) * sparse_norm
        top_ids = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for idx in top_ids:
            item = self.items[int(idx)]
            results.append({
                "score": float(final_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "bm25_score": float(sparse_scores[idx]),
                "item": item,
            })

        return results

    @staticmethod
    def _normalize(scores):
        scores = np.array(scores, dtype=np.float32)
        min_s = scores.min()
        max_s = scores.max()

        if max_s - min_s < 1e-8:
            return np.zeros_like(scores)

        return (scores - min_s) / (max_s - min_s)
