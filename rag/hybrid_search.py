# -*- coding: utf-8 -*-

import json
import math
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize(text):
    text = str(text).lower().replace("ё", "е")
    for ch in ",.;:!?()[]{}«»\"'—–-":
        text = text.replace(ch, " ")
    return [x for x in text.split() if x]


class HybridSearcher:
    def __init__(
        self,
        rag_dir,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.rag_dir = Path(rag_dir)
        self.model = SentenceTransformer(model_name)

        items_path = self._find_items_path()
        self.items = load_jsonl(items_path)

        self._normalize_items()

        embeddings_path = self.rag_dir / "embeddings.npy"

        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
        else:
            print(f"embeddings.npy not found, building: {embeddings_path}")
            texts = [item.get("text", "") for item in self.items]
            self.embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
            np.save(embeddings_path, self.embeddings)

        self.doc_tokens = [tokenize(item.get("text", "")) for item in self.items]
        self.idf = self._build_idf(self.doc_tokens)

    def _find_items_path(self):
        candidates = [
            self.rag_dir / "items.jsonl",
            self.rag_dir / "text_blocks.jsonl",
            self.rag_dir / "blocks.jsonl",
            self.rag_dir / "chunks.jsonl",
            self.rag_dir / "pdf_blocks.jsonl",
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            "Не найден текстовый индекс. Ожидался один из файлов: "
            + ", ".join(str(p) for p in candidates)
        )

    def _normalize_items(self):
        for idx, item in enumerate(self.items):
            item.setdefault("id", f"item_{idx}")

            if "block_id" not in item:
                item["block_id"] = item.get("block", idx)

            if "page_image" not in item:
                page = int(item.get("page", 1))
                page_image = self.rag_dir / "pages" / f"page_{page:04d}.png"
                item["page_image"] = str(page_image).replace("\\", "/")

    def _build_idf(self, docs):
        n = len(docs)
        df = {}

        for tokens in docs:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        return {
            token: math.log((n + 1) / (freq + 1)) + 1.0
            for token, freq in df.items()
        }

    def _bm25_score(self, query_tokens, doc_tokens, avgdl, k1=1.5, b=0.75):
        if not doc_tokens:
            return 0.0

        score = 0.0
        dl = len(doc_tokens)

        tf = {}
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1

        for token in query_tokens:
            if token not in tf:
                continue

            idf = self.idf.get(token, 0.0)
            freq = tf[token]

            denom = freq + k1 * (1 - b + b * dl / max(avgdl, 1))
            score += idf * (freq * (k1 + 1)) / max(denom, 1e-8)

        return float(score)

    def search(self, query, top_k=5, alpha=0.35):
        if not self.items:
            return []

        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        query_vec = np.asarray(query_vec, dtype=np.float32)

        dense_scores = self.embeddings @ query_vec

        query_tokens = tokenize(query)
        avgdl = sum(len(x) for x in self.doc_tokens) / max(1, len(self.doc_tokens))

        bm25_scores = np.array(
            [
                self._bm25_score(query_tokens, tokens, avgdl)
                for tokens in self.doc_tokens
            ],
            dtype=np.float32,
        )

        if bm25_scores.max() > 0:
            bm25_norm = bm25_scores / bm25_scores.max()
        else:
            bm25_norm = bm25_scores

        dense_norm = dense_scores

        final_scores = alpha * dense_norm + (1.0 - alpha) * bm25_norm

        order = np.argsort(-final_scores)[:top_k]

        results = []

        for idx in order:
            results.append({
                "item": self.items[int(idx)],
                "score": float(final_scores[int(idx)]),
                "dense_score": float(dense_scores[int(idx)]),
                "bm25_score": float(bm25_scores[int(idx)]),
            })

        return results
