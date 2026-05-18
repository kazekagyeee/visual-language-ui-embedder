# -*- coding: utf-8 -*-

import json
import math
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


STOP_WORDS = {
    "где", "как", "что", "это", "найти", "нужно", "надо", "можно",
    "в", "на", "по", "для", "и", "или", "а", "с", "из", "к", "у",
}


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def normalize(text):
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return [
        token
        for token in normalize(text).split()
        if len(token) > 2 and token not in STOP_WORDS
    ]


def expand_query(query):
    q = normalize(query)
    variants = [query]

    dictionary = {
        "входной контроль": [
            "входной контроль",
            "контроль поступления",
            "проверка поступления",
            "приемка товаров",
        ],
        "организацию": [
            "организации",
            "создать организацию",
            "справочник организации",
            "новая организация",
        ],
        "организация": [
            "организации",
            "создать организацию",
            "справочник организации",
            "новая организация",
        ],
        "интернет поддержк": [
            "интернет-поддержка пользователей",
            "монитор интернет-поддержки",
            "подключить интернет-поддержку",
        ],
    }

    for key, vals in dictionary.items():
        if key in q:
            variants.extend(vals)

    return list(dict.fromkeys(variants))


class HybridSearcher:
    def __init__(
        self,
        rag_dir,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.rag_dir = Path(rag_dir)
        self.model = SentenceTransformer(model_name)

        items_path = self.rag_dir / "items.jsonl"

        if not items_path.exists():
            raise FileNotFoundError(
                f"Не найден индекс {items_path}. "
                f"Сначала запусти scripts/build_pdf_rag_multi.py"
            )

        self.items = load_jsonl(items_path)

        embeddings_path = self.rag_dir / "embeddings.npy"

        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
        else:
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

        tf = {}

        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1

        score = 0.0
        dl = len(doc_tokens)

        for token in query_tokens:
            if token not in tf:
                continue

            freq = tf[token]
            idf = self.idf.get(token, 0.0)
            denom = freq + k1 * (1 - b + b * dl / max(avgdl, 1))

            score += idf * (freq * (k1 + 1)) / max(denom, 1e-8)

        return float(score)

    def search(self, query, top_k=7, alpha=0.2):
        if not self.items:
            return []

        query_variants = expand_query(query)

        query_vecs = self.model.encode(
            query_variants,
            normalize_embeddings=True,
        )

        query_vec = np.mean(query_vecs, axis=0).astype("float32")
        query_vec = query_vec / max(np.linalg.norm(query_vec), 1e-8)

        dense_scores = self.embeddings @ query_vec

        all_query_tokens = []

        for variant in query_variants:
            all_query_tokens.extend(tokenize(variant))

        query_tokens = list(dict.fromkeys(all_query_tokens))

        avgdl = sum(len(x) for x in self.doc_tokens) / max(1, len(self.doc_tokens))

        bm25_scores = np.array(
            [
                self._bm25_score(query_tokens, doc_tokens, avgdl)
                for doc_tokens in self.doc_tokens
            ],
            dtype=np.float32,
        )

        if bm25_scores.max() > 0:
            bm25_norm = bm25_scores / bm25_scores.max()
        else:
            bm25_norm = bm25_scores

        dense_norm = (dense_scores + 1.0) / 2.0

        final_scores = alpha * dense_norm + (1.0 - alpha) * bm25_norm

        order = np.argsort(-final_scores)

        results = []
        seen = set()

        for idx in order:
            item = self.items[int(idx)]

            key = (
                item.get("doc_id"),
                item.get("page"),
                item.get("chunk_id"),
            )

            if key in seen:
                continue

            seen.add(key)

            results.append(
                {
                    "item": item,
                    "score": float(final_scores[int(idx)]),
                    "dense_score": float(dense_scores[int(idx)]),
                    "bm25_score": float(bm25_scores[int(idx)]),
                }
            )

            if len(results) >= top_k:
                break

        return results
