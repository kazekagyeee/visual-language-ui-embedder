# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from models.siamese_ui_encoder import SiameseUIEncoder
from rag.ocr_cleaning import normalize_ocr_text
from rag.ui_reranker import UIReranker


def normalize_text(text):
    return normalize_ocr_text(text)


def lexical_bonus(query, text):
    q = normalize_text(query)
    t = normalize_text(text)

    if q == t:
        return 1.4
    if q in t or t in q:
        return 1.0

    q_tokens = set(q.split())
    t_tokens = set(t.split())

    if not q_tokens:
        return 0.0

    return len(q_tokens & t_tokens) / len(q_tokens) * 0.6


class UIElementSearcher:
    def __init__(
        self,
        checkpoint="checkpoints/ui_elements_siamese/best.pt",
        index_dir="indexes/ui_elements_siamese",
        text_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.checkpoint = Path(checkpoint)
        self.index_dir = Path(index_dir)
        self.text_model_name = text_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.text_model = None
        self.items = []
        self.embeddings = None
        self.reranker = UIReranker()

        if self.checkpoint.exists() and (self.index_dir / "items.jsonl").exists():
            self.load()

    def load(self):
        self.model = SiameseUIEncoder.load(
            self.checkpoint,
            map_location=self.device,
        ).to(self.device)

        self.model.eval()
        self.text_model = SentenceTransformer(self.text_model_name)

        with open(self.index_dir / "items.jsonl", "r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f]

        self.embeddings = np.load(self.index_dir / "embeddings.npy")

    def known_ui_phrases_in_query(self, query):
        q = normalize_text(query)
        found = []

        unique_texts = sorted(
            {item["text"] for item in self.items},
            key=lambda x: len(normalize_text(x)),
            reverse=True,
        )

        for text in unique_texts:
            t = normalize_text(text)

            if len(t) < 3:
                continue

            if t in q:
                found.append(text)

        cleaned = []

        for phrase in found:
            p = normalize_text(phrase)

            nested = False
            for other in cleaned:
                o = normalize_text(other)
                if p in o and p != o:
                    nested = True
                    break

            if not nested:
                cleaned.append(phrase)

        return cleaned

    def search(self, query, top_k=8):
        if self.model is None:
            raise FileNotFoundError("UI Element Siamese model/index not found.")

        query_vec = self.text_model.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")

        query_vec = torch.tensor(query_vec, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            text_emb = self.model.encode_text(query_vec).cpu().numpy()[0]

        siamese_scores = self.embeddings @ text_emb

        results = []

        for score, item in zip(siamese_scores, self.items):
            bonus = lexical_bonus(query, item.get("text", ""))
            final_score = float(score) + bonus

            results.append({
                "score": final_score,
                "raw_score": final_score,
                "siamese_score": float(score),
                "item": item,
            })

        return self.reranker.rerank(query, results, top_k=top_k)

    def search_many(self, query, text_pages=None, per_phrase_k=3, max_total=12):
        phrases = self.known_ui_phrases_in_query(query)

        if not phrases:
            phrases = [query]

        collected = []
        seen = set()

        for phrase in phrases:
            results = self.search(phrase, top_k=per_phrase_k * 10)

            if text_pages:
                preferred = [
                    r for r in results
                    if r["item"]["page"] in text_pages
                ]

                if preferred:
                    results = preferred

            added = 0

            for result in results:
                item = result["item"]
                key = (
                    item["page"],
                    tuple(item["bbox"]),
                    normalize_text(item.get("text", "")),
                )

                if key in seen:
                    continue

                seen.add(key)
                result["matched_query"] = phrase
                collected.append(result)
                added += 1

                if added >= per_phrase_k:
                    break

        collected.sort(key=lambda x: x.get("final_score", x["score"]), reverse=True)
        return collected[:max_total]
