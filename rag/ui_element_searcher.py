# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from models.siamese_ui_encoder import SiameseUIEncoder


def normalize_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def lexical_bonus(query, text):
    q = normalize_text(query)
    t = normalize_text(text)

    if q == t:
        return 1.0

    if q in t or t in q:
        return 0.6

    q_tokens = set(q.split())
    t_tokens = set(t.split())

    if not q_tokens:
        return 0.0

    return len(q_tokens & t_tokens) / len(q_tokens) * 0.4


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
            bonus = lexical_bonus(query, item["text"])
            final_score = float(score) + bonus

            results.append({
                "score": final_score,
                "siamese_score": float(score),
                "item": item,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
