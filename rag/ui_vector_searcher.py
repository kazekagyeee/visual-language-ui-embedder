# -*- coding: utf-8 -*-

import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from models.siamese_ui_encoder import SiameseUIEncoder
from rag.ui_vector_db import LocalUIVectorDB


def normalize_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def lexical_bonus(query, text):
    q = normalize_text(query)
    t = normalize_text(text)

    if q == t:
        return 1.5
    if q in t or t in q:
        return 1.0

    q_tokens = set(q.split())
    t_tokens = set(t.split())

    if not q_tokens:
        return 0.0

    return len(q_tokens & t_tokens) / len(q_tokens) * 0.5


class UIVectorSearcher:
    def __init__(
        self,
        checkpoint="checkpoints/ui_elements_siamese/best.pt",
        db_dir="vector_db/ui_elements",
        text_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.checkpoint = Path(checkpoint)
        self.db = LocalUIVectorDB(db_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.text_model = None

        if self.checkpoint.exists() and self.db.vectors is not None:
            self.model = SiameseUIEncoder.load(
                self.checkpoint,
                map_location=self.device,
            ).to(self.device)
            self.model.eval()
            self.text_model = SentenceTransformer(text_model_name)

    def encode_query(self, query):
        query_vec = self.text_model.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")

        query_vec = torch.tensor(query_vec, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            text_emb = self.model.encode_text(query_vec).cpu().numpy()[0]

        return text_emb

    def search(self, query, top_k=10):
        if self.model is None:
            raise FileNotFoundError(
                "Vector searcher is not ready. Check checkpoint and vector_db."
            )

        query_vector = self.encode_query(query)
        results = self.db.search(query_vector, top_k=top_k * 3)

        enriched = []

        for result in results:
            item = result["item"]
            final_score = result["score"] + lexical_bonus(query, item.get("text", ""))

            enriched.append({
                "score": final_score,
                "vector_score": result["score"],
                "item": item,
            })

        enriched.sort(key=lambda x: x["score"], reverse=True)
        return enriched[:top_k]
