# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from models.siamese_ui_encoder import SiameseUIEncoder


class WordUISiameseSearcher:
    def __init__(
        self,
        checkpoint="checkpoints/ui_siamese_words/best.pt",
        index_dir="indexes/word_ui_siamese",
        text_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.checkpoint = Path(checkpoint)
        self.index_dir = Path(index_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.text_model = None
        self.items = []
        self.embeddings = None
        self.text_model_name = text_model_name

        if self.checkpoint.exists() and (self.index_dir / "items.jsonl").exists():
            self.load()

    def load(self):
        self.model = SiameseUIEncoder.load(self.checkpoint, map_location=self.device).to(self.device)
        self.model.eval()

        self.text_model = SentenceTransformer(self.text_model_name)

        self.items = []
        with open(self.index_dir / "items.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

        self.embeddings = np.load(self.index_dir / "embeddings.npy")

    def search(self, query, top_k=5):
        if self.model is None:
            raise FileNotFoundError("Word UI Siamese model/index not found.")

        query_vec = self.text_model.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")

        query_vec = torch.tensor(query_vec, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            text_emb = self.model.encode_text(query_vec).cpu().numpy()[0]

        scores = self.embeddings @ text_emb
        top_ids = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_ids:
            results.append({
                "score": float(scores[idx]),
                "item": self.items[int(idx)],
            })

        return results
