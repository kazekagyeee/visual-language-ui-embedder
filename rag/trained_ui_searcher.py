# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from models.ui_siamese_ranker import UISiameseRanker


def normalize(text):
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def page_allowed(item_page, page_filter):
    if page_filter is None:
        return True

    item_page = int(item_page)

    if isinstance(page_filter, (list, tuple, set)):
        return item_page in {int(p) for p in page_filter}

    return item_page == int(page_filter)


class TrainedUIElementSearcher:
    def __init__(
        self,
        index_dir="data/ui_trained_index",
        checkpoint="checkpoints/ui_siamese_ranker.pt",
    ):
        self.index_dir = Path(index_dir)
        self.items_path = self.index_dir / "ui_items.jsonl"
        self.embeddings_path = self.index_dir / "ui_embeddings.npy"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(checkpoint, map_location=self.device)
        self.embedder = SentenceTransformer(ckpt["model_name"])

        self.model = UISiameseRanker(input_dim=ckpt.get("input_dim", 384)).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.items = load_jsonl(self.items_path)
        self.embeddings = np.load(self.embeddings_path)

    def search(
        self,
        query,
        targets=None,
        page_filter=None,
        pdf_filter=None,
        top_k=40,
    ):
        targets = targets or []
        query_text = " ".join([query] + targets)

        base = self.embedder.encode(
            [query_text],
            normalize_embeddings=True,
        )

        with torch.no_grad():
            q = torch.tensor(base, dtype=torch.float32).to(self.device)
            q_vec = self.model.encode_query(q).cpu().numpy()[0]

        scores = self.embeddings @ q_vec

        results = []

        for idx, item in enumerate(self.items):
            if pdf_filter is not None and item.get("pdf_name") != pdf_filter:
                continue

            if not page_allowed(item.get("page", -1), page_filter):
                continue

            score = float(scores[idx])

            results.append(
                {
                    "item": item,
                    "score": score,
                    "dense_score": score,
                    "target_score": 0.0,
                    "trained_score": score,
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)

        unique = []
        seen = set()

        for result in results:
            item = result["item"]

            key = (
                item.get("pdf_name"),
                item.get("page"),
                item.get("screenshot_idx"),
                normalize(item.get("text", "")),
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(result)

            if len(unique) >= top_k:
                break

        return unique
