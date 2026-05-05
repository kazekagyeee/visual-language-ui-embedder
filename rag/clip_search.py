# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


class ClipImageSearcher:
    def __init__(self, rag_dir="data/pdf_rag", model_name="clip-ViT-B-32"):
        self.rag_dir = Path(rag_dir)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.items_path = self.rag_dir / "clip_items.jsonl"
        self.embeddings_path = self.rag_dir / "clip_embeddings.npy"

        self.items = []
        self.embeddings = None

        if self.items_path.exists() and self.embeddings_path.exists():
            self.load_index()

    def load_index(self):
        self.items = []

        with open(self.items_path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

        self.embeddings = np.load(self.embeddings_path)

    def build_index(self):
        source_items = []

        with open(self.rag_dir / "items.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                source_items.append(json.loads(line))

        images = []
        kept_items = []
        seen_crops = set()

        for item in source_items:
            crop_candidates = item.get("target_crop_images") or []

            if not crop_candidates:
                crop_candidates = [item.get("crop_image")]

            for crop in crop_candidates:
                if not crop:
                    continue

                crop_path = Path(crop)

                if not crop_path.exists():
                    continue

                if str(crop_path) in seen_crops:
                    continue

                seen_crops.add(str(crop_path))

                image = Image.open(crop_path).convert("RGB")
                images.append(image)

                new_item = dict(item)
                new_item["clip_crop_image"] = str(crop_path).replace("\\", "/")
                kept_items.append(new_item)

        embeddings = self.model.encode(
            images,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        np.save(self.embeddings_path, np.array(embeddings, dtype=np.float32))

        with open(self.items_path, "w", encoding="utf-8") as f:
            for item in kept_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        self.items = kept_items
        self.embeddings = np.array(embeddings, dtype=np.float32)

        return len(kept_items)

    def search(self, query: str, top_k: int = 5):
        if self.embeddings is None or not self.items:
            raise FileNotFoundError("CLIP index not found. Build it first.")

        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        scores = self.embeddings @ query_vec
        top_ids = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_ids:
            results.append({
                "score": float(scores[idx]),
                "item": self.items[int(idx)],
            })

        return results
