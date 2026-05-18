# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


STOP_WORDS = {
    "где", "как", "что", "найти", "создать", "сделать",
    "нужно", "надо", "можно", "в", "на", "по", "для",
    "и", "или", "а", "из", "к", "у",
}


def normalize(text):
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return [
        x for x in normalize(text).split()
        if len(x) > 2 and x not in STOP_WORDS
    ]


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def lexical_score(query, text):
    q = set(tokenize(query))
    t = set(tokenize(text))

    if not q or not t:
        return 0.0

    return len(q & t) / max(1, len(q))


def target_score(targets, text):
    if not targets:
        return 0.0

    n_text = normalize(text)
    best = 0.0

    for target in targets:
        n_target = normalize(target)

        if not n_target:
            continue

        if n_text == n_target:
            best = max(best, 1.0)

        elif n_target in n_text or n_text in n_target:
            best = max(best, 0.95)

        else:
            best = max(best, lexical_score(n_target, n_text))

    return best


def page_allowed(item_page, page_filter):
    if page_filter is None:
        return True

    item_page = int(item_page)

    if isinstance(page_filter, (list, tuple, set)):
        return item_page in {int(p) for p in page_filter}

    return item_page == int(page_filter)


class UIElementSearcher:
    def __init__(
        self,
        index_dir="data/ui_index",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        checkpoint=None,
    ):
        self.index_dir = Path(index_dir)
        self.items_path = self.index_dir / "ui_items.jsonl"
        self.embeddings_path = self.index_dir / "ui_embeddings.npy"

        self.model = SentenceTransformer(model_name)

        if not self.items_path.exists():
            self.items = []
            self.embeddings = None
            return

        self.items = load_jsonl(self.items_path)

        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
        else:
            texts = [item.get("text", "") for item in self.items]
            self.embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
            np.save(self.embeddings_path, self.embeddings)

    def search(
        self,
        query,
        targets=None,
        page_filter=None,
        pdf_filter=None,
        top_k=20,
    ):
        if not self.items or self.embeddings is None:
            return []

        targets = targets or []

        query_text = " ".join([query] + targets)

        query_vec = self.model.encode(
            [query_text],
            normalize_embeddings=True,
        )[0].astype("float32")

        dense_scores = self.embeddings @ query_vec

        results = []

        for idx, item in enumerate(self.items):
            if pdf_filter is not None and item.get("pdf_name") != pdf_filter:
                continue

            if not page_allowed(item.get("page", -1), page_filter):
                continue

            text = item.get("text", "")

            lex = lexical_score(query, text)
            targ = target_score(targets, text)
            dense = float(dense_scores[idx])

            ui_bonus = 0.0

            if item.get("ui_type") == "button":
                ui_bonus += 0.10

            if item.get("ui_type") in {"menu_item", "link"}:
                ui_bonus += 0.08

            final = (
                0.30 * ((dense + 1) / 2)
                + 0.25 * lex
                + 0.40 * targ
                + ui_bonus
            )

            if targ <= 0 and lex <= 0.05:
                continue

            results.append(
                {
                    "item": item,
                    "score": float(final),
                    "dense_score": dense,
                    "lexical_score": float(lex),
                    "target_score": float(targ),
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
