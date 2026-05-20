# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from models.ui_siamese_ranker import UISiameseRanker
from rag.ocr_cleanup import cleanup_ocr_text


STOP_WORDS = {
    "где", "как", "что", "найти", "создать", "сделать",
    "нужно", "надо", "можно", "в", "на", "по", "для",
    "и", "или", "а", "из", "к", "у",
}


def normalize(text):
    text = cleanup_ocr_text(text)
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


def lexical_score(query, text):
    q = set(tokenize(query))
    t = set(tokenize(text))

    if not q or not t:
        return 0.0

    return len(q & t) / max(1, len(q))


def target_score(targets, text):
    if not targets:
        return 0.0

    text_n = normalize(text)
    best = 0.0

    for target in targets:
        target_n = normalize(target)

        if not target_n or not text_n:
            continue

        if target_n == text_n:
            best = max(best, 1.0)
        elif target_n in text_n:
            best = max(best, 0.90)
        elif text_n in target_n:
            best = max(best, 0.75)
        else:
            tw = set(tokenize(target_n))
            xw = set(tokenize(text_n))

            if tw and xw:
                best = max(best, len(tw & xw) / max(1, len(tw)))

    return best


def is_probably_bad(item):
    text = normalize(item.get("text", ""))

    if len(text) < 2:
        return True

    if len(text.split()) > 10:
        return True

    if "000000" in text and "инн" not in text:
        return True

    return False


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
        top_k=80,
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

        trained_scores = self.embeddings @ q_vec

        results = []

        for idx, item in enumerate(self.items):
            if pdf_filter is not None and item.get("pdf_name") != pdf_filter:
                continue

            if not page_allowed(item.get("page", -1), page_filter):
                continue

            if is_probably_bad(item):
                continue

            text = item.get("text", "")

            trained = float(trained_scores[idx])
            targ = target_score(targets, text)
            lex = lexical_score(query, text)

            ui_bonus = 0.0
            if item.get("ui_type") == "button":
                ui_bonus += 0.08
            elif item.get("ui_type") in {"menu_item", "link", "merged_text"}:
                ui_bonus += 0.05

            # Главное: trained score не должен забивать точные target-совпадения.
            final = (
                0.45 * trained
                + 0.75 * targ
                + 0.20 * lex
                + ui_bonus
            )

            # target rescue: даже если trained score слабый, точный target обязан попасть наверх.
            if targ >= 0.90:
                final += 0.60

            if targ <= 0 and lex <= 0.02 and trained < 0.15:
                continue

            results.append(
                {
                    "item": item,
                    "score": float(final),
                    "dense_score": float(trained),
                    "target_score": float(targ),
                    "lexical_score": float(lex),
                    "trained_score": float(trained),
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
