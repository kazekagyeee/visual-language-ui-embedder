# -*- coding: utf-8 -*-

import re


def normalize_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class SimpleUIReranker:
    """
    Лёгкий reranker без дополнительной тяжёлой модели.
    Комбинирует:
    - siamese score;
    - точное совпадение OCR-текста;
    - совпадение токенов;
    - бонус за тип UI.
    """

    def __init__(self):
        pass

    def score(self, query, result):
        item = result["item"]

        q = normalize_text(query)
        t = normalize_text(item.get("text", ""))

        score = float(result.get("score", 0.0))

        if q == t:
            score += 1.5
        elif q in t or t in q:
            score += 1.0
        else:
            q_tokens = set(q.split())
            t_tokens = set(t.split())

            if q_tokens:
                score += len(q_tokens & t_tokens) / len(q_tokens) * 0.6

        ui_type = item.get("ui_type", "")

        if ui_type in {"button", "hyperlink", "sidebar_item"}:
            score += 0.25

        return score

    def rerank(self, query, results, top_k=10):
        reranked = []

        for result in results:
            new_result = dict(result)
            new_result["rerank_score"] = self.score(query, result)
            reranked.append(new_result)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
