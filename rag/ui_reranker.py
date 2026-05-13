# -*- coding: utf-8 -*-

import re


def norm(text):
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_score(query, text):
    q = set(norm(query).split())
    t = set(norm(text).split())

    if not q or not t:
        return 0.0

    return len(q & t) / len(q)


class UIReranker:
    def score(self, query, result):
        item = result["item"]

        text = item.get("text", "")
        ui_type = item.get("ui_type", "unknown")

        base = float(result.get("score", 0.0))
        siamese = float(result.get("siamese_score", 0.0))

        qn = norm(query)
        tn = norm(text)

        exact = 0.0
        if qn == tn:
            exact = 2.0
        elif qn in tn or tn in qn:
            exact = 1.2

        tok = token_score(query, text)

        type_bonus = 0.0
        if ui_type in {"button", "hyperlink", "sidebar_item", "tab"}:
            type_bonus = 0.35

        length_penalty = 0.0
        if len(tn.split()) > 6:
            length_penalty = -0.8

        final = base + siamese + exact + tok + type_bonus + length_penalty

        return {
            "final_score": final,
            "base_score": base,
            "siamese_score": siamese,
            "exact_bonus": exact,
            "token_score": tok,
            "type_bonus": type_bonus,
            "length_penalty": length_penalty,
        }

    def rerank(self, query, results, top_k=10):
        out = []

        for result in results:
            scoring = self.score(query, result)
            new_result = dict(result)
            new_result.update(scoring)
            out.append(new_result)

        out.sort(key=lambda x: x["final_score"], reverse=True)
        return out[:top_k]
