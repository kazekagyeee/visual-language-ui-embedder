# -*- coding: utf-8 -*-

from sentence_transformers import CrossEncoder


class CrossEncoderUIReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, results, top_k=10):
        if not results:
            return []

        pairs = [
            [query, r["item"].get("text", "")]
            for r in results
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for r, ce_score in zip(results, scores):
            nr = dict(r)
            nr["cross_encoder_score"] = float(ce_score)
            nr["final_score"] = float(r.get("score", 0.0)) + float(ce_score)
            reranked.append(nr)

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked[:top_k]
