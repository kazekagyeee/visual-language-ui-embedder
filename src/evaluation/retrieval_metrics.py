from __future__ import annotations

import torch


def _sorted_indices(query_embs, corpus_embs):
    scores = query_embs @ corpus_embs.T
    return torch.argsort(scores, dim=1, descending=True)


def compute_recall_at_k(query_embs, corpus_embs, gt_indices, ks=(1, 5, 10)) -> dict:
    ranking = _sorted_indices(query_embs, corpus_embs)
    results = {}
    gt = torch.as_tensor(gt_indices, device=ranking.device)
    for k in ks:
        hits = (ranking[:, :k] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
        results[f"recall@{k}"] = float(hits)
    return results


def compute_mrr(query_embs, corpus_embs, gt_indices) -> float:
    ranking = _sorted_indices(query_embs, corpus_embs)
    gt = torch.as_tensor(gt_indices, device=ranking.device)
    reciprocal_ranks = []
    for idx in range(ranking.size(0)):
        rank = (ranking[idx] == gt[idx]).nonzero(as_tuple=False)
        reciprocal_ranks.append(1.0 / float(rank[0].item() + 1))
    return float(sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)))


def compute_median_rank(query_embs, corpus_embs, gt_indices) -> float:
    ranking = _sorted_indices(query_embs, corpus_embs)
    gt = torch.as_tensor(gt_indices, device=ranking.device)
    ranks = []
    for idx in range(ranking.size(0)):
        rank = (ranking[idx] == gt[idx]).nonzero(as_tuple=False)
        ranks.append(float(rank[0].item() + 1))
    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    return float(ranks_tensor.median().item()) if ranks else 0.0
