from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_pos_neg_accuracy(z_text, z_pos, z_neg) -> float:
    pos_sim = F.cosine_similarity(z_text, z_pos)
    neg_sim = F.cosine_similarity(z_text, z_neg)
    return float((pos_sim > neg_sim).float().mean().item())


def compute_margin_stats(z_text, z_pos, z_neg) -> dict:
    margins = (F.cosine_similarity(z_text, z_pos) - F.cosine_similarity(z_text, z_neg)).detach().cpu().float()
    if margins.numel() == 0:
        return {"mean_margin": 0.0, "median_margin": 0.0, "p10_margin": 0.0, "p90_margin": 0.0}
    return {
        "mean_margin": float(margins.mean().item()),
        "median_margin": float(margins.median().item()),
        "p10_margin": float(torch.quantile(margins, 0.1).item()),
        "p90_margin": float(torch.quantile(margins, 0.9).item()),
    }
