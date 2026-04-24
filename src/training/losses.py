from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_loss(z_text, z_pos, temperature=0.07) -> torch.Tensor:
    logits = z_text @ z_pos.T / temperature
    targets = torch.arange(z_text.size(0), device=z_text.device)
    loss_t2i = F.cross_entropy(logits, targets)
    loss_i2t = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_t2i + loss_i2t)


def triplet_cosine_loss(z_text, z_pos, z_neg, margin=0.2) -> torch.Tensor:
    pos_sim = F.cosine_similarity(z_text, z_pos)
    neg_sim = F.cosine_similarity(z_text, z_neg)
    return F.relu(margin - pos_sim + neg_sim).mean()


def total_loss(z_text, z_pos, z_neg, lambda_triplet=0.3, temperature=0.07, margin=0.2) -> dict:
    c_loss = contrastive_loss(z_text, z_pos, temperature=temperature)
    t_loss = triplet_cosine_loss(z_text, z_pos, z_neg, margin=margin)
    loss = c_loss + lambda_triplet * t_loss
    return {
        "loss": loss,
        "contrastive_loss": c_loss,
        "triplet_loss": t_loss,
    }
