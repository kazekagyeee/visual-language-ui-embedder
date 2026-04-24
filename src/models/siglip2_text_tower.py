from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class SigLIP2TextTower(nn.Module):
    def __init__(
        self,
        model_name: str,
        out_dim: int = 256,
        freeze_backbone: bool = True,
        backbone=None,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.backbone = backbone if backbone is not None else AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_dim = self.backbone.config.text_config.hidden_size
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, text_inputs: dict) -> torch.Tensor:
        outputs = self.backbone.get_text_features(**text_inputs)
        pooled = outputs.pooler_output
        return F.normalize(self.text_proj(pooled), dim=-1)
