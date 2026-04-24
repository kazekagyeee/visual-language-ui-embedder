from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from .bbox_mlp import BBoxMLP
from .fusion_head import FusionHead


class SigLIP2ImageTower(nn.Module):
    def __init__(
        self,
        model_name: str,
        image_feat_dim: int,
        bbox_feat_dim: int = 64,
        out_dim: int = 256,
        freeze_backbone: bool = True,
        bbox_input_dim: int = 9,
        bbox_hidden_dim: int = 64,
        dropout: float = 0.1,
        backbone=None,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.backbone = backbone if backbone is not None else AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.bbox_mlp = BBoxMLP(
            in_dim=bbox_input_dim,
            hidden_dim=bbox_hidden_dim,
            out_dim=bbox_feat_dim,
            dropout=dropout,
        )
        self.fusion_head = FusionHead(
            in_dim=image_feat_dim + bbox_feat_dim,
            hidden_dim=512,
            out_dim=out_dim,
            dropout=dropout,
        )

    def forward(self, image_inputs: dict, bbox_features: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone.get_image_features(**image_inputs)
        image_feat = outputs.pooler_output
        bbox_emb = self.bbox_mlp(bbox_features)
        fused = torch.cat([image_feat, bbox_emb], dim=-1)
        return self.fusion_head(fused)
