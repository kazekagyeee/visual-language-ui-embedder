from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorProjectionAdapter(nn.Module):
    """Last-layer adapter: Qwen UI vector -> BERT/SentenceTransformer vector."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(output_dim * 2, min(input_dim, 1024))
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(vectors), dim=-1)
