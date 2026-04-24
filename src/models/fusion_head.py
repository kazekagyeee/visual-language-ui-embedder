import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)
