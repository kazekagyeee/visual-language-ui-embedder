import torch
import torch.nn as nn
import torch.nn.functional as F
from rms_norm import RMSNorm

class VisionToTextProjector(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        hidden_ratio: float = 4 / 3
    ):
        super().__init__()

        hidden_dim = int(text_dim * hidden_ratio)

        self.norm = RMSNorm(vision_dim)

        self.w1 = nn.Linear(vision_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(vision_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, text_dim, bias=False)

    def forward(self, x):
        """
        x: [B, N+1, D_vis]
        return: [B, N+1, D_txt]
        """
        x = self.norm(x)
        return self.w3(self.w1(x) * F.silu(self.w2(x)))
