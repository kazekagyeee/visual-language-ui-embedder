import torch.nn as nn


class BBoxMLP(nn.Module):
    def __init__(self, in_dim: int = 9, hidden_dim: int = 64, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
