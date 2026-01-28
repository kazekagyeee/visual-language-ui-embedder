import torch
import torch.nn as nn
from rms_norm import RMSNorm
from bbox_window_attention import BBoxWindowAttention
from swi_glu import SwiGLUFFN

class VisualEncoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        window_size,
        ffn_hidden_dim
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.attn = BBoxWindowAttention(dim, heads, window_size)

        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class VisualEncoder(nn.Module):
    def __init__(
        self,
        dim=768,
        depth=12,
        heads=12,
        window_size=8,
        ffn_ratio=4/3
    ):
        super().__init__()

        self.dim = dim
        self.global_token = nn.Parameter(torch.zeros(1, 1, dim))

        ffn_hidden_dim = int(dim * ffn_ratio)

        self.blocks = nn.ModuleList([
            VisualEncoderBlock(
                dim=dim,
                heads=heads,
                window_size=window_size,
                ffn_hidden_dim=ffn_hidden_dim
            )
            for _ in range(depth)
        ])

        self.norm = RMSNorm(dim)

    def forward(self, bbox_tokens):
        """
        bbox_tokens: [B, N, D]
        """
        B = bbox_tokens.size(0)
        global_token = self.global_token.expand(B, -1, -1)

        x = torch.cat([global_token, bbox_tokens], dim=1)

        for blk in self.blocks:
            x = blk(x)

        return self.norm(x)

