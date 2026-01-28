import torch
import torch.nn as nn
from einops import rearrange

class BBoxWindowAttention(nn.Module):
    def __init__(self, dim, heads, window_size):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.scale = (dim // heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        """
        x: [B, N+1, D]
        token 0 = global image token
        остальные = bbox tokens
        """
        B, T, D = x.shape
        H = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=H),
            qkv
        )

        out = torch.zeros_like(q)

        # 0-й токен — global → full attention
        q0 = q[:, :, :1]
        attn0 = torch.softmax(
            (q0 @ k.transpose(-2, -1)) * self.scale,
            dim=-1
        )
        out[:, :, :1] = attn0 @ v

        # bbox window attention
        for i in range(1, T, self.window_size):
            q_i = q[:, :, i:i+self.window_size]
            k_i = k[:, :, i:i+self.window_size]
            v_i = v[:, :, i:i+self.window_size]

            attn = torch.softmax(
                (q_i @ k_i.transpose(-2, -1)) * self.scale,
                dim=-1
            )
            out[:, :, i:i+self.window_size] = attn @ v_i

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)