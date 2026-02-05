import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -----------------------------------------------------------
# 1. Базовые компоненты Qwen 2.5 (RMSNorm & SwiGLU)
# -----------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Масштабирование на основе корня из среднего квадратов
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * output


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # Имена проекций как в Qwen 2.5
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # Swish(gate) * up_proj
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# -----------------------------------------------------------
# 2. Логика 2D RoPE (Rotary Positional Embeddings)
# -----------------------------------------------------------

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # cos/sin имеют форму (1, L, 1, D)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2_5_2DRoPE(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim  # head_dim
        self.base = base
        # Половина дима на высоту, половина на ширину
        inv_freq = 1.0 / (base ** (torch.arange(0, dim // 2, 2).float() / (dim // 2)))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, grid_h, grid_w):
        def get_emb(t):
            out = torch.einsum("i,j->ij", t, self.inv_freq)
            return torch.cat((out, out), dim=-1)

        emb_h = get_emb(grid_h)
        emb_w = get_emb(grid_w)
        # Собираем 2D: [height_embs, width_embs]
        combined = torch.cat([emb_h, emb_w], dim=-1)

        cos = combined.cos().unsqueeze(0).unsqueeze(2)  # (1, L, 1, D)
        sin = combined.sin().unsqueeze(0).unsqueeze(2)
        return cos, sin


# -----------------------------------------------------------
# 3. Attention и Transformer Block
# -----------------------------------------------------------

class BoxAwareAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None, rope_cos_sin=None):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask expected as (B, 1, L, L)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        return x


class Qwen2_5_ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = BoxAwareAttention(dim, num_heads=num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))

    def forward(self, x, mask=None, rope_cos_sin=None):
        x = x + self.attn(self.norm1(x), mask=mask, rope_cos_sin=rope_cos_sin)
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------------------------------------
# 4. Основной класс Энкодера
# -----------------------------------------------------------

class Qwen2_5_BoxEncoder(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=14,
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 n_boxes=5):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.n_boxes = n_boxes
        self.embed_dim = embed_dim

        # В Qwen2.5-VL входной патчинг через свертку
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Специальные токены
        self.global_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.box_tokens = nn.Parameter(torch.zeros(1, n_boxes, embed_dim))

        # 2D RoPE модуль (применяется к head_dim)
        self.rope = Qwen2_5_2DRoPE(embed_dim // num_heads)

        self.blocks = nn.ModuleList([
            Qwen2_5_ViTBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm_final = RMSNorm(embed_dim)

    def _get_rope_embeddings(self, device):
        gs = self.grid_size
        # Сетка координат патчей
        y, x = torch.meshgrid(torch.arange(gs), torch.arange(gs), indexing='ij')
        y, x = y.flatten().float().to(device), x.flatten().float().to(device)

        # Координаты для префикс-токенов (Global + Boxes) - ставим в 0 или центр
        prefix_len = 1 + self.n_boxes
        y_prefix = torch.zeros(prefix_len, device=device)
        x_prefix = torch.zeros(prefix_len, device=device)

        return self.rope(torch.cat([y_prefix, y]), torch.cat([x_prefix, x]))

    def create_box_mask(self, boxes, B, device):
        """
        Создает маску внимания.
        boxes: (B, N, 4) -> [x1, y1, x2, y2] в диапазоне [0, 1]
        """
        total_tokens = 1 + self.n_boxes + self.num_patches
        # По умолчанию разрешаем всё (Full Attention для патчей и глобального токена)
        mask = torch.ones((B, 1, total_tokens, total_tokens), device=device, dtype=torch.bool)

        gs = self.grid_size
        y_c, x_c = torch.meshgrid(torch.linspace(0, 1, gs), torch.linspace(0, 1, gs), indexing='ij')
        y_c, x_c = y_c.flatten().to(device), x_c.flatten().to(device)

        patch_start = 1 + self.n_boxes

        for i in range(self.n_boxes):
            # Извлекаем координаты i-го бокса для всего батча
            b_x1, b_y1, b_x2, b_y2 = boxes[:, i, 0:1], boxes[:, i, 1:2], boxes[:, i, 2:3], boxes[:, i, 3:4]

            # Условие попадания центра патча в бокс
            in_box = (x_c >= b_x1) & (x_c <= b_x2) & (y_c >= b_y1) & (y_c <= b_y2)

            # Ограничиваем i-й Box-токен (строка 1+i) только этими патчами
            # Зануляем всю строку внимания для патчей
            mask[:, 0, 1 + i, patch_start:] = False
            # Включаем только те, что в боксе
            mask[:, 0, 1 + i, patch_start:] = in_box

            # Box-токен видит сам себя
            mask[:, 0, 1 + i, 1:1 + self.n_boxes] = False
            mask[:, 0, 1 + i, 1 + i] = True

        return mask

    def forward(self, img, boxes):
        B = img.shape[0]

        # 1. Patchify
        x = self.patch_embed(img).flatten(2).transpose(1, 2)  # (B, M, D)

        # 2. Concat tokens: [Global, Box_1...N, Patches...]
        tokens = torch.cat([
            self.global_token.expand(B, -1, -1),
            self.box_tokens.expand(B, -1, -1),
            x
        ], dim=1)

        # 3. Positional Info (RoPE & Mask)
        rope_cos_sin = self._get_rope_embeddings(img.device)
        mask = self.create_box_mask(boxes, B, img.device)

        # 4. Transformer Layers
        for blk in self.blocks:
            tokens = blk(tokens, mask=mask, rope_cos_sin=rope_cos_sin)

        tokens = self.norm_final(tokens)

        # 5. Split output
        global_emb = tokens[:, 0]
        box_embs = tokens[:, 1:1 + self.n_boxes]

        return global_emb, box_embs