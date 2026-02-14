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
        # Для 2D RoPE нужно разделить head_dim пополам
        half_dim = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, grid_h, grid_w):
        # Создаем эмбеддинги для каждой оси
        # grid_h и grid_w имеют форму (L,) где L - количество позиций
        def get_emb(t):
            # t: (L,), inv_freq: (half_dim//2,)
            # out: (L, half_dim//2)
            out = torch.einsum("i,j->ij", t, self.inv_freq)
            # Дублируем для sin и cos пар: (L, half_dim//2) -> (L, half_dim)
            return torch.cat((out, out), dim=-1)

        emb_h = get_emb(grid_h)  # (L, half_dim)
        emb_w = get_emb(grid_w)  # (L, half_dim)
        
        # Собираем 2D: [height_embs, width_embs] -> (L, dim)
        combined = torch.cat([emb_h, emb_w], dim=-1)

        # Форма для broadcasting с (B, num_heads, L, head_dim)
        cos = combined.cos().unsqueeze(0).unsqueeze(1)  # (1, 1, L, dim)
        sin = combined.sin().unsqueeze(0).unsqueeze(1)  # (1, 1, L, dim)
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
        self.proj = nn.Linear(dim, dim, bias=True) # Qwen2-VL has bias

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
    def __init__(self, dim, num_heads, intermediate_size=3420):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = BoxAwareAttention(dim, num_heads=num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, intermediate_size)

    def forward(self, x, mask=None, rope_cos_sin=None):
        x = x + self.attn(self.norm1(x), mask=mask, rope_cos_sin=rope_cos_sin)
        x = x + self.mlp(self.norm2(x))
        return x

class SpatialMergeAdapter(nn.Module):
    """
    Adapts 1280-dim features to 5120-dim to match Qwen2-VL Projector.
    Naive implementation: Repeat 4 times (simulating 2x2 spatial merge).
    """
    def __init__(self, in_dim=1280, out_dim=5120):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
    def forward(self, x):
        # x: (..., 1280) -> (..., 5120)
        return x.repeat_interleave(4, dim=-1)

# -----------------------------------------------------------
# 4. Основной класс Энкодера
# -----------------------------------------------------------

class Qwen2_5_BoxEncoder(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=14,
                 embed_dim=1280, # Qwen2-VL-7B default
                 depth=32, # Qwen2-VL-7B default
                 num_heads=16,
                 intermediate_size=3420, # Qwen2-VL-7B default
                 use_learned_tokens=False): # Set to True for future training
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_learned_tokens = use_learned_tokens

        # В Qwen2.5-VL входной патчинг через свертку (3D Convolution for temporal)
        # Typically kernel_size=(2, 14, 14), stride=(2, 14, 14)
        # Check if depth/patch_size logic needs T dimension?
        self.patch_embed = nn.Conv3d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=(2, patch_size, patch_size),
            stride=(2, patch_size, patch_size)
        )

        # TODO: Learned tokens для будущего файнтюнинга
        # Сейчас используем ROI pooling (use_learned_tokens=False)
        # Чтобы использовать learned tokens:
        #   1. Установить use_learned_tokens=True
        #   2. Загрузить веса или обучить с нуля
        #   3. Добавить в load_qwen_weights.py загрузку visual.global_token и visual.box_token_prototype
        if use_learned_tokens:
            self.global_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.box_token_prototype = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.global_token = None
            self.box_token_prototype = None
            
        # Adapter для преобразования 1280 -> 5120 (только для ROI pooling режима)
        self.adapter = SpatialMergeAdapter(embed_dim, 5120) if not use_learned_tokens else None

        # 2D RoPE модуль (применяется к head_dim)
        self.rope = Qwen2_5_2DRoPE(embed_dim // num_heads)

        self.blocks = nn.ModuleList([
            Qwen2_5_ViTBlock(embed_dim, num_heads, intermediate_size) for _ in range(depth)
        ])
        self.norm_final = RMSNorm(embed_dim)

    def _get_rope_embeddings(self, device, n_boxes, h, w):
        # Сетка координат патчей
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y, x = y.flatten().float().to(device), x.flatten().float().to(device)

        if self.use_learned_tokens:
            # Coordinates for prefix tokens (Global + Boxes)
            prefix_len = 1 + n_boxes
            y_prefix = torch.zeros(prefix_len, device=device)
            x_prefix = torch.zeros(prefix_len, device=device)
            return self.rope(torch.cat([y_prefix, y]), torch.cat([x_prefix, x]))
        else:
            # Only patches, no prefix tokens
            return self.rope(y, x)

    def create_box_mask(self, boxes, B, device, h, w):
        """
        Создает маску внимания.
        boxes: (B, N, 4) -> [x1, y1, x2, y2] в диапазоне [0, 1]
        """
        n_boxes = boxes.shape[1]
        num_patches = h * w
        total_tokens = 1 + n_boxes + num_patches
        # По умолчанию разрешаем всё (Full Attention для патчей и глобального токена)
        mask = torch.ones((B, 1, total_tokens, total_tokens), device=device, dtype=torch.bool)

        y_c, x_c = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing='ij')
        y_c, x_c = y_c.flatten().to(device), x_c.flatten().to(device)

        patch_start = 1 + n_boxes

        for i in range(n_boxes):
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
            mask[:, 0, 1 + i, 1:1 + n_boxes] = False
            mask[:, 0, 1 + i, 1 + i] = True

        return mask

    def forward(self, img, boxes):
        """
        Args:
            img: (B, 3, H, W)
            boxes: (B, N, 4)
        """
        B = img.shape[0]
        n_boxes = boxes.shape[1]

        # 1. Patchify using Conv3d
        # img: (B, 3, H, W) -> need (B, 3, T, H, W)
        # Qwen2-VL temporal kernel is 2. So we duplicate the frame.
        x = img.unsqueeze(2).repeat(1, 1, 2, 1, 1) # (B, 3, 2, H, W)
        
        # Output: (B, EmbedDim, T_out, H_out, W_out)
        # If T=2, stride=2, kernel=2 -> T_out = 1
        x = self.patch_embed(x) 
        
        # Flatten: (B, EmbedDim, 1, H_out, W_out) -> (B, EmbedDim, H_out*W_out)
        x = x.squeeze(2).flatten(2).transpose(1, 2)  # (B, M, D)

        # 2. Concat tokens (only if using learned tokens)
        if self.use_learned_tokens:
            box_tokens = self.box_token_prototype.expand(B, n_boxes, -1)
            tokens = torch.cat([
                self.global_token.expand(B, -1, -1),
                box_tokens,
                x
            ], dim=1)
        else:
            # ROI Pooling mode: process only image patches
            tokens = x

        # 3. Positional Info (RoPE & Mask)
        # Получаем размеры сетки из выхода свертки
        # x до flatten был (B, EmbedDim, H_out, W_out)
        # Но мы уже сделали flatten. Вернемся к шагу 1.
        # patch_embed output shape: (B, EmbedDim, T, H_out, W_out) -> T=1 after pool
        # Давайте вычислим h, w из исходного изображения
        # stride=(2, 14, 14).
        # H_out = H // 14, W_out = W // 14 (assuming H, W divisible by 14)
        h_out = img.shape[2] // self.patch_size
        w_out = img.shape[3] // self.patch_size
        
        rope_cos_sin = self._get_rope_embeddings(img.device, n_boxes, h_out, w_out)
        rc, rs = rope_cos_sin
        rope_cos_sin = (rc.to(dtype=img.dtype), rs.to(dtype=img.dtype))
        
        # Mask only for learned tokens mode
        mask = self.create_box_mask(boxes, B, img.device, h_out, w_out) if self.use_learned_tokens else None

        # 4. Transformer Layers
        for blk in self.blocks:
            tokens = blk(tokens, mask=mask, rope_cos_sin=rope_cos_sin)

        tokens = self.norm_final(tokens)

        # 5. Extract embeddings
        if self.use_learned_tokens:
            # Use learned tokens
            global_emb = tokens[:, 0]  # (B, 1280)
            box_embs = tokens[:, 1:1 + n_boxes]  # (B, N, 1280)
        else:
            # ROI Pooling from feature grid
            feature_grid = tokens.view(B, h_out, w_out, self.embed_dim)
            
            # Global embedding: mean pool all patches
            global_emb = tokens.mean(dim=1)  # (B, 1280)
            
            # Box embeddings: ROI pooling
            box_embs_list = []
            y_c, x_c = torch.meshgrid(
                torch.linspace(0, 1, h_out, dtype=img.dtype), 
                torch.linspace(0, 1, w_out, dtype=img.dtype), 
                indexing='ij'
            )
            y_c, x_c = y_c.to(img.device), x_c.to(img.device)
            
            for i in range(n_boxes):
                b = boxes[:, i, :]  # (B, 4)
                b_x1 = b[:, 0].unsqueeze(-1).unsqueeze(-1)
                b_y1 = b[:, 1].unsqueeze(-1).unsqueeze(-1)
                b_x2 = b[:, 2].unsqueeze(-1).unsqueeze(-1)
                b_y2 = b[:, 3].unsqueeze(-1).unsqueeze(-1)
                
                # Create mask for patches inside box
                in_box_mask = (x_c >= b_x1) & (x_c <= b_x2) & (y_c >= b_y1) & (y_c <= b_y2)
                
                # Handle empty boxes
                empty_mask = in_box_mask.sum(dim=(1,2)) == 0
                if empty_mask.any():
                    in_box_mask[empty_mask] = True  # Fallback to full image
                    
                # Mean pooling
                mask_expanded = in_box_mask.unsqueeze(-1).to(dtype=feature_grid.dtype)
                sum_features = (feature_grid * mask_expanded).sum(dim=(1, 2))
                count = mask_expanded.sum(dim=(1, 2))
                box_emb = sum_features / (count + 1e-6)
                box_embs_list.append(box_emb)
                
            box_embs = torch.stack(box_embs_list, dim=1)  # (B, N, 1280)
            
            # Adapt dimensions 1280 -> 5120
            global_emb = self.adapter(global_emb)
            box_embs = self.adapter(box_embs)

        return global_emb, box_embs