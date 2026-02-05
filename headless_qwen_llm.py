import torch
import torch.nn as nn
import torch.nn.functional as F

from box_aware_visual_encoder import RMSNorm, SwiGLU

def rotate_half(x):
    """Вращает половину эмбеддинга для RoPE."""
    # x: (B, H, L, D)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Применяет RoPE к Query и Key.
    cos/sin обычно имеют форму (1, L, 1, D) или (1, L, D).
    Их нужно привести к (B, 1, L, D) или (B, H, L, D) для бродкастинга.
    """
    # Убеждаемся, что cos/sin подходят под размерность (Batch, Heads, Seq, Dim)
    # Обычно в Qwen2.5 они прилетают как (1, L, 1, D)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# -----------------------------------------------------------
# 1. Специфичный для LLM 1D RoPE (Rotary Positional Embedding)
# -----------------------------------------------------------

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_position_embeddings

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]


class Qwen2_5_Attention(nn.Module):
    """
    Реализация Attention, совместимая с весами Qwen2.5 (GQA + Split Projections)
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_heads']
        self.num_key_value_heads = config['num_key_value_heads']
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # В Qwen2.5 веса разделены на отдельные слои
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x, mask=None, rope_cos_sin=None):
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            # RoPE в Qwen применяется к каждой голове
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Повторяем K, V для GQA (Grouped Query Attention)
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Стандартный Scaled Dot-Product Attention
        # Для троек (Box, Global, Text) здесь можно использовать causal mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=mask is None  # Если маски нет, считаем что нужна казуальная
        )

        attn_output = attn_output.transpose(1, 2).reshape(B, L, self.hidden_size)
        return self.o_proj(attn_output)


class Qwen2_5_DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.self_attn = Qwen2_5_Attention(config)
        self.post_attention_layernorm = RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.mlp = SwiGLU(config['hidden_size'], config['intermediate_size'])

    def forward(self, x, mask=None, rope_cos_sin=None):
        # x: (B*N, 3, D)
        x = x + self.self_attn(self.input_layernorm(x), mask=mask, rope_cos_sin=rope_cos_sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class HeadlessQwen2_5(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Список слоев с правильными именами для load_state_dict
        self.layers = nn.ModuleList([Qwen2_5_DecoderLayer(config) for _ in range(config['num_layers'])])
        self.norm = RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.rotary_emb = Qwen2RotaryEmbedding(
            config['hidden_size'] // config['num_heads'],
            base=config['rope_theta']
        )

    def forward(self, triples):
        B, N, S, D = triples.shape
        x = triples.view(B * N, S, D)

        # Генерируем 1D RoPE
        cos, sin = self.rotary_emb(x, seq_len=S)

        # Создаем Causal Mask для последовательности длиной 3
        # (Чтобы текст видел бокс и глобал, но бокс не видел текст)
        mask = torch.tril(torch.ones(S, S, device=x.device)).view(1, 1, S, S)

        for layer in self.layers:
            x = layer(x, mask=mask, rope_cos_sin=(cos, sin))

        x = self.norm(x)

        # Возвращаем эмбеддинг последнего токена (текстового),
        # обогащенного визуальным контекстом бокса
        return x[:, -1, :].view(B, N, D)