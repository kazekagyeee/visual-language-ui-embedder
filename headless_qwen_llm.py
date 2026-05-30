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
        # Output: (1, 1, S, D) to broadcast with (B, H, S, D)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


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

        # Use native causal attention when no custom prefix mask is needed.
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=mask is None
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

    def forward(self, seqs_list, s_prefix: int = 0,
                s_box_starts: list = None, s_box_ends: list = None):
        """
        seqs_list:    list of (1, S_i, D) tensors, one per bbox.
                      Expected layout: [g_summary(1) | box_patches(N_b) | text+EOS(T)]
        s_prefix:     Number of leading tokens that form the bidirectional prefix
                      (g_summary tokens). Pass 0 for pure causal (legacy).
        s_box_starts/s_box_ends are accepted for compatibility with older
                      callers, but pooling is always done on the last token.
        Returns:
            Tensor of shape (1, N_boxes, D) — one embedding per bbox.
        """
        pooled_outputs = []

        for idx, x_seq in enumerate(seqs_list):
            B, S, D = x_seq.shape

            # 1D RoPE
            cos, sin = self.rotary_emb(x_seq, seq_len=S)
            cos = cos.to(dtype=x_seq.dtype)
            sin = sin.to(dtype=x_seq.dtype)

            # --- Hybrid attention mask ---
            # [g_summary prefix (P)] → bidirectional
            # [box + text (R)]       → causal, full access to prefix
            mask = None
            if s_prefix > 0 and s_prefix < S:
                P = s_prefix
                R = S - P
                mask = torch.zeros(S, S, device=x_seq.device, dtype=torch.bool)
                mask[:P, :P] = True                                                # prefix: bidirectional
                mask[P:, :P] = True                                                # suffix sees prefix
                mask[P:, P:] = torch.tril(torch.ones(R, R, device=x_seq.device, dtype=torch.bool))
                mask = mask.view(1, 1, S, S)
            elif s_prefix >= S:
                mask = torch.ones(S, S, device=x_seq.device, dtype=torch.bool).view(1, 1, S, S)

            for layer in self.layers:
                x_seq = layer(x_seq, mask=mask, rope_cos_sin=(cos, sin))

            x_seq = self.norm(x_seq)

            # --- Pooling strategy: always use EOS ---
            # EOS is guaranteed to be at [-1] position in all sequences
            # This ensures consistent semantic representation across modalities
            pooled_emb = x_seq[:, -1, :]

            pooled_outputs.append(pooled_emb)

        # (1, N_boxes, D)
        return torch.stack(pooled_outputs, dim=1)
