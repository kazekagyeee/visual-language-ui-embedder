from dataclasses import dataclass, field
from typing import Dict, Any, Optional


# -----------------------------------------------------------
# Architecture presets sourced directly from HuggingFace configs
# Keys match the HuggingFace repo ID.
# -----------------------------------------------------------
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    # Qwen2-VL-2B — smallest publicly released Qwen2-VL model (~2B params)
    # HF: Qwen/Qwen2-VL-2B-Instruct
    "Qwen/Qwen2-VL-2B-Instruct": {
        "llm_dim": 1536,
        "vocab_size": 151936,
        "llm_config": {
            "hidden_size": 1536,
            "num_heads": 12,
            "num_key_value_heads": 2,
            "intermediate_size": 8960,
            "num_layers": 28,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
        },
        # Vision encoder (ViT) — shared across all model sizes
        "vis_embed_dim": 1280,
        "vis_depth": 32,
        "vis_heads": 16,
        "vis_intermediate_size": 3420,
        # Spatial merger output dim must match llm_dim
        "merger_out_dim": 1536,
    },
    # Qwen2.5-VL-3B — smallest Qwen2.5-VL model
    # HF: Qwen/Qwen2.5-VL-3B-Instruct
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "llm_dim": 2048,
        "vocab_size": 151936,
        "llm_config": {
            "hidden_size": 2048,
            "num_heads": 16,
            "num_key_value_heads": 2,
            "intermediate_size": 11008,
            "num_layers": 36,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
        },
        "vis_embed_dim": 1280,
        "vis_depth": 32,
        "vis_heads": 16,
        "vis_intermediate_size": 3420,
        "merger_out_dim": 2048,
    },
    # Qwen2.5-VL-7B — default model
    # HF: Qwen/Qwen2.5-VL-7B-Instruct
    "Qwen/Qwen2.5-VL-7B-Instruct": {
        "llm_dim": 3584,
        "vocab_size": 152064,
        "llm_config": {
            "hidden_size": 3584,
            "num_heads": 28,
            "num_key_value_heads": 4,
            "intermediate_size": 18944,
            "num_layers": 28,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
        },
        "vis_embed_dim": 1280,
        "vis_depth": 32,
        "vis_heads": 16,
        "vis_intermediate_size": 3420,
        "merger_out_dim": 3584,
    },
    # Qwen2.5-VL-72B — largest variant (requires significant RAM/VRAM)
    # HF: Qwen/Qwen2.5-VL-72B-Instruct
    "Qwen/Qwen2.5-VL-72B-Instruct": {
        "llm_dim": 8192,
        "vocab_size": 152064,
        "llm_config": {
            "hidden_size": 8192,
            "num_heads": 64,
            "num_key_value_heads": 8,
            "intermediate_size": 29568,
            "num_layers": 80,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1000000.0,
        },
        "vis_embed_dim": 1280,
        "vis_depth": 32,
        "vis_heads": 16,
        "vis_intermediate_size": 3420,
        "merger_out_dim": 8192,
    },
}

# Convenient short aliases (e.g. "2B", "3B", "7B")
MODEL_ALIASES: Dict[str, str] = {
    "2B":  "Qwen/Qwen2-VL-2B-Instruct",
    "3B":  "Qwen/Qwen2.5-VL-3B-Instruct",
    "7B":  "Qwen/Qwen2.5-VL-7B-Instruct",
    "72B": "Qwen/Qwen2.5-VL-72B-Instruct",
}


@dataclass
class UIEmbedderConfig:
    """Configuration for UI Embedder Pipeline.

    The easiest way to switch model sizes is via the factory:
        config = UIEmbedderConfig.from_model_name("Qwen/Qwen2.5-VL-3B-Instruct")
        # or using a short alias:
        config = UIEmbedderConfig.from_model_name("3B")
    """

    # ---- Device ----
    # GPU recommended; set "cpu" if no CUDA device available.
    device: str = "cuda"

    # ---- Prompts ----
    system_prompt: str = "You are a UI context describer assistant. You answer in russian."
    context_prompt: str = (
        "Тебе нужно описать UI элемент в контексте основного изображения и его описания: "
        "первое - целое изображение UI, второе - сам описываемый компонент, "
        "а далее текст, который является контекстом к основному изображению: "
    )

    # ---- Retrieval prompt (EXPERIMENTAL) ----
    # When True, uses an E5/GTE-style instruct prompt instead of the generative
    # chat template. This is intended for future contrastive fine-tuning experiments.
    # EXPERIMENTAL: quality without fine-tuning is NOT guaranteed — the base Qwen
    # model was not trained for retrieval alignment.
    use_retrieval_prompt: bool = False
    # Instruction prepended to the text in retrieval mode.
    retrieval_instruction: str = (
        "Instruct: Given a UI screenshot and its description, retrieve the UI component "
        "that best matches the query.\nQuery: "
    )

    # ---- Global summary token ----
    # When True, g_seq is mean-pooled into a single summary token and prepended
    # as a bidirectional prefix to each per-box LLM input sequence:
    #   [g_summary(1) | box_patches(N_b) | text+EOS(T)]
    # When False (default), g_summary is omitted and the sequence is:
    #   [box_patches(N_b) | text+EOS(T)]
    # box_start/box_end indices in HeadlessQwen2_5 are shifted automatically.
    use_global_summary: bool = False

    # ---- Tokenizer ----
    # Maximum token length for the assembled prompt (text side only).
    # Prevents OOM when text_content is very long.
    max_token_length: int = 512

    # ---- Model identity ----
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # ---- LLM dimensions (derived from model_name preset) ----
    llm_dim: int = 3584
    vocab_size: int = 152064

    # ---- LLM layer config ----
    llm_config: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_size": 3584,
        "num_heads": 28,
        "num_key_value_heads": 4,
        "intermediate_size": 18944,
        "num_layers": 28,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
    })

    # ---- Vision Encoder (ViT) — same for all model sizes ----
    vis_embed_dim: int = 1280        # Internal ViT hidden size
    vis_depth: int = 32
    vis_heads: int = 16
    vis_intermediate_size: int = 3420
    # Spatial merger output dim (must equal llm_dim)
    merger_out_dim: int = 3584

    # ---- Image processing ----
    img_size: int = 224
    patch_size_encoder: int = 14
    patch_size_resize: int = 28      # 14 × 2 (Qwen2.5-VL 2×2 spatial merge)

    # ---- Detection ----
    max_dist: int = 17

    # ---- Generation ----
    max_new_tokens: int = 250

    # ---- Debug ----
    debug_decode_embeddings: bool = True
    debug_dir: str = "debug"

    # ------------------------------------------------------------------
    @classmethod
    def from_model_name(cls, model_name: str, **overrides) -> "UIEmbedderConfig":
        """Create a config pre-filled with the correct architecture for the given model.

        Args:
            model_name: Full HuggingFace repo ID  **or**  a short alias like "3B", "7B".
            **overrides: Any UIEmbedderConfig field to override (e.g. device="cuda").

        Example:
            cfg = UIEmbedderConfig.from_model_name("3B", device="cuda")
            cfg = UIEmbedderConfig.from_model_name("Qwen/Qwen2-VL-2B-Instruct")
        """
        # Resolve alias
        resolved = MODEL_ALIASES.get(model_name, model_name)

        if resolved not in MODEL_PRESETS:
            available = list(MODEL_PRESETS.keys()) + list(MODEL_ALIASES.keys())
            raise ValueError(
                f"Unknown model '{model_name}'.\n"
                f"Available presets: {available}\n"
                f"For an unsupported size, create UIEmbedderConfig() and set fields manually."
            )

        preset = MODEL_PRESETS[resolved]
        instance = cls(
            model_name=resolved,
            llm_dim=preset["llm_dim"],
            vocab_size=preset["vocab_size"],
            llm_config=preset["llm_config"].copy(),
            vis_embed_dim=preset["vis_embed_dim"],
            vis_depth=preset["vis_depth"],
            vis_heads=preset["vis_heads"],
            vis_intermediate_size=preset["vis_intermediate_size"],
            merger_out_dim=preset["merger_out_dim"],
        )

        for key, value in overrides.items():
            if not hasattr(instance, key):
                raise ValueError(f"UIEmbedderConfig has no field '{key}'.")
            setattr(instance, key, value)

        return instance
