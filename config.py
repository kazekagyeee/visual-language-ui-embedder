from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class UIEmbedderConfig:
    """Configuration for UI Embedder Pipeline"""
    # Device configuration
    # my GPU only has 16gb of vram so cpu only(
    device: str = "cpu"
    
    # Prompts
    system_prompt: str = "You are a UI context describer assistant. You answer in russian."
    context_prompt: str = "Тебе нужно описать UI элемент в контексте основного изображения и его описания: первое - целое изображение UI, второе - сам описываемый компонент, а далее текст, который является контекстом к основному изображению: "
    
    # Model architecture parameters
    # Vision Encoder outputs 3584-dim directly (after Qwen2VLSpatialMerge = visual.merger)
    # No separate projector needed — merger IS the projector per Qwen2.5-VL paper
    llm_dim: int = 3584  # Qwen2.5-7B hidden size
    heads_vis: int = 16 
    depth_vis: int = 32 # Qwen2-VL-7B model usually has deep vision/llm
    
    # Qwen2.5-7B LLM Config
    llm_config: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_size': 3584,
        'num_heads': 28,
        'num_key_value_heads': 4,
        'intermediate_size': 18944,
        'num_layers': 28, 
        'rms_norm_eps': 1e-6,
        'rope_theta': 1000000.0,
    })
    
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    # Visual encoder and image processing parameters
    img_size: int = 224 # Will be resized dynamically
    patch_size_encoder: int = 14
    patch_size_resize: int = 28 # Qwen2.5-VL uses 2×2 spatial merge → need multiple of 28 (14 × 2)
    
    # Detection parameters
    max_dist: int = 17 # Для слияния близких bbox
    
    # Debug flags
    debug_decode_embeddings: bool = True
    debug_dir: str = "debug"
