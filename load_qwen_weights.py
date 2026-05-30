"""
Utility module for loading Qwen2-VL weights from Hugging Face.
Supports loading:
1. Vision Encoder (Visual ViT)
2. Vision Projector (MLP)
3. Headless LLM (Decoder Only)
4. Text Token Embeddings
"""

import torch
import torch.nn as nn
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import gc
from typing import Dict, Optional

from config import MODEL_ALIASES, MODEL_PRESETS

# -----------------------------------------------------------
# 1. Vision Encoder Weights
# -----------------------------------------------------------

def _new_stats() -> Dict[str, int]:
    return {"loaded": 0, "shape_mismatch": 0, "errors": 0}


def _resolve_model_preset(model_name: str) -> tuple[str, Optional[Dict]]:
    resolved = MODEL_ALIASES.get(model_name, model_name)
    return resolved, MODEL_PRESETS.get(resolved)


def _validate_selected_model(
    model_name: str,
    preset: Optional[Dict],
    box_encoder=None,
    headless_llm=None,
    token_embedding=None,
    lm_head=None,
) -> None:
    if preset is None:
        print(f"[!] No local architecture preset for {model_name}; using shape checks only.")
        return

    errors = []
    llm_dim = preset["llm_dim"]
    vocab_size = preset["vocab_size"]
    llm_config = preset["llm_config"]

    if headless_llm is not None:
        actual_layers = len(headless_llm.layers)
        expected_layers = llm_config["num_layers"]
        if actual_layers != expected_layers:
            errors.append(f"LLM layers: expected {expected_layers}, got {actual_layers}")

        if actual_layers:
            first = headless_llm.layers[0]
            actual_hidden = first.self_attn.hidden_size
            if actual_hidden != llm_dim:
                errors.append(f"LLM hidden size: expected {llm_dim}, got {actual_hidden}")
            actual_heads = first.self_attn.num_heads
            if actual_heads != llm_config["num_heads"]:
                errors.append(f"LLM heads: expected {llm_config['num_heads']}, got {actual_heads}")
            actual_kv_heads = first.self_attn.num_key_value_heads
            if actual_kv_heads != llm_config["num_key_value_heads"]:
                errors.append(
                    f"LLM KV heads: expected {llm_config['num_key_value_heads']}, got {actual_kv_heads}"
                )

    if token_embedding is not None:
        expected = (vocab_size, llm_dim)
        actual = tuple(token_embedding.weight.shape)
        if actual != expected:
            errors.append(f"token embedding shape: expected {expected}, got {actual}")

    if lm_head is not None:
        expected = (vocab_size, llm_dim)
        actual = tuple(lm_head.weight.shape)
        if actual != expected:
            errors.append(f"lm_head shape: expected {expected}, got {actual}")

    if box_encoder is not None:
        if box_encoder.embed_dim != preset["vis_embed_dim"]:
            errors.append(
                f"ViT embed dim: expected {preset['vis_embed_dim']}, got {box_encoder.embed_dim}"
            )
        if len(box_encoder.blocks) != preset["vis_depth"]:
            errors.append(
                f"ViT depth: expected {preset['vis_depth']}, got {len(box_encoder.blocks)}"
            )
        if box_encoder.spatial_merger is not None:
            out_dim = box_encoder.spatial_merger.mlp[2].out_features
            if out_dim != preset["merger_out_dim"]:
                errors.append(f"spatial merger out dim: expected {preset['merger_out_dim']}, got {out_dim}")

    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(
            f"Initialized modules do not match selected model '{model_name}'.\n"
            f"  - {joined}\n"
            "Create the pipeline with UIEmbedderConfig.from_model_name(...) for the same model_name "
            "that will be loaded."
        )


def safe_load(param, new_tensor, name, stats=None):
    """Helper to safely load tensor into parameter."""
    if param is None:
        # If model doesn't have this param (e.g. bias=False), skip loading
        # print(f"    [!] Warning: Skipping {name} (param is None in model)")
        return False
        
    try:
        # Check shapes
        if param.shape != new_tensor.shape:
            if stats is not None:
                stats["shape_mismatch"] += 1
            print(f"    [!] Shape mismatch for {name}: model {param.shape} vs loaded {new_tensor.shape}")
            return False
        param.data.copy_(new_tensor)
        if stats is not None:
            stats["loaded"] += 1
        # print(f"    [.] Loaded {name}")
        return True
    except Exception as e:
        if stats is not None:
            stats["errors"] += 1
        print(f"    [!] Error loading {name}: {e}")
        return False

def load_vit_weights(box_encoder, state_dict, stats=None):
    """
    Maps 'visual.blocks.*', 'visual.patch_embed', 'visual.merger' etc.
    """
    print("[-] Loading Vision Encoder weights...")
    
    # 1. Patch Embed
    # Qwen2-VL: visual.patch_embed.proj.weight (usually Conv3d)
    if 'visual.patch_embed.proj.weight' in state_dict:
        w = state_dict['visual.patch_embed.proj.weight']
        print(f"    [i] Found patch_embed weight: {w.shape}")
        
        # Check if our model has Conv3d or Conv2d
        if isinstance(box_encoder.patch_embed, nn.Conv2d) and w.ndim == 5:
            print("    [!] Warning: Loading 5D weight into Conv2d. Trying to squash Temporal dim...")
            # w shape: (Out, In, T, H, W) -> (Out, In, H, W) ?
            # Usually T=2. We can take mean or slice?
            # Or better: The user should have updated the model to Conv3d.
            # We will try to load it safely.
            pass
            
        safe_load(box_encoder.patch_embed.weight, w, "patch_embed.weight", stats)
        
        if 'visual.patch_embed.proj.bias' in state_dict:
            b = state_dict['visual.patch_embed.proj.bias']
            if box_encoder.patch_embed.bias is not None:
                safe_load(box_encoder.patch_embed.bias, b, "patch_embed.bias", stats)
        print("  [+] Processed patch_embed")

    # 2. Blocks
    loaded_blocks = 0
    for i in range(len(box_encoder.blocks)):
        prefix = f"visual.blocks.{i}."
        
        # Check if any key for this block exists
        if f"{prefix}norm1.weight" not in state_dict:
            # We might check other keys just in case
            pass
            
        block = box_encoder.blocks[i]
        
        # Helper to load if exists
        def load_w(attr, key):
             if key in state_dict:
                 safe_load(attr, state_dict[key], key, stats)

        load_w(block.norm1.weight, f"{prefix}norm1.weight")
        load_w(block.norm2.weight, f"{prefix}norm2.weight")
        
        # Attention
        load_w(block.attn.qkv.weight, f"{prefix}attn.qkv.weight")
        load_w(block.attn.qkv.bias, f"{prefix}attn.qkv.bias") # Qwen2-VL usually has bias
        load_w(block.attn.proj.weight, f"{prefix}attn.proj.weight")
        load_w(block.attn.proj.bias, f"{prefix}attn.proj.bias")

        # MLP
        load_w(block.mlp.gate_proj.weight, f"{prefix}mlp.gate_proj.weight")
        load_w(block.mlp.up_proj.weight, f"{prefix}mlp.up_proj.weight")
        load_w(block.mlp.down_proj.weight, f"{prefix}mlp.down_proj.weight")
        
        # Check if we loaded something
        if f"{prefix}norm1.weight" in state_dict:
            loaded_blocks += 1
        
    print(f"  [+] Scanned {len(box_encoder.blocks)} blocks, updated relevant weights.")
    
    # 3. Final Norm
    if "visual.ln_post.weight" in state_dict:
        safe_load(box_encoder.norm_final.weight, state_dict["visual.ln_post.weight"], "visual.ln_post", stats)

    # 4. Spatial Merger (visual.merger) — 2×2 grouping MLP
    # Keys: visual.merger.mlp.0.weight/bias, visual.merger.mlp.2.weight/bias
    if box_encoder.spatial_merger is not None:
        m = box_encoder.spatial_merger.mlp
        if "visual.merger.mlp.0.weight" in state_dict:
            safe_load(m[0].weight, state_dict["visual.merger.mlp.0.weight"], "merger.mlp.0.weight", stats)
        if "visual.merger.mlp.0.bias" in state_dict:
            safe_load(m[0].bias, state_dict["visual.merger.mlp.0.bias"], "merger.mlp.0.bias", stats)
        if "visual.merger.mlp.2.weight" in state_dict:
            safe_load(m[2].weight, state_dict["visual.merger.mlp.2.weight"], "merger.mlp.2.weight", stats)
        if "visual.merger.mlp.2.bias" in state_dict:
            safe_load(m[2].bias, state_dict["visual.merger.mlp.2.bias"], "merger.mlp.2.bias", stats)
        print("  [+] Loaded spatial merger weights")


# -----------------------------------------------------------
# 2. Vision Projector Weights (DEPRECATED — merger is now loaded inside load_vit_weights)
# -----------------------------------------------------------
# visual.merger weights are handled in load_vit_weights → box_encoder.spatial_merger


# -----------------------------------------------------------
# 3. LLM Weights
# -----------------------------------------------------------

def load_llm_weights(headless_llm, state_dict, stats=None):
    print("[-] Loading LLM weights...")
    
    loaded_layers = 0
    for i in range(len(headless_llm.layers)):
        prefix = f"model.layers.{i}."
        layer = headless_llm.layers[i]
        
        def load_w(attr, key):
             if key in state_dict:
                 safe_load(attr, state_dict[key], key, stats)

        # Norms
        load_w(layer.input_layernorm.weight, f"{prefix}input_layernorm.weight")
        load_w(layer.post_attention_layernorm.weight, f"{prefix}post_attention_layernorm.weight")
        
        # Self Attention
        load_w(layer.self_attn.q_proj.weight, f"{prefix}self_attn.q_proj.weight")
        load_w(layer.self_attn.k_proj.weight, f"{prefix}self_attn.k_proj.weight")
        load_w(layer.self_attn.v_proj.weight, f"{prefix}self_attn.v_proj.weight")
        load_w(layer.self_attn.o_proj.weight, f"{prefix}self_attn.o_proj.weight")
        
        load_w(layer.self_attn.q_proj.bias, f"{prefix}self_attn.q_proj.bias")
        load_w(layer.self_attn.k_proj.bias, f"{prefix}self_attn.k_proj.bias")
        load_w(layer.self_attn.v_proj.bias, f"{prefix}self_attn.v_proj.bias")
            
        # MLP
        load_w(layer.mlp.gate_proj.weight, f"{prefix}mlp.gate_proj.weight")
        load_w(layer.mlp.up_proj.weight, f"{prefix}mlp.up_proj.weight")
        load_w(layer.mlp.down_proj.weight, f"{prefix}mlp.down_proj.weight")
        
        if f"{prefix}input_layernorm.weight" in state_dict:
            loaded_layers += 1
        
    print(f"  [+] Scanned layers, updated available weights.")
    
    # 2. Final Norm
    if "model.norm.weight" in state_dict:
        safe_load(headless_llm.norm.weight, state_dict["model.norm.weight"], "model.norm.weight", stats)


# -----------------------------------------------------------
# 4. Text Embedding Weights
# -----------------------------------------------------------

def load_embedding_weights(embedding_module, state_dict, stats=None):
    print("[-] Loading Token Embedding weights...")
    if "model.embed_tokens.weight" in state_dict:
        safe_load(embedding_module.weight, state_dict["model.embed_tokens.weight"], "model.embed_tokens.weight", stats)

# -----------------------------------------------------------
# 5. LM Head Weights (Optional)
# -----------------------------------------------------------

def load_lm_head_weights(lm_head, state_dict, stats=None):
    print("[-] Loading LM Head weights...")
    if "lm_head.weight" in state_dict:
        safe_load(lm_head.weight, state_dict["lm_head.weight"], "lm_head.weight", stats)



# -----------------------------------------------------------
# Main Loader Helper
# -----------------------------------------------------------

def load_all_weights(
    box_encoder=None,
    headless_llm=None,
    token_embedding=None,
    lm_head=None,
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    cache_dir=None
):
    resolved_model_name, preset = _resolve_model_preset(model_name)
    print(f"[*] Starting complete weight loading from {resolved_model_name}...")
    _validate_selected_model(
        resolved_model_name,
        preset,
        box_encoder=box_encoder,
        headless_llm=headless_llm,
        token_embedding=token_embedding,
        lm_head=lm_head,
    )
    if preset is not None:
        print(
            "[*] Loader profile: "
            f"llm_dim={preset['llm_dim']} "
            f"layers={preset['llm_config']['num_layers']} "
            f"vocab={preset['vocab_size']} "
            f"merger_out={preset['merger_out_dim']}"
        )
    stats = _new_stats()
    
    from huggingface_hub import list_repo_files
    
    try:
        all_files = list(list_repo_files(repo_id=resolved_model_name))
        safetensors_files = sorted([f for f in all_files if f.endswith('.safetensors')])
        print(f"[*] Found {len(safetensors_files)} shards.")
    except Exception as e:
        raise RuntimeError(f"Error listing files: {e}")
        
    for filename in safetensors_files:
        print(f"\n[*] Processing shard: {filename}...")
        
        try:
            file_path = hf_hub_download(repo_id=resolved_model_name, filename=filename, cache_dir=cache_dir)
            state_dict = load_file(file_path)
            
            # 1. ViT + spatial merger (visual.merger weights loaded inside)
            if box_encoder is not None:
                load_vit_weights(box_encoder, state_dict, stats)
                
            # 2. LLM
            if headless_llm is not None:
                load_llm_weights(headless_llm, state_dict, stats)

            # 4. Embeddings
            if token_embedding is not None:
                load_embedding_weights(token_embedding, state_dict, stats)
                
            # 5. LM Head
            if lm_head is not None:
                load_lm_head_weights(lm_head, state_dict, stats)

            # Cleanup
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  [!] Error processing {filename}: {e}")
            stats["errors"] += 1
            
    if stats["shape_mismatch"] or stats["errors"]:
        raise RuntimeError(
            "Weight loading failed validation: "
            f"loaded={stats['loaded']} "
            f"shape_mismatch={stats['shape_mismatch']} "
            f"errors={stats['errors']}. "
            "Check that UIEmbedderConfig.model_name and architecture fields match the selected checkpoint."
        )

    print(
        "\n[SUCCESS] Weight loading complete. "
        f"Loaded tensors: {stats['loaded']}, shape mismatches: {stats['shape_mismatch']}."
    )
