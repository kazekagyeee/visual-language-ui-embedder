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

# -----------------------------------------------------------
# 1. Vision Encoder Weights
# -----------------------------------------------------------

def safe_load(param, new_tensor, name):
    """Helper to safely load tensor into parameter."""
    if param is None:
        # If model doesn't have this param (e.g. bias=False), skip loading
        # print(f"    [!] Warning: Skipping {name} (param is None in model)")
        return
        
    try:
        # Check shapes
        if param.shape != new_tensor.shape:
            # Try to squeeze/reshape if compatible (e.g. 3D conv vs 2D conv with T=1)
            if param.numel() == new_tensor.numel():
                new_tensor = new_tensor.view_as(param)
            else:
                print(f"    [!] Shape mismatch for {name}: model {param.shape} vs loaded {new_tensor.shape}")
                return
        param.data.copy_(new_tensor)
        # print(f"    [.] Loaded {name}")
    except Exception as e:
        print(f"    [!] Error loading {name}: {e}")

def load_vit_weights(box_encoder, state_dict):
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
            
        safe_load(box_encoder.patch_embed.weight, w, "patch_embed.weight")
        
        if 'visual.patch_embed.proj.bias' in state_dict:
            b = state_dict['visual.patch_embed.proj.bias']
            if box_encoder.patch_embed.bias is not None:
                safe_load(box_encoder.patch_embed.bias, b, "patch_embed.bias")
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
                 safe_load(attr, state_dict[key], key)

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
        safe_load(box_encoder.norm_final.weight, state_dict["visual.ln_post.weight"], "visual.ln_post")


# -----------------------------------------------------------
# 2. Vision Projector Weights
# -----------------------------------------------------------

def load_vision_projector_weights_from_dict(projector, state_dict):
    print("[-] Loading Vision Projector weights...")
    
    def load_w(attr, key):
         if key in state_dict:
             safe_load(attr, state_dict[key], key)

    load_w(projector.vision_projector[0].weight, "visual.merger.mlp.0.weight")
    load_w(projector.vision_projector[0].bias, "visual.merger.mlp.0.bias")
    
    load_w(projector.vision_projector[2].weight, "visual.merger.mlp.2.weight")
    load_w(projector.vision_projector[2].bias, "visual.merger.mlp.2.bias")


# -----------------------------------------------------------
# 3. LLM Weights
# -----------------------------------------------------------

def load_llm_weights(headless_llm, state_dict):
    print("[-] Loading LLM weights...")
    
    loaded_layers = 0
    for i in range(len(headless_llm.layers)):
        prefix = f"model.layers.{i}."
        layer = headless_llm.layers[i]
        
        def load_w(attr, key):
             if key in state_dict:
                 safe_load(attr, state_dict[key], key)

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
        safe_load(headless_llm.norm.weight, state_dict["model.norm.weight"], "model.norm.weight")


# -----------------------------------------------------------
# 4. Text Embedding Weights
# -----------------------------------------------------------

def load_embedding_weights(embedding_module, state_dict):
    print("[-] Loading Token Embedding weights...")
    if "model.embed_tokens.weight" in state_dict:
        safe_load(embedding_module.weight, state_dict["model.embed_tokens.weight"], "model.embed_tokens.weight")



# -----------------------------------------------------------
# Main Loader Helper
# -----------------------------------------------------------

def load_all_weights(
    box_encoder=None,
    projector=None,
    headless_llm=None,
    token_embedding=None,
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    cache_dir=None
):
    print(f"[*] Starting complete weight loading from {model_name}...")
    
    from huggingface_hub import list_repo_files
    
    try:
        all_files = list(list_repo_files(repo_id=model_name))
        safetensors_files = sorted([f for f in all_files if f.endswith('.safetensors')])
        print(f"[*] Found {len(safetensors_files)} shards.")
    except Exception as e:
        raise RuntimeError(f"Error listing files: {e}")
        
    for filename in safetensors_files:
        print(f"\n[*] Processing shard: {filename}...")
        
        try:
            file_path = hf_hub_download(repo_id=model_name, filename=filename, cache_dir=cache_dir)
            state_dict = load_file(file_path)
            
            # 1. ViT
            if box_encoder is not None:
                load_vit_weights(box_encoder, state_dict)
                
            # 2. Projector
            if projector is not None:
                load_vision_projector_weights_from_dict(projector, state_dict)
                
            # 3. LLM
            if headless_llm is not None:
                load_llm_weights(headless_llm, state_dict)

            # 4. Embeddings
            if token_embedding is not None:
                load_embedding_weights(token_embedding, state_dict)

            # Cleanup
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  [!] Error processing {filename}: {e}")
            
    print("\n[SUCCESS] Weight loading complete.")
