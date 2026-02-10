"""
Utility module for loading Qwen2-VL vision projector weights from Hugging Face.
Only downloads the necessary weight files instead of the full model.
"""

import torch
import torch.nn as nn
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def find_projector_keys(state_dict):
    """
    Find vision projector weight keys in the state dict.
    
    Common patterns in Qwen2-VL:
    - visual.merger.*
    - visual.projector.*
    - merger.*
    - projector.*
    """
    projector_keys = {}
    
    # Search for projector-related keys
    for key in state_dict.keys():
        key_lower = key.lower()
        if any(pattern in key_lower for pattern in ['projector', 'merger']):
            if 'visual' in key_lower or 'vision' in key_lower:
                projector_keys[key] = state_dict[key]
    
    return projector_keys


def map_weights_to_projector(projector, state_dict):
    """
    Map downloaded weights to our VisualToTextEmbeddingProjector.
    
    Args:
        projector: VisualToTextEmbeddingProjector instance
        state_dict: Dictionary of weights from Hugging Face model
    """
    # Find all projector-related keys
    projector_weights = find_projector_keys(state_dict)
    
    if not projector_weights:
        raise ValueError(
            "Could not find vision projector weights in the model. "
            f"Available keys: {list(state_dict.keys())[:10]}..."
        )
    
    print(f"Found {len(projector_weights)} projector weight tensors:")
    for key in sorted(projector_weights.keys()):
        print(f"  {key}: {projector_weights[key].shape}")
    
    # Try to map weights based on common naming patterns
    # Pattern 1: visual.merger.0.weight, visual.merger.2.weight
    # Pattern 2: visual.projector.0.weight, visual.projector.2.weight
    
    mapped = False
    
    # Look for keys with in dices 0 and 2 (skipping GELU or other activation at index 1)
    # Try various naming patterns from different Qwen2-VL versions
    for base_name in ['visual.merger.mlp', 'visual.projector', 'visual.merger', 'merger', 'projector']:
        key_0_weight = f"{base_name}.0.weight"
        key_0_bias = f"{base_name}.0.bias"
        key_2_weight = f"{base_name}.2.weight"
        key_2_bias = f"{base_name}.2.bias"
        
        if all(k in state_dict for k in [key_0_weight, key_2_weight]):
            print(f"\n[+] Found matching pattern: {base_name}.*")
            
            # Get shapes
            w0_shape = state_dict[key_0_weight].shape
            w2_shape = state_dict[key_2_weight].shape
            print(f"  Layer 0 weight shape: {w0_shape}")
            print(f"  Layer 2 weight shape: {w2_shape}")
            
            # Load layer 0 (first Linear)
            # Note: PyTorch Linear weights are stored as (out_features, in_features)
            # Keep original dtype (BFloat16)
            projector.vision_projector[0].weight.data = state_dict[key_0_weight]
            if key_0_bias in state_dict:
                projector.vision_projector[0].bias.data = state_dict[key_0_bias]
            
            # Load layer 2 (second Linear, after GELU)
            projector.vision_projector[2].weight.data = state_dict[key_2_weight]
            if key_2_bias in state_dict:
                projector.vision_projector[2].bias.data = state_dict[key_2_bias]
            
            print(f"  [>] Loaded {base_name}.0 into vision_projector[0]")
            print(f"  [>] Loaded {base_name}.2 into vision_projector[2]")
            
            mapped = True
            break
    
    if not mapped:
        raise ValueError(
            "Could not automatically map weights. Please check the model structure.\n"
            f"Found keys: {sorted(projector_weights.keys())}"
        )
    
    print("\n[SUCCESS] Successfully loaded vision projector weights!")


def load_vision_projector_weights(
    projector,
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    cache_dir=None
):
    """
    Load vision projector weights from Hugging Face model.
    
    Downloads only the specific safetensors file(s) containing vision weights
    instead of the full model (~100-500MB instead of 15GB).
    
    Args:
        projector: VisualToTextEmbeddingProjector instance
        model_name: HuggingFace model identifier
        cache_dir: Optional custom cache directory
    
    Example:
        >>> from vision_to_text_projector import VisualToTextEmbeddingProjector
        >>> from load_qwen_weights import load_vision_projector_weights
        >>> 
        >>> projector = VisualToTextEmbeddingProjector(
        ...     visual_dim=1024, text_dim=3584, target_dim=3584
        ... )
        >>> load_vision_projector_weights(projector)
    """
    print(f"[*] Downloading vision projector weights from {model_name}...")
    
    # List all files in the repository to find safetensors files
    from huggingface_hub import list_repo_files
    
    print("[*] Discovering model files...")
    try:
        all_files = list(list_repo_files(repo_id=model_name))
        safetensors_files = [f for f in all_files if f.endswith('.safetensors')]
        
        if not safetensors_files:
            raise RuntimeError(f"No safetensors files found in {model_name}")
        
        print(f"[*] Found {len(safetensors_files)} safetensors file(s)")
        for f in safetensors_files:
            print(f"    - {f}")
        
    except Exception as e:
        raise RuntimeError(f"Could not list files in {model_name}: {e}")
    
    # Try to load vision projector weights from each shard
    # Vision weights are typically in the first shard
    model_file = None
    state_dict = None
    
    for filename in sorted(safetensors_files):
        try:
            print(f"\n[*] Trying {filename}...")
            model_file = hf_hub_download(
                repo_id=model_name,
                filename=filename,
                cache_dir=cache_dir
            )
            print(f"[+] Downloaded: {filename}")
            
            # Load the safetensors file
            print(f"[*] Loading weights from {filename}...")
            state_dict = load_file(model_file)
            
            # Check if this file contains vision projector weights
            projector_weights = find_projector_keys(state_dict)
            
            if projector_weights:
                print(f"[+] Found vision projector weights in {filename}!")
                break
            else:
                print(f"[-] No vision projector weights in {filename}, trying next shard...")
                state_dict = None
        
        except Exception as e:
            print(f"[-] Error loading {filename}: {e}")
            continue
    
    if state_dict is None:
        raise RuntimeError(
            f"Could not find vision projector weights in any safetensors file.\n"
            f"Checked files: {safetensors_files}"
        )
    
    # Map weights to our projector
    map_weights_to_projector(projector, state_dict)
    
    print(f"[SUCCESS] Vision projector is now loaded with pre-trained weights!")
