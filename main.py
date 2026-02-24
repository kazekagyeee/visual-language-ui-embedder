import torch
import torch.nn as nn
from PIL import Image
import json
import os
import shutil
import time
from transformers import AutoTokenizer

from box_aware_visual_encoder import Qwen2_5_BoxEncoder
from headless_qwen_llm import HeadlessQwen2_5
from uied_detector import UIEDDetector
from load_qwen_weights import load_all_weights

def smart_resize(image: Image.Image, patch_size: int = 14) -> Image.Image:
    """
    Resizes image to ensure dimensions are multiples of patch_size.
    This prevents shape mismatch errors in ViT patch embedding.
    """
    w, h = image.size
    
    # Calculate new dimensions
    new_w = (w // patch_size) * patch_size
    new_h = (h // patch_size) * patch_size
    
    # Enforce minimum size if needed (e.g. 1 patch)
    if new_w < patch_size: new_w = patch_size
    if new_h < patch_size: new_h = patch_size
    
    if (new_w, new_h) != (w, h):
        print(f"[*] Smart Resize: {w}x{h} -> {new_w}x{new_h}")
        return image.resize((new_w, new_h), resample=Image.BICUBIC)
    
    return image


def main():
    start_total = time.time()
    
    # -------------------------------------------------------------------------
    # 0. SETUP & CONFIG
    # -------------------------------------------------------------------------
    # my GPU only has 16gb of vram so cpu only(
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"[*] Using device: {device}")

    # Paths
    img_path = r"input_images\image_20_2.png"
    txt_path = r"input_images\image_20_2.txt"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Flags
    DEBUG_DECODE_EMBEDDINGS = True
    
    SYSTEM_PROMPT = "You are a UI context describer assistant. You answer in russian."
    CONTEXT_PROMPT = "Тебе нужно описать UI элемент в контексте основного изображения и его описания: первое - целое изображение UI, второе - сам описываемый компонент, а далее текст, который является контекстом к основному изображению: "
    
    # Vision Encoder outputs 3584-dim directly (after Qwen2VLSpatialMerge = visual.merger)
    # No separate projector needed — merger IS the projector per Qwen2.5-VL paper
    LLM_DIM = 3584  # Qwen2.5-7B hidden size
    HEADS_VIS = 16 
    DEPTH_VIS = 32 # Qwen2-VL-7B model usually has deep vision/llm
    
    # Qwen2.5-7B LLM Config
    llm_config = {
        'hidden_size': LLM_DIM,
        'num_heads': 28,
        'num_key_value_heads': 4,
        'intermediate_size': 18944,
        'num_layers': 28, 
        'rms_norm_eps': 1e-6,
        'rope_theta': 1000000.0,
    }

    # -------------------------------------------------------------------------
    # 1. LOAD COMPONENTS
    # -------------------------------------------------------------------------
    print("[*] Initializing models...")
    
    # 1.1 Visual Encoder (includes visual.merger = spatial merger MLP)
    box_encoder = Qwen2_5_BoxEncoder(
        img_size=224, # Will be resized dynamically
        patch_size=14,
        embed_dim=1280,  # Internal ViT dimension
        depth=DEPTH_VIS,
        num_heads=HEADS_VIS,
        use_learned_tokens=False  # Use ROI pooling (TODO: set True for future training)
    ).to(device)
    
    # 1.2 Headless LLM
    headless_llm = HeadlessQwen2_5(llm_config).to(device)
    
    # 1.3 Text Token Embeddings
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
    vocab_size = tokenizer.vocab_size 
    # Usually around 152064 for Qwen2.5
    if vocab_size < 151936: vocab_size = 152064 # Safety
    
    token_embedding = nn.Embedding(vocab_size, LLM_DIM).to(device)
    
    t_init = time.time()
    print(f"  [Time] Model Initialization: {t_init - start_total:.2f}s")



    # -------------------------------------------------------------------------
    # 2. LOAD WEIGHTS
    # -------------------------------------------------------------------------
    # Using BFloat16 is recommended for Qwen2.5
    try:
        box_encoder = box_encoder.bfloat16()
        headless_llm = headless_llm.bfloat16()
        token_embedding = token_embedding.bfloat16()
        print("[*] Converted models to BFloat16")
    except Exception as e:
        print(f"[!] BFloat16 not supported? {e}. Using float32.")
        
    try:
        load_all_weights(
            box_encoder=box_encoder,
            headless_llm=headless_llm,
            token_embedding=token_embedding,
            lm_head=None,  # Not needed for similarity analysis
            model_name="Qwen/Qwen2.5-VL-7B-Instruct"
        )
    except Exception as e:
        print(f"[!] Critical Error loading weights: {e}")

    t_load = time.time()
    print(f"  [Time] Weight Loading: {t_load - t_init:.2f}s")

    # -------------------------------------------------------------------------
    # 3. PROCESS INPUTS
    # -------------------------------------------------------------------------
    print(f"\n[*] Processing inputs...")
    
    # 3.1 Load & Detect UI
    detector = UIEDDetector()
    image = Image.open(img_path).convert("RGB")
    
    # Apply Smart Resize to align with patch grid
    # Qwen2.5-VL uses 2×2 spatial merge → need multiple of 28 (14 × 2)
    image = smart_resize(image, patch_size=28)

    max_dist = 17 # Для слияния близких bbox
    
    # Debug output path
    debug_bbox_path = os.path.join("debug", "uied_bbox_debug.png")
    
    bboxes = detector.detect(image, max_dist=max_dist, debug_output_path=debug_bbox_path)
    if not bboxes:
        print("[!] No boxes detected! Using full image box.")
        bboxes = [[0.0, 0.0, 1.0, 1.0]]
        
    print(f"  [+] Detected {len(bboxes)} boxes.")
    
    # Prepare Tensors
    # Image: (1, 3, 224, 224)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Standard ImageNet norm, or Qwen specific? Qwen usually just /255
    ])
    img_tensor = transform(image).unsqueeze(0).to(device).bfloat16()
    
    # Boxes: (1, N, 4)
    boxes_tensor = torch.tensor([bboxes]).to(device).float() # Boxes positional doesn't need bf16 usually?
    # BoxEncoder might use them in mask creation
    
    # 3.2 Tokenize Text
    with open(txt_path, 'r', encoding='utf-8') as f:
        text_content = f.read().strip()
    
    print(f"  [+] Text content: {text_content[:50]}...")
    
    # Create Chat Template (Simplified for headless usage)
    # <|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
    prompt_parts = []
    if SYSTEM_PROMPT:
        prompt_parts.append(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
    
    prompt_parts.append(f"<|im_start|>user\n{CONTEXT_PROMPT + text_content}<|im_end|>\n")
    prompt_parts.append("<|im_start|>assistant\n")
    
    prompt_text = "".join(prompt_parts)
    
    print(f"  [+] Prompt text: {prompt_text[:50]}...")
    
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device) # (1, Seq)
    
    # Get Text Embeddings
    # (1, Seq, LLM_DIM)
    text_emb = token_embedding(input_ids)

    # Use MEAN POOLING for semantic representation (better for RAG)
    # Last token is for generation, mean is for retrieval

    text_vec = text_emb.mean(dim=1)  # (1, LLM_DIM)
    #text_vec = text_emb[:, -1, :]

    t_process = time.time()
    print(f"  [Time] Input Processing & Detection: {t_process - t_load:.2f}s")

    # -------------------------------------------------------------------------
    # 4. RUN PIPELINE
    # -------------------------------------------------------------------------
    print("\n[*] Running Forward Pass...")
    
    with torch.no_grad():
        # Step 1: Visual Encoding
        # box_encoder includes visual.merger (spatial merge MLP)
        # g_emb: (B, LLM_DIM=3584), b_embs: (B, N, LLM_DIM=3584)
        g_emb, b_embs = box_encoder(img_tensor, boxes_tensor)
        
        # Step 2: Assemble triples [Box_i, Global, Text] for each box
        # All are already in LLM_DIM space — no separate projector needed
        B, N, D = b_embs.shape
        v_global_expanded = g_emb.unsqueeze(1).expand(-1, N, -1)   # (B, N, D)
        t_expanded = text_vec.unsqueeze(1).expand(-1, N, -1)        # (B, N, D)
        triples = torch.stack([b_embs, v_global_expanded, t_expanded], dim=2)  # (B, N, 3, D)
        
        # Step 3: LLM Refinement
        # fused: (B, N, LLM_DIM)
        output_embeddings = headless_llm(triples)
        
    print(f"  [+] Output Shape: {output_embeddings.shape}")

    t_forward = time.time()
    print(f"  [Time] Forward Pass: {t_forward - t_process:.2f}s")
    
    # -------------------------------------------------------------------------
    # 5. SAVE OUTPUT
    # -------------------------------------------------------------------------
    output_file = os.path.join(output_dir, "embeddings.json")
    
    # Prepare JSON
    # We have N embeddings. We can map them to boxes.
    out_list = []
    out_data = output_embeddings[0].cpu().float().numpy() # (N, Dim)
    
    for i in range(len(bboxes)):
        item = {
            "bbox": bboxes[i],
            "embedding": out_data[i].tolist()
        }
        out_list.append(item)
        
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(out_list, f)
        
    print(f"[SUCCESS] Embeddings saved to {output_file}")
    
    # -------------------------------------------------------------------------
    # 6. DEBUG: SIMILARITY ANALYSIS
    # -------------------------------------------------------------------------
    if DEBUG_DECODE_EMBEDDINGS:
        print("\n[*] Debug: Analyzing embedding similarities...")
        
        # Normalize embeddings for cosine similarity
        emb_batch = output_embeddings[0]  # (N, LLM_DIM)
        print(f"  [DEBUG] Embeddings Stats: Min={emb_batch.min().item():.4f}, Max={emb_batch.max().item():.4f}, Mean={emb_batch.mean().item():.4f}")
        
        # L2 normalize
        emb_normalized = torch.nn.functional.normalize(emb_batch, p=2, dim=1)
        
        # Also normalize text embedding for comparison
        text_vec_normalized = torch.nn.functional.normalize(text_vec, p=2, dim=1)
        
        # 1. Component-to-Component Similarity Matrix
        # Shape: (N, N) where N is number of components
        similarity_matrix = torch.mm(emb_normalized, emb_normalized.t())
        
        print(f"\n  [+] Component-to-Component Similarity Matrix:")
        print(f"      Shape: {similarity_matrix.shape}")
        print(f"      Diagonal (self-similarity): {similarity_matrix.diag().tolist()}")
        
        # 2. Text-to-Component Similarities
        # Shape: (1, N) - similarity of text prompt to each UI component
        text_to_components = torch.mm(text_vec_normalized, emb_normalized.t())
        
        print(f"\n  [+] Text-to-Component Similarities:")
        print(f"      Shape: {text_to_components.shape}")
        print(f"      Values: {text_to_components[0].tolist()}")
        
        # 3. Find most similar pairs
        # Convert to float32 first since NumPy doesn't support BFloat16
        similarity_np = similarity_matrix.float().cpu().numpy()
        debug_out = {
            "similarity_matrix": similarity_matrix.float().cpu().tolist(),
            "text_to_components": text_to_components[0].float().cpu().tolist(),
            "components": []
        }
        
        for idx in range(len(bboxes)):
            # Get similarities to other components (excluding self)
            sims = similarity_np[idx].copy()
            sims[idx] = -1  # Ignore self-similarity
            
            # Find most similar component
            most_similar_idx = sims.argmax()
            most_similar_score = sims[most_similar_idx]
            
            component_info = {
                "index": idx,
                "bbox": bboxes[idx],
                "text_similarity": text_to_components[0][idx].item(),
                "most_similar_component": {
                    "index": int(most_similar_idx),
                    "bbox": bboxes[most_similar_idx] if most_similar_idx < len(bboxes) else None,
                    "similarity": float(most_similar_score)
                },
                "average_similarity_to_others": float(sims[sims >= 0].mean())
            }
            
            debug_out["components"].append(component_info)
            
            if idx < 3:
                print(f"\n  [Component {idx}]")
                print(f"    BBox: {bboxes[idx]}")
                print(f"    Text similarity: {component_info['text_similarity']:.4f}")
                print(f"    Most similar to component {most_similar_idx}: {most_similar_score:.4f}")
        
        # Save to file
        debug_file = os.path.join("debug", "embedding_similarities.json")
        os.makedirs("debug", exist_ok=True)
        with open(debug_file, "w", encoding='utf-8') as f:
            json.dump(debug_out, f, indent=2, ensure_ascii=False)
            
        print(f"\n[DEBUG] Similarity analysis saved to {debug_file}")

    t_end = time.time()
    print(f"  [Time] Output Saving & Debug: {t_end - t_forward:.2f}s")
    print(f"[*] Total Execution Time: {t_end - start_total:.2f}s")


if __name__ == "__main__":
    main()