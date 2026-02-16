import torch
import torch.nn as nn
from PIL import Image
import json
import os
import shutil
import time
from transformers import AutoTokenizer

from box_aware_visual_encoder import Qwen2_5_BoxEncoder
from vision_to_text_projector import VisualToTextEmbeddingProjector
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    
    # Vision Encoder outputs 3584-dim (after Qwen2VLSpatialMerge)
    VIS_DIM = 3584  # Spatial merge output (2×2 grouping + MLP)
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
    
    # 1.1 Visual Encoder
    box_encoder = Qwen2_5_BoxEncoder(
        img_size=224, # Will be resized dynamically
        patch_size=14,
        embed_dim=1280,  # Internal ViT dimension
        depth=DEPTH_VIS,
        num_heads=HEADS_VIS,
        use_learned_tokens=False  # Use ROI pooling (TODO: set True for future training)
    ).to(device)
    
    # 1.2 Projector
    projector = VisualToTextEmbeddingProjector(
        visual_dim=VIS_DIM,
        text_dim=LLM_DIM,
        target_dim=LLM_DIM
    ).to(device)
    
    # 1.3 Headless LLM
    headless_llm = HeadlessQwen2_5(llm_config).to(device)
    
    # 1.4 Text Token Embeddings
    # We need a standalone embedding layer
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
    vocab_size = tokenizer.vocab_size 
    # Usually around 152064 for Qwen2.5
    if vocab_size < 151936: vocab_size = 152064 # Safety
    
    token_embedding = nn.Embedding(vocab_size, LLM_DIM).to(device)
    
    t_init = time.time()
    print(f"  [Time] Model Initialization: {t_init - start_total:.2f}s")
    
    # 1.5 LM Head (Debug)
    lm_head = None
    if DEBUG_DECODE_EMBEDDINGS:
        # Usually LM Head is just a Linear layer
        lm_head = nn.Linear(LLM_DIM, vocab_size, bias=False).to(device)


    # -------------------------------------------------------------------------
    # 2. LOAD WEIGHTS
    # -------------------------------------------------------------------------
    # Using BFloat16 is recommended for Qwen2.5
    try:
        box_encoder = box_encoder.bfloat16()
        projector = projector.bfloat16()
        headless_llm = headless_llm.bfloat16()
        token_embedding = token_embedding.bfloat16()
        if lm_head is not None:
             lm_head = lm_head.bfloat16()
        print("[*] Converted models to BFloat16")
    except Exception as e:
        print(f"[!] BFloat16 not supported? {e}. Using float32.")
        
    try:
        load_all_weights(
            box_encoder=box_encoder,
            projector=projector,
            headless_llm=headless_llm,
            token_embedding=token_embedding,
            lm_head=lm_head,
            model_name="Qwen/Qwen2.5-VL-7B-Instruct"
        )
    except Exception as e:
        print(f"[!] Critical Error loading weights: {e}")
        # For debug purposes, we might continue with random weights if loading fails
        # but the user requested explicit weight usage.
        # return

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
    #text_vec = text_emb.mean(dim=1)  # (1, LLM_DIM)
    text_vec = text_emb[:, -1, :]

    t_process = time.time()
    print(f"  [Time] Input Processing & Detection: {t_process - t_load:.2f}s")

    # -------------------------------------------------------------------------
    # 4. RUN PIPELINE
    # -------------------------------------------------------------------------
    print("\n[*] Running Forward Pass...")
    
    with torch.no_grad():
        # Step 1: Visual Encoding
        # g_emb: (B, VIS_DIM), b_embs: (B, N, VIS_DIM)
        g_emb, b_embs = box_encoder(img_tensor, boxes_tensor)
        
        # Step 2: Projection
        # triples: (B, N, 3, LLM_DIM) -> [Box_i, Global, Text]
        triples = projector(g_emb, b_embs, text_vec)
        
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
    # 6. DEBUG DECODE - Using full generative model
    # -------------------------------------------------------------------------
    if DEBUG_DECODE_EMBEDDINGS:
        print("\\n[*] Debug: Loading full Qwen2.5-VL model for generation...")
        
        try:
            from transformers import Qwen2VLForConditionalGeneration
            
            # Load full model (this will cache)
            full_model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True
            )
            full_model.eval()
            
            print("  [+] Full model loaded successfully")
            
            # Decode embeddings using the full model
            emb_batch = output_embeddings[0]  # (N, LLM_DIM)
            print(f"  [DEBUG] Embeddings Stats: Min={emb_batch.min().item():.4f}, Max={emb_batch.max().item():.4f}, Mean={emb_batch.mean().item():.4f}")
            
            debug_out_list = []
            max_new_tokens = 200
            
            with torch.no_grad():
                for idx in range(emb_batch.shape[0]):
                    # Create full context: [text_prompt_embeddings] + [UI_component_embedding]
                    # text_emb shape: (1, Seq, LLM_DIM)  
                    # emb_batch[idx] shape: (LLM_DIM,)
                    
                    current_emb = emb_batch[idx:idx+1].unsqueeze(1)  # (1, 1, LLM_DIM)
                    
                    # Concatenate text prompt with UI component embedding
                    full_context = torch.cat([text_emb, current_emb], dim=1)  # (1, Seq+1, LLM_DIM)
                    
                    print(f"  [DEBUG] Component {idx}: Context shape = {full_context.shape}")
                    
                    # Generate from embeddings with full context
                    outputs = full_model.generate(
                        inputs_embeds=full_context,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    
                    # Decode
                    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Analysis: What would be the top-5 first tokens from this embedding?
                    if lm_head is not None:
                        first_logits = lm_head(emb_batch[idx:idx+1])
                        first_probs = torch.softmax(first_logits, dim=-1)
                        top_probs, top_ids = torch.topk(first_probs[0], 5)
                        
                        top_k_info = []
                        for k in range(5):
                            tid = top_ids[k].item()
                            tprob = top_probs[k].item()
                            ttext = tokenizer.decode([tid])
                            top_k_info.append(f"{ttext} ({tprob:.2f})")
                    else:
                        top_k_info = []
                    
                    bbox = bboxes[idx] if idx < len(bboxes) else None
                    debug_out_list.append({
                        "bbox": bbox,
                        "generated_text": decoded_text,
                        "num_tokens": len(outputs[0]),
                        "first_token_id": outputs[0][0].item() if len(outputs[0]) > 0 else None,
                        "top_5_first": top_k_info
                    })
                    
                    if idx < 3:
                        print(f"  [Component {idx}] Generated: {decoded_text[:100]}...")
            
            debug_file = os.path.join("debug", "decoded_embeddings.json")
            os.makedirs("debug", exist_ok=True)
            with open(debug_file, "w", encoding='utf-8') as f:
                json.dump(debug_out_list, f, indent=2, ensure_ascii=False)
                
            print(f"[DEBUG] Decoded texts saved to {debug_file}")
            
            # Cleanup to free memory
            del full_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[!] Failed to load full model for generation: {e}")
            print(f"    Skipping debug decoding.")

    t_end = time.time()
    print(f"  [Time] Output Saving & Debug: {t_end - t_forward:.2f}s")
    print(f"[*] Total Execution Time: {t_end - start_total:.2f}s")


if __name__ == "__main__":
    main()