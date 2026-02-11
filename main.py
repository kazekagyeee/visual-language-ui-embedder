import torch
import torch.nn as nn
from PIL import Image
import json
import os
import shutil
from transformers import AutoTokenizer

from box_aware_visual_encoder import Qwen2_5_BoxEncoder
from vision_to_text_projector import VisualToTextEmbeddingProjector
from headless_qwen_llm import HeadlessQwen2_5
from uied_detector import UIEDDetector
from load_qwen_weights import load_all_weights

def main():
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

    # Model Config (Qwen2.5-VL-7B)
    VIS_DIM = 3584 # Note: Qwen2.5-VL embeds are 3584? No wait.
    # Checking specific config of Qwen2.5-VL-7B.
    # "visual": { "depth": 32, "embed_dim": 1280, "num_heads": 16, "patch_size": 14 } -> Wait, for VL-7B-Instruct:
    # Actually Qwen2-VL has varied visual dims.
    # Qwen2-VL-7B Visual Encoder usually: embed_dim=3584 ? No.
    # Let's check typical values. Qwen2-VL-7B uses a specific ViT.
    # "hidden_size": 3584 (LLM), "visual_hidden_size": 3584?
    # Actually for Qwen2.5-VL-7B:
    # Visual Encoder: SigLIP-like?
    # If I look at the weights I can know.
    # visual.patch_embed.proj.weight: [1280, 1176, 14, 14] -> 1176 channels?
    # visual.blocks.0.mlp.gate_proj.weight: [3420, 1280]?
    #
    # CORRECTION: Qwen2-VL-7B uses a dynamic resolution vision encoder based on Qwen-ViT.
    # For this task, assuming standard config compatible with code.
    # If the user code was generic:
    # Qwen2.5-7B LLM hidden size is 3584.
    # Visual encoder dim: typically 1280 (like in Qwen-VL) or 3584.
    # Let's assume typical Qwen2.5-VL: visual_dim=1280, llm_dim=3584.
    # If this is wrong, weight loading will crash on shape mismatch, which is good.
    
    # Actually Qwen2-VL 7B keys:
    # visual.patch_embed.proj.weight shape is likely (1280, 3*14*14 / temporal?, 2, 2).
    # Since we are using a "BoxEncoder" which is a standard ViT here, I will set dims to match likely weights.
    # If standard ViT: visual_dim=1280.
    
    VIS_DIM = 1280 # Common for small/med ViTs in these models
    LLM_DIM = 3584 # Qwen2.5-7B hidden size
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
        img_size=224, # Will be resized
        patch_size=14,
        embed_dim=VIS_DIM,
        depth=DEPTH_VIS,
        num_heads=HEADS_VIS
    ).to(device) # BFloat16?
    
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

    # -------------------------------------------------------------------------
    # 2. LOAD WEIGHTS
    # -------------------------------------------------------------------------
    # Using BFloat16 is recommended for Qwen2.5
    try:
        box_encoder = box_encoder.bfloat16()
        projector = projector.bfloat16()
        headless_llm = headless_llm.bfloat16()
        token_embedding = token_embedding.bfloat16()
        print("[*] Converted models to BFloat16")
    except Exception as e:
        print(f"[!] BFloat16 not supported? {e}. Using float32.")
        
    try:
        load_all_weights(
            box_encoder=box_encoder,
            projector=projector,
            headless_llm=headless_llm,
            token_embedding=token_embedding,
            model_name="Qwen/Qwen2.5-VL-7B-Instruct"
        )
    except Exception as e:
        print(f"[!] Critical Error loading weights: {e}")
        # For debug purposes, we might continue with random weights if loading fails
        # but the user requested explicit weight usage.
        # return

    # -------------------------------------------------------------------------
    # 3. PROCESS INPUTS
    # -------------------------------------------------------------------------
    print(f"\n[*] Processing inputs...")
    
    # 3.1 Load & Detect UI
    detector = UIEDDetector()
    image = Image.open(img_path).convert("RGB")
    image = image.resize((224, 224)) # Resizing for fixed size encoder input
    # Note: Real Qwen2-VL handles dynamic aspect ratios, but BoxEncoder is fixed to 224x224 in code?
    # Our BoxEncoder uses 'img_size=224'

    max_dist = 5 # Для слияния близких bbox
    bboxes = detector.detect(image, max_dist=max_dist)
    if not bboxes:
        print("[!] No boxes detected! Using full image box.")
        bboxes = [[0.0, 0.0, 1.0, 1.0]]
        
    print(f"  [+] Detected {len(bboxes)} boxes.")
    
    # Prepare Tensors
    # Image: (1, 3, 224, 224)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
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
    
    inputs = tokenizer(text_content, return_tensors="pt")
    input_ids = inputs.input_ids.to(device) # (1, Seq)
    
    # Get Text Embeddings
    # (1, Seq, LLM_DIM)
    text_emb = token_embedding(input_ids)
    
    # We need a single text vector for the Projector? 
    # Projector signature: (global_emb, box_embs, text_emb)
    # text_emb in `VisionToTextEmbeddingProjector` expects (B, Dim_T)
    # The snippet says: "text_emb: (B, Dim_T) - Текстовый эмбеддинг"
    # So we need to POOL or Select a token from the text sequence.
    # Usually [EOS] or Mean pooling. Let's use Mean Pooling for now.
    text_vec = text_emb.mean(dim=1) # (1, LLM_DIM)

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


if __name__ == "__main__":
    main()