import torch
import torch.nn as nn
from PIL import Image
import json
import os
import time
from transformers import AutoTokenizer
from typing import List, Optional, Tuple, Dict, Any

from box_aware_visual_encoder import Qwen2_5_BoxEncoder
from headless_qwen_llm import HeadlessQwen2_5
from uied_detector import UIEDDetector
from load_qwen_weights import load_all_weights
from config import UIEmbedderConfig

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

class UIEmbedderPipeline:
    def __init__(self, config: Optional[UIEmbedderConfig] = None):
        """
        Initializes the pipeline with the given configuration.
        """
        self.config = config if config is not None else UIEmbedderConfig()
        self.device = self.config.device
        print(f"[*] Using device: {self.device}")
        
        self._initialize_models()
        self._load_weights()
        
        self.detector = UIEDDetector()
        
        from torchvision import transforms
        # Prepare Tensors
        # Image: (1, 3, 224, 224)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # Standard ImageNet norm, or Qwen specific? Qwen usually just /255
        ])
        
    def _initialize_models(self):
        start_total = time.time()
        print("[*] Initializing models...")
        
        # 1.1 Visual Encoder (includes visual.merger = spatial merger MLP)
        self.box_encoder = Qwen2_5_BoxEncoder(
            img_size=self.config.img_size, # Will be resized dynamically
            patch_size=self.config.patch_size_encoder,
            embed_dim=1280,  # Internal ViT dimension
            depth=self.config.depth_vis,
            num_heads=self.config.heads_vis,
            use_learned_tokens=False  # Use ROI pooling (TODO: set True for future training)
        ).to(self.device)
        
        # 1.2 Headless LLM
        self.headless_llm = HeadlessQwen2_5(self.config.llm_config).to(self.device)
        
        # 1.3 Text Token Embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        vocab_size = self.tokenizer.vocab_size 
        # Usually around 152064 for Qwen2.5
        if vocab_size < 151936: vocab_size = 152064 # Safety
        
        self.token_embedding = nn.Embedding(vocab_size, self.config.llm_dim).to(self.device)
        
        t_init = time.time()
        print(f"  [Time] Model Initialization: {t_init - start_total:.2f}s")

    def _load_weights(self):
        t_init = time.time()
        # Using BFloat16 is recommended for Qwen2.5
        try:
            self.box_encoder = self.box_encoder.bfloat16()
            self.headless_llm = self.headless_llm.bfloat16()
            self.token_embedding = self.token_embedding.bfloat16()
            print("[*] Converted models to BFloat16")
        except Exception as e:
            print(f"[!] BFloat16 not supported? {e}. Using float32.")
            
        try:
            load_all_weights(
                box_encoder=self.box_encoder,
                headless_llm=self.headless_llm,
                token_embedding=self.token_embedding,
                lm_head=None,  # Not needed for similarity analysis
                model_name=self.config.model_name
            )
        except Exception as e:
            print(f"[!] Critical Error loading weights: {e}")

        t_load = time.time()
        print(f"  [Time] Weight Loading: {t_load - t_init:.2f}s")
        
    def extract_bboxes(self, image: Image.Image) -> List[List[float]]:
        # Debug output path
        os.makedirs(self.config.debug_dir, exist_ok=True)
        debug_bbox_path = os.path.join(self.config.debug_dir, "uied_bbox_debug.png")
        
        bboxes = self.detector.detect(image, max_dist=self.config.max_dist, debug_output_path=debug_bbox_path)
        if not bboxes:
            print("[!] No boxes detected! Using full image box.")
            bboxes = [[0.0, 0.0, 1.0, 1.0]]
            
        print(f"  [+] Detected {len(bboxes)} boxes.")
        return bboxes
        
    def _prepare_text_embeddings(self, text_content: str) -> torch.Tensor:
        print(f"  [+] Text content: {text_content[:50]}...")
        
        # Create Chat Template (Simplified for headless usage)
        # <|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
        prompt_parts = []
        if self.config.system_prompt:
            prompt_parts.append(f"<|im_start|>system\n{self.config.system_prompt}<|im_end|>\n")
        
        prompt_parts.append(f"<|im_start|>user\n{self.config.context_prompt + text_content}<|im_end|>\n")
        prompt_parts.append("<|im_start|>assistant\n")
        
        prompt_text = "".join(prompt_parts)
        
        print(f"  [+] Prompt text: {prompt_text[:50]}...")
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device) # (1, Seq)
        
        # Get Text Embeddings
        # (1, Seq, LLM_DIM)
        text_emb = self.token_embedding(input_ids)

        # Debug: verify text embedding dim matches LLM hidden size (3584 for Qwen2.5-7B)
        assert text_emb.shape[-1] == self.config.llm_dim, (
            f"[DIM MISMATCH] text_emb dim={text_emb.shape[-1]} != LLM_DIM={self.config.llm_dim}. "
            f"Token embedding table may have wrong hidden size."
        )
        print(f"  [Debug] text_emb shape: {text_emb.shape}  ✓ dim matches LLM_DIM={self.config.llm_dim}")
        return text_emb

    def _forward_pass(self, image: Image.Image, text_content: str, bboxes: Optional[List[List[float]]] = None) -> Tuple[torch.Tensor, List[List[float]], torch.Tensor]:
        """
        Internal forward pass.
        Processes image and text, returns raw embeddings and bounding boxes.
        """
        t_start = time.time()
        print(f"\n[*] Processing inputs...")
        
        # Apply Smart Resize to align with patch grid
        # Qwen2.5-VL uses 2×2 spatial merge → need multiple of 28 (14 × 2)
        image = smart_resize(image, patch_size=self.config.patch_size_resize)
        
        if bboxes is None:
            bboxes = self.extract_bboxes(image)
            
        # Image: (1, 3, 224, 224)
        img_tensor = self.transform(image).unsqueeze(0).to(self.device).bfloat16()
        
        # Boxes: (1, N, 4)
        boxes_tensor = torch.tensor([bboxes]).to(self.device).float() # Boxes positional doesn't need bf16 usually?
        
        text_emb = self._prepare_text_embeddings(text_content)
        
        t_process = time.time()
        print(f"  [Time] Input Processing & Detection: {t_process - t_start:.2f}s")
        
        print("\n[*] Running Forward Pass...")
        
        with torch.no_grad():
            # Step 1: Visual Encoding
            # box_encoder includes visual.merger (spatial merge MLP)
            # g_seq: (1, H*W, 3584) full image sequence
            # b_seqs: list of (1, N_box_patches, 3584) sequences
            g_seq, b_seqs = self.box_encoder(img_tensor, boxes_tensor)
            
            # Step 2: Assemble sequences [Global_Seq, Box_Seq, Text_Seq] for each box
            # We need a list of sequences to pass to Headless LLM because length varies per box.
            seqs_list = []
            
            # text_emb is (1, S_text, 3584) — using the full text token sequence!
            # No mean pooling here anymore, text is treated as a sequence.
            for b_seq in b_seqs:
                # Concat along sequence dimension (dim=1)
                # Resulting shape: (1, N_global + N_bbox + N_text, 3584)
                combined_seq = torch.cat([g_seq, b_seq, text_emb], dim=1)
                seqs_list.append(combined_seq)
                
            # Step 3: LLM Refinement & Pooling
            # output_embeddings: (1, N_boxes, 3584)
            output_embeddings = self.headless_llm(seqs_list)
            
        print(f"  [+] Output Shape: {output_embeddings.shape}")

        t_forward = time.time()
        print(f"  [Time] Forward Pass: {t_forward - t_process:.2f}s")
        
        return output_embeddings, bboxes, text_emb
        
    def process(self, image: Image.Image, text_content: str, bboxes: Optional[List[List[float]]] = None) -> Dict[Tuple[float, float, float, float], List[float]]:
        """
        Processes image and text, returns a dictionary mapping bounding boxes to their embeddings.
        """
        output_embeddings, bboxes_out, text_emb = self._forward_pass(image, text_content, bboxes)
        
        if self.config.debug_decode_embeddings:
            self.analyze_similarities(output_embeddings, text_emb, bboxes_out)
            
        out_dict = {}
        out_data = output_embeddings[0].cpu().float().numpy() # (N, Dim)
        
        for i, bbox in enumerate(bboxes_out):
            out_dict[tuple(bbox)] = out_data[i].tolist()
            
        return out_dict
        
    def analyze_similarities(self, output_embeddings: torch.Tensor, text_emb: torch.Tensor, bboxes: List[List[float]]) -> Dict[str, Any]:
        """
        Analyzes cosine similarities between generated embeddings and saves results.
        """
        if not self.config.debug_decode_embeddings:
            return {}
            
        print("\n[*] Debug: Analyzing embedding similarities...")
        
        # Normalize embeddings for cosine similarity
        emb_batch = output_embeddings[0]  # (N, LLM_DIM)
        print(f"  [DEBUG] Embeddings Stats: Min={emb_batch.min().item():.4f}, Max={emb_batch.max().item():.4f}, Mean={emb_batch.mean().item():.4f}")
        
        # L2 normalize
        emb_normalized = torch.nn.functional.normalize(emb_batch, p=2, dim=1)
        
        # Also normalize text embedding for comparison
        text_vec_normalized = torch.nn.functional.normalize(text_emb.mean(dim=1), p=2, dim=1)
        
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
        os.makedirs(self.config.debug_dir, exist_ok=True)
        debug_file = os.path.join(self.config.debug_dir, "embedding_similarities.json")
        with open(debug_file, "w", encoding='utf-8') as f:
            json.dump(debug_out, f, indent=2, ensure_ascii=False)
            
        print(f"\n[DEBUG] Similarity analysis saved to {debug_file}")
        return debug_out

# Example usage when running as main script
def main():
    start_total = time.time()
    
    # Optional override parameters via config
    config = UIEmbedderConfig()
    pipeline = UIEmbedderPipeline(config)
    
    # Paths
    img_path = r"input_images\image_20_2.png"
    txt_path = r"input_images\image_20_2.txt"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load inputs
    image = Image.open(img_path).convert("RGB")
    with open(txt_path, 'r', encoding='utf-8') as f:
        text_content = f.read().strip()
        
    # Process
    embeddings_dict = pipeline.process(image, text_content)
    
    # 5. SAVE OUTPUT
    output_file = os.path.join(output_dir, "embeddings.json")
    
    # Prepare JSON
    out_list = []
    for bbox, emb in embeddings_dict.items():
        item = {
            "bbox": list(bbox),
            "embedding": emb
        }
        out_list.append(item)
        
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(out_list, f)
        
    print(f"[SUCCESS] Embeddings saved to {output_file}")
    
    t_end = time.time()

    print(f"[*] Total Execution Time: {t_end - start_total:.2f}s")


if __name__ == "__main__":
    main()