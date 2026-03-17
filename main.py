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
        print(f"    Model : {self.config.model_name}")
        print(f"    LLM dim: {self.config.llm_dim}, layers: {self.config.llm_config['num_layers']}")

        # 1.1 Visual Encoder (ViT + spatial merger MLP)
        # The spatial merger projects ViT features from vis_embed_dim (1280)
        # to merger_out_dim, which must equal llm_dim so the LLM can consume them directly.
        self.box_encoder = Qwen2_5_BoxEncoder(
            img_size=self.config.img_size,
            patch_size=self.config.patch_size_encoder,
            embed_dim=self.config.vis_embed_dim,
            depth=self.config.vis_depth,
            num_heads=self.config.vis_heads,
            intermediate_size=self.config.vis_intermediate_size,
            use_learned_tokens=False,
        ).to(self.device)

        # Patch spatial merger output dimension to match the selected model's LLM dim
        # (merger_out_dim varies: 1536 for 2B, 2048 for 3B, 3584 for 7B, …)
        if self.box_encoder.spatial_merger is not None:
            from box_aware_visual_encoder import Qwen2VLSpatialMerge
            self.box_encoder.spatial_merger = Qwen2VLSpatialMerge(
                in_dim=self.config.vis_embed_dim,
                out_dim=self.config.merger_out_dim,
            ).to(self.device)

        # 1.2 Headless LLM
        self.headless_llm = HeadlessQwen2_5(self.config.llm_config).to(self.device)

        # 1.3 Text Token Embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)

        # Use vocab_size from the preset (more reliable than tokenizer.vocab_size on some models)
        vocab_size = self.config.vocab_size

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
        
    def _prepare_text_embeddings(
        self,
        text_content: str,
        bbox: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Builds a per-box tokenized prompt and returns its token embeddings.

        Point 1: bbox coordinates are embedded directly in the prompt text so that
        each UI element gets a unique textual description.  Different bboxes produce
        different token sequences → different RoPE-encoded keys/values in the LLM
        → the last-token hidden state encodes both spatial position and visual context.

        Point 2: an explicit <|im_end|> EOS token is appended *after* the assistant
        prefix so that the last token is always a well-defined "summary anchor",
        following the convention of sentence-embedding models (E5, SBERT).
        """
        print(f"  [+] Text content: {text_content[:50]}...")

        # -- Point 1: inject bbox spatial context into the prompt --
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            spatial_tag = f"[bbox: ({x1:.3f},{y1:.3f})-({x2:.3f},{y2:.3f})] "
        else:
            spatial_tag = ""

        # Chat template:
        # <|im_start|>system\n{sys}<|im_end|>\n
        # <|im_start|>user\n{spatial_tag}{context}{text}<|im_end|>\n
        # <|im_start|>assistant\n<|im_end|>   ← EOS anchor (Point 2)
        prompt_parts = []
        if self.config.system_prompt:
            prompt_parts.append(f"<|im_start|>system\n{self.config.system_prompt}<|im_end|>\n")

        prompt_parts.append(
            f"<|im_start|>user\n"
            f"{spatial_tag}{self.config.context_prompt}{text_content}"
            f"<|im_end|>\n"
        )
        # Point 2: add explicit EOS token so it becomes the last (summary) token
        prompt_parts.append("<|im_start|>assistant\n<|im_end|>")

        prompt_text = "".join(prompt_parts)
        print(f"  [+] Prompt (first 80 chars): {prompt_text[:80]}...")

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)  # (1, Seq)

        # Token embeddings: (1, Seq, LLM_DIM)
        text_emb = self.token_embedding(input_ids)

        assert text_emb.shape[-1] == self.config.llm_dim, (
            f"[DIM MISMATCH] text_emb dim={text_emb.shape[-1]} != LLM_DIM={self.config.llm_dim}."
        )
        print(f"  [Debug] text_emb shape: {text_emb.shape}  ✓ last token = <|im_end|> EOS anchor")
        return text_emb

    def _forward_pass(self, image: Image.Image, text_content: str, bboxes: Optional[List[List[float]]] = None) -> Tuple[torch.Tensor, List[List[float]]]:
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
        
        t_process = time.time()
        print(f"  [Time] Input Processing & Detection: {t_process - t_start:.2f}s")
        
        print("\n[*] Running Forward Pass...")

        with torch.no_grad():
            # Step 1: Visual Encoding
            # g_seq: (1, H_m*W_m, D) — all merged ViT patches, SAME for every box
            # b_seqs: list of (1, N_b, D)  — ROI patches, UNIQUE per box
            g_seq, b_seqs = self.box_encoder(img_tensor, boxes_tensor)

            # --- Fix 1: compress g_seq to a single mean-pooled summary token ---
            # The full g_seq (~400 tokens for a 560×560 image) was making up >95% of
            # every LLM input sequence and was IDENTICAL across all boxes.
            # The ViT attention already baked global context into every patch, so one
            # mean-pooled vector is a sufficient global summary for the LLM.
            # g_summary = g_seq.mean(dim=1, keepdim=True)  # (1, 1, D)
            s_prefix = 1  # 1 summary token is the bidirectional prefix

            # Step 2: Assemble per-box sequences:
            #   [g_summary(1) | box_patches(N_b) | text+EOS(T)]
            # Track box-patch positions for Fix 2 pooling.
            seqs_list = []
            s_box_starts = []
            s_box_ends   = []

            for i, b_seq in enumerate(b_seqs):
                bbox_coords  = bboxes[i]
                n_box_tokens = b_seq.shape[1]

                # Per-box text embeddings (includes spatial tag + EOS anchor)
                text_emb_box = self._prepare_text_embeddings(text_content, bbox=bbox_coords)

                # Sequence positions:
                #   [0]                    → g_summary
                #   [1 : 1+n_box_tokens]   → box patches  ← pool these
                #   [1+n_box_tokens : end] → text + EOS
                box_start = 1
                box_end   = 1 + n_box_tokens

                combined_seq = torch.cat([b_seq, text_emb_box], dim=1)
                total_len    = combined_seq.shape[1]

                print(f"  [SeqDebug] Box {i}: summary=1, box_patches={n_box_tokens}, "
                      f"text={text_emb_box.shape[1]}, total={total_len}  "
                      f"box_pool=[{box_start}:{box_end}]")

                seqs_list.append(combined_seq)
                s_box_starts.append(box_start)
                s_box_ends.append(box_end)

            # Step 3: LLM + pooling at box-patch positions
            output_embeddings = self.headless_llm(
                seqs_list,
                s_prefix=s_prefix,
                s_box_starts=s_box_starts,
                s_box_ends=s_box_ends,
            )
            
        print(f"  [+] Output Shape: {output_embeddings.shape}")

        t_forward = time.time()
        print(f"  [Time] Forward Pass: {t_forward - t_process:.2f}s")

        return output_embeddings, bboxes
        
    def process(self, image: Image.Image, text_content: str, bboxes: Optional[List[List[float]]] = None) -> Dict[Tuple[float, float, float, float], List[float]]:
        """
        Processes image and text, returns a dictionary mapping bounding boxes to their embeddings.
        """
        output_embeddings, bboxes_out = self._forward_pass(image, text_content, bboxes)

        if self.config.debug_decode_embeddings:
            self.analyze_similarities(output_embeddings, bboxes_out)
            
        out_dict = {}
        out_data = output_embeddings[0].cpu().float().numpy() # (N, Dim)
        
        for i, bbox in enumerate(bboxes_out):
            out_dict[tuple(bbox)] = out_data[i].tolist()
            
        return out_dict
        
    def analyze_similarities(self, output_embeddings: torch.Tensor, bboxes: List[List[float]]) -> Dict[str, Any]:
        """
        Analyzes cosine similarities between generated embeddings and saves results.
        NOTE: text-to-component similarity is no longer computed here because each box
        now has its own unique prompt (Point 1), so there is no single shared text vector.
        """
        if not self.config.debug_decode_embeddings:
            return {}

        print("\n[*] Debug: Analyzing embedding similarities...")

        # Normalize embeddings for cosine similarity
        emb_batch = output_embeddings[0]  # (N, LLM_DIM)
        print(f"  [DEBUG] Embeddings Stats: Min={emb_batch.min().item():.4f}, Max={emb_batch.max().item():.4f}, Mean={emb_batch.mean().item():.4f}")

        # L2 normalize
        emb_normalized = torch.nn.functional.normalize(emb_batch, p=2, dim=1)

        # Component-to-Component Similarity Matrix: (N, N)
        similarity_matrix = torch.mm(emb_normalized, emb_normalized.t())

        print(f"\n  [+] Component-to-Component Similarity Matrix:")
        print(f"      Shape: {similarity_matrix.shape}")
        print(f"      Diagonal (self-similarity): {similarity_matrix.diag().tolist()}")

        # Convert to float32 for NumPy
        similarity_np = similarity_matrix.float().cpu().numpy()
        debug_out = {
            "similarity_matrix": similarity_matrix.float().cpu().tolist(),
            "components": []
        }

        for idx in range(len(bboxes)):
            sims = similarity_np[idx].copy()
            sims[idx] = -1  # Ignore self-similarity

            most_similar_idx = sims.argmax()
            most_similar_score = sims[most_similar_idx]

            component_info = {
                "index": idx,
                "bbox": bboxes[idx],
                "most_similar_component": {
                    "index": int(most_similar_idx),
                    "bbox": bboxes[most_similar_idx] if most_similar_idx < len(bboxes) else None,
                    "similarity": float(most_similar_score)
                },
                "average_similarity_to_others": float(sims[sims >= 0].mean()) if len(bboxes) > 1 else 1.0
            }

            debug_out["components"].append(component_info)

            if idx < 3:
                print(f"\n  [Component {idx}]")
                print(f"    BBox: {bboxes[idx]}")
                print(f"    Most similar to component {most_similar_idx}: {most_similar_score:.4f}")
                print(f"    Avg similarity to others: {component_info['average_similarity_to_others']:.4f}")

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

    # --- Choose model size ---
    # Option A: use a short alias
    #   config = UIEmbedderConfig.from_model_name("3B")          # ~3B params, faster
    #   config = UIEmbedderConfig.from_model_name("2B")          # ~2B params, lightest
    # Option B: use the full HuggingFace repo ID
    #   config = UIEmbedderConfig.from_model_name("Qwen/Qwen2.5-VL-3B-Instruct", device="cuda")
    # Option C: keep default 7B (original behaviour)
    config = UIEmbedderConfig.from_model_name("2B")

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

    # Save output
    output_file = os.path.join(output_dir, "embeddings.json")
    out_list = [
        {"bbox": list(bbox), "embedding": emb}
        for bbox, emb in embeddings_dict.items()
    ]
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(out_list, f)

    print(f"[SUCCESS] Embeddings saved to {output_file}")
    print(f"[*] Total Execution Time: {time.time() - start_total:.2f}s")


if __name__ == "__main__":
    main()