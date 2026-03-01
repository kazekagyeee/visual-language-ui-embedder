import json
import numpy as np
import os

def check_embeddings(file_path):
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print("[!] No data in JSON.")
        return

    print(f"[*] Found {len(data)} embeddings.")
    
    embeddings = np.array([item['embedding'] for item in data])
    
    # 1. Check for duplicates (all-zeros or identical)
    unique_rows = np.unique(embeddings, axis=0)
    print(f"[*] Unique embeddings: {len(unique_rows)}/{len(embeddings)}")
    
    # 2. Check for zero embeddings
    zeros = np.all(embeddings == 0, axis=1)
    if np.any(zeros):
        print(f"[!] Warning: {np.sum(zeros)} embeddings are all zeros!")
    
    # 3. Basic stats
    print(f"[*] Stats: Min={embeddings.min():.4f}, Max={embeddings.max():.4f}, Mean={embeddings.mean():.4f}, Std={embeddings.std():.4f}")
    
    # 4. Cosine Similarity Hub
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_emb = embeddings / (norms + 1e-9)
    sim_matrix = np.dot(norm_emb, norm_emb.T)
    
    avg_sim = (np.sum(sim_matrix) - len(embeddings)) / (len(embeddings) * (len(embeddings) - 1)) if len(embeddings) > 1 else 1.0
    print(f"[*] Average inter-component cosine similarity: {avg_sim:.4f}")
    
    if avg_sim > 0.999:
        print("[!] WARNING: Extremely high similarity! Embeddings might be collapsing or weights weren't loaded correctly.")

if __name__ == "__main__":
    check_embeddings("output/embeddings.json")
 