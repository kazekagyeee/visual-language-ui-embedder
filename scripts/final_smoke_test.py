# -*- coding: utf-8 -*-

from pathlib import Path

REQUIRED = [
    "data/all_pdf_rag/items.jsonl",
    "data/all_pdf_rag/embeddings.npy",
    "data/ui_index/ui_items.jsonl",
    "data/ui_index/ui_embeddings.npy",
    "data/ui_trained_index/ui_items.jsonl",
    "data/ui_trained_index/ui_embeddings.npy",
    "checkpoints/ui_siamese_ranker.pt",
    "reports/ui_retrieval_eval.json",
    "reports/ablation_eval.json",
    "rag/streamlit_pdf_rag.py",
]

def main():
    ok = True

    print("=== FINAL SMOKE TEST ===")

    for path in REQUIRED:
        p = Path(path)

        if p.exists():
            print(f"[OK] {path}")
        else:
            print(f"[MISS] {path}")
            ok = False

    if ok:
        print("\nRESULT: OK")
    else:
        print("\nRESULT: SOME FILES ARE MISSING")

if __name__ == "__main__":
    main()
