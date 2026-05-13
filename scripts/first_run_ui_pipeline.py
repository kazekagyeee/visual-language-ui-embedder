# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n" + "=" * 100)
    print(" ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--negatives", type=int, default=5)
    args = parser.parse_args()

    py = sys.executable

    rag_dir = Path(args.rag_dir)

    if not (rag_dir / "pages").exists():
        raise FileNotFoundError(
            f"Не найдены страницы PDF: {rag_dir / 'pages'}\n"
            "Сначала собери PDF RAG через rag.build_pdf_rag."
        )

    run([
        py, "-m", "rag.build_ui_elements",
        "--rag-dir", args.rag_dir,
    ])

    run([
        py, "-m", "training.make_ui_element_siamese_dataset",
        "--rag-dir", args.rag_dir,
        "--out", "data/ui_element_pairs.jsonl",
        "--negatives-per-positive", str(args.negatives),
    ])

    run([
        py, "-m", "training.split_ui_element_pairs",
        "--pairs", "data/ui_element_pairs.jsonl",
        "--train-out", "data/ui_element_pairs.train.jsonl",
        "--test-out", "data/ui_element_pairs.test.jsonl",
        "--test-ratio", "0.2",
    ])

    run([
        py, "-m", "training.train_ui_siamese",
        "--pairs", "data/ui_element_pairs.train.jsonl",
        "--out", "checkpoints/ui_elements_siamese/best.pt",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
    ])

    run([
        py, "-m", "training.evaluate_ui_siamese",
        "--pairs", "data/ui_element_pairs.test.jsonl",
        "--checkpoint", "checkpoints/ui_elements_siamese/best.pt",
    ])

    run([
        py, "-m", "retrieval.build_ui_element_index",
        "--rag-dir", args.rag_dir,
        "--checkpoint", "checkpoints/ui_elements_siamese/best.pt",
        "--out-dir", "indexes/ui_elements_siamese",
    ])

    run([
        py, "-m", "retrieval.build_ui_vector_db",
        "--rag-dir", args.rag_dir,
        "--checkpoint", "checkpoints/ui_elements_siamese/best.pt",
        "--db-dir", "vector_db/ui_elements",
    ])

    print("\nDONE.")
    print("Run Streamlit:")
    print("python -m streamlit run rag\\streamlit_pdf_rag.py")


if __name__ == "__main__":
    main()
