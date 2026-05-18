# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print()
    print("=" * 100)
    print("RUN:", " ".join(cmd))
    print("=" * 100)

    result = subprocess.run(cmd, shell=False)

    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="data_source/services_1c.pdf")
    parser.add_argument("--train-pages", type=int, default=100)
    parser.add_argument("--test-start-page", type=int, default=101)
    parser.add_argument("--test-pages", type=int, default=90)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-elements", type=int, default=1200)
    args = parser.parse_args()

    pdf = Path(args.pdf)

    if not pdf.exists():
        raise FileNotFoundError(
            f"PDF не найден: {pdf}\n"
            f"Положи файл сюда: data_source/services_1c.pdf"
        )

    train_rag = "data/services_1c_train_rag"
    test_rag = "data/services_1c_test_rag"
    pairs = "data/services_1c_pairs.jsonl"
    checkpoint = "checkpoints/ui_elements_siamese_services/best.pt"
    index_dir = "indexes/services_1c_test_ui"

    run([
        sys.executable, "-m", "rag.build_pdf_rag",
        "--pdf", str(pdf),
        "--out", train_rag,
        "--start-page", "1",
        "--max-pages", str(args.train_pages),
        "--force",
    ])

    run([
        sys.executable, "-m", "rag.build_ui_elements",
        "--rag-dir", train_rag,
        "--max-pages", str(args.train_pages),
    ])

    run([
        sys.executable, "-m", "training.make_ui_element_siamese_dataset",
        "--rag-dir", train_rag,
        "--out", pairs,
        "--negatives-per-positive", "2",
        "--hard-negatives-ratio", "0.5",
        "--max-elements", str(args.max_elements),
    ])

    run([
        sys.executable, "-m", "training.split_ui_element_pairs",
        "--pairs", pairs,
    ])

    run([
        sys.executable, "-m", "training.train_ui_siamese",
        "--pairs", "data/services_1c_pairs.train.jsonl",
        "--out", checkpoint,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
    ])

    run([
        sys.executable, "-m", "rag.build_pdf_rag",
        "--pdf", str(pdf),
        "--out", test_rag,
        "--start-page", str(args.test_start_page),
        "--max-pages", str(args.test_pages),
        "--force",
    ])

    run([
        sys.executable, "-m", "rag.build_ui_elements",
        "--rag-dir", test_rag,
        "--max-pages", str(args.test_pages),
    ])

    run([
        sys.executable, "-m", "retrieval.build_ui_element_index",
        "--rag-dir", test_rag,
        "--checkpoint", checkpoint,
        "--out-dir", index_dir,
    ])

    print()
    print("=" * 100)
    print("ГОТОВО")
    print("=" * 100)
    print(f"Train RAG: {train_rag}")
    print(f"Test RAG: {test_rag}")
    print(f"Pairs: {pairs}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Index: {index_dir}")
    print()
    print("Теперь запускай:")
    print("python -m streamlit run rag\\streamlit_pdf_rag.py")


if __name__ == "__main__":
    main()
