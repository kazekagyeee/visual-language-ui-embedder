# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_step(name, command):
    print("=" * 80)
    print(name)
    print(" ".join(command))
    print("=" * 80)

    started = time.perf_counter()

    proc = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )

    elapsed = time.perf_counter() - started

    print(proc.stdout)

    if proc.stderr:
        print(proc.stderr)

    return {
        "name": name,
        "command": command,
        "returncode": proc.returncode,
        "time_sec": elapsed,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--max-elements", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    steps = [
        (
            "Build UI elements",
            [
                py, "-m", "rag.build_ui_elements",
                "--rag-dir", args.rag_dir,
                "--max-pages", str(args.max_pages),
            ],
        ),
        (
            "Generate siamese pairs",
            [
                py, "-m", "training.make_ui_element_siamese_dataset",
                "--rag-dir", args.rag_dir,
                "--out", "data/ui_element_pairs.jsonl",
                "--negatives-per-positive", "2",
                "--hard-negatives-ratio", "0.5",
                "--max-elements", str(args.max_elements),
            ],
        ),
        (
            "Split pairs",
            [
                py, "-m", "training.split_ui_element_pairs",
                "--pairs", "data/ui_element_pairs.jsonl",
            ],
        ),
        (
            "Train siamese",
            [
                py, "-m", "training.train_ui_siamese",
                "--pairs", "data/ui_element_pairs.train.jsonl",
                "--out", "checkpoints/ui_elements_siamese/best.pt",
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
            ],
        ),
        (
            "Evaluate siamese pair classifier",
            [
                py, "-m", "training.evaluate_ui_siamese",
                "--pairs", "data/ui_element_pairs.test.jsonl",
                "--checkpoint", "checkpoints/ui_elements_siamese/best.pt",
            ],
        ),
        (
            "Build UI element index",
            [
                py, "-m", "retrieval.build_ui_element_index",
                "--rag-dir", args.rag_dir,
                "--checkpoint", "checkpoints/ui_elements_siamese/best.pt",
                "--out-dir", "indexes/ui_elements_siamese",
            ],
        ),
        (
            "Generate automatic test queries",
            [
                py, "-m", "evaluation.generate_test_queries",
                "--rag-dir", args.rag_dir,
                "--out", "data/test_queries.json",
                "--max-single", "20",
                "--max-multi", "15",
            ],
        ),
        (
            "Benchmark full retrieval",
            [
                py, "-m", "evaluation.benchmark_full_retrieval",
                "--rag-dir", args.rag_dir,
                "--queries", "data/test_queries.json",
                "--top-k-text", "5",
                "--top-k-ui", "8",
                "--alpha", "0.35",
                "--out-dir", str(out_dir),
            ],
        ),
    ]

    results = []
    started_all = time.perf_counter()

    for name, command in steps:
        result = run_step(name, command)
        results.append(result)

        if result["returncode"] != 0:
            print(f"FAILED STEP: {name}")
            break

    total_time = time.perf_counter() - started_all

    summary = {
        "total_time_sec": total_time,
        "steps": results,
    }

    with open(out_dir / "pipeline_run_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "pipeline_run_summary.md", "w", encoding="utf-8") as f:
        f.write("# Pipeline run summary\n\n")
        f.write(f"Total time: {total_time:.2f} sec\n\n")
        f.write("| Step | Return code | Time sec |\n")
        f.write("|---|---:|---:|\n")

        for r in results:
            f.write(f"| {r['name']} | {r['returncode']} | {r['time_sec']:.2f} |\n")

    print("=" * 80)
    print(f"Saved: {out_dir / 'pipeline_run_metrics.json'}")
    print(f"Saved: {out_dir / 'pipeline_run_summary.md'}")


if __name__ == "__main__":
    main()
