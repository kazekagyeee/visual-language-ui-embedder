# -*- coding: utf-8 -*-

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    reports = Path("reports")
    path = reports / "ui_retrieval_eval.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = {
        "Success Rate": data["success_rate"],
        "Precision": data["mean_precision"],
        "Recall": data["mean_recall"],
        "F1": data["mean_f1"],
        "MRR": data["mean_mrr"],
        "Hit@1": data["hit_at_1"],
        "Hit@3": data["hit_at_3"],
        "Hit@5": data["hit_at_5"],
        "PDF Accuracy": data["pdf_accuracy"],
    }

    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(11, 5))
    plt.bar(names, values)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Value")
    plt.title("UI Retrieval Evaluation Metrics")
    plt.tight_layout()

    out = reports / "ui_retrieval_metrics.png"
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
