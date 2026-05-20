# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)


def load_json(path: str):
    p = REPORTS / path
    if not p.exists():
        print("[MISS]", p)
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_bar(labels, values, title, ylabel, filename):
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(REPORTS / filename, dpi=200)
    plt.close()
    print("[OK]", REPORTS / filename)


def main():
    ui = load_json("ui_retrieval_eval.json")
    if ui:
        summary = ui.get("summary", ui)
        labels = ["Success", "Precision", "Recall", "F1", "MRR", "Hit@1", "Hit@3", "Hit@5", "PDF Acc"]
        keys = ["success_rate", "mean_precision", "mean_recall", "mean_f1", "mean_mrr", "hit_at_1", "hit_at_3", "hit_at_5", "pdf_accuracy"]
        values = [float(summary.get(k, 0)) for k in keys]
        save_bar(labels, values, "UI Retrieval Metrics", "score", "ui_retrieval_metrics.png")

    user = load_json("user_scenario_eval.json")
    if user:
        summary = user.get("summary", user)
        labels = ["Success", "Precision", "Recall", "F1"]
        keys = ["success_rate", "mean_precision", "mean_recall", "mean_f1"]
        values = [float(summary.get(k, 0)) for k in keys]
        save_bar(labels, values, "User Scenario Evaluation", "score", "user_scenario_metrics.png")

    ablation = load_json("ablation_eval.json")
    if ablation:
        labels = []
        f1 = []
        for name, row in ablation.items():
            if isinstance(row, dict):
                labels.append(name)
                f1.append(float(row.get("f1", 0)))
        if labels:
            save_bar(labels, f1, "Ablation Study: F1", "F1", "ablation_f1.png")

    stats = load_json("dataset_statistics.json")
    if stats:
        ui_types = stats.get("ui_types", {})
        if ui_types:
            labels = list(ui_types.keys())
            values = [int(v) for v in ui_types.values()]
            save_bar(labels, values, "Distribution of UI Element Types", "count", "ui_type_distribution.png")

        pdf_distribution = stats.get("pdf_distribution", {})
        if pdf_distribution:
            labels = list(pdf_distribution.keys())
            values = [int(v) for v in pdf_distribution.values()]
            save_bar(labels, values, "PDF Distribution", "count", "pdf_distribution.png")


if __name__ == "__main__":
    main()
