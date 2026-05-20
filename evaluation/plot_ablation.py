# -*- coding: utf-8 -*-

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    reports = Path("reports")

    with open(reports / "ablation_eval.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    configs = data["configs"]

    names = [x["name"] for x in configs]
    precision = [x["precision"] for x in configs]
    recall = [x["recall"] for x in configs]
    f1 = [x["f1"] for x in configs]
    success = [x["success_rate"] for x in configs]

    metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Success Rate": success,
    }

    for metric_name, values in metrics.items():
        plt.figure(figsize=(7, 4))
        plt.bar(names, values)
        plt.ylim(0, 1.05)
        plt.ylabel(metric_name)
        plt.title(f"Ablation Study: {metric_name}")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()

        out = reports / f"ablation_{metric_name.lower().replace(' ', '_')}.png"
        plt.savefig(out, dpi=200)
        plt.close()

        print("Saved:", out)


if __name__ == "__main__":
    main()
