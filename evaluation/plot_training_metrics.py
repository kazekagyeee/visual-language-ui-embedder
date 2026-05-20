# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt


EPOCHS = list(range(1, 26))

TRAIN_LOSS = [
    0.5537, 0.3580, 0.3084, 0.2953, 0.2633,
    0.2508, 0.2357, 0.2363, 0.2420, 0.2405,
    0.2224, 0.2246, 0.1928, 0.2070, 0.1955,
    0.1942, 0.1978, 0.1851, 0.1796, 0.1859,
    0.1935, 0.2036, 0.1758, 0.1778, 0.1748,
]

VAL_LOSS = [
    0.4257, 0.3385, 0.2777, 0.3431, 0.2483,
    0.3360, 0.2655, 0.2515, 0.3070, 0.3090,
    0.2817, 0.2368, 0.2572, 0.2828, 0.3002,
    0.2666, 0.2793, 0.2644, 0.2714, 0.2878,
    0.3509, 0.2618, 0.2751, 0.2600, 0.2784,
]

VAL_ACC = [
    0.8182, 0.8084, 0.8734, 0.8474, 0.8604,
    0.8539, 0.8734, 0.8831, 0.8669, 0.8571,
    0.8636, 0.8766, 0.8636, 0.8604, 0.8766,
    0.8669, 0.8604, 0.8734, 0.8734, 0.8831,
    0.8474, 0.8864, 0.8604, 0.8701, 0.8604,
]


def main():
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(EPOCHS, TRAIN_LOSS, marker="o", label="Train Loss")
    plt.plot(EPOCHS, VAL_LOSS, marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Siamese Ranker Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "siamese_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(EPOCHS, VAL_ACC, marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.title("Siamese Ranker Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "siamese_val_accuracy.png", dpi=200)
    plt.close()

    print("Saved:")
    print("reports/siamese_loss_curve.png")
    print("reports/siamese_val_accuracy.png")


if __name__ == "__main__":
    main()
