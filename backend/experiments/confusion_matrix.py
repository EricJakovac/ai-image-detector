import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

RESULTS_DIR = "experiments/results"
OUTPUT_DIR = RESULTS_DIR


def plot_confusion_matrix(cm, title, output_path):
    plt.figure(figsize=(5, 4))

    # 1 = correct (TN, TP), 0 = error (FP, FN)
    color_mask = np.array([[1, 0], [0, 1]])

    # Blage boje: error, correct
    cmap = ListedColormap(
        [
            "#77c8f7",  # greške (FP, FN)
            "#ffffff",  # točno (TN, TP)
        ]
    )

    plt.imshow(color_mask, cmap=cmap)

    plt.title(title, fontsize=12)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks([0, 1], ["Real", "AI"])
    plt.yticks([0, 1], ["Real", "AI"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i][j],
                ha="center",
                va="center",
                fontsize=12,
                color="black",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    files = [
        f
        for f in os.listdir(RESULTS_DIR)
        if f.startswith("eval_results") and f.endswith(".json")
    ]

    print(f"Found {len(files)} result files")

    for file in files:
        file_path = os.path.join(RESULTS_DIR, file)

        with open(file_path, "r") as f:
            data = json.load(f)

        for model_name, metrics in data.items():
            cm = metrics.get("confusion_matrix")
            if cm is None:
                continue

            cm = np.array(cm)

            title = model_name  # samo ime modela
            output_name = f"confmat_{model_name}.png"
            output_path = os.path.join(OUTPUT_DIR, output_name)

            plot_confusion_matrix(cm, title, output_path)
            print(f"Saved: {output_name}")


if __name__ == "__main__":
    main()
