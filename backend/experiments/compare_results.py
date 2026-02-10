import os
import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from datetime import datetime

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

INPUT_JSON = os.path.join(RESULTS_DIR, "eval_v1_results_20260210_021222.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

print(f"📊 Učitano {len(data)} modela")

# ---------------- METRICS ----------------
METRICS = [
    "accuracy",
    "f1_score",
    "roc_auc",
    "specificity",
    "sensitivity",
    "balanced_accuracy",
]

models = list(data.keys())

results = {}

for model, vals in data.items():
    spec = vals.get("specificity", 0)
    sens = vals.get("sensitivity", 0)
    balanced_acc = (spec + sens) / 2

    results[model] = {
        "accuracy": vals["accuracy"],
        "f1_score": vals["f1_score"],
        "roc_auc": vals["roc_auc"],
        "specificity": spec,
        "sensitivity": sens,
        "balanced_accuracy": balanced_acc,
        "num_samples": vals["num_samples"],
        "num_real": vals["num_real"],
        "num_ai": vals["num_ai"],
    }


# ---------------- BAR CHART ----------------
def plot_bar(metric, filename):
    values = [results[m][metric] for m in models]

    plt.figure(figsize=(10, 6))
    plt.bar(models, values)
    plt.title(f"{metric.upper()} comparison (V1 models)")
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(axis="y")

    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    path = os.path.join(RESULTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"🖼 Spremljen graf: {path}")


plot_bar("accuracy", "v1_accuracy_bar.png")
plot_bar("f1_score", "v1_f1_bar.png")
plot_bar("roc_auc", "v1_roc_auc_bar.png")
plot_bar("balanced_accuracy", "v1_balanced_accuracy_bar.png")

# ---------------- RADAR CHART ----------------
radar_metrics = ["accuracy", "f1_score", "roc_auc", "balanced_accuracy"]

angles = [n / float(len(radar_metrics)) * 2 * pi for n in range(len(radar_metrics))]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

for model in models:
    values = [results[model][m] for m in radar_metrics]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=model)
    ax.fill(angles, values, alpha=0.1)

plt.xticks(angles[:-1], radar_metrics)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"])
plt.ylim(0, 1)
plt.title("V1 Models Radar Comparison")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

radar_path = os.path.join(RESULTS_DIR, "v1_radar_comparison.png")
plt.tight_layout()
plt.savefig(radar_path, dpi=200)
plt.close()
print(f"🖼 Spremljen radar: {radar_path}")

# ---------------- TXT REPORT ----------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
txt_path = os.path.join(RESULTS_DIR, f"v1_report_{timestamp}.txt")

with open(txt_path, "w") as f:
    f.write("V1 MODELS EVALUATION REPORT\n")
    f.write("=" * 40 + "\n\n")

    for model, vals in results.items():
        f.write(f"Model: {model}\n")
        for k, v in vals.items():
            if isinstance(v, float):
                f.write(f"  {k}: {v:.4f}\n")
            else:
                f.write(f"  {k}: {v}\n")
        f.write("\n")

print(f"📝 TXT report spremljen: {txt_path}")

# ---------------- JSON SUMMARY ----------------
summary_path = os.path.join(RESULTS_DIR, f"v1_summary_{timestamp}.json")

with open(summary_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"💾 JSON summary spremljen: {summary_path}")

print("🏁 Usporedba gotova!")
