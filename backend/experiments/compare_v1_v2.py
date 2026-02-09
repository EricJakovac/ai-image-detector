import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
JSON_FILE = os.path.join(RESULTS_DIR, "eval_results_20260209_122217.json")


def load_results():
    if not os.path.exists(JSON_FILE):
        print("❌ Ne postoji JSON:", JSON_FILE)
        return None

    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    print(f"📊 Učitano {len(data)} modela")
    return data


def normalize_model_names(results):
    grouped = {}
    for full_name, metrics in results.items():
        name = full_name.lower()

        if "_v2_2k" in name:
            base_name = name.replace("_v2_2k", "")
            version = "V2"
        elif "_v1" in name:
            base_name = name.replace("_v1", "")
            version = "V1"
        else:
            base_name = name
            version = "V?"

        grouped[full_name] = {
            "base_name": base_name,
            "version": version,
            "metrics": metrics,
        }

    return grouped


def create_plots(grouped):
    print("🎨 Crtam grafove...")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    colors = {"V1": "#1f77b4", "V2": "#ff7f0e"}
    metrics_to_plot = ["accuracy", "f1_score", "roc_auc"]
    base_models = sorted(set(v["base_name"] for v in grouped.values()))

    # -------- BAR CHART --------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    width = 0.35

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        for i, base in enumerate(base_models):
            v1 = [
                v["metrics"].get(metric, 0)
                for v in grouped.values()
                if v["base_name"] == base and v["version"] == "V1"
            ]
            v2 = [
                v["metrics"].get(metric, 0)
                for v in grouped.values()
                if v["base_name"] == base and v["version"] == "V2"
            ]

            v1_val = v1[0] if v1 else 0
            v2_val = v2[0] if v2 else 0

            ax.bar(
                i - width / 2,
                v1_val,
                width,
                color=colors["V1"],
                label="V1" if i == 0 else "",
            )
            ax.bar(
                i + width / 2,
                v2_val,
                width,
                color=colors["V2"],
                label="V2" if i == 0 else "",
            )

            ax.text(i - width / 2, v1_val + 0.01, f"{v1_val:.3f}", ha="center")
            ax.text(i + width / 2, v2_val + 0.01, f"{v2_val:.3f}", ha="center")

        ax.set_xticks(range(len(base_models)))
        ax.set_xticklabels(base_models, rotation=45)
        ax.set_ylim([0, 1])
        ax.set_title(metric.upper())
        ax.set_ylabel(metric)

    axes[0].legend()
    plt.tight_layout()
    bar_path = os.path.join(RESULTS_DIR, "v1_v2_bar_comparison_2k.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()

    # -------- RADAR CHART --------
    radar_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "specificity",
    ]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    for base in base_models:
        for version in ["V1", "V2"]:
            vals = [
                v["metrics"].get(m, 0)
                for v in grouped.values()
                if v["base_name"] == base and v["version"] == version
                for m in radar_metrics
            ]
            if vals:
                values = vals + vals[:1]
                ax.plot(
                    angles, values, label=f"{base}-{version}", color=colors[version]
                )
                ax.fill(angles, values, alpha=0.1, color=colors[version])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", " ").upper() for m in radar_metrics])
    ax.set_ylim([0, 1])
    ax.set_title("Usporedba V1 vs V2 modela", y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    radar_path = os.path.join(RESULTS_DIR, "v1_v2_radar_comparison_2k.png")
    plt.savefig(radar_path, dpi=300)
    plt.close()

    return {"bar_plot": bar_path, "radar_plot": radar_path}


def generate_report(grouped, plots):
    print("📝 Generiram izvještaj...")

    report = {
        "title": "V1 vs V2 usporedba",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
        "plots": plots,
    }

    for name, data in grouped.items():
        report["models"][name] = {
            "version": data["version"],
            "metrics": data["metrics"],
        }

    json_path = os.path.join(RESULTS_DIR, "v1_v2_comparison_report_2k.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    txt_path = os.path.join(RESULTS_DIR, "v1_v2_comparison_report_2k.txt")
    with open(txt_path, "w") as f:
        f.write("V1 vs V2 COMPARISON REPORT\n")
        f.write("=" * 50 + "\n\n")

        for base in sorted(set(v["base_name"] for v in grouped.values())):
            f.write(f"MODEL: {base}\n")
            for version in ["V1", "V2"]:
                for v in grouped.values():
                    if v["base_name"] == base and v["version"] == version:
                        f.write(f"  {version}:\n")
                        for m, val in v["metrics"].items():
                            if isinstance(val, (int, float)):
                                f.write(f"    {m}: {val:.4f}\n")
                            else:
                                f.write(f"    {m}: {val}\n")
            f.write("-" * 40 + "\n")

    return json_path, txt_path


def main():
    print("📊 Pokrećem usporedbu V1 vs V2...")

    results = load_results()
    if not results:
        return

    grouped = normalize_model_names(results)
    plots = create_plots(grouped)
    report_json, report_txt = generate_report(grouped, plots)

    print("\n✅ GOTOVO")
    print("📊 Grafovi:", plots)
    print("📋 JSON:", report_json)
    print("📄 TXT:", report_txt)


if __name__ == "__main__":
    main()
