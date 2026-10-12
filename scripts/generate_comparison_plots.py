"""
Generate Comparison Visualizations between Empirical Implementation and Published Paper.
Creates:
1. model_accuracy_comparison.png: Accuracy leaderboard across all models.
2. paper_vs_empirical_metrics.png: Side-by-side grouped bar chart comparing F1, Precision, Recall, ROC AUC.
3. accuracy_vs_f1_tradeoff.png: Scatter visualization demonstrating the accuracy vs F1 imbalance phenomenon.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Published paper reference benchmarks (Table 6 & Table 7 in Spinde et al., 2021)
PAPER_BENCHMARKS = {
    "B1: Random Classifier": {"accuracy": 0.50, "f1": 0.17, "precision": 0.10, "recall": 0.52, "roc_auc": 0.50},
    "B2: Negative Sentiment Lexicon": {"accuracy": 0.86, "f1": 0.40, "precision": 0.35, "recall": 0.47, "roc_auc": 0.69},
    "B3: Sentiment Polarity Lexicon": {"accuracy": 0.76, "f1": 0.32, "precision": 0.22, "recall": 0.57, "roc_auc": 0.68},
    "B4: Semi-Automated Bias Lexicon": {"accuracy": 0.91, "f1": 0.20, "precision": 0.62, "recall": 0.12, "roc_auc": 0.56},
    "Logistic Regression (Weighted)": {"accuracy": 0.78, "f1": 0.38, "precision": 0.26, "recall": 0.67, "roc_auc": 0.82},
    "Linear Discriminant Analysis (LDA)": {"accuracy": 0.91, "f1": 0.41, "precision": 0.50, "recall": 0.35, "roc_auc": 0.82},
    "Quadratic Discriminant Analysis (QDA)": {"accuracy": 0.46, "f1": 0.19, "precision": 0.10, "recall": 0.99, "roc_auc": 0.76},
    "Complement Naive Bayes": {"accuracy": 0.73, "f1": 0.35, "precision": 0.23, "recall": 0.74, "roc_auc": 0.82},
    "k-Nearest Neighbors (KNN)": {"accuracy": 0.90, "f1": 0.21, "precision": 0.45, "recall": 0.14, "roc_auc": 0.70},
    "Decision Tree (Weighted)": {"accuracy": 0.80, "f1": 0.31, "precision": 0.30, "recall": 0.33, "roc_auc": 0.62},
    "Random Forest (Weighted)": {"accuracy": 0.91, "f1": 0.26, "precision": 0.71, "recall": 0.16, "roc_auc": 0.84},
    "Linear SVM (Weighted)": {"accuracy": 0.78, "f1": 0.38, "precision": 0.26, "recall": 0.70, "roc_auc": 0.83},
    "SVM (RBF Kernel, Weighted)": {"accuracy": 0.87, "f1": 0.35, "precision": 0.39, "recall": 0.31, "roc_auc": 0.78},
    "XGBoost (Default)": {"accuracy": 0.85, "f1": 0.42, "precision": 0.32, "recall": 0.64, "roc_auc": 0.84},
    "Multilayer Perceptron (MLP)": {"accuracy": 0.91, "f1": 0.34, "precision": 0.35, "recall": 0.33, "roc_auc": 0.63},
    "XGBoost (Fine-Tuned & Weighted)": {"accuracy": 0.84, "f1": 0.43, "precision": 0.29, "recall": 0.77, "roc_auc": 0.79},
}


def load_empirical_results(json_path: str = "outputs/evaluation/all_models_benchmark.json") -> list:
    if not os.path.exists(json_path):
        alt_path = os.path.join("d:/news-bias-word-detection", json_path)
        if os.path.exists(alt_path):
            json_path = alt_path
        else:
            raise FileNotFoundError(f"Cannot find {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_accuracy_comparison(results: list, output_dir: str):
    """Plot 1: Horizontal Bar Chart of Accuracy across all 16 models."""
    plt.figure(figsize=(12, 8), dpi=200)
    sns.set_theme(style="whitegrid")

    # Sort by accuracy
    sorted_res = sorted(results, key=lambda x: x["accuracy"])
    names = [x["model"] for x in sorted_res]
    accs = [x["accuracy"] * 100 for x in sorted_res]

    colors = []
    for n in names:
        if "XGBoost" in n:
            colors.append("#2E7D32")  # Rich green for XGBoost
        elif "Random Forest" in n or "LDA" in n or "MLP" in n:
            colors.append("#1976D2")  # Blue for strong ML
        elif n.startswith("B"):
            colors.append("#F57C00")  # Orange for Baselines
        else:
            colors.append("#78909C")  # Slate grey

    bars = plt.barh(names, accs, color=colors, height=0.65, edgecolor="black", alpha=0.9)
    plt.xlim(0, 105)
    plt.xlabel("Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Model Accuracy Comparison (Evaluated on MBIC Held-Out Test Set)", fontsize=14, fontweight="bold", pad=15)

    for bar, acc in zip(bars, accs):
        plt.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.2f}%",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.tight_layout()
    save_path = os.path.join(output_dir, "model_accuracy_comparison.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_side_by_side_comparison(results: list, output_dir: str):
    """Plot 2: Empirical vs Published Paper side-by-side comparison for key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)
    sns.set_theme(style="whitegrid")

    # Pick representative key models
    key_models = [
        "B1: Random Classifier",
        "B2: Negative Sentiment Lexicon",
        "B3: Sentiment Polarity Lexicon",
        "B4: Semi-Automated Bias Lexicon",
        "Logistic Regression (Weighted)",
        "Complement Naive Bayes",
        "Random Forest (Weighted)",
        "XGBoost (Default)",
        "XGBoost (Fine-Tuned & Weighted)",
    ]

    short_labels = [
        "B1 Random", "B2 Neg Lex", "B3 Polarity", "B4 Bias Lex",
        "Weighted LogReg", "Comp Naive Bayes", "Random Forest",
        "XGBoost Default", "XGBoost Tuned (Champ)",
    ]

    emp_dict = {r["model"]: r for r in results}

    metrics_to_plot = [
        ("f1", "F1 Score Comparison (Empirical vs. Paper)", axes[0, 0]),
        ("roc_auc", "ROC AUC Comparison (Empirical vs. Paper)", axes[0, 1]),
        ("precision", "Precision Comparison (Empirical vs. Paper)", axes[1, 0]),
        ("recall", "Recall Comparison (Empirical vs. Paper)", axes[1, 1]),
    ]

    x = np.arange(len(key_models))
    width = 0.38

    for m_key, title, ax in metrics_to_plot:
        emp_vals = [emp_dict.get(m, {}).get(m_key, 0) for m in key_models]
        paper_vals = [PAPER_BENCHMARKS.get(m, {}).get(m_key, 0) for m in key_models]

        r1 = ax.bar(x - width / 2, emp_vals, width, label="Our Implementation (Empirical)", color="#1976D2", edgecolor="black")
        r2 = ax.bar(x + width / 2, paper_vals, width, label="Spinde et al. Paper Reported", color="#E65100", edgecolor="black")

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="upper right")

        # Value annotations
        for bar in r1:
            h = bar.get_height()
            if h > 0.05:
                ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        for bar in r2:
            h = bar.get_height()
            if h > 0.05:
                ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", fontsize=7.5)

    plt.suptitle("Benchmark Replication: Implementation vs. Spinde et al. (2021) Published Scores", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "paper_vs_empirical_comparison.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_accuracy_vs_f1_tradeoff(results: list, output_dir: str):
    """Plot 3: Accuracy vs F1 Score Scatter Plot illustrating the Imbalance Paradox."""
    plt.figure(figsize=(10, 7), dpi=200)
    sns.set_theme(style="whitegrid")

    accs = [r["accuracy"] * 100 for r in results]
    f1s = [r["f1"] for r in results]
    names = [r["model"] for r in results]

    # Bubble size based on recall
    recalls = [r["recall"] * 500 + 100 for r in results]

    scatter = plt.scatter(accs, f1s, s=recalls, c=f1s, cmap="viridis", alpha=0.85, edgecolors="black", linewidth=1.5)
    cbar = plt.colorbar(scatter)
    cbar.set_label("F1 Score", fontsize=11, fontweight="bold")

    plt.axvline(90.3, color="red", linestyle="--", alpha=0.7, label="Majority Class Baseline (90.3% Non-Biased)")
    plt.xlabel("Accuracy (%)", fontsize=12, fontweight="bold")
    plt.ylabel("F1 Score", fontsize=12, fontweight="bold")
    plt.title("Accuracy vs. F1-Score Tradeoff (Bubble Size = Recall)", fontsize=13, fontweight="bold", pad=12)

    # Annotate specific notable models
    for n, x, y in zip(names, accs, f1s):
        short = n.replace(" (Weighted)", "").replace(" Classifier", "").replace(" Lexicon", "")
        if "XGBoost" in n or "Random Forest" in n or "Random" in n or "B4" in n or "QDA" in n:
            plt.annotate(
                short,
                (x, y),
                xytext=(8, 4),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.3),
            )

    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "accuracy_vs_f1_tradeoff.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    out_dir = "outputs/evaluation"
    os.makedirs(out_dir, exist_ok=True)
    results = load_empirical_results()

    plot_accuracy_comparison(results, out_dir)
    plot_side_by_side_comparison(results, out_dir)
    plot_accuracy_vs_f1_tradeoff(results, out_dir)
    print("All comparison plots generated successfully!")


if __name__ == "__main__":
    main()
