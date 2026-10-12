#!/usr/bin/env python
"""
Run Benchmark on ALL 16 Models from Spinde et al. (2021).
Generates the complete Accuracy & Metrics Leaderboard across:
- 4 Baselines (B1-B4)
- 11 Classic ML Algorithms (Table 6)
- 1 Champion Fine-Tuned Weighted XGBoost Model (Table 7)
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import setup_logger, format_metrics_table
from src.data.dataset_loader import MBICDataLoader
from src.models.all_models import build_all_models_suite
from src.evaluation.metrics import evaluate_model, PAPER_REPORTED_BASELINES

logger = setup_logger("all_models_runner")


def main():
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE BENCHMARK: EVALUATING ALL 16 MODELS FROM SPINDE ET AL. (2021)")
    logger.info("=" * 80)

    loader = MBICDataLoader()
    splits = loader.get_train_test_split(test_size=0.10, val_size=0.10, random_state=42)

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    logger.info(f"Loaded {len(X_train)} Train tokens, {len(X_test)} Test tokens.")
    logger.info(f"Test Class Balance: {int((y_test==1).sum())} Biased ({(y_test==1).sum()/len(y_test):.1%}), {int((y_test==0).sum())} Non-Biased\n")

    models_suite = build_all_models_suite(random_state=42)
    results = []

    for idx, (name, model) in enumerate(models_suite.items(), 1):
        t0 = time.time()
        logger.info(f"[{idx:>2}/{len(models_suite)}] Fitting & Evaluating: {name}...")
        try:
            model.fit(X_train, y_train)
            metrics_dict = evaluate_model(model, X_test, y_test, model_name=name)
            elapsed = time.time() - t0
            metrics_dict["elapsed_sec"] = round(elapsed, 2)
            results.append(metrics_dict)
            logger.info(
                f"       Accuracy: {metrics_dict['accuracy']:.4f} | F1: {metrics_dict['f1']:.4f} | "
                f"P: {metrics_dict['precision']:.4f} | R: {metrics_dict['recall']:.4f} | "
                f"AUC: {metrics_dict['roc_auc']:.4f} ({elapsed:.1f}s)"
            )
        except Exception as e:
            logger.error(f"Failed to evaluate {name}: {e}")

    # Build Comprehensive Accuracy & Metrics Table
    headers = [
        "Rank", "Model", "Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC", "TP", "FP", "Time (s)"
    ]
    # Sort by F1 score descending
    sorted_results = sorted(results, key=lambda x: x["f1"], reverse=True)

    rows = []
    for rank, res in enumerate(sorted_results, 1):
        rows.append([
            str(rank),
            res["model"],
            f"{res['accuracy']:.2%}",
            f"{res['f1']:.4f}",
            f"{res['precision']:.4f}",
            f"{res['recall']:.4f}",
            f"{res['roc_auc']:.4f}",
            str(res["tp"]),
            str(res["fp"]),
            f"{res.get('elapsed_sec', 0):.1f}",
        ])

    table_md = format_metrics_table(headers, rows)

    print("\n" + "=" * 100)
    print("ALL MODELS ACCURACY & PERFORMANCE LEADERBOARD (Test Set: 3,114 tokens)")
    print("=" * 100)
    print(table_md)
    print("=" * 100)

    # Save to disk
    out_dir = "outputs/evaluation"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "all_models_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(out_dir, "all_models_leaderboard.md"), "w", encoding="utf-8") as f:
        f.write("# Comprehensive Benchmark: All 16 Models (Spinde et al., 2021)\n\n")
        f.write(table_md)
        f.write("\n\n*Evaluated on held-out 10% test split with 31,136 total tokens across 1,700 news statements.*\n")

    logger.info("Comprehensive benchmark complete! Results saved to outputs/evaluation/")


if __name__ == "__main__":
    main()
