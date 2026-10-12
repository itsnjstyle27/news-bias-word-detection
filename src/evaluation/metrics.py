from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn import metrics
from src.utils.helpers import format_metrics_table


# Benchmark figures published in Spinde et al. (2021) Table 7 (Baselines) and Table 6 (ML)
PAPER_REPORTED_BASELINES = {
    "B1: Random Classifier": {"roc_auc": 0.50, "f1": 0.17, "precision": 0.10, "recall": 0.52},
    "B2: Negative Sentiment Lexicon": {"roc_auc": 0.69, "f1": 0.40, "precision": 0.35, "recall": 0.47},
    "B3: Sentiment Polarity Lexicon": {"roc_auc": 0.68, "f1": 0.32, "precision": 0.22, "recall": 0.57},
    "B4: Semi-Automated Bias Lexicon": {"roc_auc": 0.56, "f1": 0.20, "precision": 0.62, "recall": 0.12},
    "Weighted Logistic Regression": {"roc_auc": 0.82, "f1": 0.38, "precision": 0.26, "recall": 0.67},
    "Complement Naive Bayes": {"roc_auc": 0.82, "f1": 0.35, "precision": 0.23, "recall": 0.74},
    "Random Forest (Weighted)": {"roc_auc": 0.84, "f1": 0.26, "precision": 0.71, "recall": 0.16},
}


def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_prob: Optional[Union[pd.Series, np.ndarray]] = None,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics for binary classification:
    F1-score, Precision, Recall, ROC AUC, Accuracy, Confusion Matrix, and Rates.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    f1 = float(metrics.f1_score(y_true_arr, y_pred_arr, zero_division=0))
    precision = float(metrics.precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(metrics.recall_score(y_true_arr, y_pred_arr, zero_division=0))
    accuracy = float(metrics.accuracy_score(y_true_arr, y_pred_arr))

    # ROC AUC: use predicted probabilities if available, else binary predictions
    score_for_auc = y_prob if y_prob is not None else y_pred_arr
    try:
        roc_auc = float(metrics.roc_auc_score(y_true_arr, score_for_auc))
    except Exception:
        roc_auc = 0.5

    cm = metrics.confusion_matrix(y_true_arr, y_pred_arr)
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "model": model_name,
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "roc_auc": round(roc_auc, 4),
        "accuracy": round(accuracy, 4),
        "specificity": round(specificity, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total_test_samples": len(y_true_arr),
    }


def evaluate_model(model: Any, X_test: Any, y_test: Any, model_name: str = "Model") -> Dict[str, Any]:
    """Predict and evaluate a fitted model instance."""
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)
            if probs.ndim == 2 and probs.shape[1] == 2:
                y_prob = probs[:, 1]
        except Exception:
            pass
    return evaluate_predictions(y_test, y_pred, y_prob=y_prob, model_name=model_name)


def build_benchmark_comparison_table(results: list) -> str:
    """Generate Markdown comparison table showing empirical vs paper reported baselines."""
    headers = ["Model", "Empirical F1", "Paper F1", "Empirical P", "Paper P", "Empirical R", "Paper R", "Empirical AUC", "Paper AUC"]
    rows = []

    for res in results:
        m_name = res["model"]
        paper_ref = PAPER_REPORTED_BASELINES.get(m_name, {})
        
        paper_f1 = f"{paper_ref.get('f1', '-'):.2f}" if "f1" in paper_ref else "-"
        paper_p = f"{paper_ref.get('precision', '-'):.2f}" if "precision" in paper_ref else "-"
        paper_r = f"{paper_ref.get('recall', '-'):.2f}" if "recall" in paper_ref else "-"
        paper_auc = f"{paper_ref.get('roc_auc', '-'):.2f}" if "roc_auc" in paper_ref else "-"

        rows.append([
            m_name,
            f"{res['f1']:.2f}",
            paper_f1,
            f"{res['precision']:.2f}",
            paper_p,
            f"{res['recall']:.2f}",
            paper_r,
            f"{res['roc_auc']:.2f}",
            paper_auc,
        ])

    return format_metrics_table(headers, rows)
