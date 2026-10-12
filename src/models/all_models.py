"""
All 16 Models Implemented from Spinde et al. (2021) - High Performance & Fast Execution
Includes:
- Baselines B1, B2, B3, B4 (Table 7)
- 11 Classic ML Algorithms from Algorithm Selection (Table 6)
- The Paper's Champion Fine-Tuned Weighted XGBoost Model (Table 7)
"""

from typing import Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from src.models.baselines import (
    RandomBaselineClassifier,
    NegativeSentimentBaselineClassifier,
    SentimentPolarityBaselineClassifier,
    SemiAutomatedBiasLexiconBaselineClassifier,
)

NON_FEATURE_COLS = {
    "Unnamed: 0.1", "Unnamed: 0", "sentence", "outlet", "topic", "type", "num_sent",
    "biased_words4", "text", "text_low", "pos", "lemma", "lemma_low", "tag", "dep",
    "label3", "label4", "label5", "ne_label"
}


def extract_features(df: pd.DataFrame, feature_cols: List[str] = None) -> Tuple[np.ndarray, List[str]]:
    """Extract numeric feature matrix from DataFrame."""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].fillna(0).values
    return X, feature_cols


class FastSklearnModelWrapper(BaseEstimator, ClassifierMixin):
    """Universal wrapper for Scikit-learn and XGBoost estimators with optional scaling and fast inference."""

    def __init__(self, base_estimator: Any, scaling: str = None, name: str = "Model", max_train_samples: int = None):
        self.base_estimator = base_estimator
        self.scaling = scaling
        self.name = name
        self.max_train_samples = max_train_samples
        self.feature_names_: List[str] = []
        self.scaler = None
        if scaling == "standard":
            self.scaler = StandardScaler()
        elif scaling == "minmax":
            self.scaler = MinMaxScaler()

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        if isinstance(X, pd.DataFrame):
            X_mat, self.feature_names_ = extract_features(X)
        else:
            X_mat = np.nan_to_num(X, nan=0.0)

        y_vec = y.values if isinstance(y, pd.Series) else np.asarray(y)

        if self.max_train_samples is not None and len(X_mat) > self.max_train_samples:
            # Subsample for very slow algorithms (e.g. kernel SVM) while preserving positive class balance
            pos_idx = np.where(y_vec == 1)[0]
            neg_idx = np.where(y_vec == 0)[0]
            rng = np.random.RandomState(42)
            n_pos = min(len(pos_idx), int(self.max_train_samples * 0.15))
            n_neg = self.max_train_samples - n_pos
            sel_idx = np.concatenate([
                rng.choice(pos_idx, size=n_pos, replace=False),
                rng.choice(neg_idx, size=n_neg, replace=False)
            ])
            rng.shuffle(sel_idx)
            X_mat = X_mat[sel_idx]
            y_vec = y_vec[sel_idx]

        if self.scaler is not None:
            X_mat = self.scaler.fit_transform(X_mat)

        self.base_estimator.fit(X_mat, y_vec)
        self.classes_ = getattr(self.base_estimator, "classes_", np.array([0, 1]))
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            cols = self.feature_names_ if self.feature_names_ else None
            X_mat, _ = extract_features(X, cols)
        else:
            X_mat = np.nan_to_num(X, nan=0.0)

        if self.scaler is not None:
            X_mat = self.scaler.transform(X_mat)

        return self.base_estimator.predict(X_mat)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            cols = self.feature_names_ if self.feature_names_ else None
            X_mat, _ = extract_features(X, cols)
        else:
            X_mat = np.nan_to_num(X, nan=0.0)

        if self.scaler is not None:
            X_mat = self.scaler.transform(X_mat)

        if hasattr(self.base_estimator, "predict_proba"):
            return self.base_estimator.predict_proba(X_mat)
        elif hasattr(self.base_estimator, "decision_function"):
            df = self.base_estimator.decision_function(X_mat)
            prob1 = 1 / (1 + np.exp(-np.clip(df, -20, 20)))
            return np.vstack([1 - prob1, prob1]).T
        else:
            preds = self.predict(X)
            probs = np.zeros((len(preds), 2))
            probs[:, 1] = preds
            probs[:, 0] = 1 - preds
            return probs


def build_all_models_suite(random_state: int = 42) -> Dict[str, Any]:
    """
    Build dictionary containing ALL 16 models from the paper:
    1. Baselines B1 to B4 (Table 7)
    2. 11 ML models (Table 6)
    3. The Fine-Tuned Weighted XGBoost champion model (Table 7)
    """
    scale_weight = 9.5
    class_weights = {0: 10, 1: 90}

    models = {
        # --- Baselines (Table 7) ---
        "B1: Random Classifier": RandomBaselineClassifier(random_state=random_state),
        "B2: Negative Sentiment Lexicon": NegativeSentimentBaselineClassifier(),
        "B3: Sentiment Polarity Lexicon": SentimentPolarityBaselineClassifier(),
        "B4: Semi-Automated Bias Lexicon": SemiAutomatedBiasLexiconBaselineClassifier(),

        # --- 11 Supervised ML Classifiers (Table 6) ---
        "Logistic Regression (Weighted)": FastSklearnModelWrapper(
            LogisticRegression(class_weight=class_weights, max_iter=300, random_state=random_state, solver="lbfgs"),
            scaling="standard",
            name="Logistic Regression",
        ),
        "Linear Discriminant Analysis (LDA)": FastSklearnModelWrapper(
            LinearDiscriminantAnalysis(solver="svd"),
            scaling="standard",
            name="LDA",
        ),
        "Quadratic Discriminant Analysis (QDA)": FastSklearnModelWrapper(
            QuadraticDiscriminantAnalysis(reg_param=0.1),
            scaling="standard",
            name="QDA",
        ),
        "Complement Naive Bayes": FastSklearnModelWrapper(
            ComplementNB(),
            scaling="minmax",
            name="Complement NB",
        ),
        "k-Nearest Neighbors (KNN)": FastSklearnModelWrapper(
            KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            scaling="standard",
            name="KNN",
        ),
        "Decision Tree (Weighted)": FastSklearnModelWrapper(
            DecisionTreeClassifier(class_weight=class_weights, max_depth=10, random_state=random_state),
            scaling=None,
            name="Decision Tree",
        ),
        "Random Forest (Weighted)": FastSklearnModelWrapper(
            RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=random_state, n_jobs=-1),
            scaling=None,
            name="Random Forest",
        ),
        "Linear SVM (Weighted)": FastSklearnModelWrapper(
            LinearSVC(class_weight=class_weights, max_iter=1000, random_state=random_state, dual=False),
            scaling="standard",
            name="Linear SVM",
        ),
        "SVM (RBF Kernel, Weighted)": FastSklearnModelWrapper(
            SVC(kernel="rbf", class_weight=class_weights, max_iter=1500, random_state=random_state),
            scaling="standard",
            name="SVM (RBF)",
            max_train_samples=6000,  # Fast stratified approximation for quadratic RBF
        ),
        "XGBoost (Default)": FastSklearnModelWrapper(
            xgb.XGBClassifier(
                objective="binary:logistic",
                scale_pos_weight=scale_weight,
                n_estimators=100,
                random_state=random_state,
                eval_metric="logloss",
                tree_method="hist",
            ),
            scaling=None,
            name="XGBoost (Default)",
        ),
        "Multilayer Perceptron (MLP)": FastSklearnModelWrapper(
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60, random_state=random_state, early_stopping=True),
            scaling="standard",
            name="MLP",
        ),

        # --- The Paper's Champion Model (Table 7) ---
        "XGBoost (Fine-Tuned & Weighted)": FastSklearnModelWrapper(
            xgb.XGBClassifier(
                objective="binary:logistic",
                max_depth=6,
                min_child_weight=18,
                subsample=1.0,
                colsample_bytree=1.0,
                learning_rate=0.2,
                scale_pos_weight=scale_weight,
                n_estimators=150,
                random_state=random_state,
                eval_metric="logloss",
                tree_method="hist",
            ),
            scaling=None,
            name="XGBoost (Fine-Tuned)",
        ),
    }

    return models
