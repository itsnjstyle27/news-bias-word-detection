from typing import Union, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseBaseline(BaseEstimator, ClassifierMixin):
    """Abstract base class for heuristic and lexicon-based bias detection baselines."""

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None):
        """Fit method (mostly a no-op for rule-based models)."""
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        preds = self.predict(X)
        probs = np.zeros((len(preds), 2))
        probs[:, 1] = preds
        probs[:, 0] = 1 - preds
        return probs


class RandomBaselineClassifier(BaseBaseline):
    """
    Baseline 1 (B1): Purely Random Classifier.
    Predicts whether a word is bias-inducing with uniform random guessing (p=0.5).
    Replicates the B1 baseline from Table 7 of Spinde et al. (2021).
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        n_samples = len(X)
        return rng.randint(0, 2, size=n_samples)


class NegativeSentimentBaselineClassifier(BaseBaseline):
    """
    Baseline 2 (B2): Negative Sentiment Lexicon Classifier.
    Flags words as biased if they appear in the negative sentiment lexicon.
    Replicates the B2 baseline from Table 7 of Spinde et al. (2021).
    """

    def __init__(self, feature_col: str = "negative_conc", fallback_lexicon: Optional[set] = None):
        self.feature_col = feature_col
        self.fallback_lexicon = fallback_lexicon or set()

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and self.feature_col in X.columns:
            preds = (X[self.feature_col] == 1).astype(int).values
            return preds
        elif isinstance(X, pd.DataFrame) and "text_low" in X.columns:
            preds = X["text_low"].apply(lambda w: 1 if w in self.fallback_lexicon else 0).values
            return preds
        else:
            raise ValueError(f"Feature column '{self.feature_col}' not found in input DataFrame.")


class SentimentPolarityBaselineClassifier(BaseBaseline):
    """
    Baseline 3 (B3): Positive + Negative Polarity Lexicon Classifier.
    Flags words as biased if they appear in either the negative OR positive sentiment lexicon.
    Replicates the B3 baseline from Table 7 of Spinde et al. (2021).
    """

    def __init__(
        self,
        neg_col: str = "negative_conc",
        pos_col: str = "positive_conc",
        fallback_pos_lexicon: Optional[set] = None,
        fallback_neg_lexicon: Optional[set] = None,
    ):
        self.neg_col = neg_col
        self.pos_col = pos_col
        self.fallback_pos_lexicon = fallback_pos_lexicon or set()
        self.fallback_neg_lexicon = fallback_neg_lexicon or set()

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and self.neg_col in X.columns and self.pos_col in X.columns:
            preds = ((X[self.neg_col] == 1) | (X[self.pos_col] == 1)).astype(int).values
            return preds
        elif isinstance(X, pd.DataFrame) and "text_low" in X.columns:
            combined = self.fallback_pos_lexicon.union(self.fallback_neg_lexicon)
            preds = X["text_low"].apply(lambda w: 1 if w in combined else 0).values
            return preds
        else:
            raise ValueError(f"Columns '{self.neg_col}' or '{self.pos_col}' not found in input DataFrame.")


class SemiAutomatedBiasLexiconBaselineClassifier(BaseBaseline):
    """
    Baseline 4 (B4): Semi-Automated Bias Lexicon Classifier.
    Flags words as biased if they occur in the semi-automated domain-specific bias lexicon
    extracted from HuffPost and Breitbart Word2Vec embeddings (Section 3.3).
    Replicates the B4 baseline from Table 7 of Spinde et al. (2021).
    """

    def __init__(self, feature_col: str = "bias_lexicon", fallback_bias_words: Optional[set] = None):
        self.feature_col = feature_col
        self.fallback_bias_words = fallback_bias_words or set()

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and self.feature_col in X.columns:
            preds = (X[self.feature_col] == 1).astype(int).values
            return preds
        elif isinstance(X, pd.DataFrame) and "text_low" in X.columns:
            preds = X["text_low"].apply(lambda w: 1 if w in self.fallback_bias_words else 0).values
            return preds
        else:
            raise ValueError(f"Feature column '{self.feature_col}' not found in input DataFrame.")


def get_paper_baselines(random_state: int = 42) -> Dict[str, BaseBaseline]:
    """Factory creating all four baseline models defined in Spinde et al. (2021)."""
    return {
        "B1: Random Classifier": RandomBaselineClassifier(random_state=random_state),
        "B2: Negative Sentiment Lexicon": NegativeSentimentBaselineClassifier(),
        "B3: Sentiment Polarity Lexicon": SentimentPolarityBaselineClassifier(),
        "B4: Semi-Automated Bias Lexicon": SemiAutomatedBiasLexiconBaselineClassifier(),
    }
