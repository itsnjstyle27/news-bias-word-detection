import pytest
import numpy as np
import pandas as pd
from src.models.baselines import (
    RandomBaselineClassifier,
    NegativeSentimentBaselineClassifier,
    SentimentPolarityBaselineClassifier,
    SemiAutomatedBiasLexiconBaselineClassifier,
)
from src.evaluation.metrics import evaluate_predictions


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "negative_conc": [1, 0, 1, 0, 0],
        "positive_conc": [0, 1, 0, 0, 0],
        "bias_lexicon":  [1, 0, 0, 0, 1],
        "label4":        [1, 0, 1, 0, 1],
    })


def test_random_baseline(sample_df):
    clf1 = RandomBaselineClassifier(random_state=42)
    clf2 = RandomBaselineClassifier(random_state=42)
    preds1 = clf1.predict(sample_df)
    preds2 = clf2.predict(sample_df)
    assert len(preds1) == len(sample_df)
    assert np.array_equal(preds1, preds2), "Random baseline must be deterministic with same seed"
    assert set(np.unique(preds1)).issubset({0, 1})


def test_negative_sentiment_baseline(sample_df):
    clf = NegativeSentimentBaselineClassifier()
    preds = clf.predict(sample_df)
    expected = sample_df["negative_conc"].values
    assert np.array_equal(preds, expected)


def test_sentiment_polarity_baseline(sample_df):
    clf = SentimentPolarityBaselineClassifier()
    preds = clf.predict(sample_df)
    expected = ((sample_df["negative_conc"] == 1) | (sample_df["positive_conc"] == 1)).astype(int).values
    assert np.array_equal(preds, expected)


def test_bias_lexicon_baseline(sample_df):
    clf = SemiAutomatedBiasLexiconBaselineClassifier()
    preds = clf.predict(sample_df)
    expected = sample_df["bias_lexicon"].values
    assert np.array_equal(preds, expected)


def test_evaluation_metrics():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    results = evaluate_predictions(y_true, y_pred, model_name="TestModel")
    
    assert 0.0 <= results["f1"] <= 1.0
    assert 0.0 <= results["precision"] <= 1.0
    assert 0.0 <= results["recall"] <= 1.0
    assert 0.0 <= results["roc_auc"] <= 1.0
    assert results["tp"] + results["fp"] + results["tn"] + results["fn"] == len(y_true)
