import pytest
import os
import pandas as pd
from src.data.dataset_loader import MBICDataLoader


@pytest.fixture
def data_loader():
    return MBICDataLoader()


def test_load_raw_sentences(data_loader):
    df = data_loader.load_raw_sentences()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1000, "Corpus should have at least 1,000 sentences"
    for expected_col in ["sentence", "outlet", "topic"]:
        assert expected_col in df.columns, f"Missing required column {expected_col}"


def test_load_token_dataset(data_loader):
    df = data_loader.load_token_dataset(drop_missing_tfidf=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 20000, "Token dataset should have > 20,000 words"
    assert "label4" in df.columns, "Target label4 must be present"
    assert "negative_conc" in df.columns, "negative_conc must be present for B2"
    assert "positive_conc" in df.columns, "positive_conc must be present for B3"
    assert "bias_lexicon" in df.columns, "bias_lexicon must be present for B4"


def test_train_test_split_consistency(data_loader):
    splits = data_loader.get_train_test_split(test_size=0.10, val_size=0.10, random_state=42)
    assert "X_train" in splits
    assert "X_test" in splits
    assert "y_test" in splits
    assert len(splits["X_test"]) == len(splits["y_test"])
    
    # Verify split size is approx 10%
    total = splits["total_samples"]
    test_len = len(splits["X_test"])
    assert abs(test_len / total - 0.10) < 0.02, "Test split ratio should be ~10%"
