"""
Tests for feature engineering.
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing.feature_engineering import FeatureEngineer


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A small cleaned DataFrame for Gold."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    prices = [1800 + i * 5 for i in range(10)]
    return pd.DataFrame(
        {
            "timestamp": dates,
            "commodity": ["Gold"] * 10,
            "open": [p - 2 for p in prices],
            "high": [p + 5 for p in prices],
            "low": [p - 5 for p in prices],
            "close": prices,
            "volume": [100000 + i * 1000 for i in range(10)],
        }
    )


class TestFeatureEngineer:

    def test_return_columns_added(self, clean_df):
        eng = FeatureEngineer(rolling_windows=[7])
        df = eng.generate_features(clean_df)
        assert "daily_return" in df.columns
        assert "log_return" in df.columns
        assert "price_change" in df.columns

    def test_return_first_row_is_nan(self, clean_df):
        eng = FeatureEngineer(rolling_windows=[7])
        df = eng.generate_features(clean_df)
        assert pd.isna(df["daily_return"].iloc[0])

    def test_rolling_columns_added(self, clean_df):
        eng = FeatureEngineer(rolling_windows=[7])
        df = eng.generate_features(clean_df)
        assert "rolling_mean_7" in df.columns
        assert "rolling_std_7" in df.columns

    def test_volume_change_added(self, clean_df):
        eng = FeatureEngineer(rolling_windows=[7])
        df = eng.generate_features(clean_df)
        assert "volume_change" in df.columns

    def test_return_calculation_correctness(self, clean_df):
        eng = FeatureEngineer(rolling_windows=[7])
        df = eng.generate_features(clean_df)
        # Second row: return = (1805 - 1800) / 1800
        expected = (1805 - 1800) / 1800
        actual = df["daily_return"].iloc[1]
        assert abs(actual - expected) < 1e-8
