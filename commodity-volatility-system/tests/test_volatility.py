"""
Tests for volatility engine.
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing.feature_engineering import FeatureEngineer
from analytics.volatility import VolatilityEngine


@pytest.fixture
def featured_df() -> pd.DataFrame:
    """DataFrame with features already computed."""
    dates = pd.bdate_range("2024-01-02", periods=40)
    np.random.seed(42)
    close = 1800 + np.cumsum(np.random.normal(0, 5, 40))
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "commodity": ["Gold"] * 40,
            "open": close - 2,
            "high": close + 5,
            "low": close - 5,
            "close": close,
            "volume": np.random.randint(80000, 120000, 40),
        }
    )
    eng = FeatureEngineer(rolling_windows=[7, 20])
    return eng.generate_features(df)


class TestVolatilityEngine:

    def test_volatility_columns_added(self, featured_df):
        engine = VolatilityEngine(rolling_windows=[7, 20, 30])
        df = engine.calculate(featured_df)
        assert "volatility_7" in df.columns
        assert "volatility_7_ann" in df.columns
        assert "volatility_20" in df.columns
        assert "volatility_30" in df.columns

    def test_classification_column(self, featured_df):
        engine = VolatilityEngine(rolling_windows=[7])
        df = engine.calculate(featured_df)
        assert "volatility_class" in df.columns
        valid_classes = {"LOW", "MODERATE", "HIGH", "EXTREME", "N/A"}
        assert set(df["volatility_class"].unique()).issubset(valid_classes)

    def test_spike_detection_column(self, featured_df):
        engine = VolatilityEngine(rolling_windows=[7])
        df = engine.calculate(featured_df)
        assert "is_volatility_spike" in df.columns
        assert df["is_volatility_spike"].dtype == bool

    def test_annualized_is_larger(self, featured_df):
        engine = VolatilityEngine(rolling_windows=[7])
        df = engine.calculate(featured_df)
        # Annualised should be larger than raw (multiplied by sqrt(252))
        valid = df.dropna(subset=["volatility_7", "volatility_7_ann"])
        assert (valid["volatility_7_ann"] >= valid["volatility_7"]).all()
