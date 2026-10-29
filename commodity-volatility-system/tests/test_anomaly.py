"""
Tests for anomaly detection.
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing.feature_engineering import FeatureEngineer
from analytics.anomaly_detection import AnomalyDetector


@pytest.fixture
def featured_df() -> pd.DataFrame:
    """DataFrame with features computed for anomaly detection."""
    dates = pd.bdate_range("2024-01-02", periods=60)
    np.random.seed(42)
    close = 1800 + np.cumsum(np.random.normal(0, 5, 60))
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "commodity": ["Gold"] * 60,
            "open": close - 2,
            "high": close + 5,
            "low": close - 5,
            "close": close,
            "volume": np.random.randint(80000, 120000, 60),
        }
    )
    eng = FeatureEngineer(rolling_windows=[7, 20])
    return eng.generate_features(df)


class TestAnomalyDetector:

    def test_output_columns_added(self, featured_df):
        detector = AnomalyDetector(contamination=0.1)
        df = detector.detect(featured_df)
        assert "anomaly_score" in df.columns
        assert "is_anomaly" in df.columns

    def test_anomaly_flag_is_boolean(self, featured_df):
        detector = AnomalyDetector(contamination=0.1)
        df = detector.detect(featured_df)
        assert df["is_anomaly"].dtype == bool

    def test_at_least_one_anomaly_detected(self, featured_df):
        """With 10% contamination on 60 rows we expect ≥1 anomaly."""
        detector = AnomalyDetector(contamination=0.1)
        df = detector.detect(featured_df)
        assert df["is_anomaly"].sum() >= 1

    def test_missing_features_handled_gracefully(self):
        """Detector should handle a DataFrame that lacks features."""
        df = pd.DataFrame({"foo": [1, 2, 3]})
        detector = AnomalyDetector()
        result = detector.detect(df)
        assert "is_anomaly" in result.columns
        assert result["is_anomaly"].sum() == 0
