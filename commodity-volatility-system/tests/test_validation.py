"""
Tests for data validation.
"""

import pandas as pd
import pytest

from preprocessing.validator import DataValidator


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_df() -> pd.DataFrame:
    """A small valid DataFrame."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "commodity": ["Gold", "Gold"],
            "open": [1800.0, 1810.0],
            "high": [1820.0, 1825.0],
            "low": [1790.0, 1805.0],
            "close": [1810.0, 1815.0],
            "volume": [100000, 110000],
        }
    )


@pytest.fixture
def df_with_missing():
    """DataFrame with some NaN values."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "commodity": ["Gold", "Gold"],
            "open": [1800.0, None],
            "high": [1820.0, 1825.0],
            "low": [1790.0, 1805.0],
            "close": [1810.0, 1815.0],
            "volume": [100000, 110000],
        }
    )


@pytest.fixture
def df_with_duplicates():
    """DataFrame containing exact duplicates."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "commodity": ["Gold", "Gold"],
            "open": [1800.0, 1800.0],
            "high": [1820.0, 1820.0],
            "low": [1790.0, 1790.0],
            "close": [1810.0, 1810.0],
            "volume": [100000, 100000],
        }
    )


@pytest.fixture
def df_invalid_ohlc():
    """DataFrame with an invalid OHLC row (high < low)."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "commodity": ["Gold"],
            "open": [1800.0],
            "high": [1780.0],  # high < low!
            "low": [1790.0],
            "close": [1810.0],
            "volume": [100000],
        }
    )


# ── Tests ────────────────────────────────────────────────────────────────────

class TestDataValidator:

    def test_valid_data_passes(self, valid_df):
        v = DataValidator()
        report = v.validate(valid_df)
        assert report["total_records"] == 2
        assert report["duplicate_records"] == 0
        assert report["invalid_ohlc_records"] == 0

    def test_missing_columns_detected(self):
        df = pd.DataFrame({"timestamp": [1], "commodity": ["Gold"]})
        v = DataValidator()
        report = v.validate(df)
        assert len(report["missing_columns"]) > 0

    def test_missing_values_detected(self, df_with_missing):
        v = DataValidator()
        report = v.validate(df_with_missing)
        assert report["missing_values"].get("open", 0) == 1

    def test_duplicates_detected(self, df_with_duplicates):
        v = DataValidator()
        report = v.validate(df_with_duplicates)
        assert report["duplicate_records"] == 1

    def test_invalid_ohlc_detected(self, df_invalid_ohlc):
        v = DataValidator()
        report = v.validate(df_invalid_ohlc)
        assert report["invalid_ohlc_records"] >= 1
