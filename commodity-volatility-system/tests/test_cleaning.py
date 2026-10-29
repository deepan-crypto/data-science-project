"""
Tests for data cleaning.
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing.cleaner import DataCleaner


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """DataFrame with duplicates, missing values, and an outlier."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-02"]
            ),
            "commodity": ["Gold"] * 5,
            "open": [1800, 1810, None, 1830, 1800],
            "high": [1820, 1825, 1835, 1840, 1820],
            "low": [1790, 1805, 1815, 1820, 1790],
            "close": [1810, 1815, 1825, 1835, 1810],
            "volume": [100000, None, 120000, 130000, 100000],
        }
    )


class TestDataCleaner:

    def test_removes_duplicates(self, raw_df):
        cleaner = DataCleaner()
        cleaned = cleaner.clean(raw_df)
        # The last row is a duplicate of the first
        assert len(cleaned) < len(raw_df)

    def test_fills_missing_prices(self, raw_df):
        cleaner = DataCleaner()
        cleaned = cleaner.clean(raw_df)
        assert cleaned["open"].isnull().sum() == 0

    def test_fills_missing_volume(self, raw_df):
        cleaner = DataCleaner()
        cleaned = cleaner.clean(raw_df)
        assert cleaned["volume"].isnull().sum() == 0

    def test_outlier_flag_column_exists(self, raw_df):
        cleaner = DataCleaner()
        cleaned = cleaner.clean(raw_df)
        assert "is_price_outlier" in cleaned.columns

    def test_outlier_flag_is_boolean(self, raw_df):
        cleaner = DataCleaner()
        cleaned = cleaner.clean(raw_df)
        assert cleaned["is_price_outlier"].dtype == bool
