"""
Tests for data ingestion — CSV loading.
"""

import os
import tempfile

import pandas as pd
import pytest

from ingestion.csv_loader import CSVDataLoader


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv(tmp_path) -> str:
    """Write a minimal valid CSV and return its path."""
    csv_path = tmp_path / "test_data.csv"
    csv_path.write_text(
        "timestamp,commodity,open,high,low,close,volume\n"
        "2024-01-02,Gold,1800,1820,1790,1810,100000\n"
        "2024-01-03,Gold,1810,1825,1805,1815,110000\n"
    )
    return str(csv_path)


@pytest.fixture
def missing_col_csv(tmp_path) -> str:
    """CSV that is missing the 'volume' column."""
    csv_path = tmp_path / "missing_col.csv"
    csv_path.write_text(
        "timestamp,commodity,open,high,low,close\n"
        "2024-01-02,Gold,1800,1820,1790,1810\n"
    )
    return str(csv_path)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestCSVDataLoader:
    """Tests for CSVDataLoader."""

    def test_load_valid_csv(self, sample_csv):
        """Loading a valid CSV should return a DataFrame with correct shape."""
        loader = CSVDataLoader(filepath=sample_csv)
        df = loader.load_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "timestamp" in df.columns

    def test_timestamp_conversion(self, sample_csv):
        """Timestamps should be converted to datetime64."""
        loader = CSVDataLoader(filepath=sample_csv)
        df = loader.load_data()
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_missing_file_raises(self):
        """Should raise FileNotFoundError for a non-existent path."""
        loader = CSVDataLoader(filepath="/nonexistent/file.csv")
        with pytest.raises(FileNotFoundError):
            loader.load_data()

    def test_missing_columns_raises(self, missing_col_csv):
        """Should raise ValueError when required columns are absent."""
        loader = CSVDataLoader(filepath=missing_col_csv)
        with pytest.raises(ValueError, match="missing required columns"):
            loader.load_data()

    def test_custom_required_columns(self, sample_csv):
        """Should accept a custom set of required columns."""
        loader = CSVDataLoader(
            filepath=sample_csv,
            required_columns=["timestamp", "commodity", "close"],
        )
        df = loader.load_data()
        assert len(df) == 2
