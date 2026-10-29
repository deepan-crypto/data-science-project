"""
CSV Data Loader
===============
Concrete ``DataLoader`` that reads commodity price data from a CSV file.
"""

import logging
import os

import pandas as pd

from .base_loader import DataLoader

logger = logging.getLogger(__name__)


class CSVDataLoader(DataLoader):
    """Load commodity data from a local CSV file.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.
    required_columns : list[str] | None
        Column names that must be present. Defaults to the standard set.
    timestamp_format : str | None
        ``strftime`` format for parsing the ``timestamp`` column.
        When *None*, Pandas infers the format automatically.
    """

    # Standard columns every raw dataset must have
    DEFAULT_REQUIRED_COLUMNS = [
        "timestamp",
        "commodity",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    def __init__(
        self,
        filepath: str,
        required_columns: list[str] | None = None,
        timestamp_format: str | None = None,
    ):
        self.filepath = filepath
        self.required_columns = required_columns or self.DEFAULT_REQUIRED_COLUMNS
        self.timestamp_format = timestamp_format

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Read the CSV, validate columns, convert timestamp, and return a DataFrame.

        Returns
        -------
        pd.DataFrame
            Standardised commodity-price DataFrame.

        Raises
        ------
        FileNotFoundError
            If the CSV file does not exist at ``self.filepath``.
        ValueError
            If required columns are missing.
        """
        self._check_file_exists()

        logger.info("Loading commodity data from %s", self.filepath)
        df = pd.read_csv(self.filepath)
        logger.info("%d records loaded", len(df))

        self._validate_columns(df)
        df = self._convert_timestamp(df)

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_file_exists(self) -> None:
        """Raise ``FileNotFoundError`` with a helpful message if the file is missing."""
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(
                f"Data file not found: {self.filepath}\n"
                "Please run  python scripts/generate_mock_data.py  first, "
                "or check the CSV_PATH setting in config.py."
            )

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Ensure all required columns are present in the DataFrame."""
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}. "
                f"Found columns: {list(df.columns)}"
            )
        logger.info("All required columns present")

    def _convert_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the ``timestamp`` column to ``datetime64``."""
        try:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], format=self.timestamp_format
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to parse 'timestamp' column: {exc}. "
                "Check TIMESTAMP_FORMAT in config.py."
            ) from exc
        logger.info("Timestamps converted to datetime")
        return df
