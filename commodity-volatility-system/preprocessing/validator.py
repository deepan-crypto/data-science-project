"""
Data Validator
==============
Runs quality checks on raw commodity-price data and produces a
structured validation report.  **Does not modify** the data.
"""

import logging
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate a commodity-price DataFrame against expected schema and rules.

    Parameters
    ----------
    required_columns : list[str]
        Columns that must be present.
    """

    DEFAULT_REQUIRED_COLUMNS = [
        "timestamp",
        "commodity",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    def __init__(self, required_columns: list[str] | None = None):
        self.required_columns = required_columns or self.DEFAULT_REQUIRED_COLUMNS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run all validation checks and return a summary report.

        Returns
        -------
        dict
            Keys: total_records, missing_columns, missing_values,
            duplicate_records, invalid_timestamps, negative_prices,
            invalid_ohlc_records, invalid_volume_records.
        """
        report: dict[str, Any] = {}

        report["total_records"] = len(df)

        # 1. Missing columns
        report["missing_columns"] = self._check_missing_columns(df)

        # If critical columns are missing, return early
        if report["missing_columns"]:
            logger.warning(
                "Missing columns detected: %s — skipping further checks",
                report["missing_columns"],
            )
            return report

        # 2. Missing values per column
        report["missing_values"] = self._check_missing_values(df)

        # 3. Duplicate records
        report["duplicate_records"] = self._check_duplicates(df)

        # 4. Invalid timestamps
        report["invalid_timestamps"] = self._check_timestamps(df)

        # 5. Negative prices
        report["negative_prices"] = self._check_negative_prices(df)

        # 6. Invalid OHLC relationships
        report["invalid_ohlc_records"] = self._check_ohlc(df)

        # 7. Invalid volume
        report["invalid_volume_records"] = self._check_volume(df)

        # Log summary
        self._log_report(report)

        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_missing_columns(self, df: pd.DataFrame) -> list[str]:
        missing = sorted(set(self.required_columns) - set(df.columns))
        if missing:
            logger.warning("Missing columns: %s", missing)
        return missing

    def _check_missing_values(self, df: pd.DataFrame) -> dict[str, int]:
        """Return per-column count of NaN / NaT values."""
        missing = df[self.required_columns].isnull().sum()
        result = {col: int(cnt) for col, cnt in missing.items() if cnt > 0}
        total = sum(result.values())
        if total:
            logger.warning("%d total missing values detected: %s", total, result)
        else:
            logger.info("No missing values detected")
        return result

    def _check_duplicates(self, df: pd.DataFrame) -> int:
        """Count exact duplicate rows."""
        count = int(df.duplicated().sum())
        if count:
            logger.warning("%d duplicate records detected", count)
        else:
            logger.info("No duplicate records detected")
        return count

    def _check_timestamps(self, df: pd.DataFrame) -> int:
        """Count rows where 'timestamp' is NaT or cannot be converted."""
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            try:
                ts = pd.to_datetime(df["timestamp"], errors="coerce")
            except Exception:
                return len(df)
        else:
            ts = df["timestamp"]
        invalid_count = int(ts.isna().sum())
        if invalid_count:
            logger.warning("%d invalid timestamps detected", invalid_count)
        return invalid_count

    def _check_negative_prices(self, df: pd.DataFrame) -> int:
        """Count rows with any negative price column."""
        price_cols = ["open", "high", "low", "close"]
        # Exclude NaN rows — those are caught by the missing-value check
        mask = (df[price_cols].dropna(how="any") < 0).any(axis=1)
        count = int(mask.sum())
        if count:
            logger.warning("%d records with negative prices", count)
        return count

    def _check_ohlc(self, df: pd.DataFrame) -> int:
        """Validate OHLC relationships.

        Rules:
            high >= open,  high >= close
            low  <= open,  low  <= close
            high >= low
        """
        valid_rows = df[["open", "high", "low", "close"]].dropna()
        invalid_mask = (
            (valid_rows["high"] < valid_rows["open"])
            | (valid_rows["high"] < valid_rows["close"])
            | (valid_rows["low"] > valid_rows["open"])
            | (valid_rows["low"] > valid_rows["close"])
            | (valid_rows["high"] < valid_rows["low"])
        )
        count = int(invalid_mask.sum())
        if count:
            logger.warning("%d records with invalid OHLC relationships", count)
        return count

    def _check_volume(self, df: pd.DataFrame) -> int:
        """Count rows where volume is negative."""
        valid_volume = df["volume"].dropna()
        count = int((valid_volume < 0).sum())
        if count:
            logger.warning("%d records with invalid (negative) volume", count)
        return count

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _log_report(report: dict) -> None:
        """Print a concise summary of the validation report."""
        logger.info("=" * 50)
        logger.info("VALIDATION REPORT")
        logger.info("=" * 50)
        logger.info("Total records        : %d", report["total_records"])
        logger.info("Missing values       : %s", report["missing_values"])
        logger.info("Duplicate records    : %d", report["duplicate_records"])
        logger.info("Invalid timestamps   : %d", report["invalid_timestamps"])
        logger.info("Negative prices      : %d", report["negative_prices"])
        logger.info("Invalid OHLC records : %d", report["invalid_ohlc_records"])
        logger.info("Invalid volume       : %d", report["invalid_volume_records"])
        logger.info("=" * 50)
