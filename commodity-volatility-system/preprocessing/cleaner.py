"""
Data Cleaner
=============
Cleans validated commodity-price data:
  • Removes exact duplicates
  • Fills missing prices (forward-fill → linear interpolation per commodity)
  • Fills missing volume (rolling median per commodity)
  • Flags and drops structurally invalid OHLC rows
  • Flags price outliers (IQR on daily returns) — does NOT remove them
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean commodity-price data while preserving legitimate extreme movements.

    Parameters
    ----------
    iqr_multiplier : float
        Multiplier for the IQR method when flagging outliers (default 1.5).
    zscore_threshold : float
        Absolute z-score threshold for outlier flagging (default 3.0).
    """

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
    ):
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline and return a cleaned DataFrame.

        Steps (in order):
        1. Remove exact duplicate rows
        2. Drop rows with invalid OHLC relationships
        3. Fill missing price values
        4. Fill missing volume values
        5. Flag price outliers (``is_price_outlier`` column)
        """
        df = df.copy()

        df = self._remove_duplicates(df)
        df = self._drop_invalid_ohlc(df)
        df = self._fill_missing_prices(df)
        df = self._fill_missing_volume(df)
        df = self._flag_outliers(df)

        logger.info("Cleaning completed — %d records remain", len(df))
        return df

    # ------------------------------------------------------------------
    # Step 1: Duplicate removal
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Drop exact duplicate rows."""
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)
        if removed:
            logger.info("Removed %d duplicate records", removed)
        return df

    # ------------------------------------------------------------------
    # Step 2: Invalid OHLC rows
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_invalid_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """Flag and remove rows that violate OHLC relationships."""
        ohlc = df[["open", "high", "low", "close"]].dropna(how="any")
        invalid_mask = (
            (ohlc["high"] < ohlc["open"])
            | (ohlc["high"] < ohlc["close"])
            | (ohlc["low"] > ohlc["open"])
            | (ohlc["low"] > ohlc["close"])
            | (ohlc["high"] < ohlc["low"])
        )
        invalid_indices = ohlc.index[invalid_mask]
        if len(invalid_indices):
            logger.info(
                "Dropping %d rows with invalid OHLC relationships",
                len(invalid_indices),
            )
            df = df.drop(invalid_indices).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Step 3: Fill missing prices
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_missing_prices(df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill then linearly interpolate missing price values per commodity."""
        price_cols = ["open", "high", "low", "close"]
        missing_before = df[price_cols].isnull().sum().sum()

        # Sort so forward-fill is chronological
        df = df.sort_values(["commodity", "timestamp"]).reset_index(drop=True)

        for col in price_cols:
            df[col] = df.groupby("commodity")[col].transform(
                lambda s: s.ffill().interpolate(method="linear")
            )

        missing_after = df[price_cols].isnull().sum().sum()
        filled = missing_before - missing_after
        if filled:
            logger.info("Filled %d missing price values (ffill + interpolation)", filled)
        return df

    # ------------------------------------------------------------------
    # Step 4: Fill missing volume
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_missing_volume(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing volume with per-commodity rolling median (window=5).

        Falls back to the commodity's overall median for rows that are
        still NaN after the rolling fill (e.g. the first few rows).
        """
        missing_before = df["volume"].isnull().sum()

        df["volume"] = df.groupby("commodity")["volume"].transform(
            lambda s: s.fillna(s.rolling(window=5, min_periods=1).median())
        )
        # Fallback: overall commodity median for any remaining NaN
        df["volume"] = df.groupby("commodity")["volume"].transform(
            lambda s: s.fillna(s.median())
        )

        missing_after = df["volume"].isnull().sum()
        filled = missing_before - missing_after
        if filled:
            logger.info("Filled %d missing volume values (rolling median)", filled)
        return df

    # ------------------------------------------------------------------
    # Step 5: Outlier flagging
    # ------------------------------------------------------------------

    def _flag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``is_price_outlier`` column using IQR on daily returns.

        Outliers are flagged, **not removed**, because extreme financial
        movements can be legitimate market events.
        """
        df = df.sort_values(["commodity", "timestamp"]).reset_index(drop=True)

        # Compute simple return for outlier detection
        df["_return_for_outlier"] = df.groupby("commodity")["close"].pct_change()

        df["is_price_outlier"] = False

        for commodity in df["commodity"].unique():
            mask = df["commodity"] == commodity
            returns = df.loc[mask, "_return_for_outlier"].dropna()

            if returns.empty:
                continue

            q1 = returns.quantile(0.25)
            q3 = returns.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr

            outlier_idx = returns[(returns < lower) | (returns > upper)].index
            df.loc[outlier_idx, "is_price_outlier"] = True

        outlier_count = df["is_price_outlier"].sum()
        logger.info(
            "Flagged %d price outliers (IQR multiplier=%.1f)",
            outlier_count,
            self.iqr_multiplier,
        )

        # Clean up helper column
        df = df.drop(columns=["_return_for_outlier"])

        return df
