"""
Feature Engineering
===================
Computes derived features for each commodity:
  • Daily return, log return, price change
  • Rolling mean & std (configurable windows)
  • Volume change (percentage)
"""

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generate time-series features per commodity.

    Parameters
    ----------
    rolling_windows : Sequence[int]
        List of window sizes (in trading days) for rolling statistics.
    """

    def __init__(self, rolling_windows: Sequence[int] = (7, 20)):
        self.rolling_windows = rolling_windows

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all engineered features and return the enriched DataFrame."""
        df = df.copy()
        df = df.sort_values(["commodity", "timestamp"]).reset_index(drop=True)

        df = self._add_returns(df)
        df = self._add_rolling_stats(df)
        df = self._add_volume_change(df)

        logger.info("Feature engineering completed — %d columns", df.shape[1])
        return df

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    @staticmethod
    def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Compute daily return, log return, and price change per commodity."""

        # Simple (percentage) return: (close_t - close_{t-1}) / close_{t-1}
        df["daily_return"] = df.groupby("commodity")["close"].pct_change()

        # Log return: ln(close_t / close_{t-1})
        df["log_return"] = df.groupby("commodity")["close"].transform(
            lambda s: np.log(s / s.shift(1))
        )

        # Absolute price change
        df["price_change"] = df.groupby("commodity")["close"].diff()

        logger.info("Calculated daily_return, log_return, price_change")
        return df

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling mean and std of close price for each window."""

        for window in self.rolling_windows:
            mean_col = f"rolling_mean_{window}"
            std_col = f"rolling_std_{window}"

            df[mean_col] = df.groupby("commodity")["close"].transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            )
            df[std_col] = df.groupby("commodity")["close"].transform(
                lambda s: s.rolling(window=window, min_periods=1).std()
            )

            logger.info("Calculated %s and %s", mean_col, std_col)

        return df

    # ------------------------------------------------------------------
    # Volume change
    # ------------------------------------------------------------------

    @staticmethod
    def _add_volume_change(df: pd.DataFrame) -> pd.DataFrame:
        """Compute percentage change in volume per commodity."""
        df["volume_change"] = df.groupby("commodity")["volume"].pct_change()
        logger.info("Calculated volume_change")
        return df
