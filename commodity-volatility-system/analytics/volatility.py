"""
Volatility Engine
=================
Calculates rolling volatility, annualized volatility, classifies
volatility levels, and detects volatility spikes.

IMPORTANT:
    The volatility categories (LOW / MODERATE / HIGH / EXTREME) are
    configurable project defaults.  They do NOT represent universal
    financial risk classifications.
"""

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VolatilityEngine:
    """Compute and classify volatility for commodity return series.

    Parameters
    ----------
    rolling_windows : Sequence[int]
        Rolling windows (in trading days) for volatility calculation.
    thresholds : dict[str, float]
        Mapping of level names to upper bounds for annualised volatility.
        E.g. ``{"LOW": 0.15, "MODERATE": 0.30, "HIGH": 0.50}``.
        Anything above the highest threshold is labelled EXTREME.
    spike_threshold : float
        A spike is flagged when ``current_vol / historical_avg_vol``
        exceeds this multiplicative factor.
    trading_days_per_year : int
        Used for annualisation (default 252).
    """

    def __init__(
        self,
        rolling_windows: Sequence[int] = (7, 20, 30),
        thresholds: dict[str, float] | None = None,
        spike_threshold: float = 2.0,
        trading_days_per_year: int = 252,
    ):
        self.rolling_windows = rolling_windows
        self.thresholds = thresholds or {
            "LOW": 0.15,
            "MODERATE": 0.30,
            "HIGH": 0.50,
        }
        self.spike_threshold = spike_threshold
        self.trading_days = trading_days_per_year

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility columns to the DataFrame and return it.

        Columns added per window *w*:
            volatility_{w}      — rolling std of daily_return (window *w*)
            volatility_{w}_ann  — annualised version

        Additional columns (using the *first* window for classification):
            volatility_class    — LOW / MODERATE / HIGH / EXTREME
            historical_avg_vol  — expanding mean of the primary volatility
            volatility_ratio    — current / historical average
            is_volatility_spike — True when ratio > spike_threshold
        """
        df = df.copy()
        df = df.sort_values(["commodity", "timestamp"]).reset_index(drop=True)

        if "daily_return" not in df.columns:
            raise ValueError(
                "Column 'daily_return' not found.  "
                "Run FeatureEngineer.generate_features() first."
            )

        # --- Rolling volatility for each window ---
        primary_vol_col = None
        for window in self.rolling_windows:
            vol_col = f"volatility_{window}"
            ann_col = f"volatility_{window}_ann"

            df[vol_col] = df.groupby("commodity")["daily_return"].transform(
                lambda s: s.rolling(window=window, min_periods=2).std()
            )
            # Annualise: σ_annual = σ_daily × √(trading_days)
            df[ann_col] = df[vol_col] * np.sqrt(self.trading_days)

            logger.info("Calculated %s and %s", vol_col, ann_col)

            if primary_vol_col is None:
                primary_vol_col = ann_col  # first window used for classification

        # --- Classification ---
        df["volatility_class"] = df[primary_vol_col].apply(self._classify)
        logger.info("Volatility classification applied (based on %s)", primary_vol_col)

        # --- Spike detection ---
        df["historical_avg_vol"] = df.groupby("commodity")[primary_vol_col].transform(
            lambda s: s.expanding(min_periods=2).mean()
        )
        df["volatility_ratio"] = df[primary_vol_col] / df["historical_avg_vol"]
        df["is_volatility_spike"] = df["volatility_ratio"] > self.spike_threshold

        spike_count = df["is_volatility_spike"].sum()
        logger.info(
            "Detected %d volatility spikes (threshold=%.1f×)",
            spike_count,
            self.spike_threshold,
        )

        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _classify(self, annualized_vol: float) -> str:
        """Map an annualised volatility value to a category string."""
        if pd.isna(annualized_vol):
            return "N/A"

        # Iterate thresholds in ascending order
        for level in ["LOW", "MODERATE", "HIGH"]:
            if annualized_vol < self.thresholds.get(level, float("inf")):
                return level
        return "EXTREME"
