"""
Correlation Analysis
====================
Computes pairwise Pearson correlation between commodity **returns**
(not raw prices) to measure how commodities move together.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Compute correlation matrix of commodity returns.

    Parameters
    ----------
    return_column : str
        Column containing the return values to correlate (default: ``daily_return``).
    """

    def __init__(self, return_column: str = "daily_return"):
        self.return_column = return_column

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot returns by commodity and compute Pearson correlation.

        Returns
        -------
        pd.DataFrame
            Square correlation matrix with commodity names as row/column labels.
        """
        if self.return_column not in df.columns:
            raise ValueError(
                f"Column '{self.return_column}' not found in DataFrame. "
                "Run feature engineering first."
            )

        # Pivot: rows = timestamp, columns = commodity, values = return
        pivot = df.pivot_table(
            index="timestamp",
            columns="commodity",
            values=self.return_column,
            aggfunc="mean",
        )

        corr_matrix = pivot.corr(method="pearson")
        logger.info(
            "Correlation matrix computed for %d commodities",
            len(corr_matrix),
        )

        return corr_matrix
