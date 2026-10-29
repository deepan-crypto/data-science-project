"""
Anomaly Detection
=================
Uses Isolation Forest from scikit-learn to flag **unusual market
movements** in commodity price data.

NOTE:
    An anomaly flag does NOT automatically mean a financial crisis.
    It simply indicates that the data point is statistically unusual
    compared to the rest of the dataset.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect unusual market movements using Isolation Forest.

    Parameters
    ----------
    features : list[str]
        Column names used as input features for the model.
    contamination : float
        Expected proportion of anomalies (0 < contamination ≤ 0.5).
    random_state : int
        Seed for reproducibility.
    """

    DEFAULT_FEATURES = [
        "daily_return",
        "log_return",
        "volume_change",
        "rolling_std_7",
        "rolling_std_20",
    ]

    def __init__(
        self,
        features: list[str] | None = None,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.features = features or self.DEFAULT_FEATURES
        self.contamination = contamination
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run anomaly detection and add result columns.

        Columns added:
            anomaly_score  — Isolation Forest decision function score
                             (more negative → more anomalous)
            is_anomaly     — Boolean flag (True = unusual market movement)

        Rows with NaN features are left un-scored (score = NaN, flag = False).
        """
        df = df.copy()

        # Identify rows that have all required features
        available = [f for f in self.features if f in df.columns]
        if not available:
            logger.warning(
                "None of the anomaly features %s found — skipping detection",
                self.features,
            )
            df["anomaly_score"] = np.nan
            df["is_anomaly"] = False
            return df

        feature_df = df[available]
        valid_mask = feature_df.notna().all(axis=1)
        X = feature_df.loc[valid_mask].values

        logger.info(
            "Running Isolation Forest on %d rows with features %s",
            len(X),
            available,
        )

        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        model.fit(X)

        # decision_function: higher = more normal, lower = more anomalous
        scores = model.decision_function(X)
        predictions = model.predict(X)  # 1 = normal, -1 = anomaly

        # Write results back
        df["anomaly_score"] = np.nan
        df.loc[valid_mask, "anomaly_score"] = scores

        df["is_anomaly"] = False
        df.loc[valid_mask, "is_anomaly"] = predictions == -1

        anomaly_count = df["is_anomaly"].sum()
        logger.info(
            "%d unusual market movements detected (contamination=%.2f)",
            anomaly_count,
            self.contamination,
        )

        return df
