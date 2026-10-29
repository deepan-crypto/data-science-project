"""
Configuration for the Commodity Volatility Monitoring System.

All tunable parameters are centralized here so that nothing is
hard-coded throughout the project. Adjust these values to change
system behavior without modifying pipeline code.
"""

import os

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------

# Base directory is the folder that contains this config file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Raw input data
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
CSV_PATH = os.path.join(RAW_DATA_DIR, "commodity_prices.csv")

# Processed output data
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "commodity_features.csv")

# ---------------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------------

# Required columns in the raw CSV / API response
REQUIRED_COLUMNS = [
    "timestamp",
    "commodity",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

# Commodities tracked by the system
COMMODITIES = ["Gold", "Silver", "Crude Oil", "Natural Gas", "Copper"]

# Timestamp format used in the raw data
TIMESTAMP_FORMAT = "%Y-%m-%d"

# ---------------------------------------------------------------------------
# Rolling Window Configuration
# ---------------------------------------------------------------------------

# Windows (in trading days) used for rolling statistics
ROLLING_WINDOWS = [7, 20, 30]

# ---------------------------------------------------------------------------
# Volatility Thresholds
# ---------------------------------------------------------------------------
# These thresholds are applied to *annualized* volatility (decimal form).
# They are project-specific defaults and do NOT represent universal
# financial standards.

VOLATILITY_THRESHOLDS = {
    "LOW": 0.15,       # annualized vol < 15%
    "MODERATE": 0.30,  # 15% – 30%
    "HIGH": 0.50,      # 30% – 50%
    # anything above 50% → EXTREME
}

# A volatility spike is flagged when the current rolling volatility
# exceeds the historical average by this multiplicative factor.
VOLATILITY_SPIKE_THRESHOLD = 2.0

# ---------------------------------------------------------------------------
# Anomaly Detection Configuration
# ---------------------------------------------------------------------------

# Contamination parameter for Isolation Forest.
# Represents the expected proportion of anomalies in the dataset.
ANOMALY_CONTAMINATION = 0.05

# Features fed into the Isolation Forest model
ANOMALY_FEATURES = [
    "daily_return",
    "log_return",
    "volume_change",
    "rolling_std_7",
    "rolling_std_20",
]

# Random seed for reproducibility
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Outlier Detection Configuration
# ---------------------------------------------------------------------------

# IQR multiplier for flagging price outliers
IQR_MULTIPLIER = 1.5

# Z-score threshold (absolute value) for flagging price outliers
ZSCORE_THRESHOLD = 3.0

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
