"""
Main Pipeline
=============
Orchestrates the full commodity volatility analysis pipeline:

    Load CSV → Validate → Clean → Feature Engineering
    → Volatility → Anomaly Detection → Save Processed Data

Run with:
    python main.py
"""

import logging
import os
import sys

# ── Configuration ────────────────────────────────────────────────────────────
import config
from ingestion.csv_loader import CSVDataLoader
from preprocessing.validator import DataValidator
from preprocessing.cleaner import DataCleaner
from preprocessing.feature_engineering import FeatureEngineer
from analytics.volatility import VolatilityEngine
from analytics.anomaly_detection import AnomalyDetector
from analytics.correlation import CorrelationAnalyzer

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format=config.LOG_FORMAT,
)
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """Execute the end-to-end analysis pipeline."""

    logger.info("=" * 60)
    logger.info("COMMODITY VOLATILITY ANALYSIS PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Data Ingestion ───────────────────────────────────────────
    logger.info("Step 1 -- Loading commodity data")
    loader = CSVDataLoader(
        filepath=config.CSV_PATH,
        timestamp_format=config.TIMESTAMP_FORMAT,
    )
    df = loader.load_data()

    # ── Step 2: Data Validation ──────────────────────────────────────────
    logger.info("Step 2 -- Validating data")
    validator = DataValidator(required_columns=config.REQUIRED_COLUMNS)
    report = validator.validate(df)

    # Abort if critical columns are missing
    if report.get("missing_columns"):
        logger.error("Cannot proceed -- missing columns: %s", report["missing_columns"])
        sys.exit(1)

    # ── Step 3: Data Cleaning ────────────────────────────────────────────
    logger.info("Step 3 -- Cleaning data")
    cleaner = DataCleaner(
        iqr_multiplier=config.IQR_MULTIPLIER,
        zscore_threshold=config.ZSCORE_THRESHOLD,
    )
    df = cleaner.clean(df)

    # ── Step 4: Feature Engineering ──────────────────────────────────────
    logger.info("Step 4 -- Generating features")
    engineer = FeatureEngineer(rolling_windows=config.ROLLING_WINDOWS[:2])  # 7, 20
    df = engineer.generate_features(df)

    # ── Step 5: Volatility Calculation ───────────────────────────────────
    logger.info("Step 5 -- Calculating volatility")
    vol_engine = VolatilityEngine(
        rolling_windows=config.ROLLING_WINDOWS,
        thresholds=config.VOLATILITY_THRESHOLDS,
        spike_threshold=config.VOLATILITY_SPIKE_THRESHOLD,
    )
    df = vol_engine.calculate(df)

    # ── Step 6: Anomaly Detection ────────────────────────────────────────
    logger.info("Step 6 -- Detecting anomalies")
    detector = AnomalyDetector(
        features=config.ANOMALY_FEATURES,
        contamination=config.ANOMALY_CONTAMINATION,
        random_state=config.RANDOM_SEED,
    )
    df = detector.detect(df)

    # ── Step 7: Correlation Analysis ─────────────────────────────────────
    logger.info("Step 7 -- Computing correlations")
    corr_analyzer = CorrelationAnalyzer()
    corr_matrix = corr_analyzer.compute(df)
    logger.info("Correlation matrix:\n%s", corr_matrix.to_string())

    # ── Step 8: Save Processed Data ──────────────────────────────────────
    logger.info("Step 8 -- Saving processed data")
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    logger.info("Processed data saved to %s", config.PROCESSED_DATA_PATH)

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info("Total records processed : %d", len(df))
    logger.info("Commodities             : %s", df["commodity"].unique().tolist())
    logger.info(
        "Date range              : %s -> %s",
        df["timestamp"].min().strftime("%Y-%m-%d"),
        df["timestamp"].max().strftime("%Y-%m-%d"),
    )
    logger.info("Anomalies detected      : %d", df["is_anomaly"].sum())
    logger.info("Volatility spikes       : %d", df["is_volatility_spike"].sum())
    logger.info(
        "Volatility distribution :\n%s",
        df["volatility_class"].value_counts().to_string(),
    )
    logger.info("Output file             : %s", config.PROCESSED_DATA_PATH)


if __name__ == "__main__":
    run_pipeline()
