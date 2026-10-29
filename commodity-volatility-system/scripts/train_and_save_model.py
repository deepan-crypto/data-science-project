"""
Train & Save Model
==================
Runs the feature-engineering and volatility pipeline on the raw
commodity data, trains an Isolation Forest anomaly-detection model,
and serialises it to a .pkl file for later reuse.

Run with:
    python scripts/train_and_save_model.py

Output:
    models/anomaly_detector.pkl      — trained Isolation Forest model
    models/model_metadata.json       — training metadata (features, params, date)
"""

import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Allow imports from the project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from preprocessing.feature_engineering import FeatureEngineer  # noqa: E402
from analytics.volatility import VolatilityEngine  # noqa: E402


def main():
    # ── 1. Load raw data ─────────────────────────────────────────────────
    print("[1/5] Loading raw data ...")
    df = pd.read_csv(
        config.CSV_PATH,
        parse_dates=["timestamp"],
    )
    print(f"      {len(df)} records loaded for {df['commodity'].nunique()} commodities")

    # ── 2. Feature engineering ───────────────────────────────────────────
    print("[2/5] Running feature engineering ...")
    engineer = FeatureEngineer(rolling_windows=config.ROLLING_WINDOWS[:2])
    df = engineer.generate_features(df)

    # ── 3. Volatility calculation ────────────────────────────────────────
    print("[3/5] Calculating volatility ...")
    vol_engine = VolatilityEngine(
        rolling_windows=config.ROLLING_WINDOWS,
        thresholds=config.VOLATILITY_THRESHOLDS,
        spike_threshold=config.VOLATILITY_SPIKE_THRESHOLD,
    )
    df = vol_engine.calculate(df)

    # ── 4. Train Isolation Forest ────────────────────────────────────────
    print("[4/5] Training Isolation Forest model ...")
    features = config.ANOMALY_FEATURES
    available = [f for f in features if f in df.columns]
    print(f"      Features: {available}")

    feature_df = df[available]
    valid_mask = feature_df.notna().all(axis=1)
    X = feature_df.loc[valid_mask].values

    model = IsolationForest(
        contamination=config.ANOMALY_CONTAMINATION,
        random_state=config.RANDOM_SEED,
        n_estimators=100,
    )
    model.fit(X)

    # Quick sanity check
    scores = model.decision_function(X)
    predictions = model.predict(X)
    anomaly_count = (predictions == -1).sum()
    print(f"      Trained on {len(X)} samples")
    print(f"      Anomalies in training data: {anomaly_count} ({anomaly_count / len(X) * 100:.1f}%)")
    print(f"      Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # ── 5. Save model & metadata ─────────────────────────────────────────
    print("[5/5] Saving model ...")
    models_dir = os.path.join(ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "anomaly_detector.pkl")
    joblib.dump(model, model_path)
    print(f"      Model saved -> {model_path}")

    # Save metadata for reproducibility
    metadata = {
        "model_type": "IsolationForest",
        "model_file": "anomaly_detector.pkl",
        "trained_at": datetime.now().isoformat(),
        "training_samples": int(len(X)),
        "features": available,
        "parameters": {
            "n_estimators": 100,
            "contamination": config.ANOMALY_CONTAMINATION,
            "random_state": config.RANDOM_SEED,
        },
        "training_stats": {
            "anomalies_detected": int(anomaly_count),
            "anomaly_rate": round(anomaly_count / len(X) * 100, 2),
            "score_min": round(float(scores.min()), 4),
            "score_max": round(float(scores.max()), 4),
            "score_mean": round(float(scores.mean()), 4),
        },
        "commodities": sorted(df["commodity"].unique().tolist()),
        "date_range": {
            "start": df["timestamp"].min().strftime("%Y-%m-%d"),
            "end": df["timestamp"].max().strftime("%Y-%m-%d"),
        },
    }

    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"      Metadata saved -> {metadata_path}")

    print("\n[OK] Done! Model is ready at: models/anomaly_detector.pkl")


if __name__ == "__main__":
    main()
