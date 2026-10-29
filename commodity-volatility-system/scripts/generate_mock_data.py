"""
Mock Data Generator
===================
Generates realistic historical commodity price data (OHLCV) for testing
and development. The output CSV is written to data/raw/commodity_prices.csv.

Characteristics injected:
  • Normal price fluctuations via geometric random walk
  • Different base prices and volatility per commodity
  • ~1 % missing values (randomly set to NaN)
  • ~0.5 % exact duplicate rows
  • A handful of legitimate extreme price movements
  • A few unusual volume observations
"""

import os
import sys
import numpy as np
import pandas as pd

# Allow importing config from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# Commodity-specific parameters
# ---------------------------------------------------------------------------
# base_price : starting close price
# daily_vol  : daily return standard deviation
# base_volume: average daily volume (units vary by commodity)

COMMODITY_PARAMS = {
    "Gold": {
        "base_price": 1800.0,
        "daily_vol": 0.012,
        "base_volume": 180_000,
    },
    "Silver": {
        "base_price": 25.0,
        "daily_vol": 0.018,
        "base_volume": 75_000,
    },
    "Crude Oil": {
        "base_price": 70.0,
        "daily_vol": 0.022,
        "base_volume": 350_000,
    },
    "Natural Gas": {
        "base_price": 3.50,
        "daily_vol": 0.028,
        "base_volume": 120_000,
    },
    "Copper": {
        "base_price": 4.20,
        "daily_vol": 0.016,
        "base_volume": 65_000,
    },
}

# Number of trading days to generate (~3 years ≈ 756 trading days)
NUM_DAYS = 756

# Reproducibility
np.random.seed(config.RANDOM_SEED)


def _generate_single_commodity(
    commodity: str,
    params: dict,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Generate OHLCV data for a single commodity using a random walk."""

    n = len(dates)
    base_price = params["base_price"]
    daily_vol = params["daily_vol"]
    base_volume = params["base_volume"]

    # --- Close prices via geometric random walk ---
    daily_returns = np.random.normal(loc=0.0002, scale=daily_vol, size=n)

    # Inject a few extreme movements (legitimate market shocks)
    shock_indices = np.random.choice(n, size=5, replace=False)
    for idx in shock_indices:
        # Big move: 3–6 × normal volatility, random direction
        direction = np.random.choice([-1, 1])
        daily_returns[idx] = direction * daily_vol * np.random.uniform(3, 6)

    close_prices = base_price * np.cumprod(1 + daily_returns)

    # --- Open prices: close of previous day ± small overnight gap ---
    open_prices = np.empty(n)
    open_prices[0] = base_price
    for i in range(1, n):
        gap = np.random.normal(0, daily_vol * 0.3) * close_prices[i - 1]
        open_prices[i] = close_prices[i - 1] + gap

    # --- High / Low: extend beyond open/close ---
    high_prices = np.maximum(open_prices, close_prices) * (
        1 + np.abs(np.random.normal(0, daily_vol * 0.5, size=n))
    )
    low_prices = np.minimum(open_prices, close_prices) * (
        1 - np.abs(np.random.normal(0, daily_vol * 0.5, size=n))
    )

    # --- Volume ---
    volume = np.random.lognormal(
        mean=np.log(base_volume), sigma=0.3, size=n
    ).astype(int)

    # Inject a few unusual volume observations
    vol_spike_indices = np.random.choice(n, size=4, replace=False)
    for idx in vol_spike_indices:
        volume[idx] = int(volume[idx] * np.random.uniform(3, 7))

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "commodity": commodity,
            "open": np.round(open_prices, 4),
            "high": np.round(high_prices, 4),
            "low": np.round(low_prices, 4),
            "close": np.round(close_prices, 4),
            "volume": volume,
        }
    )

    return df


def generate_mock_data() -> pd.DataFrame:
    """Generate the full mock dataset for all commodities."""

    # Trading-day calendar (Mon–Fri, US holidays excluded via offset)
    dates = pd.bdate_range(start="2021-01-04", periods=NUM_DAYS, freq="B")

    frames = []
    for commodity, params in COMMODITY_PARAMS.items():
        df = _generate_single_commodity(commodity, params, dates)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # --- Inject ~1 % missing values across price / volume columns ---
    num_missing = int(len(data) * 0.01)
    cols_to_nan = ["open", "high", "low", "close", "volume"]
    for _ in range(num_missing):
        row_idx = np.random.randint(0, len(data))
        col = np.random.choice(cols_to_nan)
        data.loc[row_idx, col] = np.nan

    # --- Inject ~0.5 % duplicate rows ---
    num_duplicates = int(len(data) * 0.005)
    dup_indices = np.random.choice(len(data), size=num_duplicates, replace=False)
    duplicates = data.iloc[dup_indices].copy()
    data = pd.concat([data, duplicates], ignore_index=True)

    # Shuffle slightly so duplicates are not just appended at the end
    data = data.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)

    # Sort by commodity then timestamp for readability
    data = data.sort_values(["commodity", "timestamp"]).reset_index(drop=True)

    return data


def main():
    """Generate and save mock data to CSV."""
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

    print("Generating mock commodity data …")
    data = generate_mock_data()
    data.to_csv(config.CSV_PATH, index=False)
    print(f"Saved {len(data)} records to {config.CSV_PATH}")
    print(f"Commodities: {data['commodity'].unique().tolist()}")
    print(f"Date range : {data['timestamp'].min()} -> {data['timestamp'].max()}")


if __name__ == "__main__":
    main()
