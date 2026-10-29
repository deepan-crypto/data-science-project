# Commodity Volatility Monitoring System

An end-to-end Data Science pipeline that ingests commodity price data, validates and cleans it, engineers features, calculates volatility, detects anomalies, analyses correlations, and presents results through an interactive Streamlit dashboard.

> **Note**: This project is for **educational purposes only**. The volatility categories and anomaly flags are project-specific and do **not** represent investment advice or universal financial standards.

---

## Architecture

```
CSV DATA SOURCE
      |
DATA INGESTION  (ingestion/csv_loader.py)
      |
RAW DATA
      |
DATA VALIDATION (preprocessing/validator.py)
      |
DATA CLEANING   (preprocessing/cleaner.py)
      |
FEATURE ENGINEERING (preprocessing/feature_engineering.py)
      |
VOLATILITY ENGINE   (analytics/volatility.py)
      |
ANOMALY DETECTION   (analytics/anomaly_detection.py)
      |
CORRELATION ANALYSIS (analytics/correlation.py)
      |
INTERACTIVE DASHBOARD (dashboard/app.py)
```

### Project Structure

```
commodity-volatility-system/
|
+-- config.py                       # Central configuration
+-- main.py                         # Pipeline orchestrator
|
+-- data/
|   +-- raw/
|   |   +-- commodity_prices.csv    # Mock historical data
|   +-- processed/
|       +-- commodity_features.csv  # Pipeline output
|
+-- scripts/
|   +-- generate_mock_data.py       # Reproducible data generator
|
+-- ingestion/
|   +-- base_loader.py              # Abstract DataLoader interface
|   +-- csv_loader.py               # CSV implementation
|
+-- preprocessing/
|   +-- validator.py                # Data quality checks
|   +-- cleaner.py                  # Cleaning + outlier flagging
|   +-- feature_engineering.py      # Returns, rolling stats, volume change
|
+-- analytics/
|   +-- volatility.py              # Rolling/annualised volatility + spikes
|   +-- anomaly_detection.py       # Isolation Forest anomaly detection
|   +-- correlation.py             # Return correlation matrix
|
+-- dashboard/
|   +-- app.py                     # Streamlit + Plotly dashboard
|
+-- tests/                         # Pytest test suite
|
+-- requirements.txt
+-- README.md
```

---

## Dataset

The mock CSV (`data/raw/commodity_prices.csv`) contains ~3,800 records covering 5 commodities over ~3 years of trading days:

| Commodity    | Approx. Base Price | Volatility Profile |
|-------------|--------------------|--------------------|
| Gold        | $1,800             | Low                |
| Silver      | $25                | Moderate           |
| Crude Oil   | $70                | Moderate-High      |
| Natural Gas | $3.50              | High               |
| Copper      | $4.20              | Moderate           |

**Columns**: `timestamp`, `commodity`, `open`, `high`, `low`, `close`, `volume`

The data intentionally includes: missing values (~1%), duplicate records (~0.5%), extreme price movements, and unusual volume spikes.

---

## Data Preprocessing

### Validation (`preprocessing/validator.py`)
- Missing columns check
- Missing values per column
- Duplicate row detection
- Invalid timestamp detection
- Negative price detection
- OHLC relationship validation (`high >= open/close`, `low <= open/close`, `high >= low`)
- Invalid volume detection

### Cleaning (`preprocessing/cleaner.py`)
- **Duplicates**: exact duplicates removed
- **Missing prices**: forward-fill then linear interpolation per commodity
- **Missing volume**: rolling median (window=5) per commodity
- **Invalid OHLC**: structurally broken rows dropped
- **Outliers**: flagged via IQR on daily returns (`is_price_outlier`), **not removed**

---

## Feature Engineering

Per commodity, sorted by timestamp:

| Feature          | Formula                                    |
|-----------------|---------------------------------------------|
| `daily_return`  | `(close_t - close_{t-1}) / close_{t-1}`    |
| `log_return`    | `ln(close_t / close_{t-1})`                |
| `price_change`  | `close_t - close_{t-1}`                    |
| `rolling_mean_N`| Rolling mean of close (N = 7, 20)          |
| `rolling_std_N` | Rolling std of close (N = 7, 20)           |
| `volume_change` | Percentage change in volume                |

---

## Volatility Calculation

- **Rolling volatility**: standard deviation of `daily_return` over windows of 7, 20, and 30 days
- **Annualised volatility**: `daily_vol * sqrt(252)`
- **Classification** (configurable thresholds on annualised volatility):

| Level    | Threshold         |
|----------|-------------------|
| LOW      | < 15%             |
| MODERATE | 15% - 30%        |
| HIGH     | 30% - 50%        |
| EXTREME  | > 50%            |

- **Spike detection**: flags when `current_vol / historical_avg_vol > 2.0`

---

## Anomaly Detection

Uses **Isolation Forest** (scikit-learn) on features: `daily_return`, `log_return`, `volume_change`, `rolling_std_7`, `rolling_std_20`.

Output columns: `anomaly_score`, `is_anomaly`.

Anomalies are labelled **"Unusual Market Movement"** — they do not imply a financial crisis.

---

## Correlation Analysis

Computes Pearson correlation matrix across commodity **returns** (not raw prices) to measure co-movement.

---

## Dashboard

Interactive Streamlit + Plotly dashboard with:

- **KPI cards**: selected commodity, latest price, daily return, volatility + status badge
- **Commodity overview table**: all commodities at a glance
- **Candlestick chart**: OHLC price movement with anomaly markers
- **Volatility chart**: dual-axis price + multi-period volatility with spike markers
- **Volatility heatmap**: all commodities across time
- **Correlation heatmap**: return correlation matrix
- **Anomaly table**: recent unusual movements
- **Volatility spike table**: recent spike events

### Filters (sidebar)
- Commodity selector
- Date range
- Volatility level
- Show anomalies only toggle

---

## Installation

```bash
# Clone and enter the project
cd commodity-volatility-system

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

### 1. Generate mock data (optional — already included)

```bash
python scripts/generate_mock_data.py
```

### 2. Run the full pipeline

```bash
python main.py
```

This produces `data/processed/commodity_features.csv`.

### 3. Launch the React Dashboard (Recommended)

Start the FastAPI backend:
```bash
python -m uvicorn api.server:app --reload --port 8000
```

In a second terminal, start the React Vite dev server:
```bash
cd frontend
npm install
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser.

### 4. Alternative: Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Future API Integration

The ingestion layer uses an **abstract `DataLoader` base class** (`ingestion/base_loader.py`):

```python
class DataLoader(ABC):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        ...
```

Currently implemented by `CSVDataLoader`. To switch to a live financial API:

1. Create `ingestion/api_loader.py` implementing `DataLoader`
2. Return the same DataFrame schema: `timestamp, commodity, open, high, low, close, volume`
3. Update `main.py` to use `APIDataLoader` instead of `CSVDataLoader`

**No changes needed** in preprocessing, analytics, or dashboard modules.

---

## Configuration

All tunable parameters are centralised in `config.py`:

| Parameter               | Description                                |
|------------------------|--------------------------------------------|
| `CSV_PATH`             | Path to raw CSV data                       |
| `PROCESSED_DATA_PATH`  | Output path for processed data             |
| `ROLLING_WINDOWS`      | Windows for rolling statistics `[7, 20, 30]` |
| `VOLATILITY_THRESHOLDS`| Classification boundaries                  |
| `VOLATILITY_SPIKE_THRESHOLD` | Spike detection multiplier            |
| `ANOMALY_CONTAMINATION`| Isolation Forest contamination rate        |
| `IQR_MULTIPLIER`       | IQR multiplier for outlier flagging        |
| `RANDOM_SEED`          | Reproducibility seed                       |

---

## License

This project is for educational and demonstration purposes.
