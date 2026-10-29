"""
FastAPI Backend Server
======================
Serves processed commodity data as JSON endpoints for the React dashboard.

Start with:
    uvicorn api.server:app --reload
"""

import os
import sys

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402

app = FastAPI(
    title="Commodity Volatility API",
    description="REST API for the Commodity Volatility Monitoring System",
    version="1.0.0",
)

# ── CORS — allow React dev server ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load data on startup ─────────────────────────────────────────────────────
DATA: pd.DataFrame = pd.DataFrame()


@app.on_event("startup")
def load_data():
    """Read processed CSV into memory on server start."""
    global DATA
    path = config.PROCESSED_DATA_PATH
    if not os.path.isfile(path):
        print(
            f"WARNING: Processed data not found at {path}. "
            "Run 'python main.py' first."
        )
        return
    DATA = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"Loaded {len(DATA)} records from {path}")

# Load immediately on import as well
load_data()


# ── Helper ────────────────────────────────────────────────────────────────────

def _safe_val(v):
    """Convert numpy/pandas types to native Python for JSON serialisation."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of JSON-safe dicts."""
    records = df.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            rec[k] = _safe_val(v)
    return records


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/commodities")
def get_commodities() -> list[str]:
    """List available commodity names."""
    if DATA.empty:
        return []
    return sorted(DATA["commodity"].unique().tolist())


@app.get("/api/summary")
def get_summary():
    """Latest price, return, volatility, and status for every commodity."""
    if DATA.empty:
        return []
    rows = []
    for comm in sorted(DATA["commodity"].unique()):
        cdf = DATA[DATA["commodity"] == comm].sort_values("timestamp")
        if cdf.empty:
            continue
        last = cdf.iloc[-1]
        rows.append({
            "commodity": comm,
            "latest_price": _safe_val(last["close"]),
            "daily_return": _safe_val(last.get("daily_return")),
            "volatility_7_ann": _safe_val(last.get("volatility_7_ann")),
            "volatility_class": _safe_val(last.get("volatility_class")),
        })
    return rows


@app.get("/api/prices")
def get_prices(
    commodity: str = Query(..., description="Commodity name"),
    start: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    end: str | None = Query(None, description="End date (YYYY-MM-DD)"),
):
    """OHLCV + feature data for a single commodity within an optional date range."""
    if DATA.empty:
        return []
    df = DATA[DATA["commodity"] == commodity].sort_values("timestamp").copy()
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end)]
    # Convert timestamp to string for JSON
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    return _df_to_records(df)


@app.get("/api/volatility-heatmap")
def get_volatility_heatmap():
    """Weekly-aggregated annualised 7-day volatility for all commodities."""
    if DATA.empty:
        return {"commodities": [], "weeks": [], "values": []}
    df = DATA.copy()
    df["week"] = df["timestamp"].dt.to_period("W").dt.start_time
    pivot = df.pivot_table(
        index="commodity",
        columns="week",
        values="volatility_7_ann",
        aggfunc="mean",
    )
    commodities = pivot.index.tolist()
    weeks = [d.strftime("%Y-%m-%d") for d in pivot.columns]
    # Replace NaN with None for JSON
    values = []
    for row in pivot.values:
        values.append([_safe_val(v * 100) if v is not None and not np.isnan(v) else None for v in row])
    return {"commodities": commodities, "weeks": weeks, "values": values}


@app.get("/api/correlation")
def get_correlation():
    """Pearson correlation matrix of daily returns across commodities."""
    if DATA.empty:
        return {"labels": [], "matrix": []}
    pivot = DATA.pivot_table(
        index="timestamp",
        columns="commodity",
        values="daily_return",
        aggfunc="mean",
    )
    corr = pivot.corr(method="pearson")
    labels = corr.columns.tolist()
    matrix = []
    for row in corr.values:
        matrix.append([round(float(v), 4) if not np.isnan(v) else None for v in row])
    return {"labels": labels, "matrix": matrix}


@app.get("/api/anomalies")
def get_anomalies(
    commodity: str | None = Query(None, description="Filter by commodity"),
    limit: int = Query(50, description="Max records to return"),
):
    """Recent unusual market movements."""
    if DATA.empty:
        return []
    df = DATA[DATA["is_anomaly"] == True].copy()  # noqa: E712
    if commodity:
        df = df[df["commodity"] == commodity]
    df = df.sort_values("timestamp", ascending=False).head(limit)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    cols = ["timestamp", "commodity", "close", "daily_return",
            "volatility_7_ann", "anomaly_score"]
    cols = [c for c in cols if c in df.columns]
    return _df_to_records(df[cols])


@app.get("/api/spikes")
def get_spikes(
    commodity: str | None = Query(None, description="Filter by commodity"),
    limit: int = Query(50, description="Max records to return"),
):
    """Recent volatility spikes."""
    if DATA.empty:
        return []
    df = DATA[DATA["is_volatility_spike"] == True].copy()  # noqa: E712
    if commodity:
        df = df[df["commodity"] == commodity]
    df = df.sort_values("timestamp", ascending=False).head(limit)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    cols = ["timestamp", "commodity", "close", "volatility_7_ann",
            "historical_avg_vol", "volatility_ratio"]
    cols = [c for c in cols if c in df.columns]
    return _df_to_records(df[cols])


@app.get("/api/stats")
def get_stats():
    """System-level metadata and dataset statistics."""
    if DATA.empty:
        return {
            "total_records": 0,
            "commodities_count": 0,
            "min_date": None,
            "max_date": None,
            "total_anomalies": 0,
            "total_spikes": 0,
        }
    return {
        "total_records": len(DATA),
        "commodities_count": int(DATA["commodity"].nunique()),
        "min_date": DATA["timestamp"].min().strftime("%Y-%m-%d"),
        "max_date": DATA["timestamp"].max().strftime("%Y-%m-%d"),
        "total_anomalies": int(DATA["is_anomaly"].sum()) if "is_anomaly" in DATA.columns else 0,
        "total_spikes": int(DATA["is_volatility_spike"].sum()) if "is_volatility_spike" in DATA.columns else 0,
    }


@app.get("/api/market-multi")
def get_market_multi():
    """Normalized price performance (base 100) across all commodities for relative comparison."""
    if DATA.empty:
        return {"dates": [], "series": {}}
    pivoted = DATA.pivot_table(index="timestamp", columns="commodity", values="close", aggfunc="mean").dropna()
    if pivoted.empty:
        return {"dates": [], "series": {}}
    # Normalize each to 100 at start
    normalized = (pivoted / pivoted.iloc[0]) * 100
    dates = [d.strftime("%Y-%m-%d") for d in normalized.index]
    series = {}
    for col in normalized.columns:
        series[col] = [round(float(v), 2) if not np.isnan(v) else None for v in normalized[col]]
    return {"dates": dates, "series": series}


@app.post("/api/pipeline/run")
def trigger_pipeline():
    """Trigger pipeline execution and reload data."""
    import subprocess
    try:
        cmd = [sys.executable, "-u", "main.py"]
        res = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(__file__), ".."), capture_output=True, text=True, timeout=120)
        load_data()
        return {
            "status": "success" if res.returncode == 0 else "error",
            "returncode": res.returncode,
            "records": len(DATA),
            "output": res.stdout[-500:] if res.stdout else res.stderr[-500:]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
