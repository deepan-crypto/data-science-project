"""
Commodity Volatility Monitoring Dashboard
=========================================
Interactive Streamlit + Plotly dashboard.

Launch with:
    streamlit run dashboard/app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402
from analytics.correlation import CorrelationAnalyzer  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Commodity Volatility Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS for premium look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Dark glassmorphism cards ── */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30,30,60,0.85), rgba(20,20,45,0.95));
        border: 1px solid rgba(120,120,255,0.15);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 32px rgba(100,100,255,0.25);
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e0e7ff;
    }
    .kpi-sub {
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }
    .kpi-up   { color: #34d399; }
    .kpi-down { color: #f87171; }

    /* ── Status badges ── */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.85rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.8px;
    }
    .badge-low      { background: #064e3b; color: #6ee7b7; }
    .badge-moderate { background: #78350f; color: #fcd34d; }
    .badge-high     { background: #7c2d12; color: #fdba74; }
    .badge-extreme  { background: #7f1d1d; color: #fca5a5; }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #c7d2fe;
        border-bottom: 2px solid rgba(120,120,255,0.3);
        padding-bottom: 0.4rem;
        margin: 2rem 0 1rem 0;
        letter-spacing: 0.5px;
    }

    /* ── Anomaly alert ── */
    .anomaly-alert {
        background: linear-gradient(135deg, rgba(127,29,29,0.6), rgba(50,10,10,0.8));
        border-left: 4px solid #f87171;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        color: #fecaca;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the processed dataset produced by main.py."""
    path = config.PROCESSED_DATA_PATH
    if not os.path.isfile(path):
        st.error(
            f"Processed data not found at `{path}`.  "
            "Run **`python main.py`** first to generate it."
        )
        st.stop()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


df = load_data()
commodities = sorted(df["commodity"].unique())

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Filters
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🎛️ Filters")

selected_commodity = st.sidebar.selectbox(
    "Commodity",
    commodities,
    index=0,
    key="commodity_filter",
)

date_min = df["timestamp"].min().date()
date_max = df["timestamp"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
    key="date_filter",
)

vol_levels = ["All", "LOW", "MODERATE", "HIGH", "EXTREME"]
selected_vol = st.sidebar.selectbox(
    "Volatility Level",
    vol_levels,
    index=0,
    key="vol_filter",
)

show_anomalies_only = st.sidebar.checkbox(
    "Show anomalies only", value=False, key="anomaly_toggle"
)

# ── Apply filters ────────────────────────────────────────────────────────────
filtered = df.copy()

# Date range
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered = filtered[(filtered["timestamp"] >= start) & (filtered["timestamp"] <= end)]

# Volatility level
if selected_vol != "All":
    filtered = filtered[filtered["volatility_class"] == selected_vol]

# Anomaly toggle
if show_anomalies_only:
    filtered = filtered[filtered["is_anomaly"] == True]  # noqa: E712

# Commodity-specific slice (for KPIs and single-commodity charts)
comm_df = filtered[filtered["commodity"] == selected_commodity].sort_values("timestamp")

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; color:#c7d2fe; letter-spacing:2px;'>"
    "📊 Commodity Volatility Monitoring System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#9ca3af; margin-top:-0.5rem;'>"
    "Real-time commodity price analysis &amp; volatility mapping</p>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI Cards
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

if len(comm_df) > 0:
    latest = comm_df.iloc[-1]
    latest_price = latest["close"]
    daily_ret = latest.get("daily_return", 0)
    vol_7_ann = latest.get("volatility_7_ann", 0)
    vol_class = latest.get("volatility_class", "N/A")

    ret_pct = (daily_ret * 100) if pd.notna(daily_ret) else 0
    vol_pct = (vol_7_ann * 100) if pd.notna(vol_7_ann) else 0
    ret_class = "kpi-up" if ret_pct >= 0 else "kpi-down"
    ret_sign = "+" if ret_pct >= 0 else ""

    badge_class = {
        "LOW": "badge-low",
        "MODERATE": "badge-moderate",
        "HIGH": "badge-high",
        "EXTREME": "badge-extreme",
    }.get(vol_class, "badge-low")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Selected Commodity</div>'
            f'<div class="kpi-value">{selected_commodity}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Latest Price</div>'
            f'<div class="kpi-value">${latest_price:,.2f}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Daily Return</div>'
            f'<div class="kpi-value {ret_class}">{ret_sign}{ret_pct:.2f}%</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Volatility (7d ann.)</div>'
            f'<div class="kpi-value">{vol_pct:.2f}%</div>'
            f'<div class="kpi-sub"><span class="badge {badge_class}">{vol_class}</span></div>'
            f"</div>",
            unsafe_allow_html=True,
        )
else:
    st.info("No data available for the selected filters.")

# ─────────────────────────────────────────────────────────────────────────────
# All-Commodity Summary Table
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Commodity Overview</div>', unsafe_allow_html=True)

summary_rows = []
for comm in commodities:
    cdf = df[df["commodity"] == comm].sort_values("timestamp")
    if cdf.empty:
        continue
    last = cdf.iloc[-1]
    summary_rows.append(
        {
            "Commodity": comm,
            "Latest Price": f"${last['close']:,.2f}",
            "Daily Return": f"{(last.get('daily_return', 0) or 0) * 100:+.2f}%",
            "Volatility (7d)": f"{(last.get('volatility_7_ann', 0) or 0) * 100:.2f}%",
            "Status": last.get("volatility_class", "N/A"),
        }
    )
if summary_rows:
    st.dataframe(
        pd.DataFrame(summary_rows).set_index("Commodity"),
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Price Chart (Candlestick)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Price Movement</div>', unsafe_allow_html=True)

if len(comm_df) > 0:
    fig_price = go.Figure()

    # Candlestick
    fig_price.add_trace(
        go.Candlestick(
            x=comm_df["timestamp"],
            open=comm_df["open"],
            high=comm_df["high"],
            low=comm_df["low"],
            close=comm_df["close"],
            name="OHLC",
            increasing_line_color="#34d399",
            decreasing_line_color="#f87171",
        )
    )

    # Mark anomalies
    anomalies = comm_df[comm_df["is_anomaly"] == True]  # noqa: E712
    if len(anomalies) > 0:
        fig_price.add_trace(
            go.Scatter(
                x=anomalies["timestamp"],
                y=anomalies["high"] * 1.01,
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#fbbf24"),
                name="Unusual Movement",
            )
        )

    fig_price.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,30,0.6)",
        xaxis_rangeslider_visible=False,
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_price, use_container_width=True)
else:
    st.info("No price data to display.")

# ─────────────────────────────────────────────────────────────────────────────
# Volatility Chart (dual-axis: price + volatility)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">Price &amp; Volatility</div>',
    unsafe_allow_html=True,
)

if len(comm_df) > 0:
    from plotly.subplots import make_subplots

    fig_vol = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.06,
    )

    # Close price
    fig_vol.add_trace(
        go.Scatter(
            x=comm_df["timestamp"],
            y=comm_df["close"],
            name="Close Price",
            line=dict(color="#818cf8", width=1.5),
        ),
        row=1,
        col=1,
    )

    # Volatility lines
    colors = {"volatility_7_ann": "#34d399", "volatility_20_ann": "#fbbf24", "volatility_30_ann": "#f87171"}
    labels = {"volatility_7_ann": "Vol 7d", "volatility_20_ann": "Vol 20d", "volatility_30_ann": "Vol 30d"}
    for col_name, color in colors.items():
        if col_name in comm_df.columns:
            fig_vol.add_trace(
                go.Scatter(
                    x=comm_df["timestamp"],
                    y=comm_df[col_name] * 100,
                    name=labels.get(col_name, col_name),
                    line=dict(color=color, width=1.2),
                ),
                row=2,
                col=1,
            )

    # Spike markers
    spikes = comm_df[comm_df["is_volatility_spike"] == True]  # noqa: E712
    if len(spikes) > 0 and "volatility_7_ann" in spikes.columns:
        fig_vol.add_trace(
            go.Scatter(
                x=spikes["timestamp"],
                y=spikes["volatility_7_ann"] * 100,
                mode="markers",
                marker=dict(symbol="star", size=9, color="#f87171"),
                name="Vol Spike",
            ),
            row=2,
            col=1,
        )

    fig_vol.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,30,0.6)",
        height=550,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.06),
    )
    fig_vol.update_yaxes(title_text="Price", row=1, col=1)
    fig_vol.update_yaxes(title_text="Volatility %", row=2, col=1)

    st.plotly_chart(fig_vol, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Volatility Heatmap (all commodities × time)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">Volatility Heatmap — All Commodities</div>',
    unsafe_allow_html=True,
)

if "volatility_7_ann" in filtered.columns:
    # Resample to weekly for readability
    heatmap_df = filtered.copy()
    heatmap_df["week"] = heatmap_df["timestamp"].dt.to_period("W").dt.start_time
    heatmap_pivot = heatmap_df.pivot_table(
        index="commodity",
        columns="week",
        values="volatility_7_ann",
        aggfunc="mean",
    )

    fig_hm = px.imshow(
        heatmap_pivot.values * 100,
        x=[d.strftime("%Y-%m-%d") for d in heatmap_pivot.columns],
        y=heatmap_pivot.index.tolist(),
        color_continuous_scale="Inferno",
        labels=dict(x="Week", y="Commodity", color="Vol %"),
        aspect="auto",
    )
    fig_hm.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,30,0.6)",
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(nticks=20),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">Return Correlation Matrix</div>',
    unsafe_allow_html=True,
)

if "daily_return" in filtered.columns:
    analyzer = CorrelationAnalyzer()
    try:
        corr = analyzer.compute(filtered)
        fig_corr = px.imshow(
            corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            labels=dict(color="Correlation"),
        )
        fig_corr.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,10,30,0.6)",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute correlations: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Section
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">⚠️ Unusual Market Movements</div>',
    unsafe_allow_html=True,
)

anomaly_df = filtered[filtered["is_anomaly"] == True].copy()  # noqa: E712

if len(anomaly_df) > 0:
    display_cols = ["timestamp", "commodity", "close", "daily_return", "volatility_7_ann", "anomaly_score"]
    display_cols = [c for c in display_cols if c in anomaly_df.columns]
    anom_display = anomaly_df[display_cols].copy()
    anom_display = anom_display.sort_values("timestamp", ascending=False).head(50)

    # Format for readability
    if "daily_return" in anom_display.columns:
        anom_display["daily_return"] = (anom_display["daily_return"] * 100).round(2).astype(str) + "%"
    if "volatility_7_ann" in anom_display.columns:
        anom_display["volatility_7_ann"] = (anom_display["volatility_7_ann"] * 100).round(2).astype(str) + "%"
    if "close" in anom_display.columns:
        anom_display["close"] = anom_display["close"].apply(lambda x: f"${x:,.2f}")
    if "anomaly_score" in anom_display.columns:
        anom_display["anomaly_score"] = anom_display["anomaly_score"].round(4)

    anom_display.columns = [c.replace("_", " ").title() for c in anom_display.columns]
    st.dataframe(anom_display, use_container_width=True, hide_index=True)
else:
    st.success("No unusual market movements detected for the selected filters.")

# ─────────────────────────────────────────────────────────────────────────────
# Volatility Spike Section
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">🔺 Volatility Spikes</div>',
    unsafe_allow_html=True,
)

spike_df = filtered[filtered["is_volatility_spike"] == True].copy()  # noqa: E712

if len(spike_df) > 0:
    spike_cols = ["timestamp", "commodity", "close", "volatility_7_ann", "historical_avg_vol", "volatility_ratio"]
    spike_cols = [c for c in spike_cols if c in spike_df.columns]
    spike_display = spike_df[spike_cols].copy()
    spike_display = spike_display.sort_values("timestamp", ascending=False).head(50)

    if "volatility_7_ann" in spike_display.columns:
        spike_display["volatility_7_ann"] = (spike_display["volatility_7_ann"] * 100).round(2).astype(str) + "%"
    if "historical_avg_vol" in spike_display.columns:
        spike_display["historical_avg_vol"] = (spike_display["historical_avg_vol"] * 100).round(2).astype(str) + "%"
    if "volatility_ratio" in spike_display.columns:
        spike_display["volatility_ratio"] = spike_display["volatility_ratio"].round(2).astype(str) + "×"
    if "close" in spike_display.columns:
        spike_display["close"] = spike_display["close"].apply(lambda x: f"${x:,.2f}")

    spike_display.columns = [c.replace("_", " ").title() for c in spike_display.columns]
    st.dataframe(spike_display, use_container_width=True, hide_index=True)
else:
    st.success("No volatility spikes detected for the selected filters.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6b7280; font-size:0.78rem;'>"
    "⚠️ This dashboard is for educational purposes only. "
    "Volatility categories are project-specific and do not represent investment advice.</p>",
    unsafe_allow_html=True,
)
