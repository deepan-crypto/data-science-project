import React, { useState, useEffect, useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Zap,
  Layers,
  RefreshCw,
  Download,
  CheckCircle,
  X,
  Play,
  Sun,
  Coffee
} from 'lucide-react';
import './App.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export default function App() {
  // ── State ──────────────────────────────────────────────────────────────────
  const [commodities, setCommodities] = useState([]);
  const [summaryList, setSummaryList] = useState([]);
  const [selectedCommodity, setSelectedCommodity] = useState('Crude Oil');
  const [stats, setStats] = useState(null);
  const [prices, setPrices] = useState([]);
  const [heatmapData, setHeatmapData] = useState({ commodities: [], weeks: [], values: [] });
  const [correlationData, setCorrelationData] = useState({ labels: [], matrix: [] });
  const [anomalies, setAnomalies] = useState([]);
  const [spikes, setSpikes] = useState([]);
  const [multiMarket, setMultiMarket] = useState({ dates: [], series: {} });

  // UI state
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('cm_theme') || 'caramel';
  });
  const [datePreset, setDatePreset] = useState('1Y'); // '1M', '3M', '6M', '1Y', 'ALL'
  const [volFilter, setVolFilter] = useState('ALL');
  const [showAnomaliesOnly, setShowAnomaliesOnly] = useState(false);
  const [activeTab, setActiveTab] = useState('price'); // 'price', 'volatility', 'multi'
  const [pipelineModalOpen, setPipelineModalOpen] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);
  const [searchAnomaly, setSearchAnomaly] = useState('');
  const [searchSpike, setSearchSpike] = useState('');

  // ── Sync Theme with HTML Attribute ──────────────────────────────────────────
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('cm_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'caramel' ? 'white' : 'caramel'));
  };

  // ── Show Toast Helper ──────────────────────────────────────────────────────
  const showToast = (msg) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 4000);
  };

  // ── Data Fetching ──────────────────────────────────────────────────────────
  const fetchGlobalData = async () => {
    try {
      const [commRes, sumRes, statsRes, hmRes, corrRes, anomRes, spikeRes, multiRes] = await Promise.all([
        fetch('/api/commodities').then((r) => r.json()),
        fetch('/api/summary').then((r) => r.json()),
        fetch('/api/stats').then((r) => r.json()),
        fetch('/api/volatility-heatmap').then((r) => r.json()),
        fetch('/api/correlation').then((r) => r.json()),
        fetch('/api/anomalies?limit=50').then((r) => r.json()),
        fetch('/api/spikes?limit=50').then((r) => r.json()),
        fetch('/api/market-multi').then((r) => r.json()),
      ]);

      setCommodities(commRes || []);
      setSummaryList(sumRes || []);
      setStats(statsRes || null);
      setHeatmapData(hmRes || { commodities: [], weeks: [], values: [] });
      setCorrelationData(corrRes || { labels: [], matrix: [] });
      setAnomalies(anomRes || []);
      setSpikes(spikeRes || []);
      setMultiMarket(multiRes || { dates: [], series: {} });

      if (commRes && commRes.length > 0 && !selectedCommodity) {
        setSelectedCommodity(commRes[0]);
      }
    } catch (err) {
      console.error('Error fetching global data:', err);
    }
  };

  const fetchCommodityPrices = async (comm) => {
    if (!comm) return;
    try {
      const res = await fetch(`/api/prices?commodity=${encodeURIComponent(comm)}`);
      const data = await res.json();
      setPrices(data || []);
    } catch (err) {
      console.error('Error fetching commodity prices:', err);
    }
  };

  useEffect(() => {
    fetchGlobalData();
  }, []);

  useEffect(() => {
    if (selectedCommodity) {
      fetchCommodityPrices(selectedCommodity);
    }
  }, [selectedCommodity]);

  // ── Re-run Pipeline ────────────────────────────────────────────────────────
  const handleRunPipeline = async () => {
    setIsRefreshing(true);
    showToast('Running automated pipeline (Ingestion → Cleaning → ML)...');
    try {
      const res = await fetch('/api/pipeline/run', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        showToast('Pipeline completed! Reloaded 3,780 records.');
        await fetchGlobalData();
        await fetchCommodityPrices(selectedCommodity);
      } else {
        showToast('Pipeline finished with warning: ' + (data.message || 'Check terminal'));
      }
    } catch (err) {
      showToast('Error triggering pipeline: ' + err.message);
    } finally {
      setIsRefreshing(false);
    }
  };

  // ── Filtered Commodity Prices ──────────────────────────────────────────────
  const filteredPrices = useMemo(() => {
    if (!prices || prices.length === 0) return [];
    let list = [...prices];

    // Filter by Date Preset
    if (datePreset !== 'ALL' && list.length > 0) {
      const lastDate = new Date(list[list.length - 1].timestamp);
      let cutoff = new Date(lastDate);
      if (datePreset === '1M') cutoff.setMonth(cutoff.getMonth() - 1);
      else if (datePreset === '3M') cutoff.setMonth(cutoff.getMonth() - 3);
      else if (datePreset === '6M') cutoff.setMonth(cutoff.getMonth() - 6);
      else if (datePreset === '1Y') cutoff.setFullYear(cutoff.getFullYear() - 1);

      list = list.filter((p) => new Date(p.timestamp) >= cutoff);
    }

    // Filter by Volatility Class
    if (volFilter !== 'ALL') {
      list = list.filter((p) => p.volatility_class === volFilter);
    }

    // Filter by Anomalies Only
    if (showAnomaliesOnly) {
      list = list.filter((p) => p.is_anomaly);
    }

    return list;
  }, [prices, datePreset, volFilter, showAnomaliesOnly]);

  // ── Active Commodity Summary ───────────────────────────────────────────────
  const activeSummary = useMemo(() => {
    return summaryList.find((s) => s.commodity === selectedCommodity) || null;
  }, [summaryList, selectedCommodity]);

  const latestPriceRecord = useMemo(() => {
    if (prices.length === 0) return null;
    return prices[prices.length - 1];
  }, [prices]);

  // ── Badge Helpers ──────────────────────────────────────────────────────────
  const getBadgeClass = (category) => {
    switch (category) {
      case 'LOW': return 'badge-low';
      case 'MODERATE': return 'badge-moderate';
      case 'HIGH': return 'badge-high';
      case 'EXTREME': return 'badge-extreme';
      default: return 'badge-low';
    }
  };

  // ── Chart 1: Price & Technicals Config ─────────────────────────────────────
  const priceChartData = useMemo(() => {
    const labels = filteredPrices.map((p) => p.timestamp);
    const closePrices = filteredPrices.map((p) => p.close);
    const ma7 = filteredPrices.map((p) => p.rolling_mean_7 || null);
    const ma30 = filteredPrices.map((p) => p.rolling_mean_30 || null);

    // Anomalies marker points
    const anomalyPoints = filteredPrices.map((p) => (p.is_anomaly ? p.close : null));

    return {
      labels,
      datasets: [
        {
          type: 'line',
          label: `${selectedCommodity} Price ($)`,
          data: closePrices,
          borderColor: '#A15C1E',
          borderWidth: 2,
          backgroundColor: 'rgba(161, 92, 30, 0.08)',
          fill: true,
          tension: 0.15,
          pointRadius: 0,
          pointHoverRadius: 5,
        },
        {
          type: 'line',
          label: '7-Day SMA',
          data: ma7,
          borderColor: '#334155',
          borderWidth: 1.5,
          borderDash: [4, 4],
          fill: false,
          tension: 0.1,
          pointRadius: 0,
        },
        {
          type: 'line',
          label: '30-Day SMA',
          data: ma30,
          borderColor: '#0D9488',
          borderWidth: 1.5,
          borderDash: [6, 4],
          fill: false,
          tension: 0.1,
          pointRadius: 0,
        },
        {
          type: 'line',
          label: 'Unusual Movement (Anomaly)',
          data: anomalyPoints,
          borderColor: '#DC2626',
          backgroundColor: '#DC2626',
          pointStyle: 'triangle',
          pointRadius: 6,
          pointHoverRadius: 9,
          showLine: false,
        },
      ],
    };
  }, [filteredPrices, selectedCommodity]);

  // ── Chart 2: Multi-Horizon Volatility Config ───────────────────────────────
  const volatilityChartData = useMemo(() => {
    const labels = filteredPrices.map((p) => p.timestamp);
    const vol7 = filteredPrices.map((p) => (p.volatility_7_ann ? p.volatility_7_ann * 100 : null));
    const vol20 = filteredPrices.map((p) => (p.volatility_20_ann ? p.volatility_20_ann * 100 : null));
    const vol30 = filteredPrices.map((p) => (p.volatility_30_ann ? p.volatility_30_ann * 100 : null));
    const spikePoints = filteredPrices.map((p) => (p.is_volatility_spike && p.volatility_7_ann ? p.volatility_7_ann * 100 : null));

    return {
      labels,
      datasets: [
        {
          label: '7-Day Volatility (Ann. %)',
          data: vol7,
          borderColor: '#B45309',
          borderWidth: 2,
          tension: 0.2,
          pointRadius: 0,
        },
        {
          label: '20-Day Volatility (Ann. %)',
          data: vol20,
          borderColor: '#D97706',
          borderWidth: 1.7,
          tension: 0.2,
          pointRadius: 0,
        },
        {
          label: '30-Day Volatility (Ann. %)',
          data: vol30,
          borderColor: '#44403C',
          borderWidth: 1.7,
          tension: 0.2,
          pointRadius: 0,
        },
        {
          label: 'Volatility Spike Alert',
          data: spikePoints,
          borderColor: '#991B1B',
          backgroundColor: '#991B1B',
          pointStyle: 'star',
          pointRadius: 7,
          pointHoverRadius: 10,
          showLine: false,
        },
      ],
    };
  }, [filteredPrices]);

  // ── Chart 3: Multi-Commodity Comparison Config ─────────────────────────────
  const multiComparisonChartData = useMemo(() => {
    const dates = multiMarket.dates || [];
    const colors = {
      'Brent Crude': '#B45309',
      'Crude Oil': '#C2410C',
      'Gold': '#D97706',
      'Silver': '#64748B',
      'Copper': '#0D9488',
      'Natural Gas': '#4338CA',
      'Corn': '#15803D',
    };

    const datasets = Object.keys(multiMarket.series || {}).map((comm) => ({
      label: comm,
      data: multiMarket.series[comm],
      borderColor: colors[comm] || '#A15C1E',
      borderWidth: comm === selectedCommodity ? 2.8 : 1.4,
      pointRadius: 0,
      tension: 0.15,
    }));

    return { labels: dates, datasets };
  }, [multiMarket, selectedCommodity]);

  const defaultChartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: theme === 'caramel' ? '#44403C' : '#334155',
          font: { family: 'Plus Jakarta Sans', size: 12, weight: '600' },
          boxWidth: 12,
          padding: 16,
        },
      },
      tooltip: {
        backgroundColor: '#FFFFFF',
        titleColor: '#1C1917',
        bodyColor: '#44403C',
        borderColor: '#E8E2D8',
        borderWidth: 1,
        padding: 12,
        titleFont: { family: 'JetBrains Mono', size: 13, weight: 'bold' },
        bodyFont: { family: 'JetBrains Mono', size: 12 },
      },
    },
    scales: {
      x: {
        grid: { color: theme === 'caramel' ? '#EFEAE1' : '#E9ECEF' },
        ticks: { color: '#78716C', maxTicksLimit: 12, font: { family: 'JetBrains Mono', size: 11 } },
      },
      y: {
        grid: { color: theme === 'caramel' ? '#EFEAE1' : '#E9ECEF' },
        ticks: { color: '#78716C', font: { family: 'JetBrains Mono', size: 11 } },
      },
    },
  }), [theme]);

  // ── Heatmap Color Calculation ──────────────────────────────────────────────
  const getHeatmapColor = (val) => {
    if (val === null || val === undefined) return '#FAF7F2';
    if (val < 18) return '#E2ECE9';
    if (val < 28) return '#FEF3C7';
    if (val < 38) return '#FDE68A';
    if (val < 48) return '#F59E0B';
    if (val < 60) return '#D97706';
    return '#C2410C';
  };

  // ── Correlation Color Calculation ──────────────────────────────────────────
  const getCorrColor = (val) => {
    if (val === null || val === undefined) return 'transparent';
    if (val === 1) return 'rgba(161, 92, 30, 0.35)';
    if (val > 0.5) return 'rgba(180, 83, 9, 0.22)';
    if (val > 0) return 'rgba(217, 119, 6, 0.10)';
    if (val > -0.5) return 'rgba(153, 27, 27, 0.10)';
    return 'rgba(153, 27, 27, 0.25)';
  };

  // ── CSV Export ─────────────────────────────────────────────────────────────
  const exportCSV = (data, filename) => {
    if (!data || data.length === 0) return;
    const headers = Object.keys(data[0]).join(',');
    const rows = data.map((obj) => Object.values(obj).join(',')).join('\n');
    const blob = new Blob([`${headers}\n${rows}`], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="dashboard-container">
      {/* ── Top Navigation Bar ── */}
      <header className="navbar">
        <div className="nav-brand">
          <div className="brand-icon">
            <Activity size={22} />
          </div>
          <div>
            <div className="brand-title">
              Commodity Volatility Monitor
              <span className="badge badge-moderate" style={{ fontSize: '0.65rem' }}>PRO</span>
            </div>
            <div className="brand-subtitle">
              Automated Data Pipeline &amp; Real-Time Risk Intelligence Engine
            </div>
          </div>
        </div>

        <div className="nav-actions">
          <div className="system-badge">
            <div className="pulse-dot"></div>
            <span>LIVE PIPELINE</span>
            {stats && (
              <span style={{ color: 'var(--text-muted)', marginLeft: 4 }}>
                ({stats.total_records.toLocaleString()} records)
              </span>
            )}
          </div>

          <button
            className="theme-toggle-btn"
            onClick={toggleTheme}
            title={theme === 'caramel' ? 'Switch to Crisp Institutional White' : 'Switch to Warm Editorial Caramel'}
          >
            {theme === 'caramel' ? <Sun size={15} /> : <Coffee size={15} />}
            <span>{theme === 'caramel' ? 'Caramel & Cream' : 'Crisp White'}</span>
          </button>

          <button
            className="btn-secondary"
            onClick={() => setPipelineModalOpen(true)}
            title="Inspect Data Pipeline Architecture"
          >
            <Layers size={16} />
            <span>Architecture</span>
          </button>

          <button
            className="btn-primary"
            onClick={handleRunPipeline}
            disabled={isRefreshing}
            title="Trigger Ingestion & Machine Learning Pipeline"
          >
            <RefreshCw size={16} className={isRefreshing ? 'animate-spin' : ''} />
            <span>{isRefreshing ? 'Computing...' : 'Sync & Re-run'}</span>
          </button>
        </div>
      </header>

      {/* ── All Commodities Ticker / Strip ── */}
      <section className="commodity-strip">
        {summaryList.map((item) => {
          const isSelected = item.commodity === selectedCommodity;
          const ret = item.daily_return || 0;
          const isUp = ret >= 0;
          return (
            <div
              key={item.commodity}
              className={`commodity-card ${isSelected ? 'active' : ''}`}
              onClick={() => setSelectedCommodity(item.commodity)}
            >
              <div className="comm-header">
                <span className="comm-name">{item.commodity}</span>
                <span className={`badge ${getBadgeClass(item.volatility_class)}`}>
                  {item.volatility_class}
                </span>
              </div>
              <div className="comm-price">${Number(item.latest_price).toFixed(2)}</div>
              <div className="comm-footer">
                <span className={`change-pill ${isUp ? 'up' : 'down'}`}>
                  {isUp ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                  {isUp ? '+' : ''}{(ret * 100).toFixed(2)}%
                </span>
                <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                  Vol: {(item.volatility_7_ann * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          );
        })}
      </section>

      {/* ── Filters & Time Horizon Controls ── */}
      <section className="controls-bar">
        <div className="filter-group">
          <span className="filter-label">Asset:</span>
          <select
            className="filter-select"
            value={selectedCommodity}
            onChange={(e) => setSelectedCommodity(e.target.value)}
          >
            {commodities.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <span className="filter-label">Horizon:</span>
          <div className="date-presets">
            {['1M', '3M', '6M', '1Y', 'ALL'].map((preset) => (
              <button
                key={preset}
                className={`date-btn ${datePreset === preset ? 'active' : ''}`}
                onClick={() => setDatePreset(preset)}
              >
                {preset}
              </button>
            ))}
          </div>
        </div>

        <div className="filter-group">
          <span className="filter-label">Volatility Regimes:</span>
          <select
            className="filter-select"
            value={volFilter}
            onChange={(e) => setVolFilter(e.target.value)}
          >
            <option value="ALL">All Regimes</option>
            <option value="LOW">Low (&lt; 20%)</option>
            <option value="MODERATE">Moderate (20 - 35%)</option>
            <option value="HIGH">High (35 - 50%)</option>
            <option value="EXTREME">Extreme (&gt; 50%)</option>
          </select>
        </div>

        <div className="filter-group">
          <label className="toggle-container">
            <input
              type="checkbox"
              checked={showAnomaliesOnly}
              onChange={(e) => setShowAnomaliesOnly(e.target.checked)}
            />
            <span>Filter Anomalies Only</span>
          </label>
        </div>
      </section>

      {/* ── Selected Commodity Key Metrics ── */}
      {latestPriceRecord && (
        <section className="kpi-grid">
          <div className="kpi-box">
            <div className="kpi-top">
              <span className="kpi-title">{selectedCommodity} Spot Price</span>
              <div className="kpi-icon"><Activity size={18} /></div>
            </div>
            <div className="kpi-main-val">${Number(latestPriceRecord.close).toFixed(2)}</div>
            <div className="kpi-subtext">
              <span style={{ color: latestPriceRecord.daily_return >= 0 ? 'var(--gain-color)' : 'var(--loss-color)', fontWeight: 600 }}>
                {latestPriceRecord.daily_return >= 0 ? '+' : ''}
                {(Number(latestPriceRecord.daily_return || 0) * 100).toFixed(2)}%
              </span>
              <span>24h session change</span>
            </div>
          </div>

          <div className="kpi-box">
            <div className="kpi-top">
              <span className="kpi-title">7-Day Realized Volatility</span>
              <div className="kpi-icon"><Zap size={18} /></div>
            </div>
            <div className="kpi-main-val">
              {(Number(latestPriceRecord.volatility_7_ann || 0) * 100).toFixed(2)}%
            </div>
            <div className="kpi-subtext">
              <span className={`badge ${getBadgeClass(latestPriceRecord.volatility_class)}`}>
                {latestPriceRecord.volatility_class}
              </span>
              <span>Annualized standard dev</span>
            </div>
          </div>

          <div className="kpi-box">
            <div className="kpi-top">
              <span className="kpi-title">30-Day Volatility Ratio</span>
              <div className="kpi-icon"><TrendingUp size={18} /></div>
            </div>
            <div className="kpi-main-val">
              {latestPriceRecord.volatility_ratio ? `${Number(latestPriceRecord.volatility_ratio).toFixed(2)}×` : '1.00×'}
            </div>
            <div className="kpi-subtext">
              <span>Historical Avg: {(Number(latestPriceRecord.historical_avg_vol || 0) * 100).toFixed(1)}%</span>
            </div>
          </div>

          <div className="kpi-box">
            <div className="kpi-top">
              <span className="kpi-title">ML Signal Status</span>
              <div className="kpi-icon"><AlertTriangle size={18} /></div>
            </div>
            <div className="kpi-main-val" style={{ fontSize: '1.45rem' }}>
              {latestPriceRecord.is_volatility_spike ? (
                <span style={{ color: 'var(--loss-color)', fontWeight: 600 }}>⚠️ Volatility Spike</span>
              ) : latestPriceRecord.is_anomaly ? (
                <span style={{ color: 'var(--accent-amber)', fontWeight: 600 }}>⚡ Anomaly Alert</span>
              ) : (
                <span style={{ color: 'var(--gain-color)', fontWeight: 600 }}>✓ Market Normal</span>
              )}
            </div>
            <div className="kpi-subtext">
              <span>Isolation Forest anomaly score: {Number(latestPriceRecord.anomaly_score || 0).toFixed(4)}</span>
            </div>
          </div>
        </section>
      )}

      {/* ── Main Interactive Chart Hub ── */}
      <section className="chart-panel">
        <div className="panel-header">
          <div className="panel-title-area">
            <h2>Market Dynamics &amp; Risk Analytics</h2>
            <p>
              Displaying {filteredPrices.length} records for {selectedCommodity} across selected timeframe
            </p>
          </div>

          <div className="tab-nav">
            <button
              className={`tab-btn ${activeTab === 'price' ? 'active' : ''}`}
              onClick={() => setActiveTab('price')}
            >
              <Activity size={15} />
              <span>Price &amp; Anomalies</span>
            </button>
            <button
              className={`tab-btn ${activeTab === 'volatility' ? 'active' : ''}`}
              onClick={() => setActiveTab('volatility')}
            >
              <Zap size={15} />
              <span>Volatility Structure</span>
            </button>
            <button
              className={`tab-btn ${activeTab === 'multi' ? 'active' : ''}`}
              onClick={() => setActiveTab('multi')}
            >
              <TrendingUp size={15} />
              <span>Multi-Asset Relative</span>
            </button>
          </div>
        </div>

        <div className="chart-wrapper">
          {activeTab === 'price' && <Line data={priceChartData} options={defaultChartOptions} />}
          {activeTab === 'volatility' && <Line data={volatilityChartData} options={defaultChartOptions} />}
          {activeTab === 'multi' && <Line data={multiComparisonChartData} options={defaultChartOptions} />}
        </div>
      </section>

      {/* ── Cross-Asset Intelligence Grid (Heatmap & Correlation) ── */}
      <section className="intelligence-grid">
        {/* Volatility Heatmap */}
        <div className="card-box">
          <div className="box-header">
            <div>
              <div className="box-title">Weekly Volatility Heatmap</div>
              <div className="box-subtitle">Cross-commodity 7-day annualized volatility over calendar weeks</div>
            </div>
            <span className="badge badge-low">All Commodities</span>
          </div>

          <div className="heatmap-container">
            <table className="heatmap-table">
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', color: 'var(--text-muted)', paddingBottom: '0.4rem' }}>
                    Commodity
                  </th>
                  {heatmapData.weeks.filter((_, idx) => idx % 4 === 0).map((wk) => (
                    <th
                      key={wk}
                      style={{
                        color: 'var(--text-muted)',
                        fontSize: '0.68rem',
                        fontWeight: 500,
                        paddingBottom: '0.4rem',
                      }}
                    >
                      {wk.slice(5)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {heatmapData.commodities.map((comm, commIdx) => (
                  <tr key={comm}>
                    <td style={{ fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap', paddingRight: '0.5rem' }}>
                      {comm}
                    </td>
                    {heatmapData.weeks.filter((_, idx) => idx % 4 === 0).map((wk, sampleIdx) => {
                      // Map back to original index
                      const originalIdx = sampleIdx * 4;
                      const val = heatmapData.values[commIdx]?.[originalIdx];
                      return (
                        <td
                          key={wk}
                          className="heatmap-cell"
                          style={{ backgroundColor: getHeatmapColor(val) }}
                          title={`${comm} (${wk}): ${val !== null ? val.toFixed(1) + '%' : 'N/A'}`}
                        />
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="heatmap-legend">
            <span>Low (&lt;20%)</span>
            <div className="legend-bar"></div>
            <span>Extreme (&gt;50%)</span>
          </div>
        </div>

        {/* Return Correlation Matrix */}
        <div className="card-box">
          <div className="box-header">
            <div>
              <div className="box-title">Return Correlation Matrix</div>
              <div className="box-subtitle">Pearson cross-asset return dependencies</div>
            </div>
            <span className="badge badge-moderate">Inter-Asset</span>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table className="corr-table">
              <thead>
                <tr>
                  <th></th>
                  {correlationData.labels.map((lbl) => (
                    <th key={lbl}>{lbl}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {correlationData.labels.map((rowLbl, rIdx) => (
                  <tr key={rowLbl}>
                    <th style={{ textAlign: 'left' }}>{rowLbl}</th>
                    {correlationData.labels.map((colLbl, cIdx) => {
                      const val = correlationData.matrix[rIdx]?.[cIdx];
                      return (
                        <td
                          key={colLbl}
                          className="corr-cell"
                          style={{ backgroundColor: getCorrColor(val) }}
                          title={`${rowLbl} vs ${colLbl}: ${val}`}
                        >
                          {val !== null && val !== undefined ? val.toFixed(2) : '-'}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            💡 <strong>Insight:</strong> Diversification benefit is highest between negatively or weakly correlated assets (e.g. Natural Gas vs Crude Oil).
          </div>
        </div>
      </section>

      {/* ── Anomaly & Volatility Spikes Feeds ── */}
      <section className="signals-section">
        {/* Unusual Movements (Isolation Forest) */}
        <div className="card-box">
          <div className="box-header">
            <div>
              <div className="box-title">Unusual Market Movements</div>
              <div className="box-subtitle">Isolation Forest ML detection on returns &amp; volume</div>
            </div>
            <button
              className="btn-secondary"
              style={{ padding: '0.3rem 0.65rem', fontSize: '0.75rem' }}
              onClick={() => exportCSV(anomalies, 'anomalies_report')}
            >
              <Download size={14} /> CSV
            </button>
          </div>

          <div className="signal-table-wrap">
            <table className="signal-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Commodity</th>
                  <th>Close</th>
                  <th>Return</th>
                  <th>Anomaly Score</th>
                </tr>
              </thead>
              <tbody>
                {anomalies
                  .filter((a) => !searchAnomaly || a.commodity.toLowerCase().includes(searchAnomaly.toLowerCase()))
                  .slice(0, 15)
                  .map((a, i) => (
                    <tr key={i}>
                      <td>{a.timestamp}</td>
                      <td style={{ fontWeight: 600 }}>{a.commodity}</td>
                      <td>${Number(a.close).toFixed(2)}</td>
                      <td style={{ color: a.daily_return >= 0 ? 'var(--gain-color)' : 'var(--loss-color)', fontWeight: 600 }}>
                        {a.daily_return >= 0 ? '+' : ''}{(a.daily_return * 100).toFixed(2)}%
                      </td>
                      <td style={{ color: 'var(--accent-caramel)', fontWeight: 600 }}>
                        {Number(a.anomaly_score).toFixed(4)}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Volatility Spikes */}
        <div className="card-box">
          <div className="box-header">
            <div>
              <div className="box-title">Volatility Spikes</div>
              <div className="box-subtitle">Realized 7d volatility &gt; 1.75× historical average</div>
            </div>
            <button
              className="btn-secondary"
              style={{ padding: '0.3rem 0.65rem', fontSize: '0.75rem' }}
              onClick={() => exportCSV(spikes, 'volatility_spikes_report')}
            >
              <Download size={14} /> CSV
            </button>
          </div>

          <div className="signal-table-wrap">
            <table className="signal-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Commodity</th>
                  <th>Close</th>
                  <th>Vol 7d</th>
                  <th>Ratio</th>
                </tr>
              </thead>
              <tbody>
                {spikes
                  .filter((s) => !searchSpike || s.commodity.toLowerCase().includes(searchSpike.toLowerCase()))
                  .slice(0, 15)
                  .map((s, i) => (
                    <tr key={i}>
                      <td>{s.timestamp}</td>
                      <td style={{ fontWeight: 600 }}>{s.commodity}</td>
                      <td>${Number(s.close).toFixed(2)}</td>
                      <td style={{ color: 'var(--loss-color)', fontWeight: 600 }}>
                        {(s.volatility_7_ann * 100).toFixed(1)}%
                      </td>
                      <td>
                        <span className="badge badge-extreme" style={{ fontSize: '0.68rem' }}>
                          {Number(s.volatility_ratio).toFixed(2)}×
                        </span>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="footer">
        <div>
          Commodity Price &amp; Volatility Mapping System — Enterprise Data Pipeline v1.0
        </div>
        <div>
          Swappable Architecture: Mock CSV ⇄ Live Financial API Ready
        </div>
      </footer>

      {/* ── Toast Notification ── */}
      {toastMessage && (
        <div className="toast-notice">
          <CheckCircle size={16} color="var(--gain-color)" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* ── Architecture Modal ── */}
      {pipelineModalOpen && (
        <div className="modal-overlay" onClick={() => setPipelineModalOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setPipelineModalOpen(false)}>
              <X size={18} />
            </button>
            <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.4rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              System Pipeline Architecture
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.25rem' }}>
              Modular decoupled data science engineering:
            </p>

            <div className="pipeline-diagram">
              <div className="pipeline-step">
                <div className="step-num">1</div>
                <div className="step-info">
                  <h4>Data Ingestion Layer (Swappable)</h4>
                  <p>Abstract BaseLoader allows swapping mock CSV with real Yahoo Finance / AlphaVantage API.</p>
                </div>
              </div>

              <div className="pipeline-step">
                <div className="step-num">2</div>
                <div className="step-info">
                  <h4>Data Validation Layer</h4>
                  <p>Enforces schema rules, date formats, non-empty criteria, and numeric validation before ingestion.</p>
                </div>
              </div>

              <div className="pipeline-step">
                <div className="step-num">3</div>
                <div className="step-info">
                  <h4>Cleaning &amp; Preprocessing</h4>
                  <p>Handles missing values with forward-fill + interpolation, flags price outliers without loss.</p>
                </div>
              </div>

              <div className="pipeline-step">
                <div className="step-num">4</div>
                <div className="step-info">
                  <h4>Feature Engineering</h4>
                  <p>Computes daily simple returns, log returns, and rolling 7-day and 30-day statistical moments.</p>
                </div>
              </div>

              <div className="pipeline-step">
                <div className="step-num">5</div>
                <div className="step-info">
                  <h4>Volatility Engine</h4>
                  <p>Calculates annualized multi-horizon volatility (7d, 20d, 30d) and assigns risk classifications.</p>
                </div>
              </div>

              <div className="pipeline-step">
                <div className="step-num">6</div>
                <div className="step-info">
                  <h4>Machine Learning Anomaly Detection</h4>
                  <p>Scikit-Learn Isolation Forest detects multidimensional price, volatility, and volume anomalies.</p>
                </div>
              </div>

              <div className="pipeline-step">
                <div className="step-num">7</div>
                <div className="step-info">
                  <h4>FastAPI REST API</h4>
                  <p>High-throughput asynchronous JSON backend delivering optimized analytical endpoints.</p>
                </div>
              </div>

              <div className="pipeline-step">
                <div className="step-num">8</div>
                <div className="step-info">
                  <h4>React Real-Time Dashboard</h4>
                  <p>Vite + React interactive interface featuring real-time charts, heatmaps, and signal monitors.</p>
                </div>
              </div>
            </div>

            <div style={{ marginTop: '1.5rem', display: 'flex', justifyContent: 'flex-end' }}>
              <button
                className="btn-primary"
                onClick={() => {
                  setPipelineModalOpen(false);
                  handleRunPipeline();
                }}
              >
                <Play size={16} /> Run Full Pipeline Now
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
