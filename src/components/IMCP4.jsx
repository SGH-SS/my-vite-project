/**
 * IMC Prosperity 4 Dashboard
 * 
 * Comprehensive visualization of tutorial round data, strategy signals,
 * and open-source strategy analysis for the P4 competition.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter,
  ComposedChart, Area, ReferenceLine, Cell
} from 'recharts';

const API = 'http://localhost:8000/api/imcp4';

const formatNum = (v, decimals = 2) => {
  if (v === null || v === undefined) return 'N/A';
  return typeof v === 'number' ? v.toFixed(decimals) : v;
};

const formatPct = (v) => {
  if (v === null || v === undefined) return 'N/A';
  return (v * 100).toFixed(3) + '%';
};

// ============================================================================
// STAT CARD
// ============================================================================
const StatCard = ({ label, value, sub, color = 'blue', isDarkMode }) => (
  <div className={`p-4 rounded-lg border transition-colors ${
    isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
  }`}>
    <div className={`text-xs font-medium uppercase tracking-wide ${
      isDarkMode ? 'text-gray-400' : 'text-gray-500'
    }`}>{label}</div>
    <div className={`text-2xl font-bold mt-1 text-${color}-500`}>{value}</div>
    {sub && <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{sub}</div>}
  </div>
);

// ============================================================================
// SECTION HEADER
// ============================================================================
const SectionHeader = ({ title, subtitle, isDarkMode }) => (
  <div className="mb-4">
    <h2 className={`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{title}</h2>
    {subtitle && <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>{subtitle}</p>}
  </div>
);

// ============================================================================
// TAB SELECTOR
// ============================================================================
const TabSelector = ({ tabs, active, onChange, isDarkMode }) => (
  <div className={`flex rounded-lg p-1 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
    {tabs.map(tab => (
      <button
        key={tab.id}
        onClick={() => onChange(tab.id)}
        className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${
          active === tab.id
            ? isDarkMode
              ? 'bg-blue-600 text-white shadow-sm'
              : 'bg-white text-blue-600 shadow-sm'
            : isDarkMode
              ? 'text-gray-300 hover:text-white'
              : 'text-gray-600 hover:text-gray-900'
        }`}
      >{tab.label}</button>
    ))}
  </div>
);

// ============================================================================
// OVERVIEW TAB
// ============================================================================
const OverviewTab = ({ overview, dailySummary, isDarkMode }) => {
  if (!overview) return <div className="text-center py-8">Loading...</div>;

  return (
    <div className="space-y-6">
      {/* Database Stats */}
      <div>
        <SectionHeader title="Database Status" subtitle="imcp4 schema in TimescaleDB" isDarkMode={isDarkMode} />
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {Object.entries(overview.tables).map(([table, count]) => (
            <StatCard
              key={table}
              label={table.replace(/_/g, ' ')}
              value={count.toLocaleString()}
              color="blue"
              isDarkMode={isDarkMode}
            />
          ))}
        </div>
      </div>

      {/* Products */}
      <div>
        <SectionHeader title="Products" subtitle="Tutorial Round tradable goods" isDarkMode={isDarkMode} />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {overview.products.map(p => (
            <div key={p.symbol} className={`p-5 rounded-lg border ${
              isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
            }`}>
              <div className="flex items-center justify-between">
                <div>
                  <span className={`text-lg font-bold ${
                    p.product_type === 'stable' ? 'text-emerald-500' : 'text-red-500'
                  }`}>{p.symbol}</span>
                  <span className={`ml-2 text-xs px-2 py-0.5 rounded-full ${
                    p.product_type === 'stable'
                      ? 'bg-emerald-900/30 text-emerald-400'
                      : 'bg-red-900/30 text-red-400'
                  }`}>{p.product_type}</span>
                </div>
                <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  Limit: {p.position_limit}
                </div>
              </div>
              <div className={`text-sm mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                {p.description}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Daily Summary Table */}
      {dailySummary && dailySummary.length > 0 && (
        <div>
          <SectionHeader title="Daily Summary" subtitle="Per-product per-day statistics" isDarkMode={isDarkMode} />
          <div className={`overflow-x-auto rounded-lg border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
            <table className="w-full text-sm">
              <thead>
                <tr className={isDarkMode ? 'bg-gray-800' : 'bg-gray-50'}>
                  {['Product', 'Day', 'Open', 'Close', 'High', 'Low', 'Range', 'Return', 'Spread', 'Trades', 'Volume', 'Volatility'].map(h => (
                    <th key={h} className={`px-3 py-2 text-left font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dailySummary.map((row, i) => (
                  <tr key={i} className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                    <td className={`px-3 py-2 font-medium ${
                      row.product === 'EMERALDS' ? 'text-emerald-500' : 'text-red-500'
                    }`}>{row.product}</td>
                    <td className={`px-3 py-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{row.day}</td>
                    <td className={`px-3 py-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{formatNum(row.open_mid, 1)}</td>
                    <td className={`px-3 py-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{formatNum(row.close_mid, 1)}</td>
                    <td className={`px-3 py-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{formatNum(row.high_mid, 1)}</td>
                    <td className={`px-3 py-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{formatNum(row.low_mid, 1)}</td>
                    <td className={`px-3 py-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{formatNum(row.daily_range, 1)}</td>
                    <td className={`px-3 py-2 ${row.daily_return >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      {formatPct(row.daily_return)}
                    </td>
                    <td className={`px-3 py-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{formatNum(row.avg_spread, 1)}</td>
                    <td className={`px-3 py-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{row.total_trades}</td>
                    <td className={`px-3 py-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{row.total_volume}</td>
                    <td className={`px-3 py-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{formatPct(row.volatility)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Data Available */}
      <div>
        <SectionHeader title="Available Data" subtitle="Round/Day combinations loaded" isDarkMode={isDarkMode} />
        <div className="flex gap-2">
          {overview.round_days.map(rd => (
            <div key={`${rd.round}-${rd.day}`} className={`px-4 py-2 rounded-lg text-sm font-medium ${
              isDarkMode ? 'bg-gray-700 text-gray-200' : 'bg-gray-100 text-gray-700'
            }`}>
              Round {rd.round}, Day {rd.day}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// PRICE CHART TAB
// ============================================================================
const PriceChartTab = ({ isDarkMode, selectedProduct, selectedDay }) => {
  const [signals, setSignals] = useState([]);
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const ds = selectedProduct === 'EMERALDS' ? 10 : 5;
        const [sigRes, trRes] = await Promise.all([
          fetch(`${API}/strategy-signals/${selectedProduct}?day=${selectedDay}&downsample=${ds}`),
          fetch(`${API}/trades/${selectedProduct}?day=${selectedDay}`)
        ]);
        const sigData = await sigRes.json();
        const trData = await trRes.json();
        setSignals(sigData);
        setTrades(trData);
      } catch (e) {
        console.error(e);
      }
      setLoading(false);
    };
    load();
  }, [selectedProduct, selectedDay]);

  if (loading) return <div className="text-center py-12">Loading chart data...</div>;

  const tradeMarkers = trades.map(t => ({
    timestamp: t.timestamp,
    price: t.price,
    size: t.quantity,
    side: t.price > (signals.find(s => s.timestamp <= t.timestamp)?.mid_price || 0) ? 'buy' : 'sell'
  }));

  return (
    <div className="space-y-6">
      {/* Main Price Chart with EMAs */}
      <div className={`p-4 rounded-lg border ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
        <SectionHeader title={`${selectedProduct} - Day ${selectedDay}`} subtitle="Mid price with EMA overlays and trade markers" isDarkMode={isDarkMode} />
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={signals} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? '#374151' : '#e5e7eb'} />
            <XAxis dataKey="timestamp" tick={{ fontSize: 10, fill: isDarkMode ? '#9ca3af' : '#6b7280' }}
              tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} />
            <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10, fill: isDarkMode ? '#9ca3af' : '#6b7280' }} />
            <Tooltip contentStyle={{ backgroundColor: isDarkMode ? '#1f2937' : '#fff', border: 'none', borderRadius: '8px' }}
              labelFormatter={v => `Timestamp: ${v}`} />
            <Legend />
            <Line type="monotone" dataKey="mid_price" stroke="#3b82f6" dot={false} strokeWidth={1.5} name="Mid" />
            <Line type="monotone" dataKey="ema_10" stroke="#f59e0b" dot={false} strokeWidth={1} name="EMA-10" strokeDasharray="3 3" />
            <Line type="monotone" dataKey="ema_50" stroke="#ef4444" dot={false} strokeWidth={1} name="EMA-50" strokeDasharray="5 5" />
            {selectedProduct !== 'EMERALDS' && (
              <Line type="monotone" dataKey="wall_mid" stroke="#10b981" dot={false} strokeWidth={1} name="Wall Mid" />
            )}
            <Scatter data={tradeMarkers} dataKey="price" fill="#a855f7" name="Trades" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Book Imbalance */}
      <div className={`p-4 rounded-lg border ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
        <SectionHeader title="Book Imbalance" subtitle="(bid_depth - ask_depth) / total_depth — predictive signal for TOMATOES" isDarkMode={isDarkMode} />
        <ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={signals} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? '#374151' : '#e5e7eb'} />
            <XAxis dataKey="timestamp" tick={{ fontSize: 10, fill: isDarkMode ? '#9ca3af' : '#6b7280' }}
              tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} />
            <YAxis tick={{ fontSize: 10, fill: isDarkMode ? '#9ca3af' : '#6b7280' }} />
            <Tooltip contentStyle={{ backgroundColor: isDarkMode ? '#1f2937' : '#fff', border: 'none', borderRadius: '8px' }} />
            <ReferenceLine y={0} stroke={isDarkMode ? '#6b7280' : '#9ca3af'} />
            <Area type="monotone" dataKey="book_imbalance" fill="#3b82f640" stroke="#3b82f6" name="Imbalance" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Volatility + RSI */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className={`p-4 rounded-lg border ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
          <SectionHeader title="Realized Volatility (20-tick)" isDarkMode={isDarkMode} />
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={signals}>
              <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? '#374151' : '#e5e7eb'} />
              <XAxis dataKey="timestamp" tick={{ fontSize: 10, fill: isDarkMode ? '#9ca3af' : '#6b7280' }}
                tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} />
              <YAxis tick={{ fontSize: 10, fill: isDarkMode ? '#9ca3af' : '#6b7280' }} />
              <Tooltip contentStyle={{ backgroundColor: isDarkMode ? '#1f2937' : '#fff', border: 'none', borderRadius: '8px' }} />
              <Line type="monotone" dataKey="realized_vol_20" stroke="#f59e0b" dot={false} name="Vol-20" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className={`p-4 rounded-lg border ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
          <SectionHeader title="RSI-14" isDarkMode={isDarkMode} />
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={signals}>
              <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? '#374151' : '#e5e7eb'} />
              <XAxis dataKey="timestamp" tick={{ fontSize: 10, fill: isDarkMode ? '#9ca3af' : '#6b7280' }}
                tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: isDarkMode ? '#9ca3af' : '#6b7280' }} />
              <Tooltip contentStyle={{ backgroundColor: isDarkMode ? '#1f2937' : '#fff', border: 'none', borderRadius: '8px' }} />
              <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" label="Overbought" />
              <ReferenceLine y={30} stroke="#10b981" strokeDasharray="3 3" label="Oversold" />
              <Line type="monotone" dataKey="rsi_14" stroke="#8b5cf6" dot={false} name="RSI" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// MICROSTRUCTURE TAB
// ============================================================================
const MicrostructureTab = ({ isDarkMode, selectedProduct }) => {
  const [spreadDist, setSpreadDist] = useState([]);
  const [tradeSizeDist, setTradeSizeDist] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      const [sRes, tRes] = await Promise.all([
        fetch(`${API}/spread-distribution/${selectedProduct}`),
        fetch(`${API}/trade-size-distribution/${selectedProduct}`)
      ]);
      setSpreadDist(await sRes.json());
      setTradeSizeDist(await tRes.json());
      setLoading(false);
    };
    load();
  }, [selectedProduct]);

  if (loading) return <div className="text-center py-8">Loading...</div>;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Spread Distribution */}
        <div className={`p-4 rounded-lg border ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
          <SectionHeader title="Spread Distribution" subtitle="Frequency of bid-ask spread values" isDarkMode={isDarkMode} />
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={spreadDist}>
              <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? '#374151' : '#e5e7eb'} />
              <XAxis dataKey="spread" tick={{ fontSize: 11, fill: isDarkMode ? '#9ca3af' : '#6b7280' }} />
              <YAxis tick={{ fontSize: 11, fill: isDarkMode ? '#9ca3af' : '#6b7280' }} />
              <Tooltip contentStyle={{ backgroundColor: isDarkMode ? '#1f2937' : '#fff', border: 'none', borderRadius: '8px' }}
                formatter={(v, name) => [name === 'pct' ? `${v.toFixed(1)}%` : v, name]} />
              <Bar dataKey="pct" fill="#3b82f6" name="% of time" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Trade Size Distribution */}
        <div className={`p-4 rounded-lg border ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
          <SectionHeader title="Trade Size Distribution" subtitle="Bot trade quantity patterns" isDarkMode={isDarkMode} />
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={tradeSizeDist}>
              <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? '#374151' : '#e5e7eb'} />
              <XAxis dataKey="quantity" tick={{ fontSize: 11, fill: isDarkMode ? '#9ca3af' : '#6b7280' }} />
              <YAxis tick={{ fontSize: 11, fill: isDarkMode ? '#9ca3af' : '#6b7280' }} />
              <Tooltip contentStyle={{ backgroundColor: isDarkMode ? '#1f2937' : '#fff', border: 'none', borderRadius: '8px' }}
                formatter={(v, name) => [name === 'pct' ? `${v.toFixed(1)}%` : v, name]} />
              <Bar dataKey="pct" fill="#10b981" name="% of trades" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// STRATEGIES TAB
// ============================================================================
const StrategiesTab = ({ isDarkMode, repos }) => {
  if (!repos) return <div className="text-center py-8">Loading...</div>;

  return (
    <div className="space-y-4">
      <SectionHeader
        title="Open Source Strategy Repository"
        subtitle="Cloned repos from top P3 teams - analyzed and mapped to our system"
        isDarkMode={isDarkMode}
      />
      {repos.map((repo, i) => (
        <div key={i} className={`p-5 rounded-lg border ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
          <div className="flex items-center justify-between mb-3">
            <div>
              <span className={`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{repo.name}</span>
              <span className={`ml-3 text-xs px-2 py-1 rounded-full ${
                repo.placement.includes('2nd') ? 'bg-yellow-900/30 text-yellow-400' :
                repo.placement.includes('Tool') ? 'bg-blue-900/30 text-blue-400' :
                'bg-purple-900/30 text-purple-400'
              }`}>{repo.placement}</span>
            </div>
            <span className={`text-xs font-mono ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{repo.key_file}</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {repo.strategies.map((s, j) => (
              <div key={j} className={`flex items-start gap-2 text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                <span className="text-emerald-500 mt-0.5">&#9679;</span>
                <span>{s}</span>
              </div>
            ))}
          </div>
        </div>
      ))}

      {/* Key Insights Panel */}
      <div className={`p-5 rounded-lg border-2 ${isDarkMode ? 'border-blue-800 bg-blue-900/20' : 'border-blue-200 bg-blue-50'}`}>
        <SectionHeader title="Key Takeaways for P4" subtitle="Distilled from all repos + our analysis" isDarkMode={isDarkMode} />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          {[
            { title: 'EMERALDS = Rainforest Resin', desc: 'Fixed fair value at 10,000. Bot spread 16. Market make inside bots. ~40k XIRECs/day potential.' },
            { title: 'TOMATOES = Kelp', desc: 'Random walk, mean-reverting at short horizons. Book imbalance corr=0.59 with next return. Wall Mid as fair value.' },
            { title: 'Lag-1 Autocorrelation: -0.41', desc: 'Strong negative autocorrelation in TOMATOES returns = mean reversion. Trade reversals.' },
            { title: 'No Cross-Product Signal', desc: 'EMERALDS and TOMATOES are independent (corr ~ 0). Each must be traded on its own merit.' },
            { title: 'No Olivia in Tutorial', desc: 'No informed trader patterns detected. Expect Olivia-style bots to appear in Round 1.' },
            { title: 'Position Limit: 80', desc: 'Both products have 80-unit limit. Size positions carefully. Frankfurt used inventory-dependent quoting.' }
          ].map((item, i) => (
            <div key={i} className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <div className={`font-semibold ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>{item.title}</div>
              <div className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>{item.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// MAIN DASHBOARD
// ============================================================================
export default function IMCP4Dashboard({ isDarkMode: propDarkMode }) {
  const isDarkMode = propDarkMode ?? true;
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedProduct, setSelectedProduct] = useState('TOMATOES');
  const [selectedDay, setSelectedDay] = useState(-1);

  const [overview, setOverview] = useState(null);
  const [dailySummary, setDailySummary] = useState([]);
  const [repos, setRepos] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadInitial = async () => {
      try {
        const [ovRes, dsRes, rpRes] = await Promise.all([
          fetch(`${API}/overview`),
          fetch(`${API}/daily-summary`),
          fetch(`${API}/repos-summary`)
        ]);
        if (!ovRes.ok) throw new Error(`Overview: ${ovRes.status}`);
        setOverview(await ovRes.json());
        setDailySummary(await dsRes.json());
        const rpData = await rpRes.json();
        setRepos(rpData.repos);
      } catch (e) {
        setError(e.message);
        console.error(e);
      }
    };
    loadInitial();
  }, []);

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'charts', label: 'Price Charts' },
    { id: 'micro', label: 'Microstructure' },
    { id: 'strategies', label: 'Strategies' },
  ];

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="p-6 bg-red-900/30 border border-red-700 rounded-lg">
          <h2 className="text-lg font-bold text-red-400">Connection Error</h2>
          <p className="text-red-300 text-sm mt-1">{error}</p>
          <p className="text-red-400 text-xs mt-2">Ensure the backend is running: python backend/main.py</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      {/* Tab Navigation + Selectors */}
      <div className="flex items-center justify-between mb-6">
        <TabSelector tabs={tabs} active={activeTab} onChange={setActiveTab} isDarkMode={isDarkMode} />
        <div className="flex items-center gap-3">
          <select
            value={selectedProduct}
            onChange={e => setSelectedProduct(e.target.value)}
            className={`text-sm rounded-lg px-3 py-2 border ${
              isDarkMode ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-white border-gray-300 text-gray-800'
            }`}
          >
            <option value="TOMATOES">TOMATOES</option>
            <option value="EMERALDS">EMERALDS</option>
          </select>
          <select
            value={selectedDay}
            onChange={e => setSelectedDay(parseInt(e.target.value))}
            className={`text-sm rounded-lg px-3 py-2 border ${
              isDarkMode ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-white border-gray-300 text-gray-800'
            }`}
          >
            <option value={-1}>Day -1</option>
            <option value={-2}>Day -2</option>
          </select>
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && <OverviewTab overview={overview} dailySummary={dailySummary} isDarkMode={isDarkMode} />}
      {activeTab === 'charts' && <PriceChartTab isDarkMode={isDarkMode} selectedProduct={selectedProduct} selectedDay={selectedDay} />}
      {activeTab === 'micro' && <MicrostructureTab isDarkMode={isDarkMode} selectedProduct={selectedProduct} />}
      {activeTab === 'strategies' && <StrategiesTab isDarkMode={isDarkMode} repos={repos} />}
    </div>
  );
}
