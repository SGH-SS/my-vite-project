import { useEffect, useMemo, useState, useCallback } from 'react';
import PropTypes from 'prop-types';

// API base for pipeline endpoints
const API_BASE = 'http://localhost:8000/api/pipeline';

// Hook to fetch freshness status
function useFreshnessStatus() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchStatus = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/freshness-status`);
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || 'Failed to fetch status');
      setStatus(data);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    // Refresh every 5 minutes
    const interval = setInterval(fetchStatus, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  return { status, loading, error, refresh: fetchStatus };
}

function useScripts() {
  const [scripts, setScripts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    fetch(`${API_BASE}/scripts`)
      .then(r => r.json())
      .then(data => { if (mounted) setScripts(data || []); })
      .catch(err => { if (mounted) setError(String(err)); })
      .finally(() => { if (mounted) setLoading(false); });
    return () => { mounted = false; };
  }, []);

  return { scripts, loading, error };
}

function Button({ label, onClick, disabled, tone = 'primary' }) {
  const base = 'px-4 py-3 rounded-md font-semibold transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-offset-2';
  const tones = {
    primary: 'bg-blue-600 hover:bg-blue-700 text-white focus:ring-blue-500',
    neutral: 'bg-gray-200 hover:bg-gray-300 text-gray-900 focus:ring-gray-400',
    subtle: 'bg-gray-100 hover:bg-gray-200 text-gray-800 focus:ring-gray-300',
    success: 'bg-emerald-600 hover:bg-emerald-700 text-white focus:ring-emerald-500',
  };
  return (
    <button className={`${base} ${tones[tone]} disabled:opacity-50`} disabled={disabled} onClick={onClick}>
      {label}
    </button>
  );
}

Button.propTypes = {
  label: PropTypes.string.isRequired,
  onClick: PropTypes.func,
  disabled: PropTypes.bool,
  tone: PropTypes.string,
};

export default function PipelineDashboard({ isDarkMode = false }) {
  const { scripts, loading, error } = useScripts();
  const [busyKey, setBusyKey] = useState(null);
  const [toast, setToast] = useState(null);
  const [showRecentData, setShowRecentData] = useState(false);
  const [recentData, setRecentData] = useState(null);
  const [recentLoading, setRecentLoading] = useState(false);
  const [recentError, setRecentError] = useState(null);
  const [recentSymbol, setRecentSymbol] = useState(null);
  const [showRules, setShowRules] = useState(false);
  const { status: freshnessStatus, loading: freshnessLoading, error: freshnessError, refresh: refreshFreshness } = useFreshnessStatus();

  const primary = useMemo(() => {
    const byKey = Object.fromEntries((scripts || []).map(s => [s.key, s]));
    return [byKey.spy_full, byKey.es_full, byKey.eurusd_full].filter(Boolean);
  }, [scripts]);

  const vectorLaunchers = useMemo(() => {
    const byKey = Object.fromEntries((scripts || []).map(s => [s.key, s]));
    return [byKey.spy_vectors, byKey.es_vectors, byKey.eurusd_vectors].filter(Boolean);
  }, [scripts]);

  const runKey = async (key) => {
    try {
      setBusyKey(key);
      const res = await fetch(`${API_BASE}/run/${encodeURIComponent(key)}`, {
        method: 'POST'
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || 'Failed to start');
      setToast({ type: 'success', text: `${data.message} (pid ${data.pid})` });
    } catch (e) {
      setToast({ type: 'error', text: String(e) });
    } finally {
      setBusyKey(null);
      setTimeout(() => setToast(null), 5000);
    }
  };

  const runSingle = async (scriptName) => {
    try {
      setBusyKey(scriptName);
      const res = await fetch(`${API_BASE}/run-script`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scriptName })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || 'Failed to start');
      setToast({ type: 'success', text: `${data.message} (pid ${data.pid})` });
    } catch (e) {
      setToast({ type: 'error', text: String(e) });
    } finally {
      setBusyKey(null);
      setTimeout(() => setToast(null), 5000);
    }
  };

  const fetchRecentData = async (symbol) => {
    try {
      setRecentLoading(true);
      setRecentError(null);
      setRecentSymbol(symbol);
      const res = await fetch(`${API_BASE}/${symbol}-recent?limit=7`);
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || `Failed to fetch ${symbol.toUpperCase()} data`);
      setRecentData(data);
      setShowRecentData(true);
    } catch (e) {
      setRecentError(String(e));
      setToast({ type: 'error', text: `Failed to fetch ${symbol.toUpperCase()} data: ${e}` });
    } finally {
      setRecentLoading(false);
    }
  };

  // Sort tables by timeframe order (1d -> 4h -> 1h -> 30m -> 15m -> 5m -> 1m)
  const sortTablesByTimeframe = (tables) => {
    const timeframeOrder = ['1d', '4h', '1h', '30m', '15m', '5m', '1m'];
    return tables.sort((a, b) => {
      const aIndex = timeframeOrder.indexOf(a.timeframe);
      const bIndex = timeframeOrder.indexOf(b.timeframe);
      if (aIndex === -1 && bIndex === -1) return a.timeframe.localeCompare(b.timeframe);
      if (aIndex === -1) return 1;
      if (bIndex === -1) return -1;
      return aIndex - bIndex;
    });
  };

  // Curated single-table script names derived from the user's docs and screenshot
  const singleScripts = [
    // SPY
    '07_spy1d_full_sync.py',
    '12_spy4h_full_sync.py',
    '13_spy1h_full_sync.py',
    '14_spy30m_full_sync.py',
    '15_spy15m_full_sync.py',
    '16_spy5m_full_sync.py',
    '17_spy1m_full_sync.py',
    // ES (v2 series)
    '26_es1d_full_sync_v2.py',
    '28_es4h_full_sync_v2.py',
    '30_es1h_full_sync_v2.py',
    '31_es30m_full_sync_v2.py',
    '32_es15m_full_sync_v2.py',
    '33_es5m_full_sync_v2.py',
    '34_es1m_full_sync_v2.py',
    // EURUSD (v2 series)
    '27_eurusd1d_full_sync_v2.py',
    '29_eurusd4h_full_sync_v2.py',
    '35_eurusd1h_full_sync_v2.py',
    '36_eurusd30m_full_sync_v2.py',
    '37_eurusd15m_full_sync_v2.py',
    '38_eurusd5m_full_sync_v2.py',
    '39_eurusd1m_full_sync_v2.py',
  ];

  return (
    <div className={`min-h-[calc(100vh-6rem)] ${isDarkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'}`}>
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h2 className="text-2xl font-bold tracking-tight">Pipeline Dashboard</h2>
          <p className={`mt-1 text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
            Update fronttest tables via reproducible scripts launched in an Anaconda prompt.
          </p>
        </div>

        {/* Vector Generation Buttons */}
        {vectorLaunchers.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {vectorLaunchers.map(s => (
              <div key={s.key} className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} rounded-xl p-5 shadow-sm`}>
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold">🧮 {s.title}</h3>
                    <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{s.description}</p>
                  </div>
                  <Button 
                    label={busyKey === s.key ? 'Starting…' : `Run`}
                    onClick={() => runKey(s.key)}
                    disabled={busyKey !== null}
                    tone="primary"
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Utilities */}
        <div className="grid grid-cols-1 md:grid-cols-1 gap-4 mb-6">
          <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} rounded-xl p-5 shadow-sm`}>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold">🛠 Fill Fronttest Future</h3>
                <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  Fill empty 'future' values across all fronttest tables.
                </p>
              </div>
              <Button 
                label={busyKey === '44_fronttest_fill_future.py' ? 'Starting…' : 'Run'}
                onClick={() => runSingle('44_fronttest_fill_future.py')}
                disabled={busyKey !== null}
                tone="primary"
              />
            </div>
          </div>
        </div>

        {/* Big 3 Buttons */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {primary.map(s => (
            <div key={s.key} className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} rounded-xl p-5 shadow-sm`}>
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-lg font-semibold">{s.title}</h3>
                  <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{s.description}</p>
                </div>
              </div>
              <div className="mt-4">
                <Button label={busyKey === s.key ? 'Starting…' : 'Run Now'} onClick={() => runKey(s.key)} disabled={busyKey !== null} />
              </div>
            </div>
          ))}
        </div>

        {/* Recent Data Buttons */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} rounded-xl p-5 shadow-sm`}>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold">📊 SPY Fronttest Data</h3>
                <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  View recent SPY data
                </p>
              </div>
              <Button 
                label={recentLoading && recentSymbol === 'spy' ? 'Loading…' : 'Show recent spy fronttest'} 
                onClick={() => fetchRecentData('spy')} 
                disabled={recentLoading || busyKey !== null}
                tone="success"
              />
            </div>
          </div>

          <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} rounded-xl p-5 shadow-sm`}>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold">📈 ES Fronttest Data</h3>
                <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  View recent ES data
                </p>
              </div>
              <Button 
                label={recentLoading && recentSymbol === 'es' ? 'Loading…' : 'Show recent es fronttest'} 
                onClick={() => fetchRecentData('es')} 
                disabled={recentLoading || busyKey !== null}
                tone="success"
              />
            </div>
          </div>

          <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} rounded-xl p-5 shadow-sm`}>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold">💱 EURUSD Fronttest Data</h3>
                <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  View recent EURUSD data
                </p>
              </div>
              <Button 
                label={recentLoading && recentSymbol === 'eurusd' ? 'Loading…' : 'Show recent eurusd fronttest'} 
                onClick={() => fetchRecentData('eurusd')} 
                disabled={recentLoading || busyKey !== null}
                tone="success"
              />
            </div>
          </div>
        </div>

        {/* Singles */}
        <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} rounded-xl p-5 shadow-sm`}>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Run a Specific Table</h3>
            {loading && <span className="text-sm opacity-70">Loading…</span>}
            {error && <span className="text-sm text-red-500">{String(error)}</span>}
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {singleScripts.map(name => (
              <Button key={name} label={name.replace('.py','')} onClick={() => runSingle(name)} disabled={busyKey !== null} tone="subtle" />
            ))}
          </div>
          <p className={`mt-3 text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            These call <code>run_pipeline_script.bat</code> with the script name. Ensure the file exists in <code>data-pipeline</code>.
          </p>
        </div>

        {/* Data Freshness Status */}
        <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} rounded-xl p-5 shadow-sm mt-6`}>
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold">📊 Data Freshness Status</h3>
              <p className={`text-xs mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                {freshnessStatus?.timestamp 
                  ? `Last checked: ${new Date(freshnessStatus.timestamp).toLocaleString()}`
                  : 'Loading...'}
              </p>
            </div>
            <button
              onClick={refreshFreshness}
              disabled={freshnessLoading}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                isDarkMode 
                  ? 'bg-gray-700 hover:bg-gray-600 text-gray-300' 
                  : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
              } disabled:opacity-50`}
            >
              {freshnessLoading ? '⏳ Refreshing...' : '🔄 Refresh'}
            </button>
          </div>

          {freshnessError && (
            <div className={`text-sm mb-4 p-3 rounded-lg ${isDarkMode ? 'bg-red-900/30 text-red-400' : 'bg-red-50 text-red-600'}`}>
              Error: {freshnessError}
            </div>
          )}

          {freshnessStatus && (
            <div className="space-y-4">
              {/* Market Status Banner */}
              <div className={`flex gap-4 p-3 rounded-lg ${isDarkMode ? 'bg-gray-700/50' : 'bg-gray-50'}`}>
                {Object.entries(freshnessStatus.market_status || {}).map(([asset, status]) => (
                  <div key={asset} className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${status === 'Open' ? 'bg-emerald-500' : 'bg-gray-400'}`}></span>
                    <span className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                      {asset.toUpperCase()}: {status}
                    </span>
                  </div>
                ))}
              </div>

              {/* Asset Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {freshnessStatus.assets?.map(asset => (
                  <div 
                    key={asset.asset} 
                    className={`rounded-lg border ${
                      asset.overall_status === 'up_to_date' 
                        ? isDarkMode ? 'border-emerald-700 bg-emerald-900/20' : 'border-emerald-200 bg-emerald-50'
                        : asset.overall_status === 'stale'
                        ? isDarkMode ? 'border-red-700 bg-red-900/20' : 'border-red-200 bg-red-50'
                        : isDarkMode ? 'border-amber-700 bg-amber-900/20' : 'border-amber-200 bg-amber-50'
                    } p-4`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold text-base">
                        {asset.asset === 'spy' && '📊'}
                        {asset.asset === 'es' && '📈'}
                        {asset.asset === 'eurusd' && '💱'}
                        {' '}{asset.asset.toUpperCase()}
                      </h4>
                      <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                        asset.overall_status === 'up_to_date'
                          ? isDarkMode ? 'bg-emerald-800 text-emerald-200' : 'bg-emerald-100 text-emerald-700'
                          : asset.overall_status === 'stale'
                          ? isDarkMode ? 'bg-red-800 text-red-200' : 'bg-red-100 text-red-700'
                          : isDarkMode ? 'bg-amber-800 text-amber-200' : 'bg-amber-100 text-amber-700'
                      }`}>
                        {asset.overall_status === 'up_to_date' ? '✓ Up to Date' : 
                         asset.overall_status === 'stale' ? '✗ Stale' : '⚠ Partial'}
                      </span>
                    </div>
                    
                    <div className="space-y-1.5">
                      {asset.timeframes?.map(tf => (
                        <div 
                          key={tf.timeframe}
                          className={`flex items-center justify-between text-sm py-1.5 px-2 rounded ${
                            isDarkMode ? 'bg-gray-800/50' : 'bg-white/70'
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            <span className={`w-2 h-2 rounded-full ${
                              tf.is_up_to_date 
                                ? 'bg-emerald-500' 
                                : tf.staleness_periods > 5 
                                ? 'bg-red-500' 
                                : 'bg-amber-500'
                            }`}></span>
                            <span className={`font-medium ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                              {tf.timeframe}
                            </span>
                          </div>
                          <span className={`text-xs ${
                            tf.is_up_to_date 
                              ? isDarkMode ? 'text-emerald-400' : 'text-emerald-600'
                              : tf.staleness_periods > 5
                              ? isDarkMode ? 'text-red-400' : 'text-red-600'
                              : isDarkMode ? 'text-amber-400' : 'text-amber-600'
                          }`}>
                            {tf.is_up_to_date ? '✓' : tf.staleness_message}
                          </span>
                        </div>
                      ))}
                    </div>

                    {/* Last Update Info */}
                    <div className={`mt-3 pt-2 border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                      <p className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                        Latest 1m: {asset.timeframes?.find(t => t.timeframe === '1m')?.last_timestamp 
                          ? new Date(asset.timeframes.find(t => t.timeframe === '1m').last_timestamp).toLocaleString()
                          : 'N/A'}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {freshnessLoading && !freshnessStatus && (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <span className={`ml-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Loading freshness data...</span>
            </div>
          )}
        </div>

        {/* Freshness Rules Explanation */}
        <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} rounded-xl p-5 shadow-sm mt-6`}>
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setShowRules(!showRules)}
          >
            <h3 className="text-lg font-semibold">📋 Data Freshness Rules</h3>
            <button className={`p-1 rounded ${isDarkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}>
              {showRules ? '▲' : '▼'}
            </button>
          </div>

          {showRules && (
            <div className="mt-4 space-y-6">
              {/* General Rules */}
              <div>
                <h4 className={`font-semibold mb-2 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                  📌 General Rules
                </h4>
                <ul className={`list-disc list-inside text-sm space-y-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  <li>Data is considered <span className="text-emerald-500 font-medium">Up to Date</span> if within 1 candle period of expected</li>
                  <li>A candle is complete when its close time has passed</li>
                  <li>All times are calculated in <span className="font-medium">Eastern Time (ET)</span></li>
                  <li>Weekends and market closures are accounted for</li>
                </ul>
              </div>

              {/* Market Hours Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className={`${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                      <th className="text-left py-2 px-3 rounded-tl-lg">Asset</th>
                      <th className="text-left py-2 px-3">Market Type</th>
                      <th className="text-left py-2 px-3">Trading Hours (ET)</th>
                      <th className="text-left py-2 px-3 rounded-tr-lg">Notes</th>
                    </tr>
                  </thead>
                  <tbody className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>
                    <tr className={`border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                      <td className="py-2 px-3 font-medium">📊 SPY</td>
                      <td className="py-2 px-3">US Equity ETF</td>
                      <td className="py-2 px-3">Mon-Fri: 9:30 AM – 4:00 PM</td>
                      <td className="py-2 px-3 text-xs">Regular trading hours only. Closed weekends & holidays.</td>
                    </tr>
                    <tr className={`border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                      <td className="py-2 px-3 font-medium">📈 ES</td>
                      <td className="py-2 px-3">E-mini Futures</td>
                      <td className="py-2 px-3">Sun 6:00 PM – Fri 5:00 PM</td>
                      <td className="py-2 px-3 text-xs">Nearly 24/5. Daily break 5-6 PM ET. Closed Sat & Sun pre-6PM.</td>
                    </tr>
                    <tr>
                      <td className="py-2 px-3 font-medium">💱 EURUSD</td>
                      <td className="py-2 px-3">Forex</td>
                      <td className="py-2 px-3">Sun 5:00 PM – Fri 5:00 PM</td>
                      <td className="py-2 px-3 text-xs">24/5 trading. Closed Saturday & Sunday pre-5PM.</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Timeframe Expectations */}
              <div>
                <h4 className={`font-semibold mb-2 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                  ⏱️ Timeframe Update Expectations
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className={`${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                        <th className="text-left py-2 px-3 rounded-tl-lg">Timeframe</th>
                        <th className="text-left py-2 px-3">Candle Duration</th>
                        <th className="text-left py-2 px-3">Expected Updates</th>
                        <th className="text-left py-2 px-3 rounded-tr-lg">Tolerance</th>
                      </tr>
                    </thead>
                    <tbody className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>
                      <tr className={`border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                        <td className="py-2 px-3 font-medium">1d</td>
                        <td className="py-2 px-3">1 Day</td>
                        <td className="py-2 px-3">Once per trading day (at market close)</td>
                        <td className="py-2 px-3 text-emerald-500">≤1 day behind</td>
                      </tr>
                      <tr className={`border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                        <td className="py-2 px-3 font-medium">4h</td>
                        <td className="py-2 px-3">4 Hours</td>
                        <td className="py-2 px-3">~6 candles per trading day</td>
                        <td className="py-2 px-3 text-emerald-500">≤1 candle (4h) behind</td>
                      </tr>
                      <tr className={`border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                        <td className="py-2 px-3 font-medium">1h</td>
                        <td className="py-2 px-3">1 Hour</td>
                        <td className="py-2 px-3">~6.5 candles per trading day (SPY)</td>
                        <td className="py-2 px-3 text-emerald-500">≤1 candle (1h) behind</td>
                      </tr>
                      <tr className={`border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                        <td className="py-2 px-3 font-medium">30m</td>
                        <td className="py-2 px-3">30 Minutes</td>
                        <td className="py-2 px-3">~13 candles per trading day (SPY)</td>
                        <td className="py-2 px-3 text-emerald-500">≤1 candle (30m) behind</td>
                      </tr>
                      <tr className={`border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                        <td className="py-2 px-3 font-medium">15m</td>
                        <td className="py-2 px-3">15 Minutes</td>
                        <td className="py-2 px-3">~26 candles per trading day (SPY)</td>
                        <td className="py-2 px-3 text-emerald-500">≤1 candle (15m) behind</td>
                      </tr>
                      <tr className={`border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                        <td className="py-2 px-3 font-medium">5m</td>
                        <td className="py-2 px-3">5 Minutes</td>
                        <td className="py-2 px-3">~78 candles per trading day (SPY)</td>
                        <td className="py-2 px-3 text-emerald-500">≤1 candle (5m) behind</td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3 font-medium">1m</td>
                        <td className="py-2 px-3">1 Minute</td>
                        <td className="py-2 px-3">~390 candles per trading day (SPY)</td>
                        <td className="py-2 px-3 text-emerald-500">≤1 candle (1m) behind</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Status Legend */}
              <div>
                <h4 className={`font-semibold mb-2 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                  🚦 Status Legend
                </h4>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className={`flex items-center gap-2 p-3 rounded-lg ${isDarkMode ? 'bg-emerald-900/30' : 'bg-emerald-50'}`}>
                    <span className="w-3 h-3 rounded-full bg-emerald-500"></span>
                    <div>
                      <p className={`font-medium text-sm ${isDarkMode ? 'text-emerald-300' : 'text-emerald-700'}`}>Up to Date</p>
                      <p className={`text-xs ${isDarkMode ? 'text-emerald-400/70' : 'text-emerald-600/70'}`}>Within 1 candle of expected</p>
                    </div>
                  </div>
                  <div className={`flex items-center gap-2 p-3 rounded-lg ${isDarkMode ? 'bg-amber-900/30' : 'bg-amber-50'}`}>
                    <span className="w-3 h-3 rounded-full bg-amber-500"></span>
                    <div>
                      <p className={`font-medium text-sm ${isDarkMode ? 'text-amber-300' : 'text-amber-700'}`}>Slightly Stale</p>
                      <p className={`text-xs ${isDarkMode ? 'text-amber-400/70' : 'text-amber-600/70'}`}>2-5 candles behind</p>
                    </div>
                  </div>
                  <div className={`flex items-center gap-2 p-3 rounded-lg ${isDarkMode ? 'bg-red-900/30' : 'bg-red-50'}`}>
                    <span className="w-3 h-3 rounded-full bg-red-500"></span>
                    <div>
                      <p className={`font-medium text-sm ${isDarkMode ? 'text-red-300' : 'text-red-700'}`}>Stale</p>
                      <p className={`text-xs ${isDarkMode ? 'text-red-400/70' : 'text-red-600/70'}`}>More than 5 candles behind</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* DST Note */}
              <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-blue-900/20 border border-blue-800' : 'bg-blue-50 border border-blue-200'}`}>
                <p className={`text-sm ${isDarkMode ? 'text-blue-300' : 'text-blue-700'}`}>
                  <span className="font-medium">💡 Daylight Saving Time:</span> All calculations use pytz for proper timezone handling. 
                  The system automatically adjusts for DST transitions (EST ↔ EDT).
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Recent Data Modal */}
        {showRecentData && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl shadow-xl max-w-6xl w-full max-h-[90vh] overflow-hidden`}>
              <div className={`flex items-center justify-between p-6 border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <div>
                  <h3 className="text-xl font-bold">
                    {recentSymbol === 'spy' && '📊'} 
                    {recentSymbol === 'es' && '📈'} 
                    {recentSymbol === 'eurusd' && '💱'} 
                    Recent {recentSymbol?.toUpperCase()} Fronttest Data
                  </h3>
                  <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                    {recentData ? `${recentData.total_tables} tables, ${recentData.total_rows} total rows` : 'Loading...'}
                  </p>
                </div>
                <button
                  onClick={() => setShowRecentData(false)}
                  className={`p-2 rounded-lg transition-colors duration-200 ${
                    isDarkMode 
                      ? 'hover:bg-gray-700 text-gray-400 hover:text-white' 
                      : 'hover:bg-gray-100 text-gray-500 hover:text-gray-700'
                  }`}
                >
                  ✕
                </button>
              </div>
              
              <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
                {recentData ? (
                  <div className="space-y-6">
                    {sortTablesByTimeframe(recentData.tables).map(table => (
                      <div key={table.table_name} className={`${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'} rounded-lg p-4`}>
                        <h4 className="text-lg font-semibold mb-3">
                          {table.table_name} ({table.timeframe})
                        </h4>
                        <div className="overflow-x-auto">
                          <table className="w-full text-sm">
                            <thead>
                              <tr className={`${isDarkMode ? 'text-gray-300' : 'text-gray-600'} border-b`}>
                                <th className="text-left py-2 px-3">Timestamp</th>
                                <th className="text-right py-2 px-3">Open</th>
                                <th className="text-right py-2 px-3">High</th>
                                <th className="text-right py-2 px-3">Low</th>
                                <th className="text-right py-2 px-3">Close</th>
                                <th className="text-right py-2 px-3">Volume</th>
                              </tr>
                            </thead>
                            <tbody>
                              {table.rows.map((row, idx) => (
                                <tr key={idx} className={`${isDarkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-100'} transition-colors`}>
                                  <td className="py-2 px-3">
                                    {row.timestamp ? new Date(row.timestamp).toLocaleString() : '-'}
                                  </td>
                                  <td className="text-right py-2 px-3">
                                    {row.open ? row.open.toFixed(2) : '-'}
                                  </td>
                                  <td className="text-right py-2 px-3">
                                    {row.high ? row.high.toFixed(2) : '-'}
                                  </td>
                                  <td className="text-right py-2 px-3">
                                    {row.low ? row.low.toFixed(2) : '-'}
                                  </td>
                                  <td className="text-right py-2 px-3">
                                    {row.close ? row.close.toFixed(2) : '-'}
                                  </td>
                                  <td className="text-right py-2 px-3">
                                    {row.volume ? row.volume.toLocaleString() : '-'}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : recentError ? (
                  <div className={`text-center py-8 ${isDarkMode ? 'text-red-400' : 'text-red-600'}`}>
                    Error: {recentError}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
                    <p className={`mt-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Loading {recentSymbol?.toUpperCase()} data...</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Toast */}
        {toast && (
          <div className={`fixed bottom-6 right-6 px-4 py-3 rounded-md shadow-lg ${toast.type === 'success' ? 'bg-emerald-600 text-white' : 'bg-red-600 text-white'}`}>
            {toast.text}
          </div>
        )}
      </div>
    </div>
  );
}

PipelineDashboard.propTypes = {
  isDarkMode: PropTypes.bool,
};


