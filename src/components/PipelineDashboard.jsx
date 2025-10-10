import { useEffect, useMemo, useState } from 'react';
import PropTypes from 'prop-types';

// API base for pipeline endpoints
const API_BASE = 'http://localhost:8000/api/pipeline';

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


