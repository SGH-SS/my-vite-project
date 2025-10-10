import { useEffect, useMemo, useState } from 'react';
// Always-available raw code imports for the two training scripts
// Vite `?raw` ensures code shows even before any run produces snapshots
import isoCodeRaw from './v2 daygent models/train_lightgbm_1d_iso.py?raw';
import v0CodeRaw from './v2 daygent models/train_lightgbm_1d_v0.py?raw';

const API_BASE = 'http://localhost:8000/api/training';

// --- Directory Viewer Component for Debugging ---
const DirNode = ({ node, isDarkMode }) => {
  const [isOpen, setIsOpen] = useState(node.type === 'dir' ? true : false);
  const isDir = node.type === 'dir';

  const icon = isDir ? (isOpen ? '📂' : '📁') : '📄';

  return (
    <div className="ml-4">
      <div onClick={() => isDir && setIsOpen(!isOpen)} className={`${isDir ? 'cursor-pointer' : ''} flex items-center text-sm`}>
        <span className="w-6 text-center">{icon}</span>
        <span className={`${isDir ? 'font-medium' : ''}`}>{node.name}</span>
      </div>
      {isDir && isOpen && node.children && (
        <div className={`pl-4 border-l ml-3 ${isDarkMode ? 'border-gray-600' : 'border-gray-300'}`}>
          {node.children.map((child, i) => <DirNode key={`${child.name}-${i}`} node={child} isDarkMode={isDarkMode} />)}
        </div>
      )}
    </div>
  );
};


export default function ModelTraining({ isDarkMode = false }) {
  const [scripts, setScripts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [busyKey, setBusyKey] = useState(null);
  const [toast, setToast] = useState(null);
  const [latest, setLatest] = useState({});
  const [codeByScript, setCodeByScript] = useState({
    'train_lightgbm_1d_iso.py': isoCodeRaw,
    'train_lightgbm_1d_v0.py': v0CodeRaw,
  });
  const [polling, setPolling] = useState(false);
  const [dirTree, setDirTree] = useState(null);
  const [treeLoading, setTreeLoading] = useState(false);

  // Build a smart preview of a Python script: docstring, key PARAMS dicts,
  // important config constants, and risk gates. Generic, resilient to edits/new scripts.
  const buildSmartPreview = (code) => {
    if (!code || typeof code !== 'string') return '';
    const lines = code.split('\n');

    const takeDocstring = () => {
      let start = -1, end = -1, q = null, seen = 0;
      for (let i = 0; i < Math.min(lines.length, 400); i++) {
        const ln = lines[i];
        const m = ln.match(/(["']{3})/);
        if (m) {
          if (start === -1) {
            start = i; q = m[1];
            if ((ln.match(new RegExp(q, 'g')) || []).length >= 2) { end = i; break; }
          } else if (q && ln.includes(q)) { end = i; break; }
        }
      }
      if (start !== -1) {
        if (end === -1) end = Math.min(start + 40, lines.length - 1);
        const block = lines.slice(start, end + 1).join('\n');
        return [`# === Header ===`, block];
      }
      return [];
    };

    const collectDictBlocks = () => {
      const blocks = [];
      const nameRegex = /^\s*([A-Za-z_][\w]*)\s*=\s*(dict\s*\(|\{)/;
      const targetish = (name) => {
        const n = name.toLowerCase();
        return n.includes('params') || n.includes('gate') || n.includes('config');
      };
      const openCloseMatchLen = (text, openChar, closeChar) => {
        let depth = 0; let consumed = 0; const arr = text.split('\n');
        for (let i = 0; i < arr.length; i++) {
          const l = arr[i];
          for (const ch of l) {
            if (ch === openChar) depth += 1; else if (ch === closeChar) depth -= 1;
          }
          consumed = i + 1;
          if (depth <= 0 && i > 0) break;
        }
        return consumed;
      };
      for (let i = 0; i < lines.length; i++) {
        const ln = lines[i];
        const m = ln.match(nameRegex);
        if (m && targetish(m[1])) {
          const usingDictCtor = m[2].startsWith('dict');
          let endIdx = i + 1;
          if (usingDictCtor) {
            // dict( ... ) capture by parentheses
            let depth = 0; let j = i; let found = false;
            while (j < lines.length) {
              const l = lines[j];
              for (const ch of l) { if (ch === '(') depth++; else if (ch === ')') depth--; }
              if (j > i && depth <= 0) { found = true; break; }
              j++;
            }
            endIdx = Math.min(j + 1, lines.length);
          } else {
            // brace block { ... }
            let depth = 0; let j = i; let found = false;
            while (j < lines.length) {
              const l = lines[j];
              for (const ch of l) { if (ch === '{') depth++; else if (ch === '}') depth--; }
              if (j > i && depth <= 0) { found = true; break; }
              j++;
            }
            endIdx = Math.min(j + 1, lines.length);
          }
          const block = lines.slice(i, endIdx).join('\n');
          blocks.push([`# === ${m[1]} ===`, block]);
          // Skip ahead to avoid duplicate captures
          i = endIdx - 1;
        }
      }
      return blocks;
    };

    const collectConstants = () => {
      // Capture compact runs of ALL_CAPS assignments (config constants)
      const caps = [];
      let run = [];
      const isCapsAssign = (ln) => /^\s*[A-Z][A-Z0-9_]*\s*=/.test(ln);
      for (let i = 0; i < Math.min(lines.length, 800); i++) {
        const ln = lines[i];
        if (isCapsAssign(ln)) { run.push(ln); }
        else {
          if (run.length >= 3) caps.push(run.slice(0, 16).join('\n'));
          run = [];
        }
      }
      if (run.length >= 3) caps.push(run.slice(0, 16).join('\n'));
      if (!caps.length) return [];
      return [[`# === Constants ===`, caps[0]]];
    };

    const parts = [];
    parts.push(...takeDocstring());
    parts.push(...collectConstants().flat());
    for (const pair of collectDictBlocks()) { parts.push(...pair); }

    const cleaned = parts.join('\n') || lines.slice(0, 60).join('\n');
    // Limit preview length to keep UI compact
    const lim = 2200;
    return cleaned.length > lim ? cleaned.slice(0, lim) + '\n...' : cleaned;
  };

  useEffect(() => {
    const fetchScripts = async () => {
      try {
        setLoading(true);
        const res = await fetch(`${API_BASE}/scripts`);
        const data = await res.json();
        if (!res.ok) throw new Error(data?.detail || 'Failed to load training scripts');
        setScripts(data);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    };
    fetchScripts();
  }, []);

  // --- Fetch Directory Tree from backend (server-side walk) ---
  useEffect(() => {
    const fetchTree = async () => {
      setTreeLoading(true);
      try {
        const res = await fetch(`${API_BASE}/dir-tree?max_depth=3`, { cache: 'no-store' });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(`Dir tree error ${res.status}: ${text}`);
        }
        const data = await res.json();
        setDirTree(data);
      } catch (e) {
        console.error('Directory tree fetch failed:', e);
        setDirTree({ name: 'v2 daygent models (error)', type: 'dir', children: [ { name: String(e), type: 'file' } ] });
      } finally {
        setTreeLoading(false);
      }
    };
    fetchTree();
  }, []);

  // --- Fetch latest results via backend (server-side lookup) ---
  const fetchLatestFromArtifacts = async () => {
    try {
      const res = await fetch(`${API_BASE}/latest-and-best-metrics`, { cache: 'no-store' });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Latest+Best metrics error ${res.status}: ${text}`);
      }
      const rows = await res.json();
      const latestMap = {};
      for (const r of rows) {
        // Keep existing UI API: latestMap[script] = { run_timestamp_utc, metrics }
        if (r.latest) {
          latestMap[r.script_name] = { run_timestamp_utc: r.latest.run_timestamp_utc, metrics: r.latest.metrics, best: r.best, criterion: r.criterion };
        } else {
          latestMap[r.script_name] = null;
        }
      }
      setLatest(latestMap);
    } catch (e) {
      console.error('Latest+Best metrics fetch failed:', e);
    }
  };

  useEffect(() => {
    fetchLatestFromArtifacts();
  }, [scripts]);


  const runScript = async (scriptName) => {
    try {
      setBusyKey(scriptName);
      setToast(null);
      const res = await fetch(`${API_BASE}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scriptName })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || 'Failed to start training');
      setToast({ type: 'success', text: `Started ${scriptName} (PID ${data.pid || 'N/A'})` });
      // Start brief polling for latest after run starts
      startPolling();
    } catch (e) {
      setToast({ type: 'error', text: String(e) });
    } finally {
      setBusyKey(null);
    }
  };

  const startPolling = () => {
    if (polling) return;
    setPolling(true);
    let ticks = 0;
    const id = setInterval(() => {
      ticks += 1;
      // Re-run the fetcher to get the latest info
      fetchLatestFromArtifacts();

      if (ticks >= 20) { // ~20 polls then stop
        clearInterval(id);
        setPolling(false);
      }
    }, 2500); // Poll slightly less frequently
  };

  const runAll = async () => {
    if (!scripts || scripts.length === 0) return;
    setToast(null);
    startPolling();
    for (const s of scripts) {
      try {
        await fetch(`${API_BASE}/run`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scriptName: s.script_name })
        });
      } catch (_) {}
    }
  };

  const fmt = (x, d = 4) => {
    if (x === null || x === undefined) return '—';
    if (typeof x === 'number') {
      if (!isFinite(x)) return '—';
      const m = Math.pow(10, d);
      return String(Math.round(x * m) / m);
    }
    return String(x);
  };

  const IsoMetrics = ({ m }) => {
    if (!m) return (<div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>No results yet.</div>);
    return (
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div><div className="opacity-70 text-xs">Val Acc</div><div className="font-medium">{fmt(m.validation_accuracy)}</div></div>
        <div><div className="opacity-70 text-xs">Val AUC</div><div className="font-medium">{fmt(m.validation_auc)}</div></div>
        <div><div className="opacity-70 text-xs">Test Acc</div><div className="font-medium">{fmt(m.test_accuracy)}</div></div>
        <div><div className="opacity-70 text-xs">Test AUC</div><div className="font-medium">{fmt(m.test_auc)}</div></div>
        <div className="col-span-2"><div className="opacity-70 text-xs">Threshold</div><div className="font-medium">{fmt(m.chosen_threshold, 3)}</div></div>
      </div>
    );
  };

  const V0Metrics = ({ m }) => {
    if (!m) return (<div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>No results yet.</div>);
    return (
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div><div className="opacity-70 text-xs">θ</div><div className="font-medium">{fmt(m.theta_deploy, 6)}</div></div>
        <div><div className="opacity-70 text-xs">best_iter</div><div className="font-medium">{fmt(m.best_iter_median, 0)}</div></div>
        {m.oos_summary && (
          <>
            <div><div className="opacity-70 text-xs">OOS trades mean</div><div className="font-medium">{fmt(m.oos_summary.trades_mean, 2)}</div></div>
            <div><div className="opacity-70 text-xs">EV/trade median</div><div className="font-medium">{fmt(m.oos_summary.ev_per_trade_median, 6)}</div></div>
            <div><div className="opacity-70 text-xs">Sharpe median</div><div className="font-medium">{fmt(m.oos_summary.sharpe_daily_median, 3)}</div></div>
            <div><div className="opacity-70 text-xs">Max DD median</div><div className="font-medium">{fmt(m.oos_summary.max_dd_median, 3)}</div></div>
          </>
        )}
      </div>
    );
  };

  return (
    <div className={`min-h-screen transition-colors p-6 ${isDarkMode ? 'bg-gray-900 text-gray-100' : 'bg-gray-50 text-gray-900'}`}>
      <div className="max-w-7xl mx-auto">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">🧪 Model Training</h1>
            <p className={`${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Run local training scripts in your conda env</p>
          </div>
          <button
            onClick={runAll}
            disabled={busyKey !== null}
            className={`px-3 py-2 rounded-md text-sm font-medium ${isDarkMode ? 'bg-indigo-600 hover:bg-indigo-500 text-white' : 'bg-indigo-600 hover:bg-indigo-700 text-white'}`}
          >Run All</button>
        </div>

        {toast && (
          <div className={`mb-4 p-3 rounded ${toast.type === 'success' ? (isDarkMode ? 'bg-green-900/30 text-green-300' : 'bg-green-100 text-green-800') : (isDarkMode ? 'bg-red-900/30 text-red-300' : 'bg-red-100 text-red-800')}`}>{toast.text}</div>
        )}

        {loading ? (
          <div className="p-6">Loading…</div>
        ) : error ? (
          <div className={`p-4 rounded ${isDarkMode ? 'bg-red-900/30 text-red-300' : 'bg-red-100 text-red-800'}`}>{error}</div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {scripts.map(s => (
              <div key={s.key} className={`${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border rounded-xl p-8 shadow-lg min-h-[600px]`}>
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-lg font-semibold">{s.title}</h3>
                    <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{s.description}</p>
                    <p className={`text-xs mt-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>{s.script_name}</p>
                  </div>
                  <button
                    onClick={() => runScript(s.script_name)}
                    disabled={busyKey !== null}
                    className={`px-4 py-2 rounded-md text-sm font-medium ${busyKey === s.script_name ? 'opacity-60' : ''} ${isDarkMode ? 'bg-blue-600 hover:bg-blue-500 text-white' : 'bg-blue-600 hover:bg-blue-700 text-white'}`}
                  >
                    {busyKey === s.script_name ? 'Starting…' : 'Run'}
                  </button>
                </div>
                {/* Code viewer - smart preview (docstring, params, constants) */}
                 <div className={`mt-5 rounded-lg border overflow-hidden ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-100 border-gray-200'}`}>
                   <div className="relative">
                   <pre className={`p-4 text-xs font-mono leading-6 overflow-auto max-h-48 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                      <code className="language-python">{buildSmartPreview(codeByScript[s.script_name] || '')}</code>
                     </pre>
                    {(codeByScript[s.script_name] || '').length > 0 && (
                       <div className={`absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t pointer-events-none ${isDarkMode ? 'from-gray-900/50' : 'from-gray-50'}`}></div>
                     )}
                   </div>
                 </div>
                {/* Latest and Best results (side by side) */}
                <div className="mt-8 grid grid-cols-1 xl:grid-cols-2 gap-6">
                  {/* Latest */}
                  <div className={`${isDarkMode ? 'bg-gray-900/40 border-gray-700' : 'bg-gray-50 border-gray-200'} rounded-lg border p-5`}>
                    <div className="flex items-center justify-between">
                      <div className="text-base font-semibold">Latest</div>
                      <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        {latest[s.script_name]?.run_timestamp_utc?.replace(/_/g, ' ') || '—'}
                      </div>
                    </div>
                    <div className={`mt-4 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                      {latest[s.script_name]?.metrics ? (
                        latest[s.script_name].metrics.oos_summary ? (
                          <V0Metrics m={latest[s.script_name].metrics} />
                        ) : (
                          <IsoMetrics m={latest[s.script_name].metrics} />
                        )
                      ) : (
                        <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>No results yet.</div>
                      )}
                    </div>
                  </div>
                  {/* Best */}
                  <div className={`${isDarkMode ? 'bg-gray-900/40 border-gray-700' : 'bg-gray-50 border-gray-200'} rounded-lg border p-5`}>
                    <div className="flex items-center justify-between">
                      <div className="text-base font-semibold">Best</div>
                      <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        {latest[s.script_name]?.best?.run_timestamp_utc?.replace(/_/g, ' ') || '—'}
                      </div>
                    </div>
                    {latest[s.script_name]?.criterion && (
                      <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        by {latest[s.script_name].criterion}
                      </div>
                    )}
                    <div className={`mt-4 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                      {latest[s.script_name]?.best?.metrics ? (
                        latest[s.script_name].best.metrics.oos_summary ? (
                          <V0Metrics m={latest[s.script_name].best.metrics} />
                        ) : (
                          <IsoMetrics m={latest[s.script_name].best.metrics} />
                        )
                      ) : (
                        <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>—</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
        {/* --- Directory Viewer for Debugging --- */}
        <div className="mt-8">
          <h2 className="text-xl font-bold mb-4">📦 Directory Debugger</h2>
          <div className={`p-4 rounded-xl border ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-100 border-gray-200'}`}>
            <p className={`text-xs mb-3 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              This shows the live file structure inside <code>/v2 daygent models/</code> as seen by the browser. Use this to verify that artifact folders and files are being created correctly.
            </p>
            {treeLoading && <div>Loading directory tree...</div>}
            {dirTree && <DirNode node={dirTree} isDarkMode={isDarkMode} />}
          </div>
        </div>
      </div>
    </div>
  );
}


