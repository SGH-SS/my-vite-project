/**
 * StratHub — multi-coin component composer + LLM analysis dashboard
 * ====================================================================
 * Left:  collapsible catalog tree per coin (BTC / ETH / SOL / SPX) with
 *        stored vs on-command badges.
 * Top:   window selector · triples toggle · model picker (Opus 4.7/4.6/4.5,
 *        Sonnet 4.6/4.5) · reasoning effort (off / low / medium / high)
 *        · run-counter · big Run button.
 * Right: scrollable response browser.  A pill list across the top lets
 *        you flip between Individual / Pair / Triple cards; the active
 *        card occupies the panel with a markdown-rendered LLM reply,
 *        a sparkline of the underlying series, and a token-cost footer.
 *
 * Comparison calls (pair / triple) include the individual analyses of
 * their constituent components in `prior_individual` so the model has the
 * per-component reasoning loaded into its context before synthesising
 * across them.
 */

import { useState, useEffect, useMemo, useCallback } from 'react';
import {
  AreaChart, Area, BarChart, Bar, Tooltip, ResponsiveContainer,
} from 'recharts';

const API = 'http://localhost:8000/api/strathub';

// ── theme ────────────────────────────────────────────────────────────────
const C = {
  bg: '#060c18', surface: '#0c1628', raised: '#101e35', border: '#1a2d4e',
  text: '#e2e8f0', sub: '#94a3b8', muted: '#475569', dim: '#64748b',
  green: '#00d4a8', red: '#f43f5e', amber: '#eab308',
  blue: '#60a5fa', purple: '#a78bfa', cyan: '#22d3ee',
  orange: '#e97316', pink: '#ec4899', emerald: '#10b981', fuchsia: '#d946ef',
};

const ASSET_COLOR = {
  btc: '#f7b50f', eth: '#8b5cf6', sol: '#22d3ee', spx: '#f43f5e',
};

const WINDOW_OPTIONS = [
  { label: '1m',  value: 60 },
  { label: '5m',  value: 300 },
  { label: '15m', value: 900 },
  { label: '1h',  value: 3600 },
  { label: '4h',  value: 14400 },
  { label: '1d',  value: 86400 },
  { label: '5d',  value: 432000 },
  { label: '1W',  value: 604800 },
  { label: '2W',  value: 1209600 },
  { label: 'ALL', value: 'all' },
];

// Effort options — backend knows which ones apply per model tier.
// "max" is supported on all; "xhigh" is Opus-4-7-only so we skip it for now.
const EFFORT_OPTIONS = ['off', 'low', 'medium', 'high', 'max'];

// ── helpers ──────────────────────────────────────────────────────────────
const fmt = (n, d = 2) => (n == null || Number.isNaN(+n) ? '--' : Number(n).toFixed(d));
const fmtSigned = (n, d = 2) => {
  if (n == null) return '--';
  const v = Number(n);
  return (v >= 0 ? '+' : '') + v.toFixed(d);
};
const fmtCount = (n) => {
  if (n == null) return '--';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
};
const fmtPct = (n, d = 2) => (n == null ? '--' : Number(n).toFixed(d) + '%');

const pairKey = (a, b) => [a, b].sort().join('|');
const tripKey = (a, b, c) => [a, b, c].sort().join('|');

const parseBucketSize = (str) => {
  const m = String(str).trim().match(/^(\d+(?:\.\d+)?)\s*(s|m|h)?$/i);
  if (!m) return 5;
  const n = parseFloat(m[1]);
  const unit = (m[2] || 's').toLowerCase();
  if (unit === 'm') return Math.max(1, Math.round(n * 60));
  if (unit === 'h') return Math.max(1, Math.round(n * 3600));
  return Math.max(1, Math.round(n));
};

// Try to extract a numeric series we can sparkline-chart from a payload.
function extractChartSeries(payload) {
  const series = payload?.series;
  if (!Array.isArray(series) || series.length === 0) return null;

  const numericKeys = [
    'mid_mean', 'mid', 'c', 'microprice', 'imbalance', 'deriv',
    'rv_window_bps', 'spread_bps', 'funding', 'open_interest',
    'premium_bps', 'drift_bps', 'expansion_ratio', 'trades_per_s',
    'snaps_per_s', 'total_depth_usd', 'doi_per_s', 'v_total',
    'ask_replenish', 'bid_replenish', 'ask_size', 'bid_size',
    'buy_volume', 'sell_volume',
  ];

  let chosen = null;
  for (const k of numericKeys) {
    if (series.some((d) => d && d[k] != null && Number.isFinite(+d[k]))) {
      chosen = k; break;
    }
  }
  if (!chosen) return null;

  const data = series
    .map((d, i) => ({ i, ts: d.ts, v: +d[chosen] }))
    .filter((d) => Number.isFinite(d.v));

  if (data.length < 2) return null;
  return { key: chosen, data };
}

// Extract multiple named series from signal_buckets for multi-channel sparkline grid.
function extractSignalBucketChannels(payload) {
  const series = payload?.series;
  if (!Array.isArray(series) || series.length < 2) return null;
  if (payload?.kind !== 'signal_buckets') return null;

  const channels = [
    { key: 'mid_mean', label: 'Mid Price', color: C.cyan },
    { key: 'ask_size', label: 'Ask Size', color: C.red },
    { key: 'bid_size', label: 'Bid Size', color: C.green },
    { key: 'buy_volume', label: 'Buy Vol', color: C.emerald },
    { key: 'sell_volume', label: 'Sell Vol', color: C.pink },
    { key: 'ask_replenish', label: 'Ask Replenish', color: C.orange },
    { key: 'bid_replenish', label: 'Bid Replenish', color: C.blue },
    { key: 'spread_mean_bps', label: 'Spread (bps)', color: C.amber },
  ];

  const result = [];
  for (const ch of channels) {
    const data = series
      .map((d, i) => ({ i, ts: d.ts, v: d[ch.key] != null ? +d[ch.key] : null }))
      .filter((d) => d.v !== null && Number.isFinite(d.v));
    if (data.length >= 2) {
      result.push({ ...ch, data });
    }
  }
  return result.length > 0 ? result : null;
}

// Lightweight markdown-ish renderer (bullets + bold + headings + linebreaks).
function renderClaudeText(text) {
  if (!text) return null;
  const lines = String(text).split(/\r?\n/);
  const out = [];
  let listBuf = [];
  const flushList = (i) => {
    if (listBuf.length) {
      out.push(
        <ul key={`ul-${i}`} style={{
          margin: '6px 0 14px 18px', paddingLeft: 14,
          color: C.text, lineHeight: 1.6,
        }}>
          {listBuf.map((it, j) => (
            <li key={j} style={{ marginBottom: 4 }}>{renderInline(it)}</li>
          ))}
        </ul>
      );
      listBuf = [];
    }
  };

  lines.forEach((raw, idx) => {
    const line = raw.replace(/\s+$/, '');
    if (/^\s*[-*•]\s+/.test(line)) {
      listBuf.push(line.replace(/^\s*[-*•]\s+/, ''));
      return;
    }
    flushList(idx);

    if (/^#{1,6}\s+/.test(line)) {
      const level = line.match(/^#+/)[0].length;
      const txt = line.replace(/^#+\s+/, '');
      const fontSize = Math.max(13, 19 - level);
      out.push(
        <div key={idx} style={{
          fontWeight: 700, fontSize, color: C.text,
          margin: '14px 0 6px',
        }}>{renderInline(txt)}</div>
      );
      return;
    }
    if (line.trim() === '') {
      out.push(<div key={idx} style={{ height: 8 }} />);
      return;
    }
    out.push(
      <div key={idx} style={{
        color: C.text, lineHeight: 1.55, fontSize: 14, margin: '4px 0',
      }}>{renderInline(line)}</div>
    );
  });
  flushList('end');
  return out;
}

function renderInline(line) {
  // **bold** and `code` only — keep it cheap.
  const out = [];
  let i = 0; let key = 0;
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let m; let last = 0;
  while ((m = re.exec(line)) !== null) {
    if (m.index > last) out.push(line.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith('**')) {
      out.push(<strong key={key++} style={{ color: C.text }}>{tok.slice(2, -2)}</strong>);
    } else {
      out.push(
        <code key={key++} style={{
          background: C.raised, padding: '1px 6px', borderRadius: 4,
          fontFamily: 'ui-monospace, SFMono-Regular, monospace',
          fontSize: 12.5, color: C.cyan,
        }}>{tok.slice(1, -1)}</code>
      );
    }
    last = m.index + tok.length;
    i = last;
  }
  if (last < line.length) out.push(line.slice(last));
  return out;
}


// ════════════════════════════════════════════════════════════════════════
// MAIN
// ════════════════════════════════════════════════════════════════════════
export default function StratHub() {
  const [catalog, setCatalog] = useState(null);
  const [selected, setSelected] = useState(() => new Set());
  const [windowSec, setWindowSec] = useState(900);
  const [bucketSize, setBucketSize] = useState('5s');
  const [includeTriples, setIncludeTriples] = useState(false);
  const [model, setModel] = useState(null);
  const [effort, setEffort] = useState('high'); // overridden from catalog on load
  const [directive, setDirective] = useState('');
  const [systemPromptExpanded, setSystemPromptExpanded] = useState(false);
  const [systemPromptOverride, setSystemPromptOverride] = useState('');  // saved override sent to API
  const [systemPromptDraft, setSystemPromptDraft] = useState('');        // in-progress editing
  const [defaultSystemPrompts, setDefaultSystemPrompts] = useState({ individual: '', compare: '' });
  const [openGroups, setOpenGroups] = useState(() => new Set(['btc:raw', 'btc:bars']));
  const [openAssets, setOpenAssets] = useState(() => new Set(['btc', 'eth', 'sol', 'spx']));
  const [customStart, setCustomStart] = useState('');
  const [customEnd, setCustomEnd] = useState('');
  const [showCompute, setShowCompute] = useState(false);

  // results: id -> { id, kind: 'individual'|'pair'|'triple', label, components,
  //                  status: 'pending'|'materializing'|'analyzing'|'done'|'error',
  //                  text, thinking, tokens_in, tokens_out, payloads: {key: payload}, error }
  const [results, setResults] = useState({});
  const [activeId, setActiveId] = useState(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState(null);
  // Pipeline progress tracking
  const [pipelinePhase, setPipelinePhase] = useState(null); // null | 'materialize' | 'analyze' | 'done'
  const [materializeProgress, setMaterializeProgress] = useState({ done: 0, total: 0, current: '', startedAt: null });

  // ── load catalog on mount ──────────────────────────────────────────────
  useEffect(() => {
    let dead = false;
    fetch(`${API}/catalog`).then((r) => r.json()).then((c) => {
      if (dead) return;
      setCatalog(c);
      if (c?.default_model) setModel((m) => m || c.default_model);
      if (c?.default_effort) setEffort((e) => (e === 'high' ? c.default_effort : e));
      if (c?.system_prompts) setDefaultSystemPrompts(c.system_prompts);
    }).catch((e) => setRunError(`catalog load failed: ${e.message}`));
    return () => { dead = true; };
    // eslint-disable-next-line
  }, []);

  // ── selection helpers ──────────────────────────────────────────────────
  const toggle = useCallback((key) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  }, []);

  const selectAllForAsset = useCallback((asset, on) => {
    if (!catalog) return;
    const allKeys = catalog.groups.flatMap((g) =>
      g.items.map((it) => `${asset}.${it.key}`)
    );
    setSelected((prev) => {
      const next = new Set(prev);
      for (const k of allKeys) on ? next.add(k) : next.delete(k);
      return next;
    });
  }, [catalog]);

  const toggleGroup = (gid) => setOpenGroups((p) => {
    const n = new Set(p); n.has(gid) ? n.delete(gid) : n.add(gid); return n;
  });
  const toggleAsset = (a) => setOpenAssets((p) => {
    const n = new Set(p); n.has(a) ? n.delete(a) : n.add(a); return n;
  });

  // ── derived: counts ─────────────────────────────────────────────────────
  const N = selected.size;
  const nPair = N >= 2 ? (N * (N - 1)) / 2 : 0;
  const nTrip = N >= 3 ? (N * (N - 1) * (N - 2)) / 6 : 0;
  const totalAnalyze = N + nPair + (includeTriples ? nTrip : 0);
  const totalMaterialize = N;
  const modelInfo = useMemo(
    () => (catalog?.models || []).find((m) => m.id === model) || null,
    [catalog, model]
  );
  const isClaudeModel = !!modelInfo && (!modelInfo.provider || modelInfo.provider === 'anthropic');
  const isClaudeAdaptive = isClaudeModel && modelInfo?.tier === 'adaptive';
  const showEffortSelector = !!modelInfo && !isClaudeModel && (modelInfo.effort_levels || []).length > 1;

  // ── run pipeline ────────────────────────────────────────────────────────
  const runPipeline = useCallback(async (overrideSince, overrideUntil) => {
    if (!catalog || running || N === 0) return;
    setRunning(true); setRunError(null);
    setShowCompute(false);
    setPipelinePhase('materialize');

    const selectedArr = [...selected];
    setMaterializeProgress({ done: 0, total: selectedArr.length, current: selectedArr[0] || '', startedAt: Date.now() });

    // 1. Pre-create all result rows so the user can switch immediately.
    const initialResults = {};
    const indivIds = [];
    selectedArr.forEach((key) => {
      const [asset, ...rest] = key.split('.');
      const item = rest.join('.');
      const id = `i:${key}`;
      indivIds.push({ id, key, asset, item });
      initialResults[id] = {
        id, kind: 'individual', label: key, components: [key],
        status: 'materializing', text: '', thinking: null,
        tokens_in: 0, tokens_out: 0, payloads: {}, error: null,
        startedAt: Date.now(),
      };
    });

    const pairs = [];
    for (let i = 0; i < selectedArr.length; i++) {
      for (let j = i + 1; j < selectedArr.length; j++) {
        const a = selectedArr[i]; const b = selectedArr[j];
        const id = `p:${pairKey(a, b)}`;
        pairs.push({ id, a, b });
        initialResults[id] = {
          id, kind: 'pair', label: `${a}  ↔  ${b}`,
          components: [a, b], status: 'pending', text: '', thinking: null,
          tokens_in: 0, tokens_out: 0, payloads: {}, error: null,
          startedAt: Date.now(),
        };
      }
    }

    const triples = [];
    if (includeTriples) {
      for (let i = 0; i < selectedArr.length; i++)
        for (let j = i + 1; j < selectedArr.length; j++)
          for (let k = j + 1; k < selectedArr.length; k++) {
            const a = selectedArr[i], b = selectedArr[j], c = selectedArr[k];
            const id = `t:${tripKey(a, b, c)}`;
            triples.push({ id, a, b, c });
            initialResults[id] = {
              id, kind: 'triple', label: `${a}  ⟷  ${b}  ⟷  ${c}`,
              components: [a, b, c], status: 'pending', text: '', thinking: null,
              tokens_in: 0, tokens_out: 0, payloads: {}, error: null,
              startedAt: Date.now(),
            };
          }
    }

    setResults(initialResults);
    setActiveId(indivIds[0]?.id || null);

    // 2. Materialize every component sequentially (to show per-component progress and avoid server overload).
    const payloadByKey = {};
    let matDone = 0;
    for (const key of selectedArr) {
      const [asset, ...rest] = key.split('.');
      const item = rest.join('.');
      setMaterializeProgress((p) => ({ ...p, current: key, done: matDone }));
      try {
        const matBody = overrideSince
          ? { asset, item, since: overrideSince, until: overrideUntil }
          : { asset, item, window_s: windowSec === 'all' ? null : windowSec };
        if (item === 'signal_buckets') {
          matBody.bucket_s = parseBucketSize(bucketSize);
        }
        const r = await fetch(`${API}/materialize`, {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify(matBody),
        });
        if (!r.ok) {
          const errText = await r.text().catch(() => '');
          throw new Error(`HTTP ${r.status}: ${errText.slice(0, 200)}`);
        }
        const j = await r.json();
        payloadByKey[key] = j.payload;
        matDone++;
        setMaterializeProgress((p) => ({ ...p, done: matDone }));
        // Mark this component's result row as materialized
        const id = `i:${key}`;
        setResults((prev) => ({
          ...prev,
          [id]: { ...prev[id], status: 'analyzing', payloads: { [key]: j.payload } },
        }));
      } catch (e) {
        const id = `i:${key}`;
        setResults((prev) => ({
          ...prev,
          [id]: { ...prev[id], status: 'error', error: `materialize failed: ${e.message}` },
        }));
        setRunError(`materialize failed for ${key}: ${e.message}`);
        setRunning(false);
        setPipelinePhase(null);
        return;
      }
    }

    // Attach payloads to pair/triple result rows for visualisation.
    setResults((prev) => {
      const next = { ...prev };
      Object.values(next).forEach((r) => {
        if (r.kind === 'pair' || r.kind === 'triple') {
          const p = {};
          r.components.forEach((k) => { if (payloadByKey[k]) p[k] = payloadByKey[k]; });
          next[r.id] = { ...r, payloads: p };
        }
      });
      return next;
    });

    // 3. Individual analyses (parallel). Transition to analyze phase.
    setPipelinePhase('analyze');
    const indivTextByKey = {};
    await Promise.all(indivIds.map(async ({ id, key, asset, item }) => {
      try {
        setResults((prev) => ({
          ...prev,
          [id]: { ...prev[id], status: 'analyzing' },
        }));
        const r = await fetch(`${API}/analyze`, {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            mode: 'individual',
            components: [{ key, asset, item, payload: payloadByKey[key] }],
            directive: directive || null,
            system_prompt: systemPromptOverride || null,
            model, effort,
          }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status} ${(await r.text()).slice(0, 200)}`);
        const j = await r.json();
        indivTextByKey[key] = j.analysis || '';
        setResults((prev) => ({
          ...prev,
          [id]: { ...prev[id], status: 'done', text: j.analysis || '',
                  thinking: j.thinking, tokens_in: j.tokens_in, tokens_out: j.tokens_out,
                  model: j.model, dataTruncation: j.data_truncation || null },
        }));
      } catch (e) {
        setResults((prev) => ({
          ...prev,
          [id]: { ...prev[id], status: 'error', error: String(e.message || e) },
        }));
      }
    }));

    // 4. Pair + triple analyses — wait until individuals resolved, then run.
    const buildPriorIndividual = (keys) =>
      keys.filter((k) => indivTextByKey[k]).map((k) => ({ key: k, text: indivTextByKey[k] }));

    const pairTasks = pairs.map(async ({ id, a, b }) => {
      try {
        setResults((prev) => ({ ...prev, [id]: { ...prev[id], status: 'analyzing' } }));
        const r = await fetch(`${API}/analyze`, {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            mode: 'pair',
            components: [a, b].map((k) => {
              const [asset, ...rest] = k.split('.');
              return { key: k, asset, item: rest.join('.'), payload: payloadByKey[k] };
            }),
            prior_individual: buildPriorIndividual([a, b]),
            directive: directive || null,
            system_prompt: systemPromptOverride || null,
            model, effort,
          }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status} ${(await r.text()).slice(0, 200)}`);
        const j = await r.json();
        setResults((prev) => ({
          ...prev,
          [id]: { ...prev[id], status: 'done', text: j.analysis || '',
                  thinking: j.thinking, tokens_in: j.tokens_in, tokens_out: j.tokens_out,
                  model: j.model, dataTruncation: j.data_truncation || null },
        }));
      } catch (e) {
        setResults((prev) => ({
          ...prev,
          [id]: { ...prev[id], status: 'error', error: String(e.message || e) },
        }));
      }
    });

    const tripleTasks = triples.map(async ({ id, a, b, c }) => {
      try {
        setResults((prev) => ({ ...prev, [id]: { ...prev[id], status: 'analyzing' } }));
        const r = await fetch(`${API}/analyze`, {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            mode: 'triple',
            components: [a, b, c].map((k) => {
              const [asset, ...rest] = k.split('.');
              return { key: k, asset, item: rest.join('.'), payload: payloadByKey[k] };
            }),
            prior_individual: buildPriorIndividual([a, b, c]),
            directive: directive || null,
            system_prompt: systemPromptOverride || null,
            model, effort,
          }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status} ${(await r.text()).slice(0, 200)}`);
        const j = await r.json();
        setResults((prev) => ({
          ...prev,
          [id]: { ...prev[id], status: 'done', text: j.analysis || '',
                  thinking: j.thinking, tokens_in: j.tokens_in, tokens_out: j.tokens_out,
                  model: j.model, dataTruncation: j.data_truncation || null },
        }));
      } catch (e) {
        setResults((prev) => ({
          ...prev,
          [id]: { ...prev[id], status: 'error', error: String(e.message || e) },
        }));
      }
    });

    await Promise.all([...pairTasks, ...tripleTasks]);
    setPipelinePhase('done');
    setRunning(false);
  }, [catalog, running, N, selected, includeTriples, windowSec, bucketSize, directive, systemPromptOverride, model, effort]);

  // ── render ──────────────────────────────────────────────────────────────
  const orderedResults = useMemo(() => {
    const arr = Object.values(results);
    const rank = { individual: 0, pair: 1, triple: 2 };
    arr.sort((a, b) => (rank[a.kind] - rank[b.kind]) || a.label.localeCompare(b.label));
    return arr;
  }, [results]);

  const active = activeId ? results[activeId] : null;

  return (
    <div style={{
      background: C.bg, color: C.text, minHeight: '85vh',
      borderRadius: 12, padding: 20, fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
    }}>
      <Header running={running} />

      {/* ── top control bar ──────────────────────────────────────────── */}
      <div style={{
        background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10,
        padding: 14, marginBottom: 16,
        display: 'grid', gap: 12,
        gridTemplateColumns: 'auto auto auto 1fr auto auto',
        alignItems: 'center',
      }}>
        <Group label="Window">
          <Segmented
            options={WINDOW_OPTIONS.map((o) => ({ label: o.label, value: o.value }))}
            value={windowSec} onChange={setWindowSec}
          />
        </Group>

        <Group label="Signal Bucket">
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <input
              type="text"
              value={bucketSize}
              onChange={(e) => setBucketSize(e.target.value)}
              placeholder="5s"
              title="Internal grouping interval for the DERIV Signal Buckets feature (e.g. 1s, 5s, 15s, 1m, 5m)"
              style={{
                width: 56, background: C.raised, color: C.text,
                border: `1px solid ${C.fuchsia}55`, borderRadius: 8,
                padding: '6px 8px', fontSize: 13, outline: 'none',
                fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                textAlign: 'center',
              }}
            />
            <span style={{
              fontSize: 9, color: C.fuchsia, fontWeight: 600,
              letterSpacing: '0.05em', textTransform: 'uppercase',
            }}>DERIV</span>
          </div>
        </Group>

        <Group label="Triples">
          <Toggle on={includeTriples} onChange={setIncludeTriples}
                  label={includeTriples ? 'on' : 'off'} />
        </Group>

        <Group label="Directive (optional)">
          <input
            type="text"
            value={directive}
            onChange={(e) => setDirective(e.target.value)}
            placeholder="e.g. focus on next 30-min reversion edge"
            style={{
              width: '100%', background: C.raised, color: C.text,
              border: `1px solid ${C.border}`, borderRadius: 8,
              padding: '8px 12px', fontSize: 13, outline: 'none',
            }}
          />
        </Group>

        <Group label="Model">
          <select
            value={model || ''}
            onChange={(e) => {
              const newModel = e.target.value;
              setModel(newModel);
              const info = catalog?.models?.find((m) => m.id === newModel);
              const provider = info?.provider || 'anthropic';
              const allowed = info?.effort_levels || EFFORT_OPTIONS;
              setEffort((prev) => {
                if (provider === 'anthropic') return prev === 'off' ? 'off' : 'high';
                return allowed.includes(prev) ? prev : (allowed[0] || 'off');
              });
            }}
            style={selectStyle}
          >
            {(() => {
              const models = catalog?.models || [];
              const openai = models.filter((m) => m.provider === 'openai');
              const anthropic = models.filter((m) => !m.provider || m.provider === 'anthropic');
              return (
                <>
                  {openai.length > 0 && (
                    <optgroup label="OpenAI (ChatGPT)">
                      {openai.map((m) => (
                        <option key={m.id} value={m.id}>{m.label}</option>
                      ))}
                    </optgroup>
                  )}
                  <optgroup label="Anthropic (Claude)">
                    {anthropic.map((m) => (
                      <option key={m.id} value={m.id}>{m.label}</option>
                    ))}
                  </optgroup>
                </>
              );
            })()}
          </select>
          {model !== 'gpt-5.5-instant' && (
            <button
              onClick={() => { setModel('gpt-5.5-instant'); setEffort('off'); }}
              style={{ ...miniBtnStyle, marginTop: 6, color: C.cyan, borderColor: `${C.cyan}55` }}
            >
              use GPT-5.5 instant
            </button>
          )}
        </Group>

        {/* System prompt viewer/editor — positioned under directive */}
        <div style={{ gridColumn: '4 / 5' }}>
          <button
            onClick={() => {
              const opening = !systemPromptExpanded;
              setSystemPromptExpanded(opening);
              if (opening) {
                setSystemPromptDraft(systemPromptOverride || defaultSystemPrompts.individual);
              }
            }}
            style={{
              background: 'transparent', border: `1px solid ${C.border}`,
              color: C.sub, padding: '5px 12px', borderRadius: 6,
              cursor: 'pointer', fontSize: 11, fontWeight: 600,
              letterSpacing: '0.04em', textTransform: 'uppercase',
              display: 'flex', alignItems: 'center', gap: 6,
            }}
          >
            <span style={{ color: C.dim, fontSize: 10 }}>{systemPromptExpanded ? '▾' : '▸'}</span>
            System Prompt
            {systemPromptOverride && (
              <span style={{
                fontSize: 9, padding: '1px 5px', borderRadius: 999,
                background: `${C.green}22`, color: C.green,
                border: `1px solid ${C.green}55`,
              }}>saved</span>
            )}
          </button>
          {systemPromptExpanded && (
            <div style={{ marginTop: 8 }}>
              <textarea
                value={systemPromptDraft}
                onChange={(e) => setSystemPromptDraft(e.target.value)}
                placeholder="System prompt will be loaded from the server..."
                style={{
                  width: '100%', minHeight: 120, resize: 'vertical',
                  background: C.raised, color: C.text,
                  border: `1px solid ${
                    systemPromptDraft !== (systemPromptOverride || defaultSystemPrompts.individual)
                      ? C.amber + '88' : C.border
                  }`,
                  borderRadius: 8, padding: '10px 12px',
                  fontSize: 12, outline: 'none',
                  fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                  lineHeight: 1.5,
                }}
              />
              <div style={{
                display: 'flex', alignItems: 'center', gap: 8, marginTop: 8,
              }}>
                <button
                  onClick={() => setSystemPromptDraft('')}
                  style={{
                    ...miniBtnStyle,
                    color: C.red, borderColor: `${C.red}55`,
                  }}
                >clear</button>
                <button
                  onClick={() => {
                    setSystemPromptDraft(defaultSystemPrompts.individual);
                    setSystemPromptOverride('');
                  }}
                  style={miniBtnStyle}
                >reset to default</button>
                <div style={{ flex: 1 }} />
                <button
                  onClick={() => {
                    const trimmed = systemPromptDraft.trim();
                    if (trimmed && trimmed !== defaultSystemPrompts.individual) {
                      setSystemPromptOverride(trimmed);
                    } else {
                      setSystemPromptOverride('');
                    }
                    setSystemPromptExpanded(false);
                  }}
                  style={{
                    background: `linear-gradient(135deg, ${C.emerald}, ${C.cyan})`,
                    color: '#0a0f1a', border: 'none',
                    padding: '5px 16px', borderRadius: 6,
                    cursor: 'pointer', fontSize: 11, fontWeight: 700,
                    letterSpacing: '0.04em', textTransform: 'uppercase',
                  }}
                >save</button>
              </div>
              <div style={{ fontSize: 10, color: C.dim, marginTop: 6 }}>
                {systemPromptOverride
                  ? 'Custom prompt saved — will be used for all calls in this run.'
                  : 'Showing the default individual prompt. Edit and save to override.'}
              </div>
            </div>
          )}
        </div>

        {/* Per-model reasoning note */}
        {model && catalog?.models && (() => {
          const info = catalog.models.find((m) => m.id === model);
          if (!info?.thinking_note) return null;
          return (
            <div style={{
              gridColumn: '1 / -1',
              fontSize: 11, color: C.dim,
              padding: '3px 10px', borderRadius: 6,
              background: C.raised, border: `1px solid ${C.border}`,
            }}>
              <span style={{ color: C.cyan, fontWeight: 700, marginRight: 6 }}>
                {info.tier === 'adaptive'
                  ? '⟳ Adaptive'
                  : info.tier === 'reasoning'
                    ? '⚡ Reasoning'
                    : info.tier === 'instant'
                      ? '⚡ Instant'
                      : '⬡ Extended'} thinking
              </span>
              {info.thinking_note}
            </div>
          );
        })()}

        {isClaudeModel ? (
          <Group label={isClaudeAdaptive ? 'Adaptive thinking' : 'Thinking'}>
            {(() => {
              const canEnable = (modelInfo?.effort_levels || []).some((e) => e !== 'off');
              return (
                <Toggle
                  on={canEnable && effort !== 'off'}
                  onChange={(on) => { if (canEnable) setEffort(on ? 'high' : 'off'); }}
                  label={canEnable ? (effort !== 'off' ? 'on' : 'off') : 'fixed off'}
                  disabled={!canEnable}
                />
              );
            })()}
          </Group>
        ) : showEffortSelector ? (
          <Group label="Reasoning effort">
            {(() => {
              const allowed = new Set(modelInfo?.effort_levels || EFFORT_OPTIONS);
              return (
                <Segmented
                  options={EFFORT_OPTIONS.map((e) => ({
                    label: e,
                    value: e,
                    disabled: !allowed.has(e),
                  }))}
                  value={effort} onChange={setEffort}
                />
              );
            })()}
          </Group>
        ) : (
          <Group label="Reasoning">
            <div style={{
              background: C.raised, color: C.sub, border: `1px solid ${C.border}`,
              borderRadius: 8, padding: '6px 10px', fontSize: 12, fontWeight: 600,
              letterSpacing: '0.04em', textTransform: 'uppercase',
            }}>
              off (instant)
            </div>
          </Group>
        )}

        {/* date range row */}
        <div style={{
          gridColumn: '1 / -1', display: 'flex', alignItems: 'flex-end',
          gap: 16, flexWrap: 'wrap',
        }}>
          <Group label="Start date">
            <input
              type="datetime-local"
              value={customStart}
              onChange={(e) => setCustomStart(e.target.value)}
              style={dateInputStyle}
            />
          </Group>
          <Group label="End date">
            {customEnd ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <input
                  type="datetime-local"
                  value={customEnd}
                  onChange={(e) => setCustomEnd(e.target.value)}
                  style={dateInputStyle}
                />
                <button onClick={() => setCustomEnd('')}
                  style={miniBtnStyle}>present</button>
              </div>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{
                  background: `${C.green}1a`, color: C.green,
                  border: `1px solid ${C.green}55`, borderRadius: 8,
                  padding: '6px 14px', fontSize: 12, fontWeight: 600,
                }}>Present</span>
                <button onClick={() => setCustomEnd(
                  new Date().toISOString().slice(0, 16)
                )} style={miniBtnStyle}>set</button>
              </div>
            )}
          </Group>
          {(customStart || customEnd) && (
            <button
              onClick={() => { setCustomStart(''); setCustomEnd(''); }}
              style={{ ...miniBtnStyle, alignSelf: 'flex-end', marginBottom: 2 }}
            >clear dates</button>
          )}
        </div>

        <div style={{ gridColumn: '1 / -1', display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap' }}>
          <CountPill label="Selected"     value={N}                color={C.cyan} />
          <CountPill label="Materialize"  value={totalMaterialize} color={C.blue} />
          <CountPill label="Individual"   value={N}                color={C.green} />
          <CountPill label="Pair"         value={nPair}            color={C.amber} />
          {includeTriples && (
            <CountPill label="Triple"     value={nTrip}            color={C.purple} />
          )}
          <CountPill label="Total LLM calls" value={totalAnalyze}  color={C.fuchsia} />

          <div style={{ flex: 1 }} />

          <button
            onClick={() => setShowCompute(true)}
            disabled={running || N === 0}
            style={{
              padding: '10px 22px', borderRadius: 10, border: 'none', cursor: 'pointer',
              fontSize: 14, fontWeight: 700, letterSpacing: '0.04em',
              color: '#0a0f1a',
              background: (running || N === 0)
                ? C.muted
                : `linear-gradient(135deg, ${C.emerald}, ${C.cyan})`,
              boxShadow: (running || N === 0) ? 'none' : `0 4px 16px ${C.emerald}40`,
              transition: 'all .15s',
            }}
          >
            {running ? 'Running…' : N === 0 ? 'Pick a component' : `Compute ${totalAnalyze} call${totalAnalyze === 1 ? '' : 's'}`}
          </button>
        </div>

        {runError && (
          <div style={{
            gridColumn: '1 / -1',
            color: C.red, background: '#3a0e1d', border: `1px solid ${C.red}`,
            padding: '8px 12px', borderRadius: 8, fontSize: 13,
          }}>{runError}</div>
        )}
      </div>

      {/* ── pipeline progress bar ──────────────────────────────────────── */}
      {running && pipelinePhase && (
        <PipelineProgress
          phase={pipelinePhase}
          materializeProgress={materializeProgress}
          results={results}
        />
      )}

      {/* ── two-column main layout ─────────────────────────────────────── */}
      <div style={{
        display: 'grid', gap: 16,
        gridTemplateColumns: 'minmax(320px, 380px) 1fr',
        alignItems: 'flex-start',
      }}>
        {/* LEFT: catalog tree */}
        <div style={{
          background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10,
          padding: 12, maxHeight: '78vh', overflowY: 'auto',
        }}>
          <SectionTitle>Catalog</SectionTitle>
          {!catalog && <div style={{ color: C.sub, padding: 10 }}>Loading…</div>}
          {catalog && catalog.assets.map((a) => (
            <CatalogAsset
              key={a.key}
              asset={a}
              groups={catalog.groups}
              selected={selected}
              toggle={toggle}
              openAssets={openAssets} toggleAsset={toggleAsset}
              openGroups={openGroups} toggleGroup={toggleGroup}
              selectAll={selectAllForAsset}
            />
          ))}
        </div>

        {/* RIGHT: response browser */}
        <div style={{
          background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10,
          padding: 14, minHeight: '60vh',
        }}>
          <ResponseBrowser
            results={orderedResults}
            activeId={activeId} setActiveId={setActiveId}
            active={active}
          />
        </div>
      </div>

      {/* ── Compute pre-flight modal ─────────────────────────────── */}
      {showCompute && (
        <ComputeModal
          selected={selected}
          windowSec={windowSec}
          bucketSize={bucketSize}
          customStart={customStart}
          customEnd={customEnd}
          includeTriples={includeTriples}
          model={model}
          effort={effort}
          directive={directive}
          catalog={catalog}
          onClose={() => setShowCompute(false)}
          onRun={(since, until) => runPipeline(since, until)}
        />
      )}
    </div>
  );
}


// ── Header ───────────────────────────────────────────────────────────────
function Header({ running }) {
  return (
    <div style={{ marginBottom: 14, display: 'flex', alignItems: 'flex-end', gap: 14 }}>
      <div>
        <h1 style={{
          margin: 0, fontSize: 28, fontWeight: 800, letterSpacing: '-0.02em',
          background: `linear-gradient(135deg, ${C.emerald}, ${C.cyan} 50%, ${C.fuchsia})`,
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
        }}>Strat Hub</h1>
        <div style={{ color: C.sub, fontSize: 13, marginTop: 2 }}>
          Multi-coin component composer · LLM-driven structural analysis
        </div>
      </div>
      <div style={{ flex: 1 }} />
      {running && (
        <div style={{
          color: C.green, fontSize: 12, letterSpacing: '0.1em', textTransform: 'uppercase',
          display: 'flex', alignItems: 'center', gap: 8,
        }}>
          <span style={{
            display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
            background: C.green, boxShadow: `0 0 12px ${C.green}`,
            animation: 'sthpulse 1.4s infinite',
          }} />
          working
          <style>{`@keyframes sthpulse{0%,100%{opacity:1}50%{opacity:.35}}`}</style>
        </div>
      )}
    </div>
  );
}

// ── Catalog asset block ──────────────────────────────────────────────────
function CatalogAsset({ asset, groups, selected, toggle,
                       openAssets, toggleAsset,
                       openGroups, toggleGroup, selectAll }) {
  const isOpen = openAssets.has(asset.key);
  const allKeys = useMemo(
    () => groups.flatMap((g) => g.items.map((it) => `${asset.key}.${it.key}`)),
    [groups, asset.key]
  );
  const selCount = allKeys.filter((k) => selected.has(k)).length;

  return (
    <div style={{ marginBottom: 8 }}>
      <div
        onClick={() => toggleAsset(asset.key)}
        style={{
          display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer',
          padding: '8px 10px', borderRadius: 8,
          background: C.raised, border: `1px solid ${C.border}`,
        }}
      >
        <span style={{
          display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
          background: ASSET_COLOR[asset.key] || C.cyan,
          boxShadow: `0 0 8px ${ASSET_COLOR[asset.key] || C.cyan}80`,
        }} />
        <div style={{ fontWeight: 700, color: C.text, fontSize: 14 }}>{asset.label}</div>
        <div style={{ fontSize: 11, color: C.dim }}>
          {selCount}/{allKeys.length} selected
        </div>
        <div style={{ flex: 1 }} />
        <button onClick={(e) => { e.stopPropagation(); selectAll(asset.key, true); }}
                style={miniBtnStyle}>all</button>
        <button onClick={(e) => { e.stopPropagation(); selectAll(asset.key, false); }}
                style={miniBtnStyle}>clear</button>
        <span style={{ color: C.dim, fontSize: 12 }}>{isOpen ? '▾' : '▸'}</span>
      </div>
      {isOpen && (
        <div style={{ paddingLeft: 6, marginTop: 4 }}>
          {groups.map((g) => {
            const gid = `${asset.key}:${g.key}`;
            const gOpen = openGroups.has(gid);
            return (
              <div key={g.key} style={{ margin: '6px 0' }}>
                <div
                  onClick={() => toggleGroup(gid)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 6,
                    cursor: 'pointer', padding: '4px 8px',
                    color: C.sub, fontSize: 12,
                    fontWeight: 600, letterSpacing: '0.04em',
                    textTransform: 'uppercase',
                  }}
                >
                  <span style={{ color: C.dim }}>{gOpen ? '▾' : '▸'}</span>
                  {g.label}
                </div>
                {gOpen && (
                  <div style={{ paddingLeft: 14 }}>
                    {g.items.map((it) => {
                      const k = `${asset.key}.${it.key}`;
                      const on = selected.has(k);
                      return (
                        <label
                          key={k}
                          style={{
                            display: 'flex', alignItems: 'center', gap: 8,
                            padding: '5px 8px', borderRadius: 6, cursor: 'pointer',
                            background: on ? `${ASSET_COLOR[asset.key]}1a` : 'transparent',
                            border: `1px solid ${on ? ASSET_COLOR[asset.key] + '55' : 'transparent'}`,
                            margin: '2px 0',
                          }}
                        >
                          <input
                            type="checkbox" checked={on}
                            onChange={() => toggle(k)}
                            style={{ accentColor: ASSET_COLOR[asset.key] || C.cyan }}
                          />
                          <span style={{ flex: 1, color: C.text, fontSize: 13 }}>{it.label}</span>
                          <Badge stored={it.stored} />
                        </label>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Badge({ stored }) {
  return stored ? (
    <span style={{
      fontSize: 10, padding: '2px 6px', borderRadius: 999,
      background: `${C.green}1a`, color: C.green, border: `1px solid ${C.green}55`,
    }}>● stored</span>
  ) : (
    <span style={{
      fontSize: 10, padding: '2px 6px', borderRadius: 999,
      background: `${C.amber}1a`, color: C.amber, border: `1px solid ${C.amber}55`,
    }}>⚠ on-cmd</span>
  );
}

// ── Response Browser ─────────────────────────────────────────────────────
function ResponseBrowser({ results, activeId, setActiveId, active }) {
  if (!results.length) {
    return (
      <div style={{ color: C.dim, padding: 30, textAlign: 'center' }}>
        Pick components on the left, then hit Run.
        <div style={{ marginTop: 6, fontSize: 12 }}>
          Individuals run first; pair / triple comparisons re-use the per-component
          analyses as prior context.
        </div>
      </div>
    );
  }

  return (
    <div>
      <ResponseSwitcher
        results={results} activeId={activeId} setActiveId={setActiveId}
      />
      <div style={{ marginTop: 14 }}>
        {active ? <ResponseCard result={active} /> : (
          <div style={{ color: C.dim, padding: 24 }}>Pick a response from above.</div>
        )}
      </div>
    </div>
  );
}

function ResponseSwitcher({ results, activeId, setActiveId }) {
  const groups = { individual: [], pair: [], triple: [] };
  results.forEach((r) => groups[r.kind].push(r));

  const renderRow = (label, arr, accent) => (
    arr.length > 0 && (
      <div style={{ marginBottom: 6 }}>
        <div style={{
          fontSize: 10, color: C.dim, letterSpacing: '0.1em',
          textTransform: 'uppercase', marginBottom: 4,
        }}>{label}</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {arr.map((r) => {
            const isActive = r.id === activeId;
            const stColor = r.status === 'done' ? C.green
              : r.status === 'error' ? C.red
              : r.status === 'analyzing' ? C.cyan
              : r.status === 'materializing' ? C.blue
              : C.amber;
            return (
              <button
                key={r.id}
                onClick={() => setActiveId(r.id)}
                title={r.label}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '5px 9px', borderRadius: 999,
                  background: isActive ? `${accent}22` : C.raised,
                  border: `1px solid ${isActive ? accent : C.border}`,
                  color: isActive ? accent : C.text,
                  fontSize: 12, fontWeight: 600, cursor: 'pointer',
                  maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                <span style={{
                  width: 8, height: 8, borderRadius: '50%', background: stColor,
                  animation: (r.status === 'pending' || r.status === 'materializing' || r.status === 'analyzing')
                    ? 'sthpulse 1.4s infinite' : 'none',
                }} />
                <span style={{
                  fontFamily: 'ui-monospace, SFMono-Regular, monospace', fontSize: 11.5,
                }}>{r.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    )
  );

  return (
    <div style={{
      borderBottom: `1px solid ${C.border}`, paddingBottom: 8,
    }}>
      {renderRow('Individual', groups.individual, C.green)}
      {renderRow('Pair', groups.pair, C.amber)}
      {renderRow('Triple', groups.triple, C.purple)}
    </div>
  );
}

// ── Response Card ────────────────────────────────────────────────────────
function ResponseCard({ result }) {
  const [showThinking, setShowThinking] = useState(false);

  const accent = result.kind === 'individual' ? C.green
    : result.kind === 'pair' ? C.amber : C.purple;

  const elapsed = result.startedAt
    ? ((Date.now() - result.startedAt) / 1000).toFixed(1)
    : null;

  return (
    <div>
      {/* header strip */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12,
      }}>
        <div style={{
          padding: '4px 10px', borderRadius: 999,
          background: `${accent}22`, color: accent,
          fontWeight: 700, fontSize: 11, letterSpacing: '0.1em',
          textTransform: 'uppercase',
          border: `1px solid ${accent}55`,
        }}>{result.kind}</div>
        <div style={{
          fontFamily: 'ui-monospace, SFMono-Regular, monospace', fontSize: 13,
          color: C.text, fontWeight: 600,
        }}>{result.label}</div>
        <div style={{ flex: 1 }} />
        {result.model && (
          <div style={{ fontSize: 11, color: C.dim }}>{result.model}</div>
        )}
        {result.text && (
          <button
            onClick={() => navigator.clipboard?.writeText(result.text)}
            style={miniBtnStyle}
          >copy</button>
        )}
      </div>

      {/* sparkline strip per component */}
      {result.payloads && Object.keys(result.payloads).length > 0 && (
        <div style={{
          display: 'grid', gap: 10, marginBottom: 14,
          gridTemplateColumns: result.components.some(
            (k) => result.payloads[k]?.kind === 'signal_buckets'
          ) ? '1fr'
            : result.components.length === 1 ? '1fr'
            : result.components.length === 2 ? '1fr 1fr' : '1fr 1fr 1fr',
        }}>
          {result.components.map((k) => (
            <ComponentSparkline key={k} ckey={k} payload={result.payloads[k]} />
          ))}
        </div>
      )}

      {/* data coverage banner */}
      {result.status === 'done' && <DataCoverageBanner truncation={result.dataTruncation} />}

      {/* analysis box (scrollable) */}
      <div style={{
        background: C.bg, border: `1px solid ${C.border}`,
        borderLeft: `3px solid ${accent}`,
        borderRadius: 10, padding: '16px 18px',
        maxHeight: '52vh', overflowY: 'auto',
      }}>
        {(result.status === 'pending' || result.status === 'materializing' || result.status === 'analyzing') && (
          <PendingBlock status={result.status} />
        )}
        {result.status === 'error' && (
          <div style={{ color: C.red, fontSize: 13 }}>
            <div style={{ fontWeight: 700, marginBottom: 6 }}>Error</div>
            <code style={{ whiteSpace: 'pre-wrap' }}>{result.error}</code>
          </div>
        )}
        {result.status === 'done' && (
          <div>
            {result.text ? renderClaudeText(result.text) : (
              <div style={{
                background: `${C.amber}1a`, border: `1px solid ${C.amber}55`,
                borderRadius: 8, padding: '12px 14px', marginBottom: 10,
              }}>
                <div style={{ color: C.amber, fontWeight: 700, fontSize: 13, marginBottom: 4 }}>
                  No response text generated
                </div>
                <div style={{ color: C.sub, fontSize: 12, lineHeight: 1.5 }}>
                  The model produced thinking but no visible output. This usually means the input
                  data exceeded the model's context window — try a shorter lookback window or
                  larger bucket size to reduce the number of rows sent.
                  {result.tokens_in > 0 && (
                    <span style={{ color: C.dim, marginLeft: 6 }}>
                      (input was ~{fmtCount(result.tokens_in)} tokens)
                    </span>
                  )}
                </div>
              </div>
            )}
            {result.thinking && (
              <div style={{ marginTop: 16, borderTop: `1px dashed ${C.border}`, paddingTop: 10 }}>
                <button
                  onClick={() => setShowThinking((s) => !s)}
                  style={miniBtnStyle}
                >
                  {showThinking ? 'hide' : 'show'} extended thinking
                </button>
                {showThinking && (
                  <pre style={{
                    marginTop: 8, color: C.sub, fontSize: 11, lineHeight: 1.55,
                    background: C.raised, padding: 10, borderRadius: 6,
                    whiteSpace: 'pre-wrap', maxHeight: 220, overflowY: 'auto',
                  }}>{result.thinking}</pre>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* footer with token cost */}
      <div style={{
        marginTop: 10, color: C.dim, fontSize: 11,
        display: 'flex', gap: 14, flexWrap: 'wrap',
      }}>
        <span>tokens in: <strong style={{ color: C.sub }}>{fmtCount(result.tokens_in)}</strong></span>
        <span>tokens out: <strong style={{ color: C.sub }}>{fmtCount(result.tokens_out)}</strong></span>
        {result.kind !== 'individual' && (
          <span>prior individuals embedded: <strong style={{ color: C.sub }}>
            {result.components.length}
          </strong></span>
        )}
        {elapsed && <span>elapsed: <strong style={{ color: C.sub }}>{elapsed}s</strong></span>}
      </div>
    </div>
  );
}

function DataCoverageBanner({ truncation }) {
  if (!truncation || truncation.length === 0) return null;

  const anyTruncated = truncation.some((t) => t.truncated);

  if (!anyTruncated) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '6px 12px', marginBottom: 8,
        background: `${C.green}0d`, border: `1px solid ${C.green}33`,
        borderRadius: 8,
      }}>
        <span style={{ color: C.green, fontSize: 13 }}>✓</span>
        <span style={{ fontSize: 11, color: C.green, fontWeight: 600 }}>
          Full data sent to model
        </span>
        {truncation.map((t) => (
          <span key={t.key} style={{
            fontSize: 10, color: C.sub,
            fontFamily: 'ui-monospace, SFMono-Regular, monospace',
          }}>
            {t.key}: {t.total_rows} rows
          </span>
        ))}
      </div>
    );
  }

  return (
    <div style={{
      padding: '8px 12px', marginBottom: 8,
      background: `${C.amber}12`, border: `1px solid ${C.amber}44`,
      borderRadius: 8,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <span style={{ color: C.amber, fontSize: 13 }}>⚠</span>
        <span style={{ fontSize: 11, color: C.amber, fontWeight: 700 }}>
          Data truncated to fit context window
        </span>
      </div>
      {truncation.filter((t) => t.truncated).map((t) => (
        <div key={t.key} style={{
          fontSize: 11, color: C.sub, marginLeft: 21,
          fontFamily: 'ui-monospace, SFMono-Regular, monospace',
        }}>
          {t.key}: {t.sent_rows} of {t.total_rows} rows sent (last {t.sent_rows} + full-range stats)
        </div>
      ))}
      <div style={{ fontSize: 10, color: C.dim, marginTop: 4, marginLeft: 21 }}>
        Use a shorter window or larger bucket size for full coverage
      </div>
    </div>
  );
}

function PendingBlock({ status }) {
  const color = status === 'materializing' ? C.blue
    : status === 'analyzing' ? C.cyan : C.amber;
  const text = status === 'materializing' ? 'Materializing from Postgres…'
    : status === 'analyzing' ? 'Analyzing with LLM…'
    : 'Queued…';
  return (
    <div style={{ color, fontSize: 13, display: 'flex', alignItems: 'center', gap: 10 }}>
      <span style={{
        width: 10, height: 10, borderRadius: '50%', background: color,
        animation: 'sthpulse 1.4s infinite',
      }} />
      {text}
    </div>
  );
}

// ── Sparkline visualization for a single component ──────────────────────
function ComponentSparkline({ ckey, payload }) {
  const [asset, ...rest] = ckey.split('.');
  const item = rest.join('.');
  const color = ASSET_COLOR[asset] || C.cyan;

  const series = payload ? extractChartSeries(payload) : null;
  const summary = payload?.summary || {};
  const signalChannels = payload ? extractSignalBucketChannels(payload) : null;

  // Volume profile = bar chart instead of line
  const isVolProfile = item === 'volume_profile';
  const profileData = isVolProfile && payload?.series
    ? payload.series.map((s) => ({
        px: ((s.px_lo + s.px_hi) / 2).toFixed(2),
        v: s.v_total || 0,
      }))
    : null;

  // Signal buckets get a multi-channel grid
  if (signalChannels) {
    return (
      <div style={{
        background: C.raised, border: `1px solid ${C.border}`, borderRadius: 8,
        padding: '10px 12px',
      }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8,
        }}>
          <span style={{
            width: 6, height: 6, borderRadius: '50%', background: color,
          }} />
          <div style={{
            fontFamily: 'ui-monospace, SFMono-Regular, monospace', fontSize: 11,
            color: C.text, fontWeight: 600,
          }}>{ckey}</div>
          <div style={{ flex: 1 }} />
          <span style={{
            fontSize: 10, padding: '2px 6px', borderRadius: 4,
            background: `${C.fuchsia}1a`, color: C.fuchsia, border: `1px solid ${C.fuchsia}55`,
          }}>
            {summary.buckets || 0} buckets · {summary.bucket_s || 0}s
          </span>
        </div>
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
          gap: 6,
        }}>
          {signalChannels.map((ch) => (
            <div key={ch.key} style={{
              background: C.bg, border: `1px solid ${C.border}`, borderRadius: 6,
              padding: '4px 6px',
            }}>
              <div style={{ fontSize: 9, color: ch.color, fontWeight: 600, marginBottom: 2 }}>
                {ch.label}
              </div>
              <div style={{ height: 36 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={ch.data} margin={{ top: 1, right: 0, left: 0, bottom: 0 }}>
                    <Area type="monotone" dataKey="v" stroke={ch.color} strokeWidth={1.2}
                          fill={ch.color} fillOpacity={0.15} dot={false} isAnimationActive={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}
        </div>
        {/* summary chips */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
          {Object.entries(summary).slice(0, 6).map(([k, v]) => (
            <span key={k} style={{
              fontSize: 10.5, color: C.sub,
              background: C.bg, border: `1px solid ${C.border}`,
              padding: '2px 6px', borderRadius: 4,
              fontFamily: 'ui-monospace, SFMono-Regular, monospace',
            }}>
              <span style={{ color: C.dim }}>{k}:</span>{' '}
              {typeof v === 'number' ? fmt(v, 4) : String(v).slice(0, 16)}
            </span>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div style={{
      background: C.raised, border: `1px solid ${C.border}`, borderRadius: 8,
      padding: '10px 12px',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6,
      }}>
        <span style={{
          width: 6, height: 6, borderRadius: '50%', background: color,
        }} />
        <div style={{
          fontFamily: 'ui-monospace, SFMono-Regular, monospace', fontSize: 11,
          color: C.text, fontWeight: 600,
        }}>{ckey}</div>
        <div style={{ flex: 1 }} />
        {payload?.window?.since && (
          <div style={{ fontSize: 10, color: C.dim }}>
            {new Date(payload.window.since).toLocaleTimeString([],
              { hour: '2-digit', minute: '2-digit' })}
            {' → '}
            {new Date(payload.window.until).toLocaleTimeString([],
              { hour: '2-digit', minute: '2-digit' })}
          </div>
        )}
      </div>

      <div style={{ height: 70, marginTop: 4 }}>
        <ResponsiveContainer width="100%" height="100%">
          {isVolProfile && profileData ? (
            <BarChart data={profileData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
              <Bar dataKey="v" fill={color} />
              <Tooltip
                contentStyle={{
                  background: C.bg, border: `1px solid ${C.border}`,
                  fontSize: 11, color: C.text,
                }}
                labelStyle={{ color: C.sub }}
                formatter={(v) => [fmt(v, 2), 'volume']}
                labelFormatter={(l) => `px ${l}`}
              />
            </BarChart>
          ) : series ? (
            <AreaChart data={series.data} margin={{ top: 2, right: 0, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id={`g_${asset}_${item}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.45} />
                  <stop offset="100%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <Area type="monotone" dataKey="v" stroke={color} strokeWidth={1.4}
                    fill={`url(#g_${asset}_${item})`} dot={false} isAnimationActive={false} />
              <Tooltip
                contentStyle={{
                  background: C.bg, border: `1px solid ${C.border}`,
                  fontSize: 11, color: C.text,
                }}
                labelStyle={{ color: C.sub }}
                formatter={(v) => [fmt(v, 4), series.key]}
                labelFormatter={(l) =>
                  series.data[l]?.ts
                    ? new Date(series.data[l].ts).toLocaleTimeString()
                    : ''
                }
              />
            </AreaChart>
          ) : (
            <div style={{
              width: '100%', height: '100%',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: C.dim, fontSize: 11,
            }}>no plottable series</div>
          )}
        </ResponsiveContainer>
      </div>

      {/* summary chips */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
        {Object.entries(summary).slice(0, 5).map(([k, v]) => (
          <span key={k} style={{
            fontSize: 10.5, color: C.sub,
            background: C.bg, border: `1px solid ${C.border}`,
            padding: '2px 6px', borderRadius: 4,
            fontFamily: 'ui-monospace, SFMono-Regular, monospace',
          }}>
            <span style={{ color: C.dim }}>{k}:</span>{' '}
            {typeof v === 'number' ? fmt(v, 4) : String(v).slice(0, 16)}
          </span>
        ))}
      </div>
    </div>
  );
}


// ── Pipeline Progress Bar ────────────────────────────────────────────────

function PipelineProgress({ phase, materializeProgress, results }) {
  const [elapsed, setElapsed] = useState(0);
  const startedAt = materializeProgress?.startedAt;

  useEffect(() => {
    if (!startedAt) return;
    const iv = setInterval(() => setElapsed(((Date.now() - startedAt) / 1000)), 500);
    return () => clearInterval(iv);
  }, [startedAt]);

  const allResults = Object.values(results);
  const doneCount = allResults.filter((r) => r.status === 'done').length;
  const errorCount = allResults.filter((r) => r.status === 'error').length;
  const analyzingCount = allResults.filter((r) => r.status === 'analyzing').length;
  const totalCount = allResults.length;

  const matPct = materializeProgress.total > 0
    ? (materializeProgress.done / materializeProgress.total) * 100
    : 0;

  return (
    <div style={{
      background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10,
      padding: '12px 16px', marginBottom: 16,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <span style={{
          width: 10, height: 10, borderRadius: '50%',
          background: phase === 'done' ? C.green : C.cyan,
          boxShadow: phase === 'done' ? 'none' : `0 0 10px ${C.cyan}`,
          animation: phase === 'done' ? 'none' : 'sthpulse 1.4s infinite',
        }} />
        <div style={{ fontSize: 13, fontWeight: 700, color: C.text }}>
          {phase === 'materialize' && 'Materializing data from Postgres…'}
          {phase === 'analyze' && 'Analyzing with LLM…'}
          {phase === 'done' && 'Pipeline complete'}
        </div>
        <div style={{ flex: 1 }} />
        <div style={{
          fontFamily: 'ui-monospace, SFMono-Regular, monospace',
          fontSize: 12, color: C.sub,
        }}>
          {elapsed.toFixed(1)}s elapsed
        </div>
      </div>

      {/* Materialize progress bar */}
      {phase === 'materialize' && (
        <div>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6,
          }}>
            <div style={{ flex: 1, height: 6, background: C.raised, borderRadius: 3, overflow: 'hidden' }}>
              <div style={{
                width: `${matPct}%`, height: '100%',
                background: `linear-gradient(90deg, ${C.emerald}, ${C.cyan})`,
                borderRadius: 3, transition: 'width 0.3s ease',
              }} />
            </div>
            <span style={{ fontSize: 11, color: C.sub, fontFamily: 'ui-monospace, monospace' }}>
              {materializeProgress.done}/{materializeProgress.total}
            </span>
          </div>
          <div style={{ fontSize: 11, color: C.dim }}>
            Current: <span style={{ color: C.cyan, fontFamily: 'ui-monospace, monospace' }}>
              {materializeProgress.current}
            </span>
          </div>
        </div>
      )}

      {/* Analyze progress summary */}
      {phase === 'analyze' && (
        <div style={{ display: 'flex', gap: 12, fontSize: 11, color: C.sub }}>
          <span>
            <span style={{ color: C.green, fontWeight: 700 }}>{doneCount}</span> done
          </span>
          <span>
            <span style={{ color: C.cyan, fontWeight: 700 }}>{analyzingCount}</span> analyzing
          </span>
          {errorCount > 0 && (
            <span>
              <span style={{ color: C.red, fontWeight: 700 }}>{errorCount}</span> failed
            </span>
          )}
          <span>
            <span style={{ color: C.dim }}>{totalCount}</span> total
          </span>
        </div>
      )}

      {phase === 'done' && (
        <div style={{ display: 'flex', gap: 12, fontSize: 11, color: C.sub }}>
          <span style={{ color: C.green }}>
            {doneCount} completed
          </span>
          {errorCount > 0 && (
            <span style={{ color: C.red }}>
              {errorCount} failed
            </span>
          )}
        </div>
      )}
    </div>
  );
}


// ── Compute Modal ────────────────────────────────────────────────────────

function ComputeModal({ selected, windowSec, bucketSize, customStart, customEnd,
                        includeTriples, model, effort, directive,
                        catalog, onClose, onRun }) {
  const [phase, setPhase] = useState('loading'); // 'loading' | 'data-range' | 'preflight' | 'done' | 'error'
  const [loadingStep, setLoadingStep] = useState('data-range');
  const [alignment, setAlignment] = useState(null);
  const [contextChecks, setContextChecks] = useState([]);
  const [pfMeta, setPfMeta] = useState(null);
  const [indivDetail, setIndivDetail] = useState([]);
  const [error, setError] = useState(null);
  const [loadingStartedAt, setLoadingStartedAt] = useState(Date.now());
  const [loadingElapsed, setLoadingElapsed] = useState(0);
  const [abortCtrl, setAbortCtrl] = useState(null);

  // Elapsed timer during loading
  useEffect(() => {
    if (phase !== 'loading') return;
    const iv = setInterval(() => setLoadingElapsed(((Date.now() - loadingStartedAt) / 1000)), 500);
    return () => clearInterval(iv);
  }, [phase, loadingStartedAt]);

  const selectedArr = useMemo(() => [...selected], [selected]);
  const N = selectedArr.length;
  const nPair = N >= 2 ? (N * (N - 1)) / 2 : 0;
  const nTrip = includeTriples && N >= 3 ? (N * (N - 1) * (N - 2)) / 6 : 0;
  const totalCalls = N + nPair + nTrip;

  const hasSignalBuckets = selectedArr.some((k) => k.endsWith('.signal_buckets'));

  const modelInfo = catalog?.models?.find((m) => m.id === model);
  const contextWindow = modelInfo?.context_window || 200_000;

  useEffect(() => {
    let dead = false;
    const ctrl = new AbortController();
    setAbortCtrl(ctrl);
    setLoadingStartedAt(Date.now());

    (async () => {
      setPhase('loading');
      setLoadingStep('data-range');
      setError(null);
      try {
        const drRes = await fetch(`${API}/data-range`, {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ components: selectedArr }),
          signal: ctrl.signal,
        });
        const data = await drRes.json();
        if (!drRes.ok) {
          const raw = data?.detail;
          const ds = Array.isArray(raw)
            ? raw.map((x) => x.msg ?? JSON.stringify(x)).join('; ')
            : (typeof raw === 'string' ? raw : null);
          throw new Error(ds || `HTTP ${drRes.status}`);
        }
        if (dead) return;
        const ranges = data.ranges || {};

        const validRanges = selectedArr
          .map((k) => ({ key: k, ...(ranges[k] || {}) }))
          .filter((r) => r.earliest && r.latest);

        let alignedSince, alignedUntil;
        let allAligned = true;
        let noData = [];

        const now = new Date();
        alignedUntil = customEnd
          ? new Date(customEnd).toISOString()
          : now.toISOString();

        if (customStart) {
          alignedSince = new Date(customStart).toISOString();
        } else if (windowSec === 'all') {
          if (validRanges.length > 0) {
            const latestEarliest = Math.max(
              ...validRanges.map((r) => new Date(r.earliest).getTime())
            );
            alignedSince = new Date(latestEarliest).toISOString();
            const earliestOfAll = Math.min(
              ...validRanges.map((r) => new Date(r.earliest).getTime())
            );
            allAligned = latestEarliest === earliestOfAll;
          } else {
            alignedSince = now.toISOString();
          }
        } else {
          alignedSince = new Date(
            new Date(alignedUntil).getTime() - (windowSec || 900) * 1000
          ).toISOString();
        }

        noData = validRanges.filter(
          (r) => new Date(r.latest).getTime() < new Date(alignedSince).getTime()
        );

        if (windowSec !== 'all' && !customStart) {
          const missing = validRanges.filter(
            (r) => new Date(r.earliest).getTime() > new Date(alignedSince).getTime()
          );
          allAligned = missing.length === 0;
        }

        const al = {
          since: alignedSince,
          until: alignedUntil,
          allAligned,
          noData: noData.map((r) => r.key),
          componentRanges: validRanges,
        };
        setAlignment(al);
        if (dead) return;

        setLoadingStep('preflight');
        const pfRes = await fetch(`${API}/preflight`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            components: selectedArr,
            since: alignedSince,
            until: alignedUntil,
            directive: directive || null,
            include_triples: includeTriples,
            model,
            effort,
            max_tokens: 4000,
            bucket_s: hasSignalBuckets ? parseBucketSize(bucketSize) : undefined,
          }),
          signal: ctrl.signal,
        });
        const pf = await pfRes.json();
        if (!pfRes.ok) {
          const raw = pf?.detail;
          const detailStr = Array.isArray(raw)
            ? raw.map((x) => x.msg ?? JSON.stringify(x)).join('; ')
            : (typeof raw === 'string' ? raw : null);
          throw new Error(detailStr || pf.message || pfRes.statusText || `HTTP ${pfRes.status}`);
        }
        if (dead) return;

        setPfMeta({
          calibration_chars_per_token: pf.calibration_chars_per_token,
          max_tokens_reserved: pf.max_tokens_reserved,
          input_token_budget: pf.input_token_budget,
          pair_triple_prior_note: pf.pair_triple_prior_note,
          combo_max_input_tokens: pf.combo_max_input_tokens,
          preflight_notes: Array.isArray(pf.preflight_notes) ? pf.preflight_notes : [],
          anthropic_count_tokens_text_bytes_limit:
            pf.anthropic_count_tokens_text_bytes_limit ?? null,
        });
        setIndivDetail(pf.individual_detail || []);
        setContextChecks(Array.isArray(pf.checks) ? pf.checks : []);
        setPhase('done');
      } catch (e) {
        if (!dead) {
          if (e.name === 'AbortError') {
            setError('Cancelled by user');
          } else {
            setError(e.message || String(e));
          }
          setPhase('error');
        }
      }
    })();
    return () => { dead = true; ctrl.abort(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- modal snapshot on open
  }, []);

  const hasNoData = alignment?.noData?.length > 0;
  const allContextOk = contextChecks.every((c) => c.ok);
  const canRun = phase === 'done' && !hasNoData && allContextOk;

  const reservedOut = pfMeta?.max_tokens_reserved ?? 4000;
  const inputBudget = pfMeta?.input_token_budget ?? Math.max(0, contextWindow - reservedOut);
  const countTokensTextBytesCap =
    pfMeta?.anthropic_count_tokens_text_bytes_limit ?? 32_000_000;

  const maxMeasuredTokens = contextChecks.length > 0
    ? Math.max(...contextChecks.map((c) => c.tokens_per_call_max ?? 0))
    : 0;

  const alternativeModels = !allContextOk && catalog?.models
    ? catalog.models.filter((m) => {
        const cw = (m.context_window || 0);
        const budget = cw - reservedOut;
        return budget > maxMeasuredTokens;
      })
    : [];

  const canClose = phase !== 'loading';

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 9999,
      background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }} onClick={canClose ? onClose : undefined}>
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: C.surface, border: `1px solid ${C.border}`, borderRadius: 14,
          padding: 24, width: '90%', maxWidth: 640, maxHeight: '85vh',
          overflowY: 'auto', color: C.text,
        }}
      >
        {/* header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          marginBottom: 18,
        }}>
          <div style={{ fontSize: 18, fontWeight: 700 }}>Pre-flight Check</div>
          <button onClick={canClose ? onClose : undefined} disabled={!canClose} style={{
            background: 'transparent', border: 'none',
            color: canClose ? C.sub : C.muted,
            fontSize: 22, cursor: canClose ? 'pointer' : 'not-allowed', lineHeight: 1,
          }}>&times;</button>
        </div>

        {phase === 'loading' && (
          <div style={{ padding: 20 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
              <span style={{
                display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
                background: C.cyan, animation: 'sthpulse 1.4s infinite',
                boxShadow: `0 0 10px ${C.cyan}`,
              }} />
              <span style={{ color: C.text, fontWeight: 600, fontSize: 14 }}>
                {loadingStep === 'data-range' && 'Checking data availability…'}
                {loadingStep === 'preflight' && 'Materializing + counting tokens…'}
              </span>
              <div style={{ flex: 1 }} />
              <span style={{
                fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                fontSize: 12, color: C.sub,
              }}>
                {loadingElapsed.toFixed(1)}s
              </span>
            </div>

            {/* Step indicators */}
            <div style={{
              display: 'flex', gap: 8, marginBottom: 16,
            }}>
              <StepPill active={loadingStep === 'data-range'} done={loadingStep === 'preflight'} label="1. Data range" />
              <StepPill active={loadingStep === 'preflight'} done={false} label="2. Materialize + tokens" />
            </div>

            {hasSignalBuckets && loadingStep === 'preflight' && (
              <div style={{
                background: `${C.amber}1a`, border: `1px solid ${C.amber}55`,
                borderRadius: 8, padding: '8px 12px', marginBottom: 12,
                fontSize: 11, color: C.amber,
              }}>
                Signal Buckets with large windows can take 30-120s to materialize.
                The server is processing millions of L2 rows.
              </div>
            )}

            <div style={{ textAlign: 'center', marginTop: 16 }}>
              <button
                onClick={() => { if (abortCtrl) abortCtrl.abort(); onClose(); }}
                style={{
                  padding: '8px 18px', borderRadius: 8,
                  background: 'transparent', color: C.red,
                  border: `1px solid ${C.red}55`, cursor: 'pointer',
                  fontSize: 12, fontWeight: 600,
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {phase === 'error' && (
          <div style={{
            color: C.red, background: '#3a0e1d', border: `1px solid ${C.red}`,
            padding: '10px 14px', borderRadius: 8, fontSize: 13,
          }}>Failed: {error}</div>
        )}

        {phase === 'done' && (
          <>
            {/* ── Time Alignment ─────────────────────────────── */}
            <div style={{ marginBottom: 18 }}>
              <SectionTitle>Time Alignment</SectionTitle>
              <div style={{
                background: C.bg, border: `1px solid ${C.border}`, borderRadius: 10,
                padding: 14,
              }}>
                {alignment?.allAligned && !hasNoData && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: C.green }}>
                    <span style={{ fontSize: 16 }}>&#10003;</span>
                    <span style={{ fontWeight: 600 }}>
                      All {N} component{N > 1 ? 's' : ''} aligned
                    </span>
                  </div>
                )}
                {!alignment?.allAligned && !hasNoData && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: C.amber, marginBottom: 10 }}>
                    <span style={{ fontSize: 14 }}>&#9888;</span>
                    <span style={{ fontWeight: 600 }}>
                      Aligned to least common denominator
                    </span>
                  </div>
                )}
                {hasNoData && (
                  <div style={{ color: C.red, fontWeight: 600, marginBottom: 10 }}>
                    &#10007; Some components have no data in this range:
                    <div style={{ fontWeight: 400, fontSize: 12, marginTop: 4 }}>
                      {alignment.noData.join(', ')}
                    </div>
                  </div>
                )}

                {alignment?.componentRanges?.length > 0 && (
                  <div style={{ marginTop: 10, fontSize: 12 }}>
                    <div style={{
                      display: 'grid', gridTemplateColumns: '1fr auto auto',
                      gap: '4px 12px', color: C.sub,
                    }}>
                      <div style={{ fontWeight: 700, color: C.dim, fontSize: 10, textTransform: 'uppercase' }}>Component</div>
                      <div style={{ fontWeight: 700, color: C.dim, fontSize: 10, textTransform: 'uppercase' }}>Earliest</div>
                      <div style={{ fontWeight: 700, color: C.dim, fontSize: 10, textTransform: 'uppercase' }}>Latest</div>
                      {alignment.componentRanges.map((r) => (
                        [
                          <div key={`${r.key}-n`} style={{
                            fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                            color: C.text,
                          }}>{r.key}</div>,
                          <div key={`${r.key}-e`}>{fmtDateShort(r.earliest)}</div>,
                          <div key={`${r.key}-l`}>{fmtDateShort(r.latest)}</div>,
                        ]
                      ))}
                    </div>
                  </div>
                )}

                <div style={{
                  marginTop: 12, padding: '8px 12px', borderRadius: 8,
                  background: C.raised, fontSize: 12, color: C.sub,
                }}>
                  <strong style={{ color: C.text }}>Effective window: </strong>
                  {fmtDateShort(alignment?.since)} &rarr; {fmtDateShort(alignment?.until)}
                </div>
              </div>
            </div>

            {/* ── Context Limits ──────────────────────────────── */}
            <div style={{ marginBottom: 18 }}>
              <SectionTitle>Context Limits</SectionTitle>
              <div style={{
                background: C.bg, border: `1px solid ${C.border}`, borderRadius: 10,
                padding: 14,
              }}>
                <div style={{ fontSize: 12, color: C.sub, marginBottom: 10 }}>
                  Prompts encode each component as <strong style={{ color: C.text }}>CSV</strong> (compact tabular —
                  matches <code>/analyze</code>, not verbose JSON).{' '}
                  {modelInfo?.provider === 'openai' ? (
                    <>
                      OpenAI preflight uses heuristic token estimates (there is no count-tokens endpoint).
                    </>
                  ) : (
                    <>
                      Anthropic <code style={{ color: C.cyan }}>count_tokens</code> rejects request bodies whose
                      text exceeds about <strong style={{ color: C.text }}>
                        {fmtCount(countTokensTextBytesCap)}
                      </strong> UTF-8 bytes
                      {pfMeta?.anthropic_count_tokens_text_bytes_limit != null
                        ? ' (from preflight)'
                        : ' (default cap)'} — halving reduces rows until it fits.
                    </>
                  )}
                </div>

                {pfMeta?.preflight_notes?.length > 0 && (
                  <div style={{
                    marginBottom: 10, padding: '8px 12px',
                    borderRadius: 8,
                    border: `1px solid ${C.cyan}55`,
                    background: `${C.cyan}12`,
                    fontSize: 11,
                    color: C.sub,
                    lineHeight: 1.45,
                  }}>
                    {pfMeta.preflight_notes.map((n, i) => (
                      <div key={i} style={{ marginTop: i ? 6 : 0 }}>{n}</div>
                    ))}
                  </div>
                )}

                <div style={{ fontSize: 12, color: C.sub, marginBottom: 10 }}>
                  <strong style={{ color: C.text }}>{modelInfo?.label || model}</strong>
                  {' · '}
                  {fmtCount(contextWindow)} context window · reserve {fmtCount(reservedOut)} for reply
                  {' → '}
                  <strong style={{ color: C.text }}>{fmtCount(inputBudget)}</strong>
                  {' '}input-token budget per call (same shaping as Messages API){' · '}
                  {modelInfo?.max_output && (
                    <span>{fmtCount(modelInfo.max_output)} max output rated</span>
                  )}
                </div>

                {pfMeta?.pair_triple_prior_note && (
                  <div style={{ fontSize: 11, color: C.dim, marginBottom: 8 }}>
                    {pfMeta.pair_triple_prior_note}
                  </div>
                )}

                <div style={{
                  display: 'grid', gridTemplateColumns: '1fr auto auto',
                  gap: '6px 16px', fontSize: 12, alignItems: 'center',
                }}>
                  <div style={{ fontWeight: 700, color: C.dim, fontSize: 10, textTransform: 'uppercase' }}>Call type</div>
                  <div style={{ fontWeight: 700, color: C.dim, fontSize: 10, textTransform: 'uppercase' }}>Input tokens</div>
                  <div style={{ fontWeight: 700, color: C.dim, fontSize: 10, textTransform: 'uppercase' }}>Status</div>
                  {contextChecks.map((ck) => (
                    [
                      <div key={`${ck.kind}-t`} style={{ color: C.text }}>
                        {ck.kind} ({ck.count})
                        {ck.kind === 'individual' && ck.worst_component && (
                          <div style={{
                            fontSize: 10, color: C.dim, fontFamily: 'ui-monospace, monospace',
                          }}>worst: {ck.worst_component}</div>
                        )}
                      </div>,
                      <div key={`${ck.kind}-e`} style={{
                        fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                        color: C.sub,
                      }}>{fmtCount(ck.tokens_per_call_max)} max / call</div>,
                      <div key={`${ck.kind}-s`} style={{
                        color: ck.ok ? C.green : C.red, fontWeight: 600,
                      }}>
                        {ck.ok ? '✓ fits budget' : '✗ over budget'}
                      </div>,
                    ]
                  ))}
                </div>

                {indivDetail.length > 0 && (
                  <details style={{
                    marginTop: 12, fontSize: 11, color: C.sub,
                  }}>
                    <summary style={{ cursor: 'pointer', color: C.cyan, fontWeight: 600 }}>
                      Per-component input tokens ({indivDetail.length})
                    </summary>
                    <div style={{ marginTop: 8 }}>
                      {indivDetail.map((row) => (
                        <div key={row.key} style={{
                          display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between',
                          gap: 6, fontFamily: 'ui-monospace, monospace', padding: '4px 0',
                          borderBottom: `1px solid ${C.border}`,
                        }}>
                          <span style={{ color: C.text }}>{row.key}</span>
                          <span style={{ color: C.sub }}>
                            <strong style={{ color: C.cyan }}>{fmtCount(row.input_tokens)}</strong>
                            {' '}tok{' '}
                            {row.count_tokens_extrapolated && (
                              <span style={{ color: C.amber }} title="scaled from Anthropic count_tokens sample">
                                (meas. {fmtCount(row.measured_input_tokens_sample)})
                              </span>
                            )}
                            {' · '}
                            ~{fmtCount(row.estimated_prompt_utf8_bytes ?? 0)} UTF-8 B
                            {row.likely_exceeds_anthropic_text_budget && (
                              <span
                                style={{ color: C.amber, marginLeft: 4 }}
                                title={`Estimated UTF-8 prompt size near/over Anthropic ${fmtCount(countTokensTextBytesCap)}-byte text limit for count_tokens`}
                              >
                                (size)
                              </span>
                            )}
                            {' · '}
                            {fmtCount(row.series_rows_total ?? 0)} rows ·{' '}
                            {fmtCount(row.prompt_csv_chars ?? 0)} CSV chars
                            {row.count_tokens_halving_steps > 0 && (
                              <span style={{ color: C.amber, marginLeft: 6 }} title="Anthropic rejected prompt size — row caps halved until count succeeded">
                                · halved {fmtCount(row.count_tokens_halving_steps)}× (413/400)
                              </span>
                            )}
                          </span>
                        </div>
                      ))}
                    </div>
                  </details>
                )}

                {pfMeta?.calibration_chars_per_token != null && (
                  <div style={{ marginTop: 8, fontSize: 10, color: C.dim }}>
                    API calibration chars/token: {pfMeta.calibration_chars_per_token} (sizes pair/triple prior stubs only)
                  </div>
                )}

                {maxMeasuredTokens > 0 && (
                  <div style={{
                    marginTop: 10, fontSize: 11, color: C.dim,
                  }}>
                    Heaviest prompt: {fmtCount(maxMeasuredTokens)} input /
                    {' '}{fmtCount(inputBudget)} budget
                    {inputBudget > 0 ? ` (${((maxMeasuredTokens / inputBudget) * 100).toFixed(2)}% of budget)` : ''}
                  </div>
                )}

                {!allContextOk && alternativeModels.length > 0 && (
                  <div style={{
                    marginTop: 10, padding: '8px 12px', borderRadius: 8,
                    background: `${C.amber}1a`, border: `1px solid ${C.amber}55`,
                    fontSize: 12, color: C.amber,
                  }}>
                    Models that support this context:
                    {alternativeModels.map((m) => (
                      <span key={m.id} style={{
                        display: 'inline-block', margin: '2px 4px',
                        padding: '2px 8px', borderRadius: 6,
                        background: C.raised, color: C.text,
                      }}>{m.label} ({fmtCount(m.context_window)})</span>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* ── Action ────────────────────────────────────── */}
            <div style={{ textAlign: 'center', paddingTop: 4 }}>
              {canRun ? (
                <>
                  <div style={{ color: C.green, fontWeight: 600, marginBottom: 10, fontSize: 13 }}>
                    &#10003; All checks passed
                  </div>
                  <button
                    onClick={() => onRun(alignment.since, alignment.until)}
                    style={{
                      padding: '12px 28px', borderRadius: 10, border: 'none',
                      cursor: 'pointer', fontSize: 15, fontWeight: 700,
                      letterSpacing: '0.04em', color: '#0a0f1a',
                      background: `linear-gradient(135deg, ${C.emerald}, ${C.cyan})`,
                      boxShadow: `0 4px 16px ${C.emerald}40`,
                    }}
                  >
                    Run {totalCalls} call{totalCalls === 1 ? '' : 's'}
                  </button>
                </>
              ) : (
                <div style={{
                  background: `${C.red}1a`, border: `1px solid ${C.red}55`,
                  borderRadius: 10, padding: 14, fontSize: 13, color: C.red,
                  textAlign: 'left',
                }}>
                  <div style={{ fontWeight: 700, marginBottom: 8 }}>
                    Cannot run — action required:
                  </div>
                  <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.7 }}>
                    {hasNoData && (
                      <li>Remove components with no data in the selected range, or widen the window.</li>
                    )}
                    {!allContextOk && (
                      <li>
                        Measured prompt input (Anthropic <code style={{ fontSize: 11 }}>count_tokens</code>)
                        exceeds the input budget after reserving space for replies.
                        {alternativeModels.length > 0
                          ? ' Switch to a model with a larger window (below).'
                          : ' Narrow the window, drop components, or raise the deployment context limit.'}
                      </li>
                    )}
                  </ul>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function fmtDateShort(iso) {
  if (!iso) return '--';
  try {
    const d = new Date(iso);
    return d.toLocaleString([], {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
      hour12: false,
    });
  } catch { return iso.slice(0, 16); }
}


// ── small UI atoms ───────────────────────────────────────────────────────
function Group({ label, children }) {
  return (
    <div>
      <div style={{
        fontSize: 10, color: C.dim, letterSpacing: '0.1em',
        textTransform: 'uppercase', marginBottom: 4,
      }}>{label}</div>
      {children}
    </div>
  );
}

function Segmented({ options, value, onChange }) {
  return (
    <div style={{
      display: 'inline-flex', background: C.raised,
      border: `1px solid ${C.border}`, borderRadius: 8, padding: 2,
    }}>
      {options.map((o) => {
        const on = o.value === value;
        const dis = !!o.disabled;
        return (
          <button
            key={String(o.value)}
            onClick={() => { if (!dis) onChange(o.value); }}
            title={dis ? 'Not available for this model' : undefined}
            style={{
              padding: '5px 10px', borderRadius: 6, border: 'none',
              cursor: dis ? 'not-allowed' : 'pointer',
              background: on ? C.cyan + '22' : 'transparent',
              color: dis ? C.muted : on ? C.cyan : C.sub,
              fontSize: 12, fontWeight: 600,
              textTransform: 'uppercase', letterSpacing: '0.04em',
              opacity: dis ? 0.4 : 1,
            }}
          >{o.label}</button>
        );
      })}
    </div>
  );
}

function Toggle({ on, onChange, label, disabled = false }) {
  return (
    <button
      onClick={() => { if (!disabled) onChange(!on); }}
      style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '5px 10px', borderRadius: 8,
        background: on ? C.purple + '22' : C.raised,
        color: on ? C.purple : C.sub, cursor: disabled ? 'not-allowed' : 'pointer',
        border: `1px solid ${on ? C.purple + '88' : C.border}`,
        fontSize: 12, fontWeight: 600, letterSpacing: '0.04em',
        textTransform: 'uppercase',
        opacity: disabled ? 0.5 : 1,
      }}
    >
      <span style={{
        width: 24, height: 12, borderRadius: 999, background: on ? C.purple : C.muted,
        position: 'relative', transition: 'all .15s',
      }}>
        <span style={{
          position: 'absolute', top: 1, left: on ? 13 : 1,
          width: 10, height: 10, borderRadius: '50%', background: '#fff',
          transition: 'all .15s',
        }} />
      </span>
      {label}
    </button>
  );
}

function CountPill({ label, value, color }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 6,
      padding: '4px 10px', borderRadius: 999,
      background: `${color}1a`, border: `1px solid ${color}55`,
      color, fontSize: 11.5, fontWeight: 700, letterSpacing: '0.04em',
      textTransform: 'uppercase',
    }}>
      <span style={{ color: C.dim, textTransform: 'none' }}>{label}</span>
      <span style={{
        background: `${color}22`, padding: '1px 7px', borderRadius: 999,
        fontFamily: 'ui-monospace, SFMono-Regular, monospace', fontSize: 12,
      }}>{value}</span>
    </div>
  );
}

function StepPill({ active, done, label }) {
  const color = done ? C.green : active ? C.cyan : C.muted;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '4px 10px', borderRadius: 999,
      background: active ? `${color}1a` : 'transparent',
      border: `1px solid ${color}55`,
      fontSize: 11, fontWeight: 600, color,
    }}>
      {done ? '✓' : active ? '●' : '○'} {label}
    </span>
  );
}

function SectionTitle({ children }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      padding: '6px 10px 12px', color: C.text,
      fontWeight: 700, fontSize: 13, letterSpacing: '0.04em',
    }}>
      <div style={{ width: 3, height: 16, background: C.cyan, borderRadius: 2 }} />
      {children}
    </div>
  );
}

const miniBtnStyle = {
  background: 'transparent', color: C.sub, border: `1px solid ${C.border}`,
  padding: '2px 8px', borderRadius: 6, cursor: 'pointer',
  fontSize: 11, letterSpacing: '0.04em', textTransform: 'uppercase',
};

const dateInputStyle = {
  background: C.raised, color: C.text, border: `1px solid ${C.border}`,
  borderRadius: 8, padding: '6px 10px', fontSize: 12, outline: 'none',
  fontFamily: 'ui-monospace, SFMono-Regular, monospace',
  colorScheme: 'dark',
};

const selectStyle = {
  background: C.raised, color: C.text, border: `1px solid ${C.border}`,
  borderRadius: 8, padding: '6px 10px', fontSize: 13, outline: 'none',
  fontWeight: 600, minWidth: 130, cursor: 'pointer',
};
