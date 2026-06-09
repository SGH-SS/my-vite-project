/**
 * PerpDashboard — generic Hyperliquid perp dashboard (BTC / ETH / SPX)
 * =====================================================================
 * Two tabs: Overview (status + calendar) and Price Feed (live ticks +
 * trades).  Identical to the Overview / Price Feed (WS) tabs of Sol.jsx
 * but parameterised by asset so we can reuse it for every perp instrument.
 *
 * The SOL dashboard remains in Sol.jsx because it also hosts the
 * Hyperliquid account / BTC trading sub-tab — which is account-level
 * functionality unrelated to a specific market data feed.
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

// ── Colors (shared across all perps) ─────────────────────────────────────
const C = {
  green: '#00d4a8', red: '#f43f5e', amber: '#eab308', blue: '#60a5fa',
  purple: '#a78bfa', cyan: '#22d3ee', orange: '#e97316',
  grayOrange: '#8b6d42', future: '#1e293b',
  bg: '#060c18', surface: '#0c1628', raised: '#101e35', border: '#1a2d4e',
  text: '#e2e8f0', sub: '#94a3b8', muted: '#475569', dim: '#64748b',
};

// ── Helpers ──────────────────────────────────────────────────────────────
const fmt = (n, d = 2) => (n == null ? '--' : Number(n).toFixed(d));
const fmtK = (n) => {
  if (n == null) return '--';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
};
const ago = (s) => {
  if (s == null) return '--';
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m ago`;
  return `${Math.floor(s / 86400)}d ago`;
};
const tsShort = (iso) => {
  if (!iso) return '';
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
};

const MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December'];
const WEEKDAYS = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];

// ── Stat pill ────────────────────────────────────────────────────────────
const Pill = ({ label, value, color = C.blue }) => (
  <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: '14px 16px' }}>
    <div style={{ fontSize: 10, letterSpacing: '0.1em', color: C.dim, marginBottom: 4, textTransform: 'uppercase' }}>{label}</div>
    <div style={{ fontSize: 22, fontWeight: 700, color, lineHeight: 1.1 }}>{value}</div>
  </div>
);

// ── Section heading ──────────────────────────────────────────────────────
const Heading = ({ children, accent = C.green }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 10, margin: '24px 0 12px' }}>
    <div style={{ width: 3, height: 18, background: accent, borderRadius: 2 }} />
    <div style={{ fontSize: 15, fontWeight: 600, letterSpacing: '0.03em', color: C.text }}>{children}</div>
  </div>
);

// ── Live dot ─────────────────────────────────────────────────────────────
function LiveDot({ active }) {
  return (
    <span style={{
      display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
      background: active ? C.green : C.red,
      boxShadow: active ? `0 0 8px ${C.green}80` : 'none',
      animation: active ? 'perppulse 2s infinite' : 'none',
    }}>
      <style>{`@keyframes perppulse{0%,100%{opacity:1}50%{opacity:.4}}`}</style>
    </span>
  );
}


// ═════════════════════════════════════════════════════════════════════════
// MAIN
// ═════════════════════════════════════════════════════════════════════════
export default function PerpDashboard({
  asset,                             // 'btc' | 'eth' | 'spx'
  displayName,                       // e.g. 'BTC-PERP'
  subtitle = 'Hyperliquid Perpetual Futures · L2 + Trades',
  gradient = [C.purple, C.cyan],     // [from, to] for the title gradient
  priceDecimals = 2,                 // mid / bid / ask formatting
  spreadDecimals = 4,                // spread formatting
}) {
  const API = `http://localhost:8000/api/perp/${asset}`;
  const WS_URL = `ws://localhost:8000/api/perp/${asset}/ws`;

  const CH_TICKS  = `${asset}_ticks`;
  const CH_TRADES = `${asset}_trades`;
  const CH_MARKS  = `${asset}_marks`;

  const [status, setStatus] = useState(null);
  const [ticks, setTicks] = useState([]);
  const [trades, setTrades] = useState([]);
  const [latestTick, setLatestTick] = useState(null);
  const [mark, setMark] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);
  const [tab, setTab] = useState('overview');
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef(null);

  const loadStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API}/status`);
      setStatus(await res.json());
      setErr(null);
    } catch (e) { setErr(e.message); }
    setLoading(false);
  }, [API]);

  const loadInitialFeedData = useCallback(async () => {
    try {
      const [tRes, trRes] = await Promise.all([
        fetch(`${API}/recent-ticks?limit=300`),
        fetch(`${API}/recent-trades?limit=80`),
      ]);
      const tJson = await tRes.json();
      const trJson = await trRes.json();
      if (Array.isArray(tJson)) setTicks(tJson);
      if (Array.isArray(trJson)) setTrades(trJson);
    } catch (_) {}
  }, [API]);

  // Reset state when asset changes (so switching tabs / dashboards is clean)
  useEffect(() => {
    setStatus(null);
    setTicks([]);
    setTrades([]);
    setLatestTick(null);
    setMark(null);
    setLoading(true);
    setErr(null);
    setWsConnected(false);
  }, [asset]);

  useEffect(() => {
    loadStatus();
    loadInitialFeedData();
  }, [loadStatus, loadInitialFeedData]);

  // Overview tab: poll status every 10s
  useEffect(() => {
    if (tab === 'overview') {
      const sid = setInterval(loadStatus, 10_000);
      return () => clearInterval(sid);
    }
  }, [tab, loadStatus]);

  // Feed tab: open WebSocket for live push, close on tab/asset switch
  useEffect(() => {
    if (tab !== 'feed') return;

    let ws;
    let reconnectTimer;
    let alive = true;

    const connect = () => {
      if (!alive) return;
      ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
        setErr(null);
      };

      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          const { channel, data } = msg;

          if (channel === CH_TICKS) {
            setLatestTick({ available: true, ...data });
            setTicks(prev => {
              const next = [...prev, data];
              return next.length > 600 ? next.slice(next.length - 600) : next;
            });
          } else if (channel === CH_TRADES) {
            if (Array.isArray(data) && data.length > 0) {
              setTrades(prev => {
                const merged = [...prev, ...data];
                return merged.length > 200 ? merged.slice(merged.length - 200) : merged;
              });
            }
          } else if (channel === CH_MARKS) {
            const now = new Date();
            const tsDate = new Date(data.ts);
            const ageSec = (now - tsDate) / 1000;
            setMark({
              available: true, is_fresh: ageSec < 10, age_s: ageSec,
              ...data,
            });
          }
        } catch (_) {}
      };

      ws.onclose = () => {
        setWsConnected(false);
        wsRef.current = null;
        if (alive) reconnectTimer = setTimeout(connect, 2000);
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    connect();
    loadInitialFeedData();

    return () => {
      alive = false;
      clearTimeout(reconnectTimer);
      if (ws && ws.readyState <= 1) ws.close();
      wsRef.current = null;
      setWsConnected(false);
    };
  }, [tab, WS_URL, CH_TICKS, CH_TRADES, CH_MARKS, loadInitialFeedData]);

  if (loading) {
    return <div style={{ minHeight: '40vh', display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.dim }}>
      Loading {displayName} data...
    </div>;
  }
  if (err) {
    return (
      <div style={{ padding: 32, background: '#1a0a0a', border: '1px solid #7f1d1d', borderRadius: 10, margin: 24 }}>
        <div style={{ fontSize: 16, fontWeight: 600, color: C.red }}>Connection Error</div>
        <div style={{ fontSize: 13, color: '#fca5a5', marginTop: 6 }}>{err}</div>
        <div style={{ fontSize: 11, color: '#991b1b', marginTop: 8 }}>Ensure backend is running: python backend/main.py</div>
      </div>
    );
  }

  const isIngesting = status?.freshness?.is_ingesting;
  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'feed', label: wsConnected ? 'Price Feed (WS)' : 'Price Feed' },
  ];

  return (
    <div style={{ fontFamily: "'DM Sans','Segoe UI',sans-serif", color: C.text, padding: '0 0 48px' }}>

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20, paddingBottom: 16, borderBottom: `1px solid ${C.border}` }}>
        <div>
          <div style={{ fontSize: 28, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ background: `linear-gradient(135deg, ${gradient[0]}, ${gradient[1]})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>{displayName}</span>
            <LiveDot active={isIngesting} />
          </div>
          <div style={{ fontSize: 12, color: C.muted, marginTop: 4 }}>{subtitle}</div>
        </div>
        <div style={{
          background: isIngesting ? '#071812' : '#1a1a1a',
          border: `1px solid ${isIngesting ? C.green + '60' : '#333'}`,
          borderRadius: 10, padding: '12px 18px', textAlign: 'right',
        }}>
          <div style={{ fontSize: 9, letterSpacing: '0.15em', color: isIngesting ? C.green + 'cc' : '#666', marginBottom: 4 }}>
            {isIngesting ? 'INGESTING LIVE' : 'COLLECTOR OFFLINE'}
          </div>
          {status?.freshness?.l2_age_s != null && (
            <div style={{ fontSize: 13, color: isIngesting ? C.green : C.red }}>Last tick: {ago(status.freshness.l2_age_s)}</div>
          )}
          {status?.collector && (
            <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>
              {fmtK(status.collector.snaps_inserted)} snaps · {fmtK(status.collector.trades_inserted)} trades · {status.collector.reconnects} reconn
            </div>
          )}
        </div>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, background: '#0f172a', borderRadius: 8, padding: 4, maxWidth: 320 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            flex: 1, padding: '8px 0', borderRadius: 6, border: 'none',
            fontSize: 13, fontWeight: 500, cursor: 'pointer', transition: 'all 0.15s',
            background: tab === t.id ? '#1e293b' : 'transparent',
            color: tab === t.id ? C.text : C.dim,
            boxShadow: tab === t.id ? '0 1px 4px #0004' : 'none',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab status={status} priceDecimals={priceDecimals} spreadDecimals={spreadDecimals} API={API} />}
      {tab === 'feed' && <FeedTab ticks={ticks} trades={trades} status={status} mark={mark} latestTick={latestTick}
                                  priceDecimals={priceDecimals} spreadDecimals={spreadDecimals} />}
    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════
// OVERVIEW TAB
// ═════════════════════════════════════════════════════════════════════════
function OverviewTab({ status, priceDecimals, spreadDecimals, API }) {
  const tick = status?.latest_tick;

  return (
    <div>
      {/* Current Market */}
      {tick && (
        <>
          <Heading accent={C.cyan}>Current Market</Heading>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
            <Pill label="Mid Price" value={'$' + fmt(tick.mid, priceDecimals)} color={C.cyan} />
            <Pill label="Best Bid"  value={'$' + fmt(tick.best_bid, priceDecimals)} color={C.green} />
            <Pill label="Best Ask"  value={'$' + fmt(tick.best_ask, priceDecimals)} color={C.red} />
            <Pill label="Spread"    value={'$' + fmt(tick.spread, spreadDecimals)} color={C.purple} />
            <Pill label="Bid Levels" value={tick.bid_levels} color={C.blue} />
            <Pill label="Ask Levels" value={tick.ask_levels} color={C.blue} />
          </div>
        </>
      )}

      {/* Collector Session */}
      {status?.collector && (
        <>
          <Heading accent={C.blue}>Collector Session</Heading>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 10 }}>
            <Pill label="Session ID"    value={`#${status.collector.id}`} color={C.blue} />
            <Pill label="Started"       value={new Date(status.collector.started_at).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })} color={C.blue} />
            <Pill label="Last Heartbeat" value={ago(status.collector.heartbeat_age_s)} color={status.collector.is_live ? C.green : C.red} />
            <Pill label="Reconnects"    value={status.collector.reconnects} color={status.collector.reconnects > 3 ? C.amber : C.green} />
          </div>
        </>
      )}

      <DataCalendar isIngesting={status?.freshness?.is_ingesting} API={API} />
    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════
// DATA CALENDAR
// ═════════════════════════════════════════════════════════════════════════
function DataCalendar({ isIngesting, API }) {
  const now = new Date();
  const utcYear = now.getUTCFullYear();
  const utcMonth = now.getUTCMonth();
  const utcDay = now.getUTCDate();

  const [month, setMonth] = useState(utcMonth);
  const [year, setYear] = useState(utcYear);
  const [selectedDay, setSelectedDay] = useState(null);
  const [dayDetail, setDayDetail] = useState(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [calData, setCalData] = useState(null);
  const [calLoading, setCalLoading] = useState(true);

  const todayStr = `${utcYear}-${String(utcMonth + 1).padStart(2, '0')}-${String(utcDay).padStart(2, '0')}`;

  const utcTimeStr = now.toLocaleString('en-US', { timeZone: 'UTC', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  const pstTimeStr = now.toLocaleString('en-US', { timeZone: 'America/Los_Angeles', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true });

  useEffect(() => {
    let cancelled = false;
    setCalLoading(true);
    fetch(`${API}/calendar?year=${year}&month=${month + 1}`)
      .then(r => r.json())
      .then(data => { if (!cancelled) { setCalData(data); setCalLoading(false); } })
      .catch(() => { if (!cancelled) setCalLoading(false); });
    return () => { cancelled = true; };
  }, [year, month, API]);

  const dayMap = useMemo(() => {
    const m = {};
    (calData?.days || []).forEach(d => { m[d.date] = d; });
    return m;
  }, [calData]);

  const getDayColor = useCallback((dateStr) => {
    const isToday = dateStr === todayStr;
    const d = dayMap[dateStr];

    if (!d) {
      return { bg: 'transparent', border: isToday ? C.red : 'transparent', label: 'no-data' };
    }

    const hasLive = d.has_live_l2 || d.has_live_trades;
    const hasBackfill = d.has_bf_l2 || d.has_bf_trades;

    if (isToday) {
      if (isIngesting && hasLive) return { bg: 'transparent', border: C.green, label: 'today-live' };
      if (hasLive || hasBackfill) return { bg: 'transparent', border: C.amber, label: 'today-data' };
      return { bg: 'transparent', border: C.red, label: 'today-nodata' };
    }

    if (hasLive) return { bg: '#059669', border: 'transparent', label: 'live' };
    if (hasBackfill) return { bg: '#c2410c', border: 'transparent', label: 'backfill' };
    return { bg: 'transparent', border: 'transparent', label: 'no-data' };
  }, [dayMap, todayStr, isIngesting]);

  const handleDayClick = useCallback(async (dateStr) => {
    if (selectedDay === dateStr) { setSelectedDay(null); setDayDetail(null); return; }
    setSelectedDay(dateStr);
    setLoadingDetail(true);
    try {
      const r = await fetch(`${API}/day-detail/${dateStr}`);
      setDayDetail(await r.json());
    } catch (_) { setDayDetail(null); }
    setLoadingDetail(false);
  }, [selectedDay, API]);

  const firstDow = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();

  const cells = useMemo(() => {
    const arr = [];
    for (let i = 0; i < firstDow; i++) arr.push({ type: 'empty', key: `e${i}` });
    for (let d = 1; d <= daysInMonth; d++) {
      const ds = `${year}-${String(month + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
      const colors = getDayColor(ds);
      const data = dayMap[ds];
      const isToday = ds === todayStr;
      arr.push({ type: 'day', key: ds, day: d, dateStr: ds, colors, data, isToday, isSelected: ds === selectedDay });
    }
    return arr;
  }, [year, month, firstDow, daysInMonth, getDayColor, dayMap, todayStr, selectedDay]);

  const prevMonth = () => { if (month === 0) { setMonth(11); setYear(y => y - 1); } else setMonth(m => m - 1); };
  const nextMonth = () => { if (month === 11) { setMonth(0); setYear(y => y + 1); } else setMonth(m => m + 1); };

  return (
    <>
      <Heading accent={C.purple}>Data Coverage Calendar</Heading>

      <div style={{ display: 'flex', gap: 24, marginBottom: 12, fontSize: 12 }}>
        <div>
          <span style={{ color: C.cyan, fontWeight: 600 }}>UTC: </span>
          <span style={{ color: C.text }}>{utcTimeStr}</span>
        </div>
        <div>
          <span style={{ color: C.purple, fontWeight: 600 }}>PST: </span>
          <span style={{ color: C.sub }}>{pstTimeStr}</span>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 16, marginBottom: 14, flexWrap: 'wrap' }}>
        {[
          { color: '#059669', label: 'Live data' },
          { color: '#c2410c', label: 'Backfill only' },
          { color: C.red, label: 'No data', outline: true },
        ].map(l => (
          <div key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: C.sub }}>
            <div style={{
              width: 12, height: 12, borderRadius: 3,
              background: l.outline ? 'transparent' : l.color + '60',
              border: l.outline ? `2px solid ${l.color}` : `1px solid ${l.color}`,
            }} />
            {l.label}
          </div>
        ))}
      </div>

      <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: 20, position: 'relative' }}>
        {calLoading && (
          <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.35)', borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 2, fontSize: 13, color: C.dim }}>
            Loading...
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <button onClick={prevMonth} style={{ background: 'none', border: 'none', color: C.sub, cursor: 'pointer', fontSize: 13, padding: '4px 8px' }}>← Prev</button>
          <div style={{ fontSize: 16, fontWeight: 600, color: C.text }}>{MONTHS[month]} {year}</div>
          <button onClick={nextMonth} style={{ background: 'none', border: 'none', color: C.sub, cursor: 'pointer', fontSize: 13, padding: '4px 8px' }}>Next →</button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 4, marginBottom: 4 }}>
          {WEEKDAYS.map(w => (
            <div key={w} style={{ textAlign: 'center', fontSize: 11, fontWeight: 600, color: C.muted, padding: '4px 0' }}>{w}</div>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 4 }}>
          {cells.map(cell => {
            if (cell.type === 'empty') return <div key={cell.key} style={{ height: 56 }} />;
            const { day, dateStr, colors, data, isToday, isSelected } = cell;
            const hasData = !!data;
            return (
              <button
                key={cell.key}
                onClick={() => hasData && handleDayClick(dateStr)}
                style={{
                  height: 56, borderRadius: 6, border: 'none',
                  background: colors.bg !== 'transparent' ? colors.bg + '45' : (isSelected ? C.blue + '15' : C.raised),
                  outline: colors.border !== 'transparent' ? `2px solid ${colors.border}` : isSelected ? `2px solid ${C.blue}` : 'none',
                  outlineOffset: -2,
                  cursor: hasData ? 'pointer' : 'default',
                  display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                  transition: 'all 0.12s',
                  opacity: hasData ? 1 : 0.3,
                }}
              >
                <div style={{ fontSize: 14, fontWeight: isToday ? 700 : 500, color: isToday ? C.cyan : C.text }}>{day}</div>
                {hasData && (
                  <div style={{ display: 'flex', gap: 2, marginTop: 2 }}>
                    {[
                      { has: data.has_live_l2 || data.has_bf_l2, c: C.cyan, label: 'L' },
                      { has: data.has_live_trades || data.has_bf_trades, c: C.green, label: 'T' },
                      { has: data.has_live_mark || data.has_bf_mark, c: C.purple, label: 'M' },
                    ].map(s => (
                      <div key={s.label} style={{
                        fontSize: 7, fontWeight: 700, lineHeight: 1,
                        color: s.has ? s.c : C.muted + '40',
                      }}>{s.label}</div>
                    ))}
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {selectedDay && (
        <DayTimeline dateStr={selectedDay} detail={dayDetail} loading={loadingDetail} />
      )}
    </>
  );
}


// ═════════════════════════════════════════════════════════════════════════
// 24-HOUR TIMELINE
// ═════════════════════════════════════════════════════════════════════════
function DayTimeline({ dateStr, detail, loading }) {
  if (loading) return <div style={{ padding: 20, color: C.dim, textAlign: 'center' }}>Loading timeline...</div>;
  if (!detail || !detail.hours) return null;

  const now = new Date();
  const todayUTC = `${now.getUTCFullYear()}-${String(now.getUTCMonth() + 1).padStart(2, '0')}-${String(now.getUTCDate()).padStart(2, '0')}`;
  const isToday = dateStr === todayUTC;
  const currentHourUTC = now.getUTCHours();
  const isFutureDay = dateStr > todayUTC;

  const pastHours = detail.hours.filter(h => {
    if (isFutureDay) return false;
    if (!isToday) return true;
    return h.hour <= currentHourUTC;
  });

  const COMPONENTS = [
    { key: 'l2',     liveKey: 'live_l2',     bfKey: 'bf_l2',     label: 'L2 Book',    color: C.cyan },
    { key: 'trades', liveKey: 'live_trades',  bfKey: 'bf_trades', label: 'Trades',     color: C.green },
    { key: 'mark',   liveKey: 'live_mark',    bfKey: 'bf_mark',   label: 'Mark Price', color: C.purple },
  ];

  return (
    <div style={{ marginTop: 16, background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: '16px 20px' }}>
      <div style={{ fontSize: 14, fontWeight: 600, color: C.text, marginBottom: 4 }}>
        {dateStr} — 24-Hour Timeline <span style={{ fontSize: 11, color: C.dim, fontWeight: 400 }}>(UTC)</span>
      </div>
      {isToday && (
        <div style={{ fontSize: 11, color: C.dim, marginBottom: 10 }}>
          Current hour: {currentHourUTC}:00 UTC — future hours shown as dark
        </div>
      )}

      {COMPONENTS.map(comp => (
        <div key={comp.key} style={{ marginBottom: 6 }}>
          <div style={{ fontSize: 10, color: comp.color, fontWeight: 600, marginBottom: 2, letterSpacing: '0.05em' }}>{comp.label}</div>
          <div style={{ display: 'flex', gap: 2 }}>
            {detail.hours.map(h => {
              const isFuture = isFutureDay || (isToday && h.hour > currentHourUTC);
              const liveN = h[comp.liveKey] || 0;
              const bfN = h[comp.bfKey] || 0;

              let bg, borderClr, textClr;
              if (isFuture) {
                bg = C.future; borderClr = '#2d3748'; textClr = '#334155';
              } else if (liveN > 0) {
                bg = comp.color + '30'; borderClr = comp.color; textClr = comp.color;
              } else if (bfN > 0) {
                bg = C.orange + '25'; borderClr = C.orange; textClr = C.orange;
              } else {
                bg = C.red + '18'; borderClr = C.red + '60'; textClr = C.red + '80';
              }

              return (
                <div key={h.hour}
                  title={isFuture
                    ? `${h.hour}:00 — not yet`
                    : `${h.hour}:00 — live: ${fmtK(liveN)}, bf: ${fmtK(bfN)}`
                  }
                  style={{
                    flex: 1, height: 20, borderRadius: 2,
                    background: bg, borderBottom: `2px solid ${borderClr}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 8, color: textClr,
                  }}
                >
                  {h.hour}
                </div>
              );
            })}
          </div>
        </div>
      ))}

      <div style={{ display: 'flex', gap: 14, fontSize: 11, color: C.sub, marginTop: 8, flexWrap: 'wrap' }}>
        {COMPONENTS.map(comp => {
          const liveTotal = pastHours.reduce((s, h) => s + (h[comp.liveKey] || 0), 0);
          const bfTotal = pastHours.reduce((s, h) => s + (h[comp.bfKey] || 0), 0);
          const liveHrs = pastHours.filter(h => (h[comp.liveKey] || 0) > 0).length;
          return (
            <span key={comp.key}>
              <span style={{ color: comp.color, fontWeight: 600 }}>{comp.label}: </span>
              {liveHrs > 0 ? `${fmtK(liveTotal)} live (${liveHrs}h)` : ''}
              {liveHrs > 0 && bfTotal > 0 ? ' + ' : ''}
              {bfTotal > 0 ? <span style={{ color: C.orange }}>{fmtK(bfTotal)} bf</span> : ''}
              {liveHrs === 0 && bfTotal === 0 ? <span style={{ color: C.red }}>none</span> : ''}
            </span>
          );
        })}
      </div>

      <div style={{ display: 'flex', gap: 14, marginTop: 8 }}>
        {[
          { color: C.cyan, label: 'Live' },
          { color: C.orange, label: 'Backfill' },
          { color: C.red, label: 'Gap' },
          { color: '#334155', label: 'Future' },
        ].map(l => (
          <div key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: C.sub }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: l.color + '50', borderBottom: `2px solid ${l.color}` }} />
            {l.label}
          </div>
        ))}
      </div>
    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════
// PRICE FEED TAB
// ═════════════════════════════════════════════════════════════════════════
function FeedTab({ ticks, trades, status, mark, latestTick, priceDecimals, spreadDecimals }) {
  const latest = latestTick || (ticks.length > 0 ? ticks[ticks.length - 1] : null);
  const tick = latest ? {
    mid: latest.mid,
    best_bid: latest.best_bid,
    best_ask: latest.best_ask,
    spread: latest.spread,
  } : status?.latest_tick;

  const lastTrade = trades && trades.length > 0 ? trades[trades.length - 1] : null;

  const markOk    = mark && mark.available && mark.is_fresh;
  const markPx    = markOk ? mark.mark_px : null;
  const markStale = mark && mark.available && !mark.is_fresh;
  const markVal = markPx != null
    ? '$' + fmt(markPx, priceDecimals)
    : (mark?.available === false ? 'no data' : '—');

  return (
    <div>
      {tick && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, marginBottom: 16 }}>
          <Pill label={markStale ? `Mark (${Math.round(mark.age_s)}s old)` : 'Mark'}
                value={markVal} color={C.purple} />
          <Pill label="Mid"        value={'$' + fmt(tick.mid, priceDecimals)}      color={C.cyan} />
          <Pill label="Last Trade" value={lastTrade ? '$' + fmt(lastTrade.px, priceDecimals) : '—'}
                color={lastTrade && lastTrade.side === 'B' ? C.green : C.red} />
          <Pill label="Bid"        value={'$' + fmt(tick.best_bid, priceDecimals)} color={C.green} />
          <Pill label="Ask"        value={'$' + fmt(tick.best_ask, priceDecimals)} color={C.red} />
        </div>
      )}

      <Heading accent={C.cyan}>Live Mid Price</Heading>
      {ticks.length > 0 && (
        <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: '12px 8px' }}>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={ticks.map(t => ({ ...t, t: tsShort(t.ts) }))} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="t" tick={{ fontSize: 9, fill: C.dim }} interval={Math.floor(ticks.length / 8)} />
              <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10, fill: C.dim }} tickFormatter={v => '$' + Number(v).toFixed(priceDecimals)} />
              <Tooltip contentStyle={{ background: '#1e293b', border: 'none', borderRadius: 8, fontSize: 12 }}
                formatter={(v) => ['$' + Number(v).toFixed(priceDecimals + 1), 'Mid']} />
              <Line type="monotone" dataKey="mid" stroke={C.cyan} dot={false} strokeWidth={1.5} name="Mid" isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <Heading accent={C.purple}>Spread</Heading>
      {ticks.length > 0 && (
        <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: '12px 8px' }}>
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={ticks.map(t => ({ ...t, t: tsShort(t.ts) }))} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="t" tick={{ fontSize: 9, fill: C.dim }} interval={Math.floor(ticks.length / 6)} />
              <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10, fill: C.dim }} tickFormatter={v => '$' + Number(v).toFixed(spreadDecimals)} />
              <Tooltip contentStyle={{ background: '#1e293b', border: 'none', borderRadius: 8, fontSize: 12 }}
                formatter={(v) => ['$' + Number(v).toFixed(spreadDecimals + 1), 'Spread']} />
              <Line type="monotone" dataKey="spread" stroke={C.purple} dot={false} strokeWidth={1} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <Heading accent={C.green}>Recent Trades</Heading>
      <TradesTape trades={trades} priceDecimals={priceDecimals} />
    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════
// TRADES TAPE
// ═════════════════════════════════════════════════════════════════════════
function TradesTape({ trades, priceDecimals }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current) containerRef.current.scrollTop = 0;
  }, [trades]);

  const reversed = useMemo(() => [...trades].reverse(), [trades]);

  if (!trades.length) return <div style={{ color: C.dim, padding: 12 }}>No trades yet.</div>;

  return (
    <div ref={containerRef} style={{
      background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8,
      maxHeight: 360, overflowY: 'auto', overflowX: 'hidden',
    }}>
      <div style={{
        display: 'grid', gridTemplateColumns: '100px 50px 110px 80px',
        background: C.raised, borderBottom: `1px solid ${C.border}`,
        position: 'sticky', top: 0, zIndex: 1,
      }}>
        {['Time', 'Side', 'Price', 'Size'].map(h => (
          <div key={h} style={{ padding: '6px 10px', fontSize: 10, letterSpacing: '0.1em', color: C.dim }}>{h}</div>
        ))}
      </div>
      {reversed.map((tr, i) => (
        <div key={i} style={{
          display: 'grid', gridTemplateColumns: '100px 50px 110px 80px',
          borderBottom: `1px solid ${C.border}08`,
          background: i % 2 === 0 ? 'transparent' : '#0f172a',
        }}>
          <div style={{ padding: '3px 10px', fontSize: 11, color: C.dim }}>{tsShort(tr.ts)}</div>
          <div style={{ padding: '3px 10px', fontSize: 11, fontWeight: 600, color: tr.side === 'B' ? C.green : C.red }}>
            {tr.side === 'B' ? 'BUY' : 'SELL'}
          </div>
          <div style={{ padding: '3px 10px', fontSize: 11, color: C.text, fontFamily: 'monospace' }}>${fmt(tr.px, priceDecimals)}</div>
          <div style={{ padding: '3px 10px', fontSize: 11, color: C.sub, fontFamily: 'monospace' }}>{fmt(tr.sz, 4)}</div>
        </div>
      ))}
    </div>
  );
}
