/**
 * SOL-PERP Dashboard
 * Two tabs: Overview (status + calendar) and Price Feed (live ticks + trades)
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend
} from 'recharts';

const API = 'http://localhost:8000/api/sol';
const WS_URL = 'ws://localhost:8000/api/sol/ws';

// ── Colors ───────────────────────────────────────────────────────────────
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
      animation: active ? 'solpulse 2s infinite' : 'none',
    }}>
      <style>{`@keyframes solpulse{0%,100%{opacity:1}50%{opacity:.4}}`}</style>
    </span>
  );
}


// ═════════════════════════════════════════════════════════════════════════
// MAIN
// ═════════════════════════════════════════════════════════════════════════
export default function SolDashboard({ isDarkMode }) {
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

  // Hyperliquid connection state (managed inside HyperliquidTab)

  // Status is lightweight — safe to poll at 10 s
  const loadStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API}/status`);
      setStatus(await res.json());
      setErr(null);
    } catch (e) { setErr(e.message); }
    setLoading(false);
  }, []);

  // Initial REST loads for chart history + trade tape seed
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
  }, []);

  // On mount: load status (fast → unblocks render) + seed feed data
  useEffect(() => {
    loadStatus();
    loadInitialFeedData();
  }, [loadStatus, loadInitialFeedData]);

  // Overview tab: only status polls (lightweight); calendar loaded once on mount
  useEffect(() => {
    if (tab === 'overview') {
      const sid = setInterval(loadStatus, 10_000);
      return () => clearInterval(sid);
    }
  }, [tab, loadStatus]);

  // Feed tab: open WebSocket for live push, close on tab switch
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

          if (channel === 'sol_ticks') {
            setLatestTick({ available: true, ...data });
            setTicks(prev => {
              const next = [...prev, data];
              return next.length > 600 ? next.slice(next.length - 600) : next;
            });
          } else if (channel === 'sol_trades') {
            if (Array.isArray(data) && data.length > 0) {
              setTrades(prev => {
                const merged = [...prev, ...data];
                return merged.length > 200 ? merged.slice(merged.length - 200) : merged;
              });
            }
          } else if (channel === 'sol_marks') {
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
  }, [tab, loadInitialFeedData]);

  if (loading) {
    return <div style={{ minHeight: '40vh', display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.dim }}>Loading SOL-PERP data...</div>;
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
    { id: 'hyperliquid', label: 'Hyperliquid' },
  ];

  return (
    <div style={{ fontFamily: "'DM Sans','Segoe UI',sans-serif", color: C.text, padding: '0 0 48px' }}>

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20, paddingBottom: 16, borderBottom: `1px solid ${C.border}` }}>
        <div>
          <div style={{ fontSize: 28, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ background: `linear-gradient(135deg, ${C.purple}, ${C.cyan})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>SOL-PERP</span>
            <LiveDot active={isIngesting} />
          </div>
          <div style={{ fontSize: 12, color: C.muted, marginTop: 4 }}>Hyperliquid Perpetual Futures &middot; L2 + Trades</div>
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
              {fmtK(status.collector.snaps_inserted)} snaps &middot; {fmtK(status.collector.trades_inserted)} trades &middot; {status.collector.reconnects} reconn
            </div>
          )}
        </div>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, background: '#0f172a', borderRadius: 8, padding: 4, maxWidth: 420 }}>
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

      {tab === 'overview' && <OverviewTab status={status} />}
      {tab === 'feed' && <FeedTab ticks={ticks} trades={trades} status={status} mark={mark} latestTick={latestTick} />}
      {tab === 'hyperliquid' && <HyperliquidTab />}
    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════
// OVERVIEW TAB
// ═════════════════════════════════════════════════════════════════════════
function OverviewTab({ status }) {
  const tick = status?.latest_tick;

  return (
    <div>
      {/* Current Market */}
      {tick && (
        <>
          <Heading accent={C.cyan}>Current Market</Heading>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
            <Pill label="Mid Price" value={'$' + fmt(tick.mid, 3)} color={C.cyan} />
            <Pill label="Best Bid" value={'$' + fmt(tick.best_bid, 3)} color={C.green} />
            <Pill label="Best Ask" value={'$' + fmt(tick.best_ask, 3)} color={C.red} />
            <Pill label="Spread" value={'$' + fmt(tick.spread, 4)} color={C.purple} />
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
            <Pill label="Session ID" value={`#${status.collector.id}`} color={C.blue} />
            <Pill label="Started" value={new Date(status.collector.started_at).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })} color={C.blue} />
            <Pill label="Last Heartbeat" value={ago(status.collector.heartbeat_age_s)} color={status.collector.is_live ? C.green : C.red} />
            <Pill label="Reconnects" value={status.collector.reconnects} color={status.collector.reconnects > 3 ? C.amber : C.green} />
          </div>
        </>
      )}

      <DataCalendar isIngesting={status?.freshness?.is_ingesting} />
    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════
// DATA CALENDAR
// ═════════════════════════════════════════════════════════════════════════
function DataCalendar({ isIngesting }) {
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

  // Fetch calendar data when viewed month changes (backend param is 1-indexed)
  useEffect(() => {
    let cancelled = false;
    setCalLoading(true);
    fetch(`${API}/calendar?year=${year}&month=${month + 1}`)
      .then(r => r.json())
      .then(data => { if (!cancelled) { setCalData(data); setCalLoading(false); } })
      .catch(() => { if (!cancelled) setCalLoading(false); });
    return () => { cancelled = true; };
  }, [year, month]);

  const dayMap = useMemo(() => {
    const m = {};
    (calData?.days || []).forEach(d => { m[d.date] = d; });
    return m;
  }, [calData]);

  // Determine color for a day — lightweight boolean check only.
  // Full/partial/hourly detail is shown on click via /day-detail.
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

  // Load day detail when clicked
  const handleDayClick = useCallback(async (dateStr) => {
    if (selectedDay === dateStr) { setSelectedDay(null); setDayDetail(null); return; }
    setSelectedDay(dateStr);
    setLoadingDetail(true);
    try {
      const r = await fetch(`${API}/day-detail/${dateStr}`);
      setDayDetail(await r.json());
    } catch (_) { setDayDetail(null); }
    setLoadingDetail(false);
  }, [selectedDay]);

  // Calendar grid
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

      {/* UTC / PST clocks */}
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

      {/* Legend — simplified; full/partial detail shows on click */}
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

        {/* Month nav */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <button onClick={prevMonth} style={{ background: 'none', border: 'none', color: C.sub, cursor: 'pointer', fontSize: 13, padding: '4px 8px' }}>← Prev</button>
          <div style={{ fontSize: 16, fontWeight: 600, color: C.text }}>{MONTHS[month]} {year}</div>
          <button onClick={nextMonth} style={{ background: 'none', border: 'none', color: C.sub, cursor: 'pointer', fontSize: 13, padding: '4px 8px' }}>Next →</button>
        </div>

        {/* Weekday headers */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 4, marginBottom: 4 }}>
          {WEEKDAYS.map(w => (
            <div key={w} style={{ textAlign: 'center', fontSize: 11, fontWeight: 600, color: C.muted, padding: '4px 0' }}>{w}</div>
          ))}
        </div>

        {/* Grid */}
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

      {/* Day detail timeline */}
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
        {dateStr} &mdash; 24-Hour Timeline <span style={{ fontSize: 11, color: C.dim, fontWeight: 400 }}>(UTC)</span>
      </div>
      {isToday && (
        <div style={{ fontSize: 11, color: C.dim, marginBottom: 10 }}>
          Current hour: {currentHourUTC}:00 UTC &mdash; future hours shown as dark
        </div>
      )}

      {/* One row per component */}
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

      {/* Stats row */}
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

      {/* Legend */}
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
function FeedTab({ ticks, trades, status, mark, latestTick }) {
  // Prefer the 300ms fast-loop tick; fall back to chart data, then to status snapshot.
  const latest = latestTick || (ticks.length > 0 ? ticks[ticks.length - 1] : null);
  const tick = latest ? {
    mid: latest.mid,
    best_bid: latest.best_bid,
    best_ask: latest.best_ask,
    spread: latest.spread,
  } : status?.latest_tick;

  // Last trade = most recent trade row from our DB (1s poll)
  const lastTrade = trades && trades.length > 0 ? trades[trades.length - 1] : null;

  // Mark price row (1s poll from sol.mark_price)
  const markOk   = mark && mark.available && mark.is_fresh;
  const markPx   = markOk ? mark.mark_px : null;
  const markStale = mark && mark.available && !mark.is_fresh;
  const markVal = markPx != null
    ? '$' + fmt(markPx, 3)
    : (mark?.available === false ? 'no data' : '—');

  return (
    <div>
      {/* Current price pills */}
      {tick && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, marginBottom: 16 }}>
          <Pill label={markStale ? `Mark (${Math.round(mark.age_s)}s old)` : 'Mark'}
                value={markVal} color={C.purple} />
          <Pill label="Mid"        value={'$' + fmt(tick.mid, 3)}      color={C.cyan} />
          <Pill label="Last Trade" value={lastTrade ? '$' + fmt(lastTrade.px, 3) : '—'}
                color={lastTrade && lastTrade.side === 'B' ? C.green : C.red} />
          <Pill label="Bid"        value={'$' + fmt(tick.best_bid, 3)} color={C.green} />
          <Pill label="Ask"        value={'$' + fmt(tick.best_ask, 3)} color={C.red} />
        </div>
      )}

      {/* Mid price chart */}
      <Heading accent={C.cyan}>Live Mid Price</Heading>
      {ticks.length > 0 && (
        <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: '12px 8px' }}>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={ticks.map(t => ({ ...t, t: tsShort(t.ts) }))} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="t" tick={{ fontSize: 9, fill: C.dim }} interval={Math.floor(ticks.length / 8)} />
              <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10, fill: C.dim }} tickFormatter={v => '$' + v.toFixed(2)} />
              <Tooltip contentStyle={{ background: '#1e293b', border: 'none', borderRadius: 8, fontSize: 12 }}
                formatter={(v) => ['$' + Number(v).toFixed(4), 'Mid']} />
              <Line type="monotone" dataKey="mid" stroke={C.cyan} dot={false} strokeWidth={1.5} name="Mid" isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Spread chart */}
      <Heading accent={C.purple}>Spread</Heading>
      {ticks.length > 0 && (
        <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: '12px 8px' }}>
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={ticks.map(t => ({ ...t, t: tsShort(t.ts) }))} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="t" tick={{ fontSize: 9, fill: C.dim }} interval={Math.floor(ticks.length / 6)} />
              <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10, fill: C.dim }} tickFormatter={v => '$' + v.toFixed(4)} />
              <Tooltip contentStyle={{ background: '#1e293b', border: 'none', borderRadius: 8, fontSize: 12 }}
                formatter={(v) => ['$' + Number(v).toFixed(5), 'Spread']} />
              <Line type="monotone" dataKey="spread" stroke={C.purple} dot={false} strokeWidth={1} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Recent trades */}
      <Heading accent={C.green}>Recent Trades</Heading>
      <TradesTape trades={trades} />
    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════
// HYPERLIQUID TAB — with proxy selection, latency, and BTC Trading
// ═════════════════════════════════════════════════════════════════════════
function HyperliquidTab() {
  const [connected, setConnected] = useState(false);
  const [proxyIndex, setProxyIndex] = useState(-1);
  const [proxyLabel, setProxyLabel] = useState('');
  const [proxies, setProxies] = useState(null);
  const [proxiesLoading, setProxiesLoading] = useState(true);

  const [hlData, setHlData] = useState(null);
  const [hlLoading, setHlLoading] = useState(false);
  const [hlErr, setHlErr] = useState(null);

  const [latency, setLatency] = useState(null);
  const [latLoading, setLatLoading] = useState(false);

  const [hlSubTab, setHlSubTab] = useState('balances');

  useEffect(() => {
    setProxiesLoading(true);
    fetch(`${API}/hl-proxies`)
      .then(r => r.json())
      .then(d => { setProxies(d.proxies || []); setProxiesLoading(false); })
      .catch(() => setProxiesLoading(false));
  }, []);

  const loadBalances = useCallback(async (pi) => {
    setHlLoading(true);
    setHlErr(null);
    try {
      const res = await fetch(`${API}/hl-balances?proxy_index=${pi}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setHlData(data);
    } catch (e) { setHlErr(e.message); }
    setHlLoading(false);
  }, []);

  const loadLatency = useCallback(async (pi) => {
    setLatLoading(true);
    try {
      const res = await fetch(`${API}/hl-latency?proxy_index=${pi}`);
      const data = await res.json();
      setLatency(data);
    } catch (_) {}
    setLatLoading(false);
  }, []);

  const handleConnect = useCallback(async (pi, label) => {
    setProxyIndex(pi);
    setProxyLabel(label);
    setConnected(true);
    setHlSubTab('balances');
    loadBalances(pi);
    loadLatency(pi);
  }, [loadBalances, loadLatency]);

  const handleDisconnect = useCallback(() => {
    setConnected(false);
    setHlData(null);
    setLatency(null);
    setHlErr(null);
    setHlSubTab('balances');
  }, []);

  if (!connected) {
    return (
      <div>
        <Heading accent={C.purple}>Connect to Hyperliquid</Heading>
        <div style={{ fontSize: 13, color: C.sub, marginBottom: 20 }}>
          Choose how to connect to the Hyperliquid API. Different proxies may have different latency characteristics.
        </div>

        {proxiesLoading ? (
          <div style={{ padding: 32, textAlign: 'center', color: C.muted }}>Loading connection options...</div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
            <button
              onClick={() => handleConnect(-1, 'Direct (no proxy)')}
              style={{
                padding: '20px 16px', borderRadius: 10, border: `2px solid ${C.cyan}50`,
                background: C.surface, cursor: 'pointer', transition: 'all 0.15s',
                textAlign: 'left',
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = C.cyan; e.currentTarget.style.background = C.raised; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = C.cyan + '50'; e.currentTarget.style.background = C.surface; }}
            >
              <div style={{ fontSize: 14, fontWeight: 700, color: C.cyan, marginBottom: 4 }}>Direct</div>
              <div style={{ fontSize: 11, color: C.muted }}>No proxy — connect directly to api.hyperliquid.xyz</div>
            </button>

            {(proxies || []).map(p => (
              <button
                key={p.index}
                onClick={() => handleConnect(p.index, p.label)}
                style={{
                  padding: '20px 16px', borderRadius: 10, border: `2px solid ${C.purple}50`,
                  background: C.surface, cursor: 'pointer', transition: 'all 0.15s',
                  textAlign: 'left',
                }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = C.purple; e.currentTarget.style.background = C.raised; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = C.purple + '50'; e.currentTarget.style.background = C.surface; }}
              >
                <div style={{ fontSize: 14, fontWeight: 700, color: C.purple, marginBottom: 4 }}>Proxy {p.index + 1}</div>
                <div style={{ fontSize: 11, color: C.muted, fontFamily: 'monospace' }}>{p.label.split('(')[1]?.replace(')', '') || p.label}</div>
              </button>
            ))}
          </div>
        )}
      </div>
    );
  }

  const HL_SUB_TABS = [
    { id: 'balances', label: 'Balances' },
    { id: 'btc-trading', label: 'BTC Trading' },
  ];

  return (
    <div>
      {/* Connection bar */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16,
        padding: '10px 16px', background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <LiveDot active={true} />
          <span style={{ fontSize: 13, color: C.text, fontWeight: 600 }}>Connected via {proxyLabel}</span>
        </div>
        <button onClick={handleDisconnect} style={{
          padding: '5px 14px', borderRadius: 6, border: `1px solid ${C.red}40`,
          background: 'transparent', color: C.red, fontSize: 12, cursor: 'pointer',
        }}>Disconnect</button>
      </div>

      {/* Sub-tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, background: '#0f172a', borderRadius: 8, padding: 4, maxWidth: 320 }}>
        {HL_SUB_TABS.map(t => (
          <button key={t.id} onClick={() => setHlSubTab(t.id)} style={{
            flex: 1, padding: '7px 0', borderRadius: 6, border: 'none',
            fontSize: 12, fontWeight: 500, cursor: 'pointer', transition: 'all 0.15s',
            background: hlSubTab === t.id ? '#1e293b' : 'transparent',
            color: hlSubTab === t.id ? C.text : C.dim,
            boxShadow: hlSubTab === t.id ? '0 1px 4px #0004' : 'none',
          }}>{t.label}</button>
        ))}
      </div>

      {hlSubTab === 'balances' && (
        <HlBalancesSubTab
          data={hlData} loading={hlLoading} error={hlErr}
          onRefresh={() => loadBalances(proxyIndex)}
          latency={latency} latLoading={latLoading}
          onRefreshLatency={() => loadLatency(proxyIndex)}
        />
      )}

      {hlSubTab === 'btc-trading' && (
        <BtcTradingSubTab proxyIndex={proxyIndex} />
      )}
    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════
// BALANCES SUB-TAB (balances + latency panel)
// ═════════════════════════════════════════════════════════════════════════
function HlBalancesSubTab({ data, loading, error, onRefresh, latency, latLoading, onRefreshLatency }) {
  const refreshedAt = data?.mainnet?.fetched_at || data?.testnet?.fetched_at;

  return (
    <div>
      {/* Balances header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <Heading accent={C.purple}>Account Balances</Heading>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {refreshedAt && (
            <span style={{ fontSize: 11, color: C.muted }}>
              {new Date(refreshedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
            </span>
          )}
          <button onClick={onRefresh} disabled={loading} style={{
            padding: '6px 14px', borderRadius: 6, border: `1px solid ${C.border}`,
            background: loading ? C.raised : C.surface, color: loading ? C.muted : C.text,
            fontSize: 12, cursor: loading ? 'not-allowed' : 'pointer',
          }}>{loading ? 'Refreshing…' : '↻ Refresh'}</button>
        </div>
      </div>

      {error && (
        <div style={{ padding: '12px 16px', background: '#1a0a0a', border: `1px solid ${C.red}60`, borderRadius: 8, marginBottom: 16 }}>
          <span style={{ fontSize: 13, color: C.red }}>{error}</span>
        </div>
      )}

      {loading && !data && (
        <div style={{ textAlign: 'center', padding: 48, color: C.muted, fontSize: 14 }}>Fetching balances…</div>
      )}

      {data && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16, marginBottom: 24 }}>
          <HlNetCard net="mainnet" netData={data.mainnet} />
          <HlNetCard net="testnet" netData={data.testnet} />
        </div>
      )}

      {/* Latency panel */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <Heading accent={C.cyan}>Connection Latency</Heading>
        <button onClick={onRefreshLatency} disabled={latLoading} style={{
          padding: '6px 14px', borderRadius: 6, border: `1px solid ${C.border}`,
          background: latLoading ? C.raised : C.surface, color: latLoading ? C.muted : C.text,
          fontSize: 12, cursor: latLoading ? 'not-allowed' : 'pointer',
        }}>{latLoading ? 'Testing…' : '↻ Re-test'}</button>
      </div>

      {latLoading && !latency && (
        <div style={{ textAlign: 'center', padding: 32, color: C.muted, fontSize: 13 }}>Running latency benchmark ({5} samples per test)...</div>
      )}

      {latency && latency.tests && (
        <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: 16, overflow: 'auto' }}>
          <div style={{ fontSize: 11, color: C.muted, marginBottom: 12 }}>
            {latency.proxy_label} &middot; {latency.runs_per_test} samples/test
            {latency.fetched_at && <span> &middot; {new Date(latency.fetched_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
            {latency.tests.map(t => (
              <div key={t.name} style={{ background: C.raised, borderRadius: 8, padding: '12px 14px' }}>
                <div style={{ fontSize: 10, color: C.dim, letterSpacing: '0.06em', marginBottom: 6, textTransform: 'uppercase' }}>{t.label}</div>
                {t.runs > 0 ? (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
                    <LatStat label="Min" value={t.min} color={C.green} />
                    <LatStat label="Max" value={t.max} color={C.red} />
                    <LatStat label="Mean" value={t.mean} color={C.cyan} />
                    <LatStat label="Median" value={t.median} color={C.blue} />
                    <LatStat label="StDev" value={t.stdev} color={C.purple} />
                    <LatStat label="OK" value={`${t.runs}/${latency.runs_per_test}`} color={C.green} unit="" />
                  </div>
                ) : (
                  <div style={{ fontSize: 12, color: C.red }}>All {latency.runs_per_test} failed</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function LatStat({ label, value, color, unit = 'ms' }) {
  return (
    <div style={{ fontSize: 11 }}>
      <span style={{ color: C.dim }}>{label}: </span>
      <span style={{ color, fontFamily: 'monospace', fontWeight: 600 }}>{value}{unit}</span>
    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════
// BTC TRADING SUB-TAB
// ═════════════════════════════════════════════════════════════════════════
function BtcTradingSubTab({ proxyIndex }) {
  const [network, setNetwork] = useState('mainnet');
  const [btcPrice, setBtcPrice] = useState(null);
  const [btcMeta, setBtcMeta] = useState(null);
  const [priceLoading, setPriceLoading] = useState(false);
  const [metaLoading, setMetaLoading] = useState(true);

  const [netBal, setNetBal] = useState(null);
  const [netBalLoading, setNetBalLoading] = useState(false);

  // Trading station state
  const [side, setSide] = useState('long');
  const [marginInput, setMarginInput] = useState('');
  const [leverage, setLeverage] = useState(30);
  const [isCross, setIsCross] = useState(false);
  const [tpInput, setTpInput] = useState('');
  const [slInput, setSlInput] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [orderResult, setOrderResult] = useState(null);
  const [showConfirm, setShowConfirm] = useState(false);

  // Open orders
  const [openOrders, setOpenOrders] = useState([]);
  const [ordersLoading, setOrdersLoading] = useState(false);
  const [cancellingOid, setCancellingOid] = useState(null);

  const loadPrice = useCallback(async () => {
    setPriceLoading(true);
    try {
      const res = await fetch(`${API}/hl-btc-price?proxy_index=${proxyIndex}&network=mainnet`);
      setBtcPrice(await res.json());
    } catch (_) {}
    setPriceLoading(false);
  }, [proxyIndex]);

  const loadMeta = useCallback(async (net) => {
    setMetaLoading(true);
    try {
      const res = await fetch(`${API}/hl-btc-meta?proxy_index=${proxyIndex}&network=${net}`);
      setBtcMeta(await res.json());
    } catch (_) {}
    setMetaLoading(false);
  }, [proxyIndex]);

  const loadNetBal = useCallback(async (net) => {
    setNetBalLoading(true);
    try {
      const res = await fetch(`${API}/hl-net-balances?network=${net}&proxy_index=${proxyIndex}`);
      const d = await res.json();
      if (!d.error) setNetBal(d);
    } catch (_) {}
    setNetBalLoading(false);
  }, [proxyIndex]);

  const loadOpenOrders = useCallback(async (net) => {
    setOrdersLoading(true);
    try {
      const res = await fetch(`${API}/hl-open-orders?network=${net}&proxy_index=${proxyIndex}`);
      const d = await res.json();
      if (d.orders) setOpenOrders(d.orders);
    } catch (_) {}
    setOrdersLoading(false);
  }, [proxyIndex]);

  useEffect(() => {
    loadPrice();
    loadMeta(network);
    loadNetBal(network);
    loadOpenOrders(network);
  }, [network, loadPrice, loadMeta, loadNetBal, loadOpenOrders]);

  const handleNetSwitch = useCallback((net) => {
    if (net === network) return;
    setNetwork(net);
    setBtcMeta(null);
    setNetBal(null);
    setOpenOrders([]);
    setOrderResult(null);
  }, [network]);

  const maxLev = btcMeta?.maxLeverage || 40;
  const markPx = btcPrice?.mark_px ? Number(btcPrice.mark_px) : null;
  const oraclePx = btcPrice?.oracle_px ? Number(btcPrice.oracle_px) : null;
  const priceFmt = markPx
    ? '$' + markPx.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })
    : '—';

  const acctValue = netBal?.account_value ? Number(netBal.account_value) : 0;
  const withdrawable = netBal?.withdrawable ? Number(netBal.withdrawable) : 0;
  const marginUsed = netBal?.total_margin_used ? Number(netBal.total_margin_used) : 0;

  const isTestnet = network === 'testnet';
  const netAccent = isTestnet ? C.amber : C.green;
  const netLabel = isTestnet ? 'TESTNET' : 'MAINNET';

  // Position calculator
  const marginVal = parseFloat(marginInput) || 0;
  const tpVal = parseFloat(tpInput.replace(/,/g, '')) || null;
  const slVal = parseFloat(slInput.replace(/,/g, '')) || null;

  const calc = useMemo(() => {
    if (!markPx || marginVal <= 0 || leverage < 1) return null;

    const notional = marginVal * leverage;
    const posSz = notional / markPx;
    const maintMarginFrac = maxLev > 0 ? 1 / (2 * maxLev) : 0.0125;
    const maintMargin = maintMarginFrac * notional;

    let liqPrice;
    if (side === 'long') {
      liqPrice = markPx - (marginVal - maintMargin) / posSz;
    } else {
      liqPrice = markPx + (marginVal - maintMargin) / posSz;
    }

    const distToLiq = Math.abs(liqPrice - markPx);
    const distPct = (distToLiq / markPx) * 100;

    let tpProfit = null, tpPct = null;
    if (tpVal) {
      if (side === 'long') tpProfit = (tpVal - markPx) * posSz;
      else tpProfit = (markPx - tpVal) * posSz;
      tpPct = (tpProfit / marginVal) * 100;
    }

    let slLoss = null, slPct = null;
    if (slVal) {
      if (side === 'long') slLoss = (slVal - markPx) * posSz;
      else slLoss = (markPx - slVal) * posSz;
      slPct = (slLoss / marginVal) * 100;
    }

    return { notional, posSz, maintMargin, maintMarginFrac, liqPrice, distToLiq, distPct, tpProfit, tpPct, slLoss, slPct };
  }, [markPx, marginVal, leverage, maxLev, side, tpVal, slVal]);

  const handleMaxMargin = () => {
    setMarginInput(fmt(acctValue, 2));
  };

  const handleSubmitClick = () => {
    setOrderResult(null);
    setShowConfirm(true);
  };

  const handleConfirmOrder = async () => {
    setShowConfirm(false);
    setSubmitting(true);
    setOrderResult(null);
    try {
      const body = {
        network,
        side,
        margin: marginVal,
        leverage,
        is_cross: isCross,
        tp_price: tpVal,
        sl_price: slVal,
        proxy_index: proxyIndex,
      };
      const res = await fetch(`${API}/hl-place-order`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      setOrderResult(data);
      if (data.success) {
        loadNetBal(network);
        loadOpenOrders(network);
      }
    } catch (e) {
      setOrderResult({ success: false, error: e.message });
    }
    setSubmitting(false);
  };

  const handleCancelOrder = async (oid) => {
    setCancellingOid(oid);
    try {
      const res = await fetch(`${API}/hl-cancel-order`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ network, coin: 'BTC', oid, proxy_index: proxyIndex }),
      });
      await res.json();
      loadOpenOrders(network);
    } catch (_) {}
    setCancellingOid(null);
  };

  const sideColor = side === 'long' ? C.green : C.red;
  const sideLabel = side === 'long' ? 'LONG' : 'SHORT';

  const levPresets = useMemo(() => {
    const base = [1, 5, 10, 20, 30];
    if (maxLev && !base.includes(maxLev)) base.push(maxLev);
    return base.filter(l => l <= (maxLev || 50));
  }, [maxLev]);

  return (
    <div>
      {/* Network toggle */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <div style={{ display: 'flex', gap: 4, background: '#0f172a', borderRadius: 8, padding: 3 }}>
          {['mainnet', 'testnet'].map(net => {
            const active = network === net;
            const ac = net === 'testnet' ? C.amber : C.green;
            return (
              <button key={net} onClick={() => handleNetSwitch(net)} style={{
                padding: '6px 16px', borderRadius: 6, border: 'none', fontSize: 12, fontWeight: 600,
                cursor: 'pointer', transition: 'all 0.15s',
                background: active ? ac + '25' : 'transparent',
                color: active ? ac : C.dim,
              }}>{net === 'testnet' ? 'TESTNET' : 'MAINNET'}</button>
            );
          })}
        </div>
        <div style={{
          fontSize: 10, padding: '3px 10px', borderRadius: 4, letterSpacing: '0.1em',
          background: netAccent + '15', color: netAccent, border: `1px solid ${netAccent}40`,
        }}>{isTestnet ? 'PAPER TRADING' : 'REAL MONEY'}</div>
      </div>

      {/* Live BTC Price */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <Heading accent={C.orange}>BTC-PERP ({netLabel})</Heading>
        <button onClick={() => loadPrice()} disabled={priceLoading} style={{
          padding: '6px 14px', borderRadius: 6, border: `1px solid ${C.border}`,
          background: priceLoading ? C.raised : C.surface, color: priceLoading ? C.muted : C.text,
          fontSize: 12, cursor: priceLoading ? 'not-allowed' : 'pointer',
        }}>{priceLoading ? 'Loading…' : '↻ Refresh Price'}</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 10, marginBottom: 10 }}>
        <Pill label="BTC Mark Price" value={priceFmt} color={C.orange} />
        <Pill label="Oracle" value={oraclePx ? '$' + oraclePx.toLocaleString() : '—'} color={C.cyan} />
        <Pill label="Max Leverage" value={metaLoading ? '…' : (maxLev ? `${maxLev}x` : '—')} color={C.purple} />
        <Pill label="Funding (8h)" value={btcPrice?.funding ? (Number(btcPrice.funding) * 100).toFixed(4) + '%' : '—'} color={C.amber} />
      </div>

      {btcPrice?.fetched_at && (
        <div style={{ fontSize: 11, color: C.muted, marginBottom: 20 }}>
          Price as of {new Date(btcPrice.fetched_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          {maxLev && <span> &middot; True max leverage from Hyperliquid meta: <strong style={{ color: C.purple }}>{maxLev}x</strong></span>}
        </div>
      )}

      {/* Account Balance */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <Heading accent={C.cyan}>Account ({netLabel})</Heading>
        <button onClick={() => loadNetBal(network)} disabled={netBalLoading} style={{
          padding: '5px 12px', borderRadius: 6, border: `1px solid ${C.border}`,
          background: C.surface, color: netBalLoading ? C.muted : C.text, fontSize: 11,
          cursor: netBalLoading ? 'not-allowed' : 'pointer',
        }}>{netBalLoading ? '…' : '↻'}</button>
      </div>

      <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: 20, marginBottom: 16 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
          <div>
            <div style={{ fontSize: 11, color: C.dim, letterSpacing: '0.06em', marginBottom: 6, textTransform: 'uppercase' }}>Account Value</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: C.green, fontFamily: 'monospace' }}>
              {netBalLoading && !netBal ? '…' : `$${fmt(acctValue, 2)}`}
            </div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: C.dim, letterSpacing: '0.06em', marginBottom: 6, textTransform: 'uppercase' }}>Withdrawable</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: C.cyan, fontFamily: 'monospace' }}>
              {netBalLoading && !netBal ? '…' : `$${fmt(withdrawable, 2)}`}
            </div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: C.dim, letterSpacing: '0.06em', marginBottom: 6, textTransform: 'uppercase' }}>Margin Used</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: marginUsed > 0 ? C.amber : C.muted, fontFamily: 'monospace' }}>
              {netBalLoading && !netBal ? '…' : `$${fmt(marginUsed, 2)}`}
            </div>
          </div>
        </div>
      </div>

      {/* ─── Open Positions ─── */}
      <BtcPositionsPanel
        positions={(netBal?.positions || []).filter(p => p.coin === 'BTC')}
        openOrders={openOrders}
        markPx={markPx}
        network={network}
        netLabel={netLabel}
        onRefresh={() => { loadNetBal(network); loadOpenOrders(network); }}
        loading={netBalLoading}
      />

      {/* ─── Trading Station ─── */}
      <Heading accent={sideColor}>BTC Leverage Trading Station</Heading>
      <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>

        {/* Side toggle */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
          <button onClick={() => setSide('long')} style={{
            flex: 1, padding: '12px 0', borderRadius: 8, cursor: 'pointer', transition: 'all 0.15s',
            border: `2px solid ${side === 'long' ? C.green : C.green + '40'}`,
            background: side === 'long' ? C.green + '18' : 'transparent',
            color: side === 'long' ? C.green : C.green + '60',
            fontSize: 15, fontWeight: 700,
          }}>LONG</button>
          <button onClick={() => setSide('short')} style={{
            flex: 1, padding: '12px 0', borderRadius: 8, cursor: 'pointer', transition: 'all 0.15s',
            border: `2px solid ${side === 'short' ? C.red : C.red + '40'}`,
            background: side === 'short' ? C.red + '18' : 'transparent',
            color: side === 'short' ? C.red : C.red + '60',
            fontSize: 15, fontWeight: 700,
          }}>SHORT</button>
        </div>

        {/* Margin + Leverage */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
          <div>
            <div style={{ fontSize: 10, color: C.dim, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Margin (USDC)</div>
            <div style={{ display: 'flex', gap: 6 }}>
              <input
                type="text"
                placeholder={fmt(acctValue, 2) || '75.00'}
                value={marginInput}
                onChange={e => setMarginInput(e.target.value.replace(/[^0-9.]/g, ''))}
                style={{
                  flex: 1, padding: '10px 12px', borderRadius: 6, border: `1px solid ${C.border}`,
                  background: C.raised, color: C.text, fontSize: 14, fontFamily: 'monospace',
                  outline: 'none',
                }}
              />
              <button onClick={handleMaxMargin} style={{
                padding: '10px 12px', borderRadius: 6, border: `1px solid ${C.purple}40`,
                background: C.purple + '15', color: C.purple, fontSize: 11, fontWeight: 600,
                cursor: 'pointer',
              }}>MAX</button>
            </div>
          </div>
          <div>
            <div style={{ fontSize: 10, color: C.dim, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Leverage</div>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <input
                type="range"
                min="1"
                max={maxLev || 50}
                value={leverage}
                onChange={e => setLeverage(Number(e.target.value))}
                style={{ flex: 1 }}
              />
              <div style={{
                padding: '8px 12px', borderRadius: 6, background: C.raised, border: `1px solid ${C.border}`,
                color: C.purple, fontFamily: 'monospace', fontWeight: 700, fontSize: 16, minWidth: 50, textAlign: 'center',
              }}>{leverage}x</div>
            </div>
          </div>
        </div>

        {/* Leverage presets */}
        <div style={{ display: 'flex', gap: 6, marginBottom: 16 }}>
          {levPresets.map(lev => (
            <button key={lev} onClick={() => setLeverage(lev)} style={{
              flex: 1, padding: '6px 0', borderRadius: 5, cursor: 'pointer',
              border: `1px solid ${leverage === lev ? C.purple : C.border}`,
              background: leverage === lev ? C.purple + '20' : C.raised,
              color: leverage === lev ? C.purple : C.dim, fontSize: 11, fontWeight: 600,
              transition: 'all 0.1s',
            }}>{lev}x</button>
          ))}
        </div>

        {/* TP / SL */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
          <div>
            <div style={{ fontSize: 10, color: C.dim, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Take Profit Price</div>
            <input
              type="text"
              placeholder="Optional"
              value={tpInput}
              onChange={e => setTpInput(e.target.value.replace(/[^0-9.,]/g, ''))}
              style={{
                width: '100%', padding: '10px 12px', borderRadius: 6, border: `1px solid ${C.green}30`,
                background: C.raised, color: C.green, fontSize: 14, fontFamily: 'monospace', boxSizing: 'border-box',
                outline: 'none',
              }}
            />
          </div>
          <div>
            <div style={{ fontSize: 10, color: C.dim, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Stop Loss Price</div>
            <input
              type="text"
              placeholder="Optional"
              value={slInput}
              onChange={e => setSlInput(e.target.value.replace(/[^0-9.,]/g, ''))}
              style={{
                width: '100%', padding: '10px 12px', borderRadius: 6, border: `1px solid ${C.red}30`,
                background: C.raised, color: C.red, fontSize: 14, fontFamily: 'monospace', boxSizing: 'border-box',
                outline: 'none',
              }}
            />
          </div>
        </div>

        {/* Position Calculator */}
        <div style={{
          background: C.raised, borderRadius: 8, padding: 14, marginBottom: 16,
          border: `1px solid ${C.amber}20`,
        }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: C.amber, letterSpacing: '0.05em', marginBottom: 10, textTransform: 'uppercase' }}>
            Position Calculator
          </div>
          {calc ? (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8 }}>
              <CalcRow label="Position Notional" value={`$${calc.notional.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`} color={C.text} />
              <CalcRow label="Position Size" value={`${calc.posSz.toFixed(5)} BTC`} color={C.cyan} />
              <CalcRow label="Maintenance Margin" value={`$${fmt(calc.maintMargin, 2)} (${(calc.maintMarginFrac * 100).toFixed(2)}%)`} color={C.amber} />
              <CalcRow label="Liquidation Price" value={`$${calc.liqPrice.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} color={C.red} />
              <CalcRow label="Distance to Liq" value={`$${calc.distToLiq.toLocaleString(undefined, { maximumFractionDigits: 0 })} (${calc.distPct.toFixed(1)}%)`} color={C.red} />
              {calc.tpProfit != null && (
                <CalcRow label="Est. TP Profit" value={`${calc.tpProfit >= 0 ? '+' : ''}$${fmt(calc.tpProfit, 2)} (${calc.tpPct >= 0 ? '+' : ''}${fmt(calc.tpPct, 1)}%)`} color={calc.tpProfit >= 0 ? C.green : C.red} />
              )}
              {calc.slLoss != null && (
                <CalcRow label="Est. SL Loss" value={`${calc.slLoss >= 0 ? '+' : ''}$${fmt(calc.slLoss, 2)} (${calc.slPct >= 0 ? '+' : ''}${fmt(calc.slPct, 1)}%)`} color={calc.slLoss < 0 ? C.red : C.green} />
              )}
            </div>
          ) : (
            <div style={{ fontSize: 12, color: C.muted, fontStyle: 'italic' }}>Enter margin and leverage to see calculations</div>
          )}
        </div>

        {/* Isolated / Cross */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
          <button onClick={() => setIsCross(false)} style={{
            flex: 1, padding: '8px 0', borderRadius: 6, cursor: 'pointer', transition: 'all 0.1s',
            border: `1px solid ${!isCross ? C.purple : C.border}`,
            background: !isCross ? C.purple + '18' : 'transparent',
            color: !isCross ? C.purple : C.dim, fontSize: 12, fontWeight: 600,
          }}>Isolated</button>
          <button onClick={() => setIsCross(true)} style={{
            flex: 1, padding: '8px 0', borderRadius: 6, cursor: 'pointer', transition: 'all 0.1s',
            border: `1px solid ${isCross ? C.purple : C.border}`,
            background: isCross ? C.purple + '18' : 'transparent',
            color: isCross ? C.purple : C.dim, fontSize: 12,
          }}>Cross</button>
        </div>

        {/* Submit */}
        <button
          onClick={handleSubmitClick}
          disabled={submitting || marginVal <= 0 || !markPx}
          style={{
            width: '100%', padding: '14px 0', borderRadius: 8,
            background: submitting || marginVal <= 0 || !markPx
              ? '#333'
              : side === 'long'
                ? `linear-gradient(135deg, ${C.green}, #059669)`
                : `linear-gradient(135deg, ${C.red}, #be123c)`,
            border: 'none', color: '#fff', fontSize: 16, fontWeight: 700,
            letterSpacing: '0.04em',
            cursor: submitting || marginVal <= 0 || !markPx ? 'not-allowed' : 'pointer',
            transition: 'all 0.15s',
          }}
        >
          {submitting ? 'Submitting…' : `Submit ${sideLabel} Order`}
        </button>

        {/* Order result */}
        {orderResult && (
          <div style={{
            marginTop: 12, padding: '12px 16px', borderRadius: 8,
            background: orderResult.success ? '#052e16' : '#1a0a0a',
            border: `1px solid ${orderResult.success ? C.green + '60' : C.red + '60'}`,
          }}>
            {orderResult.success ? (
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: C.green, marginBottom: 6 }}>Order Submitted</div>
                <div style={{ fontSize: 11, color: C.sub }}>
                  {orderResult.side?.toUpperCase()} {orderResult.size_btc?.toFixed(5)} BTC @ ~${orderResult.mark_price?.toLocaleString()}
                  {orderResult.filled && <span> — Filled {orderResult.filled.totalSz} @ ${orderResult.filled.avgPx}</span>}
                  {orderResult.resting && <span> — Resting (oid: {orderResult.resting.oid})</span>}
                </div>
              </div>
            ) : (
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: C.red, marginBottom: 4 }}>Order Failed</div>
                <div style={{ fontSize: 11, color: '#fca5a5' }}>{orderResult.error || orderResult.order_error || 'Unknown error'}</div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Confirmation Modal */}
      {showConfirm && calc && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 1000,
          background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center',
        }} onClick={() => setShowConfirm(false)}>
          <div onClick={e => e.stopPropagation()} style={{
            background: C.bg, border: `2px solid ${sideColor}40`, borderRadius: 16,
            padding: 28, maxWidth: 440, width: '90%',
          }}>
            <div style={{ fontSize: 18, fontWeight: 700, color: sideColor, marginBottom: 16, textAlign: 'center' }}>
              Confirm {sideLabel} Order
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 20 }}>
              <CalcRow label="Network" value={netLabel} color={netAccent} />
              <CalcRow label="Side" value={sideLabel} color={sideColor} />
              <CalcRow label="Margin" value={`$${fmt(marginVal, 2)}`} color={C.text} />
              <CalcRow label="Leverage" value={`${leverage}x ${isCross ? '(Cross)' : '(Isolated)'}`} color={C.purple} />
              <CalcRow label="Notional" value={`$${calc.notional.toLocaleString(undefined, { maximumFractionDigits: 2 })}`} color={C.text} />
              <CalcRow label="Size" value={`${calc.posSz.toFixed(5)} BTC`} color={C.cyan} />
              <CalcRow label="Mark Price" value={priceFmt} color={C.orange} />
              <CalcRow label="Liquidation" value={`$${calc.liqPrice.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} color={C.red} />
              {tpVal && <CalcRow label="Take Profit" value={`$${tpVal.toLocaleString()}`} color={C.green} />}
              {slVal && <CalcRow label="Stop Loss" value={`$${slVal.toLocaleString()}`} color={C.red} />}
            </div>

            {!isTestnet && (
              <div style={{
                padding: '8px 12px', borderRadius: 6, marginBottom: 16,
                background: C.red + '12', border: `1px solid ${C.red}40`,
                fontSize: 11, color: C.red, textAlign: 'center',
              }}>
                MAINNET — This will execute with real money
              </div>
            )}

            <div style={{ display: 'flex', gap: 10 }}>
              <button onClick={() => setShowConfirm(false)} style={{
                flex: 1, padding: '12px 0', borderRadius: 8, border: `1px solid ${C.border}`,
                background: 'transparent', color: C.sub, fontSize: 14, fontWeight: 600, cursor: 'pointer',
              }}>Cancel</button>
              <button onClick={handleConfirmOrder} style={{
                flex: 1, padding: '12px 0', borderRadius: 8, border: 'none',
                background: side === 'long'
                  ? `linear-gradient(135deg, ${C.green}, #059669)`
                  : `linear-gradient(135deg, ${C.red}, #be123c)`,
                color: '#fff', fontSize: 14, fontWeight: 700, cursor: 'pointer',
              }}>Confirm {sideLabel}</button>
            </div>
          </div>
        </div>
      )}

      {/* Open Orders */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 24, marginBottom: 8 }}>
        <Heading accent={C.blue}>Open BTC Orders</Heading>
        <button onClick={() => loadOpenOrders(network)} disabled={ordersLoading} style={{
          padding: '5px 12px', borderRadius: 6, border: `1px solid ${C.border}`,
          background: C.surface, color: ordersLoading ? C.muted : C.text, fontSize: 11,
          cursor: ordersLoading ? 'not-allowed' : 'pointer',
        }}>{ordersLoading ? '…' : '↻'}</button>
      </div>

      {openOrders.length > 0 ? (
        <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, overflow: 'hidden' }}>
          <div style={{
            display: 'grid', gridTemplateColumns: '90px 90px 90px 70px 60px',
            background: C.raised, borderBottom: `1px solid ${C.border}`,
          }}>
            {['Purpose', 'Trigger', 'Size', 'OID', ''].map(h => (
              <div key={h} style={{ padding: '6px 8px', fontSize: 9, color: C.dim, letterSpacing: '0.08em' }}>{h}</div>
            ))}
          </div>
          {openOrders.map((o, i) => {
            const isTP = o.isPositionTpsl && o.orderType === 'Take Profit Market';
            const isSL = o.isPositionTpsl && o.orderType === 'Stop Market';
            const isTrigger = o.isTrigger;
            const purposeLabel = isTP ? 'TAKE PROFIT'
              : isSL ? 'STOP LOSS'
              : o.reduceOnly ? 'CLOSE'
              : o.side === 'B' ? 'LONG' : 'SHORT';
            const purposeColor = isTP ? C.green : isSL ? C.red : isTrigger ? C.amber : o.side === 'B' ? C.green : C.red;

            return (
              <div key={o.oid || i} style={{
                display: 'grid', gridTemplateColumns: '90px 90px 90px 70px 60px',
                borderBottom: i < openOrders.length - 1 ? `1px solid ${C.border}20` : 'none',
                background: i % 2 ? C.raised + '80' : 'transparent',
              }}>
                <div style={{ padding: '5px 8px', fontSize: 11, fontWeight: 700, color: purposeColor }}>
                  {purposeLabel}
                </div>
                <div style={{ padding: '5px 8px', fontSize: 11, color: C.text, fontFamily: 'monospace' }}>
                  {isTrigger ? `@ $${o.triggerPx || o.limitPx}` : `$${o.limitPx}`}
                </div>
                <div style={{ padding: '5px 8px', fontSize: 11, color: C.sub, fontFamily: 'monospace' }}>{o.sz} BTC</div>
                <div style={{ padding: '5px 8px', fontSize: 10, color: C.muted, fontFamily: 'monospace' }}>{o.oid}</div>
                <div style={{ padding: '5px 8px' }}>
                  <button
                    onClick={() => handleCancelOrder(o.oid)}
                    disabled={cancellingOid === o.oid}
                    style={{
                      padding: '2px 8px', borderRadius: 4, border: `1px solid ${C.red}40`,
                      background: 'transparent', color: C.red, fontSize: 10, cursor: 'pointer',
                    }}
                  >{cancellingOid === o.oid ? '…' : '✕'}</button>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div style={{ fontSize: 12, color: C.muted, fontStyle: 'italic', padding: '8px 0' }}>
          {ordersLoading ? 'Loading…' : 'No open BTC orders'}
        </div>
      )}
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════
// BTC POSITIONS PANEL
// ═════════════════════════════════════════════════════════════════════════
function BtcPositionsPanel({ positions, openOrders, markPx, network, netLabel, onRefresh, loading }) {
  if (!positions || positions.length === 0) return null;

  const tpslOrders = openOrders.filter(o => o.coin === 'BTC' && o.isPositionTpsl);
  const tpOrder = tpslOrders.find(o => o.orderType === 'Take Profit Market');
  const slOrder = tpslOrders.find(o => o.orderType === 'Stop Market');

  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <Heading accent={C.orange}>Open Positions</Heading>
        <button onClick={onRefresh} disabled={loading} style={{
          padding: '5px 12px', borderRadius: 6, border: `1px solid ${C.border}`,
          background: C.surface, color: loading ? C.muted : C.text, fontSize: 11,
          cursor: loading ? 'not-allowed' : 'pointer',
        }}>{loading ? '…' : '↻'}</button>
      </div>

      {positions.map((pos, idx) => {
        const sz = parseFloat(pos.size) || 0;
        const isLong = sz > 0;
        const dirColor = isLong ? C.green : C.red;
        const dirLabel = isLong ? 'LONG' : 'SHORT';

        const lev = pos.leverage || '—';
        const levType = pos.leverage_type === 'isolated' ? 'Isolated' : pos.leverage_type === 'cross' ? 'Cross' : '';
        const entryPx = parseFloat(pos.entry_px) || 0;
        const posValue = parseFloat(pos.position_value) || 0;
        const marginUsedPos = parseFloat(pos.margin_used) || 0;
        const pnl = parseFloat(pos.unrealized_pnl) || 0;
        const roe = parseFloat(pos.return_on_equity) || 0;
        const roePct = (roe * 100).toFixed(2);
        const liqPx = parseFloat(pos.liquidation_px) || 0;
        const funding = parseFloat(pos.cum_funding) || 0;

        const tpPx = tpOrder ? parseFloat(tpOrder.triggerPx) : null;
        const slPx = slOrder ? parseFloat(slOrder.triggerPx) : null;

        return (
          <div key={idx} style={{
            background: C.surface, border: `1px solid ${dirColor}30`, borderRadius: 12,
            overflow: 'hidden', marginBottom: 8,
          }}>
            {/* Header bar */}
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '10px 16px', background: dirColor + '0c',
              borderBottom: `1px solid ${dirColor}20`,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ fontSize: 16, fontWeight: 800, color: dirColor }}>{pos.coin}</span>
                <span style={{
                  fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4,
                  background: dirColor + '20', color: dirColor, letterSpacing: '0.05em',
                }}>{dirLabel}</span>
                <span style={{
                  fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 4,
                  background: C.purple + '18', color: C.purple,
                }}>{lev}x {levType}</span>
              </div>
              <div style={{
                fontSize: 18, fontWeight: 800, fontFamily: 'monospace',
                color: pnl >= 0 ? C.green : C.red,
              }}>
                {pnl >= 0 ? '+' : ''}{fmt(pnl, 2)} <span style={{ fontSize: 12, fontWeight: 600 }}>({pnl >= 0 ? '+' : ''}{roePct}%)</span>
              </div>
            </div>

            {/* Data grid — panic-friendly order: what can kill me → where I profit → context */}
            <div style={{ padding: '14px 16px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
                <PosField label="Liq. Price" value={`$${liqPx.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} color={C.red} />
                <PosField label="Take Profit" value={tpPx ? `$${tpPx.toLocaleString()}` : '—'} color={C.green} />
                <PosField label="Stop Loss" value={slPx ? `$${slPx.toLocaleString()}` : '—'} color={slPx ? C.red : C.muted} />
                <PosField label="Entry Price" value={`$${entryPx.toLocaleString()}`} color={C.text} />
                <PosField label="Mark Price" value={markPx ? `$${markPx.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}` : '—'} color={C.orange} />
                <PosField label="Size" value={`$${posValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`} color={C.text} />
                <PosField label="Size (BTC)" value={`${Math.abs(sz).toFixed(5)}`} color={C.muted} />
                <PosField label="Funding" value={funding !== 0 ? `${funding >= 0 ? '+' : ''}${fmt(funding, 4)}` : '—'} color={funding >= 0 ? C.green : C.red} />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function PosField({ label, value, color }) {
  return (
    <div>
      <div style={{ fontSize: 9, color: C.dim, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 3 }}>{label}</div>
      <div style={{ fontSize: 14, fontWeight: 700, color, fontFamily: 'monospace' }}>{value}</div>
    </div>
  );
}


function CalcRow({ label, value, color }) {
  return (
    <div>
      <div style={{ fontSize: 9, color: C.dim, letterSpacing: '0.06em', textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: 13, fontWeight: 600, color, fontFamily: 'monospace' }}>{value}</div>
    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════
// HL NET CARD (shared between old and new)
// ═════════════════════════════════════════════════════════════════════════
function HlNetCard({ net, netData }) {
  const isTestnet = net === 'testnet';
  const accentColor = isTestnet ? C.amber : C.green;
  const label = isTestnet ? 'TESTNET (paper)' : 'MAINNET (real)';

  const err = netData?.error;
  const mode = netData?.account_mode || 'unknown';
  const isUnified = mode === 'unifiedAccount' || mode === 'portfolioMargin';
  const positions = netData?.positions || [];
  const spotBalances = netData?.spot_balances || [];

  return (
    <div style={{ background: C.surface, border: `1px solid ${accentColor}30`, borderRadius: 12, overflow: 'hidden' }}>
      <div style={{
        padding: '14px 20px', borderBottom: `1px solid ${accentColor}25`,
        background: accentColor + '0c',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: accentColor, letterSpacing: '0.05em' }}>{label}</div>
          <div style={{ fontSize: 10, color: C.muted, marginTop: 2, fontFamily: 'monospace' }}>
            {netData?.master_address?.slice(0, 10)}…{netData?.master_address?.slice(-6)}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          {isUnified && (
            <div style={{
              fontSize: 9, padding: '2px 6px', borderRadius: 3,
              background: C.purple + '20', color: C.purple, border: `1px solid ${C.purple}40`,
              letterSpacing: '0.06em',
            }}>UNIFIED</div>
          )}
          <div style={{
            fontSize: 10, padding: '3px 8px', borderRadius: 4,
            background: isTestnet ? '#422006' : '#052e16',
            color: accentColor, border: `1px solid ${accentColor}40`,
            letterSpacing: '0.08em',
          }}>
            {isTestnet ? 'PAPER' : 'LIVE'}
          </div>
        </div>
      </div>

      <div style={{ padding: '16px 20px' }}>
        {err ? (
          <div style={{ fontSize: 13, color: C.red, padding: '8px 12px', background: '#1a0a0a', borderRadius: 6 }}>
            Error: {err}
          </div>
        ) : (
          <>
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: C.purple, letterSpacing: '0.08em', marginBottom: 10, textTransform: 'uppercase' }}>
                Account Summary
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                <HlBalRow label="Account Value" value={netData?.account_value} prefix="$" accent={accentColor} />
                <HlBalRow label="Withdrawable" value={netData?.withdrawable} prefix="$" accent={C.cyan} />
                <HlBalRow label="Margin Used" value={netData?.total_margin_used} prefix="$" accent={C.amber} />
                <HlBalRow label="Notional Pos" value={netData?.total_ntl_pos} prefix="$" accent={C.blue} />
              </div>

              {positions.length > 0 ? (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 10, color: C.dim, marginBottom: 6, letterSpacing: '0.06em' }}>
                    OPEN POSITIONS ({positions.length})
                  </div>
                  <div style={{ border: `1px solid ${C.border}`, borderRadius: 6, overflow: 'hidden' }}>
                    <div style={{
                      display: 'grid', gridTemplateColumns: '60px 70px 80px 1fr',
                      background: C.raised, borderBottom: `1px solid ${C.border}`,
                    }}>
                      {['Coin', 'Size', 'Entry', 'Unreal PnL'].map(h => (
                        <div key={h} style={{ padding: '5px 8px', fontSize: 9, color: C.dim, letterSpacing: '0.08em' }}>{h}</div>
                      ))}
                    </div>
                    {positions.map((p, i) => {
                      const pnl = parseFloat(p.unrealized_pnl || 0);
                      return (
                        <div key={i} style={{
                          display: 'grid', gridTemplateColumns: '60px 70px 80px 1fr',
                          borderBottom: i < positions.length - 1 ? `1px solid ${C.border}20` : 'none',
                          background: i % 2 ? C.raised + '80' : 'transparent',
                        }}>
                          <div style={{ padding: '5px 8px', fontSize: 12, color: C.text, fontWeight: 600 }}>{p.coin}</div>
                          <div style={{ padding: '5px 8px', fontSize: 11, color: parseFloat(p.size) >= 0 ? C.green : C.red, fontFamily: 'monospace' }}>{p.size}</div>
                          <div style={{ padding: '5px 8px', fontSize: 11, color: C.sub, fontFamily: 'monospace' }}>${fmt(p.entry_px, 2)}</div>
                          <div style={{ padding: '5px 8px', fontSize: 11, color: pnl >= 0 ? C.green : C.red, fontFamily: 'monospace' }}>
                            {pnl >= 0 ? '+' : ''}{fmt(pnl, 2)}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : (
                <div style={{ marginTop: 8, fontSize: 12, color: C.muted, fontStyle: 'italic' }}>No open positions</div>
              )}
            </div>

            <div style={{ height: 1, background: C.border, margin: '4px 0 16px' }} />

            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: C.cyan, letterSpacing: '0.08em', marginBottom: 10, textTransform: 'uppercase' }}>
                Token Balances {isUnified ? '(source of truth)' : '(spot wallet)'}
              </div>
              {spotBalances.length > 0 ? (
                <div style={{ border: `1px solid ${C.border}`, borderRadius: 6, overflow: 'hidden' }}>
                  <div style={{
                    display: 'grid', gridTemplateColumns: '60px 1fr 80px',
                    background: C.raised, borderBottom: `1px solid ${C.border}`,
                  }}>
                    {['Coin', 'Total', 'Hold'].map(h => (
                      <div key={h} style={{ padding: '5px 8px', fontSize: 9, color: C.dim, letterSpacing: '0.08em' }}>{h}</div>
                    ))}
                  </div>
                  {spotBalances.map((b, i) => (
                    <div key={i} style={{
                      display: 'grid', gridTemplateColumns: '60px 1fr 80px',
                      borderBottom: i < spotBalances.length - 1 ? `1px solid ${C.border}20` : 'none',
                      background: i % 2 ? C.raised + '80' : 'transparent',
                    }}>
                      <div style={{ padding: '5px 8px', fontSize: 12, color: C.text, fontWeight: 600 }}>{b.coin}</div>
                      <div style={{ padding: '5px 8px', fontSize: 11, color: accentColor, fontFamily: 'monospace' }}>{fmt(b.total, 4)}</div>
                      <div style={{ padding: '5px 8px', fontSize: 11, color: C.sub, fontFamily: 'monospace' }}>{fmt(b.hold, 4)}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ fontSize: 12, color: C.muted, fontStyle: 'italic' }}>No token balances</div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function HlBalRow({ label, value, prefix = '', accent = C.text }) {
  const display = value == null ? '—' : `${prefix}${fmt(value, 2)}`;
  return (
    <div style={{ background: C.raised, borderRadius: 6, padding: '10px 12px' }}>
      <div style={{ fontSize: 9, color: C.dim, letterSpacing: '0.08em', marginBottom: 3, textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 700, color: accent, fontFamily: 'monospace' }}>{display}</div>
    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════
// TRADES TAPE
// ═════════════════════════════════════════════════════════════════════════
function TradesTape({ trades }) {
  const containerRef = useRef(null);

  // Auto-scroll to top on new data
  useEffect(() => {
    if (containerRef.current) containerRef.current.scrollTop = 0;
  }, [trades]);

  if (!trades.length) return <div style={{ color: C.dim, padding: 12 }}>No trades yet.</div>;

  // Show newest first
  const reversed = useMemo(() => [...trades].reverse(), [trades]);

  return (
    <div ref={containerRef} style={{
      background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8,
      maxHeight: 360, overflowY: 'auto', overflowX: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        display: 'grid', gridTemplateColumns: '100px 50px 90px 70px',
        background: C.raised, borderBottom: `1px solid ${C.border}`,
        position: 'sticky', top: 0, zIndex: 1,
      }}>
        {['Time', 'Side', 'Price', 'Size'].map(h => (
          <div key={h} style={{ padding: '6px 10px', fontSize: 10, letterSpacing: '0.1em', color: C.dim }}>{h}</div>
        ))}
      </div>
      {reversed.map((tr, i) => (
        <div key={i} style={{
          display: 'grid', gridTemplateColumns: '100px 50px 90px 70px',
          borderBottom: `1px solid ${C.border}08`,
          background: i % 2 === 0 ? 'transparent' : '#0f172a',
        }}>
          <div style={{ padding: '3px 10px', fontSize: 11, color: C.dim }}>{tsShort(tr.ts)}</div>
          <div style={{ padding: '3px 10px', fontSize: 11, fontWeight: 600, color: tr.side === 'B' ? C.green : C.red }}>
            {tr.side === 'B' ? 'BUY' : 'SELL'}
          </div>
          <div style={{ padding: '3px 10px', fontSize: 11, color: C.text, fontFamily: 'monospace' }}>${fmt(tr.px, 3)}</div>
          <div style={{ padding: '3px 10px', fontSize: 11, color: C.sub, fontFamily: 'monospace' }}>{fmt(tr.sz, 2)}</div>
        </div>
      ))}
    </div>
  );
}
