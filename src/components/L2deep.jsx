/**
 * L2 DEEP DASHBOARD — L2deep.jsx
 * ============================================================================
 * Coverage dashboard for the REST-polled 20-level order book tables
 * (`<schema>.l2_deep`, written by sol-perp/l2_deep_collector.py).
 *
 * Mirrors the 5s-bucket coverage view in bucket.jsx: a coin selector across the
 * top (BTC / SOL / ETH / SPX), a live poller-status strip, a month calendar of
 * per-day row counts, and an hourly breakdown (UTC) when a day is clicked — so
 * you can spot any hour/day where the 20-level deep feed was missing or shallow.
 *
 * Pure presentational + fetch component. Mounted via the `l2deep` dashboard
 * mode in TradingDashboard.jsx.
 */

import { useState, useEffect, useMemo, useCallback } from 'react';

// ── theme (mirrors bucket.jsx) ──────────────────────────────────────────────
const C = {
  bg: '#060c18', surface: '#0c1628', raised: '#101e35', border: '#1a2d4e',
  text: '#e2e8f0', sub: '#94a3b8', muted: '#475569', dim: '#64748b',
  green: '#00d4a8', red: '#f43f5e', amber: '#eab308',
  blue: '#60a5fa', purple: '#a78bfa', cyan: '#22d3ee',
  orange: '#e97316', pink: '#ec4899', emerald: '#10b981', fuchsia: '#d946ef',
};

const mono = { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' };

const API = (asset) => `http://localhost:8000/api/perp/${asset}/l2deep`;
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
  'August', 'September', 'October', 'November', 'December'];
const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

// poll-rate driven; defaults match the backend (1 Hz target). Overridden by the
// values the API reports so the two never drift.
const DEFAULT_EXPECTED_PER_HOUR = 3600;
const DEFAULT_HOUR_GREEN = 3000;
const EXPECTED_LEVELS = 20;

const agoStr = (s) => {
  if (s == null) return '--';
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m ago`;
  return `${Math.floor(s / 86400)}d ago`;
};
const kfmt = (n) => {
  if (n == null) return '--';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
};

function LiveDot({ active }) {
  return (
    <span style={{
      display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
      background: active ? C.green : C.red,
      boxShadow: active ? `0 0 8px ${C.green}80` : 'none',
      animation: active ? 'l2dpulse 2s infinite' : 'none',
    }}>
      <style>{`@keyframes l2dpulse{0%,100%{opacity:1}50%{opacity:.4}}`}</style>
    </span>
  );
}

function L2DeepCoverage({ asset }) {
  const api = API(asset);
  const now = new Date();
  const utcYear = now.getUTCFullYear();
  const utcMonth = now.getUTCMonth();
  const utcDay = now.getUTCDate();
  const todayStr = `${utcYear}-${String(utcMonth + 1).padStart(2, '0')}-${String(utcDay).padStart(2, '0')}`;

  const [status, setStatus] = useState(null);
  const [month, setMonth] = useState(utcMonth);
  const [year, setYear] = useState(utcYear);
  const [calData, setCalData] = useState(null);
  const [calLoading, setCalLoading] = useState(true);
  const [selectedDay, setSelectedDay] = useState(null);
  const [dayDetail, setDayDetail] = useState(null);
  const [loadingDetail, setLoadingDetail] = useState(false);

  const hourGreen = status?.hour_green_threshold ?? DEFAULT_HOUR_GREEN;
  const expectedPerHour = status?.expected_per_hour ?? DEFAULT_EXPECTED_PER_HOUR;
  const dayGreen = (calData?.day_green_threshold) ?? hourGreen * 24;
  const expectedPerDay = (calData?.expected_per_day) ?? expectedPerHour * 24;

  // status poll (poller live dot) every 10s
  useEffect(() => {
    let cancelled = false;
    const load = () => fetch(`${api}/status`).then(r => r.json())
      .then(d => { if (!cancelled) setStatus(d); }).catch(() => {});
    load();
    const id = setInterval(load, 10_000);
    return () => { cancelled = true; clearInterval(id); };
  }, [api]);

  // calendar load on month change
  useEffect(() => {
    let cancelled = false;
    setCalLoading(true);
    fetch(`${api}/calendar?year=${year}&month=${month + 1}`)
      .then(r => r.json())
      .then(d => { if (!cancelled) { setCalData(d); setCalLoading(false); } })
      .catch(() => { if (!cancelled) setCalLoading(false); });
    return () => { cancelled = true; };
  }, [api, year, month]);

  const dayMap = useMemo(() => {
    const m = {};
    (calData?.days || []).forEach(d => { m[d.date] = d; });
    return m;
  }, [calData]);

  const handleDayClick = useCallback(async (dateStr) => {
    if (selectedDay === dateStr) { setSelectedDay(null); setDayDetail(null); return; }
    setSelectedDay(dateStr);
    setLoadingDetail(true);
    try {
      const r = await fetch(`${api}/day-detail/${dateStr}`);
      setDayDetail(await r.json());
    } catch (_) { setDayDetail(null); }
    setLoadingDetail(false);
  }, [selectedDay, api]);

  const firstDow = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();

  const dayColor = useCallback((dateStr, rows) => {
    const isToday = dateStr === todayStr;
    const isFuture = dateStr > todayStr;
    if (isFuture) return { bg: 'transparent', border: 'transparent', faded: true };
    if (rows == null) {
      return { bg: 'transparent', border: isToday ? C.cyan : C.red + '70', faded: !isToday };
    }
    if (rows >= dayGreen) return { bg: C.green, border: isToday ? C.cyan : 'transparent', faded: false };
    if (rows > 0) return { bg: C.amber, border: isToday ? C.cyan : 'transparent', faded: false };
    return { bg: 'transparent', border: isToday ? C.cyan : C.red + '70', faded: !isToday };
  }, [todayStr, dayGreen]);

  const cells = useMemo(() => {
    const arr = [];
    for (let i = 0; i < firstDow; i++) arr.push({ type: 'empty', key: `e${i}` });
    for (let d = 1; d <= daysInMonth; d++) {
      const ds = `${year}-${String(month + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
      const rec = dayMap[ds];
      const rows = rec ? rec.rows : (ds > todayStr ? undefined : null);
      arr.push({ type: 'day', key: ds, day: d, dateStr: ds, rows, isToday: ds === todayStr });
    }
    return arr;
  }, [year, month, firstDow, daysInMonth, dayMap, todayStr]);

  const prevMonth = () => { if (month === 0) { setMonth(11); setYear(y => y - 1); } else setMonth(m => m - 1); };
  const nextMonth = () => { if (month === 11) { setMonth(0); setYear(y => y + 1); } else setMonth(m => m + 1); };

  const live = !!status?.is_live;
  const available = status?.available !== false;
  const depthOk = status?.latest_ask_levels === EXPECTED_LEVELS && status?.latest_bid_levels === EXPECTED_LEVELS;

  return (
    <div style={{ marginBottom: 36 }}>
      {/* poller status strip */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 14, marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 3, height: 20, background: C.cyan, borderRadius: 2 }} />
          <div style={{ fontSize: 17, fontWeight: 700, color: C.text }}>20-Level L2 Coverage</div>
          <code style={{ ...mono, color: C.cyan, fontSize: 12 }}>{asset}.l2_deep</code>
          {available && (
            <span style={{
              fontSize: 10, padding: '2px 8px', borderRadius: 999, fontWeight: 700,
              color: depthOk ? C.green : C.amber,
              background: (depthOk ? C.green : C.amber) + '1a',
              border: `1px solid ${(depthOk ? C.green : C.amber)}55`,
            }}>
              {status?.latest_ask_levels != null
                ? `${status.latest_bid_levels}×${status.latest_ask_levels} levels${depthOk ? ' ✓' : ' ⚠'}`
                : 'depth ?'}
            </span>
          )}
        </div>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 12,
          background: live ? '#071812' : '#1a0d12',
          border: `1px solid ${live ? C.green + '60' : C.red + '55'}`,
          borderRadius: 10, padding: '10px 16px',
        }}>
          <LiveDot active={live} />
          <div>
            <div style={{ fontSize: 9, letterSpacing: '0.15em', color: live ? C.green + 'cc' : C.red + 'cc', fontWeight: 700 }}>
              {available ? (live ? 'L2 DEEP POLLER LIVE' : 'L2 DEEP POLLER STALLED') : 'NO l2_deep TABLE'}
            </div>
            <div style={{ fontSize: 12, color: live ? C.green : C.sub, marginTop: 2 }}>
              {status?.latest_snapshot
                ? `Last snapshot: ${agoStr(status.age_s)}`
                : 'no snapshots yet'}
              {status?.total_rows != null && (
                <span style={{ color: C.dim }}> · {kfmt(status.total_rows)} total</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* legend */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 12, flexWrap: 'wrap' }}>
        {[
          { color: C.green, label: `Full day (≥ ${kfmt(dayGreen)} rows)` },
          { color: C.amber, label: 'Partial day' },
          { color: C.red, label: 'No data', outline: true },
        ].map(l => (
          <div key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: C.sub }}>
            <div style={{
              width: 12, height: 12, borderRadius: 3,
              background: l.outline ? 'transparent' : l.color + '55',
              border: l.outline ? `2px solid ${l.color}80` : `1px solid ${l.color}`,
            }} />
            {l.label}
          </div>
        ))}
      </div>

      {/* calendar */}
      <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 18, position: 'relative' }}>
        {calLoading && (
          <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.35)', borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 2, fontSize: 13, color: C.dim }}>
            Loading…
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
            if (cell.type === 'empty') return <div key={cell.key} style={{ height: 58 }} />;
            const { day, dateStr, rows, isToday } = cell;
            const col = dayColor(dateStr, rows);
            const hasData = rows != null && rows > 0;
            const isSelected = dateStr === selectedDay;
            const pct = hasData ? Math.min(100, Math.round((rows / expectedPerDay) * 100)) : null;
            return (
              <button
                key={cell.key}
                onClick={() => hasData && handleDayClick(dateStr)}
                title={rows != null ? `${dateStr}: ${kfmt(rows)} rows` : dateStr}
                style={{
                  height: 58, borderRadius: 6, border: 'none',
                  background: col.bg !== 'transparent' ? col.bg + '40' : (isSelected ? C.blue + '20' : C.raised),
                  outline: col.border !== 'transparent' ? `2px solid ${col.border}` : (isSelected ? `2px solid ${C.blue}` : 'none'),
                  outlineOffset: -2,
                  cursor: hasData ? 'pointer' : 'default',
                  display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                  transition: 'all .12s', opacity: col.faded ? 0.35 : 1,
                }}
              >
                <div style={{ fontSize: 14, fontWeight: isToday ? 700 : 500, color: isToday ? C.cyan : C.text }}>{day}</div>
                {hasData && (
                  <div style={{ fontSize: 8, fontWeight: 700, color: rows >= dayGreen ? C.green : C.amber, marginTop: 2 }}>
                    {pct}%
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {selectedDay && (
        <L2DeepHourBreakdown
          dateStr={selectedDay} detail={dayDetail} loading={loadingDetail}
          hourGreen={hourGreen} expectedPerHour={expectedPerHour}
        />
      )}
    </div>
  );
}

function L2DeepHourBreakdown({ dateStr, detail, loading, hourGreen, expectedPerHour }) {
  if (loading) return <div style={{ padding: 18, color: C.dim, textAlign: 'center' }}>Loading hourly breakdown…</div>;
  if (!detail || !detail.hours || !detail.hours.length) return null;

  const now = new Date();
  const todayUTC = `${now.getUTCFullYear()}-${String(now.getUTCMonth() + 1).padStart(2, '0')}-${String(now.getUTCDate()).padStart(2, '0')}`;
  const isToday = dateStr === todayUTC;
  const isFutureDay = dateStr > todayUTC;
  const currentHourUTC = now.getUTCHours();

  const pastHours = detail.hours.filter(h => {
    if (isFutureDay) return false;
    if (!isToday) return true;
    return h.hour <= currentHourUTC;
  });
  const healthy = pastHours.filter(h => h.rows >= hourGreen).length;
  const total = pastHours.reduce((s, h) => s + h.rows, 0);
  const anyShallow = pastHours.some(h => h.rows > 0 && h.max_levels != null && h.max_levels < EXPECTED_LEVELS);

  return (
    <div style={{ marginTop: 16, background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: '16px 20px' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: C.text }}>
          {dateStr} — Hourly 20-Level Rows <span style={{ fontSize: 11, color: C.dim, fontWeight: 400 }}>(UTC · ~{kfmt(expectedPerHour)}/hr @ 1 Hz)</span>
        </div>
        <div style={{ fontSize: 12, color: C.sub }}>
          <span style={{ color: healthy === pastHours.length ? C.green : C.amber, fontWeight: 600 }}>{healthy}/{pastHours.length}</span> hours ≥{kfmt(hourGreen)}
          <span style={{ color: C.dim }}> · {kfmt(total)} rows</span>
          {anyShallow && <span style={{ color: C.amber }}> · ⚠ some &lt;20-level</span>}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: 4 }}>
        {detail.hours.map(h => {
          const isFuture = isFutureDay || (isToday && h.hour > currentHourUTC);
          const shallow = h.rows > 0 && h.max_levels != null && h.max_levels < EXPECTED_LEVELS;
          let bg, borderClr, textClr;
          if (isFuture) {
            bg = '#0e1626'; borderClr = '#2d3748'; textClr = '#334155';
          } else if (h.rows >= hourGreen) {
            bg = C.green + '28'; borderClr = C.green; textClr = C.green;
          } else if (h.rows > 0) {
            bg = C.amber + '25'; borderClr = C.amber; textClr = C.amber;
          } else {
            bg = C.red + '1a'; borderClr = C.red + '70'; textClr = C.red + 'cc';
          }
          if (shallow) borderClr = C.orange;
          return (
            <div key={h.hour}
              title={isFuture
                ? `${h.hour}:00 — not yet`
                : `${h.hour}:00 UTC — ${kfmt(h.rows)} rows · levels ${h.min_levels ?? '?'}–${h.max_levels ?? '?'}`}
              style={{
                borderRadius: 5, background: bg, borderBottom: `2px solid ${borderClr}`,
                padding: '8px 2px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
              }}>
              <div style={{ fontSize: 10, color: C.dim }}>{String(h.hour).padStart(2, '0')}</div>
              <div style={{ fontSize: 10, fontWeight: 700, color: textClr }}>{isFuture ? '—' : kfmt(h.rows)}</div>
            </div>
          );
        })}
      </div>

      <div style={{ display: 'flex', gap: 14, marginTop: 12, flexWrap: 'wrap' }}>
        {[
          { color: C.green, label: `Healthy (≥ ${kfmt(hourGreen)})` },
          { color: C.amber, label: 'Partial' },
          { color: C.red, label: 'Empty (0)' },
          { color: C.orange, label: '<20 levels (shallow)' },
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

// ── coin selector (mirrors bucket.jsx) ──────────────────────────────────────
const L2D_ASSETS = [
  { id: 'btc', label: 'BTC', color: C.amber },
  { id: 'sol', label: 'SOL', color: C.purple },
  { id: 'eth', label: 'ETH', color: C.blue },
  { id: 'spx', label: 'SPX', color: C.green },
];

function L2DeepAssetSwitcher({ asset, onChange }) {
  return (
    <div style={{ display: 'flex', gap: 6, marginBottom: 14 }}>
      {L2D_ASSETS.map((a) => {
        const active = a.id === asset;
        return (
          <button
            key={a.id}
            onClick={() => onChange(a.id)}
            style={{
              cursor: 'pointer', padding: '7px 16px', borderRadius: 9, fontSize: 13,
              fontWeight: 700, letterSpacing: 0.4,
              color: active ? C.bg : a.color,
              background: active ? a.color : 'transparent',
              border: `1px solid ${a.color}${active ? '' : '66'}`,
              transition: 'all .12s',
            }}
          >
            {a.label}
          </button>
        );
      })}
    </div>
  );
}

// ── main ─────────────────────────────────────────────────────────────────────
export default function L2DeepDashboard() {
  const [asset, setAsset] = useState('btc');

  return (
    <div style={{ background: C.bg, minHeight: '100vh', color: C.text, padding: '24px 28px 80px' }}>
      <div style={{ maxWidth: 1180, margin: '0 auto' }}>
        {/* header */}
        <div style={{ marginBottom: 18 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 26 }}>▦</span>
            <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800, letterSpacing: 0.4,
              background: `linear-gradient(90deg, ${C.cyan}, ${C.green})`,
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              L2 DEEP DASHBOARD
            </h1>
          </div>
          <p style={{ color: C.sub, margin: '8px 0 0', fontSize: 13.5, maxWidth: 820, lineHeight: 1.6 }}>
            REST-polled full <b style={{ color: C.text }}>20-level</b> order book coverage —{' '}
            <code style={{ ...mono, color: C.cyan }}>{asset}.l2_deep</code>, written by{' '}
            <code style={{ ...mono, color: C.green }}>l2_deep_collector.py</code>. Click any day for the hourly UTC breakdown.
          </p>
        </div>

        <L2DeepAssetSwitcher asset={asset} onChange={setAsset} />
        <L2DeepCoverage key={asset} asset={asset} />
      </div>
    </div>
  );
}
