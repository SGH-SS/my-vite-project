/**
 * InstrumentChart - 1m OHLCV candlestick chart with 1m/5m/15m/4h aggregation.
 * Supports candle selection on 1m timeframe to fetch microstructure data,
 * plus Volume Profile, Trade Flow, and L1 Imbalance visualizations.
 */

import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { createChart, CandlestickSeries, ColorType, CrosshairMode } from 'lightweight-charts';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, ReferenceLine, AreaChart, Area, Cell
} from 'recharts';

const API_BASE_URL = 'http://localhost:8000/api/databento';

const CHART_COLORS = {
  light: {
    background: '#ffffff',
    text: '#374151',
    grid: '#f3f4f6',
    border: '#d1d5db',
    upColor: '#22c55e',
    downColor: '#ef4444',
  },
  dark: {
    background: '#1f2937',
    text: '#e5e7eb',
    grid: '#374151',
    border: '#4b5563',
    upColor: '#22c55e',
    downColor: '#ef4444',
  },
};

const TIMEFRAMES = [
  { key: '1m', label: '1m', minutes: 1 },
  { key: '5m', label: '5m', minutes: 5 },
  { key: '15m', label: '15m', minutes: 15 },
  { key: '4h', label: '4h', minutes: 240 },
];

const timestampToUnix = (ts) => Math.floor(new Date(ts).getTime() / 1000);

function aggregateBars(bars1m, periodMinutes) {
  if (!bars1m || bars1m.length === 0) return [];
  if (periodMinutes <= 1) return bars1m;

  const periodMs = periodMinutes * 60 * 1000;
  const buckets = new Map();

  bars1m.forEach((b) => {
    const t = new Date(b.timestamp).getTime();
    const bucketStart = new Date(Math.floor(t / periodMs) * periodMs);
    const key = bucketStart.getTime();
    if (!buckets.has(key)) {
      buckets.set(key, {
        timestamp: bucketStart.toISOString(),
        open: parseFloat(b.open),
        high: parseFloat(b.high),
        low: parseFloat(b.low),
        close: parseFloat(b.close),
        volume: b.volume != null ? Number(b.volume) : 0,
      });
    } else {
      const agg = buckets.get(key);
      agg.high = Math.max(agg.high, parseFloat(b.high));
      agg.low = Math.min(agg.low, parseFloat(b.low));
      agg.close = parseFloat(b.close);
      agg.volume += b.volume != null ? Number(b.volume) : 0;
    }
  });

  return Array.from(buckets.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([, v]) => v);
}

function timeToMinutes(timeStr) {
  const [h, m] = timeStr.split(':').map(Number);
  return h * 60 + m;
}

function minutesToTime(mins) {
  return `${String(Math.floor(mins / 60)).padStart(2, '0')}:${String(mins % 60).padStart(2, '0')}`;
}

function aggregateMicroBars(bars, periodMinutes) {
  if (!bars || bars.length === 0) return [];
  if (periodMinutes <= 1) return bars;

  const buckets = new Map();
  bars.forEach((b) => {
    const mins = timeToMinutes(b.time);
    const bucketStart = Math.floor(mins / periodMinutes) * periodMinutes;
    const key = minutesToTime(bucketStart);
    if (!buckets.has(key)) {
      buckets.set(key, { time: key, buy_volume: 0, sell_volume: 0, volume: 0 });
    }
    const agg = buckets.get(key);
    agg.buy_volume += b.buy_volume || 0;
    agg.sell_volume += b.sell_volume || 0;
    agg.volume += b.volume || 0;
  });

  return Array.from(buckets.values()).sort(
    (a, b) => timeToMinutes(a.time) - timeToMinutes(b.time)
  );
}

function barsToChartData(bars, selectedSet = new Set()) {
  return bars.map((b) => {
    const time = timestampToUnix(b.timestamp);
    const base = { time, open: b.open, high: b.high, low: b.low, close: b.close };
    if (selectedSet.has(time)) {
      return { ...base, color: '#3b82f6', borderColor: '#2563eb', wickColor: '#60a5fa' };
    }
    return base;
  });
}

// =============================================================================
// MULTI-DAY DELTA HELPERS
// =============================================================================

const DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

function getWeekMonday(dateStr) {
  const [y, m, d] = dateStr.split('-').map(Number);
  const date = new Date(Date.UTC(y, m - 1, d));
  const day = date.getUTCDay();
  const diff = day === 0 ? 6 : day - 1;
  date.setUTCDate(date.getUTCDate() - diff);
  return date.toISOString().slice(0, 10);
}

function getBiweeklyMonday(dateStr) {
  const anchor = Date.UTC(2025, 11, 1); // Dec 1, 2025 (Monday)
  const [y, m, d] = dateStr.split('-').map(Number);
  const target = Date.UTC(y, m - 1, d);
  const diffDays = Math.floor((target - anchor) / 86400000);
  if (diffDays < 0) return dateStr;
  const period = Math.floor(diffDays / 14);
  return new Date(anchor + period * 14 * 86400000).toISOString().slice(0, 10);
}

function generateWeekdayDates(startDate, endDate) {
  const dates = [];
  const [sy, sm, sd] = startDate.split('-').map(Number);
  const [ey, em, ed] = endDate.split('-').map(Number);
  let current = new Date(Date.UTC(sy, sm - 1, sd));
  const end = new Date(Date.UTC(ey, em - 1, ed));
  while (current <= end) {
    if (current.getUTCDay() >= 1 && current.getUTCDay() <= 5) {
      dates.push(current.toISOString().slice(0, 10));
    }
    current.setUTCDate(current.getUTCDate() + 1);
  }
  return dates;
}

async function fetchMultiDayMicroBars(dates, symbol, aggregateMinutes) {
  const results = await Promise.allSettled(
    dates.map((date) =>
      fetch(`${API_BASE_URL}/micro-detail/${date}/${encodeURIComponent(symbol)}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => ({ date, bars: data?.bars || [] }))
        .catch(() => ({ date, bars: [] }))
    )
  );

  const allBars = [];
  for (const result of results) {
    if (result.status !== 'fulfilled') continue;
    const { date, bars } = result.value;
    if (bars.length === 0) continue;
    const d = new Date(date + 'T12:00:00Z');
    const dayName = DAY_NAMES[d.getUTCDay()];
    const agg = aggregateMinutes > 1 ? aggregateMicroBars(bars, aggregateMinutes) : bars;
    for (const b of agg) {
      allBars.push({
        time: `${dayName} ${b.time}`,
        sortKey: `${date}T${b.time}`,
        buy_volume: b.buy_volume || 0,
        sell_volume: b.sell_volume || 0,
        volume: b.volume || 0,
      });
    }
  }
  allBars.sort((a, b) => a.sortKey.localeCompare(b.sortKey));
  return allBars;
}

function computeCumDelta(bars) {
  if (!bars || bars.length === 0) return [];
  let cumDelta = 0;
  return bars.map((b) => {
    const net = (b.buy_volume || 0) - (b.sell_volume || 0);
    cumDelta += net;
    return { time: b.time, cumDelta, netDelta: net };
  });
}


// =============================================================================
// VOLUME PROFILE CHART (pure SVG horizontal bars)
// =============================================================================

const VolumeProfileChart = ({ bars1m, selectedBarTimes, isDarkMode }) => {
  const profile = useMemo(() => {
    if (!bars1m || bars1m.length === 0 || selectedBarTimes.size === 0) return [];

    const selectedBars = bars1m.filter(
      (b) => selectedBarTimes.has(timestampToUnix(b.timestamp))
    );
    if (selectedBars.length === 0) return [];

    const tickSize = 0.25;
    const volumeByPrice = new Map();

    selectedBars.forEach((b) => {
      const open = parseFloat(b.open);
      const high = parseFloat(b.high);
      const low = parseFloat(b.low);
      const close = parseFloat(b.close);
      const vol = b.volume != null ? Number(b.volume) : 0;
      if (vol === 0) return;

      const rangeTicks = Math.max(1, Math.round((high - low) / tickSize));
      const volPerTick = vol / rangeTicks;

      for (let price = Math.floor(low / tickSize) * tickSize; price <= high + 0.001; price += tickSize) {
        const p = Math.round(price * 100) / 100;
        volumeByPrice.set(p, (volumeByPrice.get(p) || 0) + volPerTick);
      }
    });

    const entries = Array.from(volumeByPrice.entries())
      .map(([price, volume]) => ({ price, volume: Math.round(volume) }))
      .sort((a, b) => b.price - a.price);

    return entries;
  }, [bars1m, selectedBarTimes]);

  if (profile.length === 0) {
    return (
      <div className={`text-center py-8 text-sm ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
        No volume profile data for selected bars.
      </div>
    );
  }

  const maxVol = Math.max(...profile.map((p) => p.volume));
  const pocPrice = profile.reduce((a, b) => (b.volume > a.volume ? b : a)).price;

  // Value Area: 70% of total volume centered on POC
  const totalVol = profile.reduce((sum, p) => sum + p.volume, 0);
  const targetVol = totalVol * 0.7;
  const sortedByVol = [...profile].sort((a, b) => b.volume - a.volume);
  let vaVol = 0;
  const vaPrices = new Set();
  for (const p of sortedByVol) {
    vaPrices.add(p.price);
    vaVol += p.volume;
    if (vaVol >= targetVol) break;
  }

  const barH = Math.max(4, Math.min(16, 320 / profile.length));
  const chartH = barH * profile.length + 8;

  return (
    <div>
      <div className={`text-xs mb-3 flex items-center gap-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm bg-yellow-500" /> POC {pocPrice.toFixed(2)}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm bg-blue-500/40" /> Value Area (70%)
        </span>
        <span className="ml-auto font-mono">
          {profile.length} levels · {totalVol.toLocaleString()} total vol
        </span>
      </div>

      <div className="overflow-y-auto" style={{ maxHeight: '400px' }}>
        <svg width="100%" viewBox={`0 0 500 ${chartH}`} className="block">
          {profile.map((p, i) => {
            const y = i * barH + 4;
            const w = (p.volume / maxVol) * 400;
            const isPOC = p.price === pocPrice;
            const isVA = vaPrices.has(p.price);

            let fill;
            if (isPOC) fill = '#eab308';
            else if (isVA) fill = isDarkMode ? 'rgba(59,130,246,0.5)' : 'rgba(59,130,246,0.35)';
            else fill = isDarkMode ? 'rgba(107,114,128,0.4)' : 'rgba(107,114,128,0.25)';

            return (
              <g key={p.price}>
                <rect x={80} y={y} width={Math.max(2, w)} height={barH - 1} rx={2} fill={fill} />
                <text
                  x={75}
                  y={y + barH / 2 + 1}
                  textAnchor="end"
                  fill={isDarkMode ? '#9ca3af' : '#6b7280'}
                  fontSize={Math.min(11, barH - 1)}
                  fontFamily="monospace"
                  dominantBaseline="middle"
                >
                  {p.price.toFixed(2)}
                </text>
                {p.volume > maxVol * 0.15 && (
                  <text
                    x={83 + Math.max(2, w)}
                    y={y + barH / 2 + 1}
                    fill={isDarkMode ? '#d1d5db' : '#374151'}
                    fontSize={Math.min(10, barH - 1)}
                    fontFamily="monospace"
                    dominantBaseline="middle"
                  >
                    {p.volume.toLocaleString()}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
};


// =============================================================================
// TRADE FLOW CHART (Recharts stacked bar)
// =============================================================================

const TradeFlowChart = ({ bars, isDarkMode }) => {
  const data = useMemo(() => {
    if (!bars || bars.length === 0) return [];
    return bars.map((b) => ({
      time: b.time,
      buy: b.buy_volume ?? (b.volume && b.buy_pct != null ? Math.round(b.volume * b.buy_pct / 100) : 0),
      sell: b.sell_volume ?? (b.volume && b.buy_pct != null ? Math.round(b.volume * (100 - b.buy_pct) / 100) : 0),
    }));
  }, [bars]);

  if (data.length === 0) {
    return (
      <div className={`text-center py-8 text-sm ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
        No trade flow data for selected bars.
      </div>
    );
  }

  const maxVol = Math.max(...data.map((d) => d.buy + d.sell));

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload || !payload.length) return null;
    const buy = payload.find((p) => p.dataKey === 'buy')?.value || 0;
    const sell = payload.find((p) => p.dataKey === 'sell')?.value || 0;
    const total = buy + sell;
    const buyPct = total > 0 ? ((buy / total) * 100).toFixed(1) : '—';
    return (
      <div className={`px-3 py-2 rounded-lg shadow-lg text-xs font-mono ${
        isDarkMode ? 'bg-gray-800 border border-gray-600 text-gray-200' : 'bg-white border border-gray-200 text-gray-800'
      }`}>
        <div className="font-semibold mb-1">{label}</div>
        <div className="text-green-500">Buy: {buy.toLocaleString()}</div>
        <div className="text-red-500">Sell: {sell.toLocaleString()}</div>
        <div className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>
          Total: {total.toLocaleString()} · {buyPct}% buy
        </div>
      </div>
    );
  };

  return (
    <div>
      <div className={`text-xs mb-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
        {data.length} bars · Buy vs Sell volume per minute
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={isDarkMode ? '#374151' : '#e5e7eb'}
            vertical={false}
          />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 10, fill: isDarkMode ? '#9ca3af' : '#6b7280' }}
            tickLine={false}
            axisLine={{ stroke: isDarkMode ? '#4b5563' : '#d1d5db' }}
            interval={data.length > 20 ? Math.floor(data.length / 10) : 0}
          />
          <YAxis
            tick={{ fontSize: 10, fill: isDarkMode ? '#9ca3af' : '#6b7280' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(0)}K` : v}
            domain={[0, Math.ceil(maxVol * 1.05)]}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: isDarkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)' }} />
          <Bar dataKey="buy" stackId="a" fill="#22c55e" radius={[0, 0, 0, 0]} />
          <Bar dataKey="sell" stackId="a" fill="#ef4444" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};


// =============================================================================
// L1 IMBALANCE CHART (Recharts line chart)
// =============================================================================

const ImbalanceChart = ({ bars, isDarkMode }) => {
  const data = useMemo(() => {
    if (!bars || bars.length === 0) return [];
    return bars
      .filter((b) => b.l1_imbalance != null || b.imbalance != null)
      .map((b) => ({
        time: b.time,
        l1: b.l1_imbalance ?? null,
        depth: b.imbalance != null ? (b.imbalance * 2 - 1) : null,
      }));
  }, [bars]);

  if (data.length === 0) {
    return (
      <div className={`text-center py-8 text-sm ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
        No imbalance data for selected bars. L1 imbalance requires micro_1m data (Dec 2025+).
      </div>
    );
  }

  const hasL1 = data.some((d) => d.l1 != null);
  const hasDepth = data.some((d) => d.depth != null);

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload || !payload.length) return null;
    return (
      <div className={`px-3 py-2 rounded-lg shadow-lg text-xs font-mono ${
        isDarkMode ? 'bg-gray-800 border border-gray-600 text-gray-200' : 'bg-white border border-gray-200 text-gray-800'
      }`}>
        <div className="font-semibold mb-1">{label}</div>
        {payload.map((p) => (
          <div key={p.dataKey} style={{ color: p.color }}>
            {p.dataKey === 'l1' ? 'L1 Imbalance' : 'Depth Ratio'}: {p.value?.toFixed(3) ?? '—'}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div>
      <div className={`text-xs mb-2 flex items-center gap-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
        <span>{data.length} bars</span>
        <span>+1 = bid heavy (bullish) · -1 = ask heavy (bearish)</span>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={isDarkMode ? '#374151' : '#e5e7eb'}
            vertical={false}
          />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 10, fill: isDarkMode ? '#9ca3af' : '#6b7280' }}
            tickLine={false}
            axisLine={{ stroke: isDarkMode ? '#4b5563' : '#d1d5db' }}
            interval={data.length > 20 ? Math.floor(data.length / 10) : 0}
          />
          <YAxis
            domain={[-1, 1]}
            tick={{ fontSize: 10, fill: isDarkMode ? '#9ca3af' : '#6b7280' }}
            tickLine={false}
            axisLine={false}
            ticks={[-1, -0.5, 0, 0.5, 1]}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0} stroke={isDarkMode ? '#6b7280' : '#9ca3af'} strokeDasharray="3 3" />
          <ReferenceLine y={0.5} stroke="rgba(34,197,94,0.2)" strokeDasharray="2 4" />
          <ReferenceLine y={-0.5} stroke="rgba(239,68,68,0.2)" strokeDasharray="2 4" />
          {hasL1 && (
            <Line
              type="monotone"
              dataKey="l1"
              stroke="#8b5cf6"
              strokeWidth={2}
              dot={data.length <= 30 ? { r: 2, fill: '#8b5cf6' } : false}
              connectNulls
              name="L1 Imbalance"
            />
          )}
          {hasDepth && !hasL1 && (
            <Line
              type="monotone"
              dataKey="depth"
              stroke="#f59e0b"
              strokeWidth={1.5}
              dot={false}
              connectNulls
              name="Depth Ratio (rescaled)"
              strokeDasharray="4 2"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
      {!hasL1 && hasDepth && (
        <div className={`text-xs mt-1 italic ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          Showing depth_ratio_l1 rescaled to [-1,1] as fallback (micro_1m L1 data unavailable).
        </div>
      )}
    </div>
  );
};


// =============================================================================
// DELTA CHART (generic renderer for daily / weekly / biweekly cumulative delta)
// =============================================================================

const DeltaChart = ({ data, title, subtitle, gradientId, height = 200, isDarkMode, loading, selectedDate, rawBars, smoothWindow = 5 }) => {
  const [dayOnly, setDayOnly] = useState(false);

  const displayData = useMemo(() => {
    if (!dayOnly || !rawBars || rawBars.length === 0 || !selectedDate || !data) return data;
    const filtered = [];
    for (let i = 0; i < rawBars.length; i++) {
      if (rawBars[i].sortKey && rawBars[i].sortKey.startsWith(selectedDate) && data[i]) {
        filtered.push({
          ...data[i],
          time: data[i].time.includes(' ') ? data[i].time.split(' ').slice(1).join(' ') : data[i].time,
        });
      }
    }
    return filtered.length > 0 ? filtered : data;
  }, [data, dayOnly, rawBars, selectedDate]);

  const displaySubtitle = dayOnly && rawBars ? selectedDate : subtitle;

  const accelData = useMemo(() => {
    if (!displayData || displayData.length === 0) return [];
    const sw = Math.min(smoothWindow, Math.max(2, displayData.length));
    const result = [];
    for (let i = 0; i < displayData.length; i++) {
      let smoothVel = null;
      if (i >= sw - 1) {
        let sum = 0;
        for (let j = i - sw + 1; j <= i; j++) sum += displayData[j].netDelta;
        smoothVel = sum / sw;
      }
      result.push({ ...displayData[i], smoothVel, accel: null });
    }
    for (let i = 1; i < result.length; i++) {
      if (result[i].smoothVel != null && result[i - 1].smoothVel != null) {
        result[i].accel = result[i].smoothVel - result[i - 1].smoothVel;
      }
    }
    return result;
  }, [displayData, smoothWindow]);

  if (loading) {
    return (
      <div className={`flex items-center gap-2 px-4 py-3 border-t ${isDarkMode ? 'border-gray-700 text-gray-500' : 'border-gray-200 text-gray-400'}`}>
        <div className={`animate-spin rounded-full h-4 w-4 border-2 border-t-transparent ${isDarkMode ? 'border-cyan-400' : 'border-cyan-600'}`} />
        <span className="text-xs">Loading {title?.toLowerCase() || 'delta data'}…</span>
      </div>
    );
  }

  if (!displayData || displayData.length === 0) return null;

  const minDelta = Math.min(0, ...displayData.map((d) => d.cumDelta));
  const maxDelta = Math.max(0, ...displayData.map((d) => d.cumDelta));
  const range = maxDelta - minDelta || 1;
  const zeroOffset = Math.max(0.01, Math.min(0.99, maxDelta / range));
  const finalDelta = displayData[displayData.length - 1]?.cumDelta ?? 0;

  const peakDelta = displayData.reduce((best, d) =>
    Math.abs(d.cumDelta) > Math.abs(best.cumDelta) ? d : best
  , displayData[0]);

  const latestAccel = accelData.length > 0 ? accelData[accelData.length - 1]?.accel : null;

  const isMultiDay = displayData.some((d) => d.time.includes(' '));
  const dayBoundaries = [];
  const dayTicks = [];
  if (isMultiDay) {
    let lastDay = null;
    for (const d of displayData) {
      const day = d.time.slice(0, 3);
      if (day !== lastDay) {
        if (lastDay !== null) dayBoundaries.push(d.time);
        dayTicks.push(d.time);
        lastDay = day;
      }
    }
  }

  const fillId = `${gradientId}Fill`;
  const strokeId = `${gradientId}Stroke`;

  const DeltaTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    const cum = payload[0]?.value;
    const entry = payload[0]?.payload;
    const net = entry?.netDelta;
    const accel = entry?.accel;
    return (
      <div className={`px-3 py-2 rounded-lg shadow-lg text-xs font-mono ${
        isDarkMode ? 'bg-gray-800 border border-gray-600 text-gray-200' : 'bg-white border border-gray-200 text-gray-800'
      }`}>
        <div className="font-semibold mb-1">{label}</div>
        <div className={cum >= 0 ? 'text-green-500' : 'text-red-500'}>
          Cum. Delta: {cum?.toLocaleString()}
        </div>
        {net != null && (
          <div className={net >= 0 ? 'text-green-400' : 'text-red-400'}>
            Bar Delta: {net > 0 ? '+' : ''}{net.toLocaleString()}
          </div>
        )}
        {accel != null && (
          <div className={accel >= 0 ? 'text-cyan-400' : 'text-orange-400'}>
            Accel: {accel > 0 ? '+' : ''}{accel.toFixed(0)}
          </div>
        )}
      </div>
    );
  };

  const xAxisProps = isMultiDay
    ? { ticks: dayTicks, tickFormatter: (val) => val.slice(0, 3), interval: 0 }
    : { interval: displayData.length > 60 ? Math.floor(displayData.length / 8) : displayData.length > 20 ? Math.floor(displayData.length / 6) : 0 };

  return (
    <div className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
      <div className="flex items-center justify-between px-4 pt-2 pb-1">
        <div className={`flex items-center gap-3 text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          <span className={`font-semibold uppercase tracking-wider ${isDarkMode ? 'text-cyan-400' : 'text-cyan-600'}`}>
            {title || 'Cumulative Delta'}
          </span>
          {displaySubtitle && (
            <span className={isDarkMode ? 'text-gray-600' : 'text-gray-400'}>{displaySubtitle}</span>
          )}
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-sm bg-green-500" />
            Buying
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-sm bg-red-500" />
            Selling
          </span>
        </div>
        <div className={`flex items-center gap-4 text-xs font-mono ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          {rawBars && (
            <button
              onClick={() => setDayOnly((prev) => !prev)}
              className={`px-2.5 py-1 rounded font-medium font-sans transition-colors ${
                dayOnly
                  ? isDarkMode ? 'bg-cyan-600 text-white' : 'bg-cyan-500 text-white'
                  : isDarkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {dayOnly ? 'Day Only' : 'Full Range'}
            </button>
          )}
          <span>
            Peak:{' '}
            <span className={peakDelta.cumDelta >= 0 ? 'text-green-500' : 'text-red-500'}>
              {peakDelta.cumDelta > 0 ? '+' : ''}{peakDelta.cumDelta.toLocaleString()}
            </span>
            <span className={isDarkMode ? 'text-gray-600' : 'text-gray-400'}> @ {peakDelta.time}</span>
          </span>
          <span>
            Final:{' '}
            <span className={`font-bold ${finalDelta >= 0 ? 'text-green-500' : 'text-red-500'}`}>
              {finalDelta > 0 ? '+' : ''}{finalDelta.toLocaleString()}
            </span>
          </span>
          {latestAccel != null && (
            <span>
              Accel:{' '}
              <span className={latestAccel >= 0 ? 'text-cyan-400' : 'text-orange-400'}>
                {latestAccel >= 0 ? '▲' : '▼'}{latestAccel > 0 ? '+' : ''}{latestAccel.toFixed(0)}
              </span>
            </span>
          )}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={accelData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id={fillId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#22c55e" stopOpacity={0.5} />
              <stop offset={`${zeroOffset * 100}%`} stopColor="#22c55e" stopOpacity={0.05} />
              <stop offset={`${zeroOffset * 100}%`} stopColor="#ef4444" stopOpacity={0.05} />
              <stop offset="100%" stopColor="#ef4444" stopOpacity={0.5} />
            </linearGradient>
            <linearGradient id={strokeId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#22c55e" />
              <stop offset={`${zeroOffset * 100}%`} stopColor={isDarkMode ? '#06b6d4' : '#0891b2'} />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? '#374151' : '#e5e7eb'} vertical={false} />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 10, fill: isDarkMode ? '#9ca3af' : '#6b7280' }}
            tickLine={false}
            axisLine={{ stroke: isDarkMode ? '#4b5563' : '#d1d5db' }}
            {...xAxisProps}
          />
          <YAxis
            tick={{ fontSize: 10, fill: isDarkMode ? '#9ca3af' : '#6b7280' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => {
              if (Math.abs(v) >= 1000000) return `${(v / 1000000).toFixed(1)}M`;
              if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}K`;
              return v;
            }}
          />
          <Tooltip content={<DeltaTooltip />} cursor={{ stroke: isDarkMode ? '#6b7280' : '#9ca3af', strokeDasharray: '3 3' }} />
          <ReferenceLine y={0} stroke={isDarkMode ? '#6b7280' : '#9ca3af'} strokeDasharray="4 3" strokeWidth={1.5} />
          {dayBoundaries.map((t) => (
            <ReferenceLine
              key={t}
              x={t}
              stroke={isDarkMode ? '#4b5563' : '#d1d5db'}
              strokeDasharray="4 4"
              strokeWidth={1}
            />
          ))}
          <Area
            type="monotone"
            dataKey="cumDelta"
            stroke={`url(#${strokeId})`}
            strokeWidth={2}
            fill={`url(#${fillId})`}
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
      <ResponsiveContainer width="100%" height={40}>
        <BarChart data={accelData} margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
          <YAxis hide />
          <XAxis dataKey="time" hide />
          <ReferenceLine y={0} stroke={isDarkMode ? '#4b5563' : '#d1d5db'} strokeWidth={0.5} />
          <Bar dataKey="accel" isAnimationActive={false}>
            {accelData.map((d, i) => (
              <Cell
                key={i}
                fill={d.accel == null ? 'transparent' : d.accel >= 0 ? 'rgba(6,182,212,0.6)' : 'rgba(251,146,60,0.6)'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};


// =============================================================================
// CHART TAB BUTTONS
// =============================================================================

const MICRO_TABS = [
  { key: 'volume_profile', label: 'Volume Profile', icon: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="3" width="4" height="18" rx="1" />
      <rect x="10" y="8" width="4" height="13" rx="1" />
      <rect x="17" y="12" width="4" height="9" rx="1" />
    </svg>
  )},
  { key: 'trade_flow', label: 'Trade Flow', icon: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M3 21V8l4 4 4-8 4 6 6-7" />
    </svg>
  )},
  { key: 'l1_imbalance', label: 'L1 Imbalance', icon: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="3" y1="12" x2="21" y2="12" strokeDasharray="2 2" />
      <polyline points="3,16 7,8 11,14 15,6 19,10 21,12" />
    </svg>
  )},
];


// =============================================================================
// MICROSTRUCTURE PANEL (summary + chart tabs)
// =============================================================================

const StatCard = ({ label, value, color, isDarkMode }) => (
  <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
    <div className={`text-xs uppercase tracking-wide mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
      {label}
    </div>
    <div className={`text-base font-bold font-mono ${color || (isDarkMode ? 'text-white' : 'text-gray-900')}`}>
      {value ?? '—'}
    </div>
  </div>
);

const MicrostructurePanel = ({ data, loading, error, onClear, isDarkMode, bars1m, selectedBarTimes }) => {
  const [activeTab, setActiveTab] = useState(null);

  if (loading) {
    return (
      <div className={`mt-4 rounded-lg border p-6 ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
        <div className="flex items-center gap-3">
          <div className={`animate-spin rounded-full h-5 w-5 border-2 border-t-transparent ${isDarkMode ? 'border-cyan-400' : 'border-cyan-600'}`} />
          <span className={`text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
            Fetching microstructure data…
          </span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`mt-4 rounded-lg border p-4 ${isDarkMode ? 'bg-red-900/20 border-red-800' : 'bg-red-50 border-red-200'}`}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className={`font-medium ${isDarkMode ? 'text-red-300' : 'text-red-700'}`}>
              Microstructure data unavailable
            </div>
            <div className={`text-sm mt-1 ${isDarkMode ? 'text-red-400' : 'text-red-600'}`}>{error}</div>
          </div>
          <button
            onClick={onClear}
            className={`shrink-0 text-xs px-3 py-1.5 rounded transition-colors ${
              isDarkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Clear
          </button>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const { time_range, trades, book, toxicity, bars } = data;

  return (
    <div className={`mt-4 rounded-lg border ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
      {/* Header */}
      <div className={`flex items-center justify-between px-4 py-3 border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        <div>
          <span className={`font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
            Microstructure Detail
          </span>
          {time_range && (
            <span className={`ml-3 text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              {time_range.start} → {time_range.end}
              {time_range.bars != null && ` · ${time_range.bars} bar${time_range.bars !== 1 ? 's' : ''}`}
            </span>
          )}
        </div>
        <button
          onClick={onClear}
          className={`text-xs px-3 py-1.5 rounded transition-colors ${
            isDarkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          Clear Selection
        </button>
      </div>

      <div className="p-4 space-y-5">
        {/* Summary cards */}
        {trades && (
          <div>
            <div className={`text-xs font-semibold uppercase tracking-wider mb-2 ${isDarkMode ? 'text-cyan-400' : 'text-cyan-600'}`}>
              Trade Flow
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <StatCard label="Total Trades" value={trades.count?.toLocaleString()} isDarkMode={isDarkMode} />
              <StatCard label="Total Volume" value={trades.volume?.toLocaleString()} isDarkMode={isDarkMode} />
              <StatCard label="VWAP" value={trades.vwap?.toFixed(2)} isDarkMode={isDarkMode} />
              <StatCard label="Avg Trade Size" value={trades.avg_size?.toFixed(1)} isDarkMode={isDarkMode} />
            </div>
            {(trades.buy_volume != null || trades.sell_volume != null) && (() => {
              const total = (trades.buy_volume || 0) + (trades.sell_volume || 0);
              const buyPct = total > 0 ? ((trades.buy_volume || 0) / total) * 100 : 50;
              return (
                <div className="mt-3">
                  <div className={`text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    Buy / Sell Split
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex-1 h-4 rounded overflow-hidden flex bg-gray-200 dark:bg-gray-600">
                      <div className="bg-green-500 h-full transition-all" style={{ width: `${buyPct}%` }} />
                      <div className="bg-red-500 h-full transition-all" style={{ width: `${100 - buyPct}%` }} />
                    </div>
                    <span className={`text-xs font-mono w-16 text-right ${buyPct > 55 ? 'text-green-500' : buyPct < 45 ? 'text-red-500' : isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                      {buyPct.toFixed(1)}% buy
                    </span>
                  </div>
                </div>
              );
            })()}
          </div>
        )}

        {toxicity && (
          <div>
            <div className={`text-xs font-semibold uppercase tracking-wider mb-2 ${isDarkMode ? 'text-orange-400' : 'text-orange-600'}`}>
              Toxicity Signals
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <StatCard
                label="VPIN"
                value={toxicity.avg_vpin?.toFixed(3)}
                color={toxicity.avg_vpin > 0.7 ? 'text-red-500' : toxicity.avg_vpin > 0.5 ? 'text-yellow-500' : 'text-green-500'}
                isDarkMode={isDarkMode}
              />
              <StatCard label="Toxicity Score" value={toxicity.avg_toxicity_score?.toFixed(3)} isDarkMode={isDarkMode} />
              <StatCard label="Kyle's Lambda" value={toxicity.avg_kyle_lambda?.toFixed(4)} isDarkMode={isDarkMode} />
              <StatCard label="Amihud Ratio" value={toxicity.avg_amihud?.toExponential(2)} isDarkMode={isDarkMode} />
            </div>
          </div>
        )}

        {/* ─── Chart Tabs ─────────────────────────────────────────────── */}
        <div>
          <div className={`text-xs font-semibold uppercase tracking-wider mb-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
            Visualizations
          </div>
          <div className="flex gap-2 flex-wrap">
            {MICRO_TABS.map((tab) => {
              const isActive = activeTab === tab.key;
              return (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(isActive ? null : tab.key)}
                  className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? isDarkMode
                        ? 'bg-cyan-600 text-white shadow-md ring-1 ring-cyan-500'
                        : 'bg-cyan-500 text-white shadow-md ring-1 ring-cyan-400'
                      : isDarkMode
                      ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* ─── Active Chart Panel ─────────────────────────────────────── */}
        {activeTab && (
          <div className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-900/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
            {activeTab === 'volume_profile' && (
              <VolumeProfileChart bars1m={bars1m} selectedBarTimes={selectedBarTimes} isDarkMode={isDarkMode} />
            )}
            {activeTab === 'trade_flow' && (
              <TradeFlowChart bars={bars} isDarkMode={isDarkMode} />
            )}
            {activeTab === 'l1_imbalance' && (
              <ImbalanceChart bars={bars} isDarkMode={isDarkMode} />
            )}
          </div>
        )}

        {/* Per-bar table */}
        {bars && bars.length > 0 && !activeTab && (
          <div>
            <div className={`text-xs font-semibold uppercase tracking-wider mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              Per-Bar Detail ({bars.length} bar{bars.length !== 1 ? 's' : ''})
            </div>
            <div className={`overflow-x-auto rounded-lg border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <table className="min-w-full text-xs">
                <thead className={isDarkMode ? 'bg-gray-700' : 'bg-gray-100'}>
                  <tr>
                    {['Time', 'Volume', 'Buy %', 'Spread', 'Imbalance', 'VPIN', 'Toxicity'].map((h) => (
                      <th
                        key={h}
                        className={`px-3 py-2 text-left font-medium uppercase tracking-wider ${
                          isDarkMode ? 'text-gray-400' : 'text-gray-500'
                        }`}
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className={`divide-y ${isDarkMode ? 'divide-gray-700' : 'divide-gray-200'}`}>
                  {bars.map((bar, i) => {
                    const buyPct = bar.buy_pct ?? null;
                    const imb = bar.l1_imbalance ?? bar.imbalance ?? null;
                    const vpin = bar.vpin ?? null;
                    const tox = bar.toxicity ?? null;
                    return (
                      <tr key={i} className={isDarkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-50'}>
                        <td className={`px-3 py-2 font-mono ${isDarkMode ? 'text-gray-300' : 'text-gray-900'}`}>
                          {bar.time}
                        </td>
                        <td className={`px-3 py-2 font-mono ${isDarkMode ? 'text-gray-300' : 'text-gray-900'}`}>
                          {bar.volume?.toLocaleString() ?? '—'}
                        </td>
                        <td className={`px-3 py-2 font-mono ${
                          buyPct > 55 ? 'text-green-500' : buyPct < 45 ? 'text-red-500' : isDarkMode ? 'text-gray-300' : 'text-gray-700'
                        }`}>
                          {buyPct != null ? `${buyPct.toFixed(1)}%` : '—'}
                        </td>
                        <td className={`px-3 py-2 font-mono ${isDarkMode ? 'text-gray-300' : 'text-gray-900'}`}>
                          {bar.spread?.toFixed(4) ?? '—'}
                        </td>
                        <td className={`px-3 py-2 font-mono ${
                          imb > 0.2 ? 'text-green-500' : imb < -0.2 ? 'text-red-500' : isDarkMode ? 'text-gray-300' : 'text-gray-700'
                        }`}>
                          {imb != null ? imb.toFixed(3) : '—'}
                        </td>
                        <td className={`px-3 py-2 font-mono ${
                          vpin > 0.7 ? 'text-red-500' : vpin > 0.5 ? 'text-yellow-500' : isDarkMode ? 'text-gray-300' : 'text-gray-700'
                        }`}>
                          {vpin != null ? vpin.toFixed(3) : '—'}
                        </td>
                        <td className={`px-3 py-2 font-mono ${
                          tox > 0.7 ? 'text-red-500' : tox > 0.5 ? 'text-yellow-500' : isDarkMode ? 'text-gray-300' : 'text-gray-700'
                        }`}>
                          {tox != null ? tox.toFixed(3) : '—'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};


// =============================================================================
// MAIN CHART COMPONENT
// =============================================================================

export default function InstrumentChart({ selectedDate, symbol, isDarkMode }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);
  const hasFitRef = useRef(false);

  const [bars1m, setBars1m] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [timeframe, setTimeframe] = useState('1m');

  const [isSelectionMode, setIsSelectionMode] = useState(false);
  const [selectedBarTimes, setSelectedBarTimes] = useState(new Set());

  const [microData, setMicroData] = useState(null);
  const [loadingMicro, setLoadingMicro] = useState(false);
  const [microError, setMicroError] = useState(null);

  const [dayMicroBars, setDayMicroBars] = useState([]);
  const [loadingDayMicro, setLoadingDayMicro] = useState(false);

  const [weekMicroBars, setWeekMicroBars] = useState([]);
  const [loadingWeek, setLoadingWeek] = useState(false);
  const [weekDateRange, setWeekDateRange] = useState(null);

  const [biweekMicroBars, setBiweekMicroBars] = useState([]);
  const [loadingBiweek, setLoadingBiweek] = useState(false);
  const [biweekDateRange, setBiweekDateRange] = useState(null);

  const colors = useMemo(() => (isDarkMode ? CHART_COLORS.dark : CHART_COLORS.light), [isDarkMode]);

  const canSelect = timeframe === '1m' && bars1m.length > 0;

  useEffect(() => {
    setIsSelectionMode(false);
    setSelectedBarTimes(new Set());
    setMicroData(null);
    setMicroError(null);
  }, [selectedDate, symbol, timeframe]);

  useEffect(() => {
    if (!selectedDate || !symbol) {
      setBars1m([]);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    fetch(`${API_BASE_URL}/intraday/${selectedDate}/${encodeURIComponent(symbol)}?limit=2000`)
      .then((res) => {
        if (!res.ok) throw new Error(res.statusText || 'Failed to load 1m data');
        return res.json();
      })
      .then((data) => setBars1m(data.bars || []))
      .catch((err) => {
        setError(err.message);
        setBars1m([]);
      })
      .finally(() => setLoading(false));
  }, [selectedDate, symbol]);

  // Fetch full-day microstructure for cumulative delta (Dec 2025+ only)
  useEffect(() => {
    if (!selectedDate || !symbol) {
      setDayMicroBars([]);
      return;
    }
    setLoadingDayMicro(true);
    fetch(`${API_BASE_URL}/micro-detail/${selectedDate}/${encodeURIComponent(symbol)}`)
      .then((res) => {
        if (!res.ok) throw new Error('No micro data');
        return res.json();
      })
      .then((data) => setDayMicroBars(data.bars || []))
      .catch(() => setDayMicroBars([]))
      .finally(() => setLoadingDayMicro(false));
  }, [selectedDate, symbol]);

  // Fetch multi-day micro data for weekly + biweekly cumulative delta
  useEffect(() => {
    if (!selectedDate || !symbol) {
      setWeekMicroBars([]);
      setBiweekMicroBars([]);
      setWeekDateRange(null);
      setBiweekDateRange(null);
      return;
    }
    let stale = false;

    const weekMonday = getWeekMonday(selectedDate);
    const biweekMonday = getBiweeklyMonday(selectedDate);
    const weekDates = generateWeekdayDates(weekMonday, selectedDate);
    const biweekDates = generateWeekdayDates(biweekMonday, selectedDate);

    setLoadingWeek(true);
    fetchMultiDayMicroBars(weekDates, symbol, 5)
      .then((bars) => {
        if (stale) return;
        setWeekMicroBars(bars);
        setWeekDateRange({ start: weekMonday, end: selectedDate, days: weekDates.length });
      })
      .catch(() => { if (!stale) setWeekMicroBars([]); })
      .finally(() => { if (!stale) setLoadingWeek(false); });

    setLoadingBiweek(true);
    fetchMultiDayMicroBars(biweekDates, symbol, 15)
      .then((bars) => {
        if (stale) return;
        setBiweekMicroBars(bars);
        setBiweekDateRange({ start: biweekMonday, end: selectedDate, days: biweekDates.length });
      })
      .catch(() => { if (!stale) setBiweekMicroBars([]); })
      .finally(() => { if (!stale) setLoadingBiweek(false); });

    return () => { stale = true; };
  }, [selectedDate, symbol]);

  const chartBars = useMemo(() => {
    const mins = TIMEFRAMES.find((tf) => tf.key === timeframe)?.minutes ?? 1;
    return aggregateBars(bars1m, mins);
  }, [bars1m, timeframe]);

  const chartDataWithSelection = useMemo(
    () => barsToChartData(chartBars, selectedBarTimes),
    [chartBars, selectedBarTimes]
  );

  // Cumulative delta data for each time horizon
  const dailyDeltaData = useMemo(() => {
    if (!dayMicroBars || dayMicroBars.length === 0) return [];
    const mins = TIMEFRAMES.find((tf) => tf.key === timeframe)?.minutes ?? 1;
    const sorted = [...dayMicroBars].sort((a, b) => timeToMinutes(a.time) - timeToMinutes(b.time));
    const bars = mins <= 1 ? sorted : aggregateMicroBars(sorted, mins);
    return computeCumDelta(bars);
  }, [dayMicroBars, timeframe]);

  const weeklyDeltaData = useMemo(() => computeCumDelta(weekMicroBars), [weekMicroBars]);
  const biweeklyDeltaData = useMemo(() => computeCumDelta(biweekMicroBars), [biweekMicroBars]);

  const weekSubtitle = weekDateRange
    ? `${weekDateRange.start.slice(5)} → ${weekDateRange.end.slice(5)} · ${weekDateRange.days}d · 5m bars`
    : '';
  const biweekSubtitle = biweekDateRange
    ? `${biweekDateRange.start.slice(5)} → ${biweekDateRange.end.slice(5)} · ${biweekDateRange.days}d · 15m bars`
    : '';

  useEffect(() => {
    if (!containerRef.current) return;
    if (chartRef.current) {
      try { chartRef.current.remove(); } catch (_) {}
      chartRef.current = null;
      seriesRef.current = null;
    }
    hasFitRef.current = false;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 360,
      layout: {
        background: { type: ColorType.Solid, color: colors.background },
        textColor: colors.text,
      },
      grid: {
        vertLines: { color: colors.grid },
        horzLines: { color: colors.grid },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: colors.border },
        horzLine: { color: colors.border },
      },
      rightPriceScale: { borderColor: colors.border },
      timeScale: { borderColor: colors.border, timeVisible: true, secondsVisible: true },
    });

    const series = chart.addSeries(CandlestickSeries, {
      upColor: colors.upColor,
      downColor: colors.downColor,
      borderVisible: false,
      wickUpColor: colors.upColor,
      wickDownColor: colors.downColor,
    });

    chartRef.current = chart;
    seriesRef.current = series;

    const onResize = () => {
      if (chartRef.current && containerRef.current)
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
    };
    window.addEventListener('resize', onResize);

    return () => {
      window.removeEventListener('resize', onResize);
      if (chartRef.current) {
        try { chartRef.current.remove(); } catch (_) {}
      }
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, [symbol, colors]);

  useEffect(() => {
    hasFitRef.current = false;
  }, [chartBars]);

  useEffect(() => {
    if (!seriesRef.current || !chartRef.current || chartDataWithSelection.length === 0) return;
    try {
      seriesRef.current.setData(chartDataWithSelection);
      if (!hasFitRef.current) {
        chartRef.current.timeScale().fitContent();
        hasFitRef.current = true;
      }
    } catch (e) {
      console.warn('InstrumentChart setData:', e);
    }
  }, [chartDataWithSelection]);

  const handleChartClick = useCallback(
    (e) => {
      if (!isSelectionMode || !chartRef.current || chartBars.length === 0) return;
      if (!containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const logical = chartRef.current.timeScale().coordinateToLogical(x);
      if (logical === null) return;

      const index = Math.round(logical);
      if (index < 0 || index >= chartBars.length) return;

      const bar = chartBars[index];
      if (!bar) return;

      const unixTime = timestampToUnix(bar.timestamp);
      setSelectedBarTimes((prev) => {
        const next = new Set(prev);
        if (next.has(unixTime)) next.delete(unixTime);
        else next.add(unixTime);
        return next;
      });
    },
    [isSelectionMode, chartBars]
  );

  useEffect(() => {
    if (!isSelectionMode) return;
    const onKey = (e) => {
      if (e.key === 'Escape') {
        setIsSelectionMode(false);
        setSelectedBarTimes(new Set());
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isSelectionMode]);

  const handleConfirmSelection = useCallback(async () => {
    if (selectedBarTimes.size === 0) return;

    const sortedTimes = Array.from(selectedBarTimes).sort((a, b) => a - b);
    const startUnix = sortedTimes[0];
    const endUnix = sortedTimes[sortedTimes.length - 1];

    const startISO = new Date(startUnix * 1000).toISOString();
    const endISO = new Date((endUnix + 60) * 1000).toISOString();

    setLoadingMicro(true);
    setMicroError(null);
    setMicroData(null);

    try {
      const res = await fetch(
        `${API_BASE_URL}/micro-detail/${selectedDate}/${encodeURIComponent(symbol)}` +
          `?start_time=${encodeURIComponent(startISO)}&end_time=${encodeURIComponent(endISO)}`
      );
      if (!res.ok) throw new Error(`HTTP ${res.status} — endpoint may not be implemented yet`);
      const data = await res.json();
      setMicroData(data);
      setIsSelectionMode(false);
    } catch (err) {
      setMicroError(err.message);
    } finally {
      setLoadingMicro(false);
    }
  }, [selectedBarTimes, selectedDate, symbol]);

  const handleClearMicro = useCallback(() => {
    setMicroData(null);
    setMicroError(null);
    setSelectedBarTimes(new Set());
  }, []);

  // ─── Early returns ──────────────────────────────────────────────────────────
  if (!selectedDate || !symbol) return null;

  if (loading && bars1m.length === 0) {
    return (
      <div className={`rounded-lg border p-6 ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
        <div className="flex items-center gap-3">
          <div className={`animate-spin rounded-full h-6 w-6 border-2 border-t-transparent ${isDarkMode ? 'border-cyan-400' : 'border-cyan-600'}`} />
          <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
            Loading 1m data for {symbol}…
          </span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`rounded-lg border p-4 ${isDarkMode ? 'bg-red-900/20 border-red-800' : 'bg-red-50 border-red-200'}`}>
        <div className={isDarkMode ? 'text-red-300' : 'text-red-700'}>Chart unavailable: {error}</div>
      </div>
    );
  }

  if (bars1m.length === 0) {
    return (
      <div className={`rounded-lg border p-6 ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
        <div className={`text-center ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          No 1m data for {symbol} on {selectedDate}.
        </div>
      </div>
    );
  }

  // ─── Main render ────────────────────────────────────────────────────────────
  return (
    <div>
      <div className={`rounded-lg border ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
        {/* Header */}
        <div className={`flex items-center justify-between px-4 py-3 border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
          <span className={`font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
            {symbol} • {selectedDate}
          </span>
          <div className="flex items-center gap-2">
            <div className="flex gap-1">
              {TIMEFRAMES.map((tf) => (
                <button
                  key={tf.key}
                  onClick={() => setTimeframe(tf.key)}
                  className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                    timeframe === tf.key
                      ? isDarkMode ? 'bg-cyan-600 text-white' : 'bg-cyan-500 text-white'
                      : isDarkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {tf.label}
                </button>
              ))}
            </div>

            {canSelect && (
              <button
                onClick={() => {
                  if (isSelectionMode) {
                    setIsSelectionMode(false);
                    setSelectedBarTimes(new Set());
                  } else {
                    setIsSelectionMode(true);
                    setMicroData(null);
                    setMicroError(null);
                  }
                }}
                title={isSelectionMode ? 'Exit selection mode (Esc)' : 'Select candles for microstructure analysis'}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium transition-all ${
                  isSelectionMode
                    ? 'bg-blue-500 text-white shadow-md ring-2 ring-blue-400'
                    : isDarkMode
                    ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <line x1="12" y1="2" x2="12" y2="22" />
                  <line x1="2" y1="12" x2="22" y2="12" />
                  <circle cx="12" cy="12" r="3" fill="none" />
                </svg>
                {isSelectionMode ? (
                  <>
                    Select
                    {selectedBarTimes.size > 0 && (
                      <span className="ml-1 px-1.5 py-0.5 rounded-full bg-white/20 text-xs">
                        {selectedBarTimes.size}
                      </span>
                    )}
                  </>
                ) : (
                  'Select'
                )}
              </button>
            )}
          </div>
        </div>

        {isSelectionMode && (
          <div className={`px-4 py-2 text-sm flex items-center justify-between border-b ${
            isDarkMode ? 'bg-blue-900/30 border-blue-800 text-blue-300' : 'bg-blue-50 border-blue-200 text-blue-700'
          }`}>
            <span>
              <span className="mr-2">🎯</span>
              <strong>Selection Mode</strong> — click candles to select.{' '}
              <kbd className={`px-1.5 py-0.5 rounded text-xs font-mono ${isDarkMode ? 'bg-blue-800 text-blue-200' : 'bg-blue-200 text-blue-800'}`}>
                Esc
              </kbd>{' '}
              to cancel.
            </span>
            {selectedBarTimes.size > 0 && (
              <span className={`text-xs px-2 py-1 rounded-full font-medium ${isDarkMode ? 'bg-blue-800 text-blue-200' : 'bg-blue-200 text-blue-800'}`}>
                {selectedBarTimes.size} bar{selectedBarTimes.size !== 1 ? 's' : ''} selected
              </span>
            )}
          </div>
        )}

        <div
          ref={containerRef}
          onClick={handleChartClick}
          className="w-full"
          style={{ height: '360px', cursor: isSelectionMode ? 'crosshair' : 'default' }}
        />

        <div className={`px-4 py-2 text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          {chartBars.length} candles ({timeframe}) from {bars1m.length} 1m bars
        </div>

        <DeltaChart
          data={dailyDeltaData}
          title="Daily Cum. Delta"
          gradientId="dailyDelta"
          height={200}
          isDarkMode={isDarkMode}
          loading={loadingDayMicro}
        />

        <DeltaChart
          data={weeklyDeltaData}
          title="Weekly Cum. Delta"
          subtitle={weekSubtitle}
          gradientId="weeklyDelta"
          height={200}
          isDarkMode={isDarkMode}
          loading={loadingWeek}
          selectedDate={selectedDate}
          rawBars={weekMicroBars}
        />

        <DeltaChart
          data={biweeklyDeltaData}
          title="Biweekly Cum. Delta"
          subtitle={biweekSubtitle}
          gradientId="biweeklyDelta"
          height={200}
          isDarkMode={isDarkMode}
          loading={loadingBiweek}
          selectedDate={selectedDate}
          rawBars={biweekMicroBars}
        />
      </div>

      {isSelectionMode && selectedBarTimes.size > 0 && (
        <div className="mt-3 flex justify-center">
          <button
            onClick={handleConfirmSelection}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold text-sm transition-all shadow-lg ${
              isDarkMode
                ? 'bg-cyan-600 hover:bg-cyan-500 text-white ring-1 ring-cyan-500'
                : 'bg-cyan-500 hover:bg-cyan-400 text-white ring-1 ring-cyan-400'
            }`}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            Confirm Selection — {selectedBarTimes.size} bar{selectedBarTimes.size !== 1 ? 's' : ''}
          </button>
        </div>
      )}

      <MicrostructurePanel
        data={microData}
        loading={loadingMicro}
        error={microError}
        onClear={handleClearMicro}
        isDarkMode={isDarkMode}
        bars1m={bars1m}
        selectedBarTimes={selectedBarTimes}
      />
    </div>
  );
}
