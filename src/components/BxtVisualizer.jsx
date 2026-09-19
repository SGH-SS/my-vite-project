/**
 * BXT Visualizer — mid-price OHLC candles built from the stored 5s bxt buckets.
 *
 * v1 scope: get the candle spine right. Candles come from bucket mid_o/h/l/c
 * (NOT mark — mark is smoothed and sub-minute mark OHLC is pseudo-data), and
 * stacking runs through the backend's one true stacker
 * (feature_builder.signal_buckets) via /api/bxtviz/candles.
 *
 * Chart stack: lightweight-charts v5 with panes — chosen deliberately as the
 * long-term base: future indicators mount as extra series in extra panes
 * above/below the candles, and v5 pane separators are natively draggable /
 * resizable. The volume pane below is the first proof of that architecture.
 *
 * Honesty rules carried from research/:
 *  - candles are labelled by bucket START (half-open [t, t+tf))
 *  - missing buckets render as real gaps (whitespace), never bridged
 *  - degraded-feed candles are greyed, revised candles amber — never painted
 *    the same as pristine ones
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  createChart, CandlestickSeries, HistogramSeries, ColorType, CrosshairMode,
} from 'lightweight-charts';

const API = 'http://localhost:8000/api/bxtviz';
const MAX_CANDLES = 20000;

const SCHEMAS = [
  { key: 'btc', label: 'BTC' },
  { key: 'eth', label: 'ETH' },
  { key: 'sol', label: 'SOL' },
  { key: 'spx', label: 'SPX' },
];

const DURATION_PRESETS = [
  { key: '1m', label: 'Last 1m', s: 60 },
  { key: '5m', label: 'Last 5m', s: 300 },
  { key: '15m', label: 'Last 15m', s: 900 },
  { key: '1h', label: 'Last 1 hour', s: 3600 },
  { key: '4h', label: 'Last 4 hours', s: 4 * 3600 },
  { key: '12h', label: 'Last 12 hours', s: 12 * 3600 },
  { key: '1d', label: 'Last day', s: 24 * 3600 },
  { key: '2d', label: 'Last 2 days', s: 2 * 24 * 3600 },
  { key: '1w', label: 'Last week', s: 7 * 24 * 3600 },
  { key: '2w', label: 'Last 2 weeks', s: 14 * 24 * 3600 },
  { key: 'custom', label: 'Custom…', s: null },
];

const CANDLE_SIZES = [
  { s: 5, label: '5s' },
  { s: 15, label: '15s' },
  { s: 30, label: '30s' },
  { s: 60, label: '1m' },
  { s: 300, label: '5m' },
  { s: 900, label: '15m' },
  { s: 3600, label: '1h' },
  { s: 14400, label: '4h' },
  { s: 86400, label: '1d' },
];
const sizeLabel = (s) => CANDLE_SIZES.find((c) => c.s === s)?.label ?? `${s}s`;

const COLOR_SELECTED = '#3b82f6';   // selected candles (select mode)

const CUSTOM_UNITS = [
  { key: 'm', label: 'minutes', s: 60 },
  { key: 'h', label: 'hours', s: 3600 },
  { key: 'd', label: 'days', s: 24 * 3600 },
];

// candle palette — quality is encoded per-bar, never painted over
const COLOR_UP = '#26a69a';
const COLOR_DOWN = '#ef5350';
const COLOR_DEGRADED = '#6b7280';
const COLOR_REVISED = '#f59e0b';

const fmtUtc = (epochS) => {
  const d = new Date(epochS * 1000);
  return d.toISOString().replace('T', ' ').slice(0, 19);
};

// how the bid-vs-ask comparison sizes are derived for one candle
const WALL_MODES = [
  { key: 'mean',  label: 'Bucket average',   desc: 'poll-weighted mean wall across the whole candle' },
  { key: 'avgc',  label: 'Avg of 5s closes', desc: 'mean of each internal 5s bucket’s closing wall' },
  { key: 'close', label: 'Candle close',     desc: 'the wall at the candle’s last instant (_c)' },
];

// wall overlay palette (distinct from candle colors so fills read as "book")
const COLOR_ASK_WALL = '#ef4444';   // asks sit above price — the ceiling
const COLOR_BID_WALL = '#22c55e';   // bids sit below price — the floor
const WALL_BLOCK_H = 14;            // px, constant in v1 — fill encodes the winner
const WALL_GAP = 5;                 // px between wick tip and block
const FLOW_LINE_GAP = 3;            // px between flow stick and its wall block
const FLOW_THICK = 7;               // px width of the stronger flow's stick
const FLOW_THIN = 2;                // px width of the weaker flow's stick
const FLOW_OVERHANG = 2;            // px the capsule pokes past the block's ends
const COLOR_L20_AGREE = '#64748b';  // subtle — deep book confirms the shown verdict
const COLOR_L20_DIVERGE = '#f97316'; // alert — 20-level winner contradicts the 5-level one
const L20_RADIUS = 3;               // px marker dot radius
const L20_GAP = 5;                  // px above the buy stick's overhang

/**
 * Series primitive drawing one ask block above and one bid block below every
 * candle. The bigger side (COIN size, never usd) is filled, the smaller is
 * outline-only; a tie leaves both outlined. Candles where either side is null
 * (degraded / missing book data) get no blocks at all — never a fake fill.
 *
 * Beside each block sits its ATTACKING taker flow as a vertical line: green
 * buy flow left of the ask block (buyers eat the ceiling), red sell flow left
 * of the bid block (sellers eat the floor). The stronger flow (coin volume)
 * gets the clearly thicker line; a tie draws both thin, zero trades draws none.
 *
 * Comparison sizes are stacked upstream by signal_buckets:
 *   mean mode: ask_size / bid_size   — poll-weighted mean over the bucket
 *   _c  mode: ask_size_c / bid_size_c — book at the bucket's close
 */
class WallOverlayPrimitive {
  constructor() {
    this._rows = [];          // {time, high, low, ask, bid, askC, bidC, askCA, bidCA}
    this._visible = false;
    this._mode = 'mean';      // 'mean' | 'avgc' | 'close'
    this._flowWinnerOnly = false; // hide the weaker flow line
    this._showL20 = false;        // 20-level alignment markers
    this._chart = null;
    this._series = null;
    this._requestUpdate = null;
    this._paneView = { renderer: () => ({ draw: (target) => this._draw(target) }) };
  }

  attached({ chart, series, requestUpdate }) {
    this._chart = chart;
    this._series = series;
    this._requestUpdate = requestUpdate;
  }

  detached() { this._chart = null; this._series = null; }
  paneViews() { return [this._paneView]; }
  updateAllViews() {}

  setRows(rows) { this._rows = rows; this._requestUpdate?.(); }
  setVisible(v) { this._visible = v; this._requestUpdate?.(); }
  setMode(m) { this._mode = m; this._requestUpdate?.(); }
  setFlowWinnerOnly(v) { this._flowWinnerOnly = v; this._requestUpdate?.(); }
  setShowL20(v) { this._showL20 = v; this._requestUpdate?.(); }

  _draw(target) {
    if (!this._visible || !this._chart || !this._series || this._rows.length === 0) return;
    target.useBitmapCoordinateSpace((scope) => {
      const ctx = scope.context;
      const hr = scope.horizontalPixelRatio;
      const vr = scope.verticalPixelRatio;
      const timeScale = this._chart.timeScale();
      const barW = Math.max(2, timeScale.options().barSpacing * 0.7) * hr;
      const blockH = WALL_BLOCK_H * vr;
      const gap = WALL_GAP * vr;
      ctx.lineWidth = Math.max(1, 1 * hr);

      for (const r of this._rows) {
        const ask = this._mode === 'close' ? r.askC : this._mode === 'avgc' ? r.askCA : r.ask;
        const bid = this._mode === 'close' ? r.bidC : this._mode === 'avgc' ? r.bidCA : r.bid;
        if (ask == null || bid == null) continue;   // no book data — draw nothing
        const x = timeScale.timeToCoordinate(r.time);
        if (x === null) continue;
        const yHigh = this._series.priceToCoordinate(r.high);
        const yLow = this._series.priceToCoordinate(r.low);
        if (yHigh == null || yLow == null) continue;

        const left = x * hr - barW / 2;
        const askY = yHigh * vr - gap - blockH;     // above the high
        const bidY = yLow * vr + gap;               // below the low

        this._block(ctx, left, askY, barW, blockH, COLOR_ASK_WALL, ask > bid);
        this._block(ctx, left, bidY, barW, blockH, COLOR_BID_WALL, bid > ask);

        // L20 alignment marker: does the 20-level winner match the 5-level
        // verdict currently displayed? One dot above the whole stack — subtle
        // when the deep book agrees, alert-orange when it contradicts. No dot
        // when either comparison is a tie or deep coverage is missing.
        if (this._showL20 && r.l20Bid != null && r.l20Ask != null
            && r.l20Bid !== r.l20Ask && ask !== bid) {
          const deepBidWins = r.l20Bid > r.l20Ask;
          const shownBidWins = bid > ask;
          ctx.fillStyle = deepBidWins === shownBidWins ? COLOR_L20_AGREE : COLOR_L20_DIVERGE;
          ctx.beginPath();
          const cy = askY - (FLOW_OVERHANG + L20_GAP + L20_RADIUS) * vr;
          ctx.arc(x * hr, cy, L20_RADIUS * vr, 0, Math.PI * 2);
          ctx.fill();
        }

        // taker-flow lines, left of the wall they attack
        const buyV = r.buyV || 0;
        const sellV = r.sellV || 0;
        if (buyV > 0 || sellV > 0) {
          const buyW = (buyV > sellV ? FLOW_THICK : FLOW_THIN) * hr;
          const sellW = (sellV > buyV ? FLOW_THICK : FLOW_THIN) * hr;
          const flowGap = FLOW_LINE_GAP * hr;
          // winner-only: hide the weaker line (a tie has no weaker — keep both)
          const drawBuy = !this._flowWinnerOnly || buyV >= sellV;
          const drawSell = !this._flowWinnerOnly || sellV >= buyV;
          const over = FLOW_OVERHANG * vr;
          if (drawBuy) {
            // green buy stick beside the ask block (above the candle)
            this._stick(ctx, left - flowGap - buyW, askY - over, buyW, blockH + 2 * over, COLOR_BID_WALL);
          }
          if (drawSell) {
            // red sell stick beside the bid block (below the candle)
            this._stick(ctx, left - flowGap - sellW, bidY - over, sellW, blockH + 2 * over, COLOR_ASK_WALL);
          }
        }
      }
    });
  }

  // flow line as a capsule (fully rounded ends) so it reads as a glow stick,
  // never as another wall box
  _stick(ctx, x, y, w, h, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, w / 2);
    ctx.fill();
  }

  _block(ctx, x, y, w, h, color, filled) {
    if (filled) {
      ctx.fillStyle = `${color}cc`;
      ctx.fillRect(x, y, w, h);
    } else {
      ctx.strokeStyle = color;
      ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    }
  }
}

const CHART_H = 620;      // px, candle pane + time axis (volume off)
const VOL_PANE_H = 110;   // px, volume pane — added to the chart so candles never rescale

// keyboard navigation sensitivity — shared by all panels, persisted in localStorage
const NAV_STORAGE_KEY = 'bxtviz.nav';
// speeds are % of the visible range per SECOND a key is held (v2 — v1 stored
// per-keypress steps, which are far too slow as speeds, so v1 values are dropped)
const NAV_VERSION = 2;
const NAV_DEFAULTS = { pan: 100, zoom: 80, v: NAV_VERSION };
const loadNavSettings = () => {
  try {
    const v = JSON.parse(localStorage.getItem(NAV_STORAGE_KEY));
    if (v && v.v === NAV_VERSION && Number(v.pan) > 0 && Number(v.zoom) > 0) {
      return { pan: Number(v.pan), zoom: Number(v.zoom), v: NAV_VERSION };
    }
  } catch { /* no storage / corrupt value — fall through to defaults */ }
  return { ...NAV_DEFAULTS };
};

/**
 * One chart panel. `initial` (optional) seeds a BREAKOUT panel: a fixed
 * absolute window {start, end} (epoch s) instead of a "last N" duration, plus
 * the schema / candle size / fullscreen state to open with. `onBreakout(spec)`
 * is called when the user breaks selected candles out into a finer size.
 */
function ChartPanel({ id, activeIdRef, isDarkMode, onRemove, removable, nav, setNav, initial, onBreakout }) {
  const fixedWindow = initial?.fixedWindow ?? null;   // {start, end} epoch seconds, or null
  const [schema, setSchema] = useState(initial?.schema ?? 'btc');
  const [fullscreen, setFullscreen] = useState(!!initial?.fullscreen);
  const [durKey, setDurKey] = useState('1h');
  const [customVal, setCustomVal] = useState(6);
  const [customUnit, setCustomUnit] = useState('h');
  const [tfS, setTfS] = useState(initial?.tfS ?? 60);

  // ── candle selection / breakout ──────────────────────────────────────────
  const [selectMode, setSelectMode] = useState(false);
  const [selection, setSelection] = useState(() => new Set()); // selected candle start times
  const [breakoutPrompt, setBreakoutPrompt] = useState(false);
  const [notice, setNotice] = useState(null);
  const candleDataRef = useRef([]);       // base candle data (with quality colors), logical index == array index
  const selectModeRef = useRef(false);
  selectModeRef.current = selectMode;
  const selectionRef = useRef(selection);
  selectionRef.current = selection;
  const promptRef = useRef(false);
  promptRef.current = breakoutPrompt;
  const dragRef = useRef(null);
  const tfSRef = useRef(tfS);           // the key handler is bound once — read live values via refs
  tfSRef.current = tfS;
  const schemaRef = useRef(schema);
  schemaRef.current = schema;

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  const [hover, setHover] = useState(null);
  const [loadMs, setLoadMs] = useState(null);
  const [showWalls, setShowWalls] = useState(false);   // bid/ask overlay, default off
  const [wallMode, setWallMode] = useState('mean');    // 'mean' | 'avgc' | 'close'
  const [flowWinnerOnly, setFlowWinnerOnly] = useState(false); // hide weaker flow line
  const [showL20, setShowL20] = useState(false); // 20-level alignment markers
  const [showVolume, setShowVolume] = useState(true);  // volume pane, default on
  const [showQualityHelp, setShowQualityHelp] = useState(false);

  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const candleSeriesRef = useRef(null);
  const volSeriesRef = useRef(null);
  const volDataRef = useRef([]);          // last volume data, for re-adding the pane
  const wallsRef = useRef(null);          // WallOverlayPrimitive instance
  const candleMapRef = useRef(new Map()); // epoch -> raw candle row (for hover)

  const durationS = useMemo(() => {
    if (fixedWindow) return fixedWindow.end - fixedWindow.start;
    if (durKey === 'custom') {
      const unit = CUSTOM_UNITS.find((u) => u.key === customUnit);
      const v = Number(customVal);
      return v > 0 && unit ? Math.round(v * unit.s) : null;
    }
    return DURATION_PRESETS.find((d) => d.key === durKey)?.s ?? null;
  }, [durKey, customVal, customUnit, fixedWindow]);

  const estCandles = durationS ? Math.floor(durationS / tfS) : 0;
  const tooMany = estCandles > MAX_CANDLES;
  const tooFew = durationS != null && estCandles < 1;

  // ── chart lifecycle ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: isDarkMode ? '#9ca3af' : '#4b5563',
        panes: { separatorColor: isDarkMode ? '#374151' : '#e5e7eb', enableResize: true },
      },
      grid: {
        vertLines: { color: isDarkMode ? '#1f2937' : '#f3f4f6' },
        horzLines: { color: isDarkMode ? '#1f2937' : '#f3f4f6' },
      },
      crosshair: { mode: CrosshairMode.Normal },
      timeScale: { timeVisible: true, secondsVisible: true, borderColor: isDarkMode ? '#374151' : '#e5e7eb' },
      // margins are CONSTANT — sized for the overlays so toggling them never
      // rescales the candles
      rightPriceScale: {
        borderColor: isDarkMode ? '#374151' : '#e5e7eb',
        scaleMargins: { top: 0.16, bottom: 0.16 },
      },
    });

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: COLOR_UP,
      downColor: COLOR_DOWN,
      borderVisible: false,
      wickUpColor: COLOR_UP,
      wickDownColor: COLOR_DOWN,
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    }, 0);

    const vol = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'right',
    }, 1);

    // hover readout
    chart.subscribeCrosshairMove((param) => {
      if (!param.time || !param.point) { setHover(null); return; }
      const raw = candleMapRef.current.get(param.time);
      setHover(raw && raw.present ? raw : null);
    });

    const walls = new WallOverlayPrimitive();
    candles.attachPrimitive(walls);

    chartRef.current = chart;
    candleSeriesRef.current = candles;
    volSeriesRef.current = vol;
    wallsRef.current = walls;
    try { chart.panes()[1]?.setHeight(110); } catch { /* pane not created yet */ }

    // debug hook: lets devtools reach live chart instances (e.g. takeScreenshot)
    (window.__bxtCharts ||= new Set()).add(chart);

    return () => { window.__bxtCharts?.delete(chart); chart.remove(); chartRef.current = null; };
    // theme is applied via applyOptions below — chart is created once
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // theme swap without rebuilding the chart
  useEffect(() => {
    chartRef.current?.applyOptions({
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: isDarkMode ? '#9ca3af' : '#4b5563',
        panes: { separatorColor: isDarkMode ? '#374151' : '#e5e7eb', enableResize: true },
      },
      grid: {
        vertLines: { color: isDarkMode ? '#1f2937' : '#f3f4f6' },
        horzLines: { color: isDarkMode ? '#1f2937' : '#f3f4f6' },
      },
      timeScale: { borderColor: isDarkMode ? '#374151' : '#e5e7eb' },
      rightPriceScale: { borderColor: isDarkMode ? '#374151' : '#e5e7eb' },
    });
  }, [isDarkMode]);

  // ── data fetch ───────────────────────────────────────────────────────────
  const load = useCallback(async () => {
    if (!durationS || tooMany || tooFew) return;
    setLoading(true);
    setError(null);
    const t0 = performance.now();
    try {
      const end = fixedWindow ? new Date(fixedWindow.end * 1000) : new Date();
      const start = fixedWindow ? new Date(fixedWindow.start * 1000) : new Date(end.getTime() - durationS * 1000);
      const qs = new URLSearchParams({
        schema, tf_s: String(tfS),
        start: start.toISOString(), end: end.toISOString(),
        l20: showL20 ? '1' : '0',
      });
      const r = await fetch(`${API}/candles?${qs}`);
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);

      const candleData = [];
      const volData = [];
      const map = new Map();
      for (const c of body.candles) {
        map.set(c.t, c);
        if (!c.present) {
          candleData.push({ time: c.t }); // whitespace — an honest gap
          volData.push({ time: c.t });
          continue;
        }
        const bar = { time: c.t, open: c.o, high: c.h, low: c.l, close: c.c };
        if (!c.ideal) {
          const col = c.revised ? COLOR_REVISED : COLOR_DEGRADED;
          bar.color = col; bar.wickColor = col; bar.borderColor = col;
        }
        candleData.push(bar);
        volData.push({
          time: c.t,
          value: c.v,
          color: !c.ideal
            ? (c.revised ? `${COLOR_REVISED}66` : `${COLOR_DEGRADED}66`)
            : (c.c >= c.o ? `${COLOR_UP}66` : `${COLOR_DOWN}66`),
        });
      }
      candleMapRef.current = map;

      // sane price precision per asset
      const lastClose = [...body.candles].reverse().find((c) => c.present)?.c ?? 100;
      const precision = lastClose >= 10000 ? 1 : lastClose >= 100 ? 2 : 4;
      candleSeriesRef.current?.applyOptions({
        priceFormat: { type: 'price', precision, minMove: Number((10 ** -precision).toFixed(precision)) },
      });

      candleDataRef.current = candleData;
      setSelection(new Set());            // new data → selection no longer meaningful
      candleSeriesRef.current?.setData(candleData);
      volDataRef.current = volData;
      volSeriesRef.current?.setData(volData);
      wallsRef.current?.setRows(
        body.candles
          .filter((c) => c.present)
          .map((c) => ({
            time: c.t, high: c.h, low: c.l,
            ask: c.ask_size, bid: c.bid_size,
            askC: c.ask_size_c, bidC: c.bid_size_c,
            askCA: c.ask_size_c_avg, bidCA: c.bid_size_c_avg,
            buyV: c.buy_v, sellV: c.sell_v,
            l20Bid: c.l20_bid, l20Ask: c.l20_ask,
          })),
      );
      try { chartRef.current?.panes()[1]?.setHeight(110); } catch { /* ignore */ }
      resetView();

      setStats(body);
      setLoadMs(Math.round(performance.now() - t0));
    } catch (e) {
      setError(String(e.message || e));
      setStats(null);
    } finally {
      setLoading(false);
    }
  }, [schema, tfS, durationS, tooMany, tooFew, showL20, fixedWindow]);

  // ── selection rendering: selected candles go blue, everything else keeps
  //    its quality color; clearing restores the base data exactly ─────────
  useEffect(() => {
    const series = candleSeriesRef.current;
    const base = candleDataRef.current;
    if (!series || base.length === 0) return;
    if (selection.size === 0) { series.setData(base); return; }
    series.setData(base.map((b) => (
      b.open != null && selection.has(b.time)
        ? { ...b, color: COLOR_SELECTED, wickColor: COLOR_SELECTED, borderColor: COLOR_SELECTED }
        : b
    )));
  }, [selection]);

  // select mode: the chart must not pan on drag (a drag SELECTS), and leaving
  // the mode clears the highlight
  useEffect(() => {
    chartRef.current?.applyOptions({
      handleScroll: { pressedMouseMove: !selectMode, horzTouchDrag: !selectMode, vertTouchDrag: !selectMode },
      handleScale: { axisPressedMouseMove: !selectMode },
    });
    if (!selectMode) { setSelection(new Set()); setBreakoutPrompt(false); }
  }, [selectMode]);
  useEffect(() => { if (!fullscreen) setSelectMode(false); }, [fullscreen]);

  useEffect(() => {
    if (!notice) return undefined;
    const t = setTimeout(() => setNotice(null), 2500);
    return () => clearTimeout(t);
  }, [notice]);

  // click / drag selection on the chart surface
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return undefined;
    const logicalAt = (clientX) => {
      const rect = el.getBoundingClientRect();
      const l = chartRef.current?.timeScale().coordinateToLogical(clientX - rect.left);
      return l == null ? null : Math.round(l);
    };
    const presentTimesBetween = (a, b) => {
      const base = candleDataRef.current;
      const lo = Math.max(0, Math.min(a, b)), hi = Math.min(base.length - 1, Math.max(a, b));
      const out = new Set();
      for (let i = lo; i <= hi; i++) if (base[i].open != null) out.add(base[i].time);
      return out;
    };
    const onDown = (e) => {
      if (!selectModeRef.current || e.button !== 0) return;
      const l = logicalAt(e.clientX);
      if (l == null) return;
      dragRef.current = { startL: l, startX: e.clientX, moved: false };
      e.preventDefault();
    };
    const onMove = (e) => {
      const d = dragRef.current;
      if (!d) return;
      if (Math.abs(e.clientX - d.startX) > 3) d.moved = true;
      if (!d.moved) return;
      const l = logicalAt(e.clientX);
      if (l == null) return;
      setSelection(presentTimesBetween(d.startL, l));
    };
    const onUp = () => {
      const d = dragRef.current;
      if (!d) return;
      dragRef.current = null;
      if (d.moved) return;
      const bar = candleDataRef.current[d.startL];      // plain click toggles one candle
      if (!bar || bar.open == null) return;
      setSelection((prev) => {
        const next = new Set(prev);
        if (next.has(bar.time)) next.delete(bar.time); else next.add(bar.time);
        return next;
      });
    };
    el.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      el.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, []);

  // the selected run, if it is consecutive (no unselected candle inside it)
  const consecutiveRun = () => {
    const sel = selectionRef.current;
    if (sel.size === 0) return { error: 'Select at least one candle first' };
    const base = candleDataRef.current;
    const times = [...sel].sort((a, b) => a - b);
    const iFirst = base.findIndex((b) => b.time === times[0]);
    const iLast = base.findIndex((b) => b.time === times[times.length - 1]);
    for (let i = iFirst; i <= iLast; i++) {
      if (base[i].open != null && !sel.has(base[i].time)) {
        return { error: 'Selected candles must be consecutive (no unselected candle between them)' };
      }
    }
    return { start: times[0], end: times[times.length - 1] + tfSRef.current, count: times.length };
  };
  const breakoutOptions = () => {
    const run = consecutiveRun();
    if (run.error) return [];
    return CANDLE_SIZES.filter((c) => c.s < tfSRef.current && (run.end - run.start) / c.s <= MAX_CANDLES);
  };
  const requestBreakout = () => {
    const run = consecutiveRun();
    if (run.error) { setNotice(run.error); return; }
    if (breakoutOptions().length === 0) { setNotice(`No candle size smaller than ${sizeLabel(tfSRef.current)} fits this selection`); return; }
    setBreakoutPrompt(true);
  };
  const doBreakout = (toTfS) => {
    const run = consecutiveRun();
    if (run.error) { setNotice(run.error); return; }
    setBreakoutPrompt(false);
    const sch = schemaRef.current;
    onBreakout?.({
      schema: sch,
      tfS: toTfS,
      fixedWindow: { start: run.start, end: run.end },
      fullscreen: true,
      label: `${sch.toUpperCase()} ${sizeLabel(toTfS)} · ${fmtUtc(run.start).slice(5, 16)}–${fmtUtc(run.end).slice(11, 16)}`,
      from: { tfS: tfSRef.current, count: run.count },
    });
  };

  // load whenever both selections resolve to a valid window
  useEffect(() => { load(); }, [load]);

  useEffect(() => { wallsRef.current?.setVisible(showWalls); }, [showWalls]);

  // ── hotkeys (only for the panel the mouse last entered) ────────────────
  //   v      toggle volume pane
  //   space  toggle bid/ask walls, applying the preset: winning flow only,
  //          L20 off, size mode = avg of 5s closes
  //   f      toggle fullscreen
  //   esc    leave fullscreen
  const showWallsRef = useRef(showWalls);
  showWallsRef.current = showWalls;
  const fullscreenRef = useRef(fullscreen);
  fullscreenRef.current = fullscreen;
  const navRef = useRef(nav);
  navRef.current = nav;

  // ── keyboard view navigation (fullscreen only) ─────────────────────────
  // The price axis has no public "set visible range" API, so vertical pan /
  // zoom is done by handing the candle series a fixed autoscale range and
  // zeroing the scale margins (they'd otherwise re-pad the range every
  // keypress). resetView() returns to autoscale + fitContent.
  const manualRangeRef = useRef(null);   // {min, max} while the user is driving the price axis

  const currentPriceRange = () => {
    if (manualRangeRef.current) return manualRangeRef.current;
    const chart = chartRef.current, series = candleSeriesRef.current;
    const h = chart.panes()[0].getHeight();
    const top = series.coordinateToPrice(0);
    const bottom = series.coordinateToPrice(h);
    return top != null && bottom != null ? { min: bottom, max: top } : null;
  };
  const applyPriceRange = (r) => {
    manualRangeRef.current = r;
    const series = candleSeriesRef.current;
    series.priceScale().applyOptions({ scaleMargins: { top: 0, bottom: 0 } });
    series.applyOptions({
      autoscaleInfoProvider: () => ({
        priceRange: { minValue: r.min, maxValue: r.max },
        margins: { above: 0, below: 0 },
      }),
    });
  };
  const resetView = () => {
    manualRangeRef.current = null;
    const series = candleSeriesRef.current;
    series?.applyOptions({ autoscaleInfoProvider: (original) => original() });
    series?.priceScale().applyOptions({ scaleMargins: { top: 0.16, bottom: 0.16 } });
    chartRef.current?.timeScale().fitContent();
  };

  // Continuous motion: held keys live in a Set and a requestAnimationFrame
  // loop applies `dt`-scaled deltas every frame, so holding a key glides
  // instead of stepping on the OS key-repeat. Sensitivity = % of the visible
  // range per SECOND held.
  const NAV_KEYS = new Set(['w', 'a', 's', 'd', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']);
  const heldRef = useRef(new Set());
  const rafRef = useRef(0);
  const lastFrameRef = useRef(0);

  const navigateBy = (key, dt) => {
    const chart = chartRef.current;
    if (!chart) return;
    const ts = chart.timeScale();
    const lr = ts.getVisibleLogicalRange();
    const pan = (navRef.current.pan / 100) * dt;   // fraction of range this frame
    const zoom = (navRef.current.zoom / 100) * dt;

    if ((key === 'a' || key === 'd') && lr) {                 // pan time
      // same PIXEL speed as w/s: pan% of the candle pane's height per second,
      // converted to bars — otherwise the wider axis would visibly move faster
      const paneH = chart.panes()[0].getHeight();
      const pxPerBar = ts.width() / (lr.to - lr.from);
      if (!(pxPerBar > 0)) return;
      const d = (paneH * pan) / pxPerBar * (key === 'a' ? -1 : 1);
      ts.setVisibleLogicalRange({ from: lr.from + d, to: lr.to + d });
    } else if ((key === 'ArrowLeft' || key === 'ArrowRight') && lr) { // zoom time
      const c = (lr.from + lr.to) / 2;
      let half = (lr.to - lr.from) / 2;
      half *= key === 'ArrowRight' ? 1 - zoom : 1 + zoom;
      half = Math.max(2.5, half);
      ts.setVisibleLogicalRange({ from: c - half, to: c + half });
    } else if (key === 'w' || key === 's') {                  // pan price
      const r = currentPriceRange();
      if (!r) return;
      const d = (r.max - r.min) * pan * (key === 'w' ? 1 : -1);
      applyPriceRange({ min: r.min + d, max: r.max + d });
    } else if (key === 'ArrowUp' || key === 'ArrowDown') {    // zoom price
      const r = currentPriceRange();
      if (!r) return;
      const c = (r.min + r.max) / 2;
      let half = (r.max - r.min) / 2;
      half *= key === 'ArrowUp' ? 1 - zoom : 1 + zoom;
      applyPriceRange({ min: c - half, max: c + half });
    }
  };

  const frame = (now) => {
    const dt = Math.min((now - lastFrameRef.current) / 1000, 0.1); // cap: tab-switch gaps don't teleport
    lastFrameRef.current = now;
    for (const k of heldRef.current) navigateBy(k, dt);
    rafRef.current = heldRef.current.size && fullscreenRef.current ? requestAnimationFrame(frame) : 0;
  };
  const startLoop = () => {
    if (rafRef.current) return;
    lastFrameRef.current = performance.now();
    rafRef.current = requestAnimationFrame(frame);
  };
  const releaseAll = () => { heldRef.current.clear(); };

  useEffect(() => {
    const onKey = (e) => {
      if (activeIdRef.current !== id) return;
      const tag = e.target?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      // breakout prompt open: digits pick a size, esc closes, nothing else leaks through
      if (promptRef.current) {
        e.preventDefault();
        if (e.key === 'Escape') setBreakoutPrompt(false);
        else if (/^[1-9]$/.test(e.key)) {
          const opt = breakoutOptions()[Number(e.key) - 1];
          if (opt) doBreakout(opt.s);
        }
        return;
      }
      if (fullscreenRef.current) {
        const navKey = e.key.length === 1 ? e.key.toLowerCase() : e.key;
        if (e.key === 'Shift') {                       // toggle select mode
          if (!e.repeat) setSelectMode((m) => !m);
          return;
        }
        if (navKey === 'b') {
          e.preventDefault();
          if (selectModeRef.current) requestBreakout();
          else setNotice('Press shift to enter select mode, then select candles and press b');
          return;
        }
        if (e.key === 'Escape' && selectModeRef.current) { setSelectMode(false); return; }
        if (NAV_KEYS.has(navKey)) {
          e.preventDefault();
          if (!e.repeat) { heldRef.current.add(navKey); startLoop(); }
          return;
        }
        if (navKey === 'r') { e.preventDefault(); resetView(); return; }
      }
      if (e.key === 'v' || e.key === 'V') {
        e.preventDefault();
        setShowVolume((v) => !v);
      } else if (e.key === ' ') {
        e.preventDefault();
        if (!showWallsRef.current) {
          setFlowWinnerOnly(true);
          setShowL20(false);
          setWallMode('avgc');
        }
        setShowWalls(!showWallsRef.current);
      } else if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        setFullscreen((f) => !f);
      } else if (e.key === 'Escape') {
        setFullscreen(false);
      }
    };
    const onKeyUp = (e) => {
      const navKey = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      heldRef.current.delete(navKey);
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', releaseAll);   // a missed keyup must never leave a key "stuck"
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', releaseAll);
      releaseAll();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [id, activeIdRef]);

  // leaving fullscreen releases held keys
  useEffect(() => { if (!fullscreen) releaseAll(); }, [fullscreen]);

  useEffect(() => { wallsRef.current?.setMode(wallMode); }, [wallMode]);

  useEffect(() => { wallsRef.current?.setFlowWinnerOnly(flowWinnerOnly); }, [flowWinnerOnly]);

  useEffect(() => { wallsRef.current?.setShowL20(showL20); }, [showL20]);

  // volume pane toggle — removing the series removes its pane; re-adding
  // recreates the pane with the cached data
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    if (showVolume && !volSeriesRef.current) {
      const vol = chart.addSeries(HistogramSeries, {
        priceFormat: { type: 'volume' },
        priceScaleId: 'right',
      }, 1);
      vol.setData(volDataRef.current);
      volSeriesRef.current = vol;
      try { chart.panes()[1]?.setHeight(110); } catch { /* pane not created yet */ }
    } else if (!showVolume && volSeriesRef.current) {
      chart.removeSeries(volSeriesRef.current);
      volSeriesRef.current = null;
    }
  }, [showVolume]);

  // ── ui helpers ───────────────────────────────────────────────────────────
  const card = isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200';
  const selectCls = `text-sm font-medium rounded-lg px-3 py-2 border focus:ring-2 focus:ring-blue-500 focus:outline-none transition-colors duration-200 ${
    isDarkMode ? 'bg-gray-900 border-gray-700 text-gray-200' : 'bg-white border-gray-300 text-gray-800'
  }`;
  const dim = isDarkMode ? 'text-gray-400' : 'text-gray-500';

  const hoverChange = hover && hover.o ? ((hover.c - hover.o) / hover.o) * 10000 : null;

  return (
    <div
      onMouseEnter={() => { activeIdRef.current = id; }}
      className={fullscreen
        ? `fixed inset-0 z-50 flex flex-col gap-3 p-3 overflow-hidden ${isDarkMode ? 'bg-gray-900' : 'bg-gray-50'}`
        : 'space-y-4'}
    >
      {/* ── control bar ── */}
      <div className={`rounded-lg border p-4 transition-colors duration-200 ${card}`}>
        <div className="flex flex-wrap items-center gap-3">
          {/* asset */}
          <div className="flex items-center gap-2">
            <span className={`text-2xl font-black tracking-tight ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              {SCHEMAS.find((s) => s.key === schema)?.label}
            </span>
            <select value={schema} onChange={(e) => setSchema(e.target.value)} className={selectCls}>
              {SCHEMAS.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
            </select>
          </div>

          {/* duration — or, for a breakout panel, the fixed window it covers */}
          {fixedWindow ? (
            <span
              className={`text-sm font-medium rounded-lg px-3 py-2 border ${isDarkMode ? 'bg-gray-900/60 border-gray-700 text-gray-300' : 'bg-gray-50 border-gray-300 text-gray-700'}`}
              title={`Breakout of ${initial?.from?.count ?? '?'} × ${sizeLabel(initial?.from?.tfS)} candle(s)`}
            >
              ⤵ {fmtUtc(fixedWindow.start).slice(5, 16)} → {fmtUtc(fixedWindow.end).slice(11, 16)} UTC
              <span className={`ml-1.5 text-xs ${dim}`}>({initial?.from?.count} × {sizeLabel(initial?.from?.tfS)})</span>
            </span>
          ) : (
            <select value={durKey} onChange={(e) => setDurKey(e.target.value)} className={selectCls}>
              {DURATION_PRESETS.map((d) => <option key={d.key} value={d.key}>{d.label}</option>)}
            </select>
          )}
          {!fixedWindow && durKey === 'custom' && (
            <div className="flex items-center gap-1.5">
              <span className={`text-sm ${dim}`}>last</span>
              <input
                type="number" min="1" value={customVal}
                onChange={(e) => setCustomVal(e.target.value)}
                className={`${selectCls} w-20`}
              />
              <select value={customUnit} onChange={(e) => setCustomUnit(e.target.value)} className={selectCls}>
                {CUSTOM_UNITS.map((u) => <option key={u.key} value={u.key}>{u.label}</option>)}
              </select>
            </div>
          )}

          {/* candle size */}
          <div className="flex items-center gap-1.5">
            <span className={`text-sm ${dim}`}>candle size</span>
            <select value={tfS} onChange={(e) => setTfS(Number(e.target.value))} className={selectCls}>
              {CANDLE_SIZES.map((c) => <option key={c.s} value={c.s}>{c.label}</option>)}
            </select>
          </div>

          <button
            onClick={load}
            disabled={loading || tooMany || tooFew}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition-all duration-150 ${
              loading
                ? 'opacity-50 cursor-wait'
                : ''
            } ${isDarkMode ? 'bg-blue-600 hover:bg-blue-500 text-white' : 'bg-blue-600 hover:bg-blue-700 text-white'} disabled:opacity-40`}
          >
            {loading ? 'Loading…' : '↻ Refresh'}
          </button>

          <span className={`text-xs ${dim} ml-auto`}>
            {estCandles > 0 && `≈ ${estCandles.toLocaleString()} candles`}
          </span>

          <button
            onClick={() => setFullscreen((f) => !f)}
            title={fullscreen ? 'Exit full screen (f / esc)' : 'Full screen (f)'}
            className={`px-2.5 py-1 text-sm font-medium rounded-lg transition-colors duration-150 ${
              fullscreen
                ? isDarkMode ? 'bg-teal-600 text-white hover:bg-teal-500' : 'bg-teal-500 text-white hover:bg-teal-600'
                : isDarkMode ? 'text-gray-400 hover:text-white hover:bg-gray-700' : 'text-gray-500 hover:text-gray-900 hover:bg-gray-100'
            }`}
          >
            ⛶ {fullscreen ? 'Exit' : 'Full screen'} <kbd className={`px-1 rounded text-[10px] font-mono ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>f</kbd>
          </button>

          {removable && (
            <button
              onClick={onRemove}
              title="Remove this chart"
              className={`px-2 py-1 text-sm font-bold rounded-lg transition-colors duration-150 ${
                isDarkMode
                  ? 'text-gray-500 hover:text-red-400 hover:bg-gray-700'
                  : 'text-gray-400 hover:text-red-600 hover:bg-gray-100'
              }`}
            >
              ✕
            </button>
          )}
        </div>

        {tooMany && (
          <div className={`mt-3 text-sm rounded-lg px-3 py-2 ${isDarkMode ? 'bg-amber-900/20 text-amber-300' : 'bg-amber-50 text-amber-800'}`}>
            {estCandles.toLocaleString()} candles exceeds the {MAX_CANDLES.toLocaleString()} cap — pick a coarser candle size or shorter window.
          </div>
        )}
        {tooFew && (
          <div className={`mt-3 text-sm rounded-lg px-3 py-2 ${isDarkMode ? 'bg-amber-900/20 text-amber-300' : 'bg-amber-50 text-amber-800'}`}>
            Window is smaller than one candle — pick a longer duration or a finer candle size.
          </div>
        )}
        {error && (
          <div className={`mt-3 text-sm rounded-lg px-3 py-2 ${isDarkMode ? 'bg-red-900/20 text-red-300' : 'bg-red-50 text-red-800'}`}>
            ⚠️ {error}
          </div>
        )}
      </div>

      {/* ── chart ── */}
      <div className={`rounded-lg border transition-colors duration-200 relative ${card} ${fullscreen ? 'flex-1 min-h-0 flex flex-col' : ''}`}>
        {/* select-mode banner */}
        {selectMode && (
          <div className={`flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2 text-sm border-b ${
            isDarkMode ? 'bg-blue-900/30 border-blue-800 text-blue-200' : 'bg-blue-50 border-blue-200 text-blue-800'
          }`}>
            <span className="font-semibold">🎯 Selection mode</span>
            <span>click or drag to select candles</span>
            <span className="font-semibold" style={{ color: COLOR_SELECTED }}>{selection.size} selected</span>
            <span className="ml-auto flex items-center gap-3 text-xs">
              <span><kbd className={`px-1 rounded font-mono ${isDarkMode ? 'bg-blue-800 text-blue-100' : 'bg-blue-200 text-blue-900'}`}>b</kbd> break out consecutive selection</span>
              <span><kbd className={`px-1 rounded font-mono ${isDarkMode ? 'bg-blue-800 text-blue-100' : 'bg-blue-200 text-blue-900'}`}>shift</kbd> / <kbd className={`px-1 rounded font-mono ${isDarkMode ? 'bg-blue-800 text-blue-100' : 'bg-blue-200 text-blue-900'}`}>esc</kbd> exit</span>
            </span>
          </div>
        )}

        {/* transient notice */}
        {notice && (
          <div className={`absolute top-12 left-1/2 -translate-x-1/2 z-20 px-4 py-2 rounded-lg text-sm shadow-lg ${
            isDarkMode ? 'bg-amber-900/90 text-amber-100 border border-amber-700' : 'bg-amber-100 text-amber-900 border border-amber-300'
          }`}>
            {notice}
          </div>
        )}

        {/* breakout size prompt */}
        {breakoutPrompt && (() => {
          const run = consecutiveRun();
          const opts = breakoutOptions();
          return (
            <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50" onClick={() => setBreakoutPrompt(false)}>
              <div
                onClick={(e) => e.stopPropagation()}
                className={`rounded-xl border shadow-2xl p-5 w-[420px] ${isDarkMode ? 'bg-gray-800 border-gray-700 text-gray-100' : 'bg-white border-gray-200 text-gray-900'}`}
              >
                <div className="text-lg font-bold mb-1">Break out {run.count} × {sizeLabel(tfS)} candle{run.count === 1 ? '' : 's'}</div>
                <div className={`text-xs mb-4 ${dim}`}>
                  {fmtUtc(run.start)} → {fmtUtc(run.end)} UTC · pick the candle size for the breakout chart
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {opts.map((o, i) => (
                    <button
                      key={o.s}
                      onClick={() => doBreakout(o.s)}
                      className={`px-3 py-2 rounded-lg text-sm font-medium border transition-colors duration-150 ${
                        isDarkMode
                          ? 'bg-gray-900 border-gray-700 hover:border-teal-500 hover:text-teal-300'
                          : 'bg-gray-50 border-gray-300 hover:border-teal-500 hover:text-teal-700'
                      }`}
                    >
                      <kbd className={`mr-1.5 px-1 rounded text-[10px] font-mono ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>{i + 1}</kbd>
                      {o.label}
                      <span className={`block text-[10px] ${dim}`}>≈ {Math.round((run.end - run.start) / o.s).toLocaleString()} candles</span>
                    </button>
                  ))}
                </div>
                <div className={`mt-4 text-xs ${dim}`}>press a number, or <kbd className={`px-1 rounded font-mono ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>esc</kbd> to cancel</div>
              </div>
            </div>
          );
        })()}

        {/* hover readout */}
        <div className={`absolute top-2 left-3 z-10 text-xs font-mono pointer-events-none ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
          {hover ? (
            <>
              <span className={dim}>{fmtUtc(hover.t)} UTC · </span>
              O <b>{hover.o?.toLocaleString()}</b>{' '}
              H <b>{hover.h?.toLocaleString()}</b>{' '}
              L <b>{hover.l?.toLocaleString()}</b>{' '}
              C <b>{hover.c?.toLocaleString()}</b>{' '}
              <span className={hoverChange >= 0 ? 'text-emerald-500' : 'text-red-500'}>
                {hoverChange >= 0 ? '+' : ''}{hoverChange?.toFixed(1)} bps
              </span>
              <span className={dim}> · vol {hover.v?.toFixed(3)}</span>
              {showWalls && (() => {
                const a = wallMode === 'close' ? hover.ask_size_c
                  : wallMode === 'avgc' ? hover.ask_size_c_avg : hover.ask_size;
                const b = wallMode === 'close' ? hover.bid_size_c
                  : wallMode === 'avgc' ? hover.bid_size_c_avg : hover.bid_size;
                if (a == null || b == null) return <span className={dim}> · book n/a</span>;
                const modeTag = wallMode === 'close' ? 'close' : wallMode === 'avgc' ? 'avg-of-closes' : 'avg';
                const buyV = hover.buy_v || 0;
                const sellV = hover.sell_v || 0;
                return (
                  <span>
                    {' '}· <span style={{ color: COLOR_BID_WALL, fontWeight: b > a ? 700 : 400 }}>bid {b.toFixed(3)}</span>
                    {' / '}
                    <span style={{ color: COLOR_ASK_WALL, fontWeight: a > b ? 700 : 400 }}>ask {a.toFixed(3)}</span>
                    <span className={dim}> {modeTag} coins</span>
                    {' '}· <span style={{ color: COLOR_BID_WALL, fontWeight: buyV > sellV ? 700 : 400 }}>buy {buyV.toFixed(3)}</span>
                    {' / '}
                    <span style={{ color: COLOR_ASK_WALL, fontWeight: sellV > buyV ? 700 : 400 }}>sell {sellV.toFixed(3)}</span>
                    <span className={dim}> flow</span>
                    {showL20 && (hover.l20_bid != null && hover.l20_ask != null ? (
                      <span>
                        {' '}· L20 bid {hover.l20_bid.toFixed(3)} / ask {hover.l20_ask.toFixed(3)}{' '}
                        {(hover.l20_bid > hover.l20_ask) === (b > a)
                          ? <span style={{ color: COLOR_L20_AGREE }}>agrees</span>
                          : <span style={{ color: COLOR_L20_DIVERGE, fontWeight: 700 }}>DIVERGES</span>}
                      </span>
                    ) : (
                      <span className={dim}> · L20 n/a</span>
                    ))}
                  </span>
                );
              })()}
              {!hover.ideal && (
                <span className={hover.revised ? 'text-amber-500' : 'text-gray-500'}>
                  {' '}· {hover.revised ? 'revised' : 'degraded feed'}
                </span>
              )}
            </>
          ) : stats ? (
            <span className={dim}>
              {stats.schema.toUpperCase()} · {CANDLE_SIZES.find((c) => c.s === stats.tf_s)?.label} candles · bxt mid OHLC · times UTC
            </span>
          ) : null}
        </div>
        {/* height grows by exactly the volume pane, so the candle pane —
            and therefore the candle scale — is identical with volume on or off */}
        <div
          ref={containerRef}
          style={fullscreen
            ? { flex: 1, minHeight: 0 }
            : { height: CHART_H + (showVolume ? VOL_PANE_H : 0) }}
        />

        {/* overlay toggles */}
        <div className={`flex flex-wrap items-center gap-x-5 gap-y-1 px-3 py-2 border-t text-sm transition-colors duration-200 ${
          isDarkMode ? 'border-gray-700' : 'border-gray-200'
        }`}>
          <label className={`flex items-center gap-2 cursor-pointer select-none ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            <input
              type="checkbox"
              checked={showVolume}
              onChange={(e) => setShowVolume(e.target.checked)}
              className="accent-teal-500"
            />
            Volume <kbd className={`px-1 rounded text-[10px] font-mono ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>v</kbd>
            <span className={`text-xs ${dim}`}>(buy + sell coins per candle, lower pane)</span>
          </label>
          <label className={`flex items-center gap-2 cursor-pointer select-none ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            <input
              type="checkbox"
              checked={showWalls}
              onChange={(e) => setShowWalls(e.target.checked)}
              className="accent-teal-500"
            />
            Bid/Ask walls <kbd className={`px-1 rounded text-[10px] font-mono ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>space</kbd>
            <span className={`text-xs ${dim}`}>
              (<span style={{ color: COLOR_BID_WALL }}>bid</span> below · <span style={{ color: COLOR_ASK_WALL }}>ask</span> above — bigger side by coin size is filled ·
              flow lines beside each block: <span style={{ color: COLOR_BID_WALL }}>buy</span>/<span style={{ color: COLOR_ASK_WALL }}>sell</span>, thicker = stronger)
            </span>
          </label>
          {showWalls && (
            <label className={`flex items-center gap-2 cursor-pointer select-none ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              <input
                type="checkbox"
                checked={flowWinnerOnly}
                onChange={(e) => setFlowWinnerOnly(e.target.checked)}
                className="accent-teal-500"
              />
              Winning flow only
              <span className={`text-xs ${dim}`}>(hide the weaker flow line)</span>
            </label>
          )}
          {showWalls && (
            <label className={`flex items-center gap-2 cursor-pointer select-none ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              <input
                type="checkbox"
                checked={showL20}
                onChange={(e) => setShowL20(e.target.checked)}
                className="accent-teal-500"
              />
              L20 alignment
              <span className={`text-xs ${dim}`}>
                (dot per candle: <span style={{ color: COLOR_L20_AGREE }}>●</span> 20-level book agrees with shown walls · <span style={{ color: COLOR_L20_DIVERGE }}>●</span> diverges)
              </span>
            </label>
          )}
          {showWalls && (
            <div className="flex items-center gap-2">
              <span className={`text-xs ${dim}`}>size mode</span>
              <div className={`flex rounded-lg p-0.5 gap-0.5 ${isDarkMode ? 'bg-gray-900' : 'bg-gray-100'}`}>
                {WALL_MODES.map((m) => (
                  <button
                    key={m.key}
                    onClick={() => setWallMode(m.key)}
                    title={m.desc}
                    className={`px-2.5 py-1 text-xs font-medium rounded-md transition-all duration-150 ${
                      wallMode === m.key
                        ? isDarkMode
                          ? 'bg-gray-700 text-teal-300 shadow-sm ring-1 ring-teal-500/40'
                          : 'bg-white text-teal-700 shadow-sm ring-1 ring-teal-300'
                        : isDarkMode
                          ? 'text-gray-500 hover:text-gray-200'
                          : 'text-gray-500 hover:text-gray-800'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
              <span className={`text-xs ${dim}`}>
                {WALL_MODES.find((m) => m.key === wallMode)?.desc}
              </span>
            </div>
          )}

          {/* keyboard navigation sensitivity (fullscreen) */}
          <div className={`flex items-center gap-2 ml-auto text-xs ${dim}`} title="Full screen only: hold wasd to pan, ←→ to zoom time, ↑↓ to zoom price; r resets the view. Speeds are % of the visible range per second held.">
            <span>⌨ full-screen nav speed:</span>
            <label className="flex items-center gap-1">
              wasd
              <input
                type="number" min="1" max="500" step="5" value={nav.pan}
                onChange={(e) => setNav((n) => ({ ...n, pan: Math.max(1, Math.min(500, Number(e.target.value) || 1)) }))}
                onKeyDown={(e) => { if (e.key === 'Enter') e.target.blur(); }}
                className={`w-14 px-1.5 py-0.5 rounded border text-xs ${isDarkMode ? 'bg-gray-900 border-gray-700 text-gray-200' : 'bg-white border-gray-300 text-gray-800'}`}
              />%
            </label>
            <label className="flex items-center gap-1">
              arrows
              <input
                type="number" min="1" max="500" step="5" value={nav.zoom}
                onChange={(e) => setNav((n) => ({ ...n, zoom: Math.max(1, Math.min(500, Number(e.target.value) || 1)) }))}
                onKeyDown={(e) => { if (e.key === 'Enter') e.target.blur(); }}
                className={`w-14 px-1.5 py-0.5 rounded border text-xs ${isDarkMode ? 'bg-gray-900 border-gray-700 text-gray-200' : 'bg-white border-gray-300 text-gray-800'}`}
              />%
            </label>
            <span>per second held · <kbd className={`px-1 rounded text-[10px] font-mono ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>r</kbd> reset · <kbd className={`px-1 rounded text-[10px] font-mono ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>shift</kbd> select · <kbd className={`px-1 rounded text-[10px] font-mono ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>b</kbd> break out</span>
          </div>
        </div>
      </div>

      {/* ── stats + legend strip ── */}
      {stats && (
        <div className={`rounded-lg border p-3 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs transition-colors duration-200 ${card} ${dim}`}>
          <span>
            <b className={isDarkMode ? 'text-gray-200' : 'text-gray-800'}>{stats.actual.toLocaleString()}</b>
            /{stats.expected.toLocaleString()} buckets
            {stats.missing > 0 && <span className="text-red-500"> · {stats.missing.toLocaleString()} missing</span>}
          </span>
          <span>ideal <b className={isDarkMode ? 'text-gray-200' : 'text-gray-800'}>
            {stats.actual ? Math.round((100 * stats.ideal) / stats.actual) : 0}%</b>
          </span>
          {stats.revised > 0 && <span className="text-amber-500">{stats.revised} revised</span>}
          <span>{fmtUtc(new Date(stats.grid_start).getTime() / 1000)} → {fmtUtc(new Date(stats.grid_end).getTime() / 1000)} UTC</span>
          {loadMs != null && <span>{loadMs} ms</span>}
          <span className="ml-auto flex items-center gap-3">
            <span className="flex items-center gap-1" title="Passed every quality gate: enough polls, evenly sampled, contiguous, never revised.">
              <i className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: COLOR_UP }} />/<i className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: COLOR_DOWN }} /> clean
            </span>
            <span className="flex items-center gap-1" title="Bucket exists but its sampling failed a gate — values built from thinner data.">
              <i className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: COLOR_DEGRADED }} /> degraded
            </span>
            <span className="flex items-center gap-1" title="Corrected after the fact by the reconciler — a live system may have seen different values.">
              <i className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: COLOR_REVISED }} /> revised
            </span>
            <span title="No bucket row stored for that time slot — the chart leaves real whitespace, never bridges.">gap = missing bucket</span>
            <button
              onClick={() => setShowQualityHelp((v) => !v)}
              title="What do these mean?"
              className={`w-5 h-5 rounded-full text-[11px] font-bold leading-none transition-colors duration-150 ${
                showQualityHelp
                  ? isDarkMode ? 'bg-teal-600 text-white' : 'bg-teal-500 text-white'
                  : isDarkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
              }`}
            >
              i
            </button>
          </span>
        </div>
      )}

      {/* quality definitions */}
      {stats && showQualityHelp && (
        <div className={`rounded-lg border p-4 text-xs space-y-2 transition-colors duration-200 ${card} ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
          <div className="flex gap-2">
            <span className="flex gap-0.5 shrink-0 mt-0.5"><i className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: COLOR_UP }} /><i className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: COLOR_DOWN }} /></span>
            <span><b>Clean</b> — the bucket passed every quality gate: ≥7 polls per 5s of data, no polling gap over 1.5s, contiguous with the previous bucket, and never revised. This is the only tier a live system would have seen exactly as drawn.</span>
          </div>
          <div className="flex gap-2">
            <i className="w-2.5 h-2.5 rounded-sm inline-block shrink-0 mt-0.5" style={{ background: COLOR_DEGRADED }} />
            <span><b>Degraded</b> — the bucket exists but its sampling failed a gate (too few polls, a polling gap, or a break in contiguity). Its OHLC, volume, and wall values are built from thinner data — treat them with suspicion, especially during sharp moves.</span>
          </div>
          <div className="flex gap-2">
            <i className="w-2.5 h-2.5 rounded-sm inline-block shrink-0 mt-0.5" style={{ background: COLOR_REVISED }} />
            <span><b>Revised</b> — the reconciler corrected this bucket after it was first written (late trades settling). The values shown are final and accurate, but a live system trading at the time may have seen something different. Revisions cluster in volatile moments.</span>
          </div>
          <div className="flex gap-2">
            <span className={`shrink-0 mt-0.5 ${dim}`}>▢</span>
            <span><b>Gap</b> — no bucket row was stored for that time slot at all. The chart leaves real whitespace rather than interpolating across it, so a flat-looking stretch is never fabricated.</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ── breakout ⇄ URL ───────────────────────────────────────────────────────────
// A breakout opens a REAL browser tab: the selection is encoded in the URL and
// main.jsx mounts <BxtBreakoutPage> when it sees ?bxtviz=1. The originating
// tab is untouched; the new tab can itself break out further.

const breakoutUrl = (spec) => {
  const q = new URLSearchParams({
    bxtviz: '1',
    schema: spec.schema,
    tf: String(spec.tfS),
    start: String(spec.fixedWindow.start),
    end: String(spec.fixedWindow.end),
    from_tf: String(spec.from?.tfS ?? ''),
    from_n: String(spec.from?.count ?? ''),
  });
  return `${window.location.origin}${window.location.pathname}?${q}`;
};

/** Parse a breakout spec from the current URL, or null if this is not a breakout tab. */
export const breakoutSpecFromUrl = () => {
  const q = new URLSearchParams(window.location.search);
  if (q.get('bxtviz') !== '1') return null;
  const schema = q.get('schema'), tfS = Number(q.get('tf'));
  const start = Number(q.get('start')), end = Number(q.get('end'));
  if (!SCHEMAS.some((s) => s.key === schema) || !CANDLE_SIZES.some((c) => c.s === tfS)) return null;
  if (!(start > 0 && end > start)) return null;
  const fromTf = Number(q.get('from_tf')) || null, fromN = Number(q.get('from_n')) || null;
  return {
    schema, tfS,
    fixedWindow: { start, end },
    fullscreen: true,
    from: fromTf ? { tfS: fromTf, count: fromN } : null,
    label: `${schema.toUpperCase()} ${sizeLabel(tfS)} · ${fmtUtc(start).slice(5, 16)}–${fmtUtc(end).slice(11, 16)} UTC`,
  };
};

const openBreakoutTab = (spec) => {
  const w = window.open(breakoutUrl(spec), '_blank');
  if (!w) window.alert('The browser blocked the breakout tab — allow pop-ups for this site and try again.');
};

/**
 * BXT Visualizer — a vertical stack of independent ChartPanels.
 *
 * Each panel owns its full selection (asset / duration / candle size) and its
 * own chart instance; the ＋ button under the stack spawns another panel, so
 * any number of windows/assets/timeframes can be compared down the page.
 * `initialPanel` seeds the first panel (used by the breakout tab page).
 */
export default function BxtVisualizer({ isDarkMode, initialPanel = null }) {
  const [panelIds, setPanelIds] = useState([0]);
  const nextIdRef = useRef(1);
  const activeIdRef = useRef(0);   // panel that receives hotkeys (last hovered)

  // navigation sensitivity: one preference for the whole dashboard, persisted
  const [nav, setNav] = useState(loadNavSettings);
  useEffect(() => {
    try { localStorage.setItem(NAV_STORAGE_KEY, JSON.stringify(nav)); } catch { /* storage unavailable */ }
  }, [nav]);

  const addPanel = () => setPanelIds((ids) => [...ids, nextIdRef.current++]);
  const removePanel = (id) => setPanelIds((ids) => ids.filter((x) => x !== id));

  return (
    <div className="space-y-6">
      {panelIds.map((id, i) => (
        <ChartPanel
          key={id}
          id={id}
          activeIdRef={activeIdRef}
          isDarkMode={isDarkMode}
          removable={panelIds.length > 1}
          onRemove={() => removePanel(id)}
          nav={nav}
          setNav={setNav}
          initial={i === 0 ? initialPanel : null}
          onBreakout={openBreakoutTab}
        />
      ))}

      <button
        onClick={addPanel}
        className={`w-full py-3 rounded-lg border-2 border-dashed text-sm font-medium transition-colors duration-150 ${
          isDarkMode
            ? 'border-gray-700 text-gray-500 hover:border-teal-500/60 hover:text-teal-400 hover:bg-gray-800/50'
            : 'border-gray-300 text-gray-400 hover:border-teal-500/60 hover:text-teal-600 hover:bg-teal-50/50'
        }`}
      >
        ＋ Add comparison chart
      </button>
    </div>
  );
}

/**
 * Standalone page for a breakout browser tab: just the chart, opened in
 * full screen, with the same controls and hotkeys as the main dashboard.
 */
export function BxtBreakoutPage({ isDarkMode, spec }) {
  useEffect(() => { document.title = `⤵ ${spec.label}`; }, [spec.label]);
  return (
    <div className={`min-h-screen p-4 ${isDarkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'}`}>
      <div className={`mb-3 text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-500'}`}>
        Breakout tab · {spec.label}
        {spec.from && ` · from ${spec.from.count} × ${sizeLabel(spec.from.tfS)} candle${spec.from.count === 1 ? '' : 's'}`}
        {' '}· close this tab to return
      </div>
      <BxtVisualizer isDarkMode={isDarkMode} initialPanel={spec} />
    </div>
  );
}
