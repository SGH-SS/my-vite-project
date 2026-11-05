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
// length check (q back / e forward): candles smaller than the selected one;
// the run's last candle is a shade lighter so its end stands out
const COLOR_LEN_BACK = '#a855f7';
const COLOR_LEN_BACK_LAST = '#c084fc';
const COLOR_LEN_FWD = '#22d3ee';
const COLOR_LEN_FWD_LAST = '#67e8f9';

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
const COLOR_WALL_ANCHOR = '#a78bfa'; // dashed marker: first space press of a wall range

// How decisive each comparison was: the winner's SHARE of the pair,
// bigger / (bigger + smaller). FIXED cuts — the same on every coin, candle
// size, wall mode and market regime — so a mark always reads the same:
// faint = "that was closer than 60/40", glow = "that was 3-to-1 or more".
// Deliberately NOT a rolling percentile (that would answer "unusual lately?",
// a different question). Ties are the bottom of the close band.
const SHARE_CLOSE = 0.60;           // winner < 60% of the pair → close call (faint, no fill)
const SHARE_LOPSIDED = 0.75;        // winner ≥ 75% (≥ 3:1) → lopsided (glow ★)
const FLOW_MIN_TRADES = 3;          // flow glow needs ≥ this many taker prints (one print is 100/0)
// Intensity scales INSIDE each band, linearly in the share (fixed, not fit):
//   close: level 0 at 60/40 → 1 at 50/50 — fainter and blurrier the closer it was
//   boost: level 0 at 75/25 → 1 at 100/0 — brighter glow the more lopsided
const FAINT_ALPHA_MAX = 0.6;        // opacity of a close call just under 60/40
const FAINT_ALPHA_MIN = 0.15;       // opacity of a dead-even 50/50
const FAINT_BLUR_MAX = 2.5;         // px blur of a dead-even 50/50 (0 at 60/40)
const GLOW_BLUR_MIN = 3;            // px glow radius at exactly 3:1
const GLOW_BLUR_MAX = 16;           // px glow radius at 100/0
const lerp = (a, b, t) => a + (b - a) * t;
const COLOR_ASK_EDGE = '#fecaca';   // bright inner edge on a lopsided ask block
const COLOR_BID_EDGE = '#bbf7d0';   // bright inner edge on a lopsided bid block

/**
 * Classify one two-sided comparison (x vs y, both ≥ 0).
 * band: 'close' (< SHARE_CLOSE, incl. ties and an all-zero pair)
 *       'plain' (between the cuts, or lopsided but not allowed to glow)
 *       'boost' (≥ SHARE_LOPSIDED AND boostOk)
 * boostOk carries the gates that are NOT ratios: a clean, settled candle,
 * and for flow a minimum print count. Returns null when either side is null.
 */
function splitBand(x, y, boostOk) {
  if (x == null || y == null) return null;
  const tot = x + y;
  const share = tot > 0 ? Math.max(x, y) / tot : 0.5;
  const lopsided = share >= SHARE_LOPSIDED;
  const band = share < SHARE_CLOSE ? 'close' : lopsided && boostOk ? 'boost' : 'plain';
  // 0..1 depth into the band: how close (close) / how lopsided (boost)
  const level = band === 'close' ? (SHARE_CLOSE - share) / (SHARE_CLOSE - 0.5)
    : band === 'boost' ? (share - SHARE_LOPSIDED) / (1 - SHARE_LOPSIDED) : 0;
  return { share, band, level, lopsided, xWins: band !== 'close' && x > y, yWins: band !== 'close' && y > x };
}

// a candle may glow only if a live system would have seen it exactly as drawn
const isSettled = (c) => !!c.ideal && c.complete !== false;
const flowTradesOk = (c) => (c.buyN ?? 0) + (c.sellN ?? 0) >= FLOW_MIN_TRADES;

// the wall pair for the selected size mode, from a row in primitive shape
const wallPair = (r, mode) => (mode === 'close' ? [r.askC, r.bidC]
  : mode === 'avgc' ? [r.askCA, r.bidCA] : [r.ask, r.bid]);

function wallSplit(r, mode) {
  const [ask, bid] = wallPair(r, mode);
  return splitBand(ask, bid, isSettled(r));
}
function flowSplit(r) {
  const buyV = r.buyV || 0;
  const sellV = r.sellV || 0;
  if (!(buyV > 0 || sellV > 0)) return null;     // no trades — no split to draw
  return splitBand(buyV, sellV, isSettled(r) && flowTradesOk(r));
}

// raw /candles row → the shape the overlay, hover and band counts share
const toWallRow = (c) => ({
  time: c.t, high: c.h, low: c.l,
  ask: c.ask_size, bid: c.bid_size,
  askC: c.ask_size_c, bidC: c.bid_size_c,
  askCA: c.ask_size_c_avg, bidCA: c.bid_size_c_avg,
  buyV: c.buy_v, sellV: c.sell_v,
  buyN: c.buy_n, sellN: c.sell_n,
  ideal: c.ideal, complete: c.complete,
  l20Bid: c.l20_bid, l20Ask: c.l20_ask,
});

/**
 * Series primitive drawing one ask block above and one bid block below every
 * candle. The bigger side (COIN size, never usd) is filled, the smaller is
 * outline-only; a tie leaves both outlined. Candles where either side is null
 * (degraded / missing book data) get no blocks at all — never a fake fill.
 *
 * Beside each block sits its ATTACKING taker flow as a vertical line: green
 * buy flow left of the ask block (buyers eat the ceiling), red sell flow left
 * of the bid block (sellers eat the floor). The stronger flow (coin volume)
 * gets the clearly thicker line; zero trades draws none.
 *
 * Decisiveness (splitBand — same fixed cuts for walls and flow):
 *   close (winner < 60%, ties included): no winner — both walls faint
 *     outlines, both flow sticks thin and faint, no L20 dot. Graded: the
 *     closer to 50/50, the fainter and blurrier.
 *   plain (60–75%): filled winner / thick stick, as before
 *   boost (≥ 75% on a clean, settled candle; flow also ≥ 3 prints): the
 *     winner glows — solid fill with a bright edge, glowing stick. Graded:
 *     barely at 3:1, blazing at 100/0.
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
    this._range = null;           // {from, to} candle times — walls drawn only inside; null = all
    this._anchor = null;          // candle time of a pending first space press (dashed marker)
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
  setRange(r) { this._range = r; this._requestUpdate?.(); }
  setAnchor(t) { this._anchor = t; this._requestUpdate?.(); }

  _draw(target) {
    if (!this._chart || !this._series) return;
    target.useBitmapCoordinateSpace((scope) => {
      const ctx = scope.context;
      const hr = scope.horizontalPixelRatio;
      const vr = scope.verticalPixelRatio;
      const timeScale = this._chart.timeScale();

      // pending first space press: dashed vertical line through that candle
      if (this._anchor != null) {
        const ax = timeScale.timeToCoordinate(this._anchor);
        if (ax !== null) {
          ctx.save();
          ctx.strokeStyle = COLOR_WALL_ANCHOR;
          ctx.lineWidth = Math.max(1, hr);
          ctx.setLineDash([4 * vr, 4 * vr]);
          ctx.beginPath();
          ctx.moveTo(Math.round(ax * hr) + 0.5, 0);
          ctx.lineTo(Math.round(ax * hr) + 0.5, scope.bitmapSize.height);
          ctx.stroke();
          ctx.restore();
        }
      }

      if (!this._visible || this._rows.length === 0) return;
      const barW = Math.max(2, timeScale.options().barSpacing * 0.7) * hr;
      const blockH = WALL_BLOCK_H * vr;
      const gap = WALL_GAP * vr;
      ctx.lineWidth = Math.max(1, 1 * hr);

      for (const r of this._rows) {
        if (this._range && (r.time < this._range.from || r.time > this._range.to)) continue;
        const wall = wallSplit(r, this._mode);
        if (wall == null) continue;                 // no book data — draw nothing
        const x = timeScale.timeToCoordinate(r.time);
        if (x === null) continue;
        // off-screen: skip (the close-call blur filter isn't free)
        if (x * hr < -barW - 40 * hr || x * hr > scope.bitmapSize.width + 40 * hr) continue;
        const yHigh = this._series.priceToCoordinate(r.high);
        const yLow = this._series.priceToCoordinate(r.low);
        if (yHigh == null || yLow == null) continue;

        const left = x * hr - barW / 2;
        const askY = yHigh * vr - gap - blockH;     // above the high
        const bidY = yLow * vr + gap;               // below the low

        // close call: both faint outlines. Otherwise the winner is filled,
        // and glows when lopsided on a settled candle.
        const wallStyle = (wins) => (wall.band === 'close' ? 'faint'
          : !wins ? 'outline' : wall.band === 'boost' ? 'boost' : 'fill');
        this._block(ctx, left, askY, barW, blockH, COLOR_ASK_WALL, COLOR_ASK_EDGE, wallStyle(wall.xWins), wall.level, hr);
        this._block(ctx, left, bidY, barW, blockH, COLOR_BID_WALL, COLOR_BID_EDGE, wallStyle(wall.yWins), wall.level, hr);

        // L20 alignment marker: does the 20-level winner match the 5-level
        // verdict currently displayed? One dot above the whole stack — subtle
        // when the deep book agrees, alert-orange when it contradicts. No dot
        // when the shown walls have no verdict (close call, ties included),
        // the deep book ties, or deep coverage is missing.
        if (this._showL20 && r.l20Bid != null && r.l20Ask != null
            && r.l20Bid !== r.l20Ask && wall.band !== 'close') {
          const deepBidWins = r.l20Bid > r.l20Ask;
          ctx.fillStyle = deepBidWins === wall.yWins ? COLOR_L20_AGREE : COLOR_L20_DIVERGE;
          ctx.beginPath();
          const cy = askY - (FLOW_OVERHANG + L20_GAP + L20_RADIUS) * vr;
          ctx.arc(x * hr, cy, L20_RADIUS * vr, 0, Math.PI * 2);
          ctx.fill();
        }

        // taker-flow lines, left of the wall they attack
        const flow = flowSplit(r);
        if (flow) {
          const flowGap = FLOW_LINE_GAP * hr;
          const over = FLOW_OVERHANG * vr;
          const close = flow.band === 'close';
          // winner-only hides the weaker line — a close call has no winner,
          // so both stay (faint), distinguishable from a candle with no trades
          const drawBuy = close || !this._flowWinnerOnly || flow.xWins;
          const drawSell = close || !this._flowWinnerOnly || flow.yWins;
          const stickStyle = (wins) => (close ? 'faint'
            : !wins ? 'thin' : flow.band === 'boost' ? 'boost' : 'thick');
          if (drawBuy) {
            // green buy stick beside the ask block (above the candle)
            this._stick(ctx, left - flowGap, askY - over, blockH + 2 * over, COLOR_BID_WALL, stickStyle(flow.xWins), flow.level, hr);
          }
          if (drawSell) {
            // red sell stick beside the bid block (below the candle)
            this._stick(ctx, left - flowGap, bidY - over, blockH + 2 * over, COLOR_ASK_WALL, stickStyle(flow.yWins), flow.level, hr);
          }
        }
      }
    });
  }

  // flow line as a capsule (fully rounded ends) so it reads as a glow stick,
  // never as another wall box. `right` is the stick's right edge; the width
  // comes from the style: thick / boost (thick + glow) / thin / faint.
  // `level` (0..1) scales the faintness/blur or the glow — see splitBand.
  _stick(ctx, right, y, h, color, style, level, hr) {
    const w = (style === 'thick' || style === 'boost' ? FLOW_THICK : FLOW_THIN) * hr;
    ctx.save();
    if (style === 'faint') this._faint(ctx, level, hr);
    if (style === 'boost') { ctx.shadowColor = color; ctx.shadowBlur = lerp(GLOW_BLUR_MIN, GLOW_BLUR_MAX, level) * hr; }
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(right - w, y, w, h, w / 2);
    ctx.fill();
    ctx.restore();
  }

  // close call: the closer to 50/50, the fainter and blurrier
  _faint(ctx, level, hr) {
    ctx.globalAlpha = lerp(FAINT_ALPHA_MAX, FAINT_ALPHA_MIN, level);
    const blur = FAINT_BLUR_MAX * level * hr;
    if (blur >= 0.25) ctx.filter = `blur(${blur.toFixed(2)}px)`;
  }

  // style: fill (winner) / boost (lopsided winner) / outline (loser) / faint (close call)
  _block(ctx, x, y, w, h, color, edge, style, level, hr) {
    ctx.save();
    if (style === 'fill') {
      ctx.fillStyle = `${color}cc`;
      ctx.fillRect(x, y, w, h);
    } else if (style === 'boost') {
      // the more lopsided, the wider the glow and the heavier the bright edge
      ctx.shadowColor = color;
      ctx.shadowBlur = lerp(GLOW_BLUR_MIN, GLOW_BLUR_MAX, level) * hr;
      ctx.fillStyle = color;
      ctx.fillRect(x, y, w, h);
      ctx.shadowBlur = 0;
      const lw = Math.max(1, Math.round(lerp(1, 2.5, level) * hr));
      ctx.lineWidth = lw;
      ctx.strokeStyle = edge;
      ctx.strokeRect(x + lw / 2, y + lw / 2, w - lw, h - lw);
    } else {
      if (style === 'faint') this._faint(ctx, level, hr);
      ctx.strokeStyle = color;
      ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    }
    ctx.restore();
  }
}

/**
 * Preloaded-trade markers (SOL OI Viz): for each trade a translucent band from
 * entry to exit, a dashed line at each end, an entry arrow at the fill price,
 * an exit dot, and a label with the net result. Pure overlay — the candles and
 * walls underneath are untouched.
 */
const COLOR_TRADE_WIN = '#10b981';
const COLOR_TRADE_LOSS = '#f43f5e';
const COLOR_TRADE_FOCUS = '#fbbf24';
const COLOR_TRADE_ADD = '#38bdf8';
class TradeMarkerPrimitive {
  constructor() {
    this._markers = [];   // {entryT, exitT, entryPx, exitPx, netBps, label, focus, units?: [{t, px, kind, net_bps}], plannedExitT?}
    this._chart = null; this._series = null; this._requestUpdate = null;
    this._paneView = { renderer: () => ({ draw: (target) => this._draw(target) }) };
  }
  attached({ chart, series, requestUpdate }) { this._chart = chart; this._series = series; this._requestUpdate = requestUpdate; }
  detached() { this._chart = null; this._series = null; }
  paneViews() { return [this._paneView]; }
  updateAllViews() {}
  setMarkers(m) { this._markers = m; this._requestUpdate?.(); }
  _draw(target) {
    if (!this._chart || !this._series || this._markers.length === 0) return;
    target.useBitmapCoordinateSpace((scope) => {
      const ctx = scope.context, hr = scope.horizontalPixelRatio, vr = scope.verticalPixelRatio;
      const ts = this._chart.timeScale();
      const W = scope.bitmapSize.width, H = scope.bitmapSize.height;
      for (const m of this._markers) {
        const x0 = ts.timeToCoordinate(m.entryT), x1 = ts.timeToCoordinate(m.exitT);
        if (x0 === null && x1 === null) continue;
        const col = m.netBps >= 0 ? COLOR_TRADE_WIN : COLOR_TRADE_LOSS;
        const X0 = x0 === null ? -1 : x0 * hr, X1 = x1 === null ? W + 1 : x1 * hr;
        ctx.save();
        // holding band
        ctx.fillStyle = `${col}${m.focus ? '22' : '12'}`;
        ctx.fillRect(Math.min(X0, X1), 0, Math.abs(X1 - X0), H);
        // entry / exit verticals
        ctx.setLineDash([5 * vr, 4 * vr]);
        ctx.lineWidth = Math.max(1, (m.focus ? 2 : 1) * hr);
        ctx.strokeStyle = m.focus ? COLOR_TRADE_FOCUS : col;
        for (const X of [X0, X1]) {
          if (X < 0 || X > W) continue;
          ctx.beginPath(); ctx.moveTo(Math.round(X) + 0.5, 0); ctx.lineTo(Math.round(X) + 0.5, H); ctx.stroke();
        }
        ctx.setLineDash([]);
        // entry arrow at the fill price, exit dot at the exit price
        const yIn = this._series.priceToCoordinate(m.entryPx), yOut = this._series.priceToCoordinate(m.exitPx);
        const r = (m.focus ? 7 : 5) * hr;
        if (x0 !== null && yIn != null) {
          ctx.fillStyle = m.focus ? COLOR_TRADE_FOCUS : col;
          ctx.beginPath(); ctx.moveTo(X0, yIn * vr - r); ctx.lineTo(X0 - r, yIn * vr + r); ctx.lineTo(X0 + r, yIn * vr + r); ctx.closePath(); ctx.fill();
        }
        if (x1 !== null && yOut != null) {
          ctx.fillStyle = col; ctx.beginPath(); ctx.arc(X1, yOut * vr, r * 0.8, 0, Math.PI * 2); ctx.fill();
          if (x0 !== null && yIn != null) {   // entry -> exit line
            ctx.strokeStyle = `${col}aa`; ctx.lineWidth = Math.max(1, hr);
            ctx.beginPath(); ctx.moveTo(X0, yIn * vr); ctx.lineTo(X1, yOut * vr); ctx.stroke();
          }
        }
        // extra units (variants): an up-front second unit = a second triangle stacked above the entry arrow;
        // an add = a sky-blue triangle at its own fill time/price with its own line to the exit
        if (m.units && m.units.length > 1) {
          for (const u of m.units) {
            if (u.kind === 'entry') continue;
            const xu = ts.timeToCoordinate(u.t), yu = this._series.priceToCoordinate(u.px);
            if (xu === null || yu == null) continue;
            const XU = xu * hr, YU = yu * vr, ru = r * 0.85;
            if (u.kind === 'upfront') {
              ctx.fillStyle = m.focus ? COLOR_TRADE_FOCUS : col;
              ctx.beginPath(); ctx.moveTo(XU, YU - 2.6 * ru); ctx.lineTo(XU - ru, YU - 0.6 * ru); ctx.lineTo(XU + ru, YU - 0.6 * ru); ctx.closePath(); ctx.fill();
            } else {
              ctx.fillStyle = COLOR_TRADE_ADD;
              ctx.beginPath(); ctx.moveTo(XU, YU - ru); ctx.lineTo(XU - ru, YU + ru); ctx.lineTo(XU + ru, YU + ru); ctx.closePath(); ctx.fill();
              if (x1 !== null && yOut != null) { ctx.strokeStyle = `${COLOR_TRADE_ADD}aa`; ctx.lineWidth = Math.max(1, hr); ctx.beginPath(); ctx.moveTo(XU, YU); ctx.lineTo(X1, yOut * vr); ctx.stroke(); }
              ctx.font = `${Math.round(10 * hr)}px system-ui, sans-serif`; ctx.fillStyle = COLOR_TRADE_ADD; ctx.textBaseline = 'top';
              ctx.fillText(`+1 ${u.net_bps >= 0 ? '+' : ''}${Number(u.net_bps).toFixed(0)}`, XU + 1.3 * ru, YU + 1.2 * ru);
            }
          }
        }
        // extension (variants): a thin dotted line at the planned 60 m exit that was held through
        if (m.plannedExitT) {
          const xp = ts.timeToCoordinate(m.plannedExitT);
          if (xp !== null) {
            const XP = xp * hr;
            ctx.setLineDash([2 * vr, 4 * vr]); ctx.lineWidth = Math.max(1, hr); ctx.strokeStyle = `${m.focus ? COLOR_TRADE_FOCUS : col}99`;
            ctx.beginPath(); ctx.moveTo(Math.round(XP) + 0.5, 0); ctx.lineTo(Math.round(XP) + 0.5, H); ctx.stroke(); ctx.setLineDash([]);
            ctx.font = `${Math.round(10 * hr)}px system-ui, sans-serif`; ctx.fillStyle = m.focus ? COLOR_TRADE_FOCUS : col; ctx.textBaseline = 'top';
            ctx.fillText('60m → held', XP + 3 * hr, 26 * vr);
          }
        }
        // label
        if (x0 !== null) {
          const txt = `${m.label ?? ''} ${m.netBps >= 0 ? '+' : ''}${m.netBps.toFixed(1)} bps`.trim();
          ctx.font = `${Math.round((m.focus ? 12 : 11) * hr)}px system-ui, sans-serif`;
          const tw = ctx.measureText(txt).width + 10 * hr, th = 18 * vr;
          const lx = Math.min(Math.max(X0 + 6 * hr, 2), W - tw - 2), ly = 6 * vr;
          ctx.fillStyle = m.focus ? COLOR_TRADE_FOCUS : col; ctx.globalAlpha = 0.92;
          ctx.beginPath(); ctx.roundRect(lx, ly, tw, th, 4 * hr); ctx.fill();
          ctx.globalAlpha = 1; ctx.fillStyle = '#111827'; ctx.textBaseline = 'middle';
          ctx.fillText(txt, lx + 5 * hr, ly + th / 2);
        }
        ctx.restore();
      }
    });
  }
}

// full-screen key reference (tooltip on the compact bar's "⌨ keys")
const FULLSCREEN_KEYS_HELP = [
  'click — select / deselect a candle',
  '0 — deselect all',
  'q / e — length check back / forward (1 candle selected; again to hide)',
  'b — break out consecutive selection',
  'space — walls: over chart = mark range ends; off chart = hide',
  'v — volume · f / esc — exit full screen',
  'wasd — pan · ← → zoom time · ↑ ↓ zoom price',
  'shift — zoom in · ctrl — zoom out · r — reset view',
].join('\n');

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
function ChartPanel({ id, activeIdRef, isDarkMode, onRemove, removable, nav, setNav, initial, onBreakout, focus = null, markers = null, tradeNav = null }) {
  // {start, end} epoch seconds, or null. Seeded by a breakout spec; a `focus`
  // prop (SOL OI Viz "zip to trade") can re-point it at any time.
  const [fixedWindow, setFixedWindow] = useState(initial?.fixedWindow ?? null);
  const [schema, setSchema] = useState(initial?.schema ?? 'btc');
  const [fullscreen, setFullscreen] = useState(!!initial?.fullscreen);
  const [durKey, setDurKey] = useState('1h');
  const [customVal, setCustomVal] = useState(6);
  const [customUnit, setCustomUnit] = useState('h');
  const [tfS, setTfS] = useState(initial?.tfS ?? 60);

  // ── candle selection / breakout ──────────────────────────────────────────
  // Full screen IS select mode: every click on a candle toggles it.
  const [selection, setSelection] = useState(() => new Set()); // selected candle start times
  const [breakoutPrompt, setBreakoutPrompt] = useState(false);
  const [notice, setNotice] = useState(null);
  // length checks from a single selected candle: {times (nearest first), hitEdge}
  const [lenBack, setLenBack] = useState(null);
  const [lenFwd, setLenFwd] = useState(null);
  const lenRef = useRef({ back: null, fwd: null });   // live values for the once-bound key handler
  lenRef.current = { back: lenBack, fwd: lenFwd };
  const candleDataRef = useRef([]);       // base candle data (with quality colors), logical index == array index
  const selectionRef = useRef(selection);
  selectionRef.current = selection;
  const promptRef = useRef(false);
  promptRef.current = breakoutPrompt;
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
  // space-bar wall range: 1st press remembers the candle under the cursor
  // (anchor), 2nd press remembers another and shows walls ONLY for the candles
  // between them. null range = walls on every candle (the checkbox).
  const [wallRange, setWallRange] = useState(null);    // {from, to} candle times, inclusive
  const [wallAnchor, setWallAnchor] = useState(null);  // candle time of a pending 1st press
  const wallAnchorRef = useRef(null);
  wallAnchorRef.current = wallAnchor;
  const cursorXRef = useRef(null);                     // last cursor x over the chart, px
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
  const tradesRef = useRef(null);         // TradeMarkerPrimitive instance (preloaded trades)
  const candleMapRef = useRef(new Map()); // epoch -> raw candle row (for hover)

  // ── external focus (zip to a trade) and trade markers ──────────────────
  useEffect(() => {
    if (!focus) return;
    if (focus.schema) setSchema(focus.schema);
    if (focus.tfS) setTfS(focus.tfS);
    if (focus.fixedWindow) setFixedWindow({ start: focus.fixedWindow.start, end: focus.fixedWindow.end });
  }, [focus]);
  useEffect(() => { tradesRef.current?.setMarkers(markers ?? []); }, [markers]);

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
      cursorXRef.current = param.point ? param.point.x : null;
      if (!param.time || !param.point) { setHover(null); return; }
      const raw = candleMapRef.current.get(param.time);
      setHover(raw && raw.present ? raw : null);
    });

    const walls = new WallOverlayPrimitive();
    candles.attachPrimitive(walls);
    const trades = new TradeMarkerPrimitive();
    candles.attachPrimitive(trades);

    chartRef.current = chart;
    candleSeriesRef.current = candles;
    volSeriesRef.current = vol;
    wallsRef.current = walls;
    tradesRef.current = trades;
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
        panes: { separatorColor: isDarkMode ? '#374151' : '#e5e7eb', enableResize: !fullscreenRef.current },
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
      wallsRef.current?.setRows(body.candles.filter((c) => c.present).map(toWallRow));
      tradesRef.current?.setMarkers(markers ?? []);
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
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [schema, tfS, durationS, tooMany, tooFew, showL20, fixedWindow]);

  // ── selection rendering: selected candles go blue, length-check runs
  //    purple (back) / turquoise (forward), everything else keeps its quality
  //    color; clearing restores the base data exactly ─────────────────────
  useEffect(() => {
    const series = candleSeriesRef.current;
    const base = candleDataRef.current;
    if (!series || base.length === 0) return;
    const paint = new Map();   // candle time -> color
    for (const [len, col, lastCol] of [
      [lenBack, COLOR_LEN_BACK, COLOR_LEN_BACK_LAST],
      [lenFwd, COLOR_LEN_FWD, COLOR_LEN_FWD_LAST],
    ]) {
      len?.times.forEach((t, k) => paint.set(t, k === len.times.length - 1 ? lastCol : col));
    }
    for (const t of selection) paint.set(t, COLOR_SELECTED);
    if (paint.size === 0) { series.setData(base); return; }
    series.setData(base.map((b) => {
      const col = b.open != null ? paint.get(b.time) : undefined;
      return col ? { ...b, color: col, wickColor: col, borderColor: col } : b;
    }));
  }, [selection, lenBack, lenFwd]);

  // a length check belongs to the selection it was run from
  useEffect(() => { setLenBack(null); setLenFwd(null); }, [selection]);

  // Length check from ONE selected candle: walk back (dir -1) or forward (+1)
  // collecting every candle smaller than it, stopping at the first candle of
  // equal or bigger size. Size = the candle's full visual extent, high − low
  // (color irrelevant). Gaps aren't candles, so they're stepped over.
  const lengthCheck = (dir) => {
    // pressing the key again while its run is displayed hides it
    if (lenRef.current[dir < 0 ? 'back' : 'fwd']) {
      (dir < 0 ? setLenBack : setLenFwd)(null);
      return;
    }
    const sel = selectionRef.current;
    if (sel.size !== 1) {
      setNotice('Select exactly one candle, then press q (back) or e (forward)');
      return;
    }
    const base = candleDataRef.current;
    const i0 = base.findIndex((b) => b.time === [...sel][0]);
    if (i0 < 0) return;
    const size0 = base[i0].high - base[i0].low;
    const times = [];
    let hitEdge = true;
    for (let i = i0 + dir; i >= 0 && i < base.length; i += dir) {
      const b = base[i];
      if (b.open == null) continue;
      if (b.high - b.low >= size0) { hitEdge = false; break; }
      times.push(b.time);
    }
    (dir < 0 ? setLenBack : setLenFwd)({ times, hitEdge });
  };

  // leaving full screen ends selecting: clear the highlight and any prompt
  useEffect(() => {
    if (!fullscreen) { setSelection(new Set()); setBreakoutPrompt(false); }
  }, [fullscreen]);

  // Full screen: the mouse ONLY selects. No drag / wheel pan or zoom, no axis
  // drag-scaling or double-click reset, no pane-separator resizing — the view
  // is driven by the keyboard alone. Outside full screen the mouse works as usual.
  useEffect(() => {
    const m = !fullscreen;
    chartRef.current?.applyOptions({
      handleScroll: { mouseWheel: m, pressedMouseMove: m, horzTouchDrag: m, vertTouchDrag: m },
      handleScale: {
        mouseWheel: m, pinch: m,
        axisPressedMouseMove: { time: m, price: m },
        axisDoubleClickReset: { time: m, price: m },
      },
      layout: { panes: { enableResize: m } },
    });
  }, [fullscreen]);

  useEffect(() => {
    if (!notice) return undefined;
    const t = setTimeout(() => setNotice(null), 2500);
    return () => clearTimeout(t);
  }, [notice]);

  // click-to-select (full screen only): each click toggles the ONE candle under
  // the cursor. subscribeClick only fires for a real click, never at the end of
  // a drag, so dragging still pans the chart. Gaps can't be selected.
  const fullscreenRef = useRef(fullscreen);
  fullscreenRef.current = fullscreen;
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return undefined;
    const onClick = (param) => {
      if (!fullscreenRef.current || param.time == null) return;
      const raw = candleMapRef.current.get(param.time);
      if (!raw || !raw.present) return;
      setSelection((prev) => {
        const next = new Set(prev);
        if (next.has(param.time)) next.delete(param.time); else next.add(param.time);
        return next;
      });
    };
    chart.subscribeClick(onClick);
    return () => { try { chart.unsubscribeClick(onClick); } catch { /* chart already removed */ } };
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
  useEffect(() => { wallsRef.current?.setRange(wallRange); }, [wallRange]);
  useEffect(() => { wallsRef.current?.setAnchor(wallAnchor); }, [wallAnchor]);
  // a different asset / candle size / window makes the remembered candles meaningless
  useEffect(() => { setWallRange(null); setWallAnchor(null); }, [schema, tfS, durationS]);

  // the candle vertically aligned with the cursor, read from the cursor's x at
  // key time (so a wasd pan under a still mouse is accounted for). Clamped to
  // the loaded candles; null when the cursor is off the chart.
  const candleTimeAtCursor = () => {
    const x = cursorXRef.current;
    const base = candleDataRef.current;
    if (x == null || base.length === 0) return null;
    const l = chartRef.current?.timeScale().coordinateToLogical(x);
    if (l == null) return null;
    const i = Math.max(0, Math.min(base.length - 1, Math.round(l)));
    return base[i].time;
  };

  // ── hotkeys (only for the panel the mouse last entered) ────────────────
  //   v      toggle volume pane
  //   space  1st press remembers the candle under the cursor, 2nd press the
  //          other end: walls show ONLY for that candle range, with the
  //          preset (winning flow only, L20 off, avg of 5s closes).
  //          Cursor off the chart: space hides all walls.
  //   f      toggle fullscreen
  //   esc    leave fullscreen
  //   b      (fullscreen) break out the clicked-selected candles
  //   q / e  (fullscreen, 1 candle selected) length check back / forward;
  //          press again to hide that run
  //   0      (fullscreen) deselect all candles
  //   shift  (fullscreen, held) zoom in  = ↑ + →
  //   ctrl   (fullscreen, held) zoom out = ← + ↓
  const showWallsRef = useRef(showWalls);
  showWallsRef.current = showWalls;
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
  const NAV_KEYS = new Set(['w', 'a', 's', 'd', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Shift', 'Control']);
  // combo keys: shift = zoom in (↑ + →), ctrl = zoom out (← + ↓)
  const NAV_COMBOS = { Shift: ['ArrowUp', 'ArrowRight'], Control: ['ArrowLeft', 'ArrowDown'] };
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
    // expand combos into their arrows; the Set dedupes, so holding shift + ↑
    // together zooms at the normal speed, not double
    const keys = new Set();
    for (const k of heldRef.current) for (const x of NAV_COMBOS[k] ?? [k]) keys.add(x);
    for (const k of keys) navigateBy(k, dt);
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
      // breakout prompt open: digits pick a size, esc closes, nothing else leaks through
      if (promptRef.current) {
        e.preventDefault();
        e.stopPropagation();
        if (e.key === 'Escape') setBreakoutPrompt(false);
        else if (/^[1-9]$/.test(e.key)) {
          const opt = breakoutOptions()[Number(e.key) - 1];
          if (opt) doBreakout(opt.s);
        }
        return;
      }
      const navKey = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      if (fullscreenRef.current) {
        // Full screen: every keybind belongs to the chart, whatever has focus.
        // A dropdown / button keeps focus after it's used, and would otherwise
        // swallow the key (arrows change the select, space presses the button).
        // Capture phase + blur takes the key before any control can act on it;
        // non-keybind keys (digits, backspace…) still reach the number inputs.
        const isBind = e.key === ' ' || e.key === 'Escape'
          || NAV_KEYS.has(navKey) || navKey === 'b' || navKey === 'r' || navKey === 'v' || navKey === 'f'
          || navKey === 'q' || navKey === 'e' || e.key === '0';
        if (!isBind) return;
        e.preventDefault();
        e.stopPropagation();
        const focused = document.activeElement;
        if (focused && focused !== document.body) focused.blur();
      } else {
        const tag = e.target?.tagName;
        if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      }
      if (fullscreenRef.current) {
        if (navKey === 'b') { requestBreakout(); return; }
        if (navKey === 'q') { if (!e.repeat) lengthCheck(-1); return; }
        if (navKey === 'e') { if (!e.repeat) lengthCheck(1); return; }
        // deselect everything (length-check runs clear with the selection)
        if (e.key === '0') { setSelection(new Set()); return; }
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
        if (e.repeat) return;
        const t = candleTimeAtCursor();
        if (t == null) {
          // cursor off the chart: space HIDES — walls, range and any pending anchor
          setShowWalls(false);
          setWallRange(null);
          setWallAnchor(null);
          return;
        }
        if (wallAnchorRef.current == null) {
          // 1st press: remember this candle; any walls showing are cleared
          setWallAnchor(t);
          setShowWalls(false);
          setWallRange(null);
        } else {
          // 2nd press: walls for exactly the candles between the two presses,
          // with the preset: winning flow only, L20 off, avg of 5s closes
          const a = wallAnchorRef.current;
          setWallRange({ from: Math.min(a, t), to: Math.max(a, t) });
          setWallAnchor(null);
          setFlowWinnerOnly(true);
          setShowL20(false);
          setWallMode('avgc');
          setShowWalls(true);
        }
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
    // capture phase: runs before the focused control (and React's handlers) see the key
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('keyup', onKeyUp, true);
    window.addEventListener('blur', releaseAll);   // a missed keyup must never leave a key "stuck"
    return () => {
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('keyup', onKeyUp, true);
      window.removeEventListener('blur', releaseAll);
      releaseAll();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [id, activeIdRef]);

  // the full-screen panel owns the keyboard (its overlay covers every other
  // panel, so no mouseenter could hand keys elsewhere); leaving releases held keys
  useEffect(() => {
    if (fullscreen) activeIdRef.current = id;
    else releaseAll();
  }, [fullscreen]);

  useEffect(() => { wallsRef.current?.setMode(wallMode); }, [wallMode]);

  // how often each decisiveness band shows up in the loaded window (or the
  // space-bar range) — read-only context for the fixed cuts, never used to
  // move them. Walls: candles with book data; flow: candles that traded.
  const bandCounts = useMemo(() => {
    if (!stats?.candles) return null;
    const n = { wall: 0, wallClose: 0, wallBoost: 0, flow: 0, flowClose: 0, flowBoost: 0 };
    for (const c of stats.candles) {
      if (!c.present) continue;
      if (wallRange && (c.t < wallRange.from || c.t > wallRange.to)) continue;
      const r = toWallRow(c);
      const w = wallSplit(r, wallMode);
      if (w) { n.wall++; if (w.band === 'close') n.wallClose++; if (w.band === 'boost') n.wallBoost++; }
      const f = flowSplit(r);
      if (f) { n.flow++; if (f.band === 'close') n.flowClose++; if (f.band === 'boost') n.flowBoost++; }
    }
    return n;
  }, [stats, wallMode, wallRange]);

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
  // full screen = compact chrome: the chart should own nearly the whole viewport
  const selectCls = `${fullscreen ? 'text-xs rounded px-1.5 py-0.5' : 'text-sm rounded-lg px-3 py-2'} font-medium border focus:ring-2 focus:ring-blue-500 focus:outline-none transition-colors duration-200 ${
    isDarkMode ? 'bg-gray-900 border-gray-700 text-gray-200' : 'bg-white border-gray-300 text-gray-800'
  }`;
  const dim = isDarkMode ? 'text-gray-400' : 'text-gray-500';

  const hoverChange = hover && hover.o ? ((hover.c - hover.o) / hover.o) * 10000 : null;

  return (
    <div
      onMouseEnter={() => { activeIdRef.current = id; }}
      className={fullscreen
        ? `fixed inset-0 z-50 flex flex-col gap-1 p-1 overflow-hidden ${isDarkMode ? 'bg-gray-900' : 'bg-gray-50'}`
        : 'space-y-4'}
      // the parent's space-y margin would shrink the fixed overlay off the bottom
      style={fullscreen ? { margin: 0 } : undefined}
    >
      {/* ── control bar ── */}
      <div className={`rounded-lg border transition-colors duration-200 ${card} ${fullscreen ? 'px-2 py-1' : 'p-4'}`}>
        {/* full screen: one line that never wraps, so the chart height is constant */}
        <div className={`flex items-center ${fullscreen ? 'flex-nowrap gap-2 overflow-hidden whitespace-nowrap' : 'flex-wrap gap-3'}`}>
          {/* asset */}
          <div className="flex items-center gap-2">
            <span className={`${fullscreen ? 'text-base' : 'text-2xl'} font-black tracking-tight ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              {SCHEMAS.find((s) => s.key === schema)?.label}
            </span>
            <select value={schema} onChange={(e) => setSchema(e.target.value)} className={selectCls}>
              {SCHEMAS.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
            </select>
          </div>

          {/* duration — or, for a breakout panel, the fixed window it covers */}
          {fixedWindow ? (
            <span
              className={`${fullscreen ? 'text-xs rounded px-1.5 py-0.5' : 'text-sm rounded-lg px-3 py-2'} font-medium border ${isDarkMode ? 'bg-gray-900/60 border-gray-700 text-gray-300' : 'bg-gray-50 border-gray-300 text-gray-700'}`}
              title={initial?.from
                ? `Breakout of ${initial.from.count ?? '?'} × ${sizeLabel(initial.from.tfS)} candle(s)`
                : tradeNav ? 'Window around the focused trade' : 'Fixed window'}
            >
              ⤵ {fmtUtc(fixedWindow.start).slice(5, 16)} → {fmtUtc(fixedWindow.end).slice(11, 16)} UTC
              {initial?.from && (
                <span className={`ml-1.5 text-xs ${dim}`}>({initial.from.count} × {sizeLabel(initial.from.tfS)})</span>
              )}
            </span>
          ) : (
            <select value={durKey} onChange={(e) => setDurKey(e.target.value)} className={selectCls}>
              {DURATION_PRESETS.map((d) => <option key={d.key} value={d.key}>{d.label}</option>)}
            </select>
          )}

          {/* preloaded-trade stepper (SOL OI Viz): previous / next trade, in both layouts */}
          {tradeNav && tradeNav.count > 0 && (
            <span className="flex items-center gap-1 min-w-0">
              <button
                onClick={tradeNav.onPrev}
                title="Previous trade ( [ )"
                className={`${fullscreen ? 'px-2 py-0.5 text-xs rounded' : 'px-2.5 py-1 text-sm rounded-lg'} font-bold transition-colors ${
                  isDarkMode ? 'bg-amber-500 text-gray-900 hover:bg-amber-400' : 'bg-amber-500 text-white hover:bg-amber-600'}`}
              >◀</button>
              <span className={`${fullscreen ? 'text-xs' : 'text-sm'} font-mono font-semibold ${isDarkMode ? 'text-amber-300' : 'text-amber-700'}`}>
                trade {tradeNav.index + 1}/{tradeNav.count}{tradeNav.label ? ` · ${tradeNav.label}` : ''}
              </span>
              <button
                onClick={tradeNav.onNext}
                title="Next trade ( ] )"
                className={`${fullscreen ? 'px-2 py-0.5 text-xs rounded' : 'px-2.5 py-1 text-sm rounded-lg'} font-bold transition-colors ${
                  isDarkMode ? 'bg-amber-500 text-gray-900 hover:bg-amber-400' : 'bg-amber-500 text-white hover:bg-amber-600'}`}
              >▶</button>
            </span>
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
            {!fullscreen && <span className={`text-sm ${dim}`}>candle size</span>}
            <select value={tfS} onChange={(e) => setTfS(Number(e.target.value))} className={selectCls}>
              {CANDLE_SIZES.map((c) => <option key={c.s} value={c.s}>{c.label}</option>)}
            </select>
          </div>

          <button
            onClick={load}
            disabled={loading || tooMany || tooFew}
            className={`${fullscreen ? 'px-2 py-0.5 text-xs rounded' : 'px-4 py-2 text-sm rounded-lg'} font-medium transition-all duration-150 ${
              loading
                ? 'opacity-50 cursor-wait'
                : ''
            } ${isDarkMode ? 'bg-blue-600 hover:bg-blue-500 text-white' : 'bg-blue-600 hover:bg-blue-700 text-white'} disabled:opacity-40`}
          >
            {loading ? 'Loading…' : '↻ Refresh'}
          </button>

          {/* full screen: selection / length-check readout (was a strip over the chart);
              "+" = reached the edge of the loaded candles without meeting an
              equal-or-bigger one */}
          {fullscreen && (
            <span className="flex items-center gap-2 text-xs font-semibold min-w-0">
              <span style={{ color: COLOR_SELECTED }}>🎯 {selection.size} selected</span>
              {lenBack && (
                <span style={{ color: COLOR_LEN_BACK }}
                  title={lenBack.hitEdge ? 'Reached the start of the loaded candles before any equal-or-bigger candle' : undefined}>
                  ◀ {lenBack.times.length}{lenBack.hitEdge ? '+' : ''} smaller back
                </span>
              )}
              {lenFwd && (
                <span style={{ color: COLOR_LEN_FWD }}
                  title={lenFwd.hitEdge ? 'Reached the end of the loaded candles before any equal-or-bigger candle' : undefined}>
                  {lenFwd.times.length}{lenFwd.hitEdge ? '+' : ''} smaller forward ▶
                </span>
              )}
            </span>
          )}

          <span className={`text-xs ${dim} ml-auto`}>
            {estCandles > 0 && `≈ ${estCandles.toLocaleString()} candles`}
          </span>

          {fullscreen && (
            <span className={`text-xs cursor-help ${dim}`} title={FULLSCREEN_KEYS_HELP}>
              ⌨ keys
            </span>
          )}

          <button
            onClick={() => setFullscreen((f) => !f)}
            title={fullscreen ? 'Exit full screen (f / esc)' : 'Full screen (f)'}
            className={`${fullscreen ? 'px-2 py-0.5 text-xs rounded' : 'px-2.5 py-1 text-sm rounded-lg'} font-medium transition-colors duration-150 ${
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

        {/* chart area — its own positioning box so the readout sits on the
            chart, below the full-screen selection strip; crosshair cursor in
            full screen because every click there selects a candle */}
        <div className={`relative ${fullscreen ? 'bxtviz-select-surface flex-1 min-h-0 flex flex-col' : ''}`}>
        {/* !important beats the resize cursors lightweight-charts sets inline
            on the axes and pane separator */}
        {fullscreen && <style>{'.bxtviz-select-surface, .bxtviz-select-surface * { cursor: crosshair !important; }'}</style>}
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
              {showWalls && (!wallRange || (hover.t >= wallRange.from && hover.t <= wallRange.to)) && (() => {
                const row = toWallRow(hover);
                const [a, b] = wallPair(row, wallMode);
                const w = wallSplit(row, wallMode);
                if (w == null) return <span className={dim}> · book n/a</span>;
                const modeTag = wallMode === 'close' ? 'close' : wallMode === 'avgc' ? 'avg-of-closes' : 'avg';
                const buyV = hover.buy_v || 0;
                const sellV = hover.sell_v || 0;
                const f = flowSplit(row);
                const prints = (hover.buy_n ?? 0) + (hover.sell_n ?? 0);
                // winner/loser split, floored so the label never crosses a cut
                // the drawing didn't (59.9% reads 59/41, not 60/40)
                const splitTag = (s, winColor, whyNoGlow) => {
                  const p = Math.floor(s.share * 100 + 1e-9);
                  if (s.band === 'close') return <span className={dim}> close {p}/{100 - p}</span>;
                  return (
                    <span>
                      {' '}<span style={{ color: winColor, fontWeight: s.band === 'boost' ? 700 : 400 }}>
                        {p}/{100 - p}{s.band === 'boost' ? ' ★' : ''}
                      </span>
                      {s.lopsided && s.band !== 'boost' && <span className={dim}> (no ★: {whyNoGlow})</span>}
                    </span>
                  );
                };
                const unsettled = isSettled(row) ? null : 'unsettled candle';
                return (
                  <span>
                    {' '}· <span style={{ color: COLOR_BID_WALL, fontWeight: w.yWins ? 700 : 400 }}>bid {b.toFixed(3)}</span>
                    {' / '}
                    <span style={{ color: COLOR_ASK_WALL, fontWeight: w.xWins ? 700 : 400 }}>ask {a.toFixed(3)}</span>
                    <span className={dim}> {modeTag} coins</span>
                    {splitTag(w, w.yWins ? COLOR_BID_WALL : COLOR_ASK_WALL, unsettled)}
                    {' '}· <span style={{ color: COLOR_BID_WALL, fontWeight: f?.xWins ? 700 : 400 }}>buy {buyV.toFixed(3)}</span>
                    {' / '}
                    <span style={{ color: COLOR_ASK_WALL, fontWeight: f?.yWins ? 700 : 400 }}>sell {sellV.toFixed(3)}</span>
                    <span className={dim}> flow · {prints} print{prints === 1 ? '' : 's'}</span>
                    {f && splitTag(f, f.xWins ? COLOR_BID_WALL : COLOR_ASK_WALL,
                      unsettled ?? `< ${FLOW_MIN_TRADES} prints`)}
                    {showL20 && (hover.l20_bid != null && hover.l20_ask != null ? (
                      <span>
                        {' '}· L20 bid {hover.l20_bid.toFixed(3)} / ask {hover.l20_ask.toFixed(3)}{' '}
                        {w.band === 'close'
                          ? <span className={dim}>no verdict (walls close)</span>
                          : hover.l20_bid === hover.l20_ask
                            ? <span className={dim}>deep tie</span>
                            : (hover.l20_bid > hover.l20_ask) === w.yWins
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
              {hover.mac_revised && <span className="text-sky-500"> · mac-revised</span>}
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
        </div>

        {/* overlay toggles */}
        <div className={`flex items-center border-t transition-colors duration-200 ${
          fullscreen ? 'flex-nowrap gap-x-3 px-2 py-0.5 text-xs overflow-hidden whitespace-nowrap' : 'flex-wrap gap-x-5 gap-y-1 px-3 py-2 text-sm'
        } ${
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
            {!fullscreen && <span className={`text-xs ${dim}`}>(buy + sell coins per candle, lower pane)</span>}
          </label>
          <label className={`flex items-center gap-2 cursor-pointer select-none ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            <input
              type="checkbox"
              checked={showWalls}
              onChange={(e) => { setShowWalls(e.target.checked); setWallRange(null); setWallAnchor(null); }}
              className="accent-teal-500"
            />
            Bid/Ask walls <kbd className={`px-1 rounded text-[10px] font-mono ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>space</kbd>
            {wallAnchor != null ? (
              <span className="text-xs font-medium" style={{ color: COLOR_WALL_ANCHOR }}>
                from {fmtUtc(wallAnchor).slice(11, 19)} — hover the other end, press space
              </span>
            ) : showWalls && wallRange ? (
              <span className="text-xs font-medium" style={{ color: COLOR_WALL_ANCHOR }}>
                {fmtUtc(wallRange.from).slice(11, 19)} → {fmtUtc(wallRange.to).slice(11, 19)} only
              </span>
            ) : null}
            {!fullscreen && <span className={`text-xs ${dim}`}>
              (<span style={{ color: COLOR_BID_WALL }}>bid</span> below · <span style={{ color: COLOR_ASK_WALL }}>ask</span> above — bigger side by coin size is filled ·
              flow lines beside each block: <span style={{ color: COLOR_BID_WALL }}>buy</span>/<span style={{ color: COLOR_ASK_WALL }}>sell</span>, thicker = stronger ·{' '}
              <span title={`Winner's share of the pair. Under ${SHARE_CLOSE * 100}% (closer than 60/40, ties included): no winner is drawn — both sides faint, and the closer to 50/50 the fainter and blurrier. ${SHARE_LOPSIDED * 100}% or more (3-to-1): the winner glows, barely at 3:1 and brightest at 100/0 — but only on a clean, settled candle, and for flow only with at least ${FLOW_MIN_TRADES} prints. Same cuts and scaling on every coin, candle size and wall mode.`}
                className="underline decoration-dotted cursor-help">
                blurrier = closer (&lt;60/40) · brighter glow ★ = more lopsided (≥3:1)
              </span>)
            </span>}
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
              {!fullscreen && <span className={`text-xs ${dim}`}>(hide the weaker flow line)</span>}
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
              {!fullscreen && <span className={`text-xs ${dim}`}>
                (dot per candle: <span style={{ color: COLOR_L20_AGREE }}>●</span> 20-level book agrees with shown walls · <span style={{ color: COLOR_L20_DIVERGE }}>●</span> diverges)
              </span>}
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
                    className={`${fullscreen ? 'px-1.5 py-0' : 'px-2.5 py-1'} text-xs font-medium rounded-md transition-all duration-150 ${
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
              {!fullscreen && <span className={`text-xs ${dim}`}>
                {WALL_MODES.find((m) => m.key === wallMode)?.desc}
              </span>}
            </div>
          )}

          {/* keyboard navigation sensitivity (fullscreen) */}
          <div className={`flex items-center gap-2 ml-auto text-xs ${dim}`} title="Full screen only: hold wasd to pan, ←→ to zoom time, ↑↓ to zoom price, shift to zoom in (↑+→), ctrl to zoom out (←+↓); r resets the view. Speeds are % of the visible range per second held.">
            <span>{fullscreen ? 'speed' : '⌨ full-screen nav speed:'}</span>
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
            {!fullscreen && <span>per second held · <kbd className={`px-1 rounded text-[10px] font-mono ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>r</kbd> reset · <kbd className={`px-1 rounded text-[10px] font-mono ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>shift</kbd> zoom in · <kbd className={`px-1 rounded text-[10px] font-mono ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>ctrl</kbd> zoom out · <kbd className={`px-1 rounded text-[10px] font-mono ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>click</kbd> select · <kbd className={`px-1 rounded text-[10px] font-mono ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>b</kbd> break out</span>}
          </div>
          {/* full screen: the stats card is hidden — its essentials live here */}
          {fullscreen && stats && (
            <span className={`text-xs ${dim}`}>
              <b className={isDarkMode ? 'text-gray-200' : 'text-gray-800'}>{stats.actual.toLocaleString()}</b>/{stats.expected.toLocaleString()}
              {stats.missing > 0 && <span className="text-red-500"> · {stats.missing.toLocaleString()} missing</span>}
              {' '}· ideal {stats.actual ? Math.round((100 * stats.ideal) / stats.actual) : 0}%
              {stats.revised > 0 && <span className="text-amber-500"> · {stats.revised} revised</span>}
              {stats.mac_revised > 0 && <span className="text-sky-500"> · {stats.mac_revised} mac-revised</span>}
            </span>
          )}
        </div>
      </div>

      {/* ── stats + legend strip ── */}
      {stats && !fullscreen && (
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
          {stats.mac_revised > 0 && <span className="text-sky-500" title="A majority of the candle's buckets include or were changed by Mac-replica fills. Informational only; not a quality gate.">{stats.mac_revised} mac-revised</span>}
          {showWalls && bandCounts && (() => {
            const pct = (k, d) => (d ? Math.round((100 * k) / d) : 0);
            const modeTag = WALL_MODES.find((m) => m.key === wallMode)?.label.toLowerCase();
            return (
              <span title={`Share of candles in each band${wallRange ? ' (inside the space-bar range)' : ''}. Close = winner under ${SHARE_CLOSE * 100}% (drawn faint); ★ = ${SHARE_LOPSIDED * 100}%+ on a clean candle (glows). Flow counts only candles that traded.`}>
                walls <span className={dim}>({modeTag})</span>{' '}
                close <b className={isDarkMode ? 'text-gray-200' : 'text-gray-800'}>{pct(bandCounts.wallClose, bandCounts.wall)}%</b>
                {' '}· ★ <b className={isDarkMode ? 'text-gray-200' : 'text-gray-800'}>{pct(bandCounts.wallBoost, bandCounts.wall)}%</b>
                {'  '}· flow close <b className={isDarkMode ? 'text-gray-200' : 'text-gray-800'}>{pct(bandCounts.flowClose, bandCounts.flow)}%</b>
                {' '}· ★ <b className={isDarkMode ? 'text-gray-200' : 'text-gray-800'}>{pct(bandCounts.flowBoost, bandCounts.flow)}%</b>
              </span>
            );
          })()}
          <span>{fmtUtc(new Date(stats.grid_start).getTime() / 1000)} → {fmtUtc(new Date(stats.grid_end).getTime() / 1000)} UTC</span>
          {loadMs != null && <span>{loadMs} ms</span>}
          <span className="ml-auto flex items-center gap-3">
            <span className="flex items-center gap-1" title="Passed every quality gate: enough polls, evenly sampled, contiguous, never revised.">
              <i className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: COLOR_UP }} />/<i className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: COLOR_DOWN }} /> clean
            </span>
            <span className="flex items-center gap-1" title="Bucket exists but its sampling failed a gate — values built from thinner data.">
              <i className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: COLOR_DEGRADED }} /> degraded
            </span>
            <span className="flex items-center gap-1" title="A majority of the candle's buckets were corrected after the fact from PC-observed data — a live system may have seen different values. Mac-replica fills are counted separately as mac-revised.">
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
      {stats && !fullscreen && showQualityHelp && (
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
export default function BxtVisualizer({ isDarkMode, initialPanel = null, focus = null, markers = null, tradeNav = null, hideAddPanel = false }) {
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
          focus={i === 0 ? focus : null}
          markers={i === 0 ? markers : null}
          tradeNav={i === 0 ? tradeNav : null}
        />
      ))}

      {!hideAddPanel && <button
        onClick={addPanel}
        className={`w-full py-3 rounded-lg border-2 border-dashed text-sm font-medium transition-colors duration-150 ${
          isDarkMode
            ? 'border-gray-700 text-gray-500 hover:border-teal-500/60 hover:text-teal-400 hover:bg-gray-800/50'
            : 'border-gray-300 text-gray-400 hover:border-teal-500/60 hover:text-teal-600 hover:bg-teal-50/50'
        }`}
      >
        ＋ Add comparison chart
      </button>}
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
