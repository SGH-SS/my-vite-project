/**
 * DERIV BXT DASHBOARD — bucket.jsx
 * ============================================================================
 * Standalone explainer + structural map for the pre-computed 5s DERIV Signal
 * Buckets (`<coin>."5s_bxt"`, written live by sol-perp/bxtbuilder_5s.py for
 * btc / sol / eth / spx). Synced to map.md (150 components + quality flags).
 *
 * Four sections:
 *   1. VECTORS      — every vector is a clickable card; expand to see its
 *                     component columns across State 0 / 1 / 2 with formulas.
 *   2. DERIV ENGINE — how State 0 → 1 → 2 (derivatives) + replenishment work,
 *                     including contiguity gating and NULL propagation.
 *   3. STACKING     — how to roll 5s buckets up into 10s / 15s / 1m / 5m / 1h,
 *                     column-family by column-family (sum vs mean vs OHLC vs
 *                     recompute-derivative).
 *   4. STRATEGY     — the empirical findings from the 44-day IC/backtest study
 *                     (map.md §7–§12): coin verdicts, quiet gate, flip exit,
 *                     and the round-4 passive maker-fill reality check.
 *
 * Mostly presentational (coverage calendar fetches the bxt API). Lives next to
 * StratHub.jsx and is mounted via the `bxt` dashboard mode in TradingDashboard.jsx.
 */

import { useState, useEffect, useMemo, useCallback } from 'react';

// ── theme (mirrors StratHub) ───────────────────────────────────────────────
const C = {
  bg: '#060c18', surface: '#0c1628', raised: '#101e35', border: '#1a2d4e',
  text: '#e2e8f0', sub: '#94a3b8', muted: '#475569', dim: '#64748b',
  green: '#00d4a8', red: '#f43f5e', amber: '#eab308',
  blue: '#60a5fa', purple: '#a78bfa', cyan: '#22d3ee',
  orange: '#e97316', pink: '#ec4899', emerald: '#10b981', fuchsia: '#d946ef',
};

const STATE_COLOR = { 0: C.blue, 1: C.amber, 2: C.fuchsia };
const STATE_LABEL = {
  0: 'State 0 · raw level',
  1: 'State 1 · first derivative (Δ / s)',
  2: 'State 2 · second derivative (Δ² / s)',
};

// ── vector + component model (baked from 5s_buckets.md / feature_builder) ───
// Each component: { col, type, agg, blurb }
//   agg = how it aggregates across L2 polls inside a 5s bucket
const VECTORS = [
  {
    id: 'meta', tag: 'Q', name: 'Quality layer', color: C.dim,
    summary: 'Read this before any query — the flags that separate pristine buckets from partial ones.',
    born: 0, hasDeriv: false,
    components: [
      { col: 'poll_count', type: 'int', agg: 'count of polls', blurb: 'L2 snapshots captured in the window (~9 normal). Catches the June 15–24 feed throttle (fell to ~0.9). Also the weight for MEAN-stacking.' },
      { col: 'prev_contiguous', type: 'bool', agg: 'flag', blurb: 'Previous 5s bucket exists. Gates ALL State 1/2 columns (NULL when false).' },
      { col: 'poll_gap_max_ms', type: 'int', agg: 'max intra-bucket gap', blurb: 'Largest gap between polls incl. edges — catches 9 polls bunched into 1 second.' },
      { col: 'has_trades', type: 'bool', agg: 'flag', blurb: 'Any fill in the bucket. 17.5% of SOL / 33.8% of SPX 5s buckets have none — gate flow features on it.' },
      { col: 'was_revised', type: 'bool', agg: 'flag', blurb: 'Stored values changed after first write (late raw). Treat as a control, not a filter — revisions cluster in volatile moments.' },
      { col: 'bucket_ideal', type: 'bool (view)', agg: 'composite', blurb: 'poll_count≥7 AND prev_contiguous AND poll_gap_max_ms<1500 AND NOT was_revised. Query the bxt_5s view to get it for free.' },
    ],
  },
  {
    id: 'x1', tag: 'X₁', name: 'Ask Book', color: C.red,
    summary: 'Resting sell-side liquidity: per-poll means (posture) plus close-of-bucket state and cost-to-consume.',
    born: 0, hasDeriv: true,
    components: [
      { col: 'ask_size', type: 'float', agg: 'mean across polls', blurb: 'Σ size over top 5 ask levels (coins). Posture over the bucket.' },
      { col: 'ask_size_c', type: 'float', agg: 'last poll', blurb: 'Close-of-bucket ask depth — the boundary state. Only closes telescope, so replenishment accounting runs on _c. Its imbalance vs bid_size_c was the strongest predictor in the whole study.' },
      { col: 'ask_size_usd / ask_size_c_usd', type: 'float', agg: 'mean / last', blurb: 'USD-normalized variants — use for anything cross-coin.' },
      { col: 'ask_ppl', type: 'float', agg: 'mean across polls', blurb: 'Σ order count over all levels — queue density. ppl imbalance validated as a strong niche signal (round 2).' },
      { col: 'ask_fill', type: 'float 0–1', agg: 'mean across polls', blurb: 'Occupied tick slots / possible. Saturates at 1.0 on tight coins (90% of ETH) — prefer impact_bps.' },
      { col: 'ask_centroid', type: 'float 0–1', agg: 'mean across polls', blurb: 'Size-weighted distance from best. 0 = front-loaded. centroid_imb validated (round 2).' },
      { col: 'ask_impact_bps', type: 'float', agg: 'mean across polls', blurb: 'Size-weighted cost in bps to consume all 5 levels. Uncapped, tick-independent, doubles as slippage model. impact_imb was the #2 predictor.' },
      { col: 'ask_span_ticks', type: 'float', agg: 'mean across polls', blurb: 'Tick distance best→farthest level. Stretched side is thin (span_imb validated).' },
    ],
  },
  {
    id: 'x2', tag: 'X₂', name: 'Buyers (taker)', color: C.green,
    summary: 'Aggressive buy flow that lifted the ask inside the window.',
    born: 0, hasDeriv: true,
    components: [
      { col: 'buy_volume', type: 'float', agg: 'sum over trades', blurb: 'Σ size of side=B prints (coins).' },
      { col: 'buy_volume_usd', type: 'float', agg: 'sum over trades', blurb: 'Σ (size × execution px) per buy print — USD notional.' },
      { col: 'buy_count', type: 'int', agg: 'count of trades', blurb: 'Number of buy prints.' },
    ],
  },
  {
    id: 'x3', tag: 'X₃', name: 'Sellers (taker)', color: C.orange,
    summary: 'Aggressive sell flow that hit the bid inside the window.',
    born: 0, hasDeriv: true,
    components: [
      { col: 'sell_volume', type: 'float', agg: 'sum over trades', blurb: 'Σ size of side=A prints (coins).' },
      { col: 'sell_volume_usd', type: 'float', agg: 'sum over trades', blurb: 'Σ (size × execution px) per sell print — USD notional.' },
      { col: 'sell_count', type: 'int', agg: 'count of trades', blurb: 'Number of sell prints.' },
    ],
  },
  {
    id: 'x4', tag: 'X₄', name: 'Bid Book', color: C.blue,
    summary: 'Resting buy-side liquidity — same component set as the ask side.',
    born: 0, hasDeriv: true,
    components: [
      { col: 'bid_size', type: 'float', agg: 'mean across polls', blurb: 'Σ size over top 5 bid levels (coins).' },
      { col: 'bid_size_c', type: 'float', agg: 'last poll', blurb: 'Close-of-bucket bid depth. book_imb_c = (bid_size_c − ask_size_c)/(sum) — 5s IC +0.153 BTC, the single strongest feature.' },
      { col: 'bid_size_usd / bid_size_c_usd', type: 'float', agg: 'mean / last', blurb: 'USD variants for cross-coin work.' },
      { col: 'bid_ppl', type: 'float', agg: 'mean across polls', blurb: 'Σ order count over all bid levels.' },
      { col: 'bid_fill', type: 'float 0–1', agg: 'mean across polls', blurb: 'Tick occupancy of the bid stack (saturates on tight coins).' },
      { col: 'bid_centroid', type: 'float 0–1', agg: 'mean across polls', blurb: '0 = front-loaded near best, 1 = back-loaded deep.' },
      { col: 'bid_impact_bps', type: 'float', agg: 'mean across polls', blurb: 'Cost in bps to consume the whole bid stack — thin bid ⇒ cheap to push down.' },
      { col: 'bid_span_ticks', type: 'float', agg: 'mean across polls', blurb: 'Tick distance best→farthest bid level.' },
    ],
  },
  {
    id: 'x5', tag: 'X₅', name: 'Ask Replenish', color: C.pink,
    summary: 'Cross-signal: is the ask wall rebuilding as fast as buyers eat it? Born at State 1.',
    born: 1, hasDeriv: true,
    components: [
      { col: 'ask_replenish_c', type: 'float', agg: 'd_ask_size_c + buy_volume', blurb: 'The accounting-correct version — only close-based deltas telescope. Normalize by depth or it is a depth proxy.' },
      { col: 'ask_replenish', type: 'float', agg: 'd_ask_size + buy_volume', blurb: 'Legacy mean-based version (corr 0.54 with truth — smears events across buckets). Kept for continuity only.' },
      { col: 'ask_replenish_c_usd / _usd', type: 'float', agg: 'same, USD', blurb: 'Same balances in USD notional.' },
    ],
  },
  {
    id: 'x6', tag: 'X₆', name: 'Mid Price', color: C.cyan,
    summary: 'Mid OHLC + mean across polls — the price reference (never USD-normalized).',
    born: 0, hasDeriv: true,
    components: [
      { col: 'mid_o', type: 'float', agg: 'first poll', blurb: 'Open mid of the bucket.' },
      { col: 'mid_h', type: 'float', agg: 'max across polls', blurb: 'High mid.' },
      { col: 'mid_l', type: 'float', agg: 'min across polls', blurb: 'Low mid.' },
      { col: 'mid_c', type: 'float', agg: 'last poll', blurb: 'Close mid — chain mid_c bucket-over-bucket for the price path.' },
      { col: 'mid_mean', type: 'float', agg: 'mean across polls', blurb: 'Average mid — smoother than close for derivatives.' },
    ],
  },
  {
    id: 'x7', tag: 'X₇', name: 'Bid Replenish', color: C.purple,
    summary: 'Cross-signal: is the bid wall rebuilding as fast as sellers hit it? Born at State 1.',
    born: 1, hasDeriv: true,
    components: [
      { col: 'bid_replenish_c', type: 'float', agg: 'd_bid_size_c + sell_volume', blurb: 'Accounting-correct support-rebuild measure. repl imbalance had solid 5s IC (+0.071 BTC) but adds little beyond book_imb_c.' },
      { col: 'bid_replenish', type: 'float', agg: 'd_bid_size + sell_volume', blurb: 'Legacy mean-based version, kept for continuity.' },
      { col: 'bid_replenish_c_usd / _usd', type: 'float', agg: 'same, USD', blurb: 'Same balances in USD notional.' },
    ],
  },
  {
    id: 'x8', tag: 'X₈', name: 'Sweeps', color: C.emerald,
    summary: 'Parent-order flow: fills grouped by (hash, side) — orders instead of prints, plus the largest single order.',
    born: 0, hasDeriv: true,
    components: [
      { col: 'sweep_count_buy / _sell', type: 'int', agg: 'count of distinct orders', blurb: 'Distinct aggressor orders. buy_count / sweep_count_buy = fills-per-order (fragmentation / urgency).' },
      { col: 'max_sweep_volume_buy / _sell', type: 'float', agg: 'max single order', blurb: 'Largest single order volume. At 5m stacks its imbalance flips NEGATIVE — a giant sweep marks exhaustion, not continuation.' },
      { col: 'max_sweep_levels_buy / _sell', type: 'int', agg: 'max levels consumed', blurb: 'Levels the biggest sweep ate through.' },
      { col: 'max_sweep_range_buy / _sell', type: 'float', agg: 'max price range', blurb: 'Price distance the biggest sweep covered.' },
    ],
  },
  {
    id: 'x9', tag: 'X₉', name: 'Spread & Tick', color: C.amber,
    summary: 'Top-of-book spread at three scales (abs / bps / ticks) + data-derived tick size. Model on _ticks.',
    born: 0, hasDeriv: true,
    components: [
      { col: 'spread_o/h/l/c, spread_mean', type: 'float', agg: 'OHLC + mean', blurb: 'Absolute-price spread group. spread_h = liquidity stress spike.' },
      { col: 'spread_*_ticks', type: 'float', agg: 'OHLC + mean', blurb: 'THE modeling scale. ~1 tick almost always on tight coins; deviations mark stress. No directional IC anywhere — use as regime/cost gate.' },
      { col: 'spread_*_bps', type: 'float', agg: 'OHLC + mean', blurb: 'Deprecated for modeling (~97% negated mid return when dollar spread frozen). spread_mean_bps stays useful as the fee/slippage floor.' },
      { col: 'tick_size', type: 'float', agg: 'data-derived', blurb: '5-significant-figure rule; jumps 10× at powers of ten. The hardcoded config ticks are wrong near boundaries.' },
    ],
  },
];

// ── stacking rules ──────────────────────────────────────────────────────────
const STACK_RULES = [
  {
    family: 'Counts', color: C.green, rule: 'SUM',
    cols: 'poll_count, buy_count, sell_count',
    detail: 'Add the child counts. Exact — a 1m bucket has Σ of its twelve 5s poll/trade counts.',
  },
  {
    family: 'Volumes', color: C.green, rule: 'SUM',
    cols: 'buy_volume(_usd), sell_volume(_usd)',
    detail: 'Add. Volume is additive over disjoint windows — exact.',
  },
  {
    family: 'Book size / ppl', color: C.blue, rule: 'POLL-WEIGHTED MEAN',
    cols: 'ask_size(_usd), ask_ppl, bid_size(_usd), bid_ppl',
    detail: 'Σ(child_mean × child_poll_count) / Σ(child_poll_count). Exact reconstruction of the over-all-polls mean from per-bucket means.',
  },
  {
    family: 'Book fill / centroid', color: C.cyan, rule: 'POLL-WEIGHTED MEAN ≈',
    cols: 'ask_fill, ask_centroid, bid_fill, bid_centroid',
    detail: 'Same weighting, but APPROXIMATE — these are bounded ratios, so the mean-of-means is close but not identical to re-reducing raw polls. Good enough for signals; recompute from l2_snapshots if you need exactness.',
  },
  {
    family: 'Mid / Spread OHLC', color: C.amber, rule: 'OHLC ROLLUP',
    cols: 'mid_o/h/l/c, spread_o/h/l/c (+_bps)',
    detail: 'o = first child o · h = max child h · l = min child l · c = last child c. Exact.',
  },
  {
    family: 'Mid / Spread mean', color: C.amber, rule: 'POLL-WEIGHTED MEAN',
    cols: 'mid_mean, spread_mean, spread_mean_bps',
    detail: 'Σ(child_mean × child_poll_count) / Σ(child_poll_count). Exact.',
  },
  {
    family: 'Derivatives (State 1/2)', color: C.fuchsia, rule: 'RECOMPUTE — never sum',
    cols: 'd_*, dd_*',
    detail: 'Do NOT add child derivatives. Build the coarse State-0 series first, then diff consecutive coarse buckets: d = S0[t] − S0[t−1]. State 2 = diff of the coarse State 1. Summing 5s derivatives ≠ the 1m derivative.',
  },
  {
    family: 'Replenishment', color: C.pink, rule: 'RECOMPUTE',
    cols: 'ask_replenish, bid_replenish (+_usd, +velocity)',
    detail: 'Recompute at the coarse grid: ask_replenish = d_ask_size(coarse) + buy_volume(coarse-sum). It mixes a derivative with a sum, so it must be rebuilt after the State-0 rollup, not aggregated.',
  },
];

// ── small UI atoms ───────────────────────────────────────────────────────────
function Pill({ children, color, filled }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 999, fontSize: 11,
      fontWeight: 600, letterSpacing: 0.3, lineHeight: 1.6,
      color: filled ? C.bg : color,
      background: filled ? color : 'transparent',
      border: `1px solid ${color}`,
    }}>{children}</span>
  );
}

function SectionTitle({ n, title, sub }) {
  return (
    <div style={{ marginBottom: 18, marginTop: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{
          width: 30, height: 30, borderRadius: 8, background: C.raised,
          border: `1px solid ${C.border}`, display: 'flex', alignItems: 'center',
          justifyContent: 'center', color: C.fuchsia, fontWeight: 700, fontSize: 14,
        }}>{n}</div>
        <h2 style={{ margin: 0, fontSize: 19, fontWeight: 700, color: C.text }}>{title}</h2>
      </div>
      {sub && <p style={{ margin: '8px 0 0 42px', color: C.sub, fontSize: 13.5, lineHeight: 1.6, maxWidth: 900 }}>{sub}</p>}
    </div>
  );
}

function Card({ children, style }) {
  return (
    <div style={{
      background: C.surface, border: `1px solid ${C.border}`, borderRadius: 14,
      padding: 18, ...style,
    }}>{children}</div>
  );
}

const mono = { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' };

// ── vector card (clickable → component dropdown) ────────────────────────────
function VectorCard({ v, open, onToggle }) {
  const compCount = v.components.length;
  return (
    <div style={{
      background: open ? C.raised : C.surface,
      border: `1px solid ${open ? v.color : C.border}`,
      borderRadius: 14, overflow: 'hidden', transition: 'all .15s',
    }}>
      <button
        onClick={onToggle}
        style={{
          width: '100%', textAlign: 'left', cursor: 'pointer', background: 'transparent',
          border: 'none', padding: 16, display: 'flex', alignItems: 'center', gap: 14,
        }}
      >
        <div style={{
          minWidth: 46, height: 46, borderRadius: 10, background: `${v.color}22`,
          border: `1px solid ${v.color}`, display: 'flex', alignItems: 'center',
          justifyContent: 'center', color: v.color, fontWeight: 700, fontSize: 17,
        }}>{v.tag}</div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <span style={{ color: C.text, fontWeight: 700, fontSize: 15 }}>{v.name}</span>
            <Pill color={STATE_COLOR[v.born]} >born State {v.born}</Pill>
            {v.hasDeriv && <Pill color={C.dim}>+ d_ / dd_</Pill>}
          </div>
          <div style={{ color: C.sub, fontSize: 12.5, marginTop: 4, lineHeight: 1.5 }}>{v.summary}</div>
        </div>
        <div style={{ color: C.dim, fontSize: 12, ...mono, whiteSpace: 'nowrap' }}>
          {compCount} col{compCount > 1 ? 's' : ''} {open ? '▲' : '▼'}
        </div>
      </button>

      {open && (
        <div style={{ padding: '0 16px 16px' }}>
          {v.components.map((c) => (
            <div key={c.col} style={{
              borderTop: `1px solid ${C.border}`, padding: '12px 0',
              display: 'grid', gridTemplateColumns: '220px 1fr', gap: 14, alignItems: 'start',
            }}>
              <div>
                <div style={{ color: v.color, fontWeight: 600, fontSize: 13, ...mono }}>{c.col}</div>
                <div style={{ color: C.dim, fontSize: 11, marginTop: 3 }}>{c.type}</div>
                <div style={{ color: C.sub, fontSize: 11, marginTop: 6, ...mono }}>agg: {c.agg}</div>
              </div>
              <div style={{ color: C.text, fontSize: 13, lineHeight: 1.6 }}>{c.blurb}</div>
            </div>
          ))}

          {v.hasDeriv && (
            <div style={{
              borderTop: `1px solid ${C.border}`, marginTop: 4, paddingTop: 12,
              display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center',
            }}>
              <span style={{ color: C.sub, fontSize: 12 }}>Derivative track:</span>
              {v.born === 0 ? (
                <>
                  <span style={{ ...mono, fontSize: 12, color: STATE_COLOR[1] }}>d_{v.components[0].col}</span>
                  <span style={{ color: C.dim }}>·</span>
                  <span style={{ ...mono, fontSize: 12, color: STATE_COLOR[2] }}>dd_{v.components[0].col}</span>
                  <span style={{ color: C.dim, fontSize: 12 }}>… (one per State-0 metric)</span>
                </>
              ) : (
                <span style={{ ...mono, fontSize: 12, color: STATE_COLOR[2] }}>
                  d_{v.components[0].col} (velocity, State 2)
                </span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── state-progression strip ──────────────────────────────────────────────────
function StateFlow() {
  const steps = [
    { s: 0, title: 'State 0 — raw level', n: '54 cols', body: 'Per-poll book reduction → mean across polls (book) / LAST (close sizes) / OHLC+mean (mid, spread) / sum+count (trades, sweeps). Available bucket 1+.' },
    { s: 1, title: 'State 1 — first derivative', n: '48 cols', body: 'd_X = S0(now) − S0(prev bucket) on 40 diffable keys, plus 8 replenishment cross-signals. NULL when prev_contiguous is false.' },
    { s: 2, title: 'State 2 — second derivative', n: '48 cols', body: 'dd_X = d_X(now) − d_X(prev), plus d_ of the replenishment signals. Needs 3 contiguous buckets. Noise-dominated for direction at 5s — use for events/vol at 15s+.' },
  ];
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 14 }}>
      {steps.map((st) => (
        <Card key={st.s} style={{ borderColor: `${STATE_COLOR[st.s]}66` }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Pill color={STATE_COLOR[st.s]} filled>State {st.s}</Pill>
            <span style={{ color: C.dim, fontSize: 12, ...mono }}>{st.n}</span>
          </div>
          <div style={{ color: C.text, fontWeight: 700, fontSize: 14, marginTop: 10 }}>{st.title}</div>
          <div style={{ color: C.sub, fontSize: 12.5, marginTop: 6, lineHeight: 1.6 }}>{st.body}</div>
        </Card>
      ))}
    </div>
  );
}

// ── stacking visual (5s → coarse) ──────────────────────────────────────────
function StackVisual() {
  const tiers = [
    { label: '5s', n: 1, color: C.fuchsia, note: 'base grid (stored)' },
    { label: '10s', n: 2, color: C.pink, note: '2 × 5s' },
    { label: '15s', n: 3, color: C.purple, note: '3 × 5s' },
    { label: '1m', n: 12, color: C.blue, note: '12 × 5s' },
    { label: '5m', n: 60, color: C.cyan, note: '60 × 5s' },
    { label: '1h', n: 720, color: C.green, note: '720 × 5s' },
  ];
  return (
    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'stretch' }}>
      {tiers.map((t) => (
        <div key={t.label} style={{
          flex: '1 1 130px', minWidth: 120, background: C.surface,
          border: `1px solid ${t.color}55`, borderRadius: 12, padding: 14, textAlign: 'center',
        }}>
          <div style={{ color: t.color, fontWeight: 800, fontSize: 22 }}>{t.label}</div>
          <div style={{ display: 'flex', gap: 2, justifyContent: 'center', margin: '10px 0', flexWrap: 'wrap' }}>
            {Array.from({ length: Math.min(t.n, 12) }).map((_, i) => (
              <span key={i} style={{ width: 7, height: 14, borderRadius: 2, background: t.color, opacity: 0.5 + 0.5 * (i / 12) }} />
            ))}
            {t.n > 12 && <span style={{ color: t.color, fontSize: 11, alignSelf: 'center', marginLeft: 4 }}>…</span>}
          </div>
          <div style={{ color: C.dim, fontSize: 11, ...mono }}>{t.note}</div>
        </div>
      ))}
    </div>
  );
}

// ── strategy approach cards ──────────────────────────────────────────────────
const STRATEGY = [
  {
    id: 'a', badge: '1', name: 'What predicts (44-day study)', color: C.blue,
    thesis: 'book_imb_c — the close-of-bucket depth imbalance — is the strongest signal on all four coins (5s IC: BTC +0.153, ETH +0.109, SOL +0.084, SPX +0.065). Then impact_imb, rangepos, d_book_imb_c.',
    pros: [
      'Compute signals at 5s, express trades at 1–5m: stacking the signal destroys it, extending the hold grows per-trade edge ~3×.',
      'The edge is 4–8× stronger in quiet, thick, tight-spread markets — gate on vol_5m < trailing median + spread ≤ 1 tick.',
      'Cross-coin: the other coin\'s 1m momentum adds real incremental signal (SOL leads BTC); books do not transfer.',
    ],
    cons: [
      'At 5s the mid is unchanged 46–58% of the time — sub-15s return work is dominated by ties.',
      'Naive feature averaging hurts; combination must be sparse (ML composites overfit — round 3).',
    ],
    when: 'Coin ranking: SOL (best move/spread by far) ≥ BTC (strongest IC, cleanest book) > ETH (spread tax) > SPX (too small a mover; best cross-coin target).',
  },
  {
    id: 'b', badge: '2', name: 'The blueprint that works', color: C.green,
    thesis: 'Enter at rolling-1d p99 tails of book_imb_c, quiet-gated, maker-only. Exit on signal FLIP (rank crosses the opposite tail, max 30m) — not on a clock. Both legs skip-one-bucket priced.',
    pros: [
      'Flip exit ≈ doubles per-trade gross vs fixed 1m holds: BTC +1.01/+1.65 bps (train/test), SOL +1.01/+1.58, at 50–80 trades/day.',
      'All 8 study weeks positive on both coins; long and short symmetric; avg hold ~2 min.',
      'Selective variants (persist2 + impact confirm) reach +1.6…+2.3 bps/trade at 3–12/day.',
    ],
    cons: [
      'Adaptive (rolling) thresholds are a deployment requirement — fixed train-fitted cutoffs broke on BTC out-of-sample.',
      'These are mid-to-mid numbers — see card 4: passive maker fills remove ~2/3 of this gross.',
    ],
    when: 'Adopt for any deployment: quiet gate is the risk device (worst day −384 → single digits), flip exit is the profit device.',
  },
  {
    id: 'c', badge: '3', name: 'The binding constraint: fees', color: C.amber,
    thesis: 'Breakeven round-trip fee is 1.01 bps (train) / 1.61 bps (test). Hyperliquid base maker-maker is ~3.0 bps RT; staking discounts ≈ 1.8; volume tiers ≈ 1.0.',
    pros: [
      'At ~1.0 bps RT the portfolio is net-positive: +81 bps/day on the combined BTC+SOL book, 73% days > 0.',
      'Round 3 halved the fee hurdle vs round 2 (which needed ≤ 0.5 bps/side).',
      'Only 18.5% of episodes overlap across coins — both books share one capital base.',
    ],
    cons: [
      'At base tier the strategy is net-negative — taker execution is dead on arrival (9 bps RT).',
      'A small account starts at base tier: the first weeks are a fee-tier bootstrap, roughly break-even at discounted maker.',
    ],
    when: 'The research gate is passed; the fee tier is the project. Stake HYPE, run maker-only, let the strategy\'s own volume grind the tier down, then scale leverage.',
  },
  {
    id: 'd', badge: '4', name: 'Round 4: passive-fill reality (read first)', color: C.red,
    thesis: 'Every number in cards 1–3 assumed transaction at mid, at will. Round 4 replaced that with a passive maker-fill simulation (filled only if price actually reaches the posted limit) — adverse selection costs ~0.9 bps/trade and removes roughly two thirds of the gross.',
    pros: [
      'BTC p99.9 survives: +0.76 / +1.31 bps/trade (train/test) at ~6 trades/day, ~78% fill rate.',
      'Selective variants (persist3, persist2+impact) reach +1.2…+1.7 bps at 1–5 trades/day.',
      'Result is stable when the fill test requires 1–2 ticks of penetration — not a queue artifact.',
    ],
    cons: [
      'Hit rate falls ~5pp on fills — you get filled preferentially on trades that go against you (venue-independent).',
      'SOL is negative in TRAIN on every variant — SOL as a maker strategy is dead; its rank-1 verdict was a mid-to-mid artifact.',
      'On Hyperliquid base maker (3.0 bps RT) even the best BTC variant is net-negative.',
    ],
    when: 'BTC only + high selectivity + near-zero-fee venue: all three required simultaneously. The mid-to-mid backtest is retired — all future work uses the passive-fill model.',
  },
];

// ── 5s bucket coverage calendar (top of dashboard) ──────────────────────────
const BXT_API = (asset) => `http://localhost:8000/api/perp/${asset}/bxt`;
const MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December'];
const WEEKDAYS = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
const EXPECTED_PER_HOUR = 720;   // 3600s / 5s
const HOUR_GREEN = 700;          // healthy hour threshold
const DAY_GREEN = HOUR_GREEN * 24;     // 16800
const EXPECTED_PER_DAY = EXPECTED_PER_HOUR * 24; // 17280

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
      animation: active ? 'bxtpulse 2s infinite' : 'none',
    }}>
      <style>{`@keyframes bxtpulse{0%,100%{opacity:1}50%{opacity:.4}}`}</style>
    </span>
  );
}

function BxtCoverage({ asset = 'btc' }) {
  const API = BXT_API(asset);
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

  // status poll (builder live dot) every 10s
  useEffect(() => {
    let cancelled = false;
    const load = () => fetch(`${API}/status`).then(r => r.json())
      .then(d => { if (!cancelled) setStatus(d); }).catch(() => {});
    load();
    const id = setInterval(load, 10_000);
    return () => { cancelled = true; clearInterval(id); };
  }, [API]);

  // calendar load on month change
  useEffect(() => {
    let cancelled = false;
    setCalLoading(true);
    fetch(`${API}/calendar?year=${year}&month=${month + 1}`)
      .then(r => r.json())
      .then(d => { if (!cancelled) { setCalData(d); setCalLoading(false); } })
      .catch(() => { if (!cancelled) setCalLoading(false); });
    return () => { cancelled = true; };
  }, [API, year, month]);

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
      const r = await fetch(`${API}/day-detail/${dateStr}`);
      setDayDetail(await r.json());
    } catch (_) { setDayDetail(null); }
    setLoadingDetail(false);
  }, [selectedDay, API]);

  const firstDow = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();

  const dayColor = useCallback((dateStr, buckets) => {
    const isToday = dateStr === todayStr;
    const isFuture = dateStr > todayStr;
    if (isFuture) return { bg: 'transparent', border: 'transparent', faded: true };
    if (buckets == null) {
      return { bg: 'transparent', border: isToday ? C.cyan : C.red + '70', faded: !isToday };
    }
    if (buckets >= DAY_GREEN) return { bg: C.green, border: isToday ? C.cyan : 'transparent', faded: false };
    if (buckets > 0) return { bg: C.amber, border: isToday ? C.cyan : 'transparent', faded: false };
    return { bg: 'transparent', border: isToday ? C.cyan : C.red + '70', faded: !isToday };
  }, [todayStr]);

  const cells = useMemo(() => {
    const arr = [];
    for (let i = 0; i < firstDow; i++) arr.push({ type: 'empty', key: `e${i}` });
    for (let d = 1; d <= daysInMonth; d++) {
      const ds = `${year}-${String(month + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
      const rec = dayMap[ds];
      const buckets = rec ? rec.buckets : (ds > todayStr ? undefined : null);
      arr.push({ type: 'day', key: ds, day: d, dateStr: ds, buckets, isToday: ds === todayStr });
    }
    return arr;
  }, [year, month, firstDow, daysInMonth, dayMap, todayStr]);

  const prevMonth = () => { if (month === 0) { setMonth(11); setYear(y => y - 1); } else setMonth(m => m - 1); };
  const nextMonth = () => { if (month === 11) { setMonth(0); setYear(y => y + 1); } else setMonth(m => m + 1); };

  const live = !!status?.is_live;
  const builderAvailable = status?.available !== false;

  return (
    <div style={{ marginBottom: 36 }}>
      {/* builder status strip */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 14, marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 3, height: 20, background: C.green, borderRadius: 2 }} />
          <div style={{ fontSize: 17, fontWeight: 700, color: C.text }}>5s Bucket Coverage</div>
          <code style={{ ...mono, color: C.fuchsia, fontSize: 12 }}>{asset}."5s_bxt"</code>
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
              {builderAvailable ? (live ? 'BXT BUILDER LIVE' : 'BXT BUILDER STALLED') : 'NO BUCKET TABLE'}
            </div>
            <div style={{ fontSize: 12, color: live ? C.green : C.sub, marginTop: 2 }}>
              {status?.latest_bucket
                ? `Last bucket: ${agoStr(status.age_s)}`
                : 'no buckets yet'}
              {status?.total_buckets != null && (
                <span style={{ color: C.dim }}> · {kfmt(status.total_buckets)} total</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* legend */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 12, flexWrap: 'wrap' }}>
        {[
          { color: C.green, label: `Full day (≥ ${kfmt(DAY_GREEN)} buckets)` },
          { color: C.amber, label: 'Partial day' },
          { color: C.red, label: 'No buckets', outline: true },
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
            const { day, dateStr, buckets, isToday } = cell;
            const col = dayColor(dateStr, buckets);
            const hasData = buckets != null && buckets > 0;
            const isSelected = dateStr === selectedDay;
            const pct = hasData ? Math.min(100, Math.round((buckets / EXPECTED_PER_DAY) * 100)) : null;
            return (
              <button
                key={cell.key}
                onClick={() => hasData && handleDayClick(dateStr)}
                title={buckets != null ? `${dateStr}: ${kfmt(buckets)} / ${kfmt(EXPECTED_PER_DAY)} buckets` : dateStr}
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
                  <div style={{ fontSize: 8, fontWeight: 700, color: buckets >= DAY_GREEN ? C.green : C.amber, marginTop: 2 }}>
                    {pct}%
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {selectedDay && <BxtHourBreakdown dateStr={selectedDay} detail={dayDetail} loading={loadingDetail} />}
    </div>
  );
}

function BxtHourBreakdown({ dateStr, detail, loading }) {
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
  const healthy = pastHours.filter(h => h.buckets >= HOUR_GREEN).length;
  const total = pastHours.reduce((s, h) => s + h.buckets, 0);

  return (
    <div style={{ marginTop: 16, background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: '16px 20px' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: C.text }}>
          {dateStr} — Hourly 5s Buckets <span style={{ fontSize: 11, color: C.dim, fontWeight: 400 }}>(UTC · {EXPECTED_PER_HOUR}/hr expected)</span>
        </div>
        <div style={{ fontSize: 12, color: C.sub }}>
          <span style={{ color: healthy === pastHours.length ? C.green : C.amber, fontWeight: 600 }}>{healthy}/{pastHours.length}</span> hours ≥{HOUR_GREEN}
          <span style={{ color: C.dim }}> · {kfmt(total)} buckets</span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: 4 }}>
        {detail.hours.map(h => {
          const isFuture = isFutureDay || (isToday && h.hour > currentHourUTC);
          let bg, borderClr, textClr;
          if (isFuture) {
            bg = '#0e1626'; borderClr = '#2d3748'; textClr = '#334155';
          } else if (h.buckets >= HOUR_GREEN) {
            bg = C.green + '28'; borderClr = C.green; textClr = C.green;
          } else if (h.buckets > 0) {
            bg = C.amber + '25'; borderClr = C.amber; textClr = C.amber;
          } else {
            bg = C.red + '1a'; borderClr = C.red + '70'; textClr = C.red + 'cc';
          }
          return (
            <div key={h.hour}
              title={isFuture ? `${h.hour}:00 — not yet` : `${h.hour}:00 UTC — ${h.buckets} / ${EXPECTED_PER_HOUR} buckets`}
              style={{
                borderRadius: 5, background: bg, borderBottom: `2px solid ${borderClr}`,
                padding: '8px 2px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
              }}>
              <div style={{ fontSize: 10, color: C.dim }}>{String(h.hour).padStart(2, '0')}</div>
              <div style={{ fontSize: 11, fontWeight: 700, color: textClr }}>{isFuture ? '—' : h.buckets}</div>
            </div>
          );
        })}
      </div>

      <div style={{ display: 'flex', gap: 14, marginTop: 12 }}>
        {[
          { color: C.green, label: `Healthy (≥ ${HOUR_GREEN})` },
          { color: C.amber, label: 'Partial (1–699)' },
          { color: C.red, label: 'Empty (0)' },
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

// ── main ─────────────────────────────────────────────────────────────────────
const BXT_ASSETS = [
  { id: 'btc', label: 'BTC', color: C.amber },
  { id: 'sol', label: 'SOL', color: C.purple },
  { id: 'eth', label: 'ETH', color: C.blue },
  { id: 'spx', label: 'SPX', color: C.green },
];

function BxtAssetSwitcher({ asset, onChange }) {
  return (
    <div style={{ display: 'flex', gap: 6, marginBottom: 14 }}>
      {BXT_ASSETS.map((a) => {
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

export default function BucketDashboard() {
  const [openVec, setOpenVec] = useState('x1');
  const [tab, setTab] = useState('vectors');
  const [bxtAsset, setBxtAsset] = useState('btc');

  const TABS = [
    { id: 'vectors', label: 'Vectors', icon: '◫' },
    { id: 'engine', label: 'Deriv Engine', icon: '∂' },
    { id: 'stacking', label: 'Stacking', icon: '⧉' },
    { id: 'strategy', label: 'Strategy', icon: '◎' },
  ];

  return (
    <div style={{ background: C.bg, minHeight: '100vh', color: C.text, padding: '24px 28px 80px' }}>
      <div style={{ maxWidth: 1180, margin: '0 auto' }}>

        {/* ── live 5s bucket coverage (top) ── */}
        <BxtAssetSwitcher asset={bxtAsset} onChange={setBxtAsset} />
        <BxtCoverage key={bxtAsset} asset={bxtAsset} />

        {/* divider between live coverage and the static explainer */}
        <div style={{ borderTop: `1px solid ${C.border}`, margin: '8px 0 28px' }} />

        {/* header */}
        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16, marginBottom: 8 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <span style={{ fontSize: 26 }}>◧</span>
              <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800, letterSpacing: 0.4,
                background: `linear-gradient(90deg, ${C.fuchsia}, ${C.cyan})`,
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                DERIV BXT DASHBOARD
              </h1>
            </div>
            <p style={{ color: C.sub, margin: '8px 0 0', fontSize: 13.5, maxWidth: 820, lineHeight: 1.6 }}>
              Structural map of the pre-computed 5-second DERIV signal buckets —{' '}
              <code style={{ ...mono, color: C.fuchsia }}>{'<coin>'}."5s_bxt"</code> for BTC / SOL / ETH / SPX.
              150 components + quality flags · three derivative states, written live by{' '}
              <code style={{ ...mono, color: C.amber }}>bxtbuilder_5s.py</code> (healed by <code style={{ ...mono, color: C.amber }}>reconcile_5sbxt.py</code>).
            </p>
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {[['150', 'components', C.fuchsia], ['10', 'vectors', C.cyan], ['3', 'states', C.amber], ['5s', 'grid', C.green]].map(([n, l, col]) => (
              <div key={l} style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: '10px 16px', textAlign: 'center' }}>
                <div style={{ color: col, fontWeight: 800, fontSize: 20 }}>{n}</div>
                <div style={{ color: C.dim, fontSize: 11 }}>{l}</div>
              </div>
            ))}
          </div>
        </div>

        {/* tabs */}
        <div style={{ display: 'flex', gap: 6, margin: '22px 0 26px', borderBottom: `1px solid ${C.border}`, paddingBottom: 0 }}>
          {TABS.map((t) => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              cursor: 'pointer', background: 'transparent', border: 'none',
              borderBottom: `2px solid ${tab === t.id ? C.fuchsia : 'transparent'}`,
              color: tab === t.id ? C.text : C.dim, fontWeight: 600, fontSize: 14,
              padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 8,
            }}>
              <span style={{ fontSize: 16 }}>{t.icon}</span>{t.label}
            </button>
          ))}
        </div>

        {/* ── VECTORS ── */}
        {tab === 'vectors' && (
          <div>
            <SectionTitle n="1" title="The 10 vectors → 150 components"
              sub="Each vector is one aspect of the order book / tape. Click any card to expand its component columns, types, per-poll aggregation and meaning. Vectors born at State 0 also spawn a d_ (first-derivative) and dd_ (second-derivative) column for most metrics." />
            <div style={{ display: 'grid', gap: 12 }}>
              {VECTORS.map((v) => (
                <VectorCard key={v.id} v={v} open={openVec === v.id} onToggle={() => setOpenVec(openVec === v.id ? null : v.id)} />
              ))}
            </div>
            <Card style={{ marginTop: 18, borderColor: `${C.amber}55` }}>
              <div style={{ color: C.amber, fontWeight: 700, fontSize: 13 }}>Column accounting</div>
              <div style={{ color: C.sub, fontSize: 12.5, marginTop: 8, lineHeight: 1.7 }}>
                State 0 = <b style={{ color: C.text }}>54</b> raw within-bucket measurements · State 1 = <b style={{ color: C.text }}>48</b> (d_ of 40 diffable keys + 8 replenishment cross-signals) ·
                State 2 = <b style={{ color: C.text }}>48</b> (dd_ + d_ of replenishment) = <b style={{ color: C.text }}>150</b> components.
                Stored table adds <code style={{ ...mono }}>ts</code> (PK), quality flags (<code style={{ ...mono }}>prev_contiguous</code>, <code style={{ ...mono }}>has_trades</code>, <code style={{ ...mono }}>poll_gap_max_ms</code>, <code style={{ ...mono }}>was_revised</code>) and <code style={{ ...mono }}>inserted_at</code> → 157 columns in the <code style={{ ...mono }}>bxt_5s</code> view (incl. <code style={{ ...mono }}>bucket_ideal</code>).
              </div>
            </Card>
          </div>
        )}

        {/* ── ENGINE ── */}
        {tab === 'engine' && (
          <div>
            <SectionTitle n="2" title="How the derivatives & replenishment work"
              sub="Every bucket reduces L2 polls + trades to State 0, then chains backward in time to build velocity (State 1) and acceleration (State 2). Derivatives are gated on contiguity so they never reach across a gap." />
            <StateFlow />

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 14, marginTop: 18 }}>
              <Card style={{ borderColor: `${C.pink}55` }}>
                <div style={{ color: C.pink, fontWeight: 700, fontSize: 14 }}>Replenishment — the cross-signal</div>
                <div style={{ color: C.sub, fontSize: 12.5, marginTop: 8, lineHeight: 1.7 }}>
                  The book and the tape are coupled. Replenishment asks: did resting liquidity rebuild as fast as takers consumed it?
                </div>
                <div style={{ ...mono, fontSize: 12.5, marginTop: 12, background: C.bg, border: `1px solid ${C.border}`, borderRadius: 8, padding: 12, lineHeight: 1.9 }}>
                  <div><span style={{ color: C.pink }}>ask_replenish</span> = d_ask_size + buy_volume</div>
                  <div><span style={{ color: C.purple }}>bid_replenish</span> = d_bid_size + sell_volume</div>
                  <div style={{ color: C.dim, marginTop: 8 }}>velocity (State 2):</div>
                  <div><span style={{ color: C.pink }}>d_ask_replenish</span> = dd_ask_size + d_buy_volume</div>
                </div>
                <div style={{ color: C.sub, fontSize: 12, marginTop: 10, lineHeight: 1.6 }}>
                  &gt;0 = wall refilling faster than it&apos;s eaten (absorption / support). &lt;0 = wall thinning under flow (likely to give way).
                </div>
              </Card>

              <Card style={{ borderColor: `${C.fuchsia}55` }}>
                <div style={{ color: C.fuchsia, fontWeight: 700, fontSize: 14 }}>Contiguity gating & NULLs</div>
                <ul style={{ color: C.sub, fontSize: 12.5, marginTop: 8, lineHeight: 1.7, paddingLeft: 18 }}>
                  <li>State 1 needs bucket <i>i−1</i> to exist and be adjacent (<code style={{ ...mono }}>prev_start == cur_start − 5</code>), else all <code style={{ ...mono }}>d_*</code> = NULL.</li>
                  <li>State 2 needs <i>i−2, i−1, i</i> all contiguous, else all <code style={{ ...mono }}>dd_*</code> = NULL.</li>
                  <li>Empty buckets (no polls AND no trades) are skipped → break contiguity → the next real bucket restarts the derivative cycle.</li>
                  <li>A NULL sub-component propagates: a trades-only bucket has NULL ask_size → NULL d_ask_size → NULL ask_replenish → NULL dd_ask_size…</li>
                  <li><code style={{ ...mono }}>prev_contiguous</code> (boolean column) flags whether a row had a contiguous predecessor.</li>
                </ul>
              </Card>
            </div>

            <Card style={{ marginTop: 18, borderColor: `${C.amber}66` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 18 }}>⚠</span>
                <div style={{ color: C.amber, fontWeight: 700, fontSize: 14 }}>Data-integrity status (updated Aug 2026)</div>
              </div>
              <div style={{ color: C.sub, fontSize: 12.5, marginTop: 10, lineHeight: 1.7 }}>
                The stale-bucket problem is now handled: <code style={{ ...mono }}>reconcile_5sbxt.py</code> re-runs recent windows from raw and{' '}
                <code style={{ ...mono, color: C.text }}>was_revised</code> marks every bucket whose stored values changed after first write.
                Known eras to exclude or handle: <b style={{ color: C.text }}>June 15–24</b> feed throttle (poll_count filter catches it),{' '}
                <b style={{ color: C.text }}>Aug 1–4</b> collector outage (rows don&apos;t exist), and <b style={{ color: C.text }}>since ~Aug 11</b> collector ingest
                latency degraded (p95 5.8s → 9–20s), pushing revision rates to ~25% on affected days — the flag is accurate, the collector is the problem.
                For research, treat <code style={{ ...mono }}>was_revised</code> as a control variable, not a filter: revisions cluster in volatile moments,
                so dropping them biases toward calm regimes.
              </div>
            </Card>
          </div>
        )}

        {/* ── STACKING ── */}
        {tab === 'stacking' && (
          <div>
            <SectionTitle n="3" title="Stacking 5s → 10s · 15s · 1m · 5m · 1h"
              sub="The 5s table is the atom. Coarser bars are exact rollups because the grid is an absolute UTC epoch grid (floor(epoch/5)*5), so every coarse boundary lands on a 5s boundary — no realignment needed. The rule depends on the column family." />
            <StackVisual />

            <div style={{ marginTop: 20, display: 'grid', gap: 10 }}>
              {STACK_RULES.map((r) => (
                <div key={r.family} style={{
                  background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 14,
                  display: 'grid', gridTemplateColumns: '170px 150px 1fr', gap: 14, alignItems: 'center',
                }}>
                  <div style={{ color: r.color, fontWeight: 700, fontSize: 13.5 }}>{r.family}</div>
                  <div><Pill color={r.color} filled>{r.rule}</Pill></div>
                  <div>
                    <div style={{ ...mono, color: C.sub, fontSize: 11.5, marginBottom: 4 }}>{r.cols}</div>
                    <div style={{ color: C.text, fontSize: 12.5, lineHeight: 1.6 }}>{r.detail}</div>
                  </div>
                </div>
              ))}
            </div>

            <Card style={{ marginTop: 18, borderColor: `${C.green}55` }}>
              <div style={{ color: C.green, fontWeight: 700, fontSize: 13.5 }}>The golden rule of stacking</div>
              <div style={{ color: C.sub, fontSize: 12.5, marginTop: 8, lineHeight: 1.7 }}>
                <b style={{ color: C.text }}>Rebuild State 0 at the coarse grid first, then re-derive State 1 & 2 from that coarse series.</b>{' '}
                Counts/volumes sum, levels/prices are poll-weighted means, OHLC rolls up — but derivatives and replenishment must be
                recomputed on the coarse buckets, never aggregated from the 5s derivatives. A materialized view per timeframe
                (<code style={{ ...mono }}>10s_bxt</code>, <code style={{ ...mono }}>1m_bxt</code>, …) keyed off the same epoch grid keeps everything aligned across coins.
              </div>
            </Card>
          </div>
        )}

        {/* ── STRATEGY ── */}
        {tab === 'strategy' && (
          <div>
            <SectionTitle n="4" title="What the research says — BTC · SOL · ETH · SPX"
              sub="Empirical findings from the 44-day IC/backtest study plus rounds 2–4 (map.md §7–§12): train/test split 06-25→07-25→08-12, rolling thresholds, non-overlapping episodes, no lookahead on entry or exit, passive maker-fill accounting from round 4 on." />

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 14 }}>
              {STRATEGY.map((s) => (
                <div key={s.id} style={{
                  background: C.surface, border: `1px solid ${s.color}66`, borderRadius: 14, padding: 18,
                  display: 'flex', flexDirection: 'column',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div style={{ width: 32, height: 32, borderRadius: 8, background: `${s.color}22`, border: `1px solid ${s.color}`, display: 'flex', alignItems: 'center', justifyContent: 'center', color: s.color, fontWeight: 800 }}>{s.badge}</div>
                    <div style={{ color: C.text, fontWeight: 700, fontSize: 15 }}>{s.name}</div>
                  </div>
                  <div style={{ color: C.sub, fontSize: 12.5, marginTop: 10, lineHeight: 1.6 }}>{s.thesis}</div>

                  <div style={{ color: C.green, fontSize: 11, fontWeight: 700, marginTop: 14, textTransform: 'uppercase', letterSpacing: 0.5 }}>Edge</div>
                  <ul style={{ color: C.text, fontSize: 12, margin: '6px 0 0', paddingLeft: 16, lineHeight: 1.6 }}>
                    {s.pros.map((p, i) => <li key={i} style={{ marginBottom: 4 }}>{p}</li>)}
                  </ul>

                  <div style={{ color: C.red, fontSize: 11, fontWeight: 700, marginTop: 12, textTransform: 'uppercase', letterSpacing: 0.5 }}>Cost</div>
                  <ul style={{ color: C.sub, fontSize: 12, margin: '6px 0 0', paddingLeft: 16, lineHeight: 1.6 }}>
                    {s.cons.map((p, i) => <li key={i} style={{ marginBottom: 4 }}>{p}</li>)}
                  </ul>

                  <div style={{ marginTop: 'auto', paddingTop: 14 }}>
                    <div style={{ background: C.bg, border: `1px solid ${C.border}`, borderRadius: 8, padding: 10, color: s.color, fontSize: 12, lineHeight: 1.5 }}>
                      {s.when}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <Card style={{ marginTop: 18, borderColor: `${C.cyan}55` }}>
              <div style={{ color: C.cyan, fontWeight: 700, fontSize: 14 }}>What to build (map.md §12.4)</div>
              <ol style={{ color: C.sub, fontSize: 12.5, marginTop: 10, paddingLeft: 20, lineHeight: 1.8 }}>
                <li><b style={{ color: C.text }}>BTC quiet-book maker strategy, high selectivity</b> — p99.9 / persist variants of book_imb_c, rolling ranks, quiet gate, flip exit, passive-fill accounting. Realistic gross +0.76/+1.31 bps at ~6 trades/day.</li>
                <li><b style={{ color: C.text }}>Zero-fee venue</b> — Lighter Standard (0 bps maker) is the only clearly positive cell; HL base maker is −2.2/−1.7 net. Lighter collector is live; re-run the §12 model on Lighter&apos;s own spread/depth before committing capital.</li>
                <li><b style={{ color: C.text }}>Retired</b> — mid-to-mid backtests, SOL as a maker strategy, taker execution at any tier.</li>
              </ol>
              <div style={{ color: C.text, fontSize: 13, marginTop: 12, lineHeight: 1.6, fontWeight: 600 }}>
                Deferred (refuted or unstable): SOL/BTC pair z-fade (tail blowups), absorption-bounce narratives, premium/OI/funding as standalone strategies, SPX solo, ML composites.
              </div>
            </Card>
          </div>
        )}

      </div>
    </div>
  );
}
