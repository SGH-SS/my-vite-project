/**
 * DERIV BXT DASHBOARD — bucket.jsx
 * ============================================================================
 * Standalone explainer + structural map for the pre-computed 5s DERIV Signal
 * Buckets (`btc."5s_bxt"`, written live by sol-perp/bucket_builder_btc.py).
 *
 * Four sections:
 *   1. VECTORS      — every vector is a clickable card; expand to see its
 *                     component columns across State 0 / 1 / 2 with formulas.
 *   2. DERIV ENGINE — how State 0 → 1 → 2 (derivatives) + replenishment work,
 *                     including contiguity gating and NULL propagation.
 *   3. STACKING     — how to roll 5s buckets up into 10s / 15s / 1m / 5m / 1h,
 *                     column-family by column-family (sum vs mean vs OHLC vs
 *                     recompute-derivative).
 *   4. STRATEGY     — given this infra across BTC/SOL/ETH/SPX, where to point
 *                     it: individual-first vs correlation-first vs both.
 *
 * Pure presentational component (no network). Lives next to StratHub.jsx and is
 * mounted via the `bxt` dashboard mode in TradingDashboard.jsx.
 */

import { useState } from 'react';

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
    id: 'meta', tag: 'M', name: 'Metadata', color: C.dim,
    summary: 'Bookkeeping for the bucket — how many L2 polls landed in the 5s window.',
    born: 0, hasDeriv: false,
    components: [
      { col: 'poll_count', type: 'int', agg: 'count of polls', blurb: 'L2 snapshots captured in the window (~9–10 at 0.5s flush). Low values flag an under-captured / late-data bucket.' },
    ],
  },
  {
    id: 'x1', tag: 'X₁', name: 'Ask Book', color: C.red,
    summary: 'Resting sell-side liquidity, reduced over all 20 ask levels then averaged across polls.',
    born: 0, hasDeriv: true,
    components: [
      { col: 'ask_size', type: 'float', agg: 'mean across polls', blurb: 'Σ size over all ask levels. Base unit (coins).' },
      { col: 'ask_size_usd', type: 'float', agg: 'mean across polls', blurb: 'ask_size × best_ask per poll, then mean. The only USD-normalized book metric.' },
      { col: 'ask_ppl', type: 'float', agg: 'mean across polls', blurb: 'Σ order count (n) over all levels — posts-per-level / queue density.' },
      { col: 'ask_fill', type: 'float 0–1', agg: 'mean across polls', blurb: 'occupied tick slots / possible tick slots between best & farthest level. 1 = solid wall, →0 = sparse.' },
      { col: 'ask_centroid', type: 'float 0–1', agg: 'mean across polls', blurb: 'size-weighted mean distance from best as a fraction of depth. 0 = front-loaded, 1 = back-loaded.' },
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
    summary: 'Resting buy-side liquidity, reduced over all 20 bid levels then averaged across polls.',
    born: 0, hasDeriv: true,
    components: [
      { col: 'bid_size', type: 'float', agg: 'mean across polls', blurb: 'Σ size over all bid levels (coins).' },
      { col: 'bid_size_usd', type: 'float', agg: 'mean across polls', blurb: 'bid_size × best_bid per poll, then mean.' },
      { col: 'bid_ppl', type: 'float', agg: 'mean across polls', blurb: 'Σ order count over all bid levels.' },
      { col: 'bid_fill', type: 'float 0–1', agg: 'mean across polls', blurb: 'tick occupancy of the bid stack. 1 = wall, →0 = airpockets.' },
      { col: 'bid_centroid', type: 'float 0–1', agg: 'mean across polls', blurb: '0 = front-loaded near best, 1 = back-loaded deep.' },
    ],
  },
  {
    id: 'x5', tag: 'X₅', name: 'Ask Replenish', color: C.pink,
    summary: 'Cross-signal: is the ask wall rebuilding as fast as buyers eat it? Born at State 1.',
    born: 1, hasDeriv: true,
    components: [
      { col: 'ask_replenish', type: 'float', agg: 'd_ask_size + buy_volume', blurb: 'Net change in ask depth plus what got lifted. >0 = liquidity refilling faster than consumed; <0 = wall thinning.' },
      { col: 'ask_replenish_usd', type: 'float', agg: 'd_ask_size_usd + buy_volume_usd', blurb: 'Same balance in USD notional.' },
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
      { col: 'bid_replenish', type: 'float', agg: 'd_bid_size + sell_volume', blurb: 'Net change in bid depth plus what got hit. >0 = bids refilling; <0 = support evaporating.' },
      { col: 'bid_replenish_usd', type: 'float', agg: 'd_bid_size_usd + sell_volume_usd', blurb: 'Same balance in USD notional.' },
    ],
  },
  {
    id: 'x9', tag: 'X₉', name: 'Spread', color: C.amber,
    summary: 'Top-of-book spread, OHLC + mean, in both absolute price and bps.',
    born: 0, hasDeriv: true,
    components: [
      { col: 'spread_o', type: 'float', agg: 'first poll', blurb: 'Open spread (abs price).' },
      { col: 'spread_h', type: 'float', agg: 'max across polls', blurb: 'Widest spread — liquidity stress spike.' },
      { col: 'spread_l', type: 'float', agg: 'min across polls', blurb: 'Tightest spread.' },
      { col: 'spread_c', type: 'float', agg: 'last poll', blurb: 'Close spread.' },
      { col: 'spread_mean', type: 'float', agg: 'mean across polls', blurb: 'Average spread (abs).' },
      { col: 'spread_o_bps', type: 'float', agg: 'first poll', blurb: 'spread / mid × 10000, open.' },
      { col: 'spread_h_bps', type: 'float', agg: 'max across polls', blurb: 'Widest spread in bps — cross-asset comparable.' },
      { col: 'spread_l_bps', type: 'float', agg: 'min across polls', blurb: 'Tightest spread in bps.' },
      { col: 'spread_c_bps', type: 'float', agg: 'last poll', blurb: 'Close spread in bps.' },
      { col: 'spread_mean_bps', type: 'float', agg: 'mean across polls', blurb: 'Average spread in bps — the headline liquidity-cost number.' },
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
    { s: 0, title: 'State 0 — raw level', n: '32 cols', body: 'Per-poll book reduction → mean across polls (book) / OHLC+mean (mid, spread) / sum+count (trades). Available bucket 1+.' },
    { s: 1, title: 'State 1 — first derivative', n: '35 cols', body: 'd_X = S0(now) − S0(prev bucket). Plus 4 replenishment cross-signals. Available bucket 2+ (needs a contiguous predecessor).' },
    { s: 2, title: 'State 2 — second derivative', n: '35 cols', body: 'dd_X = d_X(now) − d_X(prev). Plus 4 replenishment velocities. Available bucket 3+ (needs 3 contiguous in a row).' },
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
    id: 'a', badge: 'A', name: 'Individual-first', color: C.blue,
    thesis: 'Characterize each coin in isolation, build per-coin alpha, then look for correlations as a bonus / risk overlay.',
    pros: [
      'Cleanest causal story: each signal maps to one book.',
      'Backtest + attribution are simple — one instrument, one PnL.',
      'You already have BTC live; SOL/ETH/SPX just replicate the same builder.',
    ],
    cons: [
      'Single-coin microstructure edges are the most crowded / fastest to decay.',
      'You leave the richest structure (lead-lag, divergence) on the table.',
    ],
    when: 'Start here to validate the data + signals are real on the asset you know best (BTC).',
  },
  {
    id: 'b', badge: 'B', name: 'Correlation-first', color: C.fuchsia,
    thesis: 'Treat the tradable object as the relationship between coins (spread / ratio / lead-lag), not any single book.',
    pros: [
      'BTC tends to lead; SOL/ETH react. The replenish + derivative vectors are built to catch who moves first.',
      'Relative-value / pairs are more capacity-rich and less crowded at the 5s–1m horizon.',
      'Market-neutral framing survives regime flips that kill outright direction.',
    ],
    cons: [
      'Needs all four builders live + clock-aligned (your absolute 5s UTC grid already gives this).',
      'Correlation is unstable; you must detect when it breaks, not assume it.',
    ],
    when: 'The higher-ceiling direction — but only trustworthy once each coin\'s signals are individually validated.',
  },
  {
    id: 'c', badge: 'C', name: 'Both (recommended)', color: C.green,
    thesis: 'Sequence them. Individual work is the foundation that makes correlation work interpretable; correlation is where the durable edge lives.',
    pros: [
      'Per-coin models become the "fair value" each cross-coin signal is measured against.',
      'A divergence is only meaningful once you know each coin\'s baseline behaviour.',
      'Reuses 100% of the individual work as features for the relational layer.',
    ],
    cons: [
      'More moving parts; demands the data-integrity issue (below) be fixed first so cross-coin diffs aren\'t comparing a healed book to a stale one.',
    ],
    when: 'The plan: Phase 1 individual (per-coin signal validation) → Phase 2 align the 5s grids → Phase 3 lead-lag / spread / divergence on top.',
  },
];

// ── main ─────────────────────────────────────────────────────────────────────
export default function BucketDashboard() {
  const [openVec, setOpenVec] = useState('x1');
  const [tab, setTab] = useState('vectors');

  const TABS = [
    { id: 'vectors', label: 'Vectors', icon: '◫' },
    { id: 'engine', label: 'Deriv Engine', icon: '∂' },
    { id: 'stacking', label: 'Stacking', icon: '⧉' },
    { id: 'strategy', label: 'Strategy', icon: '◎' },
  ];

  return (
    <div style={{ background: C.bg, minHeight: '100vh', color: C.text, padding: '24px 28px 80px' }}>
      <div style={{ maxWidth: 1180, margin: '0 auto' }}>

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
              <code style={{ ...mono, color: C.fuchsia }}>btc."5s_bxt"</code>. 9 vectors · 102 components ·
              three derivative states, written live by <code style={{ ...mono, color: C.amber }}>bucket_builder_btc.py</code>.
            </p>
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {[['102', 'components', C.fuchsia], ['9', 'vectors', C.cyan], ['3', 'states', C.amber], ['5s', 'grid', C.green]].map(([n, l, col]) => (
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
            <SectionTitle n="1" title="The 9 vectors → 102 components"
              sub="Each vector is one aspect of the order book / tape. Click any card to expand its component columns, types, per-poll aggregation and meaning. Vectors born at State 0 also spawn a d_ (first-derivative) and dd_ (second-derivative) column for every metric." />
            <div style={{ display: 'grid', gap: 12 }}>
              {VECTORS.map((v) => (
                <VectorCard key={v.id} v={v} open={openVec === v.id} onToggle={() => setOpenVec(openVec === v.id ? null : v.id)} />
              ))}
            </div>
            <Card style={{ marginTop: 18, borderColor: `${C.amber}55` }}>
              <div style={{ color: C.amber, fontWeight: 700, fontSize: 13 }}>Column accounting</div>
              <div style={{ color: C.sub, fontSize: 12.5, marginTop: 8, lineHeight: 1.7 }}>
                State 0 = <b style={{ color: C.text }}>32</b> (3 int counts + 29 floats) · State 1 = <b style={{ color: C.text }}>35</b> (31 d_ diffs + 4 replenish) ·
                State 2 = <b style={{ color: C.text }}>35</b> (31 dd_ diffs + 4 replenish-velocity) = <b style={{ color: C.text }}>102</b> components.
                Stored table adds <code style={{ ...mono }}>ts</code> (PK), <code style={{ ...mono }}>prev_contiguous</code>, <code style={{ ...mono }}>inserted_at</code> → 105 columns.
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

            <Card style={{ marginTop: 18, borderColor: `${C.red}66`, background: '#1a0d14' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 18 }}>⚠</span>
                <div style={{ color: C.red, fontWeight: 700, fontSize: 14 }}>Data-integrity caveat found in the live table (verified 2026-06-02)</div>
              </div>
              <div style={{ color: C.sub, fontSize: 12.5, marginTop: 10, lineHeight: 1.7 }}>
                The State 0/1/2 math is correct — <b style={{ color: C.text }}>414 / 434 buckets match feature_builder byte-for-byte</b>. But the builder reads each
                5s window <b style={{ color: C.text }}>once</b>, at close + 3s lag, and never re-reads. When the collector reconnects/backfills (303 reconnects/month) or
                ingest lag exceeds the 3s budget (p99 ≈ 5.4s, max ≈ 9s), late rows land in a window <i>after</i> it was computed. Those buckets
                (~20 here, clustered at one 02:43–02:45 reconnect) are computed on partial data and <b style={{ color: C.text }}>never healed</b>, because
                <code style={{ ...mono }}> ON CONFLICT</code> only fires if the same bucket is reprocessed. Fix: a finalization/reconcile pass that re-runs the last N
                minutes from raw a minute later, or gate finalize on an ingest watermark. (Two builders disagreeing on a stale-vs-healed book is exactly what breaks
                cross-coin correlation work.)
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
            <SectionTitle n="4" title="Where to point this — BTC · SOL · ETH · SPX"
              sub="Assume the 5s builder + clean stacking (10s/15s/1m/1h) exists for all four coins on the same absolute UTC grid. The real question: does tradability come from each coin individually, or from the relationships between them? Short answer — do both, in sequence." />

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
              <div style={{ color: C.cyan, fontWeight: 700, fontSize: 14 }}>The recommended path (concrete)</div>
              <ol style={{ color: C.sub, fontSize: 12.5, marginTop: 10, paddingLeft: 20, lineHeight: 1.8 }}>
                <li><b style={{ color: C.text }}>Define each coin.</b> On BTC first (it&apos;s live + you know it), prove individual signals are real: does negative ask_replenish predict the mid breaking up? does spread_h_bps spike before vol? Validate, then clone the builder to SOL/ETH/SPX.</li>
                <li><b style={{ color: C.text }}>Fix finalization.</b> Add the reconcile pass so no coin&apos;s buckets are stale — cross-coin diffs must compare healed books, or correlation work is built on sand.</li>
                <li><b style={{ color: C.text }}>Align grids.</b> All four share the absolute 5s UTC epoch grid already — so a 5s bucket on BTC and SOL describe the exact same wall-clock window. This is what makes lead-lag measurable.</li>
                <li><b style={{ color: C.text }}>Trade the relationship.</b> Stack to the horizon where structure lives (10s–1m for lead-lag, 5m–1h for divergence/RV). The durable edge is BTC leading SOL/ETH, or one book&apos;s replenishment diverging from the complex — not any single coin in a vacuum.</li>
              </ol>
              <div style={{ color: C.text, fontSize: 13, marginTop: 12, lineHeight: 1.6, fontWeight: 600 }}>
                Verdict: <span style={{ color: C.green }}>both, but sequenced</span> — individual work is the ruler; the correlation layer is where the high-capacity, lower-crowding edge actually lives. Individual-first as a foundation, correlation-first as the destination.
              </div>
            </Card>
          </div>
        )}

      </div>
    </div>
  );
}
