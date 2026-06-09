import React, { useState, useEffect } from 'react';

// ── Palette ───────────────────────────────────────────────────────────
const C = {
  bg:      '#060c18',
  surface: '#0c1628',
  raised:  '#101e35',
  border:  '#1a2d4e',
  green:   '#00d4a8',
  red:     '#f43f5e',
  amber:   '#f59e0b',
  blue:    '#60a5fa',
  purple:  '#a78bfa',
  pink:    '#fb7185',
  text:    '#e2e8f0',
  sub:     '#94a3b8',
  muted:   '#4a5568',
  mono:    "'JetBrains Mono', 'Fira Code', monospace",
  display: "'Bebas Neue', 'Impact', sans-serif",
  body:    "'DM Sans', 'Segoe UI', sans-serif",
};

// ── Strategy Data ─────────────────────────────────────────────────────
const COMPONENTS = [
  {
    id: 'A', weight: '±3.0', color: C.green,
    name: 'Delta–Price Divergence',
    desc: 'Price makes a new session high/low but cumulative delta fails to confirm — the canonical exhaustion read. Buyers/sellers running out of fuel.',
    logic: 'close = session_high  ∧  cum_delta < session_Δ_high  →  short (−3)',
    type: 'Divergence',
  },
  {
    id: 'B', weight: '+2.0', color: C.amber,
    name: 'Cross-Timeframe Conflict',
    desc: 'Daily delta net-positive while weekly delta remains negative — short-term demand fighting a longer structural overhead. Carries directional asymmetry.',
    logic: 'daily_cum_Δ > 0  ∧  weekly_cum_Δ < 0  →  long (+2)',
    type: 'Multi-TF',
  },
  {
    id: 'C', weight: '±2.0', color: C.purple,
    name: 'High Delta / Thin Volume',
    desc: 'Delta z-score extreme but volume well below average — a small crowd moving price, not sustained institutional flow. Mean-reversion candidate.',
    logic: 'cum_Δ_z > 1.5  ∧  vol_ratio < 0.7  →  mean-revert (±2)',
    type: 'Reversion',
  },
  {
    id: 'D', weight: '±1.0', color: C.blue,
    name: 'Context Gate',
    desc: 'Low-vol regime (below 33rd percentile, expanding) or late session >300 min RTH amplifies signal clarity. Quieter tape = cleaner delta reads.',
    logic: 'rv_30m < expand_33pct  ∨  rth_min ≥ 300  →  delta dir (±1)',
    type: 'Context',
  },
  {
    id: 'E', weight: '±0.5', color: C.pink,
    name: 'Acceleration Confirmation',
    desc: 'Second derivative of delta. If order flow is actively accelerating in the signal direction it adds 0.5 — momentum building, not fading.',
    logic: 'sign(daily_accel) = sign(cum_Δ)  →  +0.5 in signal direction',
    type: 'Momentum',
  },
];

const V_COMPARE = [
  { m: 'OOS P&L',        orig: '−$6,772', va: '+$4,570', origN: -6772, vaN: 4570  },
  { m: 'OOS Sharpe',     orig: '−1.77',   va: '+0.85',   origN: -1.77, vaN: 0.85  },
  { m: 'OOS Win Rate',   orig: '55.3%',   va: '51.2%',   origN: 55.3,  vaN: 51.2, neutral: true },
  { m: 'Profit Factor',  orig: '0.88',    va: '1.09',    origN: 0.88,  vaN: 1.09  },
  { m: 'OOS Trades',     orig: '199',     va: '125',     origN: 199,   vaN: 125,  neutral: true },
];

const MBO_ROWS = [
  { n: 'Version A baseline',                    pnl: 4843,  sh: 0.89, tr: 127, pf: 1.09, live: false },
  { n: '+ Low Phantom',                         pnl: 13756, sh: 6.50, tr: 76,  pf: 1.75, live: false },
  { n: '+ Low Rapid Cancel',                    pnl: 9145,  sh: 2.04, tr: 95,  pf: 1.24, live: false },
  { n: '+ Net Pressure Agrees',                 pnl: 9303,  sh: 2.31, tr: 88,  pf: 1.27, live: false },
  { n: '+ High Entropy',                        pnl: 8328,  sh: 2.32, tr: 88,  pf: 1.30, live: false },
  { n: '+ Low Rapid Cancel + Pressure Agrees',  pnl: 17573, sh: 6.70, tr: 42,  pf: 2.93, live: true  },
  { n: '+ Low Phantom + High Entropy',          pnl: 11109, sh: 5.62, tr: 53,  pf: 1.86, live: false },
];

const PIPELINE = [
  { step: '01', label: 'RAW INPUT',        color: C.blue,   items: ['Buy volume (1m)', 'Sell volume (1m)', 'Bar delta = B − S'] },
  { step: '02', label: 'CUMUL. DELTA',     color: C.purple, items: ['Daily reset', 'Weekly reset', 'Biweekly reset', '+ Velocity', '+ Acceleration'] },
  { step: '03', label: 'COMPOSITE ≥ ±3',  color: C.amber,  items: ['5 components', 'Score: −7.5 → +7.5', 'Entry thresh: ±3.0'] },
  { step: '04', label: 'MBO GATE',         color: C.green,  items: ['Low rapid cancel %', 'Net pressure agrees'] },
  { step: '05', label: 'TRADE',            color: C.red,    items: ['ES futures · 1 ct', 'Hold: 60 bars max', 'EOD always flat'] },
];

// ── Helpers ───────────────────────────────────────────────────────────
function fmtPnl(n) {
  return n >= 0 ? `+$${n.toLocaleString()}` : `−$${Math.abs(n).toLocaleString()}`;
}

function Label({ text, color }) {
  return (
    <span style={{
      fontSize: 9, letterSpacing: '0.12em', fontFamily: C.mono,
      color, background: `${color}18`, border: `1px solid ${color}40`,
      padding: '2px 6px', borderRadius: 3,
    }}>{text}</span>
  );
}

function SectionTitle({ children, accent = C.green }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
      <div style={{ width: 3, height: 20, background: accent, borderRadius: 2, flexShrink: 0 }} />
      <div style={{ fontFamily: C.display, fontSize: 20, letterSpacing: '0.06em', color: C.text }}>
        {children}
      </div>
    </div>
  );
}

function MonoCell({ val, color = C.sub, last = false }) {
  return (
    <div style={{
      padding: '10px 14px',
      fontFamily: C.mono, fontSize: 12,
      color, textAlign: 'right',
      borderRight: !last ? `1px solid ${C.border}` : 'none',
    }}>{val}</div>
  );
}

// ── Main Component ────────────────────────────────────────────────────
export default function DeltaResearch({ isDarkMode }) {
  const [expandedComp, setExpandedComp] = useState(null);

  useEffect(() => {
    if (!document.getElementById('dr-fonts')) {
      const link = document.createElement('link');
      link.id = 'dr-fonts';
      link.rel = 'stylesheet';
      link.href = 'https://fonts.googleapis.com/css2?family=Bebas+Neue&family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap';
      document.head.appendChild(link);
    }
  }, []);

  return (
    <div style={{ background: C.bg, minHeight: '100vh', fontFamily: C.body, color: C.text, padding: '28px 24px 48px' }}>

      {/* ══ HEADER ══════════════════════════════════════════════════ */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 32, paddingBottom: 24, borderBottom: `1px solid ${C.border}` }}>
        <div>
          <div style={{ fontFamily: C.display, fontSize: 52, letterSpacing: '0.04em', lineHeight: 1, background: `linear-gradient(135deg, ${C.green}, ${C.blue})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            DELTA RESEARCH
          </div>
          <div style={{ fontFamily: C.mono, fontSize: 11, color: C.muted, marginTop: 8, letterSpacing: '0.1em' }}>
            ES FUTURES · CUMULATIVE ORDER FLOW SIGNAL · MBO MICROSTRUCTURE
          </div>
          <div style={{ marginTop: 10, fontSize: 13, color: C.sub, maxWidth: 540, lineHeight: 1.6 }}>
            Tracks the net imbalance of aggressive buy vs sell volume across 3 timeframes,
            decomposed into 9 signals and filtered through 2 live MBO microstructure gates.
          </div>
        </div>

        {/* Live strategy badge */}
        <div style={{
          background: `linear-gradient(135deg, #071812, #0a2018)`,
          border: `1px solid ${C.green}60`,
          borderRadius: 12, padding: '18px 22px', textAlign: 'right', minWidth: 220,
          boxShadow: `0 0 32px ${C.green}18, inset 0 1px 0 ${C.green}20`,
        }}>
          <div style={{ fontFamily: C.mono, fontSize: 9, color: `${C.green}cc`, letterSpacing: '0.18em', marginBottom: 6 }}>▶ LIVE STRATEGY</div>
          <div style={{ fontFamily: C.display, fontSize: 20, letterSpacing: '0.04em', color: C.text, lineHeight: 1.1 }}>LOW RC + PRESSURE</div>
          <div style={{ fontFamily: C.mono, fontSize: 28, color: C.green, fontWeight: 600, marginTop: 6, letterSpacing: '-0.02em' }}>$17,573</div>
          <div style={{ display: 'flex', gap: 14, marginTop: 8, justifyContent: 'flex-end' }}>
            {[['Sharpe', '+6.70'], ['Trades', '42'], ['PF', '2.93']].map(([k, v]) => (
              <div key={k} style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 9, color: C.muted, fontFamily: C.mono, letterSpacing: '0.08em' }}>{k}</div>
                <div style={{ fontSize: 13, color: C.green, fontFamily: C.mono, fontWeight: 600 }}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ══ PIPELINE FLOW ════════════════════════════════════════════ */}
      <div style={{ marginBottom: 32 }}>
        <SectionTitle>Signal Architecture</SectionTitle>
        <div style={{ display: 'flex', alignItems: 'stretch' }}>
          {PIPELINE.map((s, i) => (
            <React.Fragment key={s.step}>
              <div style={{
                flex: 1, background: C.surface, border: `1px solid ${C.border}`,
                borderRadius: i === 0 ? '8px 0 0 8px' : i === PIPELINE.length - 1 ? '0 8px 8px 0' : 0,
                borderLeft: i > 0 ? 'none' : undefined,
                padding: '14px 16px',
              }}>
                <div style={{ fontFamily: C.mono, fontSize: 9, color: s.color, letterSpacing: '0.12em', marginBottom: 8 }}>
                  STEP {s.step}
                </div>
                <div style={{ fontFamily: C.display, fontSize: 13, letterSpacing: '0.05em', marginBottom: 10, color: C.text }}>
                  {s.label}
                </div>
                {s.items.map(item => (
                  <div key={item} style={{ fontSize: 11, color: C.muted, marginBottom: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ width: 3, height: 3, borderRadius: '50%', background: s.color, flexShrink: 0, display: 'inline-block' }} />
                    {item}
                  </div>
                ))}
              </div>
              {i < PIPELINE.length - 1 && (
                <div style={{ display: 'flex', alignItems: 'center', flexShrink: 0, background: C.surface, borderTop: `1px solid ${C.border}`, borderBottom: `1px solid ${C.border}`, padding: '0 2px' }}>
                  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                    <path d="M2 10 H14 M10 4 L16 10 L10 16" stroke={C.muted} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* ══ TWO-COLUMN: COMPONENTS + VERSION A ══════════════════════ */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 360px', gap: 24, marginBottom: 32 }}>

        {/* Composite Score Components */}
        <div>
          <SectionTitle accent={C.amber}>Composite Score Components</SectionTitle>
          <div style={{ fontSize: 12, color: C.muted, marginBottom: 12, fontFamily: C.mono }}>
            Click any row to expand logic · Entry fires when |score| ≥ 3.0
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {COMPONENTS.map((c) => {
              const open = expandedComp === c.id;
              return (
                <div
                  key={c.id}
                  onClick={() => setExpandedComp(open ? null : c.id)}
                  style={{
                    background: open ? `${c.color}0c` : C.surface,
                    border: `1px solid ${open ? c.color + '70' : C.border}`,
                    borderRadius: 8, padding: '12px 16px',
                    cursor: 'pointer',
                    transition: 'border-color 0.15s, background 0.15s',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      <div style={{
                        width: 30, height: 30, borderRadius: 6, flexShrink: 0,
                        background: `${c.color}18`, border: `1px solid ${c.color}50`,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontFamily: C.mono, fontSize: 13, fontWeight: 600, color: c.color,
                      }}>{c.id}</div>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 500, color: C.text, display: 'flex', alignItems: 'center', gap: 8 }}>
                          {c.name}
                          <Label text={c.type} color={c.color} />
                        </div>
                        {!open && <div style={{ fontSize: 11, color: C.muted, marginTop: 2, maxWidth: 380 }}>{c.desc.slice(0, 72)}…</div>}
                      </div>
                    </div>
                    <div style={{ fontFamily: C.mono, fontSize: 17, fontWeight: 600, color: c.color, whiteSpace: 'nowrap', marginLeft: 16 }}>
                      {c.weight}
                    </div>
                  </div>
                  {open && (
                    <div style={{ marginTop: 12, paddingTop: 12, borderTop: `1px solid ${C.border}` }}>
                      <div style={{ fontSize: 12, color: C.sub, lineHeight: 1.65, marginBottom: 10 }}>{c.desc}</div>
                      <div style={{
                        fontFamily: C.mono, fontSize: 11, color: c.color,
                        background: `${c.color}08`, padding: '8px 12px',
                        borderRadius: 4, borderLeft: `2px solid ${c.color}`,
                        letterSpacing: '0.02em',
                      }}>{c.logic}</div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Version A Comparison */}
        <div>
          <SectionTitle accent={C.blue}>Version A vs Original</SectionTitle>
          <div style={{ fontSize: 12, color: C.muted, marginBottom: 12, fontFamily: C.mono }}>
            hold=30, opp=2.0  →  hold=60, no premature exits
          </div>
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, overflow: 'hidden', marginBottom: 12 }}>
            {/* Header */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 90px 90px', background: C.raised, borderBottom: `1px solid ${C.border}` }}>
              {['Metric', 'Original', 'Version A'].map((h, i) => (
                <div key={h} style={{
                  padding: '9px 12px', fontSize: 9,
                  fontFamily: C.mono, letterSpacing: '0.12em',
                  color: i === 2 ? C.green : C.muted,
                  borderRight: i < 2 ? `1px solid ${C.border}` : 'none',
                  textAlign: i > 0 ? 'right' : 'left',
                }}>{h}</div>
              ))}
            </div>
            {V_COMPARE.map((row, i) => (
              <div key={row.m} style={{
                display: 'grid', gridTemplateColumns: '1fr 90px 90px',
                borderBottom: i < V_COMPARE.length - 1 ? `1px solid ${C.border}` : 'none',
              }}>
                <div style={{ padding: '10px 12px', fontSize: 12, color: C.sub, borderRight: `1px solid ${C.border}` }}>{row.m}</div>
                <div style={{
                  padding: '10px 12px', fontFamily: C.mono, fontSize: 12,
                  color: row.neutral ? C.muted : row.origN < 0 ? C.red : C.muted,
                  textAlign: 'right', borderRight: `1px solid ${C.border}`,
                }}>{row.orig}</div>
                <div style={{
                  padding: '10px 12px', fontFamily: C.mono, fontSize: 12,
                  color: row.neutral ? C.muted : row.vaN > row.origN ? C.green : C.muted,
                  textAlign: 'right',
                }}>{row.va}</div>
              </div>
            ))}
          </div>

          {/* Note card */}
          <div style={{ padding: '12px 14px', background: `${C.amber}08`, border: `1px solid ${C.amber}28`, borderRadius: 8, fontSize: 12, color: C.sub, lineHeight: 1.7 }}>
            <span style={{ color: C.amber, fontWeight: 600 }}>+$11,342 OOS improvement</span> from holding longer and removing premature exits. Confidence interval is wide [−$554, +$766] and P(&gt;0) = 63.8%. This is a <span style={{ color: C.sub }}>weak positive edge</span> — not a cash machine. 10K drawdown is real.
          </div>

          {/* Caveats */}
          <div style={{ marginTop: 12, padding: '12px 14px', background: `${C.red}06`, border: `1px solid ${C.red}20`, borderRadius: 8 }}>
            <div style={{ fontFamily: C.mono, fontSize: 9, color: C.red, letterSpacing: '0.12em', marginBottom: 8 }}>KNOWN CAVEATS</div>
            {[
              'Strategy selected on same 43-day IS window',
              '~20 hypotheses tested (multiple testing risk)',
              'Single regime: bullish Dec 2025 – Jan 2026',
              'No bear market / FOMC shock in this sample',
            ].map(c => (
              <div key={c} style={{ fontSize: 11, color: C.muted, marginBottom: 4, display: 'flex', gap: 6 }}>
                <span style={{ color: `${C.red}90` }}>!</span>{c}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ══ MBO FILTER RESULTS TABLE ════════════════════════════════ */}
      <div style={{ marginBottom: 32 }}>
        <SectionTitle>MBO Filter Results</SectionTitle>
        <div style={{ fontSize: 12, color: C.muted, marginBottom: 12, fontFamily: C.mono }}>
          41-day MBO period · Jan 19 – Mar 16 · Version A hold=60 · all variants use expanding median thresholds
        </div>
        <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, overflow: 'hidden' }}>
          {/* Header */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 130px 88px 80px 72px', background: C.raised, borderBottom: `1px solid ${C.border}` }}>
            {['Strategy', 'P&L', 'Sharpe', 'Trades', 'PF'].map((h, i) => (
              <div key={h} style={{
                padding: '10px 14px', fontSize: 9,
                fontFamily: C.mono, letterSpacing: '0.12em',
                color: C.muted, textAlign: i > 0 ? 'right' : 'left',
                borderRight: i < 4 ? `1px solid ${C.border}` : 'none',
              }}>{h}</div>
            ))}
          </div>
          {/* Rows */}
          {MBO_ROWS.map((row, i) => {
            const isBase = i === 0;
            return (
              <div key={row.n} style={{
                display: 'grid', gridTemplateColumns: '1fr 130px 88px 80px 72px',
                borderBottom: i < MBO_ROWS.length - 1 ? `1px solid ${C.border}` : 'none',
                background: row.live ? `${C.green}07` : 'transparent',
                position: 'relative',
              }}>
                {row.live && (
                  <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 3, background: C.green, borderRadius: '2px 0 0 2px' }} />
                )}
                <div style={{
                  padding: '11px 14px 11px 18px', fontSize: 13,
                  color: row.live ? C.green : isBase ? C.sub : C.text,
                  display: 'flex', alignItems: 'center', gap: 8,
                  fontWeight: row.live ? 500 : 400,
                }}>
                  {row.live && (
                    <span style={{
                      fontSize: 8, letterSpacing: '0.12em', fontFamily: C.mono,
                      background: C.green, color: '#000',
                      padding: '2px 6px', borderRadius: 3, flexShrink: 0,
                    }}>LIVE</span>
                  )}
                  {row.n}
                </div>
                <MonoCell val={fmtPnl(row.pnl)} color={row.live ? C.green : row.pnl > MBO_ROWS[0].pnl ? '#86efac' : C.sub} />
                <MonoCell val={`+${row.sh.toFixed(2)}`} color={row.live ? C.green : row.sh >= 2.0 ? '#86efac' : C.sub} />
                <MonoCell val={String(row.tr)} color={C.muted} />
                <MonoCell val={row.pf.toFixed(2)} color={row.live ? C.green : row.pf >= 1.5 ? '#86efac' : C.sub} last />
              </div>
            );
          })}
        </div>
      </div>

      {/* ══ FILTER DEEP DIVE ════════════════════════════════════════ */}
      <div>
        <SectionTitle>Live Filter Logic</SectionTitle>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>

          {/* Filter Card: Low Rapid Cancel */}
          <FilterCard
            color={C.blue}
            name="Low Rapid Cancel %"
            source="mbo_book_pressure_1m"
            field="rapid_cancel_pct"
            gate="< expanding median"
            badge="HFT NOISE FILTER"
            what="Percentage of limit orders added and cancelled within <100ms. High values indicate spoofing / phantom liquidity."
            why="When rapid_cancel_pct is LOW, the visible order book reflects genuine intent — real participants are staying put. Delta signals gain credibility in these conditions."
            gate_text="Enter only when rapid_cancel_pct < its expanding historical median. Filters ~26% of base signals, keeps the quieter-book bars."
            impact="+$4,302 standalone vs baseline"
          />

          {/* Filter Card: Net Pressure Agrees */}
          <FilterCard
            color={C.green}
            name="Net Add Pressure Agrees"
            source="mbo_book_pressure_1m"
            field="net_add_pressure"
            gate="sign = signal direction"
            badge="DIRECTIONAL CONFIRM"
            what="Net directional bias in order additions at L1: positive = more aggressive bid adds, negative = more aggressive ask adds."
            why="If cumulative delta says LONG but institutions are actively adding sell-side pressure, the delta signal is swimming against real-time book activity. Agreement = confluence."
            gate_text="Enter only when sign(net_add_pressure) matches delta composite direction. Filters ~31% of base signals, retains high-confluence bars."
            impact="+$4,460 standalone vs baseline"
          />
        </div>

        {/* Combined summary */}
        <div style={{
          padding: '18px 20px',
          background: `linear-gradient(135deg, ${C.green}08, ${C.blue}06)`,
          border: `1px solid ${C.green}30`,
          borderRadius: 8, fontSize: 13, color: C.sub, lineHeight: 1.8,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
            <div style={{ fontFamily: C.display, fontSize: 16, letterSpacing: '0.06em', color: C.green }}>COMBINED EFFECT</div>
            <div style={{ flex: 1, height: 1, background: `${C.green}25` }} />
            <span style={{ fontFamily: C.mono, fontSize: 12, color: C.green }}>$17,573 · Sharpe +6.70 · PF 2.93</span>
          </div>
          The two filters are largely <span style={{ color: C.text }}>orthogonal</span> — Low Rapid Cancel identifies bars with clean book liquidity,
          Net Pressure Agrees identifies bars where institutional order-add behavior confirms the delta direction.
          When <span style={{ color: C.text }}>both</span> gate simultaneously, 42 trades remain from the 127-trade baseline.
          Each represents a bar where (1) cumulative delta hit threshold, (2) the book isn't polluted with HFT noise,
          and (3) real-time L1 add pressure is pointing the same way.{' '}
          <span style={{ color: C.amber }}>The trade frequency drop is the feature, not a bug</span> — selectivity
          drives the Sharpe from +0.89 to +6.70 and the profit factor from 1.09 to 2.93.
        </div>
      </div>

    </div>
  );
}

// ── Filter Card Sub-component ─────────────────────────────────────────
function FilterCard({ color, name, source, field, gate, badge, what, why, gate_text, impact }) {
  return (
    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, overflow: 'hidden' }}>
      {/* Card header */}
      <div style={{
        padding: '14px 18px', borderBottom: `1px solid ${C.border}`,
        background: `${color}08`,
        display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <Label text={badge} color={color} />
          </div>
          <div style={{ fontFamily: C.display, fontSize: 20, letterSpacing: '0.04em', color, lineHeight: 1 }}>
            {name}
          </div>
          <div style={{ fontFamily: C.mono, fontSize: 10, color: C.muted, marginTop: 4 }}>
            {source} · {field}
          </div>
        </div>
        <div style={{
          fontFamily: C.mono, fontSize: 10, color,
          background: `${color}15`, border: `1px solid ${color}40`,
          padding: '5px 9px', borderRadius: 4, whiteSpace: 'nowrap', marginLeft: 12,
        }}>{gate}</div>
      </div>

      {/* Card body */}
      <div style={{ padding: '14px 18px' }}>
        {[
          { label: 'WHAT IT MEASURES', text: what },
          { label: 'WHY IT FILTERS WELL', text: why },
          { label: 'GATE LOGIC', text: gate_text },
        ].map(({ label, text }) => (
          <div key={label} style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 8, letterSpacing: '0.15em', fontFamily: C.mono, color: `${color}cc`, marginBottom: 5 }}>{label}</div>
            <div style={{ fontSize: 12, color: C.sub, lineHeight: 1.65 }}>{text}</div>
          </div>
        ))}
        <div style={{
          fontFamily: C.mono, fontSize: 12, color: C.green,
          background: `${C.green}0a`, padding: '7px 10px',
          borderRadius: 4, borderLeft: `2px solid ${C.green}`,
        }}>{impact}</div>
      </div>
    </div>
  );
}
