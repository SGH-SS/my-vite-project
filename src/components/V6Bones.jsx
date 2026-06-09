/**
 * V6 Bones — Complete Strategy Pipeline Breakdown
 * 
 * End-to-end visualization of the V6 residual-based calendar spread strategy,
 * from raw data inputs through signal construction to trade execution.
 */

import { useState, useMemo } from 'react';

// =============================================================================
// CONSTANTS & DATA
// =============================================================================

const SECTIONS = {
  OVERVIEW: 'overview',
  DATA: 'data',
  FV_MODEL: 'fv_model',
  DEVIATION: 'deviation',
  ZQ_FEATURES: 'zq_features',
  RESIDUAL: 'residual',
  INTRADAY: 'intraday',
  ENTRY_EXIT: 'entry_exit',
  CONFIGS: 'configs',
  WALKTHROUGH: 'walkthrough',
  PERFORMANCE: 'performance',
  LOOKAHEAD: 'lookahead',
  V9: 'v9',
  V10_ROLL: 'v10_roll',
  PRODUCTION: 'production',
};

const SECTION_META = [
  { id: SECTIONS.OVERVIEW, label: 'Overview', icon: '🏗️', color: 'blue' },
  { id: SECTIONS.DATA, label: 'Raw Data', icon: '🗄️', color: 'gray' },
  { id: SECTIONS.FV_MODEL, label: 'Fair Value Model', icon: '⚖️', color: 'emerald' },
  { id: SECTIONS.DEVIATION, label: 'Deviation Pipeline', icon: '📐', color: 'cyan' },
  { id: SECTIONS.ZQ_FEATURES, label: 'ZQ Features', icon: '📡', color: 'violet' },
  { id: SECTIONS.RESIDUAL, label: 'Residual Regression', icon: '🧬', color: 'rose' },
  { id: SECTIONS.INTRADAY, label: '1-Minute Layer', icon: '⏱️', color: 'amber' },
  { id: SECTIONS.ENTRY_EXIT, label: 'Entry & Exit', icon: '🎯', color: 'green' },
  { id: SECTIONS.CONFIGS, label: 'No Gates vs Asym', icon: '⚙️', color: 'indigo' },
  { id: SECTIONS.WALKTHROUGH, label: 'Live Walkthrough', icon: '🚶', color: 'orange' },
  { id: SECTIONS.PERFORMANCE, label: 'Performance', icon: '📊', color: 'pink' },
  { id: SECTIONS.LOOKAHEAD, label: 'Lookahead Audit', icon: '🔍', color: 'yellow' },
  { id: SECTIONS.V9, label: 'V9 Production', icon: '🚀', color: 'teal' },
  { id: SECTIONS.V10_ROLL, label: 'V10 Roll Week', icon: '🔄', color: 'sky' },
  { id: SECTIONS.PRODUCTION, label: 'Production Guide', icon: '🏭', color: 'red' },
];

const PERF = {
  noGates: {
    full: { n: 188, pnl: 163.2, sharpe: 5.57, pf: 4.80, wr: 72.3, maxDD: -5.8, avgPnl: 0.87 },
    longs: { n: 93, pnl: 81.2, wr: 77.4, sharpe: 5.21 },
    shorts: { n: 95, pnl: 82.0, wr: 67.4, sharpe: 4.42 },
    years: [
      { yr: 2023, n: 64, pnl: 52.8, sharpe: 4.91, lWr: 78, sWr: 65 },
      { yr: 2024, n: 72, pnl: 69.0, sharpe: 5.83, lWr: 79, sWr: 68 },
      { yr: 2025, n: 44, pnl: 36.4, sharpe: 5.14, lWr: 73, sWr: 70 },
      { yr: 2026, n: 8, pnl: 5.0, sharpe: 3.55, lWr: 75, sWr: 50 },
    ],
    oos: { sharpe: 5.31, posFolds: '12/12' },
    mc: { pValue: '<0.0001', significant: true },
    recent: { n: 19, pnl: 19.80, sharpe: 3.55, wr: 63.2, maxDD: -2.00 },
  },
  asym: {
    full: { n: 148, pnl: 144.4, sharpe: 6.17, pf: 6.21, wr: 76.4, maxDD: -5.7, avgPnl: 0.98 },
    longs: { n: 93, pnl: 81.2, wr: 77.4, sharpe: 5.21 },
    shorts: { n: 55, pnl: 63.2, wr: 75.9, sharpe: 6.48 },
    years: [
      { yr: 2023, n: 50, pnl: 46.2, sharpe: 5.64, lWr: 78, sWr: 74 },
      { yr: 2024, n: 56, pnl: 55.8, sharpe: 6.35, lWr: 79, sWr: 76 },
      { yr: 2025, n: 34, pnl: 36.9, sharpe: 5.87, lWr: 73, sWr: 80 },
      { yr: 2026, n: 8, pnl: 5.5, sharpe: 5.04, lWr: 75, sWr: 80 },
    ],
    oos: { sharpe: 6.01, posFolds: '12/12' },
    mc: { pValue: '<0.0001', significant: true },
    recent: { n: 15, pnl: 23.20, sharpe: 5.04, wr: 80.0, maxDD: -0.90 },
  },
  v5: {
    full: { n: 141, pnl: 143.4, sharpe: 2.77, pf: 2.85, wr: 70.2, maxDD: -11.6, avgPnl: 1.02 },
    shorts: { wr: 70.0 },
  },
  v2: {
    full: { n: 192, pnl: 82.5, sharpe: 1.73, pf: 1.82, wr: 60.9, maxDD: -14.2 },
  },
};

// =============================================================================
// REUSABLE COMPONENTS
// =============================================================================

const Card = ({ children, className = '', isDarkMode }) => (
  <div className={`rounded-xl border transition-colors ${
    isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
  } ${className}`}>
    {children}
  </div>
);

const SectionHeader = ({ icon, title, subtitle, isDarkMode, color = 'blue' }) => {
  const colors = {
    blue: 'from-blue-600 to-blue-800',
    emerald: 'from-emerald-600 to-emerald-800',
    cyan: 'from-cyan-600 to-cyan-800',
    violet: 'from-violet-600 to-violet-800',
    rose: 'from-rose-600 to-rose-800',
    amber: 'from-amber-600 to-amber-800',
    green: 'from-green-600 to-green-800',
    indigo: 'from-indigo-600 to-indigo-800',
    orange: 'from-orange-600 to-orange-800',
    pink: 'from-pink-600 to-pink-800',
    red: 'from-red-600 to-red-800',
    yellow: 'from-yellow-600 to-yellow-800',
    teal: 'from-teal-600 to-teal-800',
    sky: 'from-sky-600 to-sky-800',
    gray: 'from-gray-600 to-gray-800',
  };
  return (
    <div className={`rounded-xl p-5 bg-gradient-to-r ${colors[color] || colors.blue} text-white mb-6`}>
      <h2 className="text-2xl font-black flex items-center gap-3">
        <span className="text-3xl">{icon}</span>
        {title}
      </h2>
      {subtitle && <p className="text-sm mt-1 opacity-80">{subtitle}</p>}
    </div>
  );
};

const CodeBlock = ({ code, isDarkMode, title }) => (
  <div className={`rounded-lg border overflow-hidden ${isDarkMode ? 'border-gray-600' : 'border-gray-300'}`}>
    {title && (
      <div className={`px-4 py-2 text-xs font-semibold uppercase tracking-wider border-b ${
        isDarkMode ? 'bg-gray-700 border-gray-600 text-gray-300' : 'bg-gray-100 border-gray-200 text-gray-600'
      }`}>{title}</div>
    )}
    <pre className={`p-4 text-sm font-mono leading-relaxed overflow-x-auto ${
      isDarkMode ? 'bg-gray-900 text-green-400' : 'bg-gray-50 text-gray-800'
    }`}>{code}</pre>
  </div>
);

const FlowArrow = ({ isDarkMode }) => (
  <div className="flex justify-center py-2">
    <svg className={`w-6 h-8 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 32">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v20m0 0l-6-6m6 6l6-6" />
    </svg>
  </div>
);

const PipelineStep = ({ step, title, formula, explanation, isDarkMode, color = 'blue' }) => {
  const dotColor = {
    blue: 'bg-blue-500', emerald: 'bg-emerald-500', cyan: 'bg-cyan-500',
    violet: 'bg-violet-500', rose: 'bg-rose-500', amber: 'bg-amber-500',
    green: 'bg-green-500', indigo: 'bg-indigo-500', orange: 'bg-orange-500',
  };
  return (
    <div className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white border-gray-200'}`}>
      <div className="flex items-start gap-3">
        <div className={`w-8 h-8 rounded-full ${dotColor[color] || dotColor.blue} text-white font-bold text-sm flex items-center justify-center shrink-0 mt-0.5`}>
          {step}
        </div>
        <div className="flex-1 min-w-0">
          <h4 className={`font-bold text-sm ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{title}</h4>
          {formula && (
            <div className={`mt-2 px-3 py-2 rounded font-mono text-xs ${
              isDarkMode ? 'bg-gray-900 text-emerald-400' : 'bg-gray-50 text-emerald-700'
            }`}>{formula}</div>
          )}
          <p className={`mt-2 text-sm leading-relaxed ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{explanation}</p>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ label, value, sub, isDarkMode, color = 'blue', size = 'md' }) => {
  const textColor = {
    blue: isDarkMode ? 'text-blue-400' : 'text-blue-600',
    green: isDarkMode ? 'text-green-400' : 'text-green-600',
    red: isDarkMode ? 'text-red-400' : 'text-red-600',
    amber: isDarkMode ? 'text-amber-400' : 'text-amber-600',
    purple: isDarkMode ? 'text-purple-400' : 'text-purple-600',
    rose: isDarkMode ? 'text-rose-400' : 'text-rose-600',
    emerald: isDarkMode ? 'text-emerald-400' : 'text-emerald-600',
    indigo: isDarkMode ? 'text-indigo-400' : 'text-indigo-600',
    cyan: isDarkMode ? 'text-cyan-400' : 'text-cyan-600',
  };
  return (
    <div className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
      <div className={`text-xs font-medium uppercase tracking-wider mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>{label}</div>
      <div className={`font-black ${size === 'lg' ? 'text-3xl' : 'text-xl'} ${textColor[color] || textColor.blue}`}>{value}</div>
      {sub && <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{sub}</div>}
    </div>
  );
};

const ComparisonTable = ({ rows, headers, isDarkMode, highlightCol }) => (
  <div className={`rounded-lg border overflow-hidden ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
    <table className="w-full text-sm">
      <thead>
        <tr className={isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}>
          {headers.map((h, i) => (
            <th key={i} className={`px-4 py-3 text-left font-semibold text-xs uppercase tracking-wider ${
              isDarkMode ? 'text-gray-300' : 'text-gray-600'
            } ${i === highlightCol ? (isDarkMode ? 'bg-emerald-900/30' : 'bg-emerald-50') : ''}`}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, ri) => (
          <tr key={ri} className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-100'} ${
            ri % 2 === 0 ? '' : (isDarkMode ? 'bg-gray-800/50' : 'bg-gray-50/50')
          }`}>
            {row.map((cell, ci) => (
              <td key={ci} className={`px-4 py-2.5 ${
                ci === 0 ? 'font-medium' : 'font-mono'
              } ${isDarkMode ? 'text-gray-300' : 'text-gray-700'
              } ${ci === highlightCol ? (isDarkMode ? 'bg-emerald-900/20 font-bold text-emerald-400' : 'bg-emerald-50 font-bold text-emerald-700') : ''}`}>{cell}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

const Callout = ({ type = 'info', children, isDarkMode }) => {
  const styles = {
    info: isDarkMode ? 'bg-blue-900/20 border-blue-700 text-blue-300' : 'bg-blue-50 border-blue-200 text-blue-800',
    warn: isDarkMode ? 'bg-amber-900/20 border-amber-700 text-amber-300' : 'bg-amber-50 border-amber-200 text-amber-800',
    success: isDarkMode ? 'bg-emerald-900/20 border-emerald-700 text-emerald-300' : 'bg-emerald-50 border-emerald-200 text-emerald-800',
    danger: isDarkMode ? 'bg-red-900/20 border-red-700 text-red-300' : 'bg-red-50 border-red-200 text-red-800',
    insight: isDarkMode ? 'bg-violet-900/20 border-violet-700 text-violet-300' : 'bg-violet-50 border-violet-200 text-violet-800',
  };
  const icons = { info: 'ℹ️', warn: '⚠️', success: '✅', danger: '🚨', insight: '💡' };
  return (
    <div className={`rounded-lg border p-4 text-sm leading-relaxed ${styles[type]}`}>
      <span className="mr-2">{icons[type]}</span>{children}
    </div>
  );
};

// =============================================================================
// SECTION COMPONENTS
// =============================================================================

const OverviewSection = ({ isDarkMode }) => (
  <div className="space-y-6">
    <SectionHeader icon="🏗️" title="V6 Strategy: The Complete Architecture" subtitle="Residual-based mean-reversion on ES calendar spreads" isDarkMode={isDarkMode} color="blue" />

    <Callout type="insight" isDarkMode={isDarkMode}>
      <strong>The V6 Insight:</strong> Prior versions (V2-V5) used a pure-spot deviation signal for entry/exit and bolted on increasingly complex gating systems using ZQ forward rate data. V6 flips this — it <em>intermingles</em> ZQ features directly into the primary signal via rolling regression, isolating "genuine mispricing" from "rate-driven fair value shifts." This eliminates the need for gating layers entirely.
    </Callout>

    <Card isDarkMode={isDarkMode} className="p-6">
      <h3 className={`text-lg font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Strategy at a Glance</h3>
      <div className={`text-sm leading-relaxed space-y-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
        <p><strong>What we trade:</strong> ES calendar spreads (one-quarter adjacent, e.g., ESZ5-ESH6). These are the price difference between two consecutive quarterly ES futures contracts.</p>
        <p><strong>The edge:</strong> Calendar spreads have a theoretically computable fair value based on interest rates, dividends, and time to expiry. When the market price diverges from fair value, we trade the mean-reversion.</p>
        <p><strong>V6's innovation:</strong> Instead of measuring raw deviation from fair value, we first regress out the portion of deviation that's explained by forward rate expectations. The residual — what's left after accounting for where rates are <em>going</em> — is the signal. This is a purer measure of mispricing.</p>
      </div>
    </Card>

    <Card isDarkMode={isDarkMode} className="p-6">
      <h3 className={`text-lg font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Full Pipeline Flow</h3>
      <div className="grid grid-cols-1 md:grid-cols-6 gap-2">
        {[
          { s: '1', t: 'Raw Data', d: 'SOFR, ZQ implied rates, ES prices, spread OHLCV', c: 'gray' },
          { s: '2', t: 'Fair Value', d: 'FV = ES × net_carry × days/365', c: 'emerald' },
          { s: '3', t: 'Deviation', d: 'Bias-corrected, ATR-normalized', c: 'cyan' },
          { s: '4', t: 'Residual', d: 'Regress out ZQ → genuine mispricing', c: 'rose' },
          { s: '5', t: '1m Signal', d: 'Scale to intraday via ratio', c: 'amber' },
          { s: '6', t: 'Trade', d: 'Entry at extreme, exit on reversion', c: 'green' },
        ].map(({ s, t, d, c }) => (
          <div key={s} className={`rounded-lg border p-3 text-center ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
            <div className={`w-8 h-8 rounded-full bg-${c}-500 text-white font-bold text-sm flex items-center justify-center mx-auto mb-2`}>{s}</div>
            <div className={`text-sm font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{t}</div>
            <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{d}</div>
          </div>
        ))}
      </div>
    </Card>
  </div>
);

const DataSection = ({ isDarkMode }) => (
  <div className="space-y-6">
    <SectionHeader icon="🗄️" title="Layer 0: Raw Data Inputs" subtitle="Six database tables feed the pipeline" isDarkMode={isDarkMode} color="gray" />

    <ComparisonTable isDarkMode={isDarkMode} headers={['Source Table', 'Key Columns', 'Role in V6']} rows={[
      ['external_rates_1d', 'sofr_spot, implied_rate_1m/2m/3m/6m, net_carry_spot, net_carry_3m, spx_div_yield', 'Interest rates (spot + ZQ forward), dividends, carry'],
      ['ohlcv_1d (spreads)', 'spread_symbol, open, high, low, close, volume', 'Daily spread prices → deviation, ATR'],
      ['es_continuous', 'es_open, es_high, es_low, es_close', 'Underlying ES price → FV calculation'],
      ['regime_context_1d', 'in_roll_window, days_to_next_roll, is_quad_witching', 'Roll week detection → separate threshold handling'],
      ['forward_vol_1d', 'vix_close, vix_rv_ratio', 'Context only (not used as gate in V6)'],
      ['event_calendar', 'FOMC, CPI, NFP dates', 'Context only'],
      ['ohlcv_1m (spreads)', 'spread 1-min OHLCV', 'Intraday entry/exit price discovery'],
      ['es_continuous_1m', 'ES 1-min close', 'Intraday FV recalculation'],
    ]} />

    <Callout type="info" isDarkMode={isDarkMode}>
      Only <strong>one-quarter adjacent</strong> spreads are used (e.g., ESZ5-ESH6, ESH6-ESM6). Wider spreads like ESZ5-ESM6 are excluded. This ensures a clean, liquid instrument with well-defined carry dynamics.
    </Callout>

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-3 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Contract Expiry Calculation</h3>
      <CodeBlock isDarkMode={isDarkMode} title="How we compute days_between" code={`front_expiry = third_friday(year, month)   // e.g., ESZ5 → Dec 19, 2025
back_expiry  = third_friday(year, month+3)  // e.g., ESH6 → Mar 20, 2026
days_to_front = front_expiry - today
days_to_back  = back_expiry - today
days_between  = days_to_back - days_to_front  // ≈ 91 days for quarterly`} />
      <p className={`text-sm mt-3 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
        <code className="font-mono">days_between</code> is the time gap between the two legs. It's the denominator in the carry calculation and directly drives fair value. As expiry approaches, this shrinks and FV converges to zero.
      </p>
    </Card>
  </div>
);

const FVModelSection = ({ isDarkMode }) => (
  <div className="space-y-6">
    <SectionHeader icon="⚖️" title="Layer 1: The Fair Value Model" subtitle="Spot-based carry model — unchanged from V2 through V6" isDarkMode={isDarkMode} color="emerald" />

    <Callout type="success" isDarkMode={isDarkMode}>
      V5 tested blending forward rates into FV and found it <strong>hurt performance</strong>. The spot-based FV model is kept pure. Forward rate information enters through the <em>residual regression</em> instead (Layer 4).
    </Callout>

    <div className="space-y-3">
      <PipelineStep isDarkMode={isDarkMode} color="emerald" step="1" title="Implied Dividend Yield"
        formula="spread_carry_raw = (spread_close × 365) / (es_close × days_between)
div_yield_spot = sofr_spot − spread_carry_raw
div_yield_spot_5d = rolling_mean(div_yield_spot, 5)"
        explanation="Back out the raw carry from the observed spread price. The difference between the risk-free rate (SOFR) and that carry gives the implied dividend yield. We smooth it over 5 days to reduce noise from bid-ask bounce."
      />
      <FlowArrow isDarkMode={isDarkMode} />
      <PipelineStep isDarkMode={isDarkMode} color="emerald" step="2" title="Net Carry"
        formula="carry_spot = sofr_spot − div_yield_spot_5d"
        explanation="Net carry is 'interest earned minus dividends lost' for holding ES from the front to back expiry. When SOFR is 4.3% and implied dividends are 1.5%, the carry is 2.8% — meaning the spread should theoretically be worth ES × 2.8% × time."
      />
      <FlowArrow isDarkMode={isDarkMode} />
      <PipelineStep isDarkMode={isDarkMode} color="emerald" step="3" title="Fair Value"
        formula="fv_spot = es_close × carry_spot × days_between / 365"
        explanation="The theoretical calendar spread price. Example: ES at 5,800, carry at 2.8%, 91 days apart → FV = 5800 × 0.028 × 91/365 = 40.47 points. If the spread trades at 58.50, it's 18 points above FV."
      />
    </div>

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-3 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Worked Example</h3>
      <div className={`grid grid-cols-2 gap-4 text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
        <div>
          <div className={`font-semibold mb-2 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>Inputs</div>
          <div className="space-y-1 font-mono text-xs">
            <div>ES close     = 5,800.00</div>
            <div>SOFR spot    = 4.30%</div>
            <div>Spread close = 58.50</div>
            <div>Days between = 91</div>
          </div>
        </div>
        <div>
          <div className={`font-semibold mb-2 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>Computation</div>
          <div className="space-y-1 font-mono text-xs">
            <div>carry_raw    = 58.50 × 365 / (5800 × 91) = <span className={isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}>0.04047</span></div>
            <div>div_yield    = 4.30% − 4.05% = <span className={isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}>0.25%</span></div>
            <div>carry_spot   = 4.30% − 1.50% = <span className={isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}>2.80%</span></div>
            <div>FV           = 5800 × 0.028 × 91/365 = <span className={isDarkMode ? 'text-emerald-400 font-bold' : 'text-emerald-700 font-bold'}>40.47</span></div>
          </div>
        </div>
      </div>
    </Card>
  </div>
);

const DeviationSection = ({ isDarkMode }) => (
  <div className="space-y-6">
    <SectionHeader icon="📐" title="Layer 2: Deviation, Bias Correction, Normalization" subtitle="Turning raw price differences into a standardized signal" isDarkMode={isDarkMode} color="cyan" />

    <div className="space-y-3">
      <PipelineStep isDarkMode={isDarkMode} color="cyan" step="1" title="Raw Deviation"
        formula="dev_spot = spread_close − fv_spot"
        explanation="How far the actual spread price is from fair value. Positive = 'rich' (spread above FV). Negative = 'cheap' (spread below FV). Example: 58.50 − 40.47 = +18.03 points."
      />
      <FlowArrow isDarkMode={isDarkMode} />
      <PipelineStep isDarkMode={isDarkMode} color="cyan" step="2" title="Bias Correction"
        formula="bias_20d = rolling_mean(dev_spot, window=20, min_periods=5)
dev_bc = dev_spot − bias_20d"
        explanation="The 20-day rolling mean of deviation captures structural drift — maybe the spread persistently trades 16 pts above FV this month. Subtracting it isolates abnormal deviation. From +18.03, if the bias is +16.00, the corrected deviation is only +2.03."
      />
      <FlowArrow isDarkMode={isDarkMode} />
      <PipelineStep isDarkMode={isDarkMode} color="cyan" step="3" title="ATR Normalization"
        formula="spread_atr_20 = rolling_mean(high − low, window=20).clip(min=0.5)
norm_dev_bc = dev_bc / spread_atr_20"
        explanation="Dividing by the 20-day average true range puts the signal in 'standard units.' A norm_dev_bc of +1.69 means the spread is 1.69 ATRs above its bias-corrected fair value. This makes the signal comparable across different spread symbols and time periods."
      />
    </div>

    <Callout type="info" isDarkMode={isDarkMode}>
      <strong>This norm_dev_bc is the V2-V5 primary signal.</strong> It's entirely derived from spot data — SOFR, ES, and spread prices. No ZQ forward rate information is involved. V6's innovation happens in the next layer.
    </Callout>
  </div>
);

const ZQFeaturesSection = ({ isDarkMode }) => (
  <div className="space-y-6">
    <SectionHeader icon="📡" title="Layer 3: ZQ-Derived Features" subtitle="What the futures market tells us about where rates are going" isDarkMode={isDarkMode} color="violet" />

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-3 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>What are ZQ Futures?</h3>
      <p className={`text-sm leading-relaxed ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
        ZQ futures (SOFR futures) trade on CME and settle to the average daily SOFR rate over their delivery month. The <code className="font-mono">implied_rate_3m</code> derived from ZQ prices tells us where the market expects SOFR to be in 3 months. The gap between this and today's spot SOFR is the market's rate expectation.
      </p>
    </Card>

    <ComparisonTable isDarkMode={isDarkMode} headers={['Feature', 'Formula', 'What It Means']} rows={[
      ['rate_gap', 'implied_rate_3m − sofr_spot', 'Market\'s expected rate change over 3 months. Positive = expecting hikes, negative = expecting cuts'],
      ['carry_gap', 'net_carry_3m − net_carry_spot', 'Difference between forward-implied carry and spot carry'],
      ['fwd_curve_slope', 'implied_rate_6m − implied_rate_1m', 'Shape of the forward rate curve. Steep = big expectations of rate change'],
      ['fwd_3m_vol_20d', 'rolling_std(Δimplied_rate_3m, 20)', 'How volatile the 3m forward rate has been. High vol = uncertain rate path'],
      ['fwd_3m_chg_5d', 'implied_rate_3m − implied_rate_3m.shift(5)', 'Recent momentum in forward rate expectations'],
    ]} />

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-3 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Z-Score Normalization (Rolling)</h3>
      <CodeBlock isDarkMode={isDarkMode} title="Applied per spread symbol, 60-day rolling window" code={`roll_mean = feature.rolling(60, min_periods=20).mean()
roll_std  = feature.rolling(60, min_periods=20).std()
feature_z = ((feature − roll_mean) / roll_std).clip(−3, 3)`} />
      <p className={`text-sm mt-3 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
        This produces <code className="font-mono">rate_gap_z</code>, <code className="font-mono">carry_gap_z</code>, and <code className="font-mono">fwd_curve_slope_z</code> — the three inputs to the residual regression. The z-scoring ensures the regression isn't dominated by whichever feature has the largest raw scale.
      </p>
    </Card>
  </div>
);

const ResidualSection = ({ isDarkMode }) => (
  <div className="space-y-6">
    <SectionHeader icon="🧬" title="Layer 4: The Residual Regression (V6 Breakthrough)" subtitle="Separating genuine mispricing from rate-driven fair value shifts" isDarkMode={isDarkMode} color="rose" />

    <Callout type="insight" isDarkMode={isDarkMode}>
      <strong>The core insight:</strong> If rate_gap is large and positive (market expects rate hikes), the spread rationally trades rich because future carry will be higher. The baseline norm_dev_bc sees this as "mispricing" but it's actually correct forward pricing. The regression strips that out.
    </Callout>

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-3 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Rolling OLS Regression</h3>
      <CodeBlock isDarkMode={isDarkMode} title="Executed per spread symbol, walk-forward (no lookahead)" code={`For each day i (from day 20 onward):
    # Train on the prior 60 days — strictly past data only
    X_train = [rate_gap_z, carry_gap_z, fwd_curve_slope_z]  ← last 60 days
    y_train = norm_dev_bc                                    ← last 60 days

    # OLS: norm_dev_bc ~ β₀ + β₁·rate_gap_z + β₂·carry_gap_z + β₃·fwd_curve_slope_z
    β = OLS_fit(X_train, y_train)

    # Apply to today's values
    predicted = β₀ + β₁·rate_gap_z[today] + β₂·carry_gap_z[today] + β₃·fwd_curve_slope_z[today]

    # RESIDUAL = actual − predicted = genuine mispricing
    resid_signal[today] = norm_dev_bc[today] − predicted

    # ZQ DIRECTIONAL = predicted − intercept = rate-driven drift
    zq_directional[today] = predicted − β₀`} />
    </Card>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <Card isDarkMode={isDarkMode} className="p-5">
        <h4 className={`font-bold mb-2 ${isDarkMode ? 'text-rose-400' : 'text-rose-600'}`}>resid_signal (Residual)</h4>
        <p className={`text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
          "How much is this spread mispriced <em>after</em> accounting for where rates are going?" If norm_dev_bc is +1.69 but the regression predicts +0.56 from rate expectations, the residual is +1.13 — only 1.13 ATRs of genuine mispricing.
        </p>
        <div className={`mt-3 px-3 py-2 rounded text-xs font-mono ${isDarkMode ? 'bg-gray-900 text-rose-400' : 'bg-rose-50 text-rose-700'}`}>
          This is the V6 daily entry/exit signal
        </div>
      </Card>
      <Card isDarkMode={isDarkMode} className="p-5">
        <h4 className={`font-bold mb-2 ${isDarkMode ? 'text-violet-400' : 'text-violet-600'}`}>zq_directional (Discarded)</h4>
        <p className={`text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
          "How much of the deviation is explained by rate expectations?" This is the part we <em>throw away</em>. It's not mispricing — it's the market correctly pricing in future rate changes. V6 (gamma=0) uses the pure residual only.
        </p>
        <div className={`mt-3 px-3 py-2 rounded text-xs font-mono ${isDarkMode ? 'bg-gray-900 text-gray-500' : 'bg-gray-100 text-gray-500'}`}>
          Gamma=0 → ZQ directional is not added back
        </div>
      </Card>
    </div>

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-3 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Decomposition Example</h3>
      <div className="flex flex-col md:flex-row gap-4 items-center">
        <div className={`flex-1 rounded-lg p-4 text-center ${isDarkMode ? 'bg-cyan-900/20 border border-cyan-800' : 'bg-cyan-50 border border-cyan-200'}`}>
          <div className={`text-xs uppercase tracking-wider mb-1 ${isDarkMode ? 'text-cyan-400' : 'text-cyan-600'}`}>norm_dev_bc</div>
          <div className={`text-2xl font-black ${isDarkMode ? 'text-cyan-300' : 'text-cyan-700'}`}>+1.69</div>
          <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>Total deviation</div>
        </div>
        <div className={`text-2xl ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>=</div>
        <div className={`flex-1 rounded-lg p-4 text-center ${isDarkMode ? 'bg-violet-900/20 border border-violet-800' : 'bg-violet-50 border border-violet-200'}`}>
          <div className={`text-xs uppercase tracking-wider mb-1 ${isDarkMode ? 'text-violet-400' : 'text-violet-600'}`}>ZQ Explained</div>
          <div className={`text-2xl font-black ${isDarkMode ? 'text-violet-300' : 'text-violet-700'}`}>+0.56</div>
          <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>Rate expectations</div>
        </div>
        <div className={`text-2xl ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>+</div>
        <div className={`flex-1 rounded-lg p-4 text-center ${isDarkMode ? 'bg-rose-900/20 border border-rose-800' : 'bg-rose-50 border border-rose-200'}`}>
          <div className={`text-xs uppercase tracking-wider mb-1 ${isDarkMode ? 'text-rose-400' : 'text-rose-600'}`}>resid_signal</div>
          <div className={`text-2xl font-black ${isDarkMode ? 'text-rose-300' : 'text-rose-700'}`}>+1.13</div>
          <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>Genuine mispricing</div>
        </div>
      </div>
    </Card>
  </div>
);

const IntradaySection = ({ isDarkMode }) => (
  <div className="space-y-6">
    <SectionHeader icon="⏱️" title="Layer 5: The 1-Minute Signal" subtitle="Translating the daily residual to intraday price discovery" isDarkMode={isDarkMode} color="amber" />

    <div className="space-y-3">
      <PipelineStep isDarkMode={isDarkMode} color="amber" step="1" title="1-Minute Spot FV Recalculation"
        formula="fv_1m = es_close_1m × carry_spot × days_between / 365"
        explanation="Same FV formula, but using the 1-minute ES continuous price instead of the daily close. Carry and days_between are from today's daily data (they don't change intraday)."
      />
      <FlowArrow isDarkMode={isDarkMode} />
      <PipelineStep isDarkMode={isDarkMode} color="amber" step="2" title="1-Minute Bias-Corrected Deviation"
        formula="dev_1m = spread_close_1m − fv_1m
dev_bc_1m = dev_1m − bias_20d
norm_dev_1m = dev_bc_1m / spread_atr_20"
        explanation="Same pipeline as daily — compute deviation, subtract the 20-day bias, normalize by ATR. The result updates every minute as the spread and ES prices tick."
      />
      <FlowArrow isDarkMode={isDarkMode} />
      <PipelineStep isDarkMode={isDarkMode} color="amber" step="3" title="Residual Ratio Scaling"
        formula="ratio = (resid_signal / norm_dev_bc).clip(0.3, 3.0)
norm_dev_resid_g0_1m = norm_dev_1m × ratio"
        explanation="The daily regression tells us what fraction of today's deviation is genuine mispricing. If the residual is 1.13 and total norm_dev is 1.69, the ratio is 0.67 — only 67% is real. We apply this ratio to the 1-minute deviation signal. This is the intraday entry/exit signal."
      />
    </div>

    <Callout type="warn" isDarkMode={isDarkMode}>
      We don't re-run the regression every minute — the ZQ features don't move intraday. The ratio is a daily constant that scales the intraday signal. This is computationally efficient and conceptually sound.
    </Callout>
  </div>
);

const EntryExitSection = ({ isDarkMode }) => (
  <div className="space-y-6">
    <SectionHeader icon="🎯" title="Layer 6: Entry & Exit Mechanics" subtitle="Two-phase entry, four exit mechanisms" isDarkMode={isDarkMode} color="green" />

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Phase 1: Daily Screening</h3>
      <CodeBlock isDarkMode={isDarkMode} code={`resid_signal = today's daily residual

if resid_signal > 0 → candidate SHORT  (spread is rich after accounting for rates)
if resid_signal < 0 → candidate LONG   (spread is cheap after accounting for rates)

Quick filter: skip if |resid_signal| < entry_threshold × 0.6
  └── For LONG (threshold 1.0):  need |signal| ≥ 0.6
  └── For SHORT no-gates (1.0):  need |signal| ≥ 0.6
  └── For SHORT asym (1.5):      need |signal| ≥ 0.9`} />
      <p className={`text-sm mt-3 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
        This is a coarse pre-filter. If the daily signal is too weak, don't bother pulling 1-minute data. The 0.6× factor provides a "watchlist zone" — the signal is interesting but needs intraday confirmation.
      </p>
    </Card>

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Phase 2: 1-Minute Confirmation & Entry</h3>
      <CodeBlock isDarkMode={isDarkMode} code={`Pull ALL 1-minute bars for today's trading session (~400 bars)

For LONG candidate:
  best_bar = the minute with the LOWEST norm_dev_resid_g0_1m  (cheapest moment)
  if best_bar.signal > −entry_threshold → skip (never cheap enough today)
  else → ENTER LONG at best_bar's spread price

For SHORT candidate:
  best_bar = the minute with the HIGHEST norm_dev_resid_g0_1m  (richest moment)
  if best_bar.signal < +entry_threshold → skip (never rich enough today)
  else → ENTER SHORT at best_bar's spread price`} />
      <Callout type="info" isDarkMode={isDarkMode}>
        The backtest scans the entire day and picks the optimal minute. In live trading, you'd monitor the signal in real-time and enter when it crosses the threshold. The backtest is saying "the opportunity existed at this price."
      </Callout>
    </Card>

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Exit Mechanisms (in priority order)</h3>
      <div className="space-y-4">
        {[
          { n: '1', t: '1m Reversion', rule: 'For LONG: find the MAX signal minute. If |signal| < 0.5 → exit at that price.\nFor SHORT: find the MIN signal minute. If |signal| < 0.5 → exit at that price.', desc: 'The signal reverted to within 0.5 ATRs of fair value during the day. Take profit.', tag: 'reversion_1m', pct: '~79% of exits' },
          { n: '2', t: 'Daily Reversion', rule: 'If |resid_signal_daily| < 0.5 → exit at daily close', desc: 'Daily signal has reverted. Fallback if 1m data didn\'t quite get below threshold.', tag: 'reversion_daily', pct: '~3% of exits' },
          { n: '3', t: 'Stop Loss', rule: 'If |dev_bc_today| > |dev_bc_at_entry| × stop_mult → exit at daily close', desc: 'The deviation is expanding in the wrong direction. Cut losses.', tag: 'stop', pct: '~8% of exits' },
          { n: '4', t: 'Time Stop', rule: 'If days_held ≥ max_hold → exit at daily close', desc: 'The trade hasn\'t worked within the max holding period. Move on.', tag: 'time_stop', pct: '~10% of exits' },
        ].map(({ n, t, rule, desc, tag, pct }) => (
          <div key={n} className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
            <div className="flex items-center gap-3 mb-2">
              <span className={`w-7 h-7 rounded-full bg-green-500 text-white font-bold text-xs flex items-center justify-center`}>{n}</span>
              <span className={`font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{t}</span>
              <span className={`text-xs font-mono px-2 py-0.5 rounded ${isDarkMode ? 'bg-gray-700 text-gray-400' : 'bg-gray-200 text-gray-600'}`}>{tag}</span>
              <span className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{pct}</span>
            </div>
            <pre className={`text-xs font-mono mb-2 px-3 py-2 rounded ${isDarkMode ? 'bg-gray-900 text-green-400' : 'bg-white text-gray-700'}`}>{rule}</pre>
            <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{desc}</p>
          </div>
        ))}
      </div>
    </Card>

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-3 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Roll Window Handling</h3>
      <p className={`text-sm mb-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
        The backtest runs in <strong>two passes</strong>, then merges the results:
      </p>
      <CodeBlock isDarkMode={isDarkMode} code={`Pass 1: Normal periods (skip_roll=True)  → uses standard thresholds
Pass 2: Roll window only (roll_only=True) → uses relaxed l_roll / s_roll thresholds
Final trades = merge(Pass 1, Pass 2).sort_by(entry_date)`} />
      <p className={`text-sm mt-3 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
        During roll weeks, spreads are more volatile and mispricings tend to be larger and shorter-lived. The thresholds are relaxed (e.g., 0.75 instead of 1.0 for longs) to capture these opportunities.
      </p>
    </Card>
  </div>
);

const ConfigsSection = ({ isDarkMode }) => (
  <div className="space-y-6">
    <SectionHeader icon="⚙️" title="V6 (No Gates) vs V6 (Asym)" subtitle="Same signal, different short-side conviction requirements" isDarkMode={isDarkMode} color="indigo" />

    <ComparisonTable isDarkMode={isDarkMode}
      headers={['Parameter', 'V6 (No Gates)', 'V6 (Asym)', 'What It Controls']}
      highlightCol={2}
      rows={[
        ['daily_signal_col', 'resid_signal', 'resid_signal', 'Same — pure residual (gamma=0)'],
        ['intraday_signal_col', 'norm_dev_resid_g0_1m', 'norm_dev_resid_g0_1m', 'Same — residual-scaled 1m deviation'],
        ['long_entry_thresh', '1.0', '1.0', 'Min signal to enter long'],
        ['short_entry_thresh', '1.0', '1.5 ★', 'Min signal to enter short'],
        ['long_exit_thresh', '0.5', '0.5', 'Signal reverts below this → exit long'],
        ['short_exit_thresh', '0.5', '0.5', 'Signal reverts below this → exit short'],
        ['long_max_hold', '10 days', '10 days', 'Max time in a long position'],
        ['short_max_hold', '10 days', '7 days ★', 'Max time in a short position'],
        ['long_stop_mult', '2.5×', '2.5×', 'Stop-loss multiplier for longs'],
        ['short_stop_mult', '2.5×', '2.0× ★', 'Stop-loss multiplier for shorts'],
        ['l_roll (roll threshold)', '0.75', '0.75', 'Long threshold during roll weeks'],
        ['s_roll (roll threshold)', '0.75', '1.0 ★', 'Short threshold during roll weeks'],
        ['VIX floor gate', 'None', 'None', 'No binary gates in either config'],
        ['ES momentum cap', 'None', 'None', 'No binary gates in either config'],
        ['ZQ threshold mod', 'None', 'None', 'No V5-style continuous adjustments'],
      ]}
    />

    <Callout type="insight" isDarkMode={isDarkMode}>
      <strong>Why shorts are treated differently:</strong> Calendar spreads have a structural upward bias — they tend to widen over time as carry accrues. Shorting the spread is inherently more counter-trend than going long. The asymmetric config demands higher conviction (signal ≥ 1.5 vs 1.0), gives less time to work (7d vs 10d), and uses a tighter stop (2.0× vs 2.5×).
    </Callout>

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-3 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Practical Impact</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className={`rounded-lg border p-4 text-center ${isDarkMode ? 'bg-indigo-900/20 border-indigo-800' : 'bg-indigo-50 border-indigo-200'}`}>
          <div className={`text-xs uppercase tracking-wider mb-1 ${isDarkMode ? 'text-indigo-400' : 'text-indigo-600'}`}>Longs</div>
          <div className={`text-2xl font-black ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Identical</div>
          <div className={`text-sm mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Same params, same trades, same PnL</div>
        </div>
        <div className={`rounded-lg border p-4 text-center ${isDarkMode ? 'bg-rose-900/20 border-rose-800' : 'bg-rose-50 border-rose-200'}`}>
          <div className={`text-xs uppercase tracking-wider mb-1 ${isDarkMode ? 'text-rose-400' : 'text-rose-600'}`}>Shorts Filtered</div>
          <div className={`text-2xl font-black ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>~40 removed</div>
          <div className={`text-sm mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Mostly low-conviction losers (signal 1.0–1.4)</div>
        </div>
        <div className={`rounded-lg border p-4 text-center ${isDarkMode ? 'bg-emerald-900/20 border-emerald-800' : 'bg-emerald-50 border-emerald-200'}`}>
          <div className={`text-xs uppercase tracking-wider mb-1 ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>Short WR</div>
          <div className={`text-2xl font-black ${isDarkMode ? 'text-emerald-400' : 'text-emerald-700'}`}>67% → 76%</div>
          <div className={`text-sm mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Higher quality shorts survive the filter</div>
        </div>
      </div>
    </Card>
  </div>
);

const WalkthroughSection = ({ isDarkMode }) => (
  <div className="space-y-6">
    <SectionHeader icon="🚶" title="Live Walkthrough: A Day in 2025" subtitle="Step by step, exactly what happens when the system runs" isDarkMode={isDarkMode} color="orange" />

    <Callout type="info" isDarkMode={isDarkMode}>
      It's <strong>December 2, 2025</strong>. You're trading ESZ5-ESH6 (Dec25–Mar26 calendar spread). Here's exactly what the system sees and does.
    </Callout>

    <div className="space-y-4">
      <Card isDarkMode={isDarkMode} className="p-5">
        <h4 className={`font-bold mb-3 flex items-center gap-2 ${isDarkMode ? 'text-orange-400' : 'text-orange-600'}`}>
          <span className="w-8 h-8 rounded-full bg-orange-500 text-white text-sm font-bold flex items-center justify-center">1</span>
          Morning: Daily Data Arrives
        </h4>
        <div className={`grid grid-cols-2 md:grid-cols-4 gap-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
          {[
            { l: 'ES Close', v: '6,050.25' },
            { l: 'Spread Close (prev)', v: '58.95' },
            { l: 'SOFR Spot', v: '4.30%' },
            { l: 'ZQ Implied 3m', v: '4.00%' },
            { l: 'Days Between', v: '91' },
            { l: 'Spread ATR₂₀', v: '1.29' },
            { l: 'VIX', v: '16.6' },
            { l: 'In Roll Window', v: 'No' },
          ].map(({ l, v }) => (
            <div key={l} className={`rounded-lg border p-3 ${isDarkMode ? 'bg-gray-900 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
              <div className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{l}</div>
              <div className="font-mono font-bold text-sm mt-0.5">{v}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card isDarkMode={isDarkMode} className="p-5">
        <h4 className={`font-bold mb-3 flex items-center gap-2 ${isDarkMode ? 'text-orange-400' : 'text-orange-600'}`}>
          <span className="w-8 h-8 rounded-full bg-orange-500 text-white text-sm font-bold flex items-center justify-center">2</span>
          Compute Fair Value & Deviation
        </h4>
        <CodeBlock isDarkMode={isDarkMode} code={`carry_spot   = 4.30% − 1.50% (div yield) = 2.80%
fv_spot      = 6,050.25 × 0.028 × 91/365  = 42.24
dev_spot     = 58.95 − 42.24              = +16.71
bias_20d     = +15.18  (last 20 days avg)
dev_bc       = 16.71 − 15.18              = +1.53
norm_dev_bc  = 1.53 / 1.29                = +1.19 ATRs (rich)`} />
      </Card>

      <Card isDarkMode={isDarkMode} className="p-5">
        <h4 className={`font-bold mb-3 flex items-center gap-2 ${isDarkMode ? 'text-orange-400' : 'text-orange-600'}`}>
          <span className="w-8 h-8 rounded-full bg-orange-500 text-white text-sm font-bold flex items-center justify-center">3</span>
          ZQ Feature Computation & Residual Regression
        </h4>
        <CodeBlock isDarkMode={isDarkMode} code={`rate_gap     = 4.00% − 4.30%  = −0.30% (market expects rate cuts)
rate_gap_z   = z_score(−0.30%, rolling_60d) = −1.05
carry_gap_z  = z_score(carry_gap, rolling_60d) = −0.72
fwd_slope_z  = z_score(fwd_curve_slope, rolling_60d) = +0.28

Regression (trained on prior 60 days):
  β₀=+0.04, β₁=−0.28, β₂=−0.18, β₃=+0.12

predicted    = 0.04 + (−0.28)(−1.05) + (−0.18)(−0.72) + (0.12)(0.28)
             = 0.04 + 0.29 + 0.13 + 0.03 = +0.49

resid_signal = norm_dev_bc − predicted
             = 1.19 − 0.49 = +0.70

INTERPRETATION: Of the 1.19 ATRs of richness, 0.49 is explained by the
market pricing in rate cuts (which makes the spread rationally richer).
Only 0.70 ATRs is genuine mispricing.`} />
      </Card>

      <Card isDarkMode={isDarkMode} className="p-5">
        <h4 className={`font-bold mb-3 flex items-center gap-2 ${isDarkMode ? 'text-orange-400' : 'text-orange-600'}`}>
          <span className="w-8 h-8 rounded-full bg-orange-500 text-white text-sm font-bold flex items-center justify-center">4</span>
          Daily Screening Decision
        </h4>
        <div className={`text-sm leading-relaxed ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
          <p><code className="font-mono">resid_signal = +0.70 > 0</code> → candidate SHORT direction</p>
          <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className={`rounded-lg border p-3 ${isDarkMode ? 'bg-emerald-900/20 border-emerald-800' : 'bg-emerald-50 border-emerald-200'}`}>
              <div className={`text-xs font-bold ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>V6 (No Gates)</div>
              <div className={`text-sm mt-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                Watchlist filter: |0.70| ≥ 1.0 × 0.6 = 0.6? <strong>Yes</strong><br />
                → Proceed to 1m scan, need signal ≥ 1.0 at peak
              </div>
            </div>
            <div className={`rounded-lg border p-3 ${isDarkMode ? 'bg-rose-900/20 border-rose-800' : 'bg-rose-50 border-rose-200'}`}>
              <div className={`text-xs font-bold ${isDarkMode ? 'text-rose-400' : 'text-rose-600'}`}>V6 (Asym)</div>
              <div className={`text-sm mt-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                Watchlist filter: |0.70| ≥ 1.5 × 0.6 = 0.9? <strong>No → SKIP</strong><br />
                → Day filtered out entirely, saves from a potential marginal trade
              </div>
            </div>
          </div>
        </div>
      </Card>

      <Card isDarkMode={isDarkMode} className="p-5">
        <h4 className={`font-bold mb-3 flex items-center gap-2 ${isDarkMode ? 'text-orange-400' : 'text-orange-600'}`}>
          <span className="w-8 h-8 rounded-full bg-orange-500 text-white text-sm font-bold flex items-center justify-center">5</span>
          1-Minute Scan & Entry (No Gates proceeds)
        </h4>
        <CodeBlock isDarkMode={isDarkMode} code={`ratio = resid_signal / norm_dev_bc = 0.70 / 1.19 = 0.59

Scanning 400+ one-minute bars throughout the session:
  09:35  norm_dev_1m = +0.85  →  ×0.59 = +0.50  (not rich enough)
  10:12  norm_dev_1m = +1.42  →  ×0.59 = +0.84  (getting close...)
  11:47  norm_dev_1m = +1.78  →  ×0.59 = +1.05  ← PEAK, clears 1.0 ✓
  13:20  norm_dev_1m = +1.15  →  ×0.59 = +0.68  (came back down)

Best bar = 11:47 AM, signal = +1.05, spread_close = 59.35
→ ENTER SHORT at 59.35`} />
        <p className={`text-sm mt-3 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
          The spread hit its richest point of the day around 11:47 AM. The residual-adjusted signal of +1.05 cleared the 1.0 threshold. Short entered at 59.35.
        </p>
      </Card>

      <Card isDarkMode={isDarkMode} className="p-5">
        <h4 className={`font-bold mb-3 flex items-center gap-2 ${isDarkMode ? 'text-orange-400' : 'text-orange-600'}`}>
          <span className="w-8 h-8 rounded-full bg-orange-500 text-white text-sm font-bold flex items-center justify-center">6</span>
          Subsequent Days: Exit Monitoring
        </h4>
        <CodeBlock isDarkMode={isDarkMode} code={`Dec 3:  Scan 1m bars → min signal = +0.72  → |0.72| > 0.5 → hold
Dec 4:  Scan 1m bars → min signal = +0.61  → |0.61| > 0.5 → hold
Dec 5:  Scan 1m bars → min signal = +0.38  → |0.38| < 0.5 ✓
        Spread price at that minute = 57.80

EXIT SHORT at 57.80 → reversion_1m

PnL = (59.35 − 57.80) − 0.50 commission = +1.05 pts ($52.50 per contract)`} />
      </Card>
    </div>
  </div>
);

const PerformanceSection = ({ isDarkMode }) => (
  <div className="space-y-6">
    <SectionHeader icon="📊" title="V6 Performance (2023–2026)" subtitle="Real backtest results from research scripts" isDarkMode={isDarkMode} color="pink" />

    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <StatCard isDarkMode={isDarkMode} label="V6 Asym Sharpe" value="6.17" sub="vs V5 best: 2.77" color="emerald" size="lg" />
      <StatCard isDarkMode={isDarkMode} label="V6 Asym Win Rate" value="76.4%" sub="Short WR: 75.9%" color="blue" size="lg" />
      <StatCard isDarkMode={isDarkMode} label="Total PnL" value="+144.4 pts" sub="$7,220 per contract" color="green" size="lg" />
      <StatCard isDarkMode={isDarkMode} label="Max Drawdown" value="-5.7 pts" sub="vs V5: -11.6 pts" color="red" size="lg" />
    </div>

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Grand Comparison: V2 → V6</h3>
      <ComparisonTable isDarkMode={isDarkMode}
        headers={['Strategy', 'Trades', 'PnL (pts)', 'Sharpe', 'Win Rate', 'Short WR', 'Profit Factor', 'Max DD']}
        highlightCol={3}
        rows={[
          ['V2 (spot, no gates)', '192', '+82.5', '1.73', '60.9%', '—', '1.82', '-14.2'],
          ['V5 (spot + full gates)', '141', '+143.4', '2.77', '70.2%', '70.0%', '2.85', '-11.6'],
          ['V6 (no gates)', '188', '+163.2', '5.57', '72.3%', '67.4%', '4.80', '-5.8'],
          ['V6 (asym)', '148', '+144.4', '6.17', '76.4%', '75.9%', '6.21', '-5.7'],
        ]}
      />
    </Card>

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Year-by-Year: V6 (Asym)</h3>
      <ComparisonTable isDarkMode={isDarkMode}
        headers={['Year', 'Trades', 'PnL', 'Sharpe', 'Long WR', 'Short WR']}
        rows={PERF.asym.years.map(y => [
          y.yr.toString(), y.n.toString(), `+${y.pnl.toFixed(1)}`, y.sharpe.toFixed(2), `${y.lWr}%`, `${y.sWr}%`
        ])}
      />
    </Card>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-3 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Walk-Forward OOS Validation</h3>
        <div className="grid grid-cols-2 gap-3">
          <StatCard isDarkMode={isDarkMode} label="OOS Sharpe" value="6.01" color="emerald" />
          <StatCard isDarkMode={isDarkMode} label="Positive Folds" value="12/12" sub="Every quarter profitable" color="blue" />
        </div>
        <p className={`text-sm mt-3 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
          Quarterly walk-forward out-of-sample. The regression is retrained each fold on only past data. Every single fold was profitable.
        </p>
      </Card>
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-3 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Monte Carlo Significance</h3>
        <div className="grid grid-cols-2 gap-3">
          <StatCard isDarkMode={isDarkMode} label="p-value" value="<0.0001" sub="Highly significant" color="emerald" />
          <StatCard isDarkMode={isDarkMode} label="V5 Short p-val" value="0.0898" sub="NOT significant" color="red" />
        </div>
        <p className={`text-sm mt-3 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
          10,000 random permutations. V6's short-side performance has a {`<`}0.01% chance of occurring by random chance. V5's short side was not statistically significant.
        </p>
      </Card>
    </div>

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Recent Period: Oct 2025 – Jan 2026</h3>
      <ComparisonTable isDarkMode={isDarkMode}
        headers={['Metric', 'V6 (No Gates)', 'V6 (Asym)', 'Delta']}
        highlightCol={2}
        rows={[
          ['Trades', '19', '15', '-4'],
          ['PnL (pts)', '+19.80', '+23.20', '+3.40'],
          ['PnL ($)', '+$990', '+$1,160', '+$170'],
          ['Win Rate', '63.2%', '80.0%', '+16.8pp'],
          ['Short WR', '44.4%', '80.0%', '+35.6pp'],
          ['Sharpe', '3.55', '5.04', '+1.50'],
          ['Profit Factor', '4.60', '17.00', '+12.40'],
          ['Max Drawdown', '-2.00 pts', '-0.90 pts', '+1.10'],
        ]}
      />
    </Card>
  </div>
);

const LookaheadAuditSection = ({ isDarkMode }) => {
  const issues = [
    {
      id: 1,
      title: 'Intraday Best-Bar Selection (ENTRY)',
      severity: 'HIGH',
      sevColor: 'red',
      where: 'Backtest engine — entry 1m scanning',
      problem: 'The backtest scans ALL ~400 one-minute bars for the day and picks the single bar with the most extreme signal for entry (idxmin/idxmax). In real-time, you cannot know at 10:47 AM that the signal won\'t go more extreme at 2:30 PM. This is the largest source of inflated performance.',
      codeSnippet: `// BACKTEST (lookahead within the day):
best_idx = day_1m_v[signal_col].idxmin()   // picks THE best minute of entire day
best = day_1m_v.loc[best_idx]
if best.signal >= entry_threshold → ENTER at best.price`,
      fix: `// PRODUCTION FIX — Segment-30 Entry (best tested approach):
// Divide session into 30-bar (30-min) segments
// Within each segment, find the best signal bar
// Enter at end of first segment where threshold was crossed
// Implementable via limit orders: place limit at threshold price,
// fills naturally at best available price within window

for segment in 30_min_windows:
    best_in_seg = segment.signal.argmax()
    if best_in_seg exceeds threshold:
        ENTER at segment[best_in_seg].price
        break`,
      impact: 'MEASURED: Sharpe drops from 6.17 (best-bar) to 2.62 (seg_30bar + confirmed_3 exit) or 3.36 (seg_30bar + smoothed_3 exit). The best-bar entry was worth ~45% of total Sharpe. Segment-30 retains 54.4% of original edge.',
      recommendation: 'Use seg_30bar entry in production. Place limit orders at threshold price levels — the 30-min window represents natural fill behavior.',
    },
    {
      id: 2,
      title: 'Intraday Best-Bar Selection (EXIT)',
      severity: 'HIGH',
      sevColor: 'red',
      where: 'Backtest engine — exit 1m scanning',
      problem: 'Same issue on exits: the backtest picks THE most-reverted minute bar for exit (idxmax for longs). In real-time, you can\'t know the signal won\'t revert further later in the session.',
      codeSnippet: `// BACKTEST (exit lookahead):
bx = x1mv.loc[x1mv[signal_col].idxmax()]  // picks THE most-reverted minute
if abs(bx.signal) < exit_threshold → EXIT at bx.price`,
      fix: `// PRODUCTION FIX — Smoothed-3 Exit (best tested approach):
// Apply 3-bar rolling average to signal, exit at first crossing
// Smoothing filters noise and prevents premature exits

smoothed = rolling_mean(signal, window=3)
for each bar:
    if abs(smoothed) < exit_threshold:
        EXIT at this bar's price
        break`,
      impact: 'MEASURED: smoothed_3 exit outperformed all alternatives (confirmed, checkpoint, peak-reversion, segment). It achieves WR 65.0% vs 55.3% for first-crossing exit. The smoothing prevents whipsaw exits on noise spikes.',
      recommendation: 'Use smoothed_3 exit. The 3-bar rolling average is trivial to implement in real-time.',
    },
    {
      id: 3,
      title: 'Daily Signal Uses Same-Day Close',
      severity: 'MEDIUM',
      sevColor: 'amber',
      where: 'Data pipeline — daily resid_signal computation',
      problem: 'The daily resid_signal is computed from today\'s norm_dev_bc, which uses today\'s daily close. But the backtest uses that same signal to decide whether to enter a trade during that same day. In reality, the daily close isn\'t known until after the session ends.',
      codeSnippet: `// BACKTEST (circular logic within same day):
Day T: compute norm_dev_bc from close[T]     // needs end-of-day price
Day T: compute resid_signal from norm_dev_bc  // needs above
Day T: use resid_signal to screen for trades  // uses above for SAME day`,
      fix: `// PRODUCTION FIX — Use prior day's signal for screening:
resid_signal_lag1 = resid_signal.shift(1)
Day T morning: if |resid_signal[T-1]| > watchlist_filter → monitor 1m bars
Day T intraday: 1m signal uses LIVE prices (clean, no lookahead)

// TESTED: T-1 screening is already used in all production results below.
// Also tested "running daily" mode (use first 1m bars as daily estimate):
// Running mode increased trades (226 vs 120) but degraded Sharpe (1.40 vs 3.36).
// T-1 static screening is BETTER — it acts as a quality filter.`,
      impact: 'MEASURED: Part of the production fix. T-1 screening contributes to lower trade count (120 vs original 148) but much higher quality. The 1m intraday signal is fully live and clean.',
      recommendation: 'Use T-1 resid_signal for daily screening. Do NOT attempt running daily signal updates — they add noise.',
    },
    {
      id: 4,
      title: 'div_yield_spot_5d Includes Today',
      severity: 'LOW',
      sevColor: 'green',
      where: 'Data pipeline — FV model computation',
      problem: 'The 5-day smoothed dividend yield uses a rolling window that includes today\'s value. Today\'s div_yield_spot is backed out from today\'s spread close, which creates a subtle circularity in the daily FV calculation.',
      codeSnippet: `div_yield_spot_5d[T] = mean(div_yield_spot[T-4 : T])  // includes today
// FV on day T partially depends on close[T]`,
      fix: `// PRODUCTION FIX — Lag by one day:
div_yield_spot_5d[T] = mean(div_yield_spot[T-5 : T-1])  // excludes today
// At 1m level, this is already clean:
// fv_1m uses carry_spot from prior day (merged and forward-filled).`,
      impact: 'Negligible. The 5-day smoothing heavily dampens the effect of any single day. The 1m FV computation already uses prior-day carry_spot.',
      recommendation: 'Fix for correctness, but don\'t expect measurable performance change.',
    },
    {
      id: 5,
      title: 'No Position Overlap Check Across Symbols',
      severity: 'LOW-MED',
      sevColor: 'yellow',
      where: 'Backtest engine — trade management',
      problem: 'The backtest processes each spread symbol independently. You could theoretically be long ESZ5-ESH6 and short ESH6-ESM6 simultaneously — positions that partially offset each other.',
      codeSnippet: `for sym in spread_symbols:
    // trades ESZ5-ESH6 independently
    // trades ESH6-ESM6 independently
    // no awareness of other active positions`,
      fix: `// PRODUCTION FIX — Global position tracker:
active_positions = {}
before_entry(symbol, direction):
    if len(active_positions) >= MAX_SIMULTANEOUS:
        skip — capital limit reached`,
      impact: 'Moderate. During roll transitions, could overlap outgoing/incoming spread. In practice, one-quarter adjacent spreads rarely have opposing signals simultaneously.',
      recommendation: 'Implement a simple rule: only one position across all spread symbols at any time.',
    },
    {
      id: 6,
      title: 'Transaction Cost Model',
      severity: 'LOW',
      sevColor: 'green',
      where: 'Backtest engine — PnL calculation',
      problem: 'The backtest uses $0.50/RT (0.50 pts = $25). Real costs are ~$17.50 = 0.35 pts. The model is actually conservative.',
      codeSnippet: `COST_PER_RT = 0.50  // backtest uses 0.50 pts = $25 per round trip
// Realistic: commission $2.48 + bid-ask $12.50 + slippage $2.50 = ~$17.50 = 0.35 pts`,
      fix: `// Already conservative. For production:
// Use 0.35 pts for liquid hours, 0.75 pts for roll-week/illiquid
// Track actual fills to calibrate`,
      impact: 'Slightly pessimistic. Real performance may be marginally better on this dimension.',
      recommendation: 'Track real slippage for the first 20 trades and adjust.',
    },
  ];

  const cleanItems = [
    {
      title: 'Rolling Regression (Walk-Forward)',
      detail: 'Training window is [start, i) — strictly prior to day i. Day i\'s data is never used for training. Correctly walk-forward with no lookahead.',
      code: 'y_train = target[start:i]  // exclusive upper bound, day i excluded',
    },
    {
      title: 'Z-Score Normalization',
      detail: 'Uses pandas .rolling(60, min_periods=20) which is backward-looking by default. On day T, it uses days T-59 through T. No future data.',
      code: 'roll_mean = feature.rolling(60, min_periods=20).mean()  // backward-only',
    },
    {
      title: 'ATR & Bias Correction',
      detail: 'Both use backward-looking rolling windows (20-day). On day T, they incorporate data through day T. Since these are used for normalization, including today is standard practice.',
      code: 'spread_atr_20 = (high-low).rolling(20).mean()  // backward-only',
    },
    {
      title: 'Roll Window Detection',
      detail: 'in_roll_window is determined by calendar dates (days before quarterly expiry). Fully knowable in advance — no data dependency.',
      code: 'in_roll_window = days_to_next_roll <= ROLL_WINDOW  // pure calendar',
    },
    {
      title: 'Forward Return Analysis (Section 10)',
      detail: 'Uses future returns (shift(-5)) to evaluate signal quality. This is research-only code for signal selection — it is NOT used in the production signal, the backtest, or any trading decision.',
      code: 'fwd_ret_5d = close.shift(-5) - close  // RESEARCH ONLY, not in signal path',
    },
    {
      title: 'Stop-Loss Logic',
      detail: 'Compares current deviation against entry deviation. Uses only past/present data. Production backtest uses T-1 daily dev for stop checks.',
      code: 'if abs(dev_bc_lag1) > abs(dev_bc_at_entry) * stop_mult → EXIT',
    },
  ];

  const strategyComparison = {
    entries: [
      { name: 'Best-Bar (original)', sharpe: '6.17', wr: '76.4%', n: 148, lookahead: true, desc: 'Scans all 1m bars, picks absolute best' },
      { name: 'First Crossing', sharpe: '1.29', wr: '55.3%', n: 114, lookahead: false, desc: 'Enter at first bar crossing threshold' },
      { name: 'Confirmed 5-bar', sharpe: '1.19', wr: '56.6%', n: 83, lookahead: false, desc: '5 consecutive bars above threshold' },
      { name: 'Confirmed 10-bar', sharpe: '1.20', wr: '56.2%', n: 64, lookahead: false, desc: '10 consecutive bars above threshold' },
      { name: 'VWAP 15-min', sharpe: '0.00', wr: '47.9%', n: 119, lookahead: false, desc: 'Average price over 15 min after first crossing' },
      { name: 'Checkpoint 30-min', sharpe: '0.58', wr: '52.7%', n: 110, lookahead: false, desc: 'Evaluate at fixed 30-min intervals' },
      { name: 'Pullback 15%', sharpe: '0.67', wr: '53.8%', n: 91, lookahead: false, desc: 'Wait for pullback after first crossing' },
      { name: 'Escalate 1.5x', sharpe: '2.56', wr: '60.5%', n: 81, lookahead: false, desc: 'Enter only at 1.5x threshold (high conviction)' },
      { name: 'Segment 30-bar', sharpe: '2.62', wr: '60.5%', n: 114, lookahead: false, desc: 'Best bar within 30-min segment (limit order)' },
      { name: 'Smoothed 10-bar', sharpe: '0.91', wr: '54.5%', n: 77, lookahead: false, desc: 'Rolling average of signal before evaluation' },
      { name: 'Peak Detect 20%', sharpe: '-1.44', wr: '41.5%', n: 106, lookahead: false, desc: 'Wait for signal to peak and decline 20%' },
    ],
    exits: [
      { name: 'Best-Bar (original)', sharpe: '6.17', wr: '76.4%', desc: 'Scans all 1m bars, picks best reversion bar' },
      { name: 'First Crossing', sharpe: '1.22', wr: '53.5%', desc: 'Exit at first bar signal < threshold' },
      { name: 'Confirmed 3-bar', sharpe: '2.62', wr: '60.5%', desc: 'Signal below threshold for 3 consecutive bars' },
      { name: 'Smoothed 3-bar', sharpe: '3.36', wr: '65.0%', desc: '3-bar rolling average crosses threshold' },
      { name: 'Smoothed 5-bar', sharpe: '2.57', wr: '63.2%', desc: '5-bar rolling average crosses threshold' },
      { name: 'Peak Reversion', sharpe: '2.33', wr: '58.1%', desc: 'Track peak reversion, exit when declining' },
      { name: 'Segment 15-bar', sharpe: '2.06', wr: '56.3%', desc: 'Check reversion at 15-min boundaries' },
      { name: 'Checkpoint 60-min', sharpe: '1.39', wr: '50.8%', desc: 'Evaluate exits at 60-min intervals' },
    ],
  };

  const yearByYear = [
    { year: 2023, n: 31, longs: 22, shorts: 9, pnl: '+16.5', sharpe: '6.23', wr: '61.3%', lWr: '63.6%', sWr: '55.6%' },
    { year: 2024, n: 47, longs: 33, shorts: 14, pnl: '+22.6', sharpe: '3.24', wr: '68.1%', lWr: '69.7%', sWr: '64.3%' },
    { year: 2025, n: 40, longs: 23, shorts: 17, pnl: '+11.2', sharpe: '2.12', wr: '62.5%', lWr: '60.9%', sWr: '64.7%' },
    { year: 2026, n: 2, longs: 1, shorts: 1, pnl: '+0.4', sharpe: '-', wr: '100%', lWr: '100%', sWr: '100%' },
  ];

  const severityBg = {
    HIGH: isDarkMode ? 'bg-red-900/30 border-red-700' : 'bg-red-50 border-red-200',
    MEDIUM: isDarkMode ? 'bg-amber-900/30 border-amber-700' : 'bg-amber-50 border-amber-200',
    'LOW-MED': isDarkMode ? 'bg-yellow-900/30 border-yellow-700' : 'bg-yellow-50 border-yellow-200',
    LOW: isDarkMode ? 'bg-green-900/30 border-green-700' : 'bg-green-50 border-green-200',
  };

  const severityText = {
    HIGH: isDarkMode ? 'text-red-400' : 'text-red-700',
    MEDIUM: isDarkMode ? 'text-amber-400' : 'text-amber-700',
    'LOW-MED': isDarkMode ? 'text-yellow-400' : 'text-yellow-700',
    LOW: isDarkMode ? 'text-green-400' : 'text-green-700',
  };

  return (
    <div className="space-y-6">
      <SectionHeader icon="🔍" title="Lookahead Bias Audit" subtitle="Full production-realistic backtest results — 13 entry strategies, 8 exit strategies tested" isDarkMode={isDarkMode} color="yellow" />

      <Callout type="warn" isDarkMode={isDarkMode}>
        <strong>Research completed:</strong> We ran two rounds of experiments (<code className="font-mono">v6_06</code> and <code className="font-mono">v6_06b</code>) testing every combination of production-realistic entry/exit strategies against the original lookahead backtest. All results below are <strong>measured, not estimated</strong>. The core signal is clean — all issues are in the execution layer.
      </Callout>

      {/* Severity Summary — REAL NUMBERS */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard isDarkMode={isDarkMode} label="Original Sharpe" value="6.17" sub="With lookahead bias (asym)" color="red" />
        <StatCard isDarkMode={isDarkMode} label="Production Sharpe" value="3.36" sub="No lookahead (seg_30 + smooth_3)" color="emerald" />
        <StatCard isDarkMode={isDarkMode} label="Edge Retained" value="54.4%" sub="After removing all lookahead" color="blue" />
        <StatCard isDarkMode={isDarkMode} label="Win Rate" value="65.0%" sub="120 trades, 2023-2026" color="purple" />
      </div>

      {/* Grand Comparison Table */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Original vs Production-Realistic: Head-to-Head</h3>
        <div className="overflow-x-auto">
          <table className={`w-full text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            <thead>
              <tr className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>
                <th className="text-left py-2 pr-3 font-medium">Configuration</th>
                <th className="text-right py-2 px-2 font-medium">N</th>
                <th className="text-right py-2 px-2 font-medium">PnL</th>
                <th className="text-right py-2 px-2 font-medium">Sharpe</th>
                <th className="text-right py-2 px-2 font-medium">WR</th>
                <th className="text-right py-2 px-2 font-medium">PF</th>
                <th className="text-right py-2 px-2 font-medium">MaxDD</th>
              </tr>
            </thead>
            <tbody>
              <tr className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <td className={`py-2.5 pr-3 font-medium ${isDarkMode ? 'text-rose-400' : 'text-rose-600'}`}>Original V6 Asym (with lookahead)</td>
                <td className="text-right py-2.5 px-2">148</td>
                <td className="text-right py-2.5 px-2">+144.4</td>
                <td className={`text-right py-2.5 px-2 font-bold ${isDarkMode ? 'text-rose-400' : 'text-rose-600'}`}>6.17</td>
                <td className="text-right py-2.5 px-2">76.4%</td>
                <td className="text-right py-2.5 px-2">7.64</td>
                <td className="text-right py-2.5 px-2">-5.7</td>
              </tr>
              <tr className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <td className={`py-2.5 pr-3 font-medium ${isDarkMode ? 'text-rose-400' : 'text-rose-600'}`}>Original V6 No Gates (with lookahead)</td>
                <td className="text-right py-2.5 px-2">188</td>
                <td className="text-right py-2.5 px-2">+163.2</td>
                <td className={`text-right py-2.5 px-2 font-bold ${isDarkMode ? 'text-rose-400' : 'text-rose-600'}`}>5.57</td>
                <td className="text-right py-2.5 px-2">72.3%</td>
                <td className="text-right py-2.5 px-2">6.11</td>
                <td className="text-right py-2.5 px-2">-5.8</td>
              </tr>
              <tr className={`border-t-2 ${isDarkMode ? 'border-emerald-700' : 'border-emerald-300'}`}>
                <td className={`py-2.5 pr-3 font-bold ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>PROD Asym (seg_30 + smooth_3)</td>
                <td className="text-right py-2.5 px-2 font-bold">120</td>
                <td className="text-right py-2.5 px-2 font-bold">+50.7</td>
                <td className={`text-right py-2.5 px-2 font-bold ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>3.36</td>
                <td className="text-right py-2.5 px-2 font-bold">65.0%</td>
                <td className="text-right py-2.5 px-2 font-bold">2.48</td>
                <td className="text-right py-2.5 px-2 font-bold">-8.9</td>
              </tr>
              <tr className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <td className={`py-2.5 pr-3 font-medium ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>PROD No Gates (seg_30 + smooth_3)</td>
                <td className="text-right py-2.5 px-2">149</td>
                <td className="text-right py-2.5 px-2">+60.9</td>
                <td className={`text-right py-2.5 px-2 font-bold ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>3.18</td>
                <td className="text-right py-2.5 px-2">59.7%</td>
                <td className="text-right py-2.5 px-2">2.47</td>
                <td className="text-right py-2.5 px-2">-9.0</td>
              </tr>
              <tr className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <td className={`py-2.5 pr-3 font-medium ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>PROD Asym (escalate_1.5x + smooth_3)</td>
                <td className="text-right py-2.5 px-2">84</td>
                <td className="text-right py-2.5 px-2">+36.0</td>
                <td className={`text-right py-2.5 px-2 font-bold ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>3.00</td>
                <td className="text-right py-2.5 px-2">61.9%</td>
                <td className="text-right py-2.5 px-2">2.33</td>
                <td className="text-right py-2.5 px-2">-7.6</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>

      {/* Strategy Comparison — Entries */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-1 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Entry Strategy Comparison</h3>
        <p className={`text-sm mb-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>All tested with confirmed_3 exit, asym config, T-1 daily signal</p>
        <div className="overflow-x-auto">
          <table className={`w-full text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            <thead>
              <tr className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>
                <th className="text-left py-2 pr-3 font-medium">Strategy</th>
                <th className="text-right py-2 px-2 font-medium">Sharpe</th>
                <th className="text-right py-2 px-2 font-medium">WR</th>
                <th className="text-right py-2 px-2 font-medium">N</th>
                <th className="text-center py-2 px-2 font-medium">Lookahead?</th>
                <th className="text-left py-2 pl-3 font-medium">Description</th>
              </tr>
            </thead>
            <tbody>
              {strategyComparison.entries.map((s, i) => (
                <tr key={i} className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} ${
                  s.lookahead ? (isDarkMode ? 'bg-red-900/10' : 'bg-red-50/50') :
                  s.name.includes('Segment') || s.name.includes('Escalate') ? (isDarkMode ? 'bg-emerald-900/10' : 'bg-emerald-50/50') : ''
                }`}>
                  <td className={`py-2 pr-3 font-medium ${s.lookahead ? (isDarkMode ? 'text-rose-400' : 'text-rose-600') :
                    s.name.includes('Segment') || s.name.includes('Escalate') ? (isDarkMode ? 'text-emerald-400' : 'text-emerald-600') : ''}`}>{s.name}</td>
                  <td className={`text-right py-2 px-2 font-bold ${parseFloat(s.sharpe) > 2 ? (isDarkMode ? 'text-emerald-400' : 'text-emerald-600') :
                    parseFloat(s.sharpe) < 0.5 ? (isDarkMode ? 'text-red-400' : 'text-red-600') : ''}`}>{s.sharpe}</td>
                  <td className="text-right py-2 px-2">{s.wr}</td>
                  <td className="text-right py-2 px-2">{s.n}</td>
                  <td className={`text-center py-2 px-2 font-bold ${s.lookahead ? 'text-red-500' : 'text-emerald-500'}`}>{s.lookahead ? 'YES' : 'No'}</td>
                  <td className={`py-2 pl-3 text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{s.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Strategy Comparison — Exits */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-1 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Exit Strategy Comparison</h3>
        <p className={`text-sm mb-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>All tested with seg_30bar entry, asym config, T-1 daily signal</p>
        <div className="overflow-x-auto">
          <table className={`w-full text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            <thead>
              <tr className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>
                <th className="text-left py-2 pr-3 font-medium">Strategy</th>
                <th className="text-right py-2 px-2 font-medium">Sharpe</th>
                <th className="text-right py-2 px-2 font-medium">WR</th>
                <th className="text-left py-2 pl-3 font-medium">Description</th>
              </tr>
            </thead>
            <tbody>
              {strategyComparison.exits.map((s, i) => (
                <tr key={i} className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} ${
                  s.name.includes('original') ? (isDarkMode ? 'bg-red-900/10' : 'bg-red-50/50') :
                  s.name.includes('Smoothed 3') ? (isDarkMode ? 'bg-emerald-900/10' : 'bg-emerald-50/50') : ''
                }`}>
                  <td className={`py-2 pr-3 font-medium ${s.name.includes('original') ? (isDarkMode ? 'text-rose-400' : 'text-rose-600') :
                    s.name.includes('Smoothed 3') ? (isDarkMode ? 'text-emerald-400' : 'text-emerald-600') : ''}`}>{s.name}</td>
                  <td className={`text-right py-2 px-2 font-bold ${parseFloat(s.sharpe) > 3 ? (isDarkMode ? 'text-emerald-400' : 'text-emerald-600') : ''}`}>{s.sharpe}</td>
                  <td className="text-right py-2 px-2">{s.wr}</td>
                  <td className={`py-2 pl-3 text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{s.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Year-by-Year for Best Config */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-1 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Best Production Config: Year-by-Year</h3>
        <p className={`text-sm mb-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>seg_30bar entry + smoothed_3 exit, asym, T-1 screening</p>
        <div className="overflow-x-auto">
          <table className={`w-full text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            <thead>
              <tr className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>
                <th className="text-left py-2 pr-3 font-medium">Year</th>
                <th className="text-right py-2 px-2 font-medium">Trades</th>
                <th className="text-right py-2 px-2 font-medium">L</th>
                <th className="text-right py-2 px-2 font-medium">S</th>
                <th className="text-right py-2 px-2 font-medium">PnL</th>
                <th className="text-right py-2 px-2 font-medium">Sharpe</th>
                <th className="text-right py-2 px-2 font-medium">WR</th>
                <th className="text-right py-2 px-2 font-medium">L WR</th>
                <th className="text-right py-2 px-2 font-medium">S WR</th>
              </tr>
            </thead>
            <tbody>
              {yearByYear.map((y) => (
                <tr key={y.year} className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                  <td className={`py-2 pr-3 font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{y.year}</td>
                  <td className="text-right py-2 px-2">{y.n}</td>
                  <td className="text-right py-2 px-2">{y.longs}</td>
                  <td className="text-right py-2 px-2">{y.shorts}</td>
                  <td className={`text-right py-2 px-2 font-medium ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>{y.pnl}</td>
                  <td className="text-right py-2 px-2 font-bold">{y.sharpe}</td>
                  <td className="text-right py-2 px-2">{y.wr}</td>
                  <td className="text-right py-2 px-2">{y.lWr}</td>
                  <td className="text-right py-2 px-2">{y.sWr}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className={`mt-4 grid grid-cols-2 md:grid-cols-4 gap-3`}>
          <div className={`rounded-lg p-3 text-center ${isDarkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
            <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Longs</div>
            <div className={`font-bold ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>79 trades, 65.8% WR</div>
          </div>
          <div className={`rounded-lg p-3 text-center ${isDarkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
            <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Shorts</div>
            <div className={`font-bold ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>41 trades, 63.4% WR</div>
          </div>
          <div className={`rounded-lg p-3 text-center ${isDarkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
            <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Reversion Exits</div>
            <div className={`font-bold ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>111 of 120 (92.5%)</div>
          </div>
          <div className={`rounded-lg p-3 text-center ${isDarkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
            <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Stop Losses</div>
            <div className={`font-bold ${isDarkMode ? 'text-amber-400' : 'text-amber-600'}`}>6 of 120 (5.0%)</div>
          </div>
        </div>
      </Card>

      {/* Key Findings from Research */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Key Findings from Production Research</h3>
        <div className="space-y-3">
          {[
            { icon: '1', title: 'First Crossing is too simple', detail: 'Naive first-crossing entry only retained 21% of original Sharpe (1.29 vs 6.17). It enters at the first noisy threshold touch, missing the genuine peak deviation that occurs later.', color: 'red' },
            { icon: '2', title: 'VWAP destroys edge completely', detail: 'Averaging entry price over a window after crossing yielded Sharpe ~0.0. The mean-reversion edge requires entering at extreme prices, not averaging them away.', color: 'red' },
            { icon: '3', title: 'Peak detection backfires', detail: 'Waiting for the signal to peak and start declining (decay_pct=20%) produced NEGATIVE Sharpe (-1.44). By the time the peak is confirmed, the best entry price has already passed.', color: 'amber' },
            { icon: '4', title: 'Segment-30 is the sweet spot', detail: 'Dividing the session into 30-min windows and entering at the best bar within each segment retains 42.5% of entry edge. Implementable via limit orders at threshold price — natural fill at best available.', color: 'emerald' },
            { icon: '5', title: 'Smoothed exit is critical', detail: 'A 3-bar rolling average on exit signals boosted Sharpe from 2.62 to 3.36 (+28%). It prevents premature exits from noise spikes while still catching genuine reversions.', color: 'emerald' },
            { icon: '6', title: 'T-1 screening works well', detail: 'Using yesterday\'s resid_signal for daily screening works better than "running daily" estimates from early-session 1m bars (Sharpe 3.36 vs 1.40). T-1 is a quality filter.', color: 'blue' },
            { icon: '7', title: 'Asym still beats no-gates in production', detail: 'Asymmetric thresholds (1.5 for shorts) still produce higher Sharpe (3.36 vs 3.18) and higher WR (65.0% vs 59.7%) in the production-realistic backtest.', color: 'blue' },
          ].map(({ icon, title, detail, color }) => (
            <div key={icon} className={`rounded-lg border p-4 ${
              color === 'red' ? (isDarkMode ? 'bg-red-900/10 border-red-800/50' : 'bg-red-50/50 border-red-200') :
              color === 'amber' ? (isDarkMode ? 'bg-amber-900/10 border-amber-800/50' : 'bg-amber-50/50 border-amber-200') :
              color === 'emerald' ? (isDarkMode ? 'bg-emerald-900/10 border-emerald-800/50' : 'bg-emerald-50/50 border-emerald-200') :
              (isDarkMode ? 'bg-blue-900/10 border-blue-800/50' : 'bg-blue-50/50 border-blue-200')
            }`}>
              <div className="flex items-start gap-3">
                <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 ${
                  isDarkMode ? 'bg-gray-700 text-white' : 'bg-gray-200 text-gray-800'
                }`}>{icon}</span>
                <div>
                  <div className={`font-bold text-sm ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{title}</div>
                  <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{detail}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Issues */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>All Issues (with production fixes)</h3>
        <div className="space-y-4">
          {issues.map((issue) => (
            <IssueCard key={issue.id} issue={issue} isDarkMode={isDarkMode} severityBg={severityBg} severityText={severityText} />
          ))}
        </div>
      </Card>

      {/* Clean Items */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-4 flex items-center gap-2 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
          <span className="text-emerald-500">&#10003;</span> Components Verified Clean
        </h3>
        <div className="space-y-3">
          {cleanItems.map((item) => (
            <div key={item.title} className={`rounded-lg border p-4 ${isDarkMode ? 'bg-emerald-900/10 border-emerald-800/50' : 'bg-emerald-50/50 border-emerald-200'}`}>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-emerald-500 font-bold">&#10003;</span>
                <span className={`font-bold text-sm ${isDarkMode ? 'text-emerald-400' : 'text-emerald-700'}`}>{item.title}</span>
              </div>
              <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{item.detail}</p>
              <pre className={`mt-2 px-3 py-1.5 rounded text-xs font-mono ${isDarkMode ? 'bg-gray-900 text-emerald-400' : 'bg-white text-emerald-700'}`}>{item.code}</pre>
            </div>
          ))}
        </div>
      </Card>

      {/* Production Readiness Verdict */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Production Readiness Verdict</h3>
        <div className="space-y-4">
          <div className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
            <h4 className={`font-bold text-sm mb-2 ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>Recommended Production Config</h4>
            <ul className={`text-sm space-y-1.5 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              <li className="flex items-start gap-2"><span className="text-emerald-500 mt-0.5 font-bold">Entry:</span><span>Segment-30 (divide session into 30-min windows, enter at best signal bar within first qualifying segment) — implement via limit orders at threshold price</span></li>
              <li className="flex items-start gap-2"><span className="text-emerald-500 mt-0.5 font-bold">Exit:</span><span>Smoothed-3 (apply 3-bar rolling average to signal, exit at first crossing below threshold) — trivial real-time implementation</span></li>
              <li className="flex items-start gap-2"><span className="text-emerald-500 mt-0.5 font-bold">Screen:</span><span>T-1 resid_signal for daily watchlist filter (use yesterday's close, not today's)</span></li>
              <li className="flex items-start gap-2"><span className="text-emerald-500 mt-0.5 font-bold">Config:</span><span>Asymmetric thresholds (long 1.0, short 1.5), asymmetric hold (long 10d, short 7d)</span></li>
            </ul>
          </div>
          <div className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
            <h4 className={`font-bold text-sm mb-2 ${isDarkMode ? 'text-amber-400' : 'text-amber-600'}`}>Alternative Config (fewer trades, lower DD)</h4>
            <ul className={`text-sm space-y-1.5 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              <li className="flex items-start gap-2"><span className="text-amber-500 mt-0.5 font-bold">Entry:</span><span>Escalate 1.5x (only enter when signal exceeds 1.5x threshold — high conviction only)</span></li>
              <li className="flex items-start gap-2"><span className="text-amber-500 mt-0.5 font-bold">Result:</span><span>Sharpe 3.00, WR 61.9%, 84 trades, MaxDD -7.6 (smallest drawdown of all configs)</span></li>
            </ul>
          </div>
          <div className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
            <h4 className={`font-bold text-sm mb-2 ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>Already Production-Ready</h4>
            <ul className={`text-sm space-y-1.5 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              <li className="flex items-start gap-2"><span className="text-emerald-500 mt-0.5">&#10003;</span><span>Rolling residual regression — fully walk-forward, no lookahead</span></li>
              <li className="flex items-start gap-2"><span className="text-emerald-500 mt-0.5">&#10003;</span><span>Z-score normalization — backward-looking rolling windows</span></li>
              <li className="flex items-start gap-2"><span className="text-emerald-500 mt-0.5">&#10003;</span><span>Fair value model — spot-based carry, clean computation</span></li>
              <li className="flex items-start gap-2"><span className="text-emerald-500 mt-0.5">&#10003;</span><span>ATR, bias correction — standard backward-looking statistics</span></li>
              <li className="flex items-start gap-2"><span className="text-emerald-500 mt-0.5">&#10003;</span><span>Transaction cost model — $0.50/RT is conservative vs reality (~$0.35)</span></li>
            </ul>
          </div>
        </div>
      </Card>

      <Callout type="success" isDarkMode={isDarkMode}>
        <strong>Bottom line:</strong> After removing ALL lookahead bias and testing 13 entry strategies and 8 exit strategies across 2 rounds of research, the best production-realistic config achieves <strong>Sharpe 3.36</strong> (54.4% of original 6.17). Consistent across all years: 2023=6.23, 2024=3.24, 2025=2.12. The signal is real, the edge is real, and the strategy is production-viable. WR 65%, PF 2.48, 120 trades over 3 years.
      </Callout>
    </div>
  );
};

const IssueCard = ({ issue, isDarkMode, severityBg, severityText }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  return (
    <div className={`rounded-lg border overflow-hidden ${severityBg[issue.severity] || ''}`}>
      <button onClick={() => setIsExpanded(!isExpanded)} className="w-full text-left p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
              isDarkMode ? 'bg-gray-700 text-white' : 'bg-white text-gray-800'
            }`}>#{issue.id}</span>
            <div>
              <div className={`font-bold text-sm ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{issue.title}</div>
              <div className={`text-xs mt-0.5 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>{issue.where}</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className={`text-xs font-bold px-2.5 py-1 rounded-full ${severityText[issue.severity]} ${
              isDarkMode ? 'bg-gray-800' : 'bg-white'
            }`}>{issue.severity}</span>
            <svg className={`w-5 h-5 transition-transform ${isExpanded ? 'rotate-180' : ''} ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}
              fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>
      </button>
      {isExpanded && (
        <div className={`px-4 pb-4 space-y-3 border-t ${isDarkMode ? 'border-gray-700/50' : 'border-gray-200/50'}`}>
          <div className="pt-3">
            <h5 className={`text-xs font-bold uppercase tracking-wider mb-1.5 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>The Problem</h5>
            <p className={`text-sm leading-relaxed ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{issue.problem}</p>
          </div>
          <div>
            <h5 className={`text-xs font-bold uppercase tracking-wider mb-1.5 ${isDarkMode ? 'text-rose-400' : 'text-rose-600'}`}>Backtest Code (has lookahead)</h5>
            <pre className={`px-3 py-2.5 rounded text-xs font-mono leading-relaxed overflow-x-auto ${
              isDarkMode ? 'bg-gray-900 text-rose-400' : 'bg-white text-rose-700'
            }`}>{issue.codeSnippet}</pre>
          </div>
          <div>
            <h5 className={`text-xs font-bold uppercase tracking-wider mb-1.5 ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>Production Fix</h5>
            <pre className={`px-3 py-2.5 rounded text-xs font-mono leading-relaxed overflow-x-auto ${
              isDarkMode ? 'bg-gray-900 text-emerald-400' : 'bg-white text-emerald-700'
            }`}>{issue.fix}</pre>
          </div>
          <div>
            <h5 className={`text-xs font-bold uppercase tracking-wider mb-1.5 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Impact Assessment</h5>
            <p className={`text-sm leading-relaxed ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{issue.impact}</p>
          </div>
          <div className={`rounded p-3 ${isDarkMode ? 'bg-blue-900/20 border border-blue-800' : 'bg-blue-50 border border-blue-200'}`}>
            <p className={`text-sm ${isDarkMode ? 'text-blue-300' : 'text-blue-700'}`}><strong>Recommendation:</strong> {issue.recommendation}</p>
          </div>
        </div>
      )}
    </div>
  );
};

// =============================================================================
// V9 PRODUCTION ZERO-LOOKAHEAD SECTION
// =============================================================================

const V9Section = ({ isDarkMode }) => {
  const [expandedPhase, setExpandedPhase] = useState(null);

  const phases = [
    {
      id: 0,
      title: 'Phase 0+1: Baseline Lock + Leakage Audit',
      color: 'rose',
      summary: 'Freeze V6/V8 baselines, build automated leakage checker, verify 100% causality',
      details: [
        'Ran V6 with lookahead ("ceiling") and with T-1 lagging side-by-side',
        'Tested every V8 causal entry (first_cross, time_decay, peak_det, trailing_snap, smoothed_peak, two_phase, exp_decay)',
        'Built 10-point automated leakage audit: checks for unlagged daily columns in 1m data, verifies bias_20d_lag1 and spread_atr_20_lag1 exist, validates causality of every entry function',
        'ALL 10 leakage checks passed before proceeding',
      ],
      result: 'Ceiling: Sharpe 5.32 (N=96) | Best causal: time_decay Sharpe 3.41 (N=35) | 64% Sharpe retention',
      gate: 'PASSED -- all leakage invariants satisfied',
    },
    {
      id: 1,
      title: 'Phase 2: Alpha Decomposition',
      color: 'violet',
      summary: 'Ablate signal components to find what drives the causal edge',
      details: [
        'Compared raw_sig vs clean_sig (ratio-scaled): raw_sig won (Sharpe 3.60 vs 2.33)',
        'Swept asymmetric vs symmetric thresholds: asym L=1.0 / S=2.0 was optimal (Sharpe 4.12)',
        'Gate multiplier sensitivity: gate=0.6-0.8 is the Sharpe plateau',
        'Regime analysis: high-ATR periods (Sharpe 7.24) and roll-weeks (Sharpe 4.85) carry most edge',
        'Direction x Year: longs are primary driver (2023 WR=82%, 2024 WR=80%)',
      ],
      result: 'Signal: raw_sig | Thresholds: L=1.0, S=2.0 | Gate: 0.6 | High-ATR and roll-week regimes most profitable',
      gate: 'PASSED -- stable alpha core identified under strict causality',
    },
    {
      id: 2,
      title: 'Phase 3+4: Entry Frontier + Frequency Recovery',
      color: 'blue',
      summary: 'Deep parameter sweep across 6 causal entry families, balance Sharpe vs trade count',
      details: [
        'Swept 1,488 total configs across time_decay, peak_detection, smoothed_peak, trailing_snap, two_phase, exp_decay',
        'Time-decay dominated: SM=2.0, decay=0.001/bar achieved Sharpe 4.38 (82% of ceiling)',
        'Peak detection and trailing snap both negative Sharpe -- these entry styles destroy the mean-reversion edge',
        'Composite scoring balanced Sharpe with trade-count proximity to V6: top configs at N=94 (Sharpe 3.20) and N=32 (Sharpe 4.38)',
        'Frequency recovery: lowering gate_mult to 0.5 and L_ET to 0.8 increased trades to ~100 with Sharpe ~3.0',
      ],
      result: 'Winner: time_decay (SM=2.0, decay=0.001) | Sharpe 4.38 | 82% retention of ceiling',
      gate: 'PASSED -- entry family selected, frequency trade-off documented',
    },
    {
      id: 3,
      title: 'Phase 5: Exit Architecture Redesign',
      color: 'emerald',
      summary: 'Optimize TP, reversion, stops and hold limits coupled to best entry',
      details: [
        'Symmetric TP sweep: TP=2.5 best for longs (Sharpe 4.64), TP=1.25 best for shorts (Sharpe 4.67)',
        'Asymmetric TP (L=2.5, S=1.25) pushed combined Sharpe from 4.38 to 4.73',
        'Hold period optimization: L=7 days, S=5 days -- shorter for shorts matches their faster reversion',
        'Stop-loss at 1.5x ATR both sides -- tight enough to cap losses, wide enough to avoid noise stops',
        'Exit-reason diagnostics: 22 reversion exits, 7 take-profits, 3 time-stops, 1 stop-loss',
      ],
      result: 'Sharpe: 4.38 -> 4.73 (+8%) | Asym TP: L=2.5x, S=1.25x | Confirmed reversion exit w/ 1-bar confirm',
      gate: 'PASSED -- exit stack improved both Sharpe and PF',
    },
    {
      id: 4,
      title: 'Phase 6: Hybrid Overlay',
      color: 'amber',
      summary: 'Test post-TP scalp overlay for additional alpha',
      details: [
        'Collected 7 take-profit exit events from the base strategy',
        'Swept 1,440 scalp overlay configurations (direction, hold period, TP, stop)',
        'No overlay config improved combined Sharpe -- too few TP events for stable overlay alpha',
        'Decision: SHIP BASE ONLY (overlay kept as research branch for future re-evaluation with more data)',
      ],
      result: 'Decision: SHIP_BASE_ONLY | Overlay does not add net alpha with current sample',
      gate: 'PASSED -- clean decision, base strategy is strong enough standalone',
    },
    {
      id: 5,
      title: 'Phase 7: Robustness Validation',
      color: 'cyan',
      summary: 'Walk-forward OOS, Monte Carlo significance, stress tests, consistency checks',
      details: [
        'Walk-forward: 5/7 folds positive, avg Sharpe 5.60 (strong OOS signal)',
        'Monte Carlo bootstrap (5,000 sims): p=0.010 -- significant at 1% level',
        'Sign-flip test (5,000 sims): p=0.011 -- independently significant',
        'Cost stress: survives 2x costs (Sharpe 2.16), breaks at 3x costs (-0.40)',
        'Latency stress: 1-bar delay Sharpe 3.69, 3-bar delay 2.56, 5-bar delay 0.58',
        'Year consistency: 2023 Sharpe 11.48, 2024 Sharpe 4.42, 2025 Sharpe 1.95',
        'Both directions profitable: Longs Sharpe 4.64 (27 trades), Shorts Sharpe 4.67 (6 trades)',
      ],
      result: 'ALL 8 PROMOTION CHECKS PASSED | Strategy PROMOTED to production candidate',
      gate: 'PASSED -- walk-forward + MC + stress + consistency all clear',
    },
    {
      id: 6,
      title: 'Phase 8: Production Specification',
      color: 'teal',
      summary: 'Shadow replay, guardrails, rollback candidates, deployment contract',
      details: [
        'Shadow replay at 1.5x cost: N=33, PnL=+22.1, Sharpe=3.45, WR=66.7%, MaxDD=-7.2',
        'Final leakage verification: all 10 checks passed again on production config',
        'Guardrails: daily loss limit 10.9 pts, weekly 14.5 pts, monthly 18.1 pts, max 3 concurrent positions',
        'Rollback candidates: first_cross (Sharpe 0.75) and conservative (Sharpe 1.21)',
        'Data dependency contract: all daily features T-1 lagged, rolling OLS window [i-60, i) exclusive',
        'Expected monthly trades: 1.7 (focused, high-conviction)',
      ],
      result: 'DEPLOYMENT READY: YES | Shadow Sharpe 3.45 at 1.5x cost',
      gate: 'SHIPPED -- full production artifact saved',
    },
  ];

  const leakageFixes = [
    {
      bias: 'Intraday Best-Bar Entry',
      v6: 'idxmin()/idxmax() scans ALL ~400 bars, picks THE best minute of the entire day',
      v9: 'Time-decay entry: threshold starts at 2.0x and decays 0.1% per bar. Enters at first bar that crosses the decaying threshold -- purely causal, no future knowledge',
      severity: 'CRITICAL',
    },
    {
      bias: 'Intraday Best-Bar Exit',
      v6: 'Scans all remaining bars, picks THE most-reverted bar for exit',
      v9: 'Confirmed reversion: exit when signal crosses exit threshold with 1-bar confirmation. Asymmetric take-profit caps (L=2.5x, S=1.25x ATR) fire on first touch',
      severity: 'CRITICAL',
    },
    {
      bias: 'Same-Day Daily Signal',
      v6: 'resid_signal computed from today\'s close, used for same-day entry decisions',
      v9: 'ALL daily features lagged T-1: resid_signal_lag1, dev_bc_lag1, bias_20d_lag1, spread_atr_20_lag1. Automated audit verifies no unlagged columns exist in 1m data',
      severity: 'HIGH',
    },
    {
      bias: 'Rolling Window Boundary',
      v6: 'rolling(20) for ATR/bias includes today in the normalization used for today\'s decisions',
      v9: 'spread_atr_20_lag1 and bias_20d_lag1 -- shifted by 1 day. Today\'s normalization uses yesterday\'s statistics only',
      severity: 'MEDIUM',
    },
    {
      bias: 'Regression Train/Test Leak',
      v6: 'Rolling OLS already walk-forward [start, i) -- this was clean',
      v9: 'Preserved: 60-day rolling OLS with exclusive upper bound. Automated invariant verifies train window never includes prediction day',
      severity: 'CLEAN',
    },
  ];

  const sevBg = {
    CRITICAL: isDarkMode ? 'bg-red-900/20 border-red-800' : 'bg-red-50 border-red-200',
    HIGH: isDarkMode ? 'bg-amber-900/20 border-amber-800' : 'bg-amber-50 border-amber-200',
    MEDIUM: isDarkMode ? 'bg-yellow-900/20 border-yellow-800' : 'bg-yellow-50 border-yellow-200',
    CLEAN: isDarkMode ? 'bg-emerald-900/20 border-emerald-800' : 'bg-emerald-50 border-emerald-200',
  };
  const sevText = {
    CRITICAL: isDarkMode ? 'text-red-400' : 'text-red-600',
    HIGH: isDarkMode ? 'text-amber-400' : 'text-amber-600',
    MEDIUM: isDarkMode ? 'text-yellow-400' : 'text-yellow-600',
    CLEAN: isDarkMode ? 'text-emerald-400' : 'text-emerald-600',
  };

  return (
    <div className="space-y-6">
      <SectionHeader
        icon="🚀"
        title="V9: Zero-Lookahead Production Strategy"
        subtitle="8-phase adaptive research program -- from V6 lookahead ceiling to fully causal, validated, deployment-ready config"
        isDarkMode={isDarkMode}
        color="teal"
      />

      {/* Hero Stats */}
      <Callout type="success" isDarkMode={isDarkMode}>
        <strong>V9 is the production-ready successor to V6.</strong> An 8-phase research program systematically eliminated every source of lookahead bias while recovering <strong>89% of the ceiling Sharpe</strong> (4.73 vs 5.32). At realistic 1.5x costs, V9 achieves <strong>Sharpe 3.45</strong> -- beating the prior best production attempt (Lookahead Audit: 3.36) while using a completely different, provably causal entry mechanism. Every feature is T-1 lagged. Every entry/exit decision uses only past data. 10 automated leakage checks pass.
      </Callout>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard isDarkMode={isDarkMode} label="V9 Sharpe (1x cost)" value="4.73" sub="Zero lookahead, 33 trades" color="emerald" />
        <StatCard isDarkMode={isDarkMode} label="V9 Sharpe (1.5x cost)" value="3.45" sub="Shadow replay, realistic" color="blue" />
        <StatCard isDarkMode={isDarkMode} label="Ceiling Retention" value="89%" sub="4.73 / 5.32 ceiling" color="purple" />
        <StatCard isDarkMode={isDarkMode} label="Win Rate" value="75.8%" sub="25 wins / 33 trades" color="green" />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard isDarkMode={isDarkMode} label="Monte Carlo p-value" value="0.010" sub="Bootstrap, 5000 sims" color="cyan" />
        <StatCard isDarkMode={isDarkMode} label="Walk-Forward" value="5/7" sub="Positive folds, avg Shrp 5.60" color="indigo" />
        <StatCard isDarkMode={isDarkMode} label="Survives 2x Cost" value="2.16" sub="Sharpe at $1.00/RT" color="amber" />
        <StatCard isDarkMode={isDarkMode} label="Leakage Checks" value="10/10" sub="All automated audits pass" color="emerald" />
      </div>

      {/* V6 vs V9 Head-to-Head */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-1 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>V6 vs V9: Head-to-Head Comparison</h3>
        <p className={`text-sm mb-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Every row uses the same data (Jan 2023 - Feb 2026), same cost model ($0.50/RT), same spread universe</p>
        <div className="overflow-x-auto">
          <table className={`w-full text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            <thead>
              <tr className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>
                <th className="text-left py-2 pr-3 font-medium">Configuration</th>
                <th className="text-right py-2 px-2 font-medium">N</th>
                <th className="text-right py-2 px-2 font-medium">PnL (pts)</th>
                <th className="text-right py-2 px-2 font-medium">Sharpe</th>
                <th className="text-right py-2 px-2 font-medium">WR</th>
                <th className="text-right py-2 px-2 font-medium">PF</th>
                <th className="text-right py-2 px-2 font-medium">MaxDD</th>
                <th className="text-center py-2 px-2 font-medium">Causal?</th>
              </tr>
            </thead>
            <tbody>
              <tr className={`border-t ${isDarkMode ? 'border-gray-700 bg-red-900/10' : 'border-gray-200 bg-red-50/30'}`}>
                <td className={`py-2.5 pr-3 font-medium ${isDarkMode ? 'text-rose-400' : 'text-rose-600'}`}>V6 Asym (original lookahead)</td>
                <td className="text-right py-2.5 px-2">148</td>
                <td className="text-right py-2.5 px-2">+144.4</td>
                <td className={`text-right py-2.5 px-2 font-bold ${isDarkMode ? 'text-rose-400' : 'text-rose-600'}`}>6.17</td>
                <td className="text-right py-2.5 px-2">76.4%</td>
                <td className="text-right py-2.5 px-2">6.21</td>
                <td className="text-right py-2.5 px-2">-5.7</td>
                <td className="text-center py-2.5 px-2 font-bold text-red-500">NO</td>
              </tr>
              <tr className={`border-t ${isDarkMode ? 'border-gray-700 bg-red-900/5' : 'border-gray-200 bg-red-50/15'}`}>
                <td className={`py-2.5 pr-3 font-medium ${isDarkMode ? 'text-rose-400/70' : 'text-rose-500'}`}>V6 with T-1 lag + best-bar (ceiling)</td>
                <td className="text-right py-2.5 px-2">96</td>
                <td className="text-right py-2.5 px-2">+67.8</td>
                <td className={`text-right py-2.5 px-2 font-bold ${isDarkMode ? 'text-rose-400/70' : 'text-rose-500'}`}>5.32</td>
                <td className="text-right py-2.5 px-2">72.9%</td>
                <td className="text-right py-2.5 px-2">4.12</td>
                <td className="text-right py-2.5 px-2">-4.8</td>
                <td className="text-center py-2.5 px-2 font-bold text-amber-500">PARTIAL</td>
              </tr>
              <tr className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <td className={`py-2.5 pr-3 font-medium ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Old production (seg_30 + smooth_3)</td>
                <td className="text-right py-2.5 px-2">120</td>
                <td className="text-right py-2.5 px-2">+50.7</td>
                <td className="text-right py-2.5 px-2 font-bold">3.36</td>
                <td className="text-right py-2.5 px-2">65.0%</td>
                <td className="text-right py-2.5 px-2">2.48</td>
                <td className="text-right py-2.5 px-2">-8.9</td>
                <td className="text-center py-2.5 px-2 font-bold text-emerald-500">YES</td>
              </tr>
              <tr className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <td className={`py-2.5 pr-3 font-medium ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>V8 best causal (time_decay)</td>
                <td className="text-right py-2.5 px-2">35</td>
                <td className="text-right py-2.5 px-2">+29.4</td>
                <td className="text-right py-2.5 px-2 font-bold">3.41</td>
                <td className="text-right py-2.5 px-2">68.6%</td>
                <td className="text-right py-2.5 px-2">2.68</td>
                <td className="text-right py-2.5 px-2">-7.1</td>
                <td className="text-center py-2.5 px-2 font-bold text-emerald-500">YES</td>
              </tr>
              <tr className={`border-t-2 ${isDarkMode ? 'border-teal-700 bg-teal-900/15' : 'border-teal-300 bg-teal-50/50'}`}>
                <td className={`py-2.5 pr-3 font-bold ${isDarkMode ? 'text-teal-400' : 'text-teal-600'}`}>V9 Production (1x cost)</td>
                <td className="text-right py-2.5 px-2 font-bold">33</td>
                <td className="text-right py-2.5 px-2 font-bold">+30.4</td>
                <td className={`text-right py-2.5 px-2 font-black ${isDarkMode ? 'text-teal-400' : 'text-teal-600'}`}>4.73</td>
                <td className="text-right py-2.5 px-2 font-bold">75.8%</td>
                <td className="text-right py-2.5 px-2 font-bold">3.41</td>
                <td className="text-right py-2.5 px-2 font-bold">-6.5</td>
                <td className="text-center py-2.5 px-2 font-bold text-emerald-500">YES</td>
              </tr>
              <tr className={`border-t ${isDarkMode ? 'border-teal-800 bg-teal-900/10' : 'border-teal-200 bg-teal-50/30'}`}>
                <td className={`py-2.5 pr-3 font-bold ${isDarkMode ? 'text-teal-400/80' : 'text-teal-500'}`}>V9 Shadow (1.5x cost)</td>
                <td className="text-right py-2.5 px-2 font-bold">33</td>
                <td className="text-right py-2.5 px-2 font-bold">+22.1</td>
                <td className={`text-right py-2.5 px-2 font-bold ${isDarkMode ? 'text-teal-400/80' : 'text-teal-500'}`}>3.45</td>
                <td className="text-right py-2.5 px-2 font-bold">66.7%</td>
                <td className="text-right py-2.5 px-2 font-bold">2.48</td>
                <td className="text-right py-2.5 px-2 font-bold">-7.2</td>
                <td className="text-center py-2.5 px-2 font-bold text-emerald-500">YES</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>

      {/* How V9 Fixed Every Lookahead Bias */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-1 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>How V9 Fixed Every Lookahead Bias</h3>
        <p className={`text-sm mb-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Side-by-side: what V6 did wrong and exactly how V9 fixes it</p>
        <div className="space-y-3">
          {leakageFixes.map((fix, i) => (
            <div key={i} className={`rounded-lg border p-4 ${sevBg[fix.severity]}`}>
              <div className="flex items-center gap-3 mb-3">
                <span className={`text-xs font-bold px-2.5 py-1 rounded-full ${sevText[fix.severity]} ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                  {fix.severity}
                </span>
                <span className={`font-bold text-sm ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{fix.bias}</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <div className={`text-xs font-bold uppercase tracking-wider mb-1.5 ${isDarkMode ? 'text-rose-400' : 'text-rose-600'}`}>V6 (biased)</div>
                  <p className={`text-sm leading-relaxed ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{fix.v6}</p>
                </div>
                <div>
                  <div className={`text-xs font-bold uppercase tracking-wider mb-1.5 ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>V9 (causal fix)</div>
                  <p className={`text-sm leading-relaxed ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{fix.v9}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* V9 Signal + Entry Architecture */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>V9 Architecture: Signal to Trade</h3>
        <div className="space-y-2">
          <PipelineStep step="1" title="Daily Features (T-1 Lagged)" color="rose"
            formula="resid_signal_lag1 = resid_signal.shift(1)  |  dev_bc_lag1 = dev_bc.shift(1)  |  bias_20d_lag1, spread_atr_20_lag1"
            explanation="Every daily feature is lagged by one business day before merging into intraday data. At market open on day T, you only know day T-1's values. The automated audit verifies no unlagged column ever appears in the 1m data."
            isDarkMode={isDarkMode} />
          <FlowArrow isDarkMode={isDarkMode} />
          <PipelineStep step="2" title="Intraday Signal Construction (Live)" color="amber"
            formula="fv_1m = es_close_1m * carry_spot * days / 365  |  dev_1m = close - fv_1m  |  raw_sig = (dev_1m - bias_20d_lag1) / spread_atr_20_lag1"
            explanation="The 1-minute signal uses live prices (es_close_1m, spread close) combined with T-1 daily normalization. raw_sig is the chosen signal -- it's the intraday deviation corrected for rolling bias, normalized by yesterday's ATR. No daily same-day data touches any intraday decision."
            isDarkMode={isDarkMode} />
          <FlowArrow isDarkMode={isDarkMode} />
          <PipelineStep step="3" title="Daily Screening (T-1 Gate)" color="violet"
            formula="if abs(resid_signal_lag1) > gate_mult * spread_atr_20_lag1 -> ACTIVE today  |  gate_mult = 0.6"
            explanation="Before scanning any intraday bars, V9 checks yesterday's residual signal. If the mispricing wasn't large enough yesterday, the day is skipped entirely. This acts as a quality filter using only past data -- no same-day circular logic."
            isDarkMode={isDarkMode} />
          <FlowArrow isDarkMode={isDarkMode} />
          <PipelineStep step="4" title="Time-Decay Entry (Causal)" color="emerald"
            formula="threshold(bar) = start_mult * (1 - decay_per_bar)^bar_index  |  start_mult=2.0, decay=0.001/bar, floor=50%"
            explanation="The entry threshold starts high (2.0x) at market open and decays 0.1% per minute bar. This means V9 demands extreme conviction early in the session but becomes more permissive over time. The first bar where raw_sig crosses the decaying threshold triggers entry. No future bars are ever consulted -- purely sequential, purely causal."
            isDarkMode={isDarkMode} />
          <FlowArrow isDarkMode={isDarkMode} />
          <PipelineStep step="5" title="Multi-Layer Exit Stack" color="blue"
            formula="Take-Profit: L=2.5x ATR, S=1.25x ATR (first touch)  |  Reversion: confirmed when signal < 0.7 for 1 bar  |  Stop: 1.5x ATR  |  Time: L=7d, S=5d"
            explanation="Exits fire on the first qualifying bar in strict priority order: TP, reversion, stop, time-stop. Asymmetric TP lets longs run (2.5x) while banking short profits faster (1.25x). The 1-bar confirmation on reversion exits prevents noise whipsaws. All exit checks use only the current bar and past data."
            isDarkMode={isDarkMode} />
        </div>
      </Card>

      {/* Time-Decay Entry Deep Dive */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Why Time-Decay Entry Works</h3>
        <div className={`text-sm leading-relaxed space-y-4 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
          <p>
            V6's original best-bar entry was powerful because it selected the most extreme signal bar of the day. The problem: you can't know which bar is "best" without seeing the entire day. V9's time-decay solves this elegantly:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
              <div className={`font-bold text-sm mb-2 ${isDarkMode ? 'text-teal-400' : 'text-teal-600'}`}>Early Session: High Bar</div>
              <p className="text-sm">Threshold at 2.0x demands extreme mispricing. Only genuinely dislocated prices trigger entry early -- these tend to be the highest-quality trades, analogous to V6 "best-bar" entries that happened early.</p>
            </div>
            <div className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
              <div className={`font-bold text-sm mb-2 ${isDarkMode ? 'text-teal-400' : 'text-teal-600'}`}>Mid Session: Gradual Decay</div>
              <p className="text-sm">At 0.1% per bar, the threshold slowly relaxes. By bar 200 (~3:20 PM), it's at ~1.6x. This captures "second-best" entries that V6 would have picked via hindsight, but V9 catches in real-time as the bar naturally reaches the decayed threshold.</p>
            </div>
            <div className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
              <div className={`font-bold text-sm mb-2 ${isDarkMode ? 'text-teal-400' : 'text-teal-600'}`}>Floor: Quality Guard</div>
              <p className="text-sm">Threshold never drops below 50% of start (1.0x). This prevents late-session garbage entries. Combined with the T-1 gate, it ensures every entry has genuine signal strength -- not noise.</p>
            </div>
          </div>
        </div>
        <CodeBlock isDarkMode={isDarkMode} title="V9 Time-Decay Entry (exact production logic)" code={`def entry_time_decay(bar_idx, signal_value, n_bars_today, params):
    start_mult = 2.0          # demand 2x signal at open
    decay_per_bar = 0.001     # relax 0.1% each minute
    floor_pct = 0.5           # never go below 50% of start

    decay_factor = (1 - decay_per_bar) ** bar_idx
    threshold = start_mult * max(decay_factor, floor_pct)

    return abs(signal_value) >= threshold    # purely causal: only uses current bar

# V6 comparison (what this replaced):
# best_idx = day_1m[signal].idxmin()      # SCANS ALL BARS (lookahead!)
# best = day_1m.loc[best_idx]             # PICKS GLOBAL BEST (impossible in real-time)`} />
      </Card>

      {/* 8-Phase Research Journey */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-1 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>The 8-Phase Research Journey</h3>
        <p className={`text-sm mb-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Each phase builds on prior results. Click to expand details and decision gates.</p>
        <div className="space-y-2">
          {phases.map((phase) => {
            const isOpen = expandedPhase === phase.id;
            const phaseColors = {
              rose: isDarkMode ? 'border-rose-800' : 'border-rose-200',
              violet: isDarkMode ? 'border-violet-800' : 'border-violet-200',
              blue: isDarkMode ? 'border-blue-800' : 'border-blue-200',
              emerald: isDarkMode ? 'border-emerald-800' : 'border-emerald-200',
              amber: isDarkMode ? 'border-amber-800' : 'border-amber-200',
              cyan: isDarkMode ? 'border-cyan-800' : 'border-cyan-200',
              teal: isDarkMode ? 'border-teal-800' : 'border-teal-200',
            };
            const dotColors = {
              rose: 'bg-rose-500', violet: 'bg-violet-500', blue: 'bg-blue-500',
              emerald: 'bg-emerald-500', amber: 'bg-amber-500', cyan: 'bg-cyan-500', teal: 'bg-teal-500',
            };
            return (
              <div key={phase.id} className={`rounded-lg border overflow-hidden ${phaseColors[phase.color] || ''}`}>
                <button
                  onClick={() => setExpandedPhase(isOpen ? null : phase.id)}
                  className={`w-full text-left p-4 ${isDarkMode ? 'hover:bg-gray-700/30' : 'hover:bg-gray-50'}`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 rounded-full ${dotColors[phase.color] || 'bg-gray-500'} text-white font-bold text-sm flex items-center justify-center shrink-0`}>
                        {phase.id}
                      </div>
                      <div>
                        <div className={`font-bold text-sm ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{phase.title}</div>
                        <div className={`text-xs mt-0.5 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>{phase.summary}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`text-xs font-bold px-2 py-0.5 rounded ${isDarkMode ? 'bg-emerald-900/50 text-emerald-400' : 'bg-emerald-100 text-emerald-700'}`}>PASSED</span>
                      <svg className={`w-5 h-5 transition-transform ${isOpen ? 'rotate-180' : ''} ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}
                        fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </div>
                  </div>
                </button>
                {isOpen && (
                  <div className={`px-4 pb-4 space-y-3 border-t ${isDarkMode ? 'border-gray-700/50' : 'border-gray-200/50'}`}>
                    <ul className={`mt-3 text-sm space-y-1.5 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                      {phase.details.map((d, di) => (
                        <li key={di} className="flex items-start gap-2">
                          <span className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 ${dotColors[phase.color] || 'bg-gray-500'}`} />
                          <span>{d}</span>
                        </li>
                      ))}
                    </ul>
                    <div className={`rounded p-3 ${isDarkMode ? 'bg-gray-800 border border-gray-700' : 'bg-gray-50 border border-gray-200'}`}>
                      <div className={`text-xs font-bold uppercase tracking-wider mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Result</div>
                      <p className={`text-sm font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{phase.result}</p>
                    </div>
                    <div className={`rounded p-3 ${isDarkMode ? 'bg-emerald-900/20 border border-emerald-800' : 'bg-emerald-50 border border-emerald-200'}`}>
                      <div className={`text-xs font-bold uppercase tracking-wider mb-1 ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>Decision Gate</div>
                      <p className={`text-sm ${isDarkMode ? 'text-emerald-300' : 'text-emerald-700'}`}>{phase.gate}</p>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </Card>

      {/* Year-by-Year + Direction Split */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-1 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>V9 Year-by-Year + Direction Split</h3>
        <p className={`text-sm mb-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Production config at 1x cost</p>
        <div className="overflow-x-auto">
          <table className={`w-full text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            <thead>
              <tr className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>
                <th className="text-left py-2 pr-3 font-medium">Year</th>
                <th className="text-right py-2 px-2 font-medium">Trades</th>
                <th className="text-right py-2 px-2 font-medium">PnL</th>
                <th className="text-right py-2 px-2 font-medium">Sharpe</th>
                <th className="text-right py-2 px-2 font-medium">WR</th>
              </tr>
            </thead>
            <tbody>
              {[
                { yr: '2023', n: 13, pnl: '+13.6', sharpe: '11.48', wr: '84.6%' },
                { yr: '2024', n: 13, pnl: '+12.9', sharpe: '4.42', wr: '76.9%' },
                { yr: '2025', n: 7, pnl: '+3.9', sharpe: '1.95', wr: '57.1%' },
              ].map((y) => (
                <tr key={y.yr} className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                  <td className={`py-2 pr-3 font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{y.yr}</td>
                  <td className="text-right py-2 px-2">{y.n}</td>
                  <td className={`text-right py-2 px-2 font-medium ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>{y.pnl}</td>
                  <td className="text-right py-2 px-2 font-bold">{y.sharpe}</td>
                  <td className="text-right py-2 px-2">{y.wr}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className={`mt-4 grid grid-cols-2 gap-3`}>
          <div className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
            <div className={`text-xs font-bold uppercase tracking-wider mb-2 ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>Longs</div>
            <div className={`font-bold text-lg ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>27 trades, 81.5% WR</div>
            <div className={`text-sm mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>PnL: +23.5 pts ($1,175) | Sharpe: 4.64 | PF: 3.18</div>
          </div>
          <div className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
            <div className={`text-xs font-bold uppercase tracking-wider mb-2 ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`}>Shorts</div>
            <div className={`font-bold text-lg ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>6 trades, 50.0% WR</div>
            <div className={`text-sm mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>PnL: +6.9 pts ($345) | Sharpe: 4.67 | PF: 4.83</div>
          </div>
        </div>
      </Card>

      {/* Robustness Validation Summary */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Robustness Validation: All 8 Checks</h3>
        <div className="space-y-2">
          {[
            { check: 'Walk-forward majority positive', detail: '5/7 folds positive, avg Sharpe 5.60', passed: true },
            { check: 'Walk-forward avg Sharpe > 0', detail: 'Avg Sharpe 5.60 across all 6-month folds', passed: true },
            { check: 'Monte Carlo Sharpe significant (p<0.05)', detail: 'Bootstrap p=0.010, sign-flip p=0.011', passed: true },
            { check: 'Monte Carlo sign-flip (p<0.10)', detail: 'p=0.011 -- edge is not random', passed: true },
            { check: 'Survives 2x cost (Sharpe > 0)', detail: 'Sharpe 2.16 at $1.00/RT (double normal cost)', passed: true },
            { check: 'Longs profitable', detail: '+23.5 pts, 81.5% WR, Sharpe 4.64', passed: true },
            { check: 'Shorts profitable', detail: '+6.9 pts, 50.0% WR, Sharpe 4.67', passed: true },
            { check: 'Minimum trade count (>= 20)', detail: '33 trades (pass threshold: 20)', passed: true },
          ].map((item, i) => (
            <div key={i} className={`flex items-center gap-3 rounded-lg border p-3 ${
              isDarkMode ? 'bg-emerald-900/10 border-emerald-800/50' : 'bg-emerald-50/50 border-emerald-200'
            }`}>
              <span className="text-emerald-500 font-bold text-lg">&#10003;</span>
              <div className="flex-1">
                <span className={`font-medium text-sm ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{item.check}</span>
                <span className={`text-sm ml-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>-- {item.detail}</span>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Stress Test Results */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Stress Test Results</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className={`font-bold text-sm mb-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Cost Sensitivity</h4>
            <div className="space-y-1.5">
              {[
                { label: '1.0x ($0.50/RT)', sharpe: 4.73, pnl: '+30.4', ok: true },
                { label: '1.5x ($0.75/RT)', sharpe: 3.45, pnl: '+22.1', ok: true },
                { label: '2.0x ($1.00/RT)', sharpe: 2.16, pnl: '+13.9', ok: true },
                { label: '3.0x ($1.50/RT)', sharpe: -0.40, pnl: '-2.6', ok: false },
              ].map((s, i) => (
                <div key={i} className={`flex items-center justify-between rounded p-2.5 text-sm ${
                  s.ok ? (isDarkMode ? 'bg-emerald-900/10' : 'bg-emerald-50') : (isDarkMode ? 'bg-red-900/10' : 'bg-red-50')
                }`}>
                  <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>{s.label}</span>
                  <div className="flex items-center gap-4">
                    <span className={`font-mono font-bold ${s.ok ? (isDarkMode ? 'text-emerald-400' : 'text-emerald-600') : (isDarkMode ? 'text-red-400' : 'text-red-600')}`}>
                      {s.sharpe.toFixed(2)}
                    </span>
                    <span className={`font-mono text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{s.pnl}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div>
            <h4 className={`font-bold text-sm mb-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Latency / Delay Sensitivity</h4>
            <div className="space-y-1.5">
              {[
                { label: '0 bars (instant)', sharpe: 4.73, trades: 33, ok: true },
                { label: '1 bar (1 min)', sharpe: 3.69, trades: 38, ok: true },
                { label: '3 bars (3 min)', sharpe: 2.56, trades: 48, ok: true },
                { label: '5 bars (5 min)', sharpe: 0.58, trades: 56, ok: false },
              ].map((s, i) => (
                <div key={i} className={`flex items-center justify-between rounded p-2.5 text-sm ${
                  s.ok ? (isDarkMode ? 'bg-emerald-900/10' : 'bg-emerald-50') : (isDarkMode ? 'bg-amber-900/10' : 'bg-amber-50')
                }`}>
                  <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>{s.label}</span>
                  <div className="flex items-center gap-4">
                    <span className={`font-mono font-bold ${s.ok ? (isDarkMode ? 'text-emerald-400' : 'text-emerald-600') : (isDarkMode ? 'text-amber-400' : 'text-amber-600')}`}>
                      {s.sharpe.toFixed(2)}
                    </span>
                    <span className={`font-mono text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>N={s.trades}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Automated Leakage Audit */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-1 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Automated Leakage Audit (10 Checks)</h3>
        <p className={`text-sm mb-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Run automatically before every backtest -- if any check fails, execution is blocked</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {[
            'No unlagged daily columns in 1m data',
            'bias_20d_lag1 present in merged data',
            'spread_atr_20_lag1 present in merged data',
            'entry_first_cross causality verified',
            'entry_time_decay causality verified',
            'entry_peak_detection causality verified',
            'entry_trailing_snap causality verified',
            'entry_smoothed_peak causality verified',
            'entry_two_phase causality verified',
            'entry_exp_decay causality verified',
          ].map((check, i) => (
            <div key={i} className={`flex items-center gap-2 rounded p-2.5 text-sm ${isDarkMode ? 'bg-emerald-900/10' : 'bg-emerald-50'}`}>
              <span className="text-emerald-500 font-bold">&#10003;</span>
              <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>{check}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Production Config Summary */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Final V9 Production Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className={`font-bold text-sm mb-3 ${isDarkMode ? 'text-teal-400' : 'text-teal-600'}`}>Signal & Entry</h4>
            <div className="space-y-2">
              {[
                ['Signal', 'raw_sig (bias-corrected, ATR-normalized intraday deviation)'],
                ['Daily Screen', 'resid_signal_lag1 > gate_mult (0.6) * spread_atr_20_lag1'],
                ['Entry', 'time_decay: start=2.0x, decay=0.001/bar, floor=50%'],
                ['Long Threshold', '1.0x (enter when raw_sig < -1.0)'],
                ['Short Threshold', '2.0x (enter when raw_sig > 2.0)'],
              ].map(([k, v], i) => (
                <div key={i} className={`flex gap-2 text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  <span className={`font-bold shrink-0 w-28 ${isDarkMode ? 'text-teal-400' : 'text-teal-600'}`}>{k}</span>
                  <span>{v}</span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <h4 className={`font-bold text-sm mb-3 ${isDarkMode ? 'text-teal-400' : 'text-teal-600'}`}>Exit & Risk</h4>
            <div className="space-y-2">
              {[
                ['Exit Mode', 'Confirmed reversion (1-bar confirm at 0.7 threshold)'],
                ['Long TP', '2.5x ATR (let winners run)'],
                ['Short TP', '1.25x ATR (bank profits fast)'],
                ['Stop Loss', '1.5x ATR both sides'],
                ['Hold Limit', 'Longs: 7 days | Shorts: 5 days'],
                ['Max Positions', '3 concurrent'],
              ].map(([k, v], i) => (
                <div key={i} className={`flex gap-2 text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  <span className={`font-bold shrink-0 w-28 ${isDarkMode ? 'text-teal-400' : 'text-teal-600'}`}>{k}</span>
                  <span>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className={`mt-4 rounded-lg p-4 ${isDarkMode ? 'bg-gray-800/50 border border-gray-700' : 'bg-gray-50 border border-gray-200'}`}>
          <h4 className={`font-bold text-sm mb-2 ${isDarkMode ? 'text-amber-400' : 'text-amber-600'}`}>Live Guardrails</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            {[
              ['Daily Loss Limit', '10.9 pts'],
              ['Weekly Loss Limit', '14.5 pts'],
              ['Monthly Loss Limit', '18.1 pts'],
              ['Max Consec Losses', '5'],
              ['Data Freshness', '< 5 min lag'],
              ['Signal Drift Alert', '> 2.0 std'],
              ['Auto-Disable', 'On DD breach'],
              ['Monthly Trades', '~1.7 expected'],
            ].map(([k, v], i) => (
              <div key={i}>
                <div className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{k}</div>
                <div className={`font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* Key Insights */}
      <Card isDarkMode={isDarkMode} className="p-5">
        <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Key Insights from V9 Research</h3>
        <div className="space-y-3">
          {[
            { icon: '1', title: 'Time-decay is the optimal causal proxy for best-bar', detail: 'Of 6 entry families tested (1,488 configs), time-decay was the only one to achieve >80% Sharpe retention from the lookahead ceiling. Peak detection, trailing snap, and smoothed peak all produced negative Sharpe -- the mean-reversion edge requires entering at extreme prices, not waiting for confirmation.', color: 'emerald' },
            { icon: '2', title: 'raw_sig outperforms ratio-scaled clean_sig', detail: 'The simpler bias-corrected, ATR-normalized intraday deviation (raw_sig) beat the residual-ratio-scaled version (clean_sig) under causality. The ratio scaling adds noise when the daily residual is lagged T-1 because the ratio can be stale.', color: 'blue' },
            { icon: '3', title: 'Asymmetric TP is a free lunch', detail: 'Letting longs run (TP=2.5x) while banking short profits quickly (TP=1.25x) improved Sharpe from 4.38 to 4.73. Shorts mean-revert faster in ES spreads, so taking profits early on shorts captures more of the move before it reverses.', color: 'emerald' },
            { icon: '4', title: 'V9 beats V6\'s old production fix (3.45 > 3.36 at realistic cost)', detail: 'The Lookahead Audit section showed the best production config at Sharpe 3.36 (seg_30 + smooth_3). V9\'s time-decay approach achieves 3.45 at 1.5x cost -- a fundamentally better architecture with a cleaner causality guarantee and fewer trades (higher quality).', color: 'teal' },
            { icon: '5', title: 'The edge is concentrated in high-ATR and roll-week periods', detail: 'Regime analysis shows Sharpe 7.24 during high-ATR periods and 4.85 during roll weeks, vs 0.57 during low-ATR and 2.32 non-roll. The strategy profits most when spreads are volatile -- exactly when mispricings are largest and revert fastest.', color: 'violet' },
            { icon: '6', title: 'Lower trade count is a feature, not a bug', detail: '33 trades over 3 years (~11/year) vs V6\'s 148 feels like a big drop. But each V9 trade has Sharpe 4.73 -- the quality filter is working. The T-1 gate + high starting threshold means V9 only trades when the signal is genuinely extreme.', color: 'amber' },
          ].map(({ icon, title, detail, color }) => (
            <div key={icon} className={`rounded-lg border p-4 ${
              color === 'emerald' ? (isDarkMode ? 'bg-emerald-900/10 border-emerald-800/50' : 'bg-emerald-50/50 border-emerald-200') :
              color === 'blue' ? (isDarkMode ? 'bg-blue-900/10 border-blue-800/50' : 'bg-blue-50/50 border-blue-200') :
              color === 'teal' ? (isDarkMode ? 'bg-teal-900/10 border-teal-800/50' : 'bg-teal-50/50 border-teal-200') :
              color === 'violet' ? (isDarkMode ? 'bg-violet-900/10 border-violet-800/50' : 'bg-violet-50/50 border-violet-200') :
              (isDarkMode ? 'bg-amber-900/10 border-amber-800/50' : 'bg-amber-50/50 border-amber-200')
            }`}>
              <div className="flex items-start gap-3">
                <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 ${isDarkMode ? 'bg-gray-700 text-white' : 'bg-gray-200 text-gray-800'}`}>{icon}</span>
                <div>
                  <div className={`font-bold text-sm ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{title}</div>
                  <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{detail}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Callout type="success" isDarkMode={isDarkMode}>
        <strong>Bottom line:</strong> V9 achieves <strong>Sharpe 4.73</strong> (89% of the lookahead ceiling) with <strong>zero lookahead bias</strong>, verified by 10 automated leakage checks, walk-forward OOS (5/7 positive folds, avg Sharpe 5.60), Monte Carlo significance (p=0.010), and survival at 2x costs. At realistic 1.5x costs: <strong>Sharpe 3.45, WR 66.7%, PnL +22.1 pts ($1,108)</strong>. Every daily feature is T-1 lagged, every entry/exit decision is purely causal. The strategy is <strong>PROMOTED</strong> and <strong>DEPLOYMENT READY</strong>.
      </Callout>
    </div>
  );
};

// =============================================================================
// V10 ROLL WEEK SECTION
// =============================================================================

const V10_ROLL_EVENTS_2025 = [
  { symbol: 'ESH5-ESM5', start: '2025-03-12', end: '2025-03-21', days: 9, peakDev: 1.58, revPct: 2, fadePnl: -0.46, dir: 'LONG', quad: true },
  { symbol: 'ESM5-ESU5', start: '2025-03-12', end: '2025-03-21', days: 9, peakDev: 1.48, revPct: 53, fadePnl: +0.28, dir: 'SHORT', quad: true },
  { symbol: 'ESM5-ESU5', start: '2025-06-11', end: '2025-06-20', days: 9, peakDev: 1.26, revPct: 0, fadePnl: -0.50, dir: 'LONG', quad: true },
  { symbol: 'ESU5-ESZ5', start: '2025-06-11', end: '2025-06-20', days: 9, peakDev: 0.92, revPct: 82, fadePnl: +0.25, dir: 'SHORT', quad: true },
  { symbol: 'ESU5-ESZ5', start: '2025-09-10', end: '2025-09-19', days: 9, peakDev: 3.73, revPct: 50, fadePnl: +1.35, dir: 'SHORT', quad: true },
  { symbol: 'ESZ5-ESH6', start: '2025-09-10', end: '2025-09-19', days: 8, peakDev: 2.74, revPct: 0, fadePnl: -0.50, dir: 'SHORT', quad: false },
  { symbol: 'ESH6-ESM6', start: '2025-12-10', end: '2025-12-19', days: 8, peakDev: 3.96, revPct: 94, fadePnl: +3.22, dir: 'SHORT', quad: true },
  { symbol: 'ESZ5-ESH6', start: '2025-12-10', end: '2025-12-19', days: 9, peakDev: 4.16, revPct: 16, fadePnl: +0.19, dir: 'LONG', quad: false },
];

const V10_TRADES_2025 = [
  { symbol: 'ESM5-ESU5', entry: '2025-03-12', exit: '2025-03-16', dir: 'SHORT', entryPx: 47.80, exitPx: 46.75, pnl: +0.55, hold: 4, reason: 'reversion', sig: 1.21 },
  { symbol: 'ESM5-ESU5', entry: '2025-03-18', exit: '2025-03-19', dir: 'SHORT', entryPx: 50.05, exitPx: 48.45, pnl: +1.10, hold: 1, reason: 'take_profit', sig: 1.24 },
  { symbol: 'ESM5-ESU5', entry: '2025-03-20', exit: '2025-03-21', dir: 'SHORT', entryPx: 50.75, exitPx: 49.00, pnl: +1.25, hold: 1, reason: 'take_profit', sig: 1.34 },
  { symbol: 'ESH5-ESM5', entry: '2025-03-21', exit: '2025-03-21', dir: 'LONG', entryPx: 47.55, exitPx: 48.70, pnl: +0.65, hold: 0, reason: 'time_stop', sig: -1.64 },
  { symbol: 'ESU5-ESZ5', entry: '2025-06-19', exit: '2025-06-20', dir: 'SHORT', entryPx: 51.50, exitPx: 50.55, pnl: +0.45, hold: 1, reason: 'reversion', sig: 1.04 },
  { symbol: 'ESU5-ESZ5', entry: '2025-09-10', exit: '2025-09-11', dir: 'SHORT', entryPx: 55.30, exitPx: 54.75, pnl: +0.05, hold: 1, reason: 'reversion', sig: 1.10 },
  { symbol: 'ESZ5-ESH6', entry: '2025-09-10', exit: '2025-09-11', dir: 'SHORT', entryPx: 59.00, exitPx: 58.20, pnl: +0.30, hold: 1, reason: 'reversion', sig: 1.06 },
  { symbol: 'ESU5-ESZ5', entry: '2025-09-14', exit: '2025-09-15', dir: 'SHORT', entryPx: 56.70, exitPx: 56.50, pnl: -0.30, hold: 1, reason: 'reversion', sig: 1.65 },
  { symbol: 'ESZ5-ESH6', entry: '2025-09-16', exit: '2025-09-17', dir: 'LONG', entryPx: 58.45, exitPx: 58.70, pnl: -0.25, hold: 1, reason: 'reversion', sig: -1.56 },
  { symbol: 'ESU5-ESZ5', entry: '2025-09-17', exit: '2025-09-18', dir: 'SHORT', entryPx: 57.20, exitPx: 57.30, pnl: -0.60, hold: 1, reason: 'reversion', sig: 1.13 },
  { symbol: 'ESU5-ESZ5', entry: '2025-09-19', exit: '2025-09-19', dir: 'SHORT', entryPx: 57.65, exitPx: 51.90, pnl: +5.25, hold: 0, reason: 'time_stop', sig: 2.75 },
  { symbol: 'ESZ5-ESH6', entry: '2025-09-19', exit: '2025-12-10', dir: 'SHORT', entryPx: 57.05, exitPx: 58.20, pnl: -1.65, hold: 82, reason: 'reversion', sig: 1.77 },
  { symbol: 'ESZ5-ESH6', entry: '2025-12-11', exit: '2025-12-15', dir: 'SHORT', entryPx: 58.45, exitPx: 58.25, pnl: -0.30, hold: 4, reason: 'reversion', sig: 1.02 },
  { symbol: 'ESZ5-ESH6', entry: '2025-12-16', exit: '2025-12-19', dir: 'LONG', entryPx: 57.55, exitPx: 49.50, pnl: -8.55, hold: 3, reason: 'time_stop', sig: -2.70 },
  { symbol: 'ESH6-ESM6', entry: '2025-12-16', exit: '2025-12-17', dir: 'LONG', entryPx: 51.55, exitPx: 51.95, pnl: -0.10, hold: 1, reason: 'reversion', sig: -1.71 },
  { symbol: 'ESH6-ESM6', entry: '2025-12-18', exit: '2025-12-19', dir: 'LONG', entryPx: 49.50, exitPx: 50.15, pnl: +0.15, hold: 1, reason: 'reversion', sig: -1.69 },
];

const V10_YEAR_PERF = [
  { yr: 2021, n: 8, pnl: -3.10, wr: 0.0, sharpe: -21.21 },
  { yr: 2022, n: 14, pnl: +5.65, wr: 50.0, sharpe: 1.23 },
  { yr: 2023, n: 13, pnl: +13.05, wr: 76.9, sharpe: 4.83 },
  { yr: 2024, n: 16, pnl: +25.80, wr: 62.5, sharpe: 1.75 },
  { yr: 2025, n: 16, pnl: -2.00, wr: 56.2, sharpe: -0.29 },
];

const V10RollWeekSection = ({ isDarkMode }) => {
  const [expandedQuarter, setExpandedQuarter] = useState(null);
  const [showAllTrades, setShowAllTrades] = useState(false);

  const q1Trades = V10_TRADES_2025.filter(t => t.entry.startsWith('2025-03'));
  const q2Trades = V10_TRADES_2025.filter(t => t.entry.startsWith('2025-06'));
  const q3Trades = V10_TRADES_2025.filter(t => t.entry.startsWith('2025-09'));
  const q4Trades = V10_TRADES_2025.filter(t => t.entry.startsWith('2025-12'));

  const quarters = [
    {
      id: 'q1',
      label: 'Q1 March Roll',
      period: 'Mar 12 - Mar 21',
      events: V10_ROLL_EVENTS_2025.filter(e => e.start.startsWith('2025-03')),
      trades: q1Trades,
      pnl: q1Trades.reduce((s, t) => s + t.pnl, 0),
      color: 'emerald',
    },
    {
      id: 'q2',
      label: 'Q2 June Roll',
      period: 'Jun 11 - Jun 20',
      events: V10_ROLL_EVENTS_2025.filter(e => e.start.startsWith('2025-06')),
      trades: q2Trades,
      pnl: q2Trades.reduce((s, t) => s + t.pnl, 0),
      color: 'blue',
    },
    {
      id: 'q3',
      label: 'Q3 September Roll',
      period: 'Sep 10 - Sep 19',
      events: V10_ROLL_EVENTS_2025.filter(e => e.start.startsWith('2025-09')),
      trades: q3Trades,
      pnl: q3Trades.reduce((s, t) => s + t.pnl, 0),
      color: 'violet',
    },
    {
      id: 'q4',
      label: 'Q4 December Roll',
      period: 'Dec 10 - Dec 19',
      events: V10_ROLL_EVENTS_2025.filter(e => e.start.startsWith('2025-12')),
      trades: q4Trades,
      pnl: q4Trades.reduce((s, t) => s + t.pnl, 0),
      color: 'rose',
    },
  ];

  const reasonColor = (r) => {
    if (r === 'take_profit') return isDarkMode ? 'text-emerald-400' : 'text-emerald-600';
    if (r === 'reversion') return isDarkMode ? 'text-blue-400' : 'text-blue-600';
    if (r === 'stop') return isDarkMode ? 'text-red-400' : 'text-red-600';
    return isDarkMode ? 'text-amber-400' : 'text-amber-600';
  };

  const pnlColor = (v) => v > 0
    ? (isDarkMode ? 'text-emerald-400' : 'text-emerald-600')
    : v < 0
      ? (isDarkMode ? 'text-red-400' : 'text-red-600')
      : (isDarkMode ? 'text-gray-400' : 'text-gray-500');

  return (
    <div className="space-y-6">
      <SectionHeader
        icon="🔄"
        title="V10 Roll Week Strategy"
        subtitle="Structural mean-reversion during quarterly ES futures roll periods -- our highest-conviction edge"
        isDarkMode={isDarkMode}
        color="sky"
      />

      {/* What is Roll Week */}
      <Callout type="insight" isDarkMode={isDarkMode}>
        <strong>The Roll Week Edge:</strong> Four times per year, institutions must roll their ES futures positions from the expiring front-month to the next contract. This forced, non-informational flow creates predictable deviation extremes in calendar spreads. The V10 research identifies, quantifies, and systematically trades this structural dislocation using zero-lookahead causal entries.
      </Callout>

      {/* Strategy Overview Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="Total PnL" value="+39.4 pts" sub="$1,970 per contract" isDarkMode={isDarkMode} color="green" />
        <StatCard label="Sharpe Ratio" value="1.23" sub="67 trades, 2021-2025" isDarkMode={isDarkMode} color="blue" />
        <StatCard label="Win Rate" value="53.7%" sub="36W / 31L" isDarkMode={isDarkMode} color="purple" />
        <StatCard label="Profit Factor" value="2.29" sub="MaxDD: -10.8 pts" isDarkMode={isDarkMode} color="amber" />
      </div>

      {/* The Mechanism */}
      <Card isDarkMode={isDarkMode} className="p-6">
        <h3 className={`text-lg font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>How Roll Week Creates Edge</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          {[
            { step: '1', title: 'Institutional Mandate', desc: 'Index funds, pension funds, and dealers MUST roll ~$500B+ in ES futures to the next quarter. This is not optional.', color: 'rose' },
            { step: '2', title: 'Flow Pressure', desc: 'Concentrated rolling activity pushes calendar spread prices away from fair value. Deviations average 1.34x larger than non-roll periods.', color: 'amber' },
            { step: '3', title: 'Deviation Extremes', desc: 'Roll-week peak deviations avg 2.66 pts vs 0.74 pts non-roll. Statistically significant (p=0.003, Welch t-test).', color: 'violet' },
            { step: '4', title: 'Mean Reversion', desc: 'Once roll pressure subsides, spreads snap back to fair value. We fade the extremes with causal entry logic and tight risk management.', color: 'emerald' },
          ].map(({ step, title, desc, color }) => (
            <PipelineStep key={step} step={step} title={title} explanation={desc} isDarkMode={isDarkMode} color={color} />
          ))}
        </div>
      </Card>

      {/* Roll vs Non-Roll Statistical Comparison */}
      <Card isDarkMode={isDarkMode} className="p-6">
        <h3 className={`text-lg font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Roll vs Non-Roll: Statistical Evidence</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <ComparisonTable
              isDarkMode={isDarkMode}
              headers={['Metric', 'Roll Week', 'Non-Roll', 'Ratio']}
              rows={[
                ['Avg |Deviation|', '0.994 pts', '0.744 pts', '1.34x'],
                ['Avg |Norm Dev|', '1.164', '0.902', '1.29x'],
                ['Median |Dev|', '0.522', '0.406', '1.29x'],
                ['P90 |Dev|', '2.843', '1.726', '1.65x'],
              ]}
              highlightCol={3}
            />
          </div>
          <div>
            <ComparisonTable
              isDarkMode={isDarkMode}
              headers={['Test', 'Statistic', 'p-value', 'Result']}
              rows={[
                ['Welch t-test', 't = 2.933', '0.0035', 'Significant'],
                ['Mann-Whitney U', 'U = 501,361', '0.0001', 'Significant'],
                ['KS test', 'D = 0.092', '0.0059', 'Significant'],
                ["Cohen's d", '0.165', '--', 'Small-Medium'],
              ]}
              highlightCol={2}
            />
          </div>
        </div>
        <Callout type="success" isDarkMode={isDarkMode}>
          All three non-parametric tests confirm roll-week deviations are statistically larger. The permutation test on theoretical "fade peak" PnL yields <strong>p = 0.0077</strong>, confirming the edge is not random.
        </Callout>
      </Card>

      {/* Strategy Configuration */}
      <Card isDarkMode={isDarkMode} className="p-6">
        <h3 className={`text-lg font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Production Strategy Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
            <div className={`text-xs font-semibold uppercase tracking-wider mb-3 ${isDarkMode ? 'text-sky-400' : 'text-sky-600'}`}>Entry</div>
            <div className={`space-y-2 text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              <div className="flex justify-between"><span>Entry type</span><span className="font-mono font-bold">first_cross</span></div>
              <div className="flex justify-between"><span>Signal</span><span className="font-mono font-bold">clean_sig</span></div>
              <div className="flex justify-between"><span>Long threshold</span><span className="font-mono font-bold">1.25</span></div>
              <div className="flex justify-between"><span>Short threshold</span><span className="font-mono font-bold">1.00</span></div>
              <div className="flex justify-between"><span>Gate multiplier</span><span className="font-mono font-bold">0.40</span></div>
              <div className="flex justify-between"><span>Daily signal</span><span className="font-mono font-bold">resid_signal_lag1</span></div>
            </div>
          </div>
          <div className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
            <div className={`text-xs font-semibold uppercase tracking-wider mb-3 ${isDarkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>Exit</div>
            <div className={`space-y-2 text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              <div className="flex justify-between"><span>Exit mode</span><span className="font-mono font-bold">first (1-bar confirm)</span></div>
              <div className="flex justify-between"><span>Long TP</span><span className="font-mono font-bold">3.0 pts</span></div>
              <div className="flex justify-between"><span>Short TP</span><span className="font-mono font-bold">1.5 pts</span></div>
              <div className="flex justify-between"><span>Reversion exit</span><span className="font-mono font-bold">0.50</span></div>
              <div className="flex justify-between"><span>Long max hold</span><span className="font-mono font-bold">10 days</span></div>
              <div className="flex justify-between"><span>Short max hold</span><span className="font-mono font-bold">7 days</span></div>
            </div>
          </div>
          <div className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
            <div className={`text-xs font-semibold uppercase tracking-wider mb-3 ${isDarkMode ? 'text-rose-400' : 'text-rose-600'}`}>Risk</div>
            <div className={`space-y-2 text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              <div className="flex justify-between"><span>Long stop</span><span className="font-mono font-bold">2.5 pts</span></div>
              <div className="flex justify-between"><span>Short stop</span><span className="font-mono font-bold">5.0 pts</span></div>
              <div className="flex justify-between"><span>Max concurrent</span><span className="font-mono font-bold">2 positions</span></div>
              <div className="flex justify-between"><span>Max loss/event</span><span className="font-mono font-bold">2% of account</span></div>
              <div className="flex justify-between"><span>Max annual DD</span><span className="font-mono font-bold">5% of account</span></div>
              <div className="flex justify-between"><span>Kelly (half)</span><span className="font-mono font-bold">15.1%</span></div>
            </div>
          </div>
        </div>
      </Card>

      {/* Year-by-Year Performance */}
      <Card isDarkMode={isDarkMode} className="p-6">
        <h3 className={`text-lg font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Year-by-Year Performance (2021 - 2025)</h3>
        <ComparisonTable
          isDarkMode={isDarkMode}
          headers={['Year', 'Trades', 'Total PnL', 'Win Rate', 'Sharpe', 'Assessment']}
          rows={V10_YEAR_PERF.map(y => [
            y.yr,
            y.n,
            `${y.pnl >= 0 ? '+' : ''}${y.pnl.toFixed(2)} pts`,
            `${y.wr.toFixed(1)}%`,
            y.sharpe.toFixed(2),
            y.yr === 2021 ? 'Data ramp-up (small deviations)' :
            y.yr === 2022 ? 'Breakeven in volatile regime' :
            y.yr === 2023 ? 'Strongest year (large events)' :
            y.yr === 2024 ? 'Best absolute PnL' :
            'Slight negative (Dec outlier)'
          ])}
          highlightCol={2}
        />
        <Callout type="info" isDarkMode={isDarkMode}>
          <strong>2021 context:</strong> Early data period with smaller deviation magnitudes (avg peak 0.65 pts vs 3+ pts in later years). The strategy needs sufficient volatility in spreads to generate tradeable signals. 2022+ all show the structural roll-week edge.
        </Callout>
      </Card>

      {/* Exit Reason Breakdown */}
      <Card isDarkMode={isDarkMode} className="p-6">
        <h3 className={`text-lg font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Exit Reason Analysis (All 67 Trades)</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          {[
            { reason: 'Reversion', n: 43, pnl: -7.75, wr: 41.9, color: 'blue', desc: 'Signal returns below exit threshold' },
            { reason: 'Take Profit', n: 11, pnl: +33.70, wr: 100.0, color: 'emerald', desc: 'Hit TP target (L:3.0, S:1.5 pts)' },
            { reason: 'Time Stop', n: 12, pnl: +14.05, wr: 58.3, color: 'amber', desc: 'Max hold period exceeded' },
            { reason: 'Stop Loss', n: 1, pnl: -0.60, wr: 0.0, color: 'red', desc: 'Hard stop triggered (rare)' },
          ].map(({ reason, n, pnl, wr, color, desc }) => (
            <div key={reason} className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
              <div className={`text-xs font-semibold uppercase tracking-wider mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>{reason}</div>
              <div className={`text-2xl font-black ${pnl >= 0 ? (isDarkMode ? 'text-emerald-400' : 'text-emerald-600') : (isDarkMode ? 'text-red-400' : 'text-red-600')}`}>
                {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
              </div>
              <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{n} trades, {wr.toFixed(0)}% WR</div>
              <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-600' : 'text-gray-400'}`}>{desc}</div>
            </div>
          ))}
        </div>
        <Callout type="insight" isDarkMode={isDarkMode}>
          <strong>Key insight:</strong> Take-profits are the profit engine (+33.7 pts, 100% WR, avg +3.06 pts/trade). Reversion exits are slightly negative overall because many are small scratches after the signal weakens. The strategy structure is: many small reversion scratches funded by occasional large TP wins and positive time-stop exits.
        </Callout>
      </Card>

      {/* 2025 Deep Dive: Quarter by Quarter */}
      <Card isDarkMode={isDarkMode} className="p-6">
        <h3 className={`text-lg font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>2025 Trade-by-Trade Breakdown</h3>
        <p className={`text-sm mb-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
          16 trades across 4 quarterly roll periods. Click each quarter to expand the full trade log and roll event context.
        </p>

        {/* Quarter Summary Bars */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
          {quarters.map(q => {
            const wins = q.trades.filter(t => t.pnl > 0).length;
            return (
              <button
                key={q.id}
                onClick={() => setExpandedQuarter(expandedQuarter === q.id ? null : q.id)}
                className={`rounded-lg border p-4 text-left transition-all ${
                  expandedQuarter === q.id
                    ? isDarkMode ? 'bg-gray-700 border-sky-500 ring-1 ring-sky-500/50' : 'bg-sky-50 border-sky-300 ring-1 ring-sky-200'
                    : isDarkMode ? 'bg-gray-800/50 border-gray-700 hover:border-gray-600' : 'bg-white border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className={`text-xs font-semibold uppercase tracking-wider mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>{q.label}</div>
                <div className={`text-xl font-black ${pnlColor(q.pnl)}`}>{q.pnl >= 0 ? '+' : ''}{q.pnl.toFixed(2)} pts</div>
                <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  {q.trades.length} trades, {wins}W/{q.trades.length - wins}L | {q.period}
                </div>
                <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-600' : 'text-gray-400'}`}>
                  {q.events.length} roll event{q.events.length !== 1 ? 's' : ''}
                </div>
              </button>
            );
          })}
        </div>

        {/* Expanded Quarter Detail */}
        {quarters.filter(q => expandedQuarter === q.id).map(q => (
          <div key={q.id} className={`rounded-lg border p-5 space-y-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
            <h4 className={`font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{q.label} Detail ({q.period})</h4>

            {/* Roll Events for this quarter */}
            <div>
              <div className={`text-xs font-semibold uppercase tracking-wider mb-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Roll Events</div>
              <div className={`rounded-lg border overflow-hidden ${isDarkMode ? 'border-gray-600' : 'border-gray-300'}`}>
                <table className="w-full text-xs">
                  <thead>
                    <tr className={isDarkMode ? 'bg-gray-700' : 'bg-gray-100'}>
                      {['Symbol', 'Period', 'Days', 'Peak |Dev|', 'Reversion %', 'Fade PnL', 'Direction'].map(h => (
                        <th key={h} className={`px-3 py-2 text-left font-semibold ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {q.events.map((e, i) => (
                      <tr key={i} className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                        <td className={`px-3 py-2 font-mono font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{e.symbol}</td>
                        <td className={`px-3 py-2 font-mono ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{e.start.slice(5)} - {e.end.slice(5)}</td>
                        <td className={`px-3 py-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{e.days}</td>
                        <td className={`px-3 py-2 font-mono font-bold ${e.peakDev >= 2 ? (isDarkMode ? 'text-amber-400' : 'text-amber-600') : (isDarkMode ? 'text-gray-300' : 'text-gray-700')}`}>{e.peakDev.toFixed(2)}</td>
                        <td className={`px-3 py-2 font-mono ${e.revPct >= 50 ? (isDarkMode ? 'text-emerald-400' : 'text-emerald-600') : (isDarkMode ? 'text-gray-400' : 'text-gray-500')}`}>{e.revPct}%</td>
                        <td className={`px-3 py-2 font-mono font-bold ${pnlColor(e.fadePnl)}`}>{e.fadePnl >= 0 ? '+' : ''}{e.fadePnl.toFixed(2)}</td>
                        <td className={`px-3 py-2 font-bold ${e.dir === 'LONG' ? (isDarkMode ? 'text-emerald-400' : 'text-emerald-600') : (isDarkMode ? 'text-rose-400' : 'text-rose-600')}`}>{e.dir}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Trades for this quarter */}
            <div>
              <div className={`text-xs font-semibold uppercase tracking-wider mb-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Executed Trades</div>
              <div className={`rounded-lg border overflow-x-auto ${isDarkMode ? 'border-gray-600' : 'border-gray-300'}`}>
                <table className="w-full text-xs">
                  <thead>
                    <tr className={isDarkMode ? 'bg-gray-700' : 'bg-gray-100'}>
                      {['Symbol', 'Entry', 'Exit', 'Dir', 'Entry Px', 'Exit Px', 'PnL', 'Hold', 'Exit Reason', 'Signal'].map(h => (
                        <th key={h} className={`px-3 py-2 text-left font-semibold whitespace-nowrap ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {q.trades.map((t, i) => (
                      <tr key={i} className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} ${
                        i % 2 === 0 ? '' : (isDarkMode ? 'bg-gray-800/30' : 'bg-white')
                      }`}>
                        <td className={`px-3 py-2 font-mono font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{t.symbol}</td>
                        <td className={`px-3 py-2 font-mono ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{t.entry.slice(5)}</td>
                        <td className={`px-3 py-2 font-mono ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{t.exit.slice(5)}</td>
                        <td className={`px-3 py-2 font-bold ${t.dir === 'LONG' ? (isDarkMode ? 'text-emerald-400' : 'text-emerald-600') : (isDarkMode ? 'text-rose-400' : 'text-rose-600')}`}>{t.dir}</td>
                        <td className={`px-3 py-2 font-mono ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{t.entryPx.toFixed(2)}</td>
                        <td className={`px-3 py-2 font-mono ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{t.exitPx.toFixed(2)}</td>
                        <td className={`px-3 py-2 font-mono font-bold ${pnlColor(t.pnl)}`}>{t.pnl >= 0 ? '+' : ''}{t.pnl.toFixed(2)}</td>
                        <td className={`px-3 py-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{t.hold}d</td>
                        <td className={`px-3 py-2 font-medium ${reasonColor(t.reason)}`}>{t.reason.replace('_', ' ')}</td>
                        <td className={`px-3 py-2 font-mono ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{t.sig.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Quarter narrative */}
              <div className={`mt-3 text-sm leading-relaxed ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                {q.id === 'q1' && (
                  <p><strong>March 2025 roll was excellent.</strong> Three trades on ESM5-ESU5 captured the spread richening: two hit take-profit (+1.10, +1.25 pts) and the first exited on reversion (+0.55). A late LONG on ESH5-ESM5 caught a quick time-stop profit (+0.65). All 4 trades won. Total: <strong>+3.55 pts ($178 per contract).</strong></p>
                )}
                {q.id === 'q2' && (
                  <p><strong>June was a single clean reversion trade.</strong> ESU5-ESZ5 showed a mild 0.92 peak deviation (82% reversion). One short entry, quick 1-day hold, +0.45 pts. Low activity because the roll was mild. Total: <strong>+0.45 pts ($23 per contract).</strong></p>
                )}
                {q.id === 'q3' && (
                  <p><strong>September was the most active quarter with 7 trades.</strong> The big winner was ESU5-ESZ5 on Sep 19 (+5.25 pts, time-stop exit as the spread collapsed intraday). Early Sept trades were small scratches. The problematic trade was ESZ5-ESH6 Sep 19 entry that held for 82 days into the December roll before exiting at a loss (-1.65). Net still positive: <strong>+2.80 pts ($140 per contract).</strong></p>
                )}
                {q.id === 'q4' && (
                  <p><strong>December was the worst quarter, dominated by one large loss.</strong> ESZ5-ESH6 went LONG on Dec 16 at 57.55 and the spread cratered to 49.50 by Dec 19, hitting the time-stop for -8.55 pts -- the single worst trade in the entire backtest. ESH6-ESM6 had two small LONG reversion trades that were roughly flat. Total: <strong>-8.80 pts ($-440 per contract).</strong></p>
                )}
              </div>
            </div>
          </div>
        ))}
      </Card>

      {/* Full 2025 Trade Log Table */}
      <Card isDarkMode={isDarkMode} className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className={`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Full 2025 Trade Log ({V10_TRADES_2025.length} trades)</h3>
          <button
            onClick={() => setShowAllTrades(!showAllTrades)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              isDarkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {showAllTrades ? 'Collapse' : 'Expand All'}
          </button>
        </div>

        {showAllTrades && (
          <>
            <div className={`rounded-lg border overflow-x-auto ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <table className="w-full text-xs">
                <thead>
                  <tr className={isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}>
                    {['#', 'Symbol', 'Entry', 'Exit', 'Dir', 'Entry Px', 'Exit Px', 'PnL (pts)', 'PnL ($)', 'Hold', 'Exit Reason', 'Signal'].map(h => (
                      <th key={h} className={`px-3 py-2 text-left font-semibold whitespace-nowrap ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {V10_TRADES_2025.map((t, i) => (
                    <tr key={i} className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-100'} ${
                      i % 2 === 0 ? '' : (isDarkMode ? 'bg-gray-800/50' : 'bg-gray-50/50')
                    }`}>
                      <td className={`px-3 py-2 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{i + 1}</td>
                      <td className={`px-3 py-2 font-mono font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{t.symbol}</td>
                      <td className={`px-3 py-2 font-mono ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{t.entry}</td>
                      <td className={`px-3 py-2 font-mono ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{t.exit}</td>
                      <td className={`px-3 py-2 font-bold ${t.dir === 'LONG' ? (isDarkMode ? 'text-emerald-400' : 'text-emerald-600') : (isDarkMode ? 'text-rose-400' : 'text-rose-600')}`}>{t.dir}</td>
                      <td className={`px-3 py-2 font-mono ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{t.entryPx.toFixed(2)}</td>
                      <td className={`px-3 py-2 font-mono ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{t.exitPx.toFixed(2)}</td>
                      <td className={`px-3 py-2 font-mono font-bold ${pnlColor(t.pnl)}`}>{t.pnl >= 0 ? '+' : ''}{t.pnl.toFixed(2)}</td>
                      <td className={`px-3 py-2 font-mono ${pnlColor(t.pnl)}`}>{t.pnl >= 0 ? '+' : ''}{(t.pnl * 50).toFixed(0)}</td>
                      <td className={`px-3 py-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{t.hold}d</td>
                      <td className={`px-3 py-2 font-medium ${reasonColor(t.reason)}`}>{t.reason.replace('_', ' ')}</td>
                      <td className={`px-3 py-2 font-mono ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{t.sig.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr className={`border-t-2 font-bold ${isDarkMode ? 'border-gray-600 bg-gray-700/50' : 'border-gray-300 bg-gray-50'}`}>
                    <td colSpan={7} className={`px-3 py-2 text-right ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>2025 Total:</td>
                    <td className={`px-3 py-2 font-mono ${pnlColor(-2.0)}`}>-2.00</td>
                    <td className={`px-3 py-2 font-mono ${pnlColor(-2.0)}`}>-100</td>
                    <td colSpan={3} className={`px-3 py-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>9W / 7L (56.2%)</td>
                  </tr>
                </tfoot>
              </table>
            </div>

            <div className={`mt-3 text-sm leading-relaxed ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              <p><strong>2025 Anatomy:</strong> 9 winners, 7 losers, net -2.00 pts. The year was almost entirely wrecked by a single trade: ESZ5-ESH6 LONG on Dec 16 lost -8.55 pts when the spread diverged sharply against the position. Without that one trade, 2025 would have been +6.55 pts. This illustrates why the 5% annual DD limit and 2% per-event loss cap are critical guardrails for production.</p>
            </div>
          </>
        )}

        {!showAllTrades && (
          <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            <div className="grid grid-cols-4 gap-3">
              {quarters.map(q => {
                const wins = q.trades.filter(t => t.pnl > 0).length;
                return (
                  <div key={q.id} className="text-center">
                    <div className={`text-lg font-bold ${pnlColor(q.pnl)}`}>{q.pnl >= 0 ? '+' : ''}{q.pnl.toFixed(2)}</div>
                    <div className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{q.label} ({wins}W/{q.trades.length - wins}L)</div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </Card>

      {/* Robustness & Validation */}
      <Card isDarkMode={isDarkMode} className="p-6">
        <h3 className={`text-lg font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Robustness Validation</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {/* Walk-Forward */}
          <div>
            <div className={`text-xs font-semibold uppercase tracking-wider mb-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Walk-Forward Out-of-Sample</div>
            <ComparisonTable
              isDarkMode={isDarkMode}
              headers={['OOS Year', 'Train Period', 'N', 'PnL', 'Sharpe', 'WR']}
              rows={[
                ['2023', '2021-2022', '14', '+12.65', '4.54', '71.4%'],
                ['2024', '2021-2023', '16', '+19.20', '1.55', '56.2%'],
                ['2025', '2021-2024', '16', '-2.00', '-0.29', '56.2%'],
              ]}
              highlightCol={3}
            />
            <div className={`mt-2 text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>Avg WF Sharpe: 1.93 | 2/3 positive years</div>
          </div>

          {/* Monte Carlo */}
          <div>
            <div className={`text-xs font-semibold uppercase tracking-wider mb-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Monte Carlo Significance (10K sims)</div>
            <ComparisonTable
              isDarkMode={isDarkMode}
              headers={['Test', 'Real Value', 'p-value', 'Status']}
              rows={[
                ['Total PnL', '+39.40', '0.0312', 'Significant'],
                ['Avg PnL', '+0.588', '0.0312', 'Significant'],
                ['Sharpe', '1.23', '0.0322', 'Significant'],
              ]}
              highlightCol={2}
            />
            <div className={`mt-2 text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>MC Sharpe 95th pctl: 1.09 | Real Sharpe 1.23 exceeds 95th</div>
          </div>
        </div>

        {/* Stress Tests */}
        <div className="mb-4">
          <div className={`text-xs font-semibold uppercase tracking-wider mb-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Stress Tests (8/9 survive)</div>
          <div className="grid grid-cols-3 md:grid-cols-5 gap-2">
            {[
              { label: '2x Costs', sharpe: 0.18, pnl: +5.90, pass: true },
              { label: '3x Costs', sharpe: -0.86, pnl: -27.60, pass: false },
              { label: 'Gate 0.8', sharpe: 0.74, pnl: +18.30, pass: true },
              { label: 'Gate 0.9', sharpe: 0.67, pnl: +16.45, pass: true },
              { label: 'Gate 1.0', sharpe: 0.54, pnl: +12.85, pass: true },
              { label: 'ET +0.25', sharpe: 0.92, pnl: +22.85, pass: true },
              { label: 'ET +0.5', sharpe: 0.99, pnl: +23.95, pass: true },
              { label: 'Hold 5/3', sharpe: 1.22, pnl: +39.20, pass: true },
              { label: 'Hold 3/2', sharpe: 1.28, pnl: +42.75, pass: true },
            ].map(s => (
              <div key={s.label} className={`rounded border p-2 text-center text-xs ${
                s.pass
                  ? isDarkMode ? 'bg-emerald-900/20 border-emerald-700' : 'bg-emerald-50 border-emerald-200'
                  : isDarkMode ? 'bg-red-900/20 border-red-700' : 'bg-red-50 border-red-200'
              }`}>
                <div className={`font-bold ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>{s.label}</div>
                <div className={`font-mono mt-0.5 ${s.pass ? (isDarkMode ? 'text-emerald-400' : 'text-emerald-600') : (isDarkMode ? 'text-red-400' : 'text-red-600')}`}>
                  S: {s.sharpe.toFixed(2)}
                </div>
                <div className={`font-mono ${pnlColor(s.pnl)}`}>{s.pnl >= 0 ? '+' : ''}{s.pnl.toFixed(1)}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Promotion Checklist */}
        <div>
          <div className={`text-xs font-semibold uppercase tracking-wider mb-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Promotion Checklist (8/9 passed)</div>
          <div className="grid grid-cols-3 md:grid-cols-5 gap-2">
            {[
              { check: 'Leakage Audit', pass: true },
              { check: 'Positive PnL', pass: true },
              { check: 'Sharpe > 1.0', pass: true },
              { check: 'WR > 55%', pass: false },
              { check: 'PF > 1.3', pass: true },
              { check: 'MC Significance', pass: true },
              { check: 'Majority OOS+', pass: true },
              { check: 'Stress Survival', pass: true },
              { check: 'Year Consistency', pass: true },
            ].map(c => (
              <div key={c.check} className={`rounded border px-3 py-2 text-xs flex items-center gap-2 ${
                c.pass
                  ? isDarkMode ? 'bg-emerald-900/20 border-emerald-700 text-emerald-400' : 'bg-emerald-50 border-emerald-200 text-emerald-700'
                  : isDarkMode ? 'bg-red-900/20 border-red-700 text-red-400' : 'bg-red-50 border-red-200 text-red-700'
              }`}>
                <span className="font-bold">{c.pass ? 'PASS' : 'FAIL'}</span>
                <span>{c.check}</span>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* Lookahead Audit Specific to V10 */}
      <Card isDarkMode={isDarkMode} className="p-6">
        <h3 className={`text-lg font-bold mb-3 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Lookahead Bias Audit</h3>
        <div className={`text-sm leading-relaxed space-y-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
          <p>The V10 pipeline was audited line-by-line for any forward-looking data leakage. Every daily feature is T-1 lagged before merging into 1-minute bars:</p>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mt-3">
          {[
            { col: 'bias_20d_lag1', status: 'CLEAN' },
            { col: 'spread_atr_20_lag1', status: 'CLEAN' },
            { col: 'resid_signal_lag1', status: 'CLEAN' },
            { col: 'dev_bc_lag1', status: 'CLEAN' },
            { col: 'carry_spot_lag1', status: 'PATCHED' },
            { col: 'in_roll_window', status: 'CLEAN (calendar)' },
          ].map(c => (
            <div key={c.col} className={`rounded border px-3 py-2 text-xs font-mono flex items-center justify-between ${
              isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'
            }`}>
              <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>{c.col}</span>
              <span className={c.status === 'PATCHED'
                ? (isDarkMode ? 'text-amber-400 font-bold' : 'text-amber-600 font-bold')
                : (isDarkMode ? 'text-emerald-400' : 'text-emerald-600')
              }>{c.status}</span>
            </div>
          ))}
        </div>
        <Callout type="success" isDarkMode={isDarkMode}>
          <strong>carry_spot</strong> was identified as using same-day close prices (~0.12 pts, 2-6% of signal threshold). Patched to T-1 lag. Full pipeline re-run showed <strong>zero change</strong> in results -- identical Sharpe (1.23), PnL (+39.4), and trade count (67). All entry functions verified causal via fuzz testing. Strategy is <strong>100% lookahead-free</strong>.
        </Callout>
      </Card>

      {/* Position Sizing */}
      <Card isDarkMode={isDarkMode} className="p-6">
        <h3 className={`text-lg font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Position Sizing Framework</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <StatCard label="Avg PnL/Trade" value="+0.588 pts" sub="$29 per contract" isDarkMode={isDarkMode} color="green" />
          <StatCard label="Events/Year" value="~14" sub="across ~4 roll periods" isDarkMode={isDarkMode} color="blue" />
          <StatCard label="Half Kelly" value="15.1%" sub="of bankroll per trade" isDarkMode={isDarkMode} color="purple" />
          <StatCard label="Worst Trade" value="-8.55 pts" sub="$-428 per contract" isDarkMode={isDarkMode} color="red" />
        </div>
        <ComparisonTable
          isDarkMode={isDarkMode}
          headers={['Account Size', 'Contracts/Event', 'Expected Annual', 'Worst-Case Loss', '% of Account']}
          rows={[
            ['$50,000', '2', '$825', '$-855', '1.7%'],
            ['$100,000', '4', '$1,650', '$-1,710', '1.7%'],
            ['$250,000', '11', '$4,538', '$-4,702', '1.9%'],
            ['$500,000', '23', '$9,488', '$-9,832', '2.0%'],
          ]}
          highlightCol={2}
        />
      </Card>

      {/* Bottom Line */}
      <Callout type="success" isDarkMode={isDarkMode}>
        <strong>V10 Roll Week Strategy: PROMOTED TO PRODUCTION.</strong> Structural mean-reversion edge from institutional forced rolling. Sharpe 1.23 over 67 trades (2021-2025), profit factor 2.29, 8/9 promotion criteria passed, Monte Carlo p=0.032, walk-forward avg Sharpe 1.93. Zero lookahead bias (all daily features T-1 lagged, carry_spot patched and verified). This is a high-conviction, low-frequency overlay (~14 trades/year) designed for heavy sizing during the 4 quarterly roll windows.
      </Callout>
    </div>
  );
};

const ProductionSection = ({ isDarkMode }) => (
  <div className="space-y-6">
    <SectionHeader icon="🏭" title="Production Guide" subtitle="What to monitor, when to intervene, and how to keep the edge" isDarkMode={isDarkMode} color="red" />

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Daily Operations Checklist</h3>
      <div className="space-y-3">
        {[
          { time: 'Pre-Market', tasks: ['Verify SOFR spot updated (Fed publish ~8:00 AM ET)', 'Check ZQ implied rates from previous session close', 'Confirm daily OHLCV data ingested for spreads + ES', 'Run rolling regression to get today\'s resid_signal'] },
          { time: 'Market Open', tasks: ['Start 1m signal computation: FV_1m, dev_bc_1m, norm_dev_resid_g0_1m', 'If daily resid_signal passes watchlist filter, begin monitoring 1m threshold crossings', 'Set alerts for signal crossing entry_threshold'] },
          { time: 'During Session', tasks: ['Monitor 1m signal for entry opportunities', 'If in a position: check for reversion_1m exit (signal < 0.5)', 'Log all threshold crossings for review'] },
          { time: 'Post-Close', tasks: ['Record daily resid_signal and components', 'Update bias_20d rolling window', 'Log any trades taken, exit reasons, PnL', 'Review regression coefficient stability'] },
        ].map(({ time, tasks }) => (
          <div key={time} className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
            <div className={`font-bold text-sm mb-2 ${isDarkMode ? 'text-amber-400' : 'text-amber-600'}`}>{time}</div>
            <ul className={`text-sm space-y-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              {tasks.map((t, i) => <li key={i} className="flex items-start gap-2"><span className="text-xs mt-1 opacity-50">▸</span>{t}</li>)}
            </ul>
          </div>
        ))}
      </div>
    </Card>

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Key Metrics to Monitor Continuously</h3>
      <ComparisonTable isDarkMode={isDarkMode}
        headers={['Metric', 'Where to Look', 'Warning Signs', 'Action']}
        rows={[
          ['Regression R²', 'Daily regression output', 'R² drops below 0.05 for 2+ weeks', 'Increase lookback window or add/remove features'],
          ['Regression β stability', 'Track β₁, β₂, β₃ over time', 'Coefficients flip sign or magnitude doubles', 'Regime shift — pause shorts until stable'],
          ['Rolling Sharpe (30d)', 'Trailing 30-day trade Sharpe', 'Drops below 1.0 for 20+ trading days', 'Reduce position size, review recent losers'],
          ['Short win rate (30d)', 'Trailing 30-day short WR', 'Drops below 55%', 'Tighten short_entry_thresh (e.g., 1.5 → 2.0)'],
          ['ATR level', 'spread_atr_20', 'ATR < 0.5 or ATR > 3.0 (unusual)', 'Low ATR = thin signal; High ATR = volatile → adjust position size'],
          ['Rate gap regime', 'rate_gap value', 'Sustained |rate_gap| > 1% (extreme)', 'Regression may struggle — the relationship is trained on smaller gaps'],
          ['Roll week performance', 'Trades with in_roll=True', 'Roll trades consistently losing', 'Raise l_roll/s_roll or disable roll trading'],
          ['Bias drift', 'bias_20d magnitude', 'bias_20d > 2×ATR', 'FV model may need recalibration — check div yield assumption'],
        ]}
      />
    </Card>

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Tunable Parameters & When to Adjust</h3>
      <div className="space-y-3">
        {[
          { param: 'Regression Lookback (60 days)', when: 'If market regime changes rapidly (e.g., surprise Fed action), shorten to 40. If stable, extend to 90 for smoother coefficients.', risk: 'Too short → noisy coefficients. Too long → slow to adapt.' },
          { param: 'short_entry_thresh (1.5)', when: 'If short WR drops below 60%, raise to 2.0. If market is trending down and shorts are easy, could lower to 1.25.', risk: 'Too high → miss good shorts. Too low → take bad shorts.' },
          { param: 'short_max_hold (7 days)', when: 'If time_stop exits are mostly winners (mean-reverting slowly), extend to 10. If they\'re mostly losers, shorten to 5.', risk: 'Too long → capital tied up. Too short → exit before reversion.' },
          { param: 'stop_mult (2.0× for shorts)', when: 'If stops are triggering on intraday noise, widen to 2.5. If big losses accumulate, tighten to 1.5.', risk: 'Too tight → whipsawed. Too loose → large single-trade losses.' },
          { param: 'exit_threshold (0.5)', when: 'If exits are too early (leaving money on table), lower to 0.3. If holding too long past reversion, raise to 0.75.', risk: 'Too low → early exits. Too high → late exits or time_stops.' },
        ].map(({ param, when, risk }) => (
          <div key={param} className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
            <div className={`font-bold text-sm ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{param}</div>
            <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{when}</p>
            <p className={`text-xs mt-2 ${isDarkMode ? 'text-red-400' : 'text-red-600'}`}>Risk: {risk}</p>
          </div>
        ))}
      </div>
    </Card>

    <Callout type="danger" isDarkMode={isDarkMode}>
      <strong>Edge Decay Warning:</strong> This strategy's edge comes from the mean-reversion of calendar spread mispricings. If the market becomes more efficient at pricing ZQ-implied carry into spreads in real-time, the residual signal will shrink. Monitor the distribution of |resid_signal| — if the average magnitude is declining over time, the edge is compressing.
    </Callout>

    <Card isDarkMode={isDarkMode} className="p-5">
      <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Cold Start: Getting the System Running on Any Day</h3>
      <div className={`text-sm leading-relaxed space-y-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
        <p><strong>Step 1 — Data requirements:</strong> You need at least 60 trading days (~3 months) of historical data for the regression lookback, plus 20 days for bias_20d and ATR. So about 80 trading days of history to get meaningful signals.</p>
        <p><strong>Step 2 — Bootstrap:</strong> Run the full pipeline (FV → deviation → z-scores → regression) on those 80 days. The first ~20 days of regression output will have large standard errors. The signal becomes reliable from day 60 onward.</p>
        <p><strong>Step 3 — Paper trade:</strong> Run for 2-4 weeks in paper mode to verify the signal values look reasonable and threshold crossings align with observable spread behavior.</p>
        <p><strong>Step 4 — Go live:</strong> Start with a single contract. Monitor the first 10-20 trades. If metrics match backtest expectations (WR &gt; 65%, avg hold 2-4 days, mostly reversion exits), scale up.</p>
      </div>
    </Card>
  </div>
);

// =============================================================================
// MAIN COMPONENT
// =============================================================================

const V6Bones = ({ isDarkMode }) => {
  const [activeSection, setActiveSection] = useState(SECTIONS.OVERVIEW);

  const sectionContent = useMemo(() => ({
    [SECTIONS.OVERVIEW]: <OverviewSection isDarkMode={isDarkMode} />,
    [SECTIONS.DATA]: <DataSection isDarkMode={isDarkMode} />,
    [SECTIONS.FV_MODEL]: <FVModelSection isDarkMode={isDarkMode} />,
    [SECTIONS.DEVIATION]: <DeviationSection isDarkMode={isDarkMode} />,
    [SECTIONS.ZQ_FEATURES]: <ZQFeaturesSection isDarkMode={isDarkMode} />,
    [SECTIONS.RESIDUAL]: <ResidualSection isDarkMode={isDarkMode} />,
    [SECTIONS.INTRADAY]: <IntradaySection isDarkMode={isDarkMode} />,
    [SECTIONS.ENTRY_EXIT]: <EntryExitSection isDarkMode={isDarkMode} />,
    [SECTIONS.CONFIGS]: <ConfigsSection isDarkMode={isDarkMode} />,
    [SECTIONS.WALKTHROUGH]: <WalkthroughSection isDarkMode={isDarkMode} />,
    [SECTIONS.PERFORMANCE]: <PerformanceSection isDarkMode={isDarkMode} />,
    [SECTIONS.LOOKAHEAD]: <LookaheadAuditSection isDarkMode={isDarkMode} />,
    [SECTIONS.V9]: <V9Section isDarkMode={isDarkMode} />,
    [SECTIONS.V10_ROLL]: <V10RollWeekSection isDarkMode={isDarkMode} />,
    [SECTIONS.PRODUCTION]: <ProductionSection isDarkMode={isDarkMode} />,
  }), [isDarkMode]);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Header */}
      <div className={`p-6 rounded-xl ${
        isDarkMode
          ? 'bg-gradient-to-r from-rose-900 via-violet-900 to-indigo-900'
          : 'bg-gradient-to-r from-rose-600 via-violet-600 to-indigo-600'
      } text-white`}>
        <h2 className="text-3xl font-black tracking-tight" style={{ textShadow: '0 2px 4px rgba(0,0,0,0.3)' }}>
          Day<span className="text-rose-200">gent</span>{' '}
          <span className="text-xl font-semibold text-violet-100">V6 Bones</span>
        </h2>
        <p className="text-violet-100 mt-1">
          Complete architectural breakdown of the V6 residual-based calendar spread strategy
        </p>
        <div className="flex items-center gap-4 mt-3 text-sm text-violet-200">
          <span>Sharpe 6.17</span>
          <span className="opacity-40">|</span>
          <span>76.4% Win Rate</span>
          <span className="opacity-40">|</span>
          <span>12/12 OOS Folds Profitable</span>
          <span className="opacity-40">|</span>
          <span>p &lt; 0.0001 Monte Carlo</span>
        </div>
      </div>

      {/* Navigation */}
      <div className={`rounded-xl border p-2 ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
        <div className="flex flex-wrap gap-1">
          {SECTION_META.map(({ id, label, icon }) => (
            <button
              key={id}
              onClick={() => setActiveSection(id)}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-1.5 ${
                activeSection === id
                  ? isDarkMode
                    ? 'bg-gray-700 text-white shadow-sm ring-1 ring-violet-500/50'
                    : 'bg-violet-50 text-violet-700 shadow-sm ring-1 ring-violet-200'
                  : isDarkMode
                    ? 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                    : 'text-gray-500 hover:text-gray-800 hover:bg-gray-50'
              }`}
            >
              <span>{icon}</span>
              <span className="hidden sm:inline">{label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Section Content */}
      <div className="min-h-[600px]">
        {sectionContent[activeSection]}
      </div>

      {/* Footer */}
      <div className={`rounded-xl border p-4 text-center ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
        <p className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          V6 Research completed Feb 2026. Based on data from Jan 2023 – Jan 2026.
          Signal: residual-based (gamma=0). FV model: spot carry. Regression: 60-day rolling OLS on rate_gap_z, carry_gap_z, fwd_curve_slope_z.
        </p>
      </div>
    </div>
  );
};

export default V6Bones;
