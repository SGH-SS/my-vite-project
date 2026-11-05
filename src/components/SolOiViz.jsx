/**
 * SOL OI Viz — the BXT Visualizer, pointed at the preloaded trades of the SOL
 * liquidation-cascade strategies (map.md §17–17.2, research/retrace_20260930).
 *
 * Everything the BXT Visualizer can do still works (candle sizes, walls, flow,
 * L20 dots, full screen, hotkeys, breakouts). On top: pick a strategy, step
 * through its trades (buttons, the list, or [ / ] keys) and the chart zips to
 * a window around that trade with the entry/exit marked.
 *
 * Trades are PRELOADED from /sol_oi_trades.json (written by
 * research/retrace_20260930/r26_export_trades.py). Nothing is computed here —
 * live strategy evaluation is a later step.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import BxtVisualizer from './BxtVisualizer.jsx';

const TRADES_URL = '/sol_oi_trades.json';

// window around the focused trade: seconds before entry / after exit
const WINDOWS = [
  { key: 'tight',  label: 'Tight (30m / 30m)', before: 30 * 60,  after: 30 * 60 },
  { key: 'normal', label: 'Normal (2h / 1h)',  before: 2 * 3600, after: 3600 },
  { key: 'wide',   label: 'Wide (6h / 3h)',    before: 6 * 3600, after: 3 * 3600 },
  { key: 'day',    label: 'Day (12h / 6h)',    before: 12 * 3600, after: 6 * 3600 },
];

// How each strategy works — written for the dashboard, numbers from the Sep 30 / Oct 1 2026 runs.
const DESCRIPTIONS = {
  v2: {
    title: 'V2 — consecutive red candles, large run, open interest collapsed',
    mechanism: 'A leveraged long that breaches maintenance margin is market-sold by the exchange. That sell pushes price down and liquidates the next tier of longs: a cascade. The selling is forced, not informed, and it ends when the last account is closed. V2 buys the recovery after a cascade identified by three or more consecutive red 1-minute candles.',
    rules: [
      '3 or more consecutive red 1-minute candles (all complete, clean buckets).',
      'The run\'s total drop is at least 3 standard deviations of the trailing day\'s 1-minute moves.',
      'Open interest at the last completed minute is at least 50 bp (0.5 %) below OI the minute before the run started.',
      'Buy at the next 5-second close (taker). Hold 60 minutes, sell (taker). No stop, no target.',
    ],
    breakdown: 'The OI drop is the base condition and its response is monotone: the more OI collapsed during the run, the bigger the bounce. The extension in sd units is monotone too. Nothing at the 5 s / 15 s level (walls, flow, deep book) adds to it. Entry timing inside the first 5 minutes does not matter; fixed 60–120 minute holds beat every mechanism exit tested. This is the least-screened rule in the study and the anchor for the paper run.',
    caveats: 'SOL only (ETH does not replicate at cost, BTC is marginal). ~110 trades over five months. Capacity is tens of thousands of dollars per trade (depth-walked slippage 0.3–0.4 bps at $20–50k).',
  },
  v5: {
    title: 'V5 — 15-minute drop, OI collapse, SOL over-sold vs BTC, high cascade intensity',
    mechanism: 'Same liquidation-cascade mechanism as V2, but the event is "any 1-minute bar whose trailing 15-minute return is at or below −2 sd of the day" (no candle-colour pattern — only 29 % of sharp drops happen inside a 3-red-candle run). Two more fingerprints are added: SOL\'s excess drop versus BTC over the same 15 minutes (forced SOL-specific selling on top of the market move), and the breadth of forced selling read from the BXT sweep / book families.',
    rules: [
      'Trailing 15-minute return ≤ −2 sd of the trailing day\'s 1-minute moves.',
      'OI fell more than 50 bp over those 15 minutes.',
      'SOL under-performed BTC by more than 40 bp over the same 15 minutes.',
      'Cascade-intensity composite in the top half: equal-weight ranks of sell-sweep count (5 m), taker volume vs daily average, max levels eaten by one sweep, bid-ladder span in ticks, bid replenishment over 5 m.',
      'Buy at the next 5-second close, hold 60 minutes, sell. One position at a time.',
    ],
    breakdown: 'Both added mechanisms are monotone on their own: OI change over 15 m (fell >200 bp → +44 bps @60 m … rose → +2) and excess vs BTC (>80 bp → +43 … outperformed → −5). The intensity composite is monotone inside the OI universe (+5 → +52 bps by quintile, positive rank correlation every month). It is the breadth of forced selling — many separate sweeps, holes in the bid ladder, bids being rebuilt — that predicts the bounce, not the imbalance of what is left resting.',
    caveats: 'More screened than V2 (the same features were examined on three coins and several cuts). Run it beside V2, not instead of it. Market-wide cascades bounce MORE, so a BTC hedge kills the trade.',
  },
  v6: {
    title: 'V6 — 15-minute drop, OI collapse, cascade intensity in the top quintile',
    mechanism: 'The intensity bundle alone, as a filter on the OI-collapse universe. It picks the cascades where forced selling was broadest: many distinct sell sweeps, volume far above the daily average, single sweeps eating many levels, a hollowed-out bid ladder, and bids being replenished as the cascade ends.',
    rules: [
      'Trailing 15-minute return ≤ −2 sd; OI fell more than 50 bp over the 15 minutes.',
      'Cascade-intensity composite (same five BXT features as V5, equal-weight ranks) in the top quintile of that universe.',
      'Buy at the next 5-second close, hold 60 minutes, sell. One position at a time.',
    ],
    breakdown: 'Best risk-adjusted line in the study: positive in every month at every hold from 30 to 120 minutes, lowest drawdown of all variants. Only ~60 trades in five months, so it is better used as a size multiplier inside V2 / V5 than as a standalone line.',
    caveats: 'The top-quintile cut is computed on the whole sample; live it must be a trailing quantile. Few trades — one bad week moves the numbers.',
  },
  lgbm5: {
    title: 'LightGBM — out-of-fold top 5 % score',
    mechanism: 'Gradient-boosted trees trained on the 15-minute-drop bars (≈5,200 SOL bars) with 64 mechanism features from the 1-minute BXT panel: OI changes over 1/5/15/60 m, SOL-vs-BTC excess, BTC\'s own move and OI, taker imbalance, sweep counts / concentration / levels, replenishment, ladder spans, order counts, impact, deep-book bands, basis. No time-of-day or regime features (an all-bars model learned only those and was month-unstable). Target: the 60-minute forward return, clipped at ±300 bps. Fixed parameters, nothing searched.',
    rules: [
      'Leave-one-month-out: each month is scored by a model trained on the other months.',
      'Threshold: the 95th percentile of the OTHER months\' out-of-fold scores (so the cut is causal by month).',
      'Trade every bar whose score is above that threshold: buy at the next 5-second close, hold 60 minutes. One position at a time.',
    ],
    breakdown: 'Top importance: 60-minute return, 60-minute OI change, excess vs BTC, BTC\'s return and OI, then deep-bid depth, taker imbalance, fills-per-sweep and max sweep levels — the same fingerprints as the hand rules. The strict walk-forward test (train only on earlier months) is useless with two months of training and strong with three or more: Jul–Sep top 5 % = 11 trades, +133 bps net, 91 % hit, t 3.2.',
    caveats: 'Leave-one-month-out lets a model trained on later months score earlier ones; the walk-forward numbers are the honest ones and cover only Jul–Sep. Trades once or twice a week. June was negative in this version. Use as a scorer / size multiplier inside V2 and V5 until it has prospective months behind it.',
  },
  lgbm10: {
    title: 'LightGBM — out-of-fold top 10 % score',
    mechanism: 'Same model as the top-5 % line, with the gate at the 90th percentile of the other months\' scores: roughly twice the trades at roughly half the per-trade edge.',
    rules: [
      'Leave-one-month-out scoring; threshold = 90th percentile of the other months\' out-of-fold scores.',
      'Buy at the next 5-second close when the score clears it, hold 60 minutes. One position at a time.',
    ],
    breakdown: 'Lands where the hand rules already are (≈ +19 bps net at 60 m, better at 80–120 m). It confirms the hand rules; it does not beat them.',
    caveats: 'Same as the top-5 % line. Out-of-fold by month, not walk-forward.',
  },
  v2_add: {
    title: 'V2 + Add — a second unit when the cascade keeps going after entry',
    mechanism: 'Bigger forced selling means a bigger bounce. If, after entry, the price keeps falling AND open interest keeps dropping, more longs are being liquidated than the signal bar showed; the eventual bounce is larger and the second unit buys it lower. Price alone carries most of the information, OI alone little (OI fell >25 bp alone adds +4…+9 bps over an unconditional add; both together +23…+30); a new run low alone is worse than adding blind.',
    rules: [
      'V2 entry as before (one unit at the signal bar\'s close).',
      'Within 10 minutes of entry, at the first 1-minute close where price is at least 20 bps below the entry AND OI is at least 25 bp below OI at the entry minute: buy a second unit (taker).',
      'Both units exit at the original 60-minute mark. No stop, no target.',
    ],
    breakdown: 'Fires on about a quarter of the trades. The second unit alone nets +46 bps (median +44, hit 79 %, positive every month, +38…+72). Per unit of capital the strategy goes from +22.5 to +27.3 bps (ex the Aug 22 flash crash), drawdown −305 → −355. Placebo (always add at minute 2–15): +16…+18 on the second unit, i.e. no information.',
    caveats: 'Same 110 trades, one coin. The Aug 22 05:10 UTC flash crash (−8 % in 85 s) alone adds +650 bps to any add rule; the honest numbers exclude it (toggle in the stats card).',
  },
  v2_ext: {
    title: 'V2 + Fresh-Liq — hold on at 60 m while open interest is still falling',
    mechanism: 'The 60-minute exit is a clock, not a signal. If at the 60-minute mark OI has fallen again over the last 30 minutes, liquidations are still happening and the bounce is not finished. Forward return from the 60 m mark by that OI change: fell >50 bp → +46, 5–20 → +22, flat → −15, rose → −2. The mirror placebo (hold on when OI ROSE) returns the base result.',
    rules: [
      'V2 entry and hold as before.',
      'At the 60-minute mark: if OI is more than 10 bp below OI 30 minutes earlier, hold 60 more minutes (exit at 120 m). Otherwise exit at 60 m.',
    ],
    breakdown: 'Extends 37 % of trades; the extension increment is +27 bps (median +13, t 2.8; months +26/+23/−9/+34/+54). Strategy +22.5 → +29.3 per trade ex-crash, t 3.8, drawdown −321. Cuts of 10/25/50 bp all work (+26…+46 per extension).',
    caveats: 'P&L-band extensions ("hold if only a small loser") do not survive any stop or target; this one is conditioned on the mechanism instead. Still ~40 extended trades.',
  },
  v2_up: {
    title: 'V2 + Up-front sizing — two units when the cascade was fast and the OI drop large',
    mechanism: 'Cascade speed (run drop in bps per minute of the run) is the strongest single descriptor of the bounce: 60 m net by speed quartile +1 / +10 / +23 / +55. Forced selling is fast; a slow ten-candle bleed is positioning. Combined with the size of the OI drop over the run, it picks the cascades where the most longs were closed in the shortest time.',
    rules: [
      'V2 entry as before.',
      'If the run\'s speed is at least 15 bps per minute AND OI fell more than 100 bp over the run: buy two units at entry instead of one.',
      'Hold 60 minutes, exit. No stop, no target.',
    ],
    breakdown: 'Two units on 23 % of trades. Per unit +22.5 → +31.7 ex-crash; drawdown −305 → −465. Known at entry, so it is pure sizing, not new information after the fill.',
    caveats: 'Median speed swings 13 → 23 bps/min by month, so a fixed cut fires 30–70 % of the time depending on regime. The cut (15) is a round number near the median, not fitted — but it is a cut.',
  },
  v2_add_ext: {
    title: 'V2 + Add + Fresh-Liq',
    mechanism: 'The add acts in the first 10 minutes (more liquidation after entry → second unit lower); the extension acts at minute 60 (liquidation still running → hold on). They read the same mechanism at two different times and do not overlap.',
    rules: ['V2 entry.', 'Add rule (price ≤ entry −20 bps AND OI −25 bp since entry, within 10 min).', 'At 60 m, if OI fell >10 bp over the last 30 min, hold all units to 120 m.'],
    breakdown: 'Ex-crash: +43.3 per trade, +34.6 per unit, t 4.7, drawdown −434. The recommended form for the paper run.',
    caveats: 'Same 110 trades. Two conditional layers on one coin over five months.',
  },
  v2_add_up: {
    title: 'V2 + Add + Up-front sizing',
    mechanism: 'Sizing on what is known at entry (speed and the run\'s OI drop) plus the add on what happens after entry. Both say "more forced selling → bigger bounce", so they add in size and in drawdown together.',
    rules: ['V2 entry; two units if speed ≥ 15 bps/min AND run OI drop > 100 bp.', 'Add rule within 10 min.', 'All units out at 60 m.'],
    breakdown: 'Ex-crash: +50.6 per trade, +34.2 per unit, t 4.0, drawdown −477; average 1.5 units.',
    caveats: 'Up to three units in a cascade: capacity is tens of thousands of dollars per unit at 0.3–0.4 bps slippage.',
  },
  v2_ext_up: {
    title: 'V2 + Fresh-Liq + Up-front sizing',
    mechanism: 'Size at entry on the cascade\'s speed and OI drop; at 60 m hold on if OI is still falling.',
    rules: ['V2 entry; two units if speed ≥ 15 bps/min AND run OI drop > 100 bp.', 'At 60 m, if OI fell >10 bp over the last 30 min, hold all units to 120 m.'],
    breakdown: 'Ex-crash: +47.8 per trade, +38.8 per unit, t 3.4, drawdown −465.',
    caveats: 'The sized trades are the biggest cascades, which are usually done by 60 m (I4 in map.md §17.5); the extension mostly fires on the others.',
  },
  v2_all: {
    title: 'V2 + Add + Fresh-Liq + Up-front sizing',
    mechanism: 'All three layers: size at entry, add on continued liquidation, hold on while OI is still falling.',
    rules: ['V2 entry; two units if speed ≥ 15 AND run OI drop > 100 bp.', 'Add within 10 min on price ≤ −20 bps AND OI −25 bp.', 'At 60 m, if OI fell >10 bp over the last 30 min, hold to 120 m.'],
    breakdown: 'Ex-crash: +61.8 per trade, +41.6 per unit, t 4.0, drawdown −477; average 1.5 units, 82 m average hold.',
    caveats: 'Each layer is the same fact measured at a different time, so the drawdown grows with the size. Nothing here is proven until the paper run.',
  },
};

const fmtUtc = (s) => new Date(s * 1000).toISOString().replace('T', ' ').slice(0, 16);
const fmtBps = (v) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}`);
const CTX_LABELS = {
  run_candles: 'red candles in run', ext_bps: 'run drop (bps)', ext_sd: 'run drop (sd)', oi_change_bp: 'OI change over run (bp)',
  drop15_bps: '15m drop (bps)', drop15_sd: '15m drop (sd)', oi_change15_bp: 'OI change 15m (bp)', excess_vs_btc_bp: 'excess vs BTC (bp)',
  intensity: 'cascade intensity (0–1)', sweeps_sell5: 'sell sweeps 5m', vol_rel15: 'volume vs daily avg', msl_sell5: 'max sweep levels 5m',
  span_bid: 'bid span (ticks)', repl_bid5: 'bid replenish 5m (×depth)', model_score: 'model score (bps)', speed_bps_min: 'cascade speed (bps/min)',
};
const STD_KEYS = new Set(['signal_t', 'entry_t', 'exit_t', 'entry_px', 'exit_px', 'gross_bps', 'net_bps', 'hold_min', 'month', 'day', 'units', 'units_n', 'added', 'extended', 'upfront']);
const unitsLabel = (t) => (t.units_n > 1 || t.extended) ? [t.upfront ? '2 up front' : null, t.added ? '+1 add' : null, t.extended ? 'ext 120m' : null].filter(Boolean).join(', ') : '';

export default function SolOiViz({ isDarkMode }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [stratId, setStratId] = useState('v2');
  const [idx, setIdx] = useState(0);
  const [winKey, setWinKey] = useState('normal');
  const [showAll, setShowAll] = useState(true);     // mark every trade in the window, not just the focused one
  const [showDesc, setShowDesc] = useState(true);
  const [exCrash, setExCrash] = useState(true);     // stats ex the Aug 22 flash crash, when the strategy carries them
  const [focusSeq, setFocusSeq] = useState(0);      // bumps so re-zipping to the same trade re-fires the focus

  useEffect(() => {
    fetch(TRADES_URL).then(async (r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setData(await r.json());
    }).catch((e) => setError(String(e.message || e)));
  }, []);

  const strat = useMemo(() => data?.strategies.find((s) => s.id === stratId) ?? null, [data, stratId]);
  const trades = strat?.trades ?? [];
  const trade = trades[idx] ?? null;
  const win = WINDOWS.find((w) => w.key === winKey) ?? WINDOWS[1];

  const goTo = useCallback((i) => {
    if (!trades.length) return;
    const n = ((i % trades.length) + trades.length) % trades.length;
    setIdx(n); setFocusSeq((s) => s + 1);
  }, [trades.length]);
  const prev = useCallback(() => goTo(idx - 1), [goTo, idx]);
  const next = useCallback(() => goTo(idx + 1), [goTo, idx]);

  // switching strategy starts at its first trade
  useEffect(() => { setIdx(0); setFocusSeq((s) => s + 1); }, [stratId]);

  // [ / ] step trades; ignored while typing in a field
  useEffect(() => {
    const onKey = (e) => {
      const tag = (e.target?.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'select' || tag === 'textarea' || e.target?.isContentEditable) return;
      if (e.key === '[') { e.preventDefault(); prev(); }
      if (e.key === ']') { e.preventDefault(); next(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [prev, next]);

  // the chart window around the focused trade (fixed absolute window, like a breakout)
  const focus = useMemo(() => {
    if (!trade) return null;
    return {
      schema: 'sol',
      fixedWindow: { start: trade.entry_t - win.before, end: trade.exit_t + win.after },
      seq: focusSeq,
    };
  }, [trade, win, focusSeq]);

  // markers: every trade overlapping the window (or only the focused one)
  const markers = useMemo(() => {
    if (!trade || !focus) return [];
    const { start, end } = focus.fixedWindow;
    const list = showAll ? trades.filter((t) => t.exit_t >= start && t.entry_t <= end) : [trade];
    return list.map((t) => ({
      entryT: t.entry_t, exitT: t.exit_t, entryPx: t.entry_px, exitPx: t.exit_px, netBps: t.net_bps,
      label: `#${trades.indexOf(t) + 1}`, focus: t === trade,
      units: t.units ?? null, extended: !!t.extended, plannedExitT: t.extended ? t.entry_t + 3600 : null,
    }));
  }, [trade, focus, trades, showAll]);

  const initialPanel = useMemo(() => (trade ? {
    schema: 'sol', tfS: 60, fullscreen: false,
    fixedWindow: { start: trade.entry_t - win.before, end: trade.exit_t + win.after },
  } : null), [trade != null]); // eslint-disable-line react-hooks/exhaustive-deps

  const card = `rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`;
  const dim = isDarkMode ? 'text-gray-400' : 'text-gray-500';
  const strong = isDarkMode ? 'text-gray-100' : 'text-gray-900';
  const btn = (active) => `px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${active
    ? (isDarkMode ? 'bg-amber-500 text-gray-900' : 'bg-amber-500 text-white')
    : (isDarkMode ? 'bg-gray-700 text-gray-200 hover:bg-gray-600' : 'bg-gray-100 text-gray-700 hover:bg-gray-200')}`;

  if (error) return <div className={card}>Could not load {TRADES_URL}: {error}</div>;
  if (!data || !strat) return <div className={card}>Loading preloaded trades…</div>;

  const d = DESCRIPTIONS[strat.id];
  const st = exCrash && strat.stats_ex_crash ? strat.stats_ex_crash : strat.stats;
  const ctxEntries = trade ? Object.entries(trade).filter(([k, v]) => !STD_KEYS.has(k) && v != null) : [];

  return (
    <div className="space-y-4">
      {/* strategy tabs */}
      <div className={card}>
        <div className="flex flex-wrap items-center gap-2">
          {data.strategies.map((s) => (
            <button key={s.id} className={btn(s.id === stratId)} onClick={() => setStratId(s.id)} title={s.name}>
              {s.name.split(' · ')[0]} <span className="opacity-70">· {s.stats.trades} trades · {fmtBps(s.stats.net_bps_per_trade)} bps{s.stats.net_bps_per_unit != null && s.stats.avg_units > 1 ? ` (${fmtBps(s.stats.net_bps_per_unit)}/unit)` : ''}</span>
            </button>
          ))}
          <span className={`ml-auto text-xs ${dim}`}>
            preloaded · SOL · data {data.data_from} → {data.data_to} UTC ({data.days} days) · base hold {data.hold_min} m · taker {data.cost_bps_rt} bps RT per unit · generated {data.generated_utc} UTC
          </span>
        </div>
      </div>

      {/* description + stats */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className={`${card} lg:col-span-2`}>
          <div className="flex items-start justify-between gap-3">
            <h2 className={`text-lg font-semibold ${strong}`}>{d?.title ?? strat.name}</h2>
            <button className={`text-xs ${dim} hover:underline`} onClick={() => setShowDesc((v) => !v)}>{showDesc ? 'hide' : 'show'} breakdown</button>
          </div>
          {showDesc && d && (
            <div className={`mt-2 space-y-3 text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              <p><span className={`font-semibold ${strong}`}>Mechanism. </span>{d.mechanism}</p>
              <div>
                <div className={`font-semibold ${strong}`}>Rules, exactly as simulated</div>
                <ol className="list-decimal ml-5 mt-1 space-y-0.5">{d.rules.map((r, i) => <li key={i}>{r}</li>)}</ol>
              </div>
              <p><span className={`font-semibold ${strong}`}>Breakdown. </span>{d.breakdown}</p>
              <p className={dim}><span className="font-semibold">Caveats. </span>{d.caveats}</p>
            </div>
          )}
        </div>
        <div className={card}>
          <div className={`flex items-center justify-between text-xs uppercase tracking-wide ${dim}`}>
            <span>Stats (one position at a time, net of costs)</span>
            {strat.stats_ex_crash && (
              <label className="flex items-center gap-1 normal-case tracking-normal" title="Aug 22 05:10 UTC: SOL −8 % in 85 s, +5 % back in 3 min — one event that dominates any entry/size rule">
                <input type="checkbox" checked={exCrash} onChange={(e) => setExCrash(e.target.checked)} /> ex Aug 22 crash
              </label>
            )}
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2 text-sm">
            {[
              ['trades', st.trades], ['per week', st.per_week], ['net bps / trade', fmtBps(st.net_bps_per_trade)], ['gross bps / trade', fmtBps(st.gross_bps_per_trade)],
              ['hit rate', `${st.hit_pct}%`], ['t (by day)', st.t ?? '—'], ['months positive', st.months_positive], ['max drawdown', `${st.max_drawdown_bps} bps`],
              ['sum', `${fmtBps(st.sum_bps)} bps`], ['hold', `${st.hold_min} m`],
              ...(st.net_bps_per_unit != null ? [['net bps / unit', fmtBps(st.net_bps_per_unit)], ['avg units', st.avg_units]] : []),
              ...(st.added_pct ? [['added %', `${st.added_pct}%`]] : []), ...(st.extended_pct ? [['extended %', `${st.extended_pct}%`]] : []), ...(st.upfront_pct ? [['2 up front %', `${st.upfront_pct}%`]] : []),
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between gap-2"><span className={dim}>{k}</span><span className={`font-mono ${strong}`}>{v}</span></div>
            ))}
          </div>
          <div className={`mt-3 text-xs ${dim}`}>by month (sum, bps)</div>
          <div className="flex flex-wrap gap-1 mt-1">
            {Object.entries(st.by_month).map(([m, v]) => (
              <span key={m} className={`px-2 py-0.5 rounded font-mono text-xs ${v >= 0
                ? (isDarkMode ? 'bg-emerald-900/40 text-emerald-300' : 'bg-emerald-100 text-emerald-800')
                : (isDarkMode ? 'bg-rose-900/40 text-rose-300' : 'bg-rose-100 text-rose-800')}`}>{m}: {fmtBps(v)}</span>
            ))}
          </div>
        </div>
      </div>

      {/* trade stepper + current trade */}
      <div className={card}>
        <div className="flex flex-wrap items-center gap-2">
          <button className={btn(false)} onClick={prev} title="[">◀ prev</button>
          <span className={`font-mono text-sm ${strong}`}>trade {idx + 1} / {trades.length}</span>
          <button className={btn(false)} onClick={next} title="]">next ▶</button>
          <span className={`text-xs ${dim}`}>keys: [ and ]</span>
          <span className={`ml-4 text-xs ${dim}`}>window</span>
          <select value={winKey} onChange={(e) => { setWinKey(e.target.value); setFocusSeq((s) => s + 1); }}
            className={`text-sm rounded px-2 py-1 ${isDarkMode ? 'bg-gray-700 text-gray-100' : 'bg-gray-100 text-gray-900'}`}>
            {WINDOWS.map((w) => <option key={w.key} value={w.key}>{w.label}</option>)}
          </select>
          <label className={`flex items-center gap-1 text-xs ${dim}`}>
            <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} /> mark every trade in the window
          </label>
          <button className={`${btn(false)} ml-auto`} onClick={() => setFocusSeq((s) => s + 1)} title="re-centre the chart on this trade">⟲ re-zip</button>
        </div>
        {trade && (
          <div className="mt-3 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-x-4 gap-y-1 text-sm">
            <div><span className={dim}>signal bar </span><span className={`font-mono ${strong}`}>{fmtUtc(trade.signal_t)}</span></div>
            <div><span className={dim}>entry </span><span className={`font-mono ${strong}`}>{fmtUtc(trade.entry_t)} @ {trade.entry_px}</span></div>
            <div><span className={dim}>exit </span><span className={`font-mono ${strong}`}>{fmtUtc(trade.exit_t)} @ {trade.exit_px}</span></div>
            <div><span className={dim}>gross </span><span className={`font-mono ${strong}`}>{fmtBps(trade.gross_bps)} bps</span></div>
            <div><span className={dim}>net </span><span className={`font-mono font-semibold ${trade.net_bps >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>{fmtBps(trade.net_bps)} bps</span></div>
            <div><span className={dim}>hold </span><span className={`font-mono ${strong}`}>{trade.hold_min} m{trade.extended ? ' (extended)' : ''}</span></div>
            {trade.units && trade.units.length > 1 && trade.units.map((u, i) => (
              <div key={i}><span className={dim}>{u.kind === 'add' ? 'add' : u.kind === 'upfront' ? '2nd unit (up front)' : 'unit 1'} </span>
                <span className={`font-mono ${strong}`}>{fmtUtc(u.t).slice(6)} @ {u.px} → <span className={u.net_bps >= 0 ? 'text-emerald-500' : 'text-rose-500'}>{fmtBps(u.net_bps)}</span></span></div>
            ))}
            {ctxEntries.map(([k, v]) => (
              <div key={k}><span className={dim}>{CTX_LABELS[k] ?? k} </span><span className={`font-mono ${strong}`}>{typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(2)) : String(v)}</span></div>
            ))}
          </div>
        )}
      </div>

      {/* trade list */}
      <div className={`${card} max-h-56 overflow-auto p-0`}>
        <table className="w-full text-xs">
          <thead className={`sticky top-0 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
            <tr className={dim}>
              <th className="text-left px-3 py-1.5">#</th><th className="text-left px-3 py-1.5">entry (UTC)</th><th className="text-right px-3 py-1.5">entry px</th>
              <th className="text-right px-3 py-1.5">exit px</th><th className="text-right px-3 py-1.5">gross</th><th className="text-right px-3 py-1.5">net bps</th><th className="text-left px-3 py-1.5">units</th><th className="text-left px-3 py-1.5">context</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t, i) => (
              <tr key={t.entry_t} onClick={() => goTo(i)}
                className={`cursor-pointer ${i === idx ? (isDarkMode ? 'bg-amber-500/20' : 'bg-amber-100') : (isDarkMode ? 'hover:bg-gray-700/60' : 'hover:bg-gray-50')}`}>
                <td className="px-3 py-1 font-mono">{i + 1}</td>
                <td className="px-3 py-1 font-mono">{fmtUtc(t.entry_t)}</td>
                <td className="px-3 py-1 font-mono text-right">{t.entry_px}</td>
                <td className="px-3 py-1 font-mono text-right">{t.exit_px}</td>
                <td className="px-3 py-1 font-mono text-right">{fmtBps(t.gross_bps)}</td>
                <td className={`px-3 py-1 font-mono text-right font-semibold ${t.net_bps >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>{fmtBps(t.net_bps)}</td>
                <td className={`px-3 py-1 ${isDarkMode ? 'text-sky-300' : 'text-sky-700'}`}>{unitsLabel(t)}</td>
                <td className={`px-3 py-1 ${dim}`}>
                  {Object.entries(t).filter(([k, v]) => !STD_KEYS.has(k) && v != null).slice(0, 4).map(([k, v]) => `${CTX_LABELS[k] ?? k}: ${typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(1)) : v}`).join(' · ')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* the chart — the full BXT Visualizer, zipped to the focused trade */}
      {initialPanel && (
        <BxtVisualizer
          isDarkMode={isDarkMode} initialPanel={initialPanel} focus={focus} markers={markers} hideAddPanel
          tradeNav={{ index: idx, count: trades.length, label: trade ? `${fmtBps(trade.net_bps)} bps` : '', onPrev: prev, onNext: next }}
        />
      )}
    </div>
  );
}
