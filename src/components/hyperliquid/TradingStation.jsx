import { useEffect, useMemo, useState } from 'react';
import { hlGet, hlPost, fmtUsd, fmtNum, fmtPct } from './api.js';
import { cardCls, dimCls, strongCls, inputCls, btnCls, Heading, KV, Confirm, ErrorBanner } from './ui.jsx';

const COINS = [{ id: 'BTC', label: 'BTC' }, { id: 'SOL', label: 'SOL' }, { id: 'ETH', label: 'ETH' }, { id: 'xyz:SP500', label: 'SPX' }];
const coinLabel = (c) => COINS.find((x) => x.id === c)?.label || c;
const isBuilderDex = (c) => c.includes(':');   // HIP-3 markets (xyz:…): traded isolated here so the liquidation is exact

/** Place a leveraged perp order: estimate (incl. liquidation) before, the exchange's actual liquidation after. */
export default function TradingStation({ dark, network, proxyIndex, account, refresh }) {
  const [coin, setCoin] = useState('BTC');
  const [side, setSide] = useState('long');
  const [margin, setMargin] = useState('');
  const [leverage, setLeverage] = useState(5);
  const [isCross, setIsCross] = useState(false);
  const [tp, setTp] = useState('');
  const [sl, setSl] = useState('');
  const [market, setMarket] = useState(null);
  const [est, setEst] = useState(null);
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState(null);
  const [strat, setStrat] = useState(null);

  useEffect(() => { if (isBuilderDex(coin)) setIsCross(false); }, [coin]);
  // the armed SOL strategy needs free margin for its entry + add; warn before a trade eats into it
  useEffect(() => {
    let alive = true;
    const load = async () => { try { const s = await hlGet('/strategy/status'); if (alive) setStrat(s); } catch { /* panel shows its own errors */ } };
    load(); const id = setInterval(load, 30000); return () => { alive = false; clearInterval(id); };
  }, []);

  // market (mark, funding, meta) every 5 s for the chosen coin
  useEffect(() => {
    let alive = true;
    const load = async () => { try { const m = await hlGet('/market', { coin, network, proxy_index: proxyIndex }); if (alive) { if (m.error) setErr(m.error); else { setMarket(m); setErr(null); } } } catch (e) { if (alive) setErr(e.message); } };
    load(); const id = setInterval(load, 5000); return () => { alive = false; clearInterval(id); };
  }, [coin, network, proxyIndex]);

  const maxLev = market?.maxLeverage || 50;
  useEffect(() => { if (leverage > maxLev) setLeverage(maxLev); }, [maxLev]); // eslint-disable-line react-hooks/exhaustive-deps

  // server-side estimate (uses margin tiers) whenever the inputs change
  const marginVal = Number(margin) || 0;
  useEffect(() => {
    if (!(marginVal > 0)) { setEst(null); return; }
    let alive = true; const t = setTimeout(async () => {
      try { const e = await hlGet('/estimate', { coin, side, margin: marginVal, leverage, is_cross: isCross, network, proxy_index: proxyIndex }); if (alive) { if (e.error) setErr(e.error); else { setEst(e); setErr(null); } } } catch (e) { if (alive) setErr(e.message); }
    }, 250);
    return () => { alive = false; clearTimeout(t); };
  }, [coin, side, marginVal, leverage, isCross, network, proxyIndex, market?.mark_px]);

  const tpVal = Number(tp) || null, slVal = Number(sl) || null;
  const tpPnl = useMemo(() => (est && tpVal ? (side === 'long' ? tpVal - est.mark_px : est.mark_px - tpVal) * est.size : null), [est, tpVal, side]);
  const slPnl = useMemo(() => (est && slVal ? (side === 'long' ? slVal - est.mark_px : est.mark_px - slVal) * est.size : null), [est, slVal, side]);
  const available = account?.available_to_trade ?? account?.withdrawable ?? 0;
  const stratNeed = strat?.running && strat?.heartbeat?.armed && !strat?.heartbeat?.position && strat?.heartbeat?.network === network ? 2 * (strat.heartbeat.unit_margin || 0) : 0;
  const leftAfter = available - (isCross ? 0 : marginVal);

  const submit = async () => {
    setBusy(true); setResult(null); setErr(null);
    try {
      const r = await hlPost('/order', { coin, side, margin: marginVal, leverage, is_cross: isCross, tp_price: tpVal, sl_price: slVal, network, proxy_index: proxyIndex });
      setResult(r); refresh();
    } catch (e) { setErr(e.message); }
    setBusy(false); setConfirm(false);
  };

  const levPresets = [1, 2, 3, 5, 6, 10, 15, 20, 25, 40, 50].filter((l) => l <= maxLev);
  return (
    <div className={cardCls(dark)}>
      <Heading dark={dark} right={market && <span className={`text-xs font-mono ${dimCls(dark)}`}>mark {fmtNum(market.mark_px, 4)} · oracle {fmtNum(market.oracle_px, 4)} · funding {fmtPct(market.funding * 100, 4)}/h · OI {fmtNum(market.open_interest, 0)}</span>}>
        Trading station · {network}
      </Heading>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="space-y-2">
          <div className="flex gap-2 flex-wrap">
            {COINS.map((c) => <button key={c.id} className={btnCls(dark, coin === c.id)} onClick={() => setCoin(c.id)} title={c.id}>{c.label}</button>)}
            <span className={`ml-auto text-xs ${dimCls(dark)}`}>max {maxLev}x</span>
          </div>
          <div className="flex gap-2">
            <button className={btnCls(dark, side === 'long', 'green')} onClick={() => setSide('long')}>Long</button>
            <button className={btnCls(dark, side === 'short', 'red')} onClick={() => setSide('short')}>Short</button>
            <button className={btnCls(dark, !isCross)} onClick={() => setIsCross(false)}>isolated</button>
            {!isBuilderDex(coin) && <button className={btnCls(dark, isCross)} onClick={() => setIsCross(true)}>cross</button>}
          </div>
          <div className="flex gap-2 items-center">
            <span className={`text-sm w-20 ${dimCls(dark)}`}>margin $</span>
            <input className={inputCls(dark)} value={margin} onChange={(e) => setMargin(e.target.value)} placeholder="USDC to post" />
            <button className={btnCls(dark)} onClick={() => setMargin((Math.floor(available * 100) / 100).toString())} title="all available">max</button>
          </div>
          <div className={`text-xs ${dimCls(dark)}`}>available to trade {fmtUsd(available)}{marginVal > 0 && !isCross ? ` · after this trade ${fmtUsd(leftAfter)}` : ''}</div>
          <div className="flex gap-2 items-center">
            <span className={`text-sm w-20 ${dimCls(dark)}`}>leverage</span>
            <input type="range" min={1} max={maxLev} value={leverage} onChange={(e) => setLeverage(Number(e.target.value))} className="flex-1" />
            <input className={`${inputCls(dark)} w-16`} value={leverage} onChange={(e) => setLeverage(Math.max(1, Math.min(maxLev, Number(e.target.value) || 1)))} />
          </div>
          <div className="flex gap-1 flex-wrap">{levPresets.map((l) => <button key={l} className={btnCls(dark, leverage === l)} onClick={() => setLeverage(l)}>{l}x</button>)}</div>
          <div className="flex gap-2 items-center">
            <span className={`text-sm w-20 ${dimCls(dark)}`}>TP / SL</span>
            <input className={inputCls(dark)} value={tp} onChange={(e) => setTp(e.target.value)} placeholder="take profit px (optional)" />
            <input className={inputCls(dark)} value={sl} onChange={(e) => setSl(e.target.value)} placeholder="stop loss px (optional)" />
          </div>
          <button className={btnCls(dark, true, side === 'long' ? 'green' : 'red')} disabled={!est || est.size <= 0 || busy} onClick={() => { setResult(null); setConfirm(true); }}>
            {side === 'long' ? 'Buy / Long' : 'Sell / Short'} {coinLabel(coin)} {leverage}x
          </button>
        </div>
        <div className="space-y-1">
          <div className={`text-xs uppercase tracking-wide ${dimCls(dark)}`}>Estimate (before the fill)</div>
          {est ? (<>
            <KV dark={dark} label="notional" value={fmtUsd(est.notional)} />
            <KV dark={dark} label="size" value={`${fmtNum(est.size, est.sz_decimals)} ${coinLabel(coin)}`} />
            <KV dark={dark} label="entry (mark now)" value={fmtNum(est.mark_px, 4)} />
            <KV dark={dark} label="maintenance margin" value={`${(est.maint_frac * 100).toFixed(2)}% (${fmtUsd(est.notional * est.maint_frac)})`} />
            <KV dark={dark} label="est. liquidation" value={fmtNum(est.liquidation_px_estimate, 4)} valueClass="text-rose-400 font-mono font-semibold" />
            <KV dark={dark} label="distance to liq" value={est.liq_distance_pct != null ? `${fmtNum(est.liq_distance_pct, 2)}% (${fmtNum(Math.abs(est.mark_px - est.liquidation_px_estimate), 4)})` : '—'} />
            {tpPnl != null && <KV dark={dark} label="TP P&L" value={`${fmtUsd(tpPnl)} (${fmtPct(tpPnl / marginVal * 100, 1)} of margin)`} valueClass={tpPnl >= 0 ? 'text-emerald-500 font-mono' : 'text-rose-500 font-mono'} />}
            {slPnl != null && <KV dark={dark} label="SL P&L" value={`${fmtUsd(slPnl)} (${fmtPct(slPnl / marginVal * 100, 1)} of margin)`} valueClass={slPnl >= 0 ? 'text-emerald-500 font-mono' : 'text-rose-500 font-mono'} />}
            {!isCross && <KV dark={dark} label="loss if liquidated" value={`${fmtUsd(marginVal)} (all the margin)`} />}
            {stratNeed > 0 && !isCross && marginVal > 0 && leftAfter < stratNeed && (
              <div className="text-xs text-amber-400">Leaves {fmtUsd(leftAfter)} free; the armed SOL strategy needs {fmtUsd(stratNeed)} for a full cascade (entry + add){leftAfter < stratNeed / 2 ? ' and could not even enter' : ', so its add could be skipped'}.</div>)}
            <div className={`text-xs ${dimCls(dark)}`}>Formula: Hyperliquid's documented liquidation price with the margin tier for this notional. The exchange's own number appears after the fill.</div>
          </>) : <div className={`text-sm ${dimCls(dark)}`}>enter a margin amount</div>}
        </div>
      </div>
      {confirm && est && <Confirm dark={dark} tone={side === 'long' ? 'green' : 'red'} yesLabel={`Place ${side} ${coinLabel(coin)}`} busy={busy}
        title={`${side.toUpperCase()} ${fmtNum(est.size, est.sz_decimals)} ${coinLabel(coin)} (${fmtUsd(est.notional)}) at ${leverage}x ${isCross ? 'cross' : 'isolated'} on ${network}`}
        lines={[`Margin ${fmtUsd(marginVal)} · est. liquidation ${fmtNum(est.liquidation_px_estimate, 4)} (${fmtNum(est.liq_distance_pct, 1)}% away)`, 'IOC through the book, 3 % slippage cap; taker fee applies.', tpVal ? `TP ${tpVal}` : 'no TP', slVal ? `SL ${slVal}` : 'no SL', `Routed via ${proxyIndex < 0 ? 'direct' : `Proxy ${proxyIndex + 1}`}.`]}
        onYes={submit} onNo={() => setConfirm(false)} />}
      {result && (
        <div className={`mt-3 rounded-lg border p-3 ${result.ok ? (dark ? 'border-emerald-600/60 bg-emerald-900/20' : 'border-emerald-400 bg-emerald-50') : (dark ? 'border-rose-600/60 bg-rose-900/20' : 'border-rose-400 bg-rose-50')}`}>
          {result.ok ? (<>
            <div className={`font-semibold ${strongCls(dark)}`}>Filled {fmtNum(result.fill.filled_sz, 5)} {coinLabel(coin)} @ {fmtNum(result.fill.avg_px, 4)}</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-1 text-sm">
              <KV dark={dark} label="est. liquidation" value={fmtNum(result.liquidation_px_estimate, 4)} />
              <KV dark={dark} label="actual liquidation" value={fmtNum(result.liquidation_px_actual, 4)} valueClass="text-rose-400 font-mono font-semibold" />
              <KV dark={dark} label="difference" value={result.liquidation_diff_px != null ? `${result.liquidation_diff_px >= 0 ? '+' : ''}${fmtNum(result.liquidation_diff_px, 4)}` : '—'} />
              <KV dark={dark} label="position" value={result.position ? `${result.position.side} ${fmtNum(result.position.size, 5)} @ ${fmtNum(result.position.entry_px, 4)}` : 'not open'} />
            </div>
            <div className={`text-xs mt-1 ${dimCls(dark)}`}>The positions table above keeps the actual liquidation live from here on.</div>
          </>) : <div className="text-rose-400 text-sm">{result.error || result.fill?.error || 'order not filled'}{result.step ? ` (step: ${result.step})` : ''}</div>}
        </div>
      )}
      <ErrorBanner dark={dark} error={err} />
    </div>
  );
}
