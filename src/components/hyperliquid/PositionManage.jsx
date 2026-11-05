import { useEffect, useState } from 'react';
import { hlGet, hlPost, fmtUsd, fmtNum } from './api.js';
import { dimCls, strongCls, inputCls, btnCls, KV, Confirm } from './ui.jsx';

/**
 * Manage an open position: change leverage, add to it, reduce part of it, add/remove isolated margin.
 * Every action shows the estimated liquidation after the change, needs a confirm, and reports the exchange's
 * actual liquidation once done.
 */
const TABS = [['leverage', 'Leverage']];   // add / reduce / margin endpoints exist in the backend; the panel keeps one control

export default function PositionManage({ dark, pos, network, proxyIndex, refresh }) {
  const [tab, setTab] = useState('leverage');
  const [lev, setLev] = useState(String(Math.round(pos.leverage_effective || pos.leverage || 1)));
  const [tgtMargin, setTgtMargin] = useState(String(Math.round(pos.margin_used)));
  const [addMargin, setAddMargin] = useState('');
  const [addLev, setAddLev] = useState(String(pos.leverage || 1));
  const [reducePct, setReducePct] = useState('50');
  const [marginDelta, setMarginDelta] = useState('');
  const [preview, setPreview] = useState(null);
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);

  const params = () => {
    if (tab === 'leverage') return { new_leverage: Number(lev) || null, target_margin: Number(tgtMargin) || null };
    if (tab === 'add') return { add_margin: Number(addMargin) || 0, new_leverage: Number(addLev) !== Number(pos.leverage) ? Number(addLev) : null };
    if (tab === 'reduce') return { reduce_fraction: Math.min(1, Math.max(0, Number(reducePct) / 100 || 0)) };
    return { margin_delta: Number(marginDelta) || 0 };
  };
  const valid = () => {
    const p = params();
    if (tab === 'leverage') return p.new_leverage >= 1 && p.target_margin > 0 && (Math.abs(p.new_leverage - (pos.leverage_effective || pos.leverage)) > 0.05 || Math.abs(p.target_margin - pos.margin_used) > 1);
    if (tab === 'add') return p.add_margin > 0;
    if (tab === 'reduce') return p.reduce_fraction > 0 && p.reduce_fraction < 1;
    return p.margin_delta !== 0;
  };

  useEffect(() => {
    if (!valid()) { setPreview(null); return; }
    let alive = true; const t = setTimeout(async () => {
      try { const r = await hlGet('/position/preview', { coin: pos.coin, network, proxy_index: proxyIndex, ...params() }); if (alive) { setPreview(r.after || null); setErr(r.error || null); } }
      catch (e) { if (alive) setErr(e.message); }
    }, 250);
    return () => { alive = false; clearTimeout(t); };
  }, [tab, lev, tgtMargin, addMargin, addLev, reducePct, marginDelta, pos.mark_px]); // eslint-disable-line react-hooks/exhaustive-deps

  const run = async () => {
    setBusy(true); setRes(null); setErr(null);
    try {
      const p = params(); let r;
      if (tab === 'leverage') r = await hlPost('/position/leverage', { coin: pos.coin, leverage: p.new_leverage, margin: p.target_margin, network, proxy_index: proxyIndex });
      else if (tab === 'add') r = await hlPost('/position/add', { coin: pos.coin, margin: p.add_margin, leverage: p.new_leverage, network, proxy_index: proxyIndex });
      else if (tab === 'reduce') r = await hlPost('/position/reduce', { coin: pos.coin, fraction: p.reduce_fraction, network, proxy_index: proxyIndex });
      else r = await hlPost('/position/margin', { coin: pos.coin, amount: p.margin_delta, network, proxy_index: proxyIndex });
      setRes(r); refresh();
    } catch (e) { setErr(e.message); }
    setBusy(false); setConfirm(false);
  };

  const summary = () => {
    const p = params();
    if (tab === 'leverage') return `${pos.coin}: ${fmtUsd(p.target_margin)} of margin at ${p.new_leverage}x = ${fmtUsd(p.target_margin * p.new_leverage)} of ${pos.coin}. ${preview && preview.size_delta > 0 ? `Buys ${fmtNum(preview.size_delta, 5)} ${pos.coin} (≈ ${fmtUsd(preview.notional_delta)}) at market` : preview && preview.size_delta < 0 ? `Sells ${fmtNum(-preview.size_delta, 5)} ${pos.coin} (≈ ${fmtUsd(-preview.notional_delta)}) at market, reduce-only` : 'size unchanged'}; margin ${fmtUsd(pos.margin_used)} → ${fmtUsd(p.target_margin)}.`;
    if (tab === 'add') return `Add ${fmtUsd(p.add_margin)} margin at ${p.new_leverage || pos.leverage}x ≈ ${fmtUsd(p.add_margin * (p.new_leverage || pos.leverage))} more notional, same side (${pos.side})${p.new_leverage ? `; leverage of the whole position becomes ${p.new_leverage}x` : ''}`;
    if (tab === 'reduce') return `Close ${(p.reduce_fraction * 100).toFixed(0)}% of the position (${fmtNum(pos.size * p.reduce_fraction, 5)} ${pos.coin}) at market, reduce-only; ≈ ${fmtUsd(pos.margin_used * p.reduce_fraction)} margin freed`;
    return `${p.margin_delta > 0 ? 'Add' : 'Remove'} ${fmtUsd(Math.abs(p.margin_delta))} ${p.margin_delta > 0 ? 'to' : 'from'} the isolated margin`;
  };

  return (
    <div className={`mt-2 rounded-lg border p-3 ${dark ? 'border-gray-700 bg-gray-900/40' : 'border-gray-200 bg-gray-50'}`}>
      <div className="flex gap-2 flex-wrap items-center">
        <span className={`text-xs uppercase tracking-wide ${dimCls(dark)}`}>Manage {pos.coin}</span>
        {TABS.length > 1 && TABS.map(([k, l]) => <button key={k} className={btnCls(dark, tab === k)} onClick={() => { setTab(k); setRes(null); setConfirm(false); }}>{l}</button>)}
        <span className={`text-xs ${dimCls(dark)}`}>· margin × leverage = position size; the control buys or sells {pos.coin} to make that true, and the liquidation price moves with it</span>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-3">
        <div className="space-y-2">
          {tab === 'leverage' && (<>
            <div className="flex gap-2 items-center"><span className={`text-sm w-32 ${dimCls(dark)}`}>margin in position $</span>
              <input className={inputCls(dark)} value={tgtMargin} onChange={(e) => setTgtMargin(e.target.value)} />
              <span className={`text-xs ${dimCls(dark)}`}>now {fmtUsd(pos.margin_used)}</span></div>
            <div className="flex gap-2 items-center"><span className={`text-sm w-32 ${dimCls(dark)}`}>leverage</span>
              <input className={inputCls(dark)} value={lev} onChange={(e) => setLev(e.target.value)} />
              {[2, 3, 5, 8, 10, 15, 20].filter((l) => l <= pos.max_leverage).map((l) => <button key={l} className={btnCls(dark, Number(lev) === l)} onClick={() => setLev(String(l))}>{l}x</button>)}
              <span className={`text-xs ${dimCls(dark)}`}>max {pos.max_leverage}x · now {pos.leverage_effective ? `${fmtNum(pos.leverage_effective, 1)}x` : '—'}</span></div>
          </>)}
          {tab === 'add' && (<>
            <div className="flex gap-2 items-center"><span className={`text-sm w-32 ${dimCls(dark)}`}>margin to add $</span><input className={inputCls(dark)} value={addMargin} onChange={(e) => setAddMargin(e.target.value)} placeholder="USDC" /></div>
            <div className="flex gap-2 items-center"><span className={`text-sm w-32 ${dimCls(dark)}`}>at leverage</span><input className={inputCls(dark)} value={addLev} onChange={(e) => setAddLev(e.target.value)} />
              <span className={`text-xs ${dimCls(dark)}`}>one position per coin: a different value re-levers the whole position</span></div>
          </>)}
          {tab === 'reduce' && (
            <div className="flex gap-2 items-center"><span className={`text-sm w-32 ${dimCls(dark)}`}>close %</span>
              <input className={inputCls(dark)} value={reducePct} onChange={(e) => setReducePct(e.target.value)} />
              {[25, 50, 75].map((p) => <button key={p} className={btnCls(dark, Number(reducePct) === p)} onClick={() => setReducePct(String(p))}>{p}%</button>)}</div>
          )}
          {tab === 'margin' && (
            <div className="flex gap-2 items-center"><span className={`text-sm w-32 ${dimCls(dark)}`}>± USDC</span>
              <input className={inputCls(dark)} value={marginDelta} onChange={(e) => setMarginDelta(e.target.value)} placeholder="+500 adds, -500 removes" /></div>
          )}
          <button className={btnCls(dark, true, tab === 'reduce' ? 'red' : 'amber')} disabled={!valid() || busy} onClick={() => { setRes(null); setConfirm(true); }}>{busy ? 'working…' : 'Set leverage'}</button>
        </div>
        <div className="space-y-1">
          <div className={`text-xs uppercase tracking-wide ${dimCls(dark)}`}>After the change (estimate)</div>
          {preview ? (<>
            <KV dark={dark} label="size" value={`${fmtNum(preview.size, 5)} ${pos.coin}${preview.size_delta ? ` (${preview.size_delta > 0 ? 'buy' : 'sell'} ${fmtNum(Math.abs(preview.size_delta), 5)})` : ''}`} />
            <KV dark={dark} label="notional" value={fmtUsd(preview.notional)} />
            <KV dark={dark} label="margin" value={fmtUsd(preview.margin)} />
            <KV dark={dark} label="effective leverage" value={preview.leverage_effective ? `${fmtNum(preview.leverage_effective, 2)}x` : '—'} />
            <KV dark={dark} label="est. liquidation" value={fmtNum(preview.liquidation_px_estimate, 2)} valueClass="font-mono text-rose-400 font-semibold" />
            <KV dark={dark} label="distance to liq" value={preview.liq_distance_pct != null ? `${fmtNum(preview.liq_distance_pct, 1)}%` : '—'} />
            <div className={`text-xs ${dimCls(dark)}`}>now: liq {fmtNum(pos.liquidation_px, 2)} ({fmtNum(pos.liq_distance_pct, 1)}% away), margin {fmtUsd(pos.margin_used)}, {fmtNum(pos.leverage_effective, 1)}x effective</div>
          </>) : <div className={`text-sm ${dimCls(dark)}`}>{err || 'enter a change'}</div>}
        </div>
      </div>
      {confirm && <Confirm dark={dark} tone={tab === 'reduce' ? 'red' : 'amber'} yesLabel="Do it" busy={busy} title={summary()}
        lines={[preview ? `Est. liquidation after: ${fmtNum(preview.liquidation_px_estimate, 2)} (${fmtNum(preview.liq_distance_pct, 1)}% away)` : 'no estimate', `${network} · ${proxyIndex < 0 ? 'direct' : `Proxy ${proxyIndex + 1}`}`]} onYes={run} onNo={() => setConfirm(false)} />}
      {res && (
        <div className={`mt-2 text-sm font-mono ${res.ok ? 'text-emerald-500' : 'text-rose-500'}`}>
          {res.ok ? `done${res.fill?.filled_sz ? ` · filled ${fmtNum(res.fill.filled_sz, 5)} @ ${fmtNum(res.fill.avg_px, 2)}` : ''} · liquidation now ${fmtNum(res.liquidation_px_actual, 2)}${res.position ? ` · ${res.position.leverage}x · margin ${fmtUsd(res.position.margin_used)}` : ' · position closed'}` : (res.error || res.fill?.error || JSON.stringify(res.raw))}
        </div>
      )}
    </div>
  );
}
