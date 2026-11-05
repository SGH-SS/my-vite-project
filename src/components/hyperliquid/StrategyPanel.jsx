import { useEffect, useState } from 'react';
import { hlGet, hlPost, fmtUsd, fmtNum, fmtTime, pnlClass } from './api.js';
import { cardCls, dimCls, strongCls, inputCls, btnCls, Heading, KV, Confirm, ErrorBanner } from './ui.jsx';

/** V2 + Add + Fresh-Liq production runner: status from its heartbeat, arm / disarm / flatten / stop. */
export default function StrategyPanel({ dark, network, proxyIndex }) {
  const [st, setSt] = useState(null);
  const [unitMargin, setUnitMargin] = useState('333');
  const [leverage, setLeverage] = useState('6');
  const [allocation, setAllocation] = useState('1000');
  const [maxLoss, setMaxLoss] = useState('500');
  const [dryRun, setDryRun] = useState(false);     // runner logs signals / adds / exits but places no orders
  const [confirm, setConfirm] = useState(null);     // 'arm' | 'disarm' | 'flatten' | 'stop'
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let alive = true;
    const load = async () => { try { const s = await hlGet('/strategy/status'); if (alive) setSt(s); } catch (e) { if (alive) setErr(e.message); } };
    load(); const id = setInterval(load, 5000); return () => { alive = false; clearInterval(id); };
  }, []);

  const act = async (what) => {
    setBusy(true); setMsg(null); setErr(null);
    try {
      let r;
      if (what === 'arm') r = await hlPost('/strategy/arm', { network, proxy_index: proxyIndex, unit_margin: Number(unitMargin), leverage: Number(leverage), allocation: Number(allocation), max_daily_loss: Number(maxLoss), dry_run: dryRun });
      else if (what === 'disarm') r = await hlPost('/strategy/disarm', { flatten: false });
      else if (what === 'flatten') r = await hlPost('/strategy/flatten', {}, { network, proxy_index: proxyIndex });
      else r = await hlPost('/strategy/stop', {});
      setMsg(r.ok ? `${what}: ${r.action || r.cmd || r.via || 'ok'}` : (r.error || JSON.stringify(r)));
    } catch (e) { setErr(e.message); }
    setBusy(false); setConfirm(null);
  };

  const hb = st?.heartbeat || {}; const pos = hb.position; const running = !!st?.running;
  const dataStale = running && hb.data_lag_s != null && hb.data_lag_s > 90;
  const stateLabel = !st?.installed ? 'runner not installed' : running ? `${hb.armed ? 'ARMED' : 'running · disarmed'}${hb.dry_run ? ' · DRY RUN' : ''}` : 'stopped';
  const stateCls = !st?.installed ? dimCls(dark) : running ? (hb.armed ? 'text-emerald-400' : 'text-amber-400') : 'text-rose-400';
  const notional = Number(unitMargin) * Number(leverage);

  return (
    <div className={cardCls(dark)}>
      <Heading dark={dark} right={<span className={`text-sm font-mono font-semibold ${stateCls}`}>{stateLabel}{running ? ` · pid ${st.pid} · hb ${st.heartbeat_age_s}s` : ''}</span>}>
        Strategy · V2 + Add + Fresh-Liq (SOL)
      </Heading>
      {running && <div className={`text-xs mb-2 ${dimCls(dark)}`}>supervised{st.supervisor_alive ? ` (supervisor pid ${st.supervisor_pid}: restarts the runner within seconds if it dies)` : ' — SUPERVISOR NOT RUNNING'} · backend watchdog relaunches if the heartbeat stops for 2 min · proxy failover on network errors</div>}
      {dataStale && <div className={`text-sm font-semibold mb-2 ${dark ? 'text-rose-300' : 'text-rose-700'}`}>DATA STALE: the 5-second feed is {fmtNum(hb.data_lag_s, 0)} s behind — the runner will not enter on stale candles; check the collector / BXT builder windows.</div>}
      {st?.stopped_by_user && !running && <div className={`text-xs mb-2 ${dimCls(dark)}`}>stopped by you — the watchdog will not relaunch until you Arm again</div>}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="space-y-2">
          <div className={`text-xs uppercase tracking-wide ${dimCls(dark)}`}>Sizing (per unit)</div>
          <div className="flex gap-2 items-center"><span className={`text-sm w-28 ${dimCls(dark)}`}>unit margin $</span><input className={inputCls(dark)} value={unitMargin} onChange={(e) => setUnitMargin(e.target.value)} disabled={running} /></div>
          <div className="flex gap-2 items-center"><span className={`text-sm w-28 ${dimCls(dark)}`}>leverage</span><input className={inputCls(dark)} value={leverage} onChange={(e) => setLeverage(e.target.value)} disabled={running} /></div>
          <div className="flex gap-2 items-center"><span className={`text-sm w-28 ${dimCls(dark)}`}>allocation $</span><input className={inputCls(dark)} value={allocation} onChange={(e) => setAllocation(e.target.value)} disabled={running} /></div>
          <div className="flex gap-2 items-center"><span className={`text-sm w-28 ${dimCls(dark)}`}>max daily loss $</span><input className={inputCls(dark)} value={maxLoss} onChange={(e) => setMaxLoss(e.target.value)} disabled={running} /></div>
          <label className={`flex items-center gap-2 text-sm ${dimCls(dark)}`}><input type="checkbox" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} disabled={running} /> dry run (signals and decisions only, no orders)</label>
          <div className={`text-xs ${dimCls(dark)}`}>unit notional {fmtUsd(notional)} · with the add {fmtUsd(notional * 2)} ({fmtUsd(Number(unitMargin) * 2)} margin) · network <b>{network}</b> · route {proxyIndex < 0 ? 'direct' : `Proxy ${proxyIndex + 1}`}</div>
          <div className="flex gap-2 flex-wrap pt-1">
            <button className={btnCls(dark, true, 'green')} disabled={busy || !st?.installed || (running && hb.armed)} onClick={() => setConfirm('arm')}>Arm</button>
            <button className={btnCls(dark, true, 'amber')} disabled={busy || !running || !hb.armed} onClick={() => setConfirm('disarm')}>Disarm (keep position)</button>
            <button className={btnCls(dark, true, 'red')} disabled={busy} onClick={() => setConfirm('flatten')}>Flatten SOL now</button>
            <button className={btnCls(dark)} disabled={busy || !running} onClick={() => setConfirm('stop')}>Stop process</button>
          </div>
        </div>
        <div className="space-y-1">
          <div className={`text-xs uppercase tracking-wide ${dimCls(dark)}`}>Runner</div>
          <KV dark={dark} label="network / route" value={hb.network ? `${hb.network} / ${hb.proxy_index >= 0 ? `Proxy ${hb.proxy_index + 1}` : 'direct'}` : '—'} />
          <KV dark={dark} label="last candle" value={hb.last_candle_t ? fmtTime(hb.last_candle_t) : '—'} />
          <KV dark={dark} label="data lag" value={hb.data_lag_s != null ? `${fmtNum(hb.data_lag_s, 0)} s` : '—'} valueClass={`font-mono ${dataStale ? 'text-rose-400' : strongCls(dark)}`} />
          <KV dark={dark} label="last signal" value={hb.last_signal?.t ? `${fmtTime(hb.last_signal.t)} (${fmtNum(hb.last_signal.ext_z, 2)} sd, OI ${fmtNum(hb.last_signal.doi, 0)} bp)` : 'none yet'} />
          <KV dark={dark} label="equity" value={fmtUsd(hb.equity)} />
          <KV dark={dark} label="today P&L" value={fmtUsd(hb.today_pnl)} valueClass={`font-mono ${pnlClass(hb.today_pnl)}`} />
          <KV dark={dark} label="trades today" value={hb.trades_today ?? '—'} />
          {hb.last_error && <div className="text-xs text-rose-400 font-mono break-all">last error: {hb.last_error}</div>}
        </div>
        <div className="space-y-1">
          <div className={`text-xs uppercase tracking-wide ${dimCls(dark)}`}>Open position (runner's view)</div>
          {pos ? (<>
            <KV dark={dark} label="entered" value={fmtTime(pos.entry_t)} />
            <KV dark={dark} label="units" value={`${pos.units?.length ?? 1} (${fmtNum(pos.sz_total, 2)} SOL)`} />
            <KV dark={dark} label="avg entry" value={fmtNum(pos.avg_px, 4)} />
            <KV dark={dark} label="exit due" value={`${fmtTime(pos.exit_due_t)}${pos.extended ? ' (extended)' : ''}`} />
            <KV dark={dark} label="add" value={pos.units?.length > 1 ? `yes @ ${fmtNum(pos.units[1].px, 4)}` : 'not yet'} />
            <KV dark={dark} label="uPnL" value={fmtUsd(hb.position_upnl)} valueClass={`font-mono ${pnlClass(hb.position_upnl)}`} />
          </>) : <div className={`text-sm ${dimCls(dark)}`}>flat</div>}
          {st?.recent_trades?.length > 0 && (<>
            <div className={`text-xs uppercase tracking-wide pt-2 ${dimCls(dark)}`}>Recent trades</div>
            {st.recent_trades.slice(-5).reverse().map((t, i) => (
              <div key={i} className={`text-xs font-mono ${strongCls(dark)}`}>{fmtTime(t.entry_t).slice(5, 16)} → {fmtTime(t.exit_t).slice(11, 16)} · {t.units} u · <span className={pnlClass(t.pnl_usd)}>{fmtUsd(t.pnl_usd)}</span> ({fmtNum(t.net_bps, 1)} bps){t.extended ? ' · ext' : ''}</div>
            ))}
          </>)}
        </div>
      </div>
      {confirm === 'arm' && <Confirm dark={dark} tone="green" yesLabel="Arm" busy={busy} title={`Arm the runner on ${network}?`}
        lines={[dryRun ? 'DRY RUN: the window logs signals, adds and exits but sends no orders.' : 'LIVE orders.', `Opens its own PowerShell window and trades SOL: one unit of ${fmtUsd(notional)} notional (${fmtUsd(Number(unitMargin))} margin at ${leverage}x isolated) per signal, one add, 60/120 m holds.`, `Allocation ${fmtUsd(Number(allocation))}; disarms itself if today's realised loss exceeds ${fmtUsd(Number(maxLoss))}.`, `Route: ${proxyIndex < 0 ? 'direct' : `Proxy ${proxyIndex + 1}`}.`]} onYes={() => act('arm')} onNo={() => setConfirm(null)} />}
      {confirm === 'disarm' && <Confirm dark={dark} tone="amber" yesLabel="Disarm" busy={busy} title="Stop taking new signals?" lines={['An open position keeps being managed to its planned exit.']} onYes={() => act('disarm')} onNo={() => setConfirm(null)} />}
      {confirm === 'flatten' && <Confirm dark={dark} tone="red" yesLabel="Flatten" busy={busy} title={`Close the SOL position at market on ${network} now?`} lines={['Reduce-only IOC. The runner disarms if it is running.']} onYes={() => act('flatten')} onNo={() => setConfirm(null)} />}
      {confirm === 'stop' && <Confirm dark={dark} tone="red" yesLabel="Stop" busy={busy} title="Exit the runner process?" lines={['Any open position is left as it is and shown in Positions above. Arm again to resume managing it.']} onYes={() => act('stop')} onNo={() => setConfirm(null)} />}
      {msg && <div className={`mt-2 text-xs font-mono ${strongCls(dark)}`}>{msg}</div>}
      <ErrorBanner dark={dark} error={err} />
    </div>
  );
}
