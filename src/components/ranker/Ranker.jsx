/**
 * Ranker — dedicated dashboard for the 10-slot alt long/short strategy (sol-perp/ranker; research map.md §19.6).
 * Controls (arm / disarm / flatten all / stop / retrain), runner + collector + model health, the live book, this
 * hour's candidates, realised stats and the equity curve, recent trades.  Backend: /api/ranker.
 */
import { useEffect, useState } from 'react';
import { rGet, rPost, fmtUsd, fmtNum, fmtTime, pnlClass, readPref, writePref } from './api.js';
import { cardCls, dimCls, strongCls, inputCls, btnCls, Heading, KV, Confirm, ErrorBanner } from '../hyperliquid/ui.jsx';

const ago = (s) => (s == null ? '—' : s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);

export default function Ranker({ isDarkMode: dark }) {
  const [st, setSt] = useState(null);
  const [stats, setStats] = useState(null);
  const [err, setErr] = useState(null);
  const [msg, setMsg] = useState(null);
  const [busy, setBusy] = useState(false);
  const [confirm, setConfirm] = useState(null);
  const [margin, setMargin] = useState(() => readPref('ranker.margin.v2', '300'));
  const [leverage, setLeverage] = useState(() => readPref('ranker.leverage.v2', '2'));
  const [slots, setSlots] = useState(() => readPref('ranker.slots', '10'));
  const [proxyIndex, setProxyIndex] = useState(() => readPref('ranker.proxy', 1));
  const [dryRun, setDryRun] = useState(false);
  const [exclude, setExclude] = useState('');

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try { const s = await rGet('/status'); if (alive) { setSt(s); setErr(null); if (s.config && exclude === '' ) setExclude((s.config.exclude_coins || []).join(', ')); } } catch (e) { if (alive) setErr(e.message); }
      try { const x = await rGet('/stats'); if (alive) setStats(x); } catch { /* stats are optional */ }
    };
    load(); const id = setInterval(load, 5000); return () => { alive = false; clearInterval(id); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const act = async (what) => {
    setBusy(true); setMsg(null); setErr(null);
    try {
      let r;
      if (what === 'arm') { writePref('ranker.margin.v2', margin); writePref('ranker.leverage.v2', leverage); writePref('ranker.slots', slots); writePref('ranker.proxy', proxyIndex);
        r = await rPost('/arm', { network: 'mainnet', proxy_index: Number(proxyIndex), margin: Number(margin), leverage: Number(leverage), slots: Number(slots), cross: true, dry_run: dryRun }); }
      else if (what === 'disarm') r = await rPost('/disarm', {});
      else if (what === 'flatten') r = await rPost('/flatten', {}, { network: 'mainnet', proxy_index: Number(proxyIndex) });
      else if (what === 'stop') r = await rPost('/stop', {});
      else if (what === 'retrain') r = await rPost('/retrain', {});
      else if (what === 'collector_start') r = await rPost('/collector/start', {});
      else if (what === 'collector_stop') r = await rPost('/collector/stop', {});
      else if (what === 'config') r = await rPost('/config', { exclude_coins: exclude.split(',').map((s) => s.trim()).filter(Boolean) });
      setMsg(r.ok === false ? (r.error || JSON.stringify(r)) : `${what}: ${r.action || r.cmd || r.via || 'ok'}`);
    } catch (e) { setErr(e.message); }
    setBusy(false); setConfirm(null);
  };

  const hb = st?.heartbeat || {}; const running = !!st?.running; const col = st?.collector || {}; const model = st?.model || hb.model;
  const stateLabel = !st?.installed ? 'runner not installed' : running ? `${hb.armed ? 'ARMED' : 'running · disarmed'}${hb.dry_run ? ' · DRY RUN' : ''}` : 'stopped';
  const stateCls = !st?.installed ? dimCls(dark) : running ? (hb.armed ? 'text-emerald-400' : 'text-amber-400') : 'text-rose-400';
  const notional = Number(margin) * Number(leverage); const book = hb.book || []; const w = stats?.windows || {};
  const colStale = col.ts && col.age_s > 900;
  const upnl = book.reduce((a, b) => a + (b.upnl || 0), 0);

  return (
    <div className="space-y-4">
      {/* ── controls + runner + feeds ─────────────────────────────────────── */}
      <div className={cardCls(dark)}>
        <Heading dark={dark} right={<span className={`text-sm font-mono font-semibold ${stateCls}`}>{stateLabel}{running ? ` · pid ${st.pid} · hb ${st.heartbeat_age_s}s` : ''}</span>}>
          Strategy · Ranker — 10-slot alt long/short (HL premium + 1h/4h/12h blend + squeeze guard)
        </Heading>
        {running && <div className={`text-xs mb-2 ${dimCls(dark)}`}>supervised{st.supervisor_alive ? ` (supervisor pid ${st.supervisor_pid})` : ' — SUPERVISOR NOT RUNNING'} · backend watchdog relaunches if the heartbeat stops for 2 min · this strategy has priority over V2 (V2 is told to flatten SOL when the ranker wants it)</div>}
        {colStale && <div className={`text-sm font-semibold mb-2 ${dark ? 'text-rose-300' : 'text-rose-700'}`}>COLLECTOR STALE: last update {ago(col.age_s)} ago — the runner skips hours it has no data for. Check the RANKER COLLECTOR window.</div>}
        {!col.alive && !col.starting && <div className={`text-sm font-semibold mb-2 ${dark ? 'text-rose-300' : 'text-rose-700'}`}>COLLECTOR NOT RUNNING — the runner cannot score new hours. Click Start collector (opens its own window).</div>}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="space-y-2">
            <div className={`text-xs uppercase tracking-wide ${dimCls(dark)}`}>Sizing (per slot, cross margin)</div>
            <div className="flex gap-2 items-center"><span className={`text-sm w-28 ${dimCls(dark)}`}>margin $</span><input className={inputCls(dark)} value={margin} onChange={(e) => setMargin(e.target.value)} disabled={running} /></div>
            <div className="flex gap-2 items-center"><span className={`text-sm w-28 ${dimCls(dark)}`}>leverage</span><input className={inputCls(dark)} value={leverage} onChange={(e) => setLeverage(e.target.value)} disabled={running} /></div>
            <div className="flex gap-2 items-center"><span className={`text-sm w-28 ${dimCls(dark)}`}>slots</span><input className={inputCls(dark)} value={slots} onChange={(e) => setSlots(e.target.value)} disabled={running} /></div>
            <div className="flex gap-2 items-center"><span className={`text-sm w-28 ${dimCls(dark)}`}>HL route</span>
              <select className={inputCls(dark)} value={proxyIndex} onChange={(e) => setProxyIndex(Number(e.target.value))} disabled={running}>
                {[1, 2, 3].map((i) => <option key={i} value={i}>Proxy {i + 1}</option>)}<option value={-1}>direct</option>
              </select></div>
            <label className={`flex items-center gap-2 text-sm ${dimCls(dark)}`}><input type="checkbox" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} disabled={running} /> dry run (decisions logged, no orders)</label>
            <div className={`text-xs ${dimCls(dark)}`}>{fmtUsd(notional, 0)} notional per slot · {fmtUsd(notional * Number(slots), 0)} book · {fmtUsd(Number(margin) * Number(slots), 0)} margin when full · cross margin: the whole account backs every slot (isolated 5–6x on alts was liquidated about once a day)</div>
            <div className="flex gap-2 flex-wrap pt-1">
              <button className={btnCls(dark, true, 'green')} disabled={busy || !st?.installed || (running && hb.armed)} onClick={() => setConfirm('arm')}>Arm</button>
              <button className={btnCls(dark, true, 'amber')} disabled={busy || !running || !hb.armed} onClick={() => setConfirm('disarm')}>Disarm (keep book)</button>
              <button className={btnCls(dark, true, 'red')} disabled={busy} onClick={() => setConfirm('flatten')}>Flatten all</button>
              <button className={btnCls(dark)} disabled={busy || !running} onClick={() => setConfirm('stop')}>Stop process</button>
              <button className={btnCls(dark, true, 'blue')} disabled={busy} onClick={() => setConfirm('retrain')}>Retrain model</button>
            </div>
            <div className="flex gap-2 items-center pt-1"><span className={`text-sm w-28 ${dimCls(dark)}`}>exclude coins</span><input className={inputCls(dark)} value={exclude} onChange={(e) => setExclude(e.target.value)} placeholder="e.g. SOL, SPX" /><button className={btnCls(dark)} disabled={busy} onClick={() => act('config')}>save</button></div>
          </div>
          <div className="space-y-1">
            <div className={`text-xs uppercase tracking-wide ${dimCls(dark)}`}>Runner</div>
            <KV dark={dark} label="route" value={hb.network ? `${hb.network} / ${hb.proxy_index >= 0 ? `Proxy ${hb.proxy_index + 1}` : 'direct'}` : '—'} />
            <KV dark={dark} label="sizing" value={hb.margin ? `${hb.slots} × ${fmtUsd(hb.margin, 0)} @ ${hb.leverage}x` : '—'} />
            <KV dark={dark} label="last cycle" value={hb.cycle?.hour ? `${hb.cycle.hour.slice(0, 16)} · ran +${hb.cycle.delay_s}s · ${hb.cycle.seconds}s · ${hb.cycle.universe} coins` : hb.last_cycle_hour ? hb.last_cycle_hour.slice(0, 16) : '—'} />
            <KV dark={dark} label="last decisions" value={hb.cycle?.hour ? `${(hb.cycle.exits || []).length} exits · ${(hb.cycle.entries || []).length} entries` : '—'} />
            <KV dark={dark} label="cycles run" value={hb.cycles ?? '—'} />
            <KV dark={dark} label="equity / available" value={`${fmtUsd(hb.equity)} / ${fmtUsd(hb.available)}`} />
            <KV dark={dark} label="realised (all time)" value={w.all ? `${fmtUsd(w.all.pnl_usd)} · ${w.all.trades} trades · ${(w.all.hit * 100).toFixed(0)}% hit` : '$0.00 · 0 trades'} valueClass={`font-mono font-semibold ${pnlClass(w.all?.pnl_usd ?? 0)}`} />
            <KV dark={dark} label="today realised" value={`${fmtUsd(hb.today_pnl)} · ${hb.trades_today ?? 0} trades`} valueClass={`font-mono ${pnlClass(hb.today_pnl)}`} />
            <KV dark={dark} label="open uPnL" value={fmtUsd(upnl)} valueClass={`font-mono ${pnlClass(upnl)}`} />
            {hb.exclude_coins?.length > 0 && <KV dark={dark} label="excluded" value={hb.exclude_coins.join(', ')} />}
            {hb.last_error && <div className="text-xs text-rose-400 font-mono break-all">last error: {hb.last_error}</div>}
            {st?.stopped_by_user && !running && <div className={`text-xs ${dimCls(dark)}`}>stopped by you — the watchdog will not relaunch until you Arm again</div>}
          </div>
          <div className="space-y-1">
            <div className={`text-xs uppercase tracking-wide ${dimCls(dark)}`}>Data feed & model</div>
            <KV dark={dark} label="collector" value={col.starting ? `starting · first cycle running (~3 min) · pid ${col.pid}` : col.alive ? `${col.ok ? 'ok' : 'ERROR'} · ${ago(col.age_s)} ago · ${col.cycle_s}s/cycle · pid ${col.pid}` : col.ts ? `DOWN (last ${ago(col.age_s)} ago)` : 'not running'} valueClass={`font-mono ${col.starting ? 'text-amber-400' : col.alive && !colStale ? strongCls(dark) : 'text-rose-400'}`} />
            <div className="flex gap-2 pt-1">
              <button className={btnCls(dark, !col.alive, 'green')} disabled={busy || col.alive} onClick={() => act('collector_start')}>Start collector</button>
              <button className={btnCls(dark)} disabled={busy || !col.alive} onClick={() => act('collector_stop')}>Stop collector</button>
            </div>
            <KV dark={dark} label="panel through" value={col.panel_through ? String(col.panel_through).slice(0, 16) : '—'} />
            {col.error && <div className="text-xs text-rose-400 font-mono break-all">collector: {col.error}</div>}
            <KV dark={dark} label="model" value={model ? model.file : 'NONE — retrain'} valueClass={`font-mono ${model ? strongCls(dark) : 'text-rose-400'}`} />
            <KV dark={dark} label="trained through" value={model ? String(model.trained_through).slice(0, 16) : '—'} />
            <KV dark={dark} label="holdout IC (90 d)" value={model ? fmtNum(model.holdout_ic, 3) : '—'} />
            <KV dark={dark} label="training rows" value={model ? fmtNum(model.rows, 0) : '—'} />
            {model?.importance_top && <div className={`text-xs ${dimCls(dark)}`}>top features: {Object.entries(model.importance_top).slice(0, 6).map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(' · ')}</div>}
          </div>
        </div>
        {confirm === 'arm' && <Confirm dark={dark} tone="green" yesLabel="Arm" busy={busy} title="Arm the ranker on mainnet?"
          lines={[dryRun ? 'DRY RUN: decisions are logged, no orders are sent.' : 'LIVE orders.', `Opens its own PowerShell window. Every closed hour: up to ${slots} isolated positions of ${fmtUsd(notional, 0)} notional (${fmtUsd(Number(margin), 0)} margin at ${leverage}x), long and short, post-only entries crossed after 15 min.`, 'No daily-loss limit. This strategy has priority over V2 on SOL.', `Route: Proxy ${Number(proxyIndex) + 1}.`]} onYes={() => act('arm')} onNo={() => setConfirm(null)} />}
        {confirm === 'disarm' && <Confirm dark={dark} tone="amber" yesLabel="Disarm" busy={busy} title="Stop opening new positions?" lines={['Open positions keep being managed to their signal exits.']} onYes={() => act('disarm')} onNo={() => setConfirm(null)} />}
        {confirm === 'flatten' && <Confirm dark={dark} tone="red" yesLabel="Flatten all" busy={busy} title="Close every ranker position at market now?" lines={['Reduce-only IOC on each coin in the book; resting entries are cancelled; the runner disarms.']} onYes={() => act('flatten')} onNo={() => setConfirm(null)} />}
        {confirm === 'stop' && <Confirm dark={dark} tone="red" yesLabel="Stop" busy={busy} title="Exit the runner process?" lines={['Positions are left open on the exchange and re-adopted when you arm again.']} onYes={() => act('stop')} onNo={() => setConfirm(null)} />}
        {confirm === 'retrain' && <Confirm dark={dark} tone="blue" yesLabel="Retrain" busy={busy} title="Retrain the model on all data to now?" lines={['Runs in its own window (10-20 min). The runner switches to the new model at its next hourly cycle.']} onYes={() => act('retrain')} onNo={() => setConfirm(null)} />}
        {msg && <div className={`mt-2 text-xs font-mono ${strongCls(dark)}`}>{msg}</div>}
        <ErrorBanner dark={dark} error={err} />
      </div>

      {/* ── book + candidates ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className={`${cardCls(dark)} lg:col-span-2`}>
          <Heading dark={dark} right={<span className={`text-xs ${dimCls(dark)}`}>{book.length} / {hb.slots ?? slots} slots</span>}>Book (runner's view, uPnL from the exchange)</Heading>
          {book.length === 0 ? <div className={`text-sm ${dimCls(dark)}`}>flat</div> : (
            <table className="w-full text-xs font-mono">
              <thead><tr className={dimCls(dark)}><th className="text-left">coin</th><th>side</th><th>status</th><th>entry</th><th>size</th><th>age</th><th>pct@entry</th><th>uPnL</th><th>liq px</th></tr></thead>
              <tbody>{book.map((b) => (
                <tr key={b.coin} className={strongCls(dark)}>
                  <td className="text-left font-semibold">{b.coin}</td><td className={`text-center ${b.side > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{b.side > 0 ? 'LONG' : 'SHORT'}</td>
                  <td className="text-center">{b.status}{b.entry_kind ? ` (${b.entry_kind})` : ''}</td><td className="text-right">{b.entry_px ? fmtNum(b.entry_px, 6) : `resting @ ${fmtNum(b.limit_px, 6)}`}</td>
                  <td className="text-right">{fmtNum(b.sz, 4)}</td><td className="text-right">{b.age_h != null ? `${b.age_h} h` : '—'}</td><td className="text-right">{b.pct != null ? fmtNum(b.pct, 2) : '—'}</td>
                  <td className={`text-right ${pnlClass(b.upnl)}`}>{fmtUsd(b.upnl)}</td><td className="text-right">{b.liq_px ? fmtNum(b.liq_px, 6) : '—'}</td>
                </tr>))}</tbody>
            </table>)}
        </div>
        <div className={cardCls(dark)}>
          <Heading dark={dark}>This hour's candidates (pct ≥ .8 / ≤ .2)</Heading>
          {(hb.candidates || []).length === 0 ? <div className={`text-sm ${dimCls(dark)}`}>no cycle yet</div> : (
            <div className="grid grid-cols-2 gap-x-3 text-xs font-mono">
              {(hb.candidates || []).map((c) => (
                <div key={c.coin} className={`flex justify-between ${c.held ? strongCls(dark) : dimCls(dark)}`}>
                  <span>{c.coin}{c.held ? ' ●' : ''}{c.squeeze ? ' ⚡' : ''}</span><span className={c.pct >= 0.8 ? 'text-emerald-400' : 'text-rose-400'}>{fmtNum(c.pct, 2)}</span>
                </div>))}
            </div>)}
          <div className={`text-xs mt-2 ${dimCls(dark)}`}>● held · ⚡ squeeze flag (no short) · entries need ≥ .90 / ≤ .10 and a free slot</div>
        </div>
      </div>

      {/* ── stats + equity + trades ──────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className={cardCls(dark)}>
          <Heading dark={dark}>Realised stats</Heading>
          <table className="w-full text-xs font-mono">
            <thead><tr className={dimCls(dark)}><th className="text-left">window</th><th>trades</th><th>P&L</th><th>hit</th><th>avg bps</th><th>liq</th><th>L / S</th></tr></thead>
            <tbody>{['today', '7d', '30d', 'all'].map((k) => w[k] ? (
              <tr key={k} className={strongCls(dark)}><td className="text-left">{k}</td><td className="text-center">{w[k].trades}</td><td className={`text-right ${pnlClass(w[k].pnl_usd)}`}>{fmtUsd(w[k].pnl_usd)}</td>
                <td className="text-right">{(w[k].hit * 100).toFixed(0)}%</td><td className="text-right">{fmtNum(w[k].mean_bps, 1)}</td><td className="text-center">{w[k].liquidations}</td>
                <td className="text-right">{fmtUsd(w[k].long_pnl, 0)} / {fmtUsd(w[k].short_pnl, 0)}</td></tr>) : null)}</tbody>
          </table>
          {stats?.fills?.n > 0 && <div className={`text-xs mt-2 ${dimCls(dark)}`}>entries: {stats.fills.n} fills · maker {(stats.fills.maker_share * 100).toFixed(0)}% · realised cost vs decision mid {fmtNum(stats.fills.mean_cost_bps, 1)} bps (maker {fmtNum(stats.fills.maker_cost_bps, 1)} / crossed {fmtNum(stats.fills.taker_cost_bps, 1)}) · mean wait {fmtNum(stats.fills.mean_wait_s, 0)} s</div>}
        </div>
        <div className={cardCls(dark)}>
          <Heading dark={dark}>Equity (daily realised, cumulative)</Heading>
          <EquityChart dark={dark} rows={stats?.equity || []} />
        </div>
        <div className={cardCls(dark)}>
          <Heading dark={dark}>Recent trades</Heading>
          <div className="max-h-72 overflow-auto space-y-0.5">
            {(st?.recent_trades || []).slice().reverse().map((t, i) => (
              <div key={i} className={`text-xs font-mono ${strongCls(dark)}`}>{fmtTime(t.exit_t).slice(5, 16)} · <b>{t.coin}</b> {t.side > 0 ? 'L' : 'S'} · {fmtNum(t.hours, 1)} h · <span className={pnlClass(t.pnl_usd)}>{fmtUsd(t.pnl_usd)}</span> ({fmtNum(t.bps, 0)} bps) · {t.entry_kind}→{t.exit_kind}{t.liquidated ? ' · LIQ' : ''}{t.reason && t.reason !== 'signal' ? ` · ${t.reason}` : ''}</div>))}
            {(st?.recent_trades || []).length === 0 && <div className={`text-sm ${dimCls(dark)}`}>none yet</div>}
          </div>
        </div>
      </div>
    </div>
  );
}

function EquityChart({ dark, rows }) {
  if (!rows.length) return <div className={`text-sm ${dimCls(dark)}`}>no closed trades yet</div>;
  const W = 420, H = 140, pad = 24;
  const ys = rows.map((r) => r.cum); const lo = Math.min(0, ...ys), hi = Math.max(0, ...ys);
  const x = (i) => pad + (i * (W - 2 * pad)) / Math.max(1, rows.length - 1);
  const y = (v) => H - pad - ((v - lo) * (H - 2 * pad)) / Math.max(1e-9, hi - lo);
  const d = rows.map((r, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(r.cum).toFixed(1)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <line x1={pad} x2={W - pad} y1={y(0)} y2={y(0)} stroke={dark ? '#4b5563' : '#d1d5db'} strokeDasharray="3 3" />
      <path d={d} fill="none" stroke={rows[rows.length - 1].cum >= 0 ? '#10b981' : '#f43f5e'} strokeWidth="2" />
      <text x={pad} y={12} fontSize="10" fill={dark ? '#9ca3af' : '#6b7280'}>{rows[0].date}</text>
      <text x={W - pad} y={12} fontSize="10" textAnchor="end" fill={dark ? '#9ca3af' : '#6b7280'}>{rows[rows.length - 1].date} · {rows[rows.length - 1].cum.toFixed(0)} USD</text>
    </svg>
  );
}
