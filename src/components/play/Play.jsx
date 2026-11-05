/**
 * BTC Play — one-button price triggers.  Arm: a runner (sol-perp/play) watches the BTC mid every second and, when the
 * price crosses DOWN through a level, puts all available capital into a BTC long at that level's leverage (cross).
 * Each trigger fires once; triggers already below the price when arming are skipped.  Backend: /api/play.
 */
import { useEffect, useState } from 'react';
import { pGet, pPost, fmtUsd, fmtNum, fmtTime, pnlClass, readPref, writePref } from './api.js';
import { cardCls, dimCls, strongCls, inputCls, btnCls, Heading, KV, Confirm, ErrorBanner } from '../hyperliquid/ui.jsx';

const DEFAULT_TRIGGERS = [{ level: '84292', leverage: '7' }, { level: '83200', leverage: '15' }];
const ago = (s) => (s == null ? '—' : s < 90 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`);

export default function Play({ isDarkMode: dark }) {
  const [st, setSt] = useState(null);
  const [px, setPx] = useState(null);
  const [err, setErr] = useState(null);
  const [msg, setMsg] = useState(null);
  const [busy, setBusy] = useState(false);
  const [confirm, setConfirm] = useState(null);
  const [triggers, setTriggers] = useState(() => readPref('play.triggers.v1', DEFAULT_TRIGGERS));
  const [proxyIndex, setProxyIndex] = useState(() => readPref('play.proxy', 1));
  const [dryRun, setDryRun] = useState(false);

  const refresh = async () => {
    try { setSt(await pGet('/status')); setErr(null); } catch (e) { setErr(e.message); }
  };
  const refreshPx = async () => {
    try { setPx(await pGet('/price', { proxy_index: Number(proxyIndex) })); } catch (e) { /* price card is best-effort */ }
  };
  useEffect(() => { refresh(); const id = setInterval(refresh, 2000); return () => clearInterval(id); }, []);
  useEffect(() => { refreshPx(); const id = setInterval(refreshPx, 5000); return () => clearInterval(id); }, [proxyIndex]);

  const running = !!st?.running; const hb = st?.heartbeat || {};
  const livePx = running && hb.price ? hb.price : px?.price;
  const avail = px?.available; const acct = px?.account_value;
  const rows = triggers.map((t) => ({ level: Number(t.level), leverage: Number(t.leverage) })).filter((t) => t.level > 0 && t.leverage >= 1);
  const setT = (i, k, v) => setTriggers((arr) => arr.map((t, j) => (j === i ? { ...t, [k]: v } : t)));

  const act = async (what) => {
    setBusy(true); setMsg(null);
    try {
      let r;
      if (what === 'arm') {
        writePref('play.triggers.v1', triggers); writePref('play.proxy', proxyIndex);
        r = await pPost('/arm', { network: 'mainnet', proxy_index: Number(proxyIndex), dry_run: dryRun, triggers: rows });
      } else if (what === 'stop') r = await pPost('/stop');
      setMsg(r?.ok === false ? `Error: ${r.error}` : JSON.stringify(r));
      setTimeout(refresh, 1500);
    } catch (e) { setMsg(`Error: ${e.message}`); }
    setBusy(false); setConfirm(null);
  };

  const liqPct = (lev) => 100 / lev - 1.25;                         // BTC maintenance margin 1.25 % on HL
  return (
    <div className="space-y-4">
      <ErrorBanner dark={dark} error={err} />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className={`${cardCls(dark)} lg:col-span-2`}>
          <Heading dark={dark} right={<span className={`text-xs ${dimCls(dark)}`}>BTC {livePx ? fmtUsd(livePx, 0) : '—'} · available {avail != null ? fmtUsd(avail, 0) : '—'} of {acct != null ? fmtUsd(acct, 0) : '—'}</span>}>BTC Play — long on a touch</Heading>
          <div className="space-y-2 mt-2">
            {triggers.map((t, i) => {
              const lev = Number(t.leverage) || 0; const lvl = Number(t.level) || 0; const notional = avail != null && lev ? avail * 0.98 * lev : null;
              const dist = livePx && lvl ? (livePx / lvl - 1) * 100 : null;
              return (
                <div key={i} className="grid grid-cols-12 gap-2 items-center">
                  <span className={`col-span-2 text-sm ${dimCls(dark)}`}>trigger {i + 1}</span>
                  <div className="col-span-3 flex items-center gap-1"><span className={`text-xs ${dimCls(dark)}`}>BTC ≤</span><input className={inputCls(dark)} value={t.level} onChange={(e) => setT(i, 'level', e.target.value)} disabled={running} /></div>
                  <div className="col-span-2 flex items-center gap-1"><input className={inputCls(dark)} value={t.leverage} onChange={(e) => setT(i, 'leverage', e.target.value)} disabled={running} /><span className={`text-xs ${dimCls(dark)}`}>x</span></div>
                  <div className={`col-span-5 text-xs ${dimCls(dark)}`}>{dist != null ? `${dist >= 0 ? '+' : ''}${dist.toFixed(2)}% away` : '—'} · ≈ {notional ? fmtUsd(notional, 0) : '—'} long · liquidation ≈ {lev ? `${liqPct(lev).toFixed(1)}% below entry (≈ ${fmtUsd(lvl * (1 - liqPct(lev) / 100), 0)})` : '—'}</div>
                </div>
              );
            })}
            <div className="flex gap-3 items-center pt-1">
              <button className={`text-xs ${dimCls(dark)} underline`} onClick={() => setTriggers((a) => [...a, { level: '', leverage: '5' }])} disabled={running}>+ trigger</button>
              <button className={`text-xs ${dimCls(dark)} underline`} onClick={() => setTriggers(DEFAULT_TRIGGERS)} disabled={running}>reset to 84,292 / 7x and 83,200 / 15x</button>
              <span className={`text-sm w-20 ${dimCls(dark)}`}>HL route</span>
              <select className={inputCls(dark)} style={{ width: 110 }} value={proxyIndex} onChange={(e) => setProxyIndex(Number(e.target.value))} disabled={running}>
                {[1, 2, 3].map((i) => <option key={i} value={i}>Proxy {i + 1}</option>)}<option value={-1}>direct</option>
              </select>
              <label className={`flex items-center gap-2 text-sm ${dimCls(dark)}`}><input type="checkbox" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} disabled={running} /> dry run</label>
            </div>
            <div className="flex gap-2 pt-2">
              <button className={`${btnCls(dark, true, 'green')} text-lg px-6 py-3`} disabled={busy || running || rows.length === 0} onClick={() => setConfirm('arm')}>ARM</button>
              <button className={btnCls(dark, true, 'red')} disabled={busy || !running} onClick={() => setConfirm('stop')}>Disarm / stop</button>
            </div>
            <div className={`text-xs ${dimCls(dark)}`}>All available capital goes into the long at the trigger's leverage, cross margin, IOC through the book. A lower trigger after a higher one raises the leverage and deploys the freed margin. Positions are never closed by the play: close them from the Hyperliquid tab or the exchange.</div>
            {msg && <div className={`text-xs ${msg.startsWith('Error') ? 'text-rose-500' : dimCls(dark)}`}>{msg}</div>}
          </div>
        </div>
        <div className={cardCls(dark)}>
          <Heading dark={dark}>Runner</Heading>
          <KV dark={dark} label="process" value={running ? `running (pid ${st.pid})${st.stale ? ' · STALE' : ''}` : 'stopped'} valueClass={running ? (st.stale ? 'text-amber-500' : 'text-emerald-500') : ''} />
          <KV dark={dark} label="armed" value={running ? (hb.armed ? (hb.all_fired ? 'all triggers fired' : 'yes') : 'no') : '—'} />
          <KV dark={dark} label="mode" value={hb.dry_run ? 'DRY RUN' : running ? 'LIVE' : '—'} valueClass={hb.dry_run ? 'text-amber-500' : ''} />
          <KV dark={dark} label="heartbeat" value={ago(st?.heartbeat_age_s)} />
          <KV dark={dark} label="BTC mid" value={hb.price ? fmtUsd(hb.price, 1) : '—'} />
          <KV dark={dark} label="account" value={hb.account?.value != null ? `${fmtUsd(hb.account.value, 0)} · avail ${fmtUsd(hb.account.available, 0)}` : '—'} />
          {hb.position && <KV dark={dark} label="BTC position" value={`${fmtNum(hb.position.size, 5)} @ ${fmtUsd(hb.position.entry_px, 0)} · liq ${fmtUsd(hb.position.liquidation_px, 0)}`} />}
          {hb.position && <KV dark={dark} label="uPnL" value={fmtUsd(hb.position.unrealized_pnl)} valueClass={pnlClass(hb.position.unrealized_pnl)} />}
          {hb.last_error && <KV dark={dark} label="last error" value={hb.last_error} valueClass="text-rose-500" />}
          <div className="mt-3 space-y-1">
            {(hb.triggers || []).map((t, i) => (
              <div key={i} className={`text-xs ${dimCls(dark)}`}>
                <span className={strongCls(dark)}>{fmtUsd(t.level, 0)} → {t.leverage}x</span>{' '}
                {t.fired ? <span className="text-emerald-500">FIRED {t.fill?.dry ? '(dry) ' : ''}{t.fill?.avg_px ? `${fmtNum(t.fill.filled_sz, 5)} BTC @ ${fmtUsd(t.fill.avg_px, 0)}` : t.fill?.error ? `error: ${t.fill.error}` : ''} {t.fired_at ? fmtTime(t.fired_at) : ''}</span>
                  : t.skipped ? <span className="text-amber-500">skipped: {t.skipped}</span>
                  : <span>armed · {t.distance_pct != null ? `${t.distance_pct >= 0 ? '+' : ''}${t.distance_pct.toFixed(2)}% away` : ''}</span>}
              </div>
            ))}
          </div>
          {st?.fills?.length > 0 && <div className={`text-xs mt-3 ${dimCls(dark)}`}>fills: {st.fills.map((f, i) => <div key={i}>{fmtTime(f.t)} {fmtNum(f.filled_sz, 5)} BTC @ {fmtUsd(f.avg_px, 0)} ({f.leverage}x){f.error ? ` error ${f.error}` : ''}</div>)}</div>}
        </div>
      </div>
      {confirm === 'arm' && <Confirm dark={dark} tone="green" yesLabel="ARM" busy={busy} title={dryRun ? 'Arm the BTC play in DRY RUN?' : 'Arm the BTC play on mainnet (LIVE)?'}
        lines={[`BTC now ${livePx ? fmtUsd(livePx, 0) : '—'}; available ${avail != null ? fmtUsd(avail, 0) : '—'}.`,
          ...rows.map((t) => `BTC ≤ ${fmtUsd(t.level, 0)} → long ≈ ${avail != null ? fmtUsd(avail * 0.98 * t.leverage, 0) : '—'} at ${t.leverage}x cross; liquidation ≈ ${liqPct(t.leverage).toFixed(1)}% below entry (≈ ${fmtUsd(t.level * (1 - liqPct(t.leverage) / 100), 0)}).`),
          'Fires once per trigger on a touch from above; levels already at/below the price are skipped. No stop-loss: the liquidation is the stop.', `Route: ${Number(proxyIndex) < 0 ? 'direct' : `Proxy ${Number(proxyIndex) + 1}`}. Opens its own PowerShell window.`]}
        onYes={() => act('arm')} onNo={() => setConfirm(null)} />}
      {confirm === 'stop' && <Confirm dark={dark} tone="red" yesLabel="Stop" busy={busy} title="Disarm the play?" lines={['The runner exits. Any open BTC position stays open.']} onYes={() => act('stop')} onNo={() => setConfirm(null)} />}
    </div>
  );
}
