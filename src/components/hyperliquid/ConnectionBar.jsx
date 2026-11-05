import { useState } from 'react';
import { hlGet } from './api.js';
import { btnCls, cardCls, dimCls, strongCls, ErrorBanner } from './ui.jsx';

/** Proxy picker (Direct / Proxy 1–4, default Proxy 1), network switch, latency check. */
export default function ConnectionBar({ dark, proxies, proxyIndex, setProxyIndex, network, setNetwork, account, lastFetch }) {
  const [lat, setLat] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  const runLatency = async () => {
    setBusy(true); setErr(null);
    try { setLat(await hlGet('/latency', { proxy_index: proxyIndex, network })); } catch (e) { setErr(e.message); }
    setBusy(false);
  };

  const age = lastFetch ? Math.round((Date.now() - lastFetch) / 1000) : null;
  return (
    <div className={cardCls(dark)}>
      <div className="flex flex-wrap items-center gap-2">
        <span className={`text-xs uppercase tracking-wide ${dimCls(dark)}`}>Route</span>
        <button className={btnCls(dark, proxyIndex === -1)} onClick={() => setProxyIndex(-1)} title="No proxy — api.hyperliquid.xyz directly">Direct</button>
        {proxies.map((p) => (
          <button key={p.index} className={btnCls(dark, proxyIndex === p.index)} onClick={() => setProxyIndex(p.index)} title={p.label}>
            {p.label.split(' (')[0]} <span className="opacity-60 font-mono text-xs">{p.label.match(/\((.*)\)/)?.[1]}</span>
          </button>
        ))}
        <span className={`ml-4 text-xs uppercase tracking-wide ${dimCls(dark)}`}>Network</span>
        <button className={btnCls(dark, network === 'mainnet', 'green')} onClick={() => setNetwork('mainnet')}>mainnet</button>
        <button className={btnCls(dark, network === 'testnet', 'blue')} onClick={() => setNetwork('testnet')}>testnet</button>
        <button className={`${btnCls(dark)} ml-auto`} onClick={runLatency} disabled={busy}>{busy ? 'measuring…' : 'latency'}</button>
      </div>
      <div className={`mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs ${dimCls(dark)}`}>
        <span>account <span className={`font-mono ${strongCls(dark)}`}>{account?.master_address ?? '—'}</span></span>
        <span>mode <span className={`font-mono ${strongCls(dark)}`}>{account?.account_mode ?? '—'}</span></span>
        <span>served by <span className={`font-mono ${strongCls(dark)}`}>{account?.proxy_label ?? '—'}</span></span>
        <span>last refresh <span className={`font-mono ${strongCls(dark)}`}>{age == null ? '—' : `${age}s ago`}</span></span>
        {lat && <span>latency (median ms): {lat.tests?.map((t) => `${t.name} ${t.median_ms ?? 'err'}`).join(' · ')}</span>}
      </div>
      <ErrorBanner dark={dark} error={err || account?.error} />
    </div>
  );
}
