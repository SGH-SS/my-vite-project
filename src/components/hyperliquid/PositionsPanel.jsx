import { useState } from 'react';
import { hlPost, fmtUsd, fmtNum, fmtPct, pnlClass } from './api.js';
import { cardCls, dimCls, strongCls, btnCls, Heading, Confirm } from './ui.jsx';
import PositionManage from './PositionManage.jsx';

/** Open positions, live: the exchange's own liquidation price and the distance to it. */
export default function PositionsPanel({ dark, account, network, proxyIndex, refresh }) {
  const [closing, setClosing] = useState(null);   // coin pending confirm
  const [manage, setManage] = useState(null);     // coin whose manage panel is open
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);
  const positions = account?.positions || [];

  const doClose = async () => {
    setBusy(true); setRes(null);
    try { setRes(await hlPost('/close', { coin: closing, network, proxy_index: proxyIndex })); refresh(); } catch (e) { setRes({ error: e.message }); }
    setBusy(false); setClosing(null);
  };

  return (
    <div className={cardCls(dark)}>
      <Heading dark={dark} right={<span className={`text-xs ${dimCls(dark)}`}>refreshes every 5 s · liquidation from the exchange</span>}>Positions</Heading>
      {positions.length === 0 ? <div className={`text-sm ${dimCls(dark)}`}>no open positions on {network}</div> : (
        <div className="overflow-auto">
          <table className="w-full text-sm">
            <thead><tr className={dimCls(dark)}>
              {['coin', 'side', 'size', 'entry', 'mark', 'liquidation', 'to liq', 'lev', 'margin', 'uPnL', 'ROE', 'funding', ''].map((h) => <th key={h} className="text-left px-2 py-1 font-normal">{h}</th>)}
            </tr></thead>
            <tbody>
              {positions.map((p) => (
                <tr key={p.coin} className={`font-mono ${strongCls(dark)}`}>
                  <td className="px-2 py-1 font-semibold">{p.coin}</td>
                  <td className={`px-2 py-1 ${p.side === 'long' ? 'text-emerald-500' : 'text-rose-500'}`}>{p.side}</td>
                  <td className="px-2 py-1">{fmtNum(p.size, 5)}</td>
                  <td className="px-2 py-1">{fmtNum(p.entry_px, 4)}</td>
                  <td className="px-2 py-1">{fmtNum(p.mark_px, 4)}</td>
                  <td className="px-2 py-1 text-rose-400 font-semibold">{fmtNum(p.liquidation_px, 4)}</td>
                  <td className="px-2 py-1">{p.liq_distance_px != null ? `${fmtNum(p.liq_distance_px, 2)} (${fmtNum(p.liq_distance_pct, 1)}%)` : '—'}</td>
                  <td className="px-2 py-1">{p.leverage_effective ? `${fmtNum(p.leverage_effective, 1)}x` : `${p.leverage}x`} {p.leverage_type}</td>
                  <td className="px-2 py-1">{fmtUsd(p.margin_used)}</td>
                  <td className={`px-2 py-1 ${pnlClass(p.unrealized_pnl)}`}>{fmtUsd(p.unrealized_pnl)}</td>
                  <td className={`px-2 py-1 ${pnlClass(p.return_on_equity)}`}>{fmtPct(p.return_on_equity * 100, 1)}</td>
                  <td className={`px-2 py-1 ${pnlClass(-p.cum_funding_since_open)}`}>{fmtUsd(-p.cum_funding_since_open)}</td>
                  <td className="px-2 py-1 whitespace-nowrap"><button className={btnCls(dark, manage === p.coin, 'blue')} onClick={() => setManage(manage === p.coin ? null : p.coin)}>manage</button> <button className={btnCls(dark, false)} onClick={() => setClosing(p.coin)}>close</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {manage && positions.find((p) => p.coin === manage) && <PositionManage dark={dark} pos={positions.find((p) => p.coin === manage)} network={network} proxyIndex={proxyIndex} refresh={refresh} />}
      {closing && <Confirm dark={dark} tone="red" yesLabel={`Close ${closing}`} title={`Close the whole ${closing} position at market on ${network}?`}
        lines={['Reduce-only IOC order through the book, 3 % slippage cap.']} onYes={doClose} onNo={() => setClosing(null)} busy={busy} />}
      {res && <div className={`mt-2 text-xs font-mono ${res.ok ? 'text-emerald-500' : 'text-rose-500'}`}>{res.ok ? `closed ${fmtNum(res.fill?.filled_sz, 5)} @ ${fmtNum(res.fill?.avg_px, 4)}` : (res.error || res.fill?.error || 'close failed')}</div>}
    </div>
  );
}
