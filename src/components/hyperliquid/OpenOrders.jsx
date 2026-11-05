import { useState } from 'react';
import { hlPost, fmtNum, fmtTime } from './api.js';
import { cardCls, dimCls, strongCls, btnCls, Heading } from './ui.jsx';

/** Resting orders (incl. TP/SL triggers) with cancel. */
export default function OpenOrders({ dark, account, network, proxyIndex, refresh }) {
  const [busy, setBusy] = useState(null);
  const orders = account?.open_orders || [];
  const cancel = async (o) => {
    setBusy(o.oid);
    try { await hlPost('/cancel', { coin: o.coin, oid: o.oid, network, proxy_index: proxyIndex }); refresh(); } catch { /* shown on next refresh */ }
    setBusy(null);
  };
  return (
    <div className={cardCls(dark)}>
      <Heading dark={dark}>Open orders</Heading>
      {orders.length === 0 ? <div className={`text-sm ${dimCls(dark)}`}>none</div> : (
        <table className="w-full text-sm">
          <thead><tr className={dimCls(dark)}>{['coin', 'side', 'type', 'size', 'price', 'trigger', 'placed', ''].map((h) => <th key={h} className="text-left px-2 py-1 font-normal">{h}</th>)}</tr></thead>
          <tbody>
            {orders.map((o) => (
              <tr key={o.oid} className={`font-mono ${strongCls(dark)}`}>
                <td className="px-2 py-1">{o.coin}</td>
                <td className={`px-2 py-1 ${o.side === 'B' ? 'text-emerald-500' : 'text-rose-500'}`}>{o.side === 'B' ? 'buy' : 'sell'}</td>
                <td className="px-2 py-1">{o.orderType}{o.reduceOnly ? ' (reduce)' : ''}</td>
                <td className="px-2 py-1">{fmtNum(o.sz, 5)}</td>
                <td className="px-2 py-1">{fmtNum(o.limitPx, 4)}</td>
                <td className="px-2 py-1">{o.isTrigger ? `${o.triggerCondition} ${fmtNum(o.triggerPx, 4)}` : '—'}</td>
                <td className="px-2 py-1">{fmtTime(o.timestamp / 1000)}</td>
                <td className="px-2 py-1"><button className={btnCls(dark)} disabled={busy === o.oid} onClick={() => cancel(o)}>{busy === o.oid ? '…' : 'cancel'}</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
