import { useState } from 'react';
import { hlPost, fmtUsd, fmtNum } from './api.js';
import { cardCls, dimCls, strongCls, inputCls, btnCls, Heading, KV, ErrorBanner, Confirm } from './ui.jsx';

/** Account value (USDC + spot holdings + perp uPnL), one button to turn all BTC into USDC, transfer under "advanced". */
export default function AccountPanel({ dark, account, network, proxyIndex, refresh }) {
  const [confirmSell, setConfirmSell] = useState(false);
  const [sellBusy, setSellBusy] = useState(false);
  const [sellRes, setSellRes] = useState(null);
  const [advanced, setAdvanced] = useState(false);
  const [amt, setAmt] = useState('');
  const [toPerp, setToPerp] = useState(true);
  const [confirmXfer, setConfirmXfer] = useState(false);
  const [xferBusy, setXferBusy] = useState(false);
  const [xferRes, setXferRes] = useState(null);

  const a = account || {};
  const ubtc = (a.spot_balances || []).find((b) => b.coin === 'UBTC');
  const otherSpot = (a.spot_balances || []).filter((b) => b.coin !== 'USDC' && b.coin !== 'UBTC');

  const sellAll = async () => {
    setSellBusy(true); setSellRes(null);
    try { setSellRes(await hlPost('/convert-to-usdc', { coin: 'UBTC', slippage: 0.01, network, proxy_index: proxyIndex })); refresh(); } catch (e) { setSellRes({ error: e.message }); }
    setSellBusy(false); setConfirmSell(false);
  };
  const doTransfer = async () => {
    setXferBusy(true); setXferRes(null);
    try { setXferRes(await hlPost('/transfer', { amount: Number(amt), to_perp: toPerp, network, proxy_index: proxyIndex })); refresh(); } catch (e) { setXferRes({ error: e.message }); }
    setXferBusy(false); setConfirmXfer(false);
  };

  return (
    <div className={cardCls(dark)}>
      <Heading dark={dark}>Account</Heading>
      <div className={`text-2xl font-mono font-semibold ${strongCls(dark)}`}>{fmtUsd(a.account_value)}</div>
      {a.spot_holdings_usd != null && <div className={`text-xs ${dimCls(dark)}`}>USDC {fmtUsd(a.usdc_total)} (includes open-position uPnL {fmtUsd(a.unrealized_pnl)}) · BTC & other spot {fmtUsd(a.spot_holdings_usd)}</div>}
      <div className="space-y-1 mt-3">
        <KV dark={dark} label="USDC (tradeable margin)" value={fmtUsd(a.usdc_total)} valueClass="font-mono text-emerald-400 font-semibold" />
        <KV dark={dark} label="UBTC" value={ubtc ? `${fmtNum(ubtc.total, 8)} ≈ ${fmtUsd(ubtc.value_usd)} @ ${fmtNum(ubtc.px_usdc, 0)}` : '0'} />
        {otherSpot.map((b) => <KV key={b.coin} dark={dark} label={b.coin} value={`${fmtNum(b.total, 6)}${b.value_usd != null ? ` ≈ ${fmtUsd(b.value_usd)}` : ''}`} />)}
        <KV dark={dark} label="margin used" value={fmtUsd(a.total_margin_used)} />
        <KV dark={dark} label="available to trade" value={fmtUsd(a.available_to_trade)} />
        <KV dark={dark} label="withdrawable" value={fmtUsd(a.withdrawable)} />
      </div>

      <button className={`${btnCls(dark, true, 'amber')} w-full mt-3 py-2`} disabled={!ubtc || ubtc.total <= 0 || sellBusy} onClick={() => { setSellRes(null); setConfirmSell(true); }}>
        {sellBusy ? 'selling…' : ubtc && ubtc.total > 0 ? `Sell all BTC → USDC (${fmtNum(ubtc.total, 6)} UBTC ≈ ${fmtUsd(ubtc.value_usd)})` : 'Sell all BTC → USDC (no BTC on spot)'}
      </button>
      <div className={`text-xs mt-1 ${dimCls(dark)}`}>Spot market sell on UBTC/USDC, IOC at 1 % below mid; the USDC is then usable as perp margin.</div>
      {confirmSell && ubtc && <Confirm dark={dark} tone="amber" yesLabel="Sell all BTC" busy={sellBusy} title={`Sell ${fmtNum(ubtc.total, 8)} UBTC for about ${fmtUsd(ubtc.value_usd)} on ${network}?`}
        lines={[`Mid ${fmtNum(ubtc.px_usdc, 2)}, limit 1 % below it, IOC. Spot taker fee applies.`, `Routed via ${proxyIndex < 0 ? 'direct' : `Proxy ${proxyIndex + 1}`}.`]} onYes={sellAll} onNo={() => setConfirmSell(false)} />}
      {sellRes && (
        <div className={`mt-2 text-sm font-mono ${sellRes.ok ? 'text-emerald-500' : 'text-rose-500'}`}>
          {sellRes.ok ? `sold ${fmtNum(sellRes.fill.filled_sz, 6)} UBTC @ ${fmtNum(sellRes.fill.avg_px, 2)} → USDC now ${fmtUsd(sellRes.usdc_total)}` : (sellRes.error || sellRes.fill?.error || 'not filled')}
        </div>
      )}

      <button className={`text-xs mt-4 ${dimCls(dark)} hover:underline`} onClick={() => setAdvanced((v) => !v)}>{advanced ? 'hide' : 'show'} advanced (spot ↔ perp USDC transfer)</button>
      {advanced && (
        <div className="mt-2">
          <div className="flex gap-2">
            <input className={inputCls(dark)} placeholder="amount USDC" value={amt} onChange={(e) => setAmt(e.target.value)} />
            <button className={btnCls(dark, toPerp, 'blue')} onClick={() => setToPerp(true)}>spot → perp</button>
            <button className={btnCls(dark, !toPerp, 'blue')} onClick={() => setToPerp(false)}>perp → spot</button>
            <button className={btnCls(dark)} disabled={!(Number(amt) > 0)} onClick={() => setConfirmXfer(true)}>transfer</button>
          </div>
          <div className={`text-xs mt-1 ${dimCls(dark)}`}>Unified account: spot USDC already backs perps; only needed if the exchange reports it as not available to trade.</div>
          {confirmXfer && <Confirm dark={dark} title={`Transfer ${amt} USDC ${toPerp ? 'spot → perp' : 'perp → spot'} on ${network}?`} lines={['Signed by the master wallet.']} onYes={doTransfer} onNo={() => setConfirmXfer(false)} busy={xferBusy} />}
          {xferRes && <div className={`mt-2 text-xs font-mono ${xferRes.ok ? 'text-emerald-500' : 'text-rose-500'}`}>{xferRes.ok ? 'transfer ok' : (xferRes.error || JSON.stringify(xferRes.raw))}</div>}
        </div>
      )}
      <ErrorBanner dark={dark} error={a.spot_error || a.spot_px_error} />
    </div>
  );
}
