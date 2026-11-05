/**
 * Hyperliquid — dedicated dashboard: connection (proxies 1–4 / direct, default Proxy 1), account, live positions with
 * the exchange's liquidation price, trading station (estimate before, actual after), open orders, and the
 * V2 + Add + Fresh-Liq runner controls.  Backend: /api/hl (my-vite-project/backend/routers/hyperliquid.py).
 */
import { useCallback, useEffect, useState } from 'react';
import { hlGet, readPref, writePref } from './api.js';
import ConnectionBar from './ConnectionBar.jsx';
import AccountPanel from './AccountPanel.jsx';
import PositionsPanel from './PositionsPanel.jsx';
import TradingStation from './TradingStation.jsx';
import OpenOrders from './OpenOrders.jsx';
import StrategyPanel from './StrategyPanel.jsx';

export default function Hyperliquid({ isDarkMode: dark }) {
  const [proxies, setProxies] = useState([]);
  const [proxyIndex, setProxyIndexState] = useState(() => readPref('proxyIndex', 0));
  const [network, setNetworkState] = useState(() => readPref('network', 'mainnet'));
  const [account, setAccount] = useState(null);
  const [lastFetch, setLastFetch] = useState(null);
  const setProxyIndex = (i) => { setProxyIndexState(i); writePref('proxyIndex', i); };
  const setNetwork = (n) => { setNetworkState(n); writePref('network', n); };

  useEffect(() => { hlGet('/proxies').then((d) => setProxies(d.proxies || [])).catch(() => setProxies([])); }, []);

  const refresh = useCallback(async () => {
    try { const a = await hlGet('/account', { network, proxy_index: proxyIndex }); setAccount(a); setLastFetch(Date.now()); }
    catch (e) { setAccount((prev) => ({ ...(prev || {}), error: e.message })); }
  }, [network, proxyIndex]);

  useEffect(() => { refresh(); const id = setInterval(refresh, 5000); return () => clearInterval(id); }, [refresh]);

  return (
    <div className="space-y-4">
      <ConnectionBar dark={dark} proxies={proxies} proxyIndex={proxyIndex} setProxyIndex={setProxyIndex} network={network} setNetwork={setNetwork} account={account} lastFetch={lastFetch} />
      <PositionsPanel dark={dark} account={account} network={network} proxyIndex={proxyIndex} refresh={refresh} />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2"><TradingStation dark={dark} network={network} proxyIndex={proxyIndex} account={account} refresh={refresh} /></div>
        <AccountPanel dark={dark} account={account} network={network} proxyIndex={proxyIndex} refresh={refresh} />
      </div>
      <OpenOrders dark={dark} account={account} network={network} proxyIndex={proxyIndex} refresh={refresh} />
      <StrategyPanel dark={dark} network={network} proxyIndex={proxyIndex} />
    </div>
  );
}
