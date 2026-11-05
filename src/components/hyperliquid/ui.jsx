// Small shared presentational pieces for the Hyperliquid dashboard.
export const cardCls = (dark) => `rounded-lg border p-4 ${dark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`;
export const dimCls = (dark) => (dark ? 'text-gray-400' : 'text-gray-500');
export const strongCls = (dark) => (dark ? 'text-gray-100' : 'text-gray-900');
export const inputCls = (dark) => `w-full text-sm rounded px-2 py-1.5 border ${dark ? 'bg-gray-900 border-gray-700 text-gray-100' : 'bg-white border-gray-300 text-gray-900'}`;
export const btnCls = (dark, active = false, tone = 'amber') => {
  const tones = {
    amber: dark ? 'bg-amber-500 text-gray-900' : 'bg-amber-500 text-white',
    green: dark ? 'bg-emerald-600 text-white' : 'bg-emerald-600 text-white',
    red: dark ? 'bg-rose-600 text-white' : 'bg-rose-600 text-white',
    blue: dark ? 'bg-sky-600 text-white' : 'bg-sky-600 text-white',
  };
  return `px-3 py-1.5 rounded-md text-sm font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${active ? tones[tone] : (dark ? 'bg-gray-700 text-gray-200 hover:bg-gray-600' : 'bg-gray-100 text-gray-700 hover:bg-gray-200')}`;
};

export function Heading({ dark, children, right }) {
  return (
    <div className="flex items-center justify-between gap-2 mb-2">
      <div className={`text-xs uppercase tracking-wide ${dimCls(dark)}`}>{children}</div>
      {right}
    </div>
  );
}

export function KV({ dark, label, value, valueClass = '' }) {
  return (
    <div className="flex justify-between gap-2 text-sm">
      <span className={dimCls(dark)}>{label}</span>
      <span className={`font-mono ${valueClass || strongCls(dark)}`}>{value}</span>
    </div>
  );
}

export function ErrorBanner({ dark, error }) {
  if (!error) return null;
  return <div className={`text-xs rounded px-2 py-1 mt-2 ${dark ? 'bg-rose-900/40 text-rose-300' : 'bg-rose-100 text-rose-800'}`}>{String(error)}</div>;
}

export function Confirm({ dark, title, lines, onYes, onNo, busy, yesLabel = 'Confirm', tone = 'green' }) {
  return (
    <div className={`mt-3 rounded-lg border p-3 ${dark ? 'border-amber-500/60 bg-amber-900/20' : 'border-amber-400 bg-amber-50'}`}>
      <div className={`font-semibold ${strongCls(dark)}`}>{title}</div>
      <ul className={`mt-1 text-sm ${dark ? 'text-gray-300' : 'text-gray-700'}`}>{lines.map((l, i) => <li key={i}>{l}</li>)}</ul>
      <div className="flex gap-2 mt-2">
        <button className={btnCls(dark, true, tone)} onClick={onYes} disabled={busy}>{busy ? 'working…' : yesLabel}</button>
        <button className={btnCls(dark)} onClick={onNo} disabled={busy}>Cancel</button>
      </div>
    </div>
  );
}
