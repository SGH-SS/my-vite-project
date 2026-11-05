// Sidebar: read-only browser of the server's folders. Clicking a file opens it in the preview panel.
import { useEffect, useState } from 'react';
import { cGet, fmtSize } from './api.js';
import { dimCls, Heading, ErrorBanner } from '../hyperliquid/ui.jsx';

const SHORTCUTS = [
  { label: 'context', path: '/research/context' },
  { label: 'work', path: '/research/work' },
  { label: 'data', path: '/research/data' },
];

function Crumbs({ dark, path, onGo }) {
  const parts = path.split('/').filter(Boolean);
  return (
    <div className="flex flex-wrap items-center text-xs">
      <button className={`${dimCls(dark)} hover:underline`} onClick={() => onGo('/')}>/</button>
      {parts.map((p, i) => {
        const to = `/${parts.slice(0, i + 1).join('/')}`;
        return (
          <span key={to} className="flex items-center">
            <button className={`max-w-[11rem] truncate ${i === parts.length - 1 ? (dark ? 'text-gray-200' : 'text-gray-800') : dimCls(dark)} hover:underline`} onClick={() => onGo(to)}>{p}</button>
            {i < parts.length - 1 && <span className={`px-0.5 ${dimCls(dark)}`}>/</span>}
          </span>
        );
      })}
    </div>
  );
}

export default function FileBrowser({ dark, path, onGo, selected, onOpen }) {
  const [list, setList] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let live = true;
    setErr(null);
    cGet('/files', { path })
      .then((r) => { if (live) setList(r); })
      .catch((e) => { if (live) { setErr(e.message); setList(null); } });
    return () => { live = false; };
  }, [path]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex-none">
        <Heading dark={dark} right={
          <span className="flex gap-2 text-xs">
            {SHORTCUTS.map((s) => <button key={s.path} className={`${dimCls(dark)} hover:underline`} onClick={() => onGo(s.path)}>{s.label}</button>)}
          </span>
        }>Files</Heading>
        <Crumbs dark={dark} path={path} onGo={onGo} />
        <ErrorBanner dark={dark} error={err} />
      </div>
      <div className="mt-1 min-h-0 flex-1 overflow-y-auto">
        {list?.entries.length === 0 && <div className={`px-2 py-1 text-xs ${dimCls(dark)}`}>Empty folder</div>}
        {list?.entries.map((e) => (
          <div key={e.path} title={e.name}
            className={`flex cursor-pointer items-center gap-1.5 rounded px-1.5 py-0.5 text-[13px] ${e.path === selected ? (dark ? 'bg-gray-700' : 'bg-gray-100') : (dark ? 'hover:bg-gray-700/50' : 'hover:bg-gray-50')}`}
            onClick={() => (e.is_dir ? onGo(e.path) : onOpen(e.path))}>
            <span className="flex-none text-xs">{e.is_dir ? '📁' : '📄'}</span>
            <span className={`min-w-0 flex-1 truncate ${dark ? 'text-gray-300' : 'text-gray-700'}`}>{e.name}</span>
            {!e.is_dir && <span className={`flex-none text-[11px] ${dimCls(dark)}`}>{fmtSize(e.size)}</span>}
          </div>
        ))}
        {list?.truncated && <div className={`px-2 py-1 text-xs ${dimCls(dark)}`}>Only the first {list.entries.length} entries are shown</div>}
      </div>
    </div>
  );
}
