// Panel under the terminal: read-only preview of a server file (text, Markdown, table, image), with a download link.
import { useEffect, useState } from 'react';
import { cGet, fileUrl, fmtSize } from './api.js';
import { dimCls, strongCls, ErrorBanner } from '../hyperliquid/ui.jsx';

function inline(text, dark) {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((p, i) => {
    if (p.startsWith('**') && p.endsWith('**')) return <strong key={i}>{p.slice(2, -2)}</strong>;
    if (p.startsWith('`') && p.endsWith('`')) return <code key={i} className={`rounded px-1 text-[12px] ${dark ? 'bg-gray-900' : 'bg-gray-100'}`}>{p.slice(1, -1)}</code>;
    return p;
  });
}

function Markdown({ dark, text }) {
  const out = [];
  let code = null;
  text.split('\n').forEach((line, i) => {
    if (line.startsWith('```')) {
      if (code === null) { code = []; return; }
      out.push(<pre key={i} className={`my-1 overflow-x-auto rounded p-2 font-mono text-[12px] ${dark ? 'bg-gray-900' : 'bg-gray-100'}`}>{code.join('\n')}</pre>);
      code = null;
      return;
    }
    if (code !== null) { code.push(line); return; }
    const h = line.match(/^(#{1,4})\s+(.*)/);
    if (h) out.push(<div key={i} className={`mt-2 font-semibold ${strongCls(dark)} ${h[1].length === 1 ? 'text-base' : 'text-sm'}`}>{inline(h[2], dark)}</div>);
    else if (/^\s*[-*]\s+/.test(line)) out.push(<div key={i} className="flex gap-1.5 pl-2"><span>•</span><span>{inline(line.replace(/^\s*[-*]\s+/, ''), dark)}</span></div>);
    else if (line.trim()) out.push(<div key={i}>{inline(line, dark)}</div>);
    else out.push(<div key={i} className="h-2" />);
  });
  if (code !== null) out.push(<pre key="end" className="font-mono text-[12px]">{code.join('\n')}</pre>);
  return <div className={`text-sm leading-relaxed ${dark ? 'text-gray-300' : 'text-gray-700'}`}>{out}</div>;
}

function Table({ dark, data }) {
  return (
    <div>
      {data.num_rows != null && <div className={`mb-1 text-xs ${dimCls(dark)}`}>{data.num_rows.toLocaleString()} rows, {data.columns.length} columns (showing the first {data.rows.length})</div>}
      <table className="font-mono text-[12px]">
        <thead>
          <tr>{data.columns.map((c) => (
            <th key={c.name} className={`sticky top-0 whitespace-nowrap border-b px-2 py-1 text-left font-medium ${dark ? 'border-gray-700 bg-gray-800 text-gray-200' : 'border-gray-200 bg-white text-gray-800'}`}>
              {c.name}{c.type && <span className={`ml-1 font-normal ${dimCls(dark)}`}>{c.type}</span>}
            </th>
          ))}</tr>
        </thead>
        <tbody>
          {data.rows.map((r, i) => (
            <tr key={i}>{r.map((v, j) => <td key={j} className={`max-w-[22rem] truncate whitespace-nowrap px-2 py-0.5 ${dark ? 'text-gray-300' : 'text-gray-700'}`} title={v ?? ''}>{v ?? ''}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function FilePreview({ dark, path, onClose }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let live = true;
    setData(null);
    setErr(null);
    cGet('/file', { path })
      .then((r) => { if (live) setData(r); })
      .catch((e) => { if (live) setErr(e.message); });
    return () => { live = false; };
  }, [path]);

  const name = path.split('/').pop();
  return (
    <div className={`flex h-full min-h-0 flex-col border-t ${dark ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-white'}`}>
      <div className="flex flex-none items-center gap-3 px-3 py-1.5">
        <span className={`truncate text-sm font-medium ${strongCls(dark)}`} title={path}>{name}</span>
        {data?.size != null && <span className={`text-xs ${dimCls(dark)}`}>{fmtSize(data.size)}</span>}
        {data?.truncated && <span className={`text-xs ${dimCls(dark)}`}>showing the start only</span>}
        <span className="flex-1" />
        <a className={`text-xs hover:underline ${dimCls(dark)}`} href={fileUrl(path)}>Download</a>
        <button className={`text-sm ${dimCls(dark)}`} title="Close" onClick={onClose}>✕</button>
      </div>
      <div className="min-h-0 flex-1 overflow-auto px-3 pb-3">
        <ErrorBanner dark={dark} error={err} />
        {!data && !err && <div className={`text-xs ${dimCls(dark)}`}>Loading…</div>}
        {data?.kind === 'text' && (name.toLowerCase().endsWith('.md')
          ? <Markdown dark={dark} text={data.text} />
          : <pre className={`font-mono text-[12px] leading-relaxed ${dark ? 'text-gray-300' : 'text-gray-700'}`}>{data.text}</pre>)}
        {data?.kind === 'table' && <Table dark={dark} data={data} />}
        {data?.kind === 'image' && <img src={fileUrl(path, true)} alt={name} className="max-h-full max-w-full" />}
        {data?.kind === 'binary' && <div className={`text-sm ${dimCls(dark)}`}>No preview for this file type. Use Download.</div>}
      </div>
    </div>
  );
}
