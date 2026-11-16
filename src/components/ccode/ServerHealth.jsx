// Server health, on click only (Nov 7 2026): nothing here polls, so the dashboard never adds load to the node.
// HealthCard sits in the sidebar (button + a short summary); HealthPanel opens under the terminal with the details.
// Data: GET /api/ccode/health -> tokoyo_server/research/ccode_health.py (1 s sample; ~4 s when it rescans /research).
import { dimCls, strongCls, btnCls, ErrorBanner } from '../hyperliquid/ui.jsx';
import { fmtAge } from './api.js';

export function fmtBytes(n, digits = 1) {
  if (n == null) return '—';
  const a = Math.abs(n);
  if (a >= 1e12) return `${(n / 1e12).toFixed(2)} TB`;
  if (a >= 1e9) return `${(n / 1e9).toFixed(digits)} GB`;
  if (a >= 1e6) return `${(n / 1e6).toFixed(digits)} MB`;
  return `${Math.round(n / 1e3)} KB`;
}

const level = (pct, warn = 60, bad = 85) => (pct >= bad ? 'bad' : pct >= warn ? 'warn' : 'ok');
const BAR = { ok: 'bg-emerald-500', warn: 'bg-amber-400', bad: 'bg-red-500' };
const TEXT = (dark) => ({ ok: dark ? 'text-emerald-300' : 'text-emerald-700', warn: dark ? 'text-amber-300' : 'text-amber-700', bad: dark ? 'text-red-300' : 'text-red-700' });

function Bar({ dark, pct, tone, h = 'h-1.5' }) {
  return (
    <span className={`relative block ${h} w-full overflow-hidden rounded-full ${dark ? 'bg-gray-700' : 'bg-gray-200'}`}>
      <span className={`absolute inset-y-0 left-0 rounded-full ${BAR[tone || level(pct)]}`} style={{ width: `${Math.max(Math.min(pct, 100), 1)}%` }} />
    </span>
  );
}

function fillDate(t) {
  return new Date(t * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

// "fills in ~43 d (Dec 20)", "fills in ≥ 11 d (Nov 18)" for an upper-bound rate, "not growing"
export function fillText(d) {
  if (d.per_day == null) return 'growth unknown yet';
  if (d.days_left == null) return 'not growing';
  const days = d.days_left < 10 ? d.days_left.toFixed(1) : Math.round(d.days_left);
  return `fills in ${d.kind === 'upper_bound' ? '≥ ' : '~'}${days} d (${fillDate(d.full_at)})`;
}

function diskTone(d) {
  if (d.days_left != null && d.days_left < 7) return 'bad';
  if (d.days_left != null && d.days_left < 21) return 'warn';
  return level(d.pct, 75, 90);
}

const memPct = (m) => (100 * m.used) / m.total;
const lagTone = (lag) => (lag == null ? 'warn' : lag >= 10 ? 'bad' : lag >= 2 ? 'warn' : 'ok');

// ── sidebar card ────────────────────────────────────────────────────────────

export function HealthCard({ dark, data, busy, err, open, onCheck, onToggle }) {
  const t = TEXT(dark);
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className={`flex-1 text-xs font-semibold uppercase tracking-wide ${dimCls(dark)}`}>Server health</span>
        {data && <button className={`text-xs hover:underline ${dimCls(dark)}`} onClick={onToggle}>{open ? 'Hide' : 'Details'}</button>}
        <button className={btnCls(dark, true, 'amber')} disabled={busy} onClick={onCheck} title="Measures the server once (about 1 s, up to 5 s). Nothing polls.">
          {busy ? 'Checking…' : data ? 'Check again' : 'Check server'}
        </button>
      </div>
      {err && <ErrorBanner dark={dark} error={err} />}
      {!data && !err && <div className={`text-xs ${dimCls(dark)}`}>Click to measure disks, CPU, memory and node lag. It only runs when you click.</div>}
      {data && (
        <div className="space-y-1.5 text-[11px]">
          {data.disks.map((d) => (
            <div key={d.mount} title={`${d.label}: ${fmtBytes(d.used)} used of ${fmtBytes(d.total)}, ${fmtBytes(d.free)} free. ${d.source}`}>
              <div className="flex justify-between gap-2">
                <span className={strongCls(dark)}>{d.mount} <span className={dimCls(dark)}>{Math.round(d.pct)}%</span></span>
                <span className={t[diskTone(d)]}>{fillText(d)}</span>
              </div>
              <Bar dark={dark} pct={d.pct} tone={diskTone(d)} />
            </div>
          ))}
          <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 pt-0.5">
            <div title={`load ${data.cpu.load.join(' / ')}`}>
              <div className="flex justify-between"><span className={strongCls(dark)}>CPU</span><span className={dimCls(dark)}>{Math.round(data.cpu.busy_pct)}%</span></div>
              <Bar dark={dark} pct={data.cpu.busy_pct} />
            </div>
            <div title={`${fmtBytes(data.memory.used)} used of ${fmtBytes(data.memory.total)}`}>
              <div className="flex justify-between"><span className={strongCls(dark)}>RAM</span><span className={dimCls(dark)}>{Math.round(memPct(data.memory))}%</span></div>
              <Bar dark={dark} pct={memPct(data.memory)} />
            </div>
          </div>
          <div className={`flex justify-between ${dimCls(dark)}`}>
            <span>Node lag <span className={t[lagTone(data.node.lag_s)]}>{data.node.lag_s == null ? '?' : `${data.node.lag_s.toFixed(1)} s`}</span></span>
            <span>checked {fmtAge(data.checked)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ── details panel ───────────────────────────────────────────────────────────

function Section({ dark, title, children, className = '' }) {
  return (
    <div className={`rounded-lg border p-3 ${dark ? 'border-gray-700 bg-gray-900/40' : 'border-gray-200 bg-gray-50'} ${className}`}>
      <div className={`mb-2 text-xs font-semibold uppercase tracking-wide ${dimCls(dark)}`}>{title}</div>
      {children}
    </div>
  );
}

function Row({ dark, label, value, tone }) {
  return (
    <div className="flex justify-between gap-3 text-xs">
      <span className={dimCls(dark)}>{label}</span>
      <span className={`tabular-nums ${tone ? TEXT(dark)[tone] : strongCls(dark)}`}>{value}</span>
    </div>
  );
}

function Disks({ dark, disks }) {
  return (
    <Section dark={dark} title="Storage" className="col-span-2">
      <div className="grid gap-3 md:grid-cols-2">
        {disks.map((d) => {
          const tone = diskTone(d);
          return (
            <div key={d.mount} className="space-y-1">
              <div className="flex items-baseline justify-between gap-2">
                <span className={`text-sm font-medium ${strongCls(dark)}`}>{d.mount} <span className={`text-xs font-normal ${dimCls(dark)}`}>{d.label} · {d.device}</span></span>
                <span className={`text-sm tabular-nums ${strongCls(dark)}`}>{d.pct.toFixed(1)}%</span>
              </div>
              <Bar dark={dark} pct={d.pct} tone={tone} h="h-2.5" />
              <Row dark={dark} label="Used / size" value={`${fmtBytes(d.used)} / ${fmtBytes(d.total)}`} />
              <Row dark={dark} label="Free" value={fmtBytes(d.free)} />
              <Row dark={dark} label={d.kind === 'upper_bound' ? 'Growth (at most)' : 'Growth'}
                value={d.per_day == null ? '—' : `${d.per_day >= 0 ? '+' : ''}${fmtBytes(d.per_day)}/day`} />
              {d.scan && <Row dark={dark} label="Written last 24 h" value={fmtBytes(d.scan.last_day)} />}
              <Row dark={dark} label="At this rate" value={fillText(d)} tone={tone} />
              <div className={`text-[11px] leading-snug ${dimCls(dark)}`}>Source: {d.source}</div>
            </div>
          );
        })}
      </div>
    </Section>
  );
}

function Cpu({ dark, cpu }) {
  return (
    <Section dark={dark} title={`CPU · ${cpu.cores} cores`}>
      <div className="mb-2 flex items-baseline gap-2">
        <span className={`text-2xl font-semibold tabular-nums ${TEXT(dark)[level(cpu.busy_pct)]}`}>{Math.round(cpu.busy_pct)}%</span>
        <span className={`text-xs ${dimCls(dark)}`}>busy over the 1 s sample</span>
      </div>
      <div className="mb-2 flex h-14 items-end gap-0.5" title="Each bar is one core">
        {cpu.per_core.map((p, i) => (
          <span key={i} className={`relative h-full flex-1 overflow-hidden rounded-sm ${dark ? 'bg-gray-700' : 'bg-gray-200'}`} title={`core ${i}: ${p}%`}>
            <span className={`absolute inset-x-0 bottom-0 ${BAR[level(p)]}`} style={{ height: `${Math.max(p, 2)}%` }} />
          </span>
        ))}
      </div>
      <Row dark={dark} label="Load 1 / 5 / 15 min" value={cpu.load.map((x) => x.toFixed(2)).join(' / ')} tone={level((100 * cpu.load[0]) / cpu.cores)} />
      <Row dark={dark} label="Waiting on disk (iowait)" value={`${cpu.iowait_pct}%`} tone={level(cpu.iowait_pct, 5, 20)} />
      {cpu.model && <div className={`mt-1 truncate text-[11px] ${dimCls(dark)}`}>{cpu.model}</div>}
    </Section>
  );
}

function Memory({ dark, m }) {
  const used = (100 * m.used) / m.total;
  const cache = (100 * Math.min(m.cache, m.total - m.used)) / m.total;
  return (
    <Section dark={dark} title="Memory">
      <div className="mb-2 flex items-baseline gap-2">
        <span className={`text-2xl font-semibold tabular-nums ${TEXT(dark)[level(used, 70, 90)]}`}>{Math.round(used)}%</span>
        <span className={`text-xs ${dimCls(dark)}`}>{fmtBytes(m.used)} of {fmtBytes(m.total)} in use</span>
      </div>
      <span className={`mb-2 flex h-2.5 w-full overflow-hidden rounded-full ${dark ? 'bg-gray-700' : 'bg-gray-200'}`}>
        <span className={BAR[level(used, 70, 90)]} style={{ width: `${used}%` }} />
        <span className={dark ? 'bg-sky-700' : 'bg-sky-300'} style={{ width: `${cache}%` }} title="File cache: given back when programs need it" />
      </span>
      <Row dark={dark} label="Available" value={fmtBytes(m.available)} />
      <Row dark={dark} label="File cache (reclaimable)" value={fmtBytes(m.cache)} />
      <Row dark={dark} label="Swap used" value={`${fmtBytes(m.swap_used)} / ${fmtBytes(m.swap_total)}`} tone={m.swap_used > 1e9 ? 'warn' : undefined} />
    </Section>
  );
}

function Groups({ dark, groups, cores }) {
  return (
    <Section dark={dark} title="Who is using it">
      <div className="space-y-2">
        {groups.map((g) => (
          <div key={g.name} className="space-y-0.5">
            <div className={`text-xs font-medium ${strongCls(dark)}`}>{g.name}</div>
            <Row dark={dark} label="CPU" value={g.cpu_cores == null ? '—' : `${g.cpu_cores.toFixed(2)} of ${cores} cores`} />
            <Row dark={dark} label="Memory" value={g.memory == null ? '—' : `${fmtBytes(g.memory)}${g.memory_max ? ` of ${fmtBytes(g.memory_max)} cap` : ''}`}
              tone={g.memory_max && g.memory / g.memory_max > 0.85 ? 'bad' : undefined} />
          </div>
        ))}
        <div className={`text-[11px] leading-snug ${dimCls(dark)}`}>Memory includes each group's file cache.</div>
      </div>
    </Section>
  );
}

const PSI_NAMES = { cpu: 'CPU', memory: 'Memory', io: 'Disk' };

function Pressure({ dark, pressure }) {
  return (
    <Section dark={dark} title="Pressure (time spent waiting)">
      <div className="space-y-1">
        {Object.entries(pressure).map(([k, v]) => {
          const s = v.some || {};
          return (
            <Row key={k} dark={dark} label={`${PSI_NAMES[k] || k} · 10 s / 60 s`}
              value={`${(s.avg10 ?? 0).toFixed(1)}% / ${(s.avg60 ?? 0).toFixed(1)}%`} tone={level(Math.max(s.avg10 ?? 0, s.avg60 ?? 0), 5, 20)} />
          );
        })}
        <div className={`pt-1 text-[11px] leading-snug ${dimCls(dark)}`}>Share of time some task was stalled waiting for that resource. Above ~5% the node may feel it.</div>
      </div>
    </Section>
  );
}

function IoNet({ dark, io, net, node }) {
  return (
    <Section dark={dark} title="Disk I/O · network · node">
      <div className="space-y-1">
        {io.map((d) => (
          <Row key={d.device} dark={dark} label={`${d.mount} read / write`} value={`${d.read_mbs} / ${d.write_mbs} MB/s · ${d.busy_pct}% busy`} tone={level(d.busy_pct, 50, 85)} />
        ))}
        <Row dark={dark} label="Network in / out" value={`${net.rx_mbs} / ${net.tx_mbs} MB/s`} />
        <Row dark={dark} label="Node service" value={node.active ? 'running' : 'STOPPED'} tone={node.active ? 'ok' : 'bad'} />
        <Row dark={dark} label="Node lag" value={node.lag_s == null ? 'unknown' : `${node.lag_s.toFixed(2)} s`} tone={lagTone(node.lag_s)} />
        <Row dark={dark} label="Nodeguard" value={node.guard} tone={node.guard === 'ok' ? 'ok' : 'warn'} />
        <Row dark={dark} label="Server up" value={`${(node.uptime_s / 86400).toFixed(1)} days`} />
      </div>
    </Section>
  );
}

function Top({ dark, top }) {
  return (
    <Section dark={dark} title="Busiest processes (1 s sample)" className="col-span-2">
      <table className="w-full text-xs">
        <thead>
          <tr className={dimCls(dark)}>
            <th className="py-0.5 text-left font-normal">Process</th>
            <th className="text-left font-normal">User</th>
            <th className="text-right font-normal">PID</th>
            <th className="text-right font-normal">CPU</th>
            <th className="text-right font-normal">Memory</th>
          </tr>
        </thead>
        <tbody className="tabular-nums">
          {top.map((p) => (
            <tr key={p.pid} className={dark ? 'text-gray-300' : 'text-gray-700'}>
              <td className="py-0.5 font-mono">{p.name}</td>
              <td>{p.user}</td>
              <td className="text-right">{p.pid}</td>
              <td className={`text-right ${TEXT(dark)[level(p.cpu_pct / 2, 50, 85)]}`}>{p.cpu_pct}%</td>
              <td className="text-right">{fmtBytes(p.rss)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className={`mt-1 text-[11px] ${dimCls(dark)}`}>100% = one full core.</div>
    </Section>
  );
}

export function HealthPanel({ dark, data, busy, onCheck, onClose }) {
  return (
    <div className={`flex h-full flex-col border-t ${dark ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-white'}`}>
      <div className="flex flex-none items-center gap-2 px-3 py-2">
        <span className={`flex-1 text-sm font-medium ${strongCls(dark)}`}>
          Server health <span className={`text-xs font-normal ${dimCls(dark)}`}>
            checked {new Date(data.checked * 1000).toLocaleTimeString()} ({fmtAge(data.checked)}) · took {data.took_s} s
          </span>
        </span>
        <button className={btnCls(dark, true, 'amber')} disabled={busy} onClick={onCheck}>{busy ? 'Checking…' : 'Check again'}</button>
        <button className={btnCls(dark)} onClick={onClose}>Close</button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
        <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
          <Disks dark={dark} disks={data.disks} />
          <Cpu dark={dark} cpu={data.cpu} />
          <Memory dark={dark} m={data.memory} />
          <Groups dark={dark} groups={data.groups} cores={data.cpu.cores} />
          <Pressure dark={dark} pressure={data.pressure} />
          <IoNet dark={dark} io={data.io} net={data.net} node={data.node} />
          <Top dark={dark} top={data.top} />
        </div>
      </div>
    </div>
  );
}
