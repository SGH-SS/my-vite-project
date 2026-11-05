// Server CCode dashboard API helpers (backend: /api/ccode). 127.0.0.1, not localhost: Windows tries IPv6 first (+0.2 s).
export const CCODE_API = 'http://127.0.0.1:8000/api/ccode';
export const CCODE_WS = 'ws://127.0.0.1:8000/api/ccode/terminal';

export const MODELS = [
  { id: 'fable', label: 'Fable 5.1' },
  { id: 'opus', label: 'Opus 5.5' },
  { id: 'sonnet', label: 'Sonnet 5.5' },
  { id: 'haiku', label: 'Haiku 5.5' },
];
export const EFFORTS = [
  { id: 'low', label: 'Low' },
  { id: 'medium', label: 'Medium' },
  { id: 'high', label: 'High' },
  { id: 'xhigh', label: 'Extra' },
  { id: 'max', label: 'Max' },
];

async function unwrap(r) {
  if (r.ok) return r.json();
  let detail = `HTTP ${r.status}`;
  try { detail = (await r.json()).detail || detail; } catch { /* not JSON */ }
  throw new Error(detail);
}

export async function cGet(path, params = {}) {
  const qs = new URLSearchParams(params).toString();
  return unwrap(await fetch(`${CCODE_API}${path}${qs ? `?${qs}` : ''}`));
}

export async function cPost(path, body = {}) {
  return unwrap(await fetch(`${CCODE_API}${path}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }));
}

export const winPath = (id) => `/windows/${encodeURIComponent(id)}`;
export const fileUrl = (path, inline = false) =>
  `${CCODE_API}/download?${new URLSearchParams({ path, inline: inline ? 'true' : 'false' })}`;

export function fmtSize(n) {
  if (n == null) return '';
  if (n < 1000) return `${n} B`;
  if (n < 1e6) return `${(n / 1e3).toFixed(1)} KB`;
  if (n < 1e9) return `${(n / 1e6).toFixed(1)} MB`;
  return `${(n / 1e9).toFixed(2)} GB`;
}

export function fmtAge(t) {
  if (!t) return '';
  const s = Date.now() / 1000 - t;
  if (s < 90) return 'just now';
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  if (s < 129600) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}
