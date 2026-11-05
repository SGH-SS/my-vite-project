// Hyperliquid dashboard API helpers — every call carries the selected proxy and network.
export const HL_API = 'http://localhost:8000/api/hl';

export async function hlGet(path, params = {}) {
  const qs = new URLSearchParams(Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null))).toString();
  const r = await fetch(`${HL_API}${path}${qs ? `?${qs}` : ''}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function hlPost(path, body = {}, params = {}) {
  const qs = new URLSearchParams(Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null))).toString();
  const r = await fetch(`${HL_API}${path}${qs ? `?${qs}` : ''}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export const fmtUsd = (v, d = 2) => (v == null || Number.isNaN(Number(v)) ? '—' : `$${Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })}`);
export const fmtNum = (v, d = 2) => (v == null || Number.isNaN(Number(v)) ? '—' : Number(v).toLocaleString(undefined, { maximumFractionDigits: d }));
export const fmtPct = (v, d = 2) => (v == null || Number.isNaN(Number(v)) ? '—' : `${Number(v) >= 0 ? '+' : ''}${Number(v).toFixed(d)}%`);
export const fmtTime = (s) => (s ? new Date(s * 1000).toISOString().replace('T', ' ').slice(0, 19) + ' UTC' : '—');
export const pnlClass = (v) => (v == null ? '' : Number(v) >= 0 ? 'text-emerald-500' : 'text-rose-500');

export function readPref(key, fallback) {
  try { const v = localStorage.getItem(`hl.${key}`); return v == null ? fallback : JSON.parse(v); } catch { return fallback; }
}
export function writePref(key, value) {
  try { localStorage.setItem(`hl.${key}`, JSON.stringify(value)); } catch { /* ignore */ }
}
