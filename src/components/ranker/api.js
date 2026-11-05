// Ranker dashboard API helpers (backend: /api/ranker).
export const RANKER_API = 'http://localhost:8000/api/ranker';

export async function rGet(path, params = {}) {
  const qs = new URLSearchParams(Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null))).toString();
  const r = await fetch(`${RANKER_API}${path}${qs ? `?${qs}` : ''}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function rPost(path, body = {}, params = {}) {
  const qs = new URLSearchParams(Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null))).toString();
  const r = await fetch(`${RANKER_API}${path}${qs ? `?${qs}` : ''}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export { fmtUsd, fmtNum, fmtPct, fmtTime, pnlClass, readPref, writePref } from '../hyperliquid/api.js';
