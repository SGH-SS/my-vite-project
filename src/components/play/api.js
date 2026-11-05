// BTC Play dashboard API helpers (backend: /api/play).
export const PLAY_API = 'http://localhost:8000/api/play';

export async function pGet(path, params = {}) {
  const qs = new URLSearchParams(Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null))).toString();
  const r = await fetch(`${PLAY_API}${path}${qs ? `?${qs}` : ''}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function pPost(path, body = {}) {
  const r = await fetch(`${PLAY_API}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export { fmtUsd, fmtNum, fmtTime, pnlClass, readPref, writePref } from '../hyperliquid/api.js';
