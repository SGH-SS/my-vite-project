// How full a chat's context is after its latest reply, and how many times it has compacted (Nov 7 2026).
// Data: chat.context from /api/ccode/chats ({ tokens, window, pct, compactions, last_compact }); tokens is null right
// after a compaction until the chat's next reply.
import { dimCls } from '../hyperliquid/ui.jsx';

export function fmtTokens(n) {
  if (n == null) return '—';
  if (n >= 1e6) return `${+(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${Math.round(n / 1e3)}k`;
  return `${n}`;
}

function barColor(pct) {
  if (pct >= 70) return 'bg-red-500';
  if (pct >= 40) return 'bg-amber-400';
  return 'bg-emerald-500';
}

function describe(ctx) {
  const size = ctx.tokens == null
    ? 'Just compacted: the new size shows after the next reply.'
    : `${fmtTokens(ctx.tokens)} of ${fmtTokens(ctx.window)} tokens in context after the latest reply (${ctx.pct}%).`;
  const comp = ctx.compactions === 0
    ? 'Never compacted.'
    : `Compacted ${ctx.compactions} time${ctx.compactions === 1 ? '' : 's'}${ctx.last_compact ? `, the last at ${fmtTokens(ctx.last_compact)} tokens` : ''}.`;
  return `${size} ${comp}`;
}

export function CompactBadge({ dark, n, long }) {
  if (!n) return null;
  return (
    <span className={`flex-none rounded px-1 text-[10px] font-medium leading-4 ${dark ? 'bg-violet-900/60 text-violet-200' : 'bg-violet-100 text-violet-700'}`}>
      {long ? `compacted ${n}×` : `⟲ ${n}`}
    </span>
  );
}

// wide: the version in the terminal header, with the token count spelled out
export default function ContextMeter({ dark, ctx, wide = false }) {
  if (!ctx) return null;
  const known = ctx.tokens != null;
  const pct = known ? Math.min(100, ctx.pct) : 0;
  return (
    <span className={`flex items-center gap-1.5 ${wide ? 'w-56' : 'w-full'}`} title={describe(ctx)}>
      <span className={`relative h-1.5 min-w-0 flex-1 overflow-hidden rounded-full ${dark ? 'bg-gray-700' : 'bg-gray-200'}`}>
        {known && <span className={`absolute inset-y-0 left-0 rounded-full ${barColor(pct)}`} style={{ width: `${Math.max(pct, 1.5)}%` }} />}
      </span>
      <span className={`flex-none text-[10px] tabular-nums ${dimCls(dark)}`}>
        {known ? (wide ? `${fmtTokens(ctx.tokens)} / ${fmtTokens(ctx.window)} · ${Math.round(ctx.pct)}%` : `${Math.round(ctx.pct)}%`) : 'compacted'}
      </span>
      <CompactBadge dark={dark} n={ctx.compactions} long={wide} />
    </span>
  );
}
