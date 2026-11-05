// Sidebar: start a chat, the running chats with what each is doing, and past chats to resume.
import { MODELS, EFFORTS, fmtAge } from './api.js';
import { btnCls, dimCls, strongCls, Heading } from '../hyperliquid/ui.jsx';

const STATES = {
  working: { dot: 'bg-amber-400 animate-pulse', label: 'working' },
  waiting: { dot: 'bg-sky-400', label: 'waiting for you' },
  idle: { dot: 'bg-gray-500', label: 'idle' },
};

function selectCls(dark) {
  return `text-xs rounded px-1.5 py-1 border ${dark ? 'bg-gray-900 border-gray-700 text-gray-200' : 'bg-white border-gray-300 text-gray-800'}`;
}

function Running({ dark, chat, active, onSelect, onFolder }) {
  const s = STATES[chat.state] || STATES.idle;
  return (
    <div className={`group flex items-center gap-2 rounded px-2 py-1.5 cursor-pointer ${active ? (dark ? 'bg-gray-700' : 'bg-gray-100') : (dark ? 'hover:bg-gray-700/50' : 'hover:bg-gray-50')}`}
      onClick={() => onSelect(chat.window_id)} title={s.label}>
      <span className={`h-2 w-2 flex-none rounded-full ${s.dot}`} />
      <span className="min-w-0 flex-1">
        <span className={`block truncate text-sm ${active ? strongCls(dark) : (dark ? 'text-gray-300' : 'text-gray-700')}`}>{chat.name}</span>
        <span className={`block text-[11px] ${dimCls(dark)}`}>{s.label}</span>
      </span>
      {chat.folder && (
        <button className={`invisible text-xs group-hover:visible ${dimCls(dark)}`} title="Open this chat's folder"
          onClick={(e) => { e.stopPropagation(); onFolder(chat.folder); }}>📁</button>
      )}
    </div>
  );
}

function Past({ dark, chat, busy, onResume, onFolder }) {
  return (
    <div className={`group flex items-center gap-2 rounded px-2 py-1 ${dark ? 'hover:bg-gray-700/50' : 'hover:bg-gray-50'}`}>
      <span className="min-w-0 flex-1">
        <span className={`block truncate text-sm ${dark ? 'text-gray-300' : 'text-gray-700'}`} title={chat.name}>{chat.name}</span>
        <span className={`block text-[11px] ${dimCls(dark)}`}>{fmtAge(chat.mtime)}</span>
      </span>
      {chat.folder && (
        <button className={`invisible text-xs group-hover:visible ${dimCls(dark)}`} title="Open this chat's folder"
          onClick={() => onFolder(chat.folder)}>📁</button>
      )}
      <button className={`invisible rounded px-1.5 py-0.5 text-xs group-hover:visible ${dark ? 'bg-gray-700 text-gray-200 hover:bg-gray-600' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
        disabled={busy} onClick={() => onResume(chat.session_id)}>Resume</button>
    </div>
  );
}

export default function ChatList({ dark, data, current, busy, newModel, newEffort, setNewModel, setNewEffort, onNew, onSelect, onResume, onFolder }) {
  const running = data?.running || [];
  const past = data?.past || [];
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex-none space-y-2">
        <button className={`${btnCls(dark, true, 'amber')} w-full`} disabled={busy} onClick={onNew}>+ New chat</button>
        <div className="flex gap-2">
          <select className={`${selectCls(dark)} min-w-0 flex-1`} value={newModel} onChange={(e) => setNewModel(e.target.value)} title="Model for the new chat">
            {MODELS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
          <select className={`${selectCls(dark)} min-w-0 flex-1`} value={newEffort} onChange={(e) => setNewEffort(e.target.value)} title="Effort for the new chat">
            {EFFORTS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
        </div>
      </div>
      <div className="mt-3 min-h-0 flex-1 space-y-3 overflow-y-auto">
        <div>
          <Heading dark={dark}>Running</Heading>
          {running.length === 0 && <div className={`px-2 text-xs ${dimCls(dark)}`}>No chats running</div>}
          {running.map((c) => (
            <Running key={c.window_id} dark={dark} chat={c} active={c.window_id === current} onSelect={onSelect} onFolder={onFolder} />
          ))}
        </div>
        {past.length > 0 && (
          <div>
            <Heading dark={dark}>Past chats</Heading>
            {past.map((c) => <Past key={c.session_id} dark={dark} chat={c} busy={busy} onResume={onResume} onFolder={onFolder} />)}
          </div>
        )}
      </div>
    </div>
  );
}
