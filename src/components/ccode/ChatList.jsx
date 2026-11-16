// Sidebar: start a chat, the running chats with what each is doing, and past chats to resume.
// Rename and archive (Nov 7 2026) are display-only: kept in this browser (pref ccode.labels), never sent to the server,
// so the chats, their tmux windows and transcripts are untouched. Keyed by chat id (shell windows by window id).
import { useRef, useState } from 'react';
import { MODELS, EFFORTS, fmtAge, chatKey, displayName } from './api.js';
import { btnCls, dimCls, strongCls, Heading } from '../hyperliquid/ui.jsx';
import ContextMeter, { fmtTokens } from './ContextMeter.jsx';

const STATES = {
  working: { dot: 'bg-amber-400 animate-pulse', label: 'working' },
  waiting: { dot: 'bg-sky-400', label: 'waiting for you' },
  idle: { dot: 'bg-gray-500', label: 'idle' },
};

function selectCls(dark) {
  return `text-xs rounded px-1.5 py-1 border ${dark ? 'bg-gray-900 border-gray-700 text-gray-200' : 'bg-white border-gray-300 text-gray-800'}`;
}

const iconCls = (dark) => `invisible flex-none rounded px-1 text-xs group-hover:visible ${dark ? 'text-gray-400 hover:bg-gray-600 hover:text-gray-100' : 'text-gray-500 hover:bg-gray-200 hover:text-gray-900'}`;

// The name, or an input while renaming. Enter or leaving the box saves; Esc cancels; an empty name restores the original.
function Name(props) {
  return <NameInner key={props.editing ? 'edit' : 'view'} {...props} />;   // fresh text each time editing starts
}

function NameInner({ dark, chat, labels, editing, onDone: done, className }) {
  const [text, setText] = useState(displayName(chat, labels));
  const finished = useRef(false);                 // Esc then blur must not save
  const onDone = (v) => { if (!finished.current) { finished.current = true; done(v); } };
  if (!editing) {
    return <span className={`block truncate text-sm ${className}`} title={chat.name !== displayName(chat, labels) ? `Original name: ${chat.name}` : undefined}>{displayName(chat, labels)}</span>;
  }
  return (
    <input autoFocus value={text} onChange={(e) => setText(e.target.value)} onClick={(e) => e.stopPropagation()}
      onFocus={(e) => e.target.select()}
      onKeyDown={(e) => { if (e.key === 'Enter') onDone(text); if (e.key === 'Escape') onDone(null); }}
      onBlur={() => onDone(text)}
      className={`block w-full rounded border px-1 py-0 text-sm ${dark ? 'border-gray-600 bg-gray-900 text-gray-100' : 'border-gray-300 bg-white text-gray-900'}`} />
  );
}

function RowActions({ dark, chat, archived, onRenameStart, onArchive }) {
  return (
    <>
      <button className={iconCls(dark)} title="Rename (only in this dashboard)"
        onClick={(e) => { e.stopPropagation(); onRenameStart(); }}>✎</button>
      <button className={iconCls(dark)} title={archived ? 'Move back to the list' : 'Archive: hide from the list (only in this dashboard; the chat keeps running)'}
        onClick={(e) => { e.stopPropagation(); onArchive(chatKey(chat), !archived); }}>{archived ? '↩' : '🗄'}</button>
    </>
  );
}

function Running({ dark, chat, labels, active, archived, onSelect, onFolder, onRename, onArchive }) {
  const [editing, setEditing] = useState(false);
  const s = STATES[chat.state] || STATES.idle;
  const done = (text) => { setEditing(false); if (text !== null) onRename(chatKey(chat), text.trim() === chat.name ? '' : text.trim()); };
  return (
    <div className={`group flex items-center gap-1.5 rounded px-2 py-1.5 cursor-pointer ${active ? (dark ? 'bg-gray-700' : 'bg-gray-100') : (dark ? 'hover:bg-gray-700/50' : 'hover:bg-gray-50')}`}
      onClick={() => onSelect(chat.window_id)} onDoubleClick={() => setEditing(true)}>
      <span className={`h-2 w-2 flex-none rounded-full ${s.dot}`} />
      <span className="min-w-0 flex-1">
        <Name dark={dark} chat={chat} labels={labels} editing={editing} onDone={done}
          className={active ? strongCls(dark) : (dark ? 'text-gray-300' : 'text-gray-700')} />
        <span className={`block text-[11px] ${dimCls(dark)}`}>
          {s.label}{chat.context?.tokens != null && ` · ${fmtTokens(chat.context.tokens)}`}
        </span>
        {chat.context && !archived && <span className="mt-1 block"><ContextMeter dark={dark} ctx={chat.context} /></span>}
      </span>
      {chat.folder && (
        <button className={iconCls(dark)} title="Open this chat's folder"
          onClick={(e) => { e.stopPropagation(); onFolder(chat.folder); }}>📁</button>
      )}
      <RowActions dark={dark} chat={chat} archived={archived} onRenameStart={() => setEditing(true)} onArchive={onArchive} />
    </div>
  );
}

function Past({ dark, chat, labels, busy, archived, onResume, onFolder, onRename, onArchive }) {
  const [editing, setEditing] = useState(false);
  const done = (text) => { setEditing(false); if (text !== null) onRename(chatKey(chat), text.trim() === chat.name ? '' : text.trim()); };
  return (
    <div className={`group flex items-center gap-1.5 rounded px-2 py-1 ${dark ? 'hover:bg-gray-700/50' : 'hover:bg-gray-50'}`}
      onDoubleClick={() => setEditing(true)}>
      <span className="min-w-0 flex-1">
        <Name dark={dark} chat={chat} labels={labels} editing={editing} onDone={done} className={dark ? 'text-gray-300' : 'text-gray-700'} />
        <span className={`block text-[11px] ${dimCls(dark)}`}>{fmtAge(chat.mtime)}</span>
      </span>
      {chat.folder && (
        <button className={iconCls(dark)} title="Open this chat's folder" onClick={() => onFolder(chat.folder)}>📁</button>
      )}
      <RowActions dark={dark} chat={chat} archived={archived} onRenameStart={() => setEditing(true)} onArchive={onArchive} />
      <button className={`invisible flex-none rounded px-1.5 py-0.5 text-xs group-hover:visible ${dark ? 'bg-gray-700 text-gray-200 hover:bg-gray-600' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
        disabled={busy} onClick={() => onResume(chat.session_id)}>Resume</button>
    </div>
  );
}

export default function ChatList({ dark, data, current, busy, labels, newModel, newEffort, setNewModel, setNewEffort,
  onNew, onSelect, onResume, onFolder, onRename, onArchive }) {
  const [showArchived, setShowArchived] = useState(false);
  const isArchived = (c) => !!labels?.archived?.[chatKey(c)];
  const running = (data?.running || []).filter((c) => !isArchived(c));
  const past = (data?.past || []).filter((c) => !isArchived(c));
  const archivedRunning = (data?.running || []).filter(isArchived);
  const archivedPast = (data?.past || []).filter(isArchived);
  const nArchived = archivedRunning.length + archivedPast.length;
  const rowProps = { dark, labels, onFolder, onRename, onArchive };
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
          {running.length === 0 && <div className={`px-2 text-xs ${dimCls(dark)}`}>{archivedRunning.length ? 'All running chats are archived' : 'No chats running'}</div>}
          {running.map((c) => (
            <Running key={c.window_id} chat={c} active={c.window_id === current} archived={false} onSelect={onSelect} {...rowProps} />
          ))}
        </div>
        {past.length > 0 && (
          <div>
            <Heading dark={dark}>Past chats</Heading>
            {past.map((c) => <Past key={c.session_id} chat={c} busy={busy} archived={false} onResume={onResume} {...rowProps} />)}
          </div>
        )}
        {nArchived > 0 && (
          <div>
            <button className={`flex w-full items-center gap-1 px-2 text-xs font-semibold uppercase tracking-wide ${dimCls(dark)} hover:underline`}
              onClick={() => setShowArchived((v) => !v)}>
              <span>{showArchived ? '▾' : '▸'}</span> Archived ({nArchived})
            </button>
            {showArchived && (
              <div className="mt-1 opacity-75">
                {archivedRunning.map((c) => (
                  <Running key={c.window_id} chat={c} active={c.window_id === current} archived onSelect={onSelect} {...rowProps} />
                ))}
                {archivedPast.map((c) => <Past key={c.session_id} chat={c} busy={busy} archived onResume={onResume} {...rowProps} />)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
