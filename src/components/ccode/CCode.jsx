/**
 * Server CCode: Claude Code on the Tokyo server, from Daygent. Every chat is a tmux window on the server; the main
 * panel is the real terminal of the current one, and the sidebar starts, switches and resumes chats and browses the
 * server's files. The same chats can be open in PowerShell at the same time. Backend: /api/ccode.
 */
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { cGet, cPost, winPath, MODELS, EFFORTS } from './api.js';
import { readPref, writePref } from '../hyperliquid/api.js';
import { btnCls, dimCls, strongCls, cardCls, ErrorBanner, Confirm } from '../hyperliquid/ui.jsx';
import Terminal from './Terminal.jsx';
import ChatList from './ChatList.jsx';
import FileBrowser from './FileBrowser.jsx';
import FilePreview from './FilePreview.jsx';

const POLL_MS = 3000;
const AFTER_CHANGE_MS = 1200;

function usePanelHeight(ref) {
  const [height, setHeight] = useState(600);
  useLayoutEffect(() => {
    const update = () => {
      if (ref.current) setHeight(Math.max(460, window.innerHeight - ref.current.getBoundingClientRect().top - 20));
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, [ref]);
  return height;
}

export default function CCode({ isDarkMode: dark }) {
  const root = useRef(null);
  const height = usePanelHeight(root);
  const [data, setData] = useState(null);
  const [current, setCurrent] = useState(null);
  const [err, setErr] = useState(null);
  const [conn, setConn] = useState({ state: 'connecting', message: null });
  const [busy, setBusy] = useState(false);
  const [confirmEnd, setConfirmEnd] = useState(false);
  const [newModel, setNewModel] = useState(() => readPref('ccode.newModel', 'opus'));
  const [newEffort, setNewEffort] = useState(() => readPref('ccode.newEffort', 'high'));
  const [settings, setSettings] = useState(() => readPref('ccode.settings', {}));   // window id -> { model, effort } set from here
  const [folder, setFolder] = useState('/research');
  const [openFile, setOpenFile] = useState(null);

  const refresh = async () => {
    try {
      const r = await cGet('/chats');
      setData(r);
      setCurrent(r.current);
      setErr(null);
    } catch (e) { setErr(e.message); }
  };
  useEffect(() => { refresh(); const id = setInterval(refresh, POLL_MS); return () => clearInterval(id); }, []);
  useEffect(() => { writePref('ccode.newModel', newModel); writePref('ccode.newEffort', newEffort); }, [newModel, newEffort]);

  const act = async (fn) => {
    setBusy(true);
    try { await fn(); setErr(null); } catch (e) { setErr(e.message); }
    setBusy(false);
    setTimeout(refresh, AFTER_CHANGE_MS);
  };

  const chat = data?.running.find((c) => c.window_id === current) || null;
  const remember = (windowId, change) => {
    setSettings((m) => {
      const next = { ...m, [windowId]: { ...m[windowId], ...change } };
      writePref('ccode.settings', next);
      return next;
    });
  };

  const newChat = () => act(async () => {
    const r = await cPost('/chats', { model: newModel, effort: newEffort });
    remember(r.window_id, { model: newModel, effort: newEffort });
    setCurrent(r.window_id);
  });
  const select = (id) => { setCurrent(id); act(() => cPost(`${winPath(id)}/select`)); };
  const resume = (sid) => act(async () => { setCurrent((await cPost(`/chats/${sid}/resume`)).window_id); });
  const changeModel = (model) => act(async () => {
    await cPost(`${winPath(chat.window_id)}/model`, { model });
    remember(chat.window_id, { model });
  });
  const changeEffort = (effort) => act(async () => {
    await cPost(`${winPath(chat.window_id)}/effort`, { effort });
    remember(chat.window_id, { effort });
  });
  const endChat = () => act(async () => { await cPost(`${winPath(chat.window_id)}/end`); setConfirmEnd(false); });
  const showFolder = (path) => setFolder(path);

  const selectCls = `text-xs rounded px-1.5 py-1 border ${dark ? 'bg-gray-900 border-gray-700 text-gray-200' : 'bg-white border-gray-300 text-gray-800'}`;
  const chosen = (chat && settings[chat.window_id]) || {};
  const modelValue = chosen.model || chat?.model || '';
  const effortValue = chosen.effort || '';
  const guardPaused = data && data.guard !== 'ok' && data.guard !== 'unknown';

  return (
    <div ref={root} className="flex gap-3" style={{ height }}>
      <aside className="flex w-72 flex-none flex-col gap-3">
        <div className={`${cardCls(dark)} min-h-0 flex-[5] !p-3`}>
          <ChatList dark={dark} data={data} current={current} busy={busy} newModel={newModel} newEffort={newEffort}
            setNewModel={setNewModel} setNewEffort={setNewEffort} onNew={newChat} onSelect={select} onResume={resume} onFolder={showFolder} />
        </div>
        <div className={`${cardCls(dark)} min-h-0 flex-[4] !p-3`}>
          <FileBrowser dark={dark} path={folder} onGo={setFolder} selected={openFile} onOpen={setOpenFile} />
        </div>
      </aside>

      <section className={`flex min-w-0 flex-1 flex-col overflow-hidden rounded-lg border ${dark ? 'border-gray-700' : 'border-gray-200'}`}>
        <div className={`flex flex-none items-center gap-2 px-3 py-2 ${dark ? 'bg-gray-800' : 'bg-white'}`}>
          <span className={`min-w-0 flex-1 truncate text-sm font-medium ${strongCls(dark)}`}>{chat ? chat.name : 'Server CCode'}</span>
          {chat?.folder && <button className={`text-xs hover:underline ${dimCls(dark)}`} onClick={() => showFolder(chat.folder)}>Folder</button>}
          {chat?.kind === 'chat' && (
            <>
              <select className={selectCls} value={modelValue} disabled={busy} title="Change this chat's model" onChange={(e) => changeModel(e.target.value)}>
                {!modelValue && <option value="">Model</option>}
                {MODELS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
              <select className={selectCls} value={effortValue} disabled={busy} title="Change this chat's effort" onChange={(e) => changeEffort(e.target.value)}>
                {!effortValue && <option value="">Effort</option>}
                {EFFORTS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
              <button className={btnCls(dark)} disabled={busy} onClick={() => setConfirmEnd(true)}>End</button>
            </>
          )}
        </div>
        {confirmEnd && chat && (
          <div className="flex-none px-3 pb-3">
            <Confirm dark={dark} title={`End "${chat.name}"?`} lines={['Closes this chat. Its folder and transcript stay, and you can resume it from Past chats.']}
              yesLabel="End chat" tone="red" busy={busy} onYes={endChat} onNo={() => setConfirmEnd(false)} />
          </div>
        )}
        {(err || conn.state !== 'connected' || guardPaused) && (
          <div className="flex-none px-3 pb-2">
            <ErrorBanner dark={dark} error={err} />
            {conn.state === 'reconnecting' && <div className={`mt-1 text-xs ${dimCls(dark)}`}>Connection lost, reconnecting… Your chats keep running on the server.</div>}
            {conn.state === 'connecting' && <div className={`mt-1 text-xs ${dimCls(dark)}`}>Connecting to the server…</div>}
            {conn.state === 'error' && <ErrorBanner dark={dark} error={conn.message} />}
            {guardPaused && <div className={`mt-1 text-xs ${dark ? 'text-amber-300' : 'text-amber-700'}`}>The node is lagging, so running research jobs are paused until it recovers.</div>}
          </div>
        )}
        <div className="min-h-0 flex-1">
          <Terminal focusKey={current} onConnection={(state, message) => setConn({ state, message })} />
        </div>
        {openFile && (
          <div className="h-[45%] flex-none">
            <FilePreview dark={dark} path={openFile} onClose={() => setOpenFile(null)} />
          </div>
        )}
      </section>
    </div>
  );
}
