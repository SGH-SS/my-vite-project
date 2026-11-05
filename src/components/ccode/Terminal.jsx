/**
 * The live terminal: xterm.js drawing the real Claude Code screen from tmux on the server, over a WebSocket to the
 * backend. Bytes go straight through both ways; the server's tmux view of the current chat is what you see. Closing
 * or losing the connection only detaches: chats keep running, and this reconnects by itself.
 */
import { useEffect, useRef } from 'react';
import { Terminal as XTerm } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';
import { CCODE_WS } from './api.js';

const THEME = {
  background: '#0b0f17', foreground: '#d7dce4', cursor: '#d7dce4', cursorAccent: '#0b0f17',
  selectionBackground: '#3b4a63',
  black: '#1b2230', red: '#f47174', green: '#6fcf97', yellow: '#e8c26a', blue: '#6ea8fe', magenta: '#c792ea',
  cyan: '#5fd3e6', white: '#d7dce4', brightBlack: '#5c6780', brightRed: '#ff8f91', brightGreen: '#8fe3b0',
  brightYellow: '#f4d58a', brightBlue: '#92bfff', brightMagenta: '#dcb2f5', brightCyan: '#86e4f2', brightWhite: '#f4f6fa',
};
const RECONNECT_MS = 1500;

function base64ToText(b64) {
  return new TextDecoder().decode(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)));
}

export default function Terminal({ focusKey, onConnection }) {
  const box = useRef(null);
  const term = useRef(null);
  const report = useRef(onConnection);
  report.current = onConnection;

  useEffect(() => {
    const encoder = new TextEncoder();
    const xterm = new XTerm({
      fontFamily: '"Cascadia Mono", Consolas, "Courier New", monospace', fontSize: 13, lineHeight: 1.15,
      cursorBlink: true, scrollback: 1000, theme: THEME,
    });
    const fit = new FitAddon();
    xterm.loadAddon(fit);
    xterm.open(box.current);
    fit.fit();
    term.current = xterm;

    let ws = null;
    let closed = false;
    let retry = null;
    const send = (data) => { if (ws?.readyState === WebSocket.OPEN) ws.send(data); };
    const sendText = (text) => send(encoder.encode(text));
    const sendSize = () => send(JSON.stringify({ type: 'resize', cols: xterm.cols, rows: xterm.rows }));

    const connect = () => {
      ws = new WebSocket(`${CCODE_WS}?cols=${xterm.cols}&rows=${xterm.rows}`);
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => { report.current?.('connected'); xterm.focus(); };
      ws.onmessage = (e) => {
        if (typeof e.data === 'string') {
          const m = JSON.parse(e.data);
          if (m.type === 'error') report.current?.('error', m.message);
        } else {
          xterm.write(new Uint8Array(e.data));
        }
      };
      ws.onclose = () => {
        if (closed) return;
        report.current?.('reconnecting');
        retry = setTimeout(connect, RECONNECT_MS);
      };
    };
    connect();

    const typing = xterm.onData(sendText);
    const mouse = xterm.onBinary((d) => send(Uint8Array.from(d, (c) => c.charCodeAt(0))));

    // tmux copies a mouse selection with OSC 52: put it on the PC's clipboard
    const clipboard = xterm.parser.registerOscHandler(52, (data) => {
      const b64 = data.split(';')[1];
      if (b64) navigator.clipboard?.writeText(base64ToText(b64)).catch(() => {});
      return true;
    });
    // Ctrl+V pastes (the browser's paste event, which xterm picks up), rather than sending a control character
    xterm.attachCustomKeyEventHandler((e) => !(e.type === 'keydown' && e.ctrlKey && e.key.toLowerCase() === 'v'));

    let timer = null;
    const observer = new ResizeObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (!box.current?.offsetWidth) return;
        fit.fit();
        sendSize();
      }, 80);
    });
    observer.observe(box.current);

    return () => {
      closed = true;
      clearTimeout(retry);
      clearTimeout(timer);
      observer.disconnect();
      typing.dispose();
      mouse.dispose();
      clipboard.dispose();
      ws?.close();
      xterm.dispose();
      term.current = null;
    };
  }, []);

  useEffect(() => { term.current?.focus(); }, [focusKey]);

  return <div ref={box} className="h-full w-full" style={{ background: THEME.background, padding: '6px 0 0 8px' }} />;
}
