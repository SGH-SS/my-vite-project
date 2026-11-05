"""One SSH connection from this backend to the Tokyo server, as the sandbox user `research`.

Uses the `hlresearch` alias from ~/.ssh/config (user + key) and checks the server against ~/.ssh/known_hosts. The
terminal, the chat controls and the file browser all share this connection, which reopens itself after a drop. Chats
live in tmux on the server, so losing this connection never affects them.

tmux layout: every chat is a window of the session `claude`. The dashboard attaches through its own grouped session
`ccode-ui`, which shares those windows but has its own current window, so switching chats here does not move a
PowerShell attached to `claude`. Its status bar is off (the dashboard shows that information) and mouse is on (scroll).
"""
import asyncio
import os
import shlex

import asyncssh

HOST = "hlresearch"
SSH_DIR = os.path.expanduser("~/.ssh")
CLAUDE = "/home/research/.local/bin/claude"
SNAPSHOT = "/usr/bin/python3 /research/code/ccode_snapshot.py"
PREVIEW = "/research/venv/bin/python /research/code/ccode_preview.py"
RESTORE_DEFAULTS = "/usr/bin/python3 /research/code/ccode_defaults.py"

_conn = None
_lock = asyncio.Lock()
_sftp = None
_sftp_conn = None


async def connection():
    global _conn
    async with _lock:
        if _conn is None or _conn.is_closed():
            _conn = await asyncssh.connect(HOST, config=[os.path.join(SSH_DIR, "config")],
                                           known_hosts=os.path.join(SSH_DIR, "known_hosts"),
                                           keepalive_interval=15, keepalive_count_max=4, connect_timeout=15)
        return _conn


async def run(cmd, timeout=20):
    """Run one command; a dead connection is reopened once. Returns (exit status, stdout, stderr)."""
    global _conn
    for attempt in (1, 2):
        try:
            conn = await connection()
            r = await conn.run(cmd, check=False, timeout=timeout)
            return r.exit_status, r.stdout or "", r.stderr or ""
        except (asyncssh.ConnectionLost, asyncssh.DisconnectError, ConnectionError, OSError):
            if attempt == 2:
                raise
            _conn = None


async def ensure_sessions():
    """Create the `claude` session if it is gone and the dashboard's own view of it, with the view's options."""
    await run("tmux has-session -t claude 2>/dev/null || tmux new-session -d -s claude -c /research; "
              "tmux has-session -t ccode-ui 2>/dev/null || tmux new-session -d -t claude -s ccode-ui; "
              "tmux set-option -t ccode-ui status off >/dev/null; tmux set-option -t ccode-ui mouse on >/dev/null")


async def open_terminal(cols, rows):
    await ensure_sessions()
    conn = await connection()
    return await conn.create_process("tmux -u attach-session -t ccode-ui", term_type="xterm-256color",
                                     term_size=(cols, rows), encoding=None)


async def new_window(claude_args):
    """Start Claude in a new tmux window in /research; returns the window id."""
    await ensure_sessions()
    command = shlex.join(["tmux", "new-window", "-d", "-P", "-F", "#{window_id}", "-t", "claude:", "-c", "/research",
                          shlex.join([CLAUDE, *claude_args])])
    status, out, err = await run(command)
    if status != 0:
        raise RuntimeError(err.strip() or "tmux new-window failed")
    return out.strip()


async def select_window(window_id):
    await ensure_sessions()
    await run(f"tmux select-window -t ccode-ui:{window_id}")


async def send_line(window_id, text):
    """Type one line into a chat, as if the user had typed it."""
    await run(f"tmux send-keys -t {window_id} -l -- {shlex.quote(text)} \\; send-keys -t {window_id} Enter")


async def screen(window_id):
    """A chat's recent screen text, with some history so a line that just scrolled off still counts."""
    _, out, _ = await run(f"tmux capture-pane -p -S -300 -t {window_id}")
    return out


async def send_and_confirm(window_id, line, confirmation, wait_s=4.0):
    """Type a slash command and watch the screen for Claude's confirmation. If the line vanished with no effect it is
    sent once more; if it is still sitting in the input area (the chat is busy) it is left there. True = confirmed."""
    seen_before = (await screen(window_id)).count(confirmation)
    for _ in (1, 2):
        await send_line(window_id, line)
        text = ""
        for _ in range(int(wait_s / 0.5)):
            await asyncio.sleep(0.5)
            text = await screen(window_id)
            if text.count(confirmation) > seen_before:
                return True
        if line in "\n".join(text.splitlines()[-8:]):
            return False
    return False


async def restore_defaults_soon():
    """/model and /effort also save themselves as the default for new chats; keep the user's defaults."""
    await asyncio.sleep(3)
    await run(RESTORE_DEFAULTS)


async def sftp():
    """The shared file channel, reopened with the connection: opening one per click costs about a second."""
    global _sftp, _sftp_conn
    conn = await connection()
    if _sftp is None or _sftp_conn is not conn:
        _sftp, _sftp_conn = await conn.start_sftp_client(), conn
    return _sftp
