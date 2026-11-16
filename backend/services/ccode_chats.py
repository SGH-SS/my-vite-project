"""Turns the server's snapshot (tokoyo_server/research/ccode_snapshot.py) into the dashboard's chat list.

A running chat is a tmux window whose process tree contains a Claude Code process recorded in ~/.claude/sessions.
Its workspace folder is /research/work/<YYYYMMDD-HHMM>-<first 8 of the chat id>[-topic]. Past chats are transcripts
that are not running now.
"""
WORK = "/research/work"
# Context window of every chat on the server. Opus 5.5 chats auto-compacted at up to 898k tokens (Nov 6 2026), so the
# server's chats run with the 1M window; Claude Code does not record the window in the transcript.
WINDOW = 1_000_000


def _folder_of(session_id, folders):
    key = session_id[:8]
    for f in folders:
        p = f.split("-", 3)
        if len(p) >= 3 and p[2] == key:
            return f
    return None


def _name(folder, title, fallback):
    if folder:
        p = folder.split("-", 3)
        if len(p) == 4 and p[3]:
            return p[3]
    return title or fallback


def _model_alias(model):
    """claude-opus-5-5 -> opus (the names the dashboard's model menu uses)"""
    for alias in ("fable", "opus", "sonnet", "haiku"):
        if alias in (model or ""):
            return alias
    return None


def _context(c):
    """the snapshot's context record -> what the sidebar shows (None when the snapshot has none for this chat)"""
    if not c:
        return None
    tokens = c.get("tokens")
    return {"tokens": tokens, "window": WINDOW, "pct": None if tokens is None else round(100 * tokens / WINDOW, 1),
            "compactions": c.get("compactions") or 0, "last_compact": c.get("last_compact")}


def _session_by_pane(snapshot):
    """pane PID -> session record, walking up from each Claude process to the window's shell"""
    panes = {w["pane_pid"] for w in snapshot.get("windows") or []}
    parents = snapshot.get("parents") or {}
    out = {}
    for s in snapshot.get("sessions") or []:
        pid = s["pid"]
        for _ in range(32):
            if pid in panes:
                out[pid] = s
                break
            pid = parents.get(str(pid))
            if pid is None:
                break
    return out


def build_chats(snapshot):
    folders = snapshot.get("folders") or []
    titles = {t["session_id"]: t.get("title", "") for t in snapshot.get("transcripts") or []}
    statuses = snapshot.get("statuses") or {}
    by_pane = _session_by_pane(snapshot)
    running = []
    for w in snapshot.get("windows") or []:
        s = by_pane.get(w["pane_pid"])
        if s is None:
            running.append({"window_id": w["id"], "index": w["index"], "kind": "shell", "session_id": None,
                            "name": "shell", "folder": None, "state": "idle", "model": None, "context": None})
            continue
        sid = s["session_id"]
        folder = _folder_of(sid, folders)
        running.append({"window_id": w["id"], "index": w["index"], "kind": "chat", "session_id": sid,
                        "name": _name(folder, titles.get(sid), "new chat"),
                        "folder": f"{WORK}/{folder}" if folder else None,
                        "state": (statuses.get(sid) or {}).get("state", "idle"),
                        "model": _model_alias((snapshot.get("models") or {}).get(sid)),
                        "context": _context((snapshot.get("context") or {}).get(sid))})
    live = {r["session_id"] for r in running}
    past = []
    for t in sorted(snapshot.get("transcripts") or [], key=lambda x: -x.get("mtime", 0)):
        sid = t["session_id"]
        folder = _folder_of(sid, folders)
        if sid in live or not (t.get("title") or folder):
            continue
        past.append({"session_id": sid, "name": _name(folder, t.get("title"), sid[:8]),
                     "folder": f"{WORK}/{folder}" if folder else None, "mtime": t.get("mtime")})
    return {"running": running, "past": past, "guard": snapshot.get("guard", "unknown"),
            "current": snapshot.get("current")}
