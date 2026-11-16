"""Server CCode dashboard: Claude Code on the Tokyo server, from Daygent (/api/ccode).

Live terminal over a WebSocket, chat controls through tmux, read-only file browsing over SFTP. This PC and the local
Daygent page only (see services/ccode_guard.py).
"""
import asyncio
import csv
import io
import json
import logging
import posixpath
import shlex
import stat as statmod

import asyncssh
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, WebSocketException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.requests import HTTPConnection

from services import ccode_ssh as ssh
from services.ccode_chats import build_chats
from services.ccode_guard import (
    chat_id_ok, client_allowed, effort_ok, model_ok, origin_allowed, path_allowed, window_id_ok,
)

logger = logging.getLogger(__name__)
PREVIEW_BYTES = 2_000_000
TABLE_BYTES = 512_000
TABLE_ROWS = 200
LIST_LIMIT = 3000
IMAGES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
          ".webp": "image/webp", ".svg": "image/svg+xml"}


def local_only(conn: HTTPConnection):
    host = conn.client.host if conn.client else None
    if client_allowed(host) and origin_allowed(conn.headers.get("origin")):
        return
    logger.warning("ccode: refused %s origin=%s", host, conn.headers.get("origin"))
    if conn.scope["type"] == "websocket":
        raise WebSocketException(code=1008)
    raise HTTPException(status_code=403, detail="This PC only")


router = APIRouter(prefix="/api/ccode", tags=["ccode"], dependencies=[Depends(local_only)])


class NewChat(BaseModel):
    model: str = "opus"
    effort: str = "high"


class ModelChange(BaseModel):
    model: str


class EffortChange(BaseModel):
    effort: str


def _window(window_id):
    if not window_id_ok(window_id):
        raise HTTPException(status_code=400, detail="Bad window id")
    return window_id


def _path(path):
    if not path_allowed(path):
        raise HTTPException(status_code=403, detail="This path can't be opened here")
    return posixpath.normpath(path)


def _unreachable(e):
    logger.warning("ccode: server unreachable: %r", e)
    return HTTPException(status_code=503, detail=f"Can't reach the server: {e}")


# ── Terminal ──────────────────────────────────────────────────────────────


@router.websocket("/terminal")
async def terminal(ws: WebSocket, cols: int = 120, rows: int = 40):
    """Raw terminal bytes both ways; text frames carry control messages ({"type": "resize", ...}). Closing this only
    detaches from tmux: the chats keep running."""
    await ws.accept()
    try:
        proc = await ssh.open_terminal(cols, rows)
    except (asyncssh.Error, OSError) as e:
        await ws.send_text(json.dumps({"type": "error", "message": f"Can't reach the server: {e}"}))
        await ws.close()
        return

    async def to_browser():
        while True:
            data = await proc.stdout.read(65536)
            if not data:
                return
            await ws.send_bytes(data)

    async def from_browser():
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes"):
                proc.stdin.write(msg["bytes"])
            elif msg.get("text"):
                m = json.loads(msg["text"])
                if m.get("type") == "resize":
                    proc.change_terminal_size(int(m["cols"]), int(m["rows"]))

    tasks = [asyncio.create_task(to_browser()), asyncio.create_task(from_browser())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        pass
    finally:
        for t in tasks:
            t.cancel()
        proc.close()
        try:
            await ws.close()
        except RuntimeError:
            pass


# ── Chats ─────────────────────────────────────────────────────────────────


@router.get("/chats")
async def chats():
    try:
        status, out, err = await ssh.run(ssh.SNAPSHOT)
    except (asyncssh.Error, OSError) as e:
        raise _unreachable(e)
    if status != 0:
        raise HTTPException(status_code=500, detail=err.strip()[:300] or "Snapshot failed")
    return build_chats(json.loads(out))


@router.get("/health")
async def health():
    """One on-demand check of the server (disks + growth, CPU, memory, pressure, IO, node lag). Never polled."""
    try:
        status, out, err = await ssh.run(ssh.HEALTH, timeout=45)
    except (asyncio.TimeoutError, asyncssh.TimeoutError):
        raise HTTPException(status_code=504, detail="The server didn't answer in 45 s. If the node is lagging, "
                                                    "nodeguard pauses research processes, this check included.")
    except (asyncssh.Error, OSError) as e:
        raise _unreachable(e)
    if status != 0:
        raise HTTPException(status_code=500, detail=err.strip()[-300:] or "Health check failed")
    return json.loads(out)


@router.post("/chats")
async def new_chat(body: NewChat):
    if not model_ok(body.model) or not effort_ok(body.effort):
        raise HTTPException(status_code=400, detail="Unknown model or effort")
    try:
        window_id = await ssh.new_window(["--model", body.model, "--effort", body.effort])
        await ssh.select_window(window_id)
    except (asyncssh.Error, OSError, RuntimeError) as e:
        raise _unreachable(e)
    return {"window_id": window_id}


@router.post("/chats/{session_id}/resume")
async def resume_chat(session_id: str):
    if not chat_id_ok(session_id):
        raise HTTPException(status_code=400, detail="Bad chat id")
    try:
        window_id = await ssh.new_window(["--resume", session_id])
        await ssh.select_window(window_id)
    except (asyncssh.Error, OSError, RuntimeError) as e:
        raise _unreachable(e)
    return {"window_id": window_id}


@router.post("/windows/{window_id}/select")
async def select(window_id: str):
    try:
        await ssh.select_window(_window(window_id))
    except (asyncssh.Error, OSError) as e:
        raise _unreachable(e)
    return {"ok": True}


async def _change(window_id, line, confirmation):
    """Type a /model or /effort line and report whether Claude confirmed it on screen."""
    try:
        confirmed = await ssh.send_and_confirm(_window(window_id), line, confirmation)
    except (asyncssh.Error, OSError) as e:
        raise _unreachable(e)
    asyncio.create_task(ssh.restore_defaults_soon())
    if not confirmed:
        raise HTTPException(status_code=409, detail="Claude didn't confirm the change. If it's busy, try again when it's done.")
    return {"ok": True}


@router.post("/windows/{window_id}/model")
async def set_model(window_id: str, body: ModelChange):
    if not model_ok(body.model):
        raise HTTPException(status_code=400, detail="Unknown model")
    return await _change(window_id, f"/model {body.model}", "Set model to")


@router.post("/windows/{window_id}/effort")
async def set_effort(window_id: str, body: EffortChange):
    if not effort_ok(body.effort):
        raise HTTPException(status_code=400, detail="Unknown effort")
    return await _change(window_id, f"/effort {body.effort}", "Set effort level to")


@router.post("/windows/{window_id}/end")
async def end_chat(window_id: str):
    try:
        await ssh.send_line(_window(window_id), "/exit")
    except (asyncssh.Error, OSError) as e:
        raise _unreachable(e)
    return {"ok": True}


# ── Files (read-only) ─────────────────────────────────────────────────────


@router.get("/files")
async def list_files(path: str = Query("/research")):
    path = _path(path)
    try:
        sftp = await ssh.sftp()
        entries = []
        for e in await sftp.readdir(path):
            if e.filename in (".", ".."):
                continue
            full = posixpath.join(path, e.filename)
            if not path_allowed(full):
                continue
            mode = e.attrs.permissions or 0
            is_dir = statmod.S_ISDIR(mode)
            if statmod.S_ISLNK(mode):
                is_dir = await sftp.isdir(full)
            entries.append({"name": e.filename, "path": full, "is_dir": is_dir, "size": e.attrs.size or 0,
                            "mtime": e.attrs.mtime})
    except asyncssh.SFTPError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (asyncssh.Error, OSError) as e:
        raise _unreachable(e)
    entries.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
    return {"path": path, "entries": entries[:LIST_LIMIT], "truncated": len(entries) > LIST_LIMIT}


async def _read_start(path, n):
    sftp = await ssh.sftp()
    size = (await sftp.stat(path)).size or 0
    async with sftp.open(path, "rb") as f:
        return size, await f.read(n)


def _as_table(data, delimiter, cut_off):
    text = data.decode("utf-8", "replace")
    lines = text.splitlines()
    if cut_off and len(lines) > 1:
        lines = lines[:-1]                  # the last line may be cut in half
    rows = [r for _, r in zip(range(TABLE_ROWS + 1), csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter))]
    return {"columns": [{"name": c, "type": ""} for c in (rows[0] if rows else [])], "rows": rows[1:]}


@router.get("/file")
async def preview(path: str):
    """What the preview panel shows; `kind` says how: text, table, image or binary."""
    path = _path(path)
    ext = posixpath.splitext(path)[1].lower()
    try:
        if ext in IMAGES:
            return {"kind": "image", "path": path}
        if ext == ".parquet":
            status, out, err = await ssh.run(f"{ssh.PREVIEW} {shlex.quote(path)} 50", timeout=60)
            if status != 0:
                raise HTTPException(status_code=422, detail=err.strip()[-300:] or "Can't read this Parquet file")
            return {"kind": "table", **json.loads(out)}
        if ext in (".csv", ".tsv"):
            size, data = await _read_start(path, TABLE_BYTES)
            return {"kind": "table", "size": size, **_as_table(data, "\t" if ext == ".tsv" else ",", size > TABLE_BYTES)}
        size, data = await _read_start(path, PREVIEW_BYTES)
    except asyncssh.SFTPError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (asyncssh.Error, OSError) as e:
        raise _unreachable(e)
    if b"\0" in data[:8192]:
        return {"kind": "binary", "size": size}
    return {"kind": "text", "size": size, "text": data.decode("utf-8", "replace"), "truncated": size > PREVIEW_BYTES}


@router.get("/download")
async def download(path: str, inline: bool = False):
    path = _path(path)
    try:
        sftp = await ssh.sftp()
        f = await sftp.open(path, "rb")
    except asyncssh.SFTPError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (asyncssh.Error, OSError) as e:
        raise _unreachable(e)

    async def body():
        try:
            while chunk := await f.read(1 << 20):
                yield chunk
        finally:
            await f.close()

    name = posixpath.basename(path).replace('"', "")
    ctype = IMAGES.get(posixpath.splitext(path)[1].lower(), "application/octet-stream") if inline else "application/octet-stream"
    disposition = "inline" if inline else f'attachment; filename="{name}"'
    return StreamingResponse(body(), media_type=ctype, headers={"Content-Disposition": disposition})
