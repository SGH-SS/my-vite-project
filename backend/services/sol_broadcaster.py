"""
sol_broadcaster.py — PG LISTEN → WebSocket broadcaster for SOL-PERP
====================================================================
Maintains a single async psycopg connection that LISTENs on three
Postgres channels (sol_ticks, sol_trades, sol_marks).  When a NOTIFY
arrives (fired by collector.py after each flush), the payload is
forwarded to every connected WebSocket client.

Usage:
    from services.sol_broadcaster import broadcaster

    # In FastAPI lifespan:
    await broadcaster.start()   # on startup
    await broadcaster.stop()    # on shutdown

    # In WebSocket endpoint:
    await broadcaster.connect(ws)
    ...
    broadcaster.disconnect(ws)
"""

import asyncio
import json
import logging

import psycopg

from fastapi import WebSocket

logger = logging.getLogger(__name__)

DB_DSN = "host=localhost port=5433 dbname=trading_data user=postgres password=postgres"

CHANNELS = ("sol_ticks", "sol_trades", "sol_marks")


class SolBroadcaster:
    def __init__(self):
        self._clients: list[WebSocket] = []
        self._task: asyncio.Task | None = None

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.append(ws)
        logger.info("SOL WS client connected  (%d total)", len(self._clients))

    def disconnect(self, ws: WebSocket):
        if ws in self._clients:
            self._clients.remove(ws)
        logger.info("SOL WS client disconnected  (%d total)", len(self._clients))

    async def broadcast(self, message: str):
        dead: list[WebSocket] = []
        for ws in self._clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self):
        self._task = asyncio.create_task(self._listen_loop())
        logger.info("SolBroadcaster started — PG LISTEN loop running")

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        for ws in self._clients[:]:
            try:
                await ws.close()
            except Exception:
                pass
        self._clients.clear()
        logger.info("SolBroadcaster stopped")

    # ── PG LISTEN loop ────────────────────────────────────────────────────

    async def _listen_loop(self):
        """Reconnecting loop: connect to PG, LISTEN, forward NOTIFYs."""
        while True:
            try:
                aconn = await psycopg.AsyncConnection.connect(
                    DB_DSN, autocommit=True
                )
                async with aconn:
                    for ch in CHANNELS:
                        await aconn.execute(f"LISTEN {ch}")
                    logger.info("PG LISTEN active on %s", ", ".join(CHANNELS))

                    async for notify in aconn.notifies():
                        if not self._clients:
                            continue
                        try:
                            payload = json.loads(notify.payload)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        msg = json.dumps({
                            "channel": notify.channel,
                            "data": payload,
                        })
                        await self.broadcast(msg)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("PG LISTEN error: %s — reconnecting in 2s", exc)
                await asyncio.sleep(2)


broadcaster = SolBroadcaster()
