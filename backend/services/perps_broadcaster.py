"""
perps_broadcaster.py — PG LISTEN → WebSocket broadcaster for BTC/ETH/SPX perps
==============================================================================
Generic per-asset broadcaster that mirrors sol_broadcaster.py but routes
NOTIFY payloads to per-asset WebSocket client pools.

Each collector publishes on three channels:
  {asset}_ticks, {asset}_trades, {asset}_marks   (asset ∈ {btc, eth, spx})

WebSocket clients connect to /api/perp/{asset}/ws and only receive
broadcasts whose channel name starts with that asset.

Usage:
    from services.perps_broadcaster import perps_broadcaster

    # FastAPI lifespan
    await perps_broadcaster.start()
    await perps_broadcaster.stop()

    # WS endpoint
    await perps_broadcaster.connect("btc", ws)
    perps_broadcaster.disconnect("btc", ws)
"""

import asyncio
import json
import logging

import psycopg

from fastapi import WebSocket

logger = logging.getLogger(__name__)

DB_DSN = "host=localhost port=5433 dbname=trading_data user=postgres password=postgres"

ASSETS = ("btc", "eth", "spx")
CHANNELS = tuple(f"{a}_{kind}" for a in ASSETS for kind in ("ticks", "trades", "marks"))


class PerpsBroadcaster:
    def __init__(self):
        self._clients: dict[str, list[WebSocket]] = {a: [] for a in ASSETS}
        self._task: asyncio.Task | None = None

    def client_count(self, asset: str | None = None) -> int:
        if asset is None:
            return sum(len(v) for v in self._clients.values())
        return len(self._clients.get(asset, []))

    async def connect(self, asset: str, ws: WebSocket):
        if asset not in self._clients:
            await ws.close(code=1008)
            return
        await ws.accept()
        self._clients[asset].append(ws)
        logger.info("PERP[%s] WS client connected (%d total)",
                    asset, len(self._clients[asset]))

    def disconnect(self, asset: str, ws: WebSocket):
        if asset in self._clients and ws in self._clients[asset]:
            self._clients[asset].remove(ws)
        logger.info("PERP[%s] WS client disconnected (%d total)",
                    asset, len(self._clients.get(asset, [])))

    async def _broadcast(self, asset: str, message: str):
        clients = self._clients.get(asset, [])
        if not clients:
            return
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(asset, ws)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self):
        self._task = asyncio.create_task(self._listen_loop())
        logger.info("PerpsBroadcaster started — PG LISTEN loop running")

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        for asset, clients in self._clients.items():
            for ws in clients[:]:
                try:
                    await ws.close()
                except Exception:
                    pass
            clients.clear()
        logger.info("PerpsBroadcaster stopped")

    # ── PG LISTEN loop ────────────────────────────────────────────────────

    async def _listen_loop(self):
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
                        # Route by channel prefix → asset
                        asset = notify.channel.split("_", 1)[0]
                        if asset not in self._clients:
                            continue
                        if not self._clients[asset]:
                            continue
                        try:
                            payload = json.loads(notify.payload)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        msg = json.dumps({
                            "channel": notify.channel,
                            "data": payload,
                        })
                        await self._broadcast(asset, msg)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("PG LISTEN (perps) error: %s — reconnecting in 2s", exc)
                await asyncio.sleep(2)


perps_broadcaster = PerpsBroadcaster()
