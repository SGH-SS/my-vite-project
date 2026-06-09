"""
SOL-PERP API Router
===================
Endpoints for the Hyperliquid SOL perpetual futures dashboard.
Includes a WebSocket endpoint that pushes live ticks/trades/marks
via PG NOTIFY → broadcaster instead of client-side polling.
"""

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import text
from datetime import datetime, timezone
from pathlib import Path
import asyncio
import json
import logging
import statistics as _stats
import time as _time

import requests as _requests
from eth_account import Account as _EthAccount
from hyperliquid.exchange import Exchange as _HlExchange
from hyperliquid.utils.constants import MAINNET_API_URL as _HL_MAIN, TESTNET_API_URL as _HL_TEST

from database import engine
from services.sol_broadcaster import broadcaster

# ── Server-side caches ─────────────────────────────────────────────────────
import calendar as _cal_mod

_cal_month_cache: dict = {}   # key "YYYY-MM" → {"result": ..., "expires": float}
_CAL_CURRENT_TTL = 60         # current month re-checked every 60 s
                              # historical months cached indefinitely (data can't change)

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "sol-perp" / "config.json"

_HL_TIMEOUT = 12
_LATENCY_RUNS = 5


def _load_hl_config():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _proxy_dict_from_index(cfg: dict, index: int) -> dict | None:
    """Return requests-style proxy dict. index=-1 means direct (no proxy)."""
    if index < 0:
        return None
    proxies = cfg.get("proxies", [])
    if index >= len(proxies):
        return None
    url = proxies[index]
    return {"http": url, "https": url}


def _hl_post(info_url: str, payload: dict, proxies: dict | None = None):
    r = _requests.post(
        info_url,
        json=payload,
        headers={"Content-Type": "application/json"},
        proxies=proxies,
        timeout=_HL_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def _hl_net_urls(network: str) -> tuple[str, str]:
    """Return (base_url, info_url) for given network string."""
    if network == "testnet":
        return _HL_TEST, f"{_HL_TEST}/info"
    return _HL_MAIN, f"{_HL_MAIN}/info"


_EMPTY_META = {"universe": []}
_EMPTY_SPOT = {"universe": [], "tokens": []}


def _build_exchange_master(cfg: dict, network: str) -> _HlExchange:
    """Exchange signed by the master wallet — for user-signed actions
    (transfers, withdrawals) that cannot be delegated to an API wallet."""
    net_cfg = cfg.get(network, {})
    pk = net_cfg.get("master_private_key", "")
    if not pk:
        raise ValueError(f"master_private_key not set for {network} in config.json")
    wallet = _EthAccount.from_key(pk)
    base_url = _HL_TEST if network == "testnet" else _HL_MAIN
    return _HlExchange(wallet, base_url, meta=_EMPTY_META, spot_meta=_EMPTY_SPOT)


def _build_exchange_agent(cfg: dict, network: str, load_meta: bool = False) -> _HlExchange:
    """Exchange signed by the API/agent wallet — for L1 actions
    (orders, cancels, leverage) on behalf of the master account.

    load_meta=True fetches real meta from the API so that coin name lookups
    (needed for order/cancel) work.  False uses empty stubs for speed.
    """
    net_cfg = cfg.get(network, {})
    pk = net_cfg.get("private_key", "")
    master = net_cfg.get("master_address")
    wallet = _EthAccount.from_key(pk)
    base_url = _HL_TEST if network == "testnet" else _HL_MAIN

    if load_meta:
        return _HlExchange(
            wallet, base_url,
            meta=None, spot_meta=None,
            account_address=master,
        )

    return _HlExchange(
        wallet, base_url,
        meta=_EMPTY_META, spot_meta=_EMPTY_SPOT,
        account_address=master,
    )


class TransferRequest(BaseModel):
    network: str = "mainnet"
    amount: float
    to_perp: bool
    proxy_index: int = -1


class PlaceOrderRequest(BaseModel):
    network: str = "mainnet"
    side: str  # "long" or "short"
    margin: float
    leverage: int
    is_cross: bool = False
    tp_price: float | None = None
    sl_price: float | None = None
    proxy_index: int = -1


class CancelOrderRequest(BaseModel):
    network: str = "mainnet"
    coin: str = "BTC"
    oid: int
    proxy_index: int = -1


class UpdateLeverageRequest(BaseModel):
    network: str = "mainnet"
    leverage: int
    is_cross: bool = False
    proxy_index: int = -1


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sol", tags=["sol"])


@router.websocket("/ws")
async def sol_websocket(ws: WebSocket):
    """Live data push: ticks, trades, marks via PG NOTIFY → WS broadcast."""
    await broadcaster.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        broadcaster.disconnect(ws)


@router.get("/health")
async def health():
    try:
        with engine.connect() as conn:
            r = conn.execute(text("SELECT COUNT(*) FROM sol.l2_snapshots"))
            return {"status": "healthy", "l2_rows": r.scalar()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@router.get("/status")
async def get_status():
    """Live feed status: collector heartbeat, freshness, current market snapshot."""
    try:
        with engine.connect() as conn:
            # Collector heartbeat
            r = conn.execute(text("""
                SELECT id, started_at, last_heartbeat, trades_inserted,
                       snaps_inserted, reconnects, status
                FROM sol.collector_status ORDER BY id DESC LIMIT 1
            """))
            coll = r.fetchone()
            collector = None
            if coll:
                now_utc = datetime.now(timezone.utc)
                hb = coll[2]
                age_s = (now_utc - hb).total_seconds() if hb else 999999
                collector = {
                    "id": coll[0],
                    "started_at": coll[1].isoformat() if coll[1] else None,
                    "last_heartbeat": coll[2].isoformat() if coll[2] else None,
                    "heartbeat_age_s": round(age_s, 1),
                    "is_live": age_s < 60,
                    "trades_inserted": coll[3],
                    "snaps_inserted": coll[4],
                    "reconnects": coll[5],
                    "status": coll[6],
                }

            # Latest data freshness — explicit index scan, never a full table scan
            r = conn.execute(text("SELECT ts FROM sol.l2_snapshots ORDER BY ts DESC LIMIT 1"))
            latest_snap = r.scalar()
            r = conn.execute(text("SELECT ts FROM sol.trades ORDER BY ts DESC LIMIT 1"))
            latest_trade = r.scalar()
            now_utc = datetime.now(timezone.utc)
            freshness = {
                "l2_age_s": round((now_utc - latest_snap).total_seconds(), 1) if latest_snap else None,
                "trade_age_s": round((now_utc - latest_trade).total_seconds(), 1) if latest_trade else None,
                "is_ingesting": False,
            }
            if freshness["l2_age_s"] is not None and freshness["l2_age_s"] < 5:
                freshness["is_ingesting"] = True

            # Latest tick
            r = conn.execute(text("""
                SELECT best_bid, best_ask, mid_price, spread, n_bid_levels, n_ask_levels
                FROM sol.l2_snapshots ORDER BY ts DESC LIMIT 1
            """))
            tick = r.fetchone()
            latest_tick = None
            if tick:
                latest_tick = {
                    "best_bid": float(tick[0]) if tick[0] else None,
                    "best_ask": float(tick[1]) if tick[1] else None,
                    "mid": float(tick[2]) if tick[2] else None,
                    "spread": float(tick[3]) if tick[3] else None,
                    "bid_levels": tick[4],
                    "ask_levels": tick[5],
                }

            return {
                "collector": collector,
                "freshness": freshness,
                "latest_tick": latest_tick,
            }
    except Exception as e:
        logger.error("sol status error: %s", e)
        return {"error": str(e)}


@router.get("/calendar")
async def get_calendar(
    year: int = Query(default=None),
    month: int = Query(default=None, ge=1, le=12),
):
    """Per-day presence check for a single calendar month.

    Accepts ?year=YYYY&month=MM.  Defaults to current UTC month.
    Uses generate_series + EXISTS → ~31 × 6 index probes per request.
    Historical months are cached forever; current month is cached 60 s.
    """
    now_utc = datetime.now(timezone.utc)
    if year is None:
        year = now_utc.year
    if month is None:
        month = now_utc.month

    cache_key = f"{year}-{month:02d}"
    is_current = (year == now_utc.year and month == now_utc.month)

    cached = _cal_month_cache.get(cache_key)
    if cached is not None:
        if not is_current:
            return cached["result"]
        if _time.monotonic() < cached["expires"]:
            return cached["result"]

    _, days_in = _cal_mod.monthrange(year, month)
    first_day = f"{year}-{month:02d}-01"
    last_day = f"{year}-{month:02d}-{days_in}"

    try:
        with engine.connect() as conn:
            r = conn.execute(text("""
                SELECT
                    d::date                                                         AS day,
                    EXISTS(SELECT 1 FROM sol.l2_snapshots    WHERE ts >= d AT TIME ZONE 'UTC' AND ts < (d + INTERVAL '1 day') AT TIME ZONE 'UTC') AS has_live_l2,
                    EXISTS(SELECT 1 FROM sol.trades          WHERE ts >= d AT TIME ZONE 'UTC' AND ts < (d + INTERVAL '1 day') AT TIME ZONE 'UTC') AS has_live_trades,
                    EXISTS(SELECT 1 FROM sol.mark_price      WHERE ts >= d AT TIME ZONE 'UTC' AND ts < (d + INTERVAL '1 day') AT TIME ZONE 'UTC') AS has_live_mark,
                    EXISTS(SELECT 1 FROM sol.hist_l2_events  WHERE ts >= d AT TIME ZONE 'UTC' AND ts < (d + INTERVAL '1 day') AT TIME ZONE 'UTC') AS has_bf_l2,
                    EXISTS(SELECT 1 FROM sol.hist_trades     WHERE ts >= d AT TIME ZONE 'UTC' AND ts < (d + INTERVAL '1 day') AT TIME ZONE 'UTC') AS has_bf_trades,
                    EXISTS(SELECT 1 FROM sol.hist_mark_price WHERE ts >= d AT TIME ZONE 'UTC' AND ts < (d + INTERVAL '1 day') AT TIME ZONE 'UTC') AS has_bf_mark
                FROM generate_series(
                    CAST(:first AS date),
                    CAST(:last AS date),
                    '1 day'::interval
                ) d
                ORDER BY 1
            """), {"first": first_day, "last": last_day})
            rows = r.fetchall()

            days = []
            for day, ll2, lt, lm, bl2, bt, bm in rows:
                if not (ll2 or lt or lm or bl2 or bt or bm):
                    continue
                days.append({
                    "date": str(day),
                    "has_live_l2": bool(ll2),
                    "has_live_trades": bool(lt),
                    "has_live_mark": bool(lm),
                    "has_bf_l2": bool(bl2),
                    "has_bf_trades": bool(bt),
                    "has_bf_mark": bool(bm),
                })

            result = {"year": year, "month": month, "days": days}
            _cal_month_cache[cache_key] = {
                "result": result,
                "expires": _time.monotonic() + (_CAL_CURRENT_TTL if is_current else float("inf")),
            }
            return result
    except Exception as e:
        logger.error("sol calendar error: %s", e)
        return {"error": str(e)}


@router.get("/day-detail/{date}")
async def get_day_detail(date: str):
    """Hourly breakdown with per-component counts for live + backfill.

    Uses range bounds on the raw `ts` column so Postgres can use the PK index
    instead of evaluating the cast-to-date expression on every row.
    """
    try:
        with engine.connect() as conn:
            range_filter = "ts >= (:d)::date AT TIME ZONE 'UTC' AND ts < ((:d)::date + 1) AT TIME ZONE 'UTC'"
            queries = {
                "live_l2":     f"SELECT EXTRACT(HOUR FROM ts AT TIME ZONE 'UTC')::int, COUNT(*) FROM sol.l2_snapshots WHERE {range_filter} GROUP BY 1",
                "live_trades": f"SELECT EXTRACT(HOUR FROM ts AT TIME ZONE 'UTC')::int, COUNT(*) FROM sol.trades WHERE {range_filter} GROUP BY 1",
                "live_mark":   f"SELECT EXTRACT(HOUR FROM ts AT TIME ZONE 'UTC')::int, COUNT(*) FROM sol.mark_price WHERE {range_filter} GROUP BY 1",
                "bf_trades":   f"SELECT EXTRACT(HOUR FROM ts AT TIME ZONE 'UTC')::int, COUNT(*) FROM sol.hist_trades WHERE {range_filter} GROUP BY 1",
                "bf_l2":       f"SELECT EXTRACT(HOUR FROM ts AT TIME ZONE 'UTC')::int, COUNT(*) FROM sol.hist_l2_events WHERE {range_filter} GROUP BY 1",
                "bf_mark":     f"SELECT EXTRACT(HOUR FROM ts AT TIME ZONE 'UTC')::int, COUNT(*) FROM sol.hist_mark_price WHERE {range_filter} GROUP BY 1",
            }
            data = {}
            for key, sql in queries.items():
                r = conn.execute(text(sql), {"d": date})
                data[key] = {row[0]: row[1] for row in r.fetchall()}

            hours = []
            for h in range(24):
                hours.append({
                    "hour": h,
                    "live_l2":     data["live_l2"].get(h, 0),
                    "live_trades": data["live_trades"].get(h, 0),
                    "live_mark":   data["live_mark"].get(h, 0),
                    "bf_trades":   data["bf_trades"].get(h, 0),
                    "bf_l2":       data["bf_l2"].get(h, 0),
                    "bf_mark":     data["bf_mark"].get(h, 0),
                })

            return {"date": date, "hours": hours}
    except Exception as e:
        logger.error("sol day-detail error: %s", e)
        return {"error": str(e)}


@router.get("/recent-ticks")
async def get_recent_ticks(limit: int = Query(default=200, le=2000)):
    """Recent L2 snapshots for live price chart."""
    try:
        with engine.connect() as conn:
            r = conn.execute(text("""
                SELECT ts, best_bid, best_ask, mid_price, spread,
                       n_bid_levels, n_ask_levels
                FROM sol.l2_snapshots
                ORDER BY ts DESC
                LIMIT :lim
            """), {"lim": limit})
            rows = r.fetchall()
            data = [{
                "ts": row[0].isoformat(),
                "best_bid": float(row[1]) if row[1] else None,
                "best_ask": float(row[2]) if row[2] else None,
                "mid": float(row[3]) if row[3] else None,
                "spread": float(row[4]) if row[4] else None,
                "bid_levels": row[5],
                "ask_levels": row[6],
            } for row in rows]
            data.reverse()
            return data
    except Exception as e:
        logger.error("sol recent-ticks error: %s", e)
        return {"error": str(e)}


@router.get("/latest-tick")
async def get_latest_tick():
    """Single most-recent L2 row — lightweight endpoint for fast price-pill polling."""
    try:
        with engine.connect() as conn:
            r = conn.execute(text("""
                SELECT ts, best_bid, best_ask, mid_price, spread,
                       n_bid_levels, n_ask_levels
                FROM sol.l2_snapshots
                ORDER BY ts DESC
                LIMIT 1
            """))
            row = r.fetchone()
            if not row:
                return {"available": False}
            return {
                "available": True,
                "ts": row[0].isoformat(),
                "best_bid": float(row[1]) if row[1] else None,
                "best_ask": float(row[2]) if row[2] else None,
                "mid": float(row[3]) if row[3] else None,
                "spread": float(row[4]) if row[4] else None,
                "bid_levels": row[5],
                "ask_levels": row[6],
            }
    except Exception as e:
        logger.error("sol latest-tick error: %s", e)
        return {"error": str(e)}


@router.get("/latest-mark")
async def get_latest_mark():
    """Most recent row from sol.mark_price (fed by the live collector).

    Returns mark / oracle / mid / premium / funding / OI for the newest ts.
    """
    try:
        with engine.connect() as conn:
            r = conn.execute(text("""
                SELECT ts, coin, mark_px, oracle_px, mid_px,
                       premium, funding, open_interest,
                       day_ntl_vlm, prev_day_px
                FROM sol.mark_price
                ORDER BY ts DESC
                LIMIT 1
            """))
            row = r.fetchone()
            if not row:
                return {"available": False}
            now_utc = datetime.now(timezone.utc)
            age_s = (now_utc - row[0]).total_seconds()
            def f(x):
                return float(x) if x is not None else None
            return {
                "available": True,
                "ts": row[0].isoformat(),
                "age_s": round(age_s, 1),
                "is_fresh": age_s < 10,
                "coin": row[1],
                "mark_px":   f(row[2]),
                "oracle_px": f(row[3]),
                "mid_px":    f(row[4]),
                "premium":   f(row[5]),
                "funding":   f(row[6]),
                "open_interest": f(row[7]),
                "day_ntl_vlm":   f(row[8]),
                "prev_day_px":   f(row[9]),
            }
    except Exception as e:
        logger.error("sol latest-mark error: %s", e)
        return {"error": str(e)}


@router.get("/recent-trades")
async def get_recent_trades(
    limit: int = Query(default=100, le=2000),
    since: str | None = Query(default=None, description="ISO timestamp — only return trades strictly newer than this"),
):
    """Recent trades for the live feed.

    If `since` is provided, returns only trades with ts > since (ordered oldest→newest).
    Otherwise returns the most recent `limit` trades (ordered oldest→newest).
    """
    try:
        with engine.connect() as conn:
            if since:
                r = conn.execute(text("""
                    SELECT ts, side, px, sz
                    FROM sol.trades
                    WHERE ts > :since
                    ORDER BY ts ASC
                    LIMIT :lim
                """), {"since": since, "lim": limit})
                rows = r.fetchall()
                data = [{
                    "ts": row[0].isoformat(),
                    "side": row[1],
                    "px": float(row[2]) if row[2] else None,
                    "sz": float(row[3]) if row[3] else None,
                } for row in rows]
                return data
            r = conn.execute(text("""
                SELECT ts, side, px, sz
                FROM sol.trades
                ORDER BY ts DESC
                LIMIT :lim
            """), {"lim": limit})
            rows = r.fetchall()
            data = [{
                "ts": row[0].isoformat(),
                "side": row[1],
                "px": float(row[2]) if row[2] else None,
                "sz": float(row[3]) if row[3] else None,
            } for row in rows]
            data.reverse()
            return data
    except Exception as e:
        logger.error("sol recent-trades error: %s", e)
        return {"error": str(e)}


@router.get("/hl-balances")
def get_hl_balances(proxy_index: int = Query(-1)):
    """Fetch account balances from both mainnet and testnet Hyperliquid accounts.

    Uses the same unified-aware logic as hl-net-balances.
    proxy_index: -1 = direct (no proxy), 0-3 = proxy index from config.json
    """
    try:
        cfg = _load_hl_config()
    except Exception as e:
        logger.error("hl-balances: could not load config.json: %s", e)
        return {"error": f"Could not load config: {e}"}

    prx = _proxy_dict_from_index(cfg, proxy_index)

    result = {}
    for net in ("mainnet", "testnet"):
        net_cfg = cfg.get(net, {})
        master = net_cfg.get("master_address", "").lower()
        _, info_url = _hl_net_urls(net)

        entry: dict = {
            "master_address": net_cfg.get("master_address", ""),
            "network": net,
            "account_mode": "default",
            "account_value": "0",
            "withdrawable": "0",
            "total_margin_used": "0",
            "total_ntl_pos": "0",
            "positions": [],
            "spot_balances": [],
            "error": None,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            mode = _hl_post(info_url, {"type": "userAbstraction", "user": master}, prx)
            if isinstance(mode, str):
                entry["account_mode"] = mode
        except Exception:
            pass

        try:
            sc = _hl_post(info_url, {"type": "spotClearinghouseState", "user": master}, prx)
            balances = sc.get("balances", [])
            entry["spot_balances"] = balances

            ch = _hl_post(info_url, {"type": "clearinghouseState", "user": master}, prx)
            ms = ch.get("marginSummary", {})
            entry["total_margin_used"] = ms.get("totalMarginUsed", "0")
            entry["total_ntl_pos"] = ms.get("totalNtlPos", "0")
            entry["withdrawable"] = ch.get("withdrawable", "0")
            entry["positions"] = []
            for p in ch.get("assetPositions", []):
                pos = p.get("position", {})
                lev = pos.get("leverage", {})
                entry["positions"].append({
                    "coin": pos.get("coin"),
                    "size": pos.get("szi"),
                    "entry_px": pos.get("entryPx"),
                    "unrealized_pnl": pos.get("unrealizedPnl"),
                    "return_on_equity": pos.get("returnOnEquity"),
                    "leverage": lev.get("value"),
                    "leverage_type": lev.get("type"),
                    "margin_used": pos.get("marginUsed"),
                    "liquidation_px": pos.get("liquidationPx"),
                    "position_value": pos.get("positionValue"),
                    "max_leverage": pos.get("maxLeverage"),
                    "cum_funding": pos.get("cumFunding", {}).get("sinceOpen"),
                })

            is_unified = entry["account_mode"] in ("unifiedAccount", "portfolioMargin")
            if is_unified:
                usdc = next((b for b in balances if b.get("coin") == "USDC"), None)
                usdc_total = float(usdc["total"]) if usdc else 0.0
                unrealized = sum(float(p.get("unrealized_pnl") or 0) for p in entry["positions"])
                entry["account_value"] = str(usdc_total + unrealized)
            else:
                entry["account_value"] = ms.get("accountValue", "0")

        except Exception as e:
            logger.error("hl-balances %s error: %s", net, e)
            entry["error"] = str(e)

        result[net] = entry

    return result


@router.get("/hl-proxies")
def get_hl_proxies():
    """Return available proxy labels (no credentials exposed) + direct option."""
    try:
        cfg = _load_hl_config()
    except Exception as e:
        return {"error": str(e)}
    proxies = cfg.get("proxies", [])
    labels = []
    for i, url in enumerate(proxies):
        host_part = url.split("@")[-1] if "@" in url else url.replace("http://", "").replace("https://", "")
        labels.append({"index": i, "label": f"Proxy {i+1} ({host_part})"})
    return {"proxies": labels}


@router.get("/hl-latency")
def get_hl_latency(proxy_index: int = Query(-1)):
    """Run latency benchmark against Hyperliquid info endpoint.

    Tests: allMids (light), clearinghouseState (medium), spotClearinghouseState (medium).
    """
    try:
        cfg = _load_hl_config()
    except Exception as e:
        return {"error": str(e)}

    prx = _proxy_dict_from_index(cfg, proxy_index)
    net_cfg = cfg.get("mainnet", {})
    master = net_cfg.get("master_address", "").lower()
    info_url = "https://api.hyperliquid.xyz/info"

    tests = [
        {"name": "allMids", "label": "Light (allMids)", "payload": {"type": "allMids"}},
        {"name": "clearinghouseState", "label": "Medium (clearinghouseState)",
         "payload": {"type": "clearinghouseState", "user": master}},
        {"name": "spotClearinghouseState", "label": "Medium (spotClearinghouseState)",
         "payload": {"type": "spotClearinghouseState", "user": master}},
    ]

    results = []
    for test in tests:
        lats, errs = [], []
        for _ in range(_LATENCY_RUNS):
            t0 = _time.perf_counter()
            try:
                _hl_post(info_url, test["payload"], prx)
                lats.append((_time.perf_counter() - t0) * 1000)
            except Exception as exc:
                errs.append(str(exc)[:120])
        if lats:
            std = _stats.stdev(lats) if len(lats) > 1 else 0.0
            results.append({
                "name": test["name"], "label": test["label"],
                "min": round(min(lats), 1), "max": round(max(lats), 1),
                "mean": round(_stats.mean(lats), 1),
                "median": round(_stats.median(lats), 1),
                "stdev": round(std, 1),
                "runs": len(lats), "errors": errs,
            })
        else:
            results.append({
                "name": test["name"], "label": test["label"],
                "min": None, "max": None, "mean": None, "median": None, "stdev": None,
                "runs": 0, "errors": errs,
            })

    return {
        "proxy_index": proxy_index,
        "proxy_label": "Direct" if proxy_index < 0 else f"Proxy {proxy_index + 1}",
        "runs_per_test": _LATENCY_RUNS,
        "tests": results,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/hl-net-balances")
def get_hl_net_balances(network: str = Query("mainnet"), proxy_index: int = Query(-1)):
    """Fetch balances for a single network.

    For unified accounts the spot endpoint is the source of truth for
    account balance across both spot and perps (per Hyperliquid docs).
    We also query clearinghouseState for open positions and margin info.
    """
    try:
        cfg = _load_hl_config()
    except Exception as e:
        return {"error": str(e)}

    prx = _proxy_dict_from_index(cfg, proxy_index)
    net_cfg = cfg.get(network, {})
    master = net_cfg.get("master_address", "").lower()
    _, info_url = _hl_net_urls(network)

    entry: dict = {
        "master_address": net_cfg.get("master_address", ""),
        "network": network,
        "account_mode": "default",
        "account_value": "0",
        "withdrawable": "0",
        "total_margin_used": "0",
        "total_ntl_pos": "0",
        "positions": [],
        "spot_balances": [],
        "error": None,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        mode = _hl_post(info_url, {"type": "userAbstraction", "user": master}, prx)
        if isinstance(mode, str):
            entry["account_mode"] = mode
    except Exception:
        pass

    try:
        sc = _hl_post(info_url, {"type": "spotClearinghouseState", "user": master}, prx)
        balances = sc.get("balances", [])
        entry["spot_balances"] = balances

        ch = _hl_post(info_url, {"type": "clearinghouseState", "user": master}, prx)
        ms = ch.get("marginSummary", {})
        entry["total_margin_used"] = ms.get("totalMarginUsed", "0")
        entry["total_ntl_pos"] = ms.get("totalNtlPos", "0")
        entry["withdrawable"] = ch.get("withdrawable", "0")
        entry["positions"] = []
        for p in ch.get("assetPositions", []):
            pos = p.get("position", {})
            lev = pos.get("leverage", {})
            entry["positions"].append({
                "coin": pos.get("coin"),
                "size": pos.get("szi"),
                "entry_px": pos.get("entryPx"),
                "unrealized_pnl": pos.get("unrealizedPnl"),
                "return_on_equity": pos.get("returnOnEquity"),
                "leverage": lev.get("value"),
                "leverage_type": lev.get("type"),
                "margin_used": pos.get("marginUsed"),
                "liquidation_px": pos.get("liquidationPx"),
                "position_value": pos.get("positionValue"),
                "max_leverage": pos.get("maxLeverage"),
                "cum_funding": pos.get("cumFunding", {}).get("sinceOpen"),
            })

        is_unified = entry["account_mode"] in ("unifiedAccount", "portfolioMargin")

        if is_unified:
            usdc = next((b for b in balances if b.get("coin") == "USDC"), None)
            usdc_total = float(usdc["total"]) if usdc else 0.0
            unrealized = sum(
                float(p.get("unrealized_pnl") or 0)
                for p in entry["positions"]
            )
            entry["account_value"] = str(usdc_total + unrealized)
        else:
            entry["account_value"] = ms.get("accountValue", "0")

    except Exception as e:
        logger.error("hl-net-balances %s error: %s", network, e)
        entry["error"] = str(e)

    return entry


@router.get("/hl-btc-price")
def get_hl_btc_price(proxy_index: int = Query(-1), network: str = Query("mainnet")):
    """Get BTC-USDC perp mark price via metaAndAssetCtxs (always from mainnet)."""
    try:
        cfg = _load_hl_config()
    except Exception as e:
        return {"error": str(e)}

    prx = _proxy_dict_from_index(cfg, proxy_index)
    _, info_url = _hl_net_urls("mainnet")

    try:
        data = _hl_post(info_url, {"type": "metaAndAssetCtxs"}, prx)
        universe = data[0]["universe"] if isinstance(data, list) and len(data) >= 2 else []
        ctxs = data[1] if isinstance(data, list) and len(data) >= 2 else []

        btc_ctx = None
        for i, coin in enumerate(universe):
            if coin.get("name") == "BTC":
                if i < len(ctxs):
                    btc_ctx = ctxs[i]
                break

        if btc_ctx:
            return {
                "mark_px": btc_ctx.get("markPx"),
                "mid_px": btc_ctx.get("midPx"),
                "oracle_px": btc_ctx.get("oraclePx"),
                "funding": btc_ctx.get("funding"),
                "open_interest": btc_ctx.get("openInterest"),
                "prev_day_px": btc_ctx.get("prevDayPx"),
                "network": "mainnet",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        return {"error": "BTC not found in metaAndAssetCtxs"}
    except Exception as e:
        logger.error("hl-btc-price error: %s", e)
        return {"error": str(e)}


@router.get("/hl-btc-meta")
def get_hl_btc_meta(proxy_index: int = Query(-1), network: str = Query("mainnet")):
    """Get BTC perpetual metadata (max leverage, szDecimals, margin tiers)."""
    try:
        cfg = _load_hl_config()
    except Exception as e:
        return {"error": str(e)}

    prx = _proxy_dict_from_index(cfg, proxy_index)
    _, info_url = _hl_net_urls(network)

    try:
        meta = _hl_post(info_url, {"type": "meta"}, prx)
        universe = meta.get("universe", [])
        btc_entry = None
        btc_index = None
        for i, coin in enumerate(universe):
            if coin.get("name") == "BTC":
                btc_entry = coin
                btc_index = i
                break

        margin_tiers = None
        if btc_index is not None:
            tables = meta.get("marginTables", [])
            tid = btc_entry.get("marginTableId", btc_index)
            if tid < len(tables):
                tier_group = tables[tid]
                if tier_group and len(tier_group) > 0:
                    margin_tiers = tier_group[0].get("marginTiers", [])

        return {
            "asset_index": btc_index,
            "name": btc_entry.get("name") if btc_entry else None,
            "szDecimals": btc_entry.get("szDecimals") if btc_entry else None,
            "maxLeverage": btc_entry.get("maxLeverage") if btc_entry else None,
            "margin_tiers": margin_tiers,
            "network": network,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error("hl-btc-meta error: %s", e)
        return {"error": str(e)}


@router.post("/hl-transfer")
def hl_transfer(req: TransferRequest):
    """Transfer USDC between spot and perp wallets.

    Uses the master wallet private key (user-signed action — API wallets cannot sign transfers).
    """
    if req.amount <= 0:
        return {"error": "Amount must be positive"}
    if req.network not in ("mainnet", "testnet"):
        return {"error": "Network must be 'mainnet' or 'testnet'"}

    try:
        cfg = _load_hl_config()
    except Exception as e:
        return {"error": f"Could not load config: {e}"}

    net_cfg = cfg.get(req.network, {})
    pk = net_cfg.get("master_private_key", "")
    if not pk:
        return {"error": f"No master_private_key configured for {req.network} — transfers require the master wallet key, not the API wallet"}

    direction = "spot → perp" if req.to_perp else "perp → spot"
    logger.info("hl-transfer: %s %.4f USDC on %s", direction, req.amount, req.network)

    try:
        exchange = _build_exchange_master(cfg, req.network)
        result = exchange.usd_class_transfer(req.amount, req.to_perp)
        logger.info("hl-transfer result: %s", result)

        status = result.get("status", "unknown")
        if status != "ok":
            return {
                "success": False,
                "error": f"Hyperliquid returned status={status}",
                "raw": result,
            }

        return {
            "success": True,
            "direction": direction,
            "amount": req.amount,
            "network": req.network,
            "raw": result,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error("hl-transfer error: %s", e)
        return {"success": False, "error": str(e)}


# ── BTC Trading endpoints ──────────────────────────────────────────────────


def _get_btc_asset_index(info_url: str, proxies: dict | None) -> int:
    """Resolve BTC asset index from meta (usually 0 on mainnet)."""
    meta = _hl_post(info_url, {"type": "meta"}, proxies)
    for i, u in enumerate(meta.get("universe", [])):
        if u.get("name") == "BTC":
            return i
    raise ValueError("BTC not found in universe")


def _get_btc_sz_decimals(info_url: str, proxies: dict | None) -> int:
    """Get szDecimals for BTC from meta."""
    meta = _hl_post(info_url, {"type": "meta"}, proxies)
    for u in meta.get("universe", []):
        if u.get("name") == "BTC":
            return u.get("szDecimals", 5)
    return 5


def _round_sz(sz: float, sz_decimals: int) -> float:
    return round(sz, sz_decimals)


def _round_price(px: float) -> float:
    """Round price to 5 significant figures (Hyperliquid perp rule)."""
    if px == 0:
        return 0.0
    import math
    digits = 5 - int(math.floor(math.log10(abs(px)))) - 1
    digits = max(digits, 0)
    return round(px, digits)


@router.get("/hl-open-orders")
def get_hl_open_orders(network: str = Query("mainnet"), proxy_index: int = Query(-1)):
    """Get open orders for BTC on the specified network."""
    try:
        cfg = _load_hl_config()
        net_cfg = cfg.get(network, {})
        master = net_cfg.get("master_address")
        _, info_url = _hl_net_urls(network)
        proxies = _proxy_dict_from_index(cfg, proxy_index)

        raw = _hl_post(info_url, {"type": "frontendOpenOrders", "user": master}, proxies)
        btc_orders = [o for o in raw if o.get("coin") == "BTC"]

        return {
            "orders": btc_orders,
            "total": len(btc_orders),
            "network": network,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error("hl-open-orders error: %s", e)
        return {"error": str(e)}


@router.post("/hl-update-leverage")
def post_hl_update_leverage(req: UpdateLeverageRequest):
    """Set leverage for BTC on the specified network."""
    if req.leverage < 1:
        return {"error": "Leverage must be >= 1"}

    try:
        cfg = _load_hl_config()
        exchange = _build_exchange_agent(cfg, req.network)
        result = exchange.update_leverage(req.leverage, "BTC", is_cross=req.is_cross)
        logger.info("hl-update-leverage %s %dx cross=%s → %s", req.network, req.leverage, req.is_cross, result)
        status = result.get("status", "unknown")
        return {
            "success": status == "ok",
            "leverage": req.leverage,
            "is_cross": req.is_cross,
            "network": req.network,
            "raw": result,
        }
    except Exception as e:
        logger.error("hl-update-leverage error: %s", e)
        return {"success": False, "error": str(e)}


@router.post("/hl-place-order")
def post_hl_place_order(req: PlaceOrderRequest):
    """Place a BTC-PERP leveraged order with optional TP/SL.

    Flow: set leverage → get current mark price → compute size → place market order
    (optionally with TP/SL as trigger orders via normalTpsl grouping).
    """
    if req.margin <= 0:
        return {"error": "Margin must be positive"}
    if req.leverage < 1:
        return {"error": "Leverage must be >= 1"}
    if req.side not in ("long", "short"):
        return {"error": "Side must be 'long' or 'short'"}

    try:
        cfg = _load_hl_config()
        _, info_url = _hl_net_urls(req.network)
        proxies = _proxy_dict_from_index(cfg, req.proxy_index)

        # 1. Get current price + metadata
        meta_and_ctx = _hl_post(info_url, {"type": "metaAndAssetCtxs"}, proxies)
        meta_list = meta_and_ctx[0].get("universe", [])
        ctx_list = meta_and_ctx[1] if len(meta_and_ctx) > 1 else []

        btc_idx = None
        sz_decimals = 5
        for i, u in enumerate(meta_list):
            if u.get("name") == "BTC":
                btc_idx = i
                sz_decimals = u.get("szDecimals", 5)
                break

        if btc_idx is None:
            return {"error": "BTC not found in meta"}

        mark_px = float(ctx_list[btc_idx].get("markPx", 0)) if btc_idx < len(ctx_list) else 0
        if mark_px <= 0:
            return {"error": f"Invalid mark price: {mark_px}"}

        # 2. Set leverage + build exchange with real meta for order placement
        exchange = _build_exchange_agent(cfg, req.network, load_meta=True)
        lev_result = exchange.update_leverage(req.leverage, "BTC", is_cross=req.is_cross)
        lev_status = lev_result.get("status", "unknown")
        if lev_status != "ok":
            return {"success": False, "error": f"Failed to set leverage: {lev_result}", "step": "leverage"}

        # 3. Compute size
        is_buy = req.side == "long"
        notional = req.margin * req.leverage
        raw_sz = notional / mark_px
        sz = _round_sz(raw_sz, sz_decimals)

        if sz <= 0:
            return {"error": f"Computed size is 0 (notional={notional}, price={mark_px})"}

        # 4. Place order(s)
        slippage = 0.03  # 3% slippage for market-like execution
        if is_buy:
            limit_px = _round_price(mark_px * (1 + slippage))
        else:
            limit_px = _round_price(mark_px * (1 - slippage))

        has_tpsl = req.tp_price is not None or req.sl_price is not None

        if not has_tpsl:
            result = exchange.order("BTC", is_buy, sz, limit_px, {"limit": {"tif": "Ioc"}})
        else:
            orders = [
                {
                    "coin": "BTC",
                    "is_buy": is_buy,
                    "sz": sz,
                    "limit_px": limit_px,
                    "order_type": {"limit": {"tif": "Ioc"}},
                    "reduce_only": False,
                },
            ]

            if req.tp_price is not None:
                tp_px = _round_price(req.tp_price)
                orders.append({
                    "coin": "BTC",
                    "is_buy": not is_buy,
                    "sz": sz,
                    "limit_px": tp_px,
                    "order_type": {"trigger": {"triggerPx": tp_px, "isMarket": True, "tpsl": "tp"}},
                    "reduce_only": True,
                })

            if req.sl_price is not None:
                sl_px = _round_price(req.sl_price)
                orders.append({
                    "coin": "BTC",
                    "is_buy": not is_buy,
                    "sz": sz,
                    "limit_px": sl_px,
                    "order_type": {"trigger": {"triggerPx": sl_px, "isMarket": True, "tpsl": "sl"}},
                    "reduce_only": True,
                })

            result = exchange.bulk_orders(orders, grouping="normalTpsl")

        logger.info("hl-place-order %s %s %.5f BTC @ ~%.1f lev=%dx → %s",
                     req.network, req.side, sz, mark_px, req.leverage, result)

        status = result.get("status", "unknown")
        statuses = []
        if status == "ok" and result.get("response", {}).get("data", {}).get("statuses"):
            statuses = result["response"]["data"]["statuses"]

        first_status = statuses[0] if statuses else {}
        filled = first_status.get("filled")
        resting = first_status.get("resting")
        order_error = first_status.get("error")

        return {
            "success": status == "ok" and not order_error,
            "side": req.side,
            "size_btc": sz,
            "notional": notional,
            "mark_price": mark_px,
            "limit_price": limit_px,
            "leverage": req.leverage,
            "is_cross": req.is_cross,
            "margin": req.margin,
            "tp_price": req.tp_price,
            "sl_price": req.sl_price,
            "network": req.network,
            "filled": filled,
            "resting": resting,
            "order_error": order_error,
            "statuses": statuses,
            "raw": result,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error("hl-place-order error: %s", e)
        return {"success": False, "error": str(e)}


@router.post("/hl-cancel-order")
def post_hl_cancel_order(req: CancelOrderRequest):
    """Cancel a specific BTC order by oid."""
    try:
        cfg = _load_hl_config()
        _, info_url = _hl_net_urls(req.network)
        proxies = _proxy_dict_from_index(cfg, req.proxy_index)

        btc_idx = _get_btc_asset_index(info_url, proxies)
        exchange = _build_exchange_agent(cfg, req.network, load_meta=True)
        result = exchange.cancel(req.coin, req.oid)
        logger.info("hl-cancel-order %s oid=%d → %s", req.network, req.oid, result)

        status = result.get("status", "unknown")
        return {
            "success": status == "ok",
            "oid": req.oid,
            "network": req.network,
            "raw": result,
        }
    except Exception as e:
        logger.error("hl-cancel-order error: %s", e)
        return {"success": False, "error": str(e)}
