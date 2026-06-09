"""
PERPS API Router (BTC / ETH / SPX)
==================================
Generic perp-instrument router that mirrors the data endpoints in
``routers/sol.py`` but is parameterised by ``{asset}`` (one of
``btc``, ``eth``, ``spx``).  Each asset has its own Postgres schema
populated by the matching collector in ``sol-perp/`` (collector_btc.py,
collector_eth.py, collector_spx.py).  Each collector publishes via
``pg_notify`` on channels ``{asset}_ticks/_trades/_marks`` which are
forwarded to WebSocket clients via ``services.perps_broadcaster``.

This router intentionally does NOT include the Hyperliquid account /
trading endpoints — those remain on the SOL router because they are
account-level (not asset-data) operations.
"""

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import text
from datetime import datetime, timezone
import calendar as _cal_mod
import logging
import time as _time

from database import engine
from services.perps_broadcaster import perps_broadcaster, ASSETS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/perp", tags=["perps"])

# ── Per-asset calendar cache (mirrors the one in sol.py) ──────────────────
_cal_month_cache: dict[str, dict] = {}   # key "{asset}:YYYY-MM"
_CAL_CURRENT_TTL = 60                    # seconds

_VALID_ASSETS = set(ASSETS)


def _check_asset(asset: str) -> str:
    """Validate ``asset`` path param — also acts as the schema name."""
    a = asset.lower()
    if a not in _VALID_ASSETS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown asset '{asset}'. Valid: {sorted(_VALID_ASSETS)}",
        )
    return a


# ── WebSocket ─────────────────────────────────────────────────────────────


@router.websocket("/{asset}/ws")
async def perp_websocket(ws: WebSocket, asset: str):
    """Live data push for one asset: ticks, trades, marks via PG NOTIFY."""
    a = asset.lower()
    if a not in _VALID_ASSETS:
        await ws.close(code=1008)
        return
    await perps_broadcaster.connect(a, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        perps_broadcaster.disconnect(a, ws)


# ── REST endpoints ────────────────────────────────────────────────────────


@router.get("/{asset}/health")
async def health(asset: str):
    a = _check_asset(asset)
    try:
        with engine.connect() as conn:
            r = conn.execute(text(f"SELECT COUNT(*) FROM {a}.l2_snapshots"))
            return {"asset": a, "status": "healthy", "l2_rows": r.scalar()}
    except Exception as e:
        return {"asset": a, "status": "unhealthy", "error": str(e)}


@router.get("/{asset}/status")
async def get_status(asset: str):
    """Live feed status: collector heartbeat, freshness, current snapshot."""
    a = _check_asset(asset)
    try:
        with engine.connect() as conn:
            r = conn.execute(text(f"""
                SELECT id, started_at, last_heartbeat, trades_inserted,
                       snaps_inserted, reconnects, status
                FROM {a}.collector_status ORDER BY id DESC LIMIT 1
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

            r = conn.execute(text(f"SELECT ts FROM {a}.l2_snapshots ORDER BY ts DESC LIMIT 1"))
            latest_snap = r.scalar()
            r = conn.execute(text(f"SELECT ts FROM {a}.trades ORDER BY ts DESC LIMIT 1"))
            latest_trade = r.scalar()
            now_utc = datetime.now(timezone.utc)
            freshness = {
                "l2_age_s": round((now_utc - latest_snap).total_seconds(), 1) if latest_snap else None,
                "trade_age_s": round((now_utc - latest_trade).total_seconds(), 1) if latest_trade else None,
                "is_ingesting": False,
            }
            if freshness["l2_age_s"] is not None and freshness["l2_age_s"] < 5:
                freshness["is_ingesting"] = True

            r = conn.execute(text(f"""
                SELECT best_bid, best_ask, mid_price, spread, n_bid_levels, n_ask_levels
                FROM {a}.l2_snapshots ORDER BY ts DESC LIMIT 1
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
                "asset": a,
                "collector": collector,
                "freshness": freshness,
                "latest_tick": latest_tick,
            }
    except Exception as e:
        logger.error("perp[%s] status error: %s", a, e)
        return {"asset": a, "error": str(e)}


@router.get("/{asset}/calendar")
async def get_calendar(
    asset: str,
    year: int = Query(default=None),
    month: int = Query(default=None, ge=1, le=12),
):
    """Per-day presence check for a single calendar month.

    Mirrors sol.calendar — only the schema differs.  Backfill tables
    (``hist_*``) are queried optimistically and skipped if absent so this
    works whether or not the backfill schema has been provisioned for
    BTC/ETH/SPX.
    """
    a = _check_asset(asset)
    now_utc = datetime.now(timezone.utc)
    if year is None:
        year = now_utc.year
    if month is None:
        month = now_utc.month

    cache_key = f"{a}:{year}-{month:02d}"
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

    # Detect which backfill tables (if any) exist for this asset.
    try:
        with engine.connect() as conn:
            r = conn.execute(text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = :schema
                  AND table_name IN ('hist_l2_events','hist_trades','hist_mark_price')
            """), {"schema": a})
            present = {row[0] for row in r.fetchall()}
    except Exception:
        present = set()

    bf_l2_expr   = (f"EXISTS(SELECT 1 FROM {a}.hist_l2_events  WHERE ts >= d AT TIME ZONE 'UTC' AND ts < (d + INTERVAL '1 day') AT TIME ZONE 'UTC')"
                   if "hist_l2_events" in present else "FALSE")
    bf_tr_expr   = (f"EXISTS(SELECT 1 FROM {a}.hist_trades     WHERE ts >= d AT TIME ZONE 'UTC' AND ts < (d + INTERVAL '1 day') AT TIME ZONE 'UTC')"
                   if "hist_trades" in present else "FALSE")
    bf_mark_expr = (f"EXISTS(SELECT 1 FROM {a}.hist_mark_price WHERE ts >= d AT TIME ZONE 'UTC' AND ts < (d + INTERVAL '1 day') AT TIME ZONE 'UTC')"
                   if "hist_mark_price" in present else "FALSE")

    try:
        with engine.connect() as conn:
            r = conn.execute(text(f"""
                SELECT
                    d::date                                                         AS day,
                    EXISTS(SELECT 1 FROM {a}.l2_snapshots    WHERE ts >= d AT TIME ZONE 'UTC' AND ts < (d + INTERVAL '1 day') AT TIME ZONE 'UTC') AS has_live_l2,
                    EXISTS(SELECT 1 FROM {a}.trades          WHERE ts >= d AT TIME ZONE 'UTC' AND ts < (d + INTERVAL '1 day') AT TIME ZONE 'UTC') AS has_live_trades,
                    EXISTS(SELECT 1 FROM {a}.mark_price      WHERE ts >= d AT TIME ZONE 'UTC' AND ts < (d + INTERVAL '1 day') AT TIME ZONE 'UTC') AS has_live_mark,
                    {bf_l2_expr}   AS has_bf_l2,
                    {bf_tr_expr}   AS has_bf_trades,
                    {bf_mark_expr} AS has_bf_mark
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

            result = {"asset": a, "year": year, "month": month, "days": days}
            _cal_month_cache[cache_key] = {
                "result": result,
                "expires": _time.monotonic() + (_CAL_CURRENT_TTL if is_current else float("inf")),
            }
            return result
    except Exception as e:
        logger.error("perp[%s] calendar error: %s", a, e)
        return {"asset": a, "error": str(e)}


@router.get("/{asset}/day-detail/{date}")
async def get_day_detail(asset: str, date: str):
    """Hourly breakdown — live tables always, backfill tables only if present."""
    a = _check_asset(asset)
    try:
        with engine.connect() as conn:
            # Detect backfill tables
            r = conn.execute(text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = :schema
                  AND table_name IN ('hist_l2_events','hist_trades','hist_mark_price')
            """), {"schema": a})
            present = {row[0] for row in r.fetchall()}

            range_filter = "ts >= (:d)::date AT TIME ZONE 'UTC' AND ts < ((:d)::date + 1) AT TIME ZONE 'UTC'"
            queries = {
                "live_l2":     f"SELECT EXTRACT(HOUR FROM ts AT TIME ZONE 'UTC')::int, COUNT(*) FROM {a}.l2_snapshots WHERE {range_filter} GROUP BY 1",
                "live_trades": f"SELECT EXTRACT(HOUR FROM ts AT TIME ZONE 'UTC')::int, COUNT(*) FROM {a}.trades       WHERE {range_filter} GROUP BY 1",
                "live_mark":   f"SELECT EXTRACT(HOUR FROM ts AT TIME ZONE 'UTC')::int, COUNT(*) FROM {a}.mark_price   WHERE {range_filter} GROUP BY 1",
            }
            if "hist_trades" in present:
                queries["bf_trades"] = f"SELECT EXTRACT(HOUR FROM ts AT TIME ZONE 'UTC')::int, COUNT(*) FROM {a}.hist_trades     WHERE {range_filter} GROUP BY 1"
            if "hist_l2_events" in present:
                queries["bf_l2"]     = f"SELECT EXTRACT(HOUR FROM ts AT TIME ZONE 'UTC')::int, COUNT(*) FROM {a}.hist_l2_events  WHERE {range_filter} GROUP BY 1"
            if "hist_mark_price" in present:
                queries["bf_mark"]   = f"SELECT EXTRACT(HOUR FROM ts AT TIME ZONE 'UTC')::int, COUNT(*) FROM {a}.hist_mark_price WHERE {range_filter} GROUP BY 1"

            data = {}
            for key, sql in queries.items():
                r = conn.execute(text(sql), {"d": date})
                data[key] = {row[0]: row[1] for row in r.fetchall()}

            hours = []
            for h in range(24):
                hours.append({
                    "hour": h,
                    "live_l2":     data.get("live_l2", {}).get(h, 0),
                    "live_trades": data.get("live_trades", {}).get(h, 0),
                    "live_mark":   data.get("live_mark", {}).get(h, 0),
                    "bf_trades":   data.get("bf_trades", {}).get(h, 0),
                    "bf_l2":       data.get("bf_l2", {}).get(h, 0),
                    "bf_mark":     data.get("bf_mark", {}).get(h, 0),
                })

            return {"asset": a, "date": date, "hours": hours}
    except Exception as e:
        logger.error("perp[%s] day-detail error: %s", a, e)
        return {"asset": a, "error": str(e)}


@router.get("/{asset}/recent-ticks")
async def get_recent_ticks(asset: str, limit: int = Query(default=200, le=2000)):
    a = _check_asset(asset)
    try:
        with engine.connect() as conn:
            r = conn.execute(text(f"""
                SELECT ts, best_bid, best_ask, mid_price, spread,
                       n_bid_levels, n_ask_levels
                FROM {a}.l2_snapshots
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
        logger.error("perp[%s] recent-ticks error: %s", a, e)
        return {"error": str(e)}


@router.get("/{asset}/latest-tick")
async def get_latest_tick(asset: str):
    a = _check_asset(asset)
    try:
        with engine.connect() as conn:
            r = conn.execute(text(f"""
                SELECT ts, best_bid, best_ask, mid_price, spread,
                       n_bid_levels, n_ask_levels
                FROM {a}.l2_snapshots
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
        logger.error("perp[%s] latest-tick error: %s", a, e)
        return {"error": str(e)}


@router.get("/{asset}/latest-mark")
async def get_latest_mark(asset: str):
    a = _check_asset(asset)
    try:
        with engine.connect() as conn:
            r = conn.execute(text(f"""
                SELECT ts, coin, mark_px, oracle_px, mid_px,
                       premium, funding, open_interest,
                       day_ntl_vlm, prev_day_px
                FROM {a}.mark_price
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
        logger.error("perp[%s] latest-mark error: %s", a, e)
        return {"error": str(e)}


@router.get("/{asset}/recent-trades")
async def get_recent_trades(
    asset: str,
    limit: int = Query(default=100, le=2000),
    since: str | None = Query(default=None, description="ISO timestamp — only return trades strictly newer than this"),
):
    a = _check_asset(asset)
    try:
        with engine.connect() as conn:
            if since:
                r = conn.execute(text(f"""
                    SELECT ts, side, px, sz
                    FROM {a}.trades
                    WHERE ts > :since
                    ORDER BY ts ASC
                    LIMIT :lim
                """), {"since": since, "lim": limit})
                rows = r.fetchall()
                return [{
                    "ts": row[0].isoformat(),
                    "side": row[1],
                    "px": float(row[2]) if row[2] else None,
                    "sz": float(row[3]) if row[3] else None,
                } for row in rows]
            r = conn.execute(text(f"""
                SELECT ts, side, px, sz
                FROM {a}.trades
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
        logger.error("perp[%s] recent-trades error: %s", a, e)
        return {"error": str(e)}
