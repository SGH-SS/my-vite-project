"""
Feature builder for the Strat Hub dashboard.

Each function takes a SQLAlchemy ``Connection``, a schema name (``sol`` /
``btc`` / ``eth`` / ``spx``), and an ISO ``since`` / ``until`` window,
and returns a JSON-serialisable ``dict`` shaped roughly like::

    {
        "kind":   "<feature key>",
        "asset":  "<schema>",
        "window": {"since": ..., "until": ...},
        "summary": { ... pre-digested numbers ... },
        "series":  [ ... ],
    }

Both raw + pre-digested numbers are returned so the LLM has structure to
reason about without re-deriving everything itself.

These functions never call Anthropic — they only touch Postgres.  All
SQL uses parameterised text via SQLAlchemy and works on both the PC
(TimescaleDB) and Mac (vanilla Postgres) since we avoid ``time_bucket``.
Within the requested ``since`` / ``until`` window, time series and trade
prints are returned in full—no downsampling caps.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection


# ── helpers ───────────────────────────────────────────────────────────────


def _w(since: str, until: str) -> dict[str, str]:
    return {"since": since, "until": until}


def _f(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _bucket_seconds_sql(col: str, bucket_s: int) -> str:
    """Time-bucket expression that works on plain Postgres (no Timescale).

    Returns ``to_timestamp(floor(extract(epoch from {col})/{bucket_s})*{bucket_s})``.
    """
    return (
        f"to_timestamp(floor(extract(epoch from {col})/{bucket_s})*{bucket_s})"
        f" AT TIME ZONE 'UTC'"
    )


def _stdev(xs: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _slope(xs: list[float], ys: list[float]) -> float | None:
    """Plain OLS slope for ys ~ xs."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    if den == 0:
        return None
    return num / den


# ── raw windows ───────────────────────────────────────────────────────────


_L2_FLAT_LEVELS = 5   # per-level px/sz/n emitted flat, both feeds
_L2_DEPTH_BANDS = (5, 10, 20)  # cumulative depth bands, deep feed only


def _side_levels(raw: Any, best_px: float | None) -> tuple[list[tuple[float, float, int]], dict]:
    """Parse one side's JSONB into (levels, aggregates).

    Aggregates mirror the stored 5s_bxt definitions so a raw window can be used
    to audit a bucket: ``size`` is the top-N total, ``impact_bps`` the
    size-weighted cost of consuming them, ``span_ticks`` their reach.
    """
    lv: list[tuple[float, float, int]] = []
    for e in _parse_jsonb_levels(raw):
        if isinstance(e, str):
            try:
                e = json.loads(e)
            except Exception:
                continue
        if not isinstance(e, dict):
            continue
        px, sz = _f(e.get("px")), _f(e.get("sz"))
        if px is None or sz is None or sz <= 0:
            continue
        lv.append((px, sz, int(e.get("n") or 1)))
    if not lv:
        return [], {}

    pxs = sorted(p[0] for p in lv)
    gaps = [round(pxs[i + 1] - pxs[i], 10) for i in range(len(pxs) - 1) if pxs[i + 1] - pxs[i] > 0]
    tick = min(gaps) if gaps else None
    agg: dict[str, Any] = {"tick": tick, "n": sum(p[2] for p in lv)}
    for band in _L2_DEPTH_BANDS:
        sub = lv[:band]
        agg[f"size_{band}"] = sum(p[1] for p in sub)
    if best_px and best_px > 0:
        top = lv[:_L2_FLAT_LEVELS]
        tot = sum(p[1] for p in top)
        if tot > 0:
            vwap = sum(p[0] * p[1] for p in top) / tot
            agg["impact_bps"] = abs(10000.0 * (vwap - best_px) / best_px)
        agg["size_usd"] = agg.get(f"size_{_L2_DEPTH_BANDS[-1]}", 0.0) * best_px
        if tick:
            agg["span_ticks"] = round(abs(pxs[-1] - pxs[0]) / tick) + 1
    return lv, agg


def _raw_l2_window_impl(
    conn: Connection, schema: str, since: str, until: str,
    *, table: str, kind: str, deep: bool = False,
) -> dict[str, Any]:
    """Shared L2-window builder used by both the 5-level fast feed
    (``l2_snapshots``) and the 20-level REST feed (``l2_deep``).

    Emits per-level prices and SIZES, not just the touch. Without them the book,
    depth and replenishment features in ``signal_buckets`` cannot be audited
    against raw at all — and for ``l2_deep`` the sizes below level 5 are the only
    reason the table exists.
    """
    r = conn.execute(
        text(
            f"""
            SELECT ts, best_bid, best_ask, mid_price, spread,
                   n_bid_levels, n_ask_levels, bids, asks
            FROM {schema}.{table}
            WHERE ts >= :since AND ts < :until
            ORDER BY ts
            """
        ),
        {"since": since, "until": until},
    )
    rows = r.fetchall()
    if not rows:
        return {"kind": kind, "asset": schema, "window": _w(since, until),
                "summary": {"snapshots": 0}, "series": []}

    total = len(rows)
    series = []
    mids: list[float] = []
    spreads: list[float] = []
    bid_lv: list[int] = []
    ask_lv: list[int] = []
    bid_sizes: list[float] = []
    ask_sizes: list[float] = []
    imbalances: list[float] = []
    impacts: list[float] = []
    top_band = _L2_DEPTH_BANDS[-1] if deep else _L2_FLAT_LEVELS
    for row in rows:
        ts, bb, ba, mid, sp, nbl, nal, bids_raw, asks_raw = row
        mid_f, sp_f = _f(mid), _f(sp)
        bb_f, ba_f = _f(bb), _f(ba)
        if mid_f is not None:
            mids.append(mid_f)
        if sp_f is not None:
            spreads.append(sp_f)
        if nbl is not None:
            bid_lv.append(int(nbl))
        if nal is not None:
            ask_lv.append(int(nal))

        blv, bagg = _side_levels(bids_raw, bb_f)
        alv, aagg = _side_levels(asks_raw, ba_f)

        rec: dict[str, Any] = {
            "ts": ts.isoformat(),
            "best_bid": bb_f, "best_ask": ba_f,
            "mid": mid_f, "spread": sp_f,
            "bid_levels": nbl, "ask_levels": nal,
        }
        tick = min([t for t in (bagg.get("tick"), aagg.get("tick")) if t], default=None)
        rec["tick"] = tick
        rec["spread_ticks"] = (sp_f / tick) if (tick and sp_f is not None) else None

        for i in range(_L2_FLAT_LEVELS):
            b = blv[i] if i < len(blv) else (None, None, None)
            a = alv[i] if i < len(alv) else (None, None, None)
            rec[f"bid_px{i + 1}"], rec[f"bid_sz{i + 1}"], rec[f"bid_n{i + 1}"] = b
            rec[f"ask_px{i + 1}"], rec[f"ask_sz{i + 1}"], rec[f"ask_n{i + 1}"] = a

        bands = _L2_DEPTH_BANDS if deep else (_L2_FLAT_LEVELS,)
        for band in bands:
            rec[f"bid_size_{band}"] = bagg.get(f"size_{band}")
            rec[f"ask_size_{band}"] = aagg.get(f"size_{band}")
        rec["bid_size_usd"] = bagg.get("size_usd")
        rec["ask_size_usd"] = aagg.get("size_usd")
        rec["bid_ppl"] = bagg.get("n")
        rec["ask_ppl"] = aagg.get("n")
        rec["bid_impact_bps"] = bagg.get("impact_bps")
        rec["ask_impact_bps"] = aagg.get("impact_bps")
        rec["bid_span_ticks"] = bagg.get("span_ticks")
        rec["ask_span_ticks"] = aagg.get("span_ticks")

        bs, asz = bagg.get(f"size_{top_band}"), aagg.get(f"size_{top_band}")
        rec["imbalance"] = ((bs - asz) / (bs + asz)) if (bs and asz and (bs + asz) > 0) else None
        if bs is not None:
            bid_sizes.append(bs)
        if asz is not None:
            ask_sizes.append(asz)
        if rec["imbalance"] is not None:
            imbalances.append(rec["imbalance"])
        for v in (bagg.get("impact_bps"), aagg.get("impact_bps")):
            if v is not None:
                impacts.append(v)
        series.append(rec)

    def _avg(xs: list[float]) -> float | None:
        return (sum(xs) / len(xs)) if xs else None

    return {
        "kind": kind,
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "snapshots": total,
            "levels_flat": _L2_FLAT_LEVELS,
            "depth_bands": list(_L2_DEPTH_BANDS) if deep else [_L2_FLAT_LEVELS],
            "avg_bid_levels": _avg([float(x) for x in bid_lv]),
            "avg_ask_levels": _avg([float(x) for x in ask_lv]),
            "mid_first": mids[0] if mids else None,
            "mid_last":  mids[-1] if mids else None,
            "mid_min":   min(mids) if mids else None,
            "mid_max":   max(mids) if mids else None,
            "mid_change_bps": (
                10000.0 * (mids[-1] - mids[0]) / mids[0]
                if len(mids) >= 2 and mids[0] else None
            ),
            "spread_avg": _avg(spreads),
            "spread_max": max(spreads) if spreads else None,
            f"bid_size_{top_band}_avg": _avg(bid_sizes),
            f"ask_size_{top_band}_avg": _avg(ask_sizes),
            "imbalance_avg": _avg(imbalances),
            "impact_bps_avg": _avg(impacts),
        },
        "series": series,
    }


def raw_l2_window(
    conn: Connection, schema: str, since: str, until: str,
) -> dict[str, Any]:
    """Every 5-level fast L2 snapshot in the window, with per-level px/sz/n."""
    return _raw_l2_window_impl(
        conn, schema, since, until, table="l2_snapshots", kind="raw_l2_window",
    )


def raw_l2_deep_window(
    conn: Connection, schema: str, since: str, until: str,
) -> dict[str, Any]:
    """Every 20-level REST L2 snapshot (``l2_deep``) in the window.

    Same shape as ``raw_l2_window`` plus cumulative depth at 5 / 10 / 20 levels,
    which is what answers "do deeper levels add edge?" without emitting 80
    per-level columns per row at 1 Hz.
    """
    return _raw_l2_window_impl(
        conn, schema, since, until, table="l2_deep", kind="raw_l2_deep_window",
        deep=True,
    )


def raw_marks_window(
    conn: Connection, schema: str, since: str, until: str,
) -> dict[str, Any]:
    """Every mark row in the window + slopes."""
    r = conn.execute(
        text(
            f"""
            SELECT ts, mark_px, oracle_px, mid_px, premium,
                   funding, open_interest, day_ntl_vlm
            FROM {schema}.mark_price
            WHERE ts >= :since AND ts < :until
            ORDER BY ts
            """
        ),
        {"since": since, "until": until},
    )
    rows = r.fetchall()
    if not rows:
        return {"kind": "raw_marks_window", "asset": schema, "window": _w(since, until),
                "summary": {"rows": 0}, "series": []}

    total = len(rows)
    series = []
    funds, ois, prems = [], [], []
    ts_epoch: list[float] = []
    for row in rows:
        ts, mark, oracle, mid, premium, funding, oi, vol = row
        series.append({
            "ts": ts.isoformat(),
            "mark": _f(mark), "oracle": _f(oracle), "mid": _f(mid),
            "premium": _f(premium), "funding": _f(funding),
            "open_interest": _f(oi), "day_ntl_vlm": _f(vol),
        })
        if _f(funding) is not None: funds.append(_f(funding))
        if _f(oi) is not None: ois.append(_f(oi))
        if _f(premium) is not None: prems.append(_f(premium))
        ts_epoch.append(ts.timestamp())

    return {
        "kind": "raw_marks_window",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "rows": total,
            "funding_first": funds[0] if funds else None,
            "funding_last":  funds[-1] if funds else None,
            "funding_slope_per_s": _slope(ts_epoch[:len(funds)], funds) if len(funds) >= 2 else None,
            "oi_first": ois[0] if ois else None,
            "oi_last":  ois[-1] if ois else None,
            "oi_change_pct": (
                100.0 * (ois[-1] - ois[0]) / ois[0]
                if len(ois) >= 2 and ois[0] else None
            ),
            "premium_avg": (sum(prems) / len(prems)) if prems else None,
        },
        "series": series,
    }


def raw_trades_window(
    conn: Connection, schema: str, since: str, until: str,
) -> dict[str, Any]:
    """Every matched trade print in the window."""
    r = conn.execute(
        text(f"SELECT COUNT(*) FROM {schema}.trades WHERE ts >= :s AND ts < :u"),
        {"s": since, "u": until},
    )
    n = int(r.scalar() or 0)

    if n == 0:
        return {"kind": "raw_trades_window", "asset": schema,
                "window": _w(since, until), "summary": {"trades": 0}, "series": []}

    rows = conn.execute(
        text(
            f"""
            SELECT ts, side, px, sz
            FROM {schema}.trades
            WHERE ts >= :s AND ts < :u
            ORDER BY ts
            """
        ),
        {"s": since, "u": until},
    ).fetchall()
    series = [{"ts": r0.isoformat(), "side": s, "px": _f(p), "sz": _f(z)}
              for r0, s, p, z in rows]

    total_buy = sum((s.get("sz") or 0) for s in series if s.get("side") == "B")
    total_sell = sum((s.get("sz") or 0) for s in series if s.get("side") == "A")

    return {
        "kind": "raw_trades_window",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "trades": n,
            "form": "individual",
            "buy_volume": total_buy,
            "sell_volume": total_sell,
            "imbalance": (
                (total_buy - total_sell) / (total_buy + total_sell)
                if (total_buy + total_sell) > 0 else None
            ),
        },
        "series": series,
    }


# ── aggregated bars ───────────────────────────────────────────────────────


def ohlcv(
    conn: Connection, schema: str, since: str, until: str, interval_s: int = 60
) -> dict[str, Any]:
    bucket = _bucket_seconds_sql("ts", interval_s)
    rows = conn.execute(
        text(
            f"""
            SELECT {bucket} AS b,
                   (array_agg(px ORDER BY ts ASC))[1]  AS o,
                   MAX(px)                              AS h,
                   MIN(px)                              AS l,
                   (array_agg(px ORDER BY ts DESC))[1] AS c,
                   COALESCE(SUM(sz), 0)                 AS v,
                   COUNT(*)                             AS n
            FROM {schema}.trades
            WHERE ts >= :s AND ts < :u
            GROUP BY 1 ORDER BY 1
            """
        ),
        {"s": since, "u": until},
    ).fetchall()

    series = [
        {"ts": b.isoformat(), "o": _f(o), "h": _f(h),
         "l": _f(l), "c": _f(c), "v": _f(v), "n": int(n)}
        for b, o, h, l, c, v, n in rows
    ]

    closes = [s["c"] for s in series if s["c"] is not None]
    return {
        "kind": f"ohlcv_{interval_s}s",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "bars": len(series),
            "interval_s": interval_s,
            "first": closes[0] if closes else None,
            "last":  closes[-1] if closes else None,
            "high":  max((s["h"] for s in series if s["h"] is not None), default=None),
            "low":   min((s["l"] for s in series if s["l"] is not None), default=None),
            "total_volume": sum((s["v"] or 0) for s in series),
            "change_pct": (
                100.0 * (closes[-1] - closes[0]) / closes[0]
                if len(closes) >= 2 and closes[0] else None
            ),
        },
        "series": series,
    }


def ohlc_marks(
    conn: Connection, schema: str, since: str, until: str, interval_s: int = 60
) -> dict[str, Any]:
    """OHLC bars built from ``mark_price.mark_px`` (the exchange mark, ~1 Hz).

    Deliberately OHLC-only — no volume. ``mark_price`` carries no per-interval
    traded size (only the cumulative ``day_ntl_vlm`` running total), so a
    meaningful per-bar volume must come from the trade prints / 5s buckets
    (see ``trade_rate`` / ``taker_imbalance`` / ``signal_buckets``), not from
    the mark series. ``n`` is the number of mark observations in the bar, which
    doubles as a data-density / staleness check.
    """
    bucket = _bucket_seconds_sql("ts", interval_s)
    rows = conn.execute(
        text(
            f"""
            SELECT {bucket} AS b,
                   (array_agg(mark_px ORDER BY ts ASC))[1]  AS o,
                   MAX(mark_px)                             AS h,
                   MIN(mark_px)                             AS l,
                   (array_agg(mark_px ORDER BY ts DESC))[1] AS c,
                   COUNT(*)                                 AS n
            FROM {schema}.mark_price
            WHERE ts >= :s AND ts < :u AND mark_px IS NOT NULL
            GROUP BY 1 ORDER BY 1
            """
        ),
        {"s": since, "u": until},
    ).fetchall()

    series = [
        {"ts": b.isoformat(), "o": _f(o), "h": _f(h),
         "l": _f(l), "c": _f(c), "n": int(n)}
        for b, o, h, l, c, n in rows
    ]

    closes = [s["c"] for s in series if s["c"] is not None]
    return {
        "kind": f"ohlc_marks_{interval_s}s",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "bars": len(series),
            "interval_s": interval_s,
            "source": "mark_price.mark_px",
            "first": closes[0] if closes else None,
            "last":  closes[-1] if closes else None,
            "high":  max((s["h"] for s in series if s["h"] is not None), default=None),
            "low":   min((s["l"] for s in series if s["l"] is not None), default=None),
            "change_pct": (
                100.0 * (closes[-1] - closes[0]) / closes[0]
                if len(closes) >= 2 and closes[0] else None
            ),
        },
        "series": series,
    }


# ── microstructure (l2) ───────────────────────────────────────────────────


def _l2_top_levels(bids_or_asks: Any) -> list[tuple[float, float]]:
    """Pull (px, sz) from JSONB bids/asks; include every posted level."""
    if bids_or_asks is None:
        return []
    if isinstance(bids_or_asks, str):
        try:
            bids_or_asks = json.loads(bids_or_asks)
        except Exception:
            return []
    out: list[tuple[float, float]] = []
    for entry in (bids_or_asks or []):
        if isinstance(entry, dict):
            px = _f(entry.get("px") or entry.get("price"))
            sz = _f(entry.get("sz") or entry.get("size"))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            px = _f(entry[0])
            sz = _f(entry[1])
        else:
            continue
        if px is not None and sz is not None:
            out.append((px, sz))
    return out


def l2_imbalance(
    conn: Connection, schema: str, since: str, until: str,
) -> dict[str, Any]:
    rows = conn.execute(
        text(
            f"""
            SELECT ts, bids, asks
            FROM {schema}.l2_snapshots
            WHERE ts >= :s AND ts < :u
            ORDER BY ts
            """
        ),
        {"s": since, "u": until},
    ).fetchall()

    series, ratios = [], []
    for ts, bids, asks in rows:
        b = _l2_top_levels(bids)
        a = _l2_top_levels(asks)
        bid_sz = sum(sz for _, sz in b)
        ask_sz = sum(sz for _, sz in a)
        if (bid_sz + ask_sz) <= 0:
            continue
        imb = (bid_sz - ask_sz) / (bid_sz + ask_sz)
        ratios.append(imb)
        series.append({
            "ts": ts.isoformat(),
            "bid_sz": bid_sz,
            "ask_sz": ask_sz,
            "imbalance":   imb,  # signed -1..+1
        })

    return {
        "kind": "l2_imbalance",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "snapshots": len(series),
            "mean_imbalance": (sum(ratios) / len(ratios)) if ratios else None,
            "min_imbalance":  min(ratios) if ratios else None,
            "max_imbalance":  max(ratios) if ratios else None,
        },
        "series": series,
    }


def microprice(
    conn: Connection, schema: str, since: str, until: str,
) -> dict[str, Any]:
    rows = conn.execute(
        text(
            f"""
            SELECT ts, bids, asks, best_bid, best_ask, mid_price
            FROM {schema}.l2_snapshots
            WHERE ts >= :s AND ts < :u
            ORDER BY ts
            """
        ),
        {"s": since, "u": until},
    ).fetchall()

    series = []
    for ts, bids, asks, bb, ba, mid in rows:
        b = _l2_top_levels(bids)
        a = _l2_top_levels(asks)
        bid_px = b[0][0] if b else _f(bb)
        ask_px = a[0][0] if a else _f(ba)
        bid_sz = b[0][1] if b else None
        ask_sz = a[0][1] if a else None
        mp = None
        if (bid_px is not None and ask_px is not None
                and bid_sz is not None and ask_sz is not None
                and (bid_sz + ask_sz) > 0):
            mp = (bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz)
        mid_f = _f(mid)
        series.append({
            "ts": ts.isoformat(),
            "microprice": mp,
            "mid": mid_f,
            "skew_bps": (
                10000.0 * (mp - mid_f) / mid_f
                if mp is not None and mid_f else None
            ),
        })

    skews = [s["skew_bps"] for s in series if s["skew_bps"] is not None]
    return {
        "kind": "microprice",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "snapshots": len(series),
            "skew_avg_bps": (sum(skews) / len(skews)) if skews else None,
            "skew_max_bps": max(skews) if skews else None,
            "skew_min_bps": min(skews) if skews else None,
        },
        "series": series,
    }


def top_of_book_depth(
    conn: Connection, schema: str, since: str, until: str,
) -> dict[str, Any]:
    rows = conn.execute(
        text(
            f"""
            SELECT ts, bids, asks, mid_price
            FROM {schema}.l2_snapshots
            WHERE ts >= :s AND ts < :u
            ORDER BY ts
            """
        ),
        {"s": since, "u": until},
    ).fetchall()

    series = []
    for ts, bids, asks, mid in rows:
        b = _l2_top_levels(bids)
        a = _l2_top_levels(asks)
        bid_usd = sum(px * sz for px, sz in b)
        ask_usd = sum(px * sz for px, sz in a)
        series.append({
            "ts": ts.isoformat(),
            "bid_depth_usd": bid_usd,
            "ask_depth_usd": ask_usd,
            "total_depth_usd": bid_usd + ask_usd,
        })

    totals = [s["total_depth_usd"] for s in series]
    return {
        "kind": "top_of_book_depth",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "snapshots": len(series),
            "avg_depth_usd": (sum(totals) / len(totals)) if totals else None,
            "min_depth_usd": min(totals) if totals else None,
            "max_depth_usd": max(totals) if totals else None,
        },
        "series": series,
    }


def effective_spread(
    conn: Connection, schema: str, since: str, until: str,
) -> dict[str, Any]:
    rows = conn.execute(
        text(
            f"""
            SELECT ts, best_bid, best_ask, mid_price, spread
            FROM {schema}.l2_snapshots
            WHERE ts >= :s AND ts < :u
            ORDER BY ts
            """
        ),
        {"s": since, "u": until},
    ).fetchall()

    series = []
    bps_list: list[float] = []
    for ts, bb, ba, mid, sp in rows:
        mid_f = _f(mid); sp_f = _f(sp)
        bps = (10000.0 * sp_f / mid_f) if (sp_f is not None and mid_f) else None
        if bps is not None:
            bps_list.append(bps)
        series.append({
            "ts": ts.isoformat(),
            "spread_abs": sp_f,
            "spread_bps": bps,
        })

    return {
        "kind": "effective_spread",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "snapshots": len(series),
            "spread_bps_avg": (sum(bps_list) / len(bps_list)) if bps_list else None,
            "spread_bps_p50": sorted(bps_list)[len(bps_list)//2] if bps_list else None,
            "spread_bps_max": max(bps_list) if bps_list else None,
        },
        "series": series,
    }


def quote_churn_rate(
    conn: Connection, schema: str, since: str, until: str, bucket_s: int = 10,
) -> dict[str, Any]:
    bucket = _bucket_seconds_sql("ts", bucket_s)
    rows = conn.execute(
        text(
            f"""
            SELECT {bucket} AS b, COUNT(*) AS snaps
            FROM {schema}.l2_snapshots
            WHERE ts >= :s AND ts < :u
            GROUP BY 1 ORDER BY 1
            """
        ),
        {"s": since, "u": until},
    ).fetchall()

    series = [{"ts": b.isoformat(), "snaps_per_bucket": int(c),
               "snaps_per_s": float(c) / bucket_s} for b, c in rows]
    rates = [s["snaps_per_s"] for s in series]
    return {
        "kind": "quote_churn_rate",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "bucket_s": bucket_s,
            "buckets": len(series),
            "rate_avg": (sum(rates) / len(rates)) if rates else None,
            "rate_max": max(rates) if rates else None,
            "rate_min": min(rates) if rates else None,
        },
        "series": series,
    }


# ── price dynamics (mid derivatives + vol) ────────────────────────────────


def mid_derivatives(
    conn: Connection, schema: str, since: str, until: str,
    order: int = 1,
) -> dict[str, Any]:
    """Finite-difference derivatives on every mid observation in the window."""
    order = max(1, min(int(order), 3))
    rows = conn.execute(
        text(
            f"""
            SELECT ts, mid_price
            FROM {schema}.l2_snapshots
            WHERE ts >= :s AND ts < :u AND mid_price IS NOT NULL
            ORDER BY ts
            """
        ),
        {"s": since, "u": until},
    ).fetchall()

    if len(rows) < order + 1:
        return {"kind": f"mid_derivatives_{order}", "asset": schema,
                "window": _w(since, until),
                "summary": {"samples": len(rows), "order": order}, "series": []}

    ts_list = [r0.timestamp() for r0, _ in rows]
    vals = [float(m) for _, m in rows]

    series_per_order: list[list[float | None]] = [vals]
    for _ in range(order):
        prev = series_per_order[-1]
        cur: list[float | None] = [None]
        for i in range(1, len(prev)):
            dt = ts_list[i] - ts_list[i - 1]
            if dt <= 0 or prev[i] is None or prev[i - 1] is None:
                cur.append(None); continue
            cur.append((prev[i] - prev[i - 1]) / dt)
        series_per_order.append(cur)

    deriv = series_per_order[order]
    out_series = [
        {"ts": rows[i][0].isoformat(),
         "mid": vals[i],
         "deriv": deriv[i]}
        for i in range(len(rows))
    ]
    cleaned = [d for d in deriv if d is not None]
    return {
        "kind": f"mid_derivatives_{order}",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "order": order,
            "samples": len(out_series),
            "mean": (sum(cleaned) / len(cleaned)) if cleaned else None,
            "stdev": _stdev(cleaned),
            "abs_max": max((abs(x) for x in cleaned), default=None),
        },
        "series": out_series,
    }


def realized_vol(
    conn: Connection, schema: str, since: str, until: str,
    win_s: int = 60,
) -> dict[str, Any]:
    """Rolling standard deviation of log returns on the full mid series."""
    rows = conn.execute(
        text(
            f"""
            SELECT ts, mid_price
            FROM {schema}.l2_snapshots
            WHERE ts >= :s AND ts < :u AND mid_price > 0
            ORDER BY ts
            """
        ),
        {"s": since, "u": until},
    ).fetchall()

    if len(rows) < 3:
        return {"kind": "realized_vol", "asset": schema,
                "window": _w(since, until),
                "summary": {"samples": len(rows)}, "series": []}

    ts = [r0 for r0, _ in rows]
    log_r: list[tuple[float, float]] = []
    for i in range(1, len(rows)):
        p1 = float(rows[i - 1][1]); p2 = float(rows[i][1])
        if p1 > 0 and p2 > 0:
            log_r.append((ts[i].timestamp(), math.log(p2 / p1)))

    series = []
    for i, (t_epoch, _) in enumerate(log_r):
        window_returns = [r for (te, r) in log_r if t_epoch - te < win_s and te <= t_epoch]
        if len(window_returns) >= 3:
            sd = _stdev(window_returns)
            series.append({
                "ts": ts[i + 1].isoformat(),
                "rv_window": sd,
                "rv_window_bps": (sd * 10000.0) if sd is not None else None,
            })

    rvs = [s["rv_window"] for s in series if s["rv_window"] is not None]
    return {
        "kind": "realized_vol",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "samples": len(series),
            "win_s": win_s,
            "rv_avg": (sum(rvs) / len(rvs)) if rvs else None,
            "rv_max": max(rvs) if rvs else None,
        },
        "series": series,
    }


def range_expansion(
    conn: Connection, schema: str, since: str, until: str, bucket_s: int = 60,
) -> dict[str, Any]:
    """Bar-over-bar high–low range expansion ratio."""
    bars = ohlcv(conn, schema, since, until, interval_s=bucket_s)["series"]
    series, ratios = [], []
    prev_range = None
    for b in bars:
        if b["h"] is None or b["l"] is None:
            continue
        rng = b["h"] - b["l"]
        ratio = (rng / prev_range) if (prev_range is not None and prev_range > 0) else None
        if ratio is not None:
            ratios.append(ratio)
        series.append({"ts": b["ts"], "range": rng, "expansion_ratio": ratio})
        prev_range = rng

    return {
        "kind": "range_expansion",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "bucket_s": bucket_s,
            "bars": len(series),
            "expansion_avg": (sum(ratios) / len(ratios)) if ratios else None,
            "expansion_max": max(ratios) if ratios else None,
        },
        "series": series,
    }


# ── flow ──────────────────────────────────────────────────────────────────


def taker_imbalance(
    conn: Connection, schema: str, since: str, until: str, bucket_s: int = 5,
) -> dict[str, Any]:
    bucket = _bucket_seconds_sql("ts", bucket_s)
    rows = conn.execute(
        text(
            f"""
            SELECT {bucket} AS b,
                   COALESCE(SUM(sz) FILTER (WHERE side='B'), 0) AS v_buy,
                   COALESCE(SUM(sz) FILTER (WHERE side='A'), 0) AS v_sell
            FROM {schema}.trades
            WHERE ts >= :s AND ts < :u
            GROUP BY 1 ORDER BY 1
            """
        ),
        {"s": since, "u": until},
    ).fetchall()

    series, imbs = [], []
    for b, vb, vs in rows:
        vb_f = _f(vb) or 0.0; vs_f = _f(vs) or 0.0
        tot = vb_f + vs_f
        imb = ((vb_f - vs_f) / tot) if tot > 0 else None
        if imb is not None: imbs.append(imb)
        series.append({"ts": b.isoformat(), "v_buy": vb_f, "v_sell": vs_f,
                       "imbalance": imb})

    return {
        "kind": "taker_imbalance",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "bucket_s": bucket_s,
            "buckets": len(series),
            "imbalance_avg": (sum(imbs) / len(imbs)) if imbs else None,
            "imbalance_max": max(imbs) if imbs else None,
            "imbalance_min": min(imbs) if imbs else None,
        },
        "series": series,
    }


def trade_rate(
    conn: Connection, schema: str, since: str, until: str, bucket_s: int = 5,
) -> dict[str, Any]:
    bucket = _bucket_seconds_sql("ts", bucket_s)
    rows = conn.execute(
        text(
            f"""
            SELECT {bucket} AS b,
                   COUNT(*)             AS n,
                   COALESCE(SUM(sz), 0) AS v
            FROM {schema}.trades
            WHERE ts >= :s AND ts < :u
            GROUP BY 1 ORDER BY 1
            """
        ),
        {"s": since, "u": until},
    ).fetchall()

    series = [{"ts": b.isoformat(), "n_trades": int(n),
               "trades_per_s": float(n) / bucket_s,
               "volume": _f(v),
               "vwap_volume_per_s": (_f(v) or 0.0) / bucket_s}
              for b, n, v in rows]
    rates = [s["trades_per_s"] for s in series]
    return {
        "kind": "trade_rate",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "bucket_s": bucket_s,
            "buckets": len(series),
            "rate_avg": (sum(rates) / len(rates)) if rates else None,
            "rate_max": max(rates) if rates else None,
            "total_volume": sum((s["volume"] or 0) for s in series),
        },
        "series": series,
    }


def volume_profile(
    conn: Connection, schema: str, since: str, until: str, n_buckets: int = 30,
) -> dict[str, Any]:
    """Histogram of size-weighted trade volume by price bucket."""
    rng = conn.execute(
        text(
            f"""
            SELECT MIN(px), MAX(px) FROM {schema}.trades
            WHERE ts >= :s AND ts < :u
            """
        ),
        {"s": since, "u": until},
    ).fetchone()

    if not rng or rng[0] is None or rng[1] is None or float(rng[1]) <= float(rng[0]):
        return {"kind": "volume_profile", "asset": schema,
                "window": _w(since, until), "summary": {"buckets": 0}, "series": []}

    lo = float(rng[0]); hi = float(rng[1])
    width = (hi - lo) / n_buckets
    rows = conn.execute(
        text(
            f"""
            SELECT
                width_bucket(px, :lo, :hi, :n) AS bk,
                COALESCE(SUM(sz), 0) AS v_total,
                COALESCE(SUM(sz) FILTER (WHERE side='B'), 0) AS v_buy,
                COALESCE(SUM(sz) FILTER (WHERE side='A'), 0) AS v_sell
            FROM {schema}.trades
            WHERE ts >= :s AND ts < :u
            GROUP BY 1 ORDER BY 1
            """
        ),
        {"s": since, "u": until, "lo": lo, "hi": hi, "n": n_buckets},
    ).fetchall()

    series = []
    for bk, vt, vb, vs in rows:
        if bk is None or bk < 1 or bk > n_buckets:
            continue
        px_lo = lo + (bk - 1) * width
        series.append({
            "px_lo": px_lo, "px_hi": px_lo + width,
            "v_total": _f(vt), "v_buy": _f(vb), "v_sell": _f(vs),
        })

    poc = max(series, key=lambda s: s["v_total"] or 0) if series else None
    return {
        "kind": "volume_profile",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "buckets": n_buckets,
            "px_low": lo, "px_high": hi,
            "poc_px": ((poc["px_lo"] + poc["px_hi"]) / 2) if poc else None,
            "poc_volume": poc["v_total"] if poc else None,
            "total_volume": sum((s["v_total"] or 0) for s in series),
        },
        "series": series,
    }


# ── derivatives ───────────────────────────────────────────────────────────


def funding_trajectory(
    conn: Connection, schema: str, since: str, until: str,
) -> dict[str, Any]:
    rows = conn.execute(
        text(
            f"""
            SELECT ts, funding
            FROM {schema}.mark_price
            WHERE ts >= :s AND ts < :u AND funding IS NOT NULL
            ORDER BY ts
            """
        ),
        {"s": since, "u": until},
    ).fetchall()

    if not rows:
        return {"kind": "funding_trajectory", "asset": schema,
                "window": _w(since, until), "summary": {"rows": 0}, "series": []}

    series = [{"ts": r0.isoformat(), "funding": float(f)} for r0, f in rows]
    epochs = [r0.timestamp() for r0, _ in rows]
    vals = [float(f) for _, f in rows]
    return {
        "kind": "funding_trajectory",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "rows": len(series),
            "first": vals[0], "last": vals[-1],
            "avg":   sum(vals) / len(vals),
            "slope_per_s": _slope(epochs, vals),
            "slope_per_h": ((_slope(epochs, vals) or 0.0) * 3600.0) if len(vals) >= 2 else None,
        },
        "series": series,
    }


def oi_change_rate(
    conn: Connection, schema: str, since: str, until: str,
) -> dict[str, Any]:
    rows = conn.execute(
        text(
            f"""
            SELECT ts, open_interest
            FROM {schema}.mark_price
            WHERE ts >= :s AND ts < :u AND open_interest IS NOT NULL
            ORDER BY ts
            """
        ),
        {"s": since, "u": until},
    ).fetchall()

    if len(rows) < 2:
        return {"kind": "oi_change_rate", "asset": schema,
                "window": _w(since, until), "summary": {"rows": len(rows)}, "series": []}

    series = []
    for i in range(1, len(rows)):
        t0, o0 = rows[i - 1]; t1, o1 = rows[i]
        dt = (t1 - t0).total_seconds()
        if dt <= 0:
            continue
        do = float(o1) - float(o0)
        series.append({
            "ts": t1.isoformat(),
            "open_interest": float(o1),
            "doi_per_s": do / dt,
            "pct_change": (100.0 * do / float(o0)) if float(o0) else None,
        })

    return {
        "kind": "oi_change_rate",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "rows": len(series),
            "first_oi": float(rows[0][1]),
            "last_oi":  float(rows[-1][1]),
            "total_change_pct": (
                100.0 * (float(rows[-1][1]) - float(rows[0][1])) / float(rows[0][1])
                if float(rows[0][1]) else None
            ),
        },
        "series": series,
    }


def premium_decay(
    conn: Connection, schema: str, since: str, until: str,
) -> dict[str, Any]:
    rows = conn.execute(
        text(
            f"""
            SELECT ts, mark_px, mid_px, premium
            FROM {schema}.mark_price
            WHERE ts >= :s AND ts < :u
            ORDER BY ts
            """
        ),
        {"s": since, "u": until},
    ).fetchall()

    series = []
    bps_list: list[float] = []
    for ts, mark, mid, prem in rows:
        mark_f = _f(mark); mid_f = _f(mid)
        bps = None
        if mark_f is not None and mid_f and mid_f != 0:
            bps = 10000.0 * (mark_f - mid_f) / mid_f
            bps_list.append(bps)
        series.append({
            "ts": ts.isoformat(),
            "mark": mark_f, "mid": mid_f,
            "premium": _f(prem),
            "premium_bps": bps,
        })

    return {
        "kind": "premium_decay",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "rows": len(series),
            "premium_bps_avg": (sum(bps_list) / len(bps_list)) if bps_list else None,
            "premium_bps_max": max(bps_list) if bps_list else None,
            "premium_bps_min": min(bps_list) if bps_list else None,
        },
        "series": series,
    }


def oracle_drift(
    conn: Connection, schema: str, since: str, until: str,
) -> dict[str, Any]:
    rows = conn.execute(
        text(
            f"""
            SELECT ts, mark_px, oracle_px
            FROM {schema}.mark_price
            WHERE ts >= :s AND ts < :u
            ORDER BY ts
            """
        ),
        {"s": since, "u": until},
    ).fetchall()

    series = []
    bps_list: list[float] = []
    for ts, mark, oracle in rows:
        mk = _f(mark); oc = _f(oracle)
        bps = None
        if mk is not None and oc and oc != 0:
            bps = 10000.0 * (mk - oc) / oc
            bps_list.append(bps)
        series.append({
            "ts": ts.isoformat(),
            "mark": mk, "oracle": oc, "drift_bps": bps,
        })

    return {
        "kind": "oracle_drift",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "rows": len(series),
            "drift_bps_avg": (sum(bps_list) / len(bps_list)) if bps_list else None,
            "drift_bps_abs_max": max((abs(b) for b in bps_list), default=None),
        },
        "series": series,
    }


# ── signal buckets (DERIV framework) ──────────────────────────────────────


# Book metrics are normalized to the top-N nearest levels so that historical
# 20-level snapshots and the live 5-level ``fast`` l2Book feed are directly
# comparable across the 2026-06 feed change.
_BOOK_LEVELS = 5

# Fallback tick sizes per asset — only used when the per-snapshot spacing can't
# be inferred (< 2 levels). The fill grid normally derives an *effective tick*
# straight from the actual posted level spacing (HL's tick is price-dependent,
# so a single static value is wrong across regimes). These fallbacks reflect
# observed spacing at current price levels.
_TICK_SIZE: dict[str, float] = {
    "sol": 0.001,
    "btc": 1.0,
    "eth": 0.1,
    "spx": 0.1,
}


def _poll_metrics(
    levels: list[dict], best_px: float, side: str, tick_size: float,
) -> dict[str, float | None]:
    """Extract {size, ppl, fill, centroid, size_usd} from one side of one L2 JSONB array.

    Parameters
    ----------
    levels : parsed JSONB array (list of {px, sz, n} dicts)
    best_px : best bid or ask price for this poll
    side : 'bid' or 'ask' — determines direction for distance calculation
    tick_size : minimum price increment for fill ratio
    """
    if not levels or best_px is None or best_px <= 0:
        return {"size": None, "ppl": None, "fill": None, "centroid": None, "size_usd": None}

    # Clamp to the top-N nearest levels (HL returns best-first) so 20-level
    # history and the 5-level fast feed produce comparable metrics.
    parsed: list[tuple[float, float, int]] = []
    for entry in levels[:_BOOK_LEVELS]:
        if isinstance(entry, str):
            try:
                entry = json.loads(entry)
            except Exception:
                continue
        if not isinstance(entry, dict):
            continue
        px = _f(entry.get("px"))
        sz = _f(entry.get("sz"))
        n = entry.get("n")
        n_val = int(n) if n is not None else 1
        if px is None or sz is None or sz <= 0:
            continue
        parsed.append((px, sz, n_val))

    if not parsed:
        return {"size": None, "ppl": None, "fill": None, "centroid": None, "size_usd": None}

    # Effective tick = smallest gap between adjacent posted prices (data-derived,
    # robust to HL's price-dependent tick); fall back to the per-asset constant.
    sorted_px = sorted(p[0] for p in parsed)
    gaps = [round(sorted_px[i + 1] - sorted_px[i], 10)
            for i in range(len(sorted_px) - 1) if sorted_px[i + 1] - sorted_px[i] > 0]
    eff_tick = min(gaps) if gaps else (tick_size if tick_size and tick_size > 0 else None)

    total_size = 0.0
    total_ppl = 0
    weighted_distance_sum = 0.0
    occupied_ticks = set()
    far_px = None
    for px, sz, n_val in parsed:
        total_size += sz
        total_ppl += n_val
        dist = abs(px - best_px)
        weighted_distance_sum += sz * dist
        if eff_tick and eff_tick > 0:
            occupied_ticks.add(round(dist / eff_tick))
        if far_px is None:
            far_px = px
        else:
            far_px = max(far_px, px) if side == "ask" else min(far_px, px)

    if total_size <= 0:
        return {"size": None, "ppl": None, "fill": None, "centroid": None, "size_usd": None}

    total_distance = abs(far_px - best_px) if far_px is not None else 0.0
    if eff_tick and eff_tick > 0 and total_distance > 0:
        max_ticks = round(total_distance / eff_tick) + 1
    else:
        max_ticks = 1
    fill = len(occupied_ticks) / max_ticks if max_ticks > 0 else 1.0
    centroid = (weighted_distance_sum / total_size) / total_distance if total_distance > 0 else 0.0

    return {
        "size": total_size,
        "ppl": total_ppl,
        "fill": min(1.0, fill),
        "centroid": min(1.0, max(0.0, centroid)),
        "size_usd": total_size * best_px,
    }


def _parse_jsonb_levels(raw: Any) -> list[dict]:
    """Parse JSONB bids/asks into a list of dicts."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw, list):
        return []
    return raw


# ── stored 5s_bxt column layout (mirrors bxtbuilder_5s.py exactly) ────────
# Every key here must exist in bxtbuilder_5s.COMPONENT_KEYS and every one of
# those must appear in exactly one stacking group below, or a stacked bucket
# silently loses a column.
#
# State-0 book means (poll-averaged per 5s bucket → poll-weighted mean when stacked)
_BXT_BOOK_MEAN_KEYS = [
    "ask_size", "ask_size_usd", "ask_ppl", "ask_fill", "ask_centroid",
    "bid_size", "bid_size_usd", "bid_ppl", "bid_fill", "bid_centroid",
    # V2: uncapped book-cost metrics, also per-poll means
    "ask_impact_bps", "bid_impact_bps", "ask_span_ticks", "bid_span_ticks",
]
# State-0 flow totals (summed when stacked)
_BXT_FLOW_SUM_KEYS = [
    "buy_volume", "buy_volume_usd", "sell_volume", "sell_volume_usd",
]
_BXT_INT_SUM_KEYS = ["poll_count", "buy_count", "sell_count",
                     "sweep_count_buy", "sweep_count_sell"]
# V2 close-of-bucket depth: the coarse bucket's close is its LAST sub-bucket's
# close, which is what makes d_ask_size_c + buy_volume telescope correctly.
_BXT_CLOSE_KEYS = ["ask_size_c", "ask_size_c_usd", "bid_size_c", "bid_size_c_usd"]
# V2 tick scale: min, matching how the 5s value itself is derived (min gap over
# both sides and all polls), so stacking is self-consistent.
_BXT_MIN_KEYS = ["tick_size"]
# V2 sweep extremes: the largest sweep in a coarse bucket is the largest in any
# of its sub-buckets — sweeps are atomic within one millisecond and provably
# never cross a 5s boundary.
_BXT_MAX_KEYS = [
    "max_sweep_volume_buy", "max_sweep_volume_sell",
    "max_sweep_levels_buy", "max_sweep_levels_sell",
    "max_sweep_range_buy", "max_sweep_range_sell",
]
# State-0 OHLC groups (first/max/min/last + poll-weighted mean when stacked)
_BXT_OHLC_GROUPS = [
    ("mid_o", "mid_h", "mid_l", "mid_c", "mid_mean"),
    ("spread_o", "spread_h", "spread_l", "spread_c", "spread_mean"),
    ("spread_o_bps", "spread_h_bps", "spread_l_bps", "spread_c_bps", "spread_mean_bps"),
    # V2: price-invariant spread. The _bps group moves with the mid even when the
    # dollar spread is frozen (~97% of its variation), so prefer ticks.
    ("spread_o_ticks", "spread_h_ticks", "spread_l_ticks", "spread_c_ticks", "spread_mean_ticks"),
]
# The 40 keys that get first-differenced for State 1 (== bxtbuilder DIFF_KEYS)
_BXT_DIFF_KEYS = [
    "ask_size", "ask_size_usd", "ask_ppl", "ask_fill", "ask_centroid",
    "buy_volume", "buy_volume_usd", "buy_count",
    "sell_volume", "sell_volume_usd", "sell_count",
    "bid_size", "bid_size_usd", "bid_ppl", "bid_fill", "bid_centroid",
    "mid_o", "mid_h", "mid_l", "mid_c", "mid_mean",
    "spread_o", "spread_h", "spread_l", "spread_c", "spread_mean",
    "spread_o_bps", "spread_h_bps", "spread_l_bps", "spread_c_bps", "spread_mean_bps",
    # V2
    "ask_size_c", "ask_size_c_usd", "bid_size_c", "bid_size_c_usd",
    "spread_c_ticks", "ask_impact_bps", "bid_impact_bps",
    "sweep_count_buy", "sweep_count_sell",
]
_BXT_S1_REPLENISH = ["ask_replenish", "ask_replenish_usd", "bid_replenish", "bid_replenish_usd"]
_BXT_S2_REPLENISH = ["d_ask_replenish", "d_ask_replenish_usd", "d_bid_replenish", "d_bid_replenish_usd"]
_BXT_S1_REPLENISH_V2 = ["ask_replenish_c", "ask_replenish_c_usd",
                        "bid_replenish_c", "bid_replenish_c_usd"]
_BXT_S2_REPLENISH_V2 = ["d_ask_replenish_c", "d_ask_replenish_c_usd",
                        "d_bid_replenish_c", "d_bid_replenish_c_usd"]
_BXT_S0_KEYS = (
    _BXT_INT_SUM_KEYS + _BXT_BOOK_MEAN_KEYS + _BXT_FLOW_SUM_KEYS
    + _BXT_CLOSE_KEYS + _BXT_MIN_KEYS + _BXT_MAX_KEYS
    + [k for grp in _BXT_OHLC_GROUPS for k in grp]
)  # 5 + 14 + 4 + 4 + 1 + 6 + 20 = 54
_BXT_S1_KEYS = [f"d_{k}" for k in _BXT_DIFF_KEYS] + _BXT_S1_REPLENISH + _BXT_S1_REPLENISH_V2   # 48
_BXT_S2_KEYS = [f"dd_{k}" for k in _BXT_DIFF_KEYS] + _BXT_S2_REPLENISH + _BXT_S2_REPLENISH_V2  # 48
_BXT_COMPONENTS = _BXT_S0_KEYS + _BXT_S1_KEYS + _BXT_S2_KEYS                                   # 150

# Quality flags — stored alongside the components but deliberately not counted
# as components. bucket_ideal is derived (see _bxt_bucket_ideal), never stored.
_BXT_QUALITY_KEYS = ["prev_contiguous", "has_trades", "poll_gap_max_ms", "was_revised"]

# ts is the bucket START. See bxtbuilder_5s's module docstring; these numbers are
# echoed into every signal_buckets payload so a consumer can't silently build a
# label that reads 5 seconds of its own future.
BXT_TS_CONVENTION = "bucket_start"
BXT_BUCKET_END_S = 5      # ts + this = last instant of data in the row
BXT_FIRST_AVAILABLE_S = 8 # ts + this = when the builder actually wrote the row
BXT_FINAL_S = 120         # ts + this = past the reconciler settle window

_BXT_MIN_POLLS = 7        # below this the feed was throttled (June 15-24 2026)
_BXT_MAX_POLL_GAP_MS = 1500


def _bxt_bucket_ideal(row: dict[str, Any]) -> bool | None:
    """Composite quality gate: enough polls, contiguous, evenly sampled, not revised.

    "Ideal" not "ok": TRUE means the bucket is pristine — a FALSE bucket is not
    necessarily unusable, it just lacks one of those guarantees.

    Not a stored column — it depends on was_revised, which is computed inside the
    same ON CONFLICT clause that would have to write it, and a STORED generated
    column would force a hypertable rewrite. Fully derivable, so nothing is lost.
    """
    pc = row.get("poll_count")
    gap = row.get("poll_gap_max_ms")
    if pc is None:
        return None
    return bool(
        pc >= _BXT_MIN_POLLS
        and row.get("prev_contiguous")
        and gap is not None and gap < _BXT_MAX_POLL_GAP_MS
        and not row.get("was_revised")
    )


def _bxt_state1(cur: dict, prev: dict) -> dict[str, Any]:
    """First derivatives + replenishment cross-signals (bxtbuilder formulas)."""
    d: dict[str, Any] = {}
    for k in _BXT_DIFF_KEYS:
        cv, pv = cur.get(k), prev.get(k)
        d[f"d_{k}"] = (cv - pv) if (cv is not None and pv is not None) else None
    def _rep(delta_key: str, vol_key: str):
        dv, vol = d.get(delta_key), cur.get(vol_key)
        return (dv + vol) if (dv is not None and vol is not None) else None

    d["ask_replenish"] = _rep("d_ask_size", "buy_volume")
    d["ask_replenish_usd"] = _rep("d_ask_size_usd", "buy_volume_usd")
    d["bid_replenish"] = _rep("d_bid_size", "sell_volume")
    d["bid_replenish_usd"] = _rep("d_bid_size_usd", "sell_volume_usd")
    # Close-based variant: boundary-to-boundary depth change, so this is the one
    # whose accounting identity actually holds at any horizon.
    d["ask_replenish_c"] = _rep("d_ask_size_c", "buy_volume")
    d["ask_replenish_c_usd"] = _rep("d_ask_size_c_usd", "buy_volume_usd")
    d["bid_replenish_c"] = _rep("d_bid_size_c", "sell_volume")
    d["bid_replenish_c_usd"] = _rep("d_bid_size_c_usd", "sell_volume_usd")
    return d


def _bxt_state2(cur1: dict, prev1: dict) -> dict[str, Any]:
    """Second derivatives + replenishment velocity (bxtbuilder formulas)."""
    dd: dict[str, Any] = {}
    for k in _BXT_DIFF_KEYS:
        cv, pv = cur1.get(f"d_{k}"), prev1.get(f"d_{k}")
        dd[f"dd_{k}"] = (cv - pv) if (cv is not None and pv is not None) else None
    def _rep_v(dd_key: str, d_vol_key: str):
        ddv, dvol = dd.get(dd_key), cur1.get(d_vol_key)
        return (ddv + dvol) if (ddv is not None and dvol is not None) else None

    dd["d_ask_replenish"] = _rep_v("dd_ask_size", "d_buy_volume")
    dd["d_ask_replenish_usd"] = _rep_v("dd_ask_size_usd", "d_buy_volume_usd")
    dd["d_bid_replenish"] = _rep_v("dd_bid_size", "d_sell_volume")
    dd["d_bid_replenish_usd"] = _rep_v("dd_bid_size_usd", "d_sell_volume_usd")
    dd["d_ask_replenish_c"] = _rep_v("dd_ask_size_c", "d_buy_volume")
    dd["d_ask_replenish_c_usd"] = _rep_v("dd_ask_size_c_usd", "d_buy_volume_usd")
    dd["d_bid_replenish_c"] = _rep_v("dd_bid_size_c", "d_sell_volume")
    dd["d_bid_replenish_c_usd"] = _rep_v("dd_bid_size_c_usd", "d_sell_volume_usd")
    return dd


def _stack_state0(base: list[dict], eff_bucket_s: int) -> tuple[list[dict], list[int]]:
    """Coarsen 5s State-0 rows into ``eff_bucket_s`` buckets.

    Aggregation rules (so a stacked bucket == what the builder would produce at
    that horizon): counts/volumes/sweep-counts SUM, book means and impact/span
    are poll-weighted, mid/spread/spread-ticks are first-open / max-high /
    min-low / last-close / poll-weighted-mean, close depth takes the LAST
    sub-bucket's close, tick_size takes the MIN, sweep extremes take the MAX.

    Quality flags coarsen conservatively: poll_gap_max_ms MAX, has_trades and
    was_revised OR, prev_contiguous AND over the sub-buckets (the caller also
    requires coarse-bucket contiguity before trusting derivatives).
    """
    from collections import OrderedDict
    groups: "OrderedDict[int, list[dict]]" = OrderedDict()
    for row in base:  # base already ordered by ts ascending
        ck = int(row["_epoch"] // eff_bucket_s) * eff_bucket_s
        groups.setdefault(ck, []).append(row)

    keys = list(groups.keys())
    s0: list[dict] = []
    for ck in keys:
        subs = groups[ck]

        def _sum(key: str) -> float:
            return sum((x[key] or 0.0) for x in subs if x[key] is not None)

        def _wmean(key: str):
            num = den = 0.0
            for x in subs:
                v, w = x[key], x["poll_count"]
                if v is not None and w:
                    num += v * w
                    den += w
            return (num / den) if den > 0 else None

        def _first(key: str):
            for x in subs:
                if x[key] is not None:
                    return x[key]
            return None

        def _last(key: str):
            for x in reversed(subs):
                if x[key] is not None:
                    return x[key]
            return None

        def _mx(key: str):
            vals = [x[key] for x in subs if x[key] is not None]
            return max(vals) if vals else None

        def _mn(key: str):
            vals = [x[key] for x in subs if x[key] is not None]
            return min(vals) if vals else None

        row: dict[str, Any] = {"_bucket_ts": ck, "_n_sub": len(subs)}
        for k in _BXT_INT_SUM_KEYS:
            row[k] = int(_sum(k))
        for k in _BXT_FLOW_SUM_KEYS:
            row[k] = _sum(k)
        for k in _BXT_BOOK_MEAN_KEYS:
            row[k] = _wmean(k)
        for k in _BXT_CLOSE_KEYS:
            row[k] = _last(k)
        for k in _BXT_MIN_KEYS:
            row[k] = _mn(k)
        for k in _BXT_MAX_KEYS:
            row[k] = _mx(k)
        for (o, h, l, c, m) in _BXT_OHLC_GROUPS:
            row[o] = _first(o)
            row[h] = _mx(h)
            row[l] = _mn(l)
            row[c] = _last(c)
            row[m] = _wmean(m)

        row["poll_gap_max_ms"] = _mx("poll_gap_max_ms")
        row["has_trades"] = any(bool(x.get("has_trades")) for x in subs)
        row["was_revised"] = any(bool(x.get("was_revised")) for x in subs)
        row["prev_contiguous"] = all(bool(x.get("prev_contiguous")) for x in subs)
        s0.append(row)
    return s0, keys


def signal_buckets(
    conn: Connection,
    schema: str,
    since: str,
    until: str,
    bucket_s: int = 5,
) -> dict[str, Any]:
    """Read the pre-computed 150-component DERIV Signal Buckets from the stored
    ``<schema>."5s_bxt"`` hypertable (built live by ``bxtbuilder_5s.py``).

    * ``bucket_s`` == 5 (default): stored 5s buckets are the ground truth and are
      returned verbatim — no recomputation.
    * ``bucket_s`` a multiple of 5 (e.g. 15/30/60): the 5s buckets are stacked
      into coarser buckets. State 0 is re-aggregated (sum/poll-weighted-mean/
      OHLC/last/min/max) and State 1/2 derivatives are RECOMPUTED on the
      coarsened series — you cannot sum 5s first-differences (they telescope),
      so derivatives must be taken across the coarse buckets directly.
      Derivatives are gated on coarse-bucket contiguity so gaps don't produce
      phantom jumps.

    Every row carries four quality flags plus the derived ``bucket_ideal``. Filter
    on ``bucket_ideal`` unless you have a reason not to: it excludes the June 15-24
    2026 feed-throttle era (poll_count fell from ~9.2 to 0.93), non-contiguous
    buckets, unevenly sampled ones, and buckets whose stored values differ from
    what a live system would have seen. Note that excluding revised buckets
    slightly biases the sample toward calm regimes — revisions cluster in
    volatile moments — so for research prefer treating ``was_revised`` as a
    control rather than a filter.

    ``ts`` is the bucket START; see the ``ts_convention`` / ``first_available_s``
    fields in the summary before building any forward-looking label.
    """
    bucket_s = int(bucket_s or 5)
    stack_n = max(1, bucket_s // 5)
    eff_bucket_s = 5 * stack_n

    read_cols = _BXT_COMPONENTS + _BXT_QUALITY_KEYS
    cols_sql = ", ".join(f'"{c}"' for c in read_cols)
    rows = conn.execute(
        text(
            f'''
            SELECT ts, {cols_sql}
            FROM {schema}."5s_bxt"
            WHERE ts >= :s AND ts < :u
            ORDER BY ts
            '''
        ),
        {"s": since, "u": until},
    ).fetchall()

    if not rows:
        return {
            "kind": "signal_buckets", "asset": schema, "window": _w(since, until),
            "bucket_s": eff_bucket_s, "source": "stored 5s_bxt",
            "summary": {"buckets": 0, "stored_5s_rows": 0}, "series": [],
        }

    # Decode stored rows into dicts keyed by component name.
    base: list[dict[str, Any]] = []
    for r in rows:
        ts = r[0]
        d: dict[str, Any] = {"_ts": ts, "_epoch": ts.timestamp()}
        for i, c in enumerate(read_cols, start=1):
            d[c] = _f(r[i]) if isinstance(r[i], float) else r[i]
        base.append(d)

    series: list[dict[str, Any]] = []

    if stack_n == 1:
        # ── pass-through: stored 5s buckets verbatim ──────────────────────
        for idx, d in enumerate(base):
            row: dict[str, Any] = {"ts": d["_ts"].isoformat(), "bucket_idx": idx}
            for c in read_cols:
                row[c] = d[c]
            row["bucket_ideal"] = _bxt_bucket_ideal(d)
            series.append(row)
    else:
        # ── stack to coarser horizon, recompute derivatives ───────────────
        s0, coarse_keys = _stack_state0(base, eff_bucket_s)
        s1: list[dict[str, Any]] = [{}]
        for i in range(1, len(s0)):
            contiguous = (coarse_keys[i] - coarse_keys[i - 1]) == eff_bucket_s
            s1.append(_bxt_state1(s0[i], s0[i - 1]) if contiguous else {})
        s2: list[dict[str, Any]] = [{}, {}]
        for i in range(2, len(s0)):
            s2.append(_bxt_state2(s1[i], s1[i - 1]) if (s1[i] and s1[i - 1]) else {})

        for i, row0 in enumerate(s0):
            row = {
                "ts": datetime.fromtimestamp(row0["_bucket_ts"], tz=timezone.utc).isoformat(),
                "bucket_idx": i,
                "n_sub": row0["_n_sub"],
            }
            for k in _BXT_S0_KEYS:
                row[k] = row0.get(k)
            for k in _BXT_QUALITY_KEYS:
                row[k] = row0.get(k)
            if i >= 1:
                for k in _BXT_S1_KEYS:
                    row[k] = s1[i].get(k)
            if i >= 2:
                for k in _BXT_S2_KEYS:
                    row[k] = s2[i].get(k)
            # A coarse bucket is only OK if its sub-buckets are AND it is itself
            # contiguous with the previous coarse bucket.
            coarse_contig = i >= 1 and (coarse_keys[i] - coarse_keys[i - 1]) == eff_bucket_s
            ok = _bxt_bucket_ideal({**row0, "poll_count": (row0.get("poll_count") or 0) / stack_n})
            row["bucket_ideal"] = bool(ok) and coarse_contig
            series.append(row)

    # ── summary ───────────────────────────────────────────────────────────
    mid_means = [s["mid_mean"] for s in series if s.get("mid_mean") is not None]
    replenish_vals = [s.get("ask_replenish") for s in series if s.get("ask_replenish") is not None]
    bid_replenish_vals = [s.get("bid_replenish") for s in series if s.get("bid_replenish") is not None]
    poll_total = sum((s.get("poll_count") or 0) for s in series)

    ideal_rows = [s for s in series if s.get("bucket_ideal")]
    summary = {
        "buckets": len(series),
        "bucket_s": eff_bucket_s,
        "stacked_from_5s": stack_n,
        "stored_5s_rows": len(base),
        "polls": poll_total,
        "source": "stored 5s_bxt",
        "state0_components": len(_BXT_S0_KEYS),
        "state1_components": len(_BXT_S1_KEYS),
        "state2_components": len(_BXT_S2_KEYS),
        "total_components": len(_BXT_COMPONENTS),
        # ts is the bucket START — a naive join on ts leaks the bucket's own
        # future. Anything forward-looking must be measured from ts + 8s.
        "ts_convention": BXT_TS_CONVENTION,
        "bucket_end_s": eff_bucket_s,
        "first_available_s": eff_bucket_s + (BXT_FIRST_AVAILABLE_S - BXT_BUCKET_END_S),
        "final_s": BXT_FINAL_S,
        "buckets_ideal": len(ideal_rows),
        "buckets_ideal_pct": (100.0 * len(ideal_rows) / len(series)) if series else None,
        "buckets_no_trades": sum(1 for s in series if s.get("has_trades") is False),
        "buckets_revised": sum(1 for s in series if s.get("was_revised")),
        "buckets_low_polls": sum(1 for s in series
                                 if (s.get("poll_count") or 0) < _BXT_MIN_POLLS * stack_n),
        "mid_first": mid_means[0] if mid_means else None,
        "mid_last": mid_means[-1] if mid_means else None,
        "mid_range_bps": (
            10000.0 * (max(mid_means) - min(mid_means)) / min(mid_means)
            if len(mid_means) >= 2 and min(mid_means) > 0 else None
        ),
        "ask_replenish_mean": (sum(replenish_vals) / len(replenish_vals)) if replenish_vals else None,
        "bid_replenish_mean": (sum(bid_replenish_vals) / len(bid_replenish_vals)) if bid_replenish_vals else None,
    }

    return {
        "kind": "signal_buckets",
        "asset": schema,
        "window": _w(since, until),
        "bucket_s": eff_bucket_s,
        "summary": summary,
        "series": series,
    }


# ── dispatch ──────────────────────────────────────────────────────────────


def _dispatch(item: str):
    """Resolve a catalog item key (without the asset prefix) to a builder."""
    table = {
        # raw
        "raw_l2":      raw_l2_window,
        "raw_l2_deep": raw_l2_deep_window,
        "raw_marks":   raw_marks_window,
        "raw_trades":  raw_trades_window,
        # bars — mark-price OHLC (mark_px), OHLC-only; volume lives in flow/buckets
        "ohlcv_1s":   lambda c, s, si, u: ohlc_marks(c, s, si, u, interval_s=1),
        "ohlcv_5s":   lambda c, s, si, u: ohlc_marks(c, s, si, u, interval_s=5),
        "ohlcv_1m":   lambda c, s, si, u: ohlc_marks(c, s, si, u, interval_s=60),
        "ohlcv_5m":   lambda c, s, si, u: ohlc_marks(c, s, si, u, interval_s=300),
        # microstructure
        "l2_imbalance":      l2_imbalance,
        "microprice":        microprice,
        "top_of_book_depth": top_of_book_depth,
        "effective_spread":  effective_spread,
        "quote_churn_rate":  quote_churn_rate,
        # price dynamics
        "mid_velocity":      lambda c, s, si, u: mid_derivatives(c, s, si, u, order=1),
        "mid_acceleration":  lambda c, s, si, u: mid_derivatives(c, s, si, u, order=2),
        "mid_jerk":          lambda c, s, si, u: mid_derivatives(c, s, si, u, order=3),
        "realized_vol":      realized_vol,
        "range_expansion":   range_expansion,
        # flow
        "taker_imbalance":   taker_imbalance,
        "trade_rate":        trade_rate,
        "volume_profile":    volume_profile,
        # derivatives
        "funding_trajectory": funding_trajectory,
        "oi_change_rate":     oi_change_rate,
        "premium_decay":      premium_decay,
        "oracle_drift":       oracle_drift,
    }
    return table.get(item)


def materialize(
    conn: Connection, asset: str, item: str, since: str, until: str,
    *, bucket_s: int | None = None,
) -> dict[str, Any]:
    """Public entry point used by the strathub router.

    Parameters
    ----------
    bucket_s : override bucket size for signal_buckets item (seconds).
    """
    # Signal buckets is parameterized by bucket_s
    if item == "signal_buckets":
        return signal_buckets(conn, asset, since, until, bucket_s=bucket_s or 5)

    fn = _dispatch(item)
    if fn is None:
        return {"kind": item, "asset": asset, "error": f"unknown item '{item}'",
                "window": _w(since, until), "series": [], "summary": {}}
    return fn(conn, asset, since, until)
