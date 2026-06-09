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


def raw_l2_window(
    conn: Connection, schema: str, since: str, until: str,
) -> dict[str, Any]:
    """Every L2 snapshot in the window + summary mid/spread stats."""
    r = conn.execute(
        text(
            f"""
            SELECT ts, best_bid, best_ask, mid_price, spread,
                   n_bid_levels, n_ask_levels
            FROM {schema}.l2_snapshots
            WHERE ts >= :since AND ts < :until
            ORDER BY ts
            """
        ),
        {"since": since, "until": until},
    )
    rows = r.fetchall()
    if not rows:
        return {"kind": "raw_l2_window", "asset": schema, "window": _w(since, until),
                "summary": {"snapshots": 0}, "series": []}

    total = len(rows)
    series = []
    mids: list[float] = []
    spreads: list[float] = []
    for row in rows:
        ts, bb, ba, mid, sp, nbl, nal = row
        mid_f = _f(mid)
        sp_f = _f(sp)
        if mid_f is not None:
            mids.append(mid_f)
        if sp_f is not None:
            spreads.append(sp_f)
        series.append({
            "ts": ts.isoformat(),
            "best_bid": _f(bb), "best_ask": _f(ba),
            "mid": mid_f, "spread": sp_f,
            "bid_levels": nbl, "ask_levels": nal,
        })

    return {
        "kind": "raw_l2_window",
        "asset": schema,
        "window": _w(since, until),
        "summary": {
            "snapshots": total,
            "mid_first": mids[0] if mids else None,
            "mid_last":  mids[-1] if mids else None,
            "mid_min":   min(mids) if mids else None,
            "mid_max":   max(mids) if mids else None,
            "mid_change_bps": (
                10000.0 * (mids[-1] - mids[0]) / mids[0]
                if len(mids) >= 2 and mids[0] else None
            ),
            "spread_avg": (sum(spreads) / len(spreads)) if spreads else None,
            "spread_max": max(spreads) if spreads else None,
        },
        "series": series,
    }


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


# Default tick sizes per asset for fill ratio calculation.
_TICK_SIZE: dict[str, float] = {
    "sol": 0.001,
    "btc": 0.1,
    "eth": 0.01,
    "spx": 0.01,
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

    total_size = 0.0
    total_ppl = 0
    weighted_distance_sum = 0.0
    occupied_ticks = set()
    far_px = None

    for entry in levels:
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

        total_size += sz
        total_ppl += n_val

        dist = abs(px - best_px)
        weighted_distance_sum += sz * dist

        tick_idx = round(dist / tick_size) if tick_size > 0 else 0
        occupied_ticks.add(tick_idx)

        if far_px is None:
            far_px = px
        else:
            if side == "ask":
                far_px = max(far_px, px)
            else:
                far_px = min(far_px, px)

    if total_size <= 0:
        return {"size": None, "ppl": None, "fill": None, "centroid": None, "size_usd": None}

    total_distance = abs(far_px - best_px) if far_px is not None else 0.0
    max_ticks = round(total_distance / tick_size) + 1 if (tick_size > 0 and total_distance > 0) else 1
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


def signal_buckets(
    conn: Connection,
    schema: str,
    since: str,
    until: str,
    bucket_s: int = 5,
) -> dict[str, Any]:
    """Compute the 22-vector / 102-component DERIV Signal Buckets framework.

    Groups L2 polls and trades into time buckets, computes State 0 (raw values),
    State 1 (first derivatives + replenishment cross-signals), and State 2
    (second derivatives + replenishment velocity).

    Returns both raw and USD-normalized tracks.
    """
    tick_size = _TICK_SIZE.get(schema, 0.001)

    # Set a 120s statement timeout to prevent unbounded blocking
    conn.execute(text("SET LOCAL statement_timeout = '120s'"))

    # ── fetch L2 polls ────────────────────────────────────────────────────
    l2_rows = conn.execute(
        text(
            f"""
            SELECT ts, bids, asks, best_bid, best_ask, mid_price, spread
            FROM {schema}.l2_snapshots
            WHERE ts >= :s AND ts < :u
            ORDER BY ts
            """
        ),
        {"s": since, "u": until},
    ).fetchall()

    # ── fetch trades ──────────────────────────────────────────────────────
    trade_rows = conn.execute(
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

    if not l2_rows and not trade_rows:
        return {
            "kind": "signal_buckets",
            "asset": schema,
            "window": _w(since, until),
            "bucket_s": bucket_s,
            "summary": {"buckets": 0, "polls": 0, "trades": 0},
            "series": [],
        }

    # ── assign polls to buckets ───────────────────────────────────────────
    from collections import defaultdict

    poll_buckets: dict[int, list] = defaultdict(list)
    for row in l2_rows:
        ts, bids_raw, asks_raw, bb, ba, mid, spread = row
        epoch = ts.timestamp()
        bk = int(epoch // bucket_s) * bucket_s
        bids = _parse_jsonb_levels(bids_raw)
        asks = _parse_jsonb_levels(asks_raw)
        best_bid = _f(bb)
        best_ask = _f(ba)
        mid_f = _f(mid)
        spread_f = _f(spread)

        ask_m = _poll_metrics(asks, best_ask, "ask", tick_size) if best_ask else {
            "size": None, "ppl": None, "fill": None, "centroid": None, "size_usd": None}
        bid_m = _poll_metrics(bids, best_bid, "bid", tick_size) if best_bid else {
            "size": None, "ppl": None, "fill": None, "centroid": None, "size_usd": None}

        spread_bps = (10000.0 * spread_f / mid_f) if (spread_f is not None and mid_f and mid_f > 0) else None

        poll_buckets[bk].append({
            "ask": ask_m,
            "bid": bid_m,
            "mid": mid_f,
            "spread": spread_f,
            "spread_bps": spread_bps,
        })

    # ── assign trades to buckets ──────────────────────────────────────────
    trade_buckets: dict[int, list] = defaultdict(list)
    for row in trade_rows:
        ts, side, px, sz = row
        epoch = ts.timestamp()
        bk = int(epoch // bucket_s) * bucket_s
        px_f = _f(px)
        sz_f = _f(sz)
        trade_buckets[bk].append({"side": side, "px": px_f, "sz": sz_f})

    # ── determine bucket timeline ─────────────────────────────────────────
    all_bucket_keys = sorted(set(list(poll_buckets.keys()) + list(trade_buckets.keys())))
    if not all_bucket_keys:
        return {
            "kind": "signal_buckets",
            "asset": schema,
            "window": _w(since, until),
            "bucket_s": bucket_s,
            "summary": {"buckets": 0, "polls": len(l2_rows), "trades": len(trade_rows)},
            "series": [],
        }

    # ── compute State 0 per bucket ────────────────────────────────────────
    state0_list: list[dict[str, Any]] = []

    for bk in all_bucket_keys:
        polls = poll_buckets.get(bk, [])
        trades = trade_buckets.get(bk, [])
        poll_count = len(polls)

        # Ask book aggregation (mean across polls)
        ask_sizes = [p["ask"]["size"] for p in polls if p["ask"]["size"] is not None]
        ask_ppls = [p["ask"]["ppl"] for p in polls if p["ask"]["ppl"] is not None]
        ask_fills = [p["ask"]["fill"] for p in polls if p["ask"]["fill"] is not None]
        ask_centroids = [p["ask"]["centroid"] for p in polls if p["ask"]["centroid"] is not None]
        ask_sizes_usd = [p["ask"]["size_usd"] for p in polls if p["ask"]["size_usd"] is not None]

        # Bid book aggregation
        bid_sizes = [p["bid"]["size"] for p in polls if p["bid"]["size"] is not None]
        bid_ppls = [p["bid"]["ppl"] for p in polls if p["bid"]["ppl"] is not None]
        bid_fills = [p["bid"]["fill"] for p in polls if p["bid"]["fill"] is not None]
        bid_centroids = [p["bid"]["centroid"] for p in polls if p["bid"]["centroid"] is not None]
        bid_sizes_usd = [p["bid"]["size_usd"] for p in polls if p["bid"]["size_usd"] is not None]

        # Mid price OHLCM
        mids = [p["mid"] for p in polls if p["mid"] is not None]
        # Spread OHLCM
        spreads = [p["spread"] for p in polls if p["spread"] is not None]
        spread_bps_vals = [p["spread_bps"] for p in polls if p["spread_bps"] is not None]

        # Trade aggregation
        buys = [t for t in trades if t["side"] == "B" and t["sz"] is not None]
        sells = [t for t in trades if t["side"] == "A" and t["sz"] is not None]

        buy_volume = sum(t["sz"] for t in buys) if buys else 0.0
        buy_count = len(buys)
        sell_volume = sum(t["sz"] for t in sells) if sells else 0.0
        sell_count = len(sells)

        buy_volume_usd = sum(t["sz"] * t["px"] for t in buys if t["px"]) if buys else 0.0
        sell_volume_usd = sum(t["sz"] * t["px"] for t in sells if t["px"]) if sells else 0.0

        def _mean(xs):
            return (sum(xs) / len(xs)) if xs else None

        s0 = {
            "bucket_ts": bk,
            "poll_count": poll_count,
            # X1 - Ask Book
            "ask_size": _mean(ask_sizes),
            "ask_size_usd": _mean(ask_sizes_usd),
            "ask_ppl": _mean(ask_ppls),
            "ask_fill": _mean(ask_fills),
            "ask_centroid": _mean(ask_centroids),
            # X2 - Buyers
            "buy_volume": buy_volume,
            "buy_volume_usd": buy_volume_usd,
            "buy_count": buy_count,
            # X3 - Sellers
            "sell_volume": sell_volume,
            "sell_volume_usd": sell_volume_usd,
            "sell_count": sell_count,
            # X4 - Bid Book
            "bid_size": _mean(bid_sizes),
            "bid_size_usd": _mean(bid_sizes_usd),
            "bid_ppl": _mean(bid_ppls),
            "bid_fill": _mean(bid_fills),
            "bid_centroid": _mean(bid_centroids),
            # X6 - Mid Price OHLCM
            "mid_o": mids[0] if mids else None,
            "mid_h": max(mids) if mids else None,
            "mid_l": min(mids) if mids else None,
            "mid_c": mids[-1] if mids else None,
            "mid_mean": _mean(mids),
            # X9 - Spread OHLCM + BPS
            "spread_o": spreads[0] if spreads else None,
            "spread_h": max(spreads) if spreads else None,
            "spread_l": min(spreads) if spreads else None,
            "spread_c": spreads[-1] if spreads else None,
            "spread_mean": _mean(spreads),
            "spread_o_bps": spread_bps_vals[0] if spread_bps_vals else None,
            "spread_h_bps": max(spread_bps_vals) if spread_bps_vals else None,
            "spread_l_bps": min(spread_bps_vals) if spread_bps_vals else None,
            "spread_c_bps": spread_bps_vals[-1] if spread_bps_vals else None,
            "spread_mean_bps": _mean(spread_bps_vals),
        }
        state0_list.append(s0)

    # ── compute State 1 (first derivatives + cross-signals) ───────────────
    # Keys that get diffed for State 1
    _S0_DIFF_KEYS = [
        "ask_size", "ask_size_usd", "ask_ppl", "ask_fill", "ask_centroid",
        "buy_volume", "buy_volume_usd", "buy_count",
        "sell_volume", "sell_volume_usd", "sell_count",
        "bid_size", "bid_size_usd", "bid_ppl", "bid_fill", "bid_centroid",
        "mid_o", "mid_h", "mid_l", "mid_c", "mid_mean",
        "spread_o", "spread_h", "spread_l", "spread_c", "spread_mean",
        "spread_o_bps", "spread_h_bps", "spread_l_bps", "spread_c_bps", "spread_mean_bps",
    ]

    state1_list: list[dict[str, Any]] = [{}]  # first bucket has no derivatives
    for i in range(1, len(state0_list)):
        d = {}
        for k in _S0_DIFF_KEYS:
            cur_v = state0_list[i].get(k)
            prev_v = state0_list[i - 1].get(k)
            if cur_v is not None and prev_v is not None:
                d[f"d_{k}"] = cur_v - prev_v
            else:
                d[f"d_{k}"] = None
        # X5 - Ask Replenishment (cross-signal)
        d_ask_size = d.get("d_ask_size")
        bv = state0_list[i].get("buy_volume")
        d["ask_replenish"] = (d_ask_size + bv) if (d_ask_size is not None and bv is not None) else None

        d_ask_size_usd = d.get("d_ask_size_usd")
        bv_usd = state0_list[i].get("buy_volume_usd")
        d["ask_replenish_usd"] = (d_ask_size_usd + bv_usd) if (d_ask_size_usd is not None and bv_usd is not None) else None

        # X7 - Bid Replenishment (cross-signal)
        d_bid_size = d.get("d_bid_size")
        sv = state0_list[i].get("sell_volume")
        d["bid_replenish"] = (d_bid_size + sv) if (d_bid_size is not None and sv is not None) else None

        d_bid_size_usd = d.get("d_bid_size_usd")
        sv_usd = state0_list[i].get("sell_volume_usd")
        d["bid_replenish_usd"] = (d_bid_size_usd + sv_usd) if (d_bid_size_usd is not None and sv_usd is not None) else None

        state1_list.append(d)

    # ── compute State 2 (second derivatives + replenishment velocity) ─────
    _S1_DIFF_KEYS = [f"d_{k}" for k in _S0_DIFF_KEYS]

    state2_list: list[dict[str, Any]] = [{}, {}]  # first two buckets have no 2nd derivatives
    for i in range(2, len(state0_list)):
        dd = {}
        for k in _S1_DIFF_KEYS:
            cur_v = state1_list[i].get(k)
            prev_v = state1_list[i - 1].get(k)
            if cur_v is not None and prev_v is not None:
                dd[f"d{k}"] = cur_v - prev_v  # dd_ask_size, dd_buy_volume, etc.
            else:
                dd[f"d{k}"] = None

        # d/dx5 - Ask Replenishment Velocity
        dd_ask_size = dd.get("dd_ask_size")
        d_bv = state1_list[i].get("d_buy_volume")
        dd["d_ask_replenish"] = (dd_ask_size + d_bv) if (dd_ask_size is not None and d_bv is not None) else None

        dd_ask_size_usd = dd.get("dd_ask_size_usd")
        d_bv_usd = state1_list[i].get("d_buy_volume_usd")
        dd["d_ask_replenish_usd"] = (dd_ask_size_usd + d_bv_usd) if (dd_ask_size_usd is not None and d_bv_usd is not None) else None

        # d/dx7 - Bid Replenishment Velocity
        dd_bid_size = dd.get("dd_bid_size")
        d_sv = state1_list[i].get("d_sell_volume")
        dd["d_bid_replenish"] = (dd_bid_size + d_sv) if (dd_bid_size is not None and d_sv is not None) else None

        dd_bid_size_usd = dd.get("dd_bid_size_usd")
        d_sv_usd = state1_list[i].get("d_sell_volume_usd")
        dd["d_bid_replenish_usd"] = (dd_bid_size_usd + d_sv_usd) if (dd_bid_size_usd is not None and d_sv_usd is not None) else None

        state2_list.append(dd)

    # ── merge into output series ──────────────────────────────────────────
    from datetime import datetime as _dt, timezone as _tz

    series: list[dict[str, Any]] = []
    for i, s0 in enumerate(state0_list):
        row: dict[str, Any] = {
            "ts": _dt.fromtimestamp(s0["bucket_ts"], tz=_tz.utc).isoformat(),
            "bucket_idx": i,
        }
        # State 0
        for k, v in s0.items():
            if k == "bucket_ts":
                continue
            row[k] = _f(v) if isinstance(v, float) else v

        # State 1 (available from bucket index 1)
        if i >= 1 and i < len(state1_list):
            s1 = state1_list[i]
            for k, v in s1.items():
                row[k] = _f(v) if isinstance(v, float) else v
        # State 2 (available from bucket index 2)
        if i >= 2 and i < len(state2_list):
            s2 = state2_list[i]
            for k, v in s2.items():
                row[k] = _f(v) if isinstance(v, float) else v

        series.append(row)

    # ── summary ───────────────────────────────────────────────────────────
    mid_means = [s["mid_mean"] for s in series if s.get("mid_mean") is not None]
    replenish_vals = [s.get("ask_replenish") for s in series if s.get("ask_replenish") is not None]
    bid_replenish_vals = [s.get("bid_replenish") for s in series if s.get("bid_replenish") is not None]

    summary = {
        "buckets": len(series),
        "bucket_s": bucket_s,
        "polls": len(l2_rows),
        "trades": len(trade_rows),
        "state0_components": 32,
        "state1_components": 35,
        "state2_components": 35,
        "total_components": 102,
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
        "bucket_s": bucket_s,
        "summary": summary,
        "series": series,
    }


# ── dispatch ──────────────────────────────────────────────────────────────


def _dispatch(item: str):
    """Resolve a catalog item key (without the asset prefix) to a builder."""
    table = {
        # raw
        "raw_l2":     raw_l2_window,
        "raw_marks":  raw_marks_window,
        "raw_trades": raw_trades_window,
        # bars
        "ohlcv_1s":   lambda c, s, si, u: ohlcv(c, s, si, u, interval_s=1),
        "ohlcv_5s":   lambda c, s, si, u: ohlcv(c, s, si, u, interval_s=5),
        "ohlcv_1m":   lambda c, s, si, u: ohlcv(c, s, si, u, interval_s=60),
        "ohlcv_5m":   lambda c, s, si, u: ohlcv(c, s, si, u, interval_s=300),
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
