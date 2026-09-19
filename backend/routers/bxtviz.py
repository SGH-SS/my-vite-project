"""
BXT Visualizer API
==================

Routes on ``/api/bxtviz``:

* ``GET /candles`` — mid-price OHLC candles for one (schema, candle size,
  window), stacked from the stored 5s bxt buckets.

Design rules (research/README.md — do not re-litigate casually):

* **StratHub's ``feature_builder.signal_buckets`` is the single stacking
  implementation.** This router never re-aggregates; it only extracts the
  candle fields from stacked rows. Mid OHLC therefore rolls up
  first-open / max-high / min-low / last-close, exactly like the panel.
* **``ts`` is the bucket START**, half-open ``[ts, ts+tf)``. Candles are
  labelled by bucket start (standard charting convention); ``bucket_end``
  is derivable as ``t + tf_s``.
* **Windows snap to complete grid slots** (ceil start, floor end — the
  T12 fractional-boundary rule) and **gaps are explicit**: the response
  is reindexed to the full grid, missing buckets ship as ``present: false``
  rows so the chart can draw whitespace instead of silently bridging.
* **Quality is never silently painted over**: each candle carries
  ``ideal`` (bucket_ideal, coarse-contiguity included when stacked) and
  ``revised`` (was_revised OR'd over sub-buckets) so the frontend can
  grey/flag degraded candles.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query

from sqlalchemy import text

from database import engine
from services import feature_builder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/bxtviz", tags=["bxtviz"])

# schemas with a 5s_bxt table (mirrors research/config.py BXT_SCHEMAS)
BXT_SCHEMAS = ("btc", "sol", "eth", "spx")

# candle sizes offered by the UI — any multiple of 5 works for the stacker,
# this list is just the sanctioned menu.
CANDLE_SIZES_S = (5, 15, 30, 60, 300, 900, 3600, 14400, 86400)

# request guards: a fetch reads all 150 stored components per 5s row, so cap
# both the output candles and the INPUT rows — a "2 weeks @ 1d" request would
# otherwise pull ~240k wide rows into memory to build 14 candles.
MAX_CANDLES = 20_000
MAX_STORED_ROWS = 70_000      # ≈ 4 days of 5s rows


def _parse_iso(s: str, name: str) -> datetime:
    # an unencoded '+' in a query string arrives as a space — repair it
    s = re.sub(r" (?=\d{2}:\d{2}$)", "+", s.strip())
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(400, f"{name} is not ISO-8601: {s!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _snap(dt: datetime, tf_s: int, up: bool) -> datetime:
    """Snap to the tf grid — ceil start / floor end (research T12 rule)."""
    epoch = dt.timestamp()
    k = math.ceil(epoch / tf_s) if up else math.floor(epoch / tf_s)
    return datetime.fromtimestamp(k * tf_s, tz=timezone.utc)


@router.get("/candles")
async def candles(
    schema: str = Query("btc"),
    tf_s: int = Query(60, description="candle size in seconds"),
    start: str = Query(..., description="ISO-8601 window start"),
    end: str = Query(..., description="ISO-8601 window end"),
    l20: int = Query(0, description="1 = include 20-level deep-book sums per candle"),
) -> dict:
    if schema not in BXT_SCHEMAS:
        raise HTTPException(400, f"'{schema}' has no 5s_bxt table — available: {list(BXT_SCHEMAS)}")
    if tf_s not in CANDLE_SIZES_S:
        raise HTTPException(400, f"tf_s={tf_s} not supported — available: {list(CANDLE_SIZES_S)}")

    now = datetime.now(timezone.utc)
    start_dt = _parse_iso(start, "start")
    end_dt = min(_parse_iso(end, "end"), now)

    grid_start = _snap(start_dt, tf_s, up=True)
    grid_end = _snap(end_dt, tf_s, up=False)
    if grid_end <= grid_start:
        raise HTTPException(400, "window smaller than one grid slot after snapping")

    expected = int((grid_end - grid_start).total_seconds()) // tf_s
    if expected > MAX_CANDLES:
        raise HTTPException(
            400,
            f"{expected:,} candles requested (max {MAX_CANDLES:,}) — "
            f"pick a coarser candle size or a shorter window",
        )
    stored_rows = int((grid_end - grid_start).total_seconds()) // 5
    if stored_rows > MAX_STORED_ROWS:
        raise HTTPException(
            400,
            f"window spans {stored_rows:,} stored 5s rows (max {MAX_STORED_ROWS:,}, "
            f"≈ {MAX_STORED_ROWS * 5 / 86400:.0f} days) — shorten the window",
        )

    # ── the one true stacker, with a 1-tf LEFT WARMUP ─────────────────────
    # bucket_ideal requires contiguity with the PREVIOUS coarse bucket, which
    # the first bucket of a fetch can never prove — so fetch one extra tf on
    # the left (research-panel warmup pattern) and trim it after stacking.
    warmup_start = grid_start - timedelta(seconds=tf_s)
    with engine.connect() as conn:
        out = feature_builder.signal_buckets(
            conn, schema, warmup_start.isoformat(), grid_end.isoformat(), bucket_s=tf_s,
        )

    grid_start_epoch = int(grid_start.timestamp())
    by_epoch: dict[int, dict] = {}
    for row in out["series"]:
        ts = datetime.fromisoformat(row["ts"])
        e = int(ts.timestamp())
        if e >= grid_start_epoch:            # trim warmup rows
            by_epoch[e] = row

    # ── full-grid reindex: gaps become explicit rows ──────────────────────
    candles_out: list[dict] = []
    actual = 0
    for e in range(int(grid_start.timestamp()), int(grid_end.timestamp()), tf_s):
        row = by_epoch.get(e)
        if row is None or row.get("mid_c") is None:
            candles_out.append({"t": e, "present": False})
            continue
        actual += 1
        buy_v = row.get("buy_volume") or 0.0
        sell_v = row.get("sell_volume") or 0.0
        candles_out.append({
            "t": e,
            "present": True,
            "o": row.get("mid_o"),
            "h": row.get("mid_h"),
            "l": row.get("mid_l"),
            "c": row.get("mid_c"),
            "v": buy_v + sell_v,
            "buy_v": buy_v,
            "sell_v": sell_v,
            "ideal": bool(row.get("bucket_ideal")),
            "revised": bool(row.get("was_revised")),
            "n_sub": row.get("n_sub", 1),
            # wall sizes in COIN units (never _usd) for the above/below overlay.
            # Stacking semantics come from signal_buckets, not from here:
            #   ask_size/bid_size   — poll-weighted MEAN over the bucket
            #   ask_size_c/bid_size_c — LAST sub-bucket's close (book at bar close)
            "ask_size": row.get("ask_size"),
            "bid_size": row.get("bid_size"),
            "ask_size_c": row.get("ask_size_c"),
            "bid_size_c": row.get("bid_size_c"),
        })

    # ── avg-of-5s-closes wall sizes ───────────────────────────────────────
    # A VISUALIZATION-LAYER aggregate, not a canonical bucket component: the
    # stacker's only rules for _c are LAST (close) — the mean of the internal
    # 5s closing snapshots doesn't exist in the component set. It is computed
    # here from the ground-truth 5s rows, grouped on the exact same epoch grid
    # the stacker uses (floor(epoch/tf)*tf), as an UNWEIGHTED mean: each 5s
    # close is one instant, so poll-count weighting doesn't apply. NULL closes
    # (degraded sub-buckets) simply drop out of the mean.
    if tf_s > 5:
        with engine.connect() as conn:
            rows = conn.execute(text(f'''
                SELECT (floor(extract(epoch FROM ts) / :tf) * :tf)::bigint AS gk,
                       AVG(ask_size_c) AS a, AVG(bid_size_c) AS b
                FROM {schema}."5s_bxt"
                WHERE ts >= :s AND ts < :u
                GROUP BY gk
            '''), {"tf": tf_s, "s": grid_start.isoformat(),
                   "u": grid_end.isoformat()}).fetchall()
        avgc = {int(r[0]): (float(r[1]) if r[1] is not None else None,
                            float(r[2]) if r[2] is not None else None) for r in rows}
        for c in candles_out:
            if c["present"]:
                a, b = avgc.get(c["t"], (None, None))
                c["ask_size_c_avg"] = a
                c["bid_size_c_avg"] = b
    else:
        # at 5s there is exactly one sub-bucket — avg of closes IS the close
        for c in candles_out:
            if c["present"]:
                c["ask_size_c_avg"] = c["ask_size_c"]
                c["bid_size_c_avg"] = c["bid_size_c"]

    # ── optional 20-level deep-book sums (L20 alignment toggle) ───────────
    # l2_deep is a 1 Hz REST poll on its OWN clock (no bucket contract), so
    # this is a visualization-layer aggregate like avg-of-closes: per candle,
    # the UNWEIGHTED mean over the snapshots inside [t, t+tf) of the summed
    # 20-level size per side, in coins. ~5 snapshots per 5s candle, ~60 per
    # 1m — approximate at 5s, honest at 1m+ (same caveat as the research
    # notes on l2_deep). Missing coverage simply yields no value.
    if l20:
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(f'''
                    SELECT (floor(extract(epoch FROM ts) / :tf) * :tf)::bigint AS gk,
                           AVG(bid_sum) AS b, AVG(ask_sum) AS a, COUNT(*) AS n
                    FROM (
                        SELECT ts,
                               (SELECT SUM((e->>'sz')::float8)
                                  FROM jsonb_array_elements(bids) e) AS bid_sum,
                               (SELECT SUM((e->>'sz')::float8)
                                  FROM jsonb_array_elements(asks) e) AS ask_sum
                        FROM {schema}."l2_deep"
                        WHERE ts >= :s AND ts < :u
                    ) x
                    GROUP BY gk
                '''), {"tf": tf_s, "s": grid_start.isoformat(),
                       "u": grid_end.isoformat()}).fetchall()
            deep = {int(r[0]): (float(r[1]) if r[1] is not None else None,
                                float(r[2]) if r[2] is not None else None,
                                int(r[3])) for r in rows}
            for c in candles_out:
                if c["present"]:
                    b, a, n = deep.get(c["t"], (None, None, 0))
                    c["l20_bid"] = b
                    c["l20_ask"] = a
                    c["l20_polls"] = n
        except Exception:
            logger.exception("l2_deep aggregation failed for %s", schema)

    summary = out.get("summary", {})
    return {
        "schema": schema,
        "tf_s": tf_s,
        "stack_n": max(1, tf_s // 5),
        "grid_start": grid_start.isoformat(),
        "grid_end": grid_end.isoformat(),
        "ts_convention": "bucket_start, half-open [t, t+tf_s)",
        "expected": expected,
        "actual": actual,
        "missing": expected - actual,
        "ideal": sum(1 for c in candles_out if c.get("ideal")),
        "revised": sum(1 for c in candles_out if c.get("revised")),
        "stored_5s_rows": summary.get("stored_5s_rows", 0),
        "candles": candles_out,
    }


@router.get("/meta")
async def meta() -> dict:
    """Static config for the frontend dropdowns."""
    return {
        "schemas": list(BXT_SCHEMAS),
        "candle_sizes_s": list(CANDLE_SIZES_S),
        "max_candles": MAX_CANDLES,
    }
