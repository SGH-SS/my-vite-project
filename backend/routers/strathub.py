"""
Strat Hub API
=============

API routes on ``/api/strathub``:

* ``GET  /catalog``     — static tree (assets × groups × items) + supported
                          model + reasoning-effort options.
* ``POST /materialize`` — build the JSON payload for a single
                          (asset, item, since, until) using
                          ``services.feature_builder``.
* ``POST /preflight``   — materialize selected components over ``since/until``,
                          build the same prompts as ``/analyze`` (tabular data as CSV),
                          and measure input tokens via Anthropic ``/v1/messages/count_tokens``.
* ``POST /analyze``     — package one or more component payloads into a
                          compact CSV-shaped user prompt and proxy through
                          ``services.anthropic_client.call_claude``.

The frontend orchestrates all individual / pair / triple calls; this
router just serves single units of work so the UI can run them in
parallel.

For comparison modes ("pair" / "triple") the frontend should pass
``prior_individual`` (a list of pre-computed individual analyses) inside
the body — they get inlined into the user prompt so the comparison call
has the per-component analyses already in its context window.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from database import engine
from services import anthropic_client, openai_client, feature_builder, prompt_csv

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strathub", tags=["strathub"])


# ── catalog (static) ──────────────────────────────────────────────────────

ASSETS: list[dict[str, str]] = [
    {"key": "btc", "label": "BTC-PERP",  "color": "#f7b50f"},
    {"key": "eth", "label": "ETH-PERP",  "color": "#8b5cf6"},
    {"key": "sol", "label": "SOL-PERP",  "color": "#22d3ee"},
    {"key": "spx", "label": "SPX-PERP",  "color": "#f43f5e"},
]

GROUPS: list[dict[str, Any]] = [
    {
        "key": "raw", "label": "Raw Data",
        "items": [
            {"key": "raw_l2",     "label": "L2 snapshot window",        "stored": True},
            {"key": "raw_marks",  "label": "Mark / oracle / funding",   "stored": True},
            {"key": "raw_trades", "label": "Trade prints window",       "stored": True},
        ],
    },
    {
        "key": "signal_buckets", "label": "Signal Buckets",
        "items": [
            {"key": "signal_buckets", "label": "DERIV 22-vector framework (102 components)", "stored": False},
        ],
    },
    {
        "key": "bars", "label": "Aggregated Bars",
        "items": [
            {"key": "ohlcv_1s", "label": "OHLCV 1s",  "stored": False},
            {"key": "ohlcv_5s", "label": "OHLCV 5s",  "stored": False},
            {"key": "ohlcv_1m", "label": "OHLCV 1m",  "stored": False},
            {"key": "ohlcv_5m", "label": "OHLCV 5m",  "stored": False},
        ],
    },
    {
        "key": "micro", "label": "Microstructure",
        "items": [
            {"key": "l2_imbalance",      "label": "Full-book L2 imbalance",     "stored": False},
            {"key": "microprice",        "label": "Microprice + skew (bps)",  "stored": False},
            {"key": "top_of_book_depth", "label": "Top-of-book depth ($)",    "stored": False},
            {"key": "effective_spread",  "label": "Effective spread (bps)",   "stored": False},
            {"key": "quote_churn_rate",  "label": "Quote churn rate",         "stored": False},
        ],
    },
    {
        "key": "dyn", "label": "Price Dynamics",
        "items": [
            {"key": "mid_velocity",     "label": "Mid velocity (d/dt)",      "stored": False},
            {"key": "mid_acceleration", "label": "Mid acceleration (d²/dt²)", "stored": False},
            {"key": "mid_jerk",         "label": "Mid jerk (d³/dt³)",        "stored": False},
            {"key": "realized_vol",     "label": "Realized vol (rolling)",   "stored": False},
            {"key": "range_expansion",  "label": "Range expansion ratio",    "stored": False},
        ],
    },
    {
        "key": "flow", "label": "Flow",
        "items": [
            {"key": "taker_imbalance", "label": "Taker buy/sell imbalance", "stored": False},
            {"key": "trade_rate",      "label": "Trade rate / volume",      "stored": False},
            {"key": "volume_profile",  "label": "Price-bucket volume profile", "stored": False},
        ],
    },
    {
        "key": "deriv", "label": "Derivatives",
        "items": [
            {"key": "funding_trajectory", "label": "Funding rate trajectory", "stored": False},
            {"key": "oi_change_rate",     "label": "Open-interest change rate", "stored": False},
            {"key": "premium_decay",      "label": "Mark-vs-mid premium (bps)", "stored": False},
            {"key": "oracle_drift",       "label": "Oracle-vs-mark divergence", "stored": False},
        ],
    },
]


ALL_MODELS: list[dict[str, Any]] = anthropic_client.SUPPORTED_MODELS + openai_client.SUPPORTED_MODELS
_OPENAI_MODEL_IDS: frozenset[str] = frozenset(m["id"] for m in openai_client.SUPPORTED_MODELS)


def _is_openai_model(model_id: str) -> bool:
    return model_id in _OPENAI_MODEL_IDS


@router.get("/catalog")
async def get_catalog() -> dict[str, Any]:
    """Static tree definition consumed directly by the React tree."""
    return {
        "assets": ASSETS,
        "groups": GROUPS,
        "models": ALL_MODELS,
        "default_model": "gpt-5.5-instant",
        "default_effort": anthropic_client.DEFAULT_EFFORT,
        "efforts": anthropic_client.EFFORT_OPTIONS,
        "system_prompts": {
            "individual": SYS_INDIVIDUAL,
            "compare": SYS_COMPARE,
        },
    }


# ── materialize ───────────────────────────────────────────────────────────


class MaterializeRequest(BaseModel):
    asset: str
    item: str
    since: Optional[str] = None
    until: Optional[str] = None
    window_s: Optional[int] = 900  # used when since/until are missing
    bucket_s: Optional[int] = None  # internal bucket size for signal_buckets (seconds)


def _resolve_window(req: MaterializeRequest) -> tuple[str, str]:
    if req.since and req.until:
        return req.since, req.until
    until = datetime.now(timezone.utc)
    if req.window_s is None or req.window_s <= 0:
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)
    else:
        since = until - timedelta(seconds=req.window_s)
    return since.isoformat(), until.isoformat()


_VALID_ASSETS = {a["key"] for a in ASSETS}

# Map catalog item keys → their underlying DB table (for data-range queries).
_ITEM_TABLE: dict[str, str] = {
    "raw_l2": "l2_snapshots", "raw_marks": "mark_price", "raw_trades": "trades",
    "signal_buckets": "l2_snapshots",
    "ohlcv_1s": "trades", "ohlcv_5s": "trades", "ohlcv_1m": "trades", "ohlcv_5m": "trades",
    "l2_imbalance": "l2_snapshots", "microprice": "l2_snapshots",
    "top_of_book_depth": "l2_snapshots", "effective_spread": "l2_snapshots",
    "quote_churn_rate": "l2_snapshots",
    "mid_velocity": "l2_snapshots", "mid_acceleration": "l2_snapshots",
    "mid_jerk": "l2_snapshots", "realized_vol": "l2_snapshots",
    "range_expansion": "trades",
    "taker_imbalance": "trades", "trade_rate": "trades", "volume_profile": "trades",
    "funding_trajectory": "mark_price", "oi_change_rate": "mark_price",
    "premium_decay": "mark_price", "oracle_drift": "mark_price",
}


@router.post("/materialize")
async def materialize(req: MaterializeRequest) -> dict[str, Any]:
    asset = req.asset.lower()
    if asset not in _VALID_ASSETS:
        raise HTTPException(400, f"unknown asset '{req.asset}'")
    if not req.item:
        raise HTTPException(400, "item required")

    since, until = _resolve_window(req)
    key = f"{asset}.{req.item}"
    try:
        with engine.connect() as conn:
            payload = feature_builder.materialize(
                conn, asset, req.item, since, until,
                bucket_s=req.bucket_s,
            )
    except Exception as e:
        logger.exception("materialize failed for %s: %s", key, e)
        raise HTTPException(500, f"materialize error: {e}")

    return {"key": key, "asset": asset, "item": req.item,
            "window": {"since": since, "until": until},
            "payload": payload}


# ── data-range (preflight) ────────────────────────────────────────────────


class DataRangeRequest(BaseModel):
    components: list[str]   # ["btc.ohlcv_1m", "sol.ohlcv_1m"]


@router.post("/data-range")
async def data_range(req: DataRangeRequest) -> dict[str, Any]:
    """Return {component: {earliest, latest}} for preflight alignment checks."""
    results: dict[str, Any] = {}
    with engine.connect() as conn:
        # Batch by (asset, table) to avoid redundant queries.
        cache: dict[tuple[str, str], tuple[str | None, str | None]] = {}
        for key in req.components:
            parts = key.split(".", 1)
            if len(parts) != 2:
                results[key] = {"error": f"invalid key: {key}"}
                continue
            asset, item = parts
            if asset not in _VALID_ASSETS:
                results[key] = {"error": f"unknown asset: {asset}"}
                continue
            table = _ITEM_TABLE.get(item)
            if not table:
                results[key] = {"error": f"unknown item: {item}"}
                continue

            pair = (asset, table)
            if pair not in cache:
                row = conn.execute(
                    text(f"SELECT MIN(ts), MAX(ts) FROM {asset}.{table}")
                ).fetchone()
                if row and row[0] is not None:
                    cache[pair] = (row[0].isoformat(), row[1].isoformat())
                else:
                    cache[pair] = (None, None)

            earliest, latest = cache[pair]
            results[key] = {"earliest": earliest, "latest": latest}

    return {"ranges": results}


# ── preflight (materialize + Anthropic token count) ────────────────────────


class PreflightTokensRequest(BaseModel):
    """Build the exact same prompts as /analyze uses, then POST count_tokens."""

    components: list[str]
    since: str
    until: str
    directive: Optional[str] = None
    include_triples: bool = False
    model: Optional[str] = None
    effort: Optional[str] = anthropic_client.DEFAULT_EFFORT
    max_tokens: int = 4000  # must match /analyze; sizes prior-analysis placeholders
    bucket_s: Optional[int] = None


def _payload_prompt_csv_stats(payload: dict[str, Any]) -> dict[str, Any]:
    """Size of each component body as embedded in ``/analyze`` (full CSV block)."""
    ch, rows = prompt_csv.stats_full_csv_chars(payload)
    return {"prompt_csv_chars": ch, "series_rows_total": rows}


def _prior_stub_text(char_len: int) -> str:
    """Deterministic pseudo-prose up to char_len (pair/triple prior placeholder)."""
    unit = (
        "• Lead with the strongest microstructure signal; cite mid, spread bps, "
        "and imbalance from the payload.\n"
    )
    if char_len <= 0:
        return ""
    n = (char_len // len(unit)) + 2
    return (unit * n)[:char_len]


def _model_context_window(model_id: Optional[str]) -> int:
    target = model_id or anthropic_client.DEFAULT_MODEL
    for m in ALL_MODELS:
        if m["id"] == target:
            return int(m.get("context_window", 200_000))
    return 200_000


def _estimate_signal_buckets_payload(
    asset: str, since: str, until: str, bucket_s: int
) -> dict[str, Any]:
    """Build a synthetic payload with representative size for token estimation.

    Instead of running the full 600K-row computation, we calculate how many
    buckets the window produces and generate a single representative row
    repeated that many times.  This gives an accurate byte-count estimate
    without touching the database.
    """
    from datetime import datetime, timezone

    t0 = datetime.fromisoformat(since.replace("Z", "+00:00")).timestamp()
    t1 = datetime.fromisoformat(until.replace("Z", "+00:00")).timestamp()
    n_buckets = max(1, int((t1 - t0) / bucket_s))

    # Representative row with all 102 columns filled (worst-case size)
    sample_row = {
        "bucket_ts": since,
        "ask_size": 1234.56, "ask_size_usd": 234567.89,
        "ask_ppl": 45, "ask_fill": 0.78, "ask_centroid": 0.42,
        "bid_size": 1345.67, "bid_size_usd": 245678.90,
        "bid_ppl": 43, "bid_fill": 0.76, "bid_centroid": 0.39,
        "buy_volume": 56.78, "buy_volume_usd": 9876.54, "buy_count": 23,
        "sell_volume": 48.90, "sell_volume_usd": 8765.43, "sell_count": 19,
        "mid_o": 172.34, "mid_h": 172.89, "mid_l": 171.90,
        "mid_c": 172.56, "mid_mean": 172.42,
        "spread_o": 0.02, "spread_h": 0.04, "spread_l": 0.01,
        "spread_c": 0.03, "spread_mean": 0.025,
        "spread_o_bps": 1.16, "spread_h_bps": 2.32, "spread_l_bps": 0.58,
        "spread_c_bps": 1.74, "spread_mean_bps": 1.45,
        "poll_count": 120,
        "d_ask_size": -12.3, "d_ask_size_usd": -2345.6,
        "d_ask_ppl": -2, "d_ask_fill": -0.01, "d_ask_centroid": 0.02,
        "d_bid_size": 15.4, "d_bid_size_usd": 2890.1,
        "d_bid_ppl": 1, "d_bid_fill": 0.005, "d_bid_centroid": -0.01,
        "d_buy_volume": 3.4, "d_buy_volume_usd": 567.8, "d_buy_count": 2,
        "d_sell_volume": -5.6, "d_sell_volume_usd": -890.1, "d_sell_count": -3,
        "ask_replenish": 44.48, "ask_replenish_usd": 7530.94,
        "bid_replenish": 64.30, "bid_replenish_usd": 11655.53,
        "d_mid_o": 0.22, "d_mid_h": 0.55, "d_mid_l": -0.44,
        "d_mid_c": 0.22, "d_mid_mean": 0.08,
        "d_spread_o": 0.01, "d_spread_h": 0.02, "d_spread_l": -0.01,
        "d_spread_c": 0.01, "d_spread_mean": 0.005,
        "d_spread_o_bps": 0.58, "d_spread_h_bps": 1.16, "d_spread_l_bps": -0.58,
        "d_spread_c_bps": 0.58, "d_spread_mean_bps": 0.29,
        "dd_ask_size": 2.1, "dd_ask_size_usd": 345.6,
        "dd_ask_ppl": 1, "dd_ask_fill": 0.003, "dd_ask_centroid": -0.005,
        "dd_bid_size": -3.2, "dd_bid_size_usd": -567.8,
        "dd_bid_ppl": -1, "dd_bid_fill": -0.002, "dd_bid_centroid": 0.003,
        "dd_buy_volume": -1.2, "dd_buy_volume_usd": -200.3, "dd_buy_count": -1,
        "dd_sell_volume": 2.3, "dd_sell_volume_usd": 345.6, "dd_sell_count": 1,
        "d_ask_replenish": 5.5, "d_ask_replenish_usd": 912.4,
        "d_bid_replenish": -8.8, "d_bid_replenish_usd": -1456.7,
        "dd_mid_o": -0.11, "dd_mid_h": -0.33, "dd_mid_l": 0.22,
        "dd_mid_c": 0.0, "dd_mid_mean": -0.14,
        "dd_spread_o": -0.005, "dd_spread_h": -0.01, "dd_spread_l": 0.005,
        "dd_spread_c": 0.0, "dd_spread_mean": -0.003,
        "dd_spread_o_bps": -0.29, "dd_spread_h_bps": -0.58, "dd_spread_l_bps": 0.29,
        "dd_spread_c_bps": 0.0, "dd_spread_mean_bps": -0.15,
    }

    series = [sample_row] * min(n_buckets, 5) if n_buckets <= 5 else (
        [sample_row] * n_buckets
    )

    return {
        "kind": "signal_buckets",
        "asset": asset,
        "window": {"since": since, "until": until},
        "bucket_s": bucket_s,
        "summary": {"buckets": n_buckets, "polls": 0, "trades": 0,
                    "estimated": True},
        "series": series,
    }


@router.post("/preflight")
async def preflight_tokens(req: PreflightTokensRequest) -> dict[str, Any]:
    if not req.components:
        raise HTTPException(400, "components required")
    asset_item_by_key: dict[str, tuple[str, str]] = {}
    for key in req.components:
        parts = key.split(".", 1)
        if len(parts) != 2:
            raise HTTPException(400, f"invalid component key: {key}")
        asset = parts[0].lower()
        item = parts[1]
        if asset not in _VALID_ASSETS:
            raise HTTPException(400, f"unknown asset in {key!r}")
        asset_item_by_key[key] = (asset, item)

    payloads: dict[str, dict[str, Any]] = {}
    with engine.connect() as conn:
        for key, (asset, item) in asset_item_by_key.items():
            if item == "signal_buckets":
                # Estimate payload size without running the expensive full computation.
                # The actual materialization happens later in runPipeline.
                payloads[key] = _estimate_signal_buckets_payload(
                    asset, req.since, req.until, req.bucket_s or 5
                )
            else:
                payloads[key] = feature_builder.materialize(
                    conn, asset, item, req.since, req.until,
                    bucket_s=req.bucket_s,
                )

    keys = list(req.components)

    model = req.model or anthropic_client.DEFAULT_MODEL
    effort = req.effort or anthropic_client.DEFAULT_EFFORT
    cw = _model_context_window(model)
    use_openai_model = _is_openai_model(model)

    # Calibrate English-ish char→token ratio for this model (API-measured).
    if use_openai_model:
        chars_per_token = 4.0
    else:
        cal_sample = "word " * 6_000
        try:
            cal_tok = await anthropic_client.count_message_input_tokens(
                system="",
                user=cal_sample,
                model=model,
                effort="off",
            )
            chars_per_token = len(cal_sample) / max(cal_tok, 1)
        except Exception as e:
            logger.warning("preflight calibration failed, using 4.0 chars/token: %s", e)
            chars_per_token = 4.0

    prior_char_len = max(
        80,
        int(req.max_tokens * chars_per_token * 1.15),
    )
    prior_stub = _prior_stub_text(prior_char_len)

    LIMIT_BYTES = anthropic_client.COUNT_TOKENS_TOTAL_TEXT_BYTES_LIMIT
    TEXT_BUDGET_WARN_FRAC = 0.90
    est_bytes_by_key: dict[str, int] = {}
    likely_big_keys: list[str] = []
    for _key in keys:
        a0, item0 = asset_item_by_key[_key]
        c_sz = AnalyzeComponent(
            key=_key,
            asset=a0,
            item=item0,
            payload=payloads[_key],
        )
        areq_sz = AnalyzeRequest(
            mode="individual",
            components=[c_sz],
            directive=req.directive,
            model=model,
            effort=effort,
            max_tokens=req.max_tokens,
        )
        b_est = _estimate_prompt_utf8_bytes(areq_sz)
        est_bytes_by_key[_key] = b_est
        if b_est >= int(LIMIT_BYTES * TEXT_BUDGET_WARN_FRAC):
            likely_big_keys.append(_key)

    async def measure_tokens(
        mode: str,
        components: list[AnalyzeComponent],
        prior: Optional[list[PriorIndividual]] = None,
    ) -> tuple[int, dict[str, Any]]:
        areq_local = AnalyzeRequest(
            mode=mode,
            components=components,
            prior_individual=prior,
            directive=req.directive,
            model=model,
            effort=effort,
            max_tokens=req.max_tokens,
        )
        if use_openai_model:
            b = _estimate_prompt_utf8_bytes(areq_local)
            est_tokens = int(b / chars_per_token)
            return est_tokens, {"heuristic_estimate": True}
        return await _measure_input_tokens_halving(areq_local)

    indiv_rows: list[dict[str, Any]] = []
    indiv_tasks = []
    for key in keys:
        asset, item = asset_item_by_key[key]
        comp = AnalyzeComponent(
            key=key, asset=asset, item=item, payload=payloads[key],
        )

        async def _run_ind(k=key, c=comp) -> tuple[str, int, dict[str, Any]]:
            tok, meta = await measure_tokens("individual", [c])
            return k, tok, meta

        indiv_tasks.append(_run_ind())
    indiv_triplets = await asyncio.gather(*indiv_tasks)
    for key, tok, meta in sorted(indiv_triplets, key=lambda x: x[0]):
        extra = _payload_prompt_csv_stats(payloads[key])
        b_est = est_bytes_by_key.get(key, 0)
        indiv_rows.append({
            "key": key,
            "input_tokens": tok,
            "estimated_prompt_utf8_bytes": b_est,
            "likely_exceeds_anthropic_text_budget": key in likely_big_keys,
            **extra,
            **meta,
        })

    max_indiv = max((r["input_tokens"] for r in indiv_rows), default=0)

    # Reserve space for assistant completion (same cap as /analyze).
    max_out = max(1, int(req.max_tokens))
    eff_context = max(0, cw - max_out)

    pairs_max = 0
    pair_examples: list[dict[str, Any]] = []
    if len(keys) >= 2:
        p_tasks = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a_key, b_key = keys[i], keys[j]

                async def _run_pair(
                    ak=a_key,
                    bk=b_key,
                    ac=AnalyzeComponent(key=a_key, asset=asset_item_by_key[a_key][0],
                                        item=asset_item_by_key[a_key][1], payload=payloads[a_key]),
                    bc=AnalyzeComponent(key=b_key, asset=asset_item_by_key[b_key][0],
                                        item=asset_item_by_key[b_key][1], payload=payloads[b_key]),
                ) -> tuple[str, str, int]:
                    prior = [
                        PriorIndividual(key=ak, text=prior_stub),
                        PriorIndividual(key=bk, text=prior_stub),
                    ]
                    t, _ = await measure_tokens("pair", [ac, bc], prior=prior)
                    return ak, bk, t

                p_tasks.append(_run_pair())
        pt = await asyncio.gather(*p_tasks)
        if pt:
            pairs_max = max(x[2] for x in pt)
            worst = max(pt, key=lambda x: x[2])
            pair_examples.append({
                "a": worst[0], "b": worst[1],
                "input_tokens": worst[2],
            })

    triples_max = 0
    triple_example: dict[str, Any] = {}
    if req.include_triples and len(keys) >= 3:
        tr_tasks = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                for k in range(j + 1, len(keys)):
                    ak, bk, ck = keys[i], keys[j], keys[k]

                    async def _run_trip(
                        ka=ak, kb=bk, kc=ck,
                        comps=(
                            AnalyzeComponent(key=ak, asset=asset_item_by_key[ak][0],
                                             item=asset_item_by_key[ak][1], payload=payloads[ak]),
                            AnalyzeComponent(key=bk, asset=asset_item_by_key[bk][0],
                                             item=asset_item_by_key[bk][1], payload=payloads[bk]),
                            AnalyzeComponent(key=ck, asset=asset_item_by_key[ck][0],
                                             item=asset_item_by_key[ck][1], payload=payloads[ck]),
                        ),
                    ) -> tuple[int, tuple[str, str, str]]:
                        prior = [
                            PriorIndividual(key=ka, text=prior_stub),
                            PriorIndividual(key=kb, text=prior_stub),
                            PriorIndividual(key=kc, text=prior_stub),
                        ]
                        tt, _ = await measure_tokens("triple", list(comps), prior=prior)
                        return tt, (ka, kb, kc)

                    tr_tasks.append(_run_trip())
        tr = await asyncio.gather(*tr_tasks)
        if tr:
            worst_t = max(tr, key=lambda x: x[0])
            triples_max = worst_t[0]
            triple_example = {
                "a": worst_t[1][0], "b": worst_t[1][1], "c": worst_t[1][2],
                "input_tokens": worst_t[0],
            }

    n = len(keys)
    n_pair = n * (n - 1) // 2 if n >= 2 else 0
    n_trip = n * (n - 1) * (n - 2) // 6 if (req.include_triples and n >= 3) else 0

    checks: list[dict[str, Any]] = [
        {
            "kind": "individual",
            "count": n,
            "tokens_per_call_max": max_indiv,
            "worst_component": (
                max(indiv_rows, key=lambda x: x["input_tokens"])["key"] if indiv_rows else None
            ),
            "ok": max_indiv <= eff_context,
        },
    ]
    if n_pair:
        checks.append({
            "kind": "pair",
            "count": n_pair,
            "tokens_per_call_max": pairs_max,
            "worst_pair": pair_examples[0] if pair_examples else None,
            "ok": pairs_max <= eff_context,
        })
    if n_trip:
        checks.append({
            "kind": "triple",
            "count": n_trip,
            "tokens_per_call_max": triples_max,
            "worst_triple": triple_example or None,
            "ok": triples_max <= eff_context,
        })

    combo_max = max((c["tokens_per_call_max"] for c in checks), default=0)

    wb = int(LIMIT_BYTES * TEXT_BUDGET_WARN_FRAC)
    qual: list[str] = [
        "Component tables are compact CSV (```csv fences), not JSON — same as /analyze.",
    ]
    if likely_big_keys:
        ks = ", ".join(likely_big_keys)
        qual.insert(
            1,
            f"Warning: {len(likely_big_keys)} component(s) ({ks}) estimate system+user text at "
            f"≥ {wb:,} UTF-8 bytes (Anthropic count_tokens ceiling is ~{LIMIT_BYTES:,} bytes). "
            "Preflight still runs fully; expect count_tokens halving/scaling until the body fits.",
        )
    if any(
        r.get("count_tokens_extrapolated") or r.get("count_tokens_halving_steps", 0) > 0
        for r in indiv_rows
    ):
        qual.append(
            "Anthropic count_tokens rejected the prompt for size (HTTP 413 or HTTP 400: "
            "too many total text bytes). "
            "Per-component CSV series rows were repeatedly halved until counting succeeded; "
            "input_tokens is then scaled linearly using full_rows/sample_rows times the measured sample — "
            "see measured_input_tokens_sample and count_tokens_series_rows_* on each component row.",
        )

    return {
        "model": model,
        "effort": effort,
        "context_window": cw,
        "max_tokens_reserved": max_out,
        "input_token_budget": eff_context,
        "calibration_chars_per_token": round(chars_per_token, 4),
        "prior_placeholder_chars_each": prior_char_len,
        "pair_triple_prior_note": (
            "Prior sections are stubs sized via Anthropic count_tokens to approximate "
            f"each individual analysis capped at ~{req.max_tokens} output tokens "
            "(actual comparison calls may differ if Claude returns shorter/longer replies)."
        ),
        "preflight_notes": qual,
        "anthropic_count_tokens_text_bytes_limit": LIMIT_BYTES,
        "individual_detail": indiv_rows,
        "checks": checks,
        "combo_max_input_tokens": combo_max,
    }


# ── analyze (Claude) ──────────────────────────────────────────────────────


class AnalyzeComponent(BaseModel):
    key: str            # "btc.ohlcv_1m"
    asset: str
    item: str
    payload: dict[str, Any]


class PriorIndividual(BaseModel):
    key: str            # which component the analysis was for
    text: str


class AnalyzeRequest(BaseModel):
    mode: str           # "individual" | "pair" | "triple"
    components: list[AnalyzeComponent]
    prior_individual: Optional[list[PriorIndividual]] = None
    directive: Optional[str] = None
    system_prompt: Optional[str] = None  # override the default system prompt for this run
    model: Optional[str] = None
    effort: Optional[str] = anthropic_client.DEFAULT_EFFORT
    max_tokens: Optional[int] = 4000


SYS_INDIVIDUAL = (
    "You are a market-microstructure analyst working on Hyperliquid perpetual data. "
    "Each component is provided as compact CSV inside a fenced ```csv block: comment lines "
    "begin with # (kind, asset, window), then optional summary as two columns "
    "(metric,value), then the main time series with a header row followed by rows. "
    "Timestamps are Unix epoch in milliseconds in the ts column when present. "
    "Be precise, technical, non-verbose. Return 4-8 short bullet points. "
    "Lead with the single most actionable observation, then supporting numbers. "
    "Cite exact values from the CSV (mid, spread bps, imbalance, slope, etc.). "
    "Do not hedge. If the data is too thin to draw a conclusion, say so in one line."
)

SYS_COMPARE = (
    "You are a cross-instrument microstructure analyst on Hyperliquid perpetual data. "
    "Each component appears as compact CSV in a ```csv fence (see instructions in the "
    "individual analyst prompt for the format). "
    "You also receive individual analyses already produced for each component. Use those as "
    "prior context — do not re-derive what has already been said; instead synthesize across "
    "them. Compare: co-movement, lead/lag in seconds, divergences, regime, and a tradeable "
    "inference. Be precise, technical, non-verbose. 6-10 short bullets max. "
    "Lead with the strongest cross-instrument signal."
)


def _row_totals_for_prompt(
    areq: AnalyzeRequest,
    limits: dict[str, Optional[int]],
) -> tuple[int, int]:
    """(sampled_series_rows_sum, full_series_rows_sum) across components."""
    sampled = full = 0
    for c in areq.components:
        n = prompt_csv.payload_series_row_count(c.payload)
        full += n
        cap = limits.get(c.key)
        sampled += min(n, cap) if cap is not None else n
    return sampled, full


def _halve_row_limits(
    areq: AnalyzeRequest, limits: dict[str, Optional[int]]
) -> dict[str, Optional[int]]:
    """Halve included series rows per component (min 1 when n>0)."""
    out: dict[str, Optional[int]] = {}
    for c in areq.components:
        n = prompt_csv.payload_series_row_count(c.payload)
        if n == 0:
            out[c.key] = limits.get(c.key)
            continue
        cap = limits.get(c.key)
        included = min(n, cap) if cap is not None else n
        out[c.key] = max(1, included // 2)
    return out


def _every_nonempty_series_is_one_row(
    areq: AnalyzeRequest, limits: dict[str, Optional[int]]
) -> bool:
    """True only if there is ≥1 series row across components and every non-empty series is capped at 1."""
    saw_data = False
    for c in areq.components:
        n = prompt_csv.payload_series_row_count(c.payload)
        if n == 0:
            continue
        saw_data = True
        if limits.get(c.key) != 1:
            return False
    return saw_data


async def _measure_input_tokens_halving(areq: AnalyzeRequest) -> tuple[int, dict[str, Any]]:
    """Call ``count_tokens``; on size rejection (HTTP 413 or HTTP 400 text-byte limit), halve
    per-component ``series`` row caps and retry.

    Returns (estimated_full_prompt_input_tokens, detail_dict). When only a sample was
    counted, the first value scales measured tokens by (full_rows/sample_rows)—anchored on
    real Anthropic tokenization of the abbreviated prompt, not heuristic char guesses.
    """
    system = SYS_INDIVIDUAL if areq.mode == "individual" else SYS_COMPARE
    model = areq.model or anthropic_client.DEFAULT_MODEL
    effort = areq.effort or anthropic_client.DEFAULT_EFFORT
    limits: dict[str, Optional[int]] = {c.key: None for c in areq.components}
    halving_steps = 0
    measured_raw = 0
    anthropic_rejected_prompt_size = False

    while True:
        user = _build_user_prompt(areq, row_series_limits=limits)
        try:
            measured_raw = await anthropic_client.count_message_input_tokens(
                system=system, user=user, model=model, effort=effort,
            )
            break
        except anthropic_client.CountTokensPayloadTooLargeError:
            anthropic_rejected_prompt_size = True
            logger.warning(
                "count_tokens rejected prompt size (413 or 400); halving CSV series caps step %s",
                halving_steps + 1,
            )
            if _every_nonempty_series_is_one_row(areq, limits):
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Anthropic's count_tokens API rejected this prompt for size (HTTP 413 or HTTP 400 "
                        "total text bytes) even after reducing each non-empty ``series`` to a single row. "
                        "Try a narrower time window, fewer heavier components (e.g. raw L2), or split the run."
                    ),
                )
            limits = _halve_row_limits(areq, limits)
            halving_steps += 1

    sampled, total = _row_totals_for_prompt(areq, limits)
    extrapolated = total > 0 and sampled < total
    estimate = measured_raw if not extrapolated else max(
        measured_raw, int(round(measured_raw * total / max(1, sampled)))
    )

    detail: dict[str, Any] = {
        "measured_input_tokens_sample": measured_raw,
        "input_tokens_estimated_full": estimate,
        "count_tokens_series_rows_sampled_total": sampled,
        "count_tokens_series_rows_full_total": total,
        "count_tokens_halving_steps": halving_steps,
        "count_tokens_extrapolated": extrapolated,
        "anthropic_count_tokens_hit_413": anthropic_rejected_prompt_size or halving_steps > 0,
        "anthropic_count_tokens_prompt_size_rejected": anthropic_rejected_prompt_size,
    }
    return estimate, detail


def _build_user_prompt(
    req: AnalyzeRequest,
    row_series_limits: Optional[dict[str, Optional[int]]] = None,
) -> str:
    chunks: list[str] = []

    if req.directive:
        chunks.append(f"User directive:\n{req.directive.strip()}\n")

    if req.mode in ("pair", "triple") and req.prior_individual:
        chunks.append("Prior per-component analyses (use as context, do not repeat):\n")
        for p in req.prior_individual:
            chunks.append(f"--- {p.key} ---\n{p.text.strip()}\n")
        chunks.append("")

    chunks.append(f"Components ({len(req.components)}):")
    lim = row_series_limits or {}
    for c in req.components:
        cap = lim.get(c.key)
        chunks.append(
            prompt_csv.component_fenced_block(c.key, c.asset, c.item, c.payload,
                                              max_series_rows=cap),
        )

    if req.mode == "individual":
        chunks.append(
            "\nProduce a single individual analysis for the component above."
        )
    elif req.mode == "pair":
        chunks.append(
            "\nProduce a single pair-comparison analysis across the two components, "
            "using the prior individual analyses as starting context."
        )
    elif req.mode == "triple":
        chunks.append(
            "\nProduce a single triple-comparison analysis across the three components, "
            "using the prior individual analyses as starting context."
        )

    return "\n".join(chunks)


def _estimate_prompt_utf8_bytes(
    req: AnalyzeRequest,
    row_series_limits: Optional[dict[str, Optional[int]]] = None,
) -> int:
    """Rough UTF-8 byte size of ``system`` + ``user`` as built for Messages/count_tokens."""
    system = SYS_INDIVIDUAL if req.mode == "individual" else SYS_COMPARE
    user = _build_user_prompt(req, row_series_limits=row_series_limits)
    return len(system.encode("utf-8")) + len(user.encode("utf-8"))


@router.post("/analyze")
async def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    if req.mode not in ("individual", "pair", "triple"):
        raise HTTPException(400, "mode must be individual | pair | triple")
    if not req.components:
        raise HTTPException(400, "at least one component required")
    if req.mode == "individual" and len(req.components) != 1:
        raise HTTPException(400, "individual mode expects exactly 1 component")
    if req.mode == "pair" and len(req.components) != 2:
        raise HTTPException(400, "pair mode expects exactly 2 components")
    if req.mode == "triple" and len(req.components) != 3:
        raise HTTPException(400, "triple mode expects exactly 3 components")

    if req.system_prompt:
        system = req.system_prompt
    else:
        system = SYS_INDIVIDUAL if req.mode == "individual" else SYS_COMPARE
    use_openai = _is_openai_model(req.model or "")

    # Auto-cap signal_buckets series to fit within context window.
    # 102 columns × ~8 chars each ≈ 820 chars/row ÷ 4 chars/token ≈ 205 tokens/row.
    # Reserve ~30K tokens for system+summary+response, leaving ~170K for data on 200K models.
    # 170K ÷ 205 ≈ 830 max rows.  Use 600 as a safe cap accounting for other overhead.
    MAX_SIGNAL_BUCKET_ROWS = 600
    row_limits: dict[str, Optional[int]] = {}
    truncation_info: list[dict[str, Any]] = []
    for c in req.components:
        if c.item == "signal_buckets":
            series = c.payload.get("series") if isinstance(c.payload, dict) else None
            total_rows = len(series) if isinstance(series, list) else 0
            if total_rows > MAX_SIGNAL_BUCKET_ROWS:
                row_limits[c.key] = MAX_SIGNAL_BUCKET_ROWS
                truncation_info.append({
                    "key": c.key,
                    "total_rows": total_rows,
                    "sent_rows": MAX_SIGNAL_BUCKET_ROWS,
                    "truncated": True,
                    "method": "tail_600_with_full_range_stats",
                })
                logger.info(
                    "signal_buckets %s: capping %d rows → %d for LLM context fit",
                    c.key, total_rows, MAX_SIGNAL_BUCKET_ROWS,
                )
            else:
                truncation_info.append({
                    "key": c.key,
                    "total_rows": total_rows,
                    "sent_rows": total_rows,
                    "truncated": False,
                })

    user = _build_user_prompt(req, row_series_limits=row_limits or None)

    try:
        if use_openai:
            result = await openai_client.call_openai(
                system=system,
                user=user,
                model=req.model or "gpt-5.5-instant",
                effort="off",
                max_tokens=req.max_tokens or 12_000,
            )
        else:
            result = await anthropic_client.call_claude(
                system=system,
                user=user,
                model=req.model,
                effort=req.effort or "medium",
                max_tokens=req.max_tokens or 4000,
            )
    except anthropic_client.MessagesPayloadTooLargeError as e:
        logger.exception("analyze prompt too large for Anthropic HTTP: %s", e)
        raise HTTPException(
            status_code=413,
            detail=(
                "Anthropic Messages API rejected this prompt for size (HTTP 413 or HTTP 400 "
                "too-many-text-bytes). Narrow the Strat Hub window or drop very heavy feeds "
                f"(large raw L2 windows). Detail: {e!s}"
            ),
        )
    except Exception as e:
        logger.exception("analyze failed: %s", e)
        raise HTTPException(502, f"LLM API error: {e}")

    return {
        "mode": req.mode,
        "components": [c.key for c in req.components],
        "model": result["model"],
        "effort": req.effort,
        "analysis": result["text"],
        "thinking": result["thinking"],
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "stop_reason": result["stop_reason"],
        "data_truncation": truncation_info if truncation_info else None,
    }
