"""
Compact CSV serialization for Strat Hub LLM prompts.

Replaces verbose JSON for tabular payloads: metadata comment lines, a small
summary table, then header + data rows (token-efficient vs repeated JSON keys).
"""

from __future__ import annotations

import csv
from datetime import datetime
from io import StringIO
from math import isfinite
from typing import Any, Optional


def payload_series_row_count(payload: dict[str, Any]) -> int:
    s = payload.get("series")
    return len(s) if isinstance(s, list) else 0


def truncate_payload_series(payload: dict[str, Any], max_rows: Optional[int]) -> dict[str, Any]:
    """Shallow copy with ``series`` clipped to the first ``max_rows`` rows."""
    if max_rows is None:
        return payload
    s = payload.get("series")
    if not isinstance(s, list) or len(s) <= max_rows:
        return payload
    return {**payload, "series": s[:max_rows]}


def _col_order(keys: list[str]) -> list[str]:
    if not keys:
        return []
    ks = list(keys)
    if "ts" in ks:
        ks.remove("ts")
        return ["ts"] + sorted(ks)
    return sorted(ks)


def _parse_ts_to_ms(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, (int, float)) and isfinite(float(val)):
        v = float(val)
        if v > 1e12:  # already ms
            return int(v)
        if v > 1e9:  # unix seconds
            return int(v * 1000)
        return None
    if not isinstance(val, str):
        return None
    try:
        s = val.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _fmt_scalar(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if not isfinite(v):
            return ""
        # compact, strip trailing zeros
        s = format(v, ".10g")
        if "e" not in s.lower() and "." in s:
            s = s.rstrip("0").rstrip(".") or "0"
        return s
    if isinstance(v, str) and ("T" in v or "-" in v) and _parse_ts_to_ms(v) is not None:
        return str(_parse_ts_to_ms(v))
    s = str(v)
    if "," in s or "\n" in s or '"' in s:
        return s.replace("\r\n", " ").replace("\n", " ")
    return s


def _row_dict_to_csv_row(headers: list[str], row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for h in headers:
        v = row.get(h)
        if h == "ts":
            ms = _parse_ts_to_ms(v)
            if ms is not None:
                out.append(str(ms))
                continue
        out.append(_fmt_scalar(v))
    return out


def _write_csv_table(headers: list[str], rows: list[list[str]]) -> str:
    buf = StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(headers)
    for r in rows:
        w.writerow(r)
    return buf.getvalue().rstrip("\n")


def _signal_buckets_column_stats(series: list[dict]) -> str:
    """Compute min/max/mean/last for key numeric columns across the full series.

    Provides Claude with full-range context even when the raw series is truncated.
    """
    if not series:
        return ""
    # Focus on the most important columns for analysis
    key_cols = [
        "mid_mean", "ask_size", "bid_size", "buy_volume", "sell_volume",
        "ask_replenish", "bid_replenish", "spread_mean_bps",
        "ask_size_usd", "bid_size_usd", "buy_volume_usd", "sell_volume_usd",
        "d_ask_size", "d_bid_size", "d_buy_volume", "d_sell_volume",
    ]
    rows_out: list[str] = ["# Full-range column statistics (all buckets):"]
    rows_out.append("column,min,max,mean,first,last")
    for col in key_cols:
        vals = [float(r[col]) for r in series
                if isinstance(r, dict) and r.get(col) is not None
                and isinstance(r[col], (int, float))]
        if not vals:
            continue
        mn = min(vals)
        mx = max(vals)
        avg = sum(vals) / len(vals)
        first = vals[0]
        last = vals[-1]
        rows_out.append(
            f"{col},{_fmt_scalar(mn)},{_fmt_scalar(mx)},{_fmt_scalar(avg)},"
            f"{_fmt_scalar(first)},{_fmt_scalar(last)}"
        )
    return "\n".join(rows_out)


def payload_to_component_csv(
    payload: dict[str, Any],
    *,
    max_series_rows: Optional[int] = None,
) -> tuple[str, int, int]:
    """
    Returns (full_text, series_rows_included, series_rows_total).

    ``max_series_rows`` limits only the main ``series`` table; summary is always full.
    """
    kind = str(payload.get("kind", "unknown"))
    asset = str(payload.get("asset", ""))
    win = payload.get("window") if isinstance(payload.get("window"), dict) else {}
    since = win.get("since", "")
    until = win.get("until", "")

    lines: list[str] = [
        f"# kind={kind}",
        f"# asset={asset}",
        f"# window_since={since}",
        f"# window_until={until}",
    ]

    summary = payload.get("summary")
    if isinstance(summary, dict) and summary:
        kv: list[tuple[str, str]] = []
        for k in sorted(summary.keys()):
            kv.append((str(k), _fmt_scalar(summary[k])))
        lines.append(_write_csv_table(["metric", "value"], [[a, b] for a, b in kv]))

    series = payload.get("series")
    total = len(series) if isinstance(series, list) else 0
    if isinstance(series, list) and series:
        # For signal_buckets with truncation: take TAIL (most recent) + prepend stats summary
        is_signal_buckets = kind == "signal_buckets"
        truncated = max_series_rows is not None and total > max_series_rows

        if truncated and is_signal_buckets:
            use = series[-max_series_rows:]
            lines.append(
                f"# NOTE: showing last {max_series_rows} of {total} buckets. "
                f"Full range statistics included in summary above."
            )
            # Add per-column statistics for the full series (helps Claude understand the full picture)
            lines.append(_signal_buckets_column_stats(series))
        elif max_series_rows is not None:
            use = series[:max_series_rows]
        else:
            use = series

        first = use[0]
        if isinstance(first, dict):
            headers = _col_order(list(first.keys()))
            row_matrix = [_row_dict_to_csv_row(headers, dict(r)) if isinstance(r, dict) else [] for r in use]
            row_matrix = [r for r in row_matrix if len(r) == len(headers)]
            lines.append(_write_csv_table(headers, row_matrix))
        elif isinstance(first, (list, tuple)):
            maxlen = max(len(r) if isinstance(r, (list, tuple)) else 0 for r in use)
            hdr = [f"c{i}" for i in range(maxlen)]
            mat: list[list[str]] = []
            for r in use:
                if not isinstance(r, (list, tuple)):
                    continue
                mat.append([_fmt_scalar(r[i]) if i < len(r) else "" for i in range(maxlen)])
            lines.append(_write_csv_table(hdr, mat))
        else:
            lines.append(_write_csv_table(["value"], [[_fmt_scalar(r)] for r in use]))

    text = "\n".join(lines)
    included = min(total, max_series_rows) if max_series_rows is not None else total
    return text, included, total


def component_fenced_block(
    display_key: str,
    asset: str,
    item: str,
    payload: dict[str, Any],
    *,
    max_series_rows: Optional[int] = None,
) -> str:
    """One component section: title + fenced csv (used inside user prompt)."""
    body, _, _ = payload_to_component_csv(payload, max_series_rows=max_series_rows)
    return (
        f"\n=== {display_key} ({asset}.{item}) ===\n"
        f"```csv\n{body}\n```"
    )


def stats_full_csv_chars(payload: dict[str, Any]) -> tuple[int, int]:
    """Character length of the full (untruncated) CSV block body + total series rows."""
    body, _, total = payload_to_component_csv(payload, max_series_rows=None)
    return len(body), total
