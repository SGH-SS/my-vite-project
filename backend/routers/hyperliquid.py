"""
Hyperliquid dashboard API  (/api/hl)
====================================
Account, positions (with the exchange's liquidation price), market data, orders, transfers, and the V2 strategy
runner controls.  Every call goes through the selected proxy (0..3 = Proxy 1..4, -1 = direct; default Proxy 1).
Shared client: sol-perp/hlcore.
"""
from __future__ import annotations
import json, logging, os, subprocess, sys, time, statistics
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, Query
from pydantic import BaseModel

_SOLPERP = Path(__file__).resolve().parent.parent.parent.parent / "sol-perp"
if str(_SOLPERP) not in sys.path:
    sys.path.insert(0, str(_SOLPERP))
from hlcore import HL, load_config, proxy_label, DEFAULT_PROXY, estimate_liq, maint_frac_for   # noqa: E402

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/hl", tags=["hyperliquid"])
STRAT_DIR = _SOLPERP / "strat"
RUN_ROOT = Path(os.environ.get("V2_RUNNER_HOME") or (Path(os.environ.get("LOCALAPPDATA", str(STRAT_DIR))) / "v2runner"))   # same place the runner writes (outside OneDrive)
STATE_DIR = RUN_ROOT / "state"
PY = sys.executable


_CLIENTS: dict = {}


def _hl(network: str, proxy_index: int) -> HL:
    """One long-lived client per (network, proxy) so its meta / market caches are shared by every request
    (a fresh client per request re-downloaded meta each time and tripped Hyperliquid's per-IP rate limit)."""
    key = ("testnet" if network == "testnet" else "mainnet", proxy_index)
    if key not in _CLIENTS:
        _CLIENTS[key] = HL(*key)
    return _CLIENTS[key]


def _err(e: Exception, proxy_index: int, **extra):
    logger.error("hl: %s", e)
    return {"error": str(e), "proxy_index": proxy_index, **extra}


# ── read endpoints ─────────────────────────────────────────────────────────
@router.get("/proxies")
def get_proxies():
    cfg = load_config()
    return {"proxies": [{"index": i, "label": proxy_label(i, cfg)} for i in range(len(cfg.get("proxies", [])))], "default": DEFAULT_PROXY}


@router.get("/latency")
def get_latency(proxy_index: int = Query(DEFAULT_PROXY), network: str = Query("mainnet"), runs: int = Query(3)):
    try:
        hl = _hl(network, proxy_index); tests = []
        for name, payload in [("allMids", {"type": "allMids"}), ("clearinghouseState", {"type": "clearinghouseState", "user": hl.master}), ("metaAndAssetCtxs", {"type": "metaAndAssetCtxs"})]:
            ms = []; errors = []
            for _ in range(runs):
                t0 = time.perf_counter()
                try:
                    hl.info(payload, retries=0); ms.append((time.perf_counter() - t0) * 1000)
                except Exception as e:                  # noqa: BLE001
                    errors.append(str(e))
            tests.append({"name": name, "runs": len(ms), "median_ms": round(statistics.median(ms), 1) if ms else None, "min_ms": round(min(ms), 1) if ms else None, "max_ms": round(max(ms), 1) if ms else None, "errors": errors})
        return {"proxy_index": proxy_index, "proxy_label": proxy_label(proxy_index), "network": network, "tests": tests}
    except Exception as e:                              # noqa: BLE001
        return _err(e, proxy_index)


@router.get("/account")
def get_account(network: str = Query("mainnet"), proxy_index: int = Query(DEFAULT_PROXY)):
    try:
        hl = _hl(network, proxy_index); out = hl.account(); out["proxy_label"] = proxy_label(proxy_index)
        for p in out["positions"]:
            try:
                # mark = position value / size (the exchange values positions at mark): no extra request every refresh
                mark = p["position_value"] / p["size"] if p["size"] else None; p["mark_px"] = mark
                p["funding_rate"] = hl.cached(f"funding:{p['coin']}", 60, lambda c=p["coin"]: hl.asset_ctx(c)["funding"])
                if p.get("liquidation_px") and mark:
                    p["liq_distance_px"] = abs(mark - p["liquidation_px"]); p["liq_distance_pct"] = p["liq_distance_px"] / mark * 100
            except Exception:                           # noqa: BLE001
                pass
        return out
    except Exception as e:                              # noqa: BLE001
        return _err(e, proxy_index)


@router.get("/market")
def get_market(coin: str = Query("SOL"), network: str = Query("mainnet"), proxy_index: int = Query(DEFAULT_PROXY)):
    try:
        hl = _hl(network, proxy_index); out = hl.asset_ctx(coin); out.update(hl.asset(coin)); out["proxy_index"] = proxy_index
        out["fetched_at"] = datetime.now(timezone.utc).isoformat(); return out
    except Exception as e:                              # noqa: BLE001
        return _err(e, proxy_index)


@router.get("/coins")
def get_coins(network: str = Query("mainnet"), proxy_index: int = Query(DEFAULT_PROXY)):
    try:
        hl = _hl(network, proxy_index)
        from hlcore.client import BUILDER_DEXES
        return {"coins": [{"name": u["name"], "maxLeverage": u.get("maxLeverage"), "szDecimals": u.get("szDecimals")}
                          for dex in ("",) + tuple(BUILDER_DEXES) for u in hl.meta(dex).get("universe", []) if not u.get("isDelisted")]}
    except Exception as e:                              # noqa: BLE001
        return _err(e, proxy_index)


def _estimate(hl: HL, coin: str, side: str, margin: float, leverage: int, is_cross: bool, px: float | None = None, account_value: float | None = None):
    a = hl.asset(coin); mark = px or hl.mark_px(coin)
    notional = margin * leverage; sz = hl.round_sz(coin, notional / mark)
    mf = maint_frac_for(notional, a["maxLeverage"], a["marginTiers"])
    liq = estimate_liq(side, mark, sz, margin, mf, is_cross=is_cross, account_value=account_value)
    return {"coin": coin, "side": side, "mark_px": mark, "margin": margin, "leverage": leverage, "notional": notional, "size": sz, "sz_decimals": a["szDecimals"],
            "max_leverage": a["maxLeverage"], "maint_frac": mf, "liquidation_px_estimate": liq,
            "liq_distance_pct": (abs(mark - liq) / mark * 100) if liq else None, "is_cross": is_cross}


@router.get("/estimate")
def get_estimate(coin: str = Query("SOL"), side: str = Query("long"), margin: float = Query(...), leverage: int = Query(...), is_cross: bool = Query(False),
                 network: str = Query("mainnet"), proxy_index: int = Query(DEFAULT_PROXY)):
    try:
        hl = _hl(network, proxy_index); av = hl.account()["account_value"] if is_cross else None
        return _estimate(hl, coin, side, margin, leverage, is_cross, account_value=av)
    except Exception as e:                              # noqa: BLE001
        return _err(e, proxy_index)


# ── write endpoints ────────────────────────────────────────────────────────
class LeverageReq(BaseModel):
    coin: str; leverage: int; is_cross: bool = False; network: str = "mainnet"; proxy_index: int = DEFAULT_PROXY

class OrderReq(BaseModel):
    coin: str; side: str; margin: float; leverage: int; is_cross: bool = False
    tp_price: float | None = None; sl_price: float | None = None; slippage: float = 0.03
    network: str = "mainnet"; proxy_index: int = DEFAULT_PROXY

class CloseReq(BaseModel):
    coin: str; network: str = "mainnet"; proxy_index: int = DEFAULT_PROXY

class CancelReq(BaseModel):
    coin: str; oid: int; network: str = "mainnet"; proxy_index: int = DEFAULT_PROXY

class TransferReq(BaseModel):
    amount: float; to_perp: bool; network: str = "mainnet"; proxy_index: int = DEFAULT_PROXY


@router.post("/leverage")
def post_leverage(req: LeverageReq):
    if req.leverage < 1:
        return {"error": "leverage must be >= 1"}
    try:
        return {**_hl(req.network, req.proxy_index).set_leverage(req.coin, req.leverage, req.is_cross), "coin": req.coin, "leverage": req.leverage, "is_cross": req.is_cross}
    except Exception as e:                              # noqa: BLE001
        return _err(e, req.proxy_index)


@router.post("/order")
def post_order(req: OrderReq):
    """Set leverage, size = margin x leverage / mark, IOC market-like order (+ optional TP/SL), then re-read the
    position so the response carries the exchange's actual liquidation price beside the estimate."""
    if req.margin <= 0 or req.leverage < 1 or req.side not in ("long", "short"):
        return {"error": "bad margin / leverage / side"}
    try:
        hl = _hl(req.network, req.proxy_index)
        acct = hl.account(); est = _estimate(hl, req.coin, req.side, req.margin, req.leverage, req.is_cross, account_value=acct["account_value"] if req.is_cross else None)
        if est["size"] <= 0:
            return {"error": f"size rounds to 0 ({est['notional']:.2f} USD at {est['mark_px']})", "estimate": est}
        if not req.is_cross and req.margin > acct["available_to_trade"]:
            return {"error": f"margin {req.margin:.2f} is more than the {acct['available_to_trade']:.2f} USDC available to trade", "estimate": est}
        lev = hl.set_leverage(req.coin, req.leverage, req.is_cross)
        if not lev["ok"]:
            return {"ok": False, "error": f"leverage not set: {lev['raw']}", "step": "leverage", "estimate": est}
        is_buy = req.side == "long"
        if req.tp_price or req.sl_price:
            res = hl.order_with_tpsl(req.coin, is_buy, est["size"], req.tp_price, req.sl_price, slippage=req.slippage)
        else:
            res = hl.market_order(req.coin, is_buy, est["size"], slippage=req.slippage, px=est["mark_px"])
        time.sleep(0.6)
        pos = hl.position(req.coin)
        out = {"ok": res["ok"] and res["filled_sz"] > 0, "fill": {k: res[k] for k in ("filled_sz", "avg_px", "oid", "resting", "error", "limit_px", "mark_px")},
               "estimate": est, "position": pos, "liquidation_px_estimate": est["liquidation_px_estimate"],
               "liquidation_px_actual": pos["liquidation_px"] if pos else None, "network": req.network, "proxy_index": req.proxy_index,
               "completed_at": datetime.now(timezone.utc).isoformat(), "raw": res["raw"]}
        if out["liquidation_px_actual"] and est["liquidation_px_estimate"]:
            out["liquidation_diff_px"] = out["liquidation_px_actual"] - est["liquidation_px_estimate"]
        logger.info("hl order %s %s %s margin %.2f x%d -> %s", req.network, req.side, req.coin, req.margin, req.leverage, out["fill"])
        return out
    except Exception as e:                              # noqa: BLE001
        return _err(e, req.proxy_index)


@router.post("/close")
def post_close(req: CloseReq):
    try:
        hl = _hl(req.network, req.proxy_index); res = hl.close_position(req.coin); time.sleep(0.6)
        return {"ok": res.get("ok"), "fill": {k: res.get(k) for k in ("filled_sz", "avg_px", "error")}, "position": hl.position(req.coin), "raw": res.get("raw")}
    except Exception as e:                              # noqa: BLE001
        return _err(e, req.proxy_index)


# ── open-position management: leverage change, add, partial reduce, isolated margin add/remove, with previews ──
class PosLeverageReq(BaseModel):
    coin: str; leverage: int; margin: float | None = None; slippage: float = 0.03; network: str = "mainnet"; proxy_index: int = DEFAULT_PROXY

class PosAddReq(BaseModel):
    coin: str; margin: float; leverage: int | None = None; slippage: float = 0.03; network: str = "mainnet"; proxy_index: int = DEFAULT_PROXY

class PosReduceReq(BaseModel):
    coin: str; fraction: float | None = None; size: float | None = None; slippage: float = 0.03; network: str = "mainnet"; proxy_index: int = DEFAULT_PROXY

class PosMarginReq(BaseModel):
    coin: str; amount: float; network: str = "mainnet"; proxy_index: int = DEFAULT_PROXY


def _pos_preview(hl: HL, coin: str, pos: dict, new_lev: int | None = None, add_margin: float = 0.0, reduce_fraction: float = 0.0, margin_delta: float = 0.0, target_margin: float | None = None) -> dict:
    """Estimated liquidation of the position after the change (documented formula; the exchange's number after)."""
    a = hl.asset(coin); mark = hl.mark_px(coin); lev = new_lev or int(pos["leverage"] or 1)
    sz = pos["size"] * (1 - reduce_fraction); margin = pos["margin_used"] * (1 - reduce_fraction)
    if target_margin is not None and new_lev:                        # "this much margin at this leverage": the size follows
        margin = target_margin; sz = hl.round_sz(coin, target_margin * new_lev / mark)
        mf = maint_frac_for(sz * mark, a["maxLeverage"], a["marginTiers"])
        liq = estimate_liq(pos["side"], mark, sz, margin, mf, is_cross=pos.get("leverage_type") == "cross", account_value=None)
        return {"size": sz, "notional": sz * mark, "margin": margin, "leverage_effective": (sz * mark / margin) if margin > 0 else None, "liquidation_px_estimate": liq,
                "liq_distance_pct": (abs(mark - liq) / mark * 100) if liq else None, "mark_px": mark, "size_delta": sz - pos["size"], "notional_delta": (sz - pos["size"]) * mark,
                "margin_delta": margin - pos["margin_used"]}
    if new_lev and not reduce_fraction and not margin_delta:          # re-lever keeping the size: margin becomes notional / leverage
        margin = pos["position_value"] / new_lev
    if add_margin:
        add_sz = hl.round_sz(coin, add_margin * lev / mark); sz += add_sz; margin += add_margin
        if new_lev:
            margin = (pos["position_value"] + add_sz * mark) / new_lev
    margin += margin_delta
    mf = maint_frac_for(sz * mark, a["maxLeverage"], a["marginTiers"])
    liq = estimate_liq(pos["side"], pos["entry_px"] if not add_margin else mark, sz, margin, mf, is_cross=pos.get("leverage_type") == "cross", account_value=None)
    return {"size": sz, "notional": sz * mark, "margin": margin, "leverage_effective": (sz * mark / margin) if margin > 0 else None, "liquidation_px_estimate": liq,
            "liq_distance_pct": (abs(mark - liq) / mark * 100) if liq else None, "mark_px": mark}


@router.get("/position/preview")
def position_preview(coin: str = Query(...), new_leverage: int | None = Query(None), add_margin: float = Query(0.0), reduce_fraction: float = Query(0.0), margin_delta: float = Query(0.0),
                     target_margin: float | None = Query(None), network: str = Query("mainnet"), proxy_index: int = Query(DEFAULT_PROXY)):
    try:
        hl = _hl(network, proxy_index); pos = hl.position(coin)
        if not pos:
            return {"error": f"no open {coin} position"}
        return {"current": pos, "after": _pos_preview(hl, coin, pos, new_leverage, add_margin, reduce_fraction, margin_delta, target_margin)}
    except Exception as e:                              # noqa: BLE001
        return _err(e, proxy_index)


def _after(hl: HL, coin: str, res: dict, extra: dict | None = None) -> dict:
    time.sleep(0.6); pos = hl.position(coin)
    return {"ok": res.get("ok"), "fill": {k: res.get(k) for k in ("filled_sz", "avg_px", "error", "limit_px")} if "filled_sz" in res else None, "raw": res.get("raw"),
            "position": pos, "liquidation_px_actual": pos["liquidation_px"] if pos else None, **(extra or {})}


@router.post("/position/leverage")
def position_leverage(req: PosLeverageReq):
    try:
        hl = _hl(req.network, req.proxy_index); pos = hl.position(req.coin)
        if not pos:
            return {"error": f"no open {req.coin} position"}
        if pos.get("leverage_type") == "cross":
            res = hl.set_leverage(req.coin, req.leverage, is_cross=True)
            return _after(hl, req.coin, res, {"leverage": req.leverage, "note": "cross position: leverage setting only"})
        lv = hl.set_leverage(req.coin, req.leverage, is_cross=False)
        if not lv["ok"]:
            return {"ok": False, "error": f"leverage not set: {lv['raw']}"}
        if req.margin is not None:                                   # keep this much margin in the position; size = margin x leverage
            mark = hl.mark_px(req.coin); target_sz = hl.round_sz(req.coin, req.margin * req.leverage / mark); d_sz = round(target_sz - pos["size"], 8)
            steps = {"leverage_set": True, "size_before": pos["size"], "size_target": target_sz, "size_delta": d_sz, "margin_before": pos["margin_used"], "margin_target": req.margin}
            if d_sz > 0:
                need = d_sz * mark / req.leverage; acct = hl.account(); free = acct["available_to_trade"]
                if free < need * 1.01:
                    return {"ok": False, "error": f"buying {d_sz} {req.coin} needs {need:.2f} USDC free margin, only {free:.2f} is free", **steps}
                fill = hl.add_to_position(req.coin, d_sz, slippage=req.slippage)
            elif d_sz < 0:
                fill = hl.reduce_position(req.coin, -d_sz, slippage=req.slippage)
            else:
                fill = {"ok": True, "filled_sz": 0.0, "avg_px": None}
            steps["fill"] = {k: fill.get(k) for k in ("filled_sz", "avg_px", "error")}
            if not fill.get("ok"):
                return {"ok": False, "error": f"size change failed: {fill.get('error') or fill.get('raw')}", **steps}
            time.sleep(0.6); pos2 = hl.position(req.coin) or pos; delta = req.margin - pos2["margin_used"]; steps["margin_after_fill"] = pos2["margin_used"]
            if abs(delta) >= 1.0:
                if delta > 0:
                    acct = hl.account(); free = acct["available_to_trade"]
                    if free < delta:
                        steps["margin_note"] = f"wanted {delta:.2f} more margin but only {free:.2f} free"; delta = min(delta, free - 0.5)
                mv = hl.update_isolated_margin(req.coin, round(delta, 2) if delta > 0 else round(delta * 0.998, 2)); steps["margin_move"] = mv.get("raw")
            logger.info("position set %s %s margin %.2f x%d: size %.5f -> %.5f", req.network, req.coin, req.margin, req.leverage, pos["size"], target_sz)
            return _after(hl, req.coin, {"ok": True}, {"leverage": req.leverage, **steps})
        target = pos["position_value"] / req.leverage; delta = target - pos["margin_used"]
        steps = {"leverage_set": True, "margin_before": pos["margin_used"], "margin_target": target, "margin_delta": delta}
        if abs(delta) >= 1.0:
            if delta > 0:
                acct = hl.account(); free = acct["available_to_trade"]
                if free < delta:
                    return {"ok": False, "error": f"need {delta:.2f} USDC more margin but only {free:.2f} is free", **steps}
            mv = hl.update_isolated_margin(req.coin, round(delta, 2) if delta > 0 else round(delta * 0.998, 2))   # cents; tiny guard when removing
            steps["margin_move"] = mv.get("raw")
            if not mv["ok"]:
                return {"ok": False, "error": f"margin move failed: {mv['raw']}", **steps}
        logger.info("position re-lever %s %s -> %dx (margin %.2f -> %.2f)", req.network, req.coin, req.leverage, pos["margin_used"], target)
        return _after(hl, req.coin, {"ok": True}, {"leverage": req.leverage, **steps})
    except Exception as e:                              # noqa: BLE001
        return _err(e, req.proxy_index)


@router.post("/position/add")
def position_add(req: PosAddReq):
    """Buy more on the same side: size = margin x leverage / mark (leverage = the position's unless given, which re-sets it first)."""
    if req.margin <= 0:
        return {"error": "margin must be positive"}
    try:
        hl = _hl(req.network, req.proxy_index); pos = hl.position(req.coin)
        if not pos:
            return {"error": f"no open {req.coin} position"}
        lev = req.leverage or int(pos["leverage"])
        if req.leverage and req.leverage != int(pos["leverage"]):
            lv = hl.set_leverage(req.coin, req.leverage, is_cross=pos.get("leverage_type") == "cross")
            if not lv["ok"]:
                return {"ok": False, "error": f"leverage not set: {lv['raw']}"}
        sz = hl.round_sz(req.coin, req.margin * lev / hl.mark_px(req.coin))
        res = hl.add_to_position(req.coin, sz, slippage=req.slippage)
        logger.info("position add %s %s margin %.2f x%d -> %s", req.network, req.coin, req.margin, lev, {k: res.get(k) for k in ("filled_sz", "avg_px", "error")})
        return _after(hl, req.coin, res, {"size_added": sz, "leverage": lev})
    except Exception as e:                              # noqa: BLE001
        return _err(e, req.proxy_index)


@router.post("/position/reduce")
def position_reduce(req: PosReduceReq):
    try:
        hl = _hl(req.network, req.proxy_index); pos = hl.position(req.coin)
        if not pos:
            return {"error": f"no open {req.coin} position"}
        sz = req.size if req.size else pos["size"] * float(req.fraction or 0)
        if sz <= 0:
            return {"error": "give a fraction (0-1) or a size"}
        res = hl.reduce_position(req.coin, sz, slippage=req.slippage)
        logger.info("position reduce %s %s %.5f -> %s", req.network, req.coin, sz, {k: res.get(k) for k in ("filled_sz", "avg_px", "error")})
        return _after(hl, req.coin, res, {"size_reduced": sz})
    except Exception as e:                              # noqa: BLE001
        return _err(e, req.proxy_index)


@router.post("/position/margin")
def position_margin(req: PosMarginReq):
    """Add (positive) or remove (negative) USDC from the isolated margin of the open position."""
    if req.amount == 0:
        return {"error": "amount must be non-zero"}
    try:
        hl = _hl(req.network, req.proxy_index); res = hl.update_isolated_margin(req.coin, req.amount)
        logger.info("position margin %s %s %+.2f -> %s", req.network, req.coin, req.amount, res.get("raw"))
        return _after(hl, req.coin, res, {"amount": req.amount})
    except Exception as e:                              # noqa: BLE001
        return _err(e, req.proxy_index)


@router.post("/cancel")
def post_cancel(req: CancelReq):
    try:
        return _hl(req.network, req.proxy_index).cancel(req.coin, req.oid)
    except Exception as e:                              # noqa: BLE001
        return _err(e, req.proxy_index)


class ConvertReq(BaseModel):
    coin: str = "UBTC"; slippage: float = 0.01; network: str = "mainnet"; proxy_index: int = DEFAULT_PROXY


@router.post("/convert-to-usdc")
def post_convert(req: ConvertReq):
    """Sell the whole spot balance of `coin` for USDC (market-like IOC on the spot pair)."""
    try:
        hl = _hl(req.network, req.proxy_index); res = hl.spot_sell_all(req.coin, req.slippage); time.sleep(0.6)
        acct = hl.account()
        logger.info("convert %s -> USDC on %s: %s", req.coin, req.network, {k: res.get(k) for k in ("filled_sz", "avg_px", "error")})
        return {"ok": res.get("ok") and res.get("filled_sz", 0) > 0, "fill": {k: res.get(k) for k in ("filled_sz", "avg_px", "sz", "balance", "limit_px", "mid_px", "est_usd", "error", "pair")},
                "usdc_total": acct.get("usdc_total"), "spot_balances": acct.get("spot_balances"), "raw": res.get("raw")}
    except Exception as e:                              # noqa: BLE001
        return _err(e, req.proxy_index)


@router.post("/transfer")
def post_transfer(req: TransferReq):
    if req.amount <= 0:
        return {"error": "amount must be positive"}
    try:
        return {**_hl(req.network, req.proxy_index).usd_transfer(req.amount, req.to_perp), "amount": req.amount, "to_perp": req.to_perp}
    except Exception as e:                              # noqa: BLE001
        return _err(e, req.proxy_index)


# ── strategy runner controls ───────────────────────────────────────────────
def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return None


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, timeout=10).stdout
        return str(pid) in out
    except Exception:                                   # noqa: BLE001
        return False


def _runner_status() -> dict:
    hb = _read_json(STATE_DIR / "v2_heartbeat.json") or {}
    st = _read_json(STATE_DIR / "v2_state.json") or {}
    pid = hb.get("pid"); age = (time.time() - hb["ts"]) if hb.get("ts") else None
    alive = _pid_alive(pid) and age is not None and age < 60
    trades = []
    jp = STATE_DIR / "v2_trades.jsonl"
    if jp.exists():
        trades = [json.loads(l) for l in jp.read_text(encoding="utf-8").splitlines() if l.strip()][-20:]
    sup_pid = int((STATE_DIR / "v2_supervisor.pid").read_text()) if (STATE_DIR / "v2_supervisor.pid").exists() else None
    return {"installed": (STRAT_DIR / "runner.py").exists(), "running": alive, "pid": pid, "heartbeat_age_s": round(age, 1) if age is not None else None,
            "supervisor_pid": sup_pid, "supervisor_alive": _pid_alive(sup_pid), "stopped_by_user": (STATE_DIR / "v2_stopped_by_user").exists(),
            "launch": _read_json(STATE_DIR / "v2_launch.json"),
            "heartbeat": hb, "state": st, "recent_trades": trades, "fetched_at": datetime.now(timezone.utc).isoformat()}


class ArmReq(BaseModel):
    network: str = "mainnet"; proxy_index: int = DEFAULT_PROXY
    unit_margin: float = 333.0; leverage: int = 6; allocation: float = 1000.0; max_daily_loss: float = 500.0; dry_run: bool = False

class ControlReq(BaseModel):
    flatten: bool = False


@router.get("/strategy/status")
def strategy_status():
    return _runner_status()


@router.post("/strategy/arm")
def strategy_arm(req: ArmReq):
    """Start the runner in its own visible PowerShell window (or re-arm a running, disarmed one)."""
    st = _runner_status()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if st["running"]:
        (STATE_DIR / "v2_control.json").write_text(json.dumps({"cmd": "arm", "ts": time.time()}), encoding="utf-8")
        return {"ok": True, "action": "re-armed running process", "pid": st["pid"]}
    if not (STRAT_DIR / "runner.py").exists():
        return {"ok": False, "error": "runner not installed (sol-perp/strat/runner.py missing)"}
    args = f"-u runner.py --network {req.network} --proxy {req.proxy_index} --unit-margin {req.unit_margin} --leverage {req.leverage} --allocation {req.allocation} --max-daily-loss {req.max_daily_loss}{' --dry-run' if req.dry_run else ''}"
    title = f"V2 RUNNER {req.network} proxy{req.proxy_index + 1 if req.proxy_index >= 0 else '-direct'}{' DRY RUN' if req.dry_run else ''}"
    return _launch(args, title)


def _launch(args: str, title: str) -> dict:
    """Open the visible window running the supervisor (which keeps runner.py alive); tee its output to a console log."""
    (RUN_ROOT / "logs").mkdir(parents=True, exist_ok=True); STATE_DIR.mkdir(parents=True, exist_ok=True)
    console = RUN_ROOT / "logs" / "v2_console.log"
    sup_args = args.replace("-u runner.py ", "", 1)
    cmd = f"$host.UI.RawUI.WindowTitle = '{title}'; & '{PY}' -u supervise.py {sup_args} 2>&1 | Tee-Object -FilePath '{console}' -Append"
    try:
        for mark in ("v2_stopped_by_user",):
            (STATE_DIR / mark).unlink(missing_ok=True)
        subprocess.Popen(["powershell.exe", "-NoExit", "-Command", cmd], cwd=str(STRAT_DIR), creationflags=subprocess.CREATE_NEW_CONSOLE)
        return {"ok": True, "action": "started", "command": sup_args, "title": title}
    except Exception as e:                              # noqa: BLE001
        return {"ok": False, "error": str(e)}


# ── watchdog: if the state says armed, the user did not stop it, and nothing has heartbeat for 2 minutes, relaunch ──
import threading


def _watchdog():
    while True:
        time.sleep(60)
        try:
            st = _runner_status(); state = st["state"]; launch = _read_json(STATE_DIR / "v2_launch.json")
            stopped = (STATE_DIR / "v2_stopped_by_user").exists()
            if state.get("armed") and launch and not stopped and not st["running"] and (st["heartbeat_age_s"] is None or st["heartbeat_age_s"] > 120):
                sup_alive = _pid_alive(int((STATE_DIR / "v2_supervisor.pid").read_text()) if (STATE_DIR / "v2_supervisor.pid").exists() else None)
                if not sup_alive:
                    logger.warning("watchdog: runner and supervisor are down while armed -> relaunching")
                    _launch("-u runner.py " + " ".join(launch["args"]), "V2 RUNNER (watchdog relaunch)")
        except Exception as e:                          # noqa: BLE001
            logger.error("watchdog: %s", e)


threading.Thread(target=_watchdog, daemon=True, name="v2-watchdog").start()


@router.post("/strategy/disarm")
def strategy_disarm(req: ControlReq):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "v2_control.json").write_text(json.dumps({"cmd": "flatten" if req.flatten else "disarm", "ts": time.time()}), encoding="utf-8")
    return {"ok": True, "cmd": "flatten" if req.flatten else "disarm", "running": _runner_status()["running"]}


@router.post("/strategy/flatten")
def strategy_flatten(network: str = Query("mainnet"), proxy_index: int = Query(DEFAULT_PROXY)):
    """Flatten SOL now.  If the runner is alive it does it (and disarms); otherwise the backend closes the position directly."""
    st = _runner_status()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if st["running"]:
        (STATE_DIR / "v2_control.json").write_text(json.dumps({"cmd": "flatten", "ts": time.time()}), encoding="utf-8")
        return {"ok": True, "via": "runner"}
    try:
        res = _hl(network, proxy_index).close_position("SOL")
        return {"ok": res.get("ok"), "via": "backend", "fill": {k: res.get(k) for k in ("filled_sz", "avg_px", "error")}}
    except Exception as e:                              # noqa: BLE001
        return _err(e, proxy_index)


@router.post("/strategy/stop")
def strategy_stop():
    """Ask the runner process to exit (position, if any, is left open and reported)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "v2_control.json").write_text(json.dumps({"cmd": "stop", "ts": time.time()}), encoding="utf-8")
    return {"ok": True, "cmd": "stop"}
