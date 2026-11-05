"""
Ranker dashboard API  (/api/ranker)
===================================
Status / control of the 10-slot alt long/short runner (sol-perp/ranker), its collector and model.  Same mechanics as
/api/hl/strategy: the runner is a supervised process in a visible window; commands go through control.json.
"""
from __future__ import annotations
import json, logging, subprocess, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from fastapi import APIRouter, Query
from pydantic import BaseModel

_SOLPERP = Path(__file__).resolve().parent.parent.parent.parent / "sol-perp"
if str(_SOLPERP) not in sys.path:
    sys.path.insert(0, str(_SOLPERP))
from hlcore import HL                                                                   # noqa: E402
from ranker.paths import (STATE_DIR, LOG_DIR, STATE_F, HB_F, CTRL_F, TRADES_F, FILLS_F, SUP_PID_F, LAUNCH_F, STOP_MARK,   # noqa: E402
                          CONFIG_F, COLLECTOR_HB_F, CURRENT_MODEL_F, ensure_dirs, jload, jdump)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ranker", tags=["ranker"])
PY = sys.executable
_CLIENTS: dict = {}


def _hl(network: str, proxy_index: int) -> HL:
    key = ("testnet" if network == "testnet" else "mainnet", proxy_index)
    if key not in _CLIENTS:
        _CLIENTS[key] = HL(*key)
    return _CLIENTS[key]


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"], capture_output=True, text=True, timeout=10).stdout
        return str(pid) in out
    except Exception:                                   # noqa: BLE001
        return False


def _jsonl(p: Path, last: int | None = None) -> list[dict]:
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[-last:] if last else rows


def _collector_status(chb: dict) -> dict:
    """Liveness from the collector's pid file (written at start), heartbeat for data freshness; a live pid with a
    stale/absent heartbeat means the first cycle is still running."""
    pid_f = STATE_DIR / "collector.pid"
    pid = None
    try:
        pid = int(pid_f.read_text(encoding="utf-8").strip()) if pid_f.exists() else None
    except Exception:                                   # noqa: BLE001
        pid = None
    proc_alive = _pid_alive(pid) if pid else False
    age = round(time.time() - chb["ts"]) if chb.get("ts") else None
    hb_fresh = age is not None and age < 900 and chb.get("pid") == pid
    return {**chb, "pid": pid or chb.get("pid"), "age_s": age, "alive": proc_alive,
            "starting": proc_alive and not hb_fresh, "stale": proc_alive and hb_fresh and age > 600}


def _status() -> dict:
    hb = jload(HB_F, {}) or {}; st = jload(STATE_F, {}) or {}
    pid = hb.get("pid"); age = (time.time() - hb["ts"]) if hb.get("ts") else None
    alive = _pid_alive(pid) and age is not None and age < 60
    sup_pid = int(SUP_PID_F.read_text()) if SUP_PID_F.exists() else None
    chb = jload(COLLECTOR_HB_F, {}) or {}
    if not chb:
        try:
            chb = json.loads(COLLECTOR_HB_F.read_text(encoding="utf-8"))
        except Exception as e:                          # noqa: BLE001
            chb = {"read_error": f"{type(e).__name__}: {e}", "path": str(COLLECTOR_HB_F)}
    return {"installed": (_SOLPERP / "ranker" / "runner.py").exists(), "running": alive, "pid": pid, "heartbeat_age_s": round(age, 1) if age is not None else None,
            "supervisor_pid": sup_pid, "supervisor_alive": _pid_alive(sup_pid), "stopped_by_user": STOP_MARK.exists(), "launch": jload(LAUNCH_F, None),
            "heartbeat": hb, "state": {k: v for k, v in st.items() if k != "book"}, "recent_trades": [t for t in _jsonl(TRADES_F, 40) if "pnl_usd" in t],
            "collector": _collector_status(chb),
            "model": jload(CURRENT_MODEL_F, None), "config": jload(CONFIG_F, {"exclude_coins": []}), "fetched_at": datetime.now(timezone.utc).isoformat()}


class ArmReq(BaseModel):
    network: str = "mainnet"; proxy_index: int = 1; margin: float = 300.0; leverage: int = 2; slots: int = 10; dry_run: bool = False
    cross: bool = True      # cross margin: isolated 5-6x on alts liquidated a slot a day (Oct 6 2026)

class ConfigReq(BaseModel):
    exclude_coins: list[str] = []


def _launch(args: str, title: str) -> dict:
    ensure_dirs()
    console = LOG_DIR / "runner_console.log"
    cmd = f"$host.UI.RawUI.WindowTitle = '{title}'; $env:PYTHONIOENCODING='utf-8'; & '{PY}' -u -m ranker.supervise {args} 2>&1 | Tee-Object -FilePath '{console}' -Append"
    try:
        STOP_MARK.unlink(missing_ok=True)
        subprocess.Popen(["powershell.exe", "-NoExit", "-Command", cmd], cwd=str(_SOLPERP), creationflags=subprocess.CREATE_NEW_CONSOLE)
        return {"ok": True, "action": "started", "command": args, "title": title}
    except Exception as e:                              # noqa: BLE001
        return {"ok": False, "error": str(e)}


@router.get("/status")
def status():
    return _status()


@router.get("/debug")
def debug(path: str | None = Query(None)):
    import os
    p = Path(path) if path else STATE_DIR
    try:
        listing = os.listdir(p)
    except Exception as e:                              # noqa: BLE001
        listing = f"{type(e).__name__}: {e}"
    return {"path": str(p), "exists": p.exists(), "listing": listing, "state_dir": str(STATE_DIR), "hb_exists": COLLECTOR_HB_F.exists(),
            "cwd": os.getcwd(), "localappdata": os.environ.get("LOCALAPPDATA"), "user": os.environ.get("USERNAME"), "pid": os.getpid()}


@router.post("/arm")
def arm(req: ArmReq):
    st = _status(); ensure_dirs()
    if st["running"]:
        jdump(CTRL_F, {"cmd": "arm", "ts": time.time()}); return {"ok": True, "action": "re-armed running process", "pid": st["pid"]}
    if not CURRENT_MODEL_F.exists():
        return {"ok": False, "error": "no trained model (models/current.json) — click Retrain model first"}
    args = f"--network {req.network} --proxy {req.proxy_index} --margin {req.margin} --leverage {req.leverage} --slots {req.slots}{' --cross' if req.cross else ''}{' --dry-run' if req.dry_run else ''}"
    return _launch(args, f"RANKER RUNNER {req.network} proxy{req.proxy_index + 1}{' cross' if req.cross else ' isolated'}{' DRY RUN' if req.dry_run else ''}")


@router.post("/disarm")
def disarm():
    ensure_dirs(); jdump(CTRL_F, {"cmd": "disarm", "ts": time.time()}); return {"ok": True, "cmd": "disarm"}


@router.post("/flatten")
def flatten(network: str = Query("mainnet"), proxy_index: int = Query(1)):
    """Close every ranker position.  Runner does it (and disarms) if alive; otherwise the backend closes the book's coins."""
    st = _status(); ensure_dirs()
    if st["running"]:
        jdump(CTRL_F, {"cmd": "flatten", "ts": time.time()}); return {"ok": True, "via": "runner"}
    book = (jload(STATE_F, {}) or {}).get("book", {})
    hl = _hl(network, proxy_index); out = {}
    for coin in book:
        try:
            r = hl.close_position(coin); out[coin] = r.get("error") or f"closed {r.get('filled_sz')} @ {r.get('avg_px')}"
        except Exception as e:                          # noqa: BLE001
            out[coin] = str(e)
    return {"ok": True, "via": "backend", "results": out}


@router.post("/stop")
def stop():
    ensure_dirs(); jdump(CTRL_F, {"cmd": "stop", "ts": time.time()}); return {"ok": True, "cmd": "stop"}


@router.get("/config")
def get_config():
    return jload(CONFIG_F, {"exclude_coins": []})


@router.post("/config")
def set_config(req: ConfigReq):
    ensure_dirs(); jdump(CONFIG_F, {"exclude_coins": [c.strip().upper() for c in req.exclude_coins if c.strip()]}); return get_config()


@router.post("/retrain")
def retrain():
    """Train a new model in a separate visible window; the runner reloads it at its next cycle."""
    console = LOG_DIR / "train_console.log"
    cmd = f"$host.UI.RawUI.WindowTitle = 'RANKER TRAIN'; $env:PYTHONIOENCODING='utf-8'; & '{PY}' -u -m ranker.model train 2>&1 | Tee-Object -FilePath '{console}' -Append"
    try:
        subprocess.Popen(["powershell.exe", "-NoExit", "-Command", cmd], cwd=str(_SOLPERP), creationflags=subprocess.CREATE_NEW_CONSOLE)
        return {"ok": True, "action": "training started in its own window (10-20 min)"}
    except Exception as e:                              # noqa: BLE001
        return {"ok": False, "error": str(e)}


@router.post("/collector/start")
def collector_start():
    """Start the data collector in its own visible window (independent of the runner)."""
    st = _status()
    if st["collector"].get("alive") and _pid_alive(st["collector"].get("pid")):
        return {"ok": True, "action": "already running", "pid": st["collector"].get("pid")}
    ensure_dirs(); (STATE_DIR / "collector_stop").unlink(missing_ok=True)
    console = LOG_DIR / "collector_console.log"
    cmd = f"$host.UI.RawUI.WindowTitle = 'RANKER COLLECTOR (Binance + HL via proxies 3+)'; $env:PYTHONIOENCODING='utf-8'; & '{PY}' -u -m ranker.collector 2>&1 | Tee-Object -FilePath '{console}' -Append"
    try:
        subprocess.Popen(["powershell.exe", "-NoExit", "-Command", cmd], cwd=str(_SOLPERP), creationflags=subprocess.CREATE_NEW_CONSOLE)
        return {"ok": True, "action": "collector started in its own window"}
    except Exception as e:                              # noqa: BLE001
        return {"ok": False, "error": str(e)}


@router.post("/collector/stop")
def collector_stop():
    ensure_dirs(); (STATE_DIR / "collector_stop").write_text(str(time.time()), encoding="utf-8")
    return {"ok": True, "action": "collector will exit after its current cycle"}


@router.get("/trades")
def trades(last: int = Query(500)):
    return {"trades": [t for t in _jsonl(TRADES_F, None) if "pnl_usd" in t][-last:], "events": [t for t in _jsonl(TRADES_F, None) if "event" in t][-50:]}


@router.get("/stats")
def stats():
    T = pd.DataFrame([t for t in _jsonl(TRADES_F, None) if "pnl_usd" in t])
    F = pd.DataFrame(_jsonl(FILLS_F, None))
    out = {"windows": {}, "equity": [], "fills": {}}
    if not T.empty:
        T["exit_dt"] = pd.to_datetime(T["exit_t"], unit="s", utc=True)
        now = pd.Timestamp.now(tz="UTC")
        for name, days in (("today", None), ("7d", 7), ("30d", 30), ("all", "all")):
            d = T if days == "all" else (T[T.exit_dt >= now.normalize()] if days is None else T[T.exit_dt >= now - pd.Timedelta(days=days)])
            if d.empty:
                continue
            out["windows"][name] = {"trades": int(len(d)), "pnl_usd": float(d.pnl_usd.sum()), "hit": float((d.pnl_usd > 0).mean()), "mean_bps": float(d.bps.mean()),
                                    "mean_pnl": float(d.pnl_usd.mean()), "liquidations": int(d.liquidated.sum()) if "liquidated" in d else 0,
                                    "long_pnl": float(d[d.side > 0].pnl_usd.sum()), "short_pnl": float(d[d.side < 0].pnl_usd.sum()),
                                    "mean_hours": float(d.hours.mean()) if "hours" in d else None}
        daily = T.set_index("exit_dt").pnl_usd.resample("1D").sum()
        out["equity"] = [{"date": str(k.date()), "pnl": float(v), "cum": float(c)} for (k, v), c in zip(daily.items(), daily.cumsum())]
    if not F.empty:
        out["fills"] = {"n": int(len(F)), "maker_share": float((F.kind == "maker").mean()), "mean_cost_bps": float(F.cost_bps.mean()),
                        "maker_cost_bps": float(F[F.kind == "maker"].cost_bps.mean()) if (F.kind == "maker").any() else None,
                        "taker_cost_bps": float(F[F.kind == "taker"].cost_bps.mean()) if (F.kind == "taker").any() else None,
                        "mean_wait_s": float(F.wait_s.mean())}
    return out


def _watchdog():
    while True:
        time.sleep(60)
        try:
            st = _status(); launch = jload(LAUNCH_F, None); state = jload(STATE_F, {}) or {}
            if state.get("armed") and launch and not STOP_MARK.exists() and not st["running"] and (st["heartbeat_age_s"] is None or st["heartbeat_age_s"] > 120):
                if not st["supervisor_alive"]:
                    logger.warning("ranker watchdog: runner and supervisor down while armed -> relaunching")
                    _launch(" ".join(launch["args"]), "RANKER RUNNER (watchdog relaunch)")
        except Exception as e:                          # noqa: BLE001
            logger.error("ranker watchdog: %s", e)


threading.Thread(target=_watchdog, daemon=True, name="ranker-watchdog").start()
