"""
BTC Play API  (/api/play)
=========================
Arm / status / stop of the price-trigger runner (sol-perp/play/runner.py): on a touch of a level from above it puts all
available capital into a BTC long at that level's leverage (cross).  The runner is a visible PowerShell window; commands
go through control.json; files live in %USERPROFILE%/playrunner.
"""
from __future__ import annotations
import json, logging, subprocess, sys, time
from pathlib import Path
from fastapi import APIRouter, Query
from pydantic import BaseModel

_SOLPERP = Path(__file__).resolve().parent.parent.parent.parent / "sol-perp"
if str(_SOLPERP) not in sys.path:
    sys.path.insert(0, str(_SOLPERP))
from hlcore import HL                                                                   # noqa: E402
from play.paths import STATE_DIR, LOG_DIR, HB_F, CTRL_F, STATE_F, PID_F, LAUNCH_F, FILLS_F, ensure_dirs, jload, jdump   # noqa: E402

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/play", tags=["play"])
PY = sys.executable
_CLIENTS: dict = {}
MAINT_BTC = 0.0125


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


class Trigger(BaseModel):
    level: float
    leverage: int


class ArmReq(BaseModel):
    network: str = "mainnet"; proxy_index: int = 1; dry_run: bool = False
    triggers: list[Trigger] = [Trigger(level=84292, leverage=7), Trigger(level=83200, leverage=15)]


def _status() -> dict:
    ensure_dirs(); hb = jload(HB_F, {}) or {}; pid = None
    try:
        pid = int(PID_F.read_text(encoding="utf-8").strip()) if PID_F.exists() else None
    except Exception:                                   # noqa: BLE001
        pid = None
    alive = _pid_alive(pid); age = round(time.time() - hb["ts"]) if hb.get("ts") else None
    fills = []
    if FILLS_F.exists():
        fills = [json.loads(l) for l in FILLS_F.read_text(encoding="utf-8").splitlines() if l.strip()][-20:]
    return {"running": alive, "pid": pid, "heartbeat_age_s": age, "stale": alive and (age is None or age > 15), "heartbeat": hb,
            "state": jload(STATE_F, {}) or {}, "launch": jload(LAUNCH_F, None), "fills": fills, "log": str(LOG_DIR / "runner.log")}


@router.get("/status")
def status():
    return _status()


@router.get("/price")
def price(network: str = Query("mainnet"), proxy_index: int = Query(1)):
    """Current BTC mid + account availability, for sizing the confirm dialog before arming."""
    hl = _hl(network, proxy_index)
    try:
        px = float(hl.info({"type": "allMids"})["BTC"]); acct = hl.account()
        pos = next((p for p in acct.get("positions", []) if p["coin"] == "BTC"), None)
        return {"ok": True, "price": px, "available": acct.get("available_to_trade"), "account_value": acct.get("account_value"),
                "btc_position": pos, "maint_btc": MAINT_BTC}
    except Exception as e:                              # noqa: BLE001
        return {"ok": False, "error": str(e)}


@router.post("/arm")
def arm(req: ArmReq):
    st = _status()
    if st["running"]:
        return {"ok": False, "error": f"the play is already running (pid {st['pid']}); stop it first"}
    if not req.triggers:
        return {"ok": False, "error": "no triggers"}
    for t in req.triggers:
        if not (1 <= t.leverage <= 40) or t.level <= 0:
            return {"ok": False, "error": f"bad trigger {t}"}
    ensure_dirs(); CTRL_F.unlink(missing_ok=True)
    tf = STATE_DIR / "triggers.json"; jdump(tf, [{"level": t.level, "leverage": t.leverage} for t in req.triggers])
    args = f"--network {req.network} --proxy {req.proxy_index} --triggers-file '{tf}'{' --dry-run' if req.dry_run else ''}"
    console = LOG_DIR / "runner_console.log"
    title = f"BTC PLAY {req.network}{' DRY RUN' if req.dry_run else ''}"
    cmd = f"$host.UI.RawUI.WindowTitle = '{title}'; $env:PYTHONIOENCODING='utf-8'; & '{PY}' -u -m play.runner {args} 2>&1 | Tee-Object -FilePath '{console}' -Append"
    try:
        subprocess.Popen(["powershell.exe", "-NoExit", "-Command", cmd], cwd=str(_SOLPERP), creationflags=subprocess.CREATE_NEW_CONSOLE)
        return {"ok": True, "action": "started", "command": args, "title": title}
    except Exception as e:                              # noqa: BLE001
        return {"ok": False, "error": str(e)}


@router.post("/stop")
def stop():
    """Stop the runner (triggers disarmed).  Open positions are NOT touched."""
    ensure_dirs(); jdump(CTRL_F, {"cmd": "stop", "ts": time.time()})
    return {"ok": True, "cmd": "stop"}
