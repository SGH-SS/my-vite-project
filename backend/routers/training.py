"""
Model training routes

Provides endpoints to list and execute local training scripts (.py) that live
under the user's v2 daygent models folder. Scripts are launched via a Windows
batch file to ensure the correct Anaconda environment activation.
"""

from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Literal, Any
import subprocess
import os
import json
from datetime import datetime

router = APIRouter(prefix="/api/training", tags=["training"])


# Root folder where training scripts live
TRAINING_DIR = r"C:\Users\sham\Documents\agentic trading system\mcp\my-vite-project\src\components\v2 daygent models"

# V3 POC training directory
V3_TRAINING_DIR = r"C:\Users\sham\Documents\agentic trading system\mcp\my-vite-project\src\components\v3 models"
POC_RUNS_DIR = os.path.join(V3_TRAINING_DIR, "on_demand_runs")

# Generic launcher that opens Anaconda Prompt, activates env, and runs a script
GENERIC_TRAINING_LAUNCHER = r"C:\Users\sham\Documents\agentic trading system\mcp\run_training_script.bat"

# POC launcher for PowerShell
POC_TRAINING_LAUNCHER = os.path.join(V3_TRAINING_DIR, "run_poc_training.bat")


class TrainingScript(BaseModel):
    key: str
    title: str
    description: str
    script_name: str


def _discover_training_scripts() -> List[TrainingScript]:
    """Discover train_*.py files in TRAINING_DIR and build metadata."""
    results: List[TrainingScript] = []
    try:
        for name in os.listdir(TRAINING_DIR):
            if not name.endswith(".py"):
                continue
            if not name.startswith("train_"):
                continue
            title = name.replace("_", " ").replace(".py", "").title()
            key = name.replace(".py", "")
            desc = f"Run {name} in the configured conda environment"
            results.append(TrainingScript(key=key, title=title, description=desc, script_name=name))
    except Exception:
        # If the directory is missing, just return empty; the run endpoint will error explicitly
        pass
    return results


@router.get("/scripts", response_model=List[TrainingScript], summary="List available local training scripts")
async def list_training_scripts():
    return _discover_training_scripts()


class RunRequest(BaseModel):
    scriptName: str


class RunResult(BaseModel):
    started: bool
    key: str
    pid: int | None
    message: str


@router.post("/run", response_model=RunResult, summary="Run a training script via .bat in a new console")
async def run_training_script(body: RunRequest):
    if not os.path.exists(GENERIC_TRAINING_LAUNCHER):
        raise HTTPException(status_code=404, detail=f"Launcher not found: {GENERIC_TRAINING_LAUNCHER}")

    script_name = body.scriptName
    if not script_name.endswith(".py"):
        script_name = f"{script_name}.py"

    # Sanity: must reside in TRAINING_DIR and exist
    full_path = os.path.join(TRAINING_DIR, script_name)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"Script not found: {full_path}")

    try:
        process = subprocess.Popen([
            "cmd.exe", "/c", GENERIC_TRAINING_LAUNCHER, script_name
        ], creationflags=subprocess.CREATE_NEW_CONSOLE)

        return RunResult(
            started=True,
            key=script_name,
            pid=process.pid,
            message="Training script launcher started in a new console"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------
# Directory Debugger (server-side)
# -----------------------------------------

class DirNode(BaseModel):
    name: str
    type: Literal['dir', 'file']
    children: Optional[List["DirNode"]] = None


def _build_dir_tree(path: str, name: Optional[str] = None, max_depth: int = 3, _depth: int = 0) -> DirNode:
    node = {"name": name or os.path.basename(path) or path, "type": "dir", "children": []}
    if _depth >= max_depth:
        return DirNode(**node)
    try:
        entries = sorted(os.listdir(path))
        for entry in entries:
            full = os.path.join(path, entry)
            # Skip very large or irrelevant directories
            if entry.lower() in {"node_modules", "__pycache__", ".git"}:
                continue
            if os.path.isdir(full):
                node["children"].append(_build_dir_tree(full, entry, max_depth, _depth + 1).dict())
            else:
                node["children"].append({"name": entry, "type": "file"})
    except Exception as e:
        node["children"].append({"name": f"Error: {e}", "type": "file"})
    return DirNode(**node)


@router.get("/dir-tree", summary="Get directory tree for training folder", response_model=DirNode)
async def get_dir_tree(max_depth: int = Query(default=3, ge=1, le=8)):
    if not os.path.exists(TRAINING_DIR):
        raise HTTPException(status_code=404, detail=f"Training directory not found: {TRAINING_DIR}")
    return _build_dir_tree(TRAINING_DIR, name=os.path.basename(TRAINING_DIR), max_depth=max_depth)


# -----------------------------------------
# Latest metrics discovery (server-side)
# -----------------------------------------

class LatestMetric(BaseModel):
    script_name: str
    run_timestamp_utc: Optional[str]
    metrics: Optional[Dict[str, Any]]


def _artifact_base_for_script(script_name: str) -> Optional[str]:
    name = script_name.lower()
    if "1d_iso" in name:
        return os.path.join(TRAINING_DIR, "artifacts_lgbm_1d_iso")
    if "1d_v0" in name:
        return os.path.join(TRAINING_DIR, "artifacts_lgbm_1d_v0")
    return None


def _latest_run_id(runs_dir: str) -> Optional[str]:
    try:
        if not os.path.isdir(runs_dir):
            return None
        run_folders = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        # Expect folders like YYYYMMDD_HHMMSS
        run_folders = [d for d in run_folders if len(d) == 15 and d[8] == "_"]
        run_folders.sort()
        return run_folders[-1] if run_folders else None
    except Exception:
        return None


def _read_metrics(script_name: str, artifact_base: str, run_id: str) -> Optional[Dict[str, Any]]:
    try:
        is_iso = "1d_iso" in script_name.lower()
        results_file = "results_1d_iso.json" if is_iso else "lightgbm_1d_v0_metrics.json"
        full_path = os.path.join(artifact_base, "runs", run_id, results_file)
        if not os.path.exists(full_path):
            return None
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if is_iso:
            return data
        # Compact selection for v0 to keep payload small
        return {
            "theta_deploy": data.get("in_sample", {}).get("theta_deploy"),
            "best_iter_median": data.get("in_sample", {}).get("best_iter_median"),
            "oos_summary": data.get("oos", {}).get("summary"),
        }
    except Exception:
        return None


@router.get("/latest-metrics", response_model=List[LatestMetric], summary="Get latest metrics for discovered scripts")
async def get_latest_metrics():
    scripts = _discover_training_scripts()
    results: List[LatestMetric] = []
    for s in scripts:
        base = _artifact_base_for_script(s.script_name)
        if not base:
            # Not every train_* has artifacts – skip
            continue
        runs_dir = os.path.join(base, "runs")
        run_id = _latest_run_id(runs_dir)
        metrics = _read_metrics(s.script_name, base, run_id) if run_id else None
        results.append(LatestMetric(script_name=s.script_name, run_timestamp_utc=run_id, metrics=metrics))
    return results


# -----------------------------------------
# Latest + Best metrics
# -----------------------------------------

class LatestAndBest(BaseModel):
    script_name: str
    latest: Optional[LatestMetric] = None
    best: Optional[LatestMetric] = None
    best_score: Optional[float] = None
    criterion: Optional[str] = None


def _score_for_script(script_name: str, metrics: Dict[str, Any]) -> tuple[Optional[float], str]:
    """Return (score, criterion_label) used to choose the best run per script.

    Heuristics per script family:
      - 1d_iso: maximize test_accuracy (Test Acc), fallback validation_accuracy, then validation_auc
      - 1d_v0: maximize oos_summary.sharpe_daily_median, fallback ev_per_trade_median
    """
    name = script_name.lower()
    # Helper to safely get nested
    def g(obj, *keys):
        cur = obj
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    if "1d_iso" in name:
        # Prioritize test_accuracy (Test Acc) for ISO models
        for key in ("test_accuracy", "validation_accuracy", "validation_auc"):
            val = g(metrics, key)
            if isinstance(val, (int, float)):
                return float(val), key
        return None, "test_accuracy"

    if "1d_v0" in name:
        sharpe = g(metrics, "oos_summary", "sharpe_daily_median")
        if isinstance(sharpe, (int, float)):
            return float(sharpe), "oos_summary.sharpe_daily_median"
        ev = g(metrics, "oos_summary", "ev_per_trade_median")
        if isinstance(ev, (int, float)):
            return float(ev), "oos_summary.ev_per_trade_median"
        return None, "oos_summary.sharpe_daily_median"

    # Default: no scoring available
    return None, "unknown"


def _all_run_ids(runs_dir: str) -> List[str]:
    try:
        if not os.path.isdir(runs_dir):
            return []
        run_folders = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        run_folders = [d for d in run_folders if len(d) == 15 and d[8] == "_"]
        run_folders.sort()
        return run_folders
    except Exception:
        return []


@router.get("/latest-and-best-metrics", response_model=List[LatestAndBest], summary="Get latest and best metrics per script")
async def get_latest_and_best_metrics():
    scripts = _discover_training_scripts()
    out: List[LatestAndBest] = []
    for s in scripts:
        base = _artifact_base_for_script(s.script_name)
        if not base:
            continue
        runs_dir = os.path.join(base, "runs")
        run_ids = _all_run_ids(runs_dir)
        if not run_ids:
            out.append(LatestAndBest(script_name=s.script_name))
            continue

        # Latest
        latest_id = run_ids[-1]
        latest_metrics = _read_metrics(s.script_name, base, latest_id)
        latest = LatestMetric(script_name=s.script_name, run_timestamp_utc=latest_id, metrics=latest_metrics)

        # Best by heuristic score
        best_id: Optional[str] = None
        best_metrics: Optional[Dict[str, Any]] = None
        best_score: Optional[float] = None
        criterion = None
        for rid in run_ids:
            m = _read_metrics(s.script_name, base, rid)
            if not m:
                continue
            score, label = _score_for_script(s.script_name, m)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_id = rid
                best_metrics = m
                criterion = label

        best = None
        if best_id is not None:
            best = LatestMetric(script_name=s.script_name, run_timestamp_utc=best_id, metrics=best_metrics)

        out.append(LatestAndBest(
            script_name=s.script_name,
            latest=latest,
            best=best,
            best_score=best_score,
            criterion=criterion
        ))
    return out


# -----------------------------------------
# POC Training Endpoints
# -----------------------------------------

class PocTrainingRequest(BaseModel):
    candle_date: str  # ISO timestamp of selected candle
    train_years: float = 3.0
    test_window_size: int = 35
    params: Dict[str, float] = {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20
    }


class PocRunResult(BaseModel):
    started: bool
    run_id: str
    pid: int | None
    message: str
    output_dir: str


@router.post("/run-poc", response_model=PocRunResult, summary="Run POC training on selected candle")
async def run_poc_training(body: PocTrainingRequest):
    """
    Trigger POC training for a user-selected candle.
    Spawns a PowerShell window with the training script.
    """
    if not os.path.exists(POC_TRAINING_LAUNCHER):
        raise HTTPException(status_code=404, detail=f"POC launcher not found: {POC_TRAINING_LAUNCHER}")
    
    # Generate unique run ID
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(POC_RUNS_DIR, run_id)
    
    # Ensure on_demand_runs directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract params
    params = body.params or {}
    lr = params.get("learning_rate", 0.05)
    leaves = int(params.get("num_leaves", 31))
    depth = int(params.get("max_depth", 6))
    samples = int(params.get("min_child_samples", 20))
    
    try:
        # Spawn PowerShell with the launcher bat file
        # Arguments: candle_date, train_years, test_window_size, lr, leaves, depth, samples, output_dir
        process = subprocess.Popen(
            [
                "powershell.exe",
                "-NoExit",
                "-Command",
                f'cd "{V3_TRAINING_DIR}"; '
                f'python train_poc_simple.py '
                f'--selected_candle_date "{body.candle_date}" '
                f'--train_years {body.train_years} '
                f'--test_window_size {body.test_window_size} '
                f'--learning_rate {lr} '
                f'--num_leaves {leaves} '
                f'--max_depth {depth} '
                f'--min_child_samples {samples} '
                f'--output_dir "{output_dir}"'
            ],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        
        return PocRunResult(
            started=True,
            run_id=run_id,
            pid=process.pid,
            message=f"POC training started for candle {body.candle_date}",
            output_dir=output_dir
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start POC training: {str(e)}")


class PocRun(BaseModel):
    run_id: str
    timestamp: str
    candle_date: Optional[str] = None
    status: str  # "completed", "running", "failed"
    train_metrics: Optional[Dict[str, Any]] = None
    test_metrics: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    artifact_paths: Optional[Dict[str, str]] = None


def _parse_poc_run(run_dir: str) -> Optional[PocRun]:
    """Parse a POC run directory and extract metadata."""
    run_id = os.path.basename(run_dir)
    
    # Look for results and metadata files
    results_files = [f for f in os.listdir(run_dir) if f.startswith("results_poc_") and f.endswith(".json")]
    meta_files = [f for f in os.listdir(run_dir) if f.startswith("run_meta_") and f.endswith(".json")]
    
    if not results_files:
        # Training might still be running or failed
        return PocRun(
            run_id=run_id,
            timestamp=run_id,
            status="running",
        )
    
    # Load results
    results_path = os.path.join(run_dir, results_files[0])
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        meta_path = os.path.join(run_dir, meta_files[0]) if meta_files else None
        meta = {}
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        
        return PocRun(
            run_id=run_id,
            timestamp=results.get("run_timestamp_utc", run_id),
            candle_date=results.get("config", {}).get("selected_candle_date"),
            status=meta.get("status", "completed"),
            train_metrics=results.get("metrics", {}).get("train"),
            test_metrics=results.get("metrics", {}).get("test"),
            config=results.get("config"),
            artifact_paths=results.get("artifacts"),
        )
    except Exception as e:
        print(f"Error parsing POC run {run_id}: {e}")
        return PocRun(
            run_id=run_id,
            timestamp=run_id,
            status="failed",
        )


@router.get("/poc-runs", response_model=List[PocRun], summary="List all POC training runs")
async def list_poc_runs():
    """List all completed and running POC training runs."""
    if not os.path.exists(POC_RUNS_DIR):
        return []
    
    runs = []
    try:
        for entry in os.listdir(POC_RUNS_DIR):
            run_dir = os.path.join(POC_RUNS_DIR, entry)
            if os.path.isdir(run_dir):
                poc_run = _parse_poc_run(run_dir)
                if poc_run:
                    runs.append(poc_run)
    except Exception as e:
        print(f"Error listing POC runs: {e}")
    
    # Sort by timestamp descending (newest first)
    runs.sort(key=lambda r: r.timestamp, reverse=True)
    
    return runs


@router.get("/poc-runs/{run_id}", response_model=PocRun, summary="Get details for a specific POC run")
async def get_poc_run(run_id: str):
    """Get detailed information for a specific POC training run."""
    run_dir = os.path.join(POC_RUNS_DIR, run_id)
    
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail=f"POC run not found: {run_id}")
    
    poc_run = _parse_poc_run(run_dir)
    if not poc_run:
        raise HTTPException(status_code=500, detail=f"Failed to parse POC run: {run_id}")
    
    return poc_run


@router.get("/latest-poc-run", response_model=Optional[PocRun], summary="Get the most recent POC run")
async def get_latest_poc_run():
    """Get the most recent POC training run (for polling after training starts)."""
    runs = await list_poc_runs()
    return runs[0] if runs else None
