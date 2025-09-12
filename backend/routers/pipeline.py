"""
Pipeline management routes

Provides minimal endpoints to list and execute Windows batch scripts that
trigger the user's data-pipeline Python scripts inside a specific conda env.

This API intentionally shells out to .bat files so that environment activation
occurs exactly as the user expects in their Windows setup.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List, Dict
import subprocess
import shlex
import os
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import get_db

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


class PipelineScript(BaseModel):
    key: str
    title: str
    description: str
    bat_path: str


# Central registry of pipeline scripts and their launcher .bat files
PIPELINE_SCRIPTS: Dict[str, PipelineScript] = {
    # Primary full pipelines (Big 3 buttons)
    "spy_full": PipelineScript(
        key="spy_full",
        title="Update SPY Fronttest Tables",
        description="Runs 25_run_spy_full_pipeline.py to refresh all SPY backtest/fronttest tables",
        bat_path=r"C:\Users\sham\Documents\agentic trading system\mcp\run_pipeline_spy_full.bat",
    ),
    "es_full": PipelineScript(
        key="es_full",
        title="Update ES Fronttest Tables",
        description="Runs 42_run_es_full_pipeline.py to refresh all ES tables",
        bat_path=r"C:\Users\sham\Documents\agentic trading system\mcp\run_pipeline_es_full.bat",
    ),
    "eurusd_full": PipelineScript(
        key="eurusd_full",
        title="Update EURUSD Fronttest Tables",
        description="Runs 43_run_eurusd_full_pipeline.py to refresh all EURUSD tables",
        bat_path=r"C:\Users\sham\Documents\agentic trading system\mcp\run_pipeline_eurusd_full.bat",
    ),
}


@router.get("/scripts", response_model=List[PipelineScript], summary="List available pipeline scripts")
async def list_scripts():
    return list(PIPELINE_SCRIPTS.values())


class RunResult(BaseModel):
    started: bool
    key: str
    pid: int | None
    message: str


@router.post("/run/{key}", response_model=RunResult, summary="Run a pipeline script via .bat")
async def run_script(key: str):
    if key not in PIPELINE_SCRIPTS:
        raise HTTPException(status_code=404, detail=f"Unknown script: {key}")

    bat_path = PIPELINE_SCRIPTS[key].bat_path
    if not os.path.exists(bat_path):
        raise HTTPException(status_code=404, detail=f"Launcher not found: {bat_path}")

    try:
        # Use start to spawn detached. Creation of a new window for the batch.
        # We keep it simple: run without waiting; return the PID if available.
        process = subprocess.Popen([
            "cmd.exe", "/c", bat_path
        ], creationflags=subprocess.CREATE_NEW_CONSOLE)

        return RunResult(
            started=True,
            key=key,
            pid=process.pid,
            message="Launcher started in a new console"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Generic single-script runner using a shared .bat that accepts the filename
GENERIC_LAUNCHER = r"C:\Users\sham\Documents\agentic trading system\mcp\run_pipeline_script.bat"


class RunSingleRequest(BaseModel):
    scriptName: str


@router.post("/run-script", response_model=RunResult, summary="Run a specific data-pipeline *.py by name")
async def run_single_script(body: RunSingleRequest):
    if not os.path.exists(GENERIC_LAUNCHER):
        raise HTTPException(status_code=404, detail=f"Launcher not found: {GENERIC_LAUNCHER}")

    script_name = body.scriptName
    if not script_name.endswith(".py"):
        script_name = f"{script_name}.py"

    try:
        process = subprocess.Popen([
            "cmd.exe", "/c", GENERIC_LAUNCHER, script_name
        ], creationflags=subprocess.CREATE_NEW_CONSOLE)

        return RunResult(
            started=True,
            key=script_name,
            pid=process.pid,
            message="Script launcher started in a new console"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TableData(BaseModel):
    table_name: str
    timeframe: str
    rows: List[Dict]

class RecentResponse(BaseModel):
    tables: List[TableData]
    total_tables: int
    total_rows: int


def get_recent_data_for_symbol(symbol: str, limit: int, db: Session):
    """Helper function to get recent data for any symbol"""
    # Get all tables for the symbol
    tables_query = text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'fronttest' 
        AND table_name LIKE :symbol_pattern
        ORDER BY table_name
    """)
    
    result = db.execute(tables_query, {"symbol_pattern": f"{symbol}_%"})
    table_names = [row[0] for row in result.fetchall()]
    
    if not table_names:
        return RecentResponse(tables=[], total_tables=0, total_rows=0)
    
    tables_data = []
    total_rows = 0
    
    for table_name in table_names:
        try:
            # Extract timeframe from table name (e.g., spy_1d -> 1d)
            timeframe = table_name.replace(f'{symbol}_', '')
            
            # Get recent rows from this table
            recent_query = text(f"""
                SELECT timestamp, open, high, low, close, volume
                FROM fronttest."{table_name}"
                ORDER BY timestamp DESC
                LIMIT :limit
            """)
            
            result = db.execute(recent_query, {"limit": limit})
            rows = []
            
            for row in result.fetchall():
                rows.append({
                    "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                    "open": float(row.open) if row.open is not None else None,
                    "high": float(row.high) if row.high is not None else None,
                    "low": float(row.low) if row.low is not None else None,
                    "close": float(row.close) if row.close is not None else None,
                    "volume": float(row.volume) if row.volume is not None else None
                })
            
            if rows:  # Only include tables that have data
                tables_data.append(TableData(
                    table_name=table_name,
                    timeframe=timeframe,
                    rows=rows
                ))
                total_rows += len(rows)
                
        except Exception as e:
            # Skip tables that can't be queried (permissions, doesn't exist, etc.)
            continue
    
    return RecentResponse(
        tables=tables_data,
        total_tables=len(tables_data),
        total_rows=total_rows
    )


@router.get("/spy-recent", response_model=RecentResponse, summary="Get recent SPY fronttest data from all tables")
async def get_spy_recent_data(limit: int = Query(default=7, ge=1, le=50), db: Session = Depends(get_db)):
    """
    Get the most recent rows from all SPY fronttest tables.
    
    - **limit**: Number of recent rows to fetch per table (default: 7, max: 50)
    """
    try:
        return get_recent_data_for_symbol("spy", limit, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching SPY recent data: {str(e)}")


@router.get("/es-recent", response_model=RecentResponse, summary="Get recent ES fronttest data from all tables")
async def get_es_recent_data(limit: int = Query(default=7, ge=1, le=50), db: Session = Depends(get_db)):
    """
    Get the most recent rows from all ES fronttest tables.
    
    - **limit**: Number of recent rows to fetch per table (default: 7, max: 50)
    """
    try:
        return get_recent_data_for_symbol("es", limit, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching ES recent data: {str(e)}")


@router.get("/eurusd-recent", response_model=RecentResponse, summary="Get recent EURUSD fronttest data from all tables")
async def get_eurusd_recent_data(limit: int = Query(default=7, ge=1, le=50), db: Session = Depends(get_db)):
    """
    Get the most recent rows from all EURUSD fronttest tables.
    
    - **limit**: Number of recent rows to fetch per table (default: 7, max: 50)
    """
    try:
        return get_recent_data_for_symbol("eurusd", limit, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching EURUSD recent data: {str(e)}")


