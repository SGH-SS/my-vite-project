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
    # Vector generation launchers (per symbol)
    "spy_vectors": PipelineScript(
        key="spy_vectors",
        title="Generate SPY Vectors",
        description="Compute and backfill missing vector columns for SPY fronttest tables",
        bat_path=r"C:\Users\sham\Documents\agentic trading system\mcp\run_vectors_spy.bat",
    ),
    "es_vectors": PipelineScript(
        key="es_vectors",
        title="Generate ES Vectors",
        description="Compute and backfill missing vector columns for ES fronttest tables",
        bat_path=r"C:\Users\sham\Documents\agentic trading system\mcp\run_vectors_es.bat",
    ),
    "eurusd_vectors": PipelineScript(
        key="eurusd_vectors",
        title="Generate EURUSD Vectors",
        description="Compute and backfill missing vector columns for EURUSD fronttest tables",
        bat_path=r"C:\Users\sham\Documents\agentic trading system\mcp\run_vectors_eurusd.bat",
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


# ============ Freshness Status Endpoint ============
from datetime import datetime, timedelta
import pytz

class TableFreshness(BaseModel):
    table_name: str
    timeframe: str
    last_timestamp: str | None
    expected_timestamp: str | None
    is_up_to_date: bool
    staleness_periods: int  # How many periods behind
    staleness_message: str

class AssetFreshness(BaseModel):
    asset: str
    display_name: str
    timeframes: List[TableFreshness]
    overall_status: str  # "up_to_date", "partially_stale", "stale"

class FreshnessStatusResponse(BaseModel):
    timestamp: str
    assets: List[AssetFreshness]
    market_status: Dict[str, str]  # Current market status for each asset

def get_market_hours():
    """
    Define market hours for each asset type.
    All times are in Eastern Time (ET).
    """
    return {
        "spy": {
            "name": "SPY (US Equities)",
            "trading_days": [0, 1, 2, 3, 4],  # Monday=0, Friday=4
            "market_open": (9, 30),   # 9:30 AM ET
            "market_close": (16, 0),  # 4:00 PM ET
            "type": "equity"
        },
        "es": {
            "name": "ES (E-mini S&P 500 Futures)",
            "trading_days": [0, 1, 2, 3, 4],  # Mon-Fri (starts Sun evening)
            "market_open": (18, 0),   # 6:00 PM ET Sunday
            "market_close": (17, 0),  # 5:00 PM ET Friday
            "daily_break_start": (17, 0),  # 5:00 PM ET
            "daily_break_end": (18, 0),    # 6:00 PM ET
            "type": "futures"
        },
        "eurusd": {
            "name": "EUR/USD (Forex)",
            "trading_days": [0, 1, 2, 3, 4],  # Mon-Fri (starts Sun evening)
            "market_open": (17, 0),   # 5:00 PM ET Sunday
            "market_close": (17, 0),  # 5:00 PM ET Friday
            "type": "forex"
        }
    }

def get_timeframe_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes"""
    mapping = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440
    }
    return mapping.get(timeframe, 60)

def is_market_open(asset: str, dt: datetime, market_hours: dict) -> bool:
    """Check if market is open for a given asset at a specific datetime"""
    et = pytz.timezone('America/New_York')
    
    # Convert to Eastern Time if not already
    if dt.tzinfo is None:
        dt = et.localize(dt)
    else:
        dt = dt.astimezone(et)
    
    weekday = dt.weekday()
    hour = dt.hour
    minute = dt.minute
    current_minutes = hour * 60 + minute
    
    hours = market_hours.get(asset, {})
    
    if hours.get("type") == "equity":
        # SPY: Mon-Fri 9:30 AM - 4:00 PM ET
        if weekday not in hours["trading_days"]:
            return False
        open_minutes = hours["market_open"][0] * 60 + hours["market_open"][1]
        close_minutes = hours["market_close"][0] * 60 + hours["market_close"][1]
        return open_minutes <= current_minutes < close_minutes
    
    elif hours.get("type") == "futures":
        # ES: Sun 6pm - Fri 5pm with 1hr daily break
        # Weekend: Sat all day and Sun before 6pm
        if weekday == 5:  # Saturday
            return False
        if weekday == 6:  # Sunday
            open_minutes = hours["market_open"][0] * 60 + hours["market_open"][1]
            return current_minutes >= open_minutes
        # Mon-Fri: closed 5pm-6pm ET
        break_start = hours.get("daily_break_start", (17, 0))
        break_end = hours.get("daily_break_end", (18, 0))
        break_start_minutes = break_start[0] * 60 + break_start[1]
        break_end_minutes = break_end[0] * 60 + break_end[1]
        if weekday == 4:  # Friday
            close_minutes = hours["market_close"][0] * 60 + hours["market_close"][1]
            return current_minutes < close_minutes
        return not (break_start_minutes <= current_minutes < break_end_minutes)
    
    elif hours.get("type") == "forex":
        # EURUSD: 24/5 - Sun 5pm to Fri 5pm ET
        if weekday == 5:  # Saturday
            return False
        if weekday == 6:  # Sunday
            open_minutes = hours["market_open"][0] * 60 + hours["market_open"][1]
            return current_minutes >= open_minutes
        if weekday == 4:  # Friday
            close_minutes = hours["market_close"][0] * 60 + hours["market_close"][1]
            return current_minutes < close_minutes
        return True  # Mon-Thu: 24 hours
    
    return True

def get_expected_latest_timestamp(asset: str, timeframe: str, market_hours: dict) -> datetime:
    """
    Calculate what the expected latest timestamp should be for a given asset/timeframe.
    This considers market hours, weekends, and timeframe intervals.
    """
    et = pytz.timezone('America/New_York')
    now = datetime.now(et)
    
    tf_minutes = get_timeframe_minutes(timeframe)
    hours = market_hours.get(asset, {})
    
    # Start from current time and work backwards to find the last complete candle
    # A candle is complete when its close time has passed
    
    if hours.get("type") == "equity":
        # SPY: Regular trading hours only
        # Find the most recent trading session
        check_time = now
        
        # If we're on a weekend, go back to Friday
        while check_time.weekday() not in hours["trading_days"]:
            check_time = check_time - timedelta(days=1)
        
        # If we're before market open, go to previous trading day
        open_minutes = hours["market_open"][0] * 60 + hours["market_open"][1]
        close_minutes = hours["market_close"][0] * 60 + hours["market_close"][1]
        current_minutes = check_time.hour * 60 + check_time.minute
        
        if current_minutes < open_minutes:
            check_time = check_time - timedelta(days=1)
            while check_time.weekday() not in hours["trading_days"]:
                check_time = check_time - timedelta(days=1)
            # Set to market close of that day
            check_time = check_time.replace(hour=hours["market_close"][0], minute=hours["market_close"][1], second=0, microsecond=0)
        elif current_minutes >= close_minutes:
            # Set to market close of today
            check_time = check_time.replace(hour=hours["market_close"][0], minute=hours["market_close"][1], second=0, microsecond=0)
        
        # Now calculate the last complete candle
        if timeframe == "1d":
            # Daily candle completes at market close
            return check_time.replace(hour=hours["market_close"][0], minute=hours["market_close"][1], second=0, microsecond=0)
        else:
            # For intraday, find the last completed interval
            session_start = check_time.replace(hour=hours["market_open"][0], minute=hours["market_open"][1], second=0, microsecond=0)
            session_end = check_time.replace(hour=hours["market_close"][0], minute=hours["market_close"][1], second=0, microsecond=0)
            
            if check_time > session_end:
                check_time = session_end
            
            minutes_since_open = (check_time - session_start).total_seconds() / 60
            complete_periods = int(minutes_since_open // tf_minutes)
            
            if complete_periods < 1:
                # No complete candles yet today, go to previous day
                check_time = check_time - timedelta(days=1)
                while check_time.weekday() not in hours["trading_days"]:
                    check_time = check_time - timedelta(days=1)
                return check_time.replace(hour=hours["market_close"][0], minute=hours["market_close"][1], second=0, microsecond=0)
            
            last_candle_time = session_start + timedelta(minutes=complete_periods * tf_minutes)
            return last_candle_time
    
    elif hours.get("type") in ["futures", "forex"]:
        # ES and EURUSD: Nearly 24-hour trading
        # Find the last complete candle based on current time
        
        check_time = now
        
        # Handle weekend
        if check_time.weekday() == 5:  # Saturday
            # Go back to Friday 5pm
            check_time = check_time - timedelta(days=1)
            check_time = check_time.replace(hour=17, minute=0, second=0, microsecond=0)
        elif check_time.weekday() == 6:  # Sunday
            open_hour = hours["market_open"][0]
            if check_time.hour < open_hour:
                # Before Sunday open, go back to Friday 5pm
                check_time = check_time - timedelta(days=2)
                check_time = check_time.replace(hour=17, minute=0, second=0, microsecond=0)
        
        if timeframe == "1d":
            # Daily candles typically close at 5pm ET for futures/forex
            if check_time.hour < 17:
                check_time = check_time - timedelta(days=1)
            check_time = check_time.replace(hour=17, minute=0, second=0, microsecond=0)
            # Skip weekends for daily
            while check_time.weekday() >= 5:
                check_time = check_time - timedelta(days=1)
            return check_time
        else:
            # For intraday, align to the timeframe
            total_minutes = check_time.hour * 60 + check_time.minute
            complete_periods = total_minutes // tf_minutes
            last_candle_minutes = complete_periods * tf_minutes
            check_time = check_time.replace(hour=last_candle_minutes // 60, minute=last_candle_minutes % 60, second=0, microsecond=0)
            return check_time
    
    return now

def calculate_staleness(last_ts: datetime, expected_ts: datetime, timeframe: str) -> tuple:
    """Calculate how many periods behind the data is"""
    if last_ts is None:
        return (999, "No data available")
    
    tf_minutes = get_timeframe_minutes(timeframe)
    
    # Make both timestamps timezone-aware for comparison
    et = pytz.timezone('America/New_York')
    if last_ts.tzinfo is None:
        last_ts = et.localize(last_ts)
    if expected_ts.tzinfo is None:
        expected_ts = et.localize(expected_ts)
    
    diff = expected_ts - last_ts
    diff_minutes = diff.total_seconds() / 60
    
    if diff_minutes <= 0:
        return (0, "Up to date")
    
    periods_behind = int(diff_minutes / tf_minutes)
    
    if periods_behind == 0:
        return (0, "Up to date")
    elif periods_behind == 1:
        return (1, f"1 {timeframe} candle behind")
    else:
        return (periods_behind, f"{periods_behind} {timeframe} candles behind")

@router.get("/freshness-status", response_model=FreshnessStatusResponse, summary="Get data freshness status for all assets")
async def get_freshness_status(db: Session = Depends(get_db)):
    """
    Get the freshness status of all fronttest tables.
    
    Returns information about whether each asset/timeframe combination is up to date,
    considering market hours, weekends, and timeframe intervals.
    """
    try:
        et = pytz.timezone('America/New_York')
        now = datetime.now(et)
        market_hours = get_market_hours()
        
        assets = ["spy", "es", "eurusd"]
        timeframes = ["1d", "4h", "1h", "30m", "15m", "5m", "1m"]
        
        result_assets = []
        market_status = {}
        
        for asset in assets:
            asset_timeframes = []
            up_to_date_count = 0
            
            # Check current market status
            is_open = is_market_open(asset, now, market_hours)
            market_status[asset] = "Open" if is_open else "Closed"
            
            for tf in timeframes:
                table_name = f"{asset}_{tf}"
                
                # Get last timestamp from fronttest table
                try:
                    query = text(f"""
                        SELECT MAX(timestamp) as last_ts
                        FROM fronttest."{table_name}"
                    """)
                    result = db.execute(query).fetchone()
                    last_ts = result[0] if result and result[0] else None
                except Exception:
                    last_ts = None
                
                # Calculate expected timestamp
                expected_ts = get_expected_latest_timestamp(asset, tf, market_hours)
                
                # Calculate staleness
                staleness_periods, staleness_msg = calculate_staleness(last_ts, expected_ts, tf)
                
                # Determine if up to date (within tolerance of 1 period for data delays)
                is_up_to_date = staleness_periods <= 1
                
                if is_up_to_date:
                    up_to_date_count += 1
                
                asset_timeframes.append(TableFreshness(
                    table_name=table_name,
                    timeframe=tf,
                    last_timestamp=last_ts.isoformat() if last_ts else None,
                    expected_timestamp=expected_ts.isoformat() if expected_ts else None,
                    is_up_to_date=is_up_to_date,
                    staleness_periods=staleness_periods,
                    staleness_message=staleness_msg
                ))
            
            # Determine overall status
            if up_to_date_count == len(timeframes):
                overall_status = "up_to_date"
            elif up_to_date_count == 0:
                overall_status = "stale"
            else:
                overall_status = "partially_stale"
            
            result_assets.append(AssetFreshness(
                asset=asset,
                display_name=market_hours[asset]["name"],
                timeframes=asset_timeframes,
                overall_status=overall_status
            ))
        
        return FreshnessStatusResponse(
            timestamp=now.isoformat(),
            assets=result_assets,
            market_status=market_status
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting freshness status: {str(e)}")


