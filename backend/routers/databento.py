"""
DataBento API Router - ES Futures Outrights and Calendar Spreads

Provides endpoints for:
- Trading calendar dates with volume summary
- Instruments (outrights + spreads) for a selected date, sorted by volume
- Spread roll activity and liquidity data
"""

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text
from datetime import date, datetime
from typing import Optional, List
import logging

from database import engine

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/databento",
    tags=["databento"]
)


# =============================================================================
# Trading Calendar Endpoints
# =============================================================================

@router.get("/calendar/dates")
async def get_trading_dates(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100, description="Maximum number of dates to return")
):
    """
    Get list of trading dates with front-month volume.
    Returns dates in descending order (most recent first).
    """
    try:
        with engine.connect() as conn:
            query = """
                SELECT 
                    date_utc,
                    contract,
                    volume,
                    open,
                    high,
                    low,
                    close
                FROM databento_es_ohlcv_1d.es_continuous
                WHERE 1=1
            """
            params = {}
            
            if start_date:
                query += " AND date_utc >= :start_date"
                params["start_date"] = start_date
            if end_date:
                query += " AND date_utc <= :end_date"
                params["end_date"] = end_date
            
            query += " ORDER BY date_utc DESC LIMIT :limit"
            params["limit"] = limit
            
            result = conn.execute(text(query), params)
            rows = result.fetchall()
            
            dates = []
            for row in rows:
                dates.append({
                    "date": row[0].isoformat() if hasattr(row[0], 'isoformat') else str(row[0]),
                    "contract": row[1],
                    "volume": row[2],
                    "open": float(row[3]) if row[3] else None,
                    "high": float(row[4]) if row[4] else None,
                    "low": float(row[5]) if row[5] else None,
                    "close": float(row[6]) if row[6] else None,
                })
            
            return {
                "dates": dates,
                "count": len(dates),
                "query_params": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": limit
                }
            }
    except Exception as e:
        logger.error(f"Error fetching trading dates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calendar/date-range")
async def get_date_range():
    """
    Get the full date range available in the database.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT MIN(date_utc), MAX(date_utc), COUNT(*)
                FROM databento_es_ohlcv_1d.es_continuous
            """))
            row = result.fetchone()
            
            return {
                "earliest": row[0].isoformat() if row[0] else None,
                "latest": row[1].isoformat() if row[1] else None,
                "total_days": row[2]
            }
    except Exception as e:
        logger.error(f"Error fetching date range: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Instruments for a Specific Date (Enhanced with trade counts)
# =============================================================================

@router.get("/instruments/{target_date}")
async def get_instruments_for_date(target_date: str):
    """
    Get all tradeable instruments (outrights + spreads) for a specific date.
    Returns instruments sorted by volume in descending order.
    
    Enhanced with:
    - Trade counts for outrights from trades_agg_1m
    - Enhanced spread data (buy/sell volume, VWAP, avg trade size)
    - Daily summary statistics
    """
    try:
        with engine.connect() as conn:
            instruments = []
            
            # 1. Get trade counts for outrights from trades_agg_1m (if available)
            trade_counts = {}
            try:
                trades_query = """
                    SELECT 
                        symbol, 
                        SUM(trade_count) as total_trades,
                        SUM(total_volume) as total_volume,
                        SUM(trades_gt_100) as large_trades,
                        SUM(buy_volume) as buy_volume,
                        SUM(sell_volume) as sell_volume,
                        AVG(avg_trade_size) as avg_trade_size
                    FROM databento_es_trades.trades_agg_1m
                    WHERE date_utc = :date
                    GROUP BY symbol
                """
                trades_result = conn.execute(text(trades_query), {"date": target_date})
                for row in trades_result.fetchall():
                    trade_counts[row[0]] = {
                        "trade_count": row[1],
                        "volume": row[2],
                        "large_trades": row[3],
                        "buy_volume": row[4],
                        "sell_volume": row[5],
                        "avg_trade_size": float(row[6]) if row[6] else None
                    }
            except Exception as e:
                logger.debug(f"Trade counts not available for {target_date}: {e}")
            
            # 2. Get active outrights from metadata, then query each table
            metadata_query = """
                SELECT table_name, symbol_type, avg_daily_volume
                FROM databento_es_ohlcv_1d._metadata
                WHERE :date BETWEEN min_date AND max_date
                AND symbol_type = 'outright'
            """
            meta_result = conn.execute(text(metadata_query), {"date": target_date})
            active_outrights = meta_result.fetchall()
            
            # Query each active outright table
            for table_name, symbol_type, avg_daily_volume in active_outrights:
                try:
                    symbol_upper = table_name.upper()
                    outright_query = f"""
                        SELECT 
                            '{symbol_upper}' as symbol,
                            volume,
                            open,
                            high,
                            low,
                            close
                        FROM databento_es_ohlcv_1d.{table_name}
                        WHERE date_utc = :date AND volume > 0
                    """
                    outright_result = conn.execute(text(outright_query), {"date": target_date})
                    for row in outright_result.fetchall():
                        # Get trade data if available
                        trade_data = trade_counts.get(symbol_upper, {})
                        
                        # Calculate volume vs average
                        volume = row[1]
                        vol_vs_avg = None
                        if avg_daily_volume and float(avg_daily_volume) > 0:
                            vol_vs_avg = round((volume / float(avg_daily_volume)) * 100, 1)
                        
                        instruments.append({
                            "type": "outright",
                            "symbol": symbol_upper,
                            "volume": volume,
                            "open": float(row[2]) if row[2] else None,
                            "high": float(row[3]) if row[3] else None,
                            "low": float(row[4]) if row[4] else None,
                            "close": float(row[5]) if row[5] else None,
                            "daily_range": float(row[3] - row[4]) if row[3] and row[4] else None,
                            "trade_count": trade_data.get("trade_count"),
                            "large_trades": trade_data.get("large_trades"),
                            "buy_volume": trade_data.get("buy_volume"),
                            "sell_volume": trade_data.get("sell_volume"),
                            "avg_trade_size": trade_data.get("avg_trade_size"),
                            "vol_vs_avg_pct": vol_vs_avg
                        })
                except Exception as e:
                    logger.warning(f"Could not query outright table {table_name}: {e}")
            
            # 3. Get spreads from ohlcv_1d with full roll_activity data
            spread_query = """
                SELECT 
                    o.spread_symbol,
                    o.volume,
                    o.open,
                    o.high,
                    o.low,
                    o.close,
                    o.daily_range,
                    o.daily_change,
                    o.daily_change_pct,
                    r.trade_count,
                    r.large_trade_count,
                    r.buy_volume,
                    r.sell_volume,
                    r.vwap,
                    r.avg_trade_size,
                    r.net_roll_direction
                FROM databento_es_spreads.ohlcv_1d o
                LEFT JOIN databento_es_spreads.roll_activity_daily r
                    ON o.date_utc = r.date_utc AND o.spread_symbol = r.spread_symbol
                WHERE o.date_utc = :date AND o.volume > 0
            """
            spread_result = conn.execute(text(spread_query), {"date": target_date})
            for row in spread_result.fetchall():
                instruments.append({
                    "type": "spread",
                    "symbol": row[0],
                    "volume": row[1],
                    "open": float(row[2]) if row[2] else None,
                    "high": float(row[3]) if row[3] else None,
                    "low": float(row[4]) if row[4] else None,
                    "close": float(row[5]) if row[5] else None,
                    "daily_range": float(row[6]) if row[6] else None,
                    "daily_change": float(row[7]) if row[7] else None,
                    "daily_change_pct": float(row[8]) if row[8] else None,
                    "trade_count": row[9],
                    "large_trades": row[10],
                    "buy_volume": row[11],
                    "sell_volume": row[12],
                    "vwap": float(row[13]) if row[13] else None,
                    "avg_trade_size": float(row[14]) if row[14] else None,
                    "net_direction": row[15],
                    "vol_vs_avg_pct": None  # Not available for spreads in metadata
                })
            
            # Sort by volume descending
            instruments.sort(key=lambda x: x["volume"] or 0, reverse=True)
            
            # 4. Get front-month info for context
            front_month_query = """
                SELECT contract, volume, open, high, low, close
                FROM databento_es_ohlcv_1d.es_continuous
                WHERE date_utc = :date
            """
            front_result = conn.execute(text(front_month_query), {"date": target_date})
            front_row = front_result.fetchone()
            
            # 5. Calculate daily summary statistics
            total_outright_volume = sum(i["volume"] for i in instruments if i["type"] == "outright")
            total_spread_volume = sum(i["volume"] for i in instruments if i["type"] == "spread")
            total_trades = sum(i["trade_count"] or 0 for i in instruments if i["trade_count"])
            total_large_trades = sum(i["large_trades"] or 0 for i in instruments if i["large_trades"])
            
            return {
                "date": target_date,
                "instruments": instruments,
                "count": len(instruments),
                "front_month": {
                    "contract": front_row[0] if front_row else None,
                    "volume": front_row[1] if front_row else None,
                    "open": float(front_row[2]) if front_row and front_row[2] else None,
                    "high": float(front_row[3]) if front_row and front_row[3] else None,
                    "low": float(front_row[4]) if front_row and front_row[4] else None,
                    "close": float(front_row[5]) if front_row and front_row[5] else None,
                    "daily_range": float(front_row[3] - front_row[4]) if front_row and front_row[3] and front_row[4] else None
                } if front_row else None,
                "summary": {
                    "outright_count": sum(1 for i in instruments if i["type"] == "outright"),
                    "spread_count": sum(1 for i in instruments if i["type"] == "spread"),
                    "total_outright_volume": total_outright_volume,
                    "total_spread_volume": total_spread_volume,
                    "total_trades": total_trades,
                    "total_large_trades": total_large_trades,
                    "has_trade_data": len(trade_counts) > 0
                }
            }
    except Exception as e:
        logger.error(f"Error fetching instruments for {target_date}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Spread-Specific Endpoints
# =============================================================================

@router.get("/spreads/instruments")
async def get_spread_instruments():
    """
    Get list of all spread instruments in the database.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    spread_symbol,
                    front_month,
                    back_month,
                    months_apart,
                    first_seen,
                    last_seen
                FROM databento_es_spreads.spread_instruments
                ORDER BY spread_symbol
            """))
            rows = result.fetchall()
            
            instruments = []
            for row in rows:
                instruments.append({
                    "symbol": row[0],
                    "front_month": row[1],
                    "back_month": row[2],
                    "quarters_apart": row[3],
                    "first_seen": row[4].isoformat() if row[4] else None,
                    "last_seen": row[5].isoformat() if row[5] else None
                })
            
            return {
                "instruments": instruments,
                "count": len(instruments)
            }
    except Exception as e:
        logger.error(f"Error fetching spread instruments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spreads/roll-activity")
async def get_roll_activity(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    spread_symbol: Optional[str] = Query(None),
    limit: int = Query(100)
):
    """
    Get detailed roll activity data for spreads.
    """
    try:
        with engine.connect() as conn:
            query = """
                SELECT 
                    date_utc,
                    spread_symbol,
                    front_month,
                    back_month,
                    total_volume,
                    buy_volume,
                    sell_volume,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    vwap,
                    trade_count,
                    large_trade_count,
                    avg_trade_size,
                    net_roll_direction
                FROM databento_es_spreads.roll_activity_daily
                WHERE 1=1
            """
            params = {}
            
            if start_date:
                query += " AND date_utc >= :start_date"
                params["start_date"] = start_date
            if end_date:
                query += " AND date_utc <= :end_date"
                params["end_date"] = end_date
            if spread_symbol:
                query += " AND spread_symbol = :spread_symbol"
                params["spread_symbol"] = spread_symbol
            
            query += " ORDER BY date_utc DESC, total_volume DESC LIMIT :limit"
            params["limit"] = limit
            
            result = conn.execute(text(query), params)
            rows = result.fetchall()
            
            activities = []
            for row in rows:
                activities.append({
                    "date": row[0].isoformat() if row[0] else None,
                    "spread_symbol": row[1],
                    "front_month": row[2],
                    "back_month": row[3],
                    "total_volume": row[4],
                    "buy_volume": row[5],
                    "sell_volume": row[6],
                    "open": float(row[7]) if row[7] else None,
                    "high": float(row[8]) if row[8] else None,
                    "low": float(row[9]) if row[9] else None,
                    "close": float(row[10]) if row[10] else None,
                    "vwap": float(row[11]) if row[11] else None,
                    "trade_count": row[12],
                    "large_trades": row[13],
                    "avg_trade_size": float(row[14]) if row[14] else None,
                    "net_direction": row[15]
                })
            
            return {
                "activities": activities,
                "count": len(activities)
            }
    except Exception as e:
        logger.error(f"Error fetching roll activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Microstructure & Large Trades Endpoints
# =============================================================================

@router.get("/microstructure/{target_date}")
async def get_microstructure_summary(target_date: str):
    """
    Get microstructure summary for a date (available Dec 2025+).
    Includes bid/ask spreads, book imbalance, trade intensity.
    """
    try:
        with engine.connect() as conn:
            # Check if microstructure data exists for this date
            check_query = """
                SELECT COUNT(*) FROM databento_es_microstructure.micro_1s
                WHERE date_utc = :date
            """
            count = conn.execute(text(check_query), {"date": target_date}).scalar()
            
            if count == 0:
                return {
                    "date": target_date,
                    "available": False,
                    "message": "Microstructure data not available for this date (available Dec 2025+)"
                }
            
            # Get aggregated microstructure metrics
            micro_query = """
                SELECT 
                    symbol,
                    COUNT(*) as data_points,
                    AVG(spread) as avg_spread,
                    MIN(spread) as min_spread,
                    MAX(spread) as max_spread,
                    AVG(spread_bps) as avg_spread_bps,
                    AVG(imbalance_l1) as avg_imbalance_l1,
                    AVG(imbalance_l5) as avg_imbalance_l5,
                    AVG(book_pressure) as avg_book_pressure,
                    SUM(trade_count) as total_trades,
                    SUM(buy_volume) as total_buy_volume,
                    SUM(sell_volume) as total_sell_volume,
                    AVG(trade_intensity) as avg_trade_intensity
                FROM databento_es_microstructure.micro_1s
                WHERE date_utc = :date
                GROUP BY symbol
                ORDER BY total_trades DESC
            """
            result = conn.execute(text(micro_query), {"date": target_date})
            
            symbols = []
            for row in result.fetchall():
                symbols.append({
                    "symbol": row[0],
                    "data_points": row[1],
                    "avg_spread": float(row[2]) if row[2] else None,
                    "min_spread": float(row[3]) if row[3] else None,
                    "max_spread": float(row[4]) if row[4] else None,
                    "avg_spread_bps": float(row[5]) if row[5] else None,
                    "avg_imbalance_l1": float(row[6]) if row[6] else None,
                    "avg_imbalance_l5": float(row[7]) if row[7] else None,
                    "avg_book_pressure": float(row[8]) if row[8] else None,
                    "total_trades": row[9],
                    "buy_volume": row[10],
                    "sell_volume": row[11],
                    "avg_trade_intensity": float(row[12]) if row[12] else None
                })
            
            return {
                "date": target_date,
                "available": True,
                "symbols": symbols,
                "total_data_points": count
            }
    except Exception as e:
        logger.error(f"Error fetching microstructure for {target_date}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/large-trades/{target_date}")
async def get_large_trades(
    target_date: str,
    min_size: int = Query(100, description="Minimum trade size"),
    limit: int = Query(50, description="Maximum number of trades to return")
):
    """
    Get large trades for a specific date.
    Large trades often indicate institutional activity.
    """
    try:
        with engine.connect() as conn:
            query = """
                SELECT 
                    timestamp_utc,
                    symbol,
                    side,
                    price,
                    size,
                    bid_px,
                    ask_px,
                    spread,
                    trade_type
                FROM databento_es_trades.large_trades
                WHERE date_utc = :date AND size >= :min_size
                ORDER BY size DESC
                LIMIT :limit
            """
            result = conn.execute(text(query), {
                "date": target_date,
                "min_size": min_size,
                "limit": limit
            })
            
            trades = []
            for row in result.fetchall():
                trades.append({
                    "timestamp": row[0].isoformat() if row[0] else None,
                    "symbol": row[1],
                    "side": row[2],
                    "price": float(row[3]) if row[3] else None,
                    "size": row[4],
                    "bid_px": float(row[5]) if row[5] else None,
                    "ask_px": float(row[6]) if row[6] else None,
                    "spread": float(row[7]) if row[7] else None,
                    "trade_type": row[8]
                })
            
            # Get summary stats
            summary_query = """
                SELECT 
                    COUNT(*) as count,
                    SUM(size) as total_size,
                    AVG(size) as avg_size,
                    MAX(size) as max_size,
                    SUM(CASE WHEN side = 'B' THEN size ELSE 0 END) as buy_size,
                    SUM(CASE WHEN side = 'S' THEN size ELSE 0 END) as sell_size
                FROM databento_es_trades.large_trades
                WHERE date_utc = :date AND size >= :min_size
            """
            summary = conn.execute(text(summary_query), {
                "date": target_date,
                "min_size": min_size
            }).fetchone()
            
            return {
                "date": target_date,
                "min_size": min_size,
                "trades": trades,
                "count": len(trades),
                "summary": {
                    "total_count": summary[0] if summary else 0,
                    "total_size": summary[1] if summary else 0,
                    "avg_size": float(summary[2]) if summary and summary[2] else None,
                    "max_size": summary[3] if summary else None,
                    "buy_size": summary[4] if summary else 0,
                    "sell_size": summary[5] if summary else 0
                }
            }
    except Exception as e:
        logger.error(f"Error fetching large trades for {target_date}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/intraday/{target_date}/{symbol}")
async def get_intraday_data(
    target_date: str,
    symbol: str,
    limit: int = Query(2000, description="Maximum number of 1m bars (e.g. full day ~1440)")
):
    """
    Get 1-minute intraday data for a specific symbol and date.
    """
    try:
        table_name = symbol.lower()
        
        with engine.connect() as conn:
            # Check if it's a spread or outright
            if '-' in symbol:
                # It's a spread - check databento_es_spreads.ohlcv_1m first (MBO-derived),
                # fall back to individual table in databento_es_ohlcv_1m
                query = """
                    SELECT 
                        timestamp_utc,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM databento_es_spreads.ohlcv_1m
                    WHERE date_utc = :date AND spread_symbol = :symbol
                    ORDER BY timestamp_utc
                    LIMIT :limit
                """
                result_check = conn.execute(text(query), {"date": target_date, "symbol": symbol, "limit": limit})
                rows_check = result_check.fetchall()
                if rows_check:
                    bars = []
                    for row in rows_check:
                        bars.append({
                            "timestamp": row[0].isoformat() if row[0] else None,
                            "open": float(row[1]) if row[1] else None,
                            "high": float(row[2]) if row[2] else None,
                            "low": float(row[3]) if row[3] else None,
                            "close": float(row[4]) if row[4] else None,
                            "volume": row[5]
                        })
                    return {
                        "date": target_date,
                        "symbol": symbol,
                        "bars": bars,
                        "count": len(bars)
                    }
                # Fall back to individual table
                query = f"""
                    SELECT 
                        timestamp_utc,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM databento_es_ohlcv_1m.{table_name.replace('-', '_')}
                    WHERE date_utc = :date
                    ORDER BY timestamp_utc
                    LIMIT :limit
                """
            else:
                # It's an outright
                query = f"""
                    SELECT 
                        timestamp_utc,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM databento_es_ohlcv_1m.{table_name}
                    WHERE date_utc = :date
                    ORDER BY timestamp_utc
                    LIMIT :limit
                """
            
            result = conn.execute(text(query), {"date": target_date, "limit": limit})
            
            bars = []
            for row in result.fetchall():
                bars.append({
                    "timestamp": row[0].isoformat() if row[0] else None,
                    "open": float(row[1]) if row[1] else None,
                    "high": float(row[2]) if row[2] else None,
                    "low": float(row[3]) if row[3] else None,
                    "close": float(row[4]) if row[4] else None,
                    "volume": row[5]
                })
            
            return {
                "date": target_date,
                "symbol": symbol,
                "bars": bars,
                "count": len(bars)
            }
    except Exception as e:
        logger.error(f"Error fetching intraday data for {symbol} on {target_date}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Schema Info Endpoints
# =============================================================================

@router.get("/schemas")
async def get_databento_schemas():
    """
    Get list of all databento schemas and their tables.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name LIKE 'databento%'
                ORDER BY schema_name
            """))
            schemas = [row[0] for row in result.fetchall()]
            
            schema_info = {}
            for schema in schemas:
                tables_result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = :schema AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """), {"schema": schema})
                schema_info[schema] = [row[0] for row in tables_result.fetchall()]
            
            return {
                "schemas": schema_info,
                "schema_count": len(schemas)
            }
    except Exception as e:
        logger.error(f"Error fetching schemas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Feature Catalog & Data Endpoints (for Feature Dashboard)
# =============================================================================

@router.get("/features/catalog")
async def get_feature_catalog():
    """
    Get complete catalog of all feature tables, their columns, row counts,
    date ranges, and null percentages. Powers the Feature Dashboard.
    """
    try:
        with engine.connect() as conn:
            # Define all feature tables with metadata
            feature_tables = [
                {
                    "table": "volatility_1m",
                    "schema": "databento_es_features",
                    "group": "volatility",
                    "group_label": "Volatility Regime",
                    "description": "Multi-timeframe volatility estimators for regime detection",
                    "model_use": ["regime"],
                    "columns_meta": {
                        "realized_vol_5m": {"label": "Realized Vol 5m", "description": "5-minute realized volatility (annualized std of log returns)", "type": "continuous"},
                        "realized_vol_15m": {"label": "Realized Vol 15m", "description": "15-minute realized volatility", "type": "continuous"},
                        "realized_vol_1h": {"label": "Realized Vol 1h", "description": "1-hour realized volatility", "type": "continuous"},
                        "parkinson_vol": {"label": "Parkinson Vol", "description": "Parkinson high-low volatility estimator (more efficient than close-close)", "type": "continuous"},
                        "garman_klass_vol": {"label": "Garman-Klass Vol", "description": "Garman-Klass OHLC volatility estimator (most efficient single-bar estimator)", "type": "continuous"},
                        "vol_of_vol_15m": {"label": "Vol of Vol 15m", "description": "Volatility of volatility - measures regime instability/transitions", "type": "continuous"},
                        "range_ratio": {"label": "Range Ratio", "description": "Current bar range vs rolling average range (>1 = expanding, <1 = contracting)", "type": "ratio"},
                        "atr_20": {"label": "ATR 20", "description": "20-period Average True Range in points", "type": "continuous"},
                        "vol_zscore": {"label": "Vol Z-Score", "description": "Current vol relative to rolling mean in standard deviations", "type": "zscore"},
                        "vol_percentile": {"label": "Vol Percentile", "description": "Current vol percentile rank (0-1) over rolling window", "type": "percentile"},
                    }
                },
                {
                    "table": "rolling_features_1m",
                    "schema": "databento_es_features",
                    "group": "momentum",
                    "group_label": "Momentum & Flow",
                    "description": "Multi-timeframe VWAP, momentum, volume, and imbalance features",
                    "model_use": ["regime", "entry"],
                    "columns_meta": {
                        "vwap_5m": {"label": "VWAP 5m", "description": "5-minute rolling Volume Weighted Average Price", "type": "price"},
                        "vwap_15m": {"label": "VWAP 15m", "description": "15-minute rolling VWAP", "type": "price"},
                        "vwap_1h": {"label": "VWAP 1h", "description": "1-hour rolling VWAP", "type": "price"},
                        "price_vs_vwap_5m": {"label": "Price vs VWAP 5m", "description": "Price deviation from 5m VWAP in points (positive = above)", "type": "deviation"},
                        "price_vs_vwap_15m": {"label": "Price vs VWAP 15m", "description": "Price deviation from 15m VWAP", "type": "deviation"},
                        "price_vs_vwap_1h": {"label": "Price vs VWAP 1h", "description": "Price deviation from 1h VWAP", "type": "deviation"},
                        "imbalance_5m": {"label": "Imbalance 5m", "description": "5-min rolling order imbalance (-1 to +1, positive = buy pressure)", "type": "bounded"},
                        "imbalance_15m": {"label": "Imbalance 15m", "description": "15-min rolling order imbalance", "type": "bounded"},
                        "imbalance_1h": {"label": "Imbalance 1h", "description": "1-hour rolling order imbalance", "type": "bounded"},
                        "imbalance_divergence_5m_15m": {"label": "Imbalance Divergence", "description": "Divergence between 5m and 15m imbalance (signals short-term reversal)", "type": "divergence"},
                        "volume_5m": {"label": "Volume 5m", "description": "Rolling 5-minute total volume", "type": "volume"},
                        "volume_15m": {"label": "Volume 15m", "description": "Rolling 15-minute total volume", "type": "volume"},
                        "volume_1h": {"label": "Volume 1h", "description": "Rolling 1-hour total volume", "type": "volume"},
                        "volume_ratio_5m": {"label": "Volume Ratio 5m", "description": "Current 5m volume vs rolling average (>1 = above average)", "type": "ratio"},
                        "volume_ratio_15m": {"label": "Volume Ratio 15m", "description": "Current 15m volume vs rolling average", "type": "ratio"},
                        "momentum_5m": {"label": "Momentum 5m", "description": "5-minute price momentum (close - close[5])", "type": "continuous"},
                        "momentum_15m": {"label": "Momentum 15m", "description": "15-minute price momentum", "type": "continuous"},
                        "momentum_1h": {"label": "Momentum 1h", "description": "1-hour price momentum", "type": "continuous"},
                        "momentum_ratio": {"label": "Momentum Ratio", "description": "Short vs long momentum ratio (divergence = potential reversal)", "type": "ratio"},
                        "spread_avg_5m": {"label": "Spread Avg 5m", "description": "5-minute rolling average bid-ask spread", "type": "continuous"},
                        "spread_avg_15m": {"label": "Spread Avg 15m", "description": "15-minute rolling average spread", "type": "continuous"},
                    }
                },
                {
                    "table": "price_levels_1m",
                    "schema": "databento_es_features",
                    "group": "structure",
                    "group_label": "Price Structure & Session",
                    "description": "Session context, price location within day range, gap analysis",
                    "model_use": ["regime", "entry"],
                    "columns_meta": {
                        "session": {"label": "Session", "description": "RTH (Regular Trading Hours 9:30-16:00 ET) or ETH (Extended)", "type": "categorical"},
                        "minutes_since_session_open": {"label": "Mins Since Open", "description": "Minutes elapsed since current session opened", "type": "time"},
                        "minutes_to_session_close": {"label": "Mins To Close", "description": "Minutes remaining until current session closes", "type": "time"},
                        "day_high": {"label": "Day High", "description": "Running intraday high price", "type": "price"},
                        "day_low": {"label": "Day Low", "description": "Running intraday low price", "type": "price"},
                        "day_vwap": {"label": "Day VWAP", "description": "Running intraday VWAP", "type": "price"},
                        "dist_from_day_high": {"label": "Dist From High", "description": "Distance from day high in ATR units (0 = at high)", "type": "continuous"},
                        "dist_from_day_low": {"label": "Dist From Low", "description": "Distance from day low in ATR units (0 = at low)", "type": "continuous"},
                        "dist_from_vwap": {"label": "Dist From VWAP", "description": "Distance from day VWAP in ATR units", "type": "continuous"},
                        "price_location": {"label": "Price Location", "description": "Price position within day range (0 = at low, 1 = at high)", "type": "percentile"},
                        "overnight_gap": {"label": "Overnight Gap", "description": "Gap from prior close in points", "type": "continuous"},
                        "gap_fill_pct": {"label": "Gap Fill %", "description": "Percentage of overnight gap that has been filled (0-1+)", "type": "percentile"},
                        "session_vwap": {"label": "Session VWAP", "description": "VWAP for current session only", "type": "price"},
                        "price_vs_session_vwap": {"label": "Price vs Session VWAP", "description": "Price deviation from session VWAP", "type": "deviation"},
                        "session_volume": {"label": "Session Volume", "description": "Cumulative session volume", "type": "volume"},
                        "session_buy_volume": {"label": "Session Buy Vol", "description": "Cumulative session buy-initiated volume", "type": "volume"},
                        "session_sell_volume": {"label": "Session Sell Vol", "description": "Cumulative session sell-initiated volume", "type": "volume"},
                    }
                },
                {
                    "table": "microstructure_toxicity",
                    "schema": "databento_es_features",
                    "group": "toxicity",
                    "group_label": "Order Flow Toxicity",
                    "description": "Informed trading probability, price impact, and liquidity measures from TBBO/MBP-10",
                    "model_use": ["regime", "entry"],
                    "columns_meta": {
                        "vpin": {"label": "VPIN", "description": "Volume-Synchronized Probability of Informed Trading (0-1, >0.7 = toxic)", "type": "percentile"},
                        "vpin_bucket_count": {"label": "VPIN Buckets", "description": "Number of volume buckets used in VPIN calculation", "type": "count"},
                        "kyle_lambda": {"label": "Kyle Lambda", "description": "Kyle price impact coefficient (higher = more price impact per unit volume)", "type": "continuous"},
                        "kyle_lambda_5m": {"label": "Kyle Lambda 5m", "description": "5-minute rolling Kyle Lambda", "type": "continuous"},
                        "amihud_ratio": {"label": "Amihud Ratio", "description": "Amihud illiquidity ratio (|return|/volume, higher = less liquid)", "type": "continuous"},
                        "amihud_ratio_5m": {"label": "Amihud Ratio 5m", "description": "5-minute rolling Amihud ratio", "type": "continuous"},
                        "effective_spread": {"label": "Effective Spread", "description": "Actual cost of trading (trade price vs midpoint)", "type": "continuous"},
                        "effective_spread_avg": {"label": "Effective Spread Avg", "description": "Rolling average effective spread", "type": "continuous"},
                        "quoted_spread_avg": {"label": "Quoted Spread Avg", "description": "Rolling average quoted bid-ask spread", "type": "continuous"},
                        "spread_capture": {"label": "Spread Capture", "description": "Effective spread / Quoted spread ratio (>1 = paying more than quoted)", "type": "ratio"},
                        "toxicity_score": {"label": "Toxicity Score", "description": "Composite toxicity score combining VPIN, Lambda, Amihud", "type": "composite"},
                    }
                },
                {
                    "table": "book_shape_1m",
                    "schema": "databento_es_features",
                    "group": "book_shape",
                    "group_label": "Order Book Shape",
                    "description": "Order book depth profile, slope, concentration, and resilience from MBP-10",
                    "model_use": ["entry"],
                    "columns_meta": {
                        "book_slope_bid": {"label": "Book Slope Bid", "description": "Rate of depth decay on bid side (steeper = thinner book)", "type": "continuous"},
                        "book_slope_ask": {"label": "Book Slope Ask", "description": "Rate of depth decay on ask side", "type": "continuous"},
                        "book_asymmetry": {"label": "Book Asymmetry", "description": "Bid vs ask slope difference (positive = bid thicker, bearish lean)", "type": "bounded"},
                        "depth_ratio_l1": {"label": "Depth Ratio L1", "description": "L1 bid size / total bid+ask at L1 (imbalance at best prices)", "type": "percentile"},
                        "depth_ratio_l5": {"label": "Depth Ratio L5", "description": "Cumulative depth ratio through 5 levels", "type": "percentile"},
                        "depth_ratio_l10": {"label": "Depth Ratio L10", "description": "Cumulative depth ratio through 10 levels", "type": "percentile"},
                        "avg_gap_bid": {"label": "Avg Gap Bid", "description": "Average price gap between bid levels (ticks)", "type": "continuous"},
                        "avg_gap_ask": {"label": "Avg Gap Ask", "description": "Average price gap between ask levels (ticks)", "type": "continuous"},
                        "max_gap_bid": {"label": "Max Gap Bid", "description": "Largest gap between bid levels (thin spots)", "type": "continuous"},
                        "max_gap_ask": {"label": "Max Gap Ask", "description": "Largest gap between ask levels", "type": "continuous"},
                        "depth_concentration_bid": {"label": "Depth Conc Bid", "description": "Herfindahl index of bid depth (higher = concentrated at fewer levels)", "type": "percentile"},
                        "depth_concentration_ask": {"label": "Depth Conc Ask", "description": "Herfindahl index of ask depth", "type": "percentile"},
                        "ticks_to_absorb_100": {"label": "Ticks to Absorb 100", "description": "Number of price levels needed to absorb 100 contracts", "type": "count"},
                        "ticks_to_absorb_500": {"label": "Ticks to Absorb 500", "description": "Number of price levels needed to absorb 500 contracts", "type": "count"},
                        "bid_resilience": {"label": "Bid Resilience", "description": "How quickly bid depth recovers after being consumed", "type": "continuous"},
                        "ask_resilience": {"label": "Ask Resilience", "description": "How quickly ask depth recovers after being consumed", "type": "continuous"},
                    }
                },
                {
                    "table": "quote_dynamics_1m",
                    "schema": "databento_es_features",
                    "group": "quote_dynamics",
                    "group_label": "Quote & Cancel Dynamics",
                    "description": "Quote activity, order lifespan, and fleeting liquidity detection from MBP-10",
                    "model_use": ["entry"],
                    "columns_meta": {
                        "quote_count": {"label": "Quote Count", "description": "Total number of quote updates this minute", "type": "count"},
                        "add_count": {"label": "Add Count", "description": "New order additions this minute", "type": "count"},
                        "cancel_count": {"label": "Cancel Count", "description": "Order cancellations this minute", "type": "count"},
                        "modify_count": {"label": "Modify Count", "description": "Order modifications this minute", "type": "count"},
                        "trade_count": {"label": "Trade Count", "description": "Number of trades this minute", "type": "count"},
                        "quote_to_trade": {"label": "Quote/Trade Ratio", "description": "Ratio of quotes to trades (higher = more HFT activity)", "type": "ratio"},
                        "cancel_to_trade": {"label": "Cancel/Trade Ratio", "description": "Ratio of cancels to trades (higher = more spoofing risk)", "type": "ratio"},
                        "add_to_cancel": {"label": "Add/Cancel Ratio", "description": "Ratio of adds to cancels (< 1 = net liquidity withdrawal)", "type": "ratio"},
                        "quote_intensity": {"label": "Quote Intensity", "description": "Quotes per second (higher = more active market making)", "type": "continuous"},
                        "add_intensity": {"label": "Add Intensity", "description": "New orders per second", "type": "continuous"},
                        "cancel_intensity": {"label": "Cancel Intensity", "description": "Cancellations per second", "type": "continuous"},
                        "avg_order_lifespan_ms": {"label": "Avg Order Lifespan", "description": "Average milliseconds an order stays before fill/cancel", "type": "continuous"},
                        "median_order_lifespan_ms": {"label": "Median Order Lifespan", "description": "Median order lifespan (less sensitive to outliers)", "type": "continuous"},
                        "min_order_lifespan_ms": {"label": "Min Order Lifespan", "description": "Fastest fill/cancel time (HFT signature)", "type": "continuous"},
                        "fleeting_order_count": {"label": "Fleeting Orders", "description": "Orders cancelled within 100ms (phantom liquidity)", "type": "count"},
                        "fleeting_pct": {"label": "Fleeting %", "description": "Percentage of orders that are fleeting (>0.5 = mostly phantom)", "type": "percentile"},
                        "bid_improve_count": {"label": "Bid Improves", "description": "Times bid price improved this minute", "type": "count"},
                        "ask_improve_count": {"label": "Ask Improves", "description": "Times ask price improved this minute", "type": "count"},
                    }
                },
                {
                    "table": "trade_clustering_1m",
                    "schema": "databento_es_features",
                    "group": "trade_clustering",
                    "group_label": "Trade Clustering & Sequencing",
                    "description": "Trade run analysis, burstiness, and aggressor patterns from trade data",
                    "model_use": ["entry"],
                    "columns_meta": {
                        "max_buy_run": {"label": "Max Buy Run", "description": "Longest streak of consecutive buy-initiated trades", "type": "count"},
                        "max_sell_run": {"label": "Max Sell Run", "description": "Longest streak of consecutive sell-initiated trades", "type": "count"},
                        "avg_run_length": {"label": "Avg Run Length", "description": "Average length of same-side trade runs", "type": "continuous"},
                        "run_count": {"label": "Run Count", "description": "Number of directional runs (alternations) this minute", "type": "count"},
                        "burstiness": {"label": "Burstiness", "description": "Coefficient of variation of inter-trade times (>1 = bursty, <1 = regular)", "type": "continuous"},
                        "trade_arrival_entropy": {"label": "Arrival Entropy", "description": "Entropy of trade arrival times (higher = more random)", "type": "continuous"},
                        "avg_inter_trade_ms": {"label": "Avg Inter-Trade ms", "description": "Average milliseconds between trades", "type": "continuous"},
                        "std_inter_trade_ms": {"label": "Std Inter-Trade ms", "description": "Standard deviation of inter-trade times", "type": "continuous"},
                        "min_inter_trade_ms": {"label": "Min Inter-Trade ms", "description": "Minimum time between trades (HFT speed)", "type": "continuous"},
                        "max_inter_trade_ms": {"label": "Max Inter-Trade ms", "description": "Maximum time between trades (lull)", "type": "continuous"},
                        "size_autocorr_lag1": {"label": "Size Autocorr Lag1", "description": "Autocorrelation of trade sizes at lag 1 (institutional splitting?)", "type": "bounded"},
                        "size_autocorr_lag5": {"label": "Size Autocorr Lag5", "description": "Autocorrelation of trade sizes at lag 5", "type": "bounded"},
                        "aggressor_entropy": {"label": "Aggressor Entropy", "description": "Entropy of buy/sell aggressor sequence (0 = one-sided, 1 = balanced)", "type": "percentile"},
                        "side_persistence": {"label": "Side Persistence", "description": "Probability next trade is same side as previous (>0.5 = trending)", "type": "percentile"},
                        "clustering_coef": {"label": "Clustering Coef", "description": "Degree to which trades cluster in time (1 = perfectly clustered)", "type": "percentile"},
                    }
                },
                {
                    "table": "cross_contract_1m",
                    "schema": "databento_es_features",
                    "group": "cross_contract",
                    "group_label": "Cross-Contract & Term Structure",
                    "description": "Calendar spread dynamics, roll pressure, and term structure features",
                    "model_use": ["regime"],
                    "columns_meta": {
                        "spread_price": {"label": "Spread Price", "description": "Calendar spread price (front - back)", "type": "price"},
                        "spread_z_score": {"label": "Spread Z-Score", "description": "Spread price relative to rolling mean in std devs", "type": "zscore"},
                        "spread_percentile": {"label": "Spread Percentile", "description": "Spread price percentile rank (0-1)", "type": "percentile"},
                        "spread_vs_fair_value": {"label": "Spread vs Fair Value", "description": "Deviation from estimated fair value (carry-adjusted)", "type": "deviation"},
                        "front_volume": {"label": "Front Volume", "description": "Front month contract volume", "type": "volume"},
                        "back_volume": {"label": "Back Volume", "description": "Back month contract volume", "type": "volume"},
                        "roll_pressure": {"label": "Roll Pressure", "description": "Front/back volume ratio (high = active rolling)", "type": "ratio"},
                        "roll_momentum": {"label": "Roll Momentum", "description": "Rate of change in roll pressure", "type": "continuous"},
                        "term_structure_slope": {"label": "Term Structure Slope", "description": "Slope of the futures curve (contango vs backwardation)", "type": "continuous"},
                        "curve_convexity": {"label": "Curve Convexity", "description": "Curvature of the term structure", "type": "continuous"},
                        "spread_bid_ask": {"label": "Spread Bid-Ask", "description": "Bid-ask spread on the calendar spread instrument", "type": "continuous"},
                        "spread_imbalance": {"label": "Spread Imbalance", "description": "Order imbalance on the calendar spread", "type": "bounded"},
                    }
                },
                {
                    "table": "sweep_iceberg_events",
                    "schema": "databento_es_features",
                    "group": "events",
                    "group_label": "Sweep & Iceberg Events",
                    "description": "Detected sweep orders, iceberg orders, and stop hunt events",
                    "model_use": ["entry"],
                    "columns_meta": {
                        "event_type": {"label": "Event Type", "description": "sweep, iceberg, or stop_hunt", "type": "categorical"},
                        "side": {"label": "Side", "description": "Buy (B) or Sell (A) aggressor side", "type": "categorical"},
                        "total_size": {"label": "Total Size", "description": "Total contracts involved in the event", "type": "count"},
                        "levels_swept": {"label": "Levels Swept", "description": "Number of price levels consumed (sweeps only)", "type": "count"},
                        "price_range": {"label": "Price Range", "description": "Price range traversed during the event", "type": "continuous"},
                        "price_impact": {"label": "Price Impact", "description": "Net price impact from start to end of event", "type": "continuous"},
                        "spread_before": {"label": "Spread Before", "description": "Bid-ask spread before the event", "type": "continuous"},
                        "spread_after": {"label": "Spread After", "description": "Bid-ask spread after the event", "type": "continuous"},
                    }
                },
            ]
            
            # Get row counts and date ranges for each table
            for ft in feature_tables:
                try:
                    # Row count
                    cnt_q = f'SELECT COUNT(*) FROM {ft["schema"]}.{ft["table"]}'
                    ft["row_count"] = conn.execute(text(cnt_q)).scalar()
                    
                    # Date range - find timestamp column
                    ts_col = "timestamp_utc"
                    range_q = f'SELECT MIN({ts_col}), MAX({ts_col}) FROM {ft["schema"]}.{ft["table"]}'
                    range_row = conn.execute(text(range_q)).fetchone()
                    ft["date_range"] = {
                        "min": range_row[0].isoformat() if range_row[0] else None,
                        "max": range_row[1].isoformat() if range_row[1] else None,
                    }
                    
                    # Get null percentage for each feature column
                    num_cols = list(ft["columns_meta"].keys())
                    if ft["row_count"] > 0 and num_cols:
                        null_parts = ", ".join([
                            f'ROUND(SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END)::numeric / COUNT(*)::numeric * 100, 1) AS "{c}"'
                            for c in num_cols[:25]
                        ])
                        null_q = f'SELECT {null_parts} FROM {ft["schema"]}.{ft["table"]}'
                        null_row = conn.execute(text(null_q)).fetchone()
                        for i, c in enumerate(num_cols[:25]):
                            ft["columns_meta"][c]["null_pct"] = float(null_row[i]) if null_row[i] is not None else 0
                    
                except Exception as e:
                    logger.warning(f"Error querying {ft['schema']}.{ft['table']}: {e}")
                    ft["row_count"] = 0
                    ft["date_range"] = {"min": None, "max": None}
            
            return {
                "tables": feature_tables,
                "table_count": len(feature_tables),
                "total_features": sum(len(t["columns_meta"]) for t in feature_tables),
            }
    except Exception as e:
        logger.error(f"Error fetching feature catalog: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/data/{table_name}")
async def get_feature_data(
    table_name: str,
    columns: str = Query(None, description="Comma-separated column names (default: all)"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    limit: int = Query(500, description="Row limit"),
    offset: int = Query(0, description="Row offset for pagination"),
):
    """
    Get feature data from a specific table with column selection and date filtering.
    """
    ALLOWED_TABLES = {
        "volatility_1m", "rolling_features_1m", "price_levels_1m",
        "microstructure_toxicity", "book_shape_1m", "quote_dynamics_1m",
        "trade_clustering_1m", "cross_contract_1m", "sweep_iceberg_events"
    }
    
    if table_name not in ALLOWED_TABLES:
        raise HTTPException(status_code=400, detail=f"Invalid table: {table_name}. Allowed: {ALLOWED_TABLES}")
    
    try:
        with engine.connect() as conn:
            # Build column list
            if columns:
                col_list = [c.strip() for c in columns.split(",")]
                # Always include timestamp for alignment
                if "timestamp_utc" not in col_list:
                    col_list.insert(0, "timestamp_utc")
                select_cols = ", ".join([f'"{c}"' for c in col_list])
            else:
                select_cols = "*"
            
            query = f'SELECT {select_cols} FROM databento_es_features."{table_name}"'
            params = {}
            conditions = []
            
            if start_date:
                conditions.append("timestamp_utc >= :start_date")
                params["start_date"] = start_date
            if end_date:
                conditions.append("timestamp_utc <= :end_date")
                params["end_date"] = end_date
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp_utc ASC"
            query += f" LIMIT :limit OFFSET :offset"
            params["limit"] = limit
            params["offset"] = offset
            
            result = conn.execute(text(query), params)
            col_names = list(result.keys())
            rows = result.fetchall()
            
            data = []
            for row in rows:
                row_dict = {}
                for i, col in enumerate(col_names):
                    val = row[i]
                    if hasattr(val, 'isoformat'):
                        row_dict[col] = val.isoformat()
                    elif val is not None:
                        try:
                            row_dict[col] = float(val)
                        except (TypeError, ValueError):
                            row_dict[col] = str(val)
                    else:
                        row_dict[col] = None
                data.append(row_dict)
            
            # Get total count for pagination
            count_q = f'SELECT COUNT(*) FROM databento_es_features."{table_name}"'
            if conditions:
                count_q += " WHERE " + " AND ".join(conditions)
            total = conn.execute(text(count_q), {k: v for k, v in params.items() if k not in ('limit', 'offset')}).scalar()
            
            return {
                "table": table_name,
                "columns": col_names,
                "data": data,
                "count": len(data),
                "total": total,
                "limit": limit,
                "offset": offset,
            }
    except Exception as e:
        logger.error(f"Error fetching feature data from {table_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/sample-stats/{table_name}")
async def get_feature_sample_stats(
    table_name: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """
    Get summary statistics (min, max, mean, std, median) for numeric columns in a feature table.
    """
    ALLOWED_TABLES = {
        "volatility_1m", "rolling_features_1m", "price_levels_1m",
        "microstructure_toxicity", "book_shape_1m", "quote_dynamics_1m",
        "trade_clustering_1m", "cross_contract_1m"
    }
    
    if table_name not in ALLOWED_TABLES:
        raise HTTPException(status_code=400, detail=f"Invalid table: {table_name}")
    
    try:
        with engine.connect() as conn:
            # Get numeric columns
            col_q = """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'databento_es_features' AND table_name = :tbl
                AND data_type IN ('numeric', 'double precision', 'real', 'integer', 'bigint')
                AND column_name NOT IN ('id')
                ORDER BY ordinal_position
            """
            col_rows = conn.execute(text(col_q), {"tbl": table_name}).fetchall()
            num_cols = [r[0] for r in col_rows]
            
            if not num_cols:
                return {"table": table_name, "stats": {}}
            
            # Build stats query
            stats_parts = []
            for c in num_cols[:20]:  # Limit to 20 columns
                stats_parts.append(f'MIN("{c}") AS "{c}_min"')
                stats_parts.append(f'MAX("{c}") AS "{c}_max"')
                stats_parts.append(f'AVG("{c}") AS "{c}_mean"')
                stats_parts.append(f'STDDEV("{c}") AS "{c}_std"')
                stats_parts.append(f'PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "{c}") AS "{c}_median"')
            
            query = f'SELECT {", ".join(stats_parts)} FROM databento_es_features."{table_name}"'
            params = {}
            conditions = []
            
            if start_date:
                conditions.append("timestamp_utc >= :start_date")
                params["start_date"] = start_date
            if end_date:
                conditions.append("timestamp_utc <= :end_date")
                params["end_date"] = end_date
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            row = conn.execute(text(query), params).fetchone()
            
            stats = {}
            for i, c in enumerate(num_cols[:20]):
                base = i * 5
                stats[c] = {
                    "min": float(row[base]) if row[base] is not None else None,
                    "max": float(row[base + 1]) if row[base + 1] is not None else None,
                    "mean": float(row[base + 2]) if row[base + 2] is not None else None,
                    "std": float(row[base + 3]) if row[base + 3] is not None else None,
                    "median": float(row[base + 4]) if row[base + 4] is not None else None,
                }
            
            return {"table": table_name, "stats": stats}
    except Exception as e:
        logger.error(f"Error fetching stats for {table_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Forward-Looking Data Endpoint (Pillar 2 & 3)
# =============================================================================

@router.get("/forward-data/{target_date}")
async def get_forward_data(target_date: str):
    """
    Get all forward-looking external data for a specific date.
    Returns rates (SOFR, Fed Funds, implied forwards, div yield, net carry),
    forward volatility (VIX, VIX/RV ratio), event flags, and regime context.
    """
    try:
        with engine.connect() as conn:
            result = {}

            # 1. External rates (schema: spx_div_yield, net_carry_spot, net_carry_3m — no spx_div_yield_trail)
            rates_row = conn.execute(text("""
                SELECT sofr_spot, fed_funds_effective, fed_funds_target_upper,
                       fed_funds_target_lower, implied_rate_1m, implied_rate_2m,
                       implied_rate_3m, implied_rate_6m, rate_source,
                       spx_div_yield, net_carry_spot, net_carry_3m
                FROM databento_es_features.external_rates_1d
                WHERE date_utc = :date
            """), {"date": target_date}).fetchone()

            if rates_row:
                result["rates"] = {
                    "sofr_spot": float(rates_row[0]) if rates_row[0] is not None else None,
                    "fed_funds_effective": float(rates_row[1]) if rates_row[1] is not None else None,
                    "fed_funds_target_upper": float(rates_row[2]) if rates_row[2] is not None else None,
                    "fed_funds_target_lower": float(rates_row[3]) if rates_row[3] is not None else None,
                    "implied_rate_1m": float(rates_row[4]) if rates_row[4] is not None else None,
                    "implied_rate_2m": float(rates_row[5]) if rates_row[5] is not None else None,
                    "implied_rate_3m": float(rates_row[6]) if rates_row[6] is not None else None,
                    "implied_rate_6m": float(rates_row[7]) if rates_row[7] is not None else None,
                    "rate_source": rates_row[8],
                    "spx_div_yield": float(rates_row[9]) if rates_row[9] is not None else None,
                    "spx_div_yield_trail": float(rates_row[9]) if rates_row[9] is not None else None,
                    "net_carry_spot": float(rates_row[10]) if rates_row[10] is not None else None,
                    "net_carry_3m": float(rates_row[11]) if rates_row[11] is not None else None,
                }
            else:
                # Try the closest prior date (weekends/holidays)
                fallback_row = conn.execute(text("""
                    SELECT sofr_spot, fed_funds_effective, fed_funds_target_upper,
                           fed_funds_target_lower, implied_rate_1m, implied_rate_2m,
                           implied_rate_3m, implied_rate_6m, rate_source,
                           spx_div_yield, net_carry_spot, net_carry_3m,
                           date_utc
                    FROM databento_es_features.external_rates_1d
                    WHERE date_utc <= :date AND sofr_spot IS NOT NULL
                    ORDER BY date_utc DESC LIMIT 1
                """), {"date": target_date}).fetchone()

                if fallback_row:
                    result["rates"] = {
                        "sofr_spot": float(fallback_row[0]) if fallback_row[0] is not None else None,
                        "fed_funds_effective": float(fallback_row[1]) if fallback_row[1] is not None else None,
                        "fed_funds_target_upper": float(fallback_row[2]) if fallback_row[2] is not None else None,
                        "fed_funds_target_lower": float(fallback_row[3]) if fallback_row[3] is not None else None,
                        "implied_rate_1m": float(fallback_row[4]) if fallback_row[4] is not None else None,
                        "implied_rate_2m": float(fallback_row[5]) if fallback_row[5] is not None else None,
                        "implied_rate_3m": float(fallback_row[6]) if fallback_row[6] is not None else None,
                        "implied_rate_6m": float(fallback_row[7]) if fallback_row[7] is not None else None,
                        "rate_source": fallback_row[8],
                        "spx_div_yield": float(fallback_row[9]) if fallback_row[9] is not None else None,
                        "spx_div_yield_trail": float(fallback_row[9]) if fallback_row[9] is not None else None,
                        "net_carry_spot": float(fallback_row[10]) if fallback_row[10] is not None else None,
                        "net_carry_3m": float(fallback_row[11]) if fallback_row[11] is not None else None,
                        "as_of_date": fallback_row[12].isoformat() if fallback_row[12] else None,
                    }
                else:
                    result["rates"] = None

            # 2. Forward volatility (VIX + VIX/RV ratio)
            vol_row = conn.execute(text("""
                SELECT vix_close, realized_vol_20d, vix_rv_ratio
                FROM databento_es_features.forward_vol_1d
                WHERE date_utc <= :date AND vix_close IS NOT NULL
                ORDER BY date_utc DESC LIMIT 1
            """), {"date": target_date}).fetchone()

            if vol_row:
                result["volatility"] = {
                    "vix_close": float(vol_row[0]) if vol_row[0] is not None else None,
                    "realized_vol_20d": float(vol_row[1]) if vol_row[1] is not None else None,
                    "vix_rv_ratio": float(vol_row[2]) if vol_row[2] is not None else None,
                }
            else:
                result["volatility"] = None

            # 3. Events on and near this date (±2 trading days)
            events_rows = conn.execute(text("""
                SELECT date_utc, event_type, event_description,
                       is_pre_event_24h, is_post_event_24h
                FROM databento_es_features.event_calendar
                WHERE date_utc BETWEEN (CAST(:date AS date) - INTERVAL '3 days') AND (CAST(:date AS date) + INTERVAL '3 days')
                ORDER BY date_utc
            """), {"date": target_date}).fetchall()

            on_date_events = []
            nearby_events = []
            is_pre_event = False
            is_post_event = False

            for row in events_rows:
                evt = {
                    "date": row[0].isoformat() if row[0] else None,
                    "type": row[1],
                    "description": row[2],
                }
                if str(row[0]) == target_date:
                    if not row[1].endswith('_PRE') and not row[1].endswith('_POST'):
                        on_date_events.append(row[1])
                    if row[3]:
                        is_pre_event = True
                    if row[4]:
                        is_post_event = True
                if not row[1].endswith('_PRE') and not row[1].endswith('_POST'):
                    nearby_events.append(evt)

            result["events"] = {
                "on_date": on_date_events,
                "is_pre_event": is_pre_event,
                "is_post_event": is_post_event,
                "nearby": nearby_events,
            }

            # 4. Regime context (if available)
            regime_row = conn.execute(text("""
                SELECT vol_regime, trend_regime, in_roll_window,
                       days_to_next_roll, is_quad_witching
                FROM databento_es_features.regime_context_1d
                WHERE date_utc = :date
                LIMIT 1
            """), {"date": target_date}).fetchone()

            if regime_row:
                result["regime"] = {
                    "vol_regime": regime_row[0],
                    "trend_regime": regime_row[1],
                    "in_roll_window": regime_row[2],
                    "days_to_next_roll": regime_row[3],
                    "is_quad_witching": regime_row[4],
                }
            else:
                result["regime"] = None

            # 5. Rate change context (24h change for stability checks)
            rate_change_row = conn.execute(text("""
                SELECT
                    curr.sofr_spot - prev.sofr_spot AS sofr_1d_change,
                    curr.sofr_spot,
                    prev.sofr_spot AS sofr_prev
                FROM databento_es_features.external_rates_1d curr
                JOIN LATERAL (
                    SELECT sofr_spot
                    FROM databento_es_features.external_rates_1d
                    WHERE date_utc < curr.date_utc AND sofr_spot IS NOT NULL
                    ORDER BY date_utc DESC LIMIT 1
                ) prev ON TRUE
                WHERE curr.date_utc = :date
            """), {"date": target_date}).fetchone()

            if rate_change_row and rate_change_row[0] is not None:
                result["rate_stability"] = {
                    "sofr_1d_change_bps": round(float(rate_change_row[0]) * 10000, 1),
                    "is_stable": abs(float(rate_change_row[0]) * 10000) < 5.0,
                }
            else:
                result["rate_stability"] = None

            return {"date": target_date, **result}

    except Exception as e:
        logger.error(f"Error fetching forward data for {target_date}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/micro-detail/{target_date}/{symbol}")
async def get_micro_detail(
    target_date: str,
    symbol: str,
    start_time: Optional[str] = Query(None, description="Start time ISO (inclusive)"),
    end_time: Optional[str] = Query(None, description="End time ISO (exclusive)"),
):
    """
    Detailed microstructure data for a selected candle range.

    Queries four tables (each in its own connection to isolate failures):
      1. databento_es_trades.trades_agg_1m        (trade flow)
      2. databento_es_features.microstructure_toxicity  (VPIN, Kyle lambda, Amihud)
      3. databento_es_microstructure.micro_1m      (L1 imbalance, spread, depth)
      4. databento_es_features.book_shape_1m       (book slope, asymmetry, resilience)

    Returns aggregated summary + per-bar breakdown.
    """
    try:
        sym_upper = symbol.upper()

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def time_conditions(ts_col: str) -> tuple[str, dict]:
            conds, params = [], {}
            if start_time:
                conds.append(f"{ts_col} >= :start_time")
                params["start_time"] = start_time
            if end_time:
                conds.append(f"{ts_col} < :end_time")
                params["end_time"] = end_time
            return (" AND ".join(conds), params)

        def safe_float(v):
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        def avg(lst):
            return sum(lst) / len(lst) if lst else None

        def minute_key(ts):
            if ts is None:
                return None
            if hasattr(ts, 'replace'):
                return ts.replace(second=0, microsecond=0).isoformat()
            return str(ts)[:16]

        def _query(sql, params) -> list:
            """Run a query in its own connection so one failure can't cascade."""
            with engine.connect() as c:
                return c.execute(text(sql), params).fetchall()

        # ==================================================================
        # 1. trades_agg_1m
        # ==================================================================
        trades_rows = []
        try:
            tc, tp = time_conditions("timestamp_utc")
            where = "WHERE symbol = :symbol AND date_utc = :date"
            if tc:
                where += f" AND {tc}"
            tp.update({"symbol": sym_upper, "date": target_date})
            trades_rows = _query(f"""
                SELECT timestamp_utc, trade_count, total_volume,
                       buy_volume, sell_volume, avg_trade_size, vwap, trades_gt_100
                FROM databento_es_trades.trades_agg_1m
                {where} ORDER BY timestamp_utc
            """, tp)
        except Exception as e:
            logger.warning(f"micro-detail trades query failed: {e}")

        # ==================================================================
        # 2. microstructure_toxicity
        # ==================================================================
        tox_rows = []
        try:
            tc, tp = time_conditions("timestamp_utc")
            where = "WHERE symbol = :symbol"
            if tc:
                where += f" AND {tc}"
            tp["symbol"] = sym_upper
            tox_rows = _query(f"""
                SELECT timestamp_utc, vpin, toxicity_score, kyle_lambda,
                       amihud_ratio, quoted_spread_avg, effective_spread, spread_capture
                FROM databento_es_features.microstructure_toxicity
                {where} ORDER BY timestamp_utc
            """, tp)
        except Exception as e:
            logger.debug(f"micro-detail toxicity with symbol failed, retrying without: {e}")
            try:
                tc, tp = time_conditions("timestamp_utc")
                where = "WHERE 1=1"
                if tc:
                    where += f" AND {tc}"
                tox_rows = _query(f"""
                    SELECT timestamp_utc, vpin, toxicity_score, kyle_lambda,
                           amihud_ratio, quoted_spread_avg, effective_spread, spread_capture
                    FROM databento_es_features.microstructure_toxicity
                    {where} ORDER BY timestamp_utc
                """, tp)
            except Exception as e2:
                logger.warning(f"micro-detail toxicity fallback also failed: {e2}")

        # ==================================================================
        # 3. micro_1m  (L1 imbalance from microstructure schema)
        #    Columns: imbalance_l1_avg, imbalance_l1_std, spread_avg, bid_sz_l1_avg, ask_sz_l1_avg
        # ==================================================================
        micro_rows = []
        try:
            tc, tp = time_conditions("timestamp_utc")
            where = "WHERE symbol = :symbol"
            if tc:
                where += f" AND {tc}"
            tp["symbol"] = sym_upper
            micro_rows = _query(f"""
                SELECT timestamp_utc, imbalance_l1_avg, spread_avg,
                       bid_sz_l1_avg, ask_sz_l1_avg
                FROM databento_es_microstructure.micro_1m
                {where} ORDER BY timestamp_utc
            """, tp)
        except Exception as e:
            logger.debug(f"micro-detail micro_1m with symbol failed, retrying without: {e}")
            try:
                tc, tp = time_conditions("timestamp_utc")
                where = "WHERE 1=1"
                if tc:
                    where += f" AND {tc}"
                micro_rows = _query(f"""
                    SELECT timestamp_utc, imbalance_l1_avg, spread_avg,
                           bid_sz_l1_avg, ask_sz_l1_avg
                    FROM databento_es_microstructure.micro_1m
                    {where} ORDER BY timestamp_utc
                """, tp)
            except Exception as e2:
                logger.warning(f"micro-detail micro_1m fallback also failed: {e2}")

        # ==================================================================
        # 4. book_shape_1m
        # ==================================================================
        book_rows = []
        try:
            tc, tp = time_conditions("timestamp_utc")
            where = "WHERE symbol = :symbol"
            if tc:
                where += f" AND {tc}"
            tp["symbol"] = sym_upper
            book_rows = _query(f"""
                SELECT timestamp_utc, depth_ratio_l1, book_asymmetry,
                       book_slope_bid, book_slope_ask,
                       ticks_to_absorb_100, bid_resilience, ask_resilience
                FROM databento_es_features.book_shape_1m
                {where} ORDER BY timestamp_utc
            """, tp)
        except Exception as e:
            logger.debug(f"micro-detail book_shape with symbol failed, retrying without: {e}")
            try:
                tc, tp = time_conditions("timestamp_utc")
                where = "WHERE 1=1"
                if tc:
                    where += f" AND {tc}"
                book_rows = _query(f"""
                    SELECT timestamp_utc, depth_ratio_l1, book_asymmetry,
                           book_slope_bid, book_slope_ask,
                           ticks_to_absorb_100, bid_resilience, ask_resilience
                    FROM databento_es_features.book_shape_1m
                    {where} ORDER BY timestamp_utc
                """, tp)
            except Exception as e2:
                logger.warning(f"micro-detail book_shape fallback also failed: {e2}")

        # ==================================================================
        # Build per-bar lookup maps keyed by minute timestamp
        # ==================================================================
        tox_map = {}
        for r in tox_rows:
            k = minute_key(r[0])
            if k:
                tox_map[k] = r

        book_map = {}
        for r in book_rows:
            k = minute_key(r[0])
            if k:
                book_map[k] = r

        micro_map = {}
        for r in micro_rows:
            k = minute_key(r[0])
            if k:
                micro_map[k] = r

        # ==================================================================
        # Aggregate summaries
        # ==================================================================
        total_trades = sum(r[1] or 0 for r in trades_rows)
        total_volume = sum(r[2] or 0 for r in trades_rows)
        total_buy = sum(r[3] or 0 for r in trades_rows)
        total_sell = sum(r[4] or 0 for r in trades_rows)
        total_large = sum(r[7] or 0 for r in trades_rows)

        vwap_num = sum((safe_float(r[6]) or 0) * (r[2] or 0) for r in trades_rows)
        vwap = vwap_num / (total_volume or 1) if total_volume else None

        avg_trade_size = (
            sum(safe_float(r[5]) or 0 for r in trades_rows) / len(trades_rows)
            if trades_rows else None
        )

        vpin_vals = [safe_float(r[1]) for r in tox_rows if r[1] is not None]
        tox_vals = [safe_float(r[2]) for r in tox_rows if r[2] is not None]
        kl_vals = [safe_float(r[3]) for r in tox_rows if r[3] is not None]
        am_vals = [safe_float(r[4]) for r in tox_rows if r[4] is not None]
        spread_vals = [safe_float(r[5]) for r in tox_rows if r[5] is not None]

        dr_vals = [safe_float(r[1]) for r in book_rows if r[1] is not None]
        asym_vals = [safe_float(r[2]) for r in book_rows if r[2] is not None]

        # ==================================================================
        # Per-bar breakdown (trades table drives the bars list)
        # ==================================================================
        bars = []
        for r in trades_rows:
            ts = r[0]
            k = minute_key(ts)
            t_label = ts.strftime("%H:%M") if hasattr(ts, 'strftime') else str(ts)[11:16]

            vol = r[2] or 0
            buy_v = r[3] or 0
            sell_v = r[4] or 0
            buy_pct = (buy_v / (buy_v + sell_v) * 100) if (buy_v + sell_v) > 0 else None

            tox_r = tox_map.get(k)
            book_r = book_map.get(k)
            micro_r = micro_map.get(k)

            bars.append({
                "time": t_label,
                "volume": int(vol),
                "buy_volume": int(buy_v),
                "sell_volume": int(sell_v),
                "buy_pct": round(buy_pct, 2) if buy_pct is not None else None,
                "spread": safe_float(micro_r[2]) if micro_r else (safe_float(tox_r[5]) if tox_r else None),
                "imbalance": safe_float(book_r[1]) if book_r else None,
                "l1_imbalance": safe_float(micro_r[1]) if micro_r else None,
                "bid_depth": safe_float(micro_r[3]) if micro_r else None,
                "ask_depth": safe_float(micro_r[4]) if micro_r else None,
                "vpin": safe_float(tox_r[1]) if tox_r else None,
                "toxicity": safe_float(tox_r[2]) if tox_r else None,
                "book_asymmetry": safe_float(book_r[2]) if book_r else None,
                "kyle_lambda": safe_float(tox_r[3]) if tox_r else None,
            })

        # Fallback: build bars from tox if no trades
        if not bars and tox_rows:
            for r in tox_rows:
                ts = r[0]
                k = minute_key(ts)
                t_label = ts.strftime("%H:%M") if hasattr(ts, 'strftime') else str(ts)[11:16]
                book_r = book_map.get(k)
                micro_r = micro_map.get(k)
                bars.append({
                    "time": t_label,
                    "volume": None, "buy_volume": None, "sell_volume": None, "buy_pct": None,
                    "spread": safe_float(micro_r[2]) if micro_r else safe_float(r[5]),
                    "imbalance": safe_float(book_r[1]) if book_r else None,
                    "l1_imbalance": safe_float(micro_r[1]) if micro_r else None,
                    "bid_depth": safe_float(micro_r[3]) if micro_r else None,
                    "ask_depth": safe_float(micro_r[4]) if micro_r else None,
                    "vpin": safe_float(r[1]),
                    "toxicity": safe_float(r[2]),
                    "book_asymmetry": safe_float(book_r[2]) if book_r else None,
                    "kyle_lambda": safe_float(r[3]),
                })

        # ==================================================================
        # Build time_range
        # ==================================================================
        all_times = (
            [r[0] for r in trades_rows if r[0]] or
            [r[0] for r in tox_rows if r[0]] or
            []
        )
        if all_times:
            t_start = min(all_times)
            t_end = max(all_times)
            time_range = {
                "start": t_start.strftime("%H:%M") if hasattr(t_start, 'strftime') else str(t_start)[11:16],
                "end": t_end.strftime("%H:%M") if hasattr(t_end, 'strftime') else str(t_end)[11:16],
                "bars": len(bars),
            }
        else:
            time_range = {"start": start_time, "end": end_time, "bars": 0}

        return {
            "date": target_date,
            "symbol": sym_upper,
            "time_range": time_range,
            "trades": {
                "count": total_trades,
                "volume": total_volume,
                "buy_volume": total_buy,
                "sell_volume": total_sell,
                "large_trades": total_large,
                "vwap": round(vwap, 4) if vwap else None,
                "avg_size": round(avg_trade_size, 2) if avg_trade_size else None,
            } if trades_rows else None,
            "toxicity": {
                "avg_vpin": round(avg(vpin_vals), 4) if vpin_vals else None,
                "avg_toxicity_score": round(avg(tox_vals), 4) if tox_vals else None,
                "avg_kyle_lambda": round(avg(kl_vals), 6) if kl_vals else None,
                "avg_amihud": avg(am_vals),
                "avg_quoted_spread": round(avg(spread_vals), 4) if spread_vals else None,
            } if tox_rows else None,
            "book": {
                "avg_imbalance": round(avg(dr_vals), 4) if dr_vals else None,
                "avg_asymmetry": round(avg(asym_vals), 4) if asym_vals else None,
            } if book_rows else None,
            "bars": bars,
            "data_availability": {
                "trades_bars": len(trades_rows),
                "toxicity_bars": len(tox_rows),
                "book_bars": len(book_rows),
                "micro_bars": len(micro_rows),
            },
        }

    except Exception as e:
        logger.error(f"Error in micro-detail for {symbol} on {target_date}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check for DataBento API.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM databento_es_ohlcv_1d.es_continuous"))
            count = result.scalar()
            
            return {
                "status": "healthy",
                "es_continuous_rows": count,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"DataBento health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
