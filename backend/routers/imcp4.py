"""
IMC Prosperity 4 API Router

Provides endpoints for:
- Tutorial round data (order book, trades, features)
- Product summaries and daily stats
- Strategy signals and analysis
- Backtesting results
"""

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text
from datetime import date, datetime
from typing import Optional, List
import logging
import json

from database import engine

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/imcp4",
    tags=["imcp4"]
)


@router.get("/health")
async def health():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM imcp4.order_book"))
            count = result.scalar()
            return {"status": "healthy", "order_book_rows": count}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@router.get("/overview")
async def get_overview():
    """Get a complete overview of all IMCP4 data in the database."""
    try:
        with engine.connect() as conn:
            tables = {}
            for table in ['order_book', 'trades', 'book_features', 'rolling_features', 'daily_summary', 'products']:
                result = conn.execute(text(f"SELECT COUNT(*) FROM imcp4.{table}"))
                tables[table] = result.scalar()

            products = conn.execute(text(
                "SELECT symbol, position_limit, description, product_type FROM imcp4.products ORDER BY symbol"
            ))
            product_list = [
                {"symbol": r[0], "position_limit": r[1], "description": r[2], "product_type": r[3]}
                for r in products.fetchall()
            ]

            rounds = conn.execute(text(
                "SELECT DISTINCT round, day FROM imcp4.order_book ORDER BY round, day"
            ))
            round_days = [{"round": r[0], "day": r[1]} for r in rounds.fetchall()]

            return {
                "tables": tables,
                "products": product_list,
                "round_days": round_days,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Overview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily-summary")
async def get_daily_summary():
    """Get daily summary stats for all products and days."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT round, day, product, open_mid, close_mid, high_mid, low_mid,
                       daily_range, daily_return, avg_spread, total_trades, 
                       total_volume, avg_trade_size, volatility
                FROM imcp4.daily_summary
                ORDER BY product, round, day
            """))
            rows = result.fetchall()
            columns = result.keys()
            return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mid-prices/{product}")
async def get_mid_prices(
    product: str,
    round: int = Query(0),
    day: Optional[int] = Query(None),
    downsample: int = Query(1, description="Take every Nth point")
):
    """Get mid price time series for a product."""
    try:
        with engine.connect() as conn:
            query = """
                SELECT timestamp, mid_price
                FROM imcp4.order_book
                WHERE product = :product AND round = :round
            """
            params = {"product": product.upper(), "round": round}

            if day is not None:
                query += " AND day = :day"
                params["day"] = day

            query += " ORDER BY day, timestamp"

            result = conn.execute(text(query), params)
            rows = result.fetchall()

            if downsample > 1:
                rows = rows[::downsample]

            return [{"timestamp": r[0], "mid_price": float(r[1])} for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/order-book/{product}")
async def get_order_book_snapshot(
    product: str,
    round: int = Query(0),
    day: int = Query(-1),
    timestamp: int = Query(0)
):
    """Get a single order book snapshot."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT bid_price_1, bid_volume_1, bid_price_2, bid_volume_2,
                       bid_price_3, bid_volume_3,
                       ask_price_1, ask_volume_1, ask_price_2, ask_volume_2,
                       ask_price_3, ask_volume_3,
                       mid_price, microprice, spread
                FROM imcp4.order_book
                WHERE product = :product AND round = :round AND day = :day AND timestamp = :ts
            """), {"product": product.upper(), "round": round, "day": day, "ts": timestamp})

            row = result.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Snapshot not found")

            cols = result.keys()
            return dict(zip(cols, [float(v) if v is not None else None for v in row]))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/book-features/{product}")
async def get_book_features(
    product: str,
    round: int = Query(0),
    day: Optional[int] = Query(None),
    downsample: int = Query(1)
):
    """Get book-level features (spread, imbalance, microprice, wall_mid) time series."""
    try:
        with engine.connect() as conn:
            query = """
                SELECT timestamp, mid_price, spread, microprice, weighted_mid,
                       bid_depth_total, ask_depth_total, book_imbalance,
                       wall_mid, spread_pct
                FROM imcp4.book_features
                WHERE product = :product AND round = :round
            """
            params = {"product": product.upper(), "round": round}
            if day is not None:
                query += " AND day = :day"
                params["day"] = day

            query += " ORDER BY day, timestamp"
            result = conn.execute(text(query), params)
            rows = result.fetchall()
            columns = result.keys()

            if downsample > 1:
                rows = rows[::downsample]

            return [
                {k: (float(v) if v is not None and k != 'timestamp' else v)
                 for k, v in zip(columns, row)}
                for row in rows
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rolling-features/{product}")
async def get_rolling_features(
    product: str,
    round: int = Query(0),
    day: Optional[int] = Query(None),
    downsample: int = Query(5)
):
    """Get rolling features (returns, vol, momentum, RSI, autocorrelation) time series."""
    try:
        with engine.connect() as conn:
            query = """
                SELECT timestamp, mid_price, return_1, return_5, return_10,
                       ema_10, ema_50, sma_20, sma_100,
                       realized_vol_20, realized_vol_100,
                       momentum_20, rsi_14, autocorr_1,
                       running_high, running_low, price_location
                FROM imcp4.rolling_features
                WHERE product = :product AND round = :round
            """
            params = {"product": product.upper(), "round": round}
            if day is not None:
                query += " AND day = :day"
                params["day"] = day

            query += " ORDER BY day, timestamp"
            result = conn.execute(text(query), params)
            rows = result.fetchall()
            columns = result.keys()

            if downsample > 1:
                rows = rows[::downsample]

            return [
                {k: (float(v) if v is not None and k != 'timestamp' else v)
                 for k, v in zip(columns, row)}
                for row in rows
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades/{product}")
async def get_trades(
    product: str,
    round: int = Query(0),
    day: Optional[int] = Query(None)
):
    """Get all trades for a product."""
    try:
        with engine.connect() as conn:
            query = """
                SELECT t.timestamp, t.price, t.quantity, t.buyer, t.seller, t.day
                FROM imcp4.trades t
                WHERE t.symbol = :product AND t.round = :round
            """
            params = {"product": product.upper(), "round": round}
            if day is not None:
                query += " AND t.day = :day"
                params["day"] = day

            query += " ORDER BY t.day, t.timestamp"
            result = conn.execute(text(query), params)
            rows = result.fetchall()
            columns = result.keys()
            return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spread-distribution/{product}")
async def get_spread_distribution(
    product: str,
    round: int = Query(0)
):
    """Get spread distribution for a product."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT spread, COUNT(*) as count
                FROM imcp4.book_features
                WHERE product = :product AND round = :round
                GROUP BY spread
                ORDER BY spread
            """), {"product": product.upper(), "round": round})

            rows = result.fetchall()
            total = sum(r[1] for r in rows)
            return [
                {"spread": float(r[0]), "count": r[1], "pct": r[1] / total * 100}
                for r in rows
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trade-size-distribution/{product}")
async def get_trade_size_distribution(
    product: str,
    round: int = Query(0)
):
    """Get trade size (quantity) distribution for a product."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT quantity, COUNT(*) as count
                FROM imcp4.trades
                WHERE symbol = :product AND round = :round
                GROUP BY quantity
                ORDER BY quantity
            """), {"product": product.upper(), "round": round})

            rows = result.fetchall()
            total = sum(r[1] for r in rows)
            return [
                {"quantity": r[0], "count": r[1], "pct": r[1] / total * 100}
                for r in rows
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/imbalance-return-correlation/{product}")
async def get_imbalance_return_correlation(
    product: str,
    round: int = Query(0),
    day: Optional[int] = Query(None),
    downsample: int = Query(5)
):
    """Get book imbalance vs future return scatter data."""
    try:
        with engine.connect() as conn:
            query = """
                SELECT b.timestamp, b.book_imbalance, r.return_1, r.return_5, r.return_10
                FROM imcp4.book_features b
                JOIN imcp4.rolling_features r USING (round, day, timestamp, product)
                WHERE b.product = :product AND b.round = :round
            """
            params = {"product": product.upper(), "round": round}
            if day is not None:
                query += " AND b.day = :day"
                params["day"] = day
            query += " ORDER BY b.day, b.timestamp"

            result = conn.execute(text(query), params)
            rows = result.fetchall()
            columns = result.keys()

            if downsample > 1:
                rows = rows[::downsample]

            return [
                {k: (float(v) if v is not None else None) for k, v in zip(columns, row)}
                for row in rows
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategy-signals/{product}")
async def get_strategy_signals(
    product: str,
    round: int = Query(0),
    day: int = Query(-1),
    downsample: int = Query(1)
):
    """
    Get combined price + book features + rolling features for strategy visualization.
    Includes mid_price, wall_mid, ema, spread, imbalance, vol, RSI all in one call.
    """
    try:
        with engine.connect() as conn:
            query = """
                SELECT 
                    o.timestamp,
                    o.mid_price,
                    o.microprice,
                    b.spread,
                    b.book_imbalance,
                    b.wall_mid,
                    b.bid_depth_total,
                    b.ask_depth_total,
                    r.ema_10,
                    r.ema_50,
                    r.sma_20,
                    r.realized_vol_20,
                    r.momentum_20,
                    r.rsi_14,
                    r.autocorr_1,
                    r.running_high,
                    r.running_low,
                    r.price_location
                FROM imcp4.order_book o
                JOIN imcp4.book_features b USING (round, day, timestamp, product)
                JOIN imcp4.rolling_features r USING (round, day, timestamp, product)
                WHERE o.product = :product AND o.round = :round AND o.day = :day
                ORDER BY o.timestamp
            """
            params = {"product": product.upper(), "round": round, "day": day}
            result = conn.execute(text(query), params)
            rows = result.fetchall()
            columns = list(result.keys())

            if downsample > 1:
                rows = rows[::downsample]

            return [
                {k: (float(v) if v is not None and k != 'timestamp' else v)
                 for k, v in zip(columns, row)}
                for row in rows
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/repos-summary")
async def get_repos_summary():
    """Return summary of cloned repos and their key strategies."""
    return {
        "repos": [
            {
                "name": "Frankfurt Hedgehogs",
                "placement": "2nd Global",
                "key_file": "FrankfurtHedgehogs_polished.py",
                "strategies": [
                    "StaticTrader (Rainforest Resin - fixed fair value MM)",
                    "DynamicTrader (Kelp - Wall Mid based MM)",
                    "InkTrader (Squid Ink - Olivia detection + mean reversion)",
                    "EtfTrader (Basket arbitrage with informed threshold adjustment)",
                    "OptionTrader (Black-Scholes + IV smile scalping)",
                    "CommodityTrader (Macarons conversion arbitrage)"
                ]
            },
            {
                "name": "Alpha Animals (UCSD)",
                "placement": "9th Global, 2nd USA",
                "key_file": "trader.py",
                "strategies": [
                    "Market making with fair value estimation",
                    "Basket synthetic value divergence trading",
                    "Black-Scholes + IV for volcanic vouchers",
                    "Olivia copy trading (process_insider_trades)",
                    "Macaron conversion arbitrage"
                ]
            },
            {
                "name": "P2 2nd Place (Stanford)",
                "placement": "2nd Global P2",
                "key_file": "round5/round5_v1.py",
                "strategies": [
                    "Adaptive market making with pennying",
                    "BlackScholes class for options",
                    "Gift basket synthetic arbitrage",
                    "Precomputed trade signal strings"
                ]
            },
            {
                "name": "jmerle Backtester",
                "placement": "Tool",
                "key_file": "prosperity3bt/runner.py",
                "strategies": [
                    "Full backtest engine with order matching",
                    "Compatible with P3 data format",
                    "Visualizer integration"
                ]
            }
        ]
    }
