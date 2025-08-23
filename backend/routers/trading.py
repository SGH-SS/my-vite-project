"""
Trading data API routes
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import logging

from database import get_db
from models import (
    TradingDataResponse, 
    DatabaseStats, 
    TradingDataPoint, 
    TableInfo,
    Symbol,
    TimeFrame,
    ErrorResponse
)
from pydantic import BaseModel
from typing import Dict, Any
from services.trading_service import TradingDataService
from services.model_inference import model_inference_service
from services.csv_model_service import csv_model_service

# Shape similarity response models
class ShapeSimilarityMatrix(BaseModel):
    """Shape similarity matrix response"""
    matrix: List[List[float]]
    candles: List[TradingDataPoint]
    statistics: Dict[str, Any]
    
class ShapeSimilarityResponse(BaseModel):
    """Shape similarity analysis response"""
    symbol: str
    timeframe: str
    vector_type: str
    similarity_matrix: ShapeSimilarityMatrix
    count: int

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/trading", tags=["trading"])

# Initialize the service
trading_service = TradingDataService()

@router.get("/health", summary="Health check for trading API")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@router.get("/test-fvg", summary="Test FVG endpoint connectivity")
async def test_fvg():
    """Test endpoint to verify FVG routes are working"""
    logger.info("🧪 TEST FVG ENDPOINT CALLED")
    return {"status": "FVG endpoint is accessible", "timestamp": datetime.now()}

@router.get("/stats", response_model=DatabaseStats, summary="Get database statistics")
async def get_database_stats(db: Session = Depends(get_db)):
    """Get overall statistics about the trading database"""
    try:
        return trading_service.get_database_stats(db)
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tables", response_model=List[TableInfo], summary="Get all available trading tables")
async def get_tables(db: Session = Depends(get_db)):
    """Get list of all available trading data tables with metadata"""
    try:
        return trading_service.get_available_tables(db)
    except Exception as e:
        logger.error(f"Error getting tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data/{symbol}/{timeframe}", response_model=TradingDataResponse, summary="Get trading data")
async def get_trading_data(
    symbol: str,
    timeframe: str,
    limit: int = Query(default=100, ge=1, le=10000, description="Number of records to return"),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    start_date: Optional[datetime] = Query(default=None, description="Start date (ISO format)"),
    end_date: Optional[datetime] = Query(default=None, description="End date (ISO format)"),
    include_vectors: bool = Query(default=False, description="Include vector columns"),
    order: str = Query(default="desc", regex="^(asc|desc)$", description="Sort order: asc or desc"),
    sort_by: str = Query(default="timestamp", description="Column to sort by"),
    db: Session = Depends(get_db)
):
    """
    Get trading data for a specific symbol and timeframe
    
    - **symbol**: Trading symbol (es, eurusd, spy)
    - **timeframe**: Time interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
    - **limit**: Number of records to return (max 10,000)
    - **offset**: Number of records to skip for pagination
    - **start_date**: Filter records after this date
    - **end_date**: Filter records before this date
    - **include_vectors**: Include ML vector columns in response
    - **order**: Sort order (asc for oldest first, desc for newest first)
    - **sort_by**: Column to sort by (default: timestamp)
    """
    try:
        return trading_service.get_trading_data(
            db=db,
            symbol=symbol.lower(),
            timeframe=timeframe.lower(),
            limit=limit,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
            include_vectors=include_vectors,
            order=order,
            sort_by=sort_by
        )
    except Exception as e:
        logger.error(f"Error getting trading data for {symbol}_{timeframe}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/{symbol}/{timeframe}", summary="Run model prediction for recent candles")
async def predict_for_recent(
    symbol: str,
    timeframe: str,
    limit: int = Query(default=50, ge=1, le=500),
    model: str = Query(default="auto", description="Model name or 'auto'"),
    train_cutoff: str = Query(default="", description="ISO timestamp of last training candle (optional)"),
    db: Session = Depends(get_db)
):
    """
    Predict next-direction probabilities for recent candles using preloaded .joblib models.

    - For now supports: SPY 1d (GB), SPY 4h (GB, LightGBM_Financial)
    - Returns per-candle predictions with probability and confidence plus coverage counts.
    """
    try:
        symbol = symbol.lower()
        timeframe = timeframe.lower()
        if symbol != "spy" or timeframe not in ("1d", "4h"):
            raise HTTPException(status_code=400, detail="Only spy 1d/4h supported for now")

        # Choose default model if auto
        if model == "auto":
            if timeframe == "1d":
                model_name = "GradientBoosting"
            else:  # 4h
                model_name = (
                    "LightGBM_Financial" if model_inference_service.model_available("LightGBM_Financial", "4h")
                    else "GradientBoosting"
                )
        else:
            model_name = model

        if not model_inference_service.model_available(model_name, timeframe):
            raise HTTPException(status_code=404, detail=f"Model not available: {model_name} {timeframe}")

        # Pull recent candles from DB in ascending order for feature building
        data_response = trading_service.get_trading_data(
            db=db,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            offset=0,
            include_vectors=True,
            order="asc",
            sort_by="timestamp"
        )

        rows = [d.dict() for d in data_response.data]
        results, coverage = model_inference_service.predict_for_rows(
            model_name=model_name,
            timeframe=timeframe,
            rows=rows,
            train_cutoff=train_cutoff or None
        )

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "model": model_name,
            "count": len(results),
            "coverage": coverage,
            "predictions": [r.__dict__ for r in results],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running prediction for {symbol}_{timeframe}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latest/{symbol}/{timeframe}", response_model=TradingDataPoint, summary="Get latest data point")
async def get_latest_data(
    symbol: str,
    timeframe: str,
    db: Session = Depends(get_db)
):
    """Get the most recent trading data point for a symbol/timeframe"""
    try:
        data_point = trading_service.get_latest_data_point(
            db=db,
            symbol=symbol.lower(),
            timeframe=timeframe.lower()
        )
        if not data_point:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for {symbol}_{timeframe}"
            )
        return data_point
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest data for {symbol}_{timeframe}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/symbols", summary="Get available symbols")
async def get_symbols():
    """Get list of available trading symbols"""
    return {
        "symbols": [symbol.value for symbol in Symbol],
        "count": len(Symbol)
    }

@router.get("/timeframes", summary="Get available timeframes")
async def get_timeframes():
    """Get list of available timeframes"""
    return {
        "timeframes": [tf.value for tf in TimeFrame],
        "count": len(TimeFrame)
    }

@router.get("/search/{symbol}/{timeframe}", response_model=List[TradingDataPoint], summary="Search by date range")
async def search_by_date_range(
    symbol: str,
    timeframe: str,
    start_date: datetime = Query(..., description="Start date (required)"),
    end_date: datetime = Query(..., description="End date (required)"),
    limit: int = Query(default=1000, ge=1, le=10000, description="Max records to return"),
    db: Session = Depends(get_db)
):
    """Search trading data within a specific date range"""
    try:
        if start_date >= end_date:
            raise HTTPException(
                status_code=400,
                detail="start_date must be before end_date"
            )
        
        return trading_service.search_by_date_range(
            db=db,
            symbol=symbol.lower(),
            timeframe=timeframe.lower(),
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching data for {symbol}_{timeframe}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary/{symbol}", summary="Get symbol summary across all timeframes")
async def get_symbol_summary(
    symbol: str,
    db: Session = Depends(get_db)
):
    """Get summary statistics for a symbol across all available timeframes"""
    try:
        tables = trading_service.get_available_tables(db)
        symbol_tables = [t for t in tables if t.symbol == symbol.lower()]
        
        if not symbol_tables:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol: {symbol}"
            )
        
        # Get latest data point for each timeframe
        summary = {
            "symbol": symbol.lower(),
            "timeframes": {}
        }
        
        for table in symbol_tables:
            latest = trading_service.get_latest_data_point(
                db=db,
                symbol=symbol.lower(),
                timeframe=table.timeframe
            )
            summary["timeframes"][table.timeframe] = {
                "table_name": table.table_name,
                "row_count": table.row_count,
                "earliest_timestamp": table.earliest_timestamp,
                "latest_timestamp": table.latest_timestamp,
                "latest_data": latest.dict() if latest else None
            }
        
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting summary for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/shape-similarity/{symbol}/{timeframe}", response_model=ShapeSimilarityResponse, summary="Get shape similarity analysis")
async def get_shape_similarity(
    symbol: str,
    timeframe: str,
    vector_type: str = Query(..., description="Vector type (must be ISO vector)"),
    limit: int = Query(default=50, ge=5, le=100, description="Number of candles to analyze"),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    start_date: Optional[datetime] = Query(default=None, description="Start date (ISO format)"),
    end_date: Optional[datetime] = Query(default=None, description="End date (ISO format)"),
    db: Session = Depends(get_db)
):
    """
    Calculate shape similarity matrix for ISO vectors
    
    - **symbol**: Trading symbol (es, eurusd, spy)
    - **timeframe**: Time interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
    - **vector_type**: Must be an ISO vector type (iso_ohlc, iso_ohlcv)
    - **limit**: Number of candles to analyze (max 100 for performance)
    - **offset**: Number of records to skip for pagination
    - **start_date**: Filter records after this date
    - **end_date**: Filter records before this date
    """
    try:
        # Validate that this is an ISO vector
        if not vector_type.startswith('iso_'):
            raise HTTPException(
                status_code=400,
                detail="Shape similarity analysis is only available for ISO vectors (iso_ohlc, iso_ohlcv)"
            )
        
        return trading_service.calculate_shape_similarity(
            db=db,
            symbol=symbol.lower(),
            timeframe=timeframe.lower(),
            vector_type=vector_type,
            limit=limit,
            offset=offset,
            start_date=start_date,
            end_date=end_date
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating shape similarity for {symbol}_{timeframe}_{vector_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/labels/{symbol}/{timeframe}", summary="Get all labels from labeled table")
async def get_labels(db: Session = Depends(get_db), symbol: str = None, timeframe: str = None):
    """Fetch all rows from labels.{symbol}{timeframe}_labeled table"""
    try:
        labels = trading_service.get_labels(db, symbol, timeframe)
        # Return empty list if no labels found (table doesn't exist or is empty)
        return labels
    except Exception as e:
        logger.error(f"Error fetching labels for {symbol}{timeframe}: {e}")
        # Return empty list instead of 500 error for missing tables
        return []

@router.get("/labels/spy1h", summary="Get all labels from spy1h_labeled table")
async def get_spy1h_labels(db: Session = Depends(get_db)):
    """Fetch all rows from labels.spy1h_labeled table"""
    try:
        return trading_service.get_spy1h_labels(db)
    except Exception as e:
        logger.error(f"Error fetching spy1h labels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/labels/spy1h_swings", summary="Get all swing labels from spy1h_swings table")
async def get_spy1h_swings_labels(db: Session = Depends(get_db)):
    """Fetch all rows from labels.spy1h_swings table"""
    try:
        return trading_service.get_spy1h_swings_labels(db)
    except Exception as e:
        logger.error(f"Error fetching swing labels for spy1h_swings: {e}")
        return []

@router.get("/swing-labels/{symbol}/{timeframe}", summary="Get all swing labels from swings table")
async def get_swing_labels(db: Session = Depends(get_db), symbol: str = None, timeframe: str = None):
    """Fetch all rows from labels.{symbol}{timeframe}_swings table"""
    try:
        return trading_service.get_swing_labels(db, symbol, timeframe)
    except Exception as e:
        logger.error(f"Error fetching swing labels for {symbol}{timeframe}: {e}")
        return []

@router.get("/fvg-labels/{symbol}/{timeframe}", summary="Get all FVG labels from FVG table")
async def get_fvg_labels(db: Session = Depends(get_db), symbol: str = None, timeframe: str = None):
    """Fetch all rows from labels.{symbol}{timeframe}_fvg table"""
    logger.info(f"🟢 FVG API ENDPOINT CALLED: /fvg-labels/{symbol}/{timeframe}")
    try:
        result = trading_service.get_fvg_labels(db, symbol, timeframe)
        
        # Check if we got None (table doesn't exist) vs empty array (table exists but no data)
        if result is None:
            logger.warning(f"❌ FVG table not found for {symbol}{timeframe}")
            raise HTTPException(status_code=404, detail=f"FVG table not found for {symbol}{timeframe}")
        
        logger.info(f"🟢 FVG API RETURNING: {len(result)} records for {symbol}{timeframe}")
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise
    except Exception as e:
        logger.error(f"❌ Error in FVG API endpoint for {symbol}{timeframe}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching FVG labels: {str(e)}")

@router.get("/model-results/{symbol}/{timeframe}", summary="Get DB-backed model results for symbol/timeframe")
async def get_model_results(symbol: str, timeframe: str, db: Session = Depends(get_db)):
    """Return predictions for all available models for symbol/timeframe from v1_models schema.

    Matches on timestamp_utc; frontend will reconcile with candles.
    """
    try:
        symbol = symbol.lower()
        timeframe = timeframe.lower()
        return trading_service.get_model_results_for_timeframe(db, symbol, timeframe)
    except Exception as e:
        logger.error(f"Error fetching model results for {symbol}_{timeframe}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/csv-models", summary="Get available CSV model predictions")
async def get_csv_models():
    """Get list of available CSV-based model predictions with metadata"""
    try:
        return csv_model_service.get_available_models()
    except Exception as e:
        logger.error(f"Error getting CSV models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/csv-predictions/{symbol}/{timeframe}", summary="Get all CSV model predictions for symbol/timeframe")
async def get_csv_predictions(symbol: str, timeframe: str):
    """Get all available CSV model predictions for a specific symbol/timeframe"""
    try:
        symbol = symbol.lower()
        timeframe = timeframe.lower()
        
        predictions = csv_model_service.get_predictions_for_timeframe(symbol, timeframe)
        available_models = csv_model_service.get_models_for_symbol_timeframe(symbol, timeframe)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "available_models": available_models,
            "predictions": predictions,
            "count": len(predictions)
        }
    except Exception as e:
        logger.error(f"Error getting CSV predictions for {symbol}_{timeframe}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/csv-prediction/{model_key}/{timestamp}", summary="Get specific prediction by model and timestamp")
async def get_csv_prediction_by_timestamp(model_key: str, timestamp: str):
    """Get a specific prediction from a CSV model by timestamp"""
    try:
        # Parse timestamp
        target_timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        prediction = csv_model_service.get_prediction_for_timestamp(model_key, target_timestamp)
        
        if not prediction:
            raise HTTPException(
                status_code=404,
                detail=f"No prediction found for {model_key} at {timestamp}"
            )
        
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {e}")
    except Exception as e:
        logger.error(f"Error getting CSV prediction for {model_key} at {timestamp}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 