"""
Data models for trading data API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class TimeFrame(str, Enum):
    """Available timeframes for trading data"""
    ONE_MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    THIRTY_MINUTE = "30m"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    ONE_DAY = "1d"

class Symbol(str, Enum):
    """Available trading symbols"""
    ES = "es"
    EURUSD = "eurusd"
    SPY = "spy"

class TradingDataPoint(BaseModel):
    """Single trading data point response"""
    symbol: str
    timestamp: datetime
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    
    # Vector columns (optional)
    raw_ohlc_vec: Optional[List[float]] = None
    raw_ohlcv_vec: Optional[List[float]] = None
    norm_ohlc: Optional[List[float]] = None
    norm_ohlcv: Optional[List[float]] = None
    bert_ohlc: Optional[List[float]] = None
    bert_ohlcv: Optional[List[float]] = None

class TradingDataResponse(BaseModel):
    """Response model for trading data queries"""
    symbol: str
    timeframe: str
    count: int
    data: List[TradingDataPoint]

class TableInfo(BaseModel):
    """Information about a trading table"""
    table_name: str
    symbol: str
    timeframe: str
    row_count: Optional[int] = None
    latest_timestamp: Optional[datetime] = None
    earliest_timestamp: Optional[datetime] = None

class DatabaseStats(BaseModel):
    """Overall database statistics"""
    total_tables: int
    tables: List[TableInfo]
    total_rows: int

class QueryParams(BaseModel):
    """Query parameters for filtering trading data"""
    limit: int = Field(default=100, ge=1, le=10000, description="Number of records to return")
    offset: int = Field(default=0, ge=0, description="Number of records to skip")
    start_date: Optional[datetime] = Field(default=None, description="Start date for filtering")
    end_date: Optional[datetime] = Field(default=None, description="End date for filtering")
    include_vectors: bool = Field(default=False, description="Include vector columns in response")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now) 