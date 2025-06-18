"""
Trading data service for querying and processing trading data
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from sqlalchemy import text, func
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from models import TradingDataPoint, TradingDataResponse, TableInfo, DatabaseStats
from config import settings

logger = logging.getLogger(__name__)

class TradingDataService:
    """Service for handling trading data operations"""
    
    def __init__(self):
        self.schema = settings.SCHEMA
        
    def get_available_tables(self, db: Session) -> List[TableInfo]:
        """Get list of all available trading tables with metadata"""
        try:
            # First, get all table names
            table_query = text("""
                SELECT table_name
                FROM information_schema.tables 
                WHERE table_schema = :schema
                ORDER BY table_name
            """)
            
            table_results = db.execute(table_query, {"schema": self.schema}).fetchall()
            
            tables = []
            for table_row in table_results:
                table_name = table_row[0]
                
                # Parse symbol and timeframe from table name (e.g., "es_1d" -> symbol="es", timeframe="1d")
                parts = table_name.split("_")
                if len(parts) >= 2:
                    symbol = "_".join(parts[:-1])  # Handle cases like "eurusd_1d"
                    timeframe = parts[-1]
                else:
                    symbol = table_name
                    timeframe = "unknown"
                
                # Get metadata for each table individually
                try:
                    metadata_query = text(f"""
                        SELECT 
                            COUNT(*) as row_count,
                            MIN(timestamp) as earliest_timestamp,
                            MAX(timestamp) as latest_timestamp
                        FROM {self.schema}."{table_name}"
                    """)
                    
                    metadata_result = db.execute(metadata_query).fetchone()
                    
                    if metadata_result:
                        row_count = metadata_result[0]
                        earliest_timestamp = metadata_result[1]
                        latest_timestamp = metadata_result[2]
                    else:
                        row_count = 0
                        earliest_timestamp = None
                        latest_timestamp = None
                        
                except Exception as e:
                    logger.warning(f"Could not get metadata for table {table_name}: {e}")
                    row_count = 0
                    earliest_timestamp = None
                    latest_timestamp = None
                
                tables.append(TableInfo(
                    table_name=table_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    row_count=row_count,
                    earliest_timestamp=earliest_timestamp,
                    latest_timestamp=latest_timestamp
                ))
            
            return tables
            
        except SQLAlchemyError as e:
            logger.error(f"Error fetching table list: {e}")
            raise

    def get_database_stats(self, db: Session) -> DatabaseStats:
        """Get overall database statistics"""
        tables = self.get_available_tables(db)
        total_rows = sum(table.row_count or 0 for table in tables)
        
        return DatabaseStats(
            total_tables=len(tables),
            tables=tables,
            total_rows=total_rows
        )

    def get_available_columns(self, db: Session, table_name: str) -> List[str]:
        """Get list of columns that actually exist in the specified table"""
        try:
            query = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = :schema AND table_name = :table_name
                ORDER BY ordinal_position
            """)
            
            result = db.execute(query, {"schema": self.schema, "table_name": table_name}).fetchall()
            return [row[0] for row in result]
            
        except SQLAlchemyError as e:
            logger.error(f"Error checking columns for table {table_name}: {e}")
            # Return basic columns as fallback
            return ["symbol", "timestamp", "open", "high", "low", "close", "volume"]

    def get_trading_data(
        self,
        db: Session,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_vectors: bool = False,
        order: str = "desc",
        sort_by: str = "timestamp"
    ) -> TradingDataResponse:
        """Get trading data for a specific symbol and timeframe"""
        
        table_name = f"{symbol}_{timeframe}"
        
        # Get available columns for this table
        available_columns = self.get_available_columns(db, table_name)
        
        # Build the SELECT clause based on available columns
        base_columns = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        possible_vector_columns = ["raw_ohlc_vec", "raw_ohlcv_vec", "norm_ohlc", "norm_ohlcv", "BERT_ohlc", "BERT_ohlcv"]
        
        # Only include columns that actually exist in the table
        columns = [col for col in base_columns if col in available_columns]
        
        if include_vectors:
            existing_vector_columns = [col for col in possible_vector_columns if col in available_columns]
            columns.extend(existing_vector_columns)
            
            # Log which vector columns are available vs missing
            missing_vectors = [col for col in possible_vector_columns if col not in available_columns]
            if missing_vectors:
                logger.info(f"Vector columns not found in {table_name}: {missing_vectors}")
            if existing_vector_columns:
                logger.info(f"Vector columns available in {table_name}: {existing_vector_columns}")
        
        select_clause = ", ".join([f'"{col}"' for col in columns])
        
        # Build the WHERE clause
        where_conditions = []
        params = {"limit": limit, "offset": offset}
        
        if start_date:
            where_conditions.append("timestamp >= :start_date")
            params["start_date"] = start_date
            
        if end_date:
            where_conditions.append("timestamp <= :end_date")
            params["end_date"] = end_date
        
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # Validate sort_by column to prevent SQL injection
        valid_sort_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        if sort_by not in valid_sort_columns:
            sort_by = "timestamp"
        
        # Validate order
        if order.lower() not in ["asc", "desc"]:
            order = "desc"
        
        # Build the complete query with proper sorting
        query = text(f"""
            SELECT {select_clause}
            FROM {self.schema}."{table_name}"
            {where_clause}
            ORDER BY "{sort_by}" {order.upper()}
            LIMIT :limit OFFSET :offset
        """)
        
        try:
            result = db.execute(query, params).fetchall()
            
            # Get total count for pagination (if needed)
            count_query = text(f"""
                SELECT COUNT(*)
                FROM {self.schema}."{table_name}"
                {where_clause}
            """)
            
            # Remove limit/offset params for count query
            count_params = {k: v for k, v in params.items() if k not in ["limit", "offset"]}
            total_count = db.execute(count_query, count_params).scalar()
            
            # Convert results to TradingDataPoint objects
            data_points = []
            for row in result:
                row_dict = dict(zip(columns, row))
                
                # Convert vector strings back to lists if needed
                if include_vectors:
                    for vec_col in possible_vector_columns:
                        if vec_col in row_dict and row_dict[vec_col]:
                            try:
                                # Handle case where vectors are stored as strings
                                if isinstance(row_dict[vec_col], str):
                                    import ast
                                    row_dict[vec_col] = ast.literal_eval(row_dict[vec_col])
                            except (ValueError, SyntaxError):
                                row_dict[vec_col] = None
                
                data_points.append(TradingDataPoint(**row_dict))
            
            return TradingDataResponse(
                symbol=symbol,
                timeframe=timeframe,
                count=len(data_points),
                data=data_points,
                total_count=total_count
            )
            
        except SQLAlchemyError as e:
            logger.error(f"Error fetching trading data for {table_name}: {e}")
            raise

    def get_latest_data_point(self, db: Session, symbol: str, timeframe: str) -> Optional[TradingDataPoint]:
        """Get the most recent data point for a symbol/timeframe"""
        table_name = f"{symbol}_{timeframe}"
        
        query = text(f"""
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM {self.schema}."{table_name}"
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        
        try:
            result = db.execute(query).fetchone()
            if result:
                return TradingDataPoint(
                    symbol=result[0],
                    timestamp=result[1],
                    open=result[2],
                    high=result[3],
                    low=result[4],
                    close=result[5],
                    volume=result[6]
                )
            return None
            
        except SQLAlchemyError as e:
            logger.error(f"Error fetching latest data for {table_name}: {e}")
            raise

    def search_by_date_range(
        self,
        db: Session,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000
    ) -> List[TradingDataPoint]:
        """Search trading data within a specific date range"""
        
        return self.get_trading_data(
            db=db,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            include_vectors=False
        ).data 