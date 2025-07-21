"""
Trading data service for querying and processing trading data
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from sqlalchemy import text, func
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import numpy as np

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
        
        # Only include columns that actually exist in the table
        columns = [col for col in base_columns if col in available_columns]
        
        # Dynamically detect vector columns instead of hardcoding them
        vector_columns = []
        if include_vectors:
            # Look for any column that looks like a vector column
            # Common patterns: ends with '_vec', contains 'BERT', contains 'norm', contains 'iso_', etc.
            vector_patterns = ['_vec', 'BERT_', 'norm_', 'raw_ohlc', 'raw_ohlcv', 'iso_']
            for col in available_columns:
                if any(pattern in col for pattern in vector_patterns) and col not in base_columns:
                    vector_columns.append(col)
            
            columns.extend(vector_columns)
            
            # Log which vector columns are available
            if vector_columns:
                logger.info(f"Vector columns detected in {table_name}: {vector_columns}")
            else:
                logger.info(f"No vector columns found in {table_name}")
        
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
                if include_vectors and vector_columns:
                    for vec_col in vector_columns:
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

    def calculate_shape_similarity(
        self,
        db: Session,
        symbol: str,
        timeframe: str,
        vector_type: str,
        limit: int = 50,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """Calculate shape similarity matrix for ISO vectors"""
        
        # Get the trading data with vectors
        trading_data = self.get_trading_data(
            db=db,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
            include_vectors=True,
            order="desc",
            sort_by="timestamp"
        )
        
        if not trading_data.data:
            raise ValueError(f"No data found for {symbol}_{timeframe}")
        
        # Extract vectors and validate they exist
        vectors = []
        valid_candles = []
        
        for candle in trading_data.data:
            vector_data = getattr(candle, vector_type, None)
            if vector_data and isinstance(vector_data, list) and len(vector_data) > 0:
                vectors.append(np.array(vector_data))
                valid_candles.append(candle)
        
        if len(vectors) < 2:
            raise ValueError(f"Insufficient vector data for similarity analysis. Found {len(vectors)} valid vectors.")
        
        # Calculate similarity matrix
        n = len(vectors)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    similarity_matrix[i][j] = self._calculate_cosine_similarity(vectors[i], vectors[j])
        
        # Calculate statistics
        # Exclude diagonal (self-similarity) for statistics
        off_diagonal = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        statistics = {
            "average_similarity": float(np.mean(off_diagonal)),
            "max_similarity": float(np.max(off_diagonal)),
            "min_similarity": float(np.min(off_diagonal)),
            "std_similarity": float(np.std(off_diagonal)),
            "total_comparisons": len(off_diagonal),
            "high_similarity_pairs": int(np.sum(off_diagonal > 0.8)),
            "medium_similarity_pairs": int(np.sum((off_diagonal > 0.5) & (off_diagonal <= 0.8))),
            "low_similarity_pairs": int(np.sum(off_diagonal <= 0.5)),
            "pattern_diversity": self._calculate_pattern_diversity(off_diagonal)
        }
        
        # Import the response models here to avoid circular imports
        from routers.trading import ShapeSimilarityMatrix, ShapeSimilarityResponse
        
        similarity_matrix_obj = ShapeSimilarityMatrix(
            matrix=similarity_matrix.tolist(),
            candles=valid_candles,
            statistics=statistics
        )
        
        return ShapeSimilarityResponse(
            symbol=symbol,
            timeframe=timeframe,
            vector_type=vector_type,
            similarity_matrix=similarity_matrix_obj,
            count=len(valid_candles)
        )
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate shape similarity using multiple distance metrics for better discrimination"""
        try:
            # Handle zero vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # 1. Calculate Manhattan distance (L1 norm) - sensitive to shape differences
            manhattan_dist = np.sum(np.abs(vec1 - vec2))
            max_manhattan = np.sum(np.abs(vec1) + np.abs(vec2))
            manhattan_sim = 1.0 - (manhattan_dist / max_manhattan) if max_manhattan > 0 else 0.0
            
            # 2. Calculate Euclidean distance (L2 norm)
            euclidean_dist = np.linalg.norm(vec1 - vec2)
            max_euclidean = np.linalg.norm(vec1) + np.linalg.norm(vec2)
            euclidean_sim = 1.0 - (euclidean_dist / max_euclidean) if max_euclidean > 0 else 0.0
            
            # 3. Calculate normalized correlation
            vec1_centered = vec1 - np.mean(vec1)
            vec2_centered = vec2 - np.mean(vec2)
            correlation = np.corrcoef(vec1_centered, vec2_centered)[0, 1]
            correlation = 0.0 if np.isnan(correlation) else correlation
            
            # 4. Calculate cosine similarity
            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            
            # Combine multiple metrics with aggressive weighting for discrimination
            # Emphasize Manhattan distance as it's more sensitive to shape differences
            combined_similarity = (
                0.4 * manhattan_sim +      # Most weight on Manhattan (shape-sensitive)
                0.25 * euclidean_sim +     # Some weight on Euclidean
                0.2 * (correlation + 1) / 2 +  # Convert correlation to 0-1 range
                0.15 * (cosine_sim + 1) / 2    # Convert cosine to 0-1 range
            )
            
            # Apply aggressive contrast enhancement to spread out values
            # Use a power function to emphasize differences
            if combined_similarity > 0.5:
                # For high similarities, make them even higher but with diminishing returns
                enhanced = 0.5 + 0.5 * np.power((combined_similarity - 0.5) * 2, 1.5)
            else:
                # For low similarities, make them even lower
                enhanced = 0.5 * np.power(combined_similarity * 2, 0.7)
            
            # Apply final scaling to spread values more
            # Scale to ensure we get good range: compress high values, expand low values
            final_similarity = np.power(enhanced, 1.2)
            
            # Convert back to -1 to 1 range but bias toward lower values
            result = (final_similarity * 2) - 1
            
            # Additional aggressive compression for high values
            if result > 0.3:
                result = 0.3 + 0.7 * np.power((result - 0.3) / 0.7, 2.0)
            
            return float(np.clip(result, -1.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Error calculating shape similarity: {e}")
            return 0.0
    
    def _calculate_pattern_diversity(self, similarities: np.ndarray) -> str:
        """Calculate pattern diversity based on similarity distribution"""
        avg_sim = np.mean(similarities)
        
        if avg_sim < 0.3:
            return "High Diversity"
        elif avg_sim < 0.6:
            return "Medium Diversity"
        else:
            return "Low Diversity"

    def get_labels(self, db: Session, symbol: str, timeframe: str) -> list:
        """Fetch all rows from labels.{symbol}{timeframe}_labeled or labels.{symbol}_{timeframe}_labeled table"""
        from config import settings
        # Try both naming conventions
        table_names = [
            f"{symbol}{timeframe}_labeled",
            f"{symbol}_{timeframe}_labeled"
        ]
        chosen_table = None
        for table_name in table_names:
            check_table_query = text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = '{settings.LABELS_SCHEMA}' 
                    AND table_name = '{table_name}'
                );
            """)
            try:
                table_exists = db.execute(check_table_query).scalar()
                if table_exists:
                    chosen_table = table_name
                    break
            except Exception as e:
                continue
        if not chosen_table:
            return []
        try:
            query = text(f"""
                SELECT id, label, value, pointer
                FROM {settings.LABELS_SCHEMA}.{chosen_table}
                ORDER BY id
            """)
            result = db.execute(query)
            rows = result.fetchall()
            return [
                {
                    "id": row[0],
                    "label": row[1],
                    "value": row[2],
                    "pointer": row[3],
                }
                for row in rows
            ]
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error accessing labels table {chosen_table}: {e}")
            return []

    def get_spy1h_labels(self, db: Session) -> list:
        """Fetch all rows from labels.spy1h_labeled table"""
        from config import settings
        query = text(f"""
            SELECT id, label, value, pointer
            FROM {settings.LABELS_SCHEMA}.spy1h_labeled
            ORDER BY id
        """)
        result = db.execute(query)
        rows = result.fetchall()
        # Convert to list of dicts
        return [
            {
                "id": row[0],
                "label": row[1],
                "value": row[2],
                "pointer": row[3],
            }
            for row in rows
        ]

    def get_spy1h_swings_labels(self, db: Session) -> list:
        """Fetch all rows from labels.spy1h_swings table"""
        from config import settings
        query = text(f"""
            SELECT id, label, value, pointer
            FROM {settings.LABELS_SCHEMA}.spy1h_swings
            ORDER BY id
        """)
        result = db.execute(query)
        rows = result.fetchall()
        # Convert to list of dicts
        return [
            {
                "id": row[0],
                "label": row[1],
                "value": row[2],
                "pointer": row[3],
            }
            for row in rows
        ]

    def get_swing_labels(self, db: Session, symbol: str, timeframe: str) -> list:
        """Fetch all rows from labels.{symbol}{timeframe}_swings table"""
        from config import settings
        # Try both naming conventions for swing tables
        table_names = [
            f"{symbol}{timeframe}_swings",
            f"{symbol}_{timeframe}_swings"
        ]
        chosen_table = None
        for table_name in table_names:
            check_table_query = text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = '{settings.LABELS_SCHEMA}' 
                    AND table_name = '{table_name}'
                );
            """)
            try:
                table_exists = db.execute(check_table_query).scalar()
                if table_exists:
                    chosen_table = table_name
                    break
            except Exception as e:
                continue
        if not chosen_table:
            return []
        try:
            query = text(f"""
                SELECT id, label, value, pointer
                FROM {settings.LABELS_SCHEMA}.{chosen_table}
                ORDER BY id
            """)
            result = db.execute(query)
            rows = result.fetchall()
            return [
                {
                    "id": row[0],
                    "label": row[1],
                    "value": row[2],
                    "pointer": row[3],
                }
                for row in rows
            ]
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error accessing swing table {chosen_table}: {e}")
            return []

    def get_fvg_labels(self, db: Session, symbol: str, timeframe: str) -> list:
        """Fetch all rows from labels.{symbol}{timeframe}_fvg table"""
        from config import settings
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"🔍 FVG REQUEST: symbol='{symbol}', timeframe='{timeframe}'")
        
        # Try both naming conventions for FVG tables
        table_names = [
            f"{symbol}{timeframe}_fvg",
            f"{symbol}_{timeframe}_fvg"
        ]
        
        logger.info(f"🔍 Checking FVG table names: {table_names}")
        logger.info(f"🔍 Using labels schema: '{settings.LABELS_SCHEMA}'")
        
        chosen_table = None
        for table_name in table_names:
            check_table_query = text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = '{settings.LABELS_SCHEMA}' 
                    AND table_name = '{table_name}'
                );
            """)
            try:
                logger.info(f"🔍 Checking if table exists: {settings.LABELS_SCHEMA}.{table_name}")
                table_exists = db.execute(check_table_query).scalar()
                logger.info(f"🔍 Table {table_name} exists: {table_exists}")
                if table_exists:
                    chosen_table = table_name
                    logger.info(f"✅ Found FVG table: {chosen_table}")
                    break
            except Exception as e:
                logger.warning(f"❌ Error checking table {table_name}: {e}")
                continue
        if not chosen_table:
            logger.warning(f"❌ No FVG table found for {symbol}{timeframe}")
            return None  # Return None to indicate table doesn't exist
        
        try:
            query = text(f"""
                SELECT id, label, value, color_order, pointer
                FROM {settings.LABELS_SCHEMA}.{chosen_table}
                ORDER BY id
            """)
            logger.info(f"🔍 Executing FVG query: {query}")
            result = db.execute(query)
            rows = result.fetchall()
            logger.info(f"✅ Found {len(rows)} FVG rows in {chosen_table}")
            return [
                {
                    "id": row[0],
                    "label": row[1],
                    "value": row[2],
                    "color_order": row[3],
                    "pointer": row[4],
                }
                for row in rows
            ]
        except Exception as e:
            logger.warning(f"❌ Error accessing FVG table {chosen_table}: {e}")
            return [] 