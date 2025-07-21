#!/usr/bin/env python3
"""
Startup script for the Trading Data API
"""

import os
import sys
import subprocess
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    return logging.getLogger(__name__)

def check_requirements():
    """Check if required dependencies are installed"""
    logger = logging.getLogger(__name__)
    try:
        import fastapi
        import uvicorn
        import sqlalchemy
        import psycopg2
        import pandas
        logger.info("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Please run: pip install -r backend/requirements.txt")
        return False

def test_database_connection():
    """Test database connection"""
    logger = logging.getLogger(__name__)
    try:
        # Add backend directory to Python path
        backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
        sys.path.insert(0, backend_dir)
        
        from database import test_connection
        if test_connection():
            logger.info("✅ Database connection successful")
            return True
        else:
            logger.error("❌ Database connection failed")
            return False
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False

def show_available_routes_and_tables():
    """Display all available API routes and check which tables exist"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("🛣️  AVAILABLE API ROUTES & TABLES")
    logger.info("=" * 80)
    
    try:
        # Add backend directory to Python path
        backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
        sys.path.insert(0, backend_dir)
        
        from database import get_db
        from services.trading_service import TradingDataService
        from sqlalchemy import text
        
        # Get database session
        db_gen = get_db()
        db = next(db_gen)
        
        # Initialize trading service
        trading_service = TradingDataService()
        
        # Core API Routes
        logger.info("📊 CORE TRADING DATA ROUTES:")
        logger.info("  GET /api/trading/data/{symbol}/{timeframe}")
        logger.info("  GET /api/trading/stats")
        logger.info("  GET /api/trading/tables")
        logger.info("  GET /api/trading/date-ranges/{symbol}/{timeframe}")
        logger.info("  GET /api/trading/shape-similarity/{symbol}/{timeframe}")
        
        # Labels Routes
        logger.info("\n🏷️  LABELS ROUTES:")
        logger.info("  GET /api/trading/labels/{symbol}/{timeframe}")
        logger.info("  GET /api/trading/swing-labels/{symbol}/{timeframe}")
        logger.info("  GET /api/trading/fvg-labels/{symbol}/{timeframe}")
        logger.info("  GET /api/trading/labels/spy1h (legacy)")
        logger.info("  GET /api/trading/labels/spy1h_swings (legacy)")
        
        # Check available tables
        logger.info("\n📋 AVAILABLE DATABASE TABLES:")
        
        # Get all tables from the main schema
        from config import settings
        
        # Get main schema tables first (just names)
        main_tables_query = text(f"""
            SELECT table_name
            FROM information_schema.tables 
            WHERE table_schema = '{settings.SCHEMA}'
            ORDER BY table_name
        """)
        
        main_table_names = db.execute(main_tables_query).fetchall()
        logger.info(f"\n  📈 MAIN SCHEMA ({settings.SCHEMA}) - {len(main_table_names)} tables:")
        
        # Get row counts for main tables safely
        for (table_name,) in main_table_names:
            try:
                count_query = text(f'SELECT COUNT(*) FROM {settings.SCHEMA}."{table_name}"')
                row_count = db.execute(count_query).scalar()
                logger.info(f"    • {table_name} ({row_count:,} rows)")
            except Exception as e:
                logger.info(f"    • {table_name} (error getting count: {e})")
        
        # Get all tables from the labels schema
        labels_tables_query = text(f"""
            SELECT table_name
            FROM information_schema.tables 
            WHERE table_schema = '{settings.LABELS_SCHEMA}'
            ORDER BY table_name
        """)
        
        label_table_names = db.execute(labels_tables_query).fetchall()
        logger.info(f"\n  🏷️  LABELS SCHEMA ({settings.LABELS_SCHEMA}) - {len(label_table_names)} tables:")
        
        # Categorize label tables
        tjr_tables = []
        swing_tables = []
        fvg_tables = []
        other_tables = []
        
        for (table_name,) in label_table_names:
            try:
                count_query = text(f'SELECT COUNT(*) FROM {settings.LABELS_SCHEMA}."{table_name}"')
                row_count = db.execute(count_query).scalar()
                
                if 'labeled' in table_name:
                    tjr_tables.append((table_name, row_count))
                elif 'swing' in table_name:
                    swing_tables.append((table_name, row_count))
                elif 'fvg' in table_name:
                    fvg_tables.append((table_name, row_count))
                else:
                    other_tables.append((table_name, row_count))
            except Exception as e:
                logger.info(f"    • {table_name} (error getting count: {e})")
        
        if tjr_tables:
            logger.info(f"\n    🎯 TJR LABELED TABLES ({len(tjr_tables)}):")
            for table_name, row_count in tjr_tables:
                logger.info(f"      • {table_name} ({row_count:,} rows)")
        
        if swing_tables:
            logger.info(f"\n    🔵 SWING TABLES ({len(swing_tables)}):")
            for table_name, row_count in swing_tables:
                logger.info(f"      • {table_name} ({row_count:,} rows)")
        
        if fvg_tables:
            logger.info(f"\n    🟢 FVG TABLES ({len(fvg_tables)}):")
            for table_name, row_count in fvg_tables:
                logger.info(f"      • {table_name} ({row_count:,} rows)")
        else:
            logger.info(f"\n    🟢 FVG TABLES (0):")
            logger.info(f"      ⚠️  No FVG tables found - this explains the 404 errors!")
        
        if other_tables:
            logger.info(f"\n    📄 OTHER LABEL TABLES ({len(other_tables)}):")
            for table_name, row_count in other_tables:
                logger.info(f"      • {table_name} ({row_count:,} rows)")
        
        # Check specific SPY timeframes for each label type
        logger.info("\n🎯 SPY TIMEFRAME AVAILABILITY CHECK:")
        spy_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        
        for tf in spy_timeframes:
            # Check TJR
            tjr_available = any(f'spy{tf}_labeled' in table[0] or f'spy_{tf}_labeled' in table[0] for table in tjr_tables)
            # Check Swing  
            swing_available = any(f'spy{tf}_swing' in table[0] or f'spy_{tf}_swing' in table[0] for table in swing_tables)
            # Check FVG
            fvg_available = any(f'spy{tf}_fvg' in table[0] or f'spy_{tf}_fvg' in table[0] for table in fvg_tables)
            
            tjr_status = "✅" if tjr_available else "❌"
            swing_status = "✅" if swing_available else "❌"
            fvg_status = "✅" if fvg_available else "❌"
            
            logger.info(f"  SPY {tf:>2}: TJR {tjr_status} | Swing {swing_status} | FVG {fvg_status}")
        
        logger.info("\n📝 SUMMARY:")
        logger.info(f"  • Main tables: {len(main_table_names)}")
        logger.info(f"  • TJR labeled tables: {len(tjr_tables)}")
        logger.info(f"  • Swing tables: {len(swing_tables)}")
        logger.info(f"  • FVG tables: {len(fvg_tables)}")
        
        if len(fvg_tables) == 0:
            logger.info("\n⚠️  FVG TABLES MISSING:")
            logger.info("  The 404 errors for FVG endpoints are expected because no FVG tables exist yet.")
            logger.info("  You need to create FVG tables in the 'labels' schema to use FVG functionality.")
            logger.info("  Expected table names: spy1m_fvg, spy5m_fvg, spy15m_fvg, spy30m_fvg, spy1h_fvg, spy4h_fvg, spy1d_fvg")
        
        logger.info("=" * 80)
        
        # Clean up
        db.close()
        
    except Exception as e:
        logger.error(f"❌ Error checking routes and tables: {e}")
        logger.info("Continuing with server startup...")

def start_backend():
    """Start the FastAPI backend server"""
    logger = logging.getLogger(__name__)
    
    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    os.chdir(backend_dir)
    
    logger.info("🚀 Starting FastAPI backend server...")
    logger.info("📖 API documentation will be available at: http://localhost:8000/docs")
    logger.info("🔄 Health check available at: http://localhost:8000/health")
    
    # Start uvicorn server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Server failed to start: {e}")

def main():
    logger = setup_logging()
    
    logger.info("🏁 Starting Trading Data API...")
    
    # Check dependencies
    if not check_requirements():
        sys.exit(1)
    
    # Test database connection
    if not test_database_connection():
        logger.warning("⚠️  Database connection failed, but starting server anyway...")
        return
    
    # Show available routes and tables
    show_available_routes_and_tables()
    
    # Start the backend server
    start_backend()

if __name__ == "__main__":
    main() 