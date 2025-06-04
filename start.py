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
    
    # Start the backend server
    start_backend()

if __name__ == "__main__":
    main() 