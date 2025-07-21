"""
FastAPI main application for trading data API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

from config import settings
from database import test_connection, engine
from routers import trading

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to see all our debug logs
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Trading Data API",
    description="""
    A high-performance API for accessing trading data from PostgreSQL database.
    
    Features:
    - Access to 21 trading tables (ES, EURUSD, SPY across multiple timeframes)
    - OHLCV data with optional ML vector embeddings
    - Real-time data queries with pagination
    - Database statistics and metadata
    - Date range filtering
    
    Built with FastAPI and SQLAlchemy for optimal performance.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(trading.router)

@app.on_event("startup")
async def startup_event():
    """Test database connection on startup"""
    logger.info("Starting Trading Data API...")
    logger.info(f"FastAPI docs available at: http://{settings.HOST}:{settings.PORT}/docs")
    
    # Test database connection
    if test_connection():
        logger.info("✅ Database connection successful!")
    else:
        logger.error("❌ Database connection failed!")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down Trading Data API...")
    engine.dispose()

@app.get("/", summary="Root endpoint")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Trading Data API",
        "version": "1.0.0",
        "timestamp": datetime.now(),
        "docs": "/docs",
        "health": "/api/trading/health",
        "stats": "/api/trading/stats"
    }

@app.get("/health", summary="Health check")
async def health_check():
    """Simple health check for the entire API"""
    try:
        # Test database connection
        db_healthy = test_connection()
        
        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "timestamp": datetime.now(),
            "database": "connected" if db_healthy else "disconnected",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now(),
                "error": str(e)
            }
        )

# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "detail": "The requested resource was not found",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error", 
            "detail": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    ) 