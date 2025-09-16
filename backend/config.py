"""
Configuration settings for the trading data API
"""

import os
from typing import Optional

class Settings:
    # Database configuration - matches your fill_vectors.py script
    DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@localhost:5433/trading_data"
    SCHEMA: str = "backtest"
    FRONTTEST_SCHEMA: str = "fronttest"
    MODELS_SCHEMA: str = "v1_models"
    LABELS_SCHEMA: str = "labels"
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS settings for Vite dev server
    CORS_ORIGINS: list = [
        "http://localhost:5173",  # Vite default dev server
        "http://localhost:3000",  # Alternative dev server
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

settings = Settings() 