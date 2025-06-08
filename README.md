# Trading Data Dashboard

A modern, high-performance web application for visualizing and analyzing trading data from PostgreSQL database with 21 vectorized trading tables connected to RAG LLM infrastructure.

## Current Status ✅

**COMPLETED FEATURES:**
- ✅ **Fully functional FastAPI backend** with 21 trading tables (ES, EURUSD, SPY across multiple timeframes)
- ✅ **Modern React frontend** with Vite, Tailwind CSS, and responsive design
- ✅ **Real-time data visualization** with sortable, searchable data tables
- ✅ **Advanced filtering & pagination** with client-side and server-side sorting
- ✅ **Database statistics dashboard** showing total tables, records, and connection status
- ✅ **ML vector embeddings support** (raw_ohlc_vec, raw_ohlcv_vec, norm_ohlc, norm_ohlcv, bert_ohlc, bert_ohlcv)
- ✅ **CSV export functionality** (implementation pending testing)
- ✅ **Multi-table quick selection** with one-click table switching
- ✅ **Comprehensive API documentation** with FastAPI automatic docs

**CURRENT BUGS/TODO:**
- ⚠️ CSV export feature needs testing and validation
- 🔄 Performance optimization needed for large datasets
- 📊 Chart visualization planned (lightweight charts integration)

## Architecture

- **Frontend**: Vite + React 19.1.0 + Tailwind CSS 4.1.8
- **Backend**: FastAPI 0.104.1 + SQLAlchemy 2.0.23 + PostgreSQL
- **Database**: PostgreSQL with TimescaleDB (21 trading tables)
- **Integration**: Direct Python FastAPI backend serving trading data to React frontend

## Features

- 📊 **Real-time trading data visualization** with sortable columns and search
- 🗄️ **Access to 21 trading tables** (ES, EURUSD, SPY across multiple timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d)
- 📈 **OHLCV data with ML vector embeddings** for RAG LLM integration
- 🔍 **Advanced filtering and pagination** with real-time search across all columns
- 📱 **Responsive design** with modern Tailwind CSS styling
- 🚀 **High-performance FastAPI backend** with async support
- 📖 **Automatic API documentation** at `/docs` endpoint
- 🎯 **Row selection and bulk operations** with checkbox selection
- 📋 **CSV export functionality** for data analysis
- 🔄 **Real-time data refresh** with loading states and error handling
- 📊 **Database statistics** with table overviews and metadata

## Project Structure

```
my-vite-project/
├── src/
│   ├── components/
│   │   └── TradingDashboard.jsx    # Main trading dashboard (1067 lines)
│   ├── App.jsx                     # React app entry point
│   ├── main.jsx                    # Vite entry point
│   ├── index.css                   # Global styles
│   └── App.css                     # Component styles
├── backend/
│   ├── main.py                     # FastAPI application (142 lines)
│   ├── config.py                   # Configuration settings
│   ├── database.py                 # Database connection & session management
│   ├── models.py                   # Pydantic models for API responses
│   ├── routers/
│   │   └── trading.py              # Trading API routes (208 lines)
│   ├── services/
│   │   └── trading_service.py      # Business logic & database queries
│   └── requirements.txt            # Python dependencies
├── start.py                        # Python startup script with health checks
├── package.json                    # Node.js dependencies & scripts
├── vite.config.js                  # Vite configuration
├── tailwind.config.js              # Tailwind CSS configuration
├── postcss.config.js               # PostCSS configuration
├── eslint.config.js                # ESLint configuration
├── index.html                      # HTML entry point
└── README.md                       # This file
```

## Available Trading Tables

Based on your PostgreSQL database schema:

### Symbols
- **ES**: E-mini S&P 500 futures
- **EURUSD**: Euro/US Dollar forex pair  
- **SPY**: SPDR S&P 500 ETF

### Timeframes
- **1m**: 1 minute
- **5m**: 5 minutes
- **15m**: 15 minutes
- **30m**: 30 minutes
- **1h**: 1 hour
- **4h**: 4 hours
- **1d**: 1 day

### Table Schema
Each table contains:
- Standard OHLCV data (Open, High, Low, Close, Volume)
- ML vector embeddings for RAG LLM integration:
  - `raw_ohlc_vec`: Raw OHLC vectors
  - `raw_ohlcv_vec`: Raw OHLCV vectors
  - `norm_ohlc`: Normalized OHLC vectors
  - `norm_ohlcv`: Normalized OHLCV vectors
  - `bert_ohlc`: BERT-encoded OHLC vectors
  - `bert_ohlcv`: BERT-encoded OHLCV vectors

## Setup Instructions

### Prerequisites

1. **PostgreSQL Database**: Ensure your trading database is running on `localhost:5433`
2. **Python 3.8+**: Required for FastAPI backend
3. **Node.js 16+**: Required for Vite frontend

### Backend Setup

1. **Install Python dependencies**:
   ```bash
   cd my-vite-project
   pip install -r backend/requirements.txt
   ```

2. **Test database connection**:
   ```bash
   cd backend
   python database.py
   ```

3. **Start the FastAPI backend**:
   ```bash
   # Option 1: Use the startup script
   python start.py
   
   # Option 2: Manual start
   cd backend
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Frontend Setup

1. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

2. **Start the Vite development server**:
   ```bash
   npm run dev
   ```

### Quick Start

1. **Start backend** (in one terminal):
   ```bash
   python start.py
   ```

2. **Start frontend** (in another terminal):
   ```bash
   npm run dev
   ```

3. **Access the application**:
   - Frontend: http://localhost:5173
   - Backend API docs: http://localhost:8000/docs
   - Backend health check: http://localhost:8000/health

## API Endpoints

The FastAPI backend provides comprehensive trading data access:

### Core Endpoints
- `GET /` - Root endpoint with API information
- `GET /health` - Health check
- `GET /api/trading/health` - Trading API health check

### Data Endpoints
- `GET /api/trading/stats` - Database statistics (table count, total records)
- `GET /api/trading/tables` - List all available tables with metadata
- `GET /api/trading/data/{symbol}/{timeframe}` - Get trading data with pagination
- `GET /api/trading/latest/{symbol}/{timeframe}` - Get latest data point
- `GET /api/trading/summary/{symbol}` - Get symbol summary across timeframes
- `GET /api/trading/search/{symbol}/{timeframe}` - Search by date range

### Utility Endpoints
- `GET /api/trading/symbols` - Available symbols (ES, EURUSD, SPY)
- `GET /api/trading/timeframes` - Available timeframes (1m, 5m, 15m, 30m, 1h, 4h, 1d)

### Query Parameters
- **limit**: Records per page (1-10,000, default: 100)
- **offset**: Pagination offset (default: 0)
- **order**: Sort order ('asc'/'desc', default: 'desc')
- **sort_by**: Sort column (default: 'timestamp')
- **start_date/end_date**: Date range filtering
- **include_vectors**: Include ML vector columns (default: false)

## Configuration

### Database Configuration

Update `backend/config.py` if your database settings differ:

```python
class Settings:
    DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@localhost:5433/trading_data"
    SCHEMA: str = "backtest"
    # ... other settings
```

### CORS Configuration

The backend is configured to allow requests from Vite's dev server (localhost:5173). Update `CORS_ORIGINS` in `config.py` for production deployment.

## Usage Examples

### Get Database Stats
```bash
curl http://localhost:8000/api/trading/stats
```

### Get ES 1-day data
```bash
curl "http://localhost:8000/api/trading/data/es/1d?limit=10"
```

### Get data with vectors
```bash
curl "http://localhost:8000/api/trading/data/es/1h?include_vectors=true&limit=5"
```

## Development

### Adding New Features

1. **Backend**: Add new routes in `backend/routers/trading.py`
2. **Frontend**: Add new components in `src/components/`
3. **Database**: Extend models in `backend/models.py`

### Testing Database Connection

```bash
cd backend
python -c "from database import test_connection; test_connection()"
```

## Production Deployment

### Backend Deployment
- Use production WSGI server (e.g., Gunicorn)
- Set proper environment variables
- Configure proper database connection pooling

### Frontend Deployment
```bash
npm run build
# Deploy dist/ folder to your web server
```

## Integration Benefits

This integration provides significant advantages over the Node.js + Python script pattern used in CourseDescribe:

1. **Performance**: Direct database access without subprocess overhead
2. **Type Safety**: Automatic API validation and documentation
3. **Scalability**: Native async support for handling multiple requests
4. **ML Integration**: Direct access to Python ML libraries for RAG LLM
5. **Real-time**: WebSocket support for live trading updates
6. **Maintainability**: Clean separation of concerns with service layers

## Troubleshooting

### Common Issues

1. **Database Connection Failed**:
   - Check if PostgreSQL is running on port 5433
   - Verify database credentials in `config.py`
   - Ensure `trading_data` database exists

2. **CORS Errors**:
   - Ensure backend is running on port 8000
   - Check CORS_ORIGINS in `config.py`

3. **Import Errors**:
   - Install all requirements: `pip install -r backend/requirements.txt`
   - Check Python version (3.8+ required)

4. **Frontend Build Errors**:
   - Install Node.js dependencies: `npm install`
   - Check Node.js version (16+ required)

## License

MIT License

## Recent Updates & Current State

### Dashboard Features (as of latest)
1. **Enhanced Data Table**: Sortable columns with visual indicators
2. **Real-time Search**: Search across all columns with instant filtering
3. **Smart Pagination**: Handles both ascending (oldest first) and descending (newest first) data
4. **Row Selection**: Checkbox selection with bulk operations support
5. **Export Functionality**: CSV export (pending testing)
6. **Responsive UI**: Mobile-friendly design with Tailwind CSS
7. **Loading States**: Proper loading indicators and error handling
8. **Debug Information**: Optional debug panel for development

### Technical Implementation
- **Frontend**: 1067-line React component with advanced state management
- **Backend**: RESTful API with comprehensive error handling
- **Database**: Direct PostgreSQL access with optimized queries
- **Performance**: Client-side sorting and filtering for responsive UX

### Data Visualization
- **OHLCV Display**: Open, High, Low, Close, Volume with proper formatting
- **Candle Type Indicators**: Visual green/red/doji candle representations
- **Timestamp Formatting**: Human-readable date/time display
- **Price Formatting**: Consistent decimal formatting for financial data
- **Volume Display**: Comma-separated number formatting
