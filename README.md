# Daygent - Agentic Trading Intelligence Platform

## Overview

**Daygent** is a sophisticated agentic trading intelligence platform that combines traditional market analysis with cutting-edge AI and machine learning technologies. The system features **four integrated dashboards** within a monolithic architecture for comprehensive data analysis, vector intelligence, professional charting, and AI-powered market insights.

## 🚀 Architecture & Current Status

### **Monolithic Implementation** (Currently Active)
- **Main Component**: `TradingDashboard.jsx` (4,329 lines) - Contains all four dashboards
- **Architecture Support**: Both monolithic and modular modes available via `App.jsx`
- **Current Mode**: You are **NOT using the modular structure** - the system runs in monolithic mode
- **Dashboard Integration**: Four dashboards seamlessly integrated within single component

### **Four Integrated Dashboards**

#### 📊 **1. Data Dashboard**
- **Real-time OHLCV Data**: Advanced filtering, pagination, and search capabilities
- **Database Integration**: PostgreSQL with 21 trading tables (ES, EURUSD, SPY across 7 timeframes)
- **Advanced Controls**: Date range filtering with multiple modes, sorting, export to CSV
- **Interactive Selection**: Click-to-select candles with cross-dashboard synchronization
- **Debug Panel**: Backend sorting verification, data quality indicators, API debugging

#### 🧠 **2. Vector Intelligence Dashboard**
- **Dynamic Vector Detection**: Automatically detects available vector types from data
- **ISO Vectors**: Isolation Forest vectors for anomaly detection and shape analysis
- **Pattern Recognition**: Mathematical representation of market patterns
- **Vector Heatmaps**: Color-coded visualization for pattern analysis (limited for large vectors)
- **Shape Similarity Matrix**: Exclusive to ISO vectors with configurable dimensions
- **Comparison Tools**: Side-by-side vector analysis with similarity scoring

#### 📈 **3. Chart Analysis Dashboard**
- **Lightweight Charts v5.0**: High-performance candlestick, line, and area charts
- **Advanced Selection Modes**: 
  - Click mode (default)
  - Range selection (Shift+Click)
  - Multi-select (Ctrl/Cmd+Click)
  - Time-based quick selection (1H, 4H, 1D)
- **Interactive Candle Selection**: Drag to select multiple, with real-time highlighting
- **Labels Integration**: 🏷️ Trading labels display (tjr_high/tjr_low indicators)
- **Real-time Updates**: Synchronized with data dashboard selections
- **Keyboard Shortcuts**: Escape to exit, Delete to clear selection

#### 🤖 **4. LLM Dashboard (AI Assistant)**
- **Chat Interface**: Complete framework for AI integration (ready for GPT-4/Claude)
- **Mock Analysis**: Comprehensive UI for trading insights and market intelligence
- **Dynamic Mini Dashboard**: Resizable component display system
- **AI-Ready Framework**: Backend integration points for real AI models

## 🎯 **Advanced Features**

### **🏷️ Trading Labels System** ⭐ **NEW**
- **Label Types**: tjr_high and tjr_low indicators
- **Database Integration**: Separate labels tables for each symbol/timeframe
- **API Endpoint**: `/api/trading/labels/{symbol}{timeframe}`
- **Chart Integration**: Visual indicators on price charts
- **Label Structure**:
  ```json
  {
    "id": 1,
    "label": "tjr_high",
    "value": 4500.25,
    "pointer": [1, 2, 3]  // Reference to candle indices
  }
  ```

### **📅 Enhanced Date Range Controls**
- **Fetch Modes**:
  - `LIMIT`: Traditional record count limiting
  - `DATE_RANGE`: Time-based data fetching
- **Date Range Types**:
  - `EARLIEST_TO_DATE`: From earliest available to specified date
  - `DATE_TO_DATE`: Between two specific dates
  - `DATE_TO_LATEST`: From specified date to most recent
- **Auto-fill Logic**: Intelligent date population based on available data
- **Separate Date/Time Inputs**: Granular control with validation

### **🔍 Global State Management**
- **Shared Context** (TradingContext):
  - Symbol & Timeframe selection
  - Row limit and pagination
  - Sort order & column
  - Search term & column filter
  - Selected candles (global)
  - Date range configuration
  - Debug & filter visibility
- **Cross-Dashboard Sync**: All dashboards share the same context

### **Complete Vector System**
```
Dynamic Vector Types (auto-detected from data):
1. raw_ohlc_vec      - Direct OHLC values (4 dimensions)
2. raw_ohlcv_vec     - OHLC + Volume (5 dimensions)  
3. norm_ohlc         - Z-score normalized OHLC (4 dimensions)
4. norm_ohlcv        - Z-score normalized OHLCV (5 dimensions)
5. BERT_ohlc         - Semantic embeddings (384 dimensions)
6. BERT_ohlcv        - BERT with volume (384 dimensions)
7. iso_ohlc          - Isolation Forest features
8. iso_ohlcv         - ISO features with volume
+ Any custom vectors in the database
```

### **🔍 ISO Vectors & Shape Similarity**
- **Exclusive Features**: Shape similarity ONLY works with ISO vectors
- **Backend Processing**: Server-side similarity calculations for performance
- **Matrix Dimensions**: Configurable from 1x1 to 100x100
- **Cell Size Optimization**: Dynamic sizing based on matrix dimensions
- **Advanced Algorithms**: Manhattan, Euclidean, correlation, and cosine similarity
- **Full Range Visualization**: -100% (opposite) to +100% (identical)
- **Client-Side Fallback**: Automatic calculation if backend unavailable

## 🏗️ **Technology Stack**

### **Frontend**
```
React 19.1.0 + Vite 6.3.5
├── TailwindCSS 4.1.8 (Modern styling)
├── Lightweight Charts 5.0.7 (Performance charts)
├── Context API (Global state management)
├── Custom Hooks (Data management)
├── Selected Candles Panel (Global selection UI)
└── Dark/Light Theme Support
```

### **Backend**
```
FastAPI 0.104.1 + Python
├── PostgreSQL (Primary database)
├── SQLAlchemy 2.0.23 (ORM)
├── Pandas 2.1.4 (Data processing) 
├── NumPy 1.26.2 (Mathematical operations)
├── Pydantic 2.5.3 (Data validation)
├── Alembic 1.12.1 (Database migrations)
└── Labels System (Trading indicators)
```

### **AI/ML Pipeline**
```
Vector Generation (compute.py)
├── BERT Model: "all-mpnet-base-v2"
├── Sentence Transformers
├── Isolation Forest (ISO vectors)
├── Z-Score Normalization
└── Batch Processing (GPU accelerated)
```

## 📁 **Project Structure**

```
my-vite-project/
├── src/
│   ├── components/
│   │   ├── TradingDashboard.jsx     # 🎯 MAIN COMPONENT (4,329 lines)
│   │   │                            # Contains all 4 integrated dashboards
│   │   ├── Chart.jsx                # Chart dashboard implementation  
│   │   ├── LLMDashboard.jsx         # AI assistant dashboard
│   │   ├── AdvancedChart.jsx        # TradingView integration template
│   │   ├── shared/                  # Reusable UI components
│   │   │   ├── SelectedCandlesPanel.jsx
│   │   │   ├── InfoTooltip.jsx
│   │   │   ├── ThemeToggle.jsx
│   │   │   └── [ErrorDisplay, LoadingSpinner]
│   │   ├── data-dashboard/          # Modular components (available but unused)
│   │   │   ├── DataDashboard.jsx
│   │   │   ├── DataTable.jsx
│   │   │   ├── AdvancedFilters.jsx
│   │   │   └── [6 more components]
│   │   └── vector-dashboard/        # Modular components (available but unused)
│   │       ├── VectorDashboard.jsx
│   │       ├── VectorHeatmap.jsx
│   │       ├── VectorComparison.jsx
│   │       └── [4 more components]
│   ├── context/
│   │   └── TradingContext.jsx       # Global state management
│   ├── hooks/
│   │   ├── useTradingData.js        # Data fetching
│   │   ├── useDateRanges.js         # Date range management
│   │   └── useTheme.js              # Theme management
│   ├── utils/
│   │   ├── constants.js             # App constants
│   │   ├── formatters.js            # Data formatting
│   │   └── tooltipContent.jsx       # Help content
│   └── App.jsx                      # Architecture selector (monolithic/modular)
├── backend/
│   ├── main.py                      # FastAPI application
│   ├── routers/
│   │   └── trading.py               # Trading & shape similarity APIs
│   ├── services/
│   │   └── trading_service.py       # Business logic
│   ├── [models.py, database.py, config.py]
│   └── requirements.txt
├── compute.py                       # Vector generation script
└── package.json                     # Dependencies
```

## 🌟 **Key Features & Capabilities**

### **Data Dashboard**
- **Database Analytics**: 21 tables with statistics and metadata
- **Advanced Search**: Column-specific filtering with real-time results
- **Pagination**: Efficient data loading with configurable limits
- **Export**: CSV export with full data preservation
- **Date Range Controls**: Multiple filtering modes (earliest-to-date, date-to-date, date-to-latest)

### **Vector Intelligence**
- **Heatmap Visualization**: Color-coded matrices for pattern recognition
- **Vector Comparison**: Side-by-side analysis with similarity scoring
- **Shape Similarity**: ISO vector-specific candlestick pattern analysis
- **Statistical Analysis**: Comprehensive vector statistics and insights
- **Dynamic Controls**: Configurable matrix dimensions and view modes

### **Chart Analysis**
- **High Performance**: Lightweight-charts with 60fps rendering
- **Interactive Selection**: Click, drag, and range selection modes
- **Real-time Sync**: Selected candles sync across all dashboards
- **Multiple Chart Types**: Candlestick, line, area, and volume charts
- **TradingView Ready**: Advanced Charts integration template included

### **AI Assistant (Framework)**
- **Complete Chat UI**: Ready for GPT-4, Claude, or local model integration
- **Mock Analysis**: Trading signals, sentiment analysis, risk assessment
- **Dynamic Interface**: Resizable components and context-aware responses
- **Integration Points**: Backend endpoints ready for AI model connection

## 🔗 **API Endpoints**

### **Core Trading Data**
```
GET /api/trading/data/{symbol}/{timeframe}        # OHLCV data with optional vectors
GET /api/trading/stats                            # Database statistics  
GET /api/trading/tables                           # Available data tables
GET /api/trading/date-ranges/{symbol}/{timeframe} # Available date ranges
```

### **Trading Labels** ⭐ **NEW**
```
GET /api/trading/labels/{symbol}{timeframe}       # Trading indicators (tjr_high/low)
```

### **Shape Similarity (ISO Vectors Only)**
```
GET /api/trading/shape-similarity/{symbol}/{timeframe}  # Advanced similarity analysis
```

### **Query Parameters**
- `limit`: Records to return (1-10,000)
- `offset`: Pagination offset
- `start_date`/`end_date`: Date filtering (ISO format)
- `include_vectors`: Include vector columns (true/false)
- `vector_type`: Specific vector type (required for shape similarity)
- `order`: Sort order (asc/desc)
- `sort_by`: Column to sort by (default: timestamp)
- `count_only`: Return only count (for pagination)

## ⚙️ **Installation & Setup**

### **Prerequisites**
- **Node.js** 18+ and npm
- **Python** 3.8+ with pip  
- **PostgreSQL** database
- **GPU** (optional, for faster BERT processing)

### **Frontend Setup**
```bash
cd my-vite-project
npm install
npm run dev
```

### **Backend Setup**  
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **Vector Generation**
```bash
# Install dependencies
pip install numpy pandas torch sentence-transformers sentencepiece accelerate

# Generate vectors from CSV files
python compute.py
```

## 🎨 **User Interface Features**

### **Cross-Dashboard Functionality**
- **Selected Candles Panel**: Unified selection display across all dashboards
- **Global Candle Selection**: Select in one dashboard, see everywhere
- **Selection Persistence**: Maintains selection when switching dashboards
- **Multi-Symbol Support**: Can select candles from different symbols/timeframes
- **Theme Support**: Dark/light mode with persistent settings
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Advanced Tooltips**: Comprehensive help system throughout

### **Chart Selection Features**
- **Selection Modes**:
  - Default click to select/deselect
  - Shift+Click for range selection
  - Ctrl/Cmd+Click for multi-select
  - Drag to select multiple candles
- **Visual Feedback**:
  - Blue highlighting for selected candles
  - Hover tooltips with OHLC data
  - Selection box during drag
  - Mode indicators on screen
- **Quick Actions**:
  - Time-based selection (1H, 4H, 1D)
  - Delete key to clear selection
  - Escape to exit selection mode

### **Data Quality Indicators**
- **Backend Sorting Status**: ✅ Working / ❌ Client-side fallback
- **Data Quality Assessment**: 
  - GOOD (Historical/Recent data as expected)
  - POOR (Sorting issues detected)
- **Debug Information**:
  - API URLs for transparency
  - Actual data ranges
  - Request/response details
  - Performance metrics

## 🔬 **Shape Similarity Analysis Deep Dive**

### **Algorithm Features**
- **Multiple Distance Metrics**: Manhattan, Euclidean, correlation, cosine similarity
- **Full Range Mapping**: -100% (opposite patterns) to +100% (identical patterns)
- **Color Visualization**: Intuitive green (similar) to red (different) mapping
- **Statistical Analysis**: Comprehensive similarity statistics and pattern diversity

### **Use Cases**
- **Pattern Recognition**: Find similar candlestick formations
- **Anomaly Detection**: Identify unusual market conditions using ISO vectors
- **Market Analysis**: Compare current patterns to historical data
- **Strategy Development**: Quantify pattern similarity for trading algorithms

## 🚀 **TradingView Advanced Charts Integration**

### **Professional Features** (Template Ready)
- **100+ Technical Indicators**: Moving averages, oscillators, volume studies
- **70+ Drawing Tools**: Trend lines, Fibonacci, patterns, annotations
- **Multiple Chart Types**: Renko, Point & Figure, Kagi, Line Break
- **Volume Profile**: Market depth and volume analysis
- **Symbol Comparison**: Multi-asset overlay analysis

### **Setup Instructions**
1. Request access: https://www.tradingview.com/charting-library/
2. Download library files to `public/charting_library/`
3. Component automatically detects and initializes the library
4. Seamless integration with existing data pipeline

## 📊 **Database Schema**

### **Trading Tables Structure**
```sql
-- Example: es_1m, eurusd_5m, spy_1d, etc.
CREATE TABLE {symbol}_{timeframe} (
    symbol VARCHAR,
    timestamp TIMESTAMP,
    open DECIMAL,
    high DECIMAL, 
    low DECIMAL,
    close DECIMAL,
    volume DECIMAL,
    
    -- Vector columns (dynamically detected)
    raw_ohlc_vec DECIMAL[],      -- [open, high, low, close]
    raw_ohlcv_vec DECIMAL[],     -- [open, high, low, close, volume]
    norm_ohlc DECIMAL[],         -- Z-score normalized OHLC
    norm_ohlcv DECIMAL[],        -- Z-score normalized OHLCV  
    BERT_ohlc DECIMAL[],         -- 384-dim semantic embeddings
    BERT_ohlcv DECIMAL[],        -- 384-dim embeddings with volume
    iso_ohlc DECIMAL[],          -- Isolation forest features
    iso_ohlcv DECIMAL[]          -- ISO features with volume
);

-- Labels tables (NEW)
CREATE TABLE labels_{symbol}{timeframe} (
    id SERIAL PRIMARY KEY,
    label VARCHAR,               -- 'tjr_high' or 'tjr_low'
    value DECIMAL,               -- Price value
    pointer INTEGER[]            -- Reference to candle indices
);
```

### **Available Data**
- **Symbols**: ES (E-mini S&P 500), EURUSD (Forex), SPY (ETF)
- **Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 1d
- **Total Tables**: 21 tables (3 symbols × 7 timeframes)

## 🔧 **Configuration**

### **Environment Variables**
```env
# Database
DATABASE_URL=postgresql://user:password@localhost/trading_db

# API Settings  
API_BASE_URL=http://localhost:8000
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Vector Generation
MODEL_NAME=all-mpnet-base-v2
BATCH_SIZE=128
```

### **App Configuration**
- **Architecture Mode**: Toggle between monolithic/modular in `App.jsx`
- **Theme Settings**: Persistent dark/light mode preference
- **Default Fetch Limits**: Configurable in `constants.js`

## 🛠️ **Future Enhancements**

### **Immediate Roadmap**
- [ ] **LLM Backend Integration**: Connect GPT-4/Claude for real AI analysis
- [ ] **Real-time Data Feeds**: WebSocket integration for live market data
- [ ] **Enhanced Labels System**: Additional indicator types beyond tjr_high/low
- [ ] **Export Selected Candles**: Export analysis of selected candle groups
- [ ] **Pattern Templates**: Save and load candle selection patterns

### **Advanced Features**  
- [ ] **Backtesting Engine**: Historical strategy testing with vector patterns
- [ ] **Alert System**: Pattern-based notifications and automated signals
- [ ] **Portfolio Management**: Multi-asset position tracking and risk management
- [ ] **Machine Learning Pipeline**: Auto-pattern discovery and classification
- [ ] **Label-based Strategies**: Trading strategies using tjr indicators

## 💡 **Development Notes**

### **Architecture Decision**
- **Current**: Monolithic implementation for rapid development and feature integration
- **Available**: Modular architecture with 23+ reusable components ready for use
- **Flexibility**: Switch between architectures anytime via `App.jsx` toggle

### **Performance Considerations**
- **Vector Heatmaps**: Limited display for vectors >1000 dimensions (e.g., BERT)
- **Shape Similarity**: Backend processing recommended for matrices >20x20
- **Chart Selection**: Optimized for up to 500 simultaneous selections
- **Label Queries**: Cached for performance with trading data
- **Date Range Queries**: More efficient than large limit-based queries

### **Known Limitations**
- **BERT Heatmaps**: Too large for visual display (384 dimensions)
- **ISO Vectors Only**: Shape similarity exclusive to isolation forest vectors
- **Selection Limit**: Browser may slow with >1000 selected candles
- **Label Types**: Currently supports only tjr_high and tjr_low

## 🆘 **Troubleshooting**

### **Common Issues**
- **Backend Connection**: Ensure FastAPI server running on port 8000
- **Vector Generation**: Requires sufficient RAM for BERT model (4GB+)
- **Large Matrices**: Shape similarity >50x50 may impact browser performance
- **TradingView Charts**: Requires separate license for production use

### **Performance Tips**
- **Data Fetching**: Use date ranges instead of large record limits
- **Vector Analysis**: Start with smaller matrices for exploration
- **Memory Usage**: Monitor browser memory with large datasets
- **GPU Acceleration**: Use CUDA for faster vector generation

---

**Daygent** represents a comprehensive agentic trading intelligence platform, combining quantitative analysis, machine learning, and professional trading tools in a unified interface. The monolithic architecture ensures seamless integration while the modular components provide future scalability options.

**🎯 Current Status**: Fully functional with 4 integrated dashboards, trading labels system, advanced candle selection, dynamic vector detection including ISO vectors with shape similarity, professional charting capabilities, and AI-ready framework for future model integration.

**Latest Updates**:
- 🏷️ Trading labels integration with tjr_high/low indicators
- 🎯 Advanced chart selection modes with keyboard shortcuts
- 📅 Enhanced date range controls with multiple fetch modes
- 🔍 Debug panel for data quality verification
- 📊 Selected Candles Panel for global selection management
