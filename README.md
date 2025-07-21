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

### **🏷️ Universal TJR & Swing Marker Support** ⭐ **NEW (2024 Update)**

#### **Key Features**
- **TJR Markers** (High/Low) and **Swing Markers** (High/Low) now supported for **all SPY timeframes** (1m, 5m, 15m, 30m, 1h, 4h, 1d)
- **Dynamic Table Detection**: System automatically detects which SPY tables have labeled or swing data
- **Flexible Table Naming**: Supports both `spy1h_swings` and `spy_1h_swings` (and similar) naming conventions
- **Clean Toggle UI**: TJR and swing toggles appear for all SPY timeframes with available data
- **Combined Marker Rendering**: Both marker types are rendered together, with no visual glitches
- **Status & Legend**: UI shows which tables have data, and a legend explains marker types

#### **Marker Types**
- **TJR High**: Green circle with "T" (highest in range)
- **TJR Low**: Red circle with "⊥" (lowest in range)
- **Swing High**: Blue circle with "▲" (exact candle)
- **Swing Low**: Orange circle with "▼" (exact candle)

#### **How Swings and TJR Markers Are Defined**
- **Swing High/Low**: A swing high is the highest high, and a swing low is the lowest low, within a fixed grouping window (e.g., a week for 4h candles, or 2 weeks for 1d candles). The marker is placed on the exact candle that set the high or low. These are objective, algorithmic pivots and are not affected by the color of the candle.
- **TJR High/Low (Trend/Jump Reversal)**: A TJR high is defined as a local high where a green (bullish) candle is immediately followed by a red (bearish) candle, indicating a potential reversal from up to down. A TJR low is a local low where a red (bearish) candle is immediately followed by a green (bullish) candle, indicating a potential reversal from down to up. TJR markers are placed at the transition point and highlight possible internal liquidity shifts.

#### **Liquidity Identification Using Swings & TJR Markers**

##### **Concept**
- **External Liquidity**: Areas where price is likely to run stops or sweep liquidity above swing highs or below swing lows. These are typically targeted by large players to trigger stop orders and induce volatility.
- **Internal Liquidity**: Areas inside the current range, often defined by TJR highs/lows, where price may react, consolidate, or reverse due to internal order flow shifts.

##### **How to Use in Practice**

1. **Identify External Liquidity (Swings):**
   - Use swing highs and lows across multiple timeframes (e.g., 4h, 1d, 10d) to mark the most obvious liquidity pools.
   - When price approaches a swing high/low, anticipate possible liquidity grabs (stop runs) and watch for sharp reversals or acceleration through these levels.
   - Example strategy: Fade the first move beyond a major swing high/low if there is a strong reversal signal (e.g., TJR or volume spike), or join the breakout if price closes decisively beyond the swing with confirmation.

2. **Identify Internal Liquidity (TJR):**
   - Use TJR highs/lows to map out the most recent internal pivots within the current range.
   - These are often used by algorithms and smart money to accumulate or distribute positions before a larger move.
   - Example strategy: Enter on a retest of a TJR high/low after a sweep, or use them as targets for partial profit-taking when trading within a range.

3. **Multi-Timeframe Confluence:**
   - Look for alignment between swing levels and TJR levels across different timeframes (e.g., a 4h swing high that coincides with a 1d TJR high).
   - The more levels cluster together, the higher the probability of a significant liquidity event.

4. **Execution Plan:**
   - **Preparation:** Mark all swing and TJR levels on your chart for the relevant timeframes.
   - **Trigger:** Wait for price to approach these levels, then look for confirmation (e.g., reversal candle, order flow shift, or momentum divergence).
   - **Entry:** Enter on confirmation, with stops placed just beyond the liquidity level (for fades) or on breakout retests (for continuations).
   - **Management:** Scale out at the next internal or external liquidity level, or trail stops as price moves in your favor.

##### **Why This Works**
- **Swings** represent the most obvious liquidity pools—where retail stops and institutional resting orders accumulate. Price is drawn to these areas, and reactions are often sharp and tradeable.
- **TJR** levels capture the subtle, internal shifts in order flow that precede larger moves. They are less obvious, providing an edge for anticipating reversals or continuations before the crowd.
- **Combining both** allows you to anticipate where liquidity is likely to be found and how price is likely to react, giving you a robust, repeatable edge.

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

## 🔬 **Advanced Shape Similarity Analysis** ⭐ **NEW**

### **Multi-Metric Similarity Engine**
- **Hybrid Similarity Calculation**: Combines 4 different distance metrics for superior pattern discrimination
- **Weighted Algorithm**: Emphasizes Manhattan distance (40%) for shape-sensitive analysis
- **Full Spectrum Mapping**: -100% (opposite patterns) to +100% (identical patterns)
- **Performance Optimization**: Server-side processing for large similarity matrices

### **Similarity Metrics Breakdown**
```
1. Manhattan Distance (40% weight)    - Most sensitive to shape differences
2. Euclidean Distance (25% weight)    - Standard geometric distance
3. Correlation Analysis (20% weight)   - Pattern relationship detection
4. Cosine Similarity (15% weight)     - Directional similarity
```

### **Advanced Features**
- **Zero Vector Handling**: Graceful handling of edge cases and invalid data
- **Normalization Pipeline**: Proper vector normalization before similarity calculation
- **Statistical Robustness**: Handles NaN values and numerical instabilities
- **Matrix Optimization**: Efficient computation for large similarity matrices (up to 100x100)

### **Use Cases**
- **Pattern Recognition**: Identify similar candlestick formations across different time periods
- **Anomaly Detection**: Find unusual market patterns using ISO vector analysis
- **Strategy Development**: Quantify pattern similarity for algorithmic trading
- **Market Analysis**: Compare current patterns to historical performance

## 🎯 **Binary Classification Labels** ⭐ **NEW**

### **Future Price Direction Column**
- **Universal Implementation**: Added to all 21 backtest tables across all symbols and timeframes
- **Binary Classification**: `future` column indicates next candle's price direction
- **Label Logic**: 
  - `1` = Next candle's close > Current candle's close (bullish)
  - `0` = Next candle's close ≤ Current candle's close (bearish/neutral)
- **ML-Ready Format**: Perfect for supervised learning and pattern prediction

### **Applications**
- **Machine Learning Training**: Train models to predict price direction from current patterns
- **Backtesting Strategies**: Validate trading signals against historical outcomes
- **Pattern Analysis**: Identify which patterns historically lead to price increases
- **Risk Assessment**: Calculate probability of bullish/bearish outcomes

### **Data Structure**
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
    future INTEGER,  -- NEW: 1 = next close higher, 0 = next close lower/equal
    
    -- Vector columns (dynamically detected)
    raw_ohlc_vec DECIMAL[],
    raw_ohlcv_vec DECIMAL[],
    norm_ohlc DECIMAL[],
    norm_ohlcv DECIMAL[],
    BERT_ohlc DECIMAL[],
    BERT_ohlcv DECIMAL[],
    iso_ohlc DECIMAL[],
    iso_ohlcv DECIMAL[]
);
```

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

## 🤖 **Enhanced ML Vector Generation Pipeline** ⭐ **NEW**

### **Advanced Vector Processing (compute.py)**
- **Multi-Modal Vector Generation**: Six distinct vector types for comprehensive analysis
- **GPU Acceleration**: CUDA support for faster BERT processing with automatic fallback
- **Batch Processing**: Configurable batch sizes for optimal memory usage
- **Sentence Crafting**: Intelligent text representation of OHLC data for semantic analysis

### **Vector Types Generated**
```
1. raw_ohlc_vec    - Direct OHLC values [4 dimensions]
2. raw_ohlcv_vec   - OHLC + Volume [5 dimensions]
3. norm_ohlc       - Z-score normalized OHLC [4 dimensions]
4. norm_ohlcv      - Z-score normalized OHLCV [5 dimensions]
5. BERT_ohlc       - Semantic embeddings [384 dimensions]
6. BERT_ohlcv      - BERT with volume [384 dimensions]
```

### **Processing Features**
- **Automatic Device Detection**: Chooses CUDA/CPU based on availability
- **Progress Tracking**: Real-time progress bars for embedding generation
- **Logging System**: Comprehensive logging to both file and console
- **Error Handling**: Graceful handling of missing columns and data issues
- **Volume Normalization**: Log1p transformation for volume data before z-scoring

### **Pipeline Workflow**
1. **Data Validation**: Checks for required columns (symbol, timestamp, OHLC, volume)
2. **Sentence Generation**: Creates text representations for semantic analysis
3. **BERT Embedding**: Generates 384-dimensional semantic vectors
4. **Raw Vector Creation**: Direct numerical representations
5. **Statistical Normalization**: Z-score normalization with proper handling
6. **Output Generation**: CSV files with all vector columns appended

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

### **🏷️ Enhanced Trading Labels & Markers** ⭐ **NEW**
```
GET /api/trading/labels/{symbol}/{timeframe}      # TJR labels for any symbol/timeframe
GET /api/trading/swing-labels/{symbol}/{timeframe} # Swing high/low labels for any symbol/timeframe
GET /api/trading/labels/spy1h                     # Legacy SPY 1H labels endpoint
GET /api/trading/labels/spy1h_swings              # Legacy SPY 1H swings endpoint
```

#### **Label Types & Data Structure**
- **TJR Labels**: Trend/Jump Reversal markers with time range pointers
- **Swing Labels**: Exact pivot points with precise timestamp matching
- **Flexible Table Support**: Handles both `spy1h_labeled` and `spy_1h_labeled` naming conventions
- **Graceful Degradation**: Returns empty arrays for missing tables instead of errors

### **Trading Labels** ⭐ **NEW**
```
GET /api/trading/labels/{symbol}/{timeframe}      # TJR labels for any symbol/timeframe
GET /api/trading/swing-labels/{symbol}/{timeframe} # Swing high/low labels for any symbol/timeframe
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

### **🎯 Advanced Candle Selection System** ⭐ **NEW**

#### **Interactive Selection Modes**
- **Click Mode** (Default): Single-click to select/deselect individual candles
- **Range Mode** (Shift+Click): Select consecutive candles between two points
- **Multi-Select Mode** (Ctrl/Cmd+Click): Add individual candles to existing selection
- **Drag Selection**: Click and drag to select multiple candles in one motion
- **Time-Based Selection**: Quick buttons for 1H, 4H, 1D time ranges

#### **Visual Selection Features**
- **Blue Highlighting**: Selected candles highlighted with blue overlay
- **Real-time Hover Tooltips**: OHLC data preview on mouse hover
- **Selection Box**: Visual rectangle during drag operations
- **Custom Crosshair**: Optional crosshair overlay in selection mode
- **Selection Statistics**: Real-time count, price range, and trend analysis

#### **Keyboard Shortcuts**
- **Escape**: Exit selection mode
- **Delete**: Clear all selected candles for current symbol/timeframe
- **Shift**: Switch to range selection mode
- **Ctrl/Cmd**: Switch to multi-select mode

### **📊 Multi-Timeframe Breakdown Analysis** ⭐ **NEW**

#### **Sequential Candle Analysis**
- **Automatic Detection**: System detects when selected candles are consecutive
- **Breakdown Toggle**: Enable secondary chart for detailed analysis
- **Timeframe Compatibility**: Validates that breakdown timeframe is smaller than primary
- **Correlated Candles**: Shows all smaller timeframe candles that compose selected candles

#### **Secondary Chart Features**
- **Dual Chart Display**: Primary chart + breakdown chart side-by-side
- **Orange Overlay Mode**: Optional highlighting of breakdown candles
- **Real-time Correlation**: Automatic matching of time ranges
- **Breakdown Statistics**: Detailed analysis of candle composition

#### **Timeframe Relationships**
- **Compatible Pairs**: 1h→15m, 4h→1h, 1d→4h, etc.
- **Validation Logic**: Primary must be evenly divisible by secondary
- **Error Handling**: Clear warnings for incompatible timeframe combinations

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
- **Full Range Visualization**: -100% (opposite) to +100% (identical)
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

-- Swing tables (NEW)
CREATE TABLE swings_{symbol}{timeframe} (
    id SERIAL PRIMARY KEY,
    label VARCHAR,               -- 'swing_high' or 'swing_low'
    value DECIMAL,               -- Price value
    pointer INTEGER[]            -- Reference to candle indices
);
```

### **Available Data**
- **Symbols**: ES (E-mini S&P 500), EURUSD (Forex), SPY (ETF)
- **Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 1d
- **Total Tables**: 21 tables (3 symbols × 7 timeframes)

### **Example Table Names Supported**
- `spy1h_labeled`, `spy_1h_labeled`
- `spy1h_swings`, `spy_1h_swings`
- ...and all other SPY timeframes

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
- **TJR/Swing Toggles**: If toggles do not appear, check that your database has the appropriate labeled or swing tables for the selected timeframe

### **Performance Tips**
- **Data Fetching**: Use date ranges instead of large record limits
- **Vector Analysis**: Start with smaller matrices for exploration
- **Memory Usage**: Monitor browser memory with large datasets
- **GPU Acceleration**: Use CUDA for faster vector generation

### **Status Panel**
- The status panel will show which tables are available for TJR and swing markers
- Check the status panel to verify which SPY timeframes have labeled or swing data

## 🎯 **How to Use**

### **Basic Setup**
1. Start the backend server (`uvicorn main:app` or your preferred method)
2. Start the frontend (`npm run dev`)
3. Select any SPY symbol and timeframe
4. Toggle TJR and swing markers as desired
5. See real-time status of available tables and markers

### **Advanced Usage**
- **Multi-Dashboard Analysis**: Use all four dashboards simultaneously for comprehensive analysis
- **Vector Analysis**: Explore pattern recognition with ISO vectors and shape similarity
- **Candle Selection**: Use advanced selection modes for detailed analysis
- **Date Range Filtering**: Optimize data loading with intelligent date controls

## 🏷️ **FVG (Fair Value Gap) Labeling System** ⭐ **NEW**

### **Universal FVG Support**
- **Complete FVG Tables**: Added FVG labeling tables for all SPY timeframes in the `labels` schema
- **Fair Value Gap Detection**: Identifies gaps in price action where liquidity may be found
- **Multi-Timeframe Coverage**: FVG labels available across all 7 SPY timeframes (1m, 5m, 15m, 30m, 1h, 4h, 1d)
- **Database Integration**: Seamlessly integrated with existing labeling infrastructure

### **FVG Data Structure**
```sql
-- Example: labels.spy4h_fvg table structure
CREATE TABLE labels.spy4h_fvg (
    id SERIAL PRIMARY KEY,
    label VARCHAR,               -- 'fvg_red' or 'fvg_green'
    value DECIMAL,               -- Price value of the FVG
    color_order TEXT[],          -- Array of color sequences
    pointer TIMESTAMP WITH TIME ZONE[]  -- Time range pointers
);
```

### **FVG Types**
- **FVG Red**: Bearish fair value gaps (price gaps down)
- **FVG Green**: Bullish fair value gaps (price gaps up)
- **Color Order Tracking**: Maintains sequence of candle colors around the gap
- **Time Range Pointers**: Precise timestamp arrays for gap identification

## 🤖 **Machine Learning Prediction Models** ⭐ **NEW**

### **SPY Prediction Model v1** (`spy_prediction_model_v1.py`)
- **Multi-Timeframe Architecture**: Uses all 7 SPY timeframes (1m, 5m, 15m, 30m, 1h, 4h, 1d)
- **Feature Engineering**: Combines raw_ohlcv vectors (5D) + iso_ohlc vectors (4D) + future binary labels
- **Model Ensemble**: Random Forest, Gradient Boosting, XGBoost, Logistic Regression, LSTM
- **Hierarchical Approach**: Professional trading system methodology
- **Sequence Analysis**: 60-period LSTM sequences for temporal pattern recognition
- **Performance Metrics**: Classification reports, confusion matrices, ROC-AUC scoring

### **SPY Prediction Model v2** (`spy_prediction_model_v2_feature_expansion_demo.py`)
- **Massive Feature Expansion**: 21 → 400+ features
- **Complete Vector Integration**: All 8 vector types (raw_ohlc, raw_ohlcv, norm_ohlc, norm_ohlcv, BERT_ohlc, BERT_ohlcv, iso_ohlc, iso_ohlcv)
- **Advanced Technical Indicators**: 50+ TA-Lib indicators including RSI, MACD, Bollinger Bands, Stochastic, Williams %R
- **Multi-Timeframe Context**: Cross-timeframe correlation analysis
- **Market Regime Detection**: Volatility regimes, trend strength, trading sessions
- **Enhanced Model Architecture**: Improved LSTM, dynamic ensemble weighting

### **Model Training Pipeline**
```python
# Feature Types in v2
1. Core Vectors (789 features)
   - raw_ohlc_vec (4D), raw_ohlcv_vec (5D)
   - norm_ohlc (4D), norm_ohlcv (5D)
   - BERT_ohlc (384D), BERT_ohlcv (384D)
   - iso_ohlc (4D), iso_ohlcv (4D)

2. Technical Indicators (50+ features)
   - Momentum: RSI, MACD, Stochastic, Williams %R
   - Volume: OBV, VPT, A/D Line
   - Volatility: ATR, Bollinger Bands
   - Trend: ADX, CCI, ROC

3. Market Context (35+ features)
   - Cross-timeframe correlations
   - Market regime indicators
   - Calendar effects
```

## 🚀 **Startup & Development Tools** ⭐ **NEW**

### **Automated Startup Script** (`start.py`)
- **Intelligent Dependency Checking**: Validates all required packages before startup
- **Database Connection Testing**: Pre-flight database connectivity verification
- **Automated Server Launch**: One-command FastAPI server startup
- **Health Monitoring**: Built-in health check endpoints
- **Error Handling**: Graceful failure handling with informative logging
- **Development Mode**: Hot reload enabled for rapid development

### **Usage**
```bash
# Start the entire system with one command
python start.py

# Features:
# ✅ Dependency validation
# ✅ Database connection test
# ✅ Automatic server startup
# ✅ Health monitoring
# ✅ Development mode with hot reload
```

### **Development Workflow**
```bash
# Quick development setup
python start.py                    # Backend server
npm run dev                        # Frontend development
```

## 📊 **Advanced Chart Integration** ⭐ **NEW**

### **TradingView Advanced Charts** (`ADVANCED_CHARTS_INTEGRATION.md`)
- **Professional Chart Library**: Enterprise-grade charting capabilities
- **100+ Technical Indicators**: Moving averages, oscillators, volume studies
- **70+ Drawing Tools**: Trend lines, Fibonacci, patterns, annotations
- **Multiple Chart Types**: Renko, Point & Figure, Kagi, Line Break
- **Volume Profile**: Market depth and volume analysis
- **Symbol Comparison**: Multi-asset overlay analysis
- **Study Templates**: Pre-configured indicator combinations

### **Migration Strategy**
- **Phase 1**: Parallel implementation with existing lightweight-charts
- **Phase 2**: Advanced Charts component with enhanced features
- **Seamless Integration**: Maintains existing data pipeline and selection functionality
- **Theme Support**: Dark/light mode compatibility
- **Custom Datafeed**: Integration with existing OHLCV data structure

### **Advanced Features Available**
```javascript
// Enhanced chart capabilities
enabled_features: [
    "study_templates",
    "compare_symbol", 
    "volume_force_overlay",
    "left_toolbar",
    "header_symbol_search",
    "header_interval_dialog_button"
]
```

### **Chart Activation Guide** (`CHART_ACTIVATION.md`)
- **Quick Setup**: 3-step activation process
- **Interactive Features**: Zoom, pan, crosshair, tooltips
- **Technical Indicators**: SMA, EMA, Bollinger Bands
- **Volume Visualization**: Histogram overlay
- **Responsive Design**: Auto-resizing charts
- **Professional Styling**: TradingView-like appearance

## 🔧 **Development Infrastructure** ⭐ **NEW**

### **Modular Architecture** (`src/README.md`)
- **Component Library**: 23+ reusable components
- **Separation of Concerns**: Data, vector, and chart components
- **PropTypes Validation**: Type checking for all components
- **Context API**: Global state management
- **Custom Hooks**: Data fetching and theme management
- **Utility Functions**: Centralized formatting and calculations

### **Component Structure**
```
src/components/
├── shared/                    # Reusable UI components
│   ├── InfoTooltip.jsx       # Helpful tooltips
│   ├── ThemeToggle.jsx       # Dark/light mode
│   ├── LoadingSpinner.jsx    # Loading states
│   └── ErrorDisplay.jsx      # Error handling
├── data-dashboard/           # Data analysis components
│   ├── DataDashboard.jsx     # Main container
│   ├── DataStats.jsx         # Statistics cards
│   ├── DataControls.jsx      # Symbol/timeframe controls
│   ├── DataTable.jsx         # Trading data table
│   ├── AdvancedFilters.jsx   # Search and filters
│   ├── Pagination.jsx        # Table pagination
│   ├── TablesList.jsx        # Table overview
│   └── DebugPanel.jsx        # Debug information
└── vector-dashboard/         # Vector analysis components
    ├── VectorDashboard.jsx   # Main container
    ├── VectorControls.jsx    # Vector controls
    ├── VectorStats.jsx       # Statistics display
    ├── VectorTypeSelector.jsx # Type selection
    ├── VectorVisualization.jsx # Visualization container
    ├── VectorHeatmap.jsx     # Heatmap display
    └── VectorComparison.jsx  # Side-by-side analysis
```

### **Development Tools**
- **ESLint Configuration**: Modern React linting rules
- **PostCSS Setup**: TailwindCSS processing
- **Vite Configuration**: Fast development server
- **TypeScript Ready**: Type definitions included
- **Hot Module Replacement**: Instant code updates

## 🐛 **Current Development Status** ⭐ **NEW**

### **Active Development Areas** (`current bugs.txt`)
- **Vector Dashboard**: Enhanced heatmap interaction (click-to-info functionality)
- **LLM Dashboard**: Backend integration with LangChain/LlamaIndex
- **AI Model Integration**: Hugging Face models and GPT/Claude APIs
- **Similarity Analysis**: Global mean vector and single candle vector integration
- **Chart Marking**: Model training for chart annotation capabilities

### **Development Priorities**
```bash
# Current focus areas:
1. Backend similarity checks using global mean vectors
2. Single candle vector similarity analysis
3. Combined similarity incorporating norm_ohlc + iso_ohlc
4. Model training for chart marking (768-dimensional vectors)
5. SVG visibility for chart analysis
```

### **Technical Debt & Improvements**
- **Code Cleanup**: Remove unnecessary components and optimize performance
- **Backend Enhancement**: Strengthen API layer and error handling
- **Frontend Refinement**: Clean up UI once backend is solid
- **Feature Integration**: Implement AI-suggested improvements

## 📈 **Performance & Optimization** ⭐ **NEW**

### **Current Performance Metrics**
- **TradingDashboard.jsx**: 194KB, 4,329 lines (monolithic implementation)
- **Chart.jsx**: 145KB, 3,543 lines (comprehensive charting)
- **LLMDashboard.jsx**: 99KB, 2,288 lines (AI assistant framework)
- **Backend Services**: 22KB trading service, 12KB router logic

### **Optimization Strategies**
- **Modular Architecture**: Available but not currently active
- **Component Splitting**: 23+ reusable components ready for use
- **Lazy Loading**: Component-level code splitting
- **Memory Management**: Optimized for large datasets
- **GPU Acceleration**: CUDA support for vector processing

### **Scalability Considerations**
- **Database Optimization**: Efficient queries with proper indexing
- **Caching Strategy**: Label queries and vector calculations
- **Batch Processing**: Configurable batch sizes for ML operations
- **Memory Monitoring**: Browser memory management for large selections

## 🔬 **Research & Development** ⭐ **NEW**

### **SPY Prediction Model v2 Plan** (`SPY_Prediction_Model_v2_Plan.md`)
- **Strategic Improvements**: 21 → 400+ features
- **Advanced Architecture**: Enhanced LSTM and ensemble methods
- **Market Regime Detection**: Volatility and trend analysis
- **Multi-Timeframe Context**: Cross-timeframe correlation features
- **Performance Enhancement**: Dynamic model weighting and selection

### **Feature Expansion Strategy**
```python
# v2 Feature Categories:
1. Core Vectors (789 features)
2. Technical Indicators (50+ features)
3. Multi-Timeframe Context (35+ features)
4. Market Regime Detection (15+ features)
5. Calendar Effects (20+ features)
6. Volatility Analysis (10+ features)
```

### **Model Architecture Improvements**
- **Enhanced LSTM**: Multi-layer with attention mechanisms
- **Dynamic Ensemble**: Adaptive model weighting
- **Feature Selection**: Automated feature importance ranking
- **Hyperparameter Optimization**: Grid search and Bayesian optimization
- **Cross-Validation**: Time series specific validation strategies

## 🎯 **Integration Points** ⭐ **NEW**

### **AI/ML Pipeline Integration**
- **Vector Generation**: `compute.py` with 6 vector types
- **Model Training**: SPY prediction models v1 and v2
- **Feature Engineering**: Advanced technical indicators
- **Backtesting**: Historical strategy validation
- **Real-time Prediction**: Live market analysis

### **Database Schema Evolution**
- **Trading Tables**: 21 tables across 3 symbols and 7 timeframes
- **Label Tables**: TJR, swing, and FVG labels for all SPY timeframes
- **Vector Columns**: 8 different vector types per table
- **Future Labels**: Binary classification for price direction prediction

### **API Endpoint Expansion**
- **Core Trading**: OHLCV data with optional vectors
- **Label Systems**: TJR, swing, and FVG labels
- **Shape Similarity**: Advanced pattern analysis
- **Model Predictions**: Real-time forecasting endpoints
- **Backtesting**: Historical strategy testing

---

**Daygent** represents a comprehensive agentic trading intelligence platform, combining quantitative analysis, machine learning, and professional trading tools in a unified interface. The monolithic architecture ensures seamless integration while the modular components provide future scalability options.

**🎯 Current Status**: Fully functional with 4 integrated dashboards, universal TJR and swing marker support, advanced candle selection, dynamic vector detection including ISO vectors with shape similarity, professional charting capabilities, and AI-ready framework for future model integration.

**Latest Updates**:
- 🎯 **Advanced Candle Selection System**: Interactive multi-mode selection with drag, range, and time-based selection
- 📊 **Multi-Timeframe Breakdown Analysis**: Secondary chart showing candle composition across timeframes
- 🏷️ **Enhanced Trading Labels API**: Universal TJR and swing marker support with flexible table naming
- 🤖 **Advanced ML Vector Pipeline**: Six vector types with GPU acceleration and BERT embeddings
- 🔬 **Multi-Metric Shape Similarity**: Hybrid similarity engine with 4 distance metrics
- 📈 **Professional Chart Features**: Lightweight Charts v5.0 with advanced selection overlays
- 🔄 **Dynamic Vector Detection**: Automatic detection of available vector types from database
- 📊 **Real-time Selection Statistics**: Live analysis of selected candle patterns and trends
- 🎯 **Binary Classification Labels**: Future price direction column added to all 21 backtest tables

**Credits**:
- Universal marker support and robust table detection added July 2024
- Comprehensive agentic trading intelligence platform developed 2024
