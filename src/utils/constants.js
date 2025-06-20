/**
 * Trading Dashboard Constants
 */

// API Configuration
export const API_BASE_URL = 'http://localhost:8000/api/trading';

// Symbol Options
export const SYMBOLS = {
  es: { value: 'es', label: 'ES (E-mini S&P 500)', color: 'text-blue-600' },
  eurusd: { value: 'eurusd', label: 'EURUSD (Euro/US Dollar)', color: 'text-green-600' },
  spy: { value: 'spy', label: 'SPY (SPDR S&P 500 ETF)', color: 'text-purple-600' }
};

// Timeframe Options
export const TIMEFRAMES = {
  '1m': { value: '1m', label: '1 Minute' },
  '5m': { value: '5m', label: '5 Minutes' },
  '15m': { value: '15m', label: '15 Minutes' },
  '30m': { value: '30m', label: '30 Minutes' },
  '1h': { value: '1h', label: '1 Hour' },
  '4h': { value: '4h', label: '4 Hours' },
  '1d': { value: '1d', label: '1 Day' }
};

// Row Limit Options
export const ROW_LIMITS = [25, 50, 100, 250, 500];

// Vector Type Definitions
export const VECTOR_TYPES = [
  { 
    key: 'raw_ohlc_vec', 
    name: 'Raw OHLC', 
    color: 'bg-blue-500', 
    description: 'Direct numerical values',
    hasVolume: false
  },
  { 
    key: 'raw_ohlcv_vec', 
    name: 'Raw OHLCV', 
    color: 'bg-blue-600', 
    description: 'With volume data',
    hasVolume: true
  },
  { 
    key: 'norm_ohlc', 
    name: 'Normalized OHLC', 
    color: 'bg-green-500', 
    description: 'Z-score normalized',
    hasVolume: false
  },
  { 
    key: 'norm_ohlcv', 
    name: 'Normalized OHLCV', 
    color: 'bg-green-600', 
    description: 'Z-score with volume',
    hasVolume: true
  },
  { 
    key: 'BERT_ohlc', 
    name: 'BERT OHLC', 
    color: 'bg-purple-500', 
    description: 'Semantic embeddings',
    hasVolume: false
  },
  { 
    key: 'BERT_ohlcv', 
    name: 'BERT OHLCV', 
    color: 'bg-purple-600', 
    description: 'Semantic with volume',
    hasVolume: true
  }
];

// Dashboard Modes
export const DASHBOARD_MODES = {
  DATA: 'data',
  VECTOR: 'vector'
};

// View Modes for Vector Dashboard
export const VECTOR_VIEW_MODES = {
  HEATMAP: 'heatmap',
  COMPARISON: 'comparison'
};

// Sort Orders
export const SORT_ORDERS = {
  ASC: 'asc',
  DESC: 'desc'
};

// Default Values
export const DEFAULTS = {
  SYMBOL: 'es',
  TIMEFRAME: '1d',
  ROW_LIMIT: 50,
  SORT_ORDER: 'desc',
  SORT_COLUMN: 'timestamp',
  VECTOR_TYPE: 'raw_ohlc_vec',
  DASHBOARD_MODE: 'data'
}; 