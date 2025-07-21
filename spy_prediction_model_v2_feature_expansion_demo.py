#!/usr/bin/env python3
"""
SPY Prediction Model v2 - Feature Expansion Demo

This script demonstrates how we'll expand from v1's 21 features to v2's 400+ features
by integrating all available vector types and advanced technical indicators.
"""

import pandas as pd
import numpy as np
import talib

def extract_features_v1(df):
    """v1 Feature Extraction (21 features total)"""
    features = []
    
    for idx, row in df.iterrows():
        # Parse vectors
        raw_ohlcv = parse_vector(row['raw_ohlcv_vec'])  # 5 features
        iso_ohlc = parse_vector(row['iso_ohlc_vec'])    # 4 features
        
        # Create feature vector
        feature_vector = []
        feature_vector.extend(raw_ohlcv)                # 5 features
        feature_vector.extend(iso_ohlc)                 # 4 features
        
        # Timeframe encoding (one-hot)
        timeframe_encoding = [0] * 7  # 7 timeframes
        tf_idx = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'].index(row['timeframe'])
        timeframe_encoding[tf_idx] = 1
        feature_vector.extend(timeframe_encoding)       # 7 features
        
        # Basic technical indicators
        o, h, l, c, v = raw_ohlcv
        feature_vector.extend([
            (h - l) / c if c != 0 else 0,               # HL range
            (c - o) / o if o != 0 else 0,               # Price change
            (h - c) / c if c != 0 else 0,               # Upper shadow
            (c - l) / c if c != 0 else 0,               # Lower shadow
            v / 1000000,                                # Volume in millions
        ])                                              # 5 features
        
        features.append(feature_vector)
    
    return np.array(features)  # Shape: (n_samples, 21)

def extract_features_v2(df):
    """v2 Feature Extraction (400+ features total)"""
    features = []
    
    for idx, row in df.iterrows():
        feature_vector = []
        
        # ═══════════════════════════════════════════════════════════════
        # 1. ALL VECTOR TYPES (789 features total)
        # ═══════════════════════════════════════════════════════════════
        
        # Original v1 vectors (keep for backward compatibility)
        raw_ohlcv = parse_vector(row['raw_ohlcv_vec'])      # 5 features
        iso_ohlc = parse_vector(row['iso_ohlc_vec'])        # 4 features
        feature_vector.extend(raw_ohlcv)
        feature_vector.extend(iso_ohlc)
        
        # NEW: Additional vector types from your database
        raw_ohlc = parse_vector(row['raw_ohlc_vec'])        # 4 features  
        norm_ohlc = parse_vector(row['norm_ohlc'])          # 4 features
        norm_ohlcv = parse_vector(row['norm_ohlcv'])        # 5 features
        BERT_ohlc = parse_vector(row['BERT_ohlc'])          # 384 features
        BERT_ohlcv = parse_vector(row['BERT_ohlcv'])        # 384 features
        
        feature_vector.extend(raw_ohlc)
        feature_vector.extend(norm_ohlc) 
        feature_vector.extend(norm_ohlcv)
        feature_vector.extend(BERT_ohlc)
        feature_vector.extend(BERT_ohlcv)
        
        # Timeframe encoding (same as v1)
        timeframe_encoding = [0] * 7
        tf_idx = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'].index(row['timeframe'])
        timeframe_encoding[tf_idx] = 1
        feature_vector.extend(timeframe_encoding)           # 7 features
        
        # ═══════════════════════════════════════════════════════════════
        # 2. ADVANCED TECHNICAL INDICATORS (50+ features)
        # ═══════════════════════════════════════════════════════════════
        
        o, h, l, c, v = raw_ohlcv
        
        # Price arrays for TA-Lib (assume we have historical data)
        # In real implementation, we'd need sliding windows
        price_data = get_price_history(row, window=100)  # Helper function
        
        # Momentum Indicators (8 features)
        feature_vector.extend([
            talib.RSI(price_data['close'], timeperiod=7)[-1],
            talib.RSI(price_data['close'], timeperiod=14)[-1], 
            talib.RSI(price_data['close'], timeperiod=21)[-1],
            talib.RSI(price_data['close'], timeperiod=50)[-1],
            talib.WILLR(price_data['high'], price_data['low'], price_data['close'], timeperiod=14)[-1],
            talib.WILLR(price_data['high'], price_data['low'], price_data['close'], timeperiod=21)[-1],
            talib.ROC(price_data['close'], timeperiod=10)[-1],
            talib.ROC(price_data['close'], timeperiod=20)[-1],
        ])
        
        # MACD Family (9 features)
        macd_12_26, macd_signal_12_26, macd_hist_12_26 = talib.MACD(price_data['close'], 12, 26, 9)
        macd_8_21, macd_signal_8_21, macd_hist_8_21 = talib.MACD(price_data['close'], 8, 21, 9)
        macd_5_13, macd_signal_5_13, macd_hist_5_13 = talib.MACD(price_data['close'], 5, 13, 9)
        feature_vector.extend([
            macd_12_26[-1], macd_signal_12_26[-1], macd_hist_12_26[-1],
            macd_8_21[-1], macd_signal_8_21[-1], macd_hist_8_21[-1], 
            macd_5_13[-1], macd_signal_5_13[-1], macd_hist_5_13[-1],
        ])
        
        # Bollinger Bands (6 features)
        bb_upper_20, bb_middle_20, bb_lower_20 = talib.BBANDS(price_data['close'], timeperiod=20)
        bb_upper_50, bb_middle_50, bb_lower_50 = talib.BBANDS(price_data['close'], timeperiod=50)
        feature_vector.extend([
            bb_upper_20[-1], bb_lower_20[-1], (c - bb_lower_20[-1]) / (bb_upper_20[-1] - bb_lower_20[-1]),  # %B
            bb_upper_50[-1], bb_lower_50[-1], (c - bb_lower_50[-1]) / (bb_upper_50[-1] - bb_lower_50[-1]),  # %B
        ])
        
        # Volume Indicators (6 features)
        feature_vector.extend([
            talib.OBV(price_data['close'], price_data['volume'])[-1] / 1e6,  # On Balance Volume
            talib.SMA(price_data['volume'], timeperiod=10)[-1] / 1e6,        # Volume SMA 10
            talib.SMA(price_data['volume'], timeperiod=20)[-1] / 1e6,        # Volume SMA 20
            talib.AD(price_data['high'], price_data['low'], price_data['close'], price_data['volume'])[-1] / 1e6,  # A/D Line
            v / talib.SMA(price_data['volume'], timeperiod=20)[-1],          # Volume ratio
            talib.ROC(price_data['volume'], timeperiod=10)[-1],              # Volume ROC
        ])
        
        # Volatility Indicators (5 features)
        feature_vector.extend([
            talib.ATR(price_data['high'], price_data['low'], price_data['close'], timeperiod=14)[-1],
            talib.ATR(price_data['high'], price_data['low'], price_data['close'], timeperiod=21)[-1],
            talib.ATR(price_data['high'], price_data['low'], price_data['close'], timeperiod=50)[-1],
            talib.NATR(price_data['high'], price_data['low'], price_data['close'], timeperiod=14)[-1],  # Normalized ATR
            (h - l) / talib.SMA(price_data['close'], timeperiod=20)[-1],     # Range/Price ratio
        ])
        
        # ═══════════════════════════════════════════════════════════════
        # 3. MULTI-TIMEFRAME FEATURES (35+ features)
        # ═══════════════════════════════════════════════════════════════
        
        # Cross-timeframe correlations (would need implementation)
        tf_correlations = calculate_timeframe_correlations(row)  # 21 features
        tf_momentum_alignment = calculate_momentum_alignment(row)  # 7 features  
        tf_volatility_ratios = calculate_volatility_ratios(row)   # 7 features
        
        feature_vector.extend(tf_correlations)
        feature_vector.extend(tf_momentum_alignment)
        feature_vector.extend(tf_volatility_ratios)
        
        # ═══════════════════════════════════════════════════════════════
        # 4. MARKET REGIME FEATURES (15+ features)
        # ═══════════════════════════════════════════════════════════════
        
        # Calendar effects
        timestamp = pd.to_datetime(row['timestamp'])
        feature_vector.extend([
            timestamp.hour,                    # Hour of day
            timestamp.dayofweek,              # Day of week
            timestamp.month,                  # Month
            timestamp.quarter,                # Quarter
            is_market_open(timestamp),        # Market session
        ])
        
        # Volatility regime
        recent_volatility = calculate_recent_volatility(row)
        feature_vector.extend([
            recent_volatility,
            recent_volatility > get_volatility_percentile(75),  # High vol regime
            recent_volatility < get_volatility_percentile(25),  # Low vol regime
        ])
        
        # Trend regime
        trend_strength = calculate_trend_strength(row)
        feature_vector.extend([
            trend_strength,
            trend_strength > 0.7,             # Strong trend
            abs(trend_strength) < 0.3,        # Ranging market
        ])
        
        features.append(feature_vector)
    
    return np.array(features)  # Shape: (n_samples, 400+)

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE COMPARISON SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def compare_feature_sets():
    """Compare v1 vs v2 feature sets"""
    
    print("🔍 SPY Prediction Model - Feature Set Comparison")
    print("=" * 60)
    
    print("\n📊 v1 Features (21 total):")
    print("├── raw_ohlcv_vec      : 5 features  (Open, High, Low, Close, Volume)")
    print("├── iso_ohlc_vec       : 4 features  (Isolation forest)")  
    print("├── timeframe_encoding : 7 features  (One-hot timeframe)")
    print("└── basic_indicators   : 5 features  (HL range, price change, shadows, volume)")
    print(f"    TOTAL v1 FEATURES  : 21")
    
    print("\n🚀 v2 Features (400+ total):")
    print("├── Vector Integration:")
    print("│   ├── raw_ohlcv_vec   : 5 features   (Keep from v1)")
    print("│   ├── raw_ohlc_vec    : 4 features   (NEW)")
    print("│   ├── iso_ohlc_vec    : 4 features   (Keep from v1)")
    print("│   ├── norm_ohlc       : 4 features   (NEW)")
    print("│   ├── norm_ohlcv      : 5 features   (NEW)")
    print("│   ├── BERT_ohlc       : 384 features (NEW - Semantic)")
    print("│   └── BERT_ohlcv      : 384 features (NEW - Semantic)")
    print("│")
    print("├── Technical Indicators:")
    print("│   ├── Momentum        : 8 features   (RSI, Williams %R, ROC)")
    print("│   ├── MACD Family     : 9 features   (3 MACD variations)")
    print("│   ├── Bollinger Bands : 6 features   (2 period variations)")
    print("│   ├── Volume Analysis : 6 features   (OBV, Volume SMA, A/D)")
    print("│   └── Volatility      : 5 features   (ATR variations)")
    print("│")
    print("├── Multi-Timeframe:")
    print("│   ├── TF Correlations : 21 features  (Price correlations)")
    print("│   ├── Momentum Align  : 7 features   (Cross-TF momentum)")
    print("│   └── Volatility Ratios: 7 features  (Cross-TF volatility)")
    print("│")
    print("└── Market Regime:")
    print("    ├── Calendar Effects: 5 features   (Hour, day, month, etc.)")
    print("    ├── Volatility Regime: 3 features  (High/low vol detection)")
    print("    └── Trend Regime   : 3 features   (Trend/range detection)")
    print(f"    TOTAL v2 FEATURES  : 400+")
    
    print(f"\n📈 IMPROVEMENT: {400/21:.1f}x more features ({400-21:+d} additional features)")
    print(f"🎯 KEY ADDITIONS:")
    print(f"   • 768 BERT semantic features (384 + 384)")
    print(f"   • 9 normalized statistical features") 
    print(f"   • 34 advanced technical indicators")
    print(f"   • 35 multi-timeframe context features")
    print(f"   • 11 market regime detection features")

# Helper functions (would be implemented in full version)
def parse_vector(vector_str):
    """Parse vector string to numpy array"""
    if pd.isna(vector_str) or vector_str is None:
        return [0] * 5  # Default fallback
    if isinstance(vector_str, str):
        vector_str = vector_str.strip('[]')
        return [float(x.strip()) for x in vector_str.split(',')]
    return list(vector_str)

def get_price_history(row, window=100):
    """Get historical price data for technical indicators"""
    # This would query the database for historical data
    # For demo, return dummy data
    return {
        'high': np.random.randn(window).cumsum() + 100,
        'low': np.random.randn(window).cumsum() + 98,
        'close': np.random.randn(window).cumsum() + 99,
        'volume': np.random.randint(1000000, 10000000, window)
    }

def calculate_timeframe_correlations(row):
    """Calculate price correlations across timeframes"""
    # Implementation would correlate price movements across all 7 timeframes
    return np.random.randn(21).tolist()  # Placeholder

def calculate_momentum_alignment(row):
    """Calculate momentum alignment across timeframes"""
    return np.random.randn(7).tolist()  # Placeholder

def calculate_volatility_ratios(row):
    """Calculate volatility ratios across timeframes"""
    return np.random.randn(7).tolist()  # Placeholder

def is_market_open(timestamp):
    """Check if market is open"""
    return 1 if 9.5 <= timestamp.hour <= 16 else 0

def calculate_recent_volatility(row):
    """Calculate recent volatility"""
    return np.random.rand()  # Placeholder

def get_volatility_percentile(percentile):
    """Get volatility percentile threshold"""
    return 0.02 * (percentile / 50)  # Placeholder

def calculate_trend_strength(row):
    """Calculate trend strength"""
    return np.random.rand() * 2 - 1  # Placeholder (-1 to 1)

if __name__ == "__main__":
    compare_feature_sets() 