#!/usr/bin/env python3
"""
POC Training Script - Simplified LGBM Training for On-Demand Model Creation

This script trains a LightGBM model on a user-selected candle with configurable parameters.
No pre-test folds, no Optuna optimization, no meta-predictors - just straightforward training.

Usage:
    python train_poc_simple.py --selected_candle_date "2023-06-15" --train_years 3 --test_window_size 35 
                               --learning_rate 0.05 --num_leaves 31 --max_depth 6 --min_child_samples 20
                               --output_dir "./on_demand_runs/20231215_143022"
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
import joblib
import lightgbm as lgb


# %% COMMAND LINE ARGUMENTS
def parse_arguments():
    """Parse command line arguments for POC training."""
    parser = argparse.ArgumentParser(
        description="POC Training Script - Simplified LGBM Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--selected_candle_date', type=str, required=True,
                       help='ISO timestamp of the selected candle (end of training period)')
    parser.add_argument('--train_years', type=float, default=3.0,
                       help='Number of years before selected candle for training (default: 3)')
    parser.add_argument('--test_window_size', type=int, default=35,
                       help='Number of candles after selected candle for testing (default: 35)')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                       help='LightGBM learning rate (default: 0.05)')
    parser.add_argument('--num_leaves', type=int, default=31,
                       help='LightGBM num_leaves (default: 31)')
    parser.add_argument('--max_depth', type=int, default=6,
                       help='LightGBM max_depth (default: 6)')
    parser.add_argument('--min_child_samples', type=int, default=20,
                       help='LightGBM min_child_samples (default: 20)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save model artifacts')
    
    return parser.parse_args()


# %% CONFIG
DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5433/trading_data"

# Feature names matching omg.py structure (using basic 16 features, not the extended 20-feature set)
FEATURE_NAMES: List[str] = [
    "raw_o","raw_h","raw_l","raw_c","raw_v",
    "iso_0","iso_1","iso_2","iso_3",
    "tf_1d","tf_4h",
    "hl_range","price_change","upper_shadow","lower_shadow","volume_m"
]


# %% HELPER FUNCTIONS
def parse_vec(value) -> Optional[np.ndarray]:
    """Parse vector from database (handles various formats)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        s = value.strip().strip('[]"')
        if not s:
            return None
        try:
            return np.array([float(x.strip()) for x in s.split(',')])
        except Exception:
            return None
    try:
        return np.array(value, dtype=float)
    except Exception:
        return None


def build_feature_vector_1d(raw_ohlcv: np.ndarray, iso_ohlc: np.ndarray) -> np.ndarray:
    """Build feature vector from raw OHLCV and ISO features (basic 16-feature version)."""
    if len(raw_ohlcv) != 5 or len(iso_ohlc) != 4:
        raise ValueError("Bad vector lengths for raw_ohlcv or iso_ohlc")
    
    o, h, l, c, v = raw_ohlcv
    feats: List[float] = []
    
    # Raw OHLCV
    feats.extend([o, h, l, c, v])
    
    # ISO features
    feats.extend(list(iso_ohlc))
    
    # Timeframe one-hot encoding (1d = [1, 0], 4h = [0, 1])
    feats.extend([1, 0])
    
    # Derived features
    hl_range = (h - l) / c if c else 0.0
    price_change = (c - o) / o if o else 0.0
    upper_shadow = (h - c) / c if c else 0.0
    lower_shadow = (c - l) / c if c else 0.0
    volume_m = v / 1_000_000.0
    
    feats.extend([hl_range, price_change, upper_shadow, lower_shadow, volume_m])
    
    return np.array(feats, dtype=float)


def extract_features_row(row: pd.Series) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Extract feature vector and label from a DataFrame row."""
    raw_ohlcv = parse_vec(row.get("raw_ohlcv_vec"))
    iso_ohlc = parse_vec(row.get("iso_ohlc"))
    future = row.get("future")
    
    if raw_ohlcv is None or iso_ohlc is None or future is None or (isinstance(future, float) and np.isnan(future)):
        return None, None
    
    try:
        return build_feature_vector_1d(raw_ohlcv, iso_ohlc), int(future)
    except Exception:
        return None, None


def extract_Xy(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from a DataFrame."""
    X: List[np.ndarray] = []
    y: List[int] = []
    
    for _, row in frame.iterrows():
        fv, lbl = extract_features_row(row)
        if fv is not None:
            X.append(fv)
            y.append(lbl)
    
    return np.array(X), np.array(y)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """Compute comprehensive classification metrics."""
    has_two_classes = len(np.unique(y_true)) == 2
    
    # AUC (only if both classes present)
    try:
        auc_val = float(roc_auc_score(y_true, y_proba)) if has_two_classes else None
    except Exception:
        auc_val = None
    
    # Standard metrics
    try:
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        bacc = float(balanced_accuracy_score(y_true, y_pred))
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        acc = prec = rec = f1 = bacc = mcc = float("nan")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "balanced_accuracy": bacc,
        "mcc": mcc,
        "auc": auc_val,
    }


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """Find best threshold by maximizing F1 score."""
    thr_grid = np.round(np.arange(0.30, 0.80 + 1e-9, 0.01), 2)
    
    best_thr = 0.5
    best_f1 = -1.0
    best_metrics = None
    
    for thr in thr_grid:
        pred = (y_proba >= thr).astype(int)
        metrics = compute_metrics(y_true, pred, y_proba)
        
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thr = float(thr)
            best_metrics = metrics
    
    return best_thr, best_metrics


def class_dist(y: np.ndarray) -> Dict[str, Any]:
    """Compute class distribution statistics."""
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    rate = pos / (pos + neg) if (pos + neg) else float('nan')
    return {"pos": pos, "neg": neg, "pos_rate": float(rate)}


# %% MAIN TRAINING PIPELINE
def main():
    args = parse_arguments()
    
    print("\n" + "="*70)
    print("  POC TRAINING - SIMPLIFIED LGBM MODEL")
    print("="*70)
    print(f"  Selected Candle: {args.selected_candle_date}")
    print(f"  Train Period: {args.train_years} years before selected candle")
    print(f"  Test Window: {args.test_window_size} candles after selected candle")
    print(f"  Output Directory: {args.output_dir}")
    print("="*70 + "\n")
    
    # Parse selected candle date
    try:
        selected_date = pd.to_datetime(args.selected_candle_date)
    except Exception as e:
        print(f"❌ Error parsing selected_candle_date: {e}")
        sys.exit(1)
    
    # Calculate train period boundaries
    train_start_date = selected_date - timedelta(days=int(args.train_years * 365))
    train_end_date = selected_date
    
    print(f"📅 Train Period: {train_start_date.strftime('%Y-%m-%d')} → {train_end_date.strftime('%Y-%m-%d')}")
    
    # %% LOAD DATA FROM DATABASE
    print("\n📊 Loading SPY 1D data from database...")
    try:
        engine = create_engine(DB_URL, pool_pre_ping=True)
        query = text(
            """
            -- Deduplicate overlap between backtest and fronttest by timestamp,
            -- preferring the fronttest row when both exist
            with combined as (
              select timestamp, raw_ohlcv_vec, iso_ohlc, future, 1 as src from backtest.spy_1d
              union all
              select timestamp, raw_ohlcv_vec, iso_ohlc, future, 2 as src from fronttest.spy_1d
            ), dedup as (
              select distinct on (timestamp)
                     timestamp, raw_ohlcv_vec, iso_ohlc, future, src
              from combined
              order by timestamp, src desc
            )
            select timestamp, raw_ohlcv_vec, iso_ohlc, future
            from dedup
            order by timestamp asc
            """
        )
        df = pd.read_sql_query(query, engine)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"✅ Loaded {len(df):,} total rows from database")
    except Exception as e:
        print(f"❌ Error loading data from database: {e}")
        sys.exit(1)
    
    # %% SPLIT DATA INTO TRAIN AND TEST
    print(f"\n🔪 Splitting data into train and test...")
    
    # Find selected candle index
    selected_idx = None
    for idx, row in df.iterrows():
        if pd.to_datetime(row["timestamp"]) == selected_date:
            selected_idx = idx
            break
    
    if selected_idx is None:
        # Find closest candle
        df["time_diff"] = abs((df["timestamp"] - selected_date).dt.total_seconds())
        selected_idx = df["time_diff"].idxmin()
        actual_date = df.loc[selected_idx, "timestamp"]
        print(f"⚠️  Exact candle not found. Using closest: {actual_date.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"✅ Found selected candle at index {selected_idx}")
    
    # Train data: all candles from train_start_date up to and including selected candle
    train_mask = (df["timestamp"] >= train_start_date) & (df["timestamp"] <= selected_date)
    train_df = df[train_mask].copy().reset_index(drop=True)
    
    # Test data: next test_window_size candles after selected candle
    test_start_idx = selected_idx + 1
    test_end_idx = test_start_idx + args.test_window_size
    test_df = df.iloc[test_start_idx:test_end_idx].copy().reset_index(drop=True)
    
    print(f"  Train: {len(train_df):,} candles ({train_df['timestamp'].min()} → {train_df['timestamp'].max()})")
    print(f"  Test: {len(test_df):,} candles ({test_df['timestamp'].min()} → {test_df['timestamp'].max()})")
    
    if len(train_df) < 100:
        print(f"❌ Error: Insufficient training data ({len(train_df)} candles). Need at least 100.")
        sys.exit(1)
    
    if len(test_df) == 0:
        print(f"❌ Error: No test data available after selected candle.")
        sys.exit(1)
    
    # %% FEATURE EXTRACTION
    print(f"\n🔧 Extracting features...")
    X_train, y_train = extract_Xy(train_df)
    X_test, y_test = extract_Xy(test_df)
    
    print(f"  Train: {X_train.shape} | Labels: {class_dist(y_train)}")
    print(f"  Test: {X_test.shape} | Labels: {class_dist(y_test)}")
    
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"❌ Error: Feature extraction failed. Check data quality.")
        sys.exit(1)
    
    # %% SCALING
    print(f"\n⚖️  Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Wrap in DataFrames for LightGBM
    X_train_df = pd.DataFrame(X_train_scaled, columns=FEATURE_NAMES)
    X_test_df = pd.DataFrame(X_test_scaled, columns=FEATURE_NAMES)
    
    # %% TRAIN MODEL
    print(f"\n🎓 Training LightGBM model...")
    print(f"  Params: lr={args.learning_rate}, num_leaves={args.num_leaves}, max_depth={args.max_depth}, min_child_samples={args.min_child_samples}")
    
    lgb_params = {
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "n_estimators": 4000,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42,
    }
    
    try:
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_train_df,
            y_train,
            eval_set=[(X_test_df, y_test)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
        )
        print(f"✅ Model trained successfully")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        sys.exit(1)
    
    # %% EVALUATE ON TRAIN
    print(f"\n📊 Evaluating on training data...")
    try:
        train_proba = model.predict_proba(X_train_df)[:, 1]
        train_thr, train_metrics = find_best_threshold(y_train, train_proba)
        
        print(f"  Best Threshold: {train_thr:.2f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  F1: {train_metrics['f1']:.4f}")
        auc_str = f"{train_metrics['auc']:.4f}" if train_metrics['auc'] is not None else 'undefined'
        print(f"  AUC: {auc_str}")
        print(f"  MCC: {train_metrics['mcc']:.4f}")
    except Exception as e:
        print(f"❌ Error evaluating on train: {e}")
        train_thr = 0.5
        train_metrics = {}
    
    # %% EVALUATE ON TEST
    print(f"\n🎯 Evaluating on test data (unseen)...")
    try:
        test_proba = model.predict_proba(X_test_df)[:, 1]
        test_pred = (test_proba >= train_thr).astype(int)
        test_metrics = compute_metrics(y_test, test_pred, test_proba)
        
        print(f"  Using Threshold: {train_thr:.2f} (from train)")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        auc_str = f"{test_metrics['auc']:.4f}" if test_metrics['auc'] is not None else 'undefined'
        print(f"  AUC: {auc_str}")
        print(f"  MCC: {test_metrics['mcc']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
    except Exception as e:
        print(f"❌ Error evaluating on test: {e}")
        test_metrics = {}
    
    # %% SAVE ARTIFACTS
    print(f"\n💾 Saving artifacts to {args.output_dir}...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate run timestamp
    run_ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = os.path.join(args.output_dir, f'model_poc_{run_ts}.joblib')
    try:
        joblib.dump(model, model_path)
        print(f"  ✅ Model saved: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"  ❌ Error saving model: {e}")
        sys.exit(1)
    
    # Save scaler
    scaler_path = os.path.join(args.output_dir, f'scaler_poc_{run_ts}.joblib')
    try:
        joblib.dump(scaler, scaler_path)
        print(f"  ✅ Scaler saved: {os.path.basename(scaler_path)}")
    except Exception as e:
        print(f"  ❌ Error saving scaler: {e}")
        sys.exit(1)
    
    # Save results JSON
    results = {
        "config": {
            "selected_candle_date": str(selected_date),
            "train_years": args.train_years,
            "test_window_size": args.test_window_size,
            "params": {
                "learning_rate": args.learning_rate,
                "num_leaves": args.num_leaves,
                "max_depth": args.max_depth,
                "min_child_samples": args.min_child_samples,
            }
        },
        "data_splits": {
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
            "train_start": str(train_df['timestamp'].min()),
            "train_end": str(train_df['timestamp'].max()),
            "test_start": str(test_df['timestamp'].min()) if len(test_df) > 0 else None,
            "test_end": str(test_df['timestamp'].max()) if len(test_df) > 0 else None,
            "train_class_dist": class_dist(y_train),
            "test_class_dist": class_dist(y_test),
        },
        "metrics": {
            "train": {
                "best_threshold": float(train_thr),
                **{k: (float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None) 
                   for k, v in train_metrics.items()}
            },
            "test": {
                "threshold_used": float(train_thr),
                **{k: (float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None) 
                   for k, v in test_metrics.items()}
            }
        },
        "artifacts": {
            "model": os.path.basename(model_path),
            "scaler": os.path.basename(scaler_path),
        },
        "run_timestamp_utc": run_ts,
        "feature_names": FEATURE_NAMES,
    }
    
    results_path = os.path.join(args.output_dir, f'results_poc_{run_ts}.json')
    try:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  ✅ Results saved: {os.path.basename(results_path)}")
    except Exception as e:
        print(f"  ❌ Error saving results: {e}")
        sys.exit(1)
    
    # Save run metadata
    run_meta = {
        "script_name": os.path.basename(__file__),
        "run_timestamp_utc": run_ts,
        "run_id": os.path.basename(args.output_dir),
        "selected_candle_date": str(selected_date),
        "artifacts": {
            "model": model_path,
            "scaler": scaler_path,
            "results": results_path,
        },
        "status": "completed",
    }
    
    meta_path = os.path.join(args.output_dir, f'run_meta_{run_ts}.json')
    try:
        with open(meta_path, 'w') as f:
            json.dump(run_meta, f, indent=2)
        print(f"  ✅ Metadata saved: {os.path.basename(meta_path)}")
    except Exception as e:
        print(f"  ❌ Error saving metadata: {e}")
    
    # %% SUMMARY
    print("\n" + "="*70)
    print("  ✅ POC TRAINING COMPLETE")
    print("="*70)
    print(f"  📁 Artifacts saved to: {args.output_dir}")
    print(f"  📊 Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
    print(f"  📊 Test F1: {test_metrics.get('f1', 0):.4f}")
    print(f"  📊 Test MCC: {test_metrics.get('mcc', 0):.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

