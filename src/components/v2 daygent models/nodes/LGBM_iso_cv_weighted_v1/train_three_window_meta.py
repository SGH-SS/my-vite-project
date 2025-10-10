#!/usr/bin/env python3
"""
LightGBM 1D Training — Three-Window Meta-Optimization Version

Windows:
- Train: all data in last 6y excluding pre-test and test
- Pre-test: 1y immediately before test, split into folds (~35 bars) with purge
- Test: last 35 trading rows

Flow:
1) Build features, split into train/pre-test/test.
2) Fit baseline model on Train window.
3) Overfit in Pre-test folds: for each fold, perform a targeted hyperparameter search to maximize validation AUC/Accuracy. Collect (features/summary → chosen HParams) pairs.
4) Meta-learn mapping from prior fold summaries to chosen HParams (simple rules/regressor over fold stats).
5) Infer HParams at the end of pre-test, train final model on Train+Pre-test, and evaluate on Test.
6) Print comprehensive summary and write rich JSON artifacts.

Note: This is an experimental pipeline intended to maximize practicality and interpretability.
"""

from __future__ import annotations

import os
import json
import time
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone
from pathlib import Path

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
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import joblib
import lightgbm as lgb


# %% CONFIG
DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5433/trading_data"
BACKTEST_TABLE = 'backtest.spy_1d'
FRONTTEST_TABLE = 'fronttest.spy_1d'

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'artifacts_lgbm_1d_iso_three_window')
os.makedirs(ARTIFACT_DIR, exist_ok=True)
RUNS_DIR = os.path.join(ARTIFACT_DIR, 'runs')
os.makedirs(RUNS_DIR, exist_ok=True)
RUN_TS = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
RUN_DIR = os.path.join(RUNS_DIR, RUN_TS)
os.makedirs(RUN_DIR, exist_ok=True)


# %% HELPERS
def parse_vec(value) -> Optional[np.ndarray]:
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


FEATURE_NAMES: List[str] = [
    "raw_o","raw_h","raw_l","raw_c","raw_v",
    "iso_0","iso_1","iso_2","iso_3",
    "tf_1d","tf_4h",
    "hl_range","price_change","upper_shadow","lower_shadow","volume_m"
]


def build_feature_vector_1d(raw_ohlcv: np.ndarray, iso_ohlc: np.ndarray) -> np.ndarray:
    if len(raw_ohlcv) != 5 or len(iso_ohlc) != 4:
        raise ValueError("Bad vector lengths for raw_ohlcv or iso_ohlc")
    o, h, l, c, v = raw_ohlcv
    feats: List[float] = []
    feats.extend([o, h, l, c, v])
    feats.extend(list(iso_ohlc))
    feats.extend([1, 0])
    hl_range = (h - l) / c if c else 0.0
    price_change = (c - o) / o if o else 0.0
    upper_shadow = (h - c) / c if c else 0.0
    lower_shadow = (c - l) / c if c else 0.0
    volume_m = v / 1_000_000.0
    feats.extend([hl_range, price_change, upper_shadow, lower_shadow, volume_m])
    return np.array(feats, dtype=float)


def extract_features_row(row: pd.Series) -> Tuple[Optional[np.ndarray], Optional[int]]:
    raw_ohlcv = parse_vec(row.get("raw_ohlcv_vec"))
    iso_ohlc = parse_vec(row.get("iso_ohlc"))
    future = row.get("future")
    if raw_ohlcv is None or iso_ohlc is None or future is None or (isinstance(future, float) and np.isnan(future)):
        return None, None
    try:
        return build_feature_vector_1d(raw_ohlcv, iso_ohlc), int(future)
    except Exception:
        return None, None


# Purge config
GAP_BARS = 5
PRETEST_FOLD_SIZE = 35
TEST_ROWS = 35  # ensure test window uses exact last N rows
FOLD_TIME_LIMIT = 210  # 3.5 minutes in seconds per fold


def window_indices_by_dates(dates: np.ndarray, test_len_days: int = 35, pretest_len_days: int = 365) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    # Assumes dates are sorted ascending, unique dates array
    last_day = pd.to_datetime(dates).max()
    test_start_day = last_day - pd.Timedelta(days=test_len_days - 1)
    pretest_start_day = test_start_day - pd.Timedelta(days=pretest_len_days)
    return {
        "pretest": (pd.Timestamp.combine(pretest_start_day, pd.Timestamp.min.time()).tz_localize("UTC"),
                     pd.Timestamp.combine(test_start_day - pd.Timedelta(days=1), pd.Timestamp.max.time()).tz_localize("UTC")),
        "test": (pd.Timestamp.combine(test_start_day, pd.Timestamp.min.time()).tz_localize("UTC"),
                  pd.Timestamp.combine(last_day, pd.Timestamp.max.time()).tz_localize("UTC")),
    }


def split_three_windows(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Backwards splitting to ensure exact test rows:
    1) Keep last 6 years of data
    2) Test = last TEST_ROWS rows (exact)
    3) Pre-test = 1 year immediately before test start (time-based)
    4) Train = remaining earlier rows in last 6 years
    """
    if len(df) == 0:
        raise RuntimeError("No data available")
    latest_ts = pd.to_datetime(df["timestamp"]).max()
    cutoff_ts = latest_ts - pd.DateOffset(years=6)
    df6 = df[df["timestamp"] >= cutoff_ts].copy().sort_values("timestamp").reset_index(drop=True)

    if len(df6) < TEST_ROWS:
        raise RuntimeError(f"Not enough bars in last 6 years to form a {TEST_ROWS}-row test window")

    # 1) Test window = last TEST_ROWS rows
    test_df = df6.tail(TEST_ROWS).copy().reset_index(drop=True)
    test_start = pd.to_datetime(test_df["timestamp"]).iloc[0]
    test_end = pd.to_datetime(test_df["timestamp"]).iloc[-1]

    # 2) Pre-test = 1 year immediately before test start
    pretest_start = test_start - pd.DateOffset(years=1)
    pre_df = df6[(df6["timestamp"] >= pretest_start) & (df6["timestamp"] < test_start)].copy().reset_index(drop=True)
    # Bounds for pre-test
    pretest_end = pd.to_datetime(pre_df["timestamp"]).iloc[-1] if len(pre_df) else (test_start - pd.Timedelta(microseconds=1))

    # 3) Train = remaining rows before pre-test start within 6-year window
    train_df = df6[df6["timestamp"] < pretest_start].copy().reset_index(drop=True)

    return {
        "train": train_df,
        "pretest": pre_df,
        "test": test_df,
        "pretest_bounds": (pretest_start, pretest_end),
        "test_bounds": (test_start, test_end),
    }


def make_folds_with_purge(n: int, fold_len: int = PRETEST_FOLD_SIZE, gap: int = GAP_BARS):
    # Sequential folds of length fold_len; apply purge by shifting validation start
    start = 0
    while True:
        end = start + fold_len
        if end > n:
            break
        val_start = start + gap
        if val_start >= end:
            start = end
            continue
        tr_idx = np.arange(start, val_start)  # before purge
        va_idx = np.arange(val_start, end)
        if len(tr_idx) == 0 or len(va_idx) == 0:
            start = end
            continue
        yield tr_idx, va_idx
        start = end


def extract_Xy(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[int] = []
    for _, row in frame.iterrows():
        fv, lbl = extract_features_row(row)
        if fv is not None:
            X.append(fv)
            y.append(lbl)
    return np.array(X), np.array(y)


def default_lgb_params() -> Dict[str, Any]:
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 6,
        "min_child_samples": 60,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "min_gain_to_split": 0.0,
        "n_estimators": 4000,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42,
        "class_weight": "balanced",
    }


def generate_smart_grid(prior_best: Optional[Dict[str, Any]] = None, n_samples: int = 100) -> List[Dict[str, Any]]:
    """
    Generate smart hyperparameter grid with random + focused search.
    If prior_best is provided, sample around those values.
    """
    base = default_lgb_params()
    grid = []
    
    # Define search ranges
    lr_range = (0.01, 0.15)
    leaves_range = (15, 127)
    depth_range = (3, 10)
    min_child_range = (10, 120)
    feat_frac_range = (0.5, 1.0)
    bag_frac_range = (0.5, 1.0)
    l1_range = (0.0, 1.0)
    l2_range = (0.0, 1.0)
    
    if prior_best:
        # Focused search around prior best
        center_lr = prior_best.get("learning_rate", 0.05)
        center_leaves = prior_best.get("num_leaves", 63)
        center_depth = prior_best.get("max_depth", 6)
        center_min_child = prior_best.get("min_child_samples", 60)
        center_feat = prior_best.get("feature_fraction", 0.85)
        center_bag = prior_best.get("bagging_fraction", 0.85)
        center_l1 = prior_best.get("lambda_l1", 0.1)
        center_l2 = prior_best.get("lambda_l2", 0.1)
        
        for _ in range(n_samples):
            g = dict(base)
            g.update({
                "learning_rate": np.clip(center_lr * np.random.uniform(0.5, 1.5), *lr_range),
                "num_leaves": int(np.clip(center_leaves + np.random.randint(-30, 31), *leaves_range)),
                "max_depth": int(np.clip(center_depth + np.random.randint(-2, 3), *depth_range)),
                "min_child_samples": int(np.clip(center_min_child + np.random.randint(-30, 31), *min_child_range)),
                "feature_fraction": np.clip(center_feat + np.random.uniform(-0.15, 0.15), *feat_frac_range),
                "bagging_fraction": np.clip(center_bag + np.random.uniform(-0.15, 0.15), *bag_frac_range),
                "lambda_l1": np.clip(center_l1 * np.random.uniform(0.1, 2.0), *l1_range),
                "lambda_l2": np.clip(center_l2 * np.random.uniform(0.1, 2.0), *l2_range),
            })
            grid.append(g)
    else:
        # Random search across full ranges
        for _ in range(n_samples):
            g = dict(base)
            g.update({
                "learning_rate": np.random.uniform(*lr_range),
                "num_leaves": int(np.random.randint(*leaves_range)),
                "max_depth": int(np.random.randint(*depth_range)),
                "min_child_samples": int(np.random.randint(*min_child_range)),
                "feature_fraction": np.random.uniform(*feat_frac_range),
                "bagging_fraction": np.random.uniform(*bag_frac_range),
                "lambda_l1": np.random.uniform(*l1_range),
                "lambda_l2": np.random.uniform(*l2_range),
            })
            grid.append(g)
    
    return grid


def evaluate_thresholds(y_true: np.ndarray, proba: np.ndarray, thr_grid: np.ndarray) -> Dict[str, Any]:
    best = {
        "thr": 0.5,
        "acc": float("nan"),
        "auc": float("nan"),
        "prec": float("nan"),
        "rec": float("nan"),
        "f1": float("nan"),
        "bacc": float("nan"),
    }
    try:
        auc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) == 2 else float("nan")
    except Exception:
        auc = float("nan")
    best_auc = auc
    # Select by F1 first, then accuracy as tie-breaker
    for thr in thr_grid:
        pred = (proba >= thr).astype(int)
        acc = accuracy_score(y_true, pred)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        bacc = balanced_accuracy_score(y_true, pred)
        if (np.isnan(best["f1"]) or f1 > best["f1"]) or (f1 == best["f1"] and acc > best.get("acc", 0)):
            best = {"thr": float(thr), "acc": float(acc), "auc": float(auc), "prec": float(prec), "rec": float(rec), "f1": float(f1), "bacc": float(bacc)}
            best_auc = auc
    best["auc"] = float(best_auc)
    return best


def summary_signature(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Rich fold summary for meta-learning features.
    Extract statistical properties that help predict optimal hyperparameters.
    """
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    rate = pos / (pos + neg) if (pos + neg) else float("nan")
    
    # Feature statistics
    feat_means = np.mean(X, axis=0) if len(X) else np.zeros(X.shape[1] if X.ndim == 2 else 0)
    feat_stds = np.std(X, axis=0) if len(X) else np.zeros(X.shape[1] if X.ndim == 2 else 0)
    
    # Class-conditional statistics
    if pos > 0 and neg > 0:
        X_pos = X[y == 1]
        X_neg = X[y == 0]
        separation = np.mean(np.abs(np.mean(X_pos, axis=0) - np.mean(X_neg, axis=0)))
    else:
        separation = 0.0
    
    return {
        "n": float(len(y)),
        "pos_rate": float(rate),
        "feat_mean_avg": float(np.mean(feat_means)) if len(feat_means) else 0.0,
        "feat_std_avg": float(np.mean(feat_stds)) if len(feat_stds) else 0.0,
        "feat_separation": float(separation),
        "n_features": float(X.shape[1] if X.ndim == 2 else 0),
    }


class MetaLearner:
    """
    Advanced meta-learner that learns hyperparameter selection from prior fold results.
    Uses ensemble of regressors to predict each hyperparameter based on fold characteristics.
    """
    def __init__(self):
        self.regressors = {
            "learning_rate": Ridge(alpha=1.0),
            "num_leaves": Ridge(alpha=1.0),
            "max_depth": Ridge(alpha=1.0),
            "min_child_samples": Ridge(alpha=1.0),
            "feature_fraction": Ridge(alpha=1.0),
            "bagging_fraction": Ridge(alpha=1.0),
            "lambda_l1": Ridge(alpha=1.0),
            "lambda_l2": Ridge(alpha=1.0),
        }
        self.fitted = False
    
    def _extract_features(self, fold_meta: Dict[str, Any]) -> np.ndarray:
        """Extract meta-features from fold metadata."""
        sig = fold_meta.get("train_signature", {})
        perf = fold_meta.get("best_f1", 0.0)
        auc = fold_meta.get("best_auc", 0.5)
        if auc is None or np.isnan(auc):
            auc = 0.5
        
        features = [
            sig.get("n", 0.0),
            sig.get("pos_rate", 0.5),
            sig.get("feat_mean_avg", 0.0),
            sig.get("feat_std_avg", 0.0),
            sig.get("feat_separation", 0.0),
            sig.get("n_features", 16.0),
            perf,  # F1 performance as meta-feature
            auc,   # AUC performance as meta-feature
        ]
        return np.array(features, dtype=float)
    
    def fit(self, prior_meta: List[Dict[str, Any]]):
        """Fit meta-learner on prior fold results."""
        if len(prior_meta) < 2:
            self.fitted = False
            return
        
        X_meta = []
        y_meta = {k: [] for k in self.regressors.keys()}
        
        for m in prior_meta:
            feat = self._extract_features(m)
            X_meta.append(feat)
            params = m.get("params", {})
            y_meta["learning_rate"].append(params.get("learning_rate", 0.05))
            y_meta["num_leaves"].append(params.get("num_leaves", 63))
            y_meta["max_depth"].append(params.get("max_depth", 6))
            y_meta["min_child_samples"].append(params.get("min_child_samples", 60))
            y_meta["feature_fraction"].append(params.get("feature_fraction", 0.85))
            y_meta["bagging_fraction"].append(params.get("bagging_fraction", 0.85))
            y_meta["lambda_l1"].append(params.get("lambda_l1", 0.1))
            y_meta["lambda_l2"].append(params.get("lambda_l2", 0.1))
        
        X_meta = np.array(X_meta)
        
        # Fit each regressor
        for key, reg in self.regressors.items():
            y = np.array(y_meta[key])
            if len(y) >= 2 and np.std(y) > 1e-6:
                try:
                    reg.fit(X_meta, y)
                except Exception:
                    pass
        
        self.fitted = True
    
    def predict(self, current_signature: Dict[str, float]) -> Dict[str, Any]:
        """Predict hyperparameters for the next fold based on its characteristics."""
        if not self.fitted:
            return default_lgb_params()
        
        # Build feature vector (without performance metrics for new fold)
        feat = np.array([
            current_signature.get("n", 0.0),
            current_signature.get("pos_rate", 0.5),
            current_signature.get("feat_mean_avg", 0.0),
            current_signature.get("feat_std_avg", 0.0),
            current_signature.get("feat_separation", 0.0),
            current_signature.get("n_features", 16.0),
            0.7,  # Assume reasonable F1
            0.55,  # Assume reasonable AUC
        ]).reshape(1, -1)
        
        params = default_lgb_params()
        try:
            params.update({
                "learning_rate": float(np.clip(self.regressors["learning_rate"].predict(feat)[0], 0.01, 0.15)),
                "num_leaves": int(np.clip(round(self.regressors["num_leaves"].predict(feat)[0]), 15, 127)),
                "max_depth": int(np.clip(round(self.regressors["max_depth"].predict(feat)[0]), 3, 10)),
                "min_child_samples": int(np.clip(round(self.regressors["min_child_samples"].predict(feat)[0]), 10, 120)),
                "feature_fraction": float(np.clip(self.regressors["feature_fraction"].predict(feat)[0], 0.5, 1.0)),
                "bagging_fraction": float(np.clip(self.regressors["bagging_fraction"].predict(feat)[0], 0.5, 1.0)),
                "lambda_l1": float(np.clip(self.regressors["lambda_l1"].predict(feat)[0], 0.0, 1.0)),
                "lambda_l2": float(np.clip(self.regressors["lambda_l2"].predict(feat)[0], 0.0, 1.0)),
            })
        except Exception:
            pass
        
        return params


def meta_infer_hparams_simple(prior_meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fallback simple weighted averaging if meta-learner fails."""
    if not prior_meta:
        return default_lgb_params()
    weights = []
    lrs = []
    leaves = []
    depths = []
    min_childs = []
    feat_fracs = []
    bag_fracs = []
    l1s = []
    l2s = []
    for m in prior_meta:
        w = max(1e-6, 0.5 * m.get("best_f1", 0) + 0.5 * (m.get("best_auc", 0) if not np.isnan(m.get("best_auc", np.nan)) else 0))
        weights.append(w)
        lrs.append(m.get("params", {}).get("learning_rate", 0.05))
        leaves.append(m.get("params", {}).get("num_leaves", 63))
        depths.append(m.get("params", {}).get("max_depth", 6))
        min_childs.append(m.get("params", {}).get("min_child_samples", 60))
        feat_fracs.append(m.get("params", {}).get("feature_fraction", 0.85))
        bag_fracs.append(m.get("params", {}).get("bagging_fraction", 0.85))
        l1s.append(m.get("params", {}).get("lambda_l1", 0.1))
        l2s.append(m.get("params", {}).get("lambda_l2", 0.1))
    w = np.array(weights)
    def wavg(vals):
        vals = np.array(vals, dtype=float)
        if vals.shape[0] != w.shape[0] or vals.shape[0] == 0:
            return float(np.mean(vals)) if vals.size else 0.0
        return float(np.sum(vals * w) / np.sum(w))
    params = default_lgb_params()
    params.update({
        "learning_rate": wavg(lrs),
        "num_leaves": int(round(wavg(leaves))),
        "max_depth": int(round(wavg(depths))),
        "min_child_samples": int(round(wavg(min_childs))),
        "feature_fraction": float(wavg(feat_fracs)),
        "bagging_fraction": float(wavg(bag_fracs)),
        "lambda_l1": float(wavg(l1s)),
        "lambda_l2": float(wavg(l2s)),
    })
    return params


# %% LOAD DATA
engine = create_engine(DB_URL, pool_pre_ping=True)
query = text(
    """
    with combined as (
      select timestamp, raw_ohlcv_vec, iso_ohlc, future from backtest.spy_1d
      union all
      select timestamp, raw_ohlcv_vec, iso_ohlc, future from fronttest.spy_1d
    )
    select * from combined order by timestamp asc
    """
)
df = pd.read_sql_query(query, engine)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"Loaded rows: {len(df):,}")


# %% THREE-WINDOW SPLIT
splits = split_three_windows(df)
train_df = splits["train"]
pre_df = splits["pretest"]
test_df = splits["test"]
pretest_bounds = splits["pretest_bounds"]
test_bounds = splits["test_bounds"]

print(f"Train rows: {len(train_df):,}, Pre-test rows: {len(pre_df):,}, Test rows: {len(test_df):,}")


# %% FEATURE EXTRACTION
X_train, y_train = extract_Xy(train_df)
X_pre, y_pre = extract_Xy(pre_df)
X_test, y_test = extract_Xy(test_df)
print(f"Shapes — Train: {X_train.shape}, Pre-test: {X_pre.shape}, Test: {X_test.shape}")


# %% SCALING
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_pre_s = scaler.transform(X_pre)
X_test_s = scaler.transform(X_test) if len(X_test) else np.empty((0, X_train_s.shape[1]))


# %% BASELINE TRAIN ON TRAIN WINDOW
baseline_params = default_lgb_params()
baseline_model = lgb.LGBMClassifier(**baseline_params)
baseline_model.fit(X_train_s, y_train)


# %% PRE-TEST: FOLD OVERFIT SEARCH WITH META-LEARNING
print("\n" + "="*70)
print("  PRE-TEST HYPERPARAMETER OPTIMIZATION (TIME-LIMITED PER FOLD)")
print("="*70)

thr_grid = np.round(np.arange(0.30, 0.80 + 1e-9, 0.01), 2)
fold_meta: List[Dict[str, Any]] = []
pre_n = len(y_pre)
best_models: List[lgb.LGBMClassifier] = []
best_settings: List[Dict[str, Any]] = []
best_thr_list: List[float] = []
fold_summaries: List[Dict[str, Any]] = []

meta_learner = MetaLearner()
prior_best_params: Optional[Dict[str, Any]] = None

for fold_idx, (tr_idx, va_idx) in enumerate(make_folds_with_purge(pre_n, PRETEST_FOLD_SIZE, GAP_BARS), start=1):
    fold_start_time = time.time()
    X_tr, y_tr = X_pre_s[tr_idx], y_pre[tr_idx]
    X_va, y_va = X_pre_s[va_idx], y_pre[va_idx]
    sig = summary_signature(X_tr, y_tr)
    
    print(f"\n{'─'*70}")
    print(f"✓ FOLD {fold_idx} | Train: {len(y_tr)} bars | Val: {len(y_va)} bars")
    print(f"  Pos rate: {sig.get('pos_rate', 0.0):.3f} | Separation: {sig.get('feat_separation', 0.0):.4f}")
    print(f"  Time limit: {FOLD_TIME_LIMIT}s | Strategy: {'Smart Grid (prior-guided)' if prior_best_params else 'Random Search'}")
    print(f"{'─'*70}")
    
    # Generate search grid (smart if we have prior best)
    n_samples = 150 if prior_best_params else 100
    grid = generate_smart_grid(prior_best_params, n_samples=n_samples)
    
    best_score = -1.0
    best_info: Dict[str, Any] = {}
    best_model_fold: Optional[lgb.LGBMClassifier] = None
    
    trials = 0
    improved_count = 0
    last_improvement = 0
    
    for param_idx, params in enumerate(grid):
        # Check time limit
        elapsed = time.time() - fold_start_time
        if elapsed > FOLD_TIME_LIMIT:
            print(f"  ⏱ Time limit reached ({elapsed:.1f}s). Stopping search.")
            break
        
        # Train and evaluate
        try:
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_va)[:, 1]
            stats = evaluate_thresholds(y_va, proba, thr_grid)
            score = 0.7 * stats["f1"] + 0.3 * (stats["auc"] if not np.isnan(stats["auc"]) else 0.5)
            
            trials += 1
            
            # Update best
            if score > best_score:
                best_score = score
                best_info = {
                    "params": params,
                    "best_thr": float(stats["thr"]),
                    "best_auc": float(stats["auc"]) if not np.isnan(stats["auc"]) else None,
                    "best_acc": float(stats["acc"]),
                    "best_prec": float(stats["prec"]),
                    "best_rec": float(stats["rec"]),
                    "best_f1": float(stats["f1"]),
                    "best_bacc": float(stats["bacc"]),
                    "score": float(score),
                }
                best_model_fold = clf
                improved_count += 1
                last_improvement = trials
                
                # Print improvement
                print(f"  ✓ Trial {trials:>3}/{len(grid)} | Score: {score:.4f} → NEW BEST | "
                      f"F1={stats['f1']:.4f}, AUC={stats['auc'] if not np.isnan(stats['auc']) else float('nan'):.4f}, θ={stats['thr']:.2f} | "
                      f"lr={params['learning_rate']:.4f}, leaves={params['num_leaves']}, depth={params['max_depth']}")
            else:
                # Print progress every 10 trials
                if trials % 10 == 0:
                    print(f"  · Trial {trials:>3}/{len(grid)} | Score: {score:.4f} | Best: {best_score:.4f} (last improved: trial {last_improvement})")
        
        except Exception as e:
            # Skip bad parameter combinations
            continue
    
    fold_elapsed = time.time() - fold_start_time
    
    # Record meta and best
    meta_entry = dict(best_info)
    meta_entry.update({
        "fold": int(fold_idx),
        "train_size": int(len(y_tr)),
        "val_size": int(len(y_va)),
        "train_signature": sig,
    })
    fold_meta.append(meta_entry)
    best_models.append(best_model_fold)
    best_settings.append(best_info.get("params", default_lgb_params()))
    best_thr_list.append(best_info.get("best_thr", 0.5))
    fold_summaries.append({
        "fold": int(fold_idx),
        "train_size": int(len(y_tr)),
        "val_size": int(len(y_va)),
        "best": best_info,
        "trials": trials,
        "improvements": improved_count,
        "elapsed": fold_elapsed,
    })
    
    # Update prior best for next fold
    prior_best_params = best_info.get("params", None)
    
    # Train meta-learner after each fold
    if len(fold_meta) >= 2:
        meta_learner.fit(fold_meta)
        print(f"\n  🧠 Meta-learner updated with {len(fold_meta)} folds")
    
    print(f"\n  📊 Fold {fold_idx} Summary:")
    print(f"     Time: {fold_elapsed:.1f}s | Trials: {trials} | Improvements: {improved_count}")
    print(f"     Best Score: {best_score:.4f} | F1: {best_info.get('best_f1', 0):.4f} | AUC: {best_info.get('best_auc') if best_info.get('best_auc') is not None else float('nan'):.4f}")
    print(f"     Threshold: {best_info.get('best_thr', 0.5):.2f}")

print(f"\n{'='*70}")
print("  PRE-TEST OPTIMIZATION COMPLETE")
print(f"{'='*70}\n")


# %% META-LEARN PARAMS FROM PRIOR FOLDS
print("\n" + "="*70)
print("  META-LEARNING: INFERRING FINAL HYPERPARAMETERS")
print("="*70)

# Use advanced meta-learner if trained, else fallback to weighted average
if meta_learner.fitted and len(fold_meta) >= 2:
    # Get signature for full training set (train + pretest combined for final model)
    X_combined = np.vstack([X_train_s, X_pre_s]) if len(X_pre_s) else X_train_s
    y_combined = np.concatenate([y_train, y_pre]) if len(y_pre) else y_train
    combined_sig = summary_signature(X_combined, y_combined)
    
    inferred_params = meta_learner.predict(combined_sig)
    print(f"✓ Using ML-based meta-learner (trained on {len(fold_meta)} folds)")
    print(f"  Combined train signature: n={combined_sig.get('n')}, pos_rate={combined_sig.get('pos_rate'):.3f}, sep={combined_sig.get('feat_separation'):.4f}")
else:
    inferred_params = meta_infer_hparams_simple(fold_meta)
    print(f"✓ Using weighted-average fallback ({len(fold_meta)} folds)")

print("\n  Inferred Hyperparameters:")
for key in ["learning_rate", "num_leaves", "max_depth", "min_child_samples", "feature_fraction", "bagging_fraction", "lambda_l1", "lambda_l2"]:
    val = inferred_params.get(key)
    print(f"    {key}: {val}")
print("="*70 + "\n")


# %% FINAL TRAIN ON TRAIN + PRE-TEST
X_full = np.vstack([X_train_s, X_pre_s]) if len(X_pre_s) else X_train_s
y_full = np.concatenate([y_train, y_pre]) if len(y_pre) else y_train
final_model = lgb.LGBMClassifier(**inferred_params)
final_model.fit(X_full, y_full)

test_proba = final_model.predict_proba(X_test_s)[:, 1] if len(X_test_s) else np.array([])
test_stats = {
    "thr": None,
    "acc": float("nan"),
    "auc": float("nan"),
    "prec": float("nan"),
    "rec": float("nan"),
    "f1": float("nan"),
    "bacc": float("nan"),
}
if len(test_proba):
    # Use robust threshold: median over pre-test best thresholds
    if len(best_thr_list):
        thr_final = float(np.median(best_thr_list))
    else:
        thr_final = 0.5
    pred = (test_proba >= thr_final).astype(int)
    try:
        auc = roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) == 2 else float("nan")
    except Exception:
        auc = float("nan")
    test_stats = {
        "thr": thr_final,
        "acc": float(accuracy_score(y_test, pred)),
        "auc": float(auc),
        "prec": float(precision_score(y_test, pred, zero_division=0)),
        "rec": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "bacc": float(balanced_accuracy_score(y_test, pred)),
    }


# %% SUMMARIES
def class_dist(y: np.ndarray) -> Dict[str, Any]:
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    rate = pos / (pos + neg) if (pos + neg) else float("nan")
    return {"pos": pos, "neg": neg, "pos_rate": float(rate)}


train_bounds = (pd.to_datetime(train_df["timestamp"]).min() if len(train_df) else None,
                pd.to_datetime(train_df["timestamp"]).max() if len(train_df) else None)
print("\n================ THREE-WINDOW RUN SUMMARY ================")
print("[Data]")
print(f"  Train rows: {len(train_df):,}  |  Pre-test rows: {len(pre_df):,}  |  Test rows: {len(test_df):,}")
if train_bounds[0] is not None and train_bounds[1] is not None:
    print(f"  Train span: {train_bounds[0]}  →  {train_bounds[1]}")
print(f"  Pre-test span: {pretest_bounds[0]}  →  {pretest_bounds[1]}")
print(f"  Test span:     {test_bounds[0]}  →  {test_bounds[1]}")
print(f"  Class dist — Train: {class_dist(y_train)}")
print(f"  Class dist — Pre-test: {class_dist(y_pre)}")
print(f"  Class dist — Test: {class_dist(y_test)}")

print("\n[Baseline on Train]")
try:
    # Evaluate baseline on pre-test as sanity
    base_proba = baseline_model.predict_proba(X_pre_s)[:, 1] if len(X_pre_s) else np.array([])
    if len(base_proba):
        base_stats = evaluate_thresholds(y_pre, base_proba, thr_grid)
        print(f"  Pre-test baseline: AUC={base_stats['auc']:.4f}, F1={base_stats['f1']:.4f}, θ={base_stats['thr']:.2f}")
except Exception:
    pass

print("\n[Pre-test Overfit by Folds]")
for fs in fold_summaries:
    b = fs["best"]
    print(
        "  Fold {fold}: train={tr:,}, val={va:,}, trials={trials}, time={time:.1f}s | "
        "AUC={auc}, F1={f1}, Acc={acc}, θ={thr} | "
        "leaves={leaves}, depth={depth}, lr={lr:.4f}".format(
            fold=fs.get("fold"), tr=fs.get("train_size", 0), va=fs.get("val_size", 0),
            trials=fs.get("trials", 0), time=fs.get("elapsed", 0.0),
            auc=f"{b.get('best_auc', float('nan')):.4f}" if b.get('best_auc') is not None else "nan",
            f1=f"{b.get('best_f1', float('nan')):.4f}",
            acc=f"{b.get('best_acc', float('nan')):.4f}",
            thr=f"{b.get('best_thr', float('nan')):.2f}",
            leaves=b.get('params', {}).get('num_leaves', None),
            depth=b.get('params', {}).get('max_depth', None),
            lr=b.get('params', {}).get('learning_rate', 0.05),
        )
    )

print("\n[Meta-Inferred Params]")
print(json.dumps({k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (float, np.floating)) else v)
                 for k, v in inferred_params.items()}, indent=2))

print("\n[Test Window Metrics]")
print(json.dumps(test_stats, indent=2))
print("==========================================================\n")


# %% TEST-WINDOW THRESHOLD SWEEP (Accuracy 0.30..0.80) + TOP-5
try:
    print("\n[Test Window Threshold Sweep — Accuracy 0.30..0.80]")
    if len(test_proba):
        thr_sweep = np.round(np.arange(0.30, 0.80 + 1e-9, 0.01), 2)
        sweep_results = []
        for thr in thr_sweep:
            pred_thr = (test_proba >= thr).astype(int)
            acc_thr = accuracy_score(y_test, pred_thr)
            sweep_results.append({"thr": float(thr), "acc": float(round(acc_thr, 6))})

        # Print all accuracies
        for r in sweep_results:
            print(f"  θ={r['thr']:.2f} -> acc={r['acc']:.4f}")

        # Compute Top-5 by accuracy including ties
        unique_accs = sorted({r["acc"] for r in sweep_results}, reverse=True)
        top_accs = unique_accs[:5]

        print("\n[Top thresholds by accuracy (Top 5 with ties)]")
        for rank, acc_val in enumerate(top_accs, start=1):
            thrs = [r["thr"] for r in sweep_results if r["acc"] == acc_val]
            thrs_sorted = sorted(thrs)
            if len(thrs_sorted) == 1:
                print(f"  Rank {rank}: acc={acc_val:.4f} at θ={thrs_sorted[0]:.2f}")
            else:
                thrs_str = ", ".join([f"{t:.2f}" for t in thrs_sorted])
                print(f"  Rank {rank}: acc={acc_val:.4f} at θ=[{thrs_str}] (ties: {len(thrs_sorted)})")
    else:
        print("  No test probabilities available; skipping threshold sweep.")
except Exception:
    pass


# %% SAVE ARTIFACTS
model_path = os.path.join(RUN_DIR, 'lightgbm_three_window_final.joblib')
scaler_path = os.path.join(RUN_DIR, 'scaler_three_window.joblib')
joblib.dump(final_model, model_path)
joblib.dump(scaler, scaler_path)

results = {
    "windows": {
        "train_rows": int(len(train_df)),
        "pretest_rows": int(len(pre_df)),
        "test_rows": int(len(test_df)),
        "train_span": [str(train_bounds[0]) if train_bounds[0] is not None else None, str(train_bounds[1]) if train_bounds[1] is not None else None],
        "pretest_span": [str(pretest_bounds[0]), str(pretest_bounds[1])],
        "test_span": [str(test_bounds[0]), str(test_bounds[1])],
    },
    "class_dist": {
        "train": class_dist(y_train),
        "pretest": class_dist(y_pre),
        "test": class_dist(y_test),
    },
    "baseline_pretest": base_stats if 'base_stats' in locals() else None,
    "pretest_folds": fold_summaries,
    "inferred_params": inferred_params,
    "test_metrics": test_stats,
    "feature_names": FEATURE_NAMES,
}

results_path = os.path.join(RUN_DIR, 'results_three_window.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

run_meta = {
    "script_name": os.path.basename(__file__),
    "run_timestamp_utc": RUN_TS,
    "artifact_dir": RUN_DIR,
    "artifacts": {
        "model": model_path,
        "scaler": scaler_path,
        "results": results_path,
    },
}
with open(os.path.join(RUN_DIR, 'run_meta.json'), 'w') as f:
    json.dump(run_meta, f, indent=2)

try:
    code_txt = Path(__file__).read_text(encoding='utf-8')
    Path(RUN_DIR, f"code_{os.path.basename(__file__)}.txt").write_text(code_txt, encoding='utf-8')
except Exception:
    pass

print("Artifacts saved to:", RUN_DIR)



# %% SECOND RUN: TEST_ROWS=25, PRETEST_FOLD_SIZE=25 (same reporting and artifacts)
try:
    print("\n\n================ SECOND RUN (test=25, folds~25) ================")
    TEST_ROWS_2 = 25
    PRETEST_FOLD_SIZE_2 = 25
    RUN_DIR_25 = os.path.join(RUN_DIR, 'run_25')
    os.makedirs(RUN_DIR_25, exist_ok=True)

    def split_three_windows_rows(df_in: pd.DataFrame, test_rows: int) -> Dict[str, pd.DataFrame]:
        if len(df_in) == 0:
            raise RuntimeError("No data available")
        latest_ts2 = pd.to_datetime(df_in["timestamp"]).max()
        cutoff_ts2 = latest_ts2 - pd.DateOffset(years=6)
        df6_2 = df_in[df_in["timestamp"] >= cutoff_ts2].copy().sort_values("timestamp").reset_index(drop=True)
        if len(df6_2) < test_rows:
            raise RuntimeError(f"Not enough bars in last 6 years to form a {test_rows}-row test window")
        test_df2 = df6_2.tail(test_rows).copy().reset_index(drop=True)
        test_start2 = pd.to_datetime(test_df2["timestamp"]).iloc[0]
        test_end2 = pd.to_datetime(test_df2["timestamp"]).iloc[-1]
        pretest_start2 = test_start2 - pd.DateOffset(years=1)
        pre_df2 = df6_2[(df6_2["timestamp"] >= pretest_start2) & (df6_2["timestamp"] < test_start2)].copy().reset_index(drop=True)
        pretest_end2 = pd.to_datetime(pre_df2["timestamp"]).iloc[-1] if len(pre_df2) else (test_start2 - pd.Timedelta(microseconds=1))
        train_df2 = df6_2[df6_2["timestamp"] < pretest_start2].copy().reset_index(drop=True)
        return {
            "train": train_df2,
            "pretest": pre_df2,
            "test": test_df2,
            "pretest_bounds": (pretest_start2, pretest_end2),
            "test_bounds": (test_start2, test_end2),
        }

    splits2 = split_three_windows_rows(df, TEST_ROWS_2)
    train_df2 = splits2["train"]
    pre_df2 = splits2["pretest"]
    test_df2 = splits2["test"]
    pretest_bounds2 = splits2["pretest_bounds"]
    test_bounds2 = splits2["test_bounds"]

    print(f"Train rows: {len(train_df2):,}, Pre-test rows: {len(pre_df2):,}, Test rows: {len(test_df2):,}")

    X_train2, y_train2 = extract_Xy(train_df2)
    X_pre2, y_pre2 = extract_Xy(pre_df2)
    X_test2, y_test2 = extract_Xy(test_df2)
    print(f"Shapes — Train: {X_train2.shape}, Pre-test: {X_pre2.shape}, Test: {X_test2.shape}")

    scaler2 = StandardScaler()
    scaler2.fit(X_train2)
    X_train2_s = scaler2.transform(X_train2)
    X_pre2_s = scaler2.transform(X_pre2)
    X_test2_s = scaler2.transform(X_test2) if len(X_test2) else np.empty((0, X_train2_s.shape[1]))

    baseline_model2 = lgb.LGBMClassifier(**default_lgb_params())
    baseline_model2.fit(X_train2_s, y_train2)

    print("\n" + "="*70)
    print("  PRE-TEST HYPERPARAMETER OPTIMIZATION (TIME-LIMITED PER FOLD) — RUN2")
    print("="*70)

    thr_grid2 = np.round(np.arange(0.30, 0.80 + 1e-9, 0.01), 2)
    fold_meta2: List[Dict[str, Any]] = []
    pre_n2 = len(y_pre2)
    best_models2: List[lgb.LGBMClassifier] = []
    best_settings2: List[Dict[str, Any]] = []
    best_thr_list2: List[float] = []
    fold_summaries2: List[Dict[str, Any]] = []

    meta_learner2 = MetaLearner()
    prior_best_params2: Optional[Dict[str, Any]] = None

    for fold_idx2, (tr_idx2, va_idx2) in enumerate(make_folds_with_purge(pre_n2, PRETEST_FOLD_SIZE_2, GAP_BARS), start=1):
        fold_start_time2 = time.time()
        X_tr2, y_tr2 = X_pre2_s[tr_idx2], y_pre2[tr_idx2]
        X_va2, y_va2 = X_pre2_s[va_idx2], y_pre2[va_idx2]
        sig2 = summary_signature(X_tr2, y_tr2)

        print(f"\n{'─'*70}")
        print(f"✓ RUN2 FOLD {fold_idx2} | Train: {len(y_tr2)} bars | Val: {len(y_va2)} bars")
        print(f"  Pos rate: {sig2.get('pos_rate', 0.0):.3f} | Separation: {sig2.get('feat_separation', 0.0):.4f}")
        print(f"  Time limit: {FOLD_TIME_LIMIT}s | Strategy: {'Smart Grid (prior-guided)' if prior_best_params2 else 'Random Search'}")
        print(f"{'─'*70}")

        n_samples2 = 150 if prior_best_params2 else 100
        grid2 = generate_smart_grid(prior_best_params2, n_samples=n_samples2)

        best_score2 = -1.0
        best_info2: Dict[str, Any] = {}
        best_model_fold2: Optional[lgb.LGBMClassifier] = None
        trials2 = 0
        improved_count2 = 0
        last_improvement2 = 0

        for _, params2 in enumerate(grid2):
            elapsed2 = time.time() - fold_start_time2
            if elapsed2 > FOLD_TIME_LIMIT:
                print(f"  ⏱ Time limit reached ({elapsed2:.1f}s). Stopping search.")
                break
            try:
                clf2 = lgb.LGBMClassifier(**params2)
                clf2.fit(X_tr2, y_tr2)
                proba2 = clf2.predict_proba(X_va2)[:, 1]
                stats2 = evaluate_thresholds(y_va2, proba2, thr_grid2)
                score2 = 0.7 * stats2["f1"] + 0.3 * (stats2["auc"] if not np.isnan(stats2["auc"]) else 0.5)
                trials2 += 1
                if score2 > best_score2:
                    best_score2 = score2
                    best_info2 = {
                        "params": params2,
                        "best_thr": float(stats2["thr"]),
                        "best_auc": float(stats2["auc"]) if not np.isnan(stats2["auc"]) else None,
                        "best_acc": float(stats2["acc"]),
                        "best_prec": float(stats2["prec"]),
                        "best_rec": float(stats2["rec"]),
                        "best_f1": float(stats2["f1"]),
                        "best_bacc": float(stats2["bacc"]),
                        "score": float(score2),
                    }
                    best_model_fold2 = clf2
                    improved_count2 += 1
                    last_improvement2 = trials2
                    print(f"  ✓ Trial {trials2:>3}/{len(grid2)} | Score: {score2:.4f} → NEW BEST | "
                          f"F1={stats2['f1']:.4f}, AUC={stats2['auc'] if not np.isnan(stats2['auc']) else float('nan'):.4f}, θ={stats2['thr']:.2f} | "
                          f"lr={params2['learning_rate']:.4f}, leaves={params2['num_leaves']}, depth={params2['max_depth']}")
                else:
                    if trials2 % 10 == 0:
                        print(f"  · Trial {trials2:>3}/{len(grid2)} | Score: {score2:.4f} | Best: {best_score2:.4f} (last improved: trial {last_improvement2})")
            except Exception:
                continue

        fold_elapsed2 = time.time() - fold_start_time2
        meta_entry2 = dict(best_info2)
        meta_entry2.update({
            "fold": int(fold_idx2),
            "train_size": int(len(y_tr2)),
            "val_size": int(len(y_va2)),
            "train_signature": sig2,
        })
        fold_meta2.append(meta_entry2)
        best_models2.append(best_model_fold2)
        best_settings2.append(best_info2.get("params", default_lgb_params()))
        best_thr_list2.append(best_info2.get("best_thr", 0.5))
        fold_summaries2.append({
            "fold": int(fold_idx2),
            "train_size": int(len(y_tr2)),
            "val_size": int(len(y_va2)),
            "best": best_info2,
            "trials": trials2,
            "improvements": improved_count2,
            "elapsed": fold_elapsed2,
        })

        prior_best_params2 = best_info2.get("params", None)
        if len(fold_meta2) >= 2:
            meta_learner2.fit(fold_meta2)
            print(f"\n  🧠 Meta-learner updated with {len(fold_meta2)} folds (RUN2)")

        print(f"\n  📊 RUN2 Fold {fold_idx2} Summary:")
        print(f"     Time: {fold_elapsed2:.1f}s | Trials: {trials2} | Improvements: {improved_count2}")
        print(f"     Best Score: {best_score2:.4f} | F1: {best_info2.get('best_f1', 0):.4f} | AUC: {best_info2.get('best_auc') if best_info2.get('best_auc') is not None else float('nan'):.4f}")
        print(f"     Threshold: {best_info2.get('best_thr', 0.5):.2f}")

    print(f"\n{'='*70}")
    print("  PRE-TEST OPTIMIZATION COMPLETE — RUN2")
    print(f"{'='*70}\n")

    print("\n" + "="*70)
    print("  META-LEARNING: INFERRING FINAL HYPERPARAMETERS — RUN2")
    print("="*70)
    if meta_learner2.fitted and len(fold_meta2) >= 2:
        X_combined2 = np.vstack([X_train2_s, X_pre2_s]) if len(X_pre2_s) else X_train2_s
        y_combined2 = np.concatenate([y_train2, y_pre2]) if len(y_pre2) else y_train2
        combined_sig2 = summary_signature(X_combined2, y_combined2)
        inferred_params2 = meta_learner2.predict(combined_sig2)
        print(f"✓ Using ML-based meta-learner (trained on {len(fold_meta2)} folds)")
        print(f"  Combined train signature: n={combined_sig2.get('n')}, pos_rate={combined_sig2.get('pos_rate'):.3f}, sep={combined_sig2.get('feat_separation'):.4f}")
    else:
        inferred_params2 = meta_infer_hparams_simple(fold_meta2)
        print(f"✓ Using weighted-average fallback ({len(fold_meta2)} folds)")

    print("\n  Inferred Hyperparameters (RUN2):")
    for key in ["learning_rate", "num_leaves", "max_depth", "min_child_samples", "feature_fraction", "bagging_fraction", "lambda_l1", "lambda_l2"]:
        val2 = inferred_params2.get(key)
        print(f"    {key}: {val2}")
    print("="*70 + "\n")

    X_full2 = np.vstack([X_train2_s, X_pre2_s]) if len(X_pre2_s) else X_train2_s
    y_full2 = np.concatenate([y_train2, y_pre2]) if len(y_pre2) else y_train2
    final_model2 = lgb.LGBMClassifier(**inferred_params2)
    final_model2.fit(X_full2, y_full2)

    test_proba2 = final_model2.predict_proba(X_test2_s)[:, 1] if len(X_test2_s) else np.array([])
    test_stats2 = {"thr": None, "acc": float("nan"), "auc": float("nan"), "prec": float("nan"), "rec": float("nan"), "f1": float("nan"), "bacc": float("nan")}
    if len(test_proba2):
        thr_final2 = float(np.median(best_thr_list2)) if len(best_thr_list2) else 0.5
        pred2 = (test_proba2 >= thr_final2).astype(int)
        try:
            auc2 = roc_auc_score(y_test2, test_proba2) if len(np.unique(y_test2)) == 2 else float("nan")
        except Exception:
            auc2 = float("nan")
        test_stats2 = {
            "thr": thr_final2,
            "acc": float(accuracy_score(y_test2, pred2)),
            "auc": float(auc2),
            "prec": float(precision_score(y_test2, pred2, zero_division=0)),
            "rec": float(recall_score(y_test2, pred2, zero_division=0)),
            "f1": float(f1_score(y_test2, pred2, zero_division=0)),
            "bacc": float(balanced_accuracy_score(y_test2, pred2)),
        }

    def class_dist2(yarr: np.ndarray) -> Dict[str, Any]:
        p = int(np.sum(yarr == 1)); n = int(np.sum(yarr == 0)); rate = p/(p+n) if (p+n) else float('nan')
        return {"pos": p, "neg": n, "pos_rate": float(rate)}

    train_bounds2 = (pd.to_datetime(train_df2["timestamp"]).min() if len(train_df2) else None,
                     pd.to_datetime(train_df2["timestamp"]).max() if len(train_df2) else None)
    print("\n================ THREE-WINDOW RUN SUMMARY (RUN2) ================")
    print("[Data]")
    print(f"  Train rows: {len(train_df2):,}  |  Pre-test rows: {len(pre_df2):,}  |  Test rows: {len(test_df2):,}")
    if train_bounds2[0] is not None and train_bounds2[1] is not None:
        print(f"  Train span: {train_bounds2[0]}  →  {train_bounds2[1]}")
    print(f"  Pre-test span: {pretest_bounds2[0]}  →  {pretest_bounds2[1]}")
    print(f"  Test span:     {test_bounds2[0]}  →  {test_bounds2[1]}")
    print(f"  Class dist — Train: {class_dist2(y_train2)}")
    print(f"  Class dist — Pre-test: {class_dist2(y_pre2)}")
    print(f"  Class dist — Test: {class_dist2(y_test2)}")

    print("\n[Baseline on Train]")
    try:
        base_proba2 = baseline_model2.predict_proba(X_pre2_s)[:, 1] if len(X_pre2_s) else np.array([])
        if len(base_proba2):
            base_stats2 = evaluate_thresholds(y_pre2, base_proba2, thr_grid2)
            print(f"  Pre-test baseline: AUC={base_stats2['auc']:.4f}, F1={base_stats2['f1']:.4f}, θ={base_stats2['thr']:.2f}")
    except Exception:
        pass

    print("\n[Pre-test Overfit by Folds]")
    for fs2 in fold_summaries2:
        b2 = fs2["best"]
        print(
            "  Fold {fold}: train={tr:,}, val={va:,}, trials={trials}, time={time:.1f}s | "
            "AUC={auc}, F1={f1}, Acc={acc}, θ={thr} | "
            "leaves={leaves}, depth={depth}, lr={lr:.4f}".format(
                fold=fs2.get("fold"), tr=fs2.get("train_size", 0), va=fs2.get("val_size", 0),
                trials=fs2.get("trials", 0), time=fs2.get("elapsed", 0.0),
                auc=f"{b2.get('best_auc', float('nan')):.4f}" if b2.get('best_auc') is not None else "nan",
                f1=f"{b2.get('best_f1', float('nan')):.4f}",
                acc=f"{b2.get('best_acc', float('nan')):.4f}",
                thr=f"{b2.get('best_thr', float('nan')):.2f}",
                leaves=b2.get('params', {}).get('num_leaves', None),
                depth=b2.get('params', {}).get('max_depth', None),
                lr=b2.get('params', {}).get('learning_rate', 0.05),
            )
        )

    print("\n[Meta-Inferred Params]")
    print(json.dumps({k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (float, np.floating)) else v)
                     for k, v in inferred_params2.items()}, indent=2))

    print("\n[Test Window Metrics]")
    print(json.dumps(test_stats2, indent=2))
    print("==========================================================\n")

    print("\n[Test Window Threshold Sweep — Accuracy 0.30..0.80] (RUN2)")
    if len(test_proba2):
        thr_sweep2 = np.round(np.arange(0.30, 0.80 + 1e-9, 0.01), 2)
        sweep_results2 = []
        for thr in thr_sweep2:
            pred_thr2 = (test_proba2 >= thr).astype(int)
            acc_thr2 = accuracy_score(y_test2, pred_thr2)
            sweep_results2.append({"thr": float(thr), "acc": float(round(acc_thr2, 6))})
        for r in sweep_results2:
            print(f"  θ={r['thr']:.2f} -> acc={r['acc']:.4f}")
        unique_accs2 = sorted({r["acc"] for r in sweep_results2}, reverse=True)
        top_accs2 = unique_accs2[:5]
        print("\n[Top thresholds by accuracy (Top 5 with ties)] (RUN2)")
        for rank, acc_val2 in enumerate(top_accs2, start=1):
            thrs2 = [r["thr"] for r in sweep_results2 if r["acc"] == acc_val2]
            thrs2_sorted = sorted(thrs2)
            if len(thrs2_sorted) == 1:
                print(f"  Rank {rank}: acc={acc_val2:.4f} at θ={thrs2_sorted[0]:.2f}")
            else:
                thrs2_str = ", ".join([f"{t:.2f}" for t in thrs2_sorted])
                print(f"  Rank {rank}: acc={acc_val2:.4f} at θ=[{thrs2_str}] (ties: {len(thrs2_sorted)})")
    else:
        print("  No test probabilities available; skipping threshold sweep.")

    # Save RUN2 artifacts
    model_path2 = os.path.join(RUN_DIR_25, 'lightgbm_three_window_final_run2.joblib')
    scaler_path2 = os.path.join(RUN_DIR_25, 'scaler_three_window_run2.joblib')
    joblib.dump(final_model2, model_path2)
    joblib.dump(scaler2, scaler_path2)
    results2 = {
        "windows": {
            "train_rows": int(len(train_df2)),
            "pretest_rows": int(len(pre_df2)),
            "test_rows": int(len(test_df2)),
            "train_span": [str(train_bounds2[0]) if train_bounds2[0] is not None else None, str(train_bounds2[1]) if train_bounds2[1] is not None else None],
            "pretest_span": [str(pretest_bounds2[0]), str(pretest_bounds2[1])],
            "test_span": [str(test_bounds2[0]), str(test_bounds2[1])],
        },
        "class_dist": {"train": class_dist2(y_train2), "pretest": class_dist2(y_pre2), "test": class_dist2(y_test2)},
        "baseline_pretest": base_stats2 if 'base_stats2' in locals() else None,
        "pretest_folds": fold_summaries2,
        "inferred_params": inferred_params2,
        "test_metrics": test_stats2,
        "feature_names": FEATURE_NAMES,
    }
    results_path2 = os.path.join(RUN_DIR_25, 'results_three_window_run2.json')
    with open(results_path2, 'w') as f:
        json.dump(results2, f, indent=2)
    run_meta2 = {
        "script_name": os.path.basename(__file__),
        "run_timestamp_utc": RUN_TS,
        "artifact_dir": RUN_DIR_25,
        "artifacts": {"model": model_path2, "scaler": scaler_path2, "results": results_path2},
    }
    with open(os.path.join(RUN_DIR_25, 'run_meta.json'), 'w') as f:
        json.dump(run_meta2, f, indent=2)
    try:
        code_txt2 = Path(__file__).read_text(encoding='utf-8')
        Path(RUN_DIR_25, f"code_{os.path.basename(__file__)}.txt").write_text(code_txt2, encoding='utf-8')
    except Exception:
        pass
    print("Artifacts saved to:", RUN_DIR_25)

    # Short comparison
    try:
        print("\n============= Comparison (RUN1 vs RUN2) =============")
        print(f"Test rows: 35 vs 25")
        print(f"AUC: {test_stats.get('auc'):.4f} vs {test_stats2.get('auc'):.4f} | Acc: {test_stats.get('acc'):.4f} vs {test_stats2.get('acc'):.4f}")
    except Exception:
        pass
except Exception as e:
    print("Second run failed:", str(e))
