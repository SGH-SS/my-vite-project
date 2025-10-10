#!/usr/bin/env python3
"""
LightGBM 1D Training (ISO-style features) — Local DB Version

Loads SPY 1D data directly from Postgres (backtest + fronttest continuation),
builds 16-feature vectors (raw OHLCV + ISO OHLC + one-hot TF + engineered),
trains LightGBM, and saves artifacts locally next to the script.

Conda env: daygent-train
"""

from __future__ import annotations

import os
import json
import warnings
from typing import List, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
import lightgbm as lgb


# %% CELL 1: CONFIG
DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5433/trading_data"
BACKTEST_TABLE = 'backtest.spy_1d'
FRONTTEST_TABLE = 'fronttest.spy_1d'

# Base artifacts directory (script-local)
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'artifacts_lgbm_1d_iso')
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Timestamped run directory
RUNS_DIR = os.path.join(ARTIFACT_DIR, 'runs')
os.makedirs(RUNS_DIR, exist_ok=True)
RUN_TS = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
RUN_DIR = os.path.join(RUNS_DIR, RUN_TS)
os.makedirs(RUN_DIR, exist_ok=True)

# No longer using public/training folder - UI reads directly from artifacts


# %% CELL 2: HELPERS
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


# CV / weighting utilities
TAU_BARS = 180
N_SPLITS = 5
GAP_DAYS = 5
THR_LOW = 0.30
THR_HIGH = 0.80
THR_STEP = 0.01

def compute_recency_weights_by_bars(n_bars: int, tau_bars: int = TAU_BARS) -> np.ndarray:
    if n_bars <= 0:
        return np.array([])
    idx = np.arange(n_bars)
    delta = idx - idx.max()
    return np.exp(delta / float(tau_bars))

def purged_splits(n_samples: int, n_splits: int = N_SPLITS, gap_days: int = GAP_DAYS):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr_idx, va_idx in tscv.split(np.arange(n_samples)):
        if len(va_idx) == 0:
            continue
        va_start = va_idx[0] + gap_days
        if va_start > va_idx[-1]:
            continue
        va_idx_adj = np.arange(va_start, va_idx[-1] + 1)
        tr_idx_adj = tr_idx[tr_idx <= (va_idx_adj[0] - gap_days - 1)]
        if len(tr_idx_adj) == 0 or len(va_idx_adj) == 0:
            continue
        yield tr_idx_adj, va_idx_adj


# %% CELL 3: LOAD DATA (BACKTEST + FRONTTEST)
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


# %% CELL 4: SLICE LAST 5 YEARS + DEFINE TEST WINDOW (35D)
if len(df) == 0:
    raise RuntimeError("No data found for spy_1d in backtest/fronttest.")

latest_ts = df["timestamp"].max()
cutoff_ts = latest_ts - pd.DateOffset(years=5)
df = df[df["timestamp"] >= cutoff_ts].copy()
df = df.sort_values("timestamp").reset_index(drop=True)

all_days = list(pd.to_datetime(df["timestamp"]).dt.date.unique())
if len(all_days) < 35:
    raise RuntimeError("Not enough candles for a 65-day evaluation window after 6y filter")

selected_days = all_days[-35:]
test_start = pd.Timestamp.combine(selected_days[0], pd.Timestamp.min.time()).tz_localize("UTC")
test_end = pd.Timestamp.combine(selected_days[-1], pd.Timestamp.max.time()).tz_localize("UTC")


# %% CELL 5: FEATURE EXTRACTION
train_df = df[df["timestamp"] < test_start].copy()
test_df = df[(df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)].copy()

X_train: List[np.ndarray] = []
y_train: List[int] = []
for _, row in train_df.iterrows():
    fv, lbl = extract_features_row(row)
    if fv is not None:
        X_train.append(fv)
        y_train.append(lbl)

X_test: List[np.ndarray] = []
y_test: List[int] = []
for _, row in test_df.iterrows():
    fv, lbl = extract_features_row(row)
    if fv is not None:
        X_test.append(fv)
        y_test.append(lbl)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# %% CELL 6: SCALING
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)


# %% CELL 7: TRAIN LGBM + CALIBRATE THRESHOLD via Time-Series CV
lgb_params = {
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

def cv_calibrate_threshold(X_s: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, int, List[dict]]:
    thr_grid = np.round(np.arange(THR_LOW, THR_HIGH + 1e-9, THR_STEP), 2)
    f1_by_fold: List[float] = []
    auc_by_fold: List[float] = []
    best_thr_by_fold: List[float] = []
    best_iters: List[int] = []
    fold_stats: List[dict] = []
    n = len(y)
    if n == 0:
        return 0.5, float("nan"), float("nan"), 500, []
    for fold_idx, (tr_idx, va_idx) in enumerate(purged_splits(n), start=1):
        X_tr, y_tr = X_s[tr_idx], y[tr_idx]
        X_va, y_va = X_s[va_idx], y[va_idx]
        w_tr = compute_recency_weights_by_bars(len(y_tr), TAU_BARS)
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(
            X_tr, y_tr,
            sample_weight=w_tr if len(w_tr) == len(y_tr) else None,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
        )
        best_iter = int(getattr(clf, "best_iteration_", lgb_params.get("n_estimators", 1000)))
        best_iters.append(max(50, best_iter))
        proba = clf.predict_proba(X_va)[:, 1]
        try:
            auc = roc_auc_score(y_va, proba) if len(np.unique(y_va)) == 2 else float("nan")
        except Exception:
            auc = float("nan")
        auc_by_fold.append(float(auc))
        # Threshold sweep by F1
        f1s = []
        for thr in thr_grid:
            pred = (proba >= thr).astype(int)
            f1s.append((thr, f1_score(y_va, pred)))
        best_thr, best_f1 = max(f1s, key=lambda t: (t[1], -abs(t[0]-0.5)))
        best_thr_by_fold.append(float(best_thr))
        f1_by_fold.append(float(best_f1))
        fold_stats.append({
            "fold": int(fold_idx),
            "train_size": int(len(y_tr)),
            "val_size": int(len(y_va)),
            "auc": float(auc),
            "f1": float(best_f1),
            "best_thr": float(best_thr),
            "best_iter": int(best_iter),
        })
    if len(best_thr_by_fold) == 0:
        return 0.5, float("nan"), float("nan"), 500, fold_stats
    thr_med = float(np.median(best_thr_by_fold))
    auc_mean = float(np.nanmean(auc_by_fold))
    f1_mean = float(np.nanmean(f1_by_fold))
    best_iter_median = int(np.median(best_iters)) if len(best_iters) else 500
    return thr_med, auc_mean, f1_mean, best_iter_median, fold_stats

best_thr, cv_auc_mean, cv_f1_mean, best_iter_median, cv_fold_stats = cv_calibrate_threshold(X_train_s, y_train)

model_full = lgb.LGBMClassifier(**{**lgb_params, "n_estimators": max(300, int(best_iter_median * 2))})
w_all = compute_recency_weights_by_bars(len(y_train), TAU_BARS)
model_full.fit(X_train_s, y_train, sample_weight=w_all if len(w_all) == len(y_train) else None)


# %% CELL 8: EVAL TEST
X_test_s = scaler.transform(X_test) if len(X_test) else np.empty((0, X_train_s.shape[1]))
test_proba = model_full.predict_proba(X_test_s)[:, 1] if len(X_test_s) else np.array([])
test_pred = (test_proba >= best_thr).astype(int) if len(test_proba) else np.array([])

if len(test_pred):
    test_acc = accuracy_score(y_test, test_pred)
    try:
        test_auc = roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) == 2 else float("nan")
    except Exception:
        test_auc = float("nan")
    test_prec = precision_score(y_test, test_pred, zero_division=0)
    test_rec = recall_score(y_test, test_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    test_bacc = balanced_accuracy_score(y_test, test_pred)
else:
    test_acc = float("nan")
    test_auc = float("nan")
    test_prec = float("nan")
    test_rec = float("nan")
    test_f1 = float("nan")
    test_bacc = float("nan")

print(f"CV mean AUC: {cv_auc_mean:.4f}, CV mean F1: {cv_f1_mean:.4f}, θ: {best_thr:.2f}")
print(f"Test acc: {test_acc:.4f}, AUC: {test_auc:.4f}, F1: {test_f1:.4f}")


# ---- End-of-run Summary (human-friendly) ----
try:
    # Dataset sizes and class distributions
    n_features = int(X_train.shape[1]) if hasattr(X_train, 'shape') and len(X_train.shape) == 2 else 0
    train_size = int(len(y_train))
    test_size = int(len(y_test))
    train_pos = int(np.sum(y_train == 1)) if train_size else 0
    train_neg = int(np.sum(y_train == 0)) if train_size else 0
    test_pos = int(np.sum(y_test == 1)) if test_size else 0
    test_neg = int(np.sum(y_test == 0)) if test_size else 0
    train_pos_rate = (train_pos / (train_pos + train_neg)) if (train_pos + train_neg) > 0 else float('nan')
    test_pos_rate = (test_pos / (test_pos + test_neg)) if (test_pos + test_neg) > 0 else float('nan')

    # Time windows
    train_start_ts = pd.to_datetime(train_df["timestamp"]).min() if len(train_df) else None
    train_end_ts = pd.to_datetime(train_df["timestamp"]).max() if len(train_df) else None

    final_n_estimators = int(max(300, int(best_iter_median * 2)))

    print("\n================ RUN SUMMARY ================")
    print("[Data]")
    print(f"  Train bars: {train_size:,}  |  Test bars: {test_size:,}  |  Features: {n_features}")
    if train_start_ts is not None and train_end_ts is not None:
        print(f"  Train span: {train_start_ts}  →  {train_end_ts}")
    print(f"  Test span:  {test_start}  →  {test_end}")
    print(f"  Train class dist: pos={train_pos:,}, neg={train_neg:,}, pos_rate={train_pos_rate:.4f}")
    print(f"  Test  class dist: pos={test_pos:,}, neg={test_neg:,}, pos_rate={test_pos_rate:.4f}")

    print("\n[Cross-Validation]")
    print(f"  n_splits={N_SPLITS}, gap_days={GAP_DAYS}, tau_bars={TAU_BARS}")
    print(f"  threshold_grid=[{THR_LOW:.2f}..{THR_HIGH:.2f}] step={THR_STEP:.2f}")
    for fs in cv_fold_stats:
        print(
            "  Fold {fold}: train={tr:,}, val={va:,}, AUC={auc:.4f}, F1={f1:.4f}, "
            "best_thr={thr:.2f}, best_iter={bi}".format(
                fold=fs.get("fold"), tr=fs.get("train_size", 0), va=fs.get("val_size", 0),
                auc=fs.get("auc", float('nan')), f1=fs.get("f1", float('nan')),
                thr=fs.get("best_thr", float('nan')), bi=fs.get("best_iter", 0)
            )
        )
    print(f"  CV mean AUC={cv_auc_mean:.4f}, mean F1={cv_f1_mean:.4f}")
    print(f"  Chosen threshold (median across folds) θ={best_thr:.2f}")
    print(f"  Median best_iteration across folds={best_iter_median}")

    print("\n[Final Model]")
    print(f"  n_estimators={final_n_estimators}  |  class_weight={lgb_params.get('class_weight')}")
    print(f"  num_leaves={lgb_params.get('num_leaves')}, max_depth={lgb_params.get('max_depth')}, min_child_samples={lgb_params.get('min_child_samples')}")
    print(f"  feature_fraction={lgb_params.get('feature_fraction')}, bagging_fraction={lgb_params.get('bagging_fraction')}, learning_rate={lgb_params.get('learning_rate')}")

    print("\n[Test Window Metrics]")
    print(f"  Accuracy={test_acc:.4f}, AUC={test_auc:.4f}, Precision={test_prec:.4f}, Recall={test_rec:.4f}, F1={test_f1:.4f}, BalancedAcc={test_bacc:.4f}")
    print(f"  Applied threshold θ={best_thr:.2f}")

    # Top feature importances
    try:
        importances = getattr(model_full, "feature_importances_", None)
        if importances is not None and len(importances) == len(FEATURE_NAMES):
            order = np.argsort(importances)[::-1]
            top_k = min(10, len(order))
            print("\n[Top Feature Importances]")
            for rank in range(top_k):
                idx = int(order[rank])
                print(f"  {rank+1:>2}. {FEATURE_NAMES[idx]}: {int(importances[idx])}")
    except Exception:
        pass

    print("============================================\n")
except Exception:
    pass


# %% CELL 9: SAVE ARTIFACTS (timestamped run folder)
model_path = os.path.join(RUN_DIR, 'lightgbm_financial_1d_iso.joblib')
scaler_path = os.path.join(RUN_DIR, 'scaler_1d_iso.joblib')
joblib.dump(model_full, model_path)
joblib.dump(scaler, scaler_path)

results = {
    "cv_mean_auc": float(cv_auc_mean) if not (isinstance(cv_auc_mean, float) and np.isnan(cv_auc_mean)) else None,
    "cv_mean_f1": float(cv_f1_mean) if not (isinstance(cv_f1_mean, float) and np.isnan(cv_f1_mean)) else None,
    "test_accuracy": float(test_acc),
    "test_auc": float(test_auc) if not (isinstance(test_auc, float) and np.isnan(test_auc)) else None,
    "test_precision": float(test_prec),
    "test_recall": float(test_rec),
    "test_f1": float(test_f1),
    "test_balanced_accuracy": float(test_bacc),
    "chosen_threshold": float(best_thr),
    "cv_best_iter_median": int(best_iter_median),
    "feature_names": FEATURE_NAMES,
    # Additional detailed stats for analysis/visualization
    "cv_folds": cv_fold_stats,
}

# Enrich results with dataset, windows, and configuration details
try:
    n_features = int(X_train.shape[1]) if hasattr(X_train, 'shape') and len(X_train.shape) == 2 else 0
    train_size = int(len(y_train))
    test_size = int(len(y_test))
    train_pos = int(np.sum(y_train == 1)) if train_size else 0
    train_neg = int(np.sum(y_train == 0)) if train_size else 0
    test_pos = int(np.sum(y_test == 1)) if test_size else 0
    test_neg = int(np.sum(y_test == 0)) if test_size else 0
    train_pos_rate = (train_pos / (train_pos + train_neg)) if (train_pos + train_neg) > 0 else None
    test_pos_rate = (test_pos / (test_pos + test_neg)) if (test_pos + test_neg) > 0 else None

    train_start_ts = pd.to_datetime(train_df["timestamp"]).min() if len(train_df) else None
    train_end_ts = pd.to_datetime(train_df["timestamp"]).max() if len(train_df) else None

    results.update({
        "dataset": {
            "train_size": train_size,
            "test_size": test_size,
            "n_features": n_features,
            "train_pos": train_pos,
            "train_neg": train_neg,
            "train_pos_rate": float(train_pos_rate) if train_pos_rate is not None else None,
            "test_pos": test_pos,
            "test_neg": test_neg,
            "test_pos_rate": float(test_pos_rate) if test_pos_rate is not None else None,
        },
        "time_windows": {
            "train_start": str(train_start_ts) if train_start_ts is not None else None,
            "train_end": str(train_end_ts) if train_end_ts is not None else None,
            "test_start": str(test_start),
            "test_end": str(test_end),
        },
        "cv_config": {
            "n_splits": N_SPLITS,
            "gap_days": GAP_DAYS,
            "tau_bars": TAU_BARS,
            "thr_low": THR_LOW,
            "thr_high": THR_HIGH,
            "thr_step": THR_STEP,
        },
        "final_model": {
            "n_estimators": int(max(300, int(best_iter_median * 2))),
            "class_weight": lgb_params.get("class_weight"),
            "num_leaves": lgb_params.get("num_leaves"),
            "max_depth": lgb_params.get("max_depth"),
            "min_child_samples": lgb_params.get("min_child_samples"),
            "feature_fraction": lgb_params.get("feature_fraction"),
            "bagging_fraction": lgb_params.get("bagging_fraction"),
            "learning_rate": lgb_params.get("learning_rate"),
        }
    })
except Exception:
    pass

results_path = os.path.join(RUN_DIR, 'results_1d_iso.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

# Save run metadata and code snapshot
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

# Save feature importances
try:
    importances = getattr(model_full, "feature_importances_", None)
    if importances is not None and len(importances) == len(FEATURE_NAMES):
        imp = {FEATURE_NAMES[i]: float(importances[i]) for i in range(len(FEATURE_NAMES))}
        Path(RUN_DIR, 'feature_importances.json').write_text(json.dumps(imp, indent=2), encoding='utf-8')
except Exception:
    pass

try:
    code_txt = Path(__file__).read_text(encoding='utf-8')
    Path(RUN_DIR, f"code_{os.path.basename(__file__)}.txt").write_text(code_txt, encoding='utf-8')
except Exception:
    pass

print("Artifacts saved to:", RUN_DIR)


