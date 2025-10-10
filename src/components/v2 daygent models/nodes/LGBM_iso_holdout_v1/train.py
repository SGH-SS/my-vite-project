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
from sklearn.metrics import accuracy_score, roc_auc_score
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


# %% CELL 6: SCALING + SPLIT
scaler = StandardScaler()
split_idx = int(len(X_train) * 0.8)
scaler.fit(X_train[:split_idx] if split_idx > 0 else X_train)

X_train_s = scaler.transform(X_train)
X_tr, X_va = X_train_s[:split_idx], X_train_s[split_idx:]
y_tr, y_va = y_train[:split_idx], y_train[split_idx:]


# %% CELL 7: TRAIN LGBM + CALIBRATE THRESHOLD
lgb_params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": ["auc", "binary_logloss"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 6,
    "min_child_samples": 40,
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
}

model = lgb.LGBMClassifier(**lgb_params)
if len(X_tr) and len(y_tr):
    model.fit(X_tr, y_tr)
else:
    model.fit(X_train_s, y_train)

if len(X_va):
    val_proba = model.predict_proba(X_va)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    val_acc = accuracy_score(y_va, val_pred)
    val_auc = roc_auc_score(y_va, val_proba)
else:
    val_proba = np.array([])
    val_acc = float("nan")
    val_auc = float("nan")

best_thr = 0.5
best_val_acc = val_acc
if len(val_proba):
    for thr in np.round(np.arange(0.30, 0.801, 0.01), 2):
        preds_thr = (val_proba >= thr).astype(int)
        acc_thr = accuracy_score(y_va, preds_thr)
        if acc_thr > best_val_acc:
            best_val_acc = acc_thr
            best_thr = float(thr)

model_full = lgb.LGBMClassifier(**lgb_params)
model_full.fit(X_train_s, y_train)


# %% CELL 8: EVAL TEST
X_test_s = scaler.transform(X_test) if len(X_test) else np.empty((0, X_train_s.shape[1]))
test_proba = model_full.predict_proba(X_test_s)[:, 1] if len(X_test_s) else np.array([])
test_pred = (test_proba >= best_thr).astype(int) if len(test_proba) else np.array([])

if len(test_pred):
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) == 2 else float("nan")
else:
    test_acc = float("nan")
    test_auc = float("nan")

nb_low = max(0.30, round(best_thr - 0.10, 2))
nb_high = min(0.80, round(best_thr + 0.10, 2))

if len(val_proba):
    thr_window = np.round(np.arange(nb_low, nb_high + 0.001, 0.01), 2)
    val_acc_window = []
    for thr in thr_window:
        preds_thr = (val_proba >= thr).astype(int)
        acc_thr = accuracy_score(y_va, preds_thr)
        val_acc_window.append({"thr": float(thr), "acc": float(acc_thr)})
else:
    val_acc_window = []

if len(test_proba):
    thr_window = np.round(np.arange(nb_low, nb_high + 0.001, 0.01), 2)
    test_acc_window = []
    for thr in thr_window:
        preds_thr = (test_proba >= thr).astype(int)
        acc_thr = accuracy_score(y_test, preds_thr)
        test_acc_window.append({"thr": float(thr), "acc": float(acc_thr)})
else:
    test_acc_window = []

print(f"Chosen threshold: {best_thr:.2f}")
print(f"Validation acc: {best_val_acc:.4f}, AUC: {val_auc:.4f}")
print(f"Test acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
if len(val_acc_window):
    curve_str = " ".join([f"{pt['thr']:.2f}:{pt['acc']:.4f}" for pt in val_acc_window])
    print(f"Val accuracy window [{nb_low:.2f},{nb_high:.2f}] (thr:acc):", curve_str)
if len(test_acc_window):
    curve_str = " ".join([f"{pt['thr']:.2f}:{pt['acc']:.4f}" for pt in test_acc_window])
    print(f"Test accuracy window [{nb_low:.2f},{nb_high:.2f}] (thr:acc):", curve_str)


# %% CELL 9: SAVE ARTIFACTS (timestamped run folder)
model_path = os.path.join(RUN_DIR, 'lightgbm_financial_1d_iso.joblib')
scaler_path = os.path.join(RUN_DIR, 'scaler_1d_iso.joblib')
joblib.dump(model_full, model_path)
joblib.dump(scaler, scaler_path)

results = {
    "validation_accuracy": float(best_val_acc),
    "validation_auc": float(val_auc) if not (isinstance(val_auc, float) and np.isnan(val_auc)) else None,
    "test_accuracy": float(test_acc),
    "test_auc": float(test_auc) if not (isinstance(test_auc, float) and np.isnan(test_auc)) else None,
    "chosen_threshold": float(best_thr),
    "feature_names": FEATURE_NAMES,
}

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

try:
    code_txt = Path(__file__).read_text(encoding='utf-8')
    Path(RUN_DIR, f"code_{os.path.basename(__file__)}.txt").write_text(code_txt, encoding='utf-8')
except Exception:
    pass

print("Artifacts saved to:", RUN_DIR)


