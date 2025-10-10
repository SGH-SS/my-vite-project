#!/usr/bin/env python3
"""
LightGBM 1D Probability Calibration (ISO-style features) — Isotonic vs Platt

Loads SPY 1D data directly from Postgres (backtest + fronttest continuation),
builds 16-feature vectors (raw OHLCV + ISO OHLC + one-hot TF + engineered),
trains LightGBM on train split, calibrates probabilities on the validation split
using both Isotonic and Platt (sigmoid), evaluates on validation and test, and
saves artifacts.

Conda env: daygent-train
"""

from __future__ import annotations

import os
import json
import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    log_loss,
    brier_score_loss,
)
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


def wilson_ci(num_correct: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p_hat = num_correct / n
    denom = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2.0 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1.0 - p_hat) + (z ** 2) / (4.0 * n)) / n)) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


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


def safe_metrics(y_true: np.ndarray, proba: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    if len(y_true) == 0 or len(proba) == 0:
        return {
            "n": 0,
            "auc": float("nan"),
            "logloss": float("nan"),
            "brier": float("nan"),
            "acc": float("nan"),
            "f1": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    preds = (proba >= thr).astype(int)
    auc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) == 2 else float("nan")
    ll = log_loss(y_true, proba)
    brier = brier_score_loss(y_true, proba)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, zero_division=0)
    correct = int(np.sum(preds == y_true))
    ci_low, ci_high = wilson_ci(correct, len(y_true))
    return {
        "n": int(len(y_true)),
        "auc": float(auc) if not (isinstance(auc, float) and np.isnan(auc)) else None,
        "logloss": float(ll),
        "brier": float(brier),
        "acc": float(acc),
        "f1": float(f1),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


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


# %% CELL 7: TRAIN BASE LGBM
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

model_base = lgb.LGBMClassifier(**lgb_params)
if len(X_tr) and len(y_tr):
    model_base.fit(X_tr, y_tr)
else:
    model_base.fit(X_train_s, y_train)


# %% CELL 8: CALIBRATION (ISOTONIC + PLATT)
if len(X_va) and len(y_va):
    cal_iso = CalibratedClassifierCV(model_base, method='isotonic', cv='prefit')
    cal_iso.fit(X_va, y_va)

    cal_platt = CalibratedClassifierCV(model_base, method='sigmoid', cv='prefit')
    cal_platt.fit(X_va, y_va)
else:
    warnings.warn("Validation split is empty; cannot perform calibration. Metrics will be NaN.")
    cal_iso = None
    cal_platt = None


# %% CELL 9: EVALUATION
X_test_s = scaler.transform(X_test) if len(X_test) else np.empty((0, X_train_s.shape[1]))

metrics = {
    "val": {},
    "test": {},
}

if cal_iso is not None:
    proba_iso_val = cal_iso.predict_proba(X_va)[:, 1] if len(X_va) else np.array([])
    proba_iso_test = cal_iso.predict_proba(X_test_s)[:, 1] if len(X_test_s) else np.array([])
    metrics["val"]["isotonic"] = safe_metrics(y_va, proba_iso_val)
    metrics["test"]["isotonic"] = safe_metrics(y_test, proba_iso_test)
else:
    metrics["val"]["isotonic"] = safe_metrics(np.array([]), np.array([]))
    metrics["test"]["isotonic"] = safe_metrics(np.array([]), np.array([]))

if cal_platt is not None:
    proba_platt_val = cal_platt.predict_proba(X_va)[:, 1] if len(X_va) else np.array([])
    proba_platt_test = cal_platt.predict_proba(X_test_s)[:, 1] if len(X_test_s) else np.array([])
    metrics["val"]["platt"] = safe_metrics(y_va, proba_platt_val)
    metrics["test"]["platt"] = safe_metrics(y_test, proba_platt_test)
else:
    metrics["val"]["platt"] = safe_metrics(np.array([]), np.array([]))
    metrics["test"]["platt"] = safe_metrics(np.array([]), np.array([]))


# Generalization gap (accuracy): val -> test
def gap(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if isinstance(a, float) and np.isnan(a):
        return None
    if isinstance(b, float) and np.isnan(b):
        return None
    return float(a - b)

gap_acc_iso = gap(metrics["val"]["isotonic"].get("acc"), metrics["test"]["isotonic"].get("acc"))
gap_acc_platt = gap(metrics["val"]["platt"].get("acc"), metrics["test"]["platt"].get("acc"))


# %% CELL 10: PRINT SUMMARY (sequential: isotonic first, then platt)
def fmt_ci(m: Dict[str, float]) -> str:
    lo, hi = m.get("ci_low"), m.get("ci_high")
    if lo is None or hi is None or (isinstance(lo, float) and np.isnan(lo)) or (isinstance(hi, float) and np.isnan(hi)):
        return "[nan, nan]"
    return f"[{lo:.4f}, {hi:.4f}]"

print("=== Isotonic calibration ===")
print("Validation (n={}):".format(metrics["val"].get("isotonic", {}).get("n", 0)))
print(
    "  AUC: {auc:.4f}, LogLoss: {ll:.4f}, Brier: {br:.4f}, Acc@0.5: {ac:.4f}, F1: {f1:.4f}, CI: {ci}".format(
        auc=metrics["val"]["isotonic"].get("auc", float("nan")) or float("nan"),
        ll=metrics["val"]["isotonic"].get("logloss", float("nan")),
        br=metrics["val"]["isotonic"].get("brier", float("nan")),
        ac=metrics["val"]["isotonic"].get("acc", float("nan")),
        f1=metrics["val"]["isotonic"].get("f1", float("nan")),
        ci=fmt_ci(metrics["val"]["isotonic"]),
    )
)
print("Test (n={}):".format(metrics["test"].get("isotonic", {}).get("n", 0)))
print(
    "  AUC: {auc:.4f}, LogLoss: {ll:.4f}, Brier: {br:.4f}, Acc@0.5: {ac:.4f}, F1: {f1:.4f}, CI: {ci}".format(
        auc=metrics["test"]["isotonic"].get("auc", float("nan")) or float("nan"),
        ll=metrics["test"]["isotonic"].get("logloss", float("nan")),
        br=metrics["test"]["isotonic"].get("brier", float("nan")),
        ac=metrics["test"]["isotonic"].get("acc", float("nan")),
        f1=metrics["test"]["isotonic"].get("f1", float("nan")),
        ci=fmt_ci(metrics["test"]["isotonic"]),
    )
)
print("  Generalization gap (Acc val - test): {g:.4f}".format(
    g=(gap_acc_iso if gap_acc_iso is not None else float("nan"))
))

print("\n=== Platt (sigmoid) calibration ===")
print("Validation (n={}):".format(metrics["val"].get("platt", {}).get("n", 0)))
print(
    "  AUC: {auc:.4f}, LogLoss: {ll:.4f}, Brier: {br:.4f}, Acc@0.5: {ac:.4f}, F1: {f1:.4f}, CI: {ci}".format(
        auc=metrics["val"]["platt"].get("auc", float("nan")) or float("nan"),
        ll=metrics["val"]["platt"].get("logloss", float("nan")),
        br=metrics["val"]["platt"].get("brier", float("nan")),
        ac=metrics["val"]["platt"].get("acc", float("nan")),
        f1=metrics["val"]["platt"].get("f1", float("nan")),
        ci=fmt_ci(metrics["val"]["platt"]),
    )
)
print("Test (n={}):".format(metrics["test"].get("platt", {}).get("n", 0)))
print(
    "  AUC: {auc:.4f}, LogLoss: {ll:.4f}, Brier: {br:.4f}, Acc@0.5: {ac:.4f}, F1: {f1:.4f}, CI: {ci}".format(
        auc=metrics["test"]["platt"].get("auc", float("nan")) or float("nan"),
        ll=metrics["test"]["platt"].get("logloss", float("nan")),
        br=metrics["test"]["platt"].get("brier", float("nan")),
        ac=metrics["test"]["platt"].get("acc", float("nan")),
        f1=metrics["test"]["platt"].get("f1", float("nan")),
        ci=fmt_ci(metrics["test"]["platt"]),
    )
)
print("  Generalization gap (Acc val - test): {g:.4f}".format(
    g=(gap_acc_platt if gap_acc_platt is not None else float("nan"))
))

print("\n=== Head-to-head (summary) ===")
print("Val Acc@0.5 — Isotonic: {ai:.4f}, Platt: {ap:.4f}".format(
    ai=metrics["val"]["isotonic"].get("acc", float("nan")),
    ap=metrics["val"]["platt"].get("acc", float("nan")),
))
print("Test Acc@0.5 — Isotonic: {ai:.4f}, Platt: {ap:.4f}".format(
    ai=metrics["test"]["isotonic"].get("acc", float("nan")),
    ap=metrics["test"]["platt"].get("acc", float("nan")),
))


# %% CELL 11: SAVE ARTIFACTS
model_base_path = os.path.join(RUN_DIR, 'lightgbm_financial_1d_iso_base.joblib')
scaler_path = os.path.join(RUN_DIR, 'scaler_1d_iso.joblib')
joblib.dump(model_base, model_base_path)
joblib.dump(scaler, scaler_path)

cal_iso_path = None
cal_platt_path = None
if cal_iso is not None:
    cal_iso_path = os.path.join(RUN_DIR, 'lightgbm_1d_iso_calibrated_isotonic.joblib')
    joblib.dump(cal_iso, cal_iso_path)
if cal_platt is not None:
    cal_platt_path = os.path.join(RUN_DIR, 'lightgbm_1d_iso_calibrated_platt.joblib')
    joblib.dump(cal_platt, cal_platt_path)

results = {
    "calibration_methods": ["isotonic", "platt"],
    "feature_names": FEATURE_NAMES,
    "metrics": metrics,
    "gaps": {
        "accuracy_val_minus_test": {
            "isotonic": gap_acc_iso,
            "platt": gap_acc_platt,
        }
    },
}

results_path = os.path.join(RUN_DIR, 'results_1d_iso_calibration.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

# Save run metadata and code snapshot
run_meta = {
    "script_name": os.path.basename(__file__),
    "run_timestamp_utc": RUN_TS,
    "artifact_dir": RUN_DIR,
    "artifacts": {
        "model_base": model_base_path,
        "scaler": scaler_path,
        "calibrated_isotonic": cal_iso_path,
        "calibrated_platt": cal_platt_path,
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


