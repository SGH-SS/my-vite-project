#!/usr/bin/env python3
"""
LightGBM 1D v0 — Training Script

Purpose:
- Train a LightGBM classifier on daily SPY bars using time-ordered CV,
  recency weights, EV-based thresholding, final refit with early stopping,
  and OOS evaluation. Saves model, scaler, reports, and OOS predictions.

CLI arguments:
- None. This script uses the constants in CELL 1 for configuration.
  To change behavior, edit those constants directly.

Defaults (editable in CELL 1):
- Input CSV: daygent/data/spy_1d.csv (columns: symbol, timestamp, raw_ohlcv_vec, iso_ohlc, future)
- Costs (round-trip): 6 bps (0.0006)
- Threshold sweep: 0.30 → 0.80 (step 0.01), EV floor = 3 bps
- Recency weighting: tau_bars=180 (by bars)
- Artifacts:
  - daygent/models/lightgbm_1d_v0.joblib
  - daygent/models/scaler_1d_v0.joblib
  - daygent/reports/lightgbm_1d_v0_threshold.json
  - daygent/reports/lightgbm_1d_v0_metrics.json
  - daygent/preds/lightgbm_1d_v0_oos.csv

Drive-aware behavior:
- If running in Colab (or if a local ./daygent_v1_models exists), the script
  will use BASE_DIR = '/content/drive/MyDrive/daygent_v1_models' (Colab)
  or './daygent_v1_models' (local) for:
    - Combined data: {BASE_DIR}/combined_spy_data/combined_spy_1d.csv (single source of truth)
    - Artifacts: {BASE_DIR}/lgbm_1d_v0
"""
# %% CELL 1: CROSS-PLATFORM DEPENDENCY MANAGEMENT
print("🔧 Setting up dependencies...")

# Cross-platform dependency installation
try:
    import pandas, numpy, sklearn, matplotlib, seaborn, joblib, tqdm
    import lightgbm as lgb
    print("✅ Core dependencies already available")
except ImportError as e:
    print(f"Installing missing dependencies: {e}")
    import sys, subprocess
    pkgs = ['pandas', 'numpy', 'scikit-learn', 'lightgbm',
            'matplotlib', 'seaborn', 'joblib', 'tqdm', 'pyarrow']
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + pkgs)
    import lightgbm as lgb
    print("✅ Dependencies installed")

# Core imports
import os
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


# %% CELL 2: CONFIGURATION & CONSTANTS
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Try to mount Google Drive if available (Colab environment)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IS_COLAB = True
    print("✅ Google Drive mounted (Colab environment)")
except ImportError:
    IS_COLAB = False
    print("✅ Local environment detected")

# Prefer Drive-style BASE_DIR if available; else use local daygent/* folders
if IS_COLAB:
    BASE_DIR = Path('/content/drive/MyDrive/daygent_v1_models')  # <— your new base folder
else:
    BASE_DIR = Path('./daygent_v1_models')

print(f"✅ Base directory: {BASE_DIR}")

# Local fallback dirs for artifacts
DATA_DIR = Path("daygent/data")
MODELS_DIR = Path("daygent/models")
REPORTS_DIR = Path("daygent/reports")
PREDS_DIR = Path("daygent/preds")

for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR, PREDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Combined data dir (Drive) – single source of truth
COMBINED_DIR_DRIVE = BASE_DIR / 'combined_spy_data'
COMBINED_1D_CSV_DRIVE = COMBINED_DIR_DRIVE / 'combined_spy_1d.csv'
print(f"✅ Combined data path: {COMBINED_1D_CSV_DRIVE}")

# If Drive-style artifacts dir exists or we're on Colab, save artifacts there
ARTIFACT_DIR = None
if IS_COLAB or BASE_DIR.exists():
    ARTIFACT_DIR = BASE_DIR / 'lgbm_1d_v0'
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = ARTIFACT_DIR / "lightgbm_1d_v0.joblib"
    SCALER_PATH = ARTIFACT_DIR / "scaler_1d_v0.joblib"
    THRESHOLD_JSON = ARTIFACT_DIR / "lightgbm_1d_v0_threshold.json"
    METRICS_JSON = ARTIFACT_DIR / "lightgbm_1d_v0_metrics.json"
    OOS_PRED_CSV = ARTIFACT_DIR / "lightgbm_1d_v0_oos.csv"
    print(f"✅ Model directory: {ARTIFACT_DIR}")

# Trading/eval parameters
COST_ROUNDTRIP = 0.0006  # 6 bps default; can be replaced by per-day vector later
THRESHOLDS = np.round(np.arange(0.30, 0.801, 0.01), 2)
EV_FLOOR = 0.0003  # 3 bps minimum EV in sweep
MIN_TRADES_PER_FOLD = 30  # guard against thin selections
TAU_BARS = 180  # recency weighting in number of bars (not calendar days)

# Feature contract (ordered)
EXPECTED_FEATS = [
    "raw_o","raw_h","raw_l","raw_c","raw_v",
    "iso_ohlc_o","iso_ohlc_h","iso_ohlc_l","iso_ohlc_c",
    "tf_1d","tf_4h",
    "hl_range","price_change","upper_shadow","lower_shadow","volume_m"
]

if 'ARTIFACT_DIR' in globals() and ARTIFACT_DIR is not None:
    # Already set to Drive above
    pass
else:
    # Local single artifact directory for all outputs
    ARTIFACT_DIR = Path("daygent/lgbm_1d_v0")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = ARTIFACT_DIR / "lightgbm_1d_v0.joblib"
    SCALER_PATH = ARTIFACT_DIR / "scaler_1d_v0.joblib"
    THRESHOLD_JSON = ARTIFACT_DIR / "lightgbm_1d_v0_threshold.json"
    METRICS_JSON = ARTIFACT_DIR / "lightgbm_1d_v0_metrics.json"
    OOS_PRED_CSV = ARTIFACT_DIR / "lightgbm_1d_v0_oos.csv"


# %% CELL 3: UTILS (RECENCY WEIGHTING, METRICS, SAFE CAST)
def compute_recency_weights_by_bars(num_bars: int, tau_bars: int) -> np.ndarray:
    """Exponential recency weights by index distance (bars), not calendar days.

    Newest bar gets weight=1; weight decays backwards with tau_bars.
    """
    if num_bars <= 0:
        return np.array([])
    idx = np.arange(num_bars)
    delta = idx - idx.max()
    return np.exp(delta / float(tau_bars))


def ev_sweep(prob_up: np.ndarray,
             ret_next: np.ndarray,
             thresholds: np.ndarray,
             cost: float,
             min_trades: int = 0,
             ev_floor: float = -np.inf,
             cost_vec: np.ndarray | None = None) -> dict:
    """Sweep thresholds with guards; pick the one with max EV per trade.

    Guards:
      - require at least min_trades
      - require EV >= ev_floor
    """
    best = {
        "best_theta": None,
        "best_ev": -np.inf,
        "best_hit_net": np.nan,
        "best_hit_gross": np.nan,
        "best_trades": 0,
    }
    for th in thresholds:
        mask = prob_up >= th
        n = int(mask.sum())
        if n < min_trades:
            continue
        if cost_vec is not None:
            r_net = ret_next[mask] - cost_vec[mask]
        else:
            r_net = ret_next[mask] - cost
        ev = float(np.mean(r_net))
        if ev < ev_floor:
            continue
        hit_net = float(np.mean(r_net > 0))
        hit_gross = float(np.mean(ret_next[mask] > 0))
        if ev > best["best_ev"]:
            best.update({
                "best_theta": float(th),
                "best_ev": ev,
                "best_hit_net": hit_net,
                "best_hit_gross": hit_gross,
                "best_trades": n,
            })
    return best


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


# %% CELL 4: LOAD DATA & BUILD FEATURES/TARGETS (21-FEATURE CONTRACT)
def build_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Build feature vector (len=16) + binary label + next-day log return + dates.

    Input columns expected: symbol, timestamp, raw_ohlcv_vec, iso_ohlc, future
    Feature order (EXPECTED_FEATS):
      raw_o, raw_h, raw_l, raw_c, raw_v,
      iso_ohlc_o, iso_ohlc_h, iso_ohlc_l, iso_ohlc_c,
      tf_1d, tf_4h,
      hl_range, price_change, upper_shadow, lower_shadow, volume_m
    """
    df = df.copy()
    # Sort & sanity
    # Normalize and sort strictly by known 'timestamp' column
    if "timestamp" not in df.columns:
        raise ValueError("Input CSV must include 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"]  # maintain compatibility downstream

    if not df['date'].is_monotonic_increasing:
        raise ValueError("Dates not strictly increasing before features.")
    if df['date'].duplicated().any():
        raise ValueError("Duplicate dates found before feature construction.")

    # Basic checks
    required = ["raw_ohlcv_vec", "iso_ohlc", "future"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Raw features
    feat = pd.DataFrame(index=df.index)
    # Parse raw_ohlcv_vec column which is a stringified list [o,h,l,c,v]
    def _parse_vec(s):
        if pd.isna(s):
            return None
        if isinstance(s, (list, tuple, np.ndarray)):
            return list(s)
        try:
            s = str(s).strip().strip('[]')
            return [float(x.strip()) for x in s.split(',')]
        except Exception:
            return None

    ohlcv = df["raw_ohlcv_vec"].apply(_parse_vec)
    parts = ohlcv.apply(lambda x: x if x and len(x) == 5 else [np.nan]*5)
    feat["raw_o"] = parts.apply(lambda x: x[0])
    feat["raw_h"] = parts.apply(lambda x: x[1])
    feat["raw_l"] = parts.apply(lambda x: x[2])
    feat["raw_c"] = parts.apply(lambda x: x[3])
    feat["raw_v"] = parts.apply(lambda x: x[4])

    # Engineered simple ratios base fields
    c = feat["raw_c"].astype(float)
    o = feat["raw_o"].astype(float)
    h = feat["raw_h"].astype(float)
    l = feat["raw_l"].astype(float)
    v = feat["raw_v"].astype(float)

    eps = 1e-12

    # ISO OHLC comes as a stringified list [iso_o, iso_h, iso_l, iso_c]
    def _parse_iso(s):
        if pd.isna(s):
            return None
        if isinstance(s, (list, tuple, np.ndarray)):
            return list(s)
        try:
            s = str(s).strip().strip('[]')
            return [float(x.strip()) for x in s.split(',')]
        except Exception:
            return None

    iso = df["iso_ohlc"].apply(_parse_iso)
    iso_parts = iso.apply(lambda x: x if x and len(x) == 4 else [0.0]*4)
    feat["iso_ohlc_o"] = iso_parts.apply(lambda x: x[0])
    feat["iso_ohlc_h"] = iso_parts.apply(lambda x: x[1])
    feat["iso_ohlc_l"] = iso_parts.apply(lambda x: x[2])
    feat["iso_ohlc_c"] = iso_parts.apply(lambda x: x[3])

    # iso_ohlcv (5)
    vv = np.log1p(v)
    m5 = (o + h + l + c + vv) / 5.0
    s5 = np.sqrt(((o - m5) ** 2 + (h - m5) ** 2 + (l - m5) ** 2 + (c - m5) ** 2 + (vv - m5) ** 2) / 5.0)
    s5_safe = s5.mask(s5 <= 0, 1.0)

    feat["iso_ohlcv_o"] = ((o - m5) / s5_safe).where(s5 > 0, 0.0)
    feat["iso_ohlcv_h"] = ((h - m5) / s5_safe).where(s5 > 0, 0.0)
    feat["iso_ohlcv_l"] = ((l - m5) / s5_safe).where(s5 > 0, 0.0)
    feat["iso_ohlcv_c"] = ((c - m5) / s5_safe).where(s5 > 0, 0.0)
    feat["iso_ohlcv_v"] = ((vv - m5) / s5_safe).where(s5 > 0, 0.0)

    # Timeframe one-hot
    feat["tf_1d"] = 1.0
    feat["tf_4h"] = 0.0

    # Engineered simple ratios
    feat["hl_range"] = (h - l) / (c.replace(0, np.nan) + eps)
    feat["price_change"] = (c - o) / (o.replace(0, np.nan) + eps)
    feat["upper_shadow"] = (h - c) / (c.replace(0, np.nan) + eps)
    feat["lower_shadow"] = (c - l) / (c.replace(0, np.nan) + eps)
    feat["volume_m"] = v / 1_000_000.0

    # Targets
    # Labels from provided 'future' and returns from raw close
    ret_next = np.log(c.shift(-1) / c)
    if 'future' in df.columns:
        y_cls = df['future'].astype(int)
    else:
        y_cls = (ret_next > 0).astype(int)

    # Drop last row (no next-day)
    mask = ret_next.notna()
    feat = feat[mask].reset_index(drop=True)
    y_cls = y_cls[mask].reset_index(drop=True)
    ret_next = ret_next[mask].reset_index(drop=True)
    dates = df.loc[mask, "date"].reset_index(drop=True)

    # Enforce expected feature contract/order
    feat = feat[EXPECTED_FEATS]

    # Sanity: ensure no NaNs/Infs
    vals = feat.values
    if not np.isfinite(vals).all():
        raise ValueError("NaNs or Infs detected in features after construction.")

    return feat, y_cls, ret_next, dates


# %% CELL 5: 5-FOLD TIME-SERIES CV WITH RECENCY WEIGHTS & EV THRESHOLDING
def run_time_series_cv(X: np.ndarray,
                       y: np.ndarray,
                       r: np.ndarray,
                       dates: pd.Series,
                       cutoff_date: pd.Timestamp) -> tuple[float, list, int]:
    """Run 5-fold TimeSeriesSplit on in-sample portion (<= cutoff).

    Returns:
      - theta_deploy (median of fold-best thresholds)
      - fold_stats (list of dicts)
      - best_iter_median (median of best_iteration_ across folds)
    """
    in_sample_idx = np.where(dates.values.astype("datetime64[ns]") <= np.datetime64(cutoff_date))[0]
    tscv = TimeSeriesSplit(n_splits=5)

    fold_stats = []
    fold_thetas = []
    best_iters = []

    for k, (tr_rel, va_rel) in enumerate(tscv.split(in_sample_idx), start=1):
        tr_idx = in_sample_idx[tr_rel]
        va_idx = in_sample_idx[va_rel]

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        r_va = r[va_idx]
        d_tr = dates.iloc[tr_idx]

        # Per-fold scaler
        scaler = StandardScaler()
        X_trs = scaler.fit_transform(X_tr)
        X_vas = scaler.transform(X_va)

        # Recency weights on training slice (by bars)
        w_tr = compute_recency_weights_by_bars(len(d_tr), TAU_BARS)

        # Model config
        clf = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            metric=["auc", "binary_logloss"],
            learning_rate=0.05,
            num_leaves=63,
            max_depth=6,
            min_child_samples=40,
            feature_fraction=0.85,
            bagging_fraction=0.85,
            bagging_freq=1,
            lambda_l1=0.1,
            lambda_l2=0.1,
            min_gain_to_split=0.0,
            n_estimators=4000,
            n_jobs=-1,
            verbose=-1,
            random_state=RANDOM_STATE,
            bagging_seed=RANDOM_STATE,
            feature_fraction_seed=RANDOM_STATE,
            deterministic=True,
        )

        clf.fit(
            X_trs, y_tr,
            sample_weight=w_tr if len(w_tr) == len(y_tr) else None,
            eval_set=[(X_vas, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )

        best_iter = int(getattr(clf, "best_iteration_", clf.n_estimators))
        best_iters.append(best_iter)

        p_va = clf.predict_proba(X_vas)[:, 1]
        # threshold sweep with guards via helper
        best = ev_sweep(p_va, r_va, THRESHOLDS, COST_ROUNDTRIP,
                        min_trades=MIN_TRADES_PER_FOLD, ev_floor=EV_FLOOR)

        fold_stats.append({
            "fold": k,
            "auc": float(clf.best_score_.get("valid_0", {}).get("auc", np.nan)),
            "best_ev": safe_float(best.get("best_ev")),
            "hit_net": safe_float(best.get("best_hit_net")),
            "hit_gross": safe_float(best.get("best_hit_gross")),
            "n_trades": int(best["best_trades"]),
            "best_theta": safe_float(best.get("best_theta")),
            "best_iter": best_iter,
            "p_up_mean": float(p_va.mean()),
            "p_up_std": float(p_va.std(ddof=1)),
        })
        if best.get("best_theta") is not None:
            fold_thetas.append(best["best_theta"])

    if len(fold_thetas) == 0:
        raise RuntimeError("No threshold met EV/trade floor and min-trades guard in any fold.")
    else:
        theta_deploy = float(np.median(fold_thetas))

    best_iter_median = int(np.median(best_iters)) if len(best_iters) else 1000
    return theta_deploy, fold_stats, best_iter_median


# %% CELL 6: FINAL REFIT WITH EARLY STOPPING HOLDOUT, THEN OOS SCORING
def refit_and_score_oos(X: np.ndarray,
                        y: np.ndarray,
                        r: np.ndarray,
                        dates: pd.Series,
                        cutoff_date: pd.Timestamp,
                        theta_deploy: float,
                        best_iter_hint: int) -> dict:
    """Refit on in-sample with an ES holdout, then score OOS and compute metrics."""
    in_mask = dates.values.astype("datetime64[ns]") <= np.datetime64(cutoff_date)
    X_in, y_in = X[in_mask], y[in_mask]

    # 90/10 split for early stopping only
    cut = int(0.9 * len(X_in))
    if cut <= 0:
        raise RuntimeError("Not enough in-sample data for early-stopping split.")
    X_tr_all, X_es = X_in[:cut], X_in[cut:]
    y_tr_all, y_es = y_in[:cut], y_in[cut:]
    d_tr_all = pd.to_datetime(dates[in_mask].iloc[:cut])

    scaler_final = StandardScaler().fit(X_tr_all)
    X_tr_all_s = scaler_final.transform(X_tr_all)
    X_es_s = scaler_final.transform(X_es)

    # Recency weights for full train (by bars)
    w_tr_all = compute_recency_weights_by_bars(len(d_tr_all), TAU_BARS)

    clf_final = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        metric=["auc", "binary_logloss"],
        learning_rate=0.05,
        num_leaves=63,
        max_depth=6,
        min_child_samples=40,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=0.1,
        lambda_l2=0.1,
        min_gain_to_split=0.0,
        n_estimators=max(1000, int(best_iter_hint * 2)),  # large cap; ES will cut down
        n_jobs=-1,
        verbose=-1,
        random_state=RANDOM_STATE,
        bagging_seed=RANDOM_STATE,
        feature_fraction_seed=RANDOM_STATE,
        deterministic=True,
    )

    clf_final.fit(
        X_tr_all_s, y_tr_all,
        sample_weight=w_tr_all if len(w_tr_all) == len(y_tr_all) else None,
        eval_set=[(X_es_s, y_es)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
    )

    # Determine best iteration from ES, then refit scaler on full in-sample and model with fixed trees
    best_n = int(getattr(clf_final, "best_iteration_", clf_final.n_estimators))
    X_in = X[in_mask]
    y_in = y[in_mask]
    scaler_final = StandardScaler().fit(X_in)
    X_in_s = scaler_final.transform(X_in)

    clf_final_refit = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        metric=["auc", "binary_logloss"],
        learning_rate=0.05,
        num_leaves=63,
        max_depth=6,
        min_child_samples=40,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=0.1,
        lambda_l2=0.1,
        min_gain_to_split=0.0,
        n_estimators=max(50, best_n),
        n_jobs=-1,
        verbose=-1,
        random_state=RANDOM_STATE,
        bagging_seed=RANDOM_STATE,
        feature_fraction_seed=RANDOM_STATE,
        deterministic=True,
    )
    # Apply recency weights for final refit as well
    w_in = compute_recency_weights_by_bars(len(y_in), TAU_BARS)
    clf_final_refit.fit(X_in_s, y_in, sample_weight=w_in)

    # Score OOS — now prefer FRONTTEST rows for the OOS window (already merged in load)
    oos_mask = dates.values.astype("datetime64[ns]") > np.datetime64(cutoff_date)
    X_oos = X[oos_mask]
    r_oos = r[oos_mask]
    d_oos = pd.to_datetime(dates[oos_mask])

    X_oos_s = scaler_final.transform(X_oos) if len(X_oos) else np.empty((0, X_in_s.shape[1]))
    p_oos = clf_final_refit.predict_proba(X_oos_s)[:, 1] if len(X_oos_s) else np.array([])

    trade = (p_oos >= theta_deploy).astype(int) if len(p_oos) else np.array([])
    ev_contrib = trade * (r_oos - COST_ROUNDTRIP) if len(trade) else np.array([])

    # Metrics (report both gross and net hit)
    trades = int(trade.sum()) if len(trade) else 0
    ev_per_trade = float(ev_contrib[trade == 1].mean()) if trades > 0 else float("nan")
    hit_gross = float((r_oos[trade == 1] > 0).mean()) if trades > 0 else float("nan")
    hit_net = float(((r_oos[trade == 1] - COST_ROUNDTRIP) > 0).mean()) if trades > 0 else float("nan")
    trades_per_day = float(trades / len(r_oos)) if len(r_oos) else 0.0

    if len(ev_contrib):
        cum_log = np.cumsum(ev_contrib)
        cum_pct = np.exp(cum_log) - 1.0
        # Max drawdown on equity curve in percent space
        equity = 1.0 + cum_pct
        max_equity = np.maximum.accumulate(equity)
        dd = 1.0 - (equity / np.maximum(max_equity, 1e-12))
        max_dd = float(np.max(dd))
        sharpe_daily = float((ev_contrib.mean() / (ev_contrib.std(ddof=1) + 1e-12)) * math.sqrt(252.0))
        cum_log_last = float(cum_log[-1])
        cum_pct_last = float(cum_pct[-1])
        exposure = float((trade.mean()))
        ann_return = float((1.0 + cum_pct_last) ** (252.0 / len(r_oos)) - 1.0) if len(r_oos) else float('nan')
        bh_cum = float(np.exp(r_oos.sum()) - 1.0)
    else:
        max_dd = float("nan")
        sharpe_daily = float("nan")
        cum_log_last = 0.0
        cum_pct_last = 0.0
        exposure = 0.0
        ann_return = float('nan')
        bh_cum = float('nan')

    # Save artifacts
    joblib.dump(clf_final_refit, MODEL_PATH)
    joblib.dump(scaler_final, SCALER_PATH)

    # Save per-day OOS predictions
    pd.DataFrame({
        "date": d_oos.astype(str),
        "p_up": p_oos,
        "trade": trade,
        "ret_next": r_oos,
        "ev_contrib": ev_contrib,
    }).to_csv(OOS_PRED_CSV, index=False)

    return {
        "oos_start": str(d_oos.min().date()) if len(d_oos) else None,
        "oos_end": str(d_oos.max().date()) if len(d_oos) else None,
        "trades": trades,
        "trades_per_day": trades_per_day,
        "ev_per_trade": ev_per_trade,
        "hit_rate_gross": hit_gross,
        "hit_rate_net": hit_net,
        "cum_log_pnl": cum_log_last,
        "cum_pct_pnl": cum_pct_last,
        "max_drawdown_pct": max_dd,
        "sharpe_daily": sharpe_daily,
        "best_iteration_final": int(getattr(clf_final_refit, "n_estimators", 0)),
        "exposure": exposure,
        "annualized_return": ann_return,
        "bh_cum_pct": bh_cum,
    }


# %% CELL 7: MAIN ENTRYPOINT — COMBINED DATA ONLY
def load_combined_df() -> pd.DataFrame:
    """Load combined 1D CSV. Expect exact columns: symbol, timestamp, raw_ohlcv_vec, iso_ohlc, future."""
    if not COMBINED_1D_CSV_DRIVE.exists():
        raise FileNotFoundError(f"Combined CSV not found at {COMBINED_1D_CSV_DRIVE}")
    df = pd.read_csv(COMBINED_1D_CSV_DRIVE)
    required = ["symbol", "timestamp", "raw_ohlcv_vec", "iso_ohlc", "future"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Combined CSV missing required columns: {missing}")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def main():
    # Load combined dataset only
    print(f"📄 Using combined dataset: {COMBINED_1D_CSV_DRIVE}")
    raw_all = load_combined_df()

    # Use only the last 6 years of data (by timestamp)
    ts_all = pd.to_datetime(raw_all['timestamp'])
    last_date = ts_all.max().normalize()
    start_date = last_date - pd.DateOffset(years=6)
    raw_all = raw_all.loc[ts_all >= start_date].reset_index(drop=True)

    # Build features/targets (after 6-year filter)
    feat, y_cls, ret_next, dates = build_features_targets(raw_all)

    # Cutoff: last 63 trading rows are OOS; everything before is in-sample
    if len(dates) <= 63:
        raise RuntimeError("Not enough data after 6-year filter to allocate 63 OOS rows.")
    cutoff_idx = len(dates) - 63 - 1
    cutoff_date = pd.to_datetime(dates.iloc[cutoff_idx]).normalize()
    X = feat.values.astype(float)
    y = y_cls.values.astype(int)
    r = ret_next.values.astype(float)

    # Run CV for theta and iteration hint
    theta_deploy, fold_stats, best_iter_median = run_time_series_cv(
        X, y, r, dates, cutoff_date
    )

    # Final refit & OOS
    oos_stats = refit_and_score_oos(
        X, y, r, dates, cutoff_date, theta_deploy, best_iter_median
    )

    # Save threshold & metrics JSONs + manifest
    with open(THRESHOLD_JSON, "w") as f:
        json.dump({
            "theta": float(theta_deploy),
            "cost_bps": 6,
            "fold_stats": fold_stats,
            "theta_selection": "median_of_fold_bests",
        }, f, indent=2)

    with open(METRICS_JSON, "w") as f:
        json.dump({
            "in_sample": {
                "folds": fold_stats,
                "theta_deploy": float(theta_deploy),
            },
            "oos": {
                **oos_stats
            },
        }, f, indent=2)

    # Persist feature contract and config manifest
    feature_names = list(feat.columns)
    manifest = {
        "model_type": "LightGBMClassifier",
        "random_state": RANDOM_STATE,
        "cost_roundtrip": COST_ROUNDTRIP,
        "cost_mode": "scalar",  # upgrade to per_row in future
        "tau_bars": TAU_BARS,
        "threshold_sweep": [float(THRESHOLDS.min()), float(THRESHOLDS.max()), 0.01],
        "ev_floor": EV_FLOOR,
        "min_trades_per_fold": MIN_TRADES_PER_FOLD,
        "feature_names": feature_names,
        "cutoff_date": str(pd.to_datetime(cutoff_date).date()),
        "theta_deploy": float(theta_deploy),
        "guards": {
            "ev_floor": EV_FLOOR,
            "min_trades_per_fold": MIN_TRADES_PER_FOLD
        },
        "artifact_paths": {
            "model": str(MODEL_PATH),
            "scaler": str(SCALER_PATH),
            "threshold_json": str(THRESHOLD_JSON),
            "metrics_json": str(METRICS_JSON),
            "oos_preds_csv": str(OOS_PRED_CSV),
        },
        "lgbm_params": {
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 6,
            "min_child_samples": 40,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
        },
        "execution_convention": "enter close_t, exit close_t+1 (log returns)",
    }
    with open((ARTIFACT_DIR if ARTIFACT_DIR else REPORTS_DIR) / "model_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Minimal console output
    print("\n=== LightGBM 1D v0 — COMPLETE ===")
    print(f"Theta (deploy): {theta_deploy:.2f}")
    print(f"OOS window: {oos_stats['oos_start']} → {oos_stats['oos_end']}")
    hit_g = oos_stats.get('hit_rate_gross', float('nan'))
    hit_n = oos_stats.get('hit_rate_net', float('nan'))
    print(f"Trades: {oos_stats['trades']}  | EV/trade: {oos_stats['ev_per_trade']:.6f}  | Hit(gross/net): {hit_g:.3f}/{hit_n:.3f}")
    print(f"Cum % PnL: {oos_stats['cum_pct_pnl']*100:.2f}%  | MaxDD: {oos_stats['max_drawdown_pct']*100:.2f}%  | Sharpe(d): {oos_stats['sharpe_daily']:.2f}")
    print(f"Exposure: {oos_stats['exposure']:.3f}  | Annualized Return: {oos_stats['annualized_return']*100:.2f}%  | Baseline(BH OOS): {oos_stats['bh_cum_pct']*100:.2f}%")
    print("Artifacts saved:")
    print(f"- Model:   {MODEL_PATH}")
    print(f"- Scaler:  {SCALER_PATH}")
    print(f"- Reports: {THRESHOLD_JSON}, {METRICS_JSON}")
    print(f"- Preds:   {OOS_PRED_CSV}")


if __name__ == "__main__":
    main()


