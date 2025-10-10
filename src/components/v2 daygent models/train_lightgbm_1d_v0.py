#!/usr/bin/env python3
"""
LightGBM 1D v0 — Regression + Policy (NumPy-safe) — Local DB

Loads SPY 1D from Postgres (backtest + fronttest), builds a technical-
feature regression target (sum of next K daily log returns), finds a
deployment threshold via purged CV with a risk gate and monthly cap,
refits, runs rolling OOS, and saves artifacts locally.

Conda env: daygent-train
"""

from __future__ import annotations

# %% CELL 1: SETUP (NumPy-safe)
print("🔧 Setting up dependencies (NumPy-safe)...")
import sys, subprocess, importlib, warnings, os, json, math
from pathlib import Path
from datetime import datetime, timezone

# Only install what's missing; do NOT pin/downgrade NumPy

def _ensure(pkg: str) -> bool:
    try:
        importlib.import_module(pkg)
        return False
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        return True

# Do NOT touch numpy version — just import it
import numpy as np
for _p in ["pandas", "joblib", "tqdm", "lightgbm", "pyarrow", "scikit-learn", "sqlalchemy", "psycopg"]:
    _ensure(_p)

import pandas as pd
import joblib
from tqdm import tqdm
import lightgbm as lgb
warnings.filterwarnings("ignore")

from sqlalchemy import create_engine, text

# %% CELL 2: CONFIG
DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5433/trading_data"
ARTIFACT_DIR = Path(os.path.dirname(__file__)) / 'artifacts_lgbm_1d_v0'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamped run folder
RUNS_DIR = ARTIFACT_DIR / 'runs'
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RUN_TS = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
RUN_DIR = RUNS_DIR / RUN_TS
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RUN_DIR / 'lightgbm_1d_v0.joblib'
SCALER_PATH = RUN_DIR / 'scaler_1d_v0.joblib'  # identity scaler saved later
THRESHOLD_JSON = RUN_DIR / 'lightgbm_1d_v0_threshold.json'
METRICS_JSON = RUN_DIR / 'lightgbm_1d_v0_metrics.json'
OOS_PRED_CSV = RUN_DIR / 'lightgbm_1d_v0_oos_latest.csv'
MANIFEST_JSON = RUN_DIR / 'model_manifest.json'
FEATURES_JSON = RUN_DIR / 'feature_names.json'

# No longer using public/training folder - UI reads directly from artifacts

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

YEARS_BACK = 5
K_NEXT = 2  # sum next K daily log returns
TAU_BARS = 180
N_SPLITS = 5
GAP_DAYS = 5
EARLY_STOP_ROUNDS = 600

# Wider, lower threshold sweep (on regression scores)
THR_Q_LOW = 0.55
THR_Q_HIGH = 0.92
THR_Q_STEPS = 31
TARGET_TRADES_PM = (7, 15)  # per-month (min, max)
OOS_MONTHS = 3
WFO_YEARS_BACK = 3

# Trading cost (round-trip)
COST_ROUNDTRIP = 0.0006

# Risk gate
GATE = dict(
    use=True,
    require_macd_hist_pos=False,
    bb_z_min=-1.5,
    max_gap_abs=0.02,
)

# LGBM (Huber regression)
LGB_PARAMS = dict(
    objective='huber',
    alpha=0.90,
    boosting_type='gbdt',
    learning_rate=0.04,
    num_leaves=95,
    max_depth=-1,
    min_child_samples=150,
    feature_fraction=0.90,
    bagging_fraction=0.90,
    bagging_freq=1,
    lambda_l1=0.0,
    lambda_l2=1.0,
    n_estimators=8000,
    n_jobs=-1,
    verbose=-1,
    random_state=RANDOM_STATE,
    deterministic=True,
)

print(f"✅ NumPy {np.__version__} | pandas {pd.__version__} | LightGBM {lgb.__version__}")
print(f"✅ Artifacts: {ARTIFACT_DIR}")

# %% CELL 3: UTILITIES (scaler, weights, metrics)
class IdentityTransformer:
    """Drop-in 'scaler' that does nothing (keeps deployment interface)."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
    def fit_transform(self, X, y=None):
        return X

def compute_recency_weights_by_bars(n_bars: int, tau_bars: int) -> np.ndarray:
    if n_bars <= 0:
        return np.array([])
    idx = np.arange(n_bars)
    delta = idx - idx.max()
    return np.exp(delta / float(tau_bars))

def month_periods(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.to_period('M')

def enforce_monthly_budget(scores: np.ndarray,
                           dates: pd.Series,
                           base_mask: np.ndarray,
                           kmin_kmax=(3,8)) -> np.ndarray:
    """Offline top-K per month by score (diagnostics)."""
    months = month_periods(dates)
    mask = np.zeros_like(base_mask, dtype=bool)
    kmin, kmax = kmin_kmax
    for m in np.unique(months):
        idx = np.where(months == m)[0]
        sel = idx[base_mask[idx]]
        if sel.size == 0:
            continue
        keep = sel[np.argsort(scores[sel])[::-1][:kmax]]
        mask[keep] = True
    return mask

def online_monthly_cap(scores: np.ndarray,
                       dates: pd.Series,
                       base_mask: np.ndarray,
                       kmax: int) -> np.ndarray:
    """Causal monthly budget: fill slots as days arrive."""
    months = pd.to_datetime(dates).dt.to_period('M')
    kept = np.zeros_like(base_mask, dtype=bool)
    used: dict = {}
    for i in range(len(scores)):  # chronological
        if not base_mask[i]:
            continue
        m = months.iloc[i]
        cnt = used.get(m, 0)
        if cnt < kmax:
            kept[i] = True
            used[m] = cnt + 1
    return kept

def max_drawdown_pct(log_returns: np.ndarray) -> float:
    if len(log_returns) == 0:
        return float('nan')
    equity = np.exp(np.cumsum(log_returns))
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - equity / np.maximum(peak, 1e-12)
    return float(np.max(dd))

def sharpe_daily(log_returns: np.ndarray) -> float:
    if len(log_returns) < 2:
        return float('nan')
    mu = float(np.mean(log_returns))
    sd = float(np.std(log_returns, ddof=1))
    if sd == 0.0:
        return float('nan')
    return (mu / sd) * math.sqrt(252.0)

def ev_metrics_from_trades(trade_mask: np.ndarray,
                           r_next: np.ndarray,
                           scores: np.ndarray,
                           dates: pd.Series,
                           cost: float) -> dict:
    if trade_mask.sum() == 0:
        return dict(trades=0, ev_per_trade=float('nan'), trades_per_day=0.0,
                    hit_rate_gross=float('nan'), hit_rate_net=float('nan'),
                    cum_log_pnl=0.0, cum_pct_pnl=0.0,
                    max_drawdown_pct=float('nan'), sharpe_daily=float('nan'),
                    exposure=0.0, annualized_return=float('nan'))
    r_net = r_next[trade_mask] - cost
    ev_per_trade = float(np.mean(r_net))
    hit_gross = float((r_next[trade_mask] > 0).mean())
    hit_net = float((r_net > 0).mean())
    days_total = len(r_next)
    trades_per_day = float(trade_mask.sum() / days_total)

    ev_contrib = np.zeros_like(r_next)
    ev_contrib[trade_mask] = r_net
    cum_log = float(np.sum(ev_contrib))
    cum_pct = float(np.exp(cum_log) - 1.0)
    mdd = max_drawdown_pct(ev_contrib)
    shrp = sharpe_daily(ev_contrib)
    exposure = float(trade_mask.mean())
    ann_ret = float((1.0 + cum_pct) ** (252.0 / days_total) - 1.0)
    return dict(
        trades=int(trade_mask.sum()),
        ev_per_trade=ev_per_trade,
        trades_per_day=trades_per_day,
        hit_rate_gross=hit_gross,
        hit_rate_net=hit_net,
        cum_log_pnl=cum_log,
        cum_pct_pnl=cum_pct,
        max_drawdown_pct=mdd,
        sharpe_daily=shrp,
        exposure=exposure,
        annualized_return=ann_ret,
    )

# %% CELL 4: Technical Indicators (lagged)

def _safe_shift(s: pd.Series, k: int = 1):
    return s.shift(k)

def _logret(c: pd.Series) -> pd.Series:
    return np.log(c / _safe_shift(c))

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_c = _safe_shift(c)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr

def atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(h, l, c)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def macd(c: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(c, fast) - ema(c, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def stoch_kd(h: pd.Series, l: pd.Series, c: pd.Series, k: int = 14, d: int = 3):
    ll = l.rolling(k, min_periods=k).min()
    hh = h.rolling(k, min_periods=k).max()
    k_fast = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    d_slow = k_fast.rolling(d, min_periods=d).mean()
    return k_fast, d_slow

def bollinger(c: pd.Series, w: int = 20, mult: float = 2.0):
    mid = sma(c, w)
    sd = c.rolling(w, min_periods=w).std(ddof=0)
    upper = mid + mult * sd
    lower = mid - mult * sd
    z = (c - mid) / (sd.replace(0, np.nan))
    return mid, upper, lower, z

def parkinson_vol(h: pd.Series, l: pd.Series, window: int = 20) -> pd.Series:
    hl = (np.log(h) - np.log(l)) ** 2
    return hl.rolling(window, min_periods=window).mean() * (1.0 / (4.0 * np.log(2)))

# %% CELL 5: Feature Builder (parse raw_ohlcv_vec, lagged features + K-day target)

def parse_ohlcv_vec(s):
    if pd.isna(s):
        return [np.nan]*5
    if isinstance(s, (list, tuple, np.ndarray)):
        v = list(s)
    else:
        try:
            s = str(s).strip().strip('[]')
            v = [float(x.strip()) for x in s.split(',')]
        except Exception:
            return [np.nan]*5
    if len(v) != 5:
        return [np.nan]*5
    return v

def build_features_targets(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, list]:
    df = df_raw.copy()
    # Force tz-aware -> tz-naive UTC for safe comparisons
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Restrict to last N years
    last_date = df['timestamp'].max().normalize()
    start_date = last_date - pd.DateOffset(years=YEARS_BACK)
    df = df.loc[df['timestamp'] >= start_date].reset_index(drop=True)

    # Parse OHLCV from vector
    ohlcv = df['raw_ohlcv_vec'].apply(parse_ohlcv_vec)
    parts = pd.DataFrame(ohlcv.tolist(), columns=['o','h','l','c','v'])
    o = parts['o'].astype(float)
    h = parts['h'].astype(float)
    l = parts['l'].astype(float)
    c = parts['c'].astype(float)
    v = parts['v'].astype(float)

    # Returns (lagged)
    ret1 = _logret(c)
    ret2 = np.log(c / _safe_shift(c, 2))
    ret3 = np.log(c / _safe_shift(c, 3))
    ret5 = np.log(c / _safe_shift(c, 5))
    ret10 = np.log(c / _safe_shift(c, 10))

    # Momentum / trend
    sma5 = sma(c, 5); sma10 = sma(c, 10); sma20 = sma(c, 20); sma50 = sma(c, 50)
    ema12 = ema(c, 12); ema26 = ema(c, 26)
    macd_line, macd_sig, macd_hist = macd(c)

    # Volatility & range
    atr14 = atr(h, l, c, 14)
    tr_pct = true_range(h, l, c) / c.replace(0, np.nan)
    rv10 = ret1.rolling(10, min_periods=10).std(ddof=0)
    rv20 = ret1.rolling(20, min_periods=20).std(ddof=0)
    par20 = parkinson_vol(h, l, 20)

    # Oscillators
    rsi14 = rsi(c, 14)
    k_fast, d_slow = stoch_kd(h, l, c, 14, 3)

    # Bollinger
    bb_mid, bb_up, bb_lo, bb_z = bollinger(c, 20, 2.0)
    bb_bw = (bb_up - bb_lo) / bb_mid.replace(0, np.nan)

    # Gaps & intraday
    prev_c = _safe_shift(c)
    gap_pct = (o - prev_c) / prev_c.replace(0, np.nan)
    intraday = (c - o) / o.replace(0, np.nan)

    # Volume transforms
    v_log = np.log1p(v)
    v_z20 = (v_log - v_log.rolling(20, min_periods=20).mean()) / (
        v_log.rolling(20, min_periods=20).std(ddof=0))

    # Seasonality
    ts = pd.to_datetime(df['timestamp'])
    dow = ts.dt.weekday
    dow_ohe = pd.get_dummies(dow, prefix='dow', dtype=float)
    month = ts.dt.month
    month_ohe = pd.get_dummies(month, prefix='m', dtype=float)

    # Regime-ish
    streak_up = (ret1 > 0).astype(int)
    up_streak_len = streak_up.groupby((streak_up != streak_up.shift()).cumsum()).cumsum()
    streak_down = (ret1 < 0).astype(int)
    down_streak_len = streak_down.groupby((streak_down != streak_down.shift()).cumsum()).cumsum()
    dist_20d_high = (c - c.rolling(20, min_periods=20).max()) / c
    dist_20d_low = (c - c.rolling(20, min_periods=20).min()) / c

    feats = pd.DataFrame({
        "ret1": ret1, "ret2": ret2, "ret3": ret3, "ret5": ret5, "ret10": ret10,
        "sma5_p": sma5 / c, "sma10_p": sma10 / c, "sma20_p": sma20 / c, "sma50_p": sma50 / c,
        "ema12_p": ema12 / c, "ema26_p": ema26 / c,
        "macd": macd_line, "macd_sig": macd_sig, "macd_hist": macd_hist,
        "atr14_p": atr14 / c,
        "tr_pct": tr_pct,
        "rv10": rv10, "rv20": rv20, "par20": par20,
        "rsi14": rsi14, "stoch_k": k_fast, "stoch_d": d_slow,
        "bb_z": bb_z, "bb_bw": bb_bw,
        "gap_pct": gap_pct, "intraday": intraday,
        "v_log": v_log, "v_z20": v_z20,
        "up_streak": up_streak_len, "down_streak": down_streak_len,
        "dist_20d_high": dist_20d_high, "dist_20d_low": dist_20d_low,
    })
    feats = pd.concat([feats, dow_ohe, month_ohe], axis=1)

    # Target: sum of next K_NEXT daily log returns
    y1 = _logret(c).shift(-1)
    ysum = y1.copy()
    for k in range(2, K_NEXT + 1):
        ysum = ysum.add(_logret(c).shift(-k), fill_value=0.0)
    y_reg = ysum

    # Drop last K_NEXT rows to avoid peeking
    feats = feats.iloc[:-K_NEXT].copy()
    y_reg = y_reg.iloc[:-K_NEXT].copy()
    dates = df['timestamp'].iloc[:-K_NEXT].copy()

    # Drop rows with NaNs from rolling windows (warm-up)
    mask = feats.notna().all(axis=1) & y_reg.notna()
    feats = feats.loc[mask].reset_index(drop=True)
    y_reg = y_reg.loc[mask].reset_index(drop=True)
    dates = pd.to_datetime(dates.loc[mask]).reset_index(drop=True)  # tz-naive

    feature_names = list(feats.columns)
    return feats, y_reg, dates, feature_names

# %% CELL 6: LOAD combined (DB) and build features/targets
engine = create_engine(DB_URL, pool_pre_ping=True)
query = text(
    """
    with combined as (
      select timestamp, raw_ohlcv_vec from backtest.spy_1d
      union all
      select timestamp, raw_ohlcv_vec from fronttest.spy_1d
    )
    select * from combined order by timestamp asc
    """
)
raw_df = pd.read_sql_query(query, engine)

X_all, y_all, dates_all, FEATURE_NAMES = build_features_targets(raw_df)
with open(FEATURES_JSON, "w") as f:
    json.dump(FEATURE_NAMES, f, indent=2)
print(f"✅ Built features: X={X_all.shape}, y={y_all.shape}")
print(f"📅 Range after trims: {dates_all.min().date()} → {dates_all.max().date()}")

# %% CELL 7: Purged CV (eval_metric='huber') + risk gate + threshold sweep
from sklearn.model_selection import TimeSeriesSplit

def purged_splits(n_samples: int, n_splits: int, gap: int):
    base = TimeSeriesSplit(n_splits=n_splits)
    for tr_idx, va_idx in base.split(np.arange(n_samples)):
        if len(va_idx) == 0:
            continue
        va_start = va_idx[0] + gap
        if va_start > va_idx[-1]:
            continue
        va_idx_adj = np.arange(va_start, va_idx[-1] + 1)
        tr_idx_adj = tr_idx[tr_idx <= (va_idx_adj[0] - gap - 1)]
        yield tr_idx_adj, va_idx_adj

# Risk gate on feature array

def apply_risk_gate_array(X: np.ndarray,
                          base_mask: np.ndarray,
                          feature_names: list,
                          gate_cfg: dict) -> np.ndarray:
    if not gate_cfg.get("use", True):
        return base_mask
    keep = base_mask.copy()
    fidx = {n: i for i, n in enumerate(feature_names)}
    if gate_cfg.get("require_macd_hist_pos", False) and "macd_hist" in fidx:
        keep &= (X[:, fidx["macd_hist"]] > 0)
    if gate_cfg.get("bb_z_min", None) is not None and "bb_z" in fidx:
        keep &= (X[:, fidx["bb_z"]] >= gate_cfg["bb_z_min"])
    if gate_cfg.get("max_gap_abs", None) is not None and "gap_pct" in fidx:
        keep &= (np.abs(X[:, fidx["gap_pct"]]) <= gate_cfg["max_gap_abs"])
    return keep

# Threshold policy on validation predictions

def policy_ev_sweep(yhat: np.ndarray,
                    X: np.ndarray,
                    r_next: np.ndarray,
                    dates: pd.Series,
                    cost: float,
                    q_low: float = THR_Q_LOW,
                    q_high: float = THR_Q_HIGH,
                    q_steps: int = THR_Q_STEPS,
                    kmin_kmax = TARGET_TRADES_PM) -> dict | None:
    qs = np.linspace(q_low, q_high, q_steps)
    grid = np.quantile(yhat, qs)
    months = np.unique(pd.to_datetime(dates).dt.to_period('M'))
    n_months = max(1, len(months))
    best: dict | None = None
    for thr in np.unique(grid):
        base = (yhat >= thr)
        base = apply_risk_gate_array(X, base, FEATURE_NAMES, GATE)
        kept = online_monthly_cap(yhat, dates, base, kmax=kmin_kmax[1])
        tpm = kept.sum() / n_months
        if tpm < kmin_kmax[0]:
            continue
        metrics = ev_metrics_from_trades(kept, r_next, yhat, dates, cost)
        score = metrics['ev_per_trade']
        if (best is None) or (np.isnan(best['ev']) and not np.isnan(score)) or (score > best['ev']):
            best = dict(theta=float(thr), ev=float(score), tpm=float(tpm), trades=int(metrics['trades']))
    return best

def cv_find_threshold(X: np.ndarray,
                      y: np.ndarray,
                      dates: pd.Series,
                      lgb_params: dict) -> tuple[float, list, int]:
    n = X.shape[0]
    fold_stats, fold_thetas, best_iters = [], [], []
    for k, (tr_idx, va_idx) in enumerate(purged_splits(n, N_SPLITS, GAP_DAYS), start=1):
        X_tr, y_tr, d_tr = X[tr_idx], y[tr_idx], dates.iloc[tr_idx]
        X_va, y_va, d_va = X[va_idx], y[va_idx], dates.iloc[va_idx]
        w_tr = compute_recency_weights_by_bars(len(y_tr), TAU_BARS)
        reg = lgb.LGBMRegressor(**lgb_params)
        reg.fit(
            X_tr, y_tr,
            sample_weight=w_tr if len(w_tr) == len(y_tr) else None,
            eval_set=[(X_va, y_va)],
            eval_metric='huber',
            callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOP_ROUNDS, verbose=False)],
        )
        best_iter = int(getattr(reg, "best_iteration_", lgb_params.get("n_estimators", 1000)))
        best_iters.append(best_iter)
        yhat_va = reg.predict(X_va, num_iteration=best_iter)
        best = policy_ev_sweep(
            yhat_va, X_va, y_va, d_va, cost=COST_ROUNDTRIP,
            q_low=THR_Q_LOW, q_high=THR_Q_HIGH, q_steps=THR_Q_STEPS,
            kmin_kmax=TARGET_TRADES_PM
        )
        fold_stats.append({
            "fold": k,
            "best_theta": None if best is None else best["theta"],
            "ev_per_trade": None if best is None else best["ev"],
            "trades": None if best is None else best["trades"],
            "tpm": None if best is None else best["tpm"],
            "best_iter": best_iter,
        })
        if best and best.get("theta") is not None:
            fold_thetas.append(best["theta"])
    if len(fold_thetas) == 0:
        raise RuntimeError("No viable threshold found in any CV fold. Loosen thresholds or gates.")
    theta_deploy = float(np.median(fold_thetas))
    best_iter_median = int(np.median(best_iters)) if len(best_iters) else 1000
    return theta_deploy, fold_stats, best_iter_median

# %% CELL 8: Final Refit (eval_metric='huber')

def compute_cutoff_for_latest_oos(dates: pd.Series, oos_months: int) -> pd.Timestamp:
    last_date = pd.to_datetime(dates.max()).normalize()
    start_oos = (last_date.to_period('M') - (oos_months - 1)).to_timestamp(how='start')
    cutoff = pd.to_datetime(start_oos) - pd.Timedelta(days=1)
    return cutoff

def refit_in_sample(X: np.ndarray,
                    y: np.ndarray,
                    dates: pd.Series,
                    best_iter_hint: int,
                    lgb_params: dict) -> tuple[lgb.LGBMRegressor, pd.Timestamp]:
    cutoff = compute_cutoff_for_latest_oos(dates, OOS_MONTHS)
    in_mask = (dates <= cutoff)
    assert in_mask.any(), "Not enough in-sample data before latest OOS window."
    X_in, y_in, d_in = X[in_mask], y[in_mask], dates[in_mask]

    # 90/10 split for ES
    cut = int(0.9 * len(X_in))
    X_tr_all, X_es = X_in[:cut], X_in[cut:]
    y_tr_all, y_es = y_in[:cut], y_in[cut:]
    w_tr = compute_recency_weights_by_bars(len(y_tr_all), TAU_BARS)

    reg = lgb.LGBMRegressor(**{**lgb_params, "n_estimators": max(1500, int(best_iter_hint * 3))})
    reg.fit(
        X_tr_all, y_tr_all,
        sample_weight=w_tr if len(w_tr) == len(y_tr_all) else None,
        eval_set=[(X_es, y_es)],
        eval_metric='huber',
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOP_ROUNDS, verbose=False)],
    )
    best_n = int(getattr(reg, "best_iteration_", reg.n_estimators))

    # Final refit on all in-sample
    w_full = compute_recency_weights_by_bars(len(y_in), TAU_BARS)
    reg_final = lgb.LGBMRegressor(**{**lgb_params, "n_estimators": max(50, best_n)})
    reg_final.fit(X_in, y_in, sample_weight=w_full if len(w_full) == len(y_in) else None)
    return reg_final, cutoff

# %% CELL 9: Walk-Forward OOS (rolling 3 months, ONLINE cap + Risk Gate)

def generate_oos_windows(dates: pd.Series, months: int = 3, years_back: int = 3):
    dts = pd.to_datetime(dates)
    last_day = dts.max().normalize()
    start_bound = (last_day - pd.DateOffset(years=years_back)).to_period('M').to_timestamp(how='start')
    starts = pd.period_range(start=start_bound.to_period('M'), end=last_day.to_period('M'), freq='M').to_timestamp(how='start')
    windows = []
    for s in starts:
        e = (s.to_period('M') + (months - 1)).to_timestamp(how='end')
        if e > last_day:
            break
        windows.append((pd.to_datetime(s), pd.to_datetime(e)))
    return windows

def train_for_cutoff(X, y, dates, cutoff, lgb_params):
    in_mask = (dates <= cutoff)
    X_in, y_in, d_in = X[in_mask], y[in_mask], dates[in_mask]
    if len(y_in) < 300:
        return None  # not enough data
    cut = int(0.9 * len(X_in))
    X_tr, X_es = X_in[:cut], X_in[cut:]
    y_tr, y_es = y_in[:cut], y_in[cut:]
    w_tr = compute_recency_weights_by_bars(len(y_tr), TAU_BARS)

    reg = lgb.LGBMRegressor(**lgb_params)
    reg.fit(
        X_tr, y_tr,
        sample_weight=w_tr if len(w_tr) == len(y_tr) else None,
        eval_set=[(X_es, y_es)],
        eval_metric='huber',
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOP_ROUNDS, verbose=False)],
    )
    best_n = int(getattr(reg, "best_iteration_", reg.n_estimators))

    reg_final = lgb.LGBMRegressor(**{**lgb_params, "n_estimators": max(50, best_n)})
    w_full = compute_recency_weights_by_bars(len(y_in), TAU_BARS)
    reg_final.fit(X_in, y_in, sample_weight=w_full if len(w_full) == len(y_in) else None)
    return reg_final

def oos_roll_eval(X: np.ndarray,
                  y: np.ndarray,
                  dates: pd.Series,
                  theta_deploy: float) -> tuple[list, pd.DataFrame | None]:
    windows = generate_oos_windows(dates, months=OOS_MONTHS, years_back=WFO_YEARS_BACK)
    all_stats: list = []
    latest_preds_df: pd.DataFrame | None = None
    for (start_oos, end_oos) in tqdm(windows, desc="Rolling OOS windows"):
        cutoff = start_oos - pd.Timedelta(days=1)
        reg = train_for_cutoff(X, y, dates, cutoff, LGB_PARAMS)
        if reg is None:
            continue
        oos_mask = (dates >= start_oos) & (dates <= end_oos)
        if not oos_mask.any():
            continue
        X_oos = X[oos_mask]
        d_oos = dates[oos_mask]
        r_oos = y[oos_mask]
        yhat = reg.predict(X_oos)
        base = (yhat >= theta_deploy)
        gate = apply_risk_gate_array(X_oos, base, FEATURE_NAMES, GATE)
        kept = online_monthly_cap(yhat, d_oos, gate, kmax=TARGET_TRADES_PM[1])
        m = ev_metrics_from_trades(kept, r_oos, yhat, d_oos, cost=COST_ROUNDTRIP)
        m.update(dict(oos_start=str(start_oos.date()), oos_end=str(end_oos.date())))
        all_stats.append(m)
        latest_preds_df = pd.DataFrame({
            "date": d_oos.astype(str).values,
            "yhat": yhat,
            "trade": kept.astype(int),
            "ret_next": r_oos,
            "ev_contrib": kept * (r_oos - COST_ROUNDTRIP),
        })
    return all_stats, latest_preds_df

# %% CELL 10: MAIN — Train/CV, deploy θ, final refit, WFO OOS, save artifacts
X_np = X_all.values.astype(float)
y_np = y_all.values.astype(float)
d_ser = dates_all.copy()

# 1) Purged CV to get deploy threshold & iteration hint
theta_deploy, fold_stats, best_iter_median = cv_find_threshold(X_np, y_np, d_ser, LGB_PARAMS)
print(f"θ_deploy (median across folds): {theta_deploy:.6f} | best_iter_median: {best_iter_median}")

# 2) Final refit on in-sample (<= latest - OOS_MONTHS)
reg_final, cutoff = refit_in_sample(X_np, y_np, d_ser, best_iter_median, LGB_PARAMS)

# 3) Rolling OOS evaluation (3-month windows across last WFO_YEARS_BACK)
oos_stats, latest_df = oos_roll_eval(X_np, y_np, d_ser, theta_deploy)

# 4) Save artifacts
joblib.dump(reg_final, MODEL_PATH)
joblib.dump(IdentityTransformer(), SCALER_PATH)  # identity to keep interface consistent
if latest_df is not None:
    latest_df.to_csv(OOS_PRED_CSV, index=False)
else:
    pd.DataFrame(columns=["date","yhat","trade","ret_next","ev_contrib"]).to_csv(OOS_PRED_CSV, index=False)

with open(THRESHOLD_JSON, "w") as f:
    json.dump({
        "theta": float(theta_deploy),
        "cost_bps": int(round(COST_ROUNDTRIP * 1e4)),  # bps
        "fold_stats": fold_stats,
        "theta_selection": "median_of_fold_bests",
        "target_trades_per_month": TARGET_TRADES_PM,
    }, f, indent=2)

# Aggregate OOS stats

def _agg(stats: list, key: str, fn) -> float:
    vals = [s[key] for s in stats if (s.get(key) is not None and not np.isnan(s.get(key)))]
    return float(fn(vals)) if len(vals) else float('nan')

oos_summary = dict(
    windows=len(oos_stats),
    ev_per_trade_median=_agg(oos_stats, "ev_per_trade", np.median),
    ev_per_trade_mean=_agg(oos_stats, "ev_per_trade", np.mean),
    trades_median=_agg(oos_stats, "trades", np.median),
    trades_mean=_agg(oos_stats, "trades", np.mean),
    trades_per_day_median=_agg(oos_stats, "trades_per_day", np.median),
    exposure_median=_agg(oos_stats, "exposure", np.median),
    sharpe_daily_median=_agg(oos_stats, "sharpe_daily", np.median),
    max_dd_median=_agg(oos_stats, "max_drawdown_pct", np.median),
    cum_pct_median=_agg(oos_stats, "cum_pct_pnl", np.median),
)

with open(METRICS_JSON, "w") as f:
    json.dump({
        "in_sample": {
            "folds": fold_stats,
            "theta_deploy": float(theta_deploy),
            "best_iter_median": int(best_iter_median),
            "cutoff_date": str(cutoff.date()),
        },
        "oos": {
            "windows_stats": oos_stats,
            "summary": oos_summary
        }
    }, f, indent=2)

manifest = {
    "model_type": "LightGBMRegressor",
    "random_state": RANDOM_STATE,
    "cost_roundtrip": COST_ROUNDTRIP,
    "tau_bars": TAU_BARS,
    "threshold_quantile_grid": [THR_Q_LOW, THR_Q_HIGH, THR_Q_STEPS],
    "target_trades_per_month": TARGET_TRADES_PM,
    "oos_months": OOS_MONTHS,
    "wfo_years_back": WFO_YEARS_BACK,
    "feature_names": FEATURE_NAMES,
    "artifact_paths": {
        "model": str(MODEL_PATH),
        "scaler": str(SCALER_PATH),
        "threshold_json": str(THRESHOLD_JSON),
        "metrics_json": str(METRICS_JSON),
        "oos_preds_csv_latest": str(OOS_PRED_CSV),
        "feature_names_json": str(FEATURES_JSON),
    },
    "lgbm_params": LGB_PARAMS,
    "execution_convention": f"enter close_t, exit close_t+{K_NEXT} (log returns, regression target)",
}
with open(MANIFEST_JSON, "w") as f:
    json.dump(manifest, f, indent=2)

print("\n=== LGBM 1D v0 (Regression + Policy) — COMPLETE ===")
print(f"θ_deploy: {theta_deploy:.4f} | in-sample cutoff: {cutoff.date()}")
print("OOS Summary:")
for k, v in oos_summary.items():
    print(f" - {k}: {v}")
print("\nArtifacts saved:")
print(f"- Model: {MODEL_PATH}")
print(f"- Scaler: {SCALER_PATH}")
print(f"- Thr: {THRESHOLD_JSON}")
print(f"- Metrics: {METRICS_JSON}")
print(f"- OOS CSV: {OOS_PRED_CSV}")

# Save code snapshot
try:
    code_txt = Path(__file__).read_text(encoding='utf-8')
    (RUN_DIR / f"code_{Path(__file__).name}.txt").write_text(code_txt, encoding='utf-8')
except Exception:
    pass


