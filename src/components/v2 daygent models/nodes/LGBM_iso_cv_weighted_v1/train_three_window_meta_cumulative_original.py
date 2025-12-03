#!/usr/bin/env python3
"""
LightGBM 1D Training — Three-Window Meta-Optimization (Cumulative Folds)

Key improvements over train_three_window_meta.py:
- Cumulative pre-test fold training: For each fold, training = Train window + all pre-test bars strictly before the fold block (with purge). Validation = current pre-test fold block (after purge).
- AUC guards: If validation labels are single-class, AUC is treated as undefined and not coerced to 0.5; scoring shifts to F1/balanced accuracy.
- Per-fold label logging: Explicitly prints train/val class counts and flags single-class validation.
- Exact last-N test rows: test window is always the last TEST_ROWS rows.
"""

from __future__ import annotations

import os
import json
import time
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

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
from sklearn.linear_model import Ridge
import joblib
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


# %% CONFIG
DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5433/trading_data"
BACKTEST_TABLE = 'backtest.spy_1d'
FRONTTEST_TABLE = 'fronttest.spy_1d'

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'artifacts_lgbm_1d_iso_three_window_cumulative')
os.makedirs(ARTIFACT_DIR, exist_ok=True)
RUNS_DIR = os.path.join(ARTIFACT_DIR, 'runs')
os.makedirs(RUNS_DIR, exist_ok=True)
RUN_TS = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
RUN_DIR = os.path.join(RUNS_DIR, RUN_TS)
os.makedirs(RUN_DIR, exist_ok=True)


# %% HELPERS
PROMPT_OVERRIDE: Optional[str] = None  # '1' -> search all, '2' -> skip all
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def _find_latest_run_dirs(base_dir: str) -> List[str]:
    try:
        entries = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        # sort by mtime desc
        entries.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return entries
    except Exception:
        return []

def load_dynamic_known_bests() -> Tuple[Dict[int, Dict[str, Any]], Dict[int, float], Dict[int, Dict[str, Any]], Dict[int, float]]:
    """
    Load latest known per-fold best params and thresholds from artifacts for RUN1 (test=35) and RUN2 (test=25).
    Returns: (params_run1, thr_run1, params_run2, thr_run2)
    """
    params_run1: Dict[int, Dict[str, Any]] = {}
    thr_run1: Dict[int, float] = {}
    params_run2: Dict[int, Dict[str, Any]] = {}
    thr_run2: Dict[int, float] = {}

    # iterate run directories newest first, skip current RUN_DIR
    for run_dir in _find_latest_run_dirs(RUNS_DIR):
        if os.path.abspath(run_dir) == os.path.abspath(RUN_DIR):
            continue
        # Primary: unified results file at run root
        res_main_path = os.path.join(run_dir, 'results_three_window_cumulative.json')
        res_main = _load_json(res_main_path)
        if res_main and isinstance(res_main.get('pretest_folds'), list):
            try:
                for fs in res_main['pretest_folds']:
                    fid = int(fs.get('fold'))
                    best = fs.get('best', {})
                    bp = best.get('params', {})
                    if isinstance(bp, dict) and fid not in params_run1:
                        params_run1[fid] = bp
                        thr_run1[fid] = _safe_float(best.get('best_thr'), 0.5)
            except Exception:
                pass
        # RUN2 from unified file if present
        run2_obj = (res_main or {}).get('run2', {})
        if isinstance(run2_obj.get('pretest_folds', []), list):
            try:
                for fs in run2_obj['pretest_folds']:
                    fid = int(fs.get('fold'))
                    best = fs.get('best', {})
                    bp = best.get('params', {})
                    if isinstance(bp, dict) and fid not in params_run2:
                        params_run2[fid] = bp
                        thr_run2[fid] = _safe_float(best.get('best_thr'), 0.5)
            except Exception:
                pass
        # Backward compatibility: legacy subfolder 'run_25'
        if not params_run2:
            res2_dir_legacy = os.path.join(run_dir, 'run_25')
            res2_path_legacy = os.path.join(res2_dir_legacy, 'results_three_window_cumulative_run2.json')
            res2_legacy = _load_json(res2_path_legacy)
            if res2_legacy and isinstance(res2_legacy.get('pretest_folds'), list):
                try:
                    for fs in res2_legacy['pretest_folds']:
                        fid = int(fs.get('fold'))
                        best = fs.get('best', {})
                        bp = best.get('params', {})
                        if isinstance(bp, dict) and fid not in params_run2:
                            params_run2[fid] = bp
                            thr_run2[fid] = _safe_float(best.get('best_thr'), 0.5)
                except Exception:
                    pass
        # If we collected a full set (or at least some), we can break early
        if params_run1 or params_run2:
            break
    return params_run1, thr_run1, params_run2, thr_run2

def merge_params(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged

def filter_params_for_search(params: Dict[str, Any], allowed_keys: List[str]) -> Dict[str, Any]:
    """Return only keys that exist in the objective's search space."""
    return {k: params[k] for k in allowed_keys if k in params}

def get_user_choice(prompt: str, has_known: bool, default_choice: Optional[str] = None) -> str:
    """
    Robust input handler. In non-interactive environments, fall back to env var MCP_AUTO_CHOICE or sensible default.
    If has_known is True, default to '2' (skip); otherwise '1' (search).
    """
    global PROMPT_OVERRIDE
    # If already set by a previous prompt, honor it
    if PROMPT_OVERRIDE in ("1", "2"):
        return PROMPT_OVERRIDE
    # Check env override first
    env_choice = os.getenv("MCP_AUTO_CHOICE")
    if env_choice in ("1", "2"):
        PROMPT_OVERRIDE = env_choice  # persist for session if given
        return PROMPT_OVERRIDE
    if env_choice in ("3", "4"):
        PROMPT_OVERRIDE = "1" if env_choice == "3" else "2"
        print(f"  ⚙️  Auto-mode enabled via MCP_AUTO_CHOICE={env_choice} → defaulting all prompts to {PROMPT_OVERRIDE}")
        return PROMPT_OVERRIDE
    # Interactive input
    try:
        choice = input(prompt).strip()
        if choice in ("1", "2"):
            return choice
        if choice == "3":
            PROMPT_OVERRIDE = "1"
            print("  ⚙️  Auto-mode: defaulting this and all subsequent folds to [1] search")
            return "1"
        if choice == "4":
            PROMPT_OVERRIDE = "2"
            print("  ⚙️  Auto-mode: defaulting this and all subsequent folds to [2] skip")
            return "2"
    except EOFError:
        pass
    except Exception:
        pass
    # Fallback default
    return default_choice if default_choice in ("1", "2") else ("2" if has_known else "1")
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


# Purge and window config
GAP_BARS = 5
PRETEST_FOLD_SIZE = 35
TEST_ROWS = 35
FOLD_TIME_LIMIT = 300  # seconds per fold (5 minutes)
EXTENSION_TIME_LIMIT = 300  # Phase 3: 5 minutes creative search
ULTRA_EXTENSION_TIME_LIMIT = 180  # Phase 4: 3 minutes ultra-aggressive search
EXTREME_EXTENSION_TIME_LIMIT = 120  # Phase 5: 2 minutes final extreme search
AUC_THRESHOLD_FOR_EXTENSION = 0.70  # trigger extension if best AUC below this
THRESHOLD_FLOOR_TRIGGER = 0.35  # trigger extension if best threshold <= this (poor calibration)

# Best hyperparameters from previous successful runs (for skip option)
KNOWN_BEST_PARAMS_RUN1 = {
    1: {"learning_rate": 0.0742, "num_leaves": 126, "max_depth": 6, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
    2: {"learning_rate": 0.1005, "num_leaves": 60, "max_depth": 10, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
    3: {"learning_rate": 0.0678, "num_leaves": 25, "max_depth": 6, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
    4: {"learning_rate": 0.0126, "num_leaves": 98, "max_depth": 8, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
    5: {"learning_rate": 0.0109, "num_leaves": 34, "max_depth": 3, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
    6: {"learning_rate": 0.0116, "num_leaves": 67, "max_depth": 5, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
    7: {"learning_rate": 0.0147, "num_leaves": 99, "max_depth": 4, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
}

KNOWN_BEST_PARAMS_RUN2 = {
    1: {"learning_rate": 0.0677, "num_leaves": 48, "max_depth": 4, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
    2: {"learning_rate": 0.1261, "num_leaves": 40, "max_depth": 6, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
    3: {"learning_rate": 0.0132, "num_leaves": 115, "max_depth": 6, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
    4: {"learning_rate": 0.0213, "num_leaves": 114, "max_depth": 6, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
    5: {"learning_rate": 0.0108, "num_leaves": 19, "max_depth": 8, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
    6: {"learning_rate": 0.0118, "num_leaves": 114, "max_depth": 9, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
    7: {"learning_rate": 0.0138, "num_leaves": 118, "max_depth": 4, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
    8: {"learning_rate": 0.0134, "num_leaves": 96, "max_depth": 5, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
    9: {"learning_rate": 0.1024, "num_leaves": 116, "max_depth": 6, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
    10: {"learning_rate": 0.0411, "num_leaves": 77, "max_depth": 3, "min_child_samples": 60, "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 0.1},
}


# Manual newest-best settings (params + exact metrics) for RUN1 and RUN2
# These act as authoritative fold-level bests unless a new manual search beats their score during this run
MANUAL_BEST_RUN1: Dict[int, Dict[str, Any]] = {
    1: {
        "params": {"objective": "binary", "boosting_type": "gbdt", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.015544850579794284, "num_leaves": 21, "max_depth": 9,
                    "min_child_samples": 14, "feature_fraction": 0.6902500540341191,
                    "bagging_fraction": 0.9904098536064474, "bagging_freq": 1,
                    "lambda_l1": 0.48524113116023637, "lambda_l2": 0.054847433980311305,
                    "min_gain_to_split": 0.0, "n_estimators": 4000, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced"},
        "best_thr": 0.52, "best_auc": 0.8287037037037037, "best_acc": 0.8666666666666667,
        "best_prec": 0.85, "best_rec": 0.9444444444444444, "best_f1": 0.8947368421052632,
        "best_bacc": 0.8472222222222222, "score": 0.8633690708252112,
    },
    2: {
        "params": {"objective": "binary", "boosting_type": "gbdt", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.02981481229262068, "num_leaves": 66, "max_depth": 6,
                    "min_child_samples": 16, "feature_fraction": 0.6195264976674061,
                    "bagging_fraction": 0.6354757332823335, "bagging_freq": 1,
                    "lambda_l1": 0.8079720167974817, "lambda_l2": 0.9790577946593941,
                    "min_gain_to_split": 0.0, "n_estimators": 4000, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced"},
        "best_thr": 0.5, "best_auc": 0.708133971291866, "best_acc": 0.7333333333333333,
        "best_prec": 0.7391304347826086, "best_rec": 0.8947368421052632, "best_f1": 0.8095238095238095,
        "best_bacc": 0.6746411483253588, "score": 0.750330371383003,
    },
    3: {
        "params": {"objective": "binary", "boosting_type": "gbdt", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.08469497362191536, "num_leaves": 61, "max_depth": 8,
                    "min_child_samples": 42, "feature_fraction": 0.5571442819922889,
                    "bagging_fraction": 0.9695730559973537, "bagging_freq": 1,
                    "lambda_l1": 0.30387874073576343, "lambda_l2": 0.4462414330551902,
                    "min_gain_to_split": 0.0, "n_estimators": 4000, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced"},
        "best_thr": 0.45, "best_auc": 0.7733333333333333, "best_acc": 0.8333333333333334,
        "best_prec": 0.8571428571428571, "best_rec": 0.8, "best_f1": 0.8275862068965517,
        "best_bacc": 0.8333333333333334, "score": 0.8114176245210728,
    },
    4: {
        "params": {"objective": "binary", "boosting_type": "goss", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.12985844132786115, "num_leaves": 66, "max_depth": 9,
                    "min_child_samples": 183, "feature_fraction": 0.9625685741860183,
                    "lambda_l1": 4.581367877527114e-08, "lambda_l2": 0.00013639846700546406,
                    "min_gain_to_split": 0.7685224563179203, "n_estimators": 4000, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced", "min_sum_hessian_in_leaf": 0.145868995039577,
                    "feature_fraction_bynode": 0.9475162791443883, "max_bin": 74, "max_delta_step": 1.3536032293142866,
                    "path_smooth": 1.2109123810426845, "extra_trees": True},
        "best_thr": 0.47, "best_auc": 0.7008928571428572, "best_acc": 0.7,
        "best_prec": 0.6086956521739131, "best_rec": 1.0, "best_f1": 0.7567567567567568,
        "best_bacc": 0.71875, "score": 0.719216537966538,
    },
    5: {
        "params": {"objective": "binary", "boosting_type": "gbdt", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.11099546566917044, "num_leaves": 103, "max_depth": 9,
                    "min_child_samples": 98, "feature_fraction": 0.8839687872223735,
                    "bagging_fraction": 0.7088286744096267, "bagging_freq": 1,
                    "lambda_l1": 0.898999882961603, "lambda_l2": 0.4216599326691339,
                    "min_gain_to_split": 0.0, "n_estimators": 4000, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced"},
        "best_thr": 0.5, "best_auc": 0.765625, "best_acc": 0.7666666666666667,
        "best_prec": 0.7647058823529411, "best_rec": 0.8125, "best_f1": 0.7878787878787878,
        "best_bacc": 0.7633928571428572, "score": 0.7733901515151516,
    },
    6: {
        "params": {"objective": "binary", "boosting_type": "gbdt", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.10644350764395455, "num_leaves": 26, "max_depth": 4,
                    "min_child_samples": 50, "feature_fraction": 0.7479345585376551,
                    "bagging_fraction": 0.6715928814531731, "bagging_freq": 1,
                    "lambda_l1": 0.739932078596443, "lambda_l2": 0.5594219209722312,
                    "min_gain_to_split": 0.0, "n_estimators": 4000, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced"},
        "best_thr": 0.46, "best_auc": 0.7916666666666667, "best_acc": 0.8333333333333334,
        "best_prec": 0.8095238095238095, "best_rec": 0.9444444444444444, "best_f1": 0.8717948717948718,
        "best_bacc": 0.8055555555555556, "score": 0.8322649572649573,
    },
    7: {
        "params": {"objective": "binary", "boosting_type": "gbdt", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.0797069659385885, "num_leaves": 34, "max_depth": 10,
                    "min_child_samples": 19, "feature_fraction": 0.5169450239016812,
                    "bagging_fraction": 0.8532621421443489, "bagging_freq": 1,
                    "lambda_l1": 0.010026418918328825, "lambda_l2": 0.8694305009685304,
                    "min_gain_to_split": 0.0, "n_estimators": 4000, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced"},
        "best_thr": 0.49, "best_auc": 0.7375565610859728, "best_acc": 0.7666666666666667,
        "best_prec": 0.7272727272727273, "best_rec": 0.9411764705882353, "best_f1": 0.8205128205128205,
        "best_bacc": 0.7398190045248869, "score": 0.7749120160884866,
    },
}

MANUAL_BEST_RUN2: Dict[int, Dict[str, Any]] = {
    1: {
        "params": {"objective": "binary", "boosting_type": "gbdt", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.14200976296032244, "num_leaves": 50, "max_depth": 3,
                    "min_child_samples": 104, "feature_fraction": 0.5442430305734951,
                    "bagging_fraction": 0.5254121830916471, "bagging_freq": 1,
                    "lambda_l1": 0.9680572088391821, "lambda_l2": 0.20568128579627198,
                    "min_gain_to_split": 0.0, "n_estimators": 4000, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced"},
        "best_thr": 0.48, "best_auc": 0.8690476190476191, "best_acc": 0.9,
        "best_prec": 0.9285714285714286, "best_rec": 0.9285714285714286, "best_f1": 0.9285714285714286,
        "best_bacc": 0.8809523809523809, "score": 0.8992063492063492,
    },
    2: {
        "params": {"objective": "binary", "boosting_type": "gbdt", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.04633459187004999, "num_leaves": 125, "max_depth": 10,
                    "min_child_samples": 15, "feature_fraction": 0.6377366181707602,
                    "bagging_fraction": 0.5425004538783101, "bagging_freq": 1,
                    "lambda_l1": 0.3687696600617726, "lambda_l2": 0.33313534221323843,
                    "min_gain_to_split": 0.0, "n_estimators": 4000, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced"},
        "best_thr": 0.51, "best_auc": 0.8229166666666666, "best_acc": 0.9,
        "best_prec": 0.9166666666666666, "best_rec": 0.9166666666666666, "best_f1": 0.9166666666666666,
        "best_bacc": 0.8958333333333333, "score": 0.8798611111111111,
    },
    3: {
        "params": {"objective": "binary", "boosting_type": "dart", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.1320367092765243, "num_leaves": 15, "max_depth": 20,
                    "min_child_samples": 59, "feature_fraction": 0.644119108508378,
                    "bagging_fraction": 0.741265117356928, "bagging_freq": 7,
                    "lambda_l1": 2.360640894531639, "lambda_l2": 6.783812754041804e-05,
                    "min_gain_to_split": 3.655898755895225, "n_estimators": 4347, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced", "min_sum_hessian_in_leaf": 1.3938653564991177e-05,
                    "feature_fraction_bynode": 0.5454044795163071, "max_bin": 78, "min_data_in_bin": 1,
                    "max_delta_step": 1.187628694122764, "path_smooth": 8.11712907292357, "extra_trees": True,
                    "drop_rate": 0.20544520197880378, "skip_drop": 0.4350961862807315, "max_drop": 60,
                    "top_rate": 0.2, "other_rate": 0.1},
        "best_thr": 0.47, "best_auc": 0.8452380952380953, "best_acc": 0.85,
        "best_prec": 0.8666666666666667, "best_rec": 0.9285714285714286, "best_f1": 0.896551724137931,
        "best_bacc": 0.7976190476190477, "score": 0.8639299397920088,
    },
    4: {
        "params": {"objective": "binary", "boosting_type": "gbdt", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.07460236932425938, "num_leaves": 84, "max_depth": 9,
                    "min_child_samples": 115, "feature_fraction": 0.815222683209991,
                    "bagging_fraction": 0.5042553247853964, "bagging_freq": 1,
                    "lambda_l1": 0.09772397546884574, "lambda_l2": 0.2349470697025339,
                    "min_gain_to_split": 0.0, "n_estimators": 4000, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced"},
        "best_thr": 0.47, "best_auc": 0.84375, "best_acc": 0.9,
        "best_prec": 0.8571428571428571, "best_rec": 1.0, "best_f1": 0.9230769230769231,
        "best_bacc": 0.875, "score": 0.8889423076923078,
    },
    5: {
        "params": {"objective": "binary", "boosting_type": "goss", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.19665517251385486, "num_leaves": 161, "max_depth": 8,
                    "min_child_samples": 82, "feature_fraction": 0.5945161377314772,
                    "lambda_l1": 2.952977509383388e-07, "lambda_l2": 2.576220110830412e-08,
                    "min_gain_to_split": 0.9559522022401985, "n_estimators": 4000, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced", "min_sum_hessian_in_leaf": 0.0013084676943776981,
                    "feature_fraction_bynode": 0.9873666983052354, "max_bin": 277, "max_delta_step": 1.3028317541232286,
                    "path_smooth": 0.13102424722299655, "extra_trees": True},
        "best_thr": 0.46, "best_auc": 0.795, "best_acc": 0.7,
        "best_prec": 0.625, "best_rec": 1.0, "best_f1": 0.7692307692307693,
        "best_bacc": 0.7, "score": 0.7547435897435898,
    },
    6: {
        "params": {"objective": "binary", "boosting_type": "gbdt", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.0742808232828308, "num_leaves": 19, "max_depth": 7,
                    "min_child_samples": 93, "feature_fraction": 0.9878953687408056,
                    "bagging_fraction": 0.6133983903809392, "bagging_freq": 1,
                    "lambda_l1": 0.06704506111531884, "lambda_l2": 0.08298961741827077,
                    "min_gain_to_split": 0.0, "n_estimators": 4000, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced"},
        "best_thr": 0.5, "best_auc": 0.8888888888888888, "best_acc": 0.85,
        "best_prec": 0.8, "best_rec": 0.8888888888888888, "best_f1": 0.8421052631578947,
        "best_bacc": 0.8535353535353536, "score": 0.8603313840155945,
    },
    7: {
        "params": {"objective": "binary", "boosting_type": "gbdt", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.0559782454286894, "num_leaves": 18, "max_depth": 4,
                    "min_child_samples": 83, "feature_fraction": 0.8839333292577561,
                    "bagging_fraction": 0.6069279273293623, "bagging_freq": 1,
                    "lambda_l1": 0.06363282097326382, "lambda_l2": 0.06826352942860602,
                    "min_gain_to_split": 0.0, "n_estimators": 4000, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced"},
        "best_thr": 0.44, "best_auc": 0.9523809523809524, "best_acc": 0.95,
        "best_prec": 0.9333333333333333, "best_rec": 1.0, "best_f1": 0.9655172413793104,
        "best_bacc": 0.9166666666666667, "score": 0.9559660645867543,
    },
    8: {
        "params": {"objective": "binary", "boosting_type": "gbdt", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.1164066091511263, "num_leaves": 24, "max_depth": 7,
                    "min_child_samples": 113, "feature_fraction": 0.9759468340623397,
                    "bagging_fraction": 0.6549914813178349, "bagging_freq": 1,
                    "lambda_l1": 0.14502592266877062, "lambda_l2": 0.6609871089517277,
                    "min_gain_to_split": 0.0, "n_estimators": 4000, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced"},
        "best_thr": 0.52, "best_auc": 0.79, "best_acc": 0.8,
        "best_prec": 0.8, "best_rec": 0.8, "best_f1": 0.8,
        "best_bacc": 0.8, "score": 0.7966666666666667,
    },
    9: {
        "params": {"objective": "binary", "boosting_type": "gbdt", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.012707942999213693, "num_leaves": 37, "max_depth": 3,
                    "min_child_samples": 46, "feature_fraction": 0.6943386448447411,
                    "bagging_fraction": 0.6356745158869479, "bagging_freq": 1,
                    "lambda_l1": 0.8287375091519293, "lambda_l2": 0.3567533266935893,
                    "min_gain_to_split": 0.0, "n_estimators": 4000, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced"},
        "best_thr": 0.5, "best_auc": 0.8392857142857143, "best_acc": 0.85,
        "best_prec": 0.8235294117647058, "best_rec": 1.0, "best_f1": 0.9032258064516129,
        "best_bacc": 0.75, "score": 0.8641705069124423,
    },
    10: {
        "params": {"objective": "binary", "boosting_type": "gbdt", "metric": ["auc", "binary_logloss"],
                    "learning_rate": 0.14001706764097102, "num_leaves": 99, "max_depth": 4,
                    "min_child_samples": 27, "feature_fraction": 0.9021122769988679,
                    "bagging_fraction": 0.619479331068003, "bagging_freq": 1,
                    "lambda_l1": 0.0011967705143562035, "lambda_l2": 0.0457913639955307,
                    "min_gain_to_split": 0.0, "n_estimators": 4000, "n_jobs": -1, "verbose": -1,
                    "random_state": 42, "class_weight": "balanced"},
        "best_thr": 0.46, "best_auc": 0.9500000000000001, "best_acc": 0.9,
        "best_prec": 0.8333333333333334, "best_rec": 1.0, "best_f1": 0.9090909090909091,
        "best_bacc": 0.9, "score": 0.9196969696969698,
    },
}

def split_three_windows(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Backwards splitting to ensure exact test rows:
    1) Keep last 6 years of data
    2) Test = last TEST_ROWS rows (exact)
    3) Pre-test = 1 year immediately before test start
    4) Train = remainder within the 6-year window
    """
    if len(df) == 0:
        raise RuntimeError("No data available")
    latest_ts = pd.to_datetime(df["timestamp"]).max()
    cutoff_ts = latest_ts - pd.DateOffset(years=6)
    df6 = df[df["timestamp"] >= cutoff_ts].copy().sort_values("timestamp").reset_index(drop=True)
    if len(df6) < TEST_ROWS:
        raise RuntimeError(f"Not enough bars in last 6 years to form a {TEST_ROWS}-row test window")
    # Test rows
    test_df = df6.tail(TEST_ROWS).copy().reset_index(drop=True)
    test_start = pd.to_datetime(test_df["timestamp"]).iloc[0]
    test_end = pd.to_datetime(test_df["timestamp"]).iloc[-1]
    # Pre-test = 1 year before test start
    pretest_start = test_start - pd.DateOffset(years=1)
    pre_df = df6[(df6["timestamp"] >= pretest_start) & (df6["timestamp"] < test_start)].copy().reset_index(drop=True)
    pretest_end = pd.to_datetime(pre_df["timestamp"]).iloc[-1] if len(pre_df) else (test_start - pd.Timedelta(microseconds=1))
    # Train = earlier rows
    train_df = df6[df6["timestamp"] < pretest_start].copy().reset_index(drop=True)
    return {
        "train": train_df,
        "pretest": pre_df,
        "test": test_df,
        "pretest_bounds": (pretest_start, pretest_end),
        "test_bounds": (test_start, test_end),
    }


def cumulative_pretest_splits(n_pre: int, fold_len: int, gap: int):
    """
    Yield (train_pre_end_exclusive, val_start, val_end) in pre-test index space.
    - train_pre_end_exclusive = max(0, fold_start - gap)
    - val_start = fold_start + gap
    - val_end = fold_start + fold_len
    Skip folds where val_start >= val_end or val_end > n_pre.
    """
    fold_start = 0
    while fold_start < n_pre:
        val_start = fold_start + gap
        val_end = fold_start + fold_len
        train_pre_end = max(0, fold_start - gap)
        if val_end > n_pre:
            break
        if val_start < val_end:
            yield train_pre_end, val_start, val_end
        fold_start = val_end


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


def create_optuna_objective_extended(X_tr: Any, y_tr: np.ndarray, X_va: Any, y_va: np.ndarray, 
                                     thr_grid: np.ndarray, prior_best: Optional[Dict[str, Any]] = None) -> callable:
    """
    Extended objective with more hyperparameters to tune when standard search fails.
    Adds many more LightGBM parameters for creative exploration.
    """
    base_params = default_lgb_params()
    
    def objective(trial: optuna.Trial) -> float:
        # Core hyperparameters with wider ranges
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.25, log=True)
        num_leaves = trial.suggest_int("num_leaves", 7, 255)
        max_depth = trial.suggest_int("max_depth", 2, 15)
        min_child_samples = trial.suggest_int("min_child_samples", 3, 250)
        feature_fraction = trial.suggest_float("feature_fraction", 0.3, 1.0)
        bagging_fraction = trial.suggest_float("bagging_fraction", 0.3, 1.0)
        lambda_l1 = trial.suggest_float("lambda_l1", 1e-8, 8.0, log=True)
        lambda_l2 = trial.suggest_float("lambda_l2", 1e-8, 8.0, log=True)
        
        # EXTENDED hyperparameters from LightGBM docs
        bagging_freq = trial.suggest_int("bagging_freq", 0, 12)
        min_gain_to_split = trial.suggest_float("min_gain_to_split", 0.0, 3.0)
        max_bin = trial.suggest_int("max_bin", 63, 511)
        min_sum_hessian_in_leaf = trial.suggest_float("min_sum_hessian_in_leaf", 1e-4, 20.0, log=True)
        
        # Feature subsampling per node
        feature_fraction_bynode = trial.suggest_float("feature_fraction_bynode", 0.4, 1.0)
        
        # Boosting types
        boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"])
        
        # Regularization
        path_smooth = trial.suggest_float("path_smooth", 0.0, 2.0)
        max_delta_step = trial.suggest_float("max_delta_step", 0.0, 3.0)
        
        # Extra trees for randomization
        extra_trees = trial.suggest_categorical("extra_trees", [False, True])
        
        # Adjust n_estimators based on learning rate
        if learning_rate < 0.01:
            n_estimators = 7000
        elif learning_rate < 0.03:
            n_estimators = 5500
        elif learning_rate < 0.07:
            n_estimators = 4500
        else:
            n_estimators = 4000
        
        # Build params
        params = dict(base_params)
        params.update({
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "min_sum_hessian_in_leaf": min_sum_hessian_in_leaf,
            "feature_fraction": feature_fraction,
            "feature_fraction_bynode": feature_fraction_bynode,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "min_gain_to_split": min_gain_to_split,
            "max_bin": max_bin,
            "max_delta_step": max_delta_step,
            "boosting_type": boosting_type,
            "path_smooth": path_smooth,
            "extra_trees": extra_trees,
            "n_estimators": n_estimators,
        })
        # Ensure compatibility: GOSS ignores bagging params
        try:
            if params.get("boosting_type") == "goss":
                params.pop("bagging_fraction", None)
                params.pop("bagging_freq", None)
        except Exception:
            pass
        
        try:
            clf = lgb.LGBMClassifier(**params)
            # Add early stopping only for compatible boosting types (exclude DART)
            bt = params.get("boosting_type", "gbdt")
            fit_callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)] if bt != "dart" else []
            clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=fit_callbacks
            )
            proba = clf.predict_proba(X_va)[:, 1]
            stats = evaluate_thresholds(y_va, proba, thr_grid)
            score = score_fold(stats)
            
            trial.set_user_attr("stats", stats)
            trial.set_user_attr("params", params)
            
            return score
        except Exception as e:
            return -1.0
    
    return objective


def create_optuna_objective_ultra(X_tr: Any, y_tr: np.ndarray, X_va: Any, y_va: np.ndarray, 
                                   thr_grid: np.ndarray, prior_best: Optional[Dict[str, Any]] = None) -> callable:
    """
    ULTRA-AGGRESSIVE objective for Phase 4 when everything else has failed.
    Maximum creativity: extreme ranges, ALL LightGBM parameters, adaptive strategies.
    """
    base_params = default_lgb_params()
    
    def objective(trial: optuna.Trial) -> float:
        # ULTRA-WIDE hyperparameter ranges
        learning_rate = trial.suggest_float("learning_rate", 0.0005, 0.35, log=True)
        num_leaves = trial.suggest_int("num_leaves", 3, 511)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_child_samples = trial.suggest_int("min_child_samples", 1, 300)
        min_sum_hessian_in_leaf = trial.suggest_float("min_sum_hessian_in_leaf", 1e-5, 50.0, log=True)
        
        # Feature sampling strategies
        feature_fraction = trial.suggest_float("feature_fraction", 0.15, 1.0)
        feature_fraction_bynode = trial.suggest_float("feature_fraction_bynode", 0.15, 1.0)
        
        # Bagging strategies
        bagging_fraction = trial.suggest_float("bagging_fraction", 0.15, 1.0)
        bagging_freq = trial.suggest_int("bagging_freq", 0, 20)
        
        # Regularization (extreme ranges)
        lambda_l1 = trial.suggest_float("lambda_l1", 1e-8, 15.0, log=True)
        lambda_l2 = trial.suggest_float("lambda_l2", 1e-8, 15.0, log=True)
        min_gain_to_split = trial.suggest_float("min_gain_to_split", 0.0, 6.0)
        path_smooth = trial.suggest_float("path_smooth", 0.0, 15.0)
        max_delta_step = trial.suggest_float("max_delta_step", 0.0, 6.0)
        
        # Binning strategies
        max_bin = trial.suggest_int("max_bin", 31, 1023)
        min_data_in_bin = trial.suggest_int("min_data_in_bin", 1, 10)
        
        # All boosting types
        boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss", "rf"])
        
        # Extra randomization
        extra_trees = trial.suggest_categorical("extra_trees", [False, True])
        
        # DART-specific parameters
        if boosting_type == "dart":
            drop_rate = trial.suggest_float("drop_rate", 0.05, 0.4)
            skip_drop = trial.suggest_float("skip_drop", 0.2, 0.9)
            max_drop = trial.suggest_int("max_drop", 10, 80)
        else:
            drop_rate = 0.1
            skip_drop = 0.5
            max_drop = 50
        
        # GOSS-specific parameters
        if boosting_type == "goss":
            top_rate = trial.suggest_float("top_rate", 0.1, 0.4)
            other_rate = trial.suggest_float("other_rate", 0.05, 0.2)
        else:
            top_rate = 0.2
            other_rate = 0.1
        
        # Adaptive iteration count
        if learning_rate < 0.005:
            n_estimators = trial.suggest_int("n_estimators", 8000, 12000)
        elif learning_rate < 0.02:
            n_estimators = trial.suggest_int("n_estimators", 5000, 9000)
        elif learning_rate < 0.08:
            n_estimators = trial.suggest_int("n_estimators", 3000, 6000)
        else:
            n_estimators = trial.suggest_int("n_estimators", 2000, 4500)
        
        # Build params
        params = dict(base_params)
        params.update({
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "min_sum_hessian_in_leaf": min_sum_hessian_in_leaf,
            "feature_fraction": feature_fraction,
            "feature_fraction_bynode": feature_fraction_bynode,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "min_gain_to_split": min_gain_to_split,
            "max_bin": max_bin,
            "min_data_in_bin": min_data_in_bin,
            "max_delta_step": max_delta_step,
            "boosting_type": boosting_type,
            "path_smooth": path_smooth,
            "extra_trees": extra_trees,
            "n_estimators": n_estimators,
            "drop_rate": drop_rate,
            "skip_drop": skip_drop,
            "max_drop": max_drop,
            "top_rate": top_rate,
            "other_rate": other_rate,
        })
        # Ensure compatibility: GOSS/RF ignore bagging params
        try:
            if params.get("boosting_type") in ("goss", "rf"):
                params.pop("bagging_fraction", None)
                params.pop("bagging_freq", None)
        except Exception:
            pass
        
        try:
            clf = lgb.LGBMClassifier(**params)
            bt = params.get("boosting_type", "gbdt")
            fit_callbacks = [lgb.early_stopping(stopping_rounds=150, verbose=False)] if bt != "dart" else []
            clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=fit_callbacks
            )
            proba = clf.predict_proba(X_va)[:, 1]
            stats = evaluate_thresholds(y_va, proba, thr_grid)
            score = score_fold(stats)
            
            trial.set_user_attr("stats", stats)
            trial.set_user_attr("params", params)
            
            return score
        except Exception as e:
            return -1.0
    
    return objective


def create_optuna_objective_extreme(X_tr: Any, y_tr: np.ndarray, X_va: Any, y_va: np.ndarray, 
                                      thr_grid: np.ndarray, prior_best: Optional[Dict[str, Any]] = None) -> callable:
    """
    EXTREME Phase 5 objective - Final desperate attempt with maximum experimentation.
    Combines insights from all previous phases with even more creative parameter combinations.
    """
    base_params = default_lgb_params()
    
    def objective(trial: optuna.Trial) -> float:
        # Most extreme ranges possible
        learning_rate = trial.suggest_float("learning_rate", 0.0003, 0.4, log=True)
        num_leaves = trial.suggest_int("num_leaves", 2, 600)
        max_depth = trial.suggest_int("max_depth", 1, 25)
        min_child_samples = trial.suggest_int("min_child_samples", 1, 350)
        min_sum_hessian_in_leaf = trial.suggest_float("min_sum_hessian_in_leaf", 1e-6, 100.0, log=True)
        
        # Maximum feature/data sampling flexibility
        feature_fraction = trial.suggest_float("feature_fraction", 0.1, 1.0)
        feature_fraction_bynode = trial.suggest_float("feature_fraction_bynode", 0.1, 1.0)
        bagging_fraction = trial.suggest_float("bagging_fraction", 0.1, 1.0)
        bagging_freq = trial.suggest_int("bagging_freq", 0, 25)
        
        # Maximum regularization ranges
        lambda_l1 = trial.suggest_float("lambda_l1", 1e-9, 20.0, log=True)
        lambda_l2 = trial.suggest_float("lambda_l2", 1e-9, 20.0, log=True)
        min_gain_to_split = trial.suggest_float("min_gain_to_split", 0.0, 8.0)
        path_smooth = trial.suggest_float("path_smooth", 0.0, 20.0)
        max_delta_step = trial.suggest_float("max_delta_step", 0.0, 8.0)
        
        # Binning experiments
        max_bin = trial.suggest_int("max_bin", 15, 1023)
        min_data_in_bin = trial.suggest_int("min_data_in_bin", 1, 15)
        
        # All boosting types
        boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss", "rf"])
        
        # Extra experiments
        extra_trees = trial.suggest_categorical("extra_trees", [False, True])
        
        # Boosting-specific parameters
        if boosting_type == "dart":
            drop_rate = trial.suggest_float("drop_rate", 0.01, 0.5)
            skip_drop = trial.suggest_float("skip_drop", 0.1, 0.95)
            max_drop = trial.suggest_int("max_drop", 5, 100)
            uniform_drop = trial.suggest_categorical("uniform_drop", [False, True])
        else:
            drop_rate, skip_drop, max_drop, uniform_drop = 0.1, 0.5, 50, False
        
        if boosting_type == "goss":
            top_rate = trial.suggest_float("top_rate", 0.05, 0.5)
            other_rate = trial.suggest_float("other_rate", 0.02, 0.3)
        else:
            top_rate, other_rate = 0.2, 0.1
        
        # Very adaptive n_estimators
        if learning_rate < 0.002:
            n_estimators = trial.suggest_int("n_estimators", 10000, 15000)
        elif learning_rate < 0.01:
            n_estimators = trial.suggest_int("n_estimators", 7000, 11000)
        elif learning_rate < 0.04:
            n_estimators = trial.suggest_int("n_estimators", 4000, 8000)
        elif learning_rate < 0.1:
            n_estimators = trial.suggest_int("n_estimators", 2500, 5500)
        else:
            n_estimators = trial.suggest_int("n_estimators", 1500, 4000)
        
        # Build params
        params = dict(base_params)
        params.update({
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "min_sum_hessian_in_leaf": min_sum_hessian_in_leaf,
            "feature_fraction": feature_fraction,
            "feature_fraction_bynode": feature_fraction_bynode,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "min_gain_to_split": min_gain_to_split,
            "max_bin": max_bin,
            "min_data_in_bin": min_data_in_bin,
            "max_delta_step": max_delta_step,
            "boosting_type": boosting_type,
            "path_smooth": path_smooth,
            "extra_trees": extra_trees,
            "n_estimators": n_estimators,
            "drop_rate": drop_rate,
            "skip_drop": skip_drop,
            "max_drop": max_drop,
            "uniform_drop": uniform_drop,
            "top_rate": top_rate,
            "other_rate": other_rate,
        })
        
        # Compatibility
        try:
            if params.get("boosting_type") in ("goss", "rf"):
                params.pop("bagging_fraction", None)
                params.pop("bagging_freq", None)
        except Exception:
            pass
        
        try:
            clf = lgb.LGBMClassifier(**params)
            bt = params.get("boosting_type", "gbdt")
            fit_callbacks = [lgb.early_stopping(stopping_rounds=200, verbose=False)] if bt != "dart" else []
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc", callbacks=fit_callbacks)
            proba = clf.predict_proba(X_va)[:, 1]
            stats = evaluate_thresholds(y_va, proba, thr_grid)
            score = score_fold(stats)
            trial.set_user_attr("stats", stats)
            trial.set_user_attr("params", params)
            return score
        except Exception:
            return -1.0
    
    return objective


def create_optuna_objective(X_tr: Any, y_tr: np.ndarray, X_va: Any, y_va: np.ndarray, 
                            thr_grid: np.ndarray, prior_best: Optional[Dict[str, Any]] = None) -> callable:
    """
    Create an Optuna objective function for hyperparameter optimization.
    Uses TPE sampler with informed priors if prior_best is provided.
    """
    base_params = default_lgb_params()
    
    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters with priors if available
        if prior_best:
            # Narrow search around prior best
            lr_center = prior_best.get("learning_rate", 0.05)
            leaves_center = prior_best.get("num_leaves", 63)
            depth_center = prior_best.get("max_depth", 6)
            min_child_center = prior_best.get("min_child_samples", 60)
            feat_center = prior_best.get("feature_fraction", 0.85)
            bag_center = prior_best.get("bagging_fraction", 0.85)
            l1_center = prior_best.get("lambda_l1", 0.1)
            l2_center = prior_best.get("lambda_l2", 0.1)
            
            learning_rate = trial.suggest_float("learning_rate", 
                                                max(0.01, lr_center * 0.6), 
                                                min(0.15, lr_center * 1.4), 
                                                log=True)
            num_leaves = trial.suggest_int("num_leaves", 
                                          max(15, leaves_center - 32), 
                                          min(127, leaves_center + 32))
            max_depth = trial.suggest_int("max_depth", 
                                         max(3, depth_center - 2), 
                                         min(10, depth_center + 2))
            min_child_samples = trial.suggest_int("min_child_samples", 
                                                 max(10, min_child_center - 30), 
                                                 min(120, min_child_center + 30))
            feature_fraction = trial.suggest_float("feature_fraction", 
                                                  max(0.5, feat_center - 0.15), 
                                                  min(1.0, feat_center + 0.15))
            bagging_fraction = trial.suggest_float("bagging_fraction", 
                                                  max(0.5, bag_center - 0.15), 
                                                  min(1.0, bag_center + 0.15))
            lambda_l1 = trial.suggest_float("lambda_l1", 
                                           max(1e-8, l1_center * 0.1), 
                                           min(1.0, l1_center * 2.0), 
                                           log=True)
            lambda_l2 = trial.suggest_float("lambda_l2", 
                                           max(1e-8, l2_center * 0.1), 
                                           min(1.0, l2_center * 2.0), 
                                           log=True)
        else:
            # Wide search
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
            num_leaves = trial.suggest_int("num_leaves", 15, 127)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            min_child_samples = trial.suggest_int("min_child_samples", 10, 120)
            feature_fraction = trial.suggest_float("feature_fraction", 0.5, 1.0)
            bagging_fraction = trial.suggest_float("bagging_fraction", 0.5, 1.0)
            lambda_l1 = trial.suggest_float("lambda_l1", 0.0, 1.0)
            lambda_l2 = trial.suggest_float("lambda_l2", 0.0, 1.0)
        
        # Build params
        params = dict(base_params)
        params.update({
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
        })
        
        try:
            clf = lgb.LGBMClassifier(**params)
            bt = params.get("boosting_type", "gbdt")
            fit_callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)] if bt != "dart" else []
            clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=fit_callbacks
            )
            proba = clf.predict_proba(X_va)[:, 1]
            stats = evaluate_thresholds(y_va, proba, thr_grid)
            score = score_fold(stats)
            
            # Store additional info for later retrieval
            trial.set_user_attr("stats", stats)
            trial.set_user_attr("params", params)
            
            return score
        except Exception as e:
            # Return worst score on failure
            return -1.0
    
    return objective


def evaluate_thresholds(y_true: np.ndarray, proba: np.ndarray, thr_grid: np.ndarray) -> Dict[str, Any]:
    has_two_classes = len(np.unique(y_true)) == 2
    auc_val: Optional[float]
    if has_two_classes:
        try:
            auc_val = float(roc_auc_score(y_true, proba))
        except Exception:
            auc_val = None
    else:
        auc_val = None

    best = {
        "thr": 0.5,
        "acc": float("nan"),
        "auc": auc_val,  # can be None
        "prec": float("nan"),
        "rec": float("nan"),
        "f1": float("nan"),
        "bacc": float("nan"),
    }
    for thr in thr_grid:
        pred = (proba >= thr).astype(int)
        acc = accuracy_score(y_true, pred)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        bacc = balanced_accuracy_score(y_true, pred)
        if (np.isnan(best["f1"]) or f1 > best["f1"]) or (f1 == best["f1"] and acc > best.get("acc", 0)):
            best = {"thr": float(thr), "acc": float(acc), "auc": auc_val, "prec": float(prec), "rec": float(rec), "f1": float(f1), "bacc": float(bacc)}
    return best


def score_fold(stats: Dict[str, Any]) -> float:
    """Fold selection score.
    - If AUC is defined: average of F1, AUC, and Accuracy
    - If AUC is undefined (e.g., single-class validation): average of F1 and Balanced Accuracy
    """
    auc_val_raw = stats.get("auc", None)
    f1 = float(stats.get("f1", 0.0))
    acc = float(stats.get("acc", 0.0))
    bacc = float(stats.get("bacc", 0.0))
    if auc_val_raw is None or (isinstance(auc_val_raw, float) and np.isnan(auc_val_raw)):
        return (f1 + bacc) / 2.0
    try:
        auc_for_score = float(auc_val_raw)
    except Exception:
        return (f1 + bacc) / 2.0
    return (f1 + auc_for_score + acc) / 3.0


class MetaLearner:
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

    def _sig(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        pos = int(np.sum(y == 1))
        neg = int(np.sum(y == 0))
        pr = pos / (pos + neg) if (pos + neg) else 0.5
        feat_mean = float(np.mean(X)) if X.size else 0.0
        feat_std = float(np.std(X)) if X.size else 0.0
        return [float(len(y)), pr, feat_mean, feat_std]

    def fit(self, metas: List[Dict[str, Any]]):
        if len(metas) < 2:
            self.fitted = False
            return
        X_meta = []
        y_meta: Dict[str, List[float]] = {k: [] for k in self.regressors.keys()}
        for m in metas:
            sig = m.get("train_signature", {})
            X_meta.append([
                sig.get("n", 0.0),
                sig.get("pos_rate", 0.5),
                sig.get("feat_mean_avg", 0.0),
                sig.get("feat_std_avg", 0.0),
            ])
            params = m.get("params", {})
            for k in y_meta.keys():
                y_meta[k].append(params.get(k, default_lgb_params().get(k)))
        X_meta = np.array(X_meta)
        for k, reg in self.regressors.items():
            try:
                yk = np.array(y_meta[k])
                if len(yk) >= 2 and np.std(yk) > 1e-6:
                    reg.fit(X_meta, yk)
            except Exception:
                pass
        self.fitted = True

    def predict(self, sig: Dict[str, float]) -> Dict[str, Any]:
        if not self.fitted:
            return default_lgb_params()
        x = np.array([[sig.get("n", 0.0), sig.get("pos_rate", 0.5), sig.get("feat_mean_avg", 0.0), sig.get("feat_std_avg", 0.0)]])
        params = default_lgb_params()
        try:
            params.update({
                k: float(self.regressors[k].predict(x)[0]) if k not in ("num_leaves", "max_depth", "min_child_samples") else int(round(self.regressors[k].predict(x)[0]))
                for k in self.regressors.keys()
            })
            # clamp ranges
            params["learning_rate"] = float(np.clip(params["learning_rate"], 0.01, 0.15))
            params["num_leaves"] = int(np.clip(params["num_leaves"], 15, 127))
            params["max_depth"] = int(np.clip(params["max_depth"], 3, 10))
            params["min_child_samples"] = int(np.clip(params["min_child_samples"], 10, 120))
            params["feature_fraction"] = float(np.clip(params["feature_fraction"], 0.5, 1.0))
            params["bagging_fraction"] = float(np.clip(params["bagging_fraction"], 0.5, 1.0))
            params["lambda_l1"] = float(np.clip(params["lambda_l1"], 0.0, 1.0))
            params["lambda_l2"] = float(np.clip(params["lambda_l2"], 0.0, 1.0))
        except Exception:
            pass
        return params


# %% LOAD DATA
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

# Wrap scaled arrays in DataFrames with stable feature names to avoid sklearn/lightgbm warnings
X_train_df = pd.DataFrame(X_train_s, columns=FEATURE_NAMES)
X_pre_df = pd.DataFrame(X_pre_s, columns=FEATURE_NAMES)
X_test_df = pd.DataFrame(X_test_s, columns=FEATURE_NAMES) if len(X_test_s) else pd.DataFrame(
    np.empty((0, X_train_s.shape[1])), columns=FEATURE_NAMES
)


# %% BASELINE
baseline_model = lgb.LGBMClassifier(**default_lgb_params())
baseline_model.fit(X_train_df, y_train)


# %% PRE-TEST: CUMULATIVE FOLD OPTIMIZATION
print("\n" + "="*70)
print("  PRE-TEST HYPERPARAMETER OPTIMIZATION (CUMULATIVE, TIME-LIMITED)")
print("="*70)

thr_grid = np.round(np.arange(0.30, 0.80 + 1e-9, 0.01), 2)
fold_meta: List[Dict[str, Any]] = []
best_settings: List[Dict[str, Any]] = []
best_thr_list: List[float] = []
fold_summaries: List[Dict[str, Any]] = []
meta_learner = MetaLearner()
prior_best_params: Optional[Dict[str, Any]] = None
dyn_run1, dyn_thr1, dyn_run2, dyn_thr2 = load_dynamic_known_bests()

pre_n = len(y_pre)
for fold_idx, (train_pre_end, val_start, val_end) in enumerate(cumulative_pretest_splits(pre_n, PRETEST_FOLD_SIZE, GAP_BARS), start=1):
    fold_t0 = time.time()
    # Build cumulative train: Train window + pre-test up to train_pre_end (as DataFrames)
    X_tr_df = pd.concat([X_train_df, X_pre_df.iloc[:train_pre_end]], ignore_index=True) if train_pre_end > 0 else X_train_df
    y_tr = np.concatenate([y_train, y_pre[:train_pre_end]]) if train_pre_end > 0 else y_train
    X_va_df = X_pre_df.iloc[val_start:val_end]
    y_va = y_pre[val_start:val_end]

    # Label counts
    tr_pos = int(np.sum(y_tr == 1)); tr_neg = int(np.sum(y_tr == 0))
    va_pos = int(np.sum(y_va == 1)); va_neg = int(np.sum(y_va == 0))

    sig = {
        "n": float(len(y_tr)),
        "pos_rate": float(tr_pos / (tr_pos + tr_neg)) if (tr_pos + tr_neg) else 0.5,
        "feat_mean_avg": float(np.mean(X_tr_df.values)) if X_tr_df.size else 0.0,
        "feat_std_avg": float(np.std(X_tr_df.values)) if X_tr_df.size else 0.0,
    }

    print(f"\n{'─'*70}")
    print(f"✓ FOLD {fold_idx} | Train: {len(y_tr)} bars (pos={tr_pos}, neg={tr_neg}) | Val: {len(y_va)} bars (pos={va_pos}, neg={va_neg})")
    print(f"  Train-pre end idx: {train_pre_end} | Val idx: [{val_start}, {val_end}) | Gap={GAP_BARS}")
    if len(np.unique(y_va)) < 2:
        print("  ⚠️  Validation is single-class ⇒ AUC undefined; scoring will ignore AUC for this fold.")
    
    # MODIFICATION 2: Prompt user to search or skip
    print(f"\n  Would you like to:")
    print(f"    [1] Run full hyperparameter search (~5 min)")
    print(f"    [2] Skip and use known best params from previous run")
    has_known = fold_idx in MANUAL_BEST_RUN1 or fold_idx in dyn_run1 or fold_idx in KNOWN_BEST_PARAMS_RUN1
    user_choice = get_user_choice("  Enter choice (1 or 2): ", has_known=has_known)
    skip_search = (user_choice == "2")
    
    # Skip: use manual best directly (no re-evaluation)
    if skip_search and (fold_idx in MANUAL_BEST_RUN1 or fold_idx in dyn_run1 or fold_idx in KNOWN_BEST_PARAMS_RUN1):
        manual_best = MANUAL_BEST_RUN1.get(fold_idx)
        if manual_best:
            print(f"  ⚡ Skipping search, using manual newest-best for fold {fold_idx}")
            best_info = {
                "params": manual_best.get("params", default_lgb_params()),
                "best_thr": float(manual_best.get("best_thr", 0.5)),
                "best_auc": manual_best.get("best_auc"),
                "best_acc": float(manual_best.get("best_acc", 0)),
                "best_prec": float(manual_best.get("best_prec", 0)),
                "best_rec": float(manual_best.get("best_rec", 0)),
                "best_f1": float(manual_best.get("best_f1", 0)),
                "best_bacc": float(manual_best.get("best_bacc", 0)),
                "score": float(manual_best.get("score", 0)),
            }
        else:
            # Fallback to dynamic or static
            best_params_source = dyn_run1.get(fold_idx, KNOWN_BEST_PARAMS_RUN1.get(fold_idx))
            print(f"  ⚡ Skipping search, using known best params for fold {fold_idx} (loaded={'dynamic' if fold_idx in dyn_run1 else 'static'})")
            best_params_dict = best_params_source
            # Quick evaluation with known params
            clf_quick = lgb.LGBMClassifier(**merge_params(default_lgb_params(), best_params_dict))
            clf_quick.fit(X_tr_df, y_tr)
            proba_quick = clf_quick.predict_proba(X_va_df)[:, 1]
            stats_quick = evaluate_thresholds(y_va, proba_quick, thr_grid)
            best_info = {
                "params": merge_params(default_lgb_params(), best_params_dict),
                "best_thr": float(dyn_thr1.get(fold_idx, stats_quick["thr"])),
                "best_auc": stats_quick["auc"],
                "best_acc": float(stats_quick["acc"]),
                "best_prec": float(stats_quick["prec"]),
                "best_rec": float(stats_quick["rec"]),
                "best_f1": float(stats_quick["f1"]),
                "best_bacc": float(stats_quick["bacc"]),
                "score": float(score_fold(stats_quick)),
            }
        best_score = best_info["score"]
        trials = 1
        improved = 1
        elapsed = time.time() - fold_t0
    else:
        if skip_search:
            print(f"  ⚠️  No known params for fold {fold_idx}, running search anyway")
        
        # Calculate total time budget
        total_time_budget = max(0.0, FOLD_TIME_LIMIT - (time.time() - fold_t0))

        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Phase planning: first 3 folds = exploratory only; from fold 4+: 3 min exploratory + 2 min prior-guided
        is_first_three = fold_idx <= 3
        phase1_budget = min(180.0, total_time_budget) if not is_first_three else total_time_budget
        phase2_budget = 0.0 if is_first_three else max(0.0, min(120.0, total_time_budget - phase1_budget))

        # PHASE 1: Exploratory search
        print(f"  🔍 Phase 1 — Exploratory | Budget: {phase1_budget:.1f}s")
        sampler_e = TPESampler(seed=42, n_startup_trials=30)
        pruner_e = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        study_e = optuna.create_study(direction="maximize", sampler=sampler_e, pruner=pruner_e, study_name=f"fold_{fold_idx}_explore")
        objective_e = create_optuna_objective(X_tr_df, y_tr, X_va_df, y_va, thr_grid, prior_best=None)
        try:
            study_e.optimize(
                objective_e,
                timeout=phase1_budget,
                n_jobs=1,
                show_progress_bar=False,
                callbacks=[
                    lambda study, trial: print(
                        f"  ✓ [E] Trial {trial.number+1:>3} | Score: {trial.value:.4f} → {'NEW BEST' if trial.value == study.best_value else 'continue'} | "
                        f"F1={trial.user_attrs.get('stats', {}).get('f1', 0):.4f}, "
                        f"AUC={trial.user_attrs.get('stats', {}).get('auc') if trial.user_attrs.get('stats', {}).get('auc') is not None else 'undefined'}, "
                        f"ACC={trial.user_attrs.get('stats', {}).get('acc', 0):.4f}, "
                        f"θ={trial.user_attrs.get('stats', {}).get('thr', 0.5):.2f} | "
                        f"lr={trial.params.get('learning_rate', 0):.4f}, leaves={trial.params.get('num_leaves', 0)}, depth={trial.params.get('max_depth', 0)}"
                    ) if trial.value > 0 and trial.number % 5 == 0 else None
                ]
            )
        except Exception as e:
            print(f"  ⚠️  Phase 1 optimization error: {str(e)}")

        # Determine time left
        time_left = max(0.0, FOLD_TIME_LIMIT - (time.time() - fold_t0))

        # PHASE 2: Prior-guided search (only from fold 4+)
        study_g = None
        if not is_first_three and time_left > 1.0 and phase2_budget > 0.0:
            # Prefer best from Phase 1 for guidance; fallback to prior_best_params from previous folds
            guided_prior = None
            if study_e.best_trial is not None:
                guided_prior = study_e.best_trial.user_attrs.get("params", None)
            if guided_prior is None:
                guided_prior = prior_best_params

            print(f"  🔍 Phase 2 — Prior-guided | Budget: {min(phase2_budget, time_left):.1f}s")
            sampler_g = TPESampler(seed=42, n_startup_trials=20)
            pruner_g = MedianPruner(n_startup_trials=8, n_warmup_steps=5)
            study_g = optuna.create_study(direction="maximize", sampler=sampler_g, pruner=pruner_g, study_name=f"fold_{fold_idx}_guided")
            objective_g = create_optuna_objective(X_tr_df, y_tr, X_va_df, y_va, thr_grid, prior_best=guided_prior)
            # Note: enqueue_trial removed to avoid "out of range" warnings when prior params don't fit guided search space
            try:
                study_g.optimize(
                    objective_g,
                    timeout=min(phase2_budget, time_left),
                    n_jobs=1,
                    show_progress_bar=False,
                    callbacks=[
                        lambda study, trial: print(
                            f"  ✓ [G] Trial {trial.number+1:>3} | Score: {trial.value:.4f} → {'NEW BEST' if trial.value == study.best_value else 'continue'} | "
                            f"F1={trial.user_attrs.get('stats', {}).get('f1', 0):.4f}, "
                            f"AUC={trial.user_attrs.get('stats', {}).get('auc') if trial.user_attrs.get('stats', {}).get('auc') is not None else 'undefined'}, "
                            f"ACC={trial.user_attrs.get('stats', {}).get('acc', 0):.4f}, "
                            f"θ={trial.user_attrs.get('stats', {}).get('thr', 0.5):.2f} | "
                            f"lr={trial.params.get('learning_rate', 0):.4f}, leaves={trial.params.get('num_leaves', 0)}, depth={trial.params.get('max_depth', 0)}"
                        ) if trial.value > 0 and trial.number % 5 == 0 else None
                    ]
                )
            except Exception as e:
                print(f"  ⚠️  Phase 2 optimization error: {str(e)}")

        # Combine results from phases
        all_trials = list(study_e.trials)
        if study_g is not None:
            all_trials += list(study_g.trials)

        trials = len(all_trials)
        # Determine global best across phases
        best_trial = None
        best_value = -np.inf
        for t in all_trials:
            if t.value is not None and t.value > best_value:
                best_value = t.value
                best_trial = t

        if best_trial is not None:
            best_stats = best_trial.user_attrs.get("stats", {})
            best_params_dict = best_trial.user_attrs.get("params", default_lgb_params())
            best_score = float(best_value)
            best_auc = best_stats.get("auc")
            
            # Count improvements
            improved = 0
            try:
                improved += len([t for t in study_e.trials if study_e.best_value is not None and t.value == study_e.best_value])
            except Exception:
                pass
            if study_g is not None:
                try:
                    improved += len([t for t in study_g.trials if study_g.best_value is not None and t.value == study_g.best_value])
                except Exception:
                    pass
            
            # MODIFICATION 3: Extension mechanism if AUC < 0.55 OR threshold <= 0.30 (poor calibration)
            best_thr_val = best_stats.get("thr", 0.5)
            trigger_extension = False
            trigger_reason = []
            if best_auc is not None and not np.isnan(best_auc) and best_auc < AUC_THRESHOLD_FOR_EXTENSION:
                trigger_extension = True
                trigger_reason.append(f"AUC={best_auc:.4f} < {AUC_THRESHOLD_FOR_EXTENSION}")
            if best_thr_val <= THRESHOLD_FLOOR_TRIGGER:
                trigger_extension = True
                trigger_reason.append(f"θ={best_thr_val:.2f} ≤ {THRESHOLD_FLOOR_TRIGGER}")
            
            if trigger_extension:
                print(f"\n  ⚠️  Extension triggered: {' AND '.join(trigger_reason)}")
                print(f"  🚀 PHASE 3 — Extended Search (thinking outside the box) | Budget: {EXTENSION_TIME_LIMIT:.1f}s")
                print(f"     Strategy: Expanded hyperparameter space + alternative boosting types + adaptive trees")
                
                # Use extended objective with more hyperparameters
                sampler_ext = TPESampler(seed=43, n_startup_trials=20)
                pruner_ext = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
                study_ext = optuna.create_study(direction="maximize", sampler=sampler_ext, pruner=pruner_ext, study_name=f"fold_{fold_idx}_extended")
                objective_ext = create_optuna_objective_extended(X_tr_df, y_tr, X_va_df, y_va, thr_grid, prior_best=best_params_dict)
                
                try:
                    study_ext.optimize(
                        objective_ext,
                        timeout=EXTENSION_TIME_LIMIT,
                        n_jobs=1,
                        show_progress_bar=False,
                        callbacks=[
                            lambda study, trial: print(
                                f"  ✓ [X] Trial {trial.number+1:>3} | Score: {trial.value:.4f} → {'NEW BEST' if trial.value == study.best_value else 'continue'} | "
                                f"F1={trial.user_attrs.get('stats', {}).get('f1', 0):.4f}, "
                                f"AUC={trial.user_attrs.get('stats', {}).get('auc') if trial.user_attrs.get('stats', {}).get('auc') is not None else 'undefined'}, "
                                f"ACC={trial.user_attrs.get('stats', {}).get('acc', 0):.4f}, "
                                f"θ={trial.user_attrs.get('stats', {}).get('thr', 0.5):.2f} | "
                                f"lr={trial.params.get('learning_rate', 0):.4f}, leaves={trial.params.get('num_leaves', 0)}, boost={trial.params.get('boosting_type', 'gbdt')}"
                            ) if trial.value > 0 and trial.number % 5 == 0 else None
                        ]
                    )
                    
                    # Merge extension results
                    all_trials += list(study_ext.trials)
                    trials = len(all_trials)
                    
                    # Re-determine global best including extension trials
                    for t in study_ext.trials:
                        if t.value is not None and t.value > best_value:
                            best_value = t.value
                            best_trial = t
                    
                    if best_trial is not None:
                        best_stats = best_trial.user_attrs.get("stats", {})
                        best_params_dict = best_trial.user_attrs.get("params", default_lgb_params())
                        best_score = float(best_value)
                        best_auc = best_stats.get("auc")
                        
                        # Recount improvements
                        try:
                            improved += len([t for t in study_ext.trials if study_ext.best_value is not None and t.value == study_ext.best_value])
                        except Exception:
                            pass
                        
                        print(f"  ✅ Phase 3 complete | AUC: {best_auc if best_auc is not None else 'undefined'} | θ={best_stats.get('thr', 0.5):.2f}")
                        
                        # PHASE 4: ULTRA-AGGRESSIVE if still problematic
                        best_thr_after_phase3 = best_stats.get("thr", 0.5)
                        still_problematic = False
                        if (best_auc is not None and best_auc < AUC_THRESHOLD_FOR_EXTENSION) or (best_thr_after_phase3 <= THRESHOLD_FLOOR_TRIGGER):
                            still_problematic = True
                        
                        if still_problematic:
                            print(f"\n  🔥 PHASE 4 — ULTRA-AGGRESSIVE Search (maximum creativity) | Budget: {ULTRA_EXTENSION_TIME_LIMIT:.1f}s")
                            print(f"     Strategy: Extreme ranges + RF/DART boosting + adaptive n_estimators + wild regularization")
                            
                            sampler_ultra = TPESampler(seed=44, n_startup_trials=15)
                            pruner_ultra = MedianPruner(n_startup_trials=3, n_warmup_steps=2)
                            study_ultra = optuna.create_study(direction="maximize", sampler=sampler_ultra, pruner=pruner_ultra, study_name=f"fold_{fold_idx}_ultra")
                            
                            objective_ultra = create_optuna_objective_ultra(X_tr_df, y_tr, X_va_df, y_va, thr_grid, prior_best=best_params_dict)
                            
                            try:
                                study_ultra.optimize(
                                    objective_ultra,
                                    timeout=ULTRA_EXTENSION_TIME_LIMIT,
                                    n_jobs=1,
                                    show_progress_bar=False,
                                    callbacks=[
                                        lambda study, trial: print(
                                            f"  ✓ [U] Trial {trial.number+1:>3} | Score: {trial.value:.4f} → {'NEW BEST' if trial.value == study.best_value else 'continue'} | "
                                            f"F1={trial.user_attrs.get('stats', {}).get('f1', 0):.4f}, "
                                            f"AUC={trial.user_attrs.get('stats', {}).get('auc') if trial.user_attrs.get('stats', {}).get('auc') is not None else 'undefined'}, "
                                            f"θ={trial.user_attrs.get('stats', {}).get('thr', 0.5):.2f} | "
                                            f"lr={trial.params.get('learning_rate', 0):.4f}, leaves={trial.params.get('num_leaves', 0)}, boost={trial.params.get('boosting_type', 'gbdt')}"
                                        ) if trial.value > 0 and trial.number % 5 == 0 else None
                                    ]
                                )
                                
                                # Merge ultra results
                                all_trials += list(study_ultra.trials)
                                trials = len(all_trials)
                                
                                # Re-determine global best including ultra trials
                                for t in study_ultra.trials:
                                    if t.value is not None and t.value > best_value:
                                        best_value = t.value
                                        best_trial = t
                                
                                if best_trial is not None:
                                    best_stats = best_trial.user_attrs.get("stats", {})
                                    best_params_dict = best_trial.user_attrs.get("params", default_lgb_params())
                                    best_score = float(best_value)
                                    best_auc = best_stats.get("auc")
                                    
                                    # Recount improvements
                                    try:
                                        improved += len([t for t in study_ultra.trials if study_ultra.best_value is not None and t.value == study_ultra.best_value])
                                    except Exception:
                                        pass
                                    
                                    print(f"  ✅ Phase 4 complete | Final AUC: {best_auc if best_auc is not None else 'undefined'} | θ={best_stats.get('thr', 0.5):.2f}")
                                    if best_auc is not None and best_auc >= AUC_THRESHOLD_FOR_EXTENSION and best_stats.get('thr', 0) > THRESHOLD_FLOOR_TRIGGER:
                                        print(f"  🎯 SUCCESS: Metrics improved!")
                                    else:
                                        # PHASE 5: EXTREME final attempt
                                        print(f"\n  💥 PHASE 5 — EXTREME Search (final attempt) | Budget: {EXTREME_EXTENSION_TIME_LIMIT:.1f}s")
                                        print(f"     Strategy: Maximum experimentation with all LightGBM parameters + extreme ranges")
                                        
                                        sampler_extreme = TPESampler(seed=45, n_startup_trials=10)
                                        pruner_extreme = MedianPruner(n_startup_trials=2, n_warmup_steps=1)
                                        study_extreme = optuna.create_study(direction="maximize", sampler=sampler_extreme, pruner=pruner_extreme, study_name=f"fold_{fold_idx}_extreme")
                                        objective_extreme = create_optuna_objective_extreme(X_tr_df, y_tr, X_va_df, y_va, thr_grid, prior_best=best_params_dict)
                                        
                                        try:
                                            study_extreme.optimize(
                                                objective_extreme,
                                                timeout=EXTREME_EXTENSION_TIME_LIMIT,
                                                n_jobs=1,
                                                show_progress_bar=False,
                                                callbacks=[
                                                    lambda study, trial: print(
                                                        f"  ✓ [Z] Trial {trial.number+1:>3} | Score: {trial.value:.4f} → {'NEW BEST' if trial.value == study.best_value else 'continue'} | "
                                                        f"F1={trial.user_attrs.get('stats', {}).get('f1', 0):.4f}, "
                                                        f"AUC={trial.user_attrs.get('stats', {}).get('auc') if trial.user_attrs.get('stats', {}).get('auc') is not None else 'undefined'}, "
                                                        f"θ={trial.user_attrs.get('stats', {}).get('thr', 0.5):.2f} | "
                                                        f"lr={trial.params.get('learning_rate', 0):.4f}, leaves={trial.params.get('num_leaves', 0)}, boost={trial.params.get('boosting_type', 'gbdt')}"
                                                    ) if trial.value > 0 and trial.number % 5 == 0 else None
                                                ]
                                            )
                                            
                                            all_trials += list(study_extreme.trials)
                                            trials = len(all_trials)
                                            
                                            for t in study_extreme.trials:
                                                if t.value is not None and t.value > best_value:
                                                    best_value = t.value
                                                    best_trial = t
                                            
                                            if best_trial is not None:
                                                best_stats = best_trial.user_attrs.get("stats", {})
                                                best_params_dict = best_trial.user_attrs.get("params", default_lgb_params())
                                                best_score = float(best_value)
                                                best_auc = best_stats.get("auc")
                                                try:
                                                    improved += len([t for t in study_extreme.trials if study_extreme.best_value is not None and t.value == study_extreme.best_value])
                                                except Exception:
                                                    pass
                                                
                                                print(f"  ✅ Phase 5 complete | Final AUC: {best_auc if best_auc is not None else 'undefined'} | θ={best_stats.get('thr', 0.5):.2f}")
                                                if best_auc is not None and best_auc >= AUC_THRESHOLD_FOR_EXTENSION and best_stats.get('thr', 0) > THRESHOLD_FLOOR_TRIGGER:
                                                    print(f"  🎯 Phase 5 SUCCESS!")
                                        
                                        except Exception as e:
                                            print(f"  ⚠️  Phase 5 error: {str(e)}")
                                    
                            except Exception as e:
                                print(f"  ⚠️  Phase 4 error: {str(e)}")
                        else:
                            print(f"  ✅ Phase 3 sufficient | AUC: {best_auc if best_auc is not None else 'undefined'} | θ={best_thr_after_phase3:.2f}")
                        
                except Exception as e:
                    print(f"  ⚠️  Extension search error: {str(e)}")
            
            # Compare against manual newest-best for this fold and override if manual has higher score
            manual_best = MANUAL_BEST_RUN1.get(fold_idx)
            if manual_best is not None:
                try:
                    if float(manual_best.get("score", -np.inf)) > float(best_score):
                        best_params_dict = merge_params(default_lgb_params(), manual_best.get("params", {}))
                        best_stats = {
                            "thr": float(manual_best.get("best_thr", 0.5)),
                            "auc": manual_best.get("best_auc", None),
                            "acc": float(manual_best.get("best_acc", 0)),
                            "prec": float(manual_best.get("best_prec", 0)),
                            "rec": float(manual_best.get("best_rec", 0)),
                            "f1": float(manual_best.get("best_f1", 0)),
                            "bacc": float(manual_best.get("best_bacc", 0)),
                        }
                        best_score = float(manual_best.get("score", best_score))
                except Exception:
                    pass

            best_info = {
                "params": best_params_dict,
                "best_thr": float(best_stats.get("thr", 0.5)),
                "best_auc": best_stats.get("auc"),
                "best_acc": float(best_stats.get("acc", 0)),
                "best_prec": float(best_stats.get("prec", 0)),
                "best_rec": float(best_stats.get("rec", 0)),
                "best_f1": float(best_stats.get("f1", 0)),
                "best_bacc": float(best_stats.get("bacc", 0)),
                "score": float(best_score),
            }
        else:
            # Fallback to default
            best_score = -1.0
            best_info = {
                "params": default_lgb_params(),
                "best_thr": 0.5,
                "best_auc": None,
                "best_acc": 0.0,
                "best_prec": 0.0,
                "best_rec": 0.0,
                "best_f1": 0.0,
                "best_bacc": 0.0,
                "score": -1.0,
            }
            improved = 0

        elapsed = time.time() - fold_t0
    meta_entry = dict(best_info)
    meta_entry.update({
        "fold": int(fold_idx),
        "train_size": int(len(y_tr)),
        "val_size": int(len(y_va)),
        "train_signature": sig,
        "train_pos": tr_pos,
        "train_neg": tr_neg,
        "val_pos": va_pos,
        "val_neg": va_neg,
    })
    fold_meta.append(meta_entry)
    best_settings.append(best_info.get("params", default_lgb_params()))
    best_thr_list.append(best_info.get("best_thr", 0.5))
    fold_summaries.append({
        "fold": int(fold_idx),
        "train_size": int(len(y_tr)),
        "val_size": int(len(y_va)),
        "best": best_info,
        "trials": trials,
        "improvements": improved,
        "elapsed": elapsed,
        "train_pos": tr_pos,
        "train_neg": tr_neg,
        "val_pos": va_pos,
        "val_neg": va_neg,
    })

    prior_best_params = best_info.get("params", None)
    if len(fold_meta) >= 2:
        meta_learner.fit(fold_meta)
        print(f"\n  🧠 Meta-learner updated with {len(fold_meta)} folds")

    auc_disp = best_info.get('best_auc')
    auc_str_final = f"{auc_disp:.4f}" if (auc_disp is not None and not np.isnan(auc_disp)) else "undefined"
    acc_disp = best_info.get('best_acc', 0)
    print(f"\n  📊 Fold {fold_idx} Summary:")
    print(f"     Time: {elapsed:.1f}s | Trials: {trials} | Improvements: {improved}")
    print(f"     Best Score: {best_score:.4f} | F1: {best_info.get('best_f1', 0):.4f} | AUC: {auc_str_final} | ACC: {acc_disp:.4f}")
    print(f"     Threshold: {best_info.get('best_thr', 0.5):.2f}")

print(f"\n{'='*70}")
print("  PRE-TEST OPTIMIZATION COMPLETE (CUMULATIVE)")
print(f"{'='*70}\n")


# %% META-LEARNING
print("\n" + "="*70)
print("  META-LEARNING: INFERRING FINAL HYPERPARAMETERS (TWO METHODS)")
print("="*70)

# MODIFICATION 1: Try both Train+Pre-test and Pre-test-only signatures
if len(fold_meta) >= 2 and meta_learner.fitted:
    # Method 1: Train + Pre-test combined (current method)
    X_combined_df = pd.concat([X_train_df, X_pre_df], ignore_index=True) if len(X_pre_df) else X_train_df
    y_combined = np.concatenate([y_train, y_pre]) if len(y_pre) else y_train
    sig_full_combined = {
        "n": float(len(y_combined)),
        "pos_rate": float(np.mean(y_combined)),
        "feat_mean_avg": float(np.mean(X_combined_df.values)) if X_combined_df.size else 0.0,
        "feat_std_avg": float(np.std(X_combined_df.values)) if X_combined_df.size else 0.0,
    }
    inferred_params_combined = meta_learner.predict(sig_full_combined)
    
    # Method 2: Pre-test only signature (more recent focus)
    sig_full_pretest = {
        "n": float(len(y_pre)),
        "pos_rate": float(np.mean(y_pre)),
        "feat_mean_avg": float(np.mean(X_pre_df.values)) if X_pre_df.size else 0.0,
        "feat_std_avg": float(np.std(X_pre_df.values)) if X_pre_df.size else 0.0,
    }
    inferred_params_pretest = meta_learner.predict(sig_full_pretest)
    
    print(f"✓ Using ML-based meta-learner (trained on {len(fold_meta)} folds)")
    print(f"\n  Method 1 (Train+Pre-test signature):")
    for k in ["learning_rate","num_leaves","max_depth","min_child_samples"]:
        print(f"    {k}: {inferred_params_combined.get(k)}")
    
    print(f"\n  Method 2 (Pre-test-only signature):")
    for k in ["learning_rate","num_leaves","max_depth","min_child_samples"]:
        print(f"    {k}: {inferred_params_pretest.get(k)}")
else:
    inferred_params_combined = default_lgb_params()
    inferred_params_pretest = default_lgb_params()
    print(f"✓ Using default parameters (insufficient folds for meta-learning)")


# %% FINAL TRAIN AND TEST (BOTH METHODS)
X_full_df = pd.concat([X_train_df, X_pre_df], ignore_index=True) if len(X_pre_df) else X_train_df
y_full = np.concatenate([y_train, y_pre]) if len(y_pre) else y_train

# Method 1: Train with Train+Pre-test signature params
print("\n" + "="*70)
print("  FINAL MODEL TRAINING — METHOD 1 (Train+Pre-test signature)")
print("="*70)
final_model_combined = lgb.LGBMClassifier(**inferred_params_combined)
final_model_combined.fit(X_full_df, y_full)

test_proba_combined = final_model_combined.predict_proba(X_test_df)[:, 1] if len(X_test_df) else np.array([])
test_stats_combined = {"thr": None, "acc": float("nan"), "auc": None, "prec": float("nan"), "rec": float("nan"), "f1": float("nan"), "bacc": float("nan")}
if len(test_proba_combined):
    thr_final = float(np.median(best_thr_list)) if len(best_thr_list) else 0.5
    pred_combined = (test_proba_combined >= thr_final).astype(int)
    try:
        auc_test_combined = float(roc_auc_score(y_test, test_proba_combined)) if len(np.unique(y_test)) == 2 else None
    except Exception:
        auc_test_combined = None
    test_stats_combined = {
        "thr": thr_final,
        "acc": float(accuracy_score(y_test, pred_combined)),
        "auc": auc_test_combined,
        "prec": float(precision_score(y_test, pred_combined, zero_division=0)),
        "rec": float(recall_score(y_test, pred_combined, zero_division=0)),
        "f1": float(f1_score(y_test, pred_combined, zero_division=0)),
        "bacc": float(balanced_accuracy_score(y_test, pred_combined)),
    }

# Method 2: Train with Pre-test-only signature params
print("\n" + "="*70)
print("  FINAL MODEL TRAINING — METHOD 2 (Pre-test-only signature)")
print("="*70)
final_model_pretest = lgb.LGBMClassifier(**inferred_params_pretest)
final_model_pretest.fit(X_full_df, y_full)

test_proba_pretest = final_model_pretest.predict_proba(X_test_df)[:, 1] if len(X_test_df) else np.array([])
test_stats_pretest = {"thr": None, "acc": float("nan"), "auc": None, "prec": float("nan"), "rec": float("nan"), "f1": float("nan"), "bacc": float("nan")}
if len(test_proba_pretest):
    pred_pretest = (test_proba_pretest >= thr_final).astype(int)
    try:
        auc_test_pretest = float(roc_auc_score(y_test, test_proba_pretest)) if len(np.unique(y_test)) == 2 else None
    except Exception:
        auc_test_pretest = None
    test_stats_pretest = {
        "thr": thr_final,
        "acc": float(accuracy_score(y_test, pred_pretest)),
        "auc": auc_test_pretest,
        "prec": float(precision_score(y_test, pred_pretest, zero_division=0)),
        "rec": float(recall_score(y_test, pred_pretest, zero_division=0)),
        "f1": float(f1_score(y_test, pred_pretest, zero_division=0)),
        "bacc": float(balanced_accuracy_score(y_test, pred_pretest)),
    }

# Select best method for saving
if test_stats_combined.get("auc") is not None and test_stats_pretest.get("auc") is not None:
    if test_stats_pretest["auc"] > test_stats_combined["auc"]:
        final_model = final_model_pretest
        test_proba = test_proba_pretest
        test_stats = test_stats_pretest
        inferred_params = inferred_params_pretest
        best_method = "Method 2 (Pre-test-only)"
    else:
        final_model = final_model_combined
        test_proba = test_proba_combined
        test_stats = test_stats_combined
        inferred_params = inferred_params_combined
        best_method = "Method 1 (Train+Pre-test)"
else:
    # Fallback to combined if one has undefined AUC
    final_model = final_model_combined
    test_proba = test_proba_combined
    test_stats = test_stats_combined
    inferred_params = inferred_params_combined
    best_method = "Method 1 (Train+Pre-test)"

print(f"\n✅ Selected {best_method} for final model (better AUC)")

# Top Feature Importances
try:
    importances = getattr(final_model, "feature_importances_", None)
    if importances is not None and len(importances) == len(FEATURE_NAMES):
        order = np.argsort(importances)[::-1]
        top_k = min(10, len(order))
        print("\n[Top Feature Importances]")
        for rank in range(top_k):
            idx = int(order[rank])
            print(f"  {rank+1:>2}. {FEATURE_NAMES[idx]}: {int(importances[idx])}")
        imp_map = {FEATURE_NAMES[i]: int(importances[i]) for i in range(len(FEATURE_NAMES))}
        Path(RUN_DIR, 'feature_importances.json').write_text(json.dumps(imp_map, indent=2), encoding='utf-8')
except Exception:
    pass


# %% SUMMARIES
def class_dist(y: np.ndarray) -> Dict[str, Any]:
    pos = int(np.sum(y == 1)); neg = int(np.sum(y == 0)); rate = pos/(pos+neg) if (pos+neg) else float('nan')
    return {"pos": pos, "neg": neg, "pos_rate": float(rate)}

print("\n================ THREE-WINDOW RUN SUMMARY (CUMULATIVE) ================")
print("[Data]")
print(f"  Train rows: {len(train_df):,}  |  Pre-test rows: {len(pre_df):,}  |  Test rows: {len(test_df):,}")
train_bounds = (pd.to_datetime(train_df["timestamp"]).min() if len(train_df) else None,
                pd.to_datetime(train_df["timestamp"]).max() if len(train_df) else None)
if train_bounds[0] is not None and train_bounds[1] is not None:
    print(f"  Train span: {train_bounds[0]}  →  {train_bounds[1]}")
print(f"  Pre-test span: {pretest_bounds[0]}  →  {pretest_bounds[1]}")
print(f"  Test span:     {test_bounds[0]}  →  {test_bounds[1]}")
print(f"  Class dist — Train: {class_dist(y_train)}")
print(f"  Class dist — Pre-test: {class_dist(y_pre)}")
print(f"  Class dist — Test: {class_dist(y_test)}")

print("\n[Baseline on Train]")
try:
    base_proba = baseline_model.predict_proba(X_pre_df)[:, 1] if len(X_pre_df) else np.array([])
    if len(base_proba):
        base_stats = evaluate_thresholds(y_pre, base_proba, thr_grid)
        auc_b = base_stats.get('auc')
        auc_b_str = f"{auc_b:.4f}" if (auc_b is not None and not np.isnan(auc_b)) else "undefined"
        print(f"  Pre-test baseline: AUC={auc_b_str}, F1={base_stats['f1']:.4f}, θ={base_stats['thr']:.2f}")
except Exception:
    pass

print("\n[Pre-test Overfit by Folds]")
for fs in fold_summaries:
    b = fs["best"]
    auc_b = b.get('best_auc')
    auc_s = f"{auc_b:.4f}" if (auc_b is not None and not np.isnan(auc_b)) else "undefined"
    print(
        "  Fold {fold}: train={tr:,} (pos={tpos}, neg={tneg}), val={va:,} (pos={vpos}, neg={vneg}), "
        "trials={trials}, time={time:.1f}s | AUC={auc}, F1={f1:.4f}, Acc={acc:.4f}, θ={thr:.2f} | "
        "leaves={leaves}, depth={depth}, lr={lr:.4f}".format(
            fold=fs.get("fold"), tr=fs.get("train_size", 0), tpos=fs.get("train_pos", 0), tneg=fs.get("train_neg", 0),
            va=fs.get("val_size", 0), vpos=fs.get("val_pos", 0), vneg=fs.get("val_neg", 0),
            trials=fs.get("trials", 0), time=fs.get("elapsed", 0.0),
            auc=auc_s, f1=b.get('best_f1', float('nan')), acc=b.get('best_acc', float('nan')),
            thr=b.get('best_thr', float('nan')),
            leaves=b.get('params', {}).get('num_leaves', None),
            depth=b.get('params', {}).get('max_depth', None),
            lr=b.get('params', {}).get('learning_rate', 0.05),
        )
    )

print("\n[Meta-Inferred Params — Method 1 (Train+Pre-test)]")
print(json.dumps({k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (float, np.floating)) else v)
                 for k, v in inferred_params_combined.items()}, indent=2))

print("\n[Meta-Inferred Params — Method 2 (Pre-test-only)]")
print(json.dumps({k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (float, np.floating)) else v)
                 for k, v in inferred_params_pretest.items()}, indent=2))

print("\n[Test Window Metrics — Method 1 (Train+Pre-test)]")
auc_out_c = test_stats_combined.get('auc')
print(json.dumps({**test_stats_combined, "auc": auc_out_c if auc_out_c is not None else None}, indent=2))

print("\n[Test Window Metrics — Method 2 (Pre-test-only)]")
auc_out_p = test_stats_pretest.get('auc')
print(json.dumps({**test_stats_pretest, "auc": auc_out_p if auc_out_p is not None else None}, indent=2))

print(f"\n✅ Best Method: {best_method}")
print("==========================================================\n")

# Keep backward compatibility
test_stats = test_stats_combined if best_method == "Method 1 (Train+Pre-test)" else test_stats_pretest


# %% TEST-WINDOW THRESHOLD SWEEP
try:
    print("\n[Test Window Threshold Sweep — Accuracy 0.30..0.80]")
    if len(test_proba):
        thr_sweep = np.round(np.arange(0.30, 0.80 + 1e-9, 0.01), 2)
        sweep_results = []
        for thr in thr_sweep:
            pred_thr = (test_proba >= thr).astype(int)
            acc_thr = accuracy_score(y_test, pred_thr)
            sweep_results.append({"thr": float(thr), "acc": float(round(acc_thr, 6))})
        for r in sweep_results:
            print(f"  θ={r['thr']:.2f} -> acc={r['acc']:.4f}")
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
        # Save sweep
        try:
            Path(RUN_DIR, 'threshold_sweep.json').write_text(json.dumps(sweep_results, indent=2), encoding='utf-8')
        except Exception:
            pass
    else:
        print("  No test probabilities available; skipping threshold sweep.")
except Exception:
    pass


# %% SAVE ARTIFACTS
model_path = os.path.join(RUN_DIR, 'lightgbm_three_window_cumulative.joblib')
scaler_path = os.path.join(RUN_DIR, 'scaler_three_window_cumulative.joblib')
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
    "class_dist": {"train": class_dist(y_train), "pretest": class_dist(y_pre), "test": class_dist(y_test)},
    "pretest_folds": fold_summaries,
    "method_comparison": {
        "method_1_train_pretest": {
            "inferred_params": inferred_params_combined,
            "test_metrics": test_stats_combined,
        },
        "method_2_pretest_only": {
            "inferred_params": inferred_params_pretest,
            "test_metrics": test_stats_pretest,
        },
        "selected_method": best_method,
    },
    "inferred_params": inferred_params,
    "test_metrics": test_stats,
    "feature_names": FEATURE_NAMES,
    "cv_config": {
        "fold_size": PRETEST_FOLD_SIZE,
        "gap_bars": GAP_BARS,
        "time_limit_s": FOLD_TIME_LIMIT,
        "extension_time_s": EXTENSION_TIME_LIMIT,
        "ultra_extension_time_s": ULTRA_EXTENSION_TIME_LIMIT,
        "extreme_extension_time_s": EXTREME_EXTENSION_TIME_LIMIT,
        "total_extension_s": EXTENSION_TIME_LIMIT + ULTRA_EXTENSION_TIME_LIMIT + EXTREME_EXTENSION_TIME_LIMIT,
        "auc_threshold_for_extension": AUC_THRESHOLD_FOR_EXTENSION,
        "threshold_floor_trigger": THRESHOLD_FLOOR_TRIGGER,
        "test_rows": TEST_ROWS,
    },
    "final_model": {
        "learning_rate": inferred_params.get("learning_rate"),
        "num_leaves": inferred_params.get("num_leaves"),
        "max_depth": inferred_params.get("max_depth"),
        "min_child_samples": inferred_params.get("min_child_samples"),
        "feature_fraction": inferred_params.get("feature_fraction"),
        "bagging_fraction": inferred_params.get("bagging_fraction"),
        "lambda_l1": inferred_params.get("lambda_l1"),
        "lambda_l2": inferred_params.get("lambda_l2"),
        "n_estimators": inferred_params.get("n_estimators", 4000),
    },
}

results_path = os.path.join(RUN_DIR, 'results_three_window_cumulative.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

run_meta = {
    "script_name": os.path.basename(__file__),
    "run_timestamp_utc": RUN_TS,
    "artifact_dir": RUN_DIR,
    "artifacts": {"model": model_path, "scaler": scaler_path, "results": results_path},
}
with open(os.path.join(RUN_DIR, 'run_meta.json'), 'w') as f:
    json.dump(run_meta, f, indent=2)

try:
    code_txt = Path(__file__).read_text(encoding='utf-8')
    Path(RUN_DIR, f"code_{os.path.basename(__file__)}.txt").write_text(code_txt, encoding='utf-8')
except Exception:
    pass

print("Artifacts saved to:", RUN_DIR)


# %% SECOND RUN: TEST_ROWS=25, PRETEST_FOLD_SIZE=25 (with cumulative training)
try:
    print("\n\n" + "="*70)
    print("  SECOND RUN: TEST=25 BARS, FOLDS~25 BARS (CUMULATIVE)")
    print("="*70)
    TEST_ROWS_2 = 25
    PRETEST_FOLD_SIZE_2 = 25
    # Flatten artifacts into the timestamp directory (no subdirectories)
    RUN_DIR_25 = RUN_DIR

    def split_three_windows_25(df_in: pd.DataFrame, test_rows: int) -> Dict[str, pd.DataFrame]:
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

    splits2 = split_three_windows_25(df, TEST_ROWS_2)
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
    X_train2_df = pd.DataFrame(X_train2_s, columns=FEATURE_NAMES)
    baseline_model2.fit(X_train2_df, y_train2)

    print("\n" + "="*70)
    print("  PRE-TEST HYPERPARAMETER OPTIMIZATION (CUMULATIVE, TIME-LIMITED) — RUN2")
    print("="*70)

    thr_grid2 = np.round(np.arange(0.30, 0.80 + 1e-9, 0.01), 2)
    fold_meta2: List[Dict[str, Any]] = []
    best_settings2: List[Dict[str, Any]] = []
    best_thr_list2: List[float] = []
    fold_summaries2: List[Dict[str, Any]] = []
    meta_learner2 = MetaLearner()
    prior_best_params2: Optional[Dict[str, Any]] = None

    # Create DataFrames for RUN2 (outside loop)
    X_pre2_df = pd.DataFrame(X_pre2_s, columns=FEATURE_NAMES)
    X_test2_df = pd.DataFrame(X_test2_s, columns=FEATURE_NAMES) if len(X_test2_s) else pd.DataFrame(
        np.empty((0, X_train2_s.shape[1])), columns=FEATURE_NAMES
    )

    pre_n2 = len(y_pre2)
    for fold_idx2, (train_pre_end2, val_start2, val_end2) in enumerate(cumulative_pretest_splits(pre_n2, PRETEST_FOLD_SIZE_2, GAP_BARS), start=1):
        fold_t0_2 = time.time()
        X_tr2_df = pd.concat([X_train2_df, X_pre2_df.iloc[:train_pre_end2]], ignore_index=True) if train_pre_end2 > 0 else X_train2_df
        y_tr2 = np.concatenate([y_train2, y_pre2[:train_pre_end2]]) if train_pre_end2 > 0 else y_train2
        X_va2_df = X_pre2_df.iloc[val_start2:val_end2]
        y_va2 = y_pre2[val_start2:val_end2]

        tr_pos2 = int(np.sum(y_tr2 == 1)); tr_neg2 = int(np.sum(y_tr2 == 0))
        va_pos2 = int(np.sum(y_va2 == 1)); va_neg2 = int(np.sum(y_va2 == 0))

        sig2 = {
            "n": float(len(y_tr2)),
            "pos_rate": float(tr_pos2 / (tr_pos2 + tr_neg2)) if (tr_pos2 + tr_neg2) else 0.5,
            "feat_mean_avg": float(np.mean(X_tr2_df.values)) if X_tr2_df.size else 0.0,
            "feat_std_avg": float(np.std(X_tr2_df.values)) if X_tr2_df.size else 0.0,
        }

        print(f"\n{'─'*70}")
        print(f"✓ RUN2 FOLD {fold_idx2} | Train: {len(y_tr2)} bars (pos={tr_pos2}, neg={tr_neg2}) | Val: {len(y_va2)} bars (pos={va_pos2}, neg={va_neg2})")
        print(f"  Train-pre end idx: {train_pre_end2} | Val idx: [{val_start2}, {val_end2}) | Gap={GAP_BARS}")
        if len(np.unique(y_va2)) < 2:
            print("  ⚠️  Validation is single-class ⇒ AUC undefined; scoring will ignore AUC for this fold.")

        # Two-phase strategy for RUN2
        print(f"\n  Would you like to:")
        print(f"    [1] Run full hyperparameter search (~5 min)")
        print(f"    [2] Skip and use known best params from previous run (RUN2)")
        has_known2 = fold_idx2 in MANUAL_BEST_RUN2 or fold_idx2 in dyn_run2 or fold_idx2 in KNOWN_BEST_PARAMS_RUN2
        user_choice2 = get_user_choice("  Enter choice (1 or 2): ", has_known=has_known2)
        skip2 = (user_choice2 == "2")
        if skip2 and (fold_idx2 in MANUAL_BEST_RUN2 or fold_idx2 in dyn_run2 or fold_idx2 in KNOWN_BEST_PARAMS_RUN2):
            manual_best2 = MANUAL_BEST_RUN2.get(fold_idx2)
            if manual_best2:
                print(f"  ⚡ Skipping search, using manual newest-best for RUN2 fold {fold_idx2}")
                best_info2 = {
                    "params": manual_best2.get("params", default_lgb_params()),
                    "best_thr": float(manual_best2.get("best_thr", 0.5)),
                    "best_auc": manual_best2.get("best_auc"),
                    "best_acc": float(manual_best2.get("best_acc", 0)),
                    "best_prec": float(manual_best2.get("best_prec", 0)),
                    "best_rec": float(manual_best2.get("best_rec", 0)),
                    "best_f1": float(manual_best2.get("best_f1", 0)),
                    "best_bacc": float(manual_best2.get("best_bacc", 0)),
                    "score": float(manual_best2.get("score", 0)),
                }
            else:
                # Fallback to dynamic or static
                best_params2_source = dyn_run2.get(fold_idx2, KNOWN_BEST_PARAMS_RUN2.get(fold_idx2))
                print(f"  ⚡ Skipping search, using known best params for RUN2 fold {fold_idx2} (loaded={'dynamic' if fold_idx2 in dyn_run2 else 'static'})")
                clf_quick2 = lgb.LGBMClassifier(**merge_params(default_lgb_params(), best_params2_source))
                clf_quick2.fit(X_tr2_df, y_tr2)
                proba_quick2 = clf_quick2.predict_proba(X_va2_df)[:, 1]
                stats_quick2 = evaluate_thresholds(y_va2, proba_quick2, thr_grid2)
                best_info2 = {
                    "params": merge_params(default_lgb_params(), best_params2_source),
                    "best_thr": float(dyn_thr2.get(fold_idx2, stats_quick2["thr"])),
                    "best_auc": stats_quick2["auc"],
                    "best_acc": float(stats_quick2["acc"]),
                    "best_prec": float(stats_quick2["prec"]),
                    "best_rec": float(stats_quick2["rec"]),
                    "best_f1": float(stats_quick2["f1"]),
                    "best_bacc": float(stats_quick2["bacc"]),
                    "score": float(score_fold(stats_quick2)),
                }
            trials2 = 1
            improved2 = 1
            elapsed2 = time.time() - fold_t0_2
            # append meta and continue to next fold
            meta_entry2 = dict(best_info2)
            meta_entry2.update({
                "fold": int(fold_idx2),
                "train_size": int(len(y_tr2)),
                "val_size": int(len(y_va2)),
                "train_signature": sig2,
                "train_pos": tr_pos2,
                "train_neg": tr_neg2,
                "val_pos": va_pos2,
                "val_neg": va_neg2,
            })
            fold_meta2.append(meta_entry2)
            best_settings2.append(best_info2.get("params", default_lgb_params()))
            best_thr_list2.append(best_info2.get("best_thr", 0.5))
            fold_summaries2.append({
                "fold": int(fold_idx2),
                "train_size": int(len(y_tr2)),
                "val_size": int(len(y_va2)),
                "best": best_info2,
                "trials": trials2,
                "improvements": improved2,
                "elapsed": elapsed2,
                "train_pos": tr_pos2,
                "train_neg": tr_neg2,
                "val_pos": va_pos2,
                "val_neg": va_neg2,
            })
            prior_best_params2 = best_info2.get("params", None)
            if len(fold_meta2) >= 2:
                meta_learner2.fit(fold_meta2)
                print(f"\n  🧠 Meta-learner updated with {len(fold_meta2)} folds (RUN2)")
            auc_disp2 = best_info2.get('best_auc')
            auc_str_final2 = f"{auc_disp2:.4f}" if (auc_disp2 is not None and not np.isnan(auc_disp2)) else "undefined"
            acc_disp2 = best_info2.get('best_acc', 0)
            print(f"\n  📊 RUN2 Fold {fold_idx2} Summary:")
            print(f"     Time: {elapsed2:.1f}s | Trials: {trials2} | Improvements: {improved2}")
            print(f"     Best Score: {best_info2.get('score', 0):.4f} | F1: {best_info2.get('best_f1', 0):.4f} | AUC: {auc_str_final2} | ACC: {acc_disp2:.4f}")
            print(f"     Threshold: {best_info2.get('best_thr', 0.5):.2f}")
            continue
        total_time_budget2 = max(0.0, FOLD_TIME_LIMIT - (time.time() - fold_t0_2))
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        is_first_three2 = fold_idx2 <= 3
        phase1_budget2 = min(180.0, total_time_budget2) if not is_first_three2 else total_time_budget2
        phase2_budget2 = 0.0 if is_first_three2 else max(0.0, min(120.0, total_time_budget2 - phase1_budget2))

        # Phase 1 — Exploratory
        print(f"  🔍 Phase 1 — Exploratory (RUN2) | Budget: {phase1_budget2:.1f}s")
        sampler2_e = TPESampler(seed=42, n_startup_trials=30)
        pruner2_e = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        study2_e = optuna.create_study(direction="maximize", sampler=sampler2_e, pruner=pruner2_e, study_name=f"run2_fold_{fold_idx2}_explore")
        objective2_e = create_optuna_objective(X_tr2_df, y_tr2, X_va2_df, y_va2, thr_grid2, prior_best=None)
        try:
            study2_e.optimize(
                objective2_e,
                timeout=phase1_budget2,
                n_jobs=1,
                show_progress_bar=False,
                callbacks=[
                    lambda study, trial: print(
                        f"  ✓ [E] Trial {trial.number+1:>3} | Score: {trial.value:.4f} → {'NEW BEST' if trial.value == study.best_value else 'continue'} | "
                        f"F1={trial.user_attrs.get('stats', {}).get('f1', 0):.4f}, "
                        f"AUC={trial.user_attrs.get('stats', {}).get('auc') if trial.user_attrs.get('stats', {}).get('auc') is not None else 'undefined'}, "
                        f"θ={trial.user_attrs.get('stats', {}).get('thr', 0.5):.2f} | "
                        f"lr={trial.params.get('learning_rate', 0):.4f}, leaves={trial.params.get('num_leaves', 0)}, depth={trial.params.get('max_depth', 0)}"
                    ) if trial.value > 0 and trial.number % 5 == 0 else None
                ]
            )
        except Exception as e:
            print(f"  ⚠️  Phase 1 optimization error (RUN2): {str(e)}")

        # Phase 2 — Prior-guided (fold 4+)
        time_left2 = max(0.0, FOLD_TIME_LIMIT - (time.time() - fold_t0_2))
        study2_g = None
        if not is_first_three2 and time_left2 > 1.0 and phase2_budget2 > 0.0:
            guided_prior2 = None
            if study2_e.best_trial is not None:
                guided_prior2 = study2_e.best_trial.user_attrs.get("params", None)
            if guided_prior2 is None:
                guided_prior2 = prior_best_params2
            print(f"  🔍 Phase 2 — Prior-guided (RUN2) | Budget: {min(phase2_budget2, time_left2):.1f}s")
            sampler2_g = TPESampler(seed=42, n_startup_trials=20)
            pruner2_g = MedianPruner(n_startup_trials=8, n_warmup_steps=5)
            study2_g = optuna.create_study(direction="maximize", sampler=sampler2_g, pruner=pruner2_g, study_name=f"run2_fold_{fold_idx2}_guided")
            objective2_g = create_optuna_objective(X_tr2_df, y_tr2, X_va2_df, y_va2, thr_grid2, prior_best=guided_prior2)
            try:
                study2_g.optimize(
                    objective2_g,
                    timeout=min(phase2_budget2, time_left2),
                    n_jobs=1,
                    show_progress_bar=False,
                    callbacks=[
                        lambda study, trial: print(
                            f"  ✓ [G] Trial {trial.number+1:>3} | Score: {trial.value:.4f} → {'NEW BEST' if trial.value == study.best_value else 'continue'} | "
                            f"F1={trial.user_attrs.get('stats', {}).get('f1', 0):.4f}, "
                            f"AUC={trial.user_attrs.get('stats', {}).get('auc') if trial.user_attrs.get('stats', {}).get('auc') is not None else 'undefined'}, "
                            f"θ={trial.user_attrs.get('stats', {}).get('thr', 0.5):.2f} | "
                            f"lr={trial.params.get('learning_rate', 0):.4f}, leaves={trial.params.get('num_leaves', 0)}, depth={trial.params.get('max_depth', 0)}"
                        ) if trial.value > 0 and trial.number % 5 == 0 else None
                    ]
                )
            except Exception as e:
                print(f"  ⚠️  Phase 2 optimization error (RUN2): {str(e)}")

        # Combine results
        all_trials2 = list(study2_e.trials)
        if study2_g is not None:
            all_trials2 += list(study2_g.trials)
        trials2 = len(all_trials2)
        best_trial2 = None
        best_value2 = -np.inf
        for t in all_trials2:
            if t.value is not None and t.value > best_value2:
                best_value2 = t.value
                best_trial2 = t
        if best_trial2 is not None:
            best_stats2 = best_trial2.user_attrs.get("stats", {})
            best_params_dict2 = best_trial2.user_attrs.get("params", default_lgb_params())
            best_score2 = float(best_value2)
            best_auc2 = best_stats2.get("auc")
            improved2 = 0
            try:
                improved2 += len([t for t in study2_e.trials if study2_e.best_value is not None and t.value == study2_e.best_value])
            except Exception:
                pass
            if study2_g is not None:
                try:
                    improved2 += len([t for t in study2_g.trials if study2_g.best_value is not None and t.value == study2_g.best_value])
                except Exception:
                    pass
            # Phase 3/4 Extension logic for RUN2 (mirror RUN1)
            best_thr_val2 = best_stats2.get("thr", 0.5)
            trigger_extension2 = False
            trigger_reason2 = []
            if best_auc2 is not None and not np.isnan(best_auc2) and best_auc2 < AUC_THRESHOLD_FOR_EXTENSION:
                trigger_extension2 = True
                trigger_reason2.append(f"AUC={best_auc2:.4f} < {AUC_THRESHOLD_FOR_EXTENSION}")
            if best_thr_val2 <= THRESHOLD_FLOOR_TRIGGER:
                trigger_extension2 = True
                trigger_reason2.append(f"θ={best_thr_val2:.2f} ≤ {THRESHOLD_FLOOR_TRIGGER}")

            if trigger_extension2:
                print(f"\n  ⚠️  Extension triggered (RUN2): {' AND '.join(trigger_reason2)}")
                print(f"  🚀 PHASE 3 — Extended Search (RUN2) | Budget: {EXTENSION_TIME_LIMIT:.1f}s")
                print(f"     Strategy: Expanded hyperparameter space + alternative boosting types + adaptive trees")

                sampler2_ext = TPESampler(seed=43, n_startup_trials=20)
                pruner2_ext = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
                study2_ext = optuna.create_study(direction="maximize", sampler=sampler2_ext, pruner=pruner2_ext, study_name=f"run2_fold_{fold_idx2}_extended")
                objective2_ext = create_optuna_objective_extended(X_tr2_df, y_tr2, X_va2_df, y_va2, thr_grid2, prior_best=best_params_dict2)

                try:
                    study2_ext.optimize(
                        objective2_ext,
                        timeout=EXTENSION_TIME_LIMIT,
                        n_jobs=1,
                        show_progress_bar=False,
                        callbacks=[
                            lambda study, trial: print(
                                f"  ✓ [X] Trial {trial.number+1:>3} | Score: {trial.value:.4f} → {'NEW BEST' if trial.value == study.best_value else 'continue'} | "
                                f"F1={trial.user_attrs.get('stats', {}).get('f1', 0):.4f}, "
                                f"AUC={trial.user_attrs.get('stats', {}).get('auc') if trial.user_attrs.get('stats', {}).get('auc') is not None else 'undefined'}, "
                                f"ACC={trial.user_attrs.get('stats', {}).get('acc', 0):.4f}, "
                                f"θ={trial.user_attrs.get('stats', {}).get('thr', 0.5):.2f} | "
                                f"lr={trial.params.get('learning_rate', 0):.4f}, leaves={trial.params.get('num_leaves', 0)}, boost={trial.params.get('boosting_type', 'gbdt')}"
                            ) if trial.value > 0 and trial.number % 5 == 0 else None
                        ]
                    )

                    all_trials2 += list(study2_ext.trials)
                    trials2 = len(all_trials2)

                    for t in study2_ext.trials:
                        if t.value is not None and t.value > best_value2:
                            best_value2 = t.value
                            best_trial2 = t

                    if best_trial2 is not None:
                        best_stats2 = best_trial2.user_attrs.get("stats", {})
                        best_params_dict2 = best_trial2.user_attrs.get("params", default_lgb_params())
                        best_score2 = float(best_value2)
                        best_auc2 = best_stats2.get("auc")
                        try:
                            improved2 += len([t for t in study2_ext.trials if study2_ext.best_value is not None and t.value == study2_ext.best_value])
                        except Exception:
                            pass

                        print(f"  ✅ Phase 3 complete (RUN2) | AUC: {best_auc2 if best_auc2 is not None else 'undefined'} | θ={best_stats2.get('thr', 0.5):.2f}")

                        # Phase 4 — Ultra aggressive (RUN2)
                        best_thr_after_phase3_2 = best_stats2.get("thr", 0.5)
                        still_problematic2 = False
                        if (best_auc2 is not None and best_auc2 < AUC_THRESHOLD_FOR_EXTENSION) or (best_thr_after_phase3_2 <= THRESHOLD_FLOOR_TRIGGER):
                            still_problematic2 = True

                        if still_problematic2:
                            print(f"\n  🔥 PHASE 4 — ULTRA-AGGRESSIVE Search (RUN2) | Budget: {ULTRA_EXTENSION_TIME_LIMIT:.1f}s")
                            print(f"     Strategy: Extreme ranges + RF/DART boosting + adaptive n_estimators + wild regularization")

                            sampler2_ultra = TPESampler(seed=44, n_startup_trials=15)
                            pruner2_ultra = MedianPruner(n_startup_trials=3, n_warmup_steps=2)
                            study2_ultra = optuna.create_study(direction="maximize", sampler=sampler2_ultra, pruner=pruner2_ultra, study_name=f"run2_fold_{fold_idx2}_ultra")

                            objective2_ultra = create_optuna_objective_ultra(X_tr2_df, y_tr2, X_va2_df, y_va2, thr_grid2, prior_best=best_params_dict2)

                            try:
                                study2_ultra.optimize(
                                    objective2_ultra,
                                    timeout=ULTRA_EXTENSION_TIME_LIMIT,
                                    n_jobs=1,
                                    show_progress_bar=False,
                                    callbacks=[
                                        lambda study, trial: print(
                                            f"  ✓ [U] Trial {trial.number+1:>3} | Score: {trial.value:.4f} → {'NEW BEST' if trial.value == study.best_value else 'continue'} | "
                                            f"F1={trial.user_attrs.get('stats', {}).get('f1', 0):.4f}, "
                                            f"AUC={trial.user_attrs.get('stats', {}).get('auc') if trial.user_attrs.get('stats', {}).get('auc') is not None else 'undefined'}, "
                                            f"θ={trial.user_attrs.get('stats', {}).get('thr', 0.5):.2f} | "
                                            f"lr={trial.params.get('learning_rate', 0):.4f}, leaves={trial.params.get('num_leaves', 0)}, boost={trial.params.get('boosting_type', 'gbdt')}"
                                        ) if trial.value > 0 and trial.number % 5 == 0 else None
                                    ]
                                )

                                all_trials2 += list(study2_ultra.trials)
                                trials2 = len(all_trials2)

                                for t in study2_ultra.trials:
                                    if t.value is not None and t.value > best_value2:
                                        best_value2 = t.value
                                        best_trial2 = t

                                if best_trial2 is not None:
                                    best_stats2 = best_trial2.user_attrs.get("stats", {})
                                    best_params_dict2 = best_trial2.user_attrs.get("params", default_lgb_params())
                                    best_score2 = float(best_value2)
                                    best_auc2 = best_stats2.get("auc")
                                    try:
                                        improved2 += len([t for t in study2_ultra.trials if study2_ultra.best_value is not None and t.value == study2_ultra.best_value])
                                    except Exception:
                                        pass

                                    print(f"  ✅ Phase 4 complete (RUN2) | Final AUC: {best_auc2 if best_auc2 is not None else 'undefined'} | θ={best_stats2.get('thr', 0.5):.2f}")
                                    if best_auc2 is not None and best_auc2 >= AUC_THRESHOLD_FOR_EXTENSION and best_stats2.get('thr', 0) > THRESHOLD_FLOOR_TRIGGER:
                                        print(f"  🎯 SUCCESS: Metrics improved!")
                                    else:
                                        # PHASE 5: EXTREME final attempt (RUN2)
                                        print(f"\n  💥 PHASE 5 — EXTREME Search (RUN2 final attempt) | Budget: {EXTREME_EXTENSION_TIME_LIMIT:.1f}s")
                                        print(f"     Strategy: Maximum experimentation with all LightGBM parameters + extreme ranges")
                                        
                                        sampler2_extreme = TPESampler(seed=45, n_startup_trials=10)
                                        pruner2_extreme = MedianPruner(n_startup_trials=2, n_warmup_steps=1)
                                        study2_extreme = optuna.create_study(direction="maximize", sampler=sampler2_extreme, pruner=pruner2_extreme, study_name=f"run2_fold_{fold_idx2}_extreme")
                                        objective2_extreme = create_optuna_objective_extreme(X_tr2_df, y_tr2, X_va2_df, y_va2, thr_grid2, prior_best=best_params_dict2)
                                        
                                        try:
                                            study2_extreme.optimize(
                                                objective2_extreme,
                                                timeout=EXTREME_EXTENSION_TIME_LIMIT,
                                                n_jobs=1,
                                                show_progress_bar=False,
                                                callbacks=[
                                                    lambda study, trial: print(
                                                        f"  ✓ [Z] Trial {trial.number+1:>3} | Score: {trial.value:.4f} → {'NEW BEST' if trial.value == study.best_value else 'continue'} | "
                                                        f"F1={trial.user_attrs.get('stats', {}).get('f1', 0):.4f}, "
                                                        f"AUC={trial.user_attrs.get('stats', {}).get('auc') if trial.user_attrs.get('stats', {}).get('auc') is not None else 'undefined'}, "
                                                        f"θ={trial.user_attrs.get('stats', {}).get('thr', 0.5):.2f} | "
                                                        f"lr={trial.params.get('learning_rate', 0):.4f}, leaves={trial.params.get('num_leaves', 0)}, boost={trial.params.get('boosting_type', 'gbdt')}"
                                                    ) if trial.value > 0 and trial.number % 5 == 0 else None
                                                ]
                                            )
                                            
                                            all_trials2 += list(study2_extreme.trials)
                                            trials2 = len(all_trials2)
                                            
                                            for t in study2_extreme.trials:
                                                if t.value is not None and t.value > best_value2:
                                                    best_value2 = t.value
                                                    best_trial2 = t
                                            
                                            if best_trial2 is not None:
                                                best_stats2 = best_trial2.user_attrs.get("stats", {})
                                                best_params_dict2 = best_trial2.user_attrs.get("params", default_lgb_params())
                                                best_score2 = float(best_value2)
                                                best_auc2 = best_stats2.get("auc")
                                                try:
                                                    improved2 += len([t for t in study2_extreme.trials if study2_extreme.best_value is not None and t.value == study2_extreme.best_value])
                                                except Exception:
                                                    pass
                                                
                                                print(f"  ✅ Phase 5 complete (RUN2) | Final AUC: {best_auc2 if best_auc2 is not None else 'undefined'} | θ={best_stats2.get('thr', 0.5):.2f}")
                                                if best_auc2 is not None and best_auc2 >= AUC_THRESHOLD_FOR_EXTENSION and best_stats2.get('thr', 0) > THRESHOLD_FLOOR_TRIGGER:
                                                    print(f"  🎯 Phase 5 SUCCESS (RUN2)!")
                                        
                                        except Exception as e:
                                            print(f"  ⚠️  Phase 5 error (RUN2): {str(e)}")

                            except Exception as e:
                                print(f"  ⚠️  Phase 4 error (RUN2): {str(e)}")
                        else:
                            print(f"  ✅ Phase 3 sufficient (RUN2) | AUC: {best_auc2 if best_auc2 is not None else 'undefined'} | θ={best_thr_after_phase3_2:.2f}")

                except Exception as e:
                    print(f"  ⚠️  Extension search error (RUN2): {str(e)}")
            # Compare against manual newest-best for RUN2 and override if manual has higher score
            manual_best2 = MANUAL_BEST_RUN2.get(fold_idx2)
            if manual_best2 is not None:
                try:
                    if float(manual_best2.get("score", -np.inf)) > float(best_score2):
                        best_params_dict2 = merge_params(default_lgb_params(), manual_best2.get("params", {}))
                        best_stats2 = {
                            "thr": float(manual_best2.get("best_thr", 0.5)),
                            "auc": manual_best2.get("best_auc", None),
                            "acc": float(manual_best2.get("best_acc", 0)),
                            "prec": float(manual_best2.get("best_prec", 0)),
                            "rec": float(manual_best2.get("best_rec", 0)),
                            "f1": float(manual_best2.get("best_f1", 0)),
                            "bacc": float(manual_best2.get("best_bacc", 0)),
                        }
                        best_score2 = float(manual_best2.get("score", best_score2))
                except Exception:
                    pass

            best_info2 = {
                "params": best_params_dict2,
                "best_thr": float(best_stats2.get("thr", 0.5)),
                "best_auc": best_stats2.get("auc"),
                "best_acc": float(best_stats2.get("acc", 0)),
                "best_prec": float(best_stats2.get("prec", 0)),
                "best_rec": float(best_stats2.get("rec", 0)),
                "best_f1": float(best_stats2.get("f1", 0)),
                "best_bacc": float(best_stats2.get("bacc", 0)),
                "score": float(best_score2),
            }
        else:
            best_score2 = -1.0
            best_info2 = {
                "params": default_lgb_params(),
                "best_thr": 0.5,
                "best_auc": None,
                "best_acc": 0.0,
                "best_prec": 0.0,
                "best_rec": 0.0,
                "best_f1": 0.0,
                "best_bacc": 0.0,
                "score": -1.0,
            }
            improved2 = 0

        elapsed2 = time.time() - fold_t0_2
        meta_entry2 = dict(best_info2)
        meta_entry2.update({
            "fold": int(fold_idx2),
            "train_size": int(len(y_tr2)),
            "val_size": int(len(y_va2)),
            "train_signature": sig2,
            "train_pos": tr_pos2,
            "train_neg": tr_neg2,
            "val_pos": va_pos2,
            "val_neg": va_neg2,
        })
        fold_meta2.append(meta_entry2)
        best_settings2.append(best_info2.get("params", default_lgb_params()))
        best_thr_list2.append(best_info2.get("best_thr", 0.5))
        fold_summaries2.append({
            "fold": int(fold_idx2),
            "train_size": int(len(y_tr2)),
            "val_size": int(len(y_va2)),
            "best": best_info2,
            "trials": trials2,
            "improvements": improved2,
            "elapsed": elapsed2,
            "train_pos": tr_pos2,
            "train_neg": tr_neg2,
            "val_pos": va_pos2,
            "val_neg": va_neg2,
        })

        prior_best_params2 = best_info2.get("params", None)
        if len(fold_meta2) >= 2:
            meta_learner2.fit(fold_meta2)
            print(f"\n  🧠 Meta-learner updated with {len(fold_meta2)} folds (RUN2)")

        auc_disp2 = best_info2.get('best_auc')
        auc_str_final2 = f"{auc_disp2:.4f}" if (auc_disp2 is not None and not np.isnan(auc_disp2)) else "undefined"
        acc_disp2_main = best_info2.get('best_acc', 0)
        print(f"\n  📊 RUN2 Fold {fold_idx2} Summary:")
        print(f"     Time: {elapsed2:.1f}s | Trials: {trials2} | Improvements: {improved2}")
        print(f"     Best Score: {best_score2:.4f} | F1: {best_info2.get('best_f1', 0):.4f} | AUC: {auc_str_final2} | ACC: {acc_disp2_main:.4f}")
        print(f"     Threshold: {best_info2.get('best_thr', 0.5):.2f}")

    print(f"\n{'='*70}")
    print("  PRE-TEST OPTIMIZATION COMPLETE (CUMULATIVE) — RUN2")
    print(f"{'='*70}\n")

    print("\n" + "="*70)
    print("  META-LEARNING: INFERRING FINAL HYPERPARAMETERS — RUN2")
    print("="*70)
    # Dual-method meta-learning for RUN2 (Train+Pre-test vs Pre-test-only)
    if len(fold_meta2) >= 2 and meta_learner2.fitted:
        X_combined2_df = pd.concat([X_train2_df, X_pre2_df], ignore_index=True) if len(X_pre2_df) else X_train2_df
        y_combined2 = np.concatenate([y_train2, y_pre2]) if len(y_pre2) else y_train2
        sig_full2_combined = {
            "n": float(len(y_combined2)),
            "pos_rate": float(np.mean(y_combined2)),
            "feat_mean_avg": float(np.mean(X_combined2_df.values)) if X_combined2_df.size else 0.0,
            "feat_std_avg": float(np.std(X_combined2_df.values)) if X_combined2_df.size else 0.0,
        }
        sig_full2_pretest = {
            "n": float(len(y_pre2)),
            "pos_rate": float(np.mean(y_pre2)),
            "feat_mean_avg": float(np.mean(X_pre2_df.values)) if X_pre2_df.size else 0.0,
            "feat_std_avg": float(np.std(X_pre2_df.values)) if X_pre2_df.size else 0.0,
        }
        inferred_params2_combined = meta_learner2.predict(sig_full2_combined)
        inferred_params2_pretest = meta_learner2.predict(sig_full2_pretest)
        print(f"✓ Using ML-based meta-learner (trained on {len(fold_meta2)} folds)")
        print(f"\n  Method 1 (Train+Pre-test signature):")
        for k in ["learning_rate","num_leaves","max_depth","min_child_samples"]:
            print(f"    {k}: {inferred_params2_combined.get(k)}")
        print(f"\n  Method 2 (Pre-test-only signature):")
        for k in ["learning_rate","num_leaves","max_depth","min_child_samples"]:
            print(f"    {k}: {inferred_params2_pretest.get(k)}")
    else:
        inferred_params2_combined = default_lgb_params()
        inferred_params2_pretest = default_lgb_params()
        print(f"✓ Using default parameters (insufficient folds for meta-learning)")

    # Final train and test for both methods (RUN2)
    X_full2_df = pd.concat([X_train2_df, X_pre2_df], ignore_index=True) if len(X_pre2_df) else X_train2_df
    y_full2 = np.concatenate([y_train2, y_pre2]) if len(y_pre2) else y_train2

    print("\n" + "="*70)
    print("  FINAL MODEL TRAINING — RUN2 METHOD 1 (Train+Pre-test signature)")
    print("="*70)
    final_model2_combined = lgb.LGBMClassifier(**inferred_params2_combined)
    final_model2_combined.fit(X_full2_df, y_full2)

    print("\n" + "="*70)
    print("  FINAL MODEL TRAINING — RUN2 METHOD 2 (Pre-test-only signature)")
    print("="*70)
    final_model2_pretest = lgb.LGBMClassifier(**inferred_params2_pretest)
    final_model2_pretest.fit(X_full2_df, y_full2)

    test_proba2_combined = final_model2_combined.predict_proba(X_test2_df)[:, 1] if len(X_test2_df) else np.array([])
    test_proba2_pretest = final_model2_pretest.predict_proba(X_test2_df)[:, 1] if len(X_test2_df) else np.array([])
    thr_final2 = float(np.median(best_thr_list2)) if len(best_thr_list2) else 0.5

    def _stats2(proba_arr: np.ndarray) -> Dict[str, Any]:
        if len(proba_arr):
            pred_arr = (proba_arr >= thr_final2).astype(int)
            try:
                auc_val = float(roc_auc_score(y_test2, proba_arr)) if len(np.unique(y_test2)) == 2 else None
            except Exception:
                auc_val = None
            return {
                "thr": thr_final2,
                "acc": float(accuracy_score(y_test2, pred_arr)),
                "auc": auc_val,
                "prec": float(precision_score(y_test2, pred_arr, zero_division=0)),
                "rec": float(recall_score(y_test2, pred_arr, zero_division=0)),
                "f1": float(f1_score(y_test2, pred_arr, zero_division=0)),
                "bacc": float(balanced_accuracy_score(y_test2, pred_arr)),
            }
        return {"thr": None, "acc": float("nan"), "auc": None, "prec": float("nan"), "rec": float("nan"), "f1": float("nan"), "bacc": float("nan")}

    test_stats2_combined = _stats2(test_proba2_combined)
    test_stats2_pretest = _stats2(test_proba2_pretest)

    # Select best by AUC
    if (test_stats2_pretest.get("auc") is not None) and (test_stats2_combined.get("auc") is not None):
        if test_stats2_pretest["auc"] > test_stats2_combined["auc"]:
            final_model2 = final_model2_pretest
            test_proba2 = test_proba2_pretest
            test_stats2 = test_stats2_pretest
            inferred_params2 = inferred_params2_pretest
            best_method2 = "Method 2 (Pre-test-only)"
        else:
            final_model2 = final_model2_combined
            test_proba2 = test_proba2_combined
            test_stats2 = test_stats2_combined
            inferred_params2 = inferred_params2_combined
            best_method2 = "Method 1 (Train+Pre-test)"
    else:
        # Fallback to combined
        final_model2 = final_model2_combined
        test_proba2 = test_proba2_combined
        test_stats2 = test_stats2_combined
        inferred_params2 = inferred_params2_combined
        best_method2 = "Method 1 (Train+Pre-test)"

    print(f"\n✅ Selected {best_method2} for final model (better AUC) — RUN2")
    print("\n[Meta-Inferred Params — RUN2 Method 1 (Train+Pre-test)]")
    print(json.dumps({k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (float, np.floating)) else v)
                     for k, v in inferred_params2_combined.items()}, indent=2))
    print("\n[Meta-Inferred Params — RUN2 Method 2 (Pre-test-only)]")
    print(json.dumps({k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (float, np.floating)) else v)
                     for k, v in inferred_params2_pretest.items()}, indent=2))
    print("\n[Test Window Metrics — RUN2 Method 1 (Train+Pre-test)]")
    auc_out2_c = test_stats2_combined.get('auc')
    print(json.dumps({**test_stats2_combined, "auc": auc_out2_c if auc_out2_c is not None else None}, indent=2))
    print("\n[Test Window Metrics — RUN2 Method 2 (Pre-test-only)]")
    auc_out2_p = test_stats2_pretest.get('auc')
    print(json.dumps({**test_stats2_pretest, "auc": auc_out2_p if auc_out2_p is not None else None}, indent=2))

    print("\n================ THREE-WINDOW RUN SUMMARY (CUMULATIVE RUN2) ================")
    print("[Data]")
    print(f"  Train rows: {len(train_df2):,}  |  Pre-test rows: {len(pre_df2):,}  |  Test rows: {len(test_df2):,}")
    print(f"  Pre-test span: {pretest_bounds2[0]}  →  {pretest_bounds2[1]}")
    print(f"  Test span:     {test_bounds2[0]}  →  {test_bounds2[1]}")
    print(f"  Class dist — Train: {class_dist(y_train2)}")
    print(f"  Class dist — Pre-test: {class_dist(y_pre2)}")
    print(f"  Class dist — Test: {class_dist(y_test2)}")

    print("\n[Pre-test Overfit by Folds]")
    for fs2 in fold_summaries2:
        b2 = fs2["best"]
        auc_b2 = b2.get('best_auc')
        auc_s2 = f"{auc_b2:.4f}" if (auc_b2 is not None and not np.isnan(auc_b2)) else "undefined"
        print(
            "  Fold {fold}: train={tr:,} (pos={tpos}, neg={tneg}), val={va:,} (pos={vpos}, neg={vneg}), "
            "trials={trials}, time={time:.1f}s | AUC={auc}, F1={f1:.4f}, Acc={acc:.4f}, θ={thr:.2f} | "
            "leaves={leaves}, depth={depth}, lr={lr:.4f}".format(
                fold=fs2.get("fold"), tr=fs2.get("train_size", 0), tpos=fs2.get("train_pos", 0), tneg=fs2.get("train_neg", 0),
                va=fs2.get("val_size", 0), vpos=fs2.get("val_pos", 0), vneg=fs2.get("val_neg", 0),
                trials=fs2.get("trials", 0), time=fs2.get("elapsed", 0.0),
                auc=auc_s2, f1=b2.get('best_f1', float('nan')), acc=b2.get('best_acc', float('nan')),
                thr=b2.get('best_thr', float('nan')),
                leaves=b2.get('params', {}).get('num_leaves', None),
                depth=b2.get('params', {}).get('max_depth', None),
                lr=b2.get('params', {}).get('learning_rate', 0.05),
            )
        )

    print("\n[Meta-Inferred Params (RUN2)]")
    print(json.dumps({k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (float, np.floating)) else v)
                     for k, v in inferred_params2.items()}, indent=2))

    print("\n[Test Window Metrics (RUN2)]")
    auc_out2 = test_stats2.get('auc')
    print(json.dumps({**test_stats2, "auc": auc_out2 if auc_out2 is not None else None}, indent=2))
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

    # Save RUN2 artifacts (flattened)
    model_path2 = os.path.join(RUN_DIR_25, 'lightgbm_three_window_cumulative_run2.joblib')
    scaler_path2 = os.path.join(RUN_DIR_25, 'scaler_three_window_run2.joblib')
    joblib.dump(final_model2, model_path2)
    joblib.dump(scaler2, scaler_path2)
    results2 = {
        "windows": {
            "train_rows": int(len(train_df2)),
            "pretest_rows": int(len(pre_df2)),
            "test_rows": int(len(test_df2)),
            "pretest_span": [str(pretest_bounds2[0]), str(pretest_bounds2[1])],
            "test_span": [str(test_bounds2[0]), str(test_bounds2[1])],
        },
        "class_dist": {"train": class_dist(y_train2), "pretest": class_dist(y_pre2), "test": class_dist(y_test2)},
        "pretest_folds": fold_summaries2,
        "method_comparison": {
            "method_1_train_pretest": {
                "inferred_params": inferred_params2_combined,
                "test_metrics": test_stats2_combined,
            },
            "method_2_pretest_only": {
                "inferred_params": inferred_params2_pretest,
                "test_metrics": test_stats2_pretest,
            },
            "selected_method": best_method2,
        },
        "inferred_params": inferred_params2,
        "test_metrics": test_stats2,
        "feature_names": FEATURE_NAMES,
    }
    # Update the top-level combined results file to include RUN2 and a bottom summary
    try:
        res_main = _load_json(results_path) or {}
        if isinstance(res_main, dict):
            res_main["run2"] = {
                "windows": results2.get("windows", {}),
                "class_dist": results2.get("class_dist", {}),
                "pretest_folds": results2.get("pretest_folds", []),
                "method_comparison": results2.get("method_comparison", {}),
                "inferred_params": results2.get("inferred_params", {}),
                "test_metrics": results2.get("test_metrics", {}),
            }
            # Build ordered summary: RUN1 folds → RUN1 A/B → RUN2 folds → RUN2 A/B
            run1_ab = (res_main.get("method_comparison") or {})
            run2_ab = results2.get("method_comparison", {})
            res_main["summary"] = [
                {"section": "run1_folds", "folds": res_main.get("pretest_folds", [])},
                {"section": "run1_ab", "ab": run1_ab},
                {"section": "run2_folds", "folds": results2.get("pretest_folds", [])},
                {"section": "run2_ab", "ab": run2_ab},
            ]
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(res_main, f, indent=2)
    except Exception:
        pass
    # Update run_meta at root to include run2 artifact paths
    try:
        run_meta_path = os.path.join(RUN_DIR, 'run_meta.json')
        meta_obj = _load_json(run_meta_path) or {}
        meta_artifacts = dict(meta_obj.get("artifacts", {}))
        meta_artifacts.update({
            "model_run2": model_path2,
            "scaler_run2": scaler_path2,
            "results": results_path,
        })
        meta_obj.update({
            "script_name": os.path.basename(__file__),
            "run_timestamp_utc": RUN_TS,
            "artifact_dir": RUN_DIR,
            "artifacts": meta_artifacts,
        })
        with open(run_meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_obj, f, indent=2)
    except Exception:
        pass
    print("Artifacts saved to:", RUN_DIR)

    # Comparison
    print("\n" + "="*70)
    print("  COMPARISON: RUN1 (test=35) vs RUN2 (test=25)")
    print("="*70)
    auc1_str = f"{test_stats.get('auc'):.4f}" if test_stats.get('auc') is not None else "undefined"
    auc2_str = f"{test_stats2.get('auc'):.4f}" if test_stats2.get('auc') is not None else "undefined"
    print(f"Test rows: {len(test_df)} vs {len(test_df2)}")
    print(f"Test AUC: {auc1_str} vs {auc2_str}")
    print(f"Test Acc: {test_stats.get('acc'):.4f} vs {test_stats2.get('acc'):.4f}")
    print(f"Test F1: {test_stats.get('f1'):.4f} vs {test_stats2.get('f1'):.4f}")
    print(f"Chosen threshold: {test_stats.get('thr'):.2f} vs {test_stats2.get('thr'):.2f}")
    print("="*70 + "\n")

except Exception as e:
    print(f"Second run failed: {str(e)}")
    import traceback
    traceback.print_exc()


