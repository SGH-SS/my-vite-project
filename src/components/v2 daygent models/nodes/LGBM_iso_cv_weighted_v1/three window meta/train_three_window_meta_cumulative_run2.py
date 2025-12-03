#!/usr/bin/env python3
"""
LightGBM 1D Training — Three-Window Meta-Optimization (Cumulative Folds) — RUN2 ONLY

This script performs only RUN2: TEST=25 BARS, PRETEST_FOLD_SIZE=25 BARS

Key improvements over train_three_window_meta.py:
- Cumulative pre-test fold training: For each fold, training = Train window + all pre-test bars strictly before the fold block (with purge). Validation = current pre-test fold block (after purge).
- AUC guards: If validation labels are single-class, AUC is treated as undefined and not coerced to 0.5; scoring shifts to F1/balanced accuracy.
- Per-fold label logging: Explicitly prints train/val class counts and flags single-class validation.
- Exact last-N test rows: test window is always the last TEST_ROWS rows.

Configuration-specific hardcoded bests:
- 1T_0G: 1 test window, 0 gaps (original default)
- 1T_1G: 1 test window, 1 gap 
- 1T_2G: 1 test window, 2 gaps
- Dynamic selection based on NUM_TEST_WINDOWS and NUM_GAP_WINDOWS
"""

from __future__ import annotations

import os
# Silence joblib/loky core-count warning on Windows - must be set before any joblib imports
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import warnings
# Proactively silence noisy loky physical core warning if any upstream import preloaded joblib
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    category=UserWarning,
)
import json
import time
import argparse
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import threading
import sys
import itertools
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
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import joblib
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner


# %% COMMAND LINE ARGUMENTS
def parse_arguments():
    """Parse command line arguments for test windows, gaps, and auto-choice."""
    parser = argparse.ArgumentParser(
        description="LightGBM 1D Training with Three-Window Meta-Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py                    # Interactive mode (prompts for inputs)
  python script.py 1 0 4             # 1 test window, 0 gaps, auto-choice 4 (skip all)
  python script.py 1 1 3             # 1 test window, 1 gap, auto-choice 3 (search all)
  python script.py 2 0 2             # 2 test windows, 0 gaps, auto-choice 2 (skip when known)

Auto-choice options:
  1 = Search (run hyperparameter optimization)
  2 = Skip (use stored best params when available)
  3 = Search All (force search for all folds)
  4 = Skip All (force skip for all folds)
        """
    )
    parser.add_argument('test_windows', nargs='?', type=int, default=None,
                       help='Number of test windows (default: 1, prompted if not provided)')
    parser.add_argument('gap_windows', nargs='?', type=int, default=None,
                       help='Number of gap windows to ignore (default: 0, prompted if not provided)')
    parser.add_argument('auto_choice', nargs='?', type=str, default=None,
                       help='Auto-choice for fold optimization (1=search, 2=skip, 3=search all, 4=skip all)')
    
    return parser.parse_args()

# Parse command line arguments
ARGS = parse_arguments()

# %% CONFIG
DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5433/trading_data"
BACKTEST_TABLE = 'backtest.spy_1d'
FRONTTEST_TABLE = 'fronttest.spy_1d'

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'artifacts_lgbm_1d_iso_three_window_cumulative_run2')
os.makedirs(ARTIFACT_DIR, exist_ok=True)
RUNS_DIR = os.path.join(ARTIFACT_DIR, 'runs')
os.makedirs(RUNS_DIR, exist_ok=True)
RUN_TS = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
RUN_DIR = os.path.join(RUNS_DIR, RUN_TS)
os.makedirs(RUN_DIR, exist_ok=True)

# Silence sklearn MLP convergence warnings (we control regularization/early stopping)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Hard-coded best fold params organized by configuration
# Keys are fold validation timestamp ranges: "YYYY-MM-DD_YYYY-MM-DD"

# Configuration: 1 test window, 0 gaps (original RUN2 default)
HARDCODED_BESTS_1T_0G: Dict[str, Dict[str, Any]] = {
    "2024-09-05_2024-10-02": {
        "params": {
            "learning_rate": 0.11959430052714536,
            "num_leaves": 30,
            "max_depth": 10,
            "min_child_samples": 11,
            "feature_fraction": 0.6715280250334623,
            "bagging_fraction": 0.8684318474251898,
            "bagging_freq": 1,
            "lambda_l1": 0.1680130174765028,
            "lambda_l2": 0.4522887105614099,
        }
    },
    "2024-10-10_2024-11-06": {
        "params": {
            "learning_rate": 0.13593876510539118,
            "num_leaves": 63,
            "max_depth": 3,
            "min_child_samples": 107,
            "feature_fraction": 0.7186498058873454,
            "bagging_fraction": 0.640732412043541,
            "bagging_freq": 1,
            "lambda_l1": 0.8612175427687435,
            "lambda_l2": 0.12443461579634313,
        }
    },
    "2024-11-14_2024-12-12": {
        "params": {
            "learning_rate": 0.09091258955037097,
            "num_leaves": 15,
            "max_depth": 3,
            "min_child_samples": 34,
            "feature_fraction": 0.9380727553061008,
            "bagging_fraction": 0.6010764208744063,
            "bagging_freq": 1,
            "lambda_l1": 0.38134401213698066,
            "lambda_l2": 0.06927064999231732,
        }
    },
    "2024-12-20_2025-01-22": {
        "params": {
            "learning_rate": 0.054467668011355226,
            "num_leaves": 29,
            "max_depth": 3,
            "min_child_samples": 26,
            "feature_fraction": 0.5436276958981346,
            "bagging_fraction": 0.8289459696533197,
            "bagging_freq": 1,
            "lambda_l1": 0.1073397858871079,
            "lambda_l2": 0.02174717260673304,
        }
    },
    "2025-01-30_2025-02-27": {
        "params": {
            "learning_rate": 0.08872720754740483,
            "num_leaves": 107,
            "max_depth": 6,
            "min_child_samples": 19,
            "feature_fraction": 0.9444039774858272,
            "bagging_fraction": 0.5419031879712568,
            "bagging_freq": 1,
            "lambda_l1": 0.24189888737176313,
            "lambda_l2": 0.8941306680211418,
        }
    },
    "2025-03-07_2025-04-03": {
        "params": {
            "learning_rate": 0.02188704140430249,
            "num_leaves": 45,
            "max_depth": 10,
            "min_child_samples": 62,
            "feature_fraction": 0.766953943854924,
            "bagging_fraction": 0.6590977477013482,
            "bagging_freq": 1,
            "lambda_l1": 0.012962012550607548,
            "lambda_l2": 0.5674153737394675,
        }
    },
    "2025-04-11_2025-05-09": {
        "params": {
            "learning_rate": 0.08939403974930357,
            "num_leaves": 50,
            "max_depth": 4,
            "min_child_samples": 118,
            "feature_fraction": 0.8215141976471426,
            "bagging_fraction": 0.6828191663659788,
            "bagging_freq": 1,
            "lambda_l1": 0.838826941621103,
            "lambda_l2": 0.6218761872283909,
        }
    },
    "2025-05-19_2025-06-16": {
        "params": {
            "learning_rate": 0.0647895253805141,
            "num_leaves": 49,
            "max_depth": 7,
            "min_child_samples": 36,
            "feature_fraction": 0.9008658022288007,
            "bagging_fraction": 0.5596913242560538,
            "bagging_freq": 1,
            "lambda_l1": 0.07803628938812464,
            "lambda_l2": 0.14617555174379493,
        }
    },
    "2025-06-25_2025-07-23": {
        "params": {
            "learning_rate": 0.12399645883123124,
            "num_leaves": 106,
            "max_depth": 8,
            "min_child_samples": 106,
            "feature_fraction": 0.9018360384495572,
            "bagging_fraction": 0.5932850294430179,
            "bagging_freq": 1,
            "lambda_l1": 0.8925589984899778,
            "lambda_l2": 0.5393422419156507,
        }
    },
    "2025-07-31_2025-08-27": {
        "params": {
            "learning_rate": 0.13617368154418308,
            "num_leaves": 17,
            "max_depth": 3,
            "min_child_samples": 61,
            "feature_fraction": 0.8431599144974905,
            "bagging_fraction": 0.576869359264971,
            "bagging_freq": 1,
            "lambda_l1": 0.27488158564405374,
            "lambda_l2": 0.45166400985014016,
        }
    },
}

# Configuration: 1 test window, 1 gap
HARDCODED_BESTS_1T_1G: Dict[str, Dict[str, Any]] = {
    "2024-07-31_2024-08-27": {
        "params": {
            "learning_rate": 0.08668081843597697,
            "num_leaves": 30,
            "max_depth": 9,
            "min_child_samples": 53,
            "feature_fraction": 0.9230239715611864,
            "bagging_fraction": 0.7870665364935905,
            "bagging_freq": 1,
            "lambda_l1": 0.22113966678869973,
            "lambda_l2": 0.17831050843145024,
        }
    },
    "2024-09-05_2024-10-02": {
        "params": {
            "learning_rate": 0.1461222396044077,
            "num_leaves": 24,
            "max_depth": 9,
            "min_child_samples": 37,
            "feature_fraction": 0.5519238016871663,
            "bagging_fraction": 0.5322579798455477,
            "bagging_freq": 1,
            "lambda_l1": 0.10793917174260247,
            "lambda_l2": 0.6441883572110686,
        }
    },
    "2024-10-10_2024-11-06": {
        "params": {
            "learning_rate": 0.12661724670476074,
            "num_leaves": 66,
            "max_depth": 10,
            "min_child_samples": 115,
            "feature_fraction": 0.7940880863854085,
            "bagging_fraction": 0.5292955112766515,
            "bagging_freq": 1,
            "lambda_l1": 0.10650808185108597,
            "lambda_l2": 0.9503721764227044,
        }
    },
    "2024-11-14_2024-12-12": {
        "params": {
            "learning_rate": 0.025970201871703427,
            "num_leaves": 41,
            "max_depth": 9,
            "min_child_samples": 13,
            "feature_fraction": 0.8610668517423196,
            "bagging_fraction": 0.5311156872813736,
            "bagging_freq": 1,
            "lambda_l1": 0.025036965696547012,
            "lambda_l2": 0.39560785165289725,
        }
        
    },
    "2024-12-20_2025-01-22": {
        "params": {
            "learning_rate": 0.13868665364721203,
            "num_leaves": 87,
            "max_depth": 4,
            "min_child_samples": 29,
            "feature_fraction": 0.7298166454294942,
            "bagging_fraction": 0.675851092797703,
            "bagging_freq": 1,
            "lambda_l1": 0.017499437477805194,
            "lambda_l2": 0.7437206383909976,
        }
    },
    "2025-01-30_2025-02-27": {
        "params": {
            "learning_rate": 0.12641607542854066,
            "num_leaves": 46,
            "max_depth": 4,
            "min_child_samples": 82,
            "feature_fraction": 0.6621257601004841,
            "bagging_fraction": 0.5530667247225123,
            "bagging_freq": 1,
            "lambda_l1": 0.002219684590676189,
            "lambda_l2": 0.023008006898933574,
        }
    },
    "2025-03-07_2025-04-03": {
        "params": {
            "learning_rate": 0.14927443702219187,
            "num_leaves": 99,
            "max_depth": 3,
            "min_child_samples": 96,
            "feature_fraction": 0.8748855608371408,
            "bagging_fraction": 0.5040807797812412,
            "bagging_freq": 1,
            "lambda_l1": 0.4352454141753437,
            "lambda_l2": 0.03152191732792635,
        }
    },
    "2025-04-11_2025-05-09": {
        "params": {
            "learning_rate": 0.0364093908921386,
            "num_leaves": 25,
            "max_depth": 7,
            "min_child_samples": 58,
            "feature_fraction": 0.7573296808327552,
            "bagging_fraction": 0.7937062045440415,
            "bagging_freq": 1,
            "lambda_l1": 0.049877807862202375,
            "lambda_l2": 0.0811747228615474,
        }
    },
    "2025-05-19_2025-06-16": {
        "params": {
            "learning_rate": 0.08630400921168414,
            "num_leaves": 70,
            "max_depth": 10,
            "min_child_samples": 18,
            "feature_fraction": 0.6566634583104347,
            "bagging_fraction": 0.5838078991326159,
            "bagging_freq": 1,
            "lambda_l1": 0.6207774616449158,
            "lambda_l2": 0.24483504286973976,
        }
    },
    "2025-06-25_2025-07-23": {
        "params": {
            "learning_rate": 0.05262220273806743,
            "num_leaves": 87,
            "max_depth": 7,
            "min_child_samples": 109,
            "feature_fraction": 0.7993494149651338,
            "bagging_fraction": 0.5938125761808022,
            "bagging_freq": 1,
            "lambda_l1": 0.2140372592645098,
            "lambda_l2": 0.10754007065681197,
        }
    },
}

# Configuration: 1 test window, 2 gaps
HARDCODED_BESTS_1T_2G: Dict[str, Dict[str, Any]] = {
    "2024-06-25_2024-07-23": {
        "params": {
            "learning_rate": 0.06682954784672998,
            "num_leaves": 99,
            "max_depth": 3,
            "min_child_samples": 51,
            "feature_fraction": 0.648157234421137,
            "bagging_fraction": 0.7814191948015099,
            "bagging_freq": 1,
            "lambda_l1": 0.8093970594677655,
            "lambda_l2": 0.6304944847226946,
        }
    },
    "2024-07-31_2024-08-27": {
        "params": {
            "learning_rate": 0.12557272934387445,
            "num_leaves": 79,
            "max_depth": 6,
            "min_child_samples": 79,
            "feature_fraction": 0.6515151777780045,
            "bagging_fraction": 0.6270631360484213,
            "bagging_freq": 1,
            "lambda_l1": 0.4844342961229664,
            "lambda_l2": 0.27786353609624004,
        }
    },
    "2024-09-05_2024-10-02": {
        "params": {
            "learning_rate": 0.0643545699980001,
            "num_leaves": 83,
            "max_depth": 6,
            "min_child_samples": 39,
            "feature_fraction": 0.7781595914083934,
            "bagging_fraction": 0.8760375750261999,
            "bagging_freq": 1,
            "lambda_l1": 0.7499153682347877,
            "lambda_l2": 0.6854595535931727,
        }
    },
    "2024-10-10_2024-11-06": {
        "params": {
            "learning_rate": 0.09880376320808783,
            "num_leaves": 91,
            "max_depth": 9,
            "min_child_samples": 31,
            "feature_fraction": 0.6195194191607329,
            "bagging_fraction": 0.6508116037275448,
            "bagging_freq": 1,
            "lambda_l1": 0.15168557828295007,
            "lambda_l2": 0.10244353652433158,
        }
    },
    "2024-11-14_2024-12-12": {
        "params": {
            "learning_rate": 0.04313176867427288,
            "num_leaves": 36,
            "max_depth": 9,
            "min_child_samples": 30,
            "feature_fraction": 0.707289811031113,
            "bagging_fraction": 0.9545084423292418,
            "bagging_freq": 1,
            "lambda_l1": 0.0014952142003898372,
            "lambda_l2": 0.03428521467217052,
        }
    },
    "2024-12-20_2025-01-22": {
        "params": {
            "learning_rate": 0.061040459787998096,
            "num_leaves": 91,
            "max_depth": 9,
            "min_child_samples": 42,
            "feature_fraction": 0.8135239771689772,
            "bagging_fraction": 0.6480262669023741,
            "bagging_freq": 1,
            "lambda_l1": 0.3061319748665662,
            "lambda_l2": 0.15796887263329346,
        }
    },
    "2025-01-30_2025-02-27": {
        "params": {
            "learning_rate": 0.12465975289448059,
            "num_leaves": 74,
            "max_depth": 9,
            "min_child_samples": 85,
            "feature_fraction": 0.7665801920991799,
            "bagging_fraction": 0.5053291437910599,
            "bagging_freq": 1,
            "lambda_l1": 0.12838778153616415,
            "lambda_l2": 0.13668071713212537,
        }
    },
    "2025-03-07_2025-04-03": {
        "params": {
            "learning_rate": 0.01489401172927616,
            "num_leaves": 59,
            "max_depth": 3,
            "min_child_samples": 58,
            "feature_fraction": 0.8605507235376805,
            "bagging_fraction": 0.8552002393174061,
            "bagging_freq": 1,
            "lambda_l1": 0.12538933613571748,
            "lambda_l2": 0.20878286506864915,
        }
    },
    "2025-04-11_2025-05-09": {
        "params": {
            "learning_rate": 0.11036704299022417,
            "num_leaves": 124,
            "max_depth": 6,
            "min_child_samples": 36,
            "feature_fraction": 0.750137681127826,
            "bagging_fraction": 0.5964701572461087,
            "bagging_freq": 1,
            "lambda_l1": 0.3378587118421168,
            "lambda_l2": 0.11326540671310044,
        }
    },
    "2025-05-19_2025-06-16": {
        "params": {
            "learning_rate": 0.04985658727594961,
            "num_leaves": 35,
            "max_depth": 4,
            "min_child_samples": 66,
            "feature_fraction": 0.8816596513452112,
            "bagging_fraction": 0.6891155972274531,
            "bagging_freq": 1,
            "lambda_l1": 0.36774931759795043,
            "lambda_l2": 0.19822652417222725,
        }
    },
}

# Dynamic selection of hardcoded bests based on configuration
def get_hardcoded_bests(num_test_windows: int, num_gap_windows: int) -> Dict[str, Dict[str, Any]]:
    """Select appropriate hardcoded bests based on configuration."""
    if num_test_windows == 1 and num_gap_windows == 0:
        return HARDCODED_BESTS_1T_0G
    elif num_test_windows == 1 and num_gap_windows == 1:
        return HARDCODED_BESTS_1T_1G
    elif num_test_windows == 1 and num_gap_windows == 2:
        return HARDCODED_BESTS_1T_2G
    else:
        # For other configurations, fall back to 1T_0G as default
        return HARDCODED_BESTS_1T_0G

# Legacy variable for backward compatibility (will be set dynamically)
HARDCODED_BESTS_RUN2: Dict[str, Dict[str, Any]] = {}


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

# Legacy function removed: now using timestamp-based registry instead of fold-index

def merge_params(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged

def filter_params_for_search(params: Dict[str, Any], allowed_keys: List[str]) -> Dict[str, Any]:
    """Return only keys that exist in the objective's search space."""
    return {k: params[k] for k in allowed_keys if k in params}

def is_gbdt_params(params: Optional[Dict[str, Any]]) -> bool:
    """Return True if params indicate gbdt or don't specify a different boosting_type."""
    if not isinstance(params, dict):
        return False
    bt = params.get("boosting_type", "gbdt")
    return bt == "gbdt"

def get_user_choice(prompt: str, has_known: bool, default_choice: Optional[str] = None) -> str:
    """
    Robust input handler. In non-interactive environments, fall back to command-line args, env var MCP_AUTO_CHOICE or sensible default.
    If has_known is True, default to '2' (skip); otherwise '1' (search).
    """
    global PROMPT_OVERRIDE
    # If already set by a previous prompt, honor it
    if PROMPT_OVERRIDE in ("1", "2", "o", "O"):
        return PROMPT_OVERRIDE
    
    # Check command-line auto-choice first
    if ARGS.auto_choice is not None:
        cmd_choice = ARGS.auto_choice
        if cmd_choice in ("1", "2", "o", "O"):
            PROMPT_OVERRIDE = cmd_choice  # persist for session if given
            return PROMPT_OVERRIDE
        if cmd_choice in ("3", "4"):
            PROMPT_OVERRIDE = "1" if cmd_choice == "3" else "2"
            print(f"  ⚙️  Auto-mode enabled via command-line arg={cmd_choice} → defaulting all prompts to {PROMPT_OVERRIDE}")
            return PROMPT_OVERRIDE
    
    # Check env override second
    env_choice = os.getenv("MCP_AUTO_CHOICE")
    if env_choice in ("1", "2", "o", "O"):
        PROMPT_OVERRIDE = env_choice  # persist for session if given
        return PROMPT_OVERRIDE
    if env_choice in ("3", "4"):
        PROMPT_OVERRIDE = "1" if env_choice == "3" else "2"
        print(f"  ⚙️  Auto-mode enabled via MCP_AUTO_CHOICE={env_choice} → defaulting all prompts to {PROMPT_OVERRIDE}")
        return PROMPT_OVERRIDE
    
    # Interactive input last
    try:
        choice = input(prompt).strip()
        if choice == "0":
            return choice
        if choice in ("1", "2", "o", "O"):
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



# Helper: parse omit-folds input like "4" or "2,5"; returns sorted unique positive ints
def _parse_omit_folds_input(text: Optional[str]) -> List[int]:
    if not text:
        return []
    s = str(text).strip()
    if s == "0":
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            v = int(p)
            if v > 0:
                out.append(v)
        except Exception:
            continue
    # unique + sorted
    return sorted(list({int(v) for v in out}))


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


# Purge and window config - RUN2 SPECIFIC
GAP_BARS = 5
PRETEST_FOLD_SIZE = 25  # RUN2: 25 bars per fold
TEST_ROWS = 25  # RUN2: 25 test rows
TRAIN_TARGET_ROWS = 1234  # enforce fixed train size regardless of gaps/windows
PRETEST_TARGET_ROWS = 250  # enforce fixed pre-test size regardless of gaps/windows
FOLD_TIME_LIMIT = 300  # seconds per fold (5 minutes)
EXTENSION_TIME_LIMIT = 300  # Phase 3: 5 minutes creative search
ULTRA_EXTENSION_TIME_LIMIT = 180  # Phase 4: 3 minutes ultra-aggressive search
EXTREME_EXTENSION_TIME_LIMIT = 120  # Phase 5: 2 minutes final extreme search
HYPER_EXTENSION_TIME_LIMIT = 600  # Phase 6: 10 minutes hyper-focused search (adaptive)
ULTIMATE_EXTENSION_TIME_LIMIT = 600  # Phase 7: 10 minutes ultimate exploration (adaptive)
FLAML_EXTENSION_TIME_LIMIT = 900  # Phase 8: 15 minutes FLAML AutoML (adaptive)
AUC_THRESHOLD_FOR_EXTENSION = 0.70  # trigger extension if best AUC below this
THRESHOLD_FLOOR_TRIGGER = 0.35  # trigger extension if best threshold <= this (poor calibration)

# Overnight finetune (Phase 9) configuration
OVERNIGHT_TIME_LIMIT_S = int(os.getenv("MCP_OVERNIGHT_TIME_LIMIT_S", "43200"))  # default 12 hours
OVERNIGHT_MCC_TARGET = float(os.getenv("MCP_OVERNIGHT_MCC_TARGET", "0.60"))

# Timestamp registry and legacy best-param flags removed


def _get_num_test_windows(default: int = 1) -> int:
    """Get desired number of test windows from command-line args, env, or prompt. Defaults to 1."""
    # Command-line args first
    if ARGS.test_windows is not None:
        return max(1, ARGS.test_windows)
    
    # Env overrides second
    env_val = os.getenv("MCP_NUM_TEST_WINDOWS") or os.getenv("NUM_TEST_WINDOWS")
    if env_val:
        try:
            v = int(env_val)
            return max(1, v)
        except Exception:
            pass
    
    # Interactive prompt last
    try:
        choice = input(f"How many test windows? [default {default}]: ").strip()
        if choice == "":
            return default
        v = int(choice)
        return max(1, v)
    except EOFError:
        return default
    except Exception:
        return default

def _get_num_gap_windows(default: int = 0) -> int:
    """Get desired number of gap windows (ignored from the end) from command-line args, env, or prompt. Defaults to 0."""
    # Command-line args first
    if ARGS.gap_windows is not None:
        return max(0, ARGS.gap_windows)
    
    # Env overrides second
    env_val = os.getenv("MCP_NUM_GAP_WINDOWS") or os.getenv("NUM_GAP_WINDOWS") or os.getenv("NUM_GAPS")
    if env_val:
        try:
            v = int(env_val)
            return max(0, v)
        except Exception:
            pass
    
    # Interactive prompt last
    try:
        choice = input(f"How many gap windows to ignore at the end? [default {default}]: ").strip()
        if choice == "":
            return default
        v = int(choice)
        return max(0, v)
    except EOFError:
        return default
    except Exception:
        return default

def split_multi_windows(df: pd.DataFrame, num_test_windows: int, gap_windows: int = 0) -> Dict[str, Any]:
    """
    Multi-window split for RUN2:
    1) Use all available data (backtest + fronttest, already deduplicated by query)
    2) Test windows = last (num_test_windows * TEST_ROWS) rows after ignoring (gap_windows * TEST_ROWS),
       partitioned into contiguous windows of TEST_ROWS each (chronological order)
    3) Pre-test = fixed PRETEST_TARGET_ROWS immediately before the earliest test window
    4) Train = fixed TRAIN_TARGET_ROWS immediately before pre-test window
    
    The entire timeline shifts backward by (gap_windows * TEST_ROWS), but Train and Pre-test
    always maintain their fixed sizes.
    
    Returns dict with train, pretest, test_windows (List[pd.DataFrame]), pretest_bounds, test_bounds (List[Tuple[start,end]]).
    """
    if len(df) == 0:
        raise RuntimeError("No data available")
    
    # Use all available data, sorted by timestamp
    df_all = df.copy().sort_values("timestamp").reset_index(drop=True)
    
    gap_ignore = TEST_ROWS * max(0, int(gap_windows))
    total_test_rows = TEST_ROWS * max(1, int(num_test_windows))
    total_needed = TRAIN_TARGET_ROWS + PRETEST_TARGET_ROWS + total_test_rows + gap_ignore
    
    if len(df_all) < total_needed:
        raise RuntimeError(f"Not enough bars: need {total_needed} (train={TRAIN_TARGET_ROWS} + pre={PRETEST_TARGET_ROWS} + test={total_test_rows} + gap={gap_ignore}), have {len(df_all)}")
    
    # Work backward from the end
    # Position of last row we can use (before gaps)
    usable_end_idx = len(df_all) - gap_ignore
    
    # Test windows: total_test_rows ending at usable_end_idx, partitioned into num_test_windows
    test_start_idx = usable_end_idx - total_test_rows
    test_end_idx = usable_end_idx
    tests_concat = df_all.iloc[test_start_idx:test_end_idx].copy().reset_index(drop=True)
    
    test_windows: List[pd.DataFrame] = []
    test_bounds: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    # Partition into contiguous windows in chronological order (earliest to latest)
    for w in range(num_test_windows):
        start = w * TEST_ROWS
        end = start + TEST_ROWS
        win_df = tests_concat.iloc[start:end].copy().reset_index(drop=True)
        test_windows.append(win_df)
        if len(win_df):
            test_bounds.append((pd.to_datetime(win_df["timestamp"]).iloc[0], pd.to_datetime(win_df["timestamp"]).iloc[-1]))
        else:
            test_bounds.append((None, None))

    # Pre-test window: PRETEST_TARGET_ROWS immediately before earliest test window
    pre_start_idx = test_start_idx - PRETEST_TARGET_ROWS
    pre_end_idx = test_start_idx
    pre_df = df_all.iloc[pre_start_idx:pre_end_idx].copy().reset_index(drop=True)
    pretest_start = pd.to_datetime(pre_df["timestamp"]).iloc[0]
    pretest_end = pd.to_datetime(pre_df["timestamp"]).iloc[-1]
    
    # Train window: TRAIN_TARGET_ROWS immediately before pre-test
    train_start_idx = pre_start_idx - TRAIN_TARGET_ROWS
    train_end_idx = pre_start_idx
    train_df = df_all.iloc[train_start_idx:train_end_idx].copy().reset_index(drop=True)

    return {
        "train": train_df,
        "pretest": pre_df,
        "test_windows": test_windows,  # chronological: earliest..latest
        "pretest_bounds": (pretest_start, pretest_end),
        "test_bounds": test_bounds,
    }

def split_three_windows(df: pd.DataFrame, gap_windows: int = 0) -> Dict[str, pd.DataFrame]:
    """
    Backwards splitting to ensure exact test rows for RUN2:
    1) Use all available data (backtest + fronttest, already deduplicated by query)
    2) Test = last TEST_ROWS rows after ignoring (gap_windows * TEST_ROWS) rows from the end
    3) Pre-test = fixed PRETEST_TARGET_ROWS immediately before test window
    4) Train = fixed TRAIN_TARGET_ROWS immediately before pre-test window
    
    The entire timeline shifts backward by (gap_windows * TEST_ROWS), but Train and Pre-test
    always maintain their fixed sizes.
    """
    if len(df) == 0:
        raise RuntimeError("No data available")
    
    # Use all available data, sorted by timestamp
    df_all = df.copy().sort_values("timestamp").reset_index(drop=True)
    
    gap_ignore = TEST_ROWS * max(0, int(gap_windows))
    total_needed = TEST_ROWS + PRETEST_TARGET_ROWS + TRAIN_TARGET_ROWS + gap_ignore
    
    if len(df_all) < total_needed:
        raise RuntimeError(f"Not enough bars: need {total_needed} (train={TRAIN_TARGET_ROWS} + pre={PRETEST_TARGET_ROWS} + test={TEST_ROWS} + gap={gap_ignore}), have {len(df_all)}")
    
    # Work backward from the end
    # Position of last row we can use (before gaps)
    usable_end_idx = len(df_all) - gap_ignore
    
    # Test window: TEST_ROWS ending at usable_end_idx
    test_start_idx = usable_end_idx - TEST_ROWS
    test_end_idx = usable_end_idx
    test_df = df_all.iloc[test_start_idx:test_end_idx].copy().reset_index(drop=True)
    test_start = pd.to_datetime(test_df["timestamp"]).iloc[0]
    test_end = pd.to_datetime(test_df["timestamp"]).iloc[-1]
    
    # Pre-test window: PRETEST_TARGET_ROWS immediately before test
    pre_start_idx = test_start_idx - PRETEST_TARGET_ROWS
    pre_end_idx = test_start_idx
    pre_df = df_all.iloc[pre_start_idx:pre_end_idx].copy().reset_index(drop=True)
    pretest_start = pd.to_datetime(pre_df["timestamp"]).iloc[0]
    pretest_end = pd.to_datetime(pre_df["timestamp"]).iloc[-1]
    
    # Train window: TRAIN_TARGET_ROWS immediately before pre-test
    train_start_idx = pre_start_idx - TRAIN_TARGET_ROWS
    train_end_idx = pre_start_idx
    train_df = df_all.iloc[train_start_idx:train_end_idx].copy().reset_index(drop=True)
    
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


def clamp_lgb_param_ranges(params_in: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp/round LightGBM params to safe ranges and types."""
    params = dict(default_lgb_params())
    params.update(params_in or {})
    try:
        params["learning_rate"] = float(np.clip(float(params.get("learning_rate", 0.05)), 0.005, 0.15))
    except Exception:
        params["learning_rate"] = 0.05
    try:
        params["num_leaves"] = int(np.clip(int(round(float(params.get("num_leaves", 63)))), 15, 127))
    except Exception:
        params["num_leaves"] = 63
    try:
        params["max_depth"] = int(np.clip(int(round(float(params.get("max_depth", 6)))), 3, 12))
    except Exception:
        params["max_depth"] = 6
    try:
        params["min_child_samples"] = int(np.clip(int(round(float(params.get("min_child_samples", 60)))), 10, 160))
    except Exception:
        params["min_child_samples"] = 60
    try:
        params["feature_fraction"] = float(np.clip(float(params.get("feature_fraction", 0.85)), 0.4, 1.0))
    except Exception:
        params["feature_fraction"] = 0.85
    try:
        params["bagging_fraction"] = float(np.clip(float(params.get("bagging_fraction", 0.85)), 0.4, 1.0))
    except Exception:
        params["bagging_fraction"] = 0.85
    try:
        params["lambda_l1"] = float(np.clip(float(max(0.0, params.get("lambda_l1", 0.1))), 0.0, 1.0))
    except Exception:
        params["lambda_l1"] = 0.1
    try:
        params["lambda_l2"] = float(np.clip(float(max(0.0, params.get("lambda_l2", 0.1))), 0.0, 1.0))
    except Exception:
        params["lambda_l2"] = 0.1
    # Soft cross-parameter sanity: ensure num_leaves not absurd vs depth (not hard limit in LGBM)
    try:
        max_depth = max(1, params["max_depth"])  # avoid zero
        max_reasonable_leaves = int(np.clip(2 ** max_depth, 15, 127))
        params["num_leaves"] = int(np.clip(params["num_leaves"], 15, max_reasonable_leaves))
    except Exception:
        pass
    # Optional late-phase knobs if present (kept but clamped)
    if "min_split_gain" in params:
        try:
            params["min_split_gain"] = float(np.clip(float(params.get("min_split_gain", 0.0)), 0.0, 2.0))
        except Exception:
            params["min_split_gain"] = 0.0
    if "min_child_weight" in params:
        try:
            params["min_child_weight"] = float(np.clip(float(params.get("min_child_weight", 0.001)), 1e-4, 100.0))
        except Exception:
            params["min_child_weight"] = 0.001
    if "bagging_freq" in params:
        try:
            params["bagging_freq"] = int(np.clip(int(round(float(params.get("bagging_freq", 1)))), 0, 10))
        except Exception:
            params["bagging_freq"] = 1
    if "max_bin" in params:
        try:
            params["max_bin"] = int(np.clip(int(round(float(params.get("max_bin", 255)))), 64, 511))
        except Exception:
            params["max_bin"] = 255
    if "feature_fraction_bynode" in params:
        try:
            params["feature_fraction_bynode"] = float(np.clip(float(params.get("feature_fraction_bynode", 1.0)), 0.2, 1.0))
        except Exception:
            params["feature_fraction_bynode"] = 1.0
    return params


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
        "mcc": float("nan"),
    }
    for thr in thr_grid:
        pred = (proba >= thr).astype(int)
        acc = accuracy_score(y_true, pred)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        bacc = balanced_accuracy_score(y_true, pred)
        try:
            mcc = float(matthews_corrcoef(y_true, pred))
        except Exception:
            mcc = float("nan")
        if (np.isnan(best["f1"]) or f1 > best["f1"]) or (f1 == best["f1"] and acc > best.get("acc", 0)):
            best = {"thr": float(thr), "acc": float(acc), "auc": auc_val, "prec": float(prec), "rec": float(rec), "f1": float(f1), "bacc": float(bacc), "mcc": mcc}
    return best


def evaluate_at_threshold(y_true: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, Any]:
    """Compute metrics at a specific threshold (no threshold search)."""
    has_two_classes = len(np.unique(y_true)) == 2
    try:
        auc_val = float(roc_auc_score(y_true, proba)) if has_two_classes else None
    except Exception:
        auc_val = None
    try:
        pred = (proba >= float(thr)).astype(int)
        acc = float(accuracy_score(y_true, pred))
        prec = float(precision_score(y_true, pred, zero_division=0))
        rec = float(recall_score(y_true, pred, zero_division=0))
        f1 = float(f1_score(y_true, pred, zero_division=0))
        bacc = float(balanced_accuracy_score(y_true, pred))
        mcc = float(matthews_corrcoef(y_true, pred))
    except Exception:
        acc = float("nan"); prec = float("nan"); rec = float("nan"); f1 = float("nan"); bacc = float("nan"); mcc = float("nan")
    return {"thr": float(thr), "acc": acc, "auc": auc_val, "prec": prec, "rec": rec, "f1": f1, "bacc": bacc, "mcc": mcc}


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


def create_optuna_objective(X_tr: Any, y_tr: np.ndarray, X_va: Any, y_va: np.ndarray, 
                           thr_grid: np.ndarray, prior_best: Optional[Dict[str, Any]] = None, mcc_gap: float = 0.0,
                           include_tough_params: bool = False) -> callable:
    """
    Create an Optuna objective function for hyperparameter optimization.
    Uses TPE sampler with informed priors if prior_best is provided.
    mcc_gap: Distance from target MCC (0.60). Larger gap = wider/more aggressive search.
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
            # Safe bounds around prior, ensure low <= high and align with late-phase cap (<=160)
            _mcs_low = int(max(10, round(min_child_center - 30)))
            _mcs_high = int(min(160, round(min_child_center + 30)))
            if _mcs_high < _mcs_low:
                _mcs_high = _mcs_low
            min_child_samples = trial.suggest_int("min_child_samples", _mcs_low, _mcs_high)
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
            # Include tough params in guided mode if requested
            if include_tough_params:
                min_split_gain = trial.suggest_float("min_split_gain", 0.0, 1.0)
                min_child_weight = trial.suggest_float("min_child_weight", 1e-4, 50.0, log=True)
                feature_fraction_bynode = trial.suggest_float("feature_fraction_bynode", 0.2, 1.0)
                max_bin = trial.suggest_int("max_bin", 128, 511)
        else:
            # Wide search (extended knobs available to phases 6-8 via dynamic ranges)
            learning_rate = trial.suggest_float("learning_rate", 0.005, 0.15, log=True)
            num_leaves = trial.suggest_int("num_leaves", 15, 127)
            max_depth = trial.suggest_int("max_depth", 3, 12)
            # Ensure bounds are consistent with data size; keep <= len(y_tr) if tiny
            _global_mcs_low = int(max(5, 0.01 * len(y_tr)))
            _global_mcs_low = min(_global_mcs_low, 160)
            _global_mcs_high = int(min(160, max(_global_mcs_low, 0.5 * len(y_tr))))
            if _global_mcs_high < _global_mcs_low:
                _global_mcs_high = _global_mcs_low
            min_child_samples = trial.suggest_int("min_child_samples", _global_mcs_low, _global_mcs_high)
            feature_fraction = trial.suggest_float("feature_fraction", 0.4, 1.0)
            bagging_fraction = trial.suggest_float("bagging_fraction", 0.4, 1.0)
            lambda_l1 = trial.suggest_float("lambda_l1", 0.0, 1.0)
            lambda_l2 = trial.suggest_float("lambda_l2", 0.0, 1.0)
            # Late-phase optional knobs
            min_split_gain = trial.suggest_float("min_split_gain", 0.0, 1.0)
            min_child_weight = trial.suggest_float("min_child_weight", 1e-4, 50.0, log=True)
            feature_fraction_bynode = trial.suggest_float("feature_fraction_bynode", 0.2, 1.0)
            bagging_freq = trial.suggest_int("bagging_freq", 0, 7)
            max_bin = trial.suggest_int("max_bin", 128, 511)
        
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
            # Always gbdt as requested
            "boosting_type": "gbdt",
        })
        # Attach late-phase knobs if defined in this branch
        for k in ("min_split_gain","min_child_weight","feature_fraction_bynode","bagging_freq","max_bin"):
            if k in locals():
                params[k] = locals()[k]
        
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
            
            # Store additional info for later retrieval
            trial.set_user_attr("stats", stats)
            trial.set_user_attr("params", params)
            
            return score
        except Exception as e:
            # Return worst score on failure
            return -1.0
    
    return objective


def create_fixed_param_objective(X_tr: Any, y_tr: np.ndarray, X_va: Any, y_va: np.ndarray,
                                 thr_grid: np.ndarray, fixed_params: Dict[str, Any]) -> callable:
    """Return an objective that evaluates a fixed LightGBM parameter dict (no search)."""
    params_fixed = clamp_lgb_param_ranges(fixed_params or {})

    def objective(_: optuna.Trial) -> float:
        try:
            clf = lgb.LGBMClassifier(**params_fixed)
            bt = params_fixed.get("boosting_type", "gbdt")
            fit_callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)] if bt != "dart" else []
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc", callbacks=fit_callbacks)
            proba = clf.predict_proba(X_va)[:, 1]
            stats = evaluate_thresholds(y_va, proba, thr_grid)
            score = score_fold(stats)
        except Exception:
            stats = {"thr": 0.5, "acc": 0.0, "auc": None, "prec": 0.0, "rec": 0.0, "f1": 0.0, "bacc": 0.0, "mcc": float("nan")}
            score = -1.0
        # attach for uniform downstream handling
        try:
            # We mimic normal objective attributes so downstream code keeps working
            optuna.trial.Trial.set_user_attr(_, "stats", stats)  # type: ignore
            optuna.trial.Trial.set_user_attr(_, "params", params_fixed)  # type: ignore
        except Exception:
            pass
        return score
    
    return objective


def _has_cmaes() -> bool:
    """Return True if optional 'cmaes' dependency for Optuna is available."""
    try:
        import importlib
        importlib.import_module("cmaes")
        return True
    except Exception:
        return False

def _has_flaml() -> bool:
    """Return True if optional 'flaml' dependency is available for Phase 8."""
    try:
        import importlib
        importlib.import_module("flaml")
        return True
    except Exception:
        return False


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


class AdvancedMetaLearner:
    """
    Significantly improved meta-learner with:
    - Rich signature features (drift, volatility, regime detection, temporal patterns)
    - Ensemble of multiple architectures (chain, MLP, tree-based)
    - Cross-validation for robust meta-model training
    - Uncertainty estimation and confidence-based selection
    """
    def __init__(self):
        self.chain_order = [
            "max_depth",
            "num_leaves", 
            "min_child_samples",
            "learning_rate",
            "feature_fraction",
            "bagging_fraction",
            "lambda_l1",
            "lambda_l2",
        ]
        self.chain_models: Dict[str, Any] = {}
        self.chain_feature_names: Dict[str, List[str]] = {}  # Store feature names per target
        self.fitted_chain: bool = False
        self.param_keys = [
            "learning_rate",
            "num_leaves",
            "max_depth", 
            "min_child_samples",
            "feature_fraction",
            "bagging_fraction",
            "lambda_l1",
            "lambda_l2",
        ]
        # Multiple model heads for ensemble
        self.mlp_ensemble: List[MLPRegressor] = []
        self.tree_ensemble: List[Any] = []
        self.fitted_mlp: bool = False
        self.fitted_tree: bool = False
        self.scaler_sig = StandardScaler()
        self.fitted_scaler = False

    def _compute_rich_signature(self, X_df: pd.DataFrame, y: np.ndarray, prev_X_df: Optional[pd.DataFrame] = None, prev_y: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute rich signature with drift, volatility, regime features."""
        n = float(len(y))
        if n <= 0:
            return {"n": 0.0, "pos_rate": 0.5, "feat_mean_avg": 0.0, "feat_std_avg": 0.0}
        
        # Basic stats
        pos_rate = float(np.mean(y))
        feat_mean_avg = float(np.mean(X_df.values)) if X_df.size else 0.0
        feat_std_avg = float(np.std(X_df.values)) if X_df.size else 0.0
        
        # Volatility and distribution features
        feat_skew = float(pd.DataFrame(X_df.values).skew().mean()) if X_df.size else 0.0
        feat_kurt = float(pd.DataFrame(X_df.values).kurtosis().mean()) if X_df.size else 0.0
        feat_range_avg = float(np.mean(np.ptp(X_df.values, axis=0))) if X_df.size else 0.0
        
        # Class imbalance and entropy
        pos_count = int(np.sum(y == 1))
        neg_count = int(np.sum(y == 0))
        class_entropy = 0.0
        if pos_count > 0 and neg_count > 0:
            p_pos = pos_count / (pos_count + neg_count)
            p_neg = neg_count / (pos_count + neg_count)
            class_entropy = -p_pos * np.log2(p_pos) - p_neg * np.log2(p_neg)
        
        # Temporal patterns (if data has time structure)
        temporal_trend = 0.0
        temporal_volatility = 0.0
        if len(y) > 10:
            # Simple trend in labels over time
            time_idx = np.arange(len(y))
            try:
                temporal_trend = float(np.corrcoef(time_idx, y)[0, 1]) if not np.isnan(np.corrcoef(time_idx, y)[0, 1]) else 0.0
            except:
                temporal_trend = 0.0
            # Volatility in feature means over time (rolling window)
            if X_df.size > 0:
                window_size = min(10, len(X_df) // 3)
                if window_size >= 2:
                    rolling_means = []
                    for i in range(0, len(X_df) - window_size + 1, window_size):
                        window_mean = np.mean(X_df.iloc[i:i+window_size].values)
                        rolling_means.append(window_mean)
                    if len(rolling_means) > 1:
                        temporal_volatility = float(np.std(rolling_means))
        
        # Drift features (if previous data available)
        drift_mean = 0.0
        drift_std = 0.0
        drift_pos_rate = 0.0
        if prev_X_df is not None and prev_y is not None and len(prev_X_df) > 0:
            prev_mean = float(np.mean(prev_X_df.values)) if prev_X_df.size else 0.0
            prev_std = float(np.std(prev_X_df.values)) if prev_X_df.size else 0.0
            prev_pos_rate = float(np.mean(prev_y)) if len(prev_y) > 0 else 0.5
            
            drift_mean = feat_mean_avg - prev_mean
            drift_std = feat_std_avg - prev_std
            drift_pos_rate = pos_rate - prev_pos_rate
        
        # Feature correlation structure
        feat_corr_mean = 0.0
        if X_df.shape[1] > 1 and X_df.size > 0:
            try:
                corr_matrix = np.corrcoef(X_df.values.T)
                # Mean absolute correlation (excluding diagonal)
                mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                feat_corr_mean = float(np.mean(np.abs(corr_matrix[mask]))) if not np.isnan(corr_matrix[mask]).all() else 0.0
            except:
                feat_corr_mean = 0.0
        
        return {
            # Basic
            "n": n,
            "pos_rate": pos_rate,
            "feat_mean_avg": feat_mean_avg,
            "feat_std_avg": feat_std_avg,
            # Distribution
            "feat_skew": feat_skew,
            "feat_kurt": feat_kurt,
            "feat_range_avg": feat_range_avg,
            "class_entropy": class_entropy,
            # Temporal
            "temporal_trend": temporal_trend,
            "temporal_volatility": temporal_volatility,
            # Drift
            "drift_mean": drift_mean,
            "drift_std": drift_std,
            "drift_pos_rate": drift_pos_rate,
            # Structure
            "feat_corr_mean": feat_corr_mean,
        }

    def _sig_to_array(self, sig: Dict[str, float]) -> np.ndarray:
        """Convert rich signature to array."""
        features = [
            "n", "pos_rate", "feat_mean_avg", "feat_std_avg",
            "feat_skew", "feat_kurt", "feat_range_avg", "class_entropy",
            "temporal_trend", "temporal_volatility",
            "drift_mean", "drift_std", "drift_pos_rate",
            "feat_corr_mean"
        ]
        return np.array([float(sig.get(f, 0.0)) for f in features], dtype=float)

    def fit(self, metas: List[Dict[str, Any]]):
        if not isinstance(metas, list) or len(metas) < 3:
            self.fitted_chain = False
            self.fitted_mlp = False
            self.fitted_tree = False
            return

        # Build training matrices using rich signatures
        X_base: List[List[float]] = []
        Y_mat: List[List[float]] = []
        P_true: List[Dict[str, float]] = []
        
        for m in metas:
            # Use basic signature as fallback for compatibility
            sig = m.get("train_signature", {}) or {}
            # Convert to rich signature format (basic features only for now to maintain compatibility)
            rich_sig = {
                "n": sig.get("n", 0.0),
                "pos_rate": sig.get("pos_rate", 0.5),
                "feat_mean_avg": sig.get("feat_mean_avg", 0.0),
                "feat_std_avg": sig.get("feat_std_avg", 0.0),
                "feat_skew": 0.0,
                "feat_kurt": 0.0,
                "feat_range_avg": 0.0,
                "class_entropy": 1.0 if 0.4 <= sig.get("pos_rate", 0.5) <= 0.6 else 0.5,
                "temporal_trend": 0.0,
                "temporal_volatility": sig.get("feat_std_avg", 0.0) * 0.1,  # proxy
                "drift_mean": 0.0,
                "drift_std": 0.0,
                "drift_pos_rate": 0.0,
                "feat_corr_mean": 0.3,  # reasonable default
            }
            X_base.append(list(self._sig_to_array(rich_sig)))
            p = m.get("params", {}) or {}
            # Consistent ordering for target
            Y_mat.append([
                float(p.get("learning_rate", default_lgb_params()["learning_rate"])),
                float(p.get("num_leaves", default_lgb_params()["num_leaves"])),
                float(p.get("max_depth", default_lgb_params()["max_depth"])),
                float(p.get("min_child_samples", default_lgb_params()["min_child_samples"])),
                float(p.get("feature_fraction", default_lgb_params()["feature_fraction"])),
                float(p.get("bagging_fraction", default_lgb_params()["bagging_fraction"])),
                float(p.get("lambda_l1", default_lgb_params()["lambda_l1"])),
                float(p.get("lambda_l2", default_lgb_params()["lambda_l2"]))
            ])
            P_true.append(p)

        X_base_arr = np.array(X_base, dtype=float)
        Y_arr = np.array(Y_mat, dtype=float)
        
        # Fit scaler on signatures
        try:
            self.scaler_sig.fit(X_base_arr)
            self.fitted_scaler = True
            X_scaled = self.scaler_sig.transform(X_base_arr)
        except Exception:
            self.fitted_scaler = False
            X_scaled = X_base_arr

        # Fit ensemble of MLP heads with different architectures and regularization
        self.mlp_ensemble = []
        mlp_configs = [
            {"hidden_layer_sizes": (64, 32), "alpha": 3e-3, "activation": "relu"},
            {"hidden_layer_sizes": (32, 16, 8), "alpha": 1e-2, "activation": "tanh"},
            {"hidden_layer_sizes": (48, 24), "alpha": 7e-3, "activation": "relu"},
        ]
        
        for i, config in enumerate(mlp_configs):
            try:
                # Increase max_iter and enable early_stopping to converge and avoid warnings
                mlp = MLPRegressor(max_iter=2500, early_stopping=True, n_iter_no_change=25, validation_fraction=0.15, random_state=42+i, **config)
                mlp.fit(X_scaled, Y_arr)
                self.mlp_ensemble.append(mlp)
            except Exception:
                continue
        
        self.fitted_mlp = len(self.mlp_ensemble) > 0
        
        # Fit ensemble of tree-based regressors
        self.tree_ensemble = []
        tree_configs = [
            {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.08, "n_jobs": 1},
            {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "n_jobs": 1},
            {"n_estimators": 120, "max_depth": 8, "learning_rate": 0.06, "n_jobs": 1},
        ]
        
        for i, config in enumerate(tree_configs):
            try:
                # Use LightGBM for each param separately (multi-output tree ensemble)
                param_regressors = {}
                for j, param_name in enumerate(self.param_keys):
                    reg = lgb.LGBMRegressor(random_state=42+i, **config)
                    reg.fit(X_scaled, Y_arr[:, j])
                    param_regressors[param_name] = reg
                self.tree_ensemble.append(param_regressors)
            except Exception:
                continue
        
        self.fitted_tree = len(self.tree_ensemble) > 0

        # Fit regressor chain using LightGBM regressors
        self.chain_models = {}
        self.chain_feature_names: Dict[str, List[str]] = {}  # Store feature names per target
        prev_targets_for_train: Dict[str, np.ndarray] = {}
        base_sig_cols = [
            "sig_n","sig_pos_rate","sig_feat_mean","sig_feat_std",
            "feat_skew","feat_kurt","feat_range_avg","class_entropy",
            "temporal_trend","temporal_volatility","drift_mean","drift_std",
            "drift_pos_rate","feat_corr_mean"
        ]
        # Build base signature matrix with 14 columns (some may be zeros if not provided)
        if X_base_arr.shape[1] == 4:
            # Expand to 14 with zeros for extended features
            zeros_ext = np.zeros((X_base_arr.shape[0], 10), dtype=float)
            X_base_14 = np.concatenate([X_base_arr, zeros_ext], axis=1)
        else:
            X_base_14 = X_base_arr
        for idx, target in enumerate(self.chain_order):
            # Build design matrix = base signature + previous true targets (teacher forcing)
            feats_frames: List[pd.DataFrame] = [
                pd.DataFrame(X_base_14, columns=base_sig_cols)
            ]
            for t_prev in self.chain_order[:idx]:
                y_prev = np.array([float(pt.get(t_prev, default_lgb_params().get(t_prev, 0.0))) for pt in P_true], dtype=float)
                feats_frames.append(pd.DataFrame({f"prev_{t_prev}": y_prev}))
            X_chain_df = pd.concat(feats_frames, axis=1)
            y_chain = np.array([float(pt.get(target, default_lgb_params().get(target, 0.0))) for pt in P_true], dtype=float)
            try:
                if len(np.unique(y_chain)) < 2:
                    # Degenerate target; skip
                    continue
                # Use shallow trees to reduce overfitting and ensure fast predict
                # Set n_jobs=1 to avoid Windows subprocess issues
                reg = lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    n_jobs=1,  # Avoid parallel processing issues on Windows
                )
                reg.fit(X_chain_df, y_chain)
                self.chain_models[target] = reg
                self.chain_feature_names[target] = list(X_chain_df.columns)  # Store column names
            except Exception:
                continue
        self.fitted_chain = len(self.chain_models) > 0

    def predict_chain(self, sig: Dict[str, float]) -> Dict[str, Any]:
        if not self.fitted_chain:
            return default_lgb_params()
        # Build 14-col base signature in the same order used in training
        x_sig = self._sig_to_array(sig).reshape(1, -1)
        if x_sig.shape[1] == 14:
            x_base_14 = x_sig
        else:
            # If only 4 basic features present, pad with zeros for extended features
            if x_sig.shape[1] == 4:
                x_base_14 = np.concatenate([x_sig, np.zeros((1, 10), dtype=float)], axis=1)
            else:
                # Fallback: truncate or pad to 14
                if x_sig.shape[1] > 14:
                    x_base_14 = x_sig[:, :14]
                else:
                    x_base_14 = np.concatenate([x_sig, np.zeros((1, 14 - x_sig.shape[1]), dtype=float)], axis=1)
        base_sig_cols = [
            "sig_n","sig_pos_rate","sig_feat_mean","sig_feat_std",
            "feat_skew","feat_kurt","feat_range_avg","class_entropy",
            "temporal_trend","temporal_volatility","drift_mean","drift_std",
            "drift_pos_rate","feat_corr_mean"
        ]
        preds: Dict[str, float] = {}
        for idx, target in enumerate(self.chain_order):
            reg = self.chain_models.get(target)
            if reg is None:
                preds[target] = float(default_lgb_params().get(target, 0.0))
                continue
            # Build features: base + previously predicted targets (in order)
            feats_frames = [
                pd.DataFrame(x_base_14, columns=base_sig_cols)
            ]
            if idx > 0:
                prev_vals_df = pd.DataFrame({f"prev_{t}": [preds[t]] for t in self.chain_order[:idx]})
                feats_frames.append(prev_vals_df)
            X_star_df = pd.concat(feats_frames, axis=1)
            
            # Ensure columns match training feature names
            expected_cols = self.chain_feature_names.get(target)
            if expected_cols is not None:
                # Reorder/filter to match training columns
                try:
                    X_star_df = X_star_df[expected_cols]
                except KeyError:
                    # If mismatch, fall back to default
                    preds[target] = float(default_lgb_params().get(target, 0.0))
                    continue
            
            try:
                y_hat = float(reg.predict(X_star_df)[0])
            except Exception:
                y_hat = float(default_lgb_params().get(target, 0.0))
            preds[target] = y_hat
        # Map to param dict and clamp/round
        params = {
            "learning_rate": preds.get("learning_rate", default_lgb_params()["learning_rate"]),
            "num_leaves": preds.get("num_leaves", default_lgb_params()["num_leaves"]),
            "max_depth": preds.get("max_depth", default_lgb_params()["max_depth"]),
            "min_child_samples": preds.get("min_child_samples", default_lgb_params()["min_child_samples"]),
            "feature_fraction": preds.get("feature_fraction", default_lgb_params()["feature_fraction"]),
            "bagging_fraction": preds.get("bagging_fraction", default_lgb_params()["bagging_fraction"]),
            "lambda_l1": preds.get("lambda_l1", default_lgb_params()["lambda_l1"]),
            "lambda_l2": preds.get("lambda_l2", default_lgb_params()["lambda_l2"]),
        }
        return clamp_lgb_param_ranges(params)

    def predict_mlp(self, sig: Dict[str, float]) -> Dict[str, Any]:
        """Ensemble MLP prediction with uncertainty estimation."""
        if not self.fitted_mlp or len(self.mlp_ensemble) == 0:
            return default_lgb_params()
        
        # Convert signature to rich format for compatibility
        rich_sig = {
            "n": sig.get("n", 0.0),
            "pos_rate": sig.get("pos_rate", 0.5),
            "feat_mean_avg": sig.get("feat_mean_avg", 0.0),
            "feat_std_avg": sig.get("feat_std_avg", 0.0),
            "feat_skew": 0.0,
            "feat_kurt": 0.0,
            "feat_range_avg": 0.0,
            "class_entropy": 1.0 if 0.4 <= sig.get("pos_rate", 0.5) <= 0.6 else 0.5,
            "temporal_trend": 0.0,
            "temporal_volatility": sig.get("feat_std_avg", 0.0) * 0.1,
            "drift_mean": 0.0,
            "drift_std": 0.0,
            "drift_pos_rate": 0.0,
            "feat_corr_mean": 0.3,
        }
        
        x = self._sig_to_array(rich_sig).reshape(1, -1)
        if self.fitted_scaler:
            try:
                x = self.scaler_sig.transform(x)
            except Exception:
                pass
        
        # Ensemble prediction
        predictions = []
        for mlp in self.mlp_ensemble:
            try:
                y_hat = mlp.predict(x)[0]
                predictions.append(y_hat)
            except Exception:
                continue
        
        if not predictions:
            return default_lgb_params()
        
        # Average ensemble predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Map back to dict using self.param_keys
        out: Dict[str, Any] = {k: float(v) for k, v in zip(self.param_keys, ensemble_pred)}
        return clamp_lgb_param_ranges(out)
    
    def predict_tree(self, sig: Dict[str, float]) -> Dict[str, Any]:
        """Ensemble tree prediction."""
        if not self.fitted_tree or len(self.tree_ensemble) == 0:
            return default_lgb_params()
        
        # Convert signature to rich format for compatibility
        rich_sig = {
            "n": sig.get("n", 0.0),
            "pos_rate": sig.get("pos_rate", 0.5),
            "feat_mean_avg": sig.get("feat_mean_avg", 0.0),
            "feat_std_avg": sig.get("feat_std_avg", 0.0),
            "feat_skew": 0.0,
            "feat_kurt": 0.0,
            "feat_range_avg": 0.0,
            "class_entropy": 1.0 if 0.4 <= sig.get("pos_rate", 0.5) <= 0.6 else 0.5,
            "temporal_trend": 0.0,
            "temporal_volatility": sig.get("feat_std_avg", 0.0) * 0.1,
            "drift_mean": 0.0,
            "drift_std": 0.0,
            "drift_pos_rate": 0.0,
            "feat_corr_mean": 0.3,
        }
        
        x = self._sig_to_array(rich_sig).reshape(1, -1)
        if self.fitted_scaler:
            try:
                x = self.scaler_sig.transform(x)
            except Exception:
                pass
        
        # Ensemble prediction across tree models
        param_predictions = {k: [] for k in self.param_keys}
        
        for tree_set in self.tree_ensemble:
            for param_name in self.param_keys:
                if param_name in tree_set:
                    try:
                        pred = float(tree_set[param_name].predict(x)[0])
                        param_predictions[param_name].append(pred)
                    except Exception:
                        continue
        
        # Average predictions for each parameter
        out = {}
        for param_name in self.param_keys:
            if param_predictions[param_name]:
                out[param_name] = float(np.mean(param_predictions[param_name]))
            else:
                out[param_name] = float(default_lgb_params()[param_name])
        
        return clamp_lgb_param_ranges(out)


def refine_params_local(X_tr_df: pd.DataFrame, y_tr: np.ndarray, X_va_df: pd.DataFrame, y_va: np.ndarray,
                        thr_grid: np.ndarray, prior_params: Dict[str, Any], timeout_s: float = 60.0) -> Dict[str, Any]:
    """Run a short Optuna search centered around prior_params; return best found params or prior on failure."""
    best_params = clamp_lgb_param_ranges(prior_params)
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = TPESampler(seed=46, n_startup_trials=15)
        pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=5)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name="adv_meta_local_refine")
        objective = create_optuna_objective(X_tr_df, y_tr, X_va_df, y_va, thr_grid, prior_best=best_params)
        study.optimize(objective, timeout=float(max(1.0, timeout_s)), n_jobs=1, show_progress_bar=False)
        if study.best_trial is not None:
            cand = study.best_trial.user_attrs.get("params", None)
            if isinstance(cand, dict) and len(cand) > 0:
                return clamp_lgb_param_ranges(cand)
    except Exception:
        pass
    return best_params


# Signatures for meta-learner from DataFrame/labels
def compute_signature(X_df: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
    n = float(len(y))
    if n <= 0:
        return {"n": 0.0, "pos_rate": 0.5, "feat_mean_avg": 0.0, "feat_std_avg": 0.0}
    pos_rate = float(np.mean(y))
    feat_mean_avg = float(np.mean(X_df.values)) if X_df.size else 0.0
    feat_std_avg = float(np.std(X_df.values)) if X_df.size else 0.0
    return {"n": n, "pos_rate": pos_rate, "feat_mean_avg": feat_mean_avg, "feat_std_avg": feat_std_avg}

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


# %% THREE-WINDOW / MULTI-WINDOW SPLIT
NUM_TEST_WINDOWS = _get_num_test_windows(default=1)
NUM_GAP_WINDOWS = _get_num_gap_windows(default=0)

# Show command-line argument usage
if ARGS.test_windows is not None or ARGS.gap_windows is not None or ARGS.auto_choice is not None:
    print(f"  ⚙️  Command-line args: test_windows={ARGS.test_windows}, gap_windows={ARGS.gap_windows}, auto_choice={ARGS.auto_choice}")

# Set hardcoded bests based on configuration
HARDCODED_BESTS_RUN2 = get_hardcoded_bests(NUM_TEST_WINDOWS, NUM_GAP_WINDOWS)
config_name = f"{NUM_TEST_WINDOWS}T_{NUM_GAP_WINDOWS}G"
print(f"  ⚙️  Configuration: {config_name} | Using hardcoded bests with {len(HARDCODED_BESTS_RUN2)} fold entries")

if NUM_TEST_WINDOWS == 1:
    splits = split_three_windows(df, gap_windows=NUM_GAP_WINDOWS)
    train_df = splits["train"]
    pre_df = splits["pretest"]
    test_windows = [splits["test"]]
    pretest_bounds = splits["pretest_bounds"]
    test_bounds_list = [splits["test_bounds"]]
else:
    splits_m = split_multi_windows(df, NUM_TEST_WINDOWS, gap_windows=NUM_GAP_WINDOWS)
    train_df = splits_m["train"]
    pre_df = splits_m["pretest"]
    test_windows = splits_m["test_windows"]
    pretest_bounds = splits_m["pretest_bounds"]
    test_bounds_list = splits_m["test_bounds"]

total_test_rows = sum(len(tw) for tw in test_windows)
ignored_rows = TEST_ROWS * max(0, int(NUM_GAP_WINDOWS))
print(f"Train rows: {len(train_df):,}, Pre-test rows: {len(pre_df):,}, Test windows: {NUM_TEST_WINDOWS} (each {TEST_ROWS}, total {total_test_rows}), Gaps: {NUM_GAP_WINDOWS} (ignored {ignored_rows})")

# Default: allow prompt/skip when hardcoded bests exist for default settings
FORCE_SEARCH_ALL_FOLDS = False

SHIFT_FOLD_OFFSET = 0
if NUM_TEST_WINDOWS > 1:
    # Map known-bests by the amount pre-test start moved earlier: ceil(extra_test_rows / fold_size)
    extra_rows = (NUM_TEST_WINDOWS - 1) * TEST_ROWS
    SHIFT_FOLD_OFFSET = max(0, (extra_rows + PRETEST_FOLD_SIZE - 1) // PRETEST_FOLD_SIZE)
    if SHIFT_FOLD_OFFSET > 0:
        print(f"  ⚙️  Known-bests fold index shift applied: {SHIFT_FOLD_OFFSET} (extra_rows={extra_rows}, fold_size={PRETEST_FOLD_SIZE})")


# %% FEATURE EXTRACTION
X_train, y_train = extract_Xy(train_df)
X_pre, y_pre = extract_Xy(pre_df)
tests_Xy: List[Tuple[np.ndarray, np.ndarray]] = [extract_Xy(tw) for tw in test_windows]
print(f"Shapes — Train: {X_train.shape}, Pre-test: {X_pre.shape}, Tests: {[x.shape for x,_ in tests_Xy]}")


# %% SCALING
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_pre_s = scaler.transform(X_pre)
tests_s = [scaler.transform(Xt) if len(Xt) else np.empty((0, X_train_s.shape[1])) for Xt, _ in tests_Xy]

# Wrap scaled arrays in DataFrames with stable feature names to avoid sklearn/lightgbm warnings
X_train_df = pd.DataFrame(X_train_s, columns=FEATURE_NAMES)
X_pre_df = pd.DataFrame(X_pre_s, columns=FEATURE_NAMES)
X_tests_df: List[pd.DataFrame] = [
    pd.DataFrame(Xs, columns=FEATURE_NAMES) if len(Xs) else pd.DataFrame(np.empty((0, X_train_s.shape[1])), columns=FEATURE_NAMES)
    for Xs in tests_s
]


# %% BASELINE
baseline_model = lgb.LGBMClassifier(**default_lgb_params())
baseline_model.fit(X_train_df, y_train)


# %% PRE-TEST: CUMULATIVE FOLD OPTIMIZATION
print("\n" + "="*70)
print("  PRE-TEST HYPERPARAMETER OPTIMIZATION (CUMULATIVE, TIME-LIMITED) — RUN2")
print("="*70)

thr_grid = np.round(np.arange(0.30, 0.80 + 1e-9, 0.01), 2)
fold_meta: List[Dict[str, Any]] = []
best_settings: List[Dict[str, Any]] = []
best_thr_list: List[float] = []
fold_summaries: List[Dict[str, Any]] = []
meta_learner = MetaLearner()
prior_best_params: Optional[Dict[str, Any]] = None

pre_n = len(y_pre)
folds_list = list(cumulative_pretest_splits(pre_n, PRETEST_FOLD_SIZE, GAP_BARS))
fold_idx = 1
while fold_idx <= len(folds_list):
    train_pre_end, val_start, val_end = folds_list[fold_idx - 1]
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
    print(f"✓ RUN2 FOLD {fold_idx} | Train: {len(y_tr)} bars (pos={tr_pos}, neg={tr_neg}) | Val: {len(y_va)} bars (pos={va_pos}, neg={va_neg})")
    print(f"  Train-pre end idx: {train_pre_end} | Val idx: [{val_start}, {val_end}) | Gap={GAP_BARS}")
    if len(np.unique(y_va)) < 2:
        print("  ⚠️  Validation is single-class ⇒ AUC undefined; scoring will ignore AUC for this fold.")
    
    # Compute fold timestamp key for registry lookup
    fold_ts_key: Optional[str] = None
    fold_start_ts = None
    fold_end_ts = None
    try:
        val_ts = pre_df["timestamp"].iloc[val_start:val_end]
        fold_start_ts = pd.to_datetime(val_ts.iloc[0]) if len(val_ts) else None
        fold_end_ts = pd.to_datetime(val_ts.iloc[-1]) if len(val_ts) else None
        if fold_start_ts is not None and fold_end_ts is not None:
            fold_ts_key = f"{fold_start_ts.strftime('%Y-%m-%d')}_{fold_end_ts.strftime('%Y-%m-%d')}"
            print(f"  Val timestamps: {fold_start_ts} → {fold_end_ts}")
            print(f"  Fold key (timestamp): {fold_ts_key}")
    except Exception as e:
        print(f"  ⚠️  Failed to compute fold timestamp key: {e}")

    # Default behavior: search fresh per fold
    skip_search = False
    has_known_for_fold = False
    # If we have a hardcoded best for this fold key, prompt user (even on non-default runs)
    hardcoded_params = None
    if fold_ts_key in HARDCODED_BESTS_RUN2:
        hardcoded_params = HARDCODED_BESTS_RUN2[fold_ts_key].get("params")
        if hardcoded_params is not None and not FORCE_SEARCH_ALL_FOLDS:
            has_known_for_fold = True
            # Inform about configuration match
            if NUM_TEST_WINDOWS == 1 and NUM_GAP_WINDOWS == 0:
                print(f"  ✓ Using stored params optimized for configuration: 1T_0G")
            elif NUM_TEST_WINDOWS == 1 and NUM_GAP_WINDOWS == 1:
                print(f"  ✓ Using stored params optimized for configuration: 1T_1G")
            elif NUM_TEST_WINDOWS == 1 and NUM_GAP_WINDOWS == 2:
                print(f"  ✓ Using stored params optimized for configuration: 1T_2G")
            else:
                print(f"  ⚠️  Configuration {NUM_TEST_WINDOWS}T_{NUM_GAP_WINDOWS}G not specifically tuned. "
                      f"Using 1T_0G params as fallback.")
            # Prompt the user interactively/non-interactively
            prompt = (
                "\n  Would you like to:\n"
                "    [1] Run full hyperparameter search (~5 min)\n"
                "    [2] Skip and use best params stored for this fold\n"
                "    [o] Overnight finetune (12h, MCC-targeted, excludes tough params)\n"
                "  Enter choice (1, 2, or o): "
            )
            choice = get_user_choice(prompt, has_known=True, default_choice="2")
            if choice == "0":
                # User wants to undo - go back to PREVIOUS fold
                if fold_idx <= 1:
                    print(f"  ⚠️  Cannot undo: already at first fold")
                    continue
                prev_fold_idx = fold_idx - 1
                # Remove last entries from tracking lists
                if fold_meta:
                    fold_meta.pop()
                if best_settings:
                    best_settings.pop()
                if best_thr_list:
                    best_thr_list.pop()
                if fold_summaries:
                    fold_summaries.pop()
                # Rewind to previous fold
                fold_idx = prev_fold_idx
                print(f"  ↩️  Undo: Rewinding to RUN2 fold {prev_fold_idx} — forgetting what just happened")
                continue
            skip_search = (choice == "2")
            overnight_mode = (choice.lower() == "o")
            # Secondary choices when user opts to search despite known-bests (suppressed in overnight mode)
            search_with_best = False
            force_jump_to_phase3 = False
            force_jump_to_phase6 = False
            include_tough_params_guided = False
            if not skip_search and has_known_for_fold and not overnight_mode:
                sub_prompt = (
                    "\n  Secondary option for known-best fold:\n"
                    "    [1] Search with Best (seed from known-best and jump)\n"
                    "    [2] Full regular search from Phase 1\n"
                    "  Enter choice (1 or 2): "
                )
                sub_choice = get_user_choice(sub_prompt, has_known=False, default_choice="1")
                if sub_choice == "1":
                    search_with_best = True
                    # Choose jump target (keep historical default: Phase 3)
                    jump_prompt = (
                        "\n  Jump target for 'Search with Best':\n"
                        "    [3] Start at Phase 3 (Creative) — default and current behavior\n"
                        "    [6] Start at Phase 6 (Hyper-Focused) — skip 3/4/5\n"
                        "  Enter choice (3 or 6): "
                    )
                    try:
                        jump_choice = input(jump_prompt).strip()
                    except Exception:
                        jump_choice = "3"
                    if jump_choice == "6":
                        force_jump_to_phase6 = True
                    else:
                        force_jump_to_phase3 = True
                    include_tough_params_guided = True

    if not skip_search:
        # Calculate total time budget
        total_time_budget = max(0.0, FOLD_TIME_LIMIT - (time.time() - fold_t0))

        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Allow bypassing normal phases when overnight mode is selected
        bypass_normal_phases = ('overnight_mode' in locals() and overnight_mode)

        # Two modes: (A) full search (phases 1-2) or (B) 'Search with Best' seeded jump to Phase 3 or 6
        if ('search_with_best' in locals() and search_with_best) and not bypass_normal_phases:
            target_phase = 6 if ('force_jump_to_phase6' in locals() and force_jump_to_phase6) else 3
            print(f"  ⚡ Seeding with known-best params and jumping directly to phase {target_phase} and beyond")
            # Evaluate known-best on validation to seed best score/stats
            try:
                params_use = clamp_lgb_param_ranges(hardcoded_params or default_lgb_params())
                clf_seed = lgb.LGBMClassifier(**params_use)
                bt = params_use.get("boosting_type", "gbdt")
                fit_callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)] if bt != "dart" else []
                clf_seed.fit(
                    X_tr_df,
                    y_tr,
                    eval_set=[(X_va_df, y_va)],
                    eval_metric="auc",
                    callbacks=fit_callbacks
                )
                proba_seed = clf_seed.predict_proba(X_va_df)[:, 1] if len(X_va_df) else np.array([])
                stats_seed = evaluate_thresholds(y_va, proba_seed, thr_grid) if len(proba_seed) else {"thr":0.5,"acc":float("nan"),"auc":None,"prec":float("nan"),"rec":float("nan"),"f1":float("nan"),"bacc":float("nan"),"mcc":float("nan")}
                score_seed = float(score_fold({"f1":stats_seed.get("f1",0),"auc":stats_seed.get("auc"),"acc":stats_seed.get("acc",0),"bacc":stats_seed.get("bacc",0)}))
                # Create a minimal pseudo-trial to seed downstream logic
                pseudo = SimpleNamespace(value=score_seed, user_attrs={"stats": stats_seed, "params": params_use})
                study_e = None
                study_g = None
                all_trials = [pseudo]
                print(f"  ✓ Seeded best from known params | Score={score_seed:.4f} | F1={stats_seed.get('f1', float('nan')):.4f} | MCC={stats_seed.get('mcc', float('nan')):.4f} | θ={stats_seed.get('thr', 0.5):.2f}")
            except Exception as se:
                print(f"  ⚠️  Failed to seed from known-best params: {str(se)} — falling back to full search")
                search_with_best = False
        
        # Overnight mode: train once on known-best, seed trials, then jump straight to Phase 9
        if bypass_normal_phases:
            try:
                params_use = clamp_lgb_param_ranges(hardcoded_params or default_lgb_params())
                clf_seed = lgb.LGBMClassifier(**params_use)
                bt = params_use.get("boosting_type", "gbdt")
                fit_callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)] if bt != "dart" else []
                clf_seed.fit(
                    X_tr_df,
                    y_tr,
                    eval_set=[(X_va_df, y_va)],
                    eval_metric="auc",
                    callbacks=fit_callbacks
                )
                proba_seed = clf_seed.predict_proba(X_va_df)[:, 1] if len(X_va_df) else np.array([])
                stats_seed = evaluate_thresholds(y_va, proba_seed, thr_grid) if len(proba_seed) else {"thr":0.5,"acc":float("nan"),"auc":None,"prec":float("nan"),"rec":float("nan"),"f1":float("nan"),"bacc":float("nan"),"mcc":float("nan")}
                score_seed = float(score_fold({"f1":stats_seed.get("f1",0),"auc":stats_seed.get("auc"),"acc":stats_seed.get("acc",0),"bacc":stats_seed.get("bacc",0)}))
                pseudo = SimpleNamespace(value=score_seed, user_attrs={"stats": stats_seed, "params": params_use})
                study_e = None
                study_g = None
                all_trials = [pseudo]
                print(f"  ✓ Overnight seed from known params | Score={score_seed:.4f} | F1={stats_seed.get('f1', float('nan')):.4f} | MCC={stats_seed.get('mcc', float('nan')):.4f} | θ={stats_seed.get('thr', 0.5):.2f}")
            except Exception as se:
                print(f"  ⚠️  Failed to seed from known-best params for overnight: {str(se)} — proceeding with default seed")
                all_trials = []

        if (('search_with_best' not in locals()) or (not search_with_best)) and not bypass_normal_phases:
            # Phase planning: first 3 folds = exploratory only; from fold 4+: 3 min exploratory + 2 min prior-guided
            is_first_three = fold_idx <= 3
            phase1_budget = min(180.0, total_time_budget) if not is_first_three else total_time_budget
            phase2_budget = 0.0 if is_first_three else max(0.0, min(120.0, total_time_budget - phase1_budget))

            # PHASE 1: Exploratory search
            print(f"  🔍 Phase 1 — Exploratory (RUN2) | Budget: {phase1_budget:.1f}s")
            sampler_e = TPESampler(seed=42, n_startup_trials=30)
            pruner_e = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            study_e = optuna.create_study(direction="maximize", sampler=sampler_e, pruner=pruner_e, study_name=f"run2_fold_{fold_idx}_explore")
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
                print(f"  ⚠️  Phase 1 optimization error (RUN2): {str(e)}")
            
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
                
                print(f"  🔍 Phase 2 — Prior-guided (RUN2) | Budget: {min(phase2_budget, time_left):.1f}s")
                sampler_g = TPESampler(seed=42, n_startup_trials=20)
                pruner_g = MedianPruner(n_startup_trials=8, n_warmup_steps=5)
                study_g = optuna.create_study(direction="maximize", sampler=sampler_g, pruner=pruner_g, study_name=f"run2_fold_{fold_idx}_guided")
                objective_g = create_optuna_objective(X_tr_df, y_tr, X_va_df, y_va, thr_grid, prior_best=guided_prior)
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
                    print(f"  ⚠️  Phase 2 optimization error (RUN2): {str(e)}")
            
            # Combine results from phases (E + optional G)
            all_trials = list(study_e.trials)
            if study_g is not None:
                all_trials += list(study_g.trials)

        # OPTIONAL: Overnight finetune Phase 9 (runs after initial phases when overnight_mode)
        if 'overnight_mode' in locals() and overnight_mode:
            print(f"\n{'='*70}")
            print(f"  🌙 PHASE 9 — OVERNIGHT FINETUNE (ENHANCED)")
            print(f"{'='*70}")
            print(f"  Budget: {OVERNIGHT_TIME_LIMIT_S/3600:.1f}h | Target MCC≥{OVERNIGHT_MCC_TARGET:.2f} | Tough params excluded")
            print(f"  Strategy: Multi-stage adaptive search with checkpointing")
            
            phase9_start_time = time.time()
            
            # Determine seed prior and establish baseline
            seed_prior = None
            baseline_mcc = float('nan')
            try:
                # Use best so far if exists
                best_trial_tmp = None
                best_val_tmp = -np.inf
                for _t in all_trials:
                    if _t.value is not None and _t.value > best_val_tmp:
                        best_val_tmp = _t.value
                        best_trial_tmp = _t
                if best_trial_tmp is not None:
                    seed_prior = best_trial_tmp.user_attrs.get("params", None)
                    baseline_stats = best_trial_tmp.user_attrs.get("stats", {})
                    baseline_mcc = baseline_stats.get("mcc", float('nan'))
            except Exception:
                seed_prior = None
            if seed_prior is None:
                seed_prior = clamp_lgb_param_ranges(hardcoded_params or default_lgb_params())
            
            # Evaluate seed to get baseline if not available
            if np.isnan(baseline_mcc):
                try:
                    clf_tmp = lgb.LGBMClassifier(**seed_prior)
                    clf_tmp.fit(X_tr_df, y_tr)
                    p_tmp = clf_tmp.predict_proba(X_va_df)[:,1] if len(X_va_df) else np.array([])
                    if len(p_tmp):
                        st_tmp = evaluate_thresholds(y_va, p_tmp, thr_grid)
                        baseline_mcc = st_tmp.get("mcc", float('nan'))
                except Exception:
                    baseline_mcc = 0.0
            
            mcc_gap = float(OVERNIGHT_MCC_TARGET) - (0.0 if np.isnan(baseline_mcc) else float(baseline_mcc))
            mcc_gap = max(0.0, mcc_gap)
            
            print(f"  Baseline MCC: {baseline_mcc:.4f} | Gap to target: {mcc_gap:.4f}")
            print(f"  Seed params: lr={seed_prior.get('learning_rate', 0.05):.4f}, leaves={seed_prior.get('num_leaves', 63)}, depth={seed_prior.get('max_depth', 6)}")
            
            # Warm restart: try loading checkpoint
            checkpoint_path = Path(RUN_DIR, f"overnight_checkpoint_fold_{fold_idx}.json")
            checkpoint_loaded = False
            if checkpoint_path.exists():
                try:
                    with open(checkpoint_path, 'r') as f:
                        ckpt = json.load(f)
                    if ckpt.get("best_params") is not None and ckpt.get("best_mcc", 0.0) > baseline_mcc:
                        seed_prior = ckpt["best_params"]
                        baseline_mcc = ckpt["best_mcc"]
                        checkpoint_loaded = True
                        print(f"  ✓ Loaded checkpoint: MCC={baseline_mcc:.4f} (better than baseline)")
                except Exception:
                    pass
            
            # Safety limit: max trials
            max_trials = int(os.getenv("MCP_OVERNIGHT_MAX_TRIALS", "10000"))
            
            # Progress tracking state
            overnight_state = {
                "best_mcc": baseline_mcc,
                "best_score": best_val_tmp if best_trial_tmp is not None else 0.0,
                "best_params": seed_prior,
                "trials_since_improvement": 0,
                "last_log_time": time.time(),
                "trial_count": 0,
                "mcc_history": [baseline_mcc] if not np.isnan(baseline_mcc) else [],
                "stage": "exploration",
                "stage_start_trial": 0,
            }
            
            # Enhanced callback: progress tracking, checkpointing, smart early-stop
            def _overnight_enhanced_cb(study, trial):
                try:
                    overnight_state["trial_count"] += 1
                    st = trial.user_attrs.get("stats", {})
                    mcc_val = st.get("mcc", float("nan"))
                    score_val = trial.value if trial.value is not None else 0.0
                    
                    # Update best if improved
                    improved = False
                    if not np.isnan(mcc_val):
                        overnight_state["mcc_history"].append(float(mcc_val))
                        if float(mcc_val) > float(overnight_state["best_mcc"]):
                            overnight_state["best_mcc"] = float(mcc_val)
                            overnight_state["best_score"] = float(score_val)
                            overnight_state["best_params"] = trial.user_attrs.get("params", seed_prior)
                            overnight_state["trials_since_improvement"] = 0
                            improved = True
                        else:
                            overnight_state["trials_since_improvement"] += 1
                    
                    # Periodic progress logging (every 5 min)
                    now = time.time()
                    if now - overnight_state["last_log_time"] >= 300:
                        elapsed_h = (now - phase9_start_time) / 3600.0
                        remaining_h = max(0, (OVERNIGHT_TIME_LIMIT_S - (now - phase9_start_time)) / 3600.0)
                        recent_mcc_avg = float(np.mean(overnight_state["mcc_history"][-20:])) if len(overnight_state["mcc_history"]) >= 20 else baseline_mcc
                        print(f"    [{elapsed_h:.1f}h] Trial {overnight_state['trial_count']} | Best MCC: {overnight_state['best_mcc']:.4f} | Recent avg: {recent_mcc_avg:.4f} | Stage: {overnight_state['stage']} | {remaining_h:.1f}h left")
                        overnight_state["last_log_time"] = now
                        
                        # Save checkpoint
                        try:
                            ckpt_data = {
                                "best_mcc": float(overnight_state["best_mcc"]),
                                "best_score": float(overnight_state["best_score"]),
                                "best_params": overnight_state["best_params"],
                                "trial_count": overnight_state["trial_count"],
                                "stage": overnight_state["stage"],
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                            with open(checkpoint_path, 'w') as f:
                                json.dump(ckpt_data, f, indent=2)
                        except Exception:
                            pass
                    
                    # Smart early-stop conditions
                    # 1. MCC target reached
                    if float(overnight_state["best_mcc"]) >= float(OVERNIGHT_MCC_TARGET):
                        print(f"    🎯 TARGET REACHED: MCC={overnight_state['best_mcc']:.4f} ≥ {OVERNIGHT_MCC_TARGET:.2f} (trial {overnight_state['trial_count']})")
                        study.stop()
                        return
                    
                    # 2. Stagnation handling disabled for overnight runs to honor full time budget
                    #    We rely on time budgets and MCC target only to stop.
                    
                    # 3. Max trials safety limit
                    if overnight_state["trial_count"] >= max_trials:
                        print(f"    🛑 MAX TRIALS REACHED: {max_trials} trials completed")
                        study.stop()
                        return
                    
                    # Stage transitions: exploration -> exploitation after 40% of time or 100 trials
                    if overnight_state["stage"] == "exploration":
                        elapsed_frac = (now - phase9_start_time) / float(OVERNIGHT_TIME_LIMIT_S)
                        if elapsed_frac >= 0.40 or overnight_state["trial_count"] >= 100:
                            overnight_state["stage"] = "exploitation"
                            overnight_state["stage_start_trial"] = overnight_state["trial_count"]
                            print(f"    🔄 STAGE TRANSITION: exploration → exploitation at trial {overnight_state['trial_count']}")
                    
                except Exception as ex:
                    print(f"    ⚠️  Callback error: {str(ex)}")
            
            # Multi-stage search
            print(f"\n  🔍 Stage 1: Exploration (TPE with wide search)")
            
            # Stage 1: Exploration with TPE
            if mcc_gap > 0.15:
                sampler_stage1 = TPESampler(seed=77, n_startup_trials=80, multivariate=True)
                pruner_stage1 = MedianPruner(n_startup_trials=15, n_warmup_steps=5)
            elif mcc_gap > 0.10:
                sampler_stage1 = TPESampler(seed=77, n_startup_trials=60, multivariate=True)
                pruner_stage1 = MedianPruner(n_startup_trials=12, n_warmup_steps=5)
            else:
                sampler_stage1 = TPESampler(seed=77, n_startup_trials=40, multivariate=True)
                pruner_stage1 = MedianPruner(n_startup_trials=10, n_warmup_steps=4)
            
            objective_stage1 = create_optuna_objective(
                X_tr_df, y_tr, X_va_df, y_va, thr_grid, prior_best=seed_prior, mcc_gap=mcc_gap, include_tough_params=False
            )
            
            study_stage1 = optuna.create_study(direction="maximize", sampler=sampler_stage1, pruner=pruner_stage1, study_name=f"run2_fold_{fold_idx}_overnight_stage1")
            
            try:
                # Exploration phase: Strict 40% of total budget
                stage1_timeout = float(OVERNIGHT_TIME_LIMIT_S) * 0.40
                study_stage1.optimize(
                    objective_stage1,
                    timeout=max(1.0, stage1_timeout),
                    n_jobs=1,
                    show_progress_bar=False,
                    callbacks=[_overnight_enhanced_cb],
                    # Do not cap n_trials here; let time budget control duration
                )
                all_trials += list(study_stage1.trials)
            except Exception as e:
                if "stop" not in str(e).lower():
                    print(f"    ⚠️  Stage 1 error: {str(e)}")
            
            # Check if we should continue to stage 2
            time_left = max(0, OVERNIGHT_TIME_LIMIT_S - (time.time() - phase9_start_time))
            if time_left > 60 and overnight_state["best_mcc"] < float(OVERNIGHT_MCC_TARGET) and overnight_state["trial_count"] < max_trials:
                print(f"\n  🔍 Stage 2: Exploitation (Narrow TPE around best + CMA-ES if available)")
                overnight_state["stage"] = "exploitation"
                
                # Update seed from best found
                exploit_prior = overnight_state["best_params"]
                
                # Stage 2A: Narrow TPE exploitation
                sampler_stage2a = TPESampler(seed=78, n_startup_trials=20, multivariate=True)
                pruner_stage2a = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
                objective_stage2a = create_optuna_objective(
                    X_tr_df, y_tr, X_va_df, y_va, thr_grid, prior_best=exploit_prior, mcc_gap=mcc_gap, include_tough_params=False
                )
                study_stage2a = optuna.create_study(direction="maximize", sampler=sampler_stage2a, pruner=pruner_stage2a, study_name=f"run2_fold_{fold_idx}_overnight_stage2a")
                
                try:
                    # Exploitation phase A: take 50% of remaining time (reserve rest for CMA-ES)
                    stage2a_timeout = max(1.0, time_left * 0.50)
                    study_stage2a.optimize(
                        objective_stage2a,
                        timeout=stage2a_timeout,
                        n_jobs=1,
                        show_progress_bar=False,
                        callbacks=[_overnight_enhanced_cb],
                        # Do not hard-cap by trials; time budget controls it
                    )
                    all_trials += list(study_stage2a.trials)
                except Exception as e:
                    if "stop" not in str(e).lower():
                        print(f"    ⚠️  Stage 2A error: {str(e)}")
                
                # Stage 2B: CMA-ES if available and time remains
                time_left = max(0, OVERNIGHT_TIME_LIMIT_S - (time.time() - phase9_start_time))
                if time_left > 60 and _has_cmaes() and overnight_state["best_mcc"] < float(OVERNIGHT_MCC_TARGET) and overnight_state["trial_count"] < max_trials:
                    print(f"    🔬 Stage 2B: CMA-ES refinement")
                    exploit_prior = overnight_state["best_params"]
                    sigma_cma = 0.10 if mcc_gap > 0.10 else 0.05
                    sampler_cma = CmaEsSampler(seed=79, sigma0=sigma_cma)
                    pruner_cma = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
                    objective_cma = create_optuna_objective(
                        X_tr_df, y_tr, X_va_df, y_va, thr_grid, prior_best=exploit_prior, mcc_gap=mcc_gap, include_tough_params=False
                    )
                    study_cma = optuna.create_study(direction="maximize", sampler=sampler_cma, pruner=pruner_cma, study_name=f"run2_fold_{fold_idx}_overnight_cma")
                    
                    try:
                        # Consume remaining time; let time, not trials, control duration
                        study_cma.optimize(
                            objective_cma,
                            timeout=max(1.0, time_left),
                            n_jobs=1,
                            show_progress_bar=False,
                            callbacks=[_overnight_enhanced_cb],
                        )
                        all_trials += list(study_cma.trials)
                    except Exception as e:
                        if "stop" not in str(e).lower():
                            print(f"    ⚠️  CMA-ES error: {str(e)}")
            
            # Final summary (always at end of total budget unless MCC target reached)
            phase9_elapsed = time.time() - phase9_start_time
            improvement = overnight_state["best_mcc"] - (baseline_mcc if not np.isnan(baseline_mcc) else 0.0)
            print(f"\n{'='*70}")
            print(f"  🌙 PHASE 9 COMPLETE")
            print(f"{'='*70}")
            print(f"  Total time: {phase9_elapsed/3600:.2f}h | Trials: {overnight_state['trial_count']}")
            print(f"  Baseline MCC: {baseline_mcc:.4f} → Final MCC: {overnight_state['best_mcc']:.4f} (Δ={improvement:+.4f})")
            print(f"  Target: {OVERNIGHT_MCC_TARGET:.2f} | {'✅ REACHED' if overnight_state['best_mcc'] >= float(OVERNIGHT_MCC_TARGET) else '❌ Not reached'}")
            print(f"  Best params saved to checkpoint: {checkpoint_path.name}")
            print(f"{'='*70}\n")

        # Helper to compute current best
        def _current_best(trials_list):
            best_t = None
            best_v = -np.inf
            for _t in trials_list:
                if _t.value is not None and _t.value > best_v:
                    best_v = _t.value
                    best_t = _t
            return best_t, best_v

        # Check if we should trigger extension phases based on interim best
        best_trial, best_value = _current_best(all_trials)
        interim_stats = best_trial.user_attrs.get("stats", {}) if best_trial is not None else {}
        interim_thr = float(interim_stats.get("thr", 0.5)) if interim_stats else 0.5
        interim_auc = interim_stats.get("auc", None) if interim_stats else None
        interim_mcc = interim_stats.get("mcc", float("nan")) if interim_stats else float("nan")
        needs_extension = False
        if best_trial is not None:
            # Trigger if AUC undefined or below threshold OR threshold at/falls below floor OR MCC <= 0.60
            try:
                auc_bad = (interim_auc is None) or (not np.isnan(interim_auc) and float(interim_auc) < float(AUC_THRESHOLD_FOR_EXTENSION))
            except Exception:
                auc_bad = True
            try:
                thr_bad = (float(interim_thr) <= float(THRESHOLD_FLOOR_TRIGGER))
            except Exception:
                thr_bad = False
            try:
                mcc_bad = (np.isnan(interim_mcc) or float(interim_mcc) <= 0.60)
            except Exception:
                mcc_bad = True
            needs_extension = bool(auc_bad or thr_bad or mcc_bad)

        # PHASE 3: Creative extension (wide search) — independent time budget
        study_x = None
        if needs_extension and EXTENSION_TIME_LIMIT > 0 and (('search_with_best' not in locals()) or (not search_with_best) or ('force_jump_to_phase3' in locals() and force_jump_to_phase3)):
            reasons = []
            if auc_bad:
                reasons.append(f"AUC={'undefined' if interim_auc is None else f'{interim_auc:.4f}'} < {AUC_THRESHOLD_FOR_EXTENSION}")
            if thr_bad:
                reasons.append(f"θ≤{THRESHOLD_FLOOR_TRIGGER}")
            if mcc_bad:
                reasons.append(f"MCC={'undefined' if np.isnan(interim_mcc) else f'{interim_mcc:.4f}'} ≤ 0.60")
            reason_str = " or ".join(reasons)
            print(f"  🔍 Phase 3 — Creative Extension (RUN2) | Budget: {EXTENSION_TIME_LIMIT:.1f}s | Reason: {reason_str}")
            try:
                sampler_x = TPESampler(seed=43, n_startup_trials=20)
                pruner_x = MedianPruner(n_startup_trials=8, n_warmup_steps=5)
                study_x = optuna.create_study(direction="maximize", sampler=sampler_x, pruner=pruner_x, study_name=f"run2_fold_{fold_idx}_extend_creative")
                objective_x = create_optuna_objective(X_tr_df, y_tr, X_va_df, y_va, thr_grid, prior_best=None)
                study_x.optimize(
                    objective_x,
                    timeout=float(EXTENSION_TIME_LIMIT),
                    n_jobs=1,
                    show_progress_bar=False,
                    callbacks=[
                        lambda study, trial: print(
                            f"  ✓ [X] Trial {trial.number+1:>3} | Score: {trial.value:.4f} → {'NEW BEST' if trial.value == study.best_value else 'continue'} | "
                            f"F1={trial.user_attrs.get('stats', {}).get('f1', 0):.4f}, "
                            f"AUC={trial.user_attrs.get('stats', {}).get('auc') if trial.user_attrs.get('stats', {}).get('auc') is not None else 'undefined'}, "
                            f"ACC={trial.user_attrs.get('stats', {}).get('acc', 0):.4f}, "
                            f"θ={trial.user_attrs.get('stats', {}).get('thr', 0.5):.2f} | "
                            f"lr={trial.params.get('learning_rate', 0):.4f}, leaves={trial.params.get('num_leaves', 0)}, depth={trial.params.get('max_depth', 0)}"
                        ) if trial.value > 0 and trial.number % 5 == 0 else None
                    ]
                )
                all_trials += list(study_x.trials)
            except Exception as e:
                print(f"  ⚠️  Phase 3 extension error (RUN2): {str(e)}")

        # Reassess after Phase 3 to decide Phase 4/5
        best_trial, best_value = _current_best(all_trials)
        interim_stats = best_trial.user_attrs.get("stats", {}) if best_trial is not None else {}
        interim_thr = float(interim_stats.get("thr", 0.5)) if interim_stats else 0.5
        interim_auc = interim_stats.get("auc", None) if interim_stats else None
        def _bad_enough(auc_val, thr_val, mcc_val=None) -> bool:
            try:
                a_bad = (auc_val is None) or (not np.isnan(auc_val) and float(auc_val) < float(AUC_THRESHOLD_FOR_EXTENSION))
            except Exception:
                a_bad = True
            try:
                t_bad = (float(thr_val) <= float(THRESHOLD_FLOOR_TRIGGER))
            except Exception:
                t_bad = False
            try:
                m_bad = (mcc_val is None) or (np.isnan(mcc_val) or float(mcc_val) <= 0.60)
            except Exception:
                m_bad = True
            return bool(a_bad or t_bad or m_bad)

        # PHASE 4: Ultra-guided extension (narrow around current best)
        study_u = None
        interim_mcc_phase4 = interim_stats.get("mcc", float("nan")) if interim_stats else float("nan")
        if _bad_enough(interim_auc, interim_thr, interim_mcc_phase4) and ULTRA_EXTENSION_TIME_LIMIT > 0 and (('search_with_best' not in locals()) or (not search_with_best) or ('force_jump_to_phase3' in locals() and force_jump_to_phase3)):
            print(f"  🔍 Phase 4 — Ultra-Guided Extension (RUN2) | Budget: {ULTRA_EXTENSION_TIME_LIMIT:.1f}s")
            try:
                guided_prior = best_trial.user_attrs.get("params", None) if best_trial is not None else None
                sampler_u = TPESampler(seed=44, n_startup_trials=15)
                pruner_u = MedianPruner(n_startup_trials=6, n_warmup_steps=5)
                study_u = optuna.create_study(direction="maximize", sampler=sampler_u, pruner=pruner_u, study_name=f"run2_fold_{fold_idx}_extend_ultra")
                objective_u = create_optuna_objective(X_tr_df, y_tr, X_va_df, y_va, thr_grid, prior_best=guided_prior, include_tough_params=('include_tough_params_guided' in locals() and include_tough_params_guided))
                study_u.optimize(
                    objective_u,
                    timeout=float(ULTRA_EXTENSION_TIME_LIMIT),
                    n_jobs=1,
                    show_progress_bar=False,
                    callbacks=[
                        lambda study, trial: print(
                            f"  ✓ [U] Trial {trial.number+1:>3} | Score: {trial.value:.4f} → {'NEW BEST' if trial.value == study.best_value else 'continue'} | "
                            f"F1={trial.user_attrs.get('stats', {}).get('f1', 0):.4f}, "
                            f"AUC={trial.user_attrs.get('stats', {}).get('auc') if trial.user_attrs.get('stats', {}).get('auc') is not None else 'undefined'}, "
                            f"ACC={trial.user_attrs.get('stats', {}).get('acc', 0):.4f}, "
                            f"θ={trial.user_attrs.get('stats', {}).get('thr', 0.5):.2f} | "
                            f"lr={trial.params.get('learning_rate', 0):.4f}, leaves={trial.params.get('num_leaves', 0)}, depth={trial.params.get('max_depth', 0)}"
                        ) if trial.value > 0 and trial.number % 5 == 0 else None
                    ]
                )
                all_trials += list(study_u.trials)
            except Exception as e:
                print(f"  ⚠️  Phase 4 extension error (RUN2): {str(e)}")

        # PHASE 5: Extreme-guided extension (final push)
        study_z = None
        # Reassess again
        best_trial, best_value = _current_best(all_trials)
        interim_stats = best_trial.user_attrs.get("stats", {}) if best_trial is not None else {}
        interim_thr = float(interim_stats.get("thr", 0.5)) if interim_stats else 0.5
        interim_auc = interim_stats.get("auc", None) if interim_stats else None
        interim_mcc_phase5 = interim_stats.get("mcc", float("nan")) if interim_stats else float("nan")
        if _bad_enough(interim_auc, interim_thr, interim_mcc_phase5) and EXTREME_EXTENSION_TIME_LIMIT > 0 and (('search_with_best' not in locals()) or (not search_with_best) or ('force_jump_to_phase3' in locals() and force_jump_to_phase3)):
            print(f"  🔍 Phase 5 — Extreme-Guided Extension (RUN2) | Budget: {EXTREME_EXTENSION_TIME_LIMIT:.1f}s")
            try:
                guided_prior2 = best_trial.user_attrs.get("params", None) if best_trial is not None else None
                sampler_z = TPESampler(seed=45, n_startup_trials=10)
                pruner_z = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
                study_z = optuna.create_study(direction="maximize", sampler=sampler_z, pruner=pruner_z, study_name=f"run2_fold_{fold_idx}_extend_extreme")
                objective_z = create_optuna_objective(X_tr_df, y_tr, X_va_df, y_va, thr_grid, prior_best=guided_prior2, include_tough_params=('include_tough_params_guided' in locals() and include_tough_params_guided))
                study_z.optimize(
                    objective_z,
                    timeout=float(EXTREME_EXTENSION_TIME_LIMIT),
                    n_jobs=1,
                    show_progress_bar=False,
                    callbacks=[
                        lambda study, trial: print(
                            f"  ✓ [Z] Trial {trial.number+1:>3} | Score: {trial.value:.4f} → {'NEW BEST' if trial.value == study.best_value else 'continue'} | "
                            f"F1={trial.user_attrs.get('stats', {}).get('f1', 0):.4f}, "
                            f"AUC={trial.user_attrs.get('stats', {}).get('auc') if trial.user_attrs.get('stats', {}).get('auc') is not None else 'undefined'}, "
                            f"ACC={trial.user_attrs.get('stats', {}).get('acc', 0):.4f}, "
                            f"θ={trial.user_attrs.get('stats', {}).get('thr', 0.5):.2f} | "
                            f"lr={trial.params.get('learning_rate', 0):.4f}, leaves={trial.params.get('num_leaves', 0)}, depth={trial.params.get('max_depth', 0)}"
                        ) if trial.value > 0 and trial.number % 5 == 0 else None
                    ]
                )
                all_trials += list(study_z.trials)
            except Exception as e:
                print(f"  ⚠️  Phase 5 extension error (RUN2): {str(e)}")

        # PHASE 6: Hyper-focused extension (adaptive micro-tuning based on MCC gap)
        study_h = None
        # Reassess again
        best_trial, best_value = _current_best(all_trials)
        interim_stats = best_trial.user_attrs.get("stats", {}) if best_trial is not None else {}
        interim_thr = float(interim_stats.get("thr", 0.5)) if interim_stats else 0.5
        interim_auc = interim_stats.get("auc", None) if interim_stats else None
        interim_mcc_phase6 = interim_stats.get("mcc", float("nan")) if interim_stats else float("nan")
        if _bad_enough(interim_auc, interim_thr, interim_mcc_phase6) and HYPER_EXTENSION_TIME_LIMIT > 0 and (('search_with_best' not in locals()) or ('force_jump_to_phase3' in locals() and force_jump_to_phase3) or ('force_jump_to_phase6' in locals() and force_jump_to_phase6)):
            # Adaptive ranges based on MCC gap to 0.60
            mcc_gap = 0.60 - (float(interim_mcc_phase6) if not np.isnan(interim_mcc_phase6) else 0.0)
            mcc_gap = max(0.0, mcc_gap)
            print(f"  🔍 Phase 6 — Hyper-Focused Extension (RUN2) | Budget: {HYPER_EXTENSION_TIME_LIMIT:.1f}s")
            print(f"!!! PHASE 6 START: Hyper-Focused Micro-Tuning (Multivariate TPE) !!!")
            print(f"!!! Current Best Before Phase 6: Score={best_value:.4f}, F1={interim_stats.get('f1', float('nan')):.4f}, AUC={interim_auc if interim_auc is not None else 'undefined'}, MCC={interim_mcc_phase6:.4f}, θ={interim_thr:.2f} !!!")
            print(f"!!! Adaptive Tuning Context: mcc_gap_to_0.60={mcc_gap:.4f} (larger gap => explore wider, lower lr, allow deeper trees) !!!")
            
            phase6_start_time = time.time()
            try:
                guided_prior3 = best_trial.user_attrs.get("params", None) if best_trial is not None else None
                if guided_prior3 is None:
                    print(f"!!! WARNING: No prior params found for Phase 6, using defaults !!!")
                    guided_prior3 = default_lgb_params()
                else:
                    print(f"!!! Phase 6 Prior Params: lr={guided_prior3.get('learning_rate', 'N/A'):.4f}, leaves={guided_prior3.get('num_leaves', 'N/A')}, depth={guided_prior3.get('max_depth', 'N/A')}, min_child={guided_prior3.get('min_child_samples', 'N/A')} !!!")
                
                # Phase 6: Adaptive micro-tuning around current best
                # Larger gap = more exploration, more startup trials, less aggressive pruning
                if mcc_gap > 0.15:
                    n_startup = 40
                    n_warmup = 5
                    prune_interval = 3
                    strategy = "Very Wide (gap > 0.15)"
                elif mcc_gap > 0.10:
                    n_startup = 30
                    n_warmup = 4
                    prune_interval = 3
                    strategy = "Wide (gap > 0.10)"
                elif mcc_gap > 0.05:
                    n_startup = 25
                    n_warmup = 3
                    prune_interval = 2
                    strategy = "Moderate (gap > 0.05)"
                else:
                    n_startup = 20
                    n_warmup = 3
                    prune_interval = 2
                    strategy = "Tight (gap ≤ 0.05)"
                
                print(f"!!! Phase 6 Strategy: {strategy} | Startup trials={n_startup}, Warmup={n_warmup}, Prune interval={prune_interval} !!!")
                print(f"!!! Creating Phase 6 Study with adaptive Multivariate TPE !!!")
                sampler_h = TPESampler(seed=46, n_startup_trials=n_startup, multivariate=True)
                pruner_h = MedianPruner(n_startup_trials=5, n_warmup_steps=n_warmup, interval_steps=prune_interval)
                study_h = optuna.create_study(direction="maximize", sampler=sampler_h, pruner=pruner_h, study_name=f"run2_fold_{fold_idx}_extend_hyper")
                # Pass mcc_gap to objective for adaptive parameter ranges
                objective_h = create_optuna_objective(X_tr_df, y_tr, X_va_df, y_va, thr_grid, prior_best=guided_prior3, mcc_gap=mcc_gap, include_tough_params=('include_tough_params_guided' in locals() and include_tough_params_guided))
                
                print(f"!!! Starting Phase 6 Optimization: {HYPER_EXTENSION_TIME_LIMIT:.0f}s timeout !!!")
                study_h.optimize(
                    objective_h,
                    timeout=float(HYPER_EXTENSION_TIME_LIMIT),
                    n_jobs=1,
                    show_progress_bar=False,
                    callbacks=[
                        lambda study, trial: print(
                            f"  ✓ [H] Trial {trial.number+1:>3} | Score: {trial.value:.4f} → {'NEW BEST' if trial.value == study.best_value else 'continue'} | "
                            f"F1={trial.user_attrs.get('stats', {}).get('f1', 0):.4f}, "
                            f"AUC={trial.user_attrs.get('stats', {}).get('auc') if trial.user_attrs.get('stats', {}).get('auc') is not None else 'undefined'}, "
                            f"ACC={trial.user_attrs.get('stats', {}).get('acc', 0):.4f}, "
                            f"θ={trial.user_attrs.get('stats', {}).get('thr', 0.5):.2f} | "
                            f"lr={trial.params.get('learning_rate', 0):.4f}, leaves={trial.params.get('num_leaves', 0)}, depth={trial.params.get('max_depth', 0)}"
                        ) if trial.value > 0 and trial.number % 5 == 0 else None
                    ]
                )
                
                phase6_elapsed = time.time() - phase6_start_time
                phase6_trials = len(study_h.trials) if study_h else 0
                phase6_pruned = len([t for t in study_h.trials if t.state == optuna.trial.TrialState.PRUNED]) if study_h else 0
                
                all_trials += list(study_h.trials)
                
                # Analyze Phase 6 results
                best_after_phase6, best_score_phase6 = _current_best(all_trials)
                if best_after_phase6 is not None:
                    phase6_stats = best_after_phase6.user_attrs.get("stats", {})
                    phase6_improvement = best_score_phase6 - best_value
                    print(f"!!! Phase 6 Complete: {phase6_trials} trials ({phase6_pruned} pruned) in {phase6_elapsed:.1f}s !!!")
                    print(f"!!! Phase 6 Best: Score={best_score_phase6:.4f} (Δ={phase6_improvement:+.4f}), F1={phase6_stats.get('f1', float('nan')):.4f}, AUC={phase6_stats.get('auc') if phase6_stats.get('auc') is not None else 'undefined'}, MCC={phase6_stats.get('mcc', float('nan')):.4f} !!!")
                    if phase6_improvement > 0.001:
                        print(f"!!! Phase 6 SUCCESS: Found improvement of {phase6_improvement:.4f} !!!")
                    else:
                        print(f"!!! Phase 6 NOTE: No significant improvement (Δ={phase6_improvement:+.6f}) - may need Phase 7 !!!")
                else:
                    print(f"!!! WARNING: Phase 6 completed but no best trial found !!!")
                    
            except Exception as e:
                phase6_elapsed = time.time() - phase6_start_time
                print(f"!!! ERROR in Phase 6 after {phase6_elapsed:.1f}s: {type(e).__name__}: {str(e)} !!!")
                import traceback
                print(f"!!! Phase 6 Traceback: {traceback.format_exc()[:300]} !!!")

        # PHASE 7: Ultimate extension — two-step: top-K refinement + CMA-ES around best (adaptive)
        study_ult_refine = None
        study_ult_cma = None
        # Final reassess
        best_trial, best_value = _current_best(all_trials)
        interim_stats = best_trial.user_attrs.get("stats", {}) if best_trial is not None else {}
        interim_thr = float(interim_stats.get("thr", 0.5)) if interim_stats else 0.5
        interim_auc = interim_stats.get("auc", None) if interim_stats else None
        interim_mcc_phase7 = interim_stats.get("mcc", float("nan")) if interim_stats else float("nan")
        if _bad_enough(interim_auc, interim_thr, interim_mcc_phase7) and ULTIMATE_EXTENSION_TIME_LIMIT > 0 and (('search_with_best' not in locals()) or ('force_jump_to_phase3' in locals() and force_jump_to_phase3) or ('force_jump_to_phase6' in locals() and force_jump_to_phase6)):
            mcc_gap7 = 0.60 - (float(interim_mcc_phase7) if not np.isnan(interim_mcc_phase7) else 0.0)
            mcc_gap7 = max(0.0, mcc_gap7)
            print(f"  🔍 Phase 7 — Ultimate Extension (RUN2) | Budget: {ULTIMATE_EXTENSION_TIME_LIMIT:.1f}s")
            print(f"!!! PHASE 7 START: Two-Step Ultimate Extension (Top-K Refine + CMA-ES) !!!")
            print(f"!!! Current Best Before Phase 7: Score={best_value:.4f}, F1={interim_stats.get('f1', float('nan')):.4f}, AUC={interim_auc if interim_auc is not None else 'undefined'}, MCC={interim_mcc_phase7:.4f}, θ={interim_thr:.2f} !!!")
            print(f"!!! Adaptive Tuning Context (Phase 7): mcc_gap_to_0.60={mcc_gap7:.4f} (larger gap => explore broader candidates and bigger late-phase knobs) !!!")
            
            phase7_start_time = time.time()
            try:
                print(f"!!! Phase 7 Step 1: Collecting Top-K Candidate Params from All Prior Phases !!!")
                # Step 1: Evaluate and lightly tune top-K parameter sets via enqueued trials
                # Collect candidate params from all previous phases (best per phase when available)
                candidate_params: List[Dict[str, Any]] = []
                phase_names = ['Exploratory', 'Prior-Guided', 'Creative', 'Ultra', 'Extreme', 'Hyper']
                studies_list = [study_e, study_g, study_x, study_u, study_z, study_h]
                
                def _maybe_add(study, phase_name):
                    try:
                        if study is not None and study.best_trial is not None:
                            p = study.best_trial.user_attrs.get("params", None)
                            if isinstance(p, dict) and p not in candidate_params:
                                candidate_params.append(p)
                                print(f"!!! Added candidate from {phase_name}: lr={p.get('learning_rate', 'N/A'):.4f}, leaves={p.get('num_leaves', 'N/A')}, depth={p.get('max_depth', 'N/A')} !!!")
                                return True
                    except Exception as ex:
                        print(f"!!! WARNING: Failed to extract params from {phase_name}: {str(ex)} !!!")
                    return False
                
                for study_obj, phase_name in zip(studies_list, phase_names):
                    _maybe_add(study_obj, phase_name)
                
                # Always include current best params
                try:
                    guided_prior4 = best_trial.user_attrs.get("params", None) if best_trial is not None else None
                    if isinstance(guided_prior4, dict) and guided_prior4 not in candidate_params:
                        candidate_params.append(guided_prior4)
                        print(f"!!! Added current overall best params: lr={guided_prior4.get('learning_rate', 'N/A'):.4f}, leaves={guided_prior4.get('num_leaves', 'N/A')}, depth={guided_prior4.get('max_depth', 'N/A')} !!!")
                except Exception as ex:
                    print(f"!!! WARNING: Failed to add current best params: {str(ex)} !!!")
                    guided_prior4 = default_lgb_params()
                    
                # Limit to top-K unique candidates
                K = min(8, max(2, len(candidate_params)))
                candidate_params = candidate_params[:K]
                print(f"!!! Phase 7 Step 1: Collected {len(candidate_params)} unique candidate param sets (K={K}) !!!")

                # Create a study and enqueue fixed-parameter evaluations (acts like a leaderboard re-check)
                print(f"!!! Creating Refinement Study with Multivariate TPE !!!")
                sampler_ult_ref = TPESampler(seed=47, n_startup_trials=10, multivariate=True)
                pruner_ult_ref = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
                study_ult_refine = optuna.create_study(direction="maximize", sampler=sampler_ult_ref, pruner=pruner_ult_ref, study_name=f"run2_fold_{fold_idx}_extend_ultimate_refine")

                # We'll evaluate each candidate explicitly by binding the objective per candidate,
                # and enqueue a placeholder for each for uniform scheduling.
                for idx, _ in enumerate(candidate_params):
                    try:
                        study_ult_refine.enqueue_trial({})
                    except Exception as ex:
                        print(f"!!! WARNING: Failed to enqueue trial {idx+1}: {str(ex)} !!!")

                # Evaluate candidates sequentially within the allotted time budget half
                # using the same study to keep logging unified
                # Allocate time adaptively: larger gap => give more time to candidate refinement
                refine_budget = float(ULTIMATE_EXTENSION_TIME_LIMIT) * (0.6 if mcc_gap7 > 0.10 else 0.5)
                print(f"!!! Phase 7 Step 1: Evaluating {len(candidate_params)} candidates with {refine_budget:.0f}s budget (adaptive: {int(refine_budget/60)}min) !!!")
                
                objective_ult_ref = None
                if candidate_params:
                    # use the first candidate to satisfy API; we'll override within loop
                    objective_ult_ref = create_fixed_param_objective(
                        X_tr_df, y_tr, X_va_df, y_va, thr_grid, fixed_params=candidate_params[0]
                    )
                    print(f"!!! Evaluating Candidate 1/{len(candidate_params)} !!!")
                # Run half of Phase 7 time for refinement
                if objective_ult_ref is not None:
                    study_ult_refine.optimize(objective_ult_ref, timeout=float(ULTIMATE_EXTENSION_TIME_LIMIT) * 0.25, n_jobs=1, show_progress_bar=False)

                # If time remains, evaluate remaining candidates one-by-one with shorter slices
                remaining_time = float(ULTIMATE_EXTENSION_TIME_LIMIT) * 0.5 - sum(t.duration.total_seconds() for t in study_ult_refine.trials if t.duration is not None)
                per_cand_time = max(5.0, remaining_time / max(1, len(candidate_params)-1)) if candidate_params else 0.0
                print(f"!!! Remaining time for other candidates: {remaining_time:.1f}s, per-candidate: {per_cand_time:.1f}s !!!")
                
                for idx, cand in enumerate(candidate_params[1:], start=2):
                    if per_cand_time <= 0:
                        print(f"!!! WARNING: No time remaining for candidate {idx}/{len(candidate_params)}, skipping !!!")
                        break
                    print(f"!!! Evaluating Candidate {idx}/{len(candidate_params)} (budget: {per_cand_time:.1f}s) !!!")
                    obj = create_fixed_param_objective(X_tr_df, y_tr, X_va_df, y_va, thr_grid, fixed_params=cand)
                    try:
                        study_ult_refine.optimize(obj, timeout=per_cand_time, n_jobs=1, show_progress_bar=False)
                    except Exception as ex:
                        print(f"!!! WARNING: Candidate {idx} evaluation failed: {str(ex)} !!!")
                        
                refine_elapsed = time.time() - phase7_start_time
                refine_trials = len(study_ult_refine.trials) if study_ult_refine else 0
                all_trials += list(study_ult_refine.trials)
                print(f"!!! Phase 7 Step 1 Complete: {refine_trials} refinement evaluations in {refine_elapsed:.1f}s !!!")

                # Determine best refined params
                refined_best_params = guided_prior4
                try:
                    if study_ult_refine.best_trial is not None:
                        refined_best_params = study_ult_refine.best_trial.user_attrs.get("params", guided_prior4)
                        refine_stats = study_ult_refine.best_trial.user_attrs.get("stats", {})
                        refine_score = study_ult_refine.best_value
                        print(f"!!! Phase 7 Step 1 Best: Score={refine_score:.4f}, F1={refine_stats.get('f1', float('nan')):.4f}, AUC={refine_stats.get('auc') if refine_stats.get('auc') is not None else 'undefined'} !!!")
                        # Mild post-refinement nudge: if gap large, slightly increase regularization to stabilize
                        if mcc_gap7 > 0.10 and isinstance(refined_best_params, dict):
                            refined_best_params = dict(refined_best_params)
                            refined_best_params["lambda_l2"] = float(np.clip(float(refined_best_params.get("lambda_l2", 0.1)) * 1.25, 0.0, 1.0))
                            print(f"!!! Phase 7 Adaptive Nudge: increased lambda_l2 for stability due to large MCC gap !!!")
                    else:
                        print(f"!!! WARNING: No best trial from refinement step, using prior params !!!")
                except Exception as ex:
                    print(f"!!! WARNING: Failed to extract refined best params: {str(ex)} !!!")

                # Step 2: CMA-ES around refined best (continuous params benefit greatly)
                print(f"!!! Phase 7 Step 2: Checking CMA-ES dependency (cmaes) availability !!!")
                use_cma = _has_cmaes()
                if not use_cma:
                    print(f"!!! WARNING: Optional dependency 'cmaes' is NOT installed — falling back to guided multivariate TPE for Step 2 !!!")
                    print(f"!!! To enable CMA-ES: pip install cmaes !!!")
                
                cma_budget = float(ULTIMATE_EXTENSION_TIME_LIMIT) - refine_elapsed  # Use remaining time
                if use_cma:
                    # Adaptive sigma based on MCC gap: larger gap = wider initial search radius
                    sigma0 = 0.15 if mcc_gap7 > 0.10 else 0.1
                    print(f"!!! Phase 7 Step 2: CMA-ES Search Around Refined Best !!!")
                    print(f"!!! CMA-ES Config: sigma0={sigma0} (adaptive: wider for large gap), seed=48 !!!")
                    print(f"!!! CMA-ES Prior Params: lr={refined_best_params.get('learning_rate', 'N/A'):.4f}, leaves={refined_best_params.get('num_leaves', 'N/A')}, depth={refined_best_params.get('max_depth', 'N/A')} !!!")
                    print(f"!!! Phase 7 Step 2: Starting CMA-ES with {cma_budget:.0f}s budget ({int(cma_budget/60)}min) !!!")
                    cma_start_time = time.time()
                    try:
                        sampler_cma = CmaEsSampler(seed=48, sigma0=sigma0)
                        pruner_cma = MedianPruner(n_startup_trials=6, n_warmup_steps=4)
                        study_ult_cma = optuna.create_study(direction="maximize", sampler=sampler_cma, pruner=pruner_cma, study_name=f"run2_fold_{fold_idx}_extend_ultimate_cma")
                        objective_ult_cma = create_optuna_objective(X_tr_df, y_tr, X_va_df, y_va, thr_grid, prior_best=refined_best_params, mcc_gap=mcc_gap7, include_tough_params=('include_tough_params_guided' in locals() and include_tough_params_guided))
                        study_ult_cma.optimize(
                            objective_ult_cma,
                            timeout=cma_budget,
                            n_jobs=1,
                            show_progress_bar=False,
                            callbacks=[
                                lambda study, trial: print(
                                    f"  ✓ [ULT-CMA] Trial {trial.number+1:>3} | Score: {trial.value:.4f} → {'NEW BEST' if trial.value == study.best_value else 'continue'} | "
                                    f"F1={trial.user_attrs.get('stats', {}).get('f1', 0):.4f}, "
                                    f"AUC={trial.user_attrs.get('stats', {}).get('auc') if trial.user_attrs.get('stats', {}).get('auc') is not None else 'undefined'}, "
                                    f"ACC={trial.user_attrs.get('stats', {}).get('acc', 0):.4f}, "
                                    f"θ={trial.user_attrs.get('stats', {}).get('thr', 0.5):.2f} | "
                                    f"lr={trial.params.get('learning_rate', 0):.4f}, leaves={trial.params.get('num_leaves', 0)}, depth={trial.params.get('max_depth', 0)}"
                                ) if trial.value > 0 and trial.number % 5 == 0 else None
                            ]
                        )
                        cma_elapsed = time.time() - cma_start_time
                        cma_trials = len(study_ult_cma.trials) if study_ult_cma else 0
                        cma_pruned = len([t for t in study_ult_cma.trials if t.state == optuna.trial.TrialState.PRUNED]) if study_ult_cma else 0
                        all_trials += list(study_ult_cma.trials)
                        print(f"!!! Phase 7 Step 2 Complete: {cma_trials} CMA-ES trials ({cma_pruned} pruned) in {cma_elapsed:.1f}s !!!")
                        if study_ult_cma and study_ult_cma.best_trial is not None:
                            cma_stats = study_ult_cma.best_trial.user_attrs.get("stats", {})
                            cma_score = study_ult_cma.best_value
                            print(f"!!! Phase 7 Step 2 Best: Score={cma_score:.4f}, F1={cma_stats.get('f1', float('nan')):.4f}, AUC={cma_stats.get('auc') if cma_stats.get('auc') is not None else 'undefined'}, MCC={cma_stats.get('mcc', float('nan')):.4f} !!!")
                        else:
                            print(f"!!! WARNING: CMA-ES completed but no best trial found !!!")
                    except Exception as cma_ex:
                        cma_elapsed = time.time() - cma_start_time
                        print(f"!!! ERROR in Phase 7 Step 2 (CMA-ES) after {cma_elapsed:.1f}s: {type(cma_ex).__name__}: {str(cma_ex)} !!!")
                        import traceback
                        print(f"!!! CMA-ES Traceback: {traceback.format_exc()[:300]} !!!")
                else:
                    # Fallback: Guided multivariate TPE in the neighborhood of refined_best_params
                    print(f"!!! Phase 7 Step 2 Fallback: Guided Multivariate TPE around refined best for {cma_budget:.0f}s ({int(cma_budget/60)}min) !!!")
                    sampler_fallback = TPESampler(seed=49, n_startup_trials=15, multivariate=True)
                    pruner_fallback = MedianPruner(n_startup_trials=6, n_warmup_steps=4)
                    study_ult_cma = optuna.create_study(direction="maximize", sampler=sampler_fallback, pruner=pruner_fallback, study_name=f"run2_fold_{fold_idx}_extend_ultimate_tpe_fallback")
                    objective_fallback = create_optuna_objective(X_tr_df, y_tr, X_va_df, y_va, thr_grid, prior_best=refined_best_params, mcc_gap=mcc_gap7, include_tough_params=('include_tough_params_guided' in locals() and include_tough_params_guided))
                    fb_start = time.time()
                    study_ult_cma.optimize(objective_fallback, timeout=cma_budget, n_jobs=1, show_progress_bar=False)
                    fb_elapsed = time.time() - fb_start
                    all_trials += list(study_ult_cma.trials)
                    print(f"!!! Phase 7 Step 2 Fallback Complete: {len(study_ult_cma.trials)} trials in {fb_elapsed:.1f}s !!!")
                
                # Final Phase 7 Summary
                phase7_total_elapsed = time.time() - phase7_start_time
                best_after_phase7, best_score_phase7 = _current_best(all_trials)
                if best_after_phase7 is not None:
                    phase7_stats = best_after_phase7.user_attrs.get("stats", {})
                    phase7_improvement = best_score_phase7 - best_value
                    print(f"!!! PHASE 7 COMPLETE: Total time {phase7_total_elapsed:.1f}s !!!")
                    print(f"!!! Phase 7 Final Best: Score={best_score_phase7:.4f} (Δ={phase7_improvement:+.4f}), F1={phase7_stats.get('f1', float('nan')):.4f}, AUC={phase7_stats.get('auc') if phase7_stats.get('auc') is not None else 'undefined'}, MCC={phase7_stats.get('mcc', float('nan')):.4f} !!!")
                    if phase7_improvement > 0.001:
                        print(f"!!! PHASE 7 SUCCESS: Found improvement of {phase7_improvement:.4f} !!!")
                    else:
                        print(f"!!! PHASE 7 NOTE: No significant improvement (Δ={phase7_improvement:+.6f}) !!!")
                else:
                    print(f"!!! WARNING: Phase 7 completed but no best trial found !!!")
                    
            except Exception as e:
                phase7_elapsed = time.time() - phase7_start_time
                print(f"!!! FATAL ERROR in Phase 7 after {phase7_elapsed:.1f}s: {type(e).__name__}: {str(e)} !!!")
                import traceback
                print(f"!!! Phase 7 Full Traceback: {traceback.format_exc()[:500]} !!!")

        # PHASE 8: FLAML AutoML Extension (15 min) — only gbdt
        study_flaml = None
        # Final reassess after Phase 7
        best_trial, best_value = _current_best(all_trials)
        interim_stats = best_trial.user_attrs.get("stats", {}) if best_trial is not None else {}
        interim_thr = float(interim_stats.get("thr", 0.5)) if interim_stats else 0.5
        interim_auc = interim_stats.get("auc", None) if interim_stats else None
        interim_mcc_phase8 = interim_stats.get("mcc", float("nan")) if interim_stats else float("nan")
        if _bad_enough(interim_auc, interim_thr, interim_mcc_phase8) and FLAML_EXTENSION_TIME_LIMIT > 0:
            mcc_gap8 = 0.60 - (float(interim_mcc_phase8) if not np.isnan(interim_mcc_phase8) else 0.0)
            mcc_gap8 = max(0.0, mcc_gap8)
            print(f"  🔍 Phase 8 — FLAML AutoML (RUN2) | Budget: {FLAML_EXTENSION_TIME_LIMIT:.1f}s")
            print(f"!!! PHASE 8 START: FLAML AutoML Search (LGBM only, GBDT boosting) !!!")
            print(f"!!! Current Best Before Phase 8: Score={best_value:.4f}, F1={interim_stats.get('f1', float('nan')):.4f}, AUC={interim_auc if interim_auc is not None else 'undefined'}, MCC={interim_mcc_phase8:.4f}, θ={interim_thr:.2f} !!!")
            print(f"!!! Adaptive Tuning Context (Phase 8): mcc_gap_to_0.60={mcc_gap8:.4f} (larger gap => extended search space for FLAML) !!!")
            
            if _has_flaml():
                phase8_start_time = time.time()
                try:
                    from flaml import AutoML
                    # Try to import tune for a compatible custom search space; fallback to default space
                    try:
                        from flaml import tune as fl_tune  # type: ignore
                        has_tune = True
                    except Exception:
                        has_tune = False
                    print(f"!!! FLAML Dependency Available — starting AutoML search !!!")
                    print(f"!!! FLAML Config: estimator=lgbm, metric=roc_auc, time_budget={FLAML_EXTENSION_TIME_LIMIT}s ({int(FLAML_EXTENSION_TIME_LIMIT/60)}min) !!!")
                    
                    automl = AutoML()
                    automl_settings = {
                        "time_budget": float(FLAML_EXTENSION_TIME_LIMIT),
                        "metric": "roc_auc",
                        "estimator_list": ["lgbm"],  # Only LightGBM
                        "task": "classification",
                        "log_file_name": str(Path(RUN_DIR, f"flaml_phase8_fold_{fold_idx}.log")),
                        "verbosity": 1,
                        "n_jobs": 1,
                        "seed": 50,
                    }
                    
                    # Extended search space for large MCC gaps — use tune only if available; otherwise skip custom_hp
                    if mcc_gap8 > 0.10 and has_tune:
                        print(f"!!! FLAML Extended Search Space enabled (mcc_gap={mcc_gap8:.4f} > 0.10) with tune primitives !!!")
                        automl_settings["custom_hp"] = {
                            "lgbm": {
                                "n_estimators": {"domain": fl_tune.randint(100, 5000), "init_value": 4000},
                                "learning_rate": {"domain": fl_tune.loguniform(0.005, 0.15), "init_value": 0.05},
                                "num_leaves": {"domain": fl_tune.randint(15, 128), "init_value": 63},
                                "max_depth": {"domain": fl_tune.randint(3, 13), "init_value": 6},
                                "min_child_samples": {"domain": fl_tune.randint(10, 161), "init_value": 60},
                                "feature_fraction": {"domain": fl_tune.uniform(0.4, 1.0), "init_value": 0.85},
                                "bagging_fraction": {"domain": fl_tune.uniform(0.4, 1.0), "init_value": 0.85},
                                "lambda_l1": {"domain": fl_tune.uniform(0.0, 1.0), "init_value": 0.1},
                                "lambda_l2": {"domain": fl_tune.uniform(0.0, 1.0), "init_value": 0.1},
                                "min_split_gain": {"domain": fl_tune.uniform(0.0, 1.0), "init_value": 0.0},
                                "min_child_weight": {"domain": fl_tune.loguniform(1e-4, 50.0), "init_value": 0.001},
                                "bagging_freq": {"domain": fl_tune.randint(0, 8), "init_value": 1},
                                "max_bin": {"domain": fl_tune.randint(128, 512), "init_value": 255},
                            }
                        }
                    elif mcc_gap8 > 0.10 and not has_tune:
                        print(f"!!! FLAML tune primitives not available — proceeding without custom_hp (safe default) !!!")
                    else:
                        print(f"!!! FLAML Standard Search Space (mcc_gap={mcc_gap8:.4f} ≤ 0.10) !!!")
                    
                    print(f"!!! Starting FLAML AutoML.fit() !!!")
                    flaml_start = time.time()
                    automl.fit(X_train=X_tr_df, y_train=y_tr, **automl_settings)
                    flaml_elapsed = time.time() - flaml_start
                    
                    best_conf = getattr(automl, "best_config", {}) or {}
                    print(f"!!! FLAML COMPLETE in {flaml_elapsed:.1f}s ({int(flaml_elapsed/60)}min) !!!")
                    print(f"!!! FLAML best_config keys: {list(best_conf.keys())} !!!")
                    
                    # Convert FLAML best_config partial params into LGBM params, clamp and evaluate
                    fl_params_raw = {k: v for k, v in best_conf.items() if k in {
                        "num_leaves","learning_rate","feature_fraction","bagging_fraction",
                        "lambda_l1","lambda_l2","min_child_samples","min_split_gain",
                        "min_child_weight","bagging_freq","max_bin","feature_fraction_bynode"
                    }}
                    fl_params = clamp_lgb_param_ranges({**fl_params_raw, "boosting_type": "gbdt"})
                    print(f"!!! FLAML Extracted Params: lr={fl_params.get('learning_rate', 'N/A'):.4f}, leaves={fl_params.get('num_leaves', 'N/A')}, depth={fl_params.get('max_depth', 'N/A')}, min_child={fl_params.get('min_child_samples', 'N/A')} !!!")
                    
                    # Evaluate FLAML best config on validation set
                    try:
                        print(f"!!! Evaluating FLAML best config on validation set !!!")
                        clf_f = lgb.LGBMClassifier(**{**default_lgb_params(), **fl_params})
                        clf_f.fit(X_tr_df, y_tr, eval_set=[(X_va_df, y_va)], eval_metric="auc",
                                 callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)])
                        p_f = clf_f.predict_proba(X_va_df)[:, 1]
                        st_f = evaluate_thresholds(y_va, p_f, thr_grid)
                        sc_f = score_fold(st_f)
                        
                        # Create a pseudo-trial to add to all_trials for uniform comparison
                        # We'll create a minimal trial-like dict
                        flaml_trial_info = {
                            "value": sc_f,
                            "user_attrs": {"stats": st_f, "params": fl_params},
                            "params": fl_params,
                        }
                        auc_f_str = 'undefined' if st_f.get('auc') is None else f"{st_f.get('auc'):.4f}"
                        print(f"!!! FLAML Score={sc_f:.4f} | F1={st_f.get('f1', float('nan')):.4f} | AUC={auc_f_str} | MCC={st_f.get('mcc', float('nan')):.4f} !!!")
                        
                        # Note: Since we can't easily create a real Optuna Trial object, we track FLAML result separately
                        # and manually compare it to best_value
                        flaml_improvement = sc_f - best_value
                        if flaml_improvement > 0.001:
                            print(f"!!! FLAML SUCCESS: Found improvement of {flaml_improvement:.4f} over current best !!!")
                            # Update best manually for subsequent phases if needed
                        else:
                            print(f"!!! FLAML NOTE: No significant improvement (Δ={flaml_improvement:+.6f}) !!!")
                        
                        # Save FLAML params to file for reference
                        try:
                            flaml_params_path = Path(RUN_DIR, f"flaml_best_params_fold_{fold_idx}.json")
                            flaml_params_path.write_text(json.dumps(fl_params, indent=2), encoding='utf-8')
                            print(f"!!! FLAML params saved to {flaml_params_path.name} !!!")
                        except Exception:
                            pass
                            
                    except Exception as fe:
                        print(f"!!! WARNING: Failed to evaluate FLAML best config: {type(fe).__name__}: {str(fe)} !!!")
                        import traceback
                        print(f"!!! FLAML Evaluation Traceback: {traceback.format_exc()[:300]} !!!")
                        
                except Exception as fe2:
                    phase8_elapsed = time.time() - phase8_start_time
                    print(f"!!! FATAL ERROR: FLAML failed to run after {phase8_elapsed:.1f}s: {type(fe2).__name__}: {str(fe2)} !!!")
                    import traceback
                    print(f"!!! FLAML Full Traceback: {traceback.format_exc()[:500]} !!!")
            else:
                print(f"!!! WARNING: 'flaml' is NOT installed — skipping Phase 8 !!!")
                print(f"!!! To enable FLAML AutoML: pip install flaml !!!")

        # Final best across all phases (E, optional G, and any extensions X/U/Z/H/ULT/FLAML)
        trials = len(all_trials)
        best_trial, best_value = _current_best(all_trials)
        if best_trial is not None:
            best_stats = best_trial.user_attrs.get("stats", {})
            best_params_dict = best_trial.user_attrs.get("params", default_lgb_params())
            best_score = float(best_value)
            best_auc = best_stats.get("auc")
            
            # Count improvements across all phases
            improved = 0
            try:
                if study_e is not None and study_e.best_trial is not None and study_e.best_value is not None:
                    improved += len([t for t in study_e.trials if t.value == study_e.best_value])
            except Exception:
                pass
            try:
                if study_g is not None and study_g.best_trial is not None and study_g.best_value is not None:
                    improved += len([t for t in study_g.trials if t.value == study_g.best_value])
            except Exception:
                pass
            try:
                if study_x is not None and study_x.best_trial is not None and study_x.best_value is not None:
                    improved += len([t for t in study_x.trials if t.value == study_x.best_value])
            except Exception:
                pass
            try:
                if study_u is not None and study_u.best_trial is not None and study_u.best_value is not None:
                    improved += len([t for t in study_u.trials if t.value == study_u.best_value])
            except Exception:
                pass
            try:
                if study_z is not None and study_z.best_trial is not None and study_z.best_value is not None:
                    improved += len([t for t in study_z.trials if t.value == study_z.best_value])
            except Exception:
                pass
            try:
                if study_h is not None and study_h.best_trial is not None and study_h.best_value is not None:
                    improved += len([t for t in study_h.trials if t.value == study_h.best_value])
            except Exception:
                pass
            try:
                if study_ult_refine is not None and study_ult_refine.best_trial is not None and study_ult_refine.best_value is not None:
                    improved += len([t for t in study_ult_refine.trials if t.value == study_ult_refine.best_value])
            except Exception:
                pass
            try:
                if study_ult_cma is not None and study_ult_cma.best_trial is not None and study_ult_cma.best_value is not None:
                    improved += len([t for t in study_ult_cma.trials if t.value == study_ult_cma.best_value])
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
                "best_mcc": float(best_stats.get("mcc", 0)),
                "score": float(best_score),
            }
        else:
            # Fallback when study fails entirely: report NaNs without best labeling
            best_score = float("nan")
            best_info = {
                "params": default_lgb_params(),
                "best_thr": 0.5,
                "best_auc": None,
                "best_acc": float("nan"),
                "best_prec": float("nan"),
                "best_rec": float("nan"),
                "best_f1": float("nan"),
                "best_bacc": float("nan"),
                "best_mcc": float("nan"),
                "score": float("nan"),
            }
            improved = 0

        elapsed = time.time() - fold_t0
    else:
        # Skip search using hardcoded best params; fit model to get fresh stats
        try:
            params_use = clamp_lgb_param_ranges(hardcoded_params or default_lgb_params())
            clf = lgb.LGBMClassifier(**params_use)
            bt = params_use.get("boosting_type", "gbdt")
            fit_callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)] if bt != "dart" else []
            clf.fit(
                X_tr_df,
                y_tr,
                eval_set=[(X_va_df, y_va)],
                eval_metric="auc",
                callbacks=fit_callbacks
            )
            proba = clf.predict_proba(X_va_df)[:, 1] if len(X_va_df) else np.array([])
            stats = evaluate_thresholds(y_va, proba, thr_grid) if len(proba) else {"thr":0.5,"acc":float("nan"),"auc":None,"prec":float("nan"),"rec":float("nan"),"f1":float("nan"),"bacc":float("nan"),"mcc":float("nan")}
            best_score = float(score_fold({"f1":stats.get("f1",0),"auc":stats.get("auc"),"acc":stats.get("acc",0),"bacc":stats.get("bacc",0)}))
            best_info = {
                "params": params_use,
                "best_thr": float(stats.get("thr", 0.5)),
                "best_auc": stats.get("auc"),
                "best_acc": float(stats.get("acc", 0)),
                "best_prec": float(stats.get("prec", 0)),
                "best_rec": float(stats.get("rec", 0)),
                "best_f1": float(stats.get("f1", 0)),
                "best_bacc": float(stats.get("bacc", 0)),
                "best_mcc": float(stats.get("mcc", float("nan"))),
                "score": best_score,
            }
            trials = 1
            improved = 1
            print(f"  ⚡ Skipping search, using hardcoded best for RUN2 fold {fold_idx} ({fold_ts_key})")
        except Exception as e:
            best_score = float("nan")
            best_info = {
                "params": default_lgb_params(),
                "best_thr": 0.5,
                "best_auc": None,
                "best_acc": float("nan"),
                "best_prec": float("nan"),
                "best_rec": float("nan"),
                "best_f1": float("nan"),
                "best_bacc": float("nan"),
                "best_mcc": float("nan"),
                "score": float("nan"),
            }
            trials = 0
            improved = 0
            print(f"  ⚠️  Error using hardcoded params for fold {fold_idx}: {str(e)}")
        elapsed = time.time() - fold_t0

    # Tag source of params for meta-learning priority
    param_source = "hardcoded" if skip_search else ("search_known" if has_known_for_fold else "search")
    meta_entry = dict(best_info)
    meta_entry.update({
        "fold": int(fold_idx),
        "train_size": int(len(y_tr)),
        "val_size": int(len(y_va)),
        "train_signature": sig,
        "param_source": param_source,
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
        "fold_ts_key": fold_ts_key,
        "train_size": int(len(y_tr)),
        "val_size": int(len(y_va)),
        "summary": best_info,
        "trials": trials,
        "improvements": improved,
        "elapsed": elapsed,
        "train_pos": tr_pos,
        "train_neg": tr_neg,
        "val_pos": va_pos,
        "val_neg": va_neg,
    })

    # Priority rule for meta-learner guidance:
    # If this fold had a known-best available and the user chose to SEARCH anyway, treat these params
    # as higher-priority guidance than hardcoded ones in subsequent folds.
    if meta_entry.get("param_source") in ("search_known", "search"):
        prior_best_params = best_info.get("params", None)
    else:
        # Do not override with hardcoded unless no prior exists yet
        if prior_best_params is None:
            prior_best_params = best_info.get("params", None)
    
    if len(fold_meta) >= 2:
        # Sort folds such that searched params from known folds take precedence in training
        # This biases the meta-learner towards your interactive searches when known-bests exist
        fold_meta_sorted = sorted(
            fold_meta,
            key=lambda m: 0 if m.get("param_source") == "search_known" else (1 if m.get("param_source") == "search" else 2)
        )
        meta_learner.fit(fold_meta_sorted)
        print(f"\n  🧠 Meta-learner updated with {len(fold_meta)} folds (RUN2)")

    auc_disp = best_info.get('best_auc')
    auc_str_final = f"{auc_disp:.4f}" if (auc_disp is not None and not np.isnan(auc_disp)) else "undefined"
    acc_disp = best_info.get('best_acc', 0)
    print(f"\n  📊 RUN2 Fold {fold_idx} Summary:")
    print(f"     Time: {elapsed:.1f}s | Trials: {trials} | Improvements: {improved}")
    print(f"     Best Score: {best_score:.4f} | F1: {best_info.get('best_f1', 0):.4f} | AUC: {auc_str_final} | ACC: {acc_disp:.4f} | MCC: {best_info.get('best_mcc', float('nan')):.4f}")
    print(f"     Threshold: {best_info.get('best_thr', 0.5):.2f}")

    # Advance to next fold
    fold_idx += 1

print(f"\n{'='*70}")
print("  PRE-TEST OPTIMIZATION COMPLETE (CUMULATIVE) — RUN2")
print(f"{'='*70}\n")

# Heartbeat spinner to indicate progress during subsequent meta steps
_heartbeat_stop = threading.Event()
def _heartbeat(label: str):
    spinner = itertools.cycle(['|','/','-','\\'])
    while not _heartbeat_stop.is_set():
        sys.stdout.write(f"\r  {label}  " + next(spinner))
        sys.stdout.flush()
        time.sleep(0.2)



# %% META-LEARNING
# Optional prompt to omit specific folds from meta-learner training
omit_env = os.getenv("MCP_OMIT_FOLDS", "").strip()
omit_folds: List[int] = _parse_omit_folds_input(omit_env)
if not omit_folds:
    try:
        user_in = input("Enter fold numbers to omit from meta-learning (e.g., 4 or 2,5) [0=keep all, x=search best omissions]: ").strip()
    except Exception:
        user_in = "0"
    special_exhaustive_omit = (user_in.lower() == 'x')
    if not special_exhaustive_omit:
        omit_folds = _parse_omit_folds_input(user_in)
    else:
        omit_folds = []

# Simple terminal progress bar
def _progress_bar(current: int, total: int, label: str = ""):
    try:
        width = 28
        total = max(1, int(total))
        current = int(min(max(0, current), total))
        ratio = current / float(total)
        filled = int(width * ratio)
        bar = '#' * filled + '-' * (width - filled)
        sys.stdout.write(f"\r  {label} [{bar}] {current}/{total}")
        sys.stdout.flush()
        if current >= total:
            sys.stdout.write("\n")
    except Exception:
        pass

# Exhaustive omission search mode: only adv_chain_refined, then exit
if 'special_exhaustive_omit' in locals() and special_exhaustive_omit:
    try:
        # Gather available fold ids
        fold_ids_all = sorted({int(m.get("fold", -1)) for m in fold_meta if isinstance(m.get("fold", None), (int, np.integer)) and int(m.get("fold", -1)) > 0})
    except Exception:
        fold_ids_all = list(range(1, max(1, len(fold_meta)+1))) if isinstance(fold_meta, list) else []

    # Base data & signatures
    X_combined_df = pd.concat([X_train_df, X_pre_df], ignore_index=True) if len(X_pre_df) else X_train_df
    y_combined = np.concatenate([y_train, y_pre]) if len(y_pre) else y_train
    X_full_df = pd.concat([X_train_df, X_pre_df], ignore_index=True) if len(X_pre_df) else X_train_df
    y_full = np.concatenate([y_train, y_pre]) if len(y_pre) else y_train
    thr_final_eval = float(np.median(best_thr_list)) if len(best_thr_list) else 0.5

    def _score_model_for_current_params(params_use: Dict[str, Any]) -> float:
        try:
            model = lgb.LGBMClassifier(**params_use)
            model.fit(X_full_df, y_full)
        except Exception:
            return float('-inf')
        window_scores: List[float] = []
        for X_test_df, (_, y_test) in zip(X_tests_df, tests_Xy):
            try:
                proba = model.predict_proba(X_test_df)[:, 1] if len(X_test_df) else np.array([])
                if len(proba):
                    pred = (proba >= thr_final_eval).astype(int)
                    acc_v = float(accuracy_score(y_test, pred))
                    try:
                        auc_v = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) == 2 else None
                    except Exception:
                        auc_v = None
                    try:
                        mcc_v = float(matthews_corrcoef(y_test, pred))
                    except Exception:
                        mcc_v = float('nan')
                    if auc_v is None or (isinstance(auc_v, float) and np.isnan(auc_v)):
                        score_v = (mcc_v + acc_v) / 2.0
                    else:
                        score_v = (mcc_v + acc_v + auc_v) / 3.0
                else:
                    score_v = float('-inf')
            except Exception:
                score_v = float('-inf')
            window_scores.append(score_v)
        try:
            return float(np.nanmean(window_scores)) if window_scores else float('-inf')
        except Exception:
            return float('-inf')

    # 1-omit search
    print("\n  🔬 Exhaustive omit search [omit 1 fold] — using adv_chain_refined only")
    single_results: List[Dict[str, Any]] = []
    total_single = len(fold_ids_all)
    for idx, omit_id in enumerate(fold_ids_all, start=1):
        _progress_bar(idx, total_single, label="1-omit testing")
        fm = [m for m in fold_meta if int(m.get("fold", -1)) != int(omit_id)]
        fm_sorted = sorted(
            fm,
            key=lambda m: 0 if m.get("param_source") == "search_known" else (1 if m.get("param_source") == "search" else 2)
        )
        adv = AdvancedMetaLearner()
        try:
            adv.fit(fm_sorted)
        except Exception:
            pass
        sig_combined = compute_signature(X_combined_df, y_combined)
        base_params = adv.predict_chain(sig_combined)
        try:
            refined_params = refine_params_local(X_full_df, y_full, X_pre_df, y_pre, thr_grid, base_params, timeout_s=30.0)
        except Exception:
            refined_params = base_params
        score_avg = _score_model_for_current_params(refined_params)
        single_results.append({"omit": [int(omit_id)], "score": float(score_avg)})

    single_results_sorted = sorted(single_results, key=lambda r: r.get("score", float('-inf')), reverse=True)
    if single_results_sorted:
        best_single = single_results_sorted[0]
        print(f"  ✓ Best single omit → fold {best_single['omit'][0]} | score={best_single['score']:.4f}")
    else:
        print("  ⚠️  No results in single-omit search")

    # 2-omit combos
    print("\n  🧪 Trying 2 omit combos (all pairs)...")
    pair_results: List[Dict[str, Any]] = []
    pairs = list(itertools.combinations(fold_ids_all, 2))
    total_pairs = len(pairs)
    for idx, (a, b) in enumerate(pairs, start=1):
        _progress_bar(idx, total_pairs, label="2-omit testing")
        fm = [m for m in fold_meta if int(m.get("fold", -1)) not in (int(a), int(b))]
        fm_sorted = sorted(
            fm,
            key=lambda m: 0 if m.get("param_source") == "search_known" else (1 if m.get("param_source") == "search" else 2)
        )
        adv = AdvancedMetaLearner()
        try:
            adv.fit(fm_sorted)
        except Exception:
            pass
        sig_combined = compute_signature(X_combined_df, y_combined)
        base_params = adv.predict_chain(sig_combined)
        try:
            refined_params = refine_params_local(X_full_df, y_full, X_pre_df, y_pre, thr_grid, base_params, timeout_s=30.0)
        except Exception:
            refined_params = base_params
        score_avg = _score_model_for_current_params(refined_params)
        pair_results.append({"omit": [int(a), int(b)], "score": float(score_avg)})

    pair_results_sorted = sorted(pair_results, key=lambda r: r.get("score", float('-inf')), reverse=True)[:3]
    if pair_results_sorted:
        print("  ✓ Top 3 omit pairs (by avg of MCC, ACC, AUC):")
        for rank, res in enumerate(pair_results_sorted, start=1):
            a, b = res['omit']
            print(f"    {rank}. folds {a},{b} | score={res['score']:.4f}")
    else:
        print("  ⚠️  No results in 2-omit search")

    # Terminate after exhaustive testing as requested
    sys.stdout.flush()
    sys.exit(0)

# Build filtered fold meta list
if isinstance(fold_meta, list) and len(fold_meta) > 0 and len(omit_folds) > 0:
    fold_meta_filtered = [m for m in fold_meta if int(m.get("fold", -1)) not in set(omit_folds)]
    # Also sort guidance priority as earlier
    fold_meta_sorted = sorted(
        fold_meta_filtered,
        key=lambda m: 0 if m.get("param_source") == "search_known" else (1 if m.get("param_source") == "search" else 2)
    )
    print(f"  ⚙️  Omitting folds from meta-learning: {omit_folds} | using {len(fold_meta_sorted)}/{len(fold_meta)} folds")
else:
    fold_meta_sorted = sorted(
        fold_meta,
        key=lambda m: 0 if m.get("param_source") == "search_known" else (1 if m.get("param_source") == "search" else 2)
    ) if isinstance(fold_meta, list) else []

print("\n" + "="*70)
print("  META-LEARNING: INFERRING FINAL HYPERPARAMETERS (TWO METHODS) — RUN2")
print("="*70)
hb_thread = threading.Thread(target=_heartbeat, args=("Working on meta-learning...",), daemon=True)
hb_thread.start()

# Refit basic meta-learner on filtered folds if possible
if isinstance(fold_meta_sorted, list) and len(fold_meta_sorted) >= 2:
    meta_learner.fit(fold_meta_sorted)

# Try both Train+Pre-test and Pre-test-only signatures
if len(fold_meta_sorted) >= 2 and meta_learner.fitted:
    # Prefer guidance from folds where you searched despite known-bests
    guided_from = next((m for m in reversed(fold_meta_sorted) if m.get("param_source") == "search_known"), None)
    if guided_from is None:
        guided_from = next((m for m in reversed(fold_meta_sorted) if m.get("param_source") == "search"), None)
    
    # Method 1: Train + Pre-test combined
    X_combined_df = pd.concat([X_train_df, X_pre_df], ignore_index=True) if len(X_pre_df) else X_train_df
    y_combined = np.concatenate([y_train, y_pre]) if len(y_pre) else y_train
    sig_full_combined = {
        "n": float(len(y_combined)),
        "pos_rate": float(np.mean(y_combined)),
        "feat_mean_avg": float(np.mean(X_combined_df.values)) if X_combined_df.size else 0.0,
        "feat_std_avg": float(np.std(X_combined_df.values)) if X_combined_df.size else 0.0,
    }
    inferred_params_combined = meta_learner.predict(sig_full_combined)
    
    # Method 2: Pre-test only signature
    sig_full_pretest = {
        "n": float(len(y_pre)),
        "pos_rate": float(np.mean(y_pre)),
        "feat_mean_avg": float(np.mean(X_pre_df.values)) if X_pre_df.size else 0.0,
        "feat_std_avg": float(np.std(X_pre_df.values)) if X_pre_df.size else 0.0,
    }
    inferred_params_pretest = meta_learner.predict(sig_full_pretest)
    
    print(f"✓ Using ML-based meta-learner (trained on {len(fold_meta_sorted)} folds)")
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

# Prepare advanced meta-learner using fold meta (train after basic meta learner fits)
advanced_meta = AdvancedMetaLearner()
try:
    advanced_meta.fit(fold_meta_sorted)
    mlp_count = len(advanced_meta.mlp_ensemble) if advanced_meta.fitted_mlp else 0
    tree_count = len(advanced_meta.tree_ensemble) if advanced_meta.fitted_tree else 0
    chain_fitted = 'yes' if advanced_meta.fitted_chain else 'no'
    print(f"✓ AdvancedMetaLearner trained (chain={chain_fitted}, mlp_ensemble={mlp_count}, tree_ensemble={tree_count})")
except Exception as e:
    import traceback
    error_details = traceback.format_exc()
    print(f"⚠️ Failed to train AdvancedMetaLearner: {str(e)}")
    print(f"   Error details: {error_details[:200]}...")  # Show first 200 chars of traceback
    print("   Will fall back to defaults")
finally:
    _heartbeat_stop.set()
    hb_thread.join(timeout=1.0)
    sys.stdout.write("\r  Meta-learning steps complete.            \n")
    sys.stdout.flush()


# %% FINAL TRAIN AND TEST — MULTI-WINDOW (per window selection among up to 4 variants)
X_full_df = pd.concat([X_train_df, X_pre_df], ignore_index=True) if len(X_pre_df) else X_train_df
y_full = np.concatenate([y_train, y_pre]) if len(y_pre) else y_train

thr_final = float(np.median(best_thr_list)) if len(best_thr_list) else 0.5

def _selection_score(stats: Dict[str, Any]) -> float:
    auc_val_raw = stats.get("auc")
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

test_windows_results: List[Dict[str, Any]] = []
selected_models: List[Tuple[int, lgb.LGBMClassifier]] = []  # (window_index, model)

print("\n" + "="*70)
print("  FINAL MODEL TRAINING — PER TEST WINDOW — RUN2")
print("="*70)

for w_idx, (X_test_df, (_, y_test)) in enumerate(zip(X_tests_df, tests_Xy), start=1):

    # Base signatures (same for all windows)
    X_combined_df = pd.concat([X_train_df, X_pre_df], ignore_index=True) if len(X_pre_df) else X_train_df
    y_combined = np.concatenate([y_train, y_pre]) if len(y_pre) else y_train
    sig_combined = compute_signature(X_combined_df, y_combined)
    sig_pre_only = compute_signature(X_pre_df, y_pre)

    # Gated signatures: include all previous test windows (1..w_idx-1)
    if w_idx > 1:
        X_prev = pd.concat([X_tests_df[i] for i in range(0, w_idx-1)], ignore_index=True) if (w_idx-1) > 0 else pd.DataFrame(columns=FEATURE_NAMES)
        y_prev = np.concatenate([tests_Xy[i][1] for i in range(0, w_idx-1)]) if (w_idx-1) > 0 else np.array([])
        sig_combined_gated = compute_signature(pd.concat([X_combined_df, X_prev], ignore_index=True) if len(X_prev) else X_combined_df,
                                              np.concatenate([y_combined, y_prev]) if len(y_prev) else y_combined)
        sig_pre_only_gated = compute_signature(pd.concat([X_pre_df, X_prev], ignore_index=True) if len(X_prev) else X_pre_df,
                                              np.concatenate([y_pre, y_prev]) if len(y_prev) else y_pre)
    else:
        sig_combined_gated = sig_combined
        sig_pre_only_gated = sig_pre_only

    # Infer params
    params_combined = meta_learner.predict(sig_combined)
    params_pre_only = meta_learner.predict(sig_pre_only)
    params_combined_gated = meta_learner.predict(sig_combined_gated)
    params_pre_only_gated = meta_learner.predict(sig_pre_only_gated)

    # (Logging of signatures suppressed for concise output)

    # Advanced meta predictions (chain, MLP ensemble, tree ensemble), plus 60s local refinement around chain
    adv_chain_params = advanced_meta.predict_chain(sig_combined)
    adv_mlp_params = advanced_meta.predict_mlp(sig_combined)
    adv_tree_params = advanced_meta.predict_tree(sig_combined)
    # Small local search using pre-test as validation proxy (consistent with prior folds behavior)
    _heartbeat_stop.clear()
    hb2 = threading.Thread(target=_heartbeat, args=(f"Local refine (window {w_idx})...",), daemon=True)
    hb2.start()
    try:
        adv_chain_refined = refine_params_local(X_full_df, y_full, X_pre_df, y_pre, thr_grid, adv_chain_params, timeout_s=60.0)
    except Exception:
        adv_chain_refined = adv_chain_params
    finally:
        _heartbeat_stop.set()
        hb2.join(timeout=1.0)
        sys.stdout.write("\r  Local refine complete.                      \n")
        sys.stdout.flush()

    # Train variants on same X_full_df
    if w_idx == 1:
        variants = {
            "method1_combined": params_combined,
            "method2_pretest": params_pre_only,
            "adv_chain": adv_chain_params,
            "adv_chain_refined": adv_chain_refined,
            "adv_mlp": adv_mlp_params,
            "adv_tree": adv_tree_params,
        }
    else:
        variants = {
            "method1_combined": params_combined,
            "method2_pretest": params_pre_only,
            "method1_combined_gated": params_combined_gated,
            "method2_pretest_gated": params_pre_only_gated,
            "adv_chain": adv_chain_params,
            "adv_chain_refined": adv_chain_refined,
            "adv_mlp": adv_mlp_params,
            "adv_tree": adv_tree_params,
        }

    variant_models: Dict[str, Any] = {}
    variant_metrics: Dict[str, Dict[str, Any]] = {}
    for v_name, v_params in variants.items():
        try:
            model_v = lgb.LGBMClassifier(**v_params)
            model_v.fit(X_full_df, y_full)
            proba_v = model_v.predict_proba(X_test_df)[:, 1] if len(X_test_df) else np.array([])
            if len(proba_v):
                pred_v = (proba_v >= thr_final).astype(int)
                try:
                    auc_v = float(roc_auc_score(y_test, proba_v)) if len(np.unique(y_test)) == 2 else None
                except Exception:
                    auc_v = None
                stats_v = {
                    "thr": thr_final,
                    "acc": float(accuracy_score(y_test, pred_v)),
                    "auc": auc_v,
                    "prec": float(precision_score(y_test, pred_v, zero_division=0)),
                    "rec": float(recall_score(y_test, pred_v, zero_division=0)),
                    "f1": float(f1_score(y_test, pred_v, zero_division=0)),
                    "bacc": float(balanced_accuracy_score(y_test, pred_v)),
                    "mcc": float(matthews_corrcoef(y_test, pred_v)) if len(np.unique(y_test)) >= 1 else float("nan"),
                }
            else:
                stats_v = {"thr": thr_final, "acc": float("nan"), "auc": None, "prec": float("nan"), "rec": float("nan"), "f1": float("nan"), "bacc": float("nan"), "mcc": float("nan")}
            variant_models[v_name] = (model_v, proba_v)
            variant_metrics[v_name] = stats_v
        except Exception:
            continue

    # Print stats per variant and select best
    best_v = None
    best_score = -np.inf
    # Display order: basic + advanced methods
    display_order = [
        "method1_combined",
        "method2_pretest",
        "adv_chain",
        "adv_chain_refined",
        "adv_mlp",
        "adv_tree",
    ] if w_idx == 1 else [
        "method1_combined",
        "method2_pretest",
        "method1_combined_gated",
        "method2_pretest_gated",
        "adv_chain",
        "adv_chain_refined",
        "adv_mlp",
        "adv_tree",
    ]
    for name in display_order:
        stats_v = variant_metrics.get(name)
        if not stats_v:
            continue
        sc = _selection_score(stats_v)
        if sc > best_score:
            best_score = sc
            best_v = name
        auc_disp_v = stats_v.get('auc')
        auc_str_v = f"{auc_disp_v:.4f}" if (auc_disp_v is not None and not np.isnan(auc_disp_v)) else "undefined"
        print(f"Window {w_idx}/{NUM_TEST_WINDOWS} | {name} | Score={sc:.4f} | F1={stats_v.get('f1', float('nan')):.4f} | AUC={auc_str_v} | ACC={stats_v.get('acc', float('nan')):.4f} | MCC={stats_v.get('mcc', float('nan')):.4f} | θ={stats_v.get('thr', 0.5):.2f}")

    # Print exact params used by each method
    print(f"\n  📋 Exact Parameters Used (Window {w_idx}):")
    for name in display_order:
        if name in variants:
            params_used = variants[name]
            # Clean params for display
            params_clean = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (float, np.floating)) else v)
                           for k, v in params_used.items() if k in ["learning_rate", "num_leaves", "max_depth", "min_child_samples", "feature_fraction", "bagging_fraction", "lambda_l1", "lambda_l2"]}
            print(f"    {name}: {json.dumps(params_clean, separators=(',', ':'))}")

    # Run Optuna search on test window to find ideal params
    # Optional per-window ideal search (can be disabled with MCP_SKIP_IDEAL=1)
    skip_ideal = os.getenv("MCP_SKIP_IDEAL", "0") == "1"
    ideal_timeout_s = float(os.getenv("MCP_IDEAL_TIMEOUT_S", "300"))
    print(f"\n  🔍 Finding Ideal Parameters for Test Window {w_idx} ({int(ideal_timeout_s/60)}min search){' [skipped]' if skip_ideal else ''}:")
    _heartbeat_stop.clear()
    hb3 = threading.Thread(target=_heartbeat, args=(f"Test window {w_idx} Optuna search...",), daemon=True)
    hb3.start()
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if not skip_ideal and ideal_timeout_s > 0:
            sampler_ideal = TPESampler(seed=47, n_startup_trials=25)
            pruner_ideal = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            study_ideal = optuna.create_study(direction="maximize", sampler=sampler_ideal, pruner=pruner_ideal, study_name=f"test_window_{w_idx}_ideal")
            objective_ideal = create_optuna_objective(X_full_df, y_full, X_test_df, y_test, thr_grid, prior_best=None)
            study_ideal.optimize(objective_ideal, timeout=float(ideal_timeout_s), n_jobs=1, show_progress_bar=False)
        else:
            study_ideal = None
        
        if study_ideal is not None and study_ideal.best_trial is not None:
            ideal_stats = study_ideal.best_trial.user_attrs.get("stats", {})
            ideal_params = study_ideal.best_trial.user_attrs.get("params", default_lgb_params())
            ideal_score = float(study_ideal.best_value)
            ideal_auc = ideal_stats.get("auc")
            ideal_auc_str = f"{ideal_auc:.4f}" if (ideal_auc is not None and not np.isnan(ideal_auc)) else "undefined"
            
            print(f"    🎯 IDEAL: Score={ideal_score:.4f} | F1={ideal_stats.get('f1', 0):.4f} | AUC={ideal_auc_str} | ACC={ideal_stats.get('acc', 0):.4f} | MCC={ideal_stats.get('mcc', float('nan')):.4f} | θ={ideal_stats.get('thr', 0.5):.2f}")
            
            # Clean ideal params for display
            ideal_params_clean = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (float, np.floating)) else v)
                                 for k, v in ideal_params.items() if k in ["learning_rate", "num_leaves", "max_depth", "min_child_samples", "feature_fraction", "bagging_fraction", "lambda_l1", "lambda_l2"]}
            print(f"    🎯 IDEAL params: {json.dumps(ideal_params_clean, separators=(',', ':'))}")
        else:
            print(f"    ⚠️  Failed to find ideal params for window {w_idx}")
    except Exception as e:
        print(f"    ⚠️  Error finding ideal params for window {w_idx}: {str(e)}")
    finally:
        _heartbeat_stop.set()
        hb3.join(timeout=1.0)
        sys.stdout.write("\r  Test window Optuna search complete.        \n")
        sys.stdout.flush()

    selected_model, selected_proba = variant_models.get(best_v, (None, np.array([])))
    selected_stats = variant_metrics.get(best_v, {})

    # Save per-window artifacts
    try:
        model_path_k = os.path.join(RUN_DIR, f'lightgbm_final_window_{w_idx}_run2.joblib')
        joblib.dump(selected_model, model_path_k)
    except Exception:
        model_path_k = None

    # Threshold sweep per window (save JSON only; no console print)
    try:
        if len(selected_proba):
            thr_sweep = np.round(np.arange(0.30, 0.80 + 1e-9, 0.01), 2)
            sweep_results = []
            for thr in thr_sweep:
                pred_thr = (selected_proba >= thr).astype(int)
                acc_thr = accuracy_score(y_test, pred_thr)
                sweep_results.append({"thr": float(thr), "acc": float(round(acc_thr, 6))})
            Path(RUN_DIR, f'threshold_sweep_window_{w_idx}_run2.json').write_text(json.dumps(sweep_results, indent=2), encoding='utf-8')
    except Exception:
        pass

    # Feature importances for the selected model (optional)
    try:
        importances = getattr(selected_model, "feature_importances_", None)
        if importances is not None and len(importances) == len(FEATURE_NAMES):
            imp_map = {FEATURE_NAMES[i]: int(importances[i]) for i in range(len(FEATURE_NAMES))}
            Path(RUN_DIR, f'feature_importances_window_{w_idx}_run2.json').write_text(json.dumps(imp_map, indent=2), encoding='utf-8')
    except Exception:
        pass

    # (No extra selected-method print; lines above already show stats per method)

    test_windows_results.append({
        "window_index": int(w_idx),
        "bounds": [str(test_bounds_list[w_idx-1][0]), str(test_bounds_list[w_idx-1][1])],
        "inferred_params": {
            "method1_combined": params_combined,
            "method2_pretest": params_pre_only,
            "method1_combined_gated": params_combined_gated,
            "method2_pretest_gated": params_pre_only_gated,
        },
        "test_metrics": variant_metrics,
        "selected_variant": best_v,
        "selected_metrics": selected_stats,
        "model_path": model_path_k,
    })
    selected_models.append((w_idx, selected_model))

# For backward compatibility: treat the latest window as "final"
if selected_models:
    latest_idx, final_model = selected_models[-1]
    latest_result = next(r for r in test_windows_results if r["window_index"] == latest_idx)
    inferred_params = latest_result["inferred_params"].get(latest_result["selected_variant"]) or default_lgb_params()
    test_stats = latest_result["selected_metrics"]
    best_method = latest_result["selected_variant"]
    test_proba = np.array([])  # not used further
else:
    final_model = lgb.LGBMClassifier(**inferred_params_combined)
    final_model.fit(X_full_df, y_full)
    inferred_params = inferred_params_combined
    test_stats = {"thr": thr_final, "acc": float("nan"), "auc": None, "prec": float("nan"), "rec": float("nan"), "f1": float("nan"), "bacc": float("nan")}
    best_method = "method1_combined"


# %% SUMMARIES
def class_dist(y: np.ndarray) -> Dict[str, Any]:
    pos = int(np.sum(y == 1)); neg = int(np.sum(y == 0)); rate = pos/(pos+neg) if (pos+neg) else float('nan')
    return {"pos": pos, "neg": neg, "pos_rate": float(rate)}

print("\n================ THREE-WINDOW RUN SUMMARY (CUMULATIVE RUN2) ================")
print("[Data]")
print(f"  Train rows: {len(train_df):,}  |  Pre-test rows: {len(pre_df):,}  |  Test windows: {NUM_TEST_WINDOWS} (each {TEST_ROWS})")
train_bounds = (pd.to_datetime(train_df["timestamp"]).min() if len(train_df) else None,
                pd.to_datetime(train_df["timestamp"]).max() if len(train_df) else None)
if train_bounds[0] is not None and train_bounds[1] is not None:
    print(f"  Train span: {train_bounds[0]}  →  {train_bounds[1]}")
print(f"  Pre-test span: {pretest_bounds[0]}  →  {pretest_bounds[1]}")
for idx, (tb0, tb1) in enumerate(test_bounds_list, start=1):
    print(f"  Test[{idx}] span: {tb0}  →  {tb1}")
print(f"  Class dist — Train: {class_dist(y_train)}")
print(f"  Class dist — Pre-test: {class_dist(y_pre)}")

print("\n[Baseline on Train]")
try:
    base_proba = baseline_model.predict_proba(X_pre_df)[:, 1] if len(X_pre_df) else np.array([])
    if len(base_proba):
        base_stats = evaluate_thresholds(y_pre, base_proba, thr_grid)
        auc_b = base_stats.get('auc')
        auc_b_str = f"{auc_b:.4f}" if (auc_b is not None and not np.isnan(auc_b)) else "undefined"
        print(f"  Pre-test baseline: AUC={auc_b_str}, F1={base_stats['f1']:.4f}, MCC={base_stats.get('mcc', float('nan')):.4f}, θ={base_stats['thr']:.2f}")
except Exception:
    pass

print("\n[Pre-test Fold Summaries]")
for fs in fold_summaries:
    b = fs["summary"]
    auc_b = b.get('best_auc')
    auc_s = f"{auc_b:.4f}" if (auc_b is not None and not np.isnan(auc_b)) else "undefined"
    fold_ts_key = fs.get("fold_ts_key", "unknown")
    print(
        "  Fold {fold} ({ts_key}): train={tr:,} (pos={tpos}, neg={tneg}), val={va:,} (pos={vpos}, neg={vneg}), "
        "trials={trials}, time={time:.1f}s | AUC={auc}, F1={f1:.4f}, Acc={acc:.4f}, MCC={mcc:.4f}, θ={thr:.2f} | "
        "leaves={leaves}, depth={depth}, lr={lr:.4f}".format(
            fold=fs.get("fold"), ts_key=fold_ts_key, tr=fs.get("train_size", 0), tpos=fs.get("train_pos", 0), tneg=fs.get("train_neg", 0),
            va=fs.get("val_size", 0), vpos=fs.get("val_pos", 0), vneg=fs.get("val_neg", 0),
            trials=fs.get("trials", 0), time=fs.get("elapsed", 0.0),
            auc=auc_s, f1=b.get('best_f1', float('nan')), acc=b.get('best_acc', float('nan')),
            mcc=b.get('best_mcc', float('nan')),
            thr=b.get('best_thr', float('nan')),
            leaves=b.get('params', {}).get('num_leaves', None),
            depth=b.get('params', {}).get('max_depth', None),
            lr=b.get('params', {}).get('learning_rate', 0.05),
        )
    )
    # Print full LightGBM params used for this fold
    try:
        params_dict = b.get('params', {}) if isinstance(b.get('params', {}), dict) else {}
        params_clean = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (float, np.floating)) else v)
                        for k, v in params_dict.items()}
        print("    Params:")
        print(json.dumps(params_clean, indent=2))
    except Exception:
        try:
            print("    Params:", b.get('params', {}))
        except Exception:
            print("    Params: {}")

print("\n[Meta-Inferred Params — Method 1 (Train+Pre-test)]")
print(json.dumps({k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (float, np.floating)) else v)
                 for k, v in inferred_params_combined.items()}, indent=2))

print("\n[Meta-Inferred Params — Method 2 (Pre-test-only)]")
print(json.dumps({k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (float, np.floating)) else v)
                 for k, v in inferred_params_pretest.items()}, indent=2))

# (Detailed per-method test metrics print removed; per-window summaries already shown above.)
print(f"\n✅ Best Method (latest window): {best_method}")
print("==========================================================\n")


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
            Path(RUN_DIR, 'threshold_sweep_run2.json').write_text(json.dumps(sweep_results, indent=2), encoding='utf-8')
        except Exception:
            pass
    else:
        print("  No test probabilities available; skipping threshold sweep.")
except Exception:
    pass


# %% SAVE ARTIFACTS
model_path = os.path.join(RUN_DIR, 'lightgbm_three_window_cumulative_run2.joblib')
scaler_path = os.path.join(RUN_DIR, 'scaler_three_window_cumulative_run2.joblib')
joblib.dump(final_model, model_path)
joblib.dump(scaler, scaler_path)

results = {
    "windows": {
        "train_rows": int(len(train_df)),
        "pretest_rows": int(len(pre_df)),
        "num_test_windows": int(NUM_TEST_WINDOWS),
        "test_rows_each": int(TEST_ROWS),
        "gap_windows": int(NUM_GAP_WINDOWS),
        "gap_rows_ignored": int(ignored_rows),
        "train_span": [str(train_bounds[0]) if train_bounds[0] is not None else None, str(train_bounds[1]) if train_bounds[1] is not None else None],
        "pretest_span": [str(pretest_bounds[0]), str(pretest_bounds[1])],
        "test_spans": [[str(a), str(b)] for a, b in test_bounds_list],
    },
    "class_dist": {
        "train": class_dist(y_train),
        "pretest": class_dist(y_pre),
    },
    "pretest_folds": fold_summaries,
    "test_windows": test_windows_results,
    "feature_names": FEATURE_NAMES,
    "cv_config": {
        "fold_size": PRETEST_FOLD_SIZE,
        "gap_bars": GAP_BARS,
        "time_limit_s": FOLD_TIME_LIMIT,
        "test_rows": TEST_ROWS,
        "num_test_windows": int(NUM_TEST_WINDOWS),
        "shift_fold_offset": int(SHIFT_FOLD_OFFSET),
    },
}

results_path = os.path.join(RUN_DIR, 'results_three_window_cumulative_run2.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

run_meta = {
    "script_name": os.path.basename(__file__),
    "run_timestamp_utc": RUN_TS,
    "artifact_dir": RUN_DIR,
    "artifacts": {"model": model_path, "scaler": scaler_path, "results": results_path},
}
with open(os.path.join(RUN_DIR, 'run_meta_run2.json'), 'w') as f:
    json.dump(run_meta, f, indent=2)

try:
    code_txt = Path(__file__).read_text(encoding='utf-8')
    Path(RUN_DIR, f"code_{os.path.basename(__file__)}.txt").write_text(code_txt, encoding='utf-8')
except Exception:
    pass

print("Artifacts saved to:", RUN_DIR)

# Display adv_chain_refined stats for each test window
print(f"\n✅ RUN2 COMPLETE — adv_chain_refined Results:")
try:
    for result in test_windows_results:
        w_idx = result["window_index"]
        test_metrics = result.get("test_metrics", {})
        adv_refined_stats = test_metrics.get("adv_chain_refined", {})
        
        if adv_refined_stats:
            # Calculate selection score for adv_chain_refined
            auc_val_raw = adv_refined_stats.get("auc")
            f1 = float(adv_refined_stats.get("f1", 0.0))
            acc = float(adv_refined_stats.get("acc", 0.0))
            bacc = float(adv_refined_stats.get("bacc", 0.0))
            if auc_val_raw is None or (isinstance(auc_val_raw, float) and np.isnan(auc_val_raw)):
                score = (f1 + bacc) / 2.0
            else:
                try:
                    auc_for_score = float(auc_val_raw)
                    score = (f1 + auc_for_score + acc) / 3.0
                except Exception:
                    score = (f1 + bacc) / 2.0
            
            auc_disp = adv_refined_stats.get('auc')
            auc_str = f"{auc_disp:.4f}" if (auc_disp is not None and not np.isnan(auc_disp)) else "undefined"
            
            print(f"Window {w_idx}/{NUM_TEST_WINDOWS} | adv_chain_refined | Score={score:.4f} | F1={adv_refined_stats.get('f1', float('nan')):.4f} | AUC={auc_str} | ACC={adv_refined_stats.get('acc', float('nan')):.4f} | MCC={adv_refined_stats.get('mcc', float('nan')):.4f} | θ={adv_refined_stats.get('thr', 0.5):.2f}")
        else:
            print(f"Window {w_idx}/{NUM_TEST_WINDOWS} | adv_chain_refined | No stats available")
except Exception as e:
    print(f"\n✅ RUN2 COMPLETE — Error displaying adv_chain_refined stats: {str(e)}")
    # Fallback to original message
    try:
        latest_idx, _ = selected_models[-1]
        latest_result = next(r for r in test_windows_results if r["window_index"] == latest_idx)
        latest_stats = latest_result.get("selected_metrics", {})
        auc_disp = latest_stats.get('auc')
        auc_str = f"{auc_disp:.4f}" if (auc_disp is not None and not np.isnan(auc_disp)) else "undefined"
        print(f"Latest window {latest_idx} | AUC: {auc_str}, Acc: {latest_stats.get('acc', 0):.4f}")
    except Exception:
        print("RUN2 COMPLETE")

print("="*70)
