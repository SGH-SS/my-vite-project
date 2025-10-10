"""
LightGBM 1D Training Script (ISO-style features)

This script mirrors the structure and behavior of `lgbm_iso.ipynb` but targets
the SPY 1D timeframe using the combined full-data CSVs shown in your Drive:
  MyDrive/daygent_v1_models/combined_spy_data/combined_spy_1d.csv

Key behavior:
 - Cross-platform dependency setup (optional install) and optional Google Drive mount
 - Data source: combined_spy_data/combined_spy_1d.csv
 - Restrict to the last 6 years of data
 - Evaluation window: last 65 trading days (1D candles)
 - Feature contract: same 16 features as used in the ISO notebooks
 - Scaler fit on first 80% of pre-test training data
 - LightGBM params identical to 4H notebook
 - Saves model, scaler, results JSON, predictions CSV, detailed TXT report,
   and deployment artifacts under: daygent_v1_models/lgbm_1d
"""

from __future__ import annotations

import os
import json
import warnings
from datetime import datetime
from typing import List, Optional, Tuple

# %% CELL 1: SETUP DEPENDENCIES & ENV
print("🔧 Setting up dependencies...")

# Cross-platform dependency installation (best-effort)
try:
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    import joblib
    import lightgbm as lgb
    print("✅ Core dependencies already available")
except Exception as e:  # pragma: no cover
    print(f"Installing missing dependencies: {e}")
    import sys, subprocess
    pkgs = [
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "matplotlib",
        "seaborn",
        "joblib",
        "tqdm",
        "pyarrow",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs)
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    import joblib
    import lightgbm as lgb
    print("✅ Dependencies installed")

# Optional Google Drive mount (Colab)
try:  # pragma: no cover
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    IS_COLAB = True
    BASE_DIR = "/content/drive/MyDrive/daygent_v1_models"
    print("✅ Google Drive mounted (Colab environment)")
except Exception:
    IS_COLAB = False
    BASE_DIR = "./daygent_v1_models"
    print("✅ Local environment detected")

# %% CELL 2: DIRECTORIES
warnings.filterwarnings("ignore")

# Directories
DATA_DIR = os.path.join(BASE_DIR, "combined_spy_data")
MODEL_DIR = os.path.join(BASE_DIR, "lgbm_1d_iso")
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"✅ Model directory: {MODEL_DIR}")
print(f"✅ Data directory: {DATA_DIR}")


# %% CELL 3: HELPERS & FEATURE CONTRACT

def parse_vector_column(vector_value) -> Optional[np.ndarray]:
    """Parse a vector column (stringified list or list-like) into np.array.

    Accepts values like:
      "[1.0, 2.0, 3.0]"
      "1.0, 2.0, 3.0"
      list/array-like
    Returns None if parsing fails.
    """
    if vector_value is None or (isinstance(vector_value, float) and np.isnan(vector_value)):
        return None
    if isinstance(vector_value, str):
        s = vector_value.strip().strip("[]\"")
        if not s:
            return None
        try:
            return np.array([float(x.strip()) for x in s.split(",")])
        except Exception:
            return None
    try:
        return np.array(vector_value, dtype=float)
    except Exception:
        return None


TIMEFRAMES_ORDERED: List[str] = ["1d", "4h"]

FEATURE_NAMES: List[str] = [
    "raw_o",
    "raw_h",
    "raw_l",
    "raw_c",
    "raw_v",
    "iso_0",
    "iso_1",
    "iso_2",
    "iso_3",
    "tf_1d",
    "tf_4h",
    "hl_range",
    "price_change",
    "upper_shadow",
    "lower_shadow",
    "volume_m",
]


def build_feature_vector_1d(raw_ohlcv: np.ndarray, iso_ohlc: np.ndarray) -> np.ndarray:
    """Build the 16-feature vector for a 1D sample.

    Order matches FEATURE_NAMES above and the ISO notebooks' contract.
    """
    if len(raw_ohlcv) != 5 or len(iso_ohlc) != 4:
        raise ValueError("Bad vector lengths for raw_ohlcv or iso_ohlc")

    o, h, l, c, v = raw_ohlcv
    features: List[float] = []

    # Raw OHLCV (5)
    features.extend([o, h, l, c, v])

    # ISO (4)
    features.extend(list(iso_ohlc))

    # One-hot timeframe for ['1d', '4h'] -> 1d=[1,0]
    features.extend([1, 0])

    # Engineered (5)
    hl_range = (h - l) / c if c else 0.0
    price_change = (c - o) / o if o else 0.0
    upper_shadow = (h - c) / c if c else 0.0
    lower_shadow = (c - l) / c if c else 0.0
    volume_m = v / 1_000_000.0
    features.extend([hl_range, price_change, upper_shadow, lower_shadow, volume_m])

    return np.array(features, dtype=float)


def extract_features_1d_only(row: pd.Series) -> Tuple[Optional[np.ndarray], Optional[int]]:
    raw_ohlcv = parse_vector_column(row.get("raw_ohlcv_vec"))
    iso_ohlc = parse_vector_column(row.get("iso_ohlc"))
    future = row.get("future")
    if raw_ohlcv is None or iso_ohlc is None or future is None or (isinstance(future, float) and np.isnan(future)):
        return None, None
    try:
        fv = build_feature_vector_1d(raw_ohlcv, iso_ohlc)
        return fv, int(future)
    except Exception:
        return None, None


# %% CELL 4: LOAD 1D DATA & SLICE TO LAST 6 YEARS

print("\n📊 Loading 1D full-data CSV from combined_spy_data...")
csv_file = os.path.join(DATA_DIR, "combined_spy_1d.csv")
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"❌ {csv_file} not found. Ensure your Drive has combined_spy_data/combined_spy_1d.csv")

df_1d = pd.read_csv(csv_file)
if "timestamp" not in df_1d.columns:
    raise RuntimeError("❌ Expected a 'timestamp' column in combined_spy_1d.csv")

df_1d["timestamp"] = pd.to_datetime(df_1d["timestamp"])
df_1d = df_1d.sort_values("timestamp").reset_index(drop=True)

print(f"✅ Loaded 1d data: {len(df_1d):,} candles")
print(f"📅 Range: {df_1d['timestamp'].min()} → {df_1d['timestamp'].max()}")

# Restrict to the last 6 years
latest_ts = df_1d["timestamp"].max()
cutoff_ts = latest_ts - pd.DateOffset(years=6)
df_1d = df_1d[df_1d["timestamp"] >= cutoff_ts].copy()
df_1d = df_1d.sort_values("timestamp").reset_index(drop=True)
print(f"🗂️ Using last 6 years only: {df_1d['timestamp'].min()} → {df_1d['timestamp'].max()}  ({len(df_1d):,} rows)")


# %% CELL 5: DEFINE TEST PERIOD (LAST 65 TRADING DAYS)

all_days = list(pd.to_datetime(df_1d["timestamp"]).dt.date.unique())
if len(all_days) < 65:
    raise RuntimeError("❌ Not enough 1D candles for a 65-day evaluation window after 6y filter")

selected_days = all_days[-65:]
test_start = pd.Timestamp.combine(selected_days[0], pd.Timestamp.min.time()).tz_localize("UTC")
test_end = pd.Timestamp.combine(selected_days[-1], pd.Timestamp.max.time()).tz_localize("UTC")

print(f"\n🎯 Test period (65 trading days): {test_start.date()} → {test_end.date()}")


# %% CELL 6: FEATURE EXTRACTION (1D ONLY)

print("\n🔄 Extracting features from 1d data...")

train_df = df_1d[df_1d["timestamp"] < test_start].copy()
test_df = df_1d[(df_1d["timestamp"] >= test_start) & (df_1d["timestamp"] <= test_end)].copy()

print(f"📊 Train samples (rows before test window): {len(train_df):,}")
print(f"📊 Test samples (rows inside 65-day window): {len(test_df):,}")

# Training features
X_train: List[np.ndarray] = []
y_train: List[int] = []
for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting 1d train features"):
    fv, lbl = extract_features_1d_only(row)
    if fv is not None:
        X_train.append(fv)
        y_train.append(lbl)

X_train = np.array(X_train)
y_train = np.array(y_train)
print(f"\n✅ Training features extracted: {X_train.shape}")
if len(y_train):
    print(f"📊 Class distribution: {np.bincount(y_train)}")

# Test features + raw info for reporting
X_test: List[np.ndarray] = []
y_test: List[int] = []
test_rows_info: List[dict] = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Extracting 1d test features"):
    fv, lbl = extract_features_1d_only(row)
    if fv is not None:
        X_test.append(fv)
        y_test.append(lbl)
        test_rows_info.append({
            "timestamp": row["timestamp"],
            "raw_ohlcv": parse_vector_column(row.get("raw_ohlcv_vec")),
            "iso_ohlc": parse_vector_column(row.get("iso_ohlc")),
            "future": int(row.get("future")),
            "feature_vector": fv,
        })

X_test = np.array(X_test)
y_test = np.array(y_test)
print(f"📊 Test features extracted: {X_test.shape}")


# %% CELL 7: SCALE AND SPLIT (MATCHING W2 LOGIC)

scaler = StandardScaler()
split_idx = int(len(X_train) * 0.8)
print(f"\n🔧 Fitting scaler on first {split_idx:,} training samples...")
if split_idx > 0:
    scaler.fit(X_train[:split_idx])
else:
    scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_tr = X_train_scaled[:split_idx]
X_val = X_train_scaled[split_idx:]
y_tr = y_train[:split_idx]
y_val = y_train[split_idx:]

print(f"📊 Training set: {X_tr.shape}")
print(f"📊 Validation set: {X_val.shape}")


# %% CELL 8: TRAIN LIGHTGBM (EXACT PARAMS) + CALIBRATE THR

print("\n🚀 Training LightGBM (1d) with exact params...")

lgb_params = {
    # Match v0 script settings
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

if IS_COLAB:  # Optional GPU acceleration
    lgb_params["device_type"] = "gpu"
    lgb_params["gpu_device_id"] = 0
    print("✅ GPU acceleration enabled")

model = lgb.LGBMClassifier(**lgb_params)
print("🔄 Training LightGBM (no early stopping - full n_estimators)...")
if len(X_tr) and len(y_tr):
    model.fit(X_tr, y_tr)
else:
    # Fall back to all train data if split produced empty slice
    model.fit(X_train_scaled, y_train)

# Validation metrics and threshold calibration
if len(X_val):
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_pred_proba >= 0.5).astype(int)
    val_acc = accuracy_score(y_val, val_pred)
    val_auc = roc_auc_score(y_val, val_pred_proba)
else:
    val_pred_proba = np.array([])
    val_acc = float("nan")
    val_auc = float("nan")

print(f"\n✅ Validation Accuracy (t=0.50): {val_acc:.4f}")
print(f"✅ Validation AUC: {val_auc:.4f}")

best_thr = 0.5
best_val_acc = val_acc
if len(val_pred_proba):
    for thr in np.round(np.arange(0.30, 0.801, 0.01), 2):
        preds_thr = (val_pred_proba >= thr).astype(int)
        acc_thr = accuracy_score(y_val, preds_thr)
        if acc_thr > best_val_acc:
            best_val_acc = acc_thr
            best_thr = float(thr)

print(f"✅ Calibrated decision threshold on validation: {best_thr:.2f} (Acc={best_val_acc:.4f})")

# Refit on all in-sample (train + val)
X_full = X_train_scaled
y_full = y_train
model_full = lgb.LGBMClassifier(**lgb_params)
model_full.fit(X_full, y_full)


# %% CELL 9: TEST + DETAILED PREDICTION-BY-PRED REPORT

print(f"\n🧪 Testing on isolated {len(selected_days)}-day period (1d)...")

X_test_scaled = scaler.transform(X_test) if len(X_test) else np.empty((0, X_full.shape[1]))
test_pred_proba = model_full.predict_proba(X_test_scaled)[:, 1] if len(X_test_scaled) else np.array([])
test_pred = (test_pred_proba >= best_thr).astype(int) if len(test_pred_proba) else np.array([])

if len(test_pred):
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_pred_proba) if len(np.unique(y_test)) == 2 else float("nan")
else:
    test_acc = float("nan")
    test_auc = float("nan")

print("\n🎯 TEST RESULTS (1d):")
print(f"✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Test AUC: {test_auc:.4f}")
if len(test_pred):
    print(f"📊 Test predictions: {np.bincount(test_pred)}")
    print(f"📊 Actual labels: {np.bincount(y_test)}")

# Build detailed per-prediction table
records: List[dict] = []
for i, info in enumerate(test_rows_info):
    ts = info["timestamp"]
    fv = info["feature_vector"]
    raw = info["raw_ohlcv"]
    iso = info["iso_ohlc"]
    true = info["future"]
    proba = float(test_pred_proba[i]) if len(test_pred_proba) else float("nan")
    pred = int(test_pred[i]) if len(test_pred) else int(0)
    correct = bool(pred == true) if len(test_pred) else False
    margin = proba - best_thr if len(test_pred_proba) else float("nan")

    rec = {
        "candle_index_in_test": i + 1,
        "timestamp_utc": ts,
        "date_utc": pd.Timestamp(ts).date(),
        "pred_prob_up": proba,
        "pred_label": int(pred),
        "true_label": int(true),
        "correct": correct,
        "threshold_used": best_thr,
        "decision_margin": margin,
        # Raw OHLCV & ISO
        "raw_o": raw[0] if raw is not None else np.nan,
        "raw_h": raw[1] if raw is not None else np.nan,
        "raw_l": raw[2] if raw is not None else np.nan,
        "raw_c": raw[3] if raw is not None else np.nan,
        "raw_v": raw[4] if raw is not None else np.nan,
        "iso_0": iso[0] if iso is not None else np.nan,
        "iso_1": iso[1] if iso is not None else np.nan,
        "iso_2": iso[2] if iso is not None else np.nan,
        "iso_3": iso[3] if iso is not None else np.nan,
        # Engineered from feature vector indices
        "tf_1d": fv[FEATURE_NAMES.index("tf_1d")],
        "tf_4h": fv[FEATURE_NAMES.index("tf_4h")],
        "hl_range": fv[FEATURE_NAMES.index("hl_range")],
        "price_change": fv[FEATURE_NAMES.index("price_change")],
        "upper_shadow": fv[FEATURE_NAMES.index("upper_shadow")],
        "lower_shadow": fv[FEATURE_NAMES.index("lower_shadow")],
        "volume_m": fv[FEATURE_NAMES.index("volume_m")],
    }
    records.append(rec)

pred_df = pd.DataFrame.from_records(records).sort_values(["date_utc", "timestamp_utc"]).reset_index(drop=True)

# Save artifacts: predictions CSV and human-readable TXT
pred_csv_path = os.path.join(MODEL_DIR, "test_predictions_1d.csv")
pred_df.to_csv(pred_csv_path, index=False)

txt_lines: List[str] = []
txt_lines.append("=" * 90)
txt_lines.append("LIGHTGBM 1D — DETAILED DAY-BY-DAY / PREDICTION-BY-PREDICTION REPORT")
txt_lines.append("=" * 90)
txt_lines.append(f"Test period: {test_start.date()} → {test_end.date()}")
txt_lines.append(f"Total test candles: {len(pred_df)}")
txt_lines.append(f"Calibrated threshold: {best_thr:.2f}")
txt_lines.append(f"Overall Test Accuracy: {test_acc:.4f}")
txt_lines.append(f"Overall Test AUC: {test_auc:.4f}")
txt_lines.append("")

for day in pred_df["date_utc"].unique():
    day_block = pred_df[pred_df["date_utc"] == day]
    correct_n = int(day_block["correct"].sum())
    total_n = len(day_block)
    txt_lines.append("-" * 90)
    txt_lines.append(f"{day}  —  Day accuracy: {correct_n}/{total_n}  ({(correct_n/total_n) if total_n else 0:.3f})")
    txt_lines.append("-" * 90)
    for _, r in day_block.iterrows():
        dir_word = "UP" if r["pred_label"] == 1 else "DOWN"
        truth_word = "UP" if r["true_label"] == 1 else "DOWN"
        right_wrong = "✅ CORRECT" if r["correct"] else "❌ WRONG"
        txt_lines.append(
            f"[{int(r['candle_index_in_test']):02d}] {r['timestamp_utc']}  "
            f"pred={dir_word}  p_up={r['pred_prob_up']:.4f}  thr={r['threshold_used']:.2f}  "
            f"margin={r['decision_margin']:.4f}  truth={truth_word}  → {right_wrong}"
        )
        txt_lines.append(
            f"    OHLCV: O={r['raw_o']:.4f}, H={r['raw_h']:.4f}, L={r['raw_l']:.4f}, C={r['raw_c']:.4f}, V={r['raw_v']:.0f} | "
            f"ISO: [{r['iso_0']:.4f}, {r['iso_1']:.4f}, {r['iso_2']:.4f}, {r['iso_3']:.4f}] | "
            f"feats: hl={r['hl_range']:.4f}, dC={r['price_change']:.4f}, upSh={r['upper_shadow']:.4f}, "
            f"loSh={r['lower_shadow']:.4f}, vol_m={r['volume_m']:.4f}"
        )
    txt_lines.append("")

report_path = os.path.join(MODEL_DIR, "lgbm_1d_day_by_day.txt")
with open(report_path, "w") as f:
    f.write("\n".join(txt_lines))

print(f"\n📝 Saved detailed TXT report to: {report_path}")
print(f"🧾 Saved machine-readable predictions to: {pred_csv_path}")


# %% CELL 10: SAVE MODEL, SCALER, AND RESULTS

print("\n💾 Saving model and results...")

model_path = os.path.join(MODEL_DIR, "lightgbm_financial_1d_only.joblib")
scaler_path = os.path.join(MODEL_DIR, "scaler_1d_only.joblib")
joblib.dump(model_full, model_path)
joblib.dump(scaler, scaler_path)


def _to_py(v):
    try:
        if isinstance(v, (np.integer, np.int64, np.int32)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v
    except Exception:
        return v


results = {
    "test_accuracy": float(test_acc),
    "test_auc": float(test_auc),
    "validation_accuracy": float(best_val_acc),
    "validation_auc": float(val_auc) if not (isinstance(val_auc, float) and np.isnan(val_auc)) else None,
    "train_samples": int(len(X_tr)) if len(X_tr) else int(len(X_train_scaled)),
    "val_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "feature_count": int(X_full.shape[1]) if X_full.ndim == 2 else 0,
    "chosen_threshold": float(best_thr),
    "model_params": {k: _to_py(v) for k, v in lgb_params.items()},
    "feature_names": FEATURE_NAMES,
    "report_txt": os.path.basename(report_path),
    "predictions_csv": os.path.basename(pred_csv_path),
    "model_path": os.path.basename(model_path),
    "scaler_path": os.path.basename(scaler_path),
    "test_period": f"{test_start.date()} to {test_end.date()}",
}

with open(os.path.join(MODEL_DIR, "results_1d_only.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Model saved to: {model_path}")
print(f"✅ Scaler saved to: {scaler_path}")
print("✅ Results JSON saved as: results_1d_only.json")


# %% CELL 11: SAVE DEPLOYMENT ARTIFACTS (for your site)

deployment_config = {
    "model_type": "LightGBMClassifier",
    "timeframe": "1d",
    "feature_contract_version": "v1",
    "feature_names": FEATURE_NAMES,
    "calibrated_threshold": float(best_thr),
    "artifact_paths": {
        "model_joblib": os.path.basename(model_path),
        "scaler_joblib": os.path.basename(scaler_path),
    },
    "inference_notes": {
        "scaling": "StandardScaler fitted on first 80% of pre-test training data",
        "one_hot": {"tf_1d": 1, "tf_4h": 0},
        "expected_columns_in_csv": ["timestamp", "raw_ohlcv_vec", "iso_ohlc", "future"],
    },
    "lgbm_params": {k: _to_py(v) for k, v in lgb_params.items()},
}

config_path = os.path.join(MODEL_DIR, "deployment_config.json")
with open(config_path, "w") as f:
    json.dump(deployment_config, f, indent=2)

feature_schema = {
    "raw_ohlcv_vec": {
        "desc": "Stringified list of [open, high, low, close, volume]",
        "len": 5,
        "dtype": "float",
    },
    "iso_ohlc": {
        "desc": "Stringified list of 4 ISO-normalized OHLC values",
        "len": 4,
        "dtype": "float",
    },
    "engineered": [
        "hl_range=(H-L)/C",
        "price_change=(C-O)/O",
        "upper_shadow=(H-C)/C",
        "lower_shadow=(C-L)/C",
        "volume_m=V/1e6",
    ],
    "tf_one_hot": {"tf_1d": 1, "tf_4h": 0},
}

schema_path = os.path.join(MODEL_DIR, "feature_schema.json")
with open(schema_path, "w") as f:
    json.dump(feature_schema, f, indent=2)

readme_text = f"""
============================================
LightGBM 1D Inference — Deployment Notes
============================================

Artifacts:
- Model:       {os.path.basename(model_path)}
- Scaler:      {os.path.basename(scaler_path)}
- Config:      {os.path.basename(config_path)}
- Feature schema: feature_schema.json
- Threshold:   {best_thr:.2f}
- Predictions: test_predictions_1d.csv
- Report:      lgbm_1d_day_by_day.txt

Feature order (must match EXACTLY):
{FEATURE_NAMES}

Inference pipeline for your site:
1) Parse raw input row:
   - Parse 'raw_ohlcv_vec' -> [o,h,l,c,v]
   - Parse 'iso_ohlc'      -> [iso_0..iso_3]
   - Add one-hot: tf_1d=1, tf_4h=0
   - Compute engineered features as in feature_schema.json
   - Concatenate into a single 16-length vector in the listed order.

2) Load scaler with joblib and call scaler.transform([vector]).
3) Load model with joblib and call model.predict_proba(scaled)[0,1].
4) If prob >= {best_thr:.2f} => predict UP (1); else DOWN (0).

Notes:
- Trained with class_weight='balanced'.
- Scaler fit on the first 80% of pre-test (1d) training data.
- Keep feature order and scaling identical for consistent results.
""".strip()

readme_path = os.path.join(MODEL_DIR, "README_DEPLOY_1D.txt")
with open(readme_path, "w") as f:
    f.write(readme_text)

print("📦 Deployment artifacts saved:")
print(" -", config_path)
print(" -", schema_path)
print(" -", readme_path)


# %% CELL 12: FINAL SUMMARY

print("\n" + "=" * 70)
print("🏆 LIGHTGBM_FINANCIAL 1D-ONLY — COMPLETE")
print("=" * 70)
print(f" • Model dir:    {MODEL_DIR}")
print(f" • Test window:  {test_start.date()} → {test_end.date()}")
print(f" • Test candles: {len(X_test)}")
print(f" • Test Acc/AUC: {test_acc:.4f} / {test_auc:.4f}")
print(f" • Threshold:    {best_thr:.2f}")
print(
    " • Saved files:  "
    + ", ".join(
        [
            os.path.basename(model_path),
            os.path.basename(scaler_path),
            "deployment_config.json",
            "feature_schema.json",
            "README_DEPLOY_1D.txt",
            os.path.basename(pred_csv_path),
            os.path.basename(report_path),
            "results_1d_only.json",
        ]
    )
)
print("=" * 70)


