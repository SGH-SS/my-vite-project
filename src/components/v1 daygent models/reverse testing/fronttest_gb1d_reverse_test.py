# %% [cell] STEP 1: CROSS-PLATFORM SETUP AND PATHS
print("🔧 Setting up environment...")

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# Try optional Colab drive mount for convenience
try:
    from google.colab import drive  # type: ignore
    drive.mount('/content/drive')
    IS_COLAB = True
    BASE_DIR = '/content/drive/MyDrive/daygent_v1_models'
    print("✅ Google Drive mounted (Colab)")
except Exception:
    IS_COLAB = False
    BASE_DIR = './daygent_v1_models'
    print("✅ Local environment detected")

# Core imports (install if missing)
try:
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
    import joblib
except Exception as e:
    print(f"Installing missing packages due to: {e}")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy', 'pandas', 'scikit-learn', 'tqdm', 'joblib', 'pyarrow'])
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
    import joblib

# Locations
FRONTTEST_DIR = os.path.join(BASE_DIR, 'spy_data_fronttest')
MODEL_DIR = os.path.join(BASE_DIR, 'gb_1d')  # use the original gb_1d artifacts
OUTPUT_DIR = os.path.join(BASE_DIR, 'gb_1d_reverse_fronttest')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("🔄 GB1D Reverse Test on Fronttest SPY 1D")
print(f"📁 Fronttest dir: {FRONTTEST_DIR}")
print(f"📁 Model dir:     {MODEL_DIR}")
print(f"📁 Output dir:    {OUTPUT_DIR}")


# %% [cell] STEP 2: LOAD MODEL CONFIG + ARTIFACTS
print("\n📦 Loading model config and artifacts...")

config_path = os.path.join(MODEL_DIR, 'deployment_config.json')
results_path = os.path.join(MODEL_DIR, 'results_gb_1d.json')
model_path = os.path.join(MODEL_DIR, 'gb_1d_final.joblib')
scaler_path = os.path.join(MODEL_DIR, 'scaler_1d.joblib')

with open(config_path, 'r') as f:
    CONFIG = json.load(f)

with open(results_path, 'r') as f:
    ORIGINAL_RESULTS = json.load(f)

FEATURE_NAMES = CONFIG['feature_names']
THRESHOLD = float(CONFIG.get('calibrated_threshold', 0.5))

gb_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print(f"✅ Model loaded: {type(gb_model).__name__}")
print(f"✅ Scaler loaded: {type(scaler).__name__}")
print(f"📊 Feature contract: {len(FEATURE_NAMES)} features -> {FEATURE_NAMES}")
print(f"🎯 Threshold: {THRESHOLD}")


# %% [cell] STEP 3: LOAD FRONTTEST CSV (1d)
print("\n📥 Loading fronttest CSVs...")

csv_1d = os.path.join(FRONTTEST_DIR, 'fronttest_spy_1d.csv')
if not os.path.exists(csv_1d):
    raise FileNotFoundError(f"fronttest_spy_1d.csv not found in {FRONTTEST_DIR}")

df_1d = pd.read_csv(csv_1d)
if 'timestamp' not in df_1d.columns:
    raise ValueError("fronttest_spy_1d.csv must include 'timestamp'")

df_1d['timestamp'] = pd.to_datetime(df_1d['timestamp'])
df_1d = df_1d.sort_values('timestamp').reset_index(drop=True)

print(f"✅ 1d rows: {len(df_1d):,}; range: {df_1d['timestamp'].min()} → {df_1d['timestamp'].max()}")


# %% [cell] STEP 4: FEATURE EXTRACTION HELPERS (MATCH gb_1d CONTRACT)
print("\n🔧 Preparing feature helpers (gb_1d 16-feature contract)...")

TIMEFRAMES = ['1d', '4h']  # one-hot order used originally

def parse_vector_column(vector_str):
    if pd.isna(vector_str) or vector_str is None:
        return None
    if isinstance(vector_str, str):
        s = vector_str.strip('[]"')
        try:
            return np.array([float(x.strip()) for x in s.split(',')])
        except Exception:
            return None
    return np.array(vector_str)

def build_feature_vector(raw_ohlcv, iso_ohlc, tf, tf_list):
    o, h, l, c, v = raw_ohlcv
    feats = list(raw_ohlcv)
    feats.extend(list(iso_ohlc))
    feats.extend([1 if tf == t else 0 for t in tf_list])
    feats.extend([
        (h - l) / c if c else 0.0,           # hl_range
        (c - o) / o if o else 0.0,           # price_change
        (h - c) / c if c else 0.0,           # upper_shadow
        (c - l) / c if c else 0.0,           # lower_shadow
        (v / 1_000_000.0) if v is not None else 0.0  # volume_m
    ])
    return np.array(feats, dtype=float)

def row_to_features_and_label(row):
    raw_ohlcv = parse_vector_column(row.get('raw_ohlcv_vec'))
    iso_ohlc  = parse_vector_column(row.get('iso_ohlc'))
    future    = row.get('future')
    if raw_ohlcv is None or iso_ohlc is None or pd.isna(future):
        return None, None
    return build_feature_vector(raw_ohlcv, iso_ohlc, '1d', TIMEFRAMES), int(future)


# %% [cell] STEP 5: BUILD TEST MATRIX FROM FRONTTEST
print("\n📐 Building test matrices from fronttest data...")

X_test, y_test, meta = [], [], []
for idx, row in tqdm(df_1d.iterrows(), total=len(df_1d), desc='Fronttest rows'):
    feats, label = row_to_features_and_label(row)
    if feats is None:
        continue
    X_test.append(feats)
    y_test.append(label)
    meta.append({
        'index': int(idx),
        'timestamp': row['timestamp'],
        'close': row.get('close', np.nan),
        'future': int(label)
    })

X_test = np.array(X_test)
y_test = np.array(y_test)
print(f"✅ Test matrix: {X_test.shape} (labels: {np.bincount(y_test) if len(y_test) else '[]'})")

if X_test.size == 0:
    raise RuntimeError("No valid rows in fronttest CSV with required vectors + future label.")


# %% [cell] STEP 6: SCALE, PREDICT, METRICS
print("\n🧪 Inference + metrics...")

X_test_scaled = scaler.transform(X_test)
proba = gb_model.predict_proba(X_test_scaled)[:, 1]
pred = (proba >= THRESHOLD).astype(int)

acc = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) == 2 else float('nan')
print(f"🎯 Accuracy: {acc:.4f}")
print(f"🎯 AUC:      {auc:.4f}")

cm = confusion_matrix(y_test, pred)
print("\n📊 Confusion Matrix:\n", cm)
print("\n📋 Classification Report:\n", classification_report(y_test, pred, digits=4))


# %% [cell] STEP 7: DAY-BY-DAY ANALYSIS
print("\n📅 Day-by-day breakdown...")

results_rows = []
for i, info in enumerate(meta):
    ts = pd.Timestamp(info['timestamp'])
    date_key = ts.date()
    results_rows.append({
        'date': str(date_key),
        'timestamp': ts.isoformat(),
        'close': float(info['close']) if info['close'] is not None and not pd.isna(info['close']) else None,
        'future': int(info['future']),
        'prob_up': float(proba[i]),
        'pred': int(pred[i])
    })

df_results = pd.DataFrame(results_rows)

daily = df_results.groupby('date').apply(lambda g: pd.Series({
    'n': len(g),
    'acc': float((g['pred'] == g['future']).mean()),
    'avg_prob_up': float(g['prob_up'].mean()),
    'pred_up_rate': float((g['pred'] == 1).mean()),
    'true_up_rate': float((g['future'] == 1).mean())
})).reset_index()

summary = {
    'overall': {
        'n_samples': int(len(df_results)),
        'accuracy': float(acc),
        'auc': float(auc),
        'threshold': float(THRESHOLD)
    },
    'by_day': daily.to_dict(orient='records')
}

summary_path = os.path.join(OUTPUT_DIR, 'fronttest_summary_gb1d.json')
preds_csv = os.path.join(OUTPUT_DIR, 'fronttest_predictions_gb1d.csv')
daily_csv = os.path.join(OUTPUT_DIR, 'fronttest_daily_metrics_gb1d.csv')

with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

df_results.to_csv(preds_csv, index=False)
daily.to_csv(daily_csv, index=False)

print(f"\n✅ Saved: {summary_path}")
print(f"✅ Saved: {preds_csv}")
print(f"✅ Saved: {daily_csv}")


# %% [cell] STEP 8: DISPLAY TOP/BOTTOM DAYS
print("\n🏁 Best/Worst days by accuracy (>=3 samples/day)...")

eligible = daily[daily['n'] >= 3].copy()
if len(eligible):
    print("Top 5 days:")
    print(eligible.sort_values('acc', ascending=False).head(5))
    print("\nBottom 5 days:")
    print(eligible.sort_values('acc', ascending=True).head(5))
else:
    print("Not enough samples per-day for breakdown; showing head:")
    print(daily.head())


