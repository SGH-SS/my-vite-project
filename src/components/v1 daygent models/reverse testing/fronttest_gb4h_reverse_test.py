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
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy', 'pandas', 'scikit-learn', 'tqdm', 'joblib', 'pyarrow', 'matplotlib', 'lightgbm'])
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
    import joblib

# Locations
FRONTTEST_DIR = os.path.join(BASE_DIR, 'spy_data_fronttest')
MODEL_DIR = os.path.join(BASE_DIR, 'gb_4h')
OUTPUT_DIR = os.path.join(BASE_DIR, 'gb_4h_reverse_fronttest')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("🔄 GB4H Reverse Test on Fronttest SPY 4H")
print(f"📁 Fronttest dir: {FRONTTEST_DIR}")
print(f"📁 Model dir:     {MODEL_DIR}")
print(f"📁 Output dir:    {OUTPUT_DIR}")


# %% [cell] STEP 2: LOAD MODEL CONFIG + ARTIFACTS
print("\n📦 Loading model config and artifacts...")

def first_existing(path: str, candidates):
    for name in candidates:
        p = os.path.join(path, name)
        if os.path.exists(p):
            return p
    return None

config_path = first_existing(MODEL_DIR, ['deployment_config_4h.json', 'deployment_config.json'])
results_path = first_existing(MODEL_DIR, ['results_gb_4h_w2_style.json', 'results_gb_4h.json', 'results_4h_only.json'])
model_path = first_existing(MODEL_DIR, ['gb_4h_w2_style.joblib', 'gb_4h_final.joblib', 'gb_4h.joblib'])
scaler_path = first_existing(MODEL_DIR, ['scaler_4h_w2_style.joblib', 'scaler_4h.joblib', 'scaler_4h_only.joblib'])

if not all([config_path, model_path, scaler_path]):
    raise FileNotFoundError(f"Missing required artifacts in {MODEL_DIR}. Found: config={config_path}, model={model_path}, scaler={scaler_path}")

with open(config_path, 'r') as f:
    CONFIG = json.load(f)

ORIGINAL_RESULTS = None
if results_path and os.path.exists(results_path):
    try:
        with open(results_path, 'r') as f:
            ORIGINAL_RESULTS = json.load(f)
    except Exception:
        ORIGINAL_RESULTS = None

FEATURE_NAMES = CONFIG.get('feature_names') or CONFIG.get('features')
THRESHOLD = float(CONFIG.get('calibrated_threshold', 0.5))

gb_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print(f"✅ Model loaded: {type(gb_model).__name__}")
print(f"✅ Scaler loaded: {type(scaler).__name__}")
if FEATURE_NAMES:
    print(f"📊 Feature contract: {len(FEATURE_NAMES)} features -> {FEATURE_NAMES}")
else:
    print("📊 Feature contract: unknown (no feature_names in config)")
print(f"🎯 Threshold: {THRESHOLD}")


# %% [cell] STEP 3: LOAD FRONTTEST CSV (4h)
print("\n📥 Loading fronttest CSVs...")

def find_fronttest_csv(base_dir: str, preferred_names, prefix: str):
    for name in preferred_names:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p
    # Fallback: search by prefix
    try:
        for fname in os.listdir(base_dir):
            if fname.lower().startswith(prefix) and fname.lower().endswith('.csv'):
                return os.path.join(base_dir, fname)
    except FileNotFoundError:
        pass
    return None

csv_4h = find_fronttest_csv(
    FRONTTEST_DIR,
    ['fronttest_spy_4h.csv', 'fronttest_spy_4h_only.csv', 'fronttest_spy_4h_w2_style.csv'],
    'fronttest_spy_4h'
)

if not csv_4h or not os.path.exists(csv_4h):
    raise FileNotFoundError(f"fronttest_spy_4h*.csv not found in {FRONTTEST_DIR}")

df_4h = pd.read_csv(csv_4h)
if 'timestamp' not in df_4h.columns:
    raise ValueError("4h fronttest CSV must include 'timestamp'")

df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
df_4h = df_4h.sort_values('timestamp').reset_index(drop=True)

print(f"✅ 4h rows: {len(df_4h):,}; range: {df_4h['timestamp'].min()} → {df_4h['timestamp'].max()}")


# %% [cell] STEP 4: FEATURE EXTRACTION HELPERS (MATCH gb_4h CONTRACT)
print("\n🔧 Preparing feature helpers (gb_4h 16-feature contract)...")

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
    return build_feature_vector(raw_ohlcv, iso_ohlc, '4h', TIMEFRAMES), int(future)


# %% [cell] STEP 5: BUILD TEST MATRIX FROM FRONTTEST
print("\n📐 Building test matrices from fronttest data...")

X_test, y_test, meta = [], [], []
for idx, row in tqdm(df_4h.iterrows(), total=len(df_4h), desc='Fronttest rows'):
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

summary_path = os.path.join(OUTPUT_DIR, 'fronttest_summary_gb4h.json')
preds_csv = os.path.join(OUTPUT_DIR, 'fronttest_predictions_gb4h.csv')
daily_csv = os.path.join(OUTPUT_DIR, 'fronttest_daily_metrics_gb4h.csv')

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


# %% [cell] STEP 9: VISUALIZE DAILY ACCURACY
print("\n📈 Visualizing daily accuracy...")
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import PercentFormatter

    if len(daily):
        dplot = daily.copy()
        dplot['date_dt'] = pd.to_datetime(dplot['date'])
        dplot = dplot.sort_values('date_dt')

        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=140)
        sizes = 20 + 3.0 * dplot['n'].astype(float)
        sc = ax.scatter(
            dplot['date_dt'], dplot['acc'],
            c=dplot['n'], cmap='viridis', s=sizes,
            alpha=0.9, edgecolor='k', linewidth=0.3, label='Daily accuracy'
        )
        ax.plot(dplot['date_dt'], dplot['acc'], color='gray', alpha=0.35, linewidth=1)
        ax.axhline(acc, color='#1f77b4', linestyle='--', linewidth=1.5, label=f'Overall: {acc:.2%}')
        ax.set_title('Daily Accuracy (4h)', fontsize=13)
        ax.set_xlabel('Date'); ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='lower left')
        cbar = plt.colorbar(sc, ax=ax, pad=0.015); cbar.set_label('Samples per day (n)')
        plt.tight_layout()

        out_plot = os.path.join(OUTPUT_DIR, 'fronttest_daily_accuracy_gb4h.png')
        plt.savefig(out_plot, bbox_inches='tight'); print(f"✅ Saved plot to {out_plot}")
        plt.close(fig)
except Exception as e:
    print(f"Plotting skipped due to: {e}")


# %% [cell] STEP 10: PERIOD-SPECIFIC ANALYSIS (Apr 25 → Jun 13)
print("\n🔎 Period analysis: Apr 25 → Jun 13 (deep dive)")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve

# Safety checks
required_cols = {'date', 'timestamp', 'future', 'prob_up'}
missing_cols = required_cols - set(df_results.columns)
if missing_cols:
    raise ValueError(f"df_results is missing columns: {missing_cols}")

# Normalize dates and infer year
dfp = df_results.copy()
dfp['date_dt'] = pd.to_datetime(dfp['date'])
dfp['timestamp_dt'] = pd.to_datetime(dfp['timestamp'], utc=True).dt.tz_localize(None)
latest_year = int(dfp['date_dt'].dt.year.max())

# Define window (inclusive)
start_date = pd.Timestamp(latest_year, 4, 25)
end_date = pd.Timestamp(latest_year, 6, 13)
mask = (dfp['date_dt'] >= start_date) & (dfp['date_dt'] <= end_date)
dfw = dfp.loc[mask].sort_values('timestamp_dt').reset_index(drop=True)

if dfw.empty:
    print(f"⚠️ No samples between {start_date.date()} and {end_date.date()} (year inferred: {latest_year}).")
else:
    y_true = dfw['future'].astype(int).to_numpy()
    p_up = dfw['prob_up'].astype(float).to_numpy()
    pred_w = (p_up >= THRESHOLD).astype(int)

    # Core metrics
    acc_w = accuracy_score(y_true, pred_w)
    bal_acc_w = balanced_accuracy_score(y_true, pred_w)
    auc_w = roc_auc_score(y_true, p_up) if len(np.unique(y_true)) == 2 else float('nan')
    ap_w = average_precision_score(y_true, p_up) if len(np.unique(y_true)) == 2 else float('nan')
    brier_w = float(np.mean((p_up - y_true) ** 2))
    cm_w = confusion_matrix(y_true, pred_w)

    # Per-day summary within window
    daily_w = dfw.groupby('date_dt').apply(lambda g: pd.Series({
        'n': len(g),
        'acc': float((g['prob_up'].ge(THRESHOLD).astype(int) == g['future']).mean()),
        'avg_prob_up': float(g['prob_up'].mean()),
        'pred_up_rate': float((g['prob_up'].ge(THRESHOLD)).mean()),
        'true_up_rate': float((g['future'] == 1).mean())
    })).reset_index().sort_values('date_dt')

    # Threshold sweep for sensitivity
    def evaluate_thresholds(y, p, thresholds):
        rows = []
        for t in thresholds:
            pred_t = (p >= t).astype(int)
            acc_t = accuracy_score(y, pred_t)
            bal_acc_t = balanced_accuracy_score(y, pred_t)
            prec, rec, f1, _ = precision_recall_fscore_support(y, pred_t, average='binary', zero_division=0)
            pos_rate = float((pred_t == 1).mean())
            rows.append({
                'threshold': float(t),
                'accuracy': float(acc_t),
                'balanced_accuracy': float(bal_acc_t),
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1),
                'pred_up_rate': float(pos_rate)
            })
        return pd.DataFrame(rows)

    sweep = evaluate_thresholds(y_true, p_up, np.round(np.arange(0.30, 0.701, 0.025), 3))
    best_by_acc = sweep.sort_values('accuracy', ascending=False).head(3).reset_index(drop=True)
    best_by_f1 = sweep.sort_values('f1', ascending=False).head(3).reset_index(drop=True)

    # Print summary
    print(f"📅 Window: {start_date.date()} → {end_date.date()} (year inferred: {latest_year})")
    print(f"🧮 Samples: {len(dfw):,} | Days: {dfw['date_dt'].nunique()}")
    print(f"🎯 Accuracy: {acc_w:.4f} | Balanced Acc: {bal_acc_w:.4f}")
    print(f"📈 ROC-AUC: {auc_w:.4f} | PR-AUC: {ap_w:.4f}")
    print(f"🎯 Threshold used: {THRESHOLD:.3f} | Brier score: {brier_w:.5f}")
    print("\n📊 Confusion Matrix (rows=true, cols=pred):")
    print(cm_w)
    print("\n📋 Classification Report:")
    print(classification_report(y_true, pred_w, digits=4, zero_division=0))

    # Save artifacts
    period_tag = f"{start_date.date()}_to_{end_date.date()}"
    out_summary = {
        'period': {'start': str(start_date.date()), 'end': str(end_date.date())},
        'n_samples': int(len(dfw)),
        'n_days': int(dfw['date_dt'].nunique()),
        'threshold': float(THRESHOLD),
        'metrics': {
            'accuracy': float(acc_w),
            'balanced_accuracy': float(bal_acc_w),
            'roc_auc': float(auc_w),
            'pr_auc': float(ap_w),
            'brier': float(brier_w)
        },
        'confusion_matrix': cm_w.tolist()
    }
    summary_json = os.path.join(OUTPUT_DIR, f'period_summary_{period_tag}.json')
    sweep_csv = os.path.join(OUTPUT_DIR, f'period_threshold_sweep_{period_tag}.csv')
    preds_csv = os.path.join(OUTPUT_DIR, f'period_predictions_{period_tag}.csv')
    daily_csv_w = os.path.join(OUTPUT_DIR, f'period_daily_metrics_{period_tag}.csv')
    with open(summary_json, 'w') as f:
        json.dump(out_summary, f, indent=2)
    sweep.to_csv(sweep_csv, index=False)
    dfw[['date', 'timestamp', 'close', 'future', 'prob_up']].to_csv(preds_csv, index=False)
    daily_w.to_csv(daily_csv_w, index=False)
    print(f"\n✅ Saved summary: {summary_json}")
    print(f"✅ Saved threshold sweep: {sweep_csv}")
    print(f"✅ Saved predictions: {preds_csv}")
    print(f"✅ Saved daily metrics: {daily_csv_w}")

    # Visualization 1: Daily accuracy in window
    if len(daily_w):
        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=140)
        dplot = daily_w.copy()
        dplot['acc_ma'] = dplot['acc'].rolling(window=7, min_periods=1).mean()
        sizes = 20 + 3.0 * dplot['n'].astype(float)
        sc = ax.scatter(
            dplot['date_dt'], dplot['acc'],
            c=dplot['n'], cmap='viridis', s=sizes,
            alpha=0.9, edgecolor='k', linewidth=0.3, label='Daily accuracy'
        )
        ax.plot(dplot['date_dt'], dplot['acc'], color='gray', alpha=0.35, linewidth=1)
        ax.plot(dplot['date_dt'], dplot['acc_ma'], color='#d62728', linewidth=2.2, label='7-day moving avg')
        ax.axhline(acc_w, color='#1f77b4', linestyle='--', linewidth=1.5, label=f'Window overall: {acc_w:.2%}')
        ax.set_title(f'Daily Accuracy ({start_date.date()} → {end_date.date()})', fontsize=13)
        ax.set_xlabel('Date'); ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='lower left')
        cbar = plt.colorbar(sc, ax=ax, pad=0.015); cbar.set_label('Samples per day (n)')
        plt.tight_layout()
        out_plot1 = os.path.join(OUTPUT_DIR, f'period_daily_accuracy_{period_tag}.png')
        plt.savefig(out_plot1, bbox_inches='tight'); print(f"🖼️ Saved plot: {out_plot1}")
        plt.close(fig)

    # Visualization 2: Probability separation + reliability
    if len(np.unique(y_true)) == 2 and len(y_true) >= 6:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
        # Prob distributions
        axes[0].hist(p_up[y_true == 1], bins=15, alpha=0.7, label='True Up', color='#1f77b4')
        axes[0].hist(p_up[y_true == 0], bins=15, alpha=0.7, label='True Down', color='#ff7f0e')
        axes[0].axvline(THRESHOLD, color='k', linestyle='--', linewidth=1.2, label=f'Threshold {THRESHOLD:.2f}')
        axes[0].set_title('Probabilities by Class')
        axes[0].set_xlabel('P(up)'); axes[0].set_ylabel('Count'); axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.3)
        # Reliability curve (adaptive bins for small samples)
        n_bins = max(3, min(10, len(y_true) // 2))
        prob_true, prob_pred = calibration_curve(y_true, p_up, n_bins=n_bins, strategy='quantile')
        axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly calibrated')
        axes[1].plot(prob_pred, prob_true, marker='o', color='#2ca02c', linewidth=1.5, label='Model')
        axes[1].set_title(f'Reliability (Brier={np.mean((p_up - y_true) ** 2):.4f})')
        axes[1].set_xlabel('Predicted P(up)'); axes[1].set_ylabel('Observed frequency')
        axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1); axes[1].grid(True, linestyle='--', alpha=0.3)
        axes[1].legend()
        plt.tight_layout()
        out_plot2 = os.path.join(OUTPUT_DIR, f'period_distribution_reliability_{period_tag}.png')
        plt.savefig(out_plot2, bbox_inches='tight'); print(f"🖼️ Saved plot: {out_plot2}")
        plt.close(fig)

    # Threshold sweep highlights
    print("\n🔧 Threshold sensitivity (top by accuracy):")
    print(best_by_acc)
    print("\n🔧 Threshold sensitivity (top by F1):")
    print(best_by_f1)


