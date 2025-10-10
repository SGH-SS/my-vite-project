# Three-Window Meta-Learning Training Script - Improvements

## Overview
The `train_three_window_meta.py` script has been significantly enhanced with advanced meta-learning, time-limited hyperparameter optimization, and comprehensive progress tracking.

## Key Improvements

### 1. Advanced ML-Based Meta-Learner ✨

**Previous**: Simple weighted averaging of hyperparameters from prior folds.

**New**: Sophisticated `MetaLearner` class that:
- Uses Ridge regression ensemble to learn hyperparameter selection
- Trains 8 separate regressors (one per hyperparameter: lr, num_leaves, max_depth, etc.)
- Extracts rich meta-features from fold characteristics:
  - Sample size (n)
  - Class distribution (pos_rate)
  - Feature statistics (mean, std)
  - Class separation (distance between positive/negative class means)
  - Performance metrics (F1, AUC from prior folds)
- Predicts optimal hyperparameters based on data characteristics
- Automatically falls back to weighted averaging if insufficient folds

**Benefits**:
- Learns relationships between data properties and optimal hyperparameters
- Adapts to changing market regimes across folds
- More principled than simple averaging
- Can generalize to new, unseen folds

---

### 2. Time-Limited Hyperparameter Search ⏱️

**Previous**: Fixed grid search with no time constraints.

**New**: Smart, adaptive search with:
- **3.5-minute time limit per fold** (`FOLD_TIME_LIMIT = 210` seconds)
- Early termination when time expires
- Progress tracking with elapsed time display

**Benefits**:
- Predictable runtime (max ~7-10 minutes for typical pre-test window)
- Prevents hanging on difficult folds
- Better resource utilization

---

### 3. Smart Hyperparameter Grid Generation 🎯

**Previous**: Simple, static grid enumeration.

**New**: `generate_smart_grid()` function with:
- **Random search** for first fold (explores full space)
- **Focused search** for subsequent folds (samples around prior best)
- Wider search ranges:
  - `learning_rate`: [0.01, 0.15]
  - `num_leaves`: [15, 127]
  - `max_depth`: [3, 10]
  - `min_child_samples`: [10, 120]
  - `lambda_l1`, `lambda_l2`: [0.0, 1.0]
- Adaptive sampling (100-150 candidates per fold)

**Benefits**:
- Explores more aggressively in early folds
- Exploits prior knowledge in later folds
- Better coverage of hyperparameter space
- More likely to find optimal settings

---

### 4. Comprehensive Terminal Progress Tracking 📊

**New**: Detailed, real-time debug output showing:

#### Per-Fold Header:
```
══════════════════════════════════════════════════════════════════════
✓ FOLD 1 | Train: 5 bars | Val: 30 bars
  Pos rate: 0.533 | Separation: 0.0234
  Time limit: 210s | Strategy: Random Search
──────────────────────────────────────────────────────────────────────
```

#### Per-Trial Progress:
- **Improvement events** (every time a new best is found):
  ```
  ✓ Trial  15/150 | Score: 0.7234 → NEW BEST | F1=0.7123, AUC=0.5845, θ=0.47 | lr=0.0532, leaves=47, depth=6
  ```

- **Progress checkpoints** (every 10 trials):
  ```
  · Trial  50/150 | Score: 0.6891 | Best: 0.7234 (last improved: trial 42)
  ```

#### Fold Summary:
```
  📊 Fold 1 Summary:
     Time: 156.3s | Trials: 98 | Improvements: 12
     Best Score: 0.7234 | F1: 0.7123 | AUC: 0.5845
     Threshold: 0.47
```

#### Meta-Learner Updates:
```
  🧠 Meta-learner updated with 2 folds
```

#### Final Meta-Inference:
```
══════════════════════════════════════════════════════════════════════
  META-LEARNING: INFERRING FINAL HYPERPARAMETERS
══════════════════════════════════════════════════════════════════════
✓ Using ML-based meta-learner (trained on 7 folds)
  Combined train signature: n=1255, pos_rate=0.542, sep=0.0189

  Inferred Hyperparameters:
    learning_rate: 0.0548
    num_leaves: 52
    max_depth: 6
    min_child_samples: 65
    feature_fraction: 0.8234
    bagging_fraction: 0.8567
    lambda_l1: 0.1234
    lambda_l2: 0.0987
══════════════════════════════════════════════════════════════════════
```

**Benefits**:
- Know exactly where you are in the training process
- See improvements happening in real-time
- Track time usage per fold
- Debug stalling or poor performance immediately
- Understand meta-learner behavior

---

### 5. Enhanced Fold Summary Output 📈

**Previous**: Basic fold results without process metrics.

**New**: Comprehensive fold summaries including:
- Number of trials attempted
- Time elapsed
- Number of improvements found
- Best hyperparameters discovered

Example:
```
  Fold 2: train=10, val=30, trials=125, time=198.2s | AUC=0.5621, F1=0.7234, Acc=0.6588, θ=0.49 | leaves=58, depth=7, lr=0.0621
```

---

### 6. Improved Data Characterization 🔍

**New**: Richer fold signature extraction:
- Feature mean averaging
- Feature std deviation averaging
- Class separation (distance between class-conditional means)
- Used by meta-learner to predict optimal hyperparameters

**Previous**: Only sample size and positive rate.

---

### 7. Iterative Meta-Learner Training 🔄

**New**: Meta-learner is retrained after each fold completion.
- Fold 1: Baseline random search
- Fold 2+: Meta-learner starts training and guiding subsequent searches
- Continuously improves as more folds are processed

**Benefits**:
- Adapts dynamically during the pre-test phase
- Later folds benefit from earlier fold learnings
- More robust final hyperparameter inference

---

## Usage

Run the script as before:
```bash
python "my-vite-project/src/components/v2 daygent models/nodes/LGBM_iso_cv_weighted_v1/train_three_window_meta.py"
```

### Configuration Options

Adjust these constants at the top of the script:

```python
GAP_BARS = 5                # Purge gap between train/val in each fold
PRETEST_FOLD_SIZE = 35      # Bars per fold (~35 days)
FOLD_TIME_LIMIT = 210       # 3.5 minutes per fold (in seconds)
```

Modify `generate_smart_grid()` to change:
- Search ranges for each hyperparameter
- Number of random samples (`n_samples`)
- Exploration vs exploitation balance

---

## What to Expect

### Runtime
- **Total time**: ~5-15 minutes depending on:
  - Number of pre-test folds (typically 5-8)
  - Time limit per fold (default 3.5 min)
  - Trials completed before timeout

### Output Quality
- More diverse hyperparameter exploration
- Better adaptation to fold characteristics
- Smoother convergence to optimal settings
- Rich terminal feedback for monitoring

### Artifacts Saved
All results are saved to `artifacts_lgbm_1d_iso_three_window/runs/<timestamp>/`:
- `lightgbm_three_window_final.joblib` - Final trained model
- `scaler_three_window.joblib` - Feature scaler
- `results_three_window.json` - Full results including:
  - Per-fold hyperparameters and metrics
  - Trials and time per fold
  - Meta-learner inferred parameters
  - Test window metrics
- `run_meta.json` - Run metadata
- `code_train_three_window_meta.py.txt` - Code snapshot

---

## Next Steps / Further Improvements

Potential enhancements you could add:
1. **Bayesian Optimization**: Replace random search with Optuna/hyperopt for smarter sampling
2. **Early Stopping**: Stop fold search if no improvement for N trials
3. **Ensemble Meta-Learner**: Use RandomForest or GradientBoosting instead of Ridge
4. **Cross-Fold Validation**: Validate meta-learner predictions on held-out folds
5. **GPU Acceleration**: Add `device_type='gpu'` to LightGBM params
6. **Adaptive Time Limits**: Allocate more time to promising folds
7. **Sharpe-Based Scoring**: Optimize for trading metrics instead of F1/AUC

---

## Comparison: Simple vs Advanced Meta-Learning

| Aspect | Simple (old) | Advanced (new) |
|--------|--------------|----------------|
| **Approach** | Weighted averaging | ML regression ensemble |
| **Features Used** | Only prior performance | Data characteristics + performance |
| **Adaptability** | Static weights | Learns relationships |
| **Hyperparameters** | 6 (lr, leaves, depth, min_child, feat_frac, bag_frac) | 8 (+ lambda_l1, lambda_l2) |
| **Robustness** | Assumes all folds equally relevant | Weights by data similarity |
| **Interpretability** | High | Medium (model-based) |

---

## Technical Details

### Meta-Feature Vector (8 dimensions):
1. Sample size (n)
2. Positive class rate
3. Feature mean average
4. Feature std deviation average
5. Feature separation (class distance)
6. Number of features
7. Prior F1 score
8. Prior AUC score

### Ridge Regression:
- L2 regularization with `alpha=1.0`
- Prevents overfitting on small fold counts
- Fast training (<1ms per fold)
- Stable predictions with clipping to valid ranges

---

## Troubleshooting

### If meta-learner doesn't train:
- Check that at least 2 folds have completed
- Verify hyperparameter variance > 1e-6
- Falls back to weighted averaging automatically

### If trials are slow:
- Reduce `n_samples` in `generate_smart_grid()`
- Decrease `n_estimators` in LightGBM params
- Increase `FOLD_TIME_LIMIT` for more thorough search

### If performance is poor:
- Check fold summaries for consistent improvements
- Verify AUC > 0.5 (better than random)
- Inspect inferred hyperparameters for reasonableness
- Consider expanding search ranges or trying different meta-features

---

## Summary

The enhanced script provides:
✅ **Advanced ML-based meta-learning** for smarter hyperparameter selection  
✅ **Time-limited search** with predictable runtimes  
✅ **Rich terminal output** showing progress, improvements, and diagnostics  
✅ **Adaptive grid generation** balancing exploration and exploitation  
✅ **Comprehensive artifacts** for post-training analysis  

This makes the three-window training process more robust, interpretable, and practical for production use.

