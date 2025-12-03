# POC Training System - Interactive Candle-Based Model Training

## Overview

This POC (Proof of Concept) system enables on-demand LightGBM model training directly from the Backtest Dashboard. Click any candle on the chart to train a custom model using that candle as the end of the training period.

## System Components

### 1. Training Script
**File:** `train_poc_simple.py`

A simplified training script that:
- Loads SPY 1D data from PostgreSQL database
- Trains on a configurable period before the selected candle (default: 3 years)
- Tests on a configurable window after the selected candle (default: 35 candles)
- Uses 4-parameter LightGBM model (learning_rate, num_leaves, max_depth, min_child_samples)
- Saves model, scaler, and results to timestamped folders

**CLI Arguments:**
```bash
python train_poc_simple.py \
  --selected_candle_date "2023-06-15T00:00:00Z" \
  --train_years 3.0 \
  --test_window_size 35 \
  --learning_rate 0.05 \
  --num_leaves 31 \
  --max_depth 6 \
  --min_child_samples 20 \
  --output_dir "./on_demand_runs/20231215_143022"
```

### 2. Launcher Script
**File:** `run_poc_training.bat`

Spawns a PowerShell window and executes the training script with provided arguments.

### 3. Backend API Endpoints

#### POST `/api/training/run-poc`
Triggers POC training for a selected candle.

**Request Body:**
```json
{
  "candle_date": "2023-06-15T00:00:00Z",
  "train_years": 3.0,
  "test_window_size": 35,
  "params": {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20
  }
}
```

**Response:**
```json
{
  "started": true,
  "run_id": "20231215_143022",
  "pid": 12345,
  "message": "POC training started for candle 2023-06-15T00:00:00Z",
  "output_dir": "C:\\...\\on_demand_runs\\20231215_143022"
}
```

#### GET `/api/training/poc-runs`
Lists all POC training runs (completed, running, or failed).

**Response:**
```json
[
  {
    "run_id": "20231215_143022",
    "timestamp": "20231215_143022",
    "candle_date": "2023-06-15T00:00:00Z",
    "status": "completed",
    "train_metrics": { "accuracy": 0.65, "f1": 0.62, "auc": 0.71, "mcc": 0.35 },
    "test_metrics": { "accuracy": 0.58, "f1": 0.54, "auc": 0.63, "mcc": 0.22 },
    "config": { ... },
    "artifact_paths": { "model": "model_poc_...joblib", "scaler": "scaler_poc_...joblib" }
  }
]
```

#### GET `/api/training/poc-runs/{run_id}`
Gets details for a specific POC run.

#### GET `/api/training/latest-poc-run`
Gets the most recent POC run (used for polling after training starts).

#### POST `/api/poc-predictions/{run_id}`
Runs predictions using a trained POC model.

**Query Parameters:**
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)
- `threshold`: Classification threshold (default: 0.5)

**Response:**
```json
{
  "run_id": "20231215_143022",
  "symbol": "SPY",
  "timeframe": "1d",
  "threshold": 0.5,
  "predictions": [
    {
      "timestamp": "2023-06-16T00:00:00Z",
      "prediction": 1,
      "probability": 0.67,
      "confidence": 0.17
    }
  ],
  "total_predictions": 25
}
```

### 4. Frontend Components

#### TrainingConfigModal.jsx
Modal popup that appears when clicking a candle in train mode.

**Features:**
- Displays selected candle date
- Configurable training period (years)
- Configurable test window size
- Parameter sliders for 4 LightGBM params
- Training progress indicator
- Results display with train/test metrics
- "Load This Model" button (for future integration)

#### Backtest Dashboard Integration
Modified `backtest.jsx` to add:
- "Train Mode" toggle button in chart header
- Click overlay when train mode is active
- Opens training modal on candle click
- Visual indicator showing train mode is active

## Usage Flow

### Step 1: Activate Train Mode
1. Navigate to Backtest Dashboard
2. Click the "🎯 Train Mode" button in the Price Chart header
3. Button turns green with "Train Mode: ON" text
4. Chart cursor changes to crosshair

### Step 2: Select a Candle
1. Click any candle on the chart
2. Training Configuration Modal opens
3. Modal shows:
   - Selected candle date/time
   - Training period (default: 3 years before candle)
   - Test window (default: 35 candles after)
   - LightGBM parameter sliders

### Step 3: Configure Training
Adjust settings as needed:
- **Training Period:** How many years of data before the selected candle (0.5 - 10 years)
- **Test Window Size:** How many candles after selected candle to test on (10 - 100)
- **Learning Rate:** 0.01 - 0.2 (default: 0.05)
- **Num Leaves:** 15 - 63 (default: 31)
- **Max Depth:** 3 - 12 (default: 6)
- **Min Child Samples:** 10 - 50 (default: 20)

### Step 4: Start Training
1. Click "Start Training" button
2. PowerShell window opens showing training progress
3. Modal shows "Training in progress..." with spinner
4. Frontend polls backend every 5 seconds for completion

### Step 5: View Results
When training completes:
- Modal updates to show "Training Complete!"
- Displays train and test metrics:
  - Accuracy
  - F1 Score
  - AUC
  - MCC (Matthews Correlation Coefficient)
  - Precision, Recall
- Shows data split information
- "Load This Model" button appears

### Step 6: Use Trained Model
(Future implementation)
- Click "Load This Model" to integrate into predictions
- Model becomes available in Live Predictions panel
- Navigate chart to see predictions from your custom model

## File Structure

```
v3 models/
├── train_poc_simple.py          # POC training script
├── run_poc_training.bat          # Launcher script
├── README_POC.md                 # This file
└── on_demand_runs/               # Training outputs
    ├── 20231215_143022/          # Example run folder
    │   ├── model_poc_20231215_143022.joblib
    │   ├── scaler_poc_20231215_143022.joblib
    │   ├── results_poc_20231215_143022.json
    │   └── run_meta_20231215_143022.json
    └── 20231216_091045/          # Another run
        └── ...
```

## Training Output Files

Each training run creates a timestamped folder with:

### model_poc_{timestamp}.joblib
Trained LightGBM classifier model

### scaler_poc_{timestamp}.joblib
StandardScaler fitted on training data

### results_poc_{timestamp}.json
Complete training results including:
- Configuration (selected candle, periods, params)
- Data splits (sizes, date ranges, class distributions)
- Train metrics (accuracy, F1, AUC, MCC, best threshold)
- Test metrics (all metrics using train threshold)
- Feature names used

### run_meta_{timestamp}.json
Metadata about the run:
- Script name
- Run timestamp
- Run ID
- Selected candle date
- Artifact paths
- Status

## Technical Details

### Data Source
- **Database:** PostgreSQL at `localhost:5433`
- **Tables:** `backtest.spy_1d` and `fronttest.spy_1d` (deduplicated)
- **Features:** 16-dimensional feature vector (basic set without MA5 extensions)

### Feature Engineering
Follows the standard feature building from `omg.py`:
- Raw OHLCV (5 features)
- ISO features (4 features)
- Timeframe one-hot (2 features: [1,0] for 1D)
- Derived features (5 features: HL range, price change, shadows, volume_m)

### Model Architecture
- **Algorithm:** LightGBM Gradient Boosting
- **Parameters:** 4-param configuration (matching `3.py`)
- **Early Stopping:** 150 rounds on eval set
- **Max Iterations:** 4000
- **Evaluation Metric:** AUC

### Threshold Selection
- Train: Best threshold found by maximizing F1 score on training data
- Test: Uses the train threshold (no peeking at test labels)

## Differences from Full Training Scripts

This POC simplifies training by removing:
- ❌ Pre-test fold cross-validation
- ❌ Optuna hyperparameter optimization
- ❌ Meta-predictors (basic, advanced, MLP, tree ensembles)
- ❌ Multiple test windows
- ❌ Gap windows
- ❌ Extended feature sets (MA5 features, raw_o_next)
- ❌ Sabotage filters

POC focuses on:
- ✅ Simple train/test split
- ✅ Fixed user-selected parameters
- ✅ Direct model training with early stopping
- ✅ Fast turnaround (< 2 minutes typical)
- ✅ Easy integration and iteration

## Next Steps / Future Enhancements

### Phase 2 Enhancements
1. **Model Loading Integration**
   - Implement "Load This Model" button functionality
   - Add POC models to Live Predictions dropdown
   - Auto-refresh predictions when new model loaded

2. **Quick Optimize Button**
   - Add optional short Optuna study (30-60 seconds)
   - Optimize params before training final model
   - Show optimization progress

3. **Extended Features**
   - Option to use MA5 features (3 additional features)
   - Option to use raw_o_next
   - Feature selection checkboxes in modal

### Phase 3 Advanced Features
1. **Pre-test Fold Validation**
   - Add optional cross-validation
   - Show stability metrics across folds

2. **Meta-Predictor Integration**
   - Load pre-trained meta-learners
   - Use meta-predicted params as starting point
   - Compare meta vs user-selected params

3. **Multiple Test Windows**
   - Train on multiple test windows
   - Show per-window performance
   - Ensemble predictions

4. **Batch Training**
   - Select multiple candles for batch training
   - Compare results across different split points
   - Find optimal training periods

## Troubleshooting

### Training doesn't start
- **Check:** Is PostgreSQL running on port 5433?
- **Check:** Does selected candle have enough prior data (3 years by default)?
- **Fix:** Try shorter training period or different candle

### PowerShell window closes immediately
- **Check:** Is Python in system PATH?
- **Check:** Are required packages installed (pandas, numpy, lightgbm, scikit-learn, sqlalchemy)?
- **Fix:** Open Anaconda Prompt and verify packages

### Modal shows "Training timeout"
- **Check:** PowerShell window for actual errors
- **Check:** Database connectivity
- **Check:** Training might need more time (current timeout: 10 minutes)

### No test data available
- **Issue:** Selected candle is too recent (not enough candles after it)
- **Fix:** Select an earlier candle with at least 35 candles after it

## Development Notes

### Testing Checklist
- [ ] Backend starts without errors
- [ ] Frontend loads backtest dashboard
- [ ] Train mode toggle works
- [ ] Clicking candle opens modal
- [ ] Modal displays correct candle info
- [ ] Parameter sliders work
- [ ] Training starts and PowerShell opens
- [ ] Training completes and results appear
- [ ] Artifacts saved in on_demand_runs folder
- [ ] API endpoints return correct data

### Known Limitations (POC Phase)
- Only supports SPY 1D (hardcoded)
- No model comparison view
- "Load This Model" button placeholder (not functional yet)
- No prediction visualization on chart (yet)
- No training history management (delete old runs)
- No training cancellation

These will be addressed in future phases.

