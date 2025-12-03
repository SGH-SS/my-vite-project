@echo off
REM POC Training Launcher - Spawns PowerShell and runs Python training script

REM Arguments passed from Node/FastAPI:
REM %1 = selected_candle_date
REM %2 = train_years
REM %3 = test_window_size
REM %4 = learning_rate
REM %5 = num_leaves
REM %6 = max_depth
REM %7 = min_child_samples
REM %8 = output_dir

REM Navigate to v3 models directory
cd /d "C:\Users\sham\Documents\agentic trading system\mcp\my-vite-project\src\components\v3 models"

REM Run Python script with all arguments
python train_poc_simple.py --selected_candle_date "%~1" --train_years %2 --test_window_size %3 --learning_rate %4 --num_leaves %5 --max_depth %6 --min_child_samples %7 --output_dir "%~8"

REM Pause to keep window open so user can see results
echo.
echo ============================================================
echo Training complete! Press any key to close this window...
echo ============================================================
pause >nul

