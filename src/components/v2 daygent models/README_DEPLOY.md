# Model Training (Local)

- Conda env name: daygent-train

- Create env:

conda env create -f "C:\Users\sham\Documents\agentic trading system\mcp\my-vite-project\src\components\v2 daygent models\envs\gb1d-environment.yml"

- Activate:

conda activate daygent-train

- Scripts live in:
C:\Users\sham\Documents\agentic trading system\mcp\my-vite-project\src\components\v2 daygent models

- Run via launcher (opens Anaconda Prompt):

run_training_script.bat train_lightgbm_1d_iso.py
run_training_script.bat train_lightgbm_1d_v0.py

Backend API: /api/training
- GET /scripts — list available train_*.py
- POST /run body { "scriptName": "train_lightgbm_1d_iso.py" }

Artifacts are saved in a sibling artifacts_* directory per script.
