# Daygent | Agentic Trading Intelligence Platform

**A high-performance, monolithic trading system for quantitative analysis, vector-based pattern recognition, and on-demand AI model training.**

## 🚀 Executive Summary
Daygent is a sophisticated **full-stack quantitative trading platform** designed to bridge the gap between traditional technical analysis and modern AI. It features a **monolithic architecture** integrating six specialized dashboards for end-to-end workflow: from raw data ingestion and vector embedding generation to backtesting and live model inference. 

Built for speed and scalability, the system leverages **GPU-accelerated vector processing**, **PostgreSQL** for massive dataset management, and a **React 19 frontend** capable of rendering complex financial data at 60fps.

---

## 🛠️ Tech Stack
*   **Frontend:** React 19, Vite, TailwindCSS, Lightweight Charts 5.0 (Canvas rendering), Framer Motion.
*   **Backend:** Python 3.10+, FastAPI (Async), SQLAlchemy 2.0, Pydantic 2.5.
*   **Database:** PostgreSQL (Time-series optimized), Alembic for migrations.
*   **Machine Learning:** LightGBM, PyTorch (BERT embeddings), Scikit-learn (Isolation Forests), NumPy/Pandas.
*   **DevOps/Infrastucture:** GPU Acceleration (CUDA), Windows Native Pipeline Orchestration (.bat/PowerShell integration).

---

## 🌟 Key Differentiators

### 1. 🧪 Interactive AI Training ("Train Mode")
**Revolutionary Feature:** Train production-grade Machine Learning models directly from the UI.
*   **On-Demand Training:** Click any candle on the chart to define a "cutoff point." The system automatically slices historical data (e.g., past 3 years) to train a **LightGBM** model specifically on that market regime.
*   **Seamless Integration:** Configuration modal allows fine-tuning of hyperparameters (learning rate, leaves, depth) without touching code.
*   **Instant Feedback:** Models are trained in background processes, with results and accuracy metrics (ROC-AUC, F1) piped back to the dashboard immediately.

### 2. 🧠 Vector Intelligence Engine
Moves beyond price-action by converting market data into high-dimensional vectors.
*   **Multi-Modal Embeddings:** Generates 6 distinct vector types per candle, including **BERT Semantic Vectors** (translating price action to language) and **Isolation Forest (ISO)** anomaly scores.
*   **Shape Similarity Search:** mathematically compares current market structure against 10+ years of history to find statistically similar setups using Manhattan, Euclidean, and Cosine distance metrics.

### 3. 🔄 Hybrid Backtest Engine
*   **Smart Playback:** Seamlessly stitches "Backtest" (historical) and "Fronttest" (live pipeline) data into a single continuous stream.
*   **Visual Verification:** "Happened vs. Coming" visualization lets you replay market history candle-by-candle to validate model predictions without look-ahead bias.
*   **Live Inference:** Overlays real-time probabilities from Gradient Boosting and LSTM models directly onto the chart.

---

## 📊 Dashboard Architecture

The platform is divided into six integrated environments, all sharing a global `TradingContext` for state management:

| Dashboard | Functionality |
| :--- | :--- |
| **1. Data Dashboard** | **ETL & Analytics Center.** Advanced filtering, regex search, and SQL-backed pagination for millions of rows across 21 trading tables. |
| **2. Vector Dashboard** | **Pattern Recognition Core.** Heatmap visualizations of BERT/ISO vectors and similarity matrix computation. |
| **3. Chart Dashboard** | **Professional Technical Analysis.** High-performance canvas charting with multi-select, range-select, and synchronized cross-dashboard highlighting. |
| **4. Backtest Dashboard** | **Strategy Validation.** The flagship environment for replaying history, training LightGBM models on-the-fly, and auditing algo performance. |
| **5. Pipeline Dashboard** | **Orchestration.** Windows-native launcher for data pipelines (scrapers, feature engineering scripts) running in detached consoles. |
| **6. LLM Dashboard** | **Agentic Interface.** Chat framework ready for GPT-4/Claude integration to "talk" to the market data and perform qualitative analysis. |

---

## 📈 Quantitative Features
*   **TJR & Swing Liquidity Markers:** Algorithmic detection of internal (TJR) and external (Swing) liquidity pools, essential for institutional order flow analysis.
*   **Binary Classification Labels:** Automated labeling pipeline (`1`=Bullish, `0`=Bearish) for supervised learning tasks.
*   **Fair Value Gaps (FVG):** Automated detection of market inefficiencies and gap balances.

## 🔧 Setup & Installation

### Prerequisites
*   Node.js 18+ & Python 3.10+
*   PostgreSQL Database
*   NVIDIA GPU (Optional, for BERT acceleration)

### Quick Start
```bash
# 1. Backend (FastAPI)
cd backend
pip install -r requirements.txt
python main.py

# 2. Frontend (React)
cd my-vite-project
npm install
npm run dev
```

---

*Designed and Engineered by [Your Name]*
