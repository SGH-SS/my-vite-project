#!/usr/bin/env python3
"""
Run this inside your RunPod worker.
• Scans /workspace/input_csvs, lets you pick a CSV,
• adds six vector columns (two raw, two z-scored, two SBERT),
• writes to /workspace/output_csvs/<name>_with_vectors.csv,
• logs every major step.
"""

#pip install numpy pandas torch sentence-transformers sentencepiece accelerate

import os, sys, time, logging
from typing import List

import numpy as np
import pandas as pd
import torch
import torch

# ───── Shim for PyTorch < 2.3  (adds torch.get_default_device) ─────
if not hasattr(torch, "get_default_device"):
    def _get_default_device():
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.get_default_device = _get_default_device
# --------------------------------------------------------------------

from sentence_transformers import SentenceTransformer

# ───────────────────────── CONFIG ──────────────────────────
MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 128
INPUT_DIR  = "/workspace/input_csvs"
OUTPUT_DIR = "/workspace/output_csvs"

VECTOR_COLUMNS: List[str] = [
    "raw_ohlc_vec", "raw_ohlcv_vec",
    "norm_ohlc", "norm_ohlcv",
    "BERT_ohlc", "BERT_ohlcv",
]

# ──────────────────────── LOGGING ──────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("vector-builder")

# ──────────────────────── HELPERS ──────────────────────────
def pick_file() -> str:
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")])
    if not files:
        log.error(f"No CSV files in {INPUT_DIR}.")
        sys.exit(1)
    log.info("Available CSVs:")
    for i, f in enumerate(files, 1):
        log.info(f"{i:2d}. {f}")
    while True:
        try:
            n = int(input(f"\nSelect file [1-{len(files)}]: ").strip())
            if 1 <= n <= len(files):
                return os.path.join(INPUT_DIR, files[n-1])
        except ValueError:
            pass
        print("Invalid choice, try again.")

def zscore(col: pd.Series) -> pd.Series:
    μ, σ = col.mean(), col.std(ddof=0)
    if σ == 0:
        return pd.Series(np.zeros(len(col)), index=col.index)
    return (col - μ) / σ, μ, σ  # we’ll discard μ,σ when used inline

# ──────────────────────── PIPELINE ─────────────────────────
def main():
    path_in = pick_file()

    # 1. Load
    log.info(f"Loading {path_in} …")
    df = pd.read_csv(path_in)
    log.info(f"{len(df):,} rows read.")

    # 2. Required columns
    must = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    miss = [c for c in must if c not in df.columns]
    if miss:
        log.error(f"Missing columns {miss}.")
        sys.exit(1)

    # 3. Timestamp to datetime for final summary
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # 4. SBERT model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading SBERT ({MODEL_NAME}) on {device} …")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # 5. Sentence crafting
    df["sent_ohlc"] = (
        df["symbol"] + " " + df["timestamp"].astype(str) +
        f" O:{df['open']} H:{df['high']} L:{df['low']} C:{df['close']}"
    )
    df["sent_ohlcv"] = (
        df["sent_ohlc"] + f" V:{df['volume']}"
    )

    # 6. Embeddings
    def embed(src, tgt):
        log.info(f"Embedding → {tgt}")
        t0 = time.time()
        v = model.encode(src.tolist(), batch_size=BATCH_SIZE, show_progress_bar=True)
        df[tgt] = v.tolist()
        log.info(f"{tgt}: {time.time()-t0:.1f}s")

    embed(df["sent_ohlc"],  "BERT_ohlc")
    embed(df["sent_ohlcv"], "BERT_ohlcv")

    # 7. Raw numeric vectors
    df["raw_ohlc_vec"]  = df[["open", "high", "low", "close"]].values.tolist()
    df["raw_ohlcv_vec"] = df[["open", "high", "low", "close", "volume"]].values.tolist()

    # 8. Z-score normalised vectors
    log.info("Computing Z-score normalised vectors …")
    price_cols = ["open", "high", "low", "close"]

    # price columns
    price_z = df[price_cols].apply(lambda col: (col - col.mean()) / col.std(ddof=0))
    df["norm_ohlc"] = price_z.values.tolist()

    # volume → log1p then z-score
    vol_log = np.log1p(df["volume"])
    vol_z = (vol_log - vol_log.mean()) / vol_log.std(ddof=0)

    df["norm_ohlcv"] = price_z.assign(volume=vol_z).values.tolist()

    # 9. Cleanup temp sentences
    df.drop(columns=["sent_ohlc", "sent_ohlcv"], inplace=True)

    # 10. Write
    base = os.path.basename(path_in).replace(".csv", "")
    path_out = os.path.join(OUTPUT_DIR, f"{base}_with_vectors.csv")
    df.to_csv(path_out, index=False)
    log.info(f"Saved → {path_out}")

    # 11. Summary
    log.info("── Summary ──")
    log.info(f"Rows  : {len(df):,}")
    log.info(f"First : {df['timestamp'].min()}")
    log.info(f"Last  : {df['timestamp'].max()}")
    log.info("Columns added: " + ", ".join(VECTOR_COLUMNS))
    log.info("Done ✅")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
