"""
Model inference service for loading pre-trained .joblib models and running predictions
for supported symbol/timeframe pairs. Produces probabilities and confidence metrics.

Supported (initial):
- SPY 1d → GradientBoosting (gb_1d_base.joblib)
- SPY 4h → GradientBoosting (gb_4h_base.joblib)
- SPY 4h → LightGBM_Financial (lgb_fin_4h_base.joblib)

Notes about scaling:
- Original training likely used StandardScaler for features before fitting tree models.
- We try to load an optional scaler if provided, otherwise fall back to no-scaling.
  Tree models are not scale-sensitive conceptually, but trained thresholds expect a
  consistent transform. This fallback is a pragmatic compromise.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Ensure classes are importable for joblib deserialization
from sklearn.ensemble import GradientBoostingClassifier  # noqa: F401
try:
    from lightgbm import LGBMClassifier  # noqa: F401
except Exception:  # pragma: no cover
    LGBMClassifier = None  # type: ignore

import joblib


TIMEFRAMES_SUPPORTED = {"1d", "4h"}
SUPPORTED_SYMBOL = "spy"


@dataclass
class PredictionResult:
    timestamp: str
    pred: int
    proba: float
    confidence: float
    is_train: Optional[bool]


class ModelInferenceService:
    """Singleton-like loader for .joblib models and prediction helpers."""

    def __init__(self) -> None:
        self._models: Dict[Tuple[str, str], Any] = {}
        self._scalers: Dict[Tuple[str, str], Any] = {}
        self._load_models()

    @staticmethod
    def _project_root() -> Path:
        # backend/services -> backend -> project root
        return Path(__file__).resolve().parents[2]

    def _models_dir(self) -> Path:
        # Models are currently stored under the frontend folder per user note
        return self._project_root() / "src" / "components" / "v1 daygent models"

    def _load_models(self) -> None:
        models_dir = self._models_dir()
        # Map (model_name, timeframe) -> filename
        path_map: Dict[Tuple[str, str], str] = {
            ("GradientBoosting", "1d"): "gb_1d_versionlock/gb_1d_final.joblib",
            ("GradientBoosting", "4h"): "gb_4h/gb_4h_w2_style.joblib",
            ("LightGBM_Financial", "4h"): "lgbm_4h/lightgbm_financial_4h_only.joblib",
        }
        for key, fname in path_map.items():
            fpath = models_dir / fname
            if fpath.exists():
                try:
                    self._models[key] = joblib.load(str(fpath))
                except Exception as e:  # pragma: no cover
                    # Leave missing on failure
                    self._models[key] = None
            else:
                self._models[key] = None

        # Optional scalers if present, look in specific model directories
        # Store scalers keyed by (scaler_name, timeframe) and also (scaler_name, timeframe, model_name)
        # so model-specific scalers can be preferred.
        generic_scalers = {
            ("StandardScaler", "1d"): models_dir / "gb_1d_versionlock" / "scaler_1d.joblib",
            ("StandardScaler", "4h"): models_dir / "gb_4h" / "scaler_4h_w2_style.joblib",
        }
        model_specific_scalers = {
            # LightGBM 4H specific scaler
            ("StandardScaler", "4h", "LightGBM_Financial"): models_dir / "lgbm_4h" / "scaler_4h_only.joblib",
        }

        # Load generic scalers
        for key, scaler_path in generic_scalers.items():
            if scaler_path.exists():
                try:
                    self._scalers[key] = joblib.load(str(scaler_path))
                except Exception:
                    pass

        # Load model-specific scalers
        for key, scaler_path in model_specific_scalers.items():
            if scaler_path.exists():
                try:
                    self._scalers[key] = joblib.load(str(scaler_path))
                except Exception:
                    pass

    def model_available(self, model_name: str, timeframe: str) -> bool:
        return self._models.get((model_name, timeframe)) is not None

    @staticmethod
    def _parse_vector(value: Any, expected_len: int) -> Optional[List[float]]:
        """Parse a vector field from DB/CSV into a fixed-length list of floats.

        Supports Python lists/tuples/ndarrays and string encodings like
        "[1,2,3]" or "{1,2,3}". Pads/truncates to expected_len.
        """
        if value is None:
            return None
        # Direct array-like
        if isinstance(value, (list, tuple, np.ndarray)):
            try:
                floats = [float(x) for x in list(value)]
            except Exception:
                return None
            if len(floats) < expected_len:
                floats = (floats + [0.0] * expected_len)[:expected_len]
            else:
                floats = floats[:expected_len]
            return floats
        # String-encoded vectors
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            # Strip common wrappers
            s = s.strip('[]{}()"\'')
            if not s:
                return None
            # Normalize delimiters and split
            parts = [p for p in s.replace(';', ',').split(',') if p.strip() != ""]
            try:
                floats = [float(p.strip()) for p in parts][:expected_len]
            except Exception:
                return None
            if len(floats) < expected_len:
                floats = (floats + [0.0] * expected_len)[:expected_len]
            return floats
        return None

    @staticmethod
    def _build_feature_vector(row: Dict[str, Any], timeframe: str) -> np.ndarray:
        """Build features consistent with training pipeline.

        Layout:
        [raw_o, raw_h, raw_l, raw_c, raw_v,
         iso_0, iso_1, iso_2, iso_3,
         one_hot_1d, one_hot_4h,
         hl_range, price_change, upper_shadow, lower_shadow, volume_m]
        """
        # Preferred: parse from vector columns as in training/notebook
        raw = ModelInferenceService._parse_vector(row.get("raw_ohlcv_vec"), 5)
        if raw is None:
            # Fallback to individual columns if vector missing
            o = float(row.get("open") or 0.0)
            h = float(row.get("high") or 0.0)
            l = float(row.get("low") or 0.0)
            c = float(row.get("close") or 0.0)
            v = float(row.get("volume") or 0.0)
            raw = [o, h, l, c, v]
        o, h, l, c, v = [float(x) for x in raw[:5]]

        # ISO vector
        iso = ModelInferenceService._parse_vector(row.get("iso_ohlc") or row.get("iso_ohlc_vec"), 4)
        if iso is None:
            iso = [0.0, 0.0, 0.0, 0.0]

        tf_one_hot = [1.0 if timeframe == "1d" else 0.0, 1.0 if timeframe == "4h" else 0.0]

        hl_range = (h - l) / c if c else 0.0
        price_change = (c - o) / o if o else 0.0
        upper_shadow = (h - c) / c if c else 0.0
        lower_shadow = (c - l) / c if c else 0.0
        volume_m = v / 1_000_000.0

        features = [o, h, l, c, v]
        features.extend(iso)
        features.extend(tf_one_hot)
        features.extend([hl_range, price_change, upper_shadow, lower_shadow, volume_m])

        return np.array(features, dtype=float)

    @staticmethod
    def _decision_margin_from_proba(p: float, threshold: float) -> float:
        # Decision margin: distance from threshold, normalized to [0,1]
        # |p - threshold| gives the margin distance
        return float(abs(p - threshold))
    
    @staticmethod
    def _get_model_threshold(model_name: str, timeframe: str) -> float:
        """Get the threshold used for this model/timeframe combination"""
        # Thresholds from the model training results
        threshold_map = {
            ("GradientBoosting", "1d"): 0.57,
            ("GradientBoosting", "4h"): 0.50,
            ("LightGBM_Financial", "4h"): 0.30,
        }
        return threshold_map.get((model_name, timeframe), 0.5)  # Default to 0.5 if not found

    def _maybe_scale(self, X: np.ndarray, timeframe: str, model_name: Optional[str] = None) -> np.ndarray:
        # Prefer model-specific scaler if available
        scaler = None
        if model_name is not None:
            scaler = self._scalers.get(("StandardScaler", timeframe, model_name))
        if scaler is None:
            scaler = self._scalers.get(("StandardScaler", timeframe))
        if scaler is None:
            return X
        try:
            return scaler.transform(X)
        except Exception:
            return X

    def predict_for_rows(
        self,
        model_name: str,
        timeframe: str,
        rows: List[Dict[str, Any]],
        train_cutoff: Optional[str] = None
    ) -> Tuple[List[PredictionResult], Dict[str, int]]:
        """
        Run predictions for given rows (ordered candles) and return per-candle results
        plus coverage counts (train vs inference vs unsupported).
        """
        model = self._models.get((model_name, timeframe))
        if model is None:
            raise ValueError(f"Model not available: {model_name} {timeframe}")

        # Build feature matrix
        X_list: List[np.ndarray] = []
        timestamps: List[str] = []
        for r in rows:
            X_list.append(self._build_feature_vector(r, timeframe))
            timestamps.append(str(r.get("timestamp")))
        X = np.vstack(X_list) if X_list else np.zeros((0, 16), dtype=float)

        # Optional scaling
        X_scaled = self._maybe_scale(X, timeframe, model_name)

        # Predict
        try:
            proba = model.predict_proba(X_scaled)[:, 1].astype(float).tolist()
        except Exception:
            # Some LGBM configs can use predict() with raw scores; fallback
            y_hat = model.predict(X_scaled)
            # If not probabilistic, map to {0,1} and set p as 0.5 +/- epsilon
            proba = [0.9 if int(v) == 1 else 0.1 for v in y_hat]

        # Get the threshold for this model/timeframe
        threshold = self._get_model_threshold(model_name, timeframe)
        
        preds = [1 if p >= threshold else 0 for p in proba]
        decision_margins = [self._decision_margin_from_proba(p, threshold) for p in proba]

        # Coverage
        coverage = {"train": 0, "inference": 0, "unsupported": 0}
        results: List[PredictionResult] = []
        cutoff = None
        if train_cutoff:
            try:
                cutoff = np.datetime64(train_cutoff)
            except Exception:
                cutoff = None

        for ts, y, p, dm in zip(timestamps, preds, proba, decision_margins):
            is_train: Optional[bool]
            if cutoff is None:
                is_train = None
            else:
                try:
                    is_train = np.datetime64(ts) <= cutoff
                except Exception:
                    is_train = None
            # Ensure JSON-serializable native bool, not numpy.bool_
            if is_train is not None:
                try:
                    is_train = bool(is_train)  # type: ignore[assignment]
                except Exception:
                    # Fallback in case of unexpected types
                    is_train = True if str(is_train).lower() == "true" else False
            if is_train is None:
                # Unknown, count as inference for display purposes
                coverage["inference"] += 1
            elif is_train:
                coverage["train"] += 1
            else:
                coverage["inference"] += 1

            results.append(PredictionResult(
                timestamp=ts,
                pred=int(y),
                proba=float(p),
                confidence=float(dm),  # Now using decision margin instead of old confidence calc
                is_train=is_train
            ))

        return results, coverage


# Module-level singleton
model_inference_service = ModelInferenceService()




