"""
CSV Model Prediction Service

Loads and serves predictions from the three pre-trained models:
- GB 1D: Gradient Boosting for SPY 1d timeframe  
- GB 4H: Gradient Boosting for SPY 4h timeframe
- LightGBM 4H: LightGBM Financial for SPY 4h timeframe

Each model has a CSV file with test predictions including:
timestamp_utc, pred_prob_up, pred_label, true_label, correct, threshold_used, decision_margin
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class CSVModelService:
    """Service for loading and serving CSV-based model predictions"""
    
    def __init__(self):
        self.models_data: Dict[str, pd.DataFrame] = {}
        self.model_configs = {
            'gb_1d': {
                'name': 'Gradient Boosting 1D',
                'symbol': 'spy',
                'timeframe': '1d',
                'csv_path': 'src/components/v1 daygent models/gb_1d/test_predictions_1d.csv',
                'threshold': 0.57,
                'description': 'Gradient Boosting model for SPY daily predictions'
            },
            'gb_4h': {
                'name': 'Gradient Boosting 4H', 
                'symbol': 'spy',
                'timeframe': '4h',
                'csv_path': 'src/components/v1 daygent models/gb_4h/test_predictions_4h.csv',
                'threshold': 0.50,
                'description': 'Gradient Boosting model for SPY 4-hour predictions'
            },
            'lgbm_4h': {
                'name': 'LightGBM Financial 4H',
                'symbol': 'spy', 
                'timeframe': '4h',
                'csv_path': 'src/components/v1 daygent models/lgbm_4h/test_predictions.csv',
                'threshold': 0.30,
                'description': 'LightGBM Financial model for SPY 4-hour predictions'
            }
        }
        self._load_all_models()
    
    def _project_root(self) -> Path:
        """Get project root directory (backend/services -> backend -> project root)"""
        return Path(__file__).resolve().parents[2]
    
    def _load_all_models(self):
        """Load all CSV model prediction files"""
        project_root = self._project_root()
        
        for model_key, config in self.model_configs.items():
            csv_path = project_root / config['csv_path']
            
            if csv_path.exists():
                try:
                    # Load CSV with proper parsing
                    df = pd.read_csv(csv_path)
                    
                    # Ensure timestamp column is properly parsed
                    if 'timestamp_utc' in df.columns:
                        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
                    
                    # Store the dataframe
                    self.models_data[model_key] = df
                    logger.info(f"✅ Loaded {model_key} model predictions: {len(df)} records from {csv_path}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load {model_key} predictions from {csv_path}: {e}")
                    self.models_data[model_key] = pd.DataFrame()
            else:
                logger.warning(f"⚠️ CSV file not found for {model_key}: {csv_path}")
                self.models_data[model_key] = pd.DataFrame()
    
    def get_available_models(self) -> List[Dict]:
        """Get list of available models with metadata"""
        available = []
        for model_key, config in self.model_configs.items():
            model_data = self.models_data.get(model_key, pd.DataFrame())
            available.append({
                'key': model_key,
                'name': config['name'],
                'symbol': config['symbol'],
                'timeframe': config['timeframe'],
                'threshold': config['threshold'],
                'description': config['description'],
                'available': not model_data.empty,
                'record_count': len(model_data) if not model_data.empty else 0,
                'date_range': {
                    'start': model_data['timestamp_utc'].min().isoformat() if not model_data.empty and 'timestamp_utc' in model_data.columns else None,
                    'end': model_data['timestamp_utc'].max().isoformat() if not model_data.empty and 'timestamp_utc' in model_data.columns else None
                } if not model_data.empty else None
            })
        return available
    
    def get_prediction_for_timestamp(self, model_key: str, timestamp: datetime) -> Optional[Dict]:
        """Get prediction for a specific timestamp from a model"""
        if model_key not in self.models_data:
            return None
            
        df = self.models_data[model_key]
        if df.empty:
            return None
        
        # Convert timestamp to pandas datetime for comparison
        target_timestamp = pd.to_datetime(timestamp)
        
        # Find exact match by timestamp
        matches = df[df['timestamp_utc'] == target_timestamp]
        
        if matches.empty:
            return None
        
        # Get the first match (should be unique)
        row = matches.iloc[0]
        config = self.model_configs[model_key]
        
        # Extract the required columns
        return {
            'model_key': model_key,
            'model_name': config['name'],
            'timestamp': row['timestamp_utc'].isoformat(),
            'pred_prob_up': float(row['pred_prob_up']),
            'pred_label': int(row['pred_label']),
            'true_label': int(row['true_label']),
            'correct': bool(row['correct']),
            'threshold_used': float(row['threshold_used']),
            'decision_margin': float(row['decision_margin']),
            # Additional useful fields
            'candle_index': int(row['candle_index_in_test']) if 'candle_index_in_test' in row else None,
            'date_utc': row['date_utc'] if 'date_utc' in row else None
        }
    
    def get_predictions_for_timeframe(self, symbol: str, timeframe: str) -> List[Dict]:
        """Get all available model predictions for a symbol/timeframe combination"""
        predictions = []
        
        for model_key, config in self.model_configs.items():
            if config['symbol'] == symbol and config['timeframe'] == timeframe:
                df = self.models_data.get(model_key, pd.DataFrame())
                if not df.empty:
                    # Convert entire dataframe to list of predictions
                    for _, row in df.iterrows():
                        prediction = {
                            'model_key': model_key,
                            'model_name': config['name'],
                            'timestamp': row['timestamp_utc'].isoformat(),
                            'pred_prob_up': float(row['pred_prob_up']),
                            'pred_label': int(row['pred_label']),
                            'true_label': int(row['true_label']),
                            'correct': bool(row['correct']),
                            'threshold_used': float(row['threshold_used']),
                            'decision_margin': float(row['decision_margin']),
                            'candle_index': int(row['candle_index_in_test']) if 'candle_index_in_test' in row else None,
                            'date_utc': row['date_utc'] if 'date_utc' in row else None
                        }
                        predictions.append(prediction)
        
        # Sort by timestamp
        predictions.sort(key=lambda x: x['timestamp'])
        return predictions
    
    def get_models_for_symbol_timeframe(self, symbol: str, timeframe: str) -> List[str]:
        """Get list of available model keys for a specific symbol/timeframe"""
        available_models = []
        for model_key, config in self.model_configs.items():
            if config['symbol'] == symbol and config['timeframe'] == timeframe:
                df = self.models_data.get(model_key, pd.DataFrame())
                if not df.empty:
                    available_models.append(model_key)
        return available_models
    
    def format_prediction_string(self, prediction: Dict) -> str:
        """Format prediction in the requested format: pred=__ p_up=__ thr=__ margin=__ truth=__ → ✅/❌"""
        if not prediction:
            return "No prediction available"
        
        pred_label = "UP" if prediction['pred_label'] == 1 else "DOWN"
        truth_label = "UP" if prediction['true_label'] == 1 else "DOWN"
        prob_pct = f"{prediction['pred_prob_up']:.4f}"
        threshold = f"{prediction['threshold_used']:.2f}"
        margin = f"{prediction['decision_margin']:.4f}"
        correct_symbol = "✅ CORRECT" if prediction['correct'] else "❌ WRONG"
        
        return f"pred={pred_label} p_up={prob_pct} thr={threshold} margin={margin} truth={truth_label} → {correct_symbol}"

# Create singleton instance
csv_model_service = CSVModelService()

