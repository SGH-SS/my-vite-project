#!/usr/bin/env python3
"""
SPY Prediction Model v1
Multi-Timeframe Price Direction Prediction

Uses all 7 SPY timeframes (1m, 5m, 15m, 30m, 1h, 4h, 1d) with:
- raw_ohlcv vectors (5 dimensions)
- iso_ohlc vectors (4 dimensions) 
- future binary labels (price direction)

This model implements a hierarchical multi-timeframe approach
commonly used in professional trading systems.
"""

import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Concatenate, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'database': 'trading_db',
    'user': 'your_username',
    'password': 'your_password',
    'port': 5432
}

TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
SEQUENCE_LENGTH = 60  # For LSTM models

class SPYPredictionModel:
    def __init__(self, db_config):
        self.db_config = db_config
        self.engine = create_engine(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
        self.scalers = {}
        self.models = {}
        self.feature_columns = []
        
    def load_data_from_timeframe(self, timeframe, limit=None):
        """Load data from a specific timeframe table"""
        table_name = f"spy_{timeframe}"
        
        query = f"""
        SELECT 
            symbol,
            timestamp,
            raw_ohlcv_vec,
            iso_ohlc_vec,
            future
        FROM {table_name}
        WHERE raw_ohlcv_vec IS NOT NULL 
        AND iso_ohlc_vec IS NOT NULL 
        AND future IS NOT NULL
        ORDER BY timestamp
        """
        
        if limit:
            query += f" LIMIT {limit}"
            
        print(f"Loading data from {table_name}...")
        df = pd.read_sql_query(query, self.engine)
        df['timeframe'] = timeframe
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"Loaded {len(df)} rows from {table_name}")
        return df
    
    def parse_vector_column(self, vector_str):
        """Parse vector string to numpy array"""
        if pd.isna(vector_str) or vector_str is None:
            return None
        
        # Remove brackets and split by comma
        if isinstance(vector_str, str):
            vector_str = vector_str.strip('[]')
            return np.array([float(x.strip()) for x in vector_str.split(',')])
        return np.array(vector_str)
    
    def load_all_timeframes(self, limit_per_timeframe=10000):
        """Load and combine data from all timeframes"""
        all_data = []
        
        for timeframe in TIMEFRAMES:
            try:
                df = self.load_data_from_timeframe(timeframe, limit_per_timeframe)
                if len(df) > 0:
                    all_data.append(df)
            except Exception as e:
                print(f"Error loading {timeframe}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data loaded from any timeframe")
        
        # Combine all timeframes
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined dataset: {len(combined_df)} total rows")
        
        return combined_df
    
    def extract_features(self, df):
        """Extract features from raw_ohlcv and iso_ohlc vectors"""
        print("Extracting features from vectors...")
        
        features = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            try:
                # Parse raw_ohlcv vector (5 dimensions: O, H, L, C, V)
                raw_ohlcv = self.parse_vector_column(row['raw_ohlcv_vec'])
                if raw_ohlcv is None or len(raw_ohlcv) != 5:
                    continue
                
                # Parse iso_ohlc vector (4 dimensions: isolation forest features)
                iso_ohlc = self.parse_vector_column(row['iso_ohlc_vec'])
                if iso_ohlc is None or len(iso_ohlc) != 4:
                    continue
                
                # Create feature vector
                feature_vector = []
                
                # Raw OHLCV features
                feature_vector.extend(raw_ohlcv)  # 5 features
                
                # ISO OHLC features
                feature_vector.extend(iso_ohlc)   # 4 features
                
                # Timeframe encoding (one-hot)
                timeframe_encoding = [0] * len(TIMEFRAMES)
                tf_idx = TIMEFRAMES.index(row['timeframe'])
                timeframe_encoding[tf_idx] = 1
                feature_vector.extend(timeframe_encoding)  # 7 features
                
                # Technical indicators from raw OHLCV
                o, h, l, c, v = raw_ohlcv
                
                # Price-based features
                feature_vector.extend([
                    (h - l) / c if c != 0 else 0,  # High-Low range normalized
                    (c - o) / o if o != 0 else 0,  # Price change percentage
                    (h - c) / c if c != 0 else 0,  # Upper shadow
                    (c - l) / c if c != 0 else 0,  # Lower shadow
                    v / 1000000,  # Volume in millions
                ])  # 5 additional features
                
                features.append(feature_vector)
                valid_indices.append(idx)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        # Convert to numpy array
        X = np.array(features)
        
        # Create feature names
        feature_names = (
            ['raw_o', 'raw_h', 'raw_l', 'raw_c', 'raw_v'] +  # Raw OHLCV
            ['iso_0', 'iso_1', 'iso_2', 'iso_3'] +           # ISO features
            [f'tf_{tf}' for tf in TIMEFRAMES] +              # Timeframe encoding
            ['hl_range', 'price_change', 'upper_shadow', 'lower_shadow', 'volume_m']  # Technical indicators
        )
        
        self.feature_columns = feature_names
        
        # Get corresponding labels
        y = df.iloc[valid_indices]['future'].values
        timestamps = df.iloc[valid_indices]['timestamp'].values
        timeframes = df.iloc[valid_indices]['timeframe'].values
        
        print(f"Extracted {X.shape[0]} samples with {X.shape[1]} features")
        print(f"Feature names: {feature_names}")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X, y, timestamps, timeframes
    
    def create_sequences_for_lstm(self, X, y, timestamps, timeframes, sequence_length=60):
        """Create sequences for LSTM model (time series approach)"""
        print(f"Creating sequences with length {sequence_length}...")
        
        # Group by timeframe and create sequences
        sequences_X = []
        sequences_y = []
        sequences_timestamps = []
        
        for tf in TIMEFRAMES:
            tf_mask = timeframes == tf
            if np.sum(tf_mask) < sequence_length:
                print(f"Not enough data for {tf} timeframe ({np.sum(tf_mask)} samples)")
                continue
            
            tf_X = X[tf_mask]
            tf_y = y[tf_mask]
            tf_timestamps = timestamps[tf_mask]
            
            # Sort by timestamp
            sort_idx = np.argsort(tf_timestamps)
            tf_X = tf_X[sort_idx]
            tf_y = tf_y[sort_idx]
            tf_timestamps = tf_timestamps[sort_idx]
            
            # Create sequences
            for i in range(len(tf_X) - sequence_length):
                sequences_X.append(tf_X[i:i+sequence_length])
                sequences_y.append(tf_y[i+sequence_length])
                sequences_timestamps.append(tf_timestamps[i+sequence_length])
        
        if not sequences_X:
            raise ValueError("No sequences created. Check sequence_length parameter.")
        
        X_seq = np.array(sequences_X)
        y_seq = np.array(sequences_y)
        timestamps_seq = np.array(sequences_timestamps)
        
        print(f"Created {X_seq.shape[0]} sequences of shape {X_seq.shape[1:]}")
        return X_seq, y_seq, timestamps_seq
    
    def train_traditional_models(self, X_train, X_test, y_train, y_test):
        """Train traditional ML models"""
        print("\n=== Training Traditional ML Models ===")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['traditional'] = scaler
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=class_weights[1]/class_weights[0],
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = model.score(X_test_scaled, y_test)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        self.models['traditional'] = results
        return results
    
    def create_lstm_model(self, input_shape, num_features):
        """Create LSTM model for time series prediction"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_lstm_model(self, X_train_seq, X_test_seq, y_train_seq, y_test_seq):
        """Train LSTM model"""
        print("\n=== Training LSTM Model ===")
        
        # Scale sequences
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1]))
        X_train_scaled = X_train_scaled.reshape(X_train_seq.shape)
        
        X_test_scaled = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1]))
        X_test_scaled = X_test_scaled.reshape(X_test_seq.shape)
        
        self.scalers['lstm'] = scaler
        
        # Create model
        model = self.create_lstm_model(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            num_features=X_train_scaled.shape[2]
        )
        
        print(f"LSTM Model Architecture:")
        model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_seq), y=y_train_seq)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train_seq,
            validation_data=(X_test_scaled, y_test_seq),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Evaluate model
        y_pred_proba = model.predict(X_test_scaled).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = np.mean(y_pred == y_test_seq)
        auc = roc_auc_score(y_test_seq, y_pred_proba)
        
        print(f"LSTM - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        print(f"Classification Report:\n{classification_report(y_test_seq, y_pred)}")
        
        self.models['lstm'] = {
            'model': model,
            'history': history,
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return model, history
    
    def plot_results(self):
        """Plot training results and model comparisons"""
        print("\n=== Plotting Results ===")
        
        # Traditional models comparison
        if 'traditional' in self.models:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            models = self.models['traditional']
            model_names = list(models.keys())
            accuracies = [models[name]['accuracy'] for name in model_names]
            aucs = [models[name]['auc'] for name in model_names]
            
            # Accuracy comparison
            axes[0, 0].bar(model_names, accuracies, color='skyblue')
            axes[0, 0].set_title('Model Accuracy Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # AUC comparison
            axes[0, 1].bar(model_names, aucs, color='lightcoral')
            axes[0, 1].set_title('Model AUC Comparison')
            axes[0, 1].set_ylabel('AUC Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Feature importance (Random Forest)
            if 'RandomForest' in models:
                rf_model = models['RandomForest']['model']
                feature_importance = rf_model.feature_importances_
                sorted_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
                
                axes[1, 0].barh(range(len(sorted_idx)), feature_importance[sorted_idx])
                axes[1, 0].set_yticks(range(len(sorted_idx)))
                axes[1, 0].set_yticklabels([self.feature_columns[i] for i in sorted_idx])
                axes[1, 0].set_title('Top 10 Feature Importance (Random Forest)')
                axes[1, 0].set_xlabel('Importance')
            
            # Confusion matrix for best model
            best_model_name = max(models.keys(), key=lambda x: models[x]['auc'])
            best_predictions = models[best_model_name]['predictions']
            
            # You'll need to pass y_test to this function or store it as instance variable
            # For now, let's skip this plot
            axes[1, 1].text(0.5, 0.5, f'Best Model: {best_model_name}\nAUC: {models[best_model_name]["auc"]:.4f}', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Best Model Summary')
            
            plt.tight_layout()
            plt.savefig('spy_prediction_v1_results.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # LSTM training history
        if 'lstm' in self.models:
            history = self.models['lstm']['history']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss
            axes[0, 0].plot(history.history['loss'], label='Training Loss')
            axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            
            # Accuracy
            axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
            axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axes[0, 1].set_title('Model Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            
            # Precision
            axes[1, 0].plot(history.history['precision'], label='Training Precision')
            axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            
            # Recall
            axes[1, 1].plot(history.history['recall'], label='Training Recall')
            axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig('spy_prediction_v1_lstm.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def run_full_pipeline(self, limit_per_timeframe=10000):
        """Run the complete training pipeline"""
        print("=== SPY Prediction Model v1 ===")
        print("Multi-Timeframe Price Direction Prediction")
        print("=" * 50)
        
        # 1. Load data
        df = self.load_all_timeframes(limit_per_timeframe)
        
        # 2. Extract features
        X, y, timestamps, timeframes = self.extract_features(df)
        
        # 3. Split data (time series split)
        # Use last 20% for testing to avoid lookahead bias
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        timestamps_train = timestamps[:split_idx]
        timestamps_test = timestamps[split_idx:]
        timeframes_train = timeframes[:split_idx]
        timeframes_test = timeframes[split_idx:]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # 4. Train traditional models
        traditional_results = self.train_traditional_models(X_train, X_test, y_train, y_test)
        
        # 5. Create sequences and train LSTM
        try:
            X_train_seq, y_train_seq, _ = self.create_sequences_for_lstm(
                X_train, y_train, timestamps_train, timeframes_train
            )
            X_test_seq, y_test_seq, _ = self.create_sequences_for_lstm(
                X_test, y_test, timestamps_test, timeframes_test
            )
            
            lstm_model, lstm_history = self.train_lstm_model(
                X_train_seq, X_test_seq, y_train_seq, y_test_seq
            )
        except Exception as e:
            print(f"LSTM training failed: {e}")
        
        # 6. Plot results
        self.plot_results()
        
        # 7. Summary
        print("\n=== Model Performance Summary ===")
        if 'traditional' in self.models:
            for name, results in self.models['traditional'].items():
                print(f"{name}: Accuracy={results['accuracy']:.4f}, AUC={results['auc']:.4f}")
        
        if 'lstm' in self.models:
            lstm_results = self.models['lstm']
            print(f"LSTM: Accuracy={lstm_results['accuracy']:.4f}, AUC={lstm_results['auc']:.4f}")
        
        return self.models

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = SPYPredictionModel(DATABASE_CONFIG)
    
    # Run full pipeline
    results = model.run_full_pipeline(limit_per_timeframe=5000)  # Adjust limit as needed
    
    print("\n=== Training Complete ===")
    print("Check the generated plots for detailed results!") 