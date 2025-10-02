"""
LSTM Model for Predictive Maintenance
Time-series forecasting using TensorFlow/Keras
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
from typing import Tuple, Dict, List, Optional
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMPredictor:
    """LSTM-based predictive maintenance model"""
    
    def __init__(self, sequence_length: int = 24, prediction_horizon: int = 6):
        """
        Initialize LSTM predictor
        
        Args:
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of time steps to predict ahead
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
        
    def prepare_sequences(self, data: pd.DataFrame, 
                         feature_columns: List[str],
                         target_column: str = 'is_failure') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training
        
        Args:
            data: Input dataframe
            feature_columns: List of feature column names
            target_column: Target column name
            
        Returns:
            X: Feature sequences, y: Target sequences
        """
        # Sort by timestamp and equipment_id
        data_sorted = data.sort_values(['equipment_id', 'timestamp']).copy()
        
        X_sequences = []
        y_sequences = []
        
        # Group by equipment to maintain temporal continuity
        for equipment_id in data_sorted['equipment_id'].unique():
            equipment_data = data_sorted[data_sorted['equipment_id'] == equipment_id]
            
            if len(equipment_data) < self.sequence_length + self.prediction_horizon:
                continue
                
            features = equipment_data[feature_columns].values
            targets = equipment_data[target_column].values
            
            # Create sequences
            for i in range(len(features) - self.sequence_length - self.prediction_horizon + 1):
                X_seq = features[i:i + self.sequence_length]
                y_seq = targets[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
                
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def preprocess_data(self, data: pd.DataFrame, 
                       feature_columns: List[str],
                       fit_scaler: bool = True) -> pd.DataFrame:
        """
        Preprocess data for LSTM model
        
        Args:
            data: Input dataframe
            feature_columns: List of feature columns
            fit_scaler: Whether to fit the scaler (True for training data)
            
        Returns:
            Preprocessed dataframe
        """
        data_processed = data.copy()
        
        if fit_scaler:
            self.scaler = StandardScaler()
            data_processed[feature_columns] = self.scaler.fit_transform(data[feature_columns])
            self.feature_columns = feature_columns
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit_scaler=True first.")
            data_processed[feature_columns] = self.scaler.transform(data[feature_columns])
        
        return data_processed
    
    def build_model(self, input_shape: Tuple[int, int], 
                   lstm_units: List[int] = [64, 32],
                   dropout_rate: float = 0.2,
                   learning_rate: float = 0.001) -> Model:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=input_shape))
        
        # LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            model.add(LSTM(units, return_sequences=return_sequences))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Output layer for binary classification (failure prediction)
        model.add(Dense(self.prediction_horizon, activation='sigmoid'))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train(self, train_data: pd.DataFrame,
              val_data: pd.DataFrame,
              feature_columns: List[str],
              target_column: str = 'is_failure',
              epochs: int = 100,
              batch_size: int = 32,
              model_save_path: str = None) -> Dict:
        """
        Train the LSTM model
        
        Args:
            train_data: Training dataframe
            val_data: Validation dataframe
            feature_columns: List of feature columns
            target_column: Target column name
            epochs: Number of training epochs
            batch_size: Training batch size
            model_save_path: Path to save the trained model
            
        Returns:
            Training history
        """
        logger.info("Preprocessing training data...")
        train_processed = self.preprocess_data(train_data, feature_columns, fit_scaler=True)
        val_processed = self.preprocess_data(val_data, feature_columns, fit_scaler=False)
        
        logger.info("Preparing sequences...")
        X_train, y_train = self.prepare_sequences(train_processed, feature_columns, target_column)
        X_val, y_val = self.prepare_sequences(val_processed, feature_columns, target_column)
        
        logger.info(f"Training sequences shape: {X_train.shape}")
        logger.info(f"Training targets shape: {y_train.shape}")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        logger.info("Model architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        if model_save_path:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
            )
        
        # Train model
        logger.info("Starting training...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        logger.info("Training completed!")
        
        return history.history
    
    def predict(self, data: pd.DataFrame, 
                feature_columns: List[str] = None) -> np.ndarray:
        """
        Make predictions using trained model
        
        Args:
            data: Input dataframe
            feature_columns: List of feature columns (uses training features if None)
            
        Returns:
            Predictions array
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if feature_columns is None:
            feature_columns = self.feature_columns
        
        # Preprocess data
        data_processed = self.preprocess_data(data, feature_columns, fit_scaler=False)
        
        # Prepare sequences
        X, _ = self.prepare_sequences(data_processed, feature_columns, 'is_failure')
        
        if len(X) == 0:
            return np.array([])
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_failure_probability(self, data: pd.DataFrame,
                                  feature_columns: List[str] = None,
                                  threshold: float = 0.5) -> pd.DataFrame:
        """
        Predict failure probability for equipment
        
        Args:
            data: Input dataframe
            feature_columns: List of feature columns
            threshold: Threshold for binary classification
            
        Returns:
            Dataframe with predictions
        """
        predictions = self.predict(data, feature_columns)
        
        if len(predictions) == 0:
            return pd.DataFrame()
        
        # Create results dataframe
        results = []
        data_sorted = data.sort_values(['equipment_id', 'timestamp'])
        
        pred_idx = 0
        for equipment_id in data_sorted['equipment_id'].unique():
            equipment_data = data_sorted[data_sorted['equipment_id'] == equipment_id]
            
            if len(equipment_data) < self.sequence_length + self.prediction_horizon:
                continue
            
            for i in range(len(equipment_data) - self.sequence_length - self.prediction_horizon + 1):
                if pred_idx >= len(predictions):
                    break
                    
                timestamp = equipment_data.iloc[i + self.sequence_length]['timestamp']
                
                # Average probability across prediction horizon
                avg_prob = np.mean(predictions[pred_idx])
                max_prob = np.max(predictions[pred_idx])
                
                results.append({
                    'timestamp': timestamp,
                    'equipment_id': equipment_id,
                    'failure_probability': avg_prob,
                    'max_failure_probability': max_prob,
                    'predicted_failure': avg_prob > threshold,
                    'prediction_horizon_hours': self.prediction_horizon
                })
                
                pred_idx += 1
        
        return pd.DataFrame(results)
    
    def evaluate(self, test_data: pd.DataFrame,
                feature_columns: List[str],
                target_column: str = 'is_failure') -> Dict:
        """
        Evaluate model performance
        
        Args:
            test_data: Test dataframe
            feature_columns: List of feature columns
            target_column: Target column name
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess data
        test_processed = self.preprocess_data(test_data, feature_columns, fit_scaler=False)
        
        # Prepare sequences
        X_test, y_test = self.prepare_sequences(test_processed, feature_columns, target_column)
        
        if len(X_test) == 0:
            return {}
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        
        # For each time step in prediction horizon
        for t in range(self.prediction_horizon):
            y_true_t = y_test[:, t]
            y_pred_t = y_pred[:, t]
            y_pred_binary_t = (y_pred_t > 0.5).astype(int)
            
            metrics[f'step_{t+1}'] = {
                'mse': mean_squared_error(y_true_t, y_pred_t),
                'mae': mean_absolute_error(y_true_t, y_pred_t),
                'accuracy': np.mean(y_true_t == y_pred_binary_t),
            }
        
        # Overall metrics (average across prediction horizon)
        y_true_avg = np.mean(y_test, axis=1)
        y_pred_avg = np.mean(y_pred, axis=1)
        y_pred_binary_avg = (y_pred_avg > 0.5).astype(int)
        
        metrics['overall'] = {
            'mse': mean_squared_error(y_true_avg, y_pred_avg),
            'mae': mean_absolute_error(y_true_avg, y_pred_avg),
            'accuracy': np.mean(y_true_avg.round() == y_pred_binary_avg),
        }
        
        return metrics
    
    def save_model(self, model_path: str, scaler_path: str):
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'feature_columns': self.feature_columns
        }
        
        metadata_path = model_path.replace('.h5', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    def load_model(self, model_path: str, scaler_path: str):
        """Load trained model and scaler"""
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = model_path.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.sequence_length = metadata['sequence_length']
            self.prediction_horizon = metadata['prediction_horizon']
            self.feature_columns = metadata['feature_columns']
        
        self.is_trained = True


if __name__ == "__main__":
    # Example usage
    from src.data_simulation import IoTSensorSimulator
    
    # Generate sample data
    simulator = IoTSensorSimulator()
    data = simulator.generate_fleet_data(num_equipment=3, duration_days=10)
    
    # Define feature columns (exclude non-numeric columns)
    feature_columns = [col for col in data.columns 
                      if col not in ['timestamp', 'equipment_id', 'equipment_type', 'is_failure']]
    
    # Split data
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Initialize and train model
    lstm_model = LSTMPredictor(sequence_length=24, prediction_horizon=6)
    
    history = lstm_model.train(
        train_data=train_data,
        val_data=val_data,
        feature_columns=feature_columns,
        epochs=10  # Reduced for example
    )
    
    # Make predictions
    predictions = lstm_model.predict_failure_probability(test_data, feature_columns)
    print(f"Generated {len(predictions)} predictions")
    
    # Evaluate model
    metrics = lstm_model.evaluate(test_data, feature_columns)
    print("Evaluation metrics:", metrics)
