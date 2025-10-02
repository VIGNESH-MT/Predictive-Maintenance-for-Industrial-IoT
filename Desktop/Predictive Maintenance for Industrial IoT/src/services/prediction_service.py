"""
Prediction Service
Combines LSTM and ARIMA models for predictive maintenance
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import os

from src.models import LSTMPredictor, ARIMAPredictor
from src.pipeline import SensorReading, RedisCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsemblePredictionService:
    """Service that combines LSTM and ARIMA predictions"""
    
    def __init__(self, models_dir: str = 'models', redis_cache: RedisCache = None):
        self.models_dir = models_dir
        self.redis_cache = redis_cache
        
        # Initialize models
        self.lstm_model = LSTMPredictor()
        self.arima_model = ARIMAPredictor()
        
        # Model weights for ensemble
        self.lstm_weight = 0.7
        self.arima_weight = 0.3
        
        # Feature columns (will be set during training/loading)
        self.feature_columns = None
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Thread pool for async predictions
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def load_models(self):
        """Load pre-trained models"""
        try:
            # Load LSTM model
            lstm_model_path = os.path.join(self.models_dir, 'lstm_model.h5')
            lstm_scaler_path = os.path.join(self.models_dir, 'lstm_scaler.pkl')
            
            if os.path.exists(lstm_model_path) and os.path.exists(lstm_scaler_path):
                self.lstm_model.load_model(lstm_model_path, lstm_scaler_path)
                logger.info("LSTM model loaded successfully")
            else:
                logger.warning("LSTM model files not found")
            
            # Load ARIMA models
            arima_models_dir = os.path.join(self.models_dir, 'arima')
            if os.path.exists(arima_models_dir):
                self.arima_model.load_models(arima_models_dir)
                logger.info("ARIMA models loaded successfully")
            else:
                logger.warning("ARIMA models directory not found")
            
            # Set feature columns from LSTM model (assuming it's the primary model)
            if self.lstm_model.feature_columns:
                self.feature_columns = self.lstm_model.feature_columns
            elif self.arima_model.feature_columns:
                self.feature_columns = self.arima_model.feature_columns
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def train_models(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                    feature_columns: List[str]):
        """Train both LSTM and ARIMA models"""
        self.feature_columns = feature_columns
        
        logger.info("Training LSTM model...")
        lstm_history = self.lstm_model.train(
            train_data=train_data,
            val_data=val_data,
            feature_columns=feature_columns,
            epochs=50,
            batch_size=32,
            model_save_path=os.path.join(self.models_dir, 'lstm_model.h5')
        )
        
        logger.info("Training ARIMA models...")
        arima_results = self.arima_model.train(train_data, feature_columns)
        
        # Save models
        self.save_models()
        
        return {
            'lstm_history': lstm_history,
            'arima_results': arima_results
        }
    
    def save_models(self):
        """Save trained models"""
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Save LSTM model
        if self.lstm_model.is_trained:
            lstm_model_path = os.path.join(self.models_dir, 'lstm_model.h5')
            lstm_scaler_path = os.path.join(self.models_dir, 'lstm_scaler.pkl')
            self.lstm_model.save_model(lstm_model_path, lstm_scaler_path)
        
        # Save ARIMA models
        if self.arima_model.is_trained:
            arima_models_dir = os.path.join(self.models_dir, 'arima')
            self.arima_model.save_models(arima_models_dir)
    
    def sensor_readings_to_dataframe(self, readings: List[SensorReading]) -> pd.DataFrame:
        """Convert sensor readings to DataFrame"""
        data = []
        for reading in readings:
            row = {
                'timestamp': reading.timestamp,
                'equipment_id': reading.equipment_id,
                'equipment_type': reading.equipment_type,
                'is_failure': reading.is_failure
            }
            row.update(reading.sensor_data)
            data.append(row)
        
        return pd.DataFrame(data)
    
    def predict_lstm(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get LSTM predictions"""
        try:
            if not self.lstm_model.is_trained:
                logger.warning("LSTM model not trained")
                return None
            
            predictions = self.lstm_model.predict_failure_probability(
                data, self.feature_columns
            )
            return predictions
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return None
    
    def predict_arima(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get ARIMA predictions"""
        try:
            if not self.arima_model.is_trained:
                logger.warning("ARIMA model not trained")
                return None
            
            predictions = self.arima_model.predict_failure_probability(data)
            return predictions
            
        except Exception as e:
            logger.error(f"ARIMA prediction error: {e}")
            return None
    
    def ensemble_predict(self, equipment_id: str, recent_data: List[SensorReading]) -> Dict:
        """Make ensemble prediction combining LSTM and ARIMA"""
        try:
            # Check cache first
            cache_key = f"{equipment_id}_{len(recent_data)}"
            if cache_key in self.prediction_cache:
                cached_result, timestamp = self.prediction_cache[cache_key]
                if (datetime.now() - timestamp).seconds < self.cache_ttl:
                    return cached_result
            
            # Convert to DataFrame
            df = self.sensor_readings_to_dataframe(recent_data)
            
            if df.empty:
                return {'error': 'No data provided'}
            
            # Get predictions from both models
            lstm_pred = self.predict_lstm(df)
            arima_pred = self.predict_arima(df)
            
            # Combine predictions
            ensemble_result = {
                'equipment_id': equipment_id,
                'timestamp': datetime.now().isoformat(),
                'data_points': len(recent_data),
                'lstm_available': lstm_pred is not None,
                'arima_available': arima_pred is not None,
                'predictions': {}
            }
            
            if lstm_pred is not None and not lstm_pred.empty:
                lstm_prob = lstm_pred['failure_probability'].iloc[-1]
                ensemble_result['predictions']['lstm'] = {
                    'failure_probability': float(lstm_prob),
                    'predicted_failure': bool(lstm_prob > 0.5)
                }
            
            if arima_pred is not None and not arima_pred.empty:
                arima_prob = arima_pred['failure_probability'].iloc[-1]
                ensemble_result['predictions']['arima'] = {
                    'failure_probability': float(arima_prob),
                    'predicted_failure': bool(arima_prob > 0.5)
                }
            
            # Calculate ensemble prediction
            if lstm_pred is not None and arima_pred is not None and not lstm_pred.empty and not arima_pred.empty:
                lstm_prob = lstm_pred['failure_probability'].iloc[-1]
                arima_prob = arima_pred['failure_probability'].iloc[-1]
                
                ensemble_prob = (self.lstm_weight * lstm_prob + 
                               self.arima_weight * arima_prob)
                
                ensemble_result['predictions']['ensemble'] = {
                    'failure_probability': float(ensemble_prob),
                    'predicted_failure': bool(ensemble_prob > 0.5),
                    'lstm_weight': self.lstm_weight,
                    'arima_weight': self.arima_weight
                }
                
                # Add confidence score based on agreement
                agreement = 1 - abs(lstm_prob - arima_prob)
                ensemble_result['predictions']['ensemble']['confidence'] = float(agreement)
                
            elif lstm_pred is not None and not lstm_pred.empty:
                # Use only LSTM
                lstm_prob = lstm_pred['failure_probability'].iloc[-1]
                ensemble_result['predictions']['ensemble'] = {
                    'failure_probability': float(lstm_prob),
                    'predicted_failure': bool(lstm_prob > 0.5),
                    'confidence': 0.7,  # Lower confidence with single model
                    'note': 'LSTM only'
                }
                
            elif arima_pred is not None and not arima_pred.empty:
                # Use only ARIMA
                arima_prob = arima_pred['failure_probability'].iloc[-1]
                ensemble_result['predictions']['ensemble'] = {
                    'failure_probability': float(arima_prob),
                    'predicted_failure': bool(arima_prob > 0.5),
                    'confidence': 0.6,  # Lower confidence with single model
                    'note': 'ARIMA only'
                }
            else:
                ensemble_result['predictions']['ensemble'] = {
                    'failure_probability': 0.0,
                    'predicted_failure': False,
                    'confidence': 0.0,
                    'note': 'No models available'
                }
            
            # Cache result
            self.prediction_cache[cache_key] = (ensemble_result, datetime.now())
            
            # Store in Redis if available
            if self.redis_cache:
                self.redis_cache.store_prediction(equipment_id, ensemble_result)
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            return {
                'equipment_id': equipment_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def async_predict(self, equipment_id: str, recent_data: List[SensorReading]) -> Dict:
        """Async wrapper for ensemble prediction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.ensemble_predict, equipment_id, recent_data
        )
    
    def batch_predict(self, equipment_data: Dict[str, List[SensorReading]]) -> Dict[str, Dict]:
        """Make predictions for multiple equipment"""
        results = {}
        
        # Use thread pool for parallel predictions
        futures = {}
        for equipment_id, readings in equipment_data.items():
            future = self.executor.submit(self.ensemble_predict, equipment_id, readings)
            futures[equipment_id] = future
        
        # Collect results
        for equipment_id, future in futures.items():
            try:
                results[equipment_id] = future.result(timeout=30)
            except Exception as e:
                logger.error(f"Batch prediction error for {equipment_id}: {e}")
                results[equipment_id] = {
                    'equipment_id': equipment_id,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        return results
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for loaded models"""
        performance = {
            'lstm': {'loaded': self.lstm_model.is_trained},
            'arima': {'loaded': self.arima_model.is_trained},
            'ensemble_weights': {
                'lstm': self.lstm_weight,
                'arima': self.arima_weight
            }
        }
        
        return performance
    
    def update_ensemble_weights(self, lstm_weight: float, arima_weight: float):
        """Update ensemble weights"""
        if abs(lstm_weight + arima_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        
        self.lstm_weight = lstm_weight
        self.arima_weight = arima_weight
        
        logger.info(f"Updated ensemble weights: LSTM={lstm_weight}, ARIMA={arima_weight}")
    
    def health_check(self) -> Dict:
        """Check service health"""
        return {
            'status': 'healthy',
            'models_loaded': {
                'lstm': self.lstm_model.is_trained,
                'arima': self.arima_model.is_trained
            },
            'feature_columns': self.feature_columns,
            'cache_size': len(self.prediction_cache),
            'redis_connected': self.redis_cache is not None,
            'timestamp': datetime.now().isoformat()
        }


class PredictionServiceManager:
    """Manager for the prediction service with automatic model retraining"""
    
    def __init__(self, models_dir: str = 'models', redis_cache: RedisCache = None):
        self.prediction_service = EnsemblePredictionService(models_dir, redis_cache)
        self.retrain_threshold = 1000  # Retrain after N predictions
        self.prediction_count = 0
        self.performance_history = []
        
    def initialize(self):
        """Initialize the service"""
        try:
            self.prediction_service.load_models()
            logger.info("Prediction service initialized successfully")
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            logger.info("Service will need training before making predictions")
    
    def predict(self, equipment_id: str, recent_data: List[SensorReading]) -> Dict:
        """Make prediction and track performance"""
        result = self.prediction_service.ensemble_predict(equipment_id, recent_data)
        
        self.prediction_count += 1
        
        # Log prediction for monitoring
        if 'predictions' in result and 'ensemble' in result['predictions']:
            prob = result['predictions']['ensemble']['failure_probability']
            confidence = result['predictions']['ensemble'].get('confidence', 0.0)
            
            logger.info(f"Prediction for {equipment_id}: "
                       f"failure_prob={prob:.3f}, confidence={confidence:.3f}")
        
        return result
    
    def should_retrain(self) -> bool:
        """Check if models should be retrained"""
        return self.prediction_count >= self.retrain_threshold
    
    def reset_counters(self):
        """Reset prediction counters after retraining"""
        self.prediction_count = 0
        self.performance_history.clear()


if __name__ == "__main__":
    # Example usage
    from src.data_simulation import IoTSensorSimulator
    from src.pipeline import RedisCache
    
    # Initialize service
    redis_cache = RedisCache()
    service = EnsemblePredictionService(redis_cache=redis_cache)
    
    # Generate sample data for training
    simulator = IoTSensorSimulator()
    data = simulator.generate_fleet_data(num_equipment=5, duration_days=20)
    
    # Define feature columns
    feature_columns = [col for col in data.columns 
                      if col not in ['timestamp', 'equipment_id', 'equipment_type', 'is_failure']]
    
    # Split data
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Train models (this would typically be done offline)
    print("Training models...")
    training_results = service.train_models(train_data, val_data, feature_columns)
    
    # Test prediction
    print("Testing prediction...")
    test_equipment = test_data['equipment_id'].iloc[0]
    test_readings = []
    
    for _, row in test_data[test_data['equipment_id'] == test_equipment].head(24).iterrows():
        sensor_data = {col: row[col] for col in feature_columns}
        reading = SensorReading(
            timestamp=row['timestamp'],
            equipment_id=row['equipment_id'],
            equipment_type=row['equipment_type'],
            sensor_data=sensor_data,
            is_failure=row['is_failure']
        )
        test_readings.append(reading)
    
    # Make prediction
    prediction = service.ensemble_predict(test_equipment, test_readings)
    print(f"Prediction result: {json.dumps(prediction, indent=2)}")
    
    # Check health
    health = service.health_check()
    print(f"Service health: {json.dumps(health, indent=2)}")
