#!/usr/bin/env python3
"""
Model Training Script
Trains LSTM and ARIMA models for predictive maintenance
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.append('/app')

from src.data_simulation import IoTSensorSimulator
from src.services import EnsemblePredictionService
from src.pipeline import RedisCache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_training_data(num_equipment: int = 20, duration_days: int = 60) -> pd.DataFrame:
    """Generate training data using simulator"""
    logger.info(f"Generating training data for {num_equipment} equipment over {duration_days} days")
    
    simulator = IoTSensorSimulator()
    data = simulator.generate_fleet_data(
        num_equipment=num_equipment,
        duration_days=duration_days,
        start_date=datetime.now() - timedelta(days=duration_days)
    )
    
    logger.info(f"Generated {len(data)} data points")
    
    # Save raw data
    os.makedirs('/app/data', exist_ok=True)
    data.to_csv('/app/data/training_data.csv', index=False)
    
    return data


def prepare_features(data: pd.DataFrame) -> list:
    """Prepare feature columns"""
    feature_columns = [col for col in data.columns 
                      if col not in ['timestamp', 'equipment_id', 'equipment_type', 'is_failure']]
    
    logger.info(f"Feature columns: {feature_columns}")
    return feature_columns


def split_data(data: pd.DataFrame) -> tuple:
    """Split data into train, validation, and test sets"""
    # Sort by timestamp
    data_sorted = data.sort_values('timestamp')
    
    # Split by time to avoid data leakage
    total_size = len(data_sorted)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    
    train_data = data_sorted[:train_size]
    val_data = data_sorted[train_size:train_size + val_size]
    test_data = data_sorted[train_size + val_size:]
    
    logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def train_models(train_data: pd.DataFrame, val_data: pd.DataFrame, 
                feature_columns: list) -> dict:
    """Train both LSTM and ARIMA models"""
    logger.info("Starting model training...")
    
    # Initialize service
    redis_cache = RedisCache(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', '6379'))
    )
    
    try:
        redis_cache.connect()
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}")
        redis_cache = None
    
    service = EnsemblePredictionService(
        models_dir='/app/models',
        redis_cache=redis_cache
    )
    
    # Train models
    training_results = service.train_models(
        train_data=train_data,
        val_data=val_data,
        feature_columns=feature_columns
    )
    
    logger.info("Model training completed")
    return training_results


def evaluate_models(service: EnsemblePredictionService, test_data: pd.DataFrame, 
                   feature_columns: list) -> dict:
    """Evaluate trained models"""
    logger.info("Evaluating models...")
    
    try:
        # LSTM evaluation
        lstm_metrics = service.lstm_model.evaluate(test_data, feature_columns)
        logger.info(f"LSTM metrics: {lstm_metrics}")
        
        # ARIMA evaluation
        arima_metrics = service.arima_model.evaluate(test_data, feature_columns)
        logger.info(f"ARIMA metrics: {arima_metrics}")
        
        # Test ensemble predictions
        test_equipment = test_data['equipment_id'].unique()[:5]  # Test on 5 equipment
        ensemble_results = []
        
        for equipment_id in test_equipment:
            equipment_data = test_data[test_data['equipment_id'] == equipment_id]
            if len(equipment_data) >= 24:  # Need minimum data
                # Convert to sensor readings format
                from src.pipeline import SensorReading
                readings = []
                
                for _, row in equipment_data.head(24).iterrows():
                    sensor_data = {col: row[col] for col in feature_columns}
                    reading = SensorReading(
                        timestamp=row['timestamp'],
                        equipment_id=row['equipment_id'],
                        equipment_type=row['equipment_type'],
                        sensor_data=sensor_data,
                        is_failure=row['is_failure']
                    )
                    readings.append(reading)
                
                # Make ensemble prediction
                prediction = service.ensemble_predict(equipment_id, readings)
                ensemble_results.append(prediction)
        
        evaluation_results = {
            'lstm_metrics': lstm_metrics,
            'arima_metrics': arima_metrics,
            'ensemble_predictions': len(ensemble_results),
            'test_equipment_count': len(test_equipment)
        }
        
        logger.info("Model evaluation completed")
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return {'error': str(e)}


def save_training_report(training_results: dict, evaluation_results: dict, 
                        feature_columns: list):
    """Save training report"""
    report = {
        'training_date': datetime.now().isoformat(),
        'feature_columns': feature_columns,
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'model_versions': {
            'lstm': '1.0',
            'arima': '1.0',
            'ensemble': '1.0'
        }
    }
    
    os.makedirs('/app/models', exist_ok=True)
    with open('/app/models/training_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("Training report saved")


def main():
    """Main training function"""
    try:
        logger.info("Starting model training process...")
        
        # Check if we should use existing data or generate new
        use_existing_data = os.getenv('USE_EXISTING_DATA', 'false').lower() == 'true'
        data_path = '/app/data/training_data.csv'
        
        if use_existing_data and os.path.exists(data_path):
            logger.info("Loading existing training data...")
            data = pd.read_csv(data_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        else:
            # Generate new training data
            num_equipment = int(os.getenv('NUM_EQUIPMENT', '20'))
            duration_days = int(os.getenv('DURATION_DAYS', '60'))
            data = generate_training_data(num_equipment, duration_days)
        
        # Prepare features
        feature_columns = prepare_features(data)
        
        # Split data
        train_data, val_data, test_data = split_data(data)
        
        # Train models
        training_results = train_models(train_data, val_data, feature_columns)
        
        # Evaluate models
        service = EnsemblePredictionService(models_dir='/app/models')
        service.load_models()
        evaluation_results = evaluate_models(service, test_data, feature_columns)
        
        # Save report
        save_training_report(training_results, evaluation_results, feature_columns)
        
        logger.info("Model training process completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Training data points: {len(train_data)}")
        print(f"Validation data points: {len(val_data)}")
        print(f"Test data points: {len(test_data)}")
        print(f"Feature columns: {len(feature_columns)}")
        print(f"LSTM trained: {training_results.get('lstm_history') is not None}")
        print(f"ARIMA models trained: {training_results.get('arima_results', {}).get('total_models', 0)}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
