"""
ARIMA Model for Predictive Maintenance
Time-series forecasting using statistical methods
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional
import warnings
import logging
import joblib
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARIMAPredictor:
    """ARIMA-based predictive maintenance model"""
    
    def __init__(self, order: Tuple[int, int, int] = None, seasonal_order: Tuple[int, int, int, int] = None):
        """
        Initialize ARIMA predictor
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.models = {}  # Store models for each equipment
        self.scalers = {}  # Store scalers for each equipment
        self.is_trained = False
        self.feature_columns = None
        
    def check_stationarity(self, timeseries: pd.Series, significance_level: float = 0.05) -> Dict:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        
        Args:
            timeseries: Time series data
            significance_level: Significance level for the test
            
        Returns:
            Dictionary with test results
        """
        result = adfuller(timeseries.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] <= significance_level
        }
    
    def difference_series(self, series: pd.Series, periods: int = 1) -> pd.Series:
        """Apply differencing to make series stationary"""
        return series.diff(periods=periods).dropna()
    
    def auto_arima_order(self, series: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """
        Automatically determine ARIMA order using AIC criterion
        
        Args:
            series: Time series data
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            
        Returns:
            Best ARIMA order (p, d, q)
        """
        best_aic = float('inf')
        best_order = (0, 0, 0)
        
        # Test different combinations
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            
                    except Exception as e:
                        continue
        
        logger.info(f"Best ARIMA order: {best_order} with AIC: {best_aic:.2f}")
        return best_order
    
    def detect_outliers(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Detect and handle outliers using z-score method
        
        Args:
            series: Time series data
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Series with outliers handled
        """
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = z_scores > threshold
        
        # Replace outliers with median
        cleaned_series = series.copy()
        cleaned_series[outliers] = series.median()
        
        logger.info(f"Detected and handled {outliers.sum()} outliers")
        return cleaned_series
    
    def prepare_data(self, data: pd.DataFrame, equipment_id: str, 
                    feature_column: str, target_column: str = 'is_failure') -> Tuple[pd.Series, pd.Series]:
        """
        Prepare data for ARIMA modeling
        
        Args:
            data: Input dataframe
            equipment_id: Equipment identifier
            feature_column: Feature column to model
            target_column: Target column for failure prediction
            
        Returns:
            Feature series and target series
        """
        # Filter data for specific equipment
        equipment_data = data[data['equipment_id'] == equipment_id].copy()
        equipment_data = equipment_data.sort_values('timestamp')
        
        # Set timestamp as index
        equipment_data.set_index('timestamp', inplace=True)
        
        # Get feature and target series
        feature_series = equipment_data[feature_column]
        target_series = equipment_data[target_column]
        
        # Handle missing values
        feature_series = feature_series.fillna(method='ffill').fillna(method='bfill')
        
        # Detect and handle outliers
        feature_series = self.detect_outliers(feature_series)
        
        return feature_series, target_series
    
    def train_equipment_model(self, feature_series: pd.Series, 
                            equipment_id: str, feature_name: str) -> Dict:
        """
        Train ARIMA model for specific equipment and feature
        
        Args:
            feature_series: Time series data for the feature
            equipment_id: Equipment identifier
            feature_name: Name of the feature
            
        Returns:
            Training results
        """
        logger.info(f"Training ARIMA model for {equipment_id} - {feature_name}")
        
        # Check stationarity
        stationarity_test = self.check_stationarity(feature_series)
        logger.info(f"Stationarity test p-value: {stationarity_test['p_value']:.4f}")
        
        # Determine ARIMA order if not provided
        if self.order is None:
            order = self.auto_arima_order(feature_series)
        else:
            order = self.order
        
        try:
            # Fit ARIMA model
            model = ARIMA(feature_series, order=order, seasonal_order=self.seasonal_order)
            fitted_model = model.fit()
            
            # Store model
            model_key = f"{equipment_id}_{feature_name}"
            self.models[model_key] = fitted_model
            
            logger.info(f"Model trained successfully. AIC: {fitted_model.aic:.2f}")
            
            return {
                'equipment_id': equipment_id,
                'feature_name': feature_name,
                'order': order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'stationarity_p_value': stationarity_test['p_value']
            }
            
        except Exception as e:
            logger.error(f"Failed to train model for {equipment_id} - {feature_name}: {str(e)}")
            return None
    
    def train(self, data: pd.DataFrame, feature_columns: List[str]) -> Dict:
        """
        Train ARIMA models for all equipment and features
        
        Args:
            data: Training dataframe
            feature_columns: List of feature columns to model
            
        Returns:
            Training results summary
        """
        self.feature_columns = feature_columns
        training_results = []
        
        # Get unique equipment IDs
        equipment_ids = data['equipment_id'].unique()
        
        logger.info(f"Training ARIMA models for {len(equipment_ids)} equipment units")
        
        for equipment_id in equipment_ids:
            for feature_column in feature_columns:
                try:
                    # Prepare data for this equipment and feature
                    feature_series, _ = self.prepare_data(data, equipment_id, feature_column)
                    
                    if len(feature_series) < 10:  # Need minimum data points
                        logger.warning(f"Insufficient data for {equipment_id} - {feature_column}")
                        continue
                    
                    # Train model
                    result = self.train_equipment_model(feature_series, equipment_id, feature_column)
                    if result:
                        training_results.append(result)
                        
                except Exception as e:
                    logger.error(f"Error training {equipment_id} - {feature_column}: {str(e)}")
                    continue
        
        self.is_trained = True
        logger.info(f"Training completed. {len(training_results)} models trained successfully.")
        
        return {
            'total_models': len(training_results),
            'results': training_results
        }
    
    def predict_feature(self, equipment_id: str, feature_name: str, 
                       steps: int = 6) -> Dict:
        """
        Predict future values for a specific equipment feature
        
        Args:
            equipment_id: Equipment identifier
            feature_name: Feature name
            steps: Number of steps to forecast
            
        Returns:
            Prediction results
        """
        model_key = f"{equipment_id}_{feature_name}"
        
        if model_key not in self.models:
            raise ValueError(f"No trained model found for {equipment_id} - {feature_name}")
        
        model = self.models[model_key]
        
        try:
            # Make forecast
            forecast = model.forecast(steps=steps)
            forecast_ci = model.get_forecast(steps=steps).conf_int()
            
            return {
                'equipment_id': equipment_id,
                'feature_name': feature_name,
                'forecast': forecast.tolist(),
                'lower_ci': forecast_ci.iloc[:, 0].tolist(),
                'upper_ci': forecast_ci.iloc[:, 1].tolist(),
                'steps': steps
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for {equipment_id} - {feature_name}: {str(e)}")
            return None
    
    def predict_failure_probability(self, data: pd.DataFrame, 
                                  prediction_steps: int = 6,
                                  failure_thresholds: Dict[str, Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Predict failure probability based on feature forecasts
        
        Args:
            data: Recent data for prediction
            prediction_steps: Number of steps to predict ahead
            failure_thresholds: Thresholds for each feature (min, max) indicating failure risk
            
        Returns:
            Dataframe with failure probability predictions
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call train() first.")
        
        if failure_thresholds is None:
            # Default thresholds (these should be calibrated based on domain knowledge)
            failure_thresholds = {
                'temperature': (100, float('inf')),
                'vibration': (0.8, float('inf')),
                'pressure': (0, 1.5),
                'flow_rate': (0, 90),
                'power_consumption': (15, float('inf')),
                'current': (20, float('inf')),
                'rpm': (0, 1700),
                'power_output': (0, 400),
                'efficiency': (0, 80),
                'voltage': (0, 370)
            }
        
        results = []
        equipment_ids = data['equipment_id'].unique()
        
        for equipment_id in equipment_ids:
            equipment_predictions = {}
            failure_scores = []
            
            # Get predictions for each feature
            for feature_name in self.feature_columns:
                prediction = self.predict_feature(equipment_id, feature_name, prediction_steps)
                
                if prediction:
                    equipment_predictions[feature_name] = prediction
                    
                    # Calculate failure score based on thresholds
                    if feature_name in failure_thresholds:
                        min_thresh, max_thresh = failure_thresholds[feature_name]
                        forecast_values = prediction['forecast']
                        
                        # Calculate how many predicted values exceed thresholds
                        failure_count = sum(1 for val in forecast_values 
                                          if val < min_thresh or val > max_thresh)
                        failure_score = failure_count / len(forecast_values)
                        failure_scores.append(failure_score)
            
            # Calculate overall failure probability
            if failure_scores:
                avg_failure_prob = np.mean(failure_scores)
                max_failure_prob = np.max(failure_scores)
                
                # Get latest timestamp for this equipment
                equipment_data = data[data['equipment_id'] == equipment_id]
                latest_timestamp = equipment_data['timestamp'].max()
                
                results.append({
                    'timestamp': latest_timestamp,
                    'equipment_id': equipment_id,
                    'failure_probability': avg_failure_prob,
                    'max_failure_probability': max_failure_prob,
                    'predicted_failure': avg_failure_prob > 0.5,
                    'prediction_horizon_hours': prediction_steps,
                    'feature_predictions': equipment_predictions
                })
        
        return pd.DataFrame(results)
    
    def evaluate(self, test_data: pd.DataFrame, feature_columns: List[str]) -> Dict:
        """
        Evaluate ARIMA model performance
        
        Args:
            test_data: Test dataframe
            feature_columns: List of feature columns
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call train() first.")
        
        evaluation_results = {}
        equipment_ids = test_data['equipment_id'].unique()
        
        for equipment_id in equipment_ids:
            for feature_name in feature_columns:
                model_key = f"{equipment_id}_{feature_name}"
                
                if model_key not in self.models:
                    continue
                
                try:
                    # Prepare test data
                    feature_series, _ = self.prepare_data(test_data, equipment_id, feature_name)
                    
                    if len(feature_series) < 10:
                        continue
                    
                    # Split into train and test for evaluation
                    split_point = int(0.8 * len(feature_series))
                    train_series = feature_series[:split_point]
                    test_series = feature_series[split_point:]
                    
                    # Retrain model on train portion
                    model = ARIMA(train_series, order=self.models[model_key].model.order)
                    fitted_model = model.fit()
                    
                    # Make predictions
                    predictions = fitted_model.forecast(steps=len(test_series))
                    
                    # Calculate metrics
                    mse = mean_squared_error(test_series, predictions)
                    mae = mean_absolute_error(test_series, predictions)
                    mape = np.mean(np.abs((test_series - predictions) / test_series)) * 100
                    
                    evaluation_results[model_key] = {
                        'mse': mse,
                        'mae': mae,
                        'mape': mape,
                        'rmse': np.sqrt(mse)
                    }
                    
                except Exception as e:
                    logger.error(f"Evaluation failed for {model_key}: {str(e)}")
                    continue
        
        return evaluation_results
    
    def save_models(self, models_dir: str):
        """Save trained ARIMA models"""
        if not self.is_trained:
            raise ValueError("No models to save")
        
        os.makedirs(models_dir, exist_ok=True)
        
        # Save each model
        for model_key, model in self.models.items():
            model_path = os.path.join(models_dir, f"{model_key}_arima.pkl")
            joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'model_keys': list(self.models.keys())
        }
        
        metadata_path = os.path.join(models_dir, 'arima_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    def load_models(self, models_dir: str):
        """Load trained ARIMA models"""
        # Load metadata
        metadata_path = os.path.join(models_dir, 'arima_metadata.json')
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_columns = metadata['feature_columns']
        self.order = metadata['order']
        self.seasonal_order = metadata['seasonal_order']
        
        # Load models
        self.models = {}
        for model_key in metadata['model_keys']:
            model_path = os.path.join(models_dir, f"{model_key}_arima.pkl")
            if os.path.exists(model_path):
                self.models[model_key] = joblib.load(model_path)
        
        self.is_trained = True


if __name__ == "__main__":
    # Example usage
    from src.data_simulation import IoTSensorSimulator
    
    # Generate sample data
    simulator = IoTSensorSimulator()
    data = simulator.generate_fleet_data(num_equipment=2, duration_days=15)
    
    # Define feature columns
    feature_columns = [col for col in data.columns 
                      if col not in ['timestamp', 'equipment_id', 'equipment_type', 'is_failure']]
    
    # Split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Initialize and train ARIMA model
    arima_model = ARIMAPredictor()
    
    training_results = arima_model.train(train_data, feature_columns[:2])  # Use first 2 features for example
    print(f"Training results: {training_results}")
    
    # Make predictions
    predictions = arima_model.predict_failure_probability(test_data)
    print(f"Generated {len(predictions)} predictions")
    
    # Evaluate model
    evaluation = arima_model.evaluate(test_data, feature_columns[:2])
    print("Evaluation metrics:", evaluation)
