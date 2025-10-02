"""
FastAPI REST API for Predictive Maintenance System
Provides endpoints for predictions, monitoring, and data access
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio
import json
import os

from src.services import EnsemblePredictionService, PredictionServiceManager
from src.pipeline import RedisCache, SensorReading, DataPipeline
from src.data_simulation import IoTSensorSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="REST API for Industrial IoT Predictive Maintenance System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
redis_cache = None
prediction_service_manager = None
data_pipeline = None


# Pydantic models
class SensorData(BaseModel):
    timestamp: datetime
    equipment_id: str
    equipment_type: str
    sensor_data: Dict[str, float]
    is_failure: bool = False


class PredictionRequest(BaseModel):
    equipment_id: str
    hours_back: int = Field(default=24, ge=1, le=168)  # 1 hour to 1 week


class PredictionResponse(BaseModel):
    equipment_id: str
    timestamp: str
    failure_probability: float
    predicted_failure: bool
    confidence: float
    model_used: str
    data_points: int


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    redis_connected: bool
    pipeline_running: bool


class MetricsResponse(BaseModel):
    total_equipment: int
    active_equipment: int
    predictions_made: int
    failure_alerts: int
    system_uptime: str


# Dependency to get services
def get_redis_cache():
    global redis_cache
    if redis_cache is None:
        redis_cache = RedisCache()
        redis_cache.connect()
    return redis_cache


def get_prediction_service():
    global prediction_service_manager
    if prediction_service_manager is None:
        prediction_service_manager = PredictionServiceManager(
            models_dir='models',
            redis_cache=get_redis_cache()
        )
        prediction_service_manager.initialize()
    return prediction_service_manager


def get_data_pipeline():
    global data_pipeline
    if data_pipeline is None:
        data_pipeline = DataPipeline()
    return data_pipeline


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Predictive Maintenance API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(
    prediction_service: PredictionServiceManager = Depends(get_prediction_service),
    redis_cache: RedisCache = Depends(get_redis_cache)
):
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_connected = False
        try:
            redis_cache.redis_client.ping()
            redis_connected = True
        except:
            pass
        
        # Get service health
        service_health = prediction_service.prediction_service.health_check()
        
        return HealthResponse(
            status="healthy" if service_health['status'] == 'healthy' else "degraded",
            timestamp=datetime.now().isoformat(),
            models_loaded=service_health['models_loaded'],
            redis_connected=redis_connected,
            pipeline_running=True  # TODO: Add actual pipeline status check
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/predict", response_model=PredictionResponse)
async def predict_failure(
    request: PredictionRequest,
    prediction_service: PredictionServiceManager = Depends(get_prediction_service),
    redis_cache: RedisCache = Depends(get_redis_cache)
):
    """Predict failure for specific equipment"""
    try:
        # Get recent data from Redis
        recent_readings = redis_cache.get_recent_data(
            request.equipment_id, 
            hours=request.hours_back
        )
        
        if not recent_readings:
            raise HTTPException(
                status_code=404, 
                detail=f"No recent data found for equipment {request.equipment_id}"
            )
        
        # Make prediction
        prediction_result = prediction_service.predict(request.equipment_id, recent_readings)
        
        if 'error' in prediction_result:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {prediction_result['error']}"
            )
        
        # Extract ensemble prediction
        ensemble_pred = prediction_result['predictions'].get('ensemble', {})
        
        return PredictionResponse(
            equipment_id=request.equipment_id,
            timestamp=prediction_result['timestamp'],
            failure_probability=ensemble_pred.get('failure_probability', 0.0),
            predicted_failure=ensemble_pred.get('predicted_failure', False),
            confidence=ensemble_pred.get('confidence', 0.0),
            model_used=ensemble_pred.get('note', 'ensemble'),
            data_points=prediction_result['data_points']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/batch", response_model=Dict[str, PredictionResponse])
async def batch_predict(
    equipment_ids: str,  # Comma-separated list
    hours_back: int = 24,
    prediction_service: PredictionServiceManager = Depends(get_prediction_service),
    redis_cache: RedisCache = Depends(get_redis_cache)
):
    """Batch prediction for multiple equipment"""
    try:
        equipment_list = [eq.strip() for eq in equipment_ids.split(',')]
        
        # Collect data for all equipment
        equipment_data = {}
        for equipment_id in equipment_list:
            recent_readings = redis_cache.get_recent_data(equipment_id, hours=hours_back)
            if recent_readings:
                equipment_data[equipment_id] = recent_readings
        
        if not equipment_data:
            raise HTTPException(
                status_code=404,
                detail="No recent data found for any of the specified equipment"
            )
        
        # Make batch predictions
        batch_results = prediction_service.prediction_service.batch_predict(equipment_data)
        
        # Format response
        formatted_results = {}
        for equipment_id, result in batch_results.items():
            if 'error' not in result:
                ensemble_pred = result['predictions'].get('ensemble', {})
                formatted_results[equipment_id] = PredictionResponse(
                    equipment_id=equipment_id,
                    timestamp=result['timestamp'],
                    failure_probability=ensemble_pred.get('failure_probability', 0.0),
                    predicted_failure=ensemble_pred.get('predicted_failure', False),
                    confidence=ensemble_pred.get('confidence', 0.0),
                    model_used=ensemble_pred.get('note', 'ensemble'),
                    data_points=result['data_points']
                )
        
        return formatted_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/equipment", response_model=List[str])
async def list_equipment(redis_cache: RedisCache = Depends(get_redis_cache)):
    """List all equipment with recent data"""
    try:
        # Get all equipment keys from Redis
        keys = redis_cache.redis_client.keys("equipment:*")
        equipment_ids = [key.split(':')[1] for key in keys]
        return sorted(equipment_ids)
        
    except Exception as e:
        logger.error(f"Error listing equipment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/equipment/{equipment_id}/data", response_model=List[SensorData])
async def get_equipment_data(
    equipment_id: str,
    hours: int = 24,
    redis_cache: RedisCache = Depends(get_redis_cache)
):
    """Get recent sensor data for specific equipment"""
    try:
        recent_readings = redis_cache.get_recent_data(equipment_id, hours=hours)
        
        if not recent_readings:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for equipment {equipment_id}"
            )
        
        # Convert to response format
        sensor_data_list = []
        for reading in recent_readings:
            sensor_data_list.append(SensorData(
                timestamp=reading.timestamp,
                equipment_id=reading.equipment_id,
                equipment_type=reading.equipment_type,
                sensor_data=reading.sensor_data,
                is_failure=reading.is_failure
            ))
        
        return sensor_data_list
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting equipment data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(redis_cache: RedisCache = Depends(get_redis_cache)):
    """Get system metrics"""
    try:
        # Get equipment count
        equipment_keys = redis_cache.redis_client.keys("equipment:*")
        total_equipment = len(equipment_keys)
        
        # Count active equipment (with data in last hour)
        active_equipment = 0
        current_time = datetime.now()
        for key in equipment_keys:
            equipment_id = key.split(':')[1]
            recent_data = redis_cache.get_recent_data(equipment_id, hours=1)
            if recent_data:
                active_equipment += 1
        
        # Get prediction count (from Redis if stored)
        predictions_made = 0
        try:
            pred_keys = redis_cache.redis_client.keys("prediction:*")
            predictions_made = len(pred_keys)
        except:
            pass
        
        # Count failure alerts (predictions with high failure probability)
        failure_alerts = 0
        try:
            for key in redis_cache.redis_client.keys("prediction:*"):
                pred_data = redis_cache.redis_client.get(key)
                if pred_data:
                    pred_json = json.loads(pred_data)
                    if (pred_json.get('predictions', {}).get('ensemble', {})
                        .get('failure_probability', 0) > 0.7):
                        failure_alerts += 1
        except:
            pass
        
        return MetricsResponse(
            total_equipment=total_equipment,
            active_equipment=active_equipment,
            predictions_made=predictions_made,
            failure_alerts=failure_alerts,
            system_uptime="24h"  # TODO: Calculate actual uptime
        )
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate/start")
async def start_simulation(
    background_tasks: BackgroundTasks,
    num_equipment: int = 10,
    interval_seconds: int = 15,
    data_pipeline: DataPipeline = Depends(get_data_pipeline)
):
    """Start data simulation"""
    try:
        def run_simulation():
            # Start pipeline
            consumer_thread = data_pipeline.start_pipeline()
            
            # Add prediction callback
            def prediction_callback(equipment_id: str, recent_data: List[SensorReading]):
                try:
                    prediction_service = get_prediction_service()
                    result = prediction_service.predict(equipment_id, recent_data)
                    logger.info(f"Auto-prediction for {equipment_id}: "
                              f"{result.get('predictions', {}).get('ensemble', {}).get('failure_probability', 0):.3f}")
                except Exception as e:
                    logger.error(f"Auto-prediction error: {e}")
            
            data_pipeline.add_prediction_callback(prediction_callback)
            
            # Start simulation
            simulation_thread = data_pipeline.start_simulation(num_equipment, interval_seconds)
            
            return consumer_thread, simulation_thread
        
        background_tasks.add_task(run_simulation)
        
        return {"message": f"Simulation started with {num_equipment} equipment units"}
        
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate/stop")
async def stop_simulation(data_pipeline: DataPipeline = Depends(get_data_pipeline)):
    """Stop data simulation"""
    try:
        data_pipeline.stop_pipeline()
        return {"message": "Simulation stopped"}
        
    except Exception as e:
        logger.error(f"Error stopping simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/performance")
async def get_model_performance(
    prediction_service: PredictionServiceManager = Depends(get_prediction_service)
):
    """Get model performance metrics"""
    try:
        performance = prediction_service.prediction_service.get_model_performance()
        return performance
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/models/weights")
async def update_model_weights(
    lstm_weight: float,
    arima_weight: float,
    prediction_service: PredictionServiceManager = Depends(get_prediction_service)
):
    """Update ensemble model weights"""
    try:
        prediction_service.prediction_service.update_ensemble_weights(lstm_weight, arima_weight)
        return {"message": f"Updated weights: LSTM={lstm_weight}, ARIMA={arima_weight}"}
        
    except Exception as e:
        logger.error(f"Error updating model weights: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Prometheus metrics endpoint for Grafana
@app.get("/metrics/prometheus")
async def prometheus_metrics(redis_cache: RedisCache = Depends(get_redis_cache)):
    """Prometheus-compatible metrics endpoint"""
    try:
        # Get basic metrics
        equipment_keys = redis_cache.redis_client.keys("equipment:*")
        total_equipment = len(equipment_keys)
        
        active_equipment = 0
        failure_predictions = 0
        
        for key in equipment_keys:
            equipment_id = key.split(':')[1]
            
            # Check if active
            recent_data = redis_cache.get_recent_data(equipment_id, hours=1)
            if recent_data:
                active_equipment += 1
            
            # Check failure prediction
            pred_data = redis_cache.get_prediction(equipment_id)
            if pred_data:
                failure_prob = (pred_data.get('predictions', {})
                              .get('ensemble', {})
                              .get('failure_probability', 0))
                if failure_prob > 0.5:
                    failure_predictions += 1
        
        # Format as Prometheus metrics
        metrics = f"""# HELP equipment_total Total number of equipment units
# TYPE equipment_total gauge
equipment_total {total_equipment}

# HELP equipment_active Number of active equipment units
# TYPE equipment_active gauge
equipment_active {active_equipment}

# HELP failure_predictions Number of equipment predicted to fail
# TYPE failure_predictions gauge
failure_predictions {failure_predictions}

# HELP system_health System health status (1=healthy, 0=unhealthy)
# TYPE system_health gauge
system_health 1
"""
        
        return JSONResponse(content=metrics, media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Error generating Prometheus metrics: {e}")
        return JSONResponse(content="", media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    
    # Initialize services
    redis_cache = RedisCache()
    prediction_service_manager = PredictionServiceManager(redis_cache=redis_cache)
    
    # Run the API
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
