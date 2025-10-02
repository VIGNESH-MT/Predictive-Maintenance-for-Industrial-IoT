# API Documentation

This document provides detailed information about the Predictive Maintenance API endpoints.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production deployments, consider implementing:
- JWT tokens
- API keys
- OAuth 2.0

## Endpoints

### Health Check

#### GET /health

Check the health status of the system.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "models_loaded": {
    "lstm": true,
    "arima": true
  },
  "redis_connected": true,
  "pipeline_running": true
}
```

**Status Codes:**
- `200`: System is healthy
- `500`: System has issues

---

### Predictions

#### POST /predict

Get failure prediction for a specific equipment.

**Request Body:**
```json
{
  "equipment_id": "pump_001",
  "hours_back": 24
}
```

**Parameters:**
- `equipment_id` (string, required): Equipment identifier
- `hours_back` (integer, optional): Hours of historical data to use (1-168, default: 24)

**Response:**
```json
{
  "equipment_id": "pump_001",
  "timestamp": "2024-01-01T12:00:00Z",
  "failure_probability": 0.75,
  "predicted_failure": true,
  "confidence": 0.85,
  "model_used": "ensemble",
  "data_points": 96
}
```

**Status Codes:**
- `200`: Prediction successful
- `404`: Equipment not found or no recent data
- `500`: Prediction failed

---

#### GET /predict/batch

Get predictions for multiple equipment units.

**Parameters:**
- `equipment_ids` (string, required): Comma-separated list of equipment IDs
- `hours_back` (integer, optional): Hours of historical data (default: 24)

**Example:**
```
GET /predict/batch?equipment_ids=pump_001,motor_002,compressor_003&hours_back=48
```

**Response:**
```json
{
  "pump_001": {
    "equipment_id": "pump_001",
    "timestamp": "2024-01-01T12:00:00Z",
    "failure_probability": 0.75,
    "predicted_failure": true,
    "confidence": 0.85,
    "model_used": "ensemble",
    "data_points": 96
  },
  "motor_002": {
    "equipment_id": "motor_002",
    "timestamp": "2024-01-01T12:00:00Z",
    "failure_probability": 0.25,
    "predicted_failure": false,
    "confidence": 0.92,
    "model_used": "ensemble",
    "data_points": 96
  }
}
```

---

### Equipment Management

#### GET /equipment

List all equipment with recent data.

**Response:**
```json
[
  "pump_001",
  "pump_002",
  "motor_001",
  "compressor_001",
  "turbine_001"
]
```

---

#### GET /equipment/{equipment_id}/data

Get recent sensor data for specific equipment.

**Parameters:**
- `equipment_id` (string, required): Equipment identifier
- `hours` (integer, optional): Hours of data to retrieve (default: 24)

**Example:**
```
GET /equipment/pump_001/data?hours=48
```

**Response:**
```json
[
  {
    "timestamp": "2024-01-01T12:00:00Z",
    "equipment_id": "pump_001",
    "equipment_type": "pump",
    "sensor_data": {
      "temperature": 75.5,
      "vibration": 0.25,
      "pressure": 2.8,
      "flow_rate": 125.0,
      "power_consumption": 7.2
    },
    "is_failure": false
  }
]
```

---

### System Metrics

#### GET /metrics

Get system-wide metrics.

**Response:**
```json
{
  "total_equipment": 25,
  "active_equipment": 23,
  "predictions_made": 1547,
  "failure_alerts": 3,
  "system_uptime": "24h"
}
```

---

#### GET /metrics/prometheus

Get Prometheus-compatible metrics.

**Response:**
```
# HELP equipment_total Total number of equipment units
# TYPE equipment_total gauge
equipment_total 25

# HELP equipment_active Number of active equipment units
# TYPE equipment_active gauge
equipment_active 23

# HELP failure_predictions Number of equipment predicted to fail
# TYPE failure_predictions gauge
failure_predictions 3
```

---

### Model Management

#### GET /models/performance

Get model performance information.

**Response:**
```json
{
  "lstm": {
    "loaded": true
  },
  "arima": {
    "loaded": true
  },
  "ensemble_weights": {
    "lstm": 0.7,
    "arima": 0.3
  }
}
```

---

#### PUT /models/weights

Update ensemble model weights.

**Parameters:**
- `lstm_weight` (float, required): Weight for LSTM model (0.0-1.0)
- `arima_weight` (float, required): Weight for ARIMA model (0.0-1.0)

**Note:** Weights must sum to 1.0

**Example:**
```
PUT /models/weights?lstm_weight=0.8&arima_weight=0.2
```

**Response:**
```json
{
  "message": "Updated weights: LSTM=0.8, ARIMA=0.2"
}
```

---

### Simulation Control

#### POST /simulate/start

Start data simulation.

**Parameters:**
- `num_equipment` (integer, optional): Number of equipment to simulate (default: 10)
- `interval_seconds` (integer, optional): Interval between data points (default: 15)

**Response:**
```json
{
  "message": "Simulation started with 10 equipment units"
}
```

---

#### POST /simulate/stop

Stop data simulation.

**Response:**
```json
{
  "message": "Simulation stopped"
}
```

---

## Error Handling

All endpoints return consistent error responses:

```json
{
  "detail": "Error description",
  "status_code": 400,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Common Error Codes

- `400`: Bad Request - Invalid parameters
- `404`: Not Found - Resource not found
- `422`: Validation Error - Invalid request body
- `500`: Internal Server Error - System error

## Rate Limiting

Currently, no rate limiting is implemented. For production:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/predict")
@limiter.limit("10/minute")
async def predict_failure(request: Request, ...):
    ...
```

## SDK Examples

### Python

```python
import requests
import json

class PredictiveMaintenanceClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def get_prediction(self, equipment_id, hours_back=24):
        response = requests.post(
            f"{self.base_url}/predict",
            json={"equipment_id": equipment_id, "hours_back": hours_back}
        )
        return response.json()
    
    def get_equipment_list(self):
        response = requests.get(f"{self.base_url}/equipment")
        return response.json()
    
    def get_system_health(self):
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Usage
client = PredictiveMaintenanceClient()
prediction = client.get_prediction("pump_001")
print(f"Failure probability: {prediction['failure_probability']}")
```

### JavaScript

```javascript
class PredictiveMaintenanceClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async getPrediction(equipmentId, hoursBack = 24) {
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                equipment_id: equipmentId,
                hours_back: hoursBack
            })
        });
        return response.json();
    }
    
    async getEquipmentList() {
        const response = await fetch(`${this.baseUrl}/equipment`);
        return response.json();
    }
    
    async getSystemHealth() {
        const response = await fetch(`${this.baseUrl}/health`);
        return response.json();
    }
}

// Usage
const client = new PredictiveMaintenanceClient();
client.getPrediction('pump_001').then(prediction => {
    console.log(`Failure probability: ${prediction.failure_probability}`);
});
```

### cURL Examples

```bash
# Get prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"equipment_id": "pump_001", "hours_back": 24}'

# Get batch predictions
curl "http://localhost:8000/predict/batch?equipment_ids=pump_001,motor_001"

# Get equipment data
curl "http://localhost:8000/equipment/pump_001/data?hours=48"

# Check system health
curl "http://localhost:8000/health"

# Get system metrics
curl "http://localhost:8000/metrics"

# Update model weights
curl -X PUT "http://localhost:8000/models/weights?lstm_weight=0.8&arima_weight=0.2"
```

## WebSocket Support (Future)

For real-time updates, WebSocket support can be added:

```python
@app.websocket("/ws/predictions/{equipment_id}")
async def websocket_predictions(websocket: WebSocket, equipment_id: str):
    await websocket.accept()
    while True:
        # Send real-time predictions
        prediction = await get_latest_prediction(equipment_id)
        await websocket.send_json(prediction)
        await asyncio.sleep(60)  # Update every minute
```

## API Versioning

For future versions, implement versioning:

```python
# v1 routes
@app.get("/v1/predict")
async def predict_v1(...):
    ...

# v2 routes with enhanced features
@app.get("/v2/predict")
async def predict_v2(...):
    ...
```

## Testing

### Unit Tests

```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction():
    response = client.post("/predict", json={
        "equipment_id": "test_equipment",
        "hours_back": 24
    })
    assert response.status_code in [200, 404]  # 404 if no data
```

### Load Testing

```python
import asyncio
import aiohttp
import time

async def load_test():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(100):
            task = session.post(
                "http://localhost:8000/predict",
                json={"equipment_id": f"test_{i}", "hours_back": 24}
            )
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        print(f"Completed 100 requests in {end_time - start_time:.2f} seconds")

asyncio.run(load_test())
```
