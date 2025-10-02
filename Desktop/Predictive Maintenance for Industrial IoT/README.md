# Predictive Maintenance for Industrial IoT

A comprehensive predictive maintenance system for Industrial IoT that combines LSTM and ARIMA models for time-series forecasting to predict equipment failures and optimize maintenance schedules.

## 🚀 Features

- **Real-time Data Processing**: Kafka-based streaming pipeline for IoT sensor data
- **Dual Model Approach**: LSTM (deep learning) and ARIMA (statistical) models for robust predictions
- **Ensemble Predictions**: Combines both models for improved accuracy and confidence scoring
- **Interactive Dashboard**: Grafana dashboards for real-time monitoring and visualization
- **Scalable Architecture**: Docker-based microservices architecture
- **RESTful API**: FastAPI-based API for predictions and system management
- **Data Simulation**: Realistic IoT sensor data generator for testing and development
- **Automated Training**: Periodic model retraining with performance monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   IoT Sensors   │───▶│     Kafka       │───▶│  Data Pipeline  │
│   (Simulated)   │    │   (Streaming)   │    │   (Processing)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Grafana     │◀───│   PostgreSQL    │◀───│     Redis       │
│  (Dashboard)    │    │   (Database)    │    │    (Cache)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                                              │
         │              ┌─────────────────┐             │
         └──────────────│   FastAPI       │◀────────────┘
                        │     (API)       │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Prediction      │
                        │ Service         │
                        │ (LSTM + ARIMA)  │
                        └─────────────────┘
```

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- 8GB+ RAM recommended
- 10GB+ free disk space

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/VIGNESH-MT/Predictive-Maintenance-for-Industrial-IoT.git
cd Predictive-Maintenance-for-Industrial-IoT
```

### 2. Environment Setup

```bash
# Copy environment template
cp env.example .env

# Edit configuration as needed
nano .env
```

### 3. Start the System

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Access the Services

- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **API Documentation**: http://localhost:8000/docs
- **Prometheus Metrics**: http://localhost:9090
- **Jupyter Notebooks**: http://localhost:8888 (development profile)

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Core Services
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/predictive_maintenance
REDIS_HOST=redis
KAFKA_SERVERS=kafka:29092

# Simulation
SIMULATE_DATA=true
NUM_EQUIPMENT=10
SIMULATION_INTERVAL=15

# Model Parameters
LSTM_SEQUENCE_LENGTH=24
LSTM_PREDICTION_HORIZON=6
```

### Docker Profiles

- **Default**: Core services (API, database, monitoring)
- **Development**: Includes Jupyter notebooks
- **Training**: Includes model training service

```bash
# Start with development profile
docker-compose --profile development up -d

# Start training service
docker-compose --profile training up model_trainer
```

## 📊 Data Flow

### 1. Data Generation
- IoT sensors (or simulator) generate time-series data
- Includes temperature, vibration, pressure, power consumption, etc.
- Realistic failure patterns and equipment degradation

### 2. Data Ingestion
- Kafka streams sensor data in real-time
- Redis caches recent data for fast access
- PostgreSQL stores historical data

### 3. Prediction Pipeline
- LSTM model analyzes sequential patterns
- ARIMA model captures statistical trends
- Ensemble method combines predictions with confidence scoring

### 4. Monitoring & Alerts
- Grafana dashboards visualize metrics
- Prometheus collects system metrics
- API endpoints provide programmatic access

## 🤖 Models

### LSTM Model
- **Architecture**: Multi-layer LSTM with dropout and batch normalization
- **Input**: 24-hour sequence of sensor readings
- **Output**: 6-hour ahead failure probability
- **Features**: Temperature, vibration, pressure, power consumption, etc.

### ARIMA Model
- **Type**: Auto-regressive Integrated Moving Average
- **Auto-tuning**: Automatic order selection using AIC criterion
- **Stationarity**: Automatic differencing for non-stationary series
- **Outlier handling**: Z-score based outlier detection and correction

### Ensemble Method
- **Weighted Average**: Configurable weights (default: 70% LSTM, 30% ARIMA)
- **Confidence Scoring**: Based on model agreement
- **Fallback**: Single model predictions when one model fails

## 📈 API Usage

### Get Predictions

```bash
# Single equipment prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"equipment_id": "pump_001", "hours_back": 24}'

# Batch predictions
curl "http://localhost:8000/predict/batch?equipment_ids=pump_001,motor_001&hours_back=24"
```

### System Monitoring

```bash
# Health check
curl http://localhost:8000/health

# System metrics
curl http://localhost:8000/metrics

# Equipment list
curl http://localhost:8000/equipment
```

### Model Management

```bash
# Model performance
curl http://localhost:8000/models/performance

# Update ensemble weights
curl -X PUT "http://localhost:8000/models/weights?lstm_weight=0.8&arima_weight=0.2"
```

## 🔬 Development

### Local Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set Python path
export PYTHONPATH=$PWD

# Run individual components
python src/data_simulation/sensor_simulator.py
python src/models/lstm_model.py
python -m uvicorn src.api.main:app --reload
```

### Jupyter Notebooks

```bash
# Start with development profile
docker-compose --profile development up -d

# Access Jupyter at http://localhost:8888
# Notebooks are in ./notebooks directory
```

### Model Training

```bash
# Train models with custom parameters
docker-compose run --rm model_trainer \
  -e NUM_EQUIPMENT=50 \
  -e DURATION_DAYS=90

# Or run locally
python scripts/train_models.py
```

## 📊 Monitoring & Dashboards

### Grafana Dashboards

The system includes pre-configured Grafana dashboards:

1. **Equipment Overview**: Total equipment, active units, failure predictions
2. **Sensor Metrics**: Temperature, vibration, pressure trends
3. **Prediction Analytics**: Failure probability over time, model performance
4. **System Health**: API metrics, database performance, pipeline status

### Key Metrics

- **Equipment Status**: Active vs. total equipment count
- **Failure Predictions**: Number of equipment predicted to fail
- **Model Accuracy**: Prediction accuracy over time
- **System Performance**: API response times, throughput
- **Data Quality**: Missing data points, outlier detection

### Alerts

Configure Grafana alerts for:
- High failure probability (>80%)
- Equipment offline for >1 hour
- Model prediction accuracy drop
- System resource usage

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/models/
python -m pytest tests/pipeline/
python -m pytest tests/api/
```

### Integration Tests

```bash
# Test complete pipeline
python tests/integration/test_pipeline.py

# Test API endpoints
python tests/integration/test_api.py
```

### Load Testing

```bash
# API load testing
python tests/load/test_api_load.py

# Pipeline throughput testing
python tests/load/test_pipeline_load.py
```

## 🚀 Deployment

### Production Deployment

1. **Security Configuration**:
   ```bash
   # Generate secure passwords
   openssl rand -base64 32  # For database
   openssl rand -base64 32  # For JWT secret
   ```

2. **Resource Scaling**:
   ```yaml
   # docker-compose.prod.yml
   api:
     deploy:
       replicas: 3
       resources:
         limits:
           memory: 2G
           cpus: '1.0'
   ```

3. **SSL/TLS Setup**:
   ```bash
   # Use reverse proxy (nginx/traefik)
   # Configure SSL certificates
   # Update CORS origins
   ```

### Cloud Deployment

#### AWS
- Use ECS/EKS for container orchestration
- RDS for PostgreSQL
- ElastiCache for Redis
- MSK for Kafka

#### Azure
- Use Container Instances/AKS
- Azure Database for PostgreSQL
- Azure Cache for Redis
- Event Hubs for streaming

#### GCP
- Use Cloud Run/GKE
- Cloud SQL for PostgreSQL
- Memorystore for Redis
- Pub/Sub for messaging

## 📚 Documentation

### API Documentation
- Interactive API docs: http://localhost:8000/docs
- OpenAPI specification: http://localhost:8000/openapi.json

### Model Documentation
- [LSTM Model Architecture](docs/models/lstm.md)
- [ARIMA Model Configuration](docs/models/arima.md)
- [Ensemble Method](docs/models/ensemble.md)

### Deployment Guides
- [Docker Deployment](docs/deployment/docker.md)
- [Kubernetes Deployment](docs/deployment/kubernetes.md)
- [Cloud Deployment](docs/deployment/cloud.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Use type hints
- Add logging for debugging

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Statsmodels team for ARIMA implementation
- FastAPI team for the excellent web framework
- Grafana team for visualization tools
- Apache Kafka team for streaming platform

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/VIGNESH-MT/Predictive-Maintenance-for-Industrial-IoT/issues)
- **Discussions**: [GitHub Discussions](https://github.com/VIGNESH-MT/Predictive-Maintenance-for-Industrial-IoT/discussions)
- **Email**: [Contact](mailto:support@example.com)

## 🗺️ Roadmap

- [ ] **v1.1**: Real-time anomaly detection
- [ ] **v1.2**: Multi-variate time series forecasting
- [ ] **v1.3**: Automated hyperparameter tuning
- [ ] **v1.4**: Edge computing support
- [ ] **v2.0**: Reinforcement learning for maintenance scheduling

---

**Built with ❤️ for Industrial IoT and Predictive Maintenance**
