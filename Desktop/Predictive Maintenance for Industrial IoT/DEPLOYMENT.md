# Deployment Guide

This guide covers different deployment scenarios for the Predictive Maintenance system.

## Quick Start (Development)

```bash
# Clone repository
git clone https://github.com/VIGNESH-MT/Predictive-Maintenance-for-Industrial-IoT.git
cd Predictive-Maintenance-for-Industrial-IoT

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

## Production Deployment

### 1. Environment Configuration

```bash
# Copy and edit environment file
cp env.example .env

# Key production settings
DATABASE_URL=postgresql://user:password@prod-db:5432/predictive_maintenance
REDIS_HOST=prod-redis
KAFKA_SERVERS=prod-kafka-1:9092,prod-kafka-2:9092,prod-kafka-3:9092
SECRET_KEY=your-secure-secret-key
CORS_ORIGINS=https://yourdomain.com
```

### 2. SSL/TLS Setup

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
      - grafana
```

### 3. Resource Limits

```yaml
# Production resource limits
api:
  deploy:
    replicas: 3
    resources:
      limits:
        memory: 2G
        cpus: '1.0'
      reservations:
        memory: 1G
        cpus: '0.5'
```

## Cloud Deployment

### AWS Deployment

#### Using ECS

```bash
# Build and push images
docker build -t your-account.dkr.ecr.region.amazonaws.com/predictive-maintenance:latest .
docker push your-account.dkr.ecr.region.amazonaws.com/predictive-maintenance:latest

# Deploy using ECS CLI
ecs-cli compose --file docker-compose.aws.yml service up
```

#### Using EKS

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: predictive-maintenance-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: predictive-maintenance-api
  template:
    metadata:
      labels:
        app: predictive-maintenance-api
    spec:
      containers:
      - name: api
        image: your-account.dkr.ecr.region.amazonaws.com/predictive-maintenance:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
```

### Azure Deployment

#### Using Container Instances

```bash
# Create resource group
az group create --name predictive-maintenance --location eastus

# Deploy container
az container create \
  --resource-group predictive-maintenance \
  --name predictive-maintenance-api \
  --image your-registry.azurecr.io/predictive-maintenance:latest \
  --ports 8000 \
  --environment-variables DATABASE_URL=your-db-url
```

### GCP Deployment

#### Using Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/your-project/predictive-maintenance

# Deploy to Cloud Run
gcloud run deploy predictive-maintenance \
  --image gcr.io/your-project/predictive-maintenance \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'predictive-maintenance'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics/prometheus'
```

### Grafana Alerts

```json
{
  "alert": {
    "name": "High Failure Probability",
    "message": "Equipment failure probability > 80%",
    "frequency": "10s",
    "conditions": [
      {
        "query": {
          "queryType": "",
          "refId": "A",
          "expr": "failure_predictions > 0.8"
        }
      }
    ]
  }
}
```

## Security Considerations

### 1. Database Security

```bash
# Use strong passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32)

# Enable SSL
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require
```

### 2. API Security

```python
# Enable HTTPS only
app.add_middleware(HTTPSRedirectMiddleware)

# Add security headers
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["yourdomain.com"]
)
```

### 3. Network Security

```yaml
# docker-compose.prod.yml
networks:
  internal:
    driver: bridge
    internal: true
  external:
    driver: bridge

services:
  api:
    networks:
      - internal
      - external
  
  postgres:
    networks:
      - internal  # Only internal access
```

## Backup and Recovery

### Database Backup

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

docker exec postgres pg_dump -U postgres predictive_maintenance > \
  $BACKUP_DIR/backup_$DATE.sql

# Keep only last 7 days
find $BACKUP_DIR -name "backup_*.sql" -mtime +7 -delete
```

### Model Backup

```bash
# Backup trained models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Upload to cloud storage
aws s3 cp models_backup_*.tar.gz s3://your-backup-bucket/
```

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  api:
    deploy:
      replicas: 5
    
  data_pipeline:
    deploy:
      replicas: 3
```

### Load Balancing

```nginx
# nginx/nginx.conf
upstream api_backend {
    server api_1:8000;
    server api_2:8000;
    server api_3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://api_backend;
    }
}
```

## Troubleshooting

### Common Issues

1. **Service Won't Start**
   ```bash
   # Check logs
   docker-compose logs service_name
   
   # Check resource usage
   docker stats
   ```

2. **Database Connection Issues**
   ```bash
   # Test connection
   docker exec -it postgres psql -U postgres -d predictive_maintenance
   
   # Check network
   docker network ls
   docker network inspect network_name
   ```

3. **Model Loading Errors**
   ```bash
   # Check model files
   ls -la models/
   
   # Check permissions
   docker exec -it api ls -la /app/models/
   ```

### Performance Tuning

1. **Database Optimization**
   ```sql
   -- Add indexes
   CREATE INDEX CONCURRENTLY idx_sensor_data_timestamp_equipment 
   ON sensor_data(timestamp, equipment_id);
   
   -- Analyze tables
   ANALYZE sensor_data;
   ```

2. **Redis Optimization**
   ```bash
   # Increase memory
   redis-cli CONFIG SET maxmemory 2gb
   redis-cli CONFIG SET maxmemory-policy allkeys-lru
   ```

3. **API Optimization**
   ```python
   # Use connection pooling
   DATABASE_POOL_SIZE=20
   DATABASE_MAX_OVERFLOW=30
   ```

## Health Checks

### Service Health

```bash
#!/bin/bash
# health_check.sh

services=("api" "postgres" "redis" "kafka" "grafana")

for service in "${services[@]}"; do
    if docker-compose ps $service | grep -q "Up"; then
        echo "✅ $service is running"
    else
        echo "❌ $service is down"
    fi
done
```

### API Health

```bash
# Check API health
curl -f http://localhost:8000/health || exit 1

# Check prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"equipment_id": "test", "hours_back": 24}'
```

## Maintenance

### Regular Tasks

1. **Daily**
   - Check service health
   - Monitor disk usage
   - Review error logs

2. **Weekly**
   - Database maintenance
   - Model performance review
   - Security updates

3. **Monthly**
   - Full backup verification
   - Performance optimization
   - Capacity planning

### Update Procedure

```bash
# 1. Backup current state
docker-compose exec postgres pg_dump -U postgres predictive_maintenance > backup.sql

# 2. Pull latest images
docker-compose pull

# 3. Rolling update
docker-compose up -d --no-deps api
docker-compose up -d --no-deps data_pipeline

# 4. Verify deployment
curl -f http://localhost:8000/health
```
