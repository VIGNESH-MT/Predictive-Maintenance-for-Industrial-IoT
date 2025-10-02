@echo off
REM Predictive Maintenance for Industrial IoT - Windows Setup Script

echo.
echo 🚀 Setting up Predictive Maintenance for Industrial IoT
echo ==================================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)
echo [INFO] Docker found

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)
echo [INFO] Docker Compose found

REM Create .env file if it doesn't exist
if not exist .env (
    echo [INFO] Creating .env file from template...
    copy env.example .env
    echo [INFO] Please edit .env file with your specific configuration
) else (
    echo [INFO] .env file already exists
)

REM Create necessary directories
echo [INFO] Creating directories...
if not exist data mkdir data
if not exist logs mkdir logs
if not exist models mkdir models
if not exist notebooks mkdir notebooks
if not exist notebooks\work mkdir notebooks\work

REM Build and start services
echo [INFO] Building Docker images...
docker-compose build

echo [INFO] Starting core services...
docker-compose up -d postgres redis kafka zookeeper

echo [INFO] Waiting for services to be ready...
timeout /t 30 /nobreak >nul

echo [INFO] Starting application services...
docker-compose up -d api data_pipeline prometheus grafana

echo [INFO] All services started

REM Check service health
echo [INFO] Checking service health...
timeout /t 10 /nobreak >nul

REM Test API health
echo [INFO] Testing API health...
for /l %%i in (1,1,10) do (
    curl -s -f http://localhost:8000/health >nul 2>&1
    if not errorlevel 1 (
        echo [INFO] API is healthy
        goto :api_ready
    )
    echo [INFO] Waiting for API to be ready... (attempt %%i/10)
    timeout /t 10 /nobreak >nul
)
echo [WARNING] API health check failed after 10 attempts

:api_ready

REM Generate sample data
echo [INFO] Starting data simulation...
curl -s -X POST "http://localhost:8000/simulate/start?num_equipment=5&interval_seconds=10" >nul 2>&1

echo.
echo 🎉 Setup Complete!
echo.
echo [INFO] Your Predictive Maintenance system is now running!
echo.
echo Access URLs:
echo   📊 Grafana Dashboard:    http://localhost:3000 (admin/admin)
echo   🔧 API Documentation:    http://localhost:8000/docs
echo   📈 Prometheus Metrics:   http://localhost:9090
echo.
echo Quick Commands:
echo   View logs:               docker-compose logs -f
echo   Stop services:           docker-compose down
echo   Restart services:        docker-compose restart
echo   Check status:            docker-compose ps
echo.
echo API Examples:
echo   Health check:            curl http://localhost:8000/health
echo   List equipment:          curl http://localhost:8000/equipment
echo.
echo [INFO] For more information, see README.md and API_DOCUMENTATION.md
echo.
echo Setup completed successfully! 🎉
echo.
pause
