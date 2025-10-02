#!/bin/bash

# Predictive Maintenance for Industrial IoT - Setup Script
# This script sets up the complete system

set -e

echo "🚀 Setting up Predictive Maintenance for Industrial IoT"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_status "Docker found: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_status "Docker Compose found: $(docker-compose --version)"
    
    # Check available disk space (need at least 10GB)
    available_space=$(df . | tail -1 | awk '{print $4}')
    if [ $available_space -lt 10485760 ]; then  # 10GB in KB
        print_warning "Less than 10GB disk space available. System may run out of space."
    fi
    
    # Check available memory (recommend at least 8GB)
    if command -v free &> /dev/null; then
        available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
        if [ $available_memory -lt 4096 ]; then  # 4GB
            print_warning "Less than 4GB memory available. Performance may be affected."
        fi
    fi
    
    print_status "Prerequisites check completed"
}

# Setup environment
setup_environment() {
    print_header "Setting up Environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        print_status "Creating .env file from template..."
        cp env.example .env
        print_status "Please edit .env file with your specific configuration"
    else
        print_status ".env file already exists"
    fi
    
    # Create necessary directories
    print_status "Creating directories..."
    mkdir -p data logs models notebooks/work
    
    # Set permissions
    chmod +x scripts/*.py
    
    print_status "Environment setup completed"
}

# Build and start services
start_services() {
    print_header "Building and Starting Services..."
    
    print_status "Building Docker images..."
    docker-compose build
    
    print_status "Starting core services..."
    docker-compose up -d postgres redis kafka zookeeper
    
    print_status "Waiting for services to be ready..."
    sleep 30
    
    print_status "Starting application services..."
    docker-compose up -d api data_pipeline prometheus grafana
    
    print_status "All services started"
}

# Check service health
check_services() {
    print_header "Checking Service Health..."
    
    services=("postgres" "redis" "kafka" "api" "grafana" "prometheus")
    
    for service in "${services[@]}"; do
        if docker-compose ps $service | grep -q "Up"; then
            print_status "$service is running"
        else
            print_error "$service is not running"
        fi
    done
    
    # Test API health
    print_status "Testing API health..."
    for i in {1..10}; do
        if curl -s -f http://localhost:8000/health > /dev/null; then
            print_status "API is healthy"
            break
        else
            if [ $i -eq 10 ]; then
                print_warning "API health check failed after 10 attempts"
            else
                print_status "Waiting for API to be ready... (attempt $i/10)"
                sleep 10
            fi
        fi
    done
}

# Generate sample data
generate_sample_data() {
    print_header "Generating Sample Data..."
    
    print_status "Starting data simulation..."
    curl -s -X POST "http://localhost:8000/simulate/start?num_equipment=5&interval_seconds=10" > /dev/null
    
    print_status "Sample data generation started"
    print_status "Data will be generated continuously. You can stop it later via the API."
}

# Train initial models
train_models() {
    print_header "Training Initial Models..."
    
    print_status "This may take several minutes..."
    
    if docker-compose --profile training up model_trainer; then
        print_status "Model training completed successfully"
    else
        print_warning "Model training failed. You can train models later using:"
        print_warning "docker-compose --profile training up model_trainer"
    fi
}

# Display access information
show_access_info() {
    print_header "🎉 Setup Complete!"
    echo ""
    print_status "Your Predictive Maintenance system is now running!"
    echo ""
    echo "Access URLs:"
    echo "  📊 Grafana Dashboard:    http://localhost:3000 (admin/admin)"
    echo "  🔧 API Documentation:    http://localhost:8000/docs"
    echo "  📈 Prometheus Metrics:   http://localhost:9090"
    echo "  💻 Jupyter Notebooks:    http://localhost:8888 (if development profile)"
    echo ""
    echo "Quick Commands:"
    echo "  View logs:               docker-compose logs -f"
    echo "  Stop services:           docker-compose down"
    echo "  Restart services:        docker-compose restart"
    echo "  Check status:            docker-compose ps"
    echo ""
    echo "API Examples:"
    echo "  Health check:            curl http://localhost:8000/health"
    echo "  List equipment:          curl http://localhost:8000/equipment"
    echo "  Get prediction:          curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{\"equipment_id\":\"pump_001\",\"hours_back\":24}'"
    echo ""
    print_status "For more information, see README.md and API_DOCUMENTATION.md"
}

# Main setup function
main() {
    echo ""
    print_header "🏭 Predictive Maintenance for Industrial IoT Setup"
    echo ""
    
    # Parse command line arguments
    SKIP_TRAINING=false
    DEVELOPMENT_MODE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-training)
                SKIP_TRAINING=true
                shift
                ;;
            --development)
                DEVELOPMENT_MODE=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --skip-training    Skip initial model training"
                echo "  --development      Start with development profile (includes Jupyter)"
                echo "  --help            Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run setup steps
    check_prerequisites
    setup_environment
    start_services
    check_services
    generate_sample_data
    
    if [ "$SKIP_TRAINING" = false ]; then
        train_models
    else
        print_status "Skipping model training (use --skip-training flag)"
    fi
    
    if [ "$DEVELOPMENT_MODE" = true ]; then
        print_status "Starting development services..."
        docker-compose --profile development up -d jupyter
    fi
    
    show_access_info
    
    echo ""
    print_status "Setup completed successfully! 🎉"
    echo ""
}

# Run main function
main "$@"
