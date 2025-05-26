#!/bin/bash
# deploy.sh
set -e

echo "Starting NBA Betting ML Pipeline Deployment"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f "service-account-key.json" ]; then
    print_error "service-account-key.json not found!"
    echo "Please download your Google Cloud service account key and place it in the project root"
    exit 1
fi

if [ ! -f ".env" ]; then
    print_error ".env file not found!"
    echo "Please create a .env file with required environment variables"
    exit 1
fi

print_status "Prerequisites check passed"

# Load environment variables
source .env

# Validate required environment variables
required_vars=("PROJECT_ID" "DATASET_ID" "GOOGLE_APPLICATION_CREDENTIALS")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        print_error "Required environment variable $var is not set"
        exit 1
    fi
done

print_status "Environment variables validated"

# Create necessary directories
echo "Creating directories..."
mkdir -p airflow/logs airflow/plugins models data/logs

# Set proper ownership for Airflow
echo "Setting permissions..."
export AIRFLOW_UID=$(id -u)
sudo chown -R $AIRFLOW_UID:0 airflow/logs airflow/plugins || true
chmod -R 755 airflow/logs airflow/plugins

print_status "Directories and permissions set"

# Initialize BigQuery schema
echo "Setting up BigQuery..."
if command -v uv &> /dev/null; then
    uv run python -m data.schema
else
    python -m data.schema
fi

print_status "BigQuery schema initialized"

# Build and start Docker services
echo "Building and starting Docker services..."
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d

print_status "Docker services started"

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 45

# Check service health
echo "Checking service health..."
docker-compose ps

# Initialize Airflow
echo "Initializing Airflow..."
docker-compose run --rm airflow-cli airflow db init

# Create admin user
docker-compose run --rm airflow-cli airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin123

print_status "Airflow initialized"

# Enable DAG
echo "Enabling NBA Betting ML Pipeline DAG..."
docker-compose run --rm airflow-cli airflow dags unpause nba_betting_ml_pipeline

print_status "DAG enabled"

echo ""
echo "Deployment completed successfully!"
echo ""
echo "Access points:"
echo "   Airflow UI: http://localhost:8080 (admin/admin123)"
echo "   Flower (Celery): http://localhost:5555"
echo ""
echo "Management commands:"
echo "   Monitor logs: docker-compose logs -f airflow-scheduler"
echo "   Trigger DAG: docker-compose run --rm airflow-cli airflow dags trigger nba_betting_ml_pipeline"
echo "   Stop services: docker-compose down"
echo ""
print_status "Ready to make predictions!"
