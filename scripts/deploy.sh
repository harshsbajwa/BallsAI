#!/bin/bash

set -e

echo "NBA Pipeline Deployment Script"
echo "=================================="

if ! command -v k3s &> /dev/null; then
    echo "K3s is not installed. Please install K3s first."
    exit 1
fi

echo "Building Docker images..."
docker build -f docker/Dockerfile.api -t nba-pipeline/api:latest .
docker build -f docker/Dockerfile.airflow -t nba-pipeline/airflow:latest .

echo "Deploying to Kubernetes..."

echo "Deploying PostgreSQL..."
kubectl apply -f k8s/postgres.yaml
kubectl wait --for=condition=ready pod -l app=postgres --timeout=300s

echo "Deploying MinIO..."
kubectl apply -f k8s/minio.yaml
kubectl wait --for=condition=ready pod -l app=minio --timeout=300s

echo "Initializing database..."
python scripts/init_database.py

echo "Training ML models..."
python scripts/train_models.py

echo "Deploying FastAPI..."
kubectl apply -f k8s/fastapi.yaml
kubectl wait --for=condition=ready pod -l app=nba-api --timeout=300s

echo "Deploying Airflow..."
kubectl apply -f k8s/airflow.yaml
kubectl wait --for=condition=ready pod -l app=airflow --timeout=600s

echo "Deployment completed!"
echo ""
echo "Service URLs:"
echo "============="
echo "FastAPI:        http://$(kubectl get service nba-api-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):80"
echo "Airflow:        http://$(kubectl get service airflow-webserver-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):8080"
echo "MinIO Console:  http://$(kubectl get service minio-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):9001"
echo ""
echo "Default Credentials:"
echo "==================="
echo "Airflow: admin/admin (default for standalone, may need to be set)"
echo "MinIO: minioadmin/minioadmin"
echo ""
echo "NBA Pipeline is ready to use!"