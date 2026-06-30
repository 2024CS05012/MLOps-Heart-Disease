# Setup Guide

## Prerequisites
- Python 3.9+
- Docker Desktop (optional for container testing)
- Kubernetes tools such as Minikube or Docker Desktop Kubernetes (optional)

## Installation
```bash
cd /Users/shyamgupta/Documents/Assignmnet/MLOps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tests
```bash
pytest -q
```

## Start the API
```bash
uvicorn src.api.main:app --reload
```

## Run MLflow UI
```bash
mlflow ui --backend-store-uri file:./mlruns
```

## Build Docker image
```bash
docker build -f docker/Dockerfile -t heart-disease-api .
```
