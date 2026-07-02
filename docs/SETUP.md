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

## Download data and generate EDA
```bash
python data/raw/download_dataset.py
python -m src.data.generate_eda_artifacts
```

EDA figures are written to `artifacts/eda/`.

## Train models and log experiments
```bash
python -m src.models.train
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

MLflow records model parameters, metrics, evaluation plots, and sklearn model artifacts.

## Run tests
```bash
pytest -q
```

## Start the API
```bash
uvicorn src.api.main:app --reload
```

Sample prediction:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

Metrics endpoint:
```bash
curl http://127.0.0.1:8000/metrics
```

## Build Docker image
```bash
docker build -f docker/Dockerfile -t heart-disease-api .
docker run --rm -p 8000:8000 heart-disease-api
```

## Run API with Prometheus
```bash
docker compose -f docker/docker-compose.yml up --build
```

- API: http://127.0.0.1:8000/docs
- Prometheus: http://127.0.0.1:9090

## Kubernetes deployment
```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods
kubectl get svc heart-disease-api-service
```

For Minikube:
```bash
minikube service heart-disease-api-service
```
