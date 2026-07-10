# Setup Guide

This guide is written so a fresh user can run the full assignment project on a local machine and verify all 8 tasks.

## 1. Prerequisites
Make sure the following are installed:
- Python 3.9 or newer
- Docker Desktop
- Kubernetes support (Docker Desktop Kubernetes is enough)
- Git

## 2. Clone and set up the environment
```bash
cd /path/to/your/project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Task 1: Data Acquisition & EDA
Download the dataset and generate the EDA artifacts:
```bash
python data/raw/download_dataset.py
python -m src.data.generate_eda_artifacts
```

Generated figures will be written to:
- artifacts/eda/feature_histograms.png
- artifacts/eda/missing_values.png
- artifacts/eda/class_distribution.png
- artifacts/eda/correlation_heatmap.png
- artifacts/eda/thalach_by_target.png

These support the EDA part of the assignment.

## 4. Task 2: Feature Engineering & Model Development
Train the models and produce evaluation metrics:
```bash
python -m src.models.train
```

This trains:
- Logistic Regression
- Random Forest

It also saves evaluation artifacts under:
- artifacts/logistic_regression/
- artifacts/random_forest/

## 5. Task 3: Experiment Tracking with MLflow
Start MLflow locally:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Then open:
- http://127.0.0.1:5000

MLflow will log parameters, metrics, evaluation plots, and model artifacts for each run.

## 6. Task 4: Model Packaging & Reproducibility
The repository already saves trained models in the models/ directory as joblib artifacts. To reproduce the workflow:
```bash
python data/raw/download_dataset.py
python -m src.data.generate_eda_artifacts
python -m src.models.train
```

The required reproducibility components are:
- requirements.txt
- preprocessing pipeline in src/features/engineering.py
- saved model artifacts in models/

## 7. Task 5: CI/CD and Automated Testing
Run the tests locally:
```bash
pytest -q
```

The CI workflow is defined in:
- .github/workflows/ci.yml

It runs:
- linting with ruff
- dataset download
- EDA artifact generation
- model training
- pytest
- Docker build validation

## 8. Task 6: Containerization
Build and run the FastAPI container locally:
```bash
docker build -f docker/Dockerfile -t heart-disease-api:local .
docker run --rm -p 8000:8000 --name heart-disease-api heart-disease-api:local
```

Test the API:
```bash
curl http://127.0.0.1:8000/health
```

Sample prediction:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

## 9. Task 7: Production Deployment with Kubernetes
Apply the deployment manifest:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods
kubectl get svc heart-disease-api-service
```

For local port-forwarding:
```bash
kubectl port-forward svc/heart-disease-api-service 18000:8000
```

Then test:
```bash
curl http://127.0.0.1:18000/health
curl -X POST http://127.0.0.1:18000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

## 10. Task 8: Monitoring & Logging
Run the monitoring stack:
```bash
docker compose -f docker/docker-compose.yml up -d --build
```

Verify the metrics endpoint:
```bash
curl http://127.0.0.1:8000/metrics
```

Verify Prometheus target health:
```bash
curl http://127.0.0.1:9090/api/v1/targets
```

Open the interfaces:
- API docs: http://127.0.0.1:8000/docs
- Prometheus: http://127.0.0.1:9090

## 11. Quick verification checklist
If you want to verify everything quickly, run:
```bash
python data/raw/download_dataset.py
python -m src.data.generate_eda_artifacts
python -m src.models.train
pytest -q
docker build -f docker/Dockerfile -t heart-disease-api:local .
docker compose -f docker/docker-compose.yml up -d --build
kubectl apply -f k8s/deployment.yaml
```
