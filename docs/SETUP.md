# Setup Guide

This guide is written so a fresh user can run the full assignment project on a
local machine and verify all 8 tasks.

Run commands from the repository root unless a step says otherwise.

## 1. Prerequisites
Make sure the following are installed:
- Python 3.9 Python 3.11 is recommended and is used by CI and Docker.
- Git
- Docker Desktop with Kubernetes enabled
- `kubectl`
- `curl`

Check your tools:
```bash
python --version
git --version
docker --version
docker compose version
kubectl version --client
```

## 2. Clone and Set Up the Environment
Clone the repository, then enter the project directory:
```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Create and activate a virtual environment:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
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

## 5. Task 3: Experiment Tracking With MLflow
Start MLflow locally in a separate terminal with the virtual environment
activated:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Then open:
- http://127.0.0.1:5000

MLflow will log parameters, metrics, evaluation plots, and model artifacts for
each run.

## 6. Task 4: Model Packaging & Reproducibility
The repository saves trained models in the `models/` directory as joblib
artifacts. To reproduce the workflow:
```bash
python data/raw/download_dataset.py
python -m src.data.generate_eda_artifacts
python -m src.models.train
```

The required reproducibility components are:
- `requirements.txt`
- preprocessing pipeline in `src/features/engineering.py`
- saved model artifacts in `models/`

## 7. Task 5: CI/CD and Automated Testing
Run the tests locally:
```bash
pytest -q
```

The CI workflow is defined in:
- `.github/workflows/ci.yml`

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

Keep this terminal open while testing the standalone container.

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

When you are done testing this standalone container, stop it before starting the
Docker Compose monitoring stack in Task 8:
```bash
docker stop heart-disease-api
```

If Docker says the container does not exist or is not running, continue with the
next task.

## 9. Task 7: Production Deployment With Kubernetes
Build the local image first if you have not already done Task 6. Docker Desktop
Kubernetes can use this local image directly:
```bash
docker build -f docker/Dockerfile -t heart-disease-api:local .
```

Apply the deployment manifest:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/heart-disease-api
kubectl get svc heart-disease-api-service
```

For local port-forwarding, keep this command running in its own terminal:
```bash
kubectl port-forward svc/heart-disease-api-service 18000:8000
```
Then test from another terminal health:
```bash
curl http://127.0.0.1:18000/health
```

Then test from another terminal for prediction no heart disease:
```bash
curl http://127.0.0.1:18000/health
curl -X POST http://127.0.0.1:18000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```
Then test from another terminal for prediction heart disease:
```bash
curl -X POST http://127.0.0.1:18000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}'
```

## 10. Task 8: Monitoring & Logging
If the standalone container from Task 6 is still running, stop it first:
```bash
docker stop heart-disease-api
```

If Docker says the container does not exist or is not running, continue.

Run the monitoring stack:
```bash
docker compose -f docker/docker-compose.yml up -d --build
```

Verify the metrics endpoint:
```bash
curl http://127.0.0.1:18080/metrics
```

Verify Prometheus target health:
```bash
curl http://127.0.0.1:9090/api/v1/targets
```

Open the interfaces:
- API docs: http://127.0.0.1:18080/docs
- Prometheus: http://127.0.0.1:9090


## 11. Quick Verification Checklist
If you want to verify everything quickly from a clean checkout, run:
```bash
python data/raw/download_dataset.py
python -m src.data.generate_eda_artifacts
python -m src.models.train
pytest -q
docker build -f docker/Dockerfile -t heart-disease-api:local .
docker compose -f docker/docker-compose.yml up -d --build
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/heart-disease-api
```
