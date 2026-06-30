# Heart Disease MLOps Assignment

This project builds an end-to-end MLOps pipeline for the UCI Heart Disease dataset and is designed to satisfy the assignment requirements for EDA, model training, MLflow tracking, FastAPI deployment, Dockerization, CI/CD, and reporting.

## Project structure

- data/ for dataset download and storage
- notebooks/ for EDA and experimentation
- src/data/ for dataset loading and preprocessing
- src/features/ for feature engineering and preprocessing pipelines
- src/models/ for training and MLflow integration
- src/api/ for FastAPI prediction API
- tests/ for unit tests
- docker/ and k8s/ for deployment assets
- .github/workflows/ for CI

## Quick start

1. Create a Python environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest`
4. Start the API: `uvicorn src.api.main:app --reload`
5. Launch MLflow UI: `mlflow ui --backend-store-uri file:./mlruns`

## Current implementation highlights

- Dataset download helper in data/raw/download_dataset.py
- Baseline training for Logistic Regression and Random Forest
- MLflow experiment tracking for each trained model
- FastAPI prediction endpoint
- Docker and Kubernetes starter files
- Automated tests via pytest
