# Heart Disease MLOps Assignment

This repository implements an end-to-end MLOps workflow for predicting heart disease risk using the UCI Cleveland Heart Disease dataset. The project covers the full machine learning lifecycle, including data acquisition, exploratory data analysis, preprocessing, model training and evaluation, MLflow experiment tracking, FastAPI-based deployment, Docker and Kubernetes packaging, CI/CD automation, and monitoring.

## Project summary

- Build a binary classification model to predict whether a patient is likely to have heart disease.
- Compare and evaluate multiple models such as Logistic Regression and Random Forest.
- Package the trained pipeline for reproducible inference through a REST API.
- Deploy the service locally using Docker and Kubernetes, with Prometheus-compatible monitoring.
- Automate validation and build checks with GitHub Actions.

## Documentation links

- Setup guide: [docs/SETUP.md](docs/SETUP.md)
- Final report: [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md)
- Architecture notes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- CI workflow: [.github/workflows/ci.yml](.github/workflows/ci.yml)

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
- docs/ for the final report, architecture notes, and setup guidance

## Quick start

Use Python 3.11 for the local environment. The project supports Python 3.9
through 3.13; Python 3.14 is currently avoided.

1. Create and activate a Python virtual environment:
   - `python3.11 -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset: `python data/raw/download_dataset.py`
4. Generate EDA figures: `python -m src.data.generate_eda_artifacts`
5. Train and log models: `python -m src.models.train`
6. Run tests: `pytest`
7. Start the API: `uvicorn src.api.main:app --reload`
8. Launch MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`

### API endpoints

- `GET /health` returns service health.
- `POST /predict` returns the predicted heart disease class and confidence.
- `GET /metrics` exposes Prometheus metrics.

## Current implementation highlights

- UCI Cleveland dataset download and normalization to a binary target
- Reusable sklearn preprocessing with imputation, scaling, and one-hot encoding
- GridSearchCV tuning for Logistic Regression and Random Forest
- Cross-validation and held-out evaluation metrics
- MLflow experiment tracking for metrics, params, plots, and model artifacts
- A FastAPI prediction endpoint with a health check
- Prometheus-compatible API metrics and request logging
- Docker, Docker Compose, and Kubernetes deployment files
- Automated linting, tests, training validation, and Docker build in GitHub Actions CI

## Assignment submission notes

- The repository is published at https://github.com/2024CS05012/MLOps-Heart-Disease
- Report material, architecture notes, and setup guidance are available in the docs/ folder
