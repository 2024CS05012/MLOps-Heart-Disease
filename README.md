# Heart Disease MLOps Assignment

This repository implements an end-to-end MLOps workflow for the UCI Heart Disease dataset. It covers data loading, preprocessing, model training, experiment tracking with MLflow, API serving with FastAPI, containerization, CI/CD, and reporting assets for the assignment.

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
- docs/ for the report outline and setup notes

## Quick start

1. Create and activate a Python virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest`
4. Start the API: `uvicorn src.api.main:app --reload`
5. Launch MLflow UI: `mlflow ui --backend-store-uri file:./mlruns`

## Current implementation highlights

- A reusable data loader and preprocessing module
- Baseline training for Logistic Regression and Random Forest
- MLflow experiment tracking for each trained model
- A FastAPI prediction endpoint with a health check
- Docker and Kubernetes starter files
- Automated tests via pytest and GitHub Actions CI

## Assignment submission notes

- The repository is published at https://github.com/2024CS05012/MLOps-Heart-Disease
- The project is designed to be reproducible and easy to run locally
- The final report outline and setup guidance are available in the docs/ folder
