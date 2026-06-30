# Final Report Outline

## 1. Project Overview
- This project predicts whether a patient has heart disease using the UCI Heart Disease dataset.
- The solution follows an end-to-end MLOps workflow covering data loading, exploratory analysis, preprocessing, model training, experiment tracking, API serving, containerization, CI/CD, and deployment scaffolding.
- The goal is to build a reproducible and testable machine learning pipeline that can be reused for future experiments.

## 2. Data Acquisition and EDA
- The dataset is loaded from a local CSV when available and otherwise falls back to a small built-in sample for reproducibility.
- The data loader and preprocessing modules prepare the data for training.
- EDA focuses on basic dataset inspection, target balance, and feature readiness for modeling.
- Key insights include the need for imputation, scaling, and categorical handling before modeling.

## 3. Model Development
- The preprocessing pipeline combines numeric scaling and categorical encoding.
- Two baseline models were trained: Logistic Regression and Random Forest.
- Hyperparameter candidates were evaluated using a simple grid-style comparison.
- Evaluation metrics include accuracy, precision, recall, F1-score, and ROC-AUC.

## 4. Experiment Tracking
- MLflow is used to track model runs and store metrics and artifacts.
- Each trained model logs parameters, metrics, and a saved sklearn model artifact.
- The experiment is configured to write to the local mlruns directory.

## 5. API and Deployment
- A FastAPI application exposes a health endpoint and a prediction endpoint.
- The API accepts the heart disease feature vector and returns a prediction with confidence.
- Docker and Kubernetes starter files are included for containerized deployment.

## 6. CI/CD and Monitoring
- GitHub Actions runs pytest automatically on push and pull requests.
- The testing strategy includes unit tests for data loading, preprocessing, model training, and API creation.
- Monitoring support is scaffolded through a basic Prometheus configuration and logging-friendly project structure.

## 7. Setup Instructions
- Create and activate a Python virtual environment.
- Install dependencies with pip install -r requirements.txt.
- Run pytest to validate the project.
- Start the API with uvicorn src.api.main:app --reload.
- Run Docker or Kubernetes assets from the docker/ and k8s/ folders.

## 8. Repository Link
- GitHub repository: https://github.com/2024CS05012/MLOps-Heart-Disease
