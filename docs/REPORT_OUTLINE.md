# Final Report Outline

## 1. Project Overview
- This project predicts whether a patient has heart disease using the UCI Heart Disease dataset.
- The solution follows an end-to-end MLOps workflow covering data loading, exploratory analysis, preprocessing, model training, experiment tracking, API serving, containerization, CI/CD, and deployment scaffolding.
- The goal is to build a reproducible and testable machine learning pipeline that can be reused for future experiments.

## 2. Data Acquisition and EDA
- The dataset is downloaded from the UCI Cleveland Heart Disease source using `data/raw/download_dataset.py`.
- The original UCI target values are converted to a binary target: 0 = no disease, 1 = disease.
- Missing values represented as `?` are converted to null values and handled inside the preprocessing pipeline.
- EDA artifacts are generated with `python -m src.data.generate_eda_artifacts`.
- Include histograms, missing-value chart, class distribution, correlation heatmap, and maximum-heart-rate relationship plot from `artifacts/eda/`.

## 3. Model Development
- The preprocessing pipeline combines median imputation and scaling for numeric features with most-frequent imputation and one-hot encoding for categorical features.
- Two baseline models were trained: Logistic Regression and Random Forest.
- Hyperparameters were selected using GridSearchCV with stratified cross-validation.
- Evaluation metrics include accuracy, precision, recall, F1-score, ROC-AUC, and cross-validation summaries.

## 4. Experiment Tracking
- MLflow is used to track model runs and store metrics and artifacts.
- Each trained model logs parameters, metrics, confusion matrix, ROC curve, classification report, joblib artifact, and sklearn model artifact.
- The experiment is configured to write to the local mlruns directory.

## 5. API and Deployment
- A FastAPI application exposes a health endpoint and a prediction endpoint.
- The API accepts the heart disease feature vector and returns a prediction with confidence.
- Docker and Kubernetes files are included for containerized deployment.
- The service also exposes `/metrics` for Prometheus scraping.

## 6. CI/CD and Monitoring
- GitHub Actions runs linting, dataset download, EDA generation, model training, tests, and Docker build validation.
- The testing strategy includes unit tests for data loading, preprocessing, model training, API prediction, and monitoring metrics.
- Monitoring support includes request logging and a Prometheus configuration for scraping FastAPI metrics.

## 7. Setup Instructions
- Create and activate a Python virtual environment.
- Install dependencies with pip install -r requirements.txt.
- Download the dataset and train models.
- Run pytest to validate the project.
- Start the API with uvicorn src.api.main:app --reload.
- Run Docker or Kubernetes assets from the docker/ and k8s/ folders.

## 8. Repository Link
- GitHub repository: https://github.com/2024CS05012/MLOps-Heart-Disease
