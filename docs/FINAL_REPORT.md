# Heart Disease Risk Prediction MLOps Pipeline

## 1. Project Overview

This project implements an end-to-end MLOps workflow for predicting the presence of heart disease from patient clinical features. The solution uses the UCI Cleveland Heart Disease dataset and includes reproducible data acquisition, EDA, preprocessing, model training, experiment tracking, API serving, containerization, CI/CD, Kubernetes deployment manifests, and monitoring.

## 2. Dataset

The dataset is downloaded from the UCI Machine Learning Repository using `data/raw/download_dataset.py`. The original target field represents disease severity from 0 to 4. For this binary classification task, target value 0 is mapped to no disease and values 1-4 are mapped to disease.

The final feature set includes age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression, slope, number of vessels, and thalassemia category.

## 3. Exploratory Data Analysis

EDA figures are generated using:

```bash
python -m src.data.generate_eda_artifacts
```

The generated artifacts include:

- Numeric feature histograms
- Missing-value analysis
- Class distribution
- Correlation heatmap
- Maximum heart rate by target class

Add screenshots from `artifacts/eda/` in this section.

Store final submission screenshots in the `screenshots/` folder.

## 4. Preprocessing and Feature Engineering

The project uses a reusable sklearn `ColumnTransformer` inside the final model pipeline. Numeric features are median-imputed and scaled using `StandardScaler`. Categorical features are imputed using the most frequent value and encoded using `OneHotEncoder`.

Because preprocessing is saved inside the model pipeline, the same transformations are used during training and API inference.

## 5. Model Development

Two classification models are trained:

- Logistic Regression
- Random Forest

Hyperparameter tuning is performed using `GridSearchCV` with stratified cross-validation. The models are evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and cross-validation averages.

Training command:

```bash
python -m src.models.train
```

## 6. Experiment Tracking

MLflow tracks model parameters, metrics, evaluation plots, classification reports, serialized joblib models, and sklearn model artifacts.

MLflow UI command:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Add MLflow screenshots showing runs, metrics, parameters, and artifacts.

## 7. API Serving

The model is served using FastAPI. The API exposes:

- `GET /health`
- `POST /predict`
- `GET /metrics`

The `/predict` endpoint accepts JSON input and returns a binary prediction with a confidence score.

## 8. Containerization

The API can be containerized using Docker:

```bash
docker build -f docker/Dockerfile -t heart-disease-api .
docker run --rm -p 8000:8000 heart-disease-api
```

Add Docker build and API test screenshots.

## 9. CI/CD

GitHub Actions validates the project on push and pull request. The workflow runs linting, dataset download, EDA artifact generation, model training, unit tests, and Docker build validation.

Add GitHub Actions workflow screenshots.

## 10. Deployment and Monitoring

Kubernetes deployment files are provided in `k8s/deployment.yaml`. The service can be exposed locally through Minikube or Docker Desktop Kubernetes.

Monitoring is supported through FastAPI request logs and Prometheus-compatible `/metrics`. Docker Compose includes Prometheus for local scraping.

Add Kubernetes service/pod screenshots and Prometheus screenshots.

## 11. Repository

Repository link: https://github.com/2024CS05012/MLOps-Heart-Disease
