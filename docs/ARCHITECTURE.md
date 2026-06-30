# Architecture Overview

The project follows a simple MLOps pipeline:

1. Data ingestion from the Heart Disease dataset
2. Data preprocessing and feature engineering
3. Model training and hyperparameter tuning
4. Experiment logging with MLflow
5. Model persistence for reuse
6. FastAPI prediction service
7. Dockerized deployment and local Kubernetes exposure
8. CI/CD automation through GitHub Actions

## High-level flow

Client -> FastAPI API -> Trained Model -> Prediction Response
                ^
                |
         MLflow / Docker / CI
