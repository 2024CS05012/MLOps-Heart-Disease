from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import mlflow
import mlflow.sklearn


def configure_mlflow(tracking_uri: str | None = None) -> None:
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlruns/mlflow.db"

    Path("mlruns").mkdir(exist_ok=True)
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart-disease-mlops")


def log_model_run(metrics: Dict[str, Any], model_name: str, model: Any) -> None:
    configure_mlflow()
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", metrics.get("accuracy", 0.0))
        mlflow.log_metric("precision", metrics.get("precision", 0.0))
        mlflow.log_metric("recall", metrics.get("recall", 0.0))
        mlflow.log_metric("f1", metrics.get("f1", 0.0))
        mlflow.log_metric("roc_auc", metrics.get("roc_auc", 0.0))

        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception:
            mlflow.log_artifact(str(Path("models") / f"{model_name}.joblib"))
