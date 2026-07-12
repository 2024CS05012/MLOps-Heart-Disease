from __future__ import annotations

"""Utilities for tracking training runs with MLflow."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException


logger = logging.getLogger(__name__)


def configure_mlflow(tracking_uri: str | None = None) -> None:
    # Set a local MLflow tracking backend and experiment name for training runs.
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlruns/mlflow.db"

    Path("mlruns").mkdir(exist_ok=True)
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart-disease-mlops")


def log_model_run(
    metrics: Dict[str, Any],
    model_name: str,
    model: Any,
    artifact_paths: Iterable[Path] | None = None,
) -> None:
    configure_mlflow()
    # Log hyperparameters, metrics, and evaluation artifacts for the trained model.
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        for key, value in metrics.get("best_params", {}).items():
            mlflow.log_param(key, value)

        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mlflow.log_metric(key, float(value))

        artifact_path = Path("models") / f"{model_name}.joblib"
        if artifact_path.exists():
            try:
                mlflow.log_artifact(str(artifact_path))
            except Exception:
                pass

        for path in artifact_paths or []:
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="evaluation")

        try:
            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
                skops_trusted_types=["numpy.dtype"],
            )
        except (MlflowException, TypeError) as exc:
            logger.warning("Skipping MLflow sklearn model logging for %s: %s", model_name, exc)
