"""Persist and load trained model artifacts from disk."""

from __future__ import annotations

import logging
import joblib
from pathlib import Path
from typing import Any, Dict


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("heart_disease_api")


def save_model_artifacts(models: Dict[str, Any]) -> None:
    # Save each trained model to the models directory for later inference.
    for name, payload in models.items():
        model = payload.get("model")
        if model is None:
            continue
        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)


def _train_and_get_model(name: str) -> Any:
    # Re-train the requested model if the artifact is missing or cannot be loaded.
    from src.models.train import train_baseline_models

    metrics = train_baseline_models()
    if name not in metrics or metrics[name].get("model") is None:
        raise FileNotFoundError(f"Model artifact not found: {MODEL_DIR / f'{name}.joblib'}")
    return metrics[name]["model"]


def load_model_artifact(name: str) -> Any:
    # Load a model from disk when available; otherwise fall back to training it.
    model_path = MODEL_DIR / f"{name}.joblib"
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as exc:  # pragma: no cover - exercised in container runtime
            logger.warning("Falling back to retraining because model artifact could not be loaded: %s", exc)
            return _train_and_get_model(name)

    return _train_and_get_model(name)
