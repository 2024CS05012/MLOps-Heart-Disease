from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any, Dict


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def save_model_artifacts(models: Dict[str, Any]) -> None:
    for name, payload in models.items():
        model = payload.get("model")
        if model is None:
            continue
        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)


def load_model_artifact(name: str) -> Any:
    model_path = MODEL_DIR / f"{name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)
