from __future__ import annotations

from typing import Any, Dict

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.data.loader import load_heart_disease_data
from src.data.preprocessor import prepare_training_data
from src.features.engineering import build_preprocessing_pipeline
from src.models.mlflow_utils import log_model_run
from src.models.persistence import save_model_artifacts


def train_baseline_models() -> Dict[str, Dict[str, Any]]:
    X_train, X_test, y_train, y_test = prepare_training_data()
    df = load_heart_disease_data()
    target_col = "target" if "target" in df.columns else "num"

    candidate_grids = {
        "logistic_regression": [
            {"C": 0.1, "solver": "liblinear"},
            {"C": 1.0, "solver": "liblinear"},
            {"C": 10.0, "solver": "lbfgs"},
        ],
        "random_forest": [
            {"n_estimators": 20, "max_depth": None},
            {"n_estimators": 50, "max_depth": None},
            {"n_estimators": 80, "max_depth": 5},
        ],
    }

    def build_metrics(model_name: str, predictions: Any, true_labels: Any, best_params: Any) -> Dict[str, Any]:
        return {
            "accuracy": accuracy_score(true_labels, predictions),
            "precision": precision_score(true_labels, predictions, zero_division=0),
            "recall": recall_score(true_labels, predictions, zero_division=0),
            "f1": f1_score(true_labels, predictions, zero_division=0),
            "roc_auc": roc_auc_score(true_labels, predictions),
            "best_params": best_params,
            "model": None,
            "model_name": model_name,
        }

    metrics = {}
    for name, candidate_params in candidate_grids.items():
        base_model = LogisticRegression(max_iter=1000) if name == "logistic_regression" else RandomForestClassifier(random_state=42)
        best_model = None
        best_scores = None
        best_params = None

        for params in candidate_params:
            model = clone(base_model)
            for key, value in params.items():
                setattr(model, key, value)
            pipeline = Pipeline(steps=[("preprocessor", build_preprocessing_pipeline()), ("classifier", model)])
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            score = accuracy_score(y_test, predictions)
            if best_scores is None or score > best_scores:
                best_scores = score
                best_params = params
                best_model = pipeline

        metrics[name] = build_metrics(name, best_model.predict(X_test), y_test, best_params)
        metrics[name]["model"] = best_model

    for name, payload in metrics.items():
        log_model_run(payload, name, payload["model"])

    save_model_artifacts(metrics)
    return metrics
