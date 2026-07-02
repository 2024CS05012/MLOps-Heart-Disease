from __future__ import annotations

# ruff: noqa: E402

import json
import os
from pathlib import Path
from typing import Any, Dict

MPLCONFIGDIR = Path("artifacts/.matplotlib").resolve()
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data.preprocessor import prepare_training_data
from src.features.engineering import build_preprocessing_pipeline
from src.models.mlflow_utils import log_model_run
from src.models.persistence import save_model_artifacts

ARTIFACT_DIR = Path("artifacts")


def _build_cv(y_train: Any) -> StratifiedKFold:
    min_class_count = int(y_train.value_counts().min())
    n_splits = max(2, min(5, min_class_count))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def _positive_class_scores(model: Any, X_test: Any) -> Any:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return model.decision_function(X_test)


def _save_evaluation_artifacts(model_name: str, model: Any, X_test: Any, y_test: Any, predictions: Any, probabilities: Any) -> list[Path]:
    output_dir = ARTIFACT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    classification_report_path = output_dir / "classification_report.json"
    with classification_report_path.open("w", encoding="utf-8") as file:
        json.dump(classification_report(y_test, predictions, output_dict=True, zero_division=0), file, indent=2)

    confusion_matrix_path = output_dir / "confusion_matrix.png"
    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=160)
    plt.close()

    roc_curve_path = output_dir / "roc_curve.png"
    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title(f"{model_name} ROC Curve")
    plt.tight_layout()
    plt.savefig(roc_curve_path, dpi=160)
    plt.close()

    return [classification_report_path, confusion_matrix_path, roc_curve_path]


def train_baseline_models() -> Dict[str, Dict[str, Any]]:
    X_train, X_test, y_train, y_test = prepare_training_data()
    cv = _build_cv(y_train)

    model_specs = {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "param_grid": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__solver": ["liblinear"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "param_grid": {
                "classifier__n_estimators": [50, 100, 150],
                "classifier__max_depth": [None, 4, 8],
                "classifier__min_samples_split": [2, 5],
            },
        },
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    def build_metrics(model_name: str, model: Any, predictions: Any, probabilities: Any, best_params: Any, best_score: float) -> Dict[str, Any]:
        cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        return {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "cv_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
            "cv_precision_mean": float(cv_scores["test_precision"].mean()),
            "cv_recall_mean": float(cv_scores["test_recall"].mean()),
            "cv_f1_mean": float(cv_scores["test_f1"].mean()),
            "cv_roc_auc_mean": float(cv_scores["test_roc_auc"].mean()),
            "grid_search_best_roc_auc": best_score,
            "best_params": best_params,
            "model": None,
            "model_name": model_name,
        }

    metrics = {}
    artifact_paths: dict[str, list[Path]] = {}
    for name, spec in model_specs.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessing_pipeline()),
                ("classifier", spec["estimator"]),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec["param_grid"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        probabilities = _positive_class_scores(best_model, X_test)

        metrics[name] = build_metrics(
            name,
            best_model,
            predictions,
            probabilities,
            search.best_params_,
            float(search.best_score_),
        )
        metrics[name]["model"] = best_model
        artifact_paths[name] = _save_evaluation_artifacts(name, best_model, X_test, y_test, predictions, probabilities)

    save_model_artifacts(metrics)
    for name, payload in metrics.items():
        log_model_run(payload, name, payload["model"], artifact_paths.get(name))

    return metrics


if __name__ == "__main__":
    results = train_baseline_models()
    for model_name, payload in results.items():
        print(
            model_name,
            {
                "accuracy": round(payload["accuracy"], 4),
                "f1": round(payload["f1"], 4),
                "roc_auc": round(payload["roc_auc"], 4),
                "best_params": payload["best_params"],
            },
        )
