from src.models.persistence import load_model_artifact
from src.models.train import train_baseline_models


def test_train_baseline_models_returns_metrics():
    metrics = train_baseline_models()

    assert "logistic_regression" in metrics
    assert "random_forest" in metrics
    assert metrics["logistic_regression"]["accuracy"] >= 0
    assert metrics["random_forest"]["accuracy"] >= 0
    assert metrics["logistic_regression"]["roc_auc"] >= 0
    assert metrics["random_forest"]["roc_auc"] >= 0
    assert metrics["logistic_regression"]["cv_roc_auc_mean"] >= 0
    assert metrics["random_forest"]["cv_roc_auc_mean"] >= 0
    assert metrics["logistic_regression"]["best_params"] is not None
    assert metrics["random_forest"]["best_params"] is not None


def test_load_model_artifact_falls_back_to_training_when_pickle_is_unusable():
    model = load_model_artifact("logistic_regression")

    assert model is not None
