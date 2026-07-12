"""Tests for the FastAPI prediction service."""

from fastapi.testclient import TestClient

from src.api.main import PredictRequest, create_app


def test_create_app_returns_fastapi_instance():
    app = create_app()
    assert app is not None


def test_predict_endpoint_returns_prediction():
    client = TestClient(create_app())
    response = client.post(
        "/predict",
        json={
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1,
        },
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()


def test_metrics_endpoint_is_exposed():
    client = TestClient(create_app())
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests" in response.text


def test_predict_request_accepts_thal_field():
    payload = PredictRequest(
        age=63,
        sex=1,
        cp=3,
        trestbps=145,
        chol=233,
        fbs=1,
        restecg=0,
        thalach=150,
        exang=0,
        oldpeak=2.3,
        slope=0,
        ca=0,
        thal=1,
    )

    assert payload.thal == 1
