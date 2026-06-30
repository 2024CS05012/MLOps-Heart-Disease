from src.api.main import PredictRequest, create_app


def test_create_app_returns_fastapi_instance():
    app = create_app()
    assert app is not None


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
