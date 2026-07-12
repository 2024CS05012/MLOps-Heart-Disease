"""FastAPI application for serving heart disease predictions."""

import logging
import time

import pandas as pd
from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.api.utils import build_feature_row
from src.features.engineering import get_feature_columns
from src.models.persistence import load_model_artifact

logger = logging.getLogger("heart_disease_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class PredictRequest(BaseModel):
    """Schema for incoming prediction requests."""
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


class PredictResponse(BaseModel):
    """Schema for prediction responses returned by the API."""
    prediction: int
    confidence: float


def create_app() -> FastAPI:
    # Build the FastAPI app and expose Prometheus metrics for monitoring.
    app = FastAPI(title="Heart Disease MLOps API")
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "method=%s path=%s status_code=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.get("/health")
    # Simple readiness endpoint for deployment and smoke testing.
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    # Load the model, build a feature row, and return the prediction plus confidence.
    def predict(payload: PredictRequest) -> PredictResponse:
        model = load_model_artifact("logistic_regression")
        row = build_feature_row(payload)
        feature_frame = pd.DataFrame([row], columns=get_feature_columns())
        prediction = int(model.predict(feature_frame)[0])
        probabilities = model.predict_proba(feature_frame)[0]
        confidence = float(max(probabilities))
        return PredictResponse(prediction=prediction, confidence=confidence)

    return app


app = create_app()
