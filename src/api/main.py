from fastapi import FastAPI
from pydantic import BaseModel

from src.models.persistence import load_model_artifact


class PredictRequest(BaseModel):
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
    th: int


class PredictResponse(BaseModel):
    prediction: int
    confidence: float


def create_app() -> FastAPI:
    app = FastAPI(title="Heart Disease MLOps API")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        model = load_model_artifact("logistic_regression")
        row = [
            payload.age,
            payload.sex,
            payload.cp,
            payload.trestbps,
            payload.chol,
            payload.fbs,
            payload.restecg,
            payload.thalach,
            payload.exang,
            payload.oldpeak,
            payload.slope,
            payload.ca,
            payload.th,
        ]
        prediction = int(model.predict([row])[0])
        probabilities = model.predict_proba([row])[0]
        confidence = float(max(probabilities))
        return PredictResponse(prediction=prediction, confidence=confidence)

    return app


app = create_app()
