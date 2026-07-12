"""Pydantic schemas used by the API endpoints."""

from pydantic import BaseModel


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
