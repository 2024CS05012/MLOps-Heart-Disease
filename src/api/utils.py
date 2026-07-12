"""Helper utilities for converting API payloads into model input rows."""

from typing import Any, List


def build_feature_row(payload: Any) -> List[float]:
    """Convert a request payload into the ordered feature vector expected by the model."""
    return [
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
        payload.thal,
    ]
