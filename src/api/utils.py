from typing import Any, List


def build_feature_row(payload: Any) -> List[float]:
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
        payload.th,
    ]
