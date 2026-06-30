from pathlib import Path
from typing import Optional

import pandas as pd


def load_heart_disease_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load the heart disease dataset from a CSV file if present, otherwise use a small built-in sample."""
    if csv_path is None:
        candidate_paths = [
            Path("data/raw/heart.csv"),
            Path("data/raw/processed.cleveland.data"),
            Path("heart.csv"),
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                return pd.read_csv(candidate)

    if csv_path is not None:
        return pd.read_csv(csv_path)

    # Fallback sample to keep the project runnable without the dataset upfront.
    return pd.DataFrame(
        {
            "age": [63, 67, 68, 44, 62],
            "sex": [1, 1, 0, 1, 0],
            "cp": [3, 2, 0, 1, 2],
            "trestbps": [145, 160, 138, 120, 140],
            "chol": [233, 286, 166, 263, 268],
            "fbs": [1, 0, 0, 0, 0],
            "restecg": [0, 1, 0, 0, 1],
            "thalach": [150, 108, 125, 173, 160],
            "exang": [0, 1, 1, 0, 0],
            "oldpeak": [2.3, 1.5, 0.2, 0.0, 3.6],
            "slope": [0, 1, 2, 2, 2],
            "ca": [0, 3, 1, 0, 2],
            "thal": [1, 2, 2, 3, 2],
            "target": [1, 1, 0, 0, 1],
        }
    )
