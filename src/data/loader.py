"""Helpers for loading and normalizing the heart disease dataset."""

from pathlib import Path
from typing import Optional

import pandas as pd


HEART_DISEASE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def normalize_heart_disease_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return the heart disease data with stable column names and binary target."""
    df = df.copy()

    # Handle the common column layout used by the Cleveland dataset.
    if list(df.columns) == list(range(len(HEART_DISEASE_COLUMNS))):
        df.columns = HEART_DISEASE_COLUMNS
    elif "num" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"num": "target"})

    # Ensure every required feature and the target column are present.
    missing_columns = [column for column in HEART_DISEASE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Replace missing placeholders with pandas missing values and coerce to numeric types.
    df = df[HEART_DISEASE_COLUMNS].replace("?", pd.NA)
    for column in HEART_DISEASE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["target"] = (df["target"] > 0).astype(int)
    return df


def load_heart_disease_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load the heart disease dataset from CSV, falling back to a smoke-test sample."""
    # Try the project data path first, then the common alternate filenames.
    if csv_path is None:
        candidate_paths = [
            Path("data/raw/heart.csv"),
            Path("data/raw/processed.cleveland.data"),
            Path("heart.csv"),
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                if candidate.name == "processed.cleveland.data":
                    df = pd.read_csv(candidate, names=HEART_DISEASE_COLUMNS, na_values="?")
                else:
                    df = pd.read_csv(candidate)
                return normalize_heart_disease_data(df)

    if csv_path is not None:
        path = Path(csv_path)
        if path.name == "processed.cleveland.data":
            return normalize_heart_disease_data(pd.read_csv(path, names=HEART_DISEASE_COLUMNS, na_values="?"))
        return normalize_heart_disease_data(pd.read_csv(path))

    # Offline fallback keeps tests and API smoke runs usable before downloading data.
    sample = pd.DataFrame(
        {
            "age": [63, 67, 68, 44, 62, 57, 54, 48, 52, 45, 58, 60, 41, 49, 56, 64, 43, 59, 51, 46],
            "sex": [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            "cp": [3, 2, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 3, 1, 2, 0, 1],
            "trestbps": [145, 160, 138, 120, 140, 130, 125, 118, 134, 112, 132, 142, 110, 128, 136, 148, 122, 126, 135, 115],
            "chol": [233, 286, 166, 263, 268, 236, 212, 204, 250, 210, 224, 289, 172, 230, 256, 280, 213, 218, 245, 198],
            "fbs": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            "restecg": [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
            "thalach": [150, 108, 125, 173, 160, 155, 168, 178, 142, 175, 162, 120, 182, 171, 150, 132, 179, 164, 145, 176],
            "exang": [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
            "oldpeak": [2.3, 1.5, 0.2, 0.0, 3.6, 0.4, 0.1, 0.0, 1.2, 0.0, 0.3, 2.8, 0.0, 0.2, 1.4, 2.1, 0.0, 0.5, 1.0, 0.0],
            "slope": [0, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 0, 2, 2, 1, 0, 2, 2, 1, 2],
            "ca": [0, 3, 1, 0, 2, 1, 0, 0, 2, 0, 0, 3, 0, 0, 1, 2, 0, 0, 1, 0],
            "thal": [1, 2, 2, 3, 2, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2],
            "target": [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
        }
    )
    return normalize_heart_disease_data(sample)
