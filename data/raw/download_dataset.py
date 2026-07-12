"""Download the UCI Cleveland heart disease dataset into the local data folder."""

from pathlib import Path

import pandas as pd


UCI_CLEVELAND_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

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


def download_heart_disease_dataset(output_path: str = "data/raw/heart.csv") -> Path:
    """Download and clean the UCI Cleveland heart disease dataset."""
    # Prepare the output location and read the remote CSV with the expected column names.
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        UCI_CLEVELAND_URL,
        names=HEART_DISEASE_COLUMNS,
        na_values="?",
    )
    for column in HEART_DISEASE_COLUMNS:
        # Convert every column to numeric values while tolerating missing entries.
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # The original UCI target uses 0 for no disease and 1-4 for disease severity.
    df["target"] = (df["target"] > 0).astype(int)

    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")
    return output


if __name__ == "__main__":
    download_heart_disease_dataset()
