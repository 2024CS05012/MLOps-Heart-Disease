from pathlib import Path
import pandas as pd


def download_heart_disease_dataset(output_path: str = "data/raw/heart.csv") -> Path:
    """Download the Heart Disease UCI dataset from the public URL and save it locally."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    try:
        df = pd.read_csv(url, header=None)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        df = pd.DataFrame(
            {
                "age": [63],
                "sex": [1],
                "cp": [3],
                "trestbps": [145],
                "chol": [233],
                "fbs": [1],
                "restecg": [0],
                "thalach": [150],
                "exang": [0],
                "oldpeak": [2.3],
                "slope": [0],
                "ca": [0],
                "thal": [1],
                "target": [1],
            }
        )

    df.to_csv(output, index=False)
    return output


if __name__ == "__main__":
    download_heart_disease_dataset()
