from __future__ import annotations

# ruff: noqa: E402

import os
from pathlib import Path

MPLCONFIGDIR = Path("artifacts/.matplotlib").resolve()
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.data.loader import load_heart_disease_data


EDA_DIR = Path("artifacts/eda")


def generate_eda_artifacts(output_dir: Path = EDA_DIR) -> list[Path]:
    """Generate assignment-ready EDA figures for the report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_heart_disease_data()
    generated_files: list[Path] = []

    histogram_path = output_dir / "feature_histograms.png"
    df[["age", "trestbps", "chol", "thalach", "oldpeak"]].hist(figsize=(12, 8), bins=20)
    plt.suptitle("Numeric Feature Distributions")
    plt.tight_layout()
    plt.savefig(histogram_path, dpi=160)
    plt.close()
    generated_files.append(histogram_path)

    missing_values_path = output_dir / "missing_values.png"
    missing_values = df.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing_values.index, y=missing_values.values)
    plt.title("Missing Values by Feature")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing count")
    plt.tight_layout()
    plt.savefig(missing_values_path, dpi=160)
    plt.close()
    generated_files.append(missing_values_path)

    class_distribution_path = output_dir / "class_distribution.png"
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="target")
    plt.title("Heart Disease Class Distribution")
    plt.xlabel("Target: 0 = no disease, 1 = disease")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(class_distribution_path, dpi=160)
    plt.close()
    generated_files.append(class_distribution_path)

    correlation_heatmap_path = output_dir / "correlation_heatmap.png"
    plt.figure(figsize=(12, 9))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(correlation_heatmap_path, dpi=160)
    plt.close()
    generated_files.append(correlation_heatmap_path)

    relationship_path = output_dir / "thalach_by_target.png"
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="target", y="thalach")
    plt.title("Maximum Heart Rate by Target Class")
    plt.xlabel("Target")
    plt.ylabel("Maximum heart rate achieved")
    plt.tight_layout()
    plt.savefig(relationship_path, dpi=160)
    plt.close()
    generated_files.append(relationship_path)

    return generated_files


if __name__ == "__main__":
    for path in generate_eda_artifacts():
        print(path)
