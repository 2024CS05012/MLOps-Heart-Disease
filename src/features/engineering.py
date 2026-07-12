from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.loader import load_heart_disease_data


# Define the expected numeric and categorical feature groups.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def get_feature_columns() -> list[str]:
    """Return the list of input features used by the model."""
    return FEATURE_COLUMNS


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Create a preprocessing pipeline for numeric and categorical features."""
    # Load the dataset once so the pipeline can adapt to the available columns.
    df = load_heart_disease_data()

    # Exclude the target column and keep only the feature columns that are actually present.
    available_features = set(df.drop(columns=["target"]).columns)
    numeric_features = [column for column in NUMERIC_FEATURES if column in available_features]
    categorical_features = [column for column in CATEGORICAL_FEATURES if column in available_features]

    # Apply median imputation and scaling to numeric columns.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Apply most-frequent imputation and one-hot encoding to categorical columns.
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Route each feature subset through its matching preprocessing steps.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor
