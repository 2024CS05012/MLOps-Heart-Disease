from __future__ import annotations

"""Utilities for preparing train and test splits for model training."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_heart_disease_data


def prepare_training_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Load the cleaned dataset and split features from the target label.
    df = load_heart_disease_data()
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    # Use stratified splitting when possible so both classes are represented in train and test sets.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError:
        # Fall back to a non-stratified split if the dataset is too small for stratification.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    return X_train, X_test, y_train, y_test
