from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_heart_disease_data


def prepare_training_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_heart_disease_data()
    target_col = "target" if "target" in df.columns else "num"

    if df[target_col].dtype == object:
        df[target_col] = df[target_col].astype(int)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    return X_train, X_test, y_train, y_test
