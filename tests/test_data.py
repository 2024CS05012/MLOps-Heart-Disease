import pandas as pd

from src.data.loader import load_heart_disease_data
from src.data.preprocessor import prepare_training_data


def test_load_heart_disease_data_returns_dataframe():
    df = load_heart_disease_data()

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert "target" in df.columns or "num" in df.columns


def test_prepare_training_data_returns_train_test_splits():
    X_train, X_test, y_train, y_test = prepare_training_data()

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]
