import pandas as pd

from src.data.split_dataset import TARGET, split_dataset
from src.preprocessing.pipeline import create_preprocessor


def create_dataset():
    rows = 100

    return pd.DataFrame(
        {
            "gender": ["Female", "Male"] * 50,
            "age": [20 + (i * 0.5) for i in range(rows)],
            "hypertension": [0, 1] * 50,
            "heart_disease": [0, 0, 1, 0] * 25,
            "ever_married": ["Yes", "No"] * 50,
            "work_type": ["Private", "Self-employed"] * 50,
            "Residence_type": ["Urban", "Rural"] * 50,
            "avg_glucose_level": [80 + i for i in range(rows)],
            "bmi": [20 + (i * 0.1) for i in range(rows)],
            "smoking_status": ["never smoked", "formerly smoked"] * 50,
            "stroke": [0] * 95 + [1] * 5,
        }
    )


def test_target_not_present_in_features():
    df = create_dataset()

    X_train, X_val, X_test, _, _, _ = split_dataset(df)

    assert TARGET not in X_train.columns
    assert TARGET not in X_val.columns
    assert TARGET not in X_test.columns


def test_train_validation_test_are_disjoint():
    df = create_dataset()

    X_train, X_val, X_test, _, _, _ = split_dataset(df)

    train_indexes = set(X_train.index)
    val_indexes = set(X_val.index)
    test_indexes = set(X_test.index)

    assert train_indexes.isdisjoint(val_indexes)
    assert train_indexes.isdisjoint(test_indexes)
    assert val_indexes.isdisjoint(test_indexes)


def test_preprocessor_is_fitted_only_on_train():
    df = create_dataset()

    X_train, X_val, X_test, _, _, _ = split_dataset(df)

    preprocessor = create_preprocessor()

    preprocessor.fit(X_train)

    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)

    assert X_val_transformed.shape[0] == len(X_val)
    assert X_test_transformed.shape[0] == len(X_test)


def test_validation_and_test_keep_original_size():
    df = create_dataset()

    _, X_val, X_test, _, y_val, y_test = split_dataset(df)

    original_val_size = len(X_val)
    original_test_size = len(X_test)

    assert len(X_val) == original_val_size
    assert len(y_val) == original_val_size

    assert len(X_test) == original_test_size
    assert len(y_test) == original_test_size