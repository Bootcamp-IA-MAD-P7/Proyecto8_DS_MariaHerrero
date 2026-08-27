import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.split_dataset import RANDOM_SEED, TARGET


def create_dataset():
    return pd.DataFrame(
        {
            "feature": range(100),
            "stroke": [0] * 95 + [1] * 5,
        }
    )


def make_split(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_dev,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def test_target_is_excluded_from_features():
    df = create_dataset()

    X = df.drop(columns=[TARGET])

    assert TARGET not in X.columns


def test_split_is_reproducible():
    df = create_dataset()

    first_split = make_split(df)
    second_split = make_split(df)

    for first, second in zip(first_split, second_split):
        assert first.equals(second)


def test_split_preserves_class_distribution():
    df = create_dataset()

    _, _, _, y_train, y_val, y_test = make_split(df)

    original_rate = df[TARGET].mean()

    for split in [y_train, y_val, y_test]:
        assert abs(split.mean() - original_rate) < 0.02


def test_split_sizes():
    df = create_dataset()

    X_train, X_val, X_test, _, _, _ = make_split(df)

    assert len(X_train) == 64
    assert len(X_val) == 16
    assert len(X_test) == 20